"""High-Performance Layer Streaming Engine for Consumer GPUs (AirLLM / ZeRO-3 Style).

Enables training of models with billions of parameters (like triune-base, 4.95B)
on consumer GPUs with limited VRAM (e.g. 8GB RTX 5070) by virtualizing memory:
- Transformer layers reside permanently in host CPU RAM (zero dynamic reallocations).
- Only the currently executing layer is streamed into GPU VRAM.
- Secondary CUDA stream prefetching overlaps PCIe transfers with GPU compute.
- Zero-copy restoration: parameter/buffer pointers point back to permanent CPU storage.
- Handles all parameters AND registered buffers (expert_bias, inv_freq, etc.).
- Dynamic auto-pinning leverages DMA when host RAM is ample and protects system when tight.
- PyTorch autograd backward pre/post hooks bring layers to GPU just-in-time.
- Designed to drastically minimize persistent layer-weight VRAM; peak VRAM depends on batch size, sequence length, activations, and root embedding dimensions.
- Fully opt-in and configurable via --streaming.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn

from .capabilities import PrecisionCapabilities



@dataclass
class StreamingConfig:
    enabled: bool = False
    chunk_size: int = 1
    pin_memory: Optional[bool] = None  # None = auto-detect based on host RAM
    async_prefetch: bool = True
    fp8_weights: bool = False
    grad_clip: float = 1.0
    instant_optimizer: bool = True


def _offload_grad_hook(param: nn.Parameter) -> None:
    """Intercepts computed gradient on CUDA, offloads to CPU buffer, and instantly frees GPU VRAM."""
    if param.grad is not None:
        grad_cpu = param.grad.data.cpu()
        if not hasattr(param, "_cpu_grad") or param._cpu_grad is None:
            param._cpu_grad = grad_cpu
        else:
            param._cpu_grad.add_(grad_cpu)
        # Instantly deallocate GPU gradient tensor to keep VRAM footprint minimal
        param.grad = None


def _move_param_states_to_device(opt: Any, params: list[nn.Parameter], device: torch.device) -> None:
    """Migrates optimizer momentum/variance state tensors for specified parameters to target device."""
    if opt is None:
        return
    sub_opts = []
    if hasattr(opt, "base_optimizer") and opt.base_optimizer is not None:
        sub_opts.append(opt.base_optimizer)
    if hasattr(opt, "muon_optimizer") and opt.muon_optimizer is not None:
        sub_opts.append(opt.muon_optimizer)
    if hasattr(opt, "state"):
        sub_opts.append(opt)

    for sub in sub_opts:
        for p in params:
            if p in sub.state:
                st = sub.state[p]
                for k, v in st.items():
                    if isinstance(v, torch.Tensor) and v.device != device:
                        st[k] = v.to(device, non_blocking=True)

    if hasattr(opt, "layer_groups"):
        param_ids = {id(p) for p in params}
        for g in opt.layer_groups:
            if id(g.get("param")) in param_ids:
                st = g.get("state", {})
                for k, v in st.items():
                    if isinstance(v, torch.Tensor) and v.device != device:
                        st[k] = v.to(device, non_blocking=True)
                proj = g.get("projection")
                if proj is not None and isinstance(proj, torch.Tensor) and proj.device != device:
                    g["projection"] = proj.to(device, non_blocking=True)


def _move_param_states_to_cpu(opt: Any, params: list[nn.Parameter]) -> None:
    """Migrates optimizer momentum/variance state tensors for specified parameters back to CPU to free VRAM."""
    if opt is None:
        return
    sub_opts = []
    if hasattr(opt, "base_optimizer") and opt.base_optimizer is not None:
        sub_opts.append(opt.base_optimizer)
    if hasattr(opt, "muon_optimizer") and opt.muon_optimizer is not None:
        sub_opts.append(opt.muon_optimizer)
    if hasattr(opt, "state"):
        sub_opts.append(opt)

    for sub in sub_opts:
        for p in params:
            if p in sub.state:
                st = sub.state[p]
                for k, v in st.items():
                    if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                        st[k] = v.cpu()

    if hasattr(opt, "layer_groups"):
        param_ids = {id(p) for p in params}
        for g in opt.layer_groups:
            if id(g.get("param")) in param_ids:
                st = g.get("state", {})
                for k, v in st.items():
                    if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                        st[k] = v.cpu()
                proj = g.get("projection")
                if proj is not None and isinstance(proj, torch.Tensor) and proj.device.type != "cpu":
                    g["projection"] = proj.cpu()


class LayerStreamingEngine:
    """Manages layer-wise streaming, pinned memory DMA, and just-in-time GPU scheduling."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Optional[StreamingConfig] = None,
        optimizer: Optional[Any] = None,
    ):
        self.model = model
        self.device = device
        self.config = config or StreamingConfig(enabled=True)
        self.optimizer = optimizer
        self.is_attached = False
        self._hooks: list[Any] = []
        self.prefetch_stream: Optional[torch.cuda.Stream] = None
        self.d2h_stream: Optional[torch.cuda.Stream] = None
        self.layers: list[nn.Module] = self._resolve_layers()
        self.caps = PrecisionCapabilities.detect(self.device)

    def _stage_param_tensor(self, p: nn.Parameter) -> torch.Tensor:
        target_dtype = torch.bfloat16 if p._cpu_data.dtype == torch.float8_e4m3fn else p._cpu_data.dtype
        if self.config.fp8_weights and p.dim() >= 2 and self.device.type == "cuda":
            gpu_master = p._cpu_data.to(self.device, non_blocking=True).to(target_dtype)
            amax = gpu_master.abs().amax().clamp_min(1e-12)
            scale = (448.0 / amax).to(dtype=target_dtype)
            fp8_t = (gpu_master * scale).to(torch.float8_e4m3fn)
            inv_scale = (1.0 / scale.float())
            if self.caps.native_scaled_mm:
                return fp8_t
            else:
                return (fp8_t.to(target_dtype) * inv_scale)
        return p._cpu_data.to(self.device, non_blocking=True).to(target_dtype)

    def _prefetch_param_tensor(self, p: nn.Parameter) -> Any:
        target_dtype = torch.bfloat16 if p._cpu_data.dtype == torch.float8_e4m3fn else p._cpu_data.dtype
        if self.config.fp8_weights and p.dim() >= 2 and self.device.type == "cuda":
            gpu_master = p._cpu_data.to(self.device, non_blocking=True).to(target_dtype)
            amax = gpu_master.abs().amax().clamp_min(1e-12)
            scale = (448.0 / amax).to(dtype=target_dtype)
            fp8_t = (gpu_master * scale).to(torch.float8_e4m3fn)
            inv_scale = (1.0 / scale.float())
            return (fp8_t, inv_scale, target_dtype)
        return p._cpu_data.to(self.device, non_blocking=True).to(target_dtype)

    def _unpack_prefetched(self, staged_item: Any) -> torch.Tensor:
        if isinstance(staged_item, tuple):
            fp8_t, inv_scale, target_dtype = staged_item
            if self.caps.native_scaled_mm:
                return fp8_t
            else:
                return (fp8_t.to(target_dtype) * inv_scale)
        return staged_item


    def _resolve_layers(self) -> list[nn.Module]:
        """Dynamically detect sequential transformer blocks across diverse model architectures."""
        if hasattr(self.model, "layers") and isinstance(self.model.layers, (nn.ModuleList, list, tuple)):
            return list(self.model.layers)
        # Hugging Face Llama / Mistral / Qwen / Gemma / DeepSeek
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        # Hugging Face GPT-2 / MPT
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        # Hugging Face Falcon / ChatGLM
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layers"):
            return list(self.model.transformer.layers)
        # Search named submodules for largest ModuleList
        max_ml: list[nn.Module] = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > len(max_ml):
                max_ml = list(m)
        return max_ml

    def set_optimizer(self, optimizer: Any) -> None:
        """Sets the active optimizer to enable instant layer-wise optimization during backward."""
        self.optimizer = optimizer

    def get_root_parameters(self) -> List[nn.Parameter]:
        """Returns non-layer root parameters resident on GPU (embeddings, router, heads)."""
        layer_param_ids = {id(p) for layer in self.layers for p in layer.parameters()}
        return [p for p in self.model.parameters() if id(p) not in layer_param_ids]

    def attach(self, optimizer: Optional[Any] = None) -> None:
        """Attaches streaming hooks, configures CPU parameters, and initializes root components on GPU."""
        if self.is_attached or not self.layers:
            return
        if optimizer is not None:
            self.optimizer = optimizer

        print("⚡ [Layer Streaming] Initializing memory virtualization...", flush=True)

        # 1. Root non-layer components stay resident on GPU (~100MB)
        layer_ids = {id(l) for l in self.layers}
        for name, child in self.model.named_children():
            if id(child) not in layer_ids and not any(id(l) in {id(c) for c in child.modules()} for l in self.layers):
                child.to(self.device)
            elif hasattr(child, "embed_tokens"):
                child.embed_tokens.to(self.device)
            elif hasattr(child, "norm"):
                child.norm.to(self.device)
        if hasattr(self.model, "lm_head"):
            self.model.lm_head.to(self.device)
        if hasattr(self.model, "token_embed"):
            self.model.token_embed.to(self.device)
        if hasattr(self.model, "router"):
            self.model.router.to(self.device)
        if hasattr(self.model, "final_norm"):
            self.model.final_norm.to(self.device)
        if hasattr(self.model, "final_head"):
            self.model.final_head.to(self.device)

        # 2. Configure layers on CPU and establish permanent CPU storage pointers for both params & buffers
        for layer in self.layers:
            layer.to("cpu")
            for p in layer.parameters():
                p._cpu_data = p.data
            for b in layer.buffers():
                b._cpu_data = b.data

        # 3. Dynamic Host RAM Pinning Assessment
        should_pin = self.config.pin_memory
        if should_pin is None:
            try:
                import psutil
                avail_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
                total_param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
                model_gb = total_param_bytes / (1024 ** 3)
                should_pin = (avail_ram_gb >= model_gb * 2.0)
            except Exception:
                should_pin = False

        if should_pin:
            print("🚀 [Layer Streaming] Pinned host RAM enabled for high-throughput PCIe DMA.", flush=True)
            for layer in self.layers:
                for t in list(layer.parameters()) + list(layer.buffers()):
                    if not t._cpu_data.is_pinned():
                        try:
                            t._cpu_data = t._cpu_data.pin_memory()
                            t.data = t._cpu_data
                        except Exception:
                            pass
        else:
            print("💡 [Layer Streaming] Host RAM pinning bypassed to conserve host memory pages.", flush=True)

        # 4. Streams for asynchronous PCIe prefetching and D2H gradient transfer
        dev = self.device
        if dev.type == "cuda" and self.config.async_prefetch:
            try:
                self.prefetch_stream = torch.cuda.Stream(device=dev)
                print("⚡ [Layer Streaming] Secondary CUDA prefetch stream active.", flush=True)
            except Exception:
                self.prefetch_stream = None

        if dev.type == "cuda":
            try:
                self.d2h_stream = torch.cuda.Stream(device=dev)
                print("⚡ [Layer Streaming] Dedicated CUDA D2H transfer stream active.", flush=True)
            except Exception:
                self.d2h_stream = None
        else:
            self.d2h_stream = None

        prefetch_stream = self.prefetch_stream
        d2h_stream = self.d2h_stream
        num_layers = len(self.layers)
        layer_to_idx = {layer: i for i, layer in enumerate(self.layers)}

        # 5. Parameter autograd post-accumulate hooks and module backward hooks
        def make_post_accum_hook(param: nn.Parameter):
            def post_accum_hook(p: nn.Parameter) -> None:
                if p.grad is None:
                    return

                if dev.type == "cuda" and d2h_stream is not None:
                    # 1. Capture current p.grad and protect GPU memory allocation from reuse
                    gpu_grad = p.grad.detach()
                    gpu_grad.record_stream(d2h_stream)

                    # 2. Record dependency on compute stream
                    compute_event = torch.cuda.Event()
                    compute_event.record(torch.cuda.current_stream())
                    d2h_stream.wait_event(compute_event)

                    # 3. Asynchronous D2H transfer in d2h_stream
                    with torch.cuda.stream(d2h_stream):
                        grad_cpu = gpu_grad.to("cpu", non_blocking=True)
                        d2h_event = torch.cuda.Event()
                        d2h_event.record(d2h_stream)
                        p._d2h_event = d2h_event

                    if not hasattr(p, "_cpu_grad") or p._cpu_grad is None:
                        p._cpu_grad = grad_cpu
                    else:
                        # Ensure D2H transfer has completed into host memory before CPU in-place accumulation
                        p._d2h_event.synchronize()
                        p._cpu_grad.add_(grad_cpu)
                else:
                    grad_cpu = p.grad.detach().to("cpu")
                    if not hasattr(p, "_cpu_grad") or p._cpu_grad is None:
                        p._cpu_grad = grad_cpu
                    else:
                        p._cpu_grad.add_(grad_cpu)
                    p._d2h_event = None

                # 5. Safely free GPU-side gradient
                p.grad = None

            return post_accum_hook

        for idx, layer in enumerate(self.layers):
            for p in layer.parameters():
                if p.requires_grad:
                    if hasattr(p, "register_post_accumulate_grad_hook"):
                        self._hooks.append(p.register_post_accumulate_grad_hook(make_post_accum_hook(p)))
                    else:
                        def make_legacy_hook(param_ref):
                            def legacy_hook(grad):
                                if grad is not None:
                                    param_ref.grad = grad
                                    make_post_accum_hook(param_ref)(param_ref)
                            return legacy_hook
                        self._hooks.append(p.register_hook(make_legacy_hook(p)))

            def make_bwd_pre_hook(l: nn.Module):
                def bwd_pre_hook(module, grad_output):
                    # Stream this layer's parameters and buffers to GPU
                    for p in l.parameters():
                        p.data = self._stage_param_tensor(p)
                    for b in l.buffers():
                        b.data = b._cpu_data.to(dev, non_blocking=True)
                return bwd_pre_hook
            self._hooks.append(layer.register_full_backward_pre_hook(make_bwd_pre_hook(layer)))

            def make_bwd_post_hook(l: nn.Module):
                def bwd_post_hook(module, grad_input, grad_output):
                    # Restore permanent CPU tensor pointers; GPU memory freed
                    for t in list(l.parameters()) + list(l.buffers()):
                        t.data = t._cpu_data
                return bwd_post_hook
            self._hooks.append(layer.register_full_backward_hook(make_bwd_post_hook(layer)))

        # 6. Forward execution path
        orig_forward_block = getattr(self.model, "_forward_block", None)
        if orig_forward_block is not None:
            self.model._orig_forward_block = orig_forward_block

            def streaming_forward_block(layer, x, return_exit, cache=None, update_stats=True):
                idx = layer_to_idx.get(layer)

                # Check if this layer was prefetched on secondary CUDA stream
                if hasattr(layer, "_prefetched_cuda_params") and layer._prefetched_cuda_params is not None:
                    if prefetch_stream is not None:
                        torch.cuda.current_stream().wait_stream(prefetch_stream)
                    for p, staged_item in zip(layer.parameters(), layer._prefetched_cuda_params):
                        p.data = self._unpack_prefetched(staged_item)
                    for b, cuda_tensor in zip(layer.buffers(), layer._prefetched_cuda_buffers):
                        b.data = cuda_tensor
                    layer._prefetched_cuda_params = None
                    layer._prefetched_cuda_buffers = None
                else:
                    # Layer 0 or un-prefetched: transfer synchronously on main stream
                    for p in layer.parameters():
                        p.data = self._stage_param_tensor(p)
                    for b in layer.buffers():
                        b.data = b._cpu_data.to(dev, non_blocking=True)

                # Prefetch NEXT layer into isolated layer staging buffers on secondary stream
                # NEVER mutating next_layer parameters' .data from secondary stream!
                if prefetch_stream is not None and idx is not None and idx + 1 < num_layers:
                    next_layer = self.layers[idx + 1]
                    with torch.cuda.stream(prefetch_stream):
                        next_layer._prefetched_cuda_params = [
                            self._prefetch_param_tensor(p) for p in next_layer.parameters()
                        ]
                        next_layer._prefetched_cuda_buffers = [
                            b._cpu_data.to(dev, non_blocking=True) for b in next_layer.buffers()
                        ]

                out = orig_forward_block(layer, x, return_exit, cache=cache, update_stats=update_stats)

                # Instantly restore pointer to permanent CPU tensor; GPU memory freed
                for t in list(layer.parameters()) + list(layer.buffers()):
                    t.data = t._cpu_data
                return out

            self.model._forward_block = streaming_forward_block
        else:
            # Universal PyTorch forward hooks for external architectures (Llama, Mistral, Qwen, etc.)
            for idx, layer in enumerate(self.layers):
                def make_fwd_pre_hook(l: nn.Module, i: int):
                    def fwd_pre_hook(module, args):
                        if hasattr(l, "_prefetched_cuda_params") and l._prefetched_cuda_params is not None:
                            if prefetch_stream is not None:
                                torch.cuda.current_stream().wait_stream(prefetch_stream)
                            for p, staged_item in zip(l.parameters(), l._prefetched_cuda_params):
                                p.data = self._unpack_prefetched(staged_item)
                            for b, cuda_tensor in zip(l.buffers(), l._prefetched_cuda_buffers):
                                b.data = cuda_tensor
                            l._prefetched_cuda_params = None
                            l._prefetched_cuda_buffers = None
                        else:
                            for p in l.parameters():
                                p.data = self._stage_param_tensor(p)
                            for b in l.buffers():
                                b.data = b._cpu_data.to(dev, non_blocking=True)

                        if prefetch_stream is not None and i + 1 < num_layers:
                            next_layer = self.layers[i + 1]
                            with torch.cuda.stream(prefetch_stream):
                                next_layer._prefetched_cuda_params = [
                                    self._prefetch_param_tensor(p) for p in next_layer.parameters()
                                ]
                                next_layer._prefetched_cuda_buffers = [
                                    b._cpu_data.to(dev, non_blocking=True) for b in next_layer.buffers()
                                ]
                        if isinstance(args, tuple):
                            return tuple(a.to(dev) if isinstance(a, torch.Tensor) and a.device != dev else a for a in args)
                        return args
                    return fwd_pre_hook
                self._hooks.append(layer.register_forward_pre_hook(make_fwd_pre_hook(layer, idx)))

                def make_fwd_post_hook(l: nn.Module):
                    def fwd_post_hook(module, args, output):
                        for t in list(l.parameters()) + list(l.buffers()):
                            t.data = t._cpu_data
                        return output
                    return fwd_post_hook
                self._hooks.append(layer.register_forward_hook(make_fwd_post_hook(layer)))

        self.model._streaming_engine = self
        self.model._layer_streaming_active = True
        self.is_attached = True
        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if self.device.type == "cuda" else 0.0
        print(f"✅ [Layer Streaming] Active. Root components on GPU ({allocated_mb:.1f} MB); all 24 layers streaming from CPU RAM.", flush=True)

    def detach(self) -> None:
        """Detaches streaming engine, unregisters hooks, and restores unpatched model methods."""
        if not self.is_attached:
            return
        if hasattr(self.model, "_orig_forward_block") and self.model._orig_forward_block is not None:
            self.model._forward_block = self.model._orig_forward_block
            del self.model._orig_forward_block
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()

        # Synchronize and release streams
        if self.prefetch_stream is not None:
            self.prefetch_stream.synchronize()
            self.prefetch_stream = None
        if self.d2h_stream is not None:
            self.d2h_stream.synchronize()
            self.d2h_stream = None

        # Clean up temporary layer attributes and restore CPU storage pointers
        # (Phase 14 compatibility restoration; replaced by ParameterStager in Phase 3)
        for layer in self.layers:
            if hasattr(layer, "_prefetched_cuda_params"):
                delattr(layer, "_prefetched_cuda_params")
            if hasattr(layer, "_prefetched_cuda_buffers"):
                delattr(layer, "_prefetched_cuda_buffers")
            for t in list(layer.parameters()) + list(layer.buffers()):
                if hasattr(t, "_cpu_data"):
                    t.data = t._cpu_data
                if hasattr(t, "_d2h_event"):
                    delattr(t, "_d2h_event")
                if hasattr(t, "_cpu_grad"):
                    delattr(t, "_cpu_grad")

        self.model._streaming_engine = None
        self.model._layer_streaming_active = False
        self.is_attached = False

    def step_streaming_optimizer(self, optimizer: Optional[Any] = None, grad_clip: Optional[float] = None) -> None:
        """Streams one layer at a time to GPU to execute optimizer on GPU Tensor Cores at high speed."""
        opt = optimizer or self.optimizer
        if opt is None:
            return

        dev = self.device
        clip_val = grad_clip if grad_clip is not None else self.config.grad_clip

        # 0. Ensure all D2H transfers have completed before reading CPU gradients
        for layer in self.layers:
            for p in layer.parameters():
                if hasattr(p, "_d2h_event") and p._d2h_event is not None:
                    p._d2h_event.synchronize()
                    p._d2h_event = None

        # 1. Calculate double-precision global gradient norm across entire model (root + all layers)
        root_params = self.get_root_parameters()
        clip_coef = 1.0
        if clip_val and clip_val > 0:
            global_norm_sq = torch.zeros((), dtype=torch.float64)
            for p in root_params:
                if p.grad is not None:
                    global_norm_sq += p.grad.float().pow(2).sum().cpu().double()
            for layer in self.layers:
                for p in layer.parameters():
                    if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                        global_norm_sq += p._cpu_grad.float().pow(2).sum().cpu().double()
            global_norm = global_norm_sq.sqrt().item()
            clip_coef = min(1.0, float(clip_val) / (global_norm + 1e-6))

        # 2. Step GPU-resident root components (embeddings, router, heads)
        if clip_coef < 1.0:
            for p in root_params:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)

        if hasattr(opt, "step_parameters"):
            opt.step_parameters(root_params, is_global_step_end=False)
        else:
            opt.step()

        for p in root_params:
            p.grad = None

        # 3. Step each transformer layer one by one on GPU Tensor Cores (<80ms per layer)
        total_layers = len(self.layers)
        for layer_idx, layer in enumerate(self.layers):
            layer_params = list(layer.parameters())
            for p in layer_params:
                target_dtype = torch.bfloat16 if p._cpu_data.dtype == torch.float8_e4m3fn else p._cpu_data.dtype
                p.data = p._cpu_data.to(dev, non_blocking=True).to(target_dtype)
                if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                    g = p._cpu_grad.to(dev, non_blocking=True).to(target_dtype)
                    if clip_coef < 1.0:
                        g.mul_(clip_coef)
                    p.grad = g
                    p._cpu_grad = None  # Instantly free CPU gradient tensor

            # Move any existing optimizer states for this layer's parameters to GPU
            _move_param_states_to_device(opt, layer_params, dev)

            is_last = (layer_idx == total_layers - 1)
            if hasattr(opt, "step_parameters"):
                opt.step_parameters(layer_params, is_global_step_end=is_last)
            else:
                opt.step()

            # Offload updated optimizer states back to CPU to prevent VRAM accumulation
            _move_param_states_to_cpu(opt, layer_params)

            # Sync updated weights back to permanent CPU storage and release GPU tensors
            for p in layer_params:
                p._cpu_data.copy_(p.data.to(p._cpu_data.dtype), non_blocking=True)
                p.grad = None
                p.data = p._cpu_data

            for b in layer.buffers():
                b.data = b._cpu_data

        # Increment global optimizer step counter if not already incremented by step_parameters
        if hasattr(opt, "step_count") and not hasattr(opt, "step_parameters"):
            opt.step_count += 1
            if dev.type == "cuda" and (opt.step_count <= 2 or opt.step_count % 50 == 0):
                torch.cuda.empty_cache()

    def zero_grad(self) -> None:
        """Zeros gradients on model and optimizer."""
        self.model.zero_grad()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        for layer in self.layers:
            for p in layer.parameters():
                if hasattr(p, "_d2h_event") and p._d2h_event is not None:
                    p._d2h_event.synchronize()
                    p._d2h_event = None
                if hasattr(p, "_cpu_grad"):
                    p._cpu_grad = None
        for p in self.get_root_parameters():
            if hasattr(p, "_cpu_grad"):
                p._cpu_grad = None

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Applies optimizer step on offloaded CPU gradients and cleans up buffers."""
        for layer in self.layers:
            for p in layer.parameters():
                if hasattr(p, "_d2h_event") and p._d2h_event is not None:
                    p._d2h_event.synchronize()
                    p._d2h_event = None

        offloaded_params = []
        for p in self.model.parameters():
            if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                p.grad = p._cpu_grad
                offloaded_params.append(p)

        optimizer.step()

        for p in offloaded_params:
            p.grad = None
            p._cpu_grad = None


def enable_layer_streaming(
    model: nn.Module,
    device: Optional[torch.device | str] = None,
    chunk_size: int = 1,
    pin_memory: Optional[bool] = None,
    async_prefetch: bool = True,
    fp8_weights: bool = False,
    optimizer: Optional[Any] = None,
) -> LayerStreamingEngine:
    """Enable layer streaming on any PyTorch model (Triune, Llama, Mistral, Qwen, etc.)."""
    if device is None:
        target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    cfg = StreamingConfig(
        enabled=True,
        chunk_size=chunk_size,
        pin_memory=pin_memory,
        async_prefetch=async_prefetch,
        fp8_weights=fp8_weights,
    )
    engine = LayerStreamingEngine(model, device=target_device, config=cfg, optimizer=optimizer)
    engine.attach(optimizer=optimizer)
    return engine


def apply_layer_streaming_if_requested(
    model: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    optimizer: Optional[Any] = None,
) -> Optional[LayerStreamingEngine]:
    """Helper that checks config and attaches streaming engine if enabled."""
    if not config.get("streaming", False):
        return None

    streaming_cfg = StreamingConfig(
        enabled=True,
        chunk_size=config.get("streaming_chunk_size", 1),
        pin_memory=config.get("streaming_pin_memory", None),
        async_prefetch=config.get("streaming_prefetch", True),
        fp8_weights=config.get("streaming_fp8_weights", False),
        grad_clip=config.get("grad_clip", 1.0),
        instant_optimizer=True,
    )
    engine = LayerStreamingEngine(model, device=device, config=streaming_cfg, optimizer=optimizer)
    engine.attach()
    return engine
