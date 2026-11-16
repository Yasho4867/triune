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
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .capabilities import PrecisionCapabilities
from .staging import FP8StagingBuffer, ParameterStager


@dataclass
class StreamingConfig:
    enabled: bool = False
    chunk_size: int = 1
    pin_memory: Optional[bool] = None  # None = auto-detect based on host RAM
    async_prefetch: bool = True
    fp8_weights: bool = False
    fp4_weights: bool = False
    grad_clip: float = 1.0
    instant_optimizer: bool = True
    layers_path: Optional[str] = None
    resident_gpu_layers: int = 0
    capture_forward_snapshots: bool = True  # Mode 1 (True: with snapshots) vs Mode 2 (False: deferred zero-snapshot)
    activation_checkpointing: bool = True  # Release SavedVariables & activations across streamed blocks


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
                for key_name in ("projection", "Q_L", "Q_R", "proj_basis", "last_c_orth"):
                    t_val = g.get(key_name)
                    if t_val is not None and isinstance(t_val, torch.Tensor) and t_val.device != device:
                        g[key_name] = t_val.to(device, non_blocking=True)


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
                for key_name in ("projection", "Q_L", "Q_R", "proj_basis", "last_c_orth"):
                    t_val = g.get(key_name)
                    if t_val is not None and isinstance(t_val, torch.Tensor) and t_val.device.type != "cpu":
                        g[key_name] = t_val.cpu()


def _step_optimizer_parameters(opt: Any, params: list[nn.Parameter], is_global_step_end: bool = False) -> None:
    """Safely executes optimizer step on a specific subset of parameters.

    Supports custom optimizers implementing step_parameters (CentroidSteerOptimizer)
    as well as standard PyTorch optimizers (AdamW, SGD, etc.) by temporarily isolating
    the target parameter subset in param_groups to avoid duplicate step updates.
    """
    if hasattr(opt, "step_parameters"):
        opt.step_parameters(params, is_global_step_end=is_global_step_end)
        return

    param_ids = {id(p) for p in params}
    orig_param_lists = []
    has_active_params = False
    for g in getattr(opt, "param_groups", []):
        orig_param_lists.append(g["params"])
        subset = [p for p in g["params"] if id(p) in param_ids]
        g["params"] = subset
        if subset:
            has_active_params = True

    try:
        if has_active_params:
            opt.step()
    finally:
        for g, orig in zip(getattr(opt, "param_groups", []), orig_param_lists):
            g["params"] = orig


class LayerStreamingEngine:
    """Manages layer-wise streaming, pinned memory DMA, and just-in-time GPU scheduling."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Optional[StreamingConfig] = None,
        optimizer: Optional[Any] = None,
        layers_path: Optional[str] = None,
    ):
        self.model = model
        self.device = device
        self.config = config or StreamingConfig(enabled=True)
        if layers_path is not None:
            self.config.layers_path = layers_path

        chunk_sz = int(getattr(self.config, "chunk_size", 1) or 1)
        if chunk_sz < 1:
            raise ValueError(f"StreamingConfig.chunk_size must be >= 1, got {chunk_sz}")
        self._chunk_size = chunk_sz

        self.optimizer = optimizer
        self.is_attached = False
        self._hooks: list[Any] = []
        self.prefetch_stream: Optional[torch.cuda.Stream] = None
        self.d2h_stream: Optional[torch.cuda.Stream] = None
        all_layers: list[nn.Module] = self._resolve_layers()
        raw_res = int(getattr(self.config, "resident_gpu_layers", 0) or 0)
        if self.device.type != "cuda" or not all_layers:
            res_count = 0
        elif raw_res < 0:
            res_count = self._compute_auto_resident_layers(all_layers)
        else:
            res_count = max(0, min(len(all_layers), raw_res))
        self.all_layers: list[nn.Module] = all_layers
        self.resident_layers: list[nn.Module] = all_layers[:res_count]
        self.layers: list[nn.Module] = all_layers[res_count:]
        self._layer_chunks: list[list[nn.Module]] = [
            self.layers[i : i + self._chunk_size] for i in range(0, len(self.layers), self._chunk_size)
        ]
        self.caps = PrecisionCapabilities.detect(self.device)
        self.stager: Optional[ParameterStager] = None

    def _compute_auto_resident_layers(self, all_layers: list[nn.Module]) -> int:
        """Heuristically keeps initial layers on GPU while leaving safe streaming + activation headroom."""
        if not all_layers or self.device.type != "cuda":
            return 0
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            return 0
        sample_bytes = sum(
            t.numel() * t.element_size() for t in list(all_layers[0].parameters()) + list(all_layers[0].buffers())
        )
        if sample_bytes <= 0:
            return 0
        # Reserve room for root embeddings/heads (~512MB), 2 active+prefetch chunks, and activations (~1.5GB)
        reserve = max(int(2.0 * (1024**3)), 2 * self._chunk_size * sample_bytes + int(1.0 * (1024**3)))
        avail = max(0, free_bytes - reserve)
        return max(0, min(len(all_layers), int(avail // max(1, sample_bytes))))

    def _chunk_for_layer_idx(self, idx: int) -> tuple[int, int, int]:
        """Returns (chunk_idx, start_layer_idx, end_layer_idx) for streamed layer index `idx`."""
        c_idx = idx // self._chunk_size
        start = c_idx * self._chunk_size
        end = min(len(self.layers), start + self._chunk_size) - 1
        return c_idx, start, end

    def _stage_chunk_forward(self, idx: int, use_hw_fp8: bool) -> None:
        c_idx, start, end = self._chunk_for_layer_idx(idx)
        chunk_layers = self._layer_chunks[c_idx]
        lead = chunk_layers[0]
        if getattr(lead, "_chunk_active", False):
            # Ensure target layer specifically holds its staged tensor (e.g. partial forward path)
            layer = self.layers[idx]
            has_staged = any(
                (self.stager.buffers.get(id(t)) is not None and self.stager.buffers[id(t)].gpu_tensor is not None)
                for t in list(layer.parameters()) + list(layer.buffers())
            )
            if has_staged or not list(layer.parameters()):
                return
        if getattr(lead, "_is_prefetched", False):
            if self.prefetch_stream is not None and self.device.type == "cuda":
                torch.cuda.current_stream().wait_stream(self.prefetch_stream)
            for cl in chunk_layers:
                self.stager.apply_staged_tensors(cl)
                cl._is_prefetched = False
                cl._chunk_active = True
        else:
            for cl in chunk_layers:
                self.stager.stage_layer(
                    cl,
                    stream=None,
                    target_dtype=None,
                    use_hardware_fp8=use_hw_fp8,
                    _from_backward=False,
                )
                self.stager.apply_staged_tensors(cl)
                cl._chunk_active = True

        # Prefetch the next chunk once when entering the first layer (or chunk) in forward
        next_c = c_idx + 1
        if self.prefetch_stream is not None and next_c < len(self._layer_chunks):
            next_chunk = self._layer_chunks[next_c]
            if not getattr(next_chunk[0], "_is_prefetched", False) and not getattr(
                next_chunk[0], "_chunk_active", False
            ):
                for nl in next_chunk:
                    self.stager.stage_layer(
                        nl,
                        stream=self.prefetch_stream,
                        target_dtype=None,
                        use_hardware_fp8=use_hw_fp8,
                        _from_backward=False,
                    )
                    nl._is_prefetched = True

    def _release_chunk_forward(self, idx: int) -> None:
        c_idx, _start, end = self._chunk_for_layer_idx(idx)
        if idx != end:
            return
        for cl in self._layer_chunks[c_idx]:
            self.stager.release_layer(cl)
            cl._chunk_active = False

    def _stage_chunk_backward(self, idx: int, use_hw_fp8: bool) -> None:
        c_idx, _start, _end = self._chunk_for_layer_idx(idx)
        chunk_layers = self._layer_chunks[c_idx]
        lead = chunk_layers[0]
        if getattr(self.layers[idx], "_is_bwd_staged", False):
            return
        if getattr(lead, "_is_bwd_prefetched", False):
            if self.prefetch_stream is not None and self.device.type == "cuda":
                torch.cuda.current_stream().wait_stream(self.prefetch_stream)
            for cl in chunk_layers:
                self.stager.apply_staged_tensors(cl)
                cl._is_bwd_prefetched = False
                cl._is_bwd_staged = True
        else:
            for cl in chunk_layers:
                if not getattr(cl, "_is_bwd_staged", False):
                    self.stager.stage_layer(
                        cl,
                        stream=None,
                        target_dtype=None,
                        use_hardware_fp8=use_hw_fp8,
                        _from_backward=True,
                    )
                    self.stager.apply_staged_tensors(cl)
                    cl._is_bwd_staged = True

        prev_c = c_idx - 1
        if self.prefetch_stream is not None and prev_c >= 0:
            prev_chunk = self._layer_chunks[prev_c]
            if not getattr(prev_chunk[0], "_is_bwd_prefetched", False) and not getattr(
                prev_chunk[0], "_is_bwd_staged", False
            ):
                for pl in prev_chunk:
                    self.stager.stage_layer(
                        pl,
                        stream=self.prefetch_stream,
                        target_dtype=None,
                        use_hardware_fp8=use_hw_fp8,
                        _from_backward=True,
                    )
                    pl._is_bwd_prefetched = True

    def _release_chunk_backward(self, idx: int) -> None:
        c_idx, start, _end = self._chunk_for_layer_idx(idx)
        layer = self.layers[idx]
        layer._in_backward = False
        layer._pending_bwd_count = max(0, getattr(layer, "_pending_bwd_count", 1) - 1)
        if idx != start:
            return
        for cl in self._layer_chunks[c_idx]:
            cl._is_bwd_staged = False
            cl._chunk_active = False
            self.stager.release_layer(cl, clear_snapshot=(getattr(cl, "_pending_bwd_count", 0) == 0))

    def _stage_param_tensor(self, p: nn.Parameter) -> torch.Tensor:
        """Adapts parameter staging through ParameterStager."""
        if self.stager is not None:
            buf = self.stager.buffers.get(id(p))
            if buf is not None:
                target_dtype = torch.bfloat16 if getattr(p, "_cpu_data", p.data).dtype == torch.float8_e4m3fn else None
                use_hw_fp8 = self.caps.native_scaled_mm and self.config.fp8_weights
                if isinstance(buf, FP8StagingBuffer):
                    return buf.stage_to_gpu(
                        self.device,
                        stream=None,
                        target_dtype=target_dtype,
                        non_blocking=True,
                        use_hardware_fp8=use_hw_fp8,
                    )
                return buf.stage_to_gpu(self.device, stream=None, target_dtype=target_dtype, non_blocking=True)
        return p.to(self.device)

    def _prefetch_param_tensor(self, p: nn.Parameter) -> Any:
        """Adapts parameter prefetching through ParameterStager."""
        if self.stager is not None:
            buf = self.stager.buffers.get(id(p))
            if buf is not None:
                target_dtype = torch.bfloat16 if getattr(p, "_cpu_data", p.data).dtype == torch.float8_e4m3fn else None
                use_hw_fp8 = self.caps.native_scaled_mm and self.config.fp8_weights
                if isinstance(buf, FP8StagingBuffer):
                    return buf.stage_to_gpu(
                        self.device,
                        stream=self.prefetch_stream,
                        target_dtype=target_dtype,
                        non_blocking=True,
                        use_hardware_fp8=use_hw_fp8,
                    )
                return buf.stage_to_gpu(
                    self.device, stream=self.prefetch_stream, target_dtype=target_dtype, non_blocking=True
                )
        return p.to(self.device)

    def _unpack_prefetched(self, staged_item: Any) -> torch.Tensor:
        """Unpacks staged items into appropriate tensor format."""
        if isinstance(staged_item, tuple):
            fp8_t, inv_scale, target_dtype = staged_item
            if self.caps.native_scaled_mm:
                return fp8_t
            else:
                return fp8_t.to(target_dtype) * inv_scale
        return staged_item

    def _resolve_layers(self) -> list[nn.Module]:
        """Dynamically detect sequential transformer blocks across diverse model architectures."""
        # 1. Explicit layers_path if configured
        if getattr(self.config, "layers_path", None):
            curr: Any = self.model
            try:
                for part in self.config.layers_path.split("."):
                    curr = getattr(curr, part)
            except AttributeError as e:
                raise ValueError(
                    f"Specified layers_path '{self.config.layers_path}' could not be resolved on model: {e}"
                ) from e
            if isinstance(curr, (nn.ModuleList, list, tuple)):
                return list(curr)
            raise ValueError(
                f"Specified layers_path '{self.config.layers_path}' does not resolve to ModuleList or sequence."
            )

        # 2. Standard transformer structural paths
        if hasattr(self.model, "layers") and isinstance(self.model.layers, (nn.ModuleList, list, tuple)):
            return list(self.model.layers)
        if (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "layers")
            and isinstance(self.model.model.layers, (nn.ModuleList, list, tuple))
        ):
            return list(self.model.model.layers)
        if (
            hasattr(self.model, "transformer")
            and hasattr(self.model.transformer, "h")
            and isinstance(self.model.transformer.h, (nn.ModuleList, list, tuple))
        ):
            return list(self.model.transformer.h)
        if (
            hasattr(self.model, "transformer")
            and hasattr(self.model.transformer, "layers")
            and isinstance(self.model.transformer.layers, (nn.ModuleList, list, tuple))
        ):
            return list(self.model.transformer.layers)

        # 3. Generic search across named submodules with strict ambiguity checking
        candidate_lists: list[tuple[str, nn.ModuleList]] = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.ModuleList) and len(m) > 0:
                candidate_lists.append((name, m))

        if not candidate_lists:
            return []

        max_len = max(len(m) for _, m in candidate_lists)
        top_candidates = [(name, m) for name, m in candidate_lists if len(m) == max_len]

        if len(top_candidates) > 1:
            names = [name for name, _ in top_candidates]
            raise ValueError(
                f"Ambiguous layer discovery: multiple candidate ModuleLists found with equal length {max_len} ({names}). "
                "Specify an explicit 'layers_path' in StreamingConfig to resolve ambiguity."
            )

        return list(top_candidates[0][1])

    def set_optimizer(self, optimizer: Any) -> None:
        """Sets the active optimizer to enable instant layer-wise optimization during backward."""
        self.optimizer = optimizer

    def get_root_parameters(self) -> List[nn.Parameter]:
        """Returns non-layer root parameters resident on GPU (embeddings, router, heads)."""
        all_layer_list = getattr(self, "all_layers", self.layers)
        layer_param_ids = {id(p) for layer in all_layer_list for p in layer.parameters()}
        return [p for p in self.model.parameters() if id(p) not in layer_param_ids]

    def attach(self, optimizer: Optional[Any] = None) -> None:
        """Attaches streaming hooks, configures CPU parameters, and initializes root components on GPU."""
        if self.is_attached or not getattr(self, "all_layers", self.layers):
            return
        if optimizer is not None:
            self.optimizer = optimizer

        print("[Layer Streaming] Initializing memory virtualization...", flush=True)

        # 1. Root non-layer components stay resident on GPU (~100MB)
        all_layer_list = getattr(self, "all_layers", self.layers)
        layer_ids = {id(l) for l in all_layer_list}
        for name, child in self.model.named_children():
            if id(child) not in layer_ids and not any(
                id(l) in {id(c) for c in child.modules()} for l in all_layer_list
            ):
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
        if hasattr(self.model, "token_embed") and hasattr(self.model, "final_head"):
            self.model.token_embed.weight = self.model.final_head.weight

        # 1b. Keep resident_layers permanently in GPU VRAM while offloading their gradients/optimizer state
        for layer in getattr(self, "resident_layers", []):
            layer.to(self.device)

        # 2. Configure streamed layers on CPU and establish permanent CPU storage pointers for both params & buffers
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

                avail_ram_gb = psutil.virtual_memory().available / (1024**3)
                total_param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
                model_gb = total_param_bytes / (1024**3)
                should_pin = avail_ram_gb >= model_gb * 2.0
            except Exception:
                should_pin = False

        self.is_pinned = should_pin
        if should_pin and self.layers:
            print("[Layer Streaming] Pinned host RAM enabled for high-throughput PCIe DMA.", flush=True)
            for layer in self.layers:
                for t in list(layer.parameters()) + list(layer.buffers()):
                    if not t._cpu_data.is_pinned():
                        try:
                            t._cpu_data = t._cpu_data.pin_memory()
                            t.data = t._cpu_data
                        except Exception:
                            pass
        elif self.layers:
            print("[Layer Streaming] Host RAM pinning bypassed to conserve host memory pages.", flush=True)

        # 4. Streams for asynchronous PCIe prefetching and D2H gradient transfer
        dev = self.device
        if dev.type == "cuda" and self.config.async_prefetch and self.layers:
            try:
                self.prefetch_stream = torch.cuda.Stream(device=dev)
                print("[Layer Streaming] Secondary CUDA prefetch stream active.", flush=True)
            except Exception:
                self.prefetch_stream = None

        if dev.type == "cuda":
            try:
                self.d2h_stream = torch.cuda.Stream(device=dev)
                print("[Layer Streaming] Dedicated CUDA D2H transfer stream active.", flush=True)
            except Exception:
                self.d2h_stream = None
        else:
            self.d2h_stream = None

        prefetch_stream = self.prefetch_stream
        d2h_stream = self.d2h_stream
        num_layers = len(self.layers)
        layer_to_idx = {layer: i for i, layer in enumerate(self.layers)}

        # 5. Initialize ParameterStager to manage GPU temporary staging and master weight isolation
        self.stager = ParameterStager(
            device=self.device,
            prefetch_stream=self.prefetch_stream,
            d2h_stream=self.d2h_stream,
            use_fp8=self.config.fp8_weights,
            use_fp4=getattr(self.config, "fp4_weights", False),
            capture_forward_snapshots=getattr(self.config, "capture_forward_snapshots", True),
        )
        for layer in self.layers:
            self.stager.register_module(layer, register_backward_hooks=False)

        # 6. Parameter autograd post-accumulate hooks and module backward hooks
        def make_post_accum_hook(param: nn.Parameter):
            def post_accum_hook(p: nn.Parameter) -> None:
                if p.grad is None:
                    return

                # Compute L2 norm squared on GPU while p.grad is hot in GPU L2 cache
                is_first_accum = not hasattr(p, "_cpu_grad") or p._cpu_grad is None
                if is_first_accum:
                    p._grad_norm_sq = (p.grad.detach().float() ** 2).sum()
                else:
                    p._grad_norm_sq = None

                use_async_d2h = bool(
                    dev.type == "cuda" and d2h_stream is not None and getattr(self, "is_pinned", False)
                )
                if use_async_d2h:
                    try:
                        gpu_grad = p.grad.detach()
                        gpu_grad.record_stream(d2h_stream)

                        compute_event = torch.cuda.Event()
                        compute_event.record(torch.cuda.current_stream())
                        d2h_stream.wait_event(compute_event)

                        with torch.cuda.stream(d2h_stream):
                            grad_cpu = gpu_grad.to("cpu", non_blocking=True)
                            d2h_event = torch.cuda.Event()
                            d2h_event.record(d2h_stream)
                            p._d2h_event = d2h_event

                        if is_first_accum:
                            p._cpu_grad = grad_cpu
                        else:
                            if not hasattr(p, "_pending_cpu_grads") or p._pending_cpu_grads is None:
                                p._pending_cpu_grads = []
                            p._pending_cpu_grads.append(grad_cpu)
                    except Exception:
                        use_async_d2h = False

                if not use_async_d2h:
                    grad_cpu = p.grad.detach().to("cpu", non_blocking=False)
                    if is_first_accum:
                        p._cpu_grad = grad_cpu
                    else:
                        p._cpu_grad.add_(grad_cpu)
                    p._d2h_event = None

                # Safely free GPU-side gradient
                p.grad = None

            return post_accum_hook

        # Register gradient offload hook on resident GPU layers so backward pass never accumulates full-model gradients in VRAM
        for layer in getattr(self, "resident_layers", []):
            for p in layer.parameters():
                if p.requires_grad:
                    if hasattr(p, "register_post_accumulate_grad_hook"):
                        self._hooks.append(p.register_post_accumulate_grad_hook(make_post_accum_hook(p)))
                    else:

                        def make_res_legacy_hook(param_ref):
                            def legacy_hook(grad):
                                if grad is not None:
                                    param_ref.grad = grad
                                    make_post_accum_hook(param_ref)(param_ref)

                            return legacy_hook

                        self._hooks.append(p.register_hook(make_res_legacy_hook(p)))

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

            def make_bwd_pre_hook(l: nn.Module, b_idx: int):
                def bwd_pre_hook(module, grad_output):
                    try:
                        l._in_backward = True
                        use_hw_fp8 = self.caps.native_scaled_mm and self.config.fp8_weights
                        self._stage_chunk_backward(b_idx, use_hw_fp8)
                    except Exception:
                        l._in_backward = False
                        l._is_bwd_staged = False
                        self.cleanup_resident_layers()
                        raise

                return bwd_pre_hook

            self._hooks.append(layer.register_full_backward_pre_hook(make_bwd_pre_hook(layer, idx)))

            def make_bwd_post_hook(b_idx: int):
                def bwd_post_hook(module, grad_input, grad_output):
                    self._release_chunk_backward(b_idx)

                return bwd_post_hook

            self._hooks.append(layer.register_full_backward_hook(make_bwd_post_hook(idx)))

        # 7. Forward execution path
        if self.layers:
            orig_forward_block = getattr(self.model, "_forward_block", None)
            if orig_forward_block is not None:
                self.model._orig_forward_block = orig_forward_block

                def streaming_forward_block(
                    layer,
                    x,
                    return_exit,
                    cache=None,
                    update_stats=True,
                    exit_decider=None,
                    exit_last_token_only=False,
                    active_mask=None,
                    **extra_kwargs,
                ):
                    idx = layer_to_idx.get(layer)
                    if idx is None:
                        return orig_forward_block(
                            layer,
                            x,
                            return_exit,
                            cache=cache,
                            update_stats=update_stats,
                            exit_decider=exit_decider,
                            exit_last_token_only=exit_last_token_only,
                            active_mask=active_mask,
                            **extra_kwargs,
                        )
                    in_bwd = getattr(layer, "_in_backward", False)
                    if torch.is_grad_enabled() and not in_bwd:
                        layer._pending_bwd_count = getattr(layer, "_pending_bwd_count", 0) + 1

                    if (
                        torch.is_grad_enabled()
                        and isinstance(x, torch.Tensor)
                        and x.is_floating_point()
                        and not x.requires_grad
                    ):
                        x = x.detach().requires_grad_(True)

                    use_hw_fp8 = self.caps.native_scaled_mm and self.config.fp8_weights
                    if in_bwd:
                        if not getattr(layer, "_is_bwd_staged", False):
                            self._stage_chunk_backward(idx, use_hw_fp8)
                    else:
                        self._stage_chunk_forward(idx, use_hw_fp8)

                    try:
                        out = orig_forward_block(
                            layer,
                            x,
                            return_exit,
                            cache=cache,
                            update_stats=update_stats,
                            exit_decider=exit_decider,
                            exit_last_token_only=exit_last_token_only,
                            active_mask=active_mask,
                            **extra_kwargs,
                        )
                    finally:
                        if not getattr(layer, "_in_backward", False):
                            self._release_chunk_forward(idx)
                    return out

                if getattr(self.config, "activation_checkpointing", True):
                    from triune.model.config import checkpoint_bwd_ctx, checkpoint_fwd_ctx, is_checkpoint_recomputing

                    streamed_layer_ids = {id(l) for l in self.layers}
                    raw_streaming_forward_block = streaming_forward_block

                    def checkpointed_streaming_forward_block(
                        layer,
                        x,
                        return_exit: bool = False,
                        cache=None,
                        update_stats: bool = True,
                        exit_decider=None,
                        exit_last_token_only: bool = False,
                        active_mask=None,
                        **extra_kwargs,
                    ):
                        if (
                            torch.is_grad_enabled()
                            and id(layer) in streamed_layer_ids
                            and cache is None
                            and exit_decider is None
                            and not getattr(layer, "_in_backward", False)
                            and not is_checkpoint_recomputing()
                        ):
                            x_in = (
                                x
                                if (not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.requires_grad)
                                else x.detach().requires_grad_(True)
                            )

                            def _ckpt_block(x_arg, mask_arg):
                                prev_inner_ckpt = getattr(layer, "_use_gradient_checkpointing", False)
                                if prev_inner_ckpt:
                                    layer._use_gradient_checkpointing = False
                                try:
                                    return raw_streaming_forward_block(
                                        layer,
                                        x_arg,
                                        return_exit=return_exit,
                                        cache=None,
                                        update_stats=(update_stats and not is_checkpoint_recomputing()),
                                        exit_decider=None,
                                        exit_last_token_only=exit_last_token_only,
                                        active_mask=mask_arg,
                                        **extra_kwargs,
                                    )
                                finally:
                                    if prev_inner_ckpt:
                                        layer._use_gradient_checkpointing = prev_inner_ckpt

                            return torch.utils.checkpoint.checkpoint(
                                _ckpt_block,
                                x_in,
                                active_mask,
                                use_reentrant=False,
                                preserve_rng_state=True,
                                context_fn=lambda: (checkpoint_fwd_ctx(), checkpoint_bwd_ctx()),
                            )
                        return raw_streaming_forward_block(
                            layer,
                            x,
                            return_exit=return_exit,
                            cache=cache,
                            update_stats=update_stats,
                            exit_decider=exit_decider,
                            exit_last_token_only=exit_last_token_only,
                            active_mask=active_mask,
                            **extra_kwargs,
                        )

                    self.model._forward_block = checkpointed_streaming_forward_block
                else:
                    self.model._forward_block = streaming_forward_block
            else:
                for idx, layer in enumerate(self.layers):

                    def make_fwd_pre_hook(l: nn.Module, i: int):
                        def fwd_pre_hook(module, args, kwargs):
                            in_bwd = getattr(l, "_in_backward", False)
                            if torch.is_grad_enabled() and not in_bwd:
                                l._pending_bwd_count = getattr(l, "_pending_bwd_count", 0) + 1
                            use_hw_fp8 = self.caps.native_scaled_mm and self.config.fp8_weights
                            if in_bwd:
                                if not getattr(l, "_is_bwd_staged", False):
                                    self._stage_chunk_backward(i, use_hw_fp8)
                            else:
                                self._stage_chunk_forward(i, use_hw_fp8)

                            new_args = (
                                tuple(a.to(dev) if isinstance(a, torch.Tensor) and a.device != dev else a for a in args)
                                if isinstance(args, tuple)
                                else args
                            )
                            if (
                                torch.is_grad_enabled()
                                and isinstance(new_args, tuple)
                                and new_args
                                and isinstance(new_args[0], torch.Tensor)
                                and new_args[0].is_floating_point()
                                and not new_args[0].requires_grad
                            ):
                                new_args = (new_args[0].detach().requires_grad_(True),) + new_args[1:]
                            new_kwargs = (
                                {
                                    k: v.to(dev) if isinstance(v, torch.Tensor) and v.device != dev else v
                                    for k, v in kwargs.items()
                                }
                                if kwargs
                                else kwargs
                            )
                            return new_args, new_kwargs

                        return fwd_pre_hook

                    self._hooks.append(layer.register_forward_pre_hook(make_fwd_pre_hook(layer, idx), with_kwargs=True))

                    def make_fwd_post_hook(l: nn.Module, i: int):
                        def fwd_post_hook(module, args, output):
                            if not getattr(l, "_in_backward", False):
                                self._release_chunk_forward(i)
                            return output

                        return fwd_post_hook

                    try:
                        self._hooks.append(
                            layer.register_forward_hook(make_fwd_post_hook(layer, idx), always_call=True)
                        )
                    except TypeError:
                        self._hooks.append(layer.register_forward_hook(make_fwd_post_hook(layer, idx)))

                    if getattr(self.config, "activation_checkpointing", True):
                        from triune.model.config import (
                            checkpoint_bwd_ctx,
                            checkpoint_fwd_ctx,
                            is_checkpoint_recomputing,
                        )

                        orig_layer_forward = layer.forward
                        layer._orig_forward_for_streaming = orig_layer_forward

                        def make_ckpt_layer_forward(l_ref: nn.Module, orig_fn):
                            def ckpt_forward(*f_args, **f_kwargs):
                                has_cache = bool(
                                    f_kwargs.get("use_cache", False)
                                    or f_kwargs.get("past_key_value") is not None
                                    or f_kwargs.get("past_key_values") is not None
                                )
                                first_is_tensor = len(f_args) > 0 and isinstance(f_args[0], torch.Tensor)
                                if (
                                    torch.is_grad_enabled()
                                    and not has_cache
                                    and first_is_tensor
                                    and not getattr(l_ref, "_in_backward", False)
                                    and not is_checkpoint_recomputing()
                                ):
                                    x0 = f_args[0]
                                    if x0.is_floating_point() and not x0.requires_grad:
                                        f_args = (x0.detach().requires_grad_(True),) + f_args[1:]

                                    def _ckpt_call(x_arg, *rest_args):
                                        try:
                                            return orig_fn(x_arg, *rest_args, **f_kwargs)
                                        except Exception:
                                            l_ref._in_backward = False
                                            l_ref._is_bwd_staged = False
                                            self.cleanup_resident_layers()
                                            raise

                                    return torch.utils.checkpoint.checkpoint(
                                        _ckpt_call,
                                        f_args[0],
                                        *f_args[1:],
                                        use_reentrant=False,
                                        preserve_rng_state=True,
                                        context_fn=lambda: (checkpoint_fwd_ctx(), checkpoint_bwd_ctx()),
                                    )
                                return orig_fn(*f_args, **f_kwargs)

                            return ckpt_forward

                        layer.forward = make_ckpt_layer_forward(layer, orig_layer_forward)

            def make_model_forward_hook():
                def model_forward_hook(module, args, output):
                    for l in self.layers:
                        if getattr(l, "_is_prefetched", False) or getattr(l, "_chunk_active", False):
                            self.stager.release_layer(l)
                            l._is_prefetched = False
                            l._chunk_active = False
                    return output

                return model_forward_hook

            try:
                self._hooks.append(self.model.register_forward_hook(make_model_forward_hook(), always_call=True))
            except TypeError:
                self._hooks.append(self.model.register_forward_hook(make_model_forward_hook()))

        self.model._streaming_engine = self
        self.model._layer_streaming_active = True
        self.is_attached = True
        if self.device.type == "cuda":
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        allocated_mb = torch.cuda.memory_allocated(self.device) / (1024**2) if self.device.type == "cuda" else 0.0
        res_n = len(getattr(self, "resident_layers", []))
        print(
            f"[Layer Streaming] Active. GPU Resident: {res_n} layers + root ({allocated_mb:.1f} MB VRAM); "
            f"CPU Streaming: {len(self.layers)} layers.",
            flush=True,
        )

    def cleanup_resident_layers(self) -> None:
        """Universal exception recovery: releases all resident/prefetched layers, restores
        CPU master pointers, clears temporary FP8 metadata, and synchronizes streams."""
        if self.stager is not None:
            self.stager.cleanup_all()
        for layer in self.layers:
            layer._pending_bwd_count = 0
            layer._is_bwd_staged = False
            layer._chunk_active = False
            if hasattr(layer, "_is_prefetched"):
                layer._is_prefetched = False
            if hasattr(layer, "_is_bwd_prefetched"):
                layer._is_bwd_prefetched = False
            for p in list(layer.parameters()) + list(layer.buffers()):
                if hasattr(p, "_cpu_data"):
                    p.data = p._cpu_data
                if hasattr(p, "_fp8_tensor"):
                    p._fp8_tensor = None
                if hasattr(p, "_fp8_scale"):
                    p._fp8_scale = None
                if hasattr(p, "_fp8_inv_scale"):
                    p._fp8_inv_scale = None
                if hasattr(p, "_fp8_amax"):
                    p._fp8_amax = None
                if hasattr(p, "_grad_norm_sq"):
                    p._grad_norm_sq = None
                if hasattr(p, "_d2h_event") and p._d2h_event is not None:
                    try:
                        p._d2h_event.synchronize()
                    except Exception:
                        pass
                    p._d2h_event = None
        if self.prefetch_stream is not None:
            try:
                self.prefetch_stream.synchronize()
            except Exception:
                pass
        if self.d2h_stream is not None:
            try:
                self.d2h_stream.synchronize()
            except Exception:
                pass

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

        if self.prefetch_stream is not None:
            self.prefetch_stream.synchronize()
            self.prefetch_stream = None
        if self.d2h_stream is not None:
            self.d2h_stream.synchronize()
            self.d2h_stream = None

        if self.stager is not None:
            self.stager.clear()
            self.stager = None
        if self.device.type == "cuda" and hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()

        for layer in getattr(self, "resident_layers", []):
            for t in list(layer.parameters()) + list(layer.buffers()):
                if hasattr(t, "_d2h_event"):
                    delattr(t, "_d2h_event")
                if hasattr(t, "_cpu_grad"):
                    delattr(t, "_cpu_grad")
                if hasattr(t, "_grad_norm_sq"):
                    delattr(t, "_grad_norm_sq")

        for layer in self.layers:
            if hasattr(layer, "_orig_forward_for_streaming"):
                layer.forward = layer._orig_forward_for_streaming
                delattr(layer, "_orig_forward_for_streaming")
            if hasattr(layer, "_is_bwd_staged"):
                delattr(layer, "_is_bwd_staged")
            if hasattr(layer, "_pending_bwd_count"):
                delattr(layer, "_pending_bwd_count")
            if hasattr(layer, "_chunk_active"):
                delattr(layer, "_chunk_active")
            if hasattr(layer, "_is_prefetched"):
                delattr(layer, "_is_prefetched")
            if hasattr(layer, "_is_bwd_prefetched"):
                delattr(layer, "_is_bwd_prefetched")
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
                if hasattr(t, "_grad_norm_sq"):
                    delattr(t, "_grad_norm_sq")

        self.model._streaming_engine = None
        self.model._layer_streaming_active = False
        self.is_attached = False

    def flush_d2h_transfers(self) -> None:
        if self.d2h_stream is not None and self.device.type == "cuda":
            self.d2h_stream.synchronize()
        all_layer_list = getattr(self, "all_layers", self.layers)
        for layer in all_layer_list:
            for p in layer.parameters():
                if hasattr(p, "_d2h_event"):
                    p._d2h_event = None
                if hasattr(p, "_pending_cpu_grads") and p._pending_cpu_grads:
                    for g_add in p._pending_cpu_grads:
                        if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                            p._cpu_grad.add_(g_add)
                        else:
                            p._cpu_grad = g_add
                    p._pending_cpu_grads.clear()

    def step_streaming_optimizer(self, optimizer: Optional[Any] = None, grad_clip: Optional[float] = None) -> None:
        opt = optimizer or self.optimizer
        if opt is None:
            return

        dev = self.device
        clip_val = grad_clip if grad_clip is not None else self.config.grad_clip
        resident_layers = getattr(self, "resident_layers", [])

        self.flush_d2h_transfers()
        for layer in self.layers:
            if getattr(layer, "_is_prefetched", False) or getattr(layer, "_is_bwd_prefetched", False):
                self.stager.release_layer(layer, clear_snapshot=True)
                layer._is_prefetched = False
                layer._is_bwd_prefetched = False

        # 1. Calculate double-precision global gradient norm across entire model (root + resident + streamed layers)
        root_params = self.get_root_parameters()
        clip_coef = 1.0
        if clip_val and clip_val > 0:
            global_norm_sq = torch.zeros((), dtype=torch.float64)
            gpu_norm_sqs = []
            for p in root_params:
                if p.grad is not None:
                    gpu_norm_sqs.append((p.grad.detach().float() ** 2).sum())
            for layer in resident_layers:
                for p in layer.parameters():
                    if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                        if getattr(p, "_grad_norm_sq", None) is not None:
                            gpu_norm_sqs.append(p._grad_norm_sq)
                        else:
                            global_norm_sq += p._cpu_grad.float().pow(2).sum().cpu().double()
            for layer in self.layers:
                for p in layer.parameters():
                    if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                        if getattr(p, "_grad_norm_sq", None) is not None:
                            gpu_norm_sqs.append(p._grad_norm_sq)
                        else:
                            global_norm_sq += p._cpu_grad.float().pow(2).sum().cpu().double()
            if gpu_norm_sqs:
                global_norm_sq += torch.stack([t.to(dev) for t in gpu_norm_sqs]).double().sum().cpu()
            global_norm = global_norm_sq.sqrt().item()
            clip_coef = min(1.0, float(clip_val) / (global_norm + 1e-6))

        # 2. Step GPU-resident root components (embeddings, router, heads) and offload root optimizer states
        if clip_coef < 1.0:
            for p in root_params:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)

        try:
            _move_param_states_to_device(opt, root_params, dev)
            try:
                _step_optimizer_parameters(
                    opt, root_params, is_global_step_end=(len(resident_layers) == 0 and len(self.layers) == 0)
                )
            finally:
                _move_param_states_to_cpu(opt, root_params)

            for p in root_params:
                p.grad = None

            # 2b. Step GPU-resident transformer layers one by one (gradients & optimizer states offloaded on CPU)
            total_res = len(resident_layers)
            total_stream = len(self.layers)
            for res_idx, layer in enumerate(resident_layers):
                layer_params = list(layer.parameters())
                for p in layer_params:
                    if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                        g = p._cpu_grad.to(dev, non_blocking=True).to(p.dtype)
                        if clip_coef < 1.0:
                            g.mul_(clip_coef)
                        p.grad = g
                        p._cpu_grad = None
                        p._grad_norm_sq = None
                _move_param_states_to_device(opt, layer_params, dev)
                try:
                    is_last = total_stream == 0 and res_idx == total_res - 1
                    _step_optimizer_parameters(opt, layer_params, is_global_step_end=is_last)
                finally:
                    _move_param_states_to_cpu(opt, layer_params)
                for p in layer_params:
                    p.grad = None

            # 3. Step each streamed transformer layer one by one on GPU Tensor Cores
            for layer_idx, layer in enumerate(self.layers):
                layer_params = list(layer.parameters())
                target_dtype = (
                    torch.bfloat16
                    if any(getattr(p, "_cpu_data", p.data).dtype == torch.float8_e4m3fn for p in layer_params)
                    else None
                )
                self.stager.stage_layer(layer, stream=None, target_dtype=target_dtype, use_hardware_fp8=False)
                self.stager.apply_staged_tensors(layer)

                for p in layer_params:
                    if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                        g = p._cpu_grad.to(dev, non_blocking=True).to(p.dtype)
                        if clip_coef < 1.0:
                            g.mul_(clip_coef)
                        p.grad = g
                        p._cpu_grad = None  # Instantly free CPU gradient tensor
                        p._grad_norm_sq = None

                # Move any existing optimizer states for this layer's parameters to GPU
                _move_param_states_to_device(opt, layer_params, dev)

                try:
                    is_last = layer_idx == total_stream - 1
                    _step_optimizer_parameters(opt, layer_params, is_global_step_end=is_last)
                finally:
                    # Offload updated optimizer states back to CPU to prevent VRAM accumulation
                    _move_param_states_to_cpu(opt, layer_params)

                # Clean p.grad and sync updated weights back to CPU master via ParameterStager
                for p in layer_params:
                    p.grad = None

                self.stager.sync_layer_to_cpu(layer)
        except Exception:
            self.cleanup_resident_layers()
            raise

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
        for layer in getattr(self, "resident_layers", []) + self.layers:
            for p in layer.parameters():
                if hasattr(p, "_d2h_event") and p._d2h_event is not None:
                    p._d2h_event.synchronize()
                    p._d2h_event = None
                if hasattr(p, "_cpu_grad"):
                    p._cpu_grad = None
                if hasattr(p, "_grad_norm_sq"):
                    p._grad_norm_sq = None
        for p in self.get_root_parameters():
            if hasattr(p, "_cpu_grad"):
                p._cpu_grad = None
            if hasattr(p, "_grad_norm_sq"):
                p._grad_norm_sq = None

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Applies optimizer step on offloaded CPU gradients and cleans up buffers."""
        self.flush_d2h_transfers()

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
    fp4_weights: bool = False,
    resident_gpu_layers: int = 0,
    capture_forward_snapshots: bool = True,
    optimizer: Optional[Any] = None,
    activation_checkpointing: bool = True,
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
        fp4_weights=fp4_weights,
        resident_gpu_layers=resident_gpu_layers,
        capture_forward_snapshots=capture_forward_snapshots,
        activation_checkpointing=activation_checkpointing,
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
        fp4_weights=config.get("streaming_fp4_weights", False),
        resident_gpu_layers=config.get("streaming_resident_gpu_layers", 0),
        grad_clip=config.get("grad_clip", 1.0),
        instant_optimizer=True,
        capture_forward_snapshots=config.get("streaming_capture_forward_snapshots", True),
        activation_checkpointing=config.get("streaming_activation_checkpointing", True),
    )
    engine = LayerStreamingEngine(model, device=device, config=streaming_cfg, optimizer=optimizer)
    engine.attach()
    return engine
