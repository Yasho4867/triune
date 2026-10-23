"""Parameter staging architecture for layer streaming.

Formal Invariant:
Parameter identity (nn.Parameter Python object id and autograd graph node) never changes;
only execution storage is redirected under an explicit staging guard. Forward-time weight
snapshots are captured on staging so autograd backward hooks always restage the exact
forward-time weights, preventing CPU master mutations from corrupting backward gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional
import torch
import torch.nn as nn


@dataclass
class StagingBuffer:
    """Encapsulates a single parameter's permanent CPU master storage and its active GPU execution tensor."""

    param: nn.Parameter
    cpu_master: torch.Tensor
    gpu_tensor: Optional[torch.Tensor] = None
    event: Optional[torch.cuda.Event] = None
    forward_snapshot: Optional[torch.Tensor] = None

    def stage_to_gpu(
        self,
        device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
        target_dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
        is_backward: bool = False,
    ) -> torch.Tensor:
        """Transfers CPU master tensor (or forward-time snapshot during backward) to GPU staging buffer."""
        if is_backward and self.forward_snapshot is not None:
            source = self.forward_snapshot
        else:
            source = self.cpu_master
            if torch.is_grad_enabled() or self.forward_snapshot is None:
                self.forward_snapshot = self.cpu_master.detach().clone()

        dtype = target_dtype if target_dtype is not None else (
            torch.bfloat16 if source.dtype == torch.float8_e4m3fn else source.dtype
        )
        can_non_blocking = bool(non_blocking and getattr(source, "is_pinned", lambda: False)())
        if stream is not None and device.type == "cuda":
            with torch.cuda.stream(stream):
                t = source.to(device, non_blocking=can_non_blocking).to(dtype)
                if t.data_ptr() == source.data_ptr():
                    t = t.clone()
                ev = torch.cuda.Event()
                ev.record(stream)
                self.event = ev
                self.gpu_tensor = t
        else:
            t = source.to(device, non_blocking=can_non_blocking).to(dtype)
            if t.data_ptr() == source.data_ptr():
                t = t.clone()
            self.gpu_tensor = t
            self.event = None
        return self.gpu_tensor

    def sync_to_cpu(self, non_blocking: bool = True) -> None:
        """Copies updated GPU tensor values back to permanent CPU master storage."""
        if self.gpu_tensor is not None:
            if self.event is not None:
                self.event.synchronize()
                self.event = None
            self.cpu_master.copy_(self.gpu_tensor.to(self.cpu_master.dtype), non_blocking=non_blocking)
        self.forward_snapshot = None

    def release_gpu(self, clear_snapshot: bool = False) -> None:
        """Releases the temporary GPU execution tensor and resets pointers to CPU master."""
        self.gpu_tensor = None
        self.event = None
        if clear_snapshot:
            self.forward_snapshot = None
        self.param.data = self.cpu_master


@dataclass
class FP8StagingBuffer(StagingBuffer):
    """Encapsulates CPU master storage and FP8 GPU staging with self-contained scale, inv_scale, amax.

    CRITICAL INVARIANT: The CPU master parameter is NEVER overwritten or mutated into FP8.
    Optimizer updates strictly track the unquantized master weights in BF16 or FP32.
    """

    scale: Optional[torch.Tensor] = None
    inv_scale: Optional[torch.Tensor] = None
    amax: Optional[torch.Tensor] = None
    fp8_tensor: Optional[torch.Tensor] = None

    def stage_to_gpu(
        self,
        device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
        target_dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
        use_hardware_fp8: bool = False,
        is_backward: bool = False,
    ) -> torch.Tensor:
        """Transfers CPU master (or forward snapshot during backward) to GPU as quantized FP8 tensor with scaling metadata."""
        if is_backward and self.forward_snapshot is not None:
            source = self.forward_snapshot
        else:
            source = self.cpu_master
            if torch.is_grad_enabled() or self.forward_snapshot is None:
                self.forward_snapshot = self.cpu_master.detach().clone()

        compute_dtype = target_dtype if target_dtype is not None else (
            torch.bfloat16 if source.dtype == torch.float8_e4m3fn else source.dtype
        )

        can_non_blocking = bool(non_blocking and getattr(source, "is_pinned", lambda: False)())
        if device.type != "cuda":
            t = source.to(device, non_blocking=can_non_blocking).to(compute_dtype)
            if t.data_ptr() == source.data_ptr():
                t = t.clone()
            self.gpu_tensor = t
            return self.gpu_tensor

        def _do_quantize():
            gpu_master = source.to(device, non_blocking=can_non_blocking).to(compute_dtype)
            if gpu_master.data_ptr() == source.data_ptr():
                gpu_master = gpu_master.clone()
            if use_hardware_fp8:
                amax = gpu_master.abs().amax().clamp_min(1e-12)
                scale = (448.0 / amax).to(dtype=compute_dtype)
                fp8_t = (gpu_master * scale).to(torch.float8_e4m3fn)
                inv_scale = (1.0 / scale.float())

                self.amax = amax
                self.scale = scale
                self.inv_scale = inv_scale
                self.fp8_tensor = fp8_t
            else:
                self.amax = None
                self.scale = None
                self.inv_scale = None
                self.fp8_tensor = None

            # CRITICAL INVARIANT: self.gpu_tensor is ALWAYS the master precision representation
            # so autograd, optimizer, and non-FP8 consumers always see BF16/FP32.
            self.gpu_tensor = gpu_master
            return self.gpu_tensor

        if stream is not None:
            with torch.cuda.stream(stream):
                t = _do_quantize()
                ev = torch.cuda.Event()
                ev.record(stream)
                self.event = ev
        else:
            t = _do_quantize()
            self.event = None

        return t

    def sync_to_cpu(self, non_blocking: bool = True) -> None:
        """Copies updated GPU tensor values back to permanent CPU master storage."""
        if self.gpu_tensor is not None:
            if self.event is not None:
                self.event.synchronize()
                self.event = None
            # Master weight updates are in master precision (BF16/FP32)
            self.cpu_master.copy_(self.gpu_tensor.to(self.cpu_master.dtype), non_blocking=non_blocking)
        self.forward_snapshot = None

    def release_gpu(self, clear_snapshot: bool = False) -> None:
        """Releases temporary GPU execution tensors and scale metadata, restoring CPU master."""
        self.gpu_tensor = None
        self.fp8_tensor = None
        self.scale = None
        self.inv_scale = None
        self.amax = None
        self.event = None
        if clear_snapshot:
            self.forward_snapshot = None
        self.param.data = self.cpu_master


class ParameterStager:
    """Manages layer-level GPU staging and memory lifetime for layer-streaming execution."""

    def __init__(
        self,
        device: torch.device,
        prefetch_stream: Optional[torch.cuda.Stream] = None,
        d2h_stream: Optional[torch.cuda.Stream] = None,
        use_fp8: bool = False,
    ):
        self.device = device
        self.prefetch_stream = prefetch_stream
        self.d2h_stream = d2h_stream
        self.use_fp8 = use_fp8
        self.buffers: dict[int, StagingBuffer] = {}
        self._hooks: List[Any] = []

    def register_module(self, module: nn.Module, register_backward_hooks: bool = True) -> None:
        """Initializes staging buffers for all parameters and buffers in a module.

        FP8 staging is strictly opt-in per module: only parameters owned by modules
        explicitly marked with `_triune_fp8_aware = True` (such as FP8Linear) and
        having dim >= 2 receive an FP8StagingBuffer when self.use_fp8 is enabled.
        Ordinary nn.Linear, norms, embeddings, routers, and biases are NEVER marked
        FP8-aware and always receive standard StagingBuffer.
        """
        for sub_name, sub in module.named_modules():
            is_fp8_aware = getattr(sub, "_triune_fp8_aware", False)
            for p_name, p in sub.named_parameters(recurse=False):
                p_id = id(p)
                if p_id not in self.buffers:
                    master = p.data if not hasattr(p, "_cpu_data") else p._cpu_data
                    p._cpu_data = master
                    p._master_dtype = master.dtype
                    if self.use_fp8 and is_fp8_aware and p.dim() >= 2:
                        self.buffers[p_id] = FP8StagingBuffer(param=p, cpu_master=master)
                    else:
                        self.buffers[p_id] = StagingBuffer(param=p, cpu_master=master)

        # Fallback for any unmapped parameters/buffers
        for p in module.parameters():
            p_id = id(p)
            if p_id not in self.buffers:
                master = p.data if not hasattr(p, "_cpu_data") else p._cpu_data
                p._cpu_data = master
                p._master_dtype = master.dtype
                self.buffers[p_id] = StagingBuffer(param=p, cpu_master=master)

        for b in module.buffers():
            b_id = id(b)
            if b_id not in self.buffers:
                master = b.data if not hasattr(b, "_cpu_data") else b._cpu_data
                b._cpu_data = master
                self.buffers[b_id] = StagingBuffer(param=b, cpu_master=master)

        if register_backward_hooks and not getattr(module, "_stager_hooks_registered", False):
            def _fwd_pre_hook(mod: nn.Module, args: tuple):
                if torch.is_grad_enabled() and args and isinstance(args[0], torch.Tensor) and args[0].is_floating_point() and not args[0].requires_grad:
                    return (args[0].detach().requires_grad_(True),) + args[1:]
                return args

            def _bwd_pre_hook(mod: nn.Module, grad_output):
                mod._in_backward = True
                self.stage_layer(mod, stream=None, _from_backward=True)
                self.apply_staged_tensors(mod)

            def _bwd_post_hook(mod: nn.Module, grad_input, grad_output):
                mod._in_backward = False
                self.release_layer(mod)

            self._hooks.append(module.register_forward_pre_hook(_fwd_pre_hook))
            self._hooks.append(module.register_full_backward_pre_hook(_bwd_pre_hook))
            self._hooks.append(module.register_full_backward_hook(_bwd_post_hook))
            module._stager_hooks_registered = True

    def stage_layer(
        self,
        layer: nn.Module,
        stream: Optional[torch.cuda.Stream] = None,
        target_dtype: Optional[torch.dtype] = None,
        use_hardware_fp8: bool = False,
        _from_backward: bool = False,
    ) -> None:
        """Stages a layer's parameters and buffers to GPU."""
        is_backward = _from_backward or getattr(layer, "_in_backward", False)
        for p in list(layer.parameters()) + list(layer.buffers()):
            buf = self.buffers.get(id(p))
            if buf is not None:
                if isinstance(buf, FP8StagingBuffer):
                    gpu_tensor = buf.stage_to_gpu(
                        device=self.device,
                        stream=stream,
                        target_dtype=target_dtype,
                        non_blocking=True,
                        use_hardware_fp8=use_hardware_fp8,
                        is_backward=is_backward,
                    )
                else:
                    gpu_tensor = buf.stage_to_gpu(
                        device=self.device,
                        stream=stream,
                        target_dtype=target_dtype,
                        non_blocking=True,
                        is_backward=is_backward,
                    )
                if stream is None:
                    p.data = gpu_tensor
                    if isinstance(buf, FP8StagingBuffer):
                        p._fp8_tensor = buf.fp8_tensor
                        p._fp8_scale = buf.scale
                        p._fp8_inv_scale = buf.inv_scale
                        p._fp8_amax = buf.amax

    def apply_staged_tensors(self, layer: nn.Module) -> None:
        """Points layer parameters to staged GPU tensors after waiting on prefetch stream."""
        if self.prefetch_stream is not None and self.device.type == "cuda":
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        for p in list(layer.parameters()) + list(layer.buffers()):
            buf = self.buffers.get(id(p))
            if buf is not None and buf.gpu_tensor is not None:
                if buf.event is not None and self.device.type == "cuda":
                    torch.cuda.current_stream().wait_event(buf.event)
                    buf.event = None
                p.data = buf.gpu_tensor
                if isinstance(buf, FP8StagingBuffer):
                    p._fp8_tensor = buf.fp8_tensor
                    p._fp8_scale = buf.scale
                    p._fp8_inv_scale = buf.inv_scale
                    p._fp8_amax = buf.amax

    def release_layer(self, layer: nn.Module) -> None:
        """Releases all staged GPU tensors for a layer, restoring CPU pointers."""
        for p in list(layer.parameters()) + list(layer.buffers()):
            buf = self.buffers.get(id(p))
            if buf is not None:
                buf.release_gpu()
            if hasattr(p, "_fp8_tensor"):
                p._fp8_tensor = None
            if hasattr(p, "_fp8_scale"):
                p._fp8_scale = None
            if hasattr(p, "_fp8_inv_scale"):
                p._fp8_inv_scale = None
            if hasattr(p, "_fp8_amax"):
                p._fp8_amax = None
            if hasattr(p, "_cpu_data"):
                p.data = p._cpu_data

    def sync_layer_to_cpu(self, layer: nn.Module) -> None:
        """Synchronizes updated GPU weights for a layer back to CPU master storage."""
        for p in list(layer.parameters()) + list(layer.buffers()):
            buf = self.buffers.get(id(p))
            if buf is not None:
                buf.sync_to_cpu()
                buf.release_gpu(clear_snapshot=True)
            if hasattr(p, "_fp8_tensor"):
                p._fp8_tensor = None
            if hasattr(p, "_fp8_scale"):
                p._fp8_scale = None
            if hasattr(p, "_fp8_inv_scale"):
                p._fp8_inv_scale = None
            if hasattr(p, "_fp8_amax"):
                p._fp8_amax = None
            if hasattr(p, "_cpu_data"):
                p.data = p._cpu_data

    def cleanup_all(self) -> None:
        """Universal exception cleanup: releases all GPU tensors, restores CPU pointers,
        clears temporary FP8 metadata, and synchronizes/retires pending events."""
        for buf in self.buffers.values():
            buf.release_gpu(clear_snapshot=True)
            p = buf.param
            if hasattr(p, "_fp8_tensor"):
                p._fp8_tensor = None
            if hasattr(p, "_fp8_scale"):
                p._fp8_scale = None
            if hasattr(p, "_fp8_inv_scale"):
                p._fp8_inv_scale = None
            if hasattr(p, "_fp8_amax"):
                p._fp8_amax = None
            if hasattr(p, "_cpu_data"):
                p.data = p._cpu_data

    def clear(self) -> None:
        """Releases all staging buffers and resets state."""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        self.cleanup_all()
        self.buffers.clear()
