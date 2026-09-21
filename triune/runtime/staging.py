"""Parameter staging architecture for layer streaming.

Decouples CPU master parameter identity and optimizer state from temporary GPU
staging tensors, eliminating mutable parameter.data swapping during forward and
backward passes.
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

    def stage_to_gpu(
        self,
        device: torch.device,
        stream: Optional[torch.cuda.Stream] = None,
        target_dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Transfers CPU master tensor to GPU staging buffer."""
        dtype = target_dtype if target_dtype is not None else (
            torch.bfloat16 if self.cpu_master.dtype == torch.float8_e4m3fn else self.cpu_master.dtype
        )
        if stream is not None and device.type == "cuda":
            with torch.cuda.stream(stream):
                t = self.cpu_master.to(device, non_blocking=non_blocking).to(dtype)
                ev = torch.cuda.Event()
                ev.record(stream)
                self.event = ev
                self.gpu_tensor = t
        else:
            self.gpu_tensor = self.cpu_master.to(device, non_blocking=non_blocking).to(dtype)
            self.event = None
        return self.gpu_tensor

    def sync_to_cpu(self, non_blocking: bool = True) -> None:
        """Copies updated GPU tensor values back to permanent CPU master storage."""
        if self.gpu_tensor is not None:
            if self.event is not None:
                self.event.synchronize()
                self.event = None
            self.cpu_master.copy_(self.gpu_tensor.to(self.cpu_master.dtype), non_blocking=non_blocking)

    def release_gpu(self) -> None:
        """Releases the temporary GPU execution tensor and resets pointers to CPU master."""
        self.gpu_tensor = None
        self.event = None
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
    ) -> torch.Tensor:
        """Transfers CPU master to GPU as quantized FP8 tensor with scaling metadata."""
        if device.type != "cuda":
            self.gpu_tensor = self.cpu_master
            return self.gpu_tensor

        compute_dtype = target_dtype if target_dtype is not None else (
            torch.bfloat16 if self.cpu_master.dtype == torch.float8_e4m3fn else self.cpu_master.dtype
        )

        def _do_quantize():
            gpu_master = self.cpu_master.to(device, non_blocking=non_blocking).to(compute_dtype)
            amax = gpu_master.abs().amax().clamp_min(1e-12)
            scale = (448.0 / amax).to(dtype=compute_dtype)
            fp8_t = (gpu_master * scale).to(torch.float8_e4m3fn)
            inv_scale = (1.0 / scale.float())

            self.amax = amax
            self.scale = scale
            self.inv_scale = inv_scale
            self.fp8_tensor = fp8_t

            if use_hardware_fp8:
                self.gpu_tensor = fp8_t
            else:
                self.gpu_tensor = (fp8_t.to(compute_dtype) * inv_scale)
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
            if self.gpu_tensor.dtype == torch.float8_e4m3fn and self.inv_scale is not None:
                dequant = self.gpu_tensor.to(self.cpu_master.dtype) * self.inv_scale.to(self.cpu_master.dtype)
                self.cpu_master.copy_(dequant, non_blocking=non_blocking)
            else:
                self.cpu_master.copy_(self.gpu_tensor.to(self.cpu_master.dtype), non_blocking=non_blocking)

    def release_gpu(self) -> None:
        """Releases temporary GPU execution tensors and scale metadata, restoring CPU master."""
        self.gpu_tensor = None
        self.fp8_tensor = None
        self.scale = None
        self.inv_scale = None
        self.amax = None
        self.event = None
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

    def register_module(self, module: nn.Module) -> None:
        """Initializes staging buffers for all parameters and buffers in a module."""
        for p in module.parameters():
            p_id = id(p)
            if p_id not in self.buffers:
                master = p.data if not hasattr(p, "_cpu_data") else p._cpu_data
                p._cpu_data = master
                if self.use_fp8 and p.dim() >= 2:
                    self.buffers[p_id] = FP8StagingBuffer(param=p, cpu_master=master)
                else:
                    self.buffers[p_id] = StagingBuffer(param=p, cpu_master=master)

        for b in module.buffers():
            b_id = id(b)
            if b_id not in self.buffers:
                master = b.data if not hasattr(b, "_cpu_data") else b._cpu_data
                b._cpu_data = master
                self.buffers[b_id] = StagingBuffer(param=b, cpu_master=master)

    def stage_layer(
        self,
        layer: nn.Module,
        stream: Optional[torch.cuda.Stream] = None,
        target_dtype: Optional[torch.dtype] = None,
        use_hardware_fp8: bool = False,
    ) -> None:
        """Stages a layer's parameters and buffers to GPU."""
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
                    )
                else:
                    gpu_tensor = buf.stage_to_gpu(
                        device=self.device,
                        stream=stream,
                        target_dtype=target_dtype,
                        non_blocking=True,
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

    def sync_layer_to_cpu(self, layer: nn.Module) -> None:
        """Synchronizes updated GPU weights for a layer back to CPU master storage."""
        for p in list(layer.parameters()) + list(layer.buffers()):
            buf = self.buffers.get(id(p))
            if buf is not None:
                buf.sync_to_cpu()
                buf.release_gpu()
            if hasattr(p, "_fp8_tensor"):
                p._fp8_tensor = None
            if hasattr(p, "_fp8_scale"):
                p._fp8_scale = None
            if hasattr(p, "_fp8_inv_scale"):
                p._fp8_inv_scale = None
            if hasattr(p, "_fp8_amax"):
                p._fp8_amax = None

    def clear(self) -> None:
        """Releases all staging buffers and resets state."""
        for buf in self.buffers.values():
            buf.release_gpu()
            if hasattr(buf.param, "_fp8_tensor"):
                buf.param._fp8_tensor = None
            if hasattr(buf.param, "_fp8_scale"):
                buf.param._fp8_scale = None
            if hasattr(buf.param, "_fp8_inv_scale"):
                buf.param._fp8_inv_scale = None
            if hasattr(buf.param, "_fp8_amax"):
                buf.param._fp8_amax = None
        self.buffers.clear()
