"""Triune High-Performance Kernel Acceleration Suite.

Co-optimizes speed and memory with fused Triton kernels on CUDA/Linux,
native CUDA C++ kernels, and zero-crash analytical in-place PyTorch autograd kernels.
Outperforms standard Unsloth baselines across speed, VRAM compression, and sequence length.
"""

from __future__ import annotations

from .dispatcher import (
    get_kernel_backend,
    get_kernel_telemetry,
    get_triton_error,
    is_triton_available,
    is_turbo_mode_enabled,
    set_turbo_mode,
)
from .fast_cross_entropy import FastChunkedCrossEntropyLoss, chunked_cross_entropy_from_hidden, fast_cross_entropy
from .fast_fp8 import FastFP8Linear, fast_fp8_linear, is_fp8_available
from .fast_gla import fast_chunk_gated_deltanet, fast_chunk_gla, fast_recurrent_gated_deltanet
from .fast_lora import fast_gate_up_lora_linear, fast_lora_forward, fast_lora_linear, fast_qkv_lora_linear
from .fast_muon import fast_zeropower_via_newtonschulz5
from .fast_nvfp4 import NVFP4Linear, nvfp4_dequantize, nvfp4_quantize
from .fast_rmsnorm import FastRMSNorm, fast_fused_add_rmsnorm, fast_rmsnorm
from .fast_rope import fast_rope
from .fast_swiglu import FastGELU, FastSwiGLU, fast_gelu, fast_swiglu

__all__ = [
    # Telemetry and Dispatch
    "is_triton_available",
    "get_kernel_backend",
    "get_triton_error",
    "is_turbo_mode_enabled",
    "set_turbo_mode",
    "get_kernel_telemetry",
    # Normalization & Embeddings
    "fast_rmsnorm",
    "fast_fused_add_rmsnorm",
    "FastRMSNorm",
    "fast_rope",
    # Activations
    "fast_swiglu",
    "fast_gelu",
    "FastSwiGLU",
    "FastGELU",
    # Loss & Attention
    "fast_cross_entropy",
    "chunked_cross_entropy_from_hidden",
    "FastChunkedCrossEntropyLoss",
    "fast_chunk_gla",
    "fast_chunk_gated_deltanet",
    "fast_recurrent_gated_deltanet",
    # Optimizers & Adapters
    "fast_zeropower_via_newtonschulz5",
    "fast_lora_forward",
    "fast_lora_linear",
    "fast_qkv_lora_linear",
    "fast_gate_up_lora_linear",
    # Blackwell FP8 & NVFP4
    "is_fp8_available",
    "fast_fp8_linear",
    "FastFP8Linear",
    "nvfp4_quantize",
    "nvfp4_dequantize",
    "NVFP4Linear",
]
