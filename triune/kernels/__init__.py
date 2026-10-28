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
from .fast_cross_entropy import chunked_cross_entropy_from_hidden, fast_cross_entropy
from .fast_gla import fast_chunk_gla
from .fast_lora import fast_lora_forward
from .fast_muon import fast_zeropower_via_newtonschulz5
from .fast_rmsnorm import FastRMSNorm, fast_rmsnorm
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
    "fast_chunk_gla",
    # Optimizers & Adapters
    "fast_zeropower_via_newtonschulz5",
    "fast_lora_forward",
]
