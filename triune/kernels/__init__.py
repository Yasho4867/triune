"""Triune High-Performance Kernel Acceleration Suite.

Co-optimizes speed and memory with fused Triton kernels on CUDA/Linux and
zero-crash vectorized PyTorch fallbacks on Windows/CPU.
"""

from __future__ import annotations

from .dispatcher import get_kernel_backend, is_triton_available
from .fast_cross_entropy import chunked_cross_entropy_from_hidden, fast_cross_entropy
from .fast_rmsnorm import FastRMSNorm, fast_rmsnorm
from .fast_rope import fast_rope

__all__ = [
    "is_triton_available",
    "get_kernel_backend",
    "fast_rmsnorm",
    "FastRMSNorm",
    "fast_rope",
    "fast_cross_entropy",
    "chunked_cross_entropy_from_hidden",
]
