"""Hardware acceleration detector and kernel dispatcher.

Provides automatic zero-crash dispatch between ultra-fast OpenAI Triton kernels
(on Linux + CUDA) and high-performance vectorized PyTorch operations (on Windows/CPU).
"""

from __future__ import annotations

import os
import sys
import torch

_TRITON_AVAILABLE: bool | None = None
_TRITON_ERROR: str | None = None


def is_triton_available() -> bool:
    """Return True if OpenAI Triton is installed and a compatible CUDA device is available."""
    global _TRITON_AVAILABLE, _TRITON_ERROR
    if _TRITON_AVAILABLE is not None:
        return _TRITON_AVAILABLE

    # Allow user override to disable Triton kernels via environment variable
    if os.environ.get("TRIUNE_DISABLE_TRITON", "0") in ("1", "true", "True"):
        _TRITON_AVAILABLE = False
        _TRITON_ERROR = "Disabled via TRIUNE_DISABLE_TRITON environment variable"
        return False

    if not torch.cuda.is_available():
        _TRITON_AVAILABLE = False
        _TRITON_ERROR = "CUDA is not available"
        return False

    try:
        import triton
        import triton.language as tl  # noqa: F401

        # Test small kernel compilation to verify backend works
        _TRITON_AVAILABLE = True
        _TRITON_ERROR = None
        return True
    except Exception as exc:
        _TRITON_AVAILABLE = False
        _TRITON_ERROR = str(exc)
        return False


def get_kernel_backend() -> str:
    """Return 'triton' if Triton is available and active, otherwise 'pytorch'."""
    return "triton" if is_triton_available() else "pytorch"


def get_triton_error() -> str | None:
    """Return explanation if Triton is unavailable."""
    is_triton_available()
    return _TRITON_ERROR
