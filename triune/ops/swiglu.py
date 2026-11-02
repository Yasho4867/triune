"""Fused SwiGLU Activation Operator."""

from __future__ import annotations

import torch
from triune.kernels.fast_swiglu import fast_swiglu


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Execute fused SwiGLU activation in registers."""
    return fast_swiglu(gate, up)
