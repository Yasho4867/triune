"""Fused RMSNorm Operator."""

from __future__ import annotations

import torch
from triune.kernels.fast_rmsnorm import fast_rmsnorm


def norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Execute fused RMSNorm with register caching."""
    return fast_rmsnorm(x, weight, eps=eps)
