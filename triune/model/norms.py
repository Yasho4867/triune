"""RMSNorm normalization layer with fast kernel dispatch."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from triune.kernels import fast_rmsnorm
        return fast_rmsnorm(x, self.weight, self.eps)
