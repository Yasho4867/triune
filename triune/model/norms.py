import torch
import torch.nn as nn

from .config import *
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        from triune.kernels import fast_rmsnorm
        return fast_rmsnorm(x, self.weight, self.eps)

# ─── Hybrid Attention (only GLA, no dead projections) ────────
