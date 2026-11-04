"""Muon Polar Orthogonalization Operator."""

from __future__ import annotations

import torch

from triune.kernels.fast_muon import fast_zeropower_via_newtonschulz5


def muon_orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Compute polar orthogonalization direction via iterative Newton-Schulz iterations."""
    return fast_zeropower_via_newtonschulz5(G, steps=steps, eps=eps)
