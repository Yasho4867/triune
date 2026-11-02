"""Custom Gated Linear Attention (GLA) Operator."""

from __future__ import annotations

import torch
from triune.kernels.fast_gla import fast_chunk_gla


def gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute high-efficiency chunked Gated Linear Attention (GLA) with O(N) scaling.

    Args:
        q: Query tensor [B, T, H, HD]
        k: Key tensor [B, T, H, HD]
        v: Value tensor [B, T, H, HD]
        g: Log-decay gate tensor [B, T, H, HD]
        initial_state: Optional recurrent hidden state [B, H, HD, HD]
        chunk_size: Chunk size for blockwise GEMMs (default: 64)

    Returns:
        (output, final_state)
    """
    return fast_chunk_gla(
        q, k, v, g,
        initial_state=initial_state,
        chunk_size=chunk_size
    )
