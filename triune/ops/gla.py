"""Custom Gated Linear Attention (GLA) Operator."""

from __future__ import annotations

import torch

from triune.kernels.fast_gla import fast_chunk_gated_deltanet, fast_chunk_gla


def gated_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute high-efficiency Gated DeltaNet with associative error-correcting memory updates.

    Reference: Songlin Yang et al., 'Gated Delta Networks: Improving Mamba2 with Delta Rule', ICLR 2025.
    """
    return fast_chunk_gated_deltanet(q, k, v, g, beta=beta, initial_state=initial_state, chunk_size=chunk_size)


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
    if q.shape[1] == 1 and initial_state is not None and q.device.type == "cpu" and q.dtype == torch.float32:
        from ._c_bridge import c_gla_recurrent_step_f32

        c_res = c_gla_recurrent_step_f32(
            q.squeeze(1).contiguous(),
            k.squeeze(1).contiguous(),
            v.squeeze(1).contiguous(),
            g.squeeze(1).contiguous(),
            initial_state,
        )
        if c_res is not None:
            out_step, next_state = c_res
            return out_step.unsqueeze(1), next_state

    return fast_chunk_gla(q, k, v, g, initial_state=initial_state, chunk_size=chunk_size)
