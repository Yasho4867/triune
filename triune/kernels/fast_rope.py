"""Fused & In-Place Rotary Position Embedding (RoPE) kernel.

Eliminates slicing, transposition, and allocation overheads in standard PyTorch RoPE.
"""

from __future__ import annotations

import torch

from .dispatcher import is_triton_available, is_turbo_mode_enabled

if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_kernel(
        Q,
        K,
        Cos,
        Sin,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_cost,
        stride_cosd,
        stride_sint,
        stride_sind,
        H,
        T,
        D,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program processes one (batch, head, token) vector
        pid_bh = tl.program_id(0)
        t_idx = tl.program_id(1)

        b_idx = pid_bh // H
        h_idx = pid_bh % H

        half_d = D // 2
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < half_d

        # Pointers to first half and second half of D
        q1_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + t_idx * stride_qt + cols
        q2_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + t_idx * stride_qt + (cols + half_d)

        k1_ptr = K + b_idx * stride_kb + h_idx * stride_kh + t_idx * stride_kt + cols
        k2_ptr = K + b_idx * stride_kb + h_idx * stride_kh + t_idx * stride_kt + (cols + half_d)

        cos_ptr = Cos + t_idx * stride_cost + cols
        sin_ptr = Sin + t_idx * stride_sint + cols

        q1 = tl.load(q1_ptr, mask=mask, other=0.0).to(tl.float32)
        q2 = tl.load(q2_ptr, mask=mask, other=0.0).to(tl.float32)
        k1 = tl.load(k1_ptr, mask=mask, other=0.0).to(tl.float32)
        k2 = tl.load(k2_ptr, mask=mask, other=0.0).to(tl.float32)

        cos = tl.load(cos_ptr, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr, mask=mask, other=0.0).to(tl.float32)

        # In-place rotary formula:
        # q1_rot = q1 * cos - q2 * sin
        # q2_rot = q1 * sin + q2 * cos
        out_q1 = q1 * cos - q2 * sin
        out_q2 = q1 * sin + q2 * cos

        out_k1 = k1 * cos - k2 * sin
        out_k2 = k1 * sin + k2 * cos

        tl.store(q1_ptr, out_q1.to(Q.dtype.element_ty), mask=mask)
        tl.store(q2_ptr, out_q2.to(Q.dtype.element_ty), mask=mask)
        tl.store(k1_ptr, out_k1.to(K.dtype.element_ty), mask=mask)
        tl.store(k2_ptr, out_k2.to(K.dtype.element_ty), mask=mask)

    class _FastRoPEFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            B, H, T, D = q.shape
            half_d = D // 2
            cos_2d = cos.reshape(-1, cos.size(-1)).contiguous()
            sin_2d = sin.reshape(-1, sin.size(-1)).contiguous()
            out_q = q.clone()
            out_k = k.clone()
            BLOCK_SIZE = triton.next_power_of_2(half_d)
            _rope_kernel[(B * H, T)](
                out_q,
                out_k,
                cos_2d,
                sin_2d,
                out_q.stride(0),
                out_q.stride(1),
                out_q.stride(2),
                out_q.stride(3),
                out_k.stride(0),
                out_k.stride(1),
                out_k.stride(2),
                out_k.stride(3),
                cos_2d.stride(0),
                cos_2d.stride(1),
                sin_2d.stride(0),
                sin_2d.stride(1),
                H,
                T,
                D,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            ctx.save_for_backward(cos_2d, sin_2d)
            ctx.dims = (B, H, T, D)
            return out_q, out_k

        @staticmethod
        def backward(
            ctx, grad_q: torch.Tensor, grad_k: torch.Tensor
        ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
            cos_2d, sin_2d = ctx.saved_tensors
            B, H, T, D = ctx.dims
            half_d = D // 2
            out_gq = grad_q.clone() if grad_q is not None else None
            out_gk = grad_k.clone() if grad_k is not None else None
            BLOCK_SIZE = triton.next_power_of_2(half_d)
            if out_gq is not None and out_gk is not None:
                # The inverse of a 2D rotation R(theta) is R(-theta), which corresponds to -sin
                _rope_kernel[(B * H, T)](
                    out_gq,
                    out_gk,
                    cos_2d,
                    -sin_2d,
                    out_gq.stride(0),
                    out_gq.stride(1),
                    out_gq.stride(2),
                    out_gq.stride(3),
                    out_gk.stride(0),
                    out_gk.stride(1),
                    out_gk.stride(2),
                    out_gk.stride(3),
                    cos_2d.stride(0),
                    cos_2d.stride(1),
                    sin_2d.stride(0),
                    sin_2d.stride(1),
                    H,
                    T,
                    D,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            return out_gq, out_gk, None, None


def fast_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings using fused Triton or vectorized PyTorch.

    Args:
        q: [B, H, T, D] or [B, T, H, D]
        k: [B, H, T, D] or [B, T, H, D]
        cos: [T, D] or [1, 1, T, D]
        sin: [T, D] or [1, 1, T, D]
    """
    if is_turbo_mode_enabled() and is_triton_available() and q.is_cuda and q.dim() == 4 and k.shape == q.shape:
        try:
            B, H, T, D = q.shape
            half_d = D // 2
            cos_2d = cos.reshape(-1, cos.size(-1))
            sin_2d = sin.reshape(-1, sin.size(-1))
            if cos_2d.size(0) >= T and cos_2d.size(1) >= half_d:
                return _FastRoPEFunction.apply(q, k, cos, sin)
        except Exception:
            pass

    # Vectorized PyTorch Fast Execution Fallback
    cos = cos.to(dtype=q.dtype, device=q.device)
    sin = sin.to(dtype=q.dtype, device=q.device)

    # Ensure broadcastable shape to [1, 1, T, D] or [1, T, 1, D]
    while cos.dim() < q.dim():
        if cos.size(-2) == q.size(-2):
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

    # Use rotate_half with full-D cos/sin (cos/sin are [T, D] where D = full head_dim)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotary helper: [-x2, x1] from the two halves of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
