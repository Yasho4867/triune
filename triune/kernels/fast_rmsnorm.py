"""Fused RMSNorm and Fused Residual + RMSNorm with Triton kernel and zero-crash PyTorch fallback.

Eliminates memory roundtrips to HBM and activation caching for autograd.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .dispatcher import is_triton_available, is_turbo_mode_enabled

# Try importing Triton language bindings
if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _rms_norm_fwd_kernel(
        X,
        Y,
        W,
        R,
        stride_x,
        stride_y,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x_ptrs = X + row * stride_x + cols
        y_ptrs = Y + row * stride_y + cols
        w_ptrs = W + cols

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N
        rsqrt = 1.0 / tl.sqrt(var + eps)

        # Store rsqrt for backward pass
        tl.store(R + row, rsqrt)

        y = x * rsqrt * w
        tl.store(y_ptrs, y.to(X.dtype.element_ty), mask=mask)

    @triton.jit
    def _rms_norm_bwd_kernel(
        dY,
        X,
        W,
        R,
        dX,
        stride_dy,
        stride_x,
        stride_dx,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        dy_ptrs = dY + row * stride_dy + cols
        x_ptrs = X + row * stride_x + cols
        dx_ptrs = dX + row * stride_dx + cols
        w_ptrs = W + cols

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
        rsqrt = tl.load(R + row).to(tl.float32)

        # dX = rsqrt * (dy * w - (x * rsqrt^2 / N) * sum(dy * w * x))
        dy_w = dy * w
        dy_w_x = tl.sum(dy_w * x, axis=0)
        dx = rsqrt * (dy_w - (x * (rsqrt * rsqrt) / N) * dy_w_x)

        tl.store(dx_ptrs, dx.to(X.dtype.element_ty), mask=mask)

    @triton.jit
    def _fused_add_rms_norm_fwd_kernel(
        X,
        R_in,
        Y,
        R_out,
        W,
        RSQRT,
        stride_x,
        stride_rin,
        stride_y,
        stride_rout,
        N,
        eps,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x_ptrs = X + row * stride_x + cols
        y_ptrs = Y + row * stride_y + cols
        rout_ptrs = R_out + row * stride_rout + cols
        w_ptrs = W + cols

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            rin_ptrs = R_in + row * stride_rin + cols
            r_in = tl.load(rin_ptrs, mask=mask, other=0.0).to(tl.float32)
            h = x + r_in
        else:
            h = x

        tl.store(rout_ptrs, h.to(R_out.dtype.element_ty), mask=mask)

        var = tl.sum(h * h, axis=0) / N
        rsqrt = 1.0 / tl.sqrt(var + eps)
        tl.store(RSQRT + row, rsqrt)

        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = h * rsqrt * w
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _fused_add_rms_norm_bwd_kernel(
        dY,
        dR_out,
        H,
        W,
        RSQRT,
        dX,
        dR_in,
        stride_dy,
        stride_drout,
        stride_h,
        stride_dx,
        stride_drin,
        N,
        HAS_DROUT: tl.constexpr,
        HAS_RESIDUAL_IN: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        dy_ptrs = dY + row * stride_dy + cols
        h_ptrs = H + row * stride_h + cols
        w_ptrs = W + cols
        dx_ptrs = dX + row * stride_dx + cols

        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        h = tl.load(h_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
        rsqrt = tl.load(RSQRT + row).to(tl.float32)

        dy_w = dy * w
        dy_w_h = tl.sum(dy_w * h, axis=0)
        dh = rsqrt * (dy_w - (h * (rsqrt * rsqrt) / N) * dy_w_h)

        if HAS_DROUT:
            drout_ptrs = dR_out + row * stride_drout + cols
            dr_out = tl.load(drout_ptrs, mask=mask, other=0.0).to(tl.float32)
            dh += dr_out

        tl.store(dx_ptrs, dh.to(dX.dtype.element_ty), mask=mask)
        if HAS_RESIDUAL_IN:
            drin_ptrs = dR_in + row * stride_drin + cols
            tl.store(drin_ptrs, dh.to(dR_in.dtype.element_ty), mask=mask)

    class _FastRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
            M, N = x_2d.shape

            y = torch.empty_like(x_2d)
            rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)

            BLOCK_SIZE = triton.next_power_of_2(N)
            _rms_norm_fwd_kernel[(M,)](
                x_2d,
                y,
                weight,
                rsqrt,
                x_2d.stride(0),
                y.stride(0),
                N,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            ctx.save_for_backward(x_2d, weight, rsqrt)
            ctx.orig_shape = orig_shape
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.N = N
            return y.reshape(orig_shape)

        @staticmethod
        def backward(ctx, dy: torch.Tensor):
            x_2d, weight, rsqrt = ctx.saved_tensors
            orig_shape = ctx.orig_shape
            dy_2d = dy.reshape(-1, orig_shape[-1]).contiguous()
            M, N = x_2d.shape

            dx = torch.empty_like(x_2d)
            _rms_norm_bwd_kernel[(M,)](
                dy_2d,
                x_2d,
                weight,
                rsqrt,
                dx,
                dy_2d.stride(0),
                x_2d.stride(0),
                dx.stride(0),
                N,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
            )

            # Weight gradient: sum over M of (dy * x * rsqrt)
            normed_x = x_2d * rsqrt.unsqueeze(1)
            dw = (dy_2d * normed_x).sum(dim=0).to(weight.dtype)
            return dx.reshape(orig_shape), dw, None

    class _FusedAddRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, residual: Optional[torch.Tensor], weight: torch.Tensor, eps: float = 1e-6):
            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
            M, N = x_2d.shape
            has_residual = residual is not None
            if has_residual:
                residual_2d = residual.reshape(-1, orig_shape[-1]).contiguous()
                stride_rin = residual_2d.stride(0)
            else:
                residual_2d = x_2d
                stride_rin = 0

            y = torch.empty_like(x_2d)
            r_out = torch.empty_like(x_2d)
            rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)

            BLOCK_SIZE = triton.next_power_of_2(N)
            _fused_add_rms_norm_fwd_kernel[(M,)](
                x_2d,
                residual_2d,
                y,
                r_out,
                weight,
                rsqrt,
                x_2d.stride(0),
                stride_rin,
                y.stride(0),
                r_out.stride(0),
                N,
                eps,
                HAS_RESIDUAL=has_residual,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            ctx.save_for_backward(r_out, weight, rsqrt)
            ctx.orig_shape = orig_shape
            ctx.has_residual = has_residual
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.N = N

            return y.reshape(orig_shape), r_out.reshape(orig_shape)

        @staticmethod
        def backward(ctx, dy: torch.Tensor, dr_out: Optional[torch.Tensor]):
            h_2d, weight, rsqrt = ctx.saved_tensors
            orig_shape = ctx.orig_shape
            dy_2d = dy.reshape(-1, orig_shape[-1]).contiguous()
            M, N = dy_2d.shape

            has_drout = dr_out is not None
            if has_drout:
                dr_out_2d = dr_out.reshape(-1, orig_shape[-1]).contiguous()
                stride_drout = dr_out_2d.stride(0)
            else:
                dr_out_2d = dy_2d
                stride_drout = 0

            dx = torch.empty_like(dy_2d)
            if ctx.has_residual:
                dr_in = torch.empty_like(dy_2d)
                stride_drin = dr_in.stride(0)
            else:
                dr_in = dx
                stride_drin = 0

            _fused_add_rms_norm_bwd_kernel[(M,)](
                dy_2d,
                dr_out_2d,
                h_2d,
                weight,
                rsqrt,
                dx,
                dr_in,
                dy_2d.stride(0),
                stride_drout,
                h_2d.stride(0),
                dx.stride(0),
                stride_drin,
                N,
                HAS_DROUT=has_drout,
                HAS_RESIDUAL_IN=ctx.has_residual,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
            )

            normed_h = h_2d * rsqrt.unsqueeze(1)
            dw = (dy_2d * normed_h).sum(dim=0).to(weight.dtype)

            d_residual = dr_in.reshape(orig_shape) if ctx.has_residual else None
            return dx.reshape(orig_shape), d_residual, dw, None


def fast_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dispatches to native C++ PyTorch 2.4+ kernel, fused Triton, or high-performance fallback."""
    w = weight.to(x.dtype) if weight.dtype != x.dtype else weight
    if hasattr(torch.nn.functional, "rms_norm"):
        return torch.nn.functional.rms_norm(x, (x.size(-1),), weight=w, eps=eps)
    if is_turbo_mode_enabled() and is_triton_available() and x.is_cuda:
        try:
            return _FastRMSNormFunction.apply(x, weight, eps)
        except Exception:
            pass

    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def fast_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuses residual addition (x + residual) and RMSNorm into a single SRAM pass.

    Returns:
        (normed_output, residual_output)
    """
    w = weight.to(x.dtype) if weight.dtype != x.dtype else weight
    if is_turbo_mode_enabled() and is_triton_available() and x.is_cuda:
        try:
            return _FusedAddRMSNormFunction.apply(x, residual, w, eps)
        except Exception:
            pass

    res_out = x + residual if residual is not None else x
    normed = fast_rmsnorm(res_out, w, eps=eps)
    return normed, res_out


class FastRMSNorm(nn.Module):
    """Drop-in high-performance replacement for RMSNorm with optional fused residual support."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return fast_fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return fast_rmsnorm(x, self.weight, self.eps)
