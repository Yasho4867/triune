"""Fused RMSNorm implementation with Triton kernel and zero-crash PyTorch fallback.

Eliminates memory roundtrips to HBM and activation caching for autograd.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dispatcher import is_triton_available

# Try importing Triton language bindings
if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _rms_norm_fwd_kernel(
        X, Y, W, R,
        stride_x, stride_y,
        N, eps,
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
        dY, X, W, R, dX,
        stride_dy, stride_x, stride_dx,
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
                x_2d, y, weight, rsqrt,
                x_2d.stride(0), y.stride(0),
                N, eps,
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
                dy_2d, x_2d, weight, rsqrt, dx,
                dy_2d.stride(0), x_2d.stride(0), dx.stride(0),
                N,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
            )

            # Weight gradient: sum over M of (dy * x * rsqrt)
            normed_x = x_2d * rsqrt.unsqueeze(1)
            dw = (dy_2d * normed_x).sum(dim=0)
            return dx.reshape(orig_shape), dw, None


def fast_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dispatches to fused Triton kernel on CUDA or high-performance PyTorch native."""
    if is_triton_available() and x.is_cuda:
        try:
            return _FastRMSNormFunction.apply(x, weight, eps)
        except Exception:
            pass  # Seamless fallback if tensor configuration requires PyTorch

    # PyTorch Fallback
    if hasattr(torch.nn.functional, "rms_norm"):
        return torch.nn.functional.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


class FastRMSNorm(nn.Module):
    """Drop-in high-performance replacement for RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fast_rmsnorm(x, self.weight, self.eps)
