"""Blackwell Native FP8 Scaled Linear Engine.

Accelerates GEMMs via hardware 5th-Gen Tensor Cores using torch._scaled_mm
on NVIDIA Blackwell (SM100+ / SM120) and Hopper (SM90+) architectures.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_fp8_available(device: Optional[torch.device] = None) -> bool:
    """Checks whether hardware FP8 acceleration via scaled_mm is supported."""
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda:0")
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9 and hasattr(torch, "_scaled_mm")


class _FastFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, K = x_2d.shape
        N, _ = weight.shape

        # Dynamic per-tensor scaling for E4M3 (max representable = 448.0)
        max_x = x_2d.abs().max().clamp_min(1e-12)
        scale_x = (max_x / 448.0).float()
        inv_scale_x = (1.0 / scale_x).float()
        x_fp8 = (x_2d.float() * inv_scale_x).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        max_w = weight.abs().max().clamp_min(1e-12)
        scale_w = (max_w / 448.0).float()
        inv_scale_w = (1.0 / scale_w).float()
        w_fp8 = (weight.float() * inv_scale_w).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        # torch._scaled_mm computes A @ B where A is [M, K] and B is [K, N]
        w_fp8_t = w_fp8.t().contiguous()

        out_2d = torch._scaled_mm(
            x_fp8,
            w_fp8_t,
            scale_a=scale_x,
            scale_b=scale_w,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )

        if bias is not None:
            out_2d = out_2d + bias

        ctx.save_for_backward(x_fp8, w_fp8, scale_x, scale_w)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        ctx.x_dtype = x.dtype
        ctx.w_dtype = weight.dtype

        return out_2d.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_fp8, w_fp8, scale_x, scale_w = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        M, N = grad_2d.shape
        _, K = x_fp8.shape

        # Gradient w.r.t X: grad_output @ weight
        # Use E5M2 for gradients (wider dynamic range, max representable = 57344.0)
        max_g = grad_2d.abs().max().clamp_min(1e-12)
        scale_g = (max_g / 57344.0).float()
        inv_scale_g = (1.0 / scale_g).float()
        g_fp8 = (grad_2d.float() * inv_scale_g).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)

        grad_x = torch._scaled_mm(
            g_fp8,
            w_fp8.contiguous(),
            scale_a=scale_g,
            scale_b=scale_w,
            out_dtype=ctx.x_dtype,
            use_fast_accum=True,
        )

        # Gradient w.r.t W: grad_output.T @ x
        g_fp8_t = g_fp8.t().contiguous()
        grad_w = torch._scaled_mm(
            g_fp8_t,
            x_fp8.contiguous(),
            scale_a=scale_g,
            scale_b=scale_x,
            out_dtype=ctx.w_dtype,
            use_fast_accum=True,
        )

        grad_b = grad_2d.sum(dim=0).to(ctx.w_dtype) if ctx.has_bias else None
        return grad_x.reshape(orig_shape), grad_w, grad_b


def fast_fp8_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatches to hardware Blackwell/Hopper FP8 GEMM when supported, with zero-crash fallback."""
    w = weight.to(x.dtype) if weight.dtype != x.dtype else weight
    b = bias.to(x.dtype) if (bias is not None and bias.dtype != x.dtype) else bias
    if is_fp8_available(x.device) and x.is_cuda and x.numel() > 65536:
        try:
            return _FastFP8LinearFunction.apply(x, w, b)
        except Exception:
            pass
    return F.linear(x, w, b)


class FastFP8Linear(nn.Module):
    """Drop-in hardware-accelerated FP8 Linear layer for Blackwell architectures."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fast_fp8_linear(x, self.weight, self.bias)
