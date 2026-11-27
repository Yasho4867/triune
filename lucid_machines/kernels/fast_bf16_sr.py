"""Fused in-place Triton BF16 Stochastic Rounding parameter update kernel."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:

    @triton.jit
    def _bf16_sr_update_kernel(
        W_ptr,
        dW_ptr,
        n_elements,
        beta_wd,
        alpha_upd,
        seed,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        w = tl.load(W_ptr + offs, mask=mask).to(tl.float32)
        dw = tl.load(dW_ptr + offs, mask=mask).to(tl.float32)
        w_new = w * beta_wd + dw * alpha_upd
        rand16 = (tl.randint(seed, offs) & 0xFFFF).to(tl.int32)
        w_bits = w_new.to(tl.int32, bitcast=True)
        w_sr = ((w_bits + rand16) & (-65536)).to(tl.float32, bitcast=True).to(tl.bfloat16)
        tl.store(W_ptr + offs, w_sr, mask=mask)


def bf16_stochastic_round_update_(
    param: torch.Tensor,
    delta_w: torch.Tensor,
    beta_wd: float,
    alpha_upd: float,
    seed: int = 0,
) -> None:
    """Apply `param = SR_bf16(param * beta_wd + delta_w * alpha_upd)` in-place with zero FP32/INT32 buffers on CUDA."""
    if param.numel() == 0:
        return
    if (
        HAS_TRITON
        and param.is_cuda
        and param.dtype == torch.bfloat16
        and param.is_contiguous()
        and delta_w.is_contiguous()
        and param.numel() == delta_w.numel()
        and not (hasattr(torch, "compiler") and torch.compiler.is_compiling())
    ):
        n_elements = param.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _bf16_sr_update_kernel[grid](
            param,
            delta_w,
            n_elements,
            float(beta_wd),
            float(alpha_upd),
            int(seed) & 0x7FFFFFFF,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return

    with torch.no_grad():
        w_fp32 = param.data.float()
        if beta_wd != 1.0:
            w_fp32.mul_(float(beta_wd))
        w_fp32.add_(delta_w.float(), alpha=float(alpha_upd))
        if param.dtype == torch.bfloat16:
            rand_bits = torch.randint(0, 65536, w_fp32.shape, device=w_fp32.device, dtype=torch.int32)
            w_sr = ((w_fp32.view(torch.int32) + rand_bits) & -65536).view(torch.float32).to(dtype=torch.bfloat16)
            param.data.copy_(w_sr)
        else:
            param.data.copy_(w_fp32.to(dtype=param.dtype))
