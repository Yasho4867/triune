"""High-Performance Fused Upper-Triangular AdamW Kernel in Triton.

Performs simultaneous fused upper-triangular moment accumulation, exact FP32 bias
correction, and preconditioned gradient generation directly in GPU registers,
bypassing global memory round-trips and executing in a single CUDA kernel launch.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_triangular_adamw_kernel(
        R_ptr,
        M_ptr,
        V_ptr,
        Delta_ptr,
        Bias1_ptr,
        Bias2_ptr,
        SteerBias1_ptr,
        SteerBias2_ptr,
        beta1,
        beta2,
        eps,
        rms_reg,
        stride_rn,
        stride_rm,
        stride_rk,
        stride_mn,
        stride_mm,
        stride_mk,
        stride_vn,
        stride_vm,
        stride_vk,
        stride_dn,
        stride_dm,
        stride_dk,
        DR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        IS_TRIU: tl.constexpr,
    ):
        """Triton kernel for fused core AdamW (upper-triangular or full (r+1)x(r+1)) with in-register bias correction."""
        pid = tl.program_id(0)
        b_c1_sub = tl.load(Bias1_ptr + pid)
        b_c2_sub = tl.load(Bias2_ptr + pid)
        b_c1_str = tl.load(SteerBias1_ptr + pid)
        b_c2_str = tl.load(SteerBias2_ptr + pid)

        # Vectorized coordinate grids
        row_idx = tl.arange(0, BLOCK_SIZE)[:, None]
        col_idx = tl.arange(0, BLOCK_SIZE)[None, :]
        mask = (row_idx < DR) & (col_idx < DR)
        if IS_TRIU:
            triu_mask = mask & (row_idx <= col_idx)
        else:
            triu_mask = mask

        is_steer_border = triu_mask & ((row_idx == (DR - 1)) | (col_idx == (DR - 1)))
        b_c1 = tl.where(is_steer_border, b_c1_str, b_c1_sub)
        b_c2 = tl.where(is_steer_border, b_c2_str, b_c2_sub)

        # Memory pointer offsets
        r_ptrs = R_ptr + pid * stride_rn + row_idx * stride_rm + col_idx * stride_rk
        m_ptrs = M_ptr + pid * stride_mn + row_idx * stride_mm + col_idx * stride_mk
        v_ptrs = V_ptr + pid * stride_vn + row_idx * stride_vm + col_idx * stride_vk
        d_ptrs = Delta_ptr + pid * stride_dn + row_idx * stride_dm + col_idx * stride_dk

        # In-SRAM / register loads in FP32
        r = tl.load(r_ptrs, mask=triu_mask, other=0.0).to(tl.float32)
        m = tl.load(m_ptrs, mask=triu_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=triu_mask, other=0.0).to(tl.float32)

        # In-register Adam update
        m_new = beta1 * m + (1.0 - beta1) * r
        v_new = beta2 * v + (1.0 - beta2) * (r * r)

        tl.store(m_ptrs, m_new, mask=triu_mask)
        tl.store(v_ptrs, v_new, mask=triu_mask)

        # Exact bias-corrected AdamW update directly in registers
        m_hat = m_new / b_c1
        v_hat = v_new / b_c2
        if rms_reg > 0.0:
            border_cnt = tl.maximum(1.0, tl.sum(tl.where(is_steer_border, 1.0, 0.0)))
            v_border_mean = tl.sum(tl.where(is_steer_border, v_hat, 0.0)) / border_cnt
            v_hat = tl.where(is_steer_border, (1.0 - rms_reg) * v_hat + rms_reg * v_border_mean, v_hat)
        denom = tl.sqrt(v_hat) + eps
        delta = m_hat / denom

        tl.store(d_ptrs, delta, mask=triu_mask)


def fast_triangular_adamw(
    R_batch: torch.Tensor,
    batch_mom: torch.Tensor,
    batch_var: torch.Tensor,
    batch_delta: torch.Tensor,
    step_vals: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float = 1e-8,
    mask_mode: str = "triu",
    steer_step_vals: torch.Tensor | None = None,
    rms_reg: float = 0.0,
) -> torch.Tensor:
    """Executes fused AdamW update on batched (N, rank+1, rank+1) core matrices.

    Args:
        R_batch: Projected gradient core batch [N, DR, DR]
        batch_mom: Persistent first moments [N, DR, DR]
        batch_var: Persistent second moments [N, DR, DR]
        batch_delta: Pre-allocated output tensor for update direction [N, DR, DR]
        step_vals: 1D or 3D tensor of per-parameter subspace step counts [N]
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        eps: Small constant for numerical stability
        mask_mode: ``"triu"`` for upper-triangular masking or ``"full"`` for dense (DR, DR) core
        steer_step_vals: Optional independent per-parameter steering step counts [N] for column/row DR-1
        rms_reg: Optional RMS variance regularization weight in [0, 1] on steering border

    Returns:
        batch_delta updated in-place with preconditioned directions
    """
    if mask_mode not in ("triu", "full"):
        raise ValueError(f"Unknown mask_mode={mask_mode!r}; expected 'triu' or 'full'")

    N, DR, _ = R_batch.shape
    eff_steer_steps = step_vals if steer_step_vals is None else steer_step_vals
    if not HAS_TRITON or not R_batch.is_cuda:
        # High-performance PyTorch vectorized fallback in exact FP32
        R_f32 = R_batch if R_batch.dtype == torch.float32 else R_batch.float()
        batch_mom.lerp_(R_f32, 1.0 - beta1)
        batch_var.mul_(beta2).addcmul_(R_f32, R_f32, value=1.0 - beta2)
        step_view = step_vals.view(N, 1, 1).to(torch.float32)
        bias_c1_sub = 1.0 - (beta1**step_view)
        bias_c2_sub = (1.0 - (beta2**step_view)).clamp_min(1e-7)

        if steer_step_vals is not None:
            s_view = eff_steer_steps.view(N, 1, 1).to(torch.float32)
            bias_c1_str = 1.0 - (beta1**s_view)
            bias_c2_str = (1.0 - (beta2**s_view)).clamp_min(1e-7)
            b_c1 = bias_c1_sub.expand(N, DR, DR).clone()
            b_c2 = bias_c2_sub.expand(N, DR, DR).clone()
            b_c1[:, :, DR - 1] = bias_c1_str.view(N, 1)
            b_c1[:, DR - 1, :] = bias_c1_str.view(N, 1)
            b_c2[:, :, DR - 1] = bias_c2_str.view(N, 1)
            b_c2[:, DR - 1, :] = bias_c2_str.view(N, 1)
        else:
            b_c1 = bias_c1_sub
            b_c2 = bias_c2_sub

        m_hat = batch_mom / b_c1
        v_hat = batch_var / b_c2
        if rms_reg > 0.0 and DR >= 2:
            if mask_mode == "triu":
                col_r = v_hat[:, :, DR - 1]
                v_mean = col_r.mean(dim=1, keepdim=True)
                v_hat = v_hat.clone()
                v_hat[:, :, DR - 1] = (1.0 - rms_reg) * col_r + rms_reg * v_mean
            else:
                border_mask = torch.zeros((DR, DR), dtype=torch.bool, device=R_batch.device)
                border_mask[:, DR - 1] = True
                border_mask[DR - 1, :] = True
                v_mean = v_hat[:, border_mask].mean(dim=1, keepdim=True).unsqueeze(-1)
                v_hat = torch.where(border_mask.unsqueeze(0), (1.0 - rms_reg) * v_hat + rms_reg * v_mean, v_hat)

        denom = v_hat.sqrt().add_(eps)
        delta_f32 = m_hat.div(denom)
        if mask_mode == "triu":
            delta_f32 = delta_f32.triu_()
        batch_delta.copy_(delta_f32)
        return batch_delta

    # Calculate next power of 2 for Triton tile size
    block_size = triton.next_power_of_2(DR)
    step_view = step_vals.view(N).to(torch.float32)
    bias_c1 = (1.0 - (beta1**step_view)).to(torch.float32)
    bias_c2 = (1.0 - (beta2**step_view)).clamp_min(1e-7).to(torch.float32)
    if steer_step_vals is not None:
        s_view = eff_steer_steps.view(N).to(torch.float32)
        steer_bias_c1 = (1.0 - (beta1**s_view)).to(torch.float32)
        steer_bias_c2 = (1.0 - (beta2**s_view)).clamp_min(1e-7).to(torch.float32)
    else:
        steer_bias_c1 = bias_c1
        steer_bias_c2 = bias_c2

    grid = (N,)
    _fused_triangular_adamw_kernel[grid](
        R_batch,
        batch_mom,
        batch_var,
        batch_delta,
        bias_c1,
        bias_c2,
        steer_bias_c1,
        steer_bias_c2,
        beta1,
        beta2,
        eps,
        float(rms_reg),
        R_batch.stride(0),
        R_batch.stride(1),
        R_batch.stride(2),
        batch_mom.stride(0),
        batch_mom.stride(1),
        batch_mom.stride(2),
        batch_var.stride(0),
        batch_var.stride(1),
        batch_var.stride(2),
        batch_delta.stride(0),
        batch_delta.stride(1),
        batch_delta.stride(2),
        DR=DR,
        BLOCK_SIZE=block_size,
        IS_TRIU=(mask_mode == "triu"),
    )

    return batch_delta
