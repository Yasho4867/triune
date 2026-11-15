"""NVFP4 (NVIDIA Blackwell FP4) Format Implementation.

Provides microscopic block-scaled E2M1 quantization, Triton dequantization,
and memory-saving linear layers for Blackwell (SM100+ / SM120) architectures.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .dispatcher import is_triton_available, is_turbo_mode_enabled

NVFP4_GROUP_SIZE = 16
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00)

_H16_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_hadamard16(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns cached orthogonal symmetric 16x16 Walsh-Hadamard matrix (H16 @ H16 == I)."""
    key = (device, dtype)
    h = _H16_CACHE.get(key)
    if h is None:
        base = torch.tensor([[1.0]], dtype=torch.float32)
        for _ in range(4):
            top = torch.cat([base, base], dim=1)
            bot = torch.cat([base, -base], dim=1)
            base = torch.cat([top, bot], dim=0)
        h = (base * 0.25).to(device=device, dtype=dtype).contiguous()
        _H16_CACHE[key] = h
    return h


def fwht_16(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal block-16 Walsh-Hadamard transform along the last dimension."""
    orig_shape = x.shape
    last_dim = orig_shape[-1]
    if last_dim % 16 != 0:
        raise ValueError(f"fwht_16 requires last dimension divisible by 16, got {last_dim}")
    h16 = _get_hadamard16(x.device, x.dtype)
    return (x.reshape(-1, 16) @ h16).reshape(orig_shape)


if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_e2m1(nib):
        """Bitwise branch-free E2M1 4-bit nibble -> float32 decode (exact LUT match)."""
        u = nib.to(tl.uint32)
        sign_bit = (u & 0x08) << 28
        mag = u & 0x07
        normal_bits = (((mag >> 1) + 126) << 23) | ((mag & 1) << 22)
        sub_bits = mag * 0x3F000000
        abs_bits = tl.where(mag <= 1, sub_bits, normal_bits)
        return (sign_bit | abs_bits).to(tl.float32, bitcast=True)

    @triton.jit
    def _nvfp4_quant_kernel(
        W_ptr,
        Packed_ptr,
        Scale_ptr,
        n_groups,
        BLOCK_GROUPS: tl.constexpr,
    ):
        """Fused parallel Triton kernel quantizing groups of 16 elements into packed NVFP4 E2M1."""
        pid = tl.program_id(0)
        g_offs = pid * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        g_mask = g_offs < n_groups

        idx_8 = tl.arange(0, 8)[None, :]
        w_even = tl.load(W_ptr + g_offs[:, None] * 16 + idx_8 * 2, mask=g_mask[:, None], other=0.0).to(tl.float32)
        w_odd = tl.load(W_ptr + g_offs[:, None] * 16 + idx_8 * 2 + 1, mask=g_mask[:, None], other=0.0).to(tl.float32)

        amax = tl.maximum(
            tl.max(tl.abs(w_even), axis=1),
            tl.max(tl.abs(w_odd), axis=1),
        )
        scale_bf16 = tl.maximum(amax / 6.0, 1e-8).to(Scale_ptr.dtype.element_ty)
        tl.store(Scale_ptr + g_offs, scale_bf16, mask=g_mask)

        inv_scale = (1.0 / scale_bf16.to(tl.float32))[:, None]
        ne = w_even * inv_scale
        no = w_odd * inv_scale
        ae = tl.abs(ne)
        ao = tl.abs(no)

        ce = (
            tl.where(ne < 0.0, 8, 0)
            + tl.where(ae > 0.25, 1, 0)
            + tl.where(ae > 0.75, 1, 0)
            + tl.where(ae > 1.25, 1, 0)
            + tl.where(ae > 1.75, 1, 0)
            + tl.where(ae > 2.50, 1, 0)
            + tl.where(ae > 3.50, 1, 0)
            + tl.where(ae > 5.00, 1, 0)
        )
        co = (
            tl.where(no < 0.0, 8, 0)
            + tl.where(ao > 0.25, 1, 0)
            + tl.where(ao > 0.75, 1, 0)
            + tl.where(ao > 1.25, 1, 0)
            + tl.where(ao > 1.75, 1, 0)
            + tl.where(ao > 2.50, 1, 0)
            + tl.where(ao > 3.50, 1, 0)
            + tl.where(ao > 5.00, 1, 0)
        )
        packed = ((co << 4) | ce).to(tl.uint8)
        tl.store(Packed_ptr + g_offs[:, None] * 8 + idx_8, packed, mask=g_mask[:, None])

    @triton.jit
    def _nvfp4_dequant_kernel(
        Packed_ptr,
        Scale_ptr,
        Out_ptr,
        n_bytes,
        BLOCK_BYTES: tl.constexpr,
    ):
        """Parallelized Triton NVFP4 E2M1 dequantization kernel (1024 packed bytes per CTA)."""
        pid = tl.program_id(0)
        b_offs = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        mask = b_offs < n_bytes

        byte_val = tl.load(Packed_ptr + b_offs, mask=mask, other=0).to(tl.int32)
        g_offs = b_offs // 8
        scale = tl.load(Scale_ptr + g_offs, mask=mask, other=0.0).to(tl.float32)

        low_nib = byte_val & 0x0F
        high_nib = (byte_val >> 4) & 0x0F

        s0 = _decode_e2m1(low_nib)
        s1 = _decode_e2m1(high_nib)

        out0 = (s0 * scale).to(Out_ptr.dtype.element_ty)
        out1 = (s1 * scale).to(Out_ptr.dtype.element_ty)

        tl.store(Out_ptr + b_offs * 2, out0, mask=mask)
        tl.store(Out_ptr + b_offs * 2 + 1, out1, mask=mask)

    @triton.jit
    def _nvfp4_fused_gemv_kernel(
        X_ptr,
        Packed_ptr,
        Scale_ptr,
        Bias_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_pn,
        stride_pk,
        stride_sn,
        stride_sk,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HALF_K: tl.constexpr,
    ):
        """Fused W4A16 dequant-dot kernel specialized for decode & small M (zero BF16 weight allocation)."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        if pid_m >= M:
            return

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        half_k_total = K // 2
        scale_k_total = K // 16

        x_row_ptr = X_ptr + pid_m * stride_xm
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_byte in range(0, half_k_total, HALF_K):
            j_offs = k_byte + tl.arange(0, HALF_K)
            j_mask = j_offs < half_k_total

            p_bytes = tl.load(
                Packed_ptr + n_offs[:, None] * stride_pn + j_offs[None, :] * stride_pk,
                mask=n_mask[:, None] & j_mask[None, :],
                other=0,
            ).to(tl.int32)
            g_offs = j_offs // 8
            scales = tl.load(
                Scale_ptr + n_offs[:, None] * stride_sn + g_offs[None, :] * stride_sk,
                mask=n_mask[:, None] & (g_offs[None, :] < scale_k_total),
                other=0.0,
            ).to(tl.float32)

            w0 = _decode_e2m1(p_bytes & 0x0F) * scales
            w1 = _decode_e2m1((p_bytes >> 4) & 0x0F) * scales

            x_even = tl.load(
                x_row_ptr + (j_offs * 2) * stride_xk,
                mask=(j_offs * 2) < K,
                other=0.0,
            ).to(tl.float32)
            x_odd = tl.load(
                x_row_ptr + (j_offs * 2 + 1) * stride_xk,
                mask=(j_offs * 2 + 1) < K,
                other=0.0,
            ).to(tl.float32)

            acc += tl.sum(w0 * x_even[None, :] + w1 * x_odd[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(Bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias

        tl.store(
            Out_ptr + pid_m * stride_om + n_offs * stride_on,
            acc.to(Out_ptr.dtype.element_ty),
            mask=n_mask,
        )

    @triton.jit
    def _nvfp4_fused_gemm_kernel(
        X_ptr,
        Packed_ptr,
        Scale_ptr,
        Bias_ptr,
        Out_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_pn,
        stride_pk,
        stride_sn,
        stride_sk,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        IS_FP32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HALF_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Fused W4A16 Tensor-Core GEMM decoding E2M1 nibbles in registers via even/odd K split."""
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offs < M
        n_mask = n_offs < N

        half_k_total = K // 2
        scale_k_total = K // 16
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_byte in range(0, half_k_total, HALF_K):
            j_offs = k_byte + tl.arange(0, HALF_K)
            j_mask = j_offs < half_k_total

            p_bytes = tl.load(
                Packed_ptr + n_offs[:, None] * stride_pn + j_offs[None, :] * stride_pk,
                mask=n_mask[:, None] & j_mask[None, :],
                other=0,
            ).to(tl.int32)
            g_offs = j_offs // 8
            scales = tl.load(
                Scale_ptr + n_offs[:, None] * stride_sn + g_offs[None, :] * stride_sk,
                mask=n_mask[:, None] & (g_offs[None, :] < scale_k_total),
                other=0.0,
            ).to(tl.float32)

            w0 = _decode_e2m1(p_bytes & 0x0F) * scales
            w1 = _decode_e2m1((p_bytes >> 4) & 0x0F) * scales

            x_even = tl.load(
                X_ptr + m_offs[:, None] * stride_xm + (j_offs[None, :] * 2) * stride_xk,
                mask=m_mask[:, None] & ((j_offs[None, :] * 2) < K),
                other=0.0,
            )
            x_odd = tl.load(
                X_ptr + m_offs[:, None] * stride_xm + (j_offs[None, :] * 2 + 1) * stride_xk,
                mask=m_mask[:, None] & ((j_offs[None, :] * 2 + 1) < K),
                other=0.0,
            )

            if IS_FP32:
                acc = tl.dot(x_even.to(tl.float32), tl.trans(w0), acc, input_precision="ieee")
                acc = tl.dot(x_odd.to(tl.float32), tl.trans(w1), acc, input_precision="ieee")
            else:
                acc = tl.dot(x_even.to(X_ptr.dtype.element_ty), tl.trans(w0.to(X_ptr.dtype.element_ty)), acc)
                acc = tl.dot(x_odd.to(X_ptr.dtype.element_ty), tl.trans(w1.to(X_ptr.dtype.element_ty)), acc)

        if HAS_BIAS:
            bias = tl.load(Bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        tl.store(
            Out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on,
            acc.to(Out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def nvfp4_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes a 2D tensor [rows, cols] into NVFP4 format with group size 16 block scaling.

    Returns:
        packed: uint8 tensor of shape [rows, cols // 2]
        group_scale: bfloat16 tensor of shape [rows, cols // 16]
    """
    assert tensor.dim() == 2, "Tensor must be 2D"
    rows, cols = tensor.shape
    assert cols % NVFP4_GROUP_SIZE == 0, f"cols must be a multiple of {NVFP4_GROUP_SIZE}"

    if is_turbo_mode_enabled() and is_triton_available() and tensor.is_cuda:
        t_contig = tensor.contiguous()
        n_groups = (rows * cols) // NVFP4_GROUP_SIZE
        packed = torch.empty((rows, cols // 2), device=tensor.device, dtype=torch.uint8)
        scale = torch.empty((rows, cols // NVFP4_GROUP_SIZE), device=tensor.device, dtype=torch.bfloat16)
        BLOCK_GROUPS = 64
        _nvfp4_quant_kernel[(triton.cdiv(n_groups, BLOCK_GROUPS),)](
            t_contig,
            packed,
            scale,
            n_groups,
            BLOCK_GROUPS=BLOCK_GROUPS,
            num_warps=4,
        )
        return packed, scale

    # Vectorized CPU/fallback path without 4D LUT expansion
    t_f32 = tensor.float()
    groups = t_f32.view(rows, cols // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    group_max = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    group_scale = (group_max / 6.0).to(torch.bfloat16)

    normalized = groups / group_scale.float()
    sign = (normalized < 0).to(torch.uint8)
    abs_norm = normalized.abs()
    best_idx = torch.zeros_like(sign)
    for thresh in _E2M1_MIDPOINTS:
        best_idx.add_((abs_norm > thresh).to(torch.uint8))

    nibbles_flat = ((sign << 3) | (best_idx & 0x07)).view(rows, cols)
    packed = nibbles_flat[:, 0::2] | (nibbles_flat[:, 1::2] << 4)
    return packed, group_scale.squeeze(-1)


def nvfp4_dequantize(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantizes NVFP4 packed weights with group size 16 scales back to BF16/FP32."""
    rows, half_cols = packed.shape
    cols = half_cols * 2

    if is_turbo_mode_enabled() and is_triton_available() and packed.is_cuda:
        p_contig = packed.contiguous()
        s_contig = scale.contiguous()
        out = torch.empty((rows, cols), device=packed.device, dtype=dtype)
        n_bytes = rows * half_cols
        BLOCK_BYTES = 1024
        _nvfp4_dequant_kernel[(triton.cdiv(n_bytes, BLOCK_BYTES),)](
            p_contig,
            s_contig,
            out,
            n_bytes,
            BLOCK_BYTES=BLOCK_BYTES,
            num_warps=4,
        )
        return out

    # PyTorch Vectorized Fallback
    lut = torch.tensor(_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    low = (packed & 0x0F).to(torch.long)
    high = (packed >> 4).to(torch.long)

    nibbles = torch.empty((rows, cols), device=packed.device, dtype=torch.long)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high

    mag = nibbles & 7
    sign = torch.where((nibbles & 8) != 0, -1.0, 1.0)
    vals = lut[mag] * sign

    group_scale = scale.float().repeat_interleave(NVFP4_GROUP_SIZE, dim=1)[:, :cols]
    return (vals * group_scale).to(dtype)


def nvfp4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    hadamard: bool = False,
) -> torch.Tensor:
    """Fused W4A16 linear projection over packed NVFP4 weights (without full HBM dequant allocation)."""
    x_in = fwht_16(x) if hadamard else x
    orig_shape = x_in.shape
    k = orig_shape[-1]
    n = packed_weight.size(0)
    x_2d = x_in.reshape(-1, k)
    m = x_2d.size(0)
    if m == 0:
        out = torch.empty((*orig_shape[:-1], n), device=x.device, dtype=x.dtype)
        return out

    if (
        is_turbo_mode_enabled()
        and is_triton_available()
        and x_2d.is_cuda
        and packed_weight.is_cuda
        and x_2d.dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        x_contig = x_2d.contiguous()
        p_contig = packed_weight.contiguous()
        s_contig = scale.contiguous()
        out_2d = torch.empty((m, n), device=x.device, dtype=x.dtype)
        has_bias = bias is not None
        bias_arg = bias.contiguous() if has_bias else out_2d

        if m <= 8:
            block_n = 8
            grid = (triton.cdiv(n, block_n), m)
            _nvfp4_fused_gemv_kernel[grid](
                x_contig,
                p_contig,
                s_contig,
                bias_arg,
                out_2d,
                m,
                n,
                k,
                x_contig.stride(0),
                x_contig.stride(1),
                p_contig.stride(0),
                p_contig.stride(1),
                s_contig.stride(0),
                s_contig.stride(1),
                out_2d.stride(0),
                out_2d.stride(1),
                HAS_BIAS=has_bias,
                BLOCK_N=block_n,
                HALF_K=64,
                num_warps=4,
            )
        else:
            block_m = 16 if m <= 32 else 64
            block_n = 64
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            _nvfp4_fused_gemm_kernel[grid](
                x_contig,
                p_contig,
                s_contig,
                bias_arg,
                out_2d,
                m,
                n,
                k,
                x_contig.stride(0),
                x_contig.stride(1),
                p_contig.stride(0),
                p_contig.stride(1),
                s_contig.stride(0),
                s_contig.stride(1),
                out_2d.stride(0),
                out_2d.stride(1),
                HAS_BIAS=has_bias,
                IS_FP32=(x.dtype == torch.float32),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                HALF_K=32,
                GROUP_M=8,
                num_warps=4,
                num_stages=2,
            )
        return out_2d.reshape(*orig_shape[:-1], n)

    w_deq = nvfp4_dequantize(packed_weight, scale, dtype=x.dtype)
    return torch.nn.functional.linear(x_in, w_deq, bias.to(dtype=x.dtype) if bias is not None else None)


class _NVFP4MatmulFn(torch.autograd.Function):
    """STE-backed NVFP4 block-scaled E2M1 forward pass with master-precision backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight_master: torch.Tensor,
        bias: torch.Tensor | None,
        packed_weight: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight_master, bias)
        if packed_weight is not None and scale is not None:
            return nvfp4_linear(x, packed_weight, scale, bias=bias, hadamard=False)
        elif weight_master.size(1) % NVFP4_GROUP_SIZE == 0:
            p, s = nvfp4_quantize(weight_master.data)
            return nvfp4_linear(x, p, s, bias=bias, hadamard=False)
        return torch.nn.functional.linear(
            x, weight_master.to(dtype=x.dtype), bias.to(dtype=x.dtype) if bias is not None else None
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight_master, bias = ctx.saved_tensors
        master_dtype = weight_master.dtype
        grad_out = grad_output.to(dtype=master_dtype, device=weight_master.device)
        grad_flat = grad_out.reshape(-1, grad_out.shape[-1])
        x_flat = x.to(dtype=master_dtype).reshape(-1, x.shape[-1])
        grad_x = (grad_flat @ weight_master.to(master_dtype)).reshape(x.shape).to(dtype=x.dtype)
        grad_weight = (grad_flat.t() @ x_flat).to(dtype=master_dtype)
        grad_bias = grad_flat.sum(dim=0).to(dtype=bias.dtype) if bias is not None else None
        return grad_x, grad_weight, grad_bias, None, None


class NVFP4Linear(nn.Module):
    """Linear layer storing weights in 4-bit NVFP4 block-scaled format.

    Achieves a ~3.2x physical VRAM reduction compared to BF16 (0.625 bytes/param:
    0.5 B packed E2M1 + 0.125 B BF16 scale per 16 elements), with fused W4A16
    Triton GEMM/GEMV execution and optional FWHT-16 Hadamard preconditioning.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        hadamard: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.hadamard = hadamard

        packed_cols = in_features // 2
        scale_cols = in_features // NVFP4_GROUP_SIZE
        self.register_buffer("packed_weight", torch.zeros((out_features, packed_cols), dtype=torch.uint8))
        self.register_buffer("scale", torch.ones((out_features, scale_cols), dtype=torch.bfloat16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        w = nvfp4_dequantize(self.packed_weight, self.scale, dtype=self.dtype)
        return fwht_16(w) if self.hadamard else w

    @classmethod
    def from_linear(cls, linear: nn.Linear, hadamard: bool = False) -> NVFP4Linear:
        """Constructs an NVFP4Linear from an existing nn.Linear (or FP4Linear) module."""
        w = linear.weight.data if hasattr(linear, "weight") else linear.linear.weight.data
        b = getattr(linear, "bias", None)
        if b is None and hasattr(linear, "linear"):
            b = getattr(linear.linear, "bias", None)
        mod = cls(
            w.size(1),
            w.size(0),
            bias=b is not None,
            dtype=w.dtype,
            hadamard=hadamard,
        )
        w_quant = fwht_16(w) if hadamard else w
        packed, scale = nvfp4_quantize(w_quant)
        mod.packed_weight = packed.clone()
        mod.scale = scale.clone()
        if b is not None:
            mod.bias = nn.Parameter(b.data.clone())
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nvfp4_linear(x, self.packed_weight, self.scale, bias=self.bias, hadamard=self.hadamard)
