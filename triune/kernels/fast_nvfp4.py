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

if is_triton_available():
    import triton
    import triton.language as tl

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

        s0 = tl.where((low_nib & 0x08) != 0, -1.0, 1.0)
        m0 = low_nib & 0x07
        v0 = tl.where(
            m0 == 0,
            0.0,
            tl.where(
                m0 == 1,
                0.5,
                tl.where(
                    m0 == 2,
                    1.0,
                    tl.where(m0 == 3, 1.5, tl.where(m0 == 4, 2.0, tl.where(m0 == 5, 3.0, tl.where(m0 == 6, 4.0, 6.0)))),
                ),
            ),
        )

        s1 = tl.where((high_nib & 0x08) != 0, -1.0, 1.0)
        m1 = high_nib & 0x07
        v1 = tl.where(
            m1 == 0,
            0.0,
            tl.where(
                m1 == 1,
                0.5,
                tl.where(
                    m1 == 2,
                    1.0,
                    tl.where(m1 == 3, 1.5, tl.where(m1 == 4, 2.0, tl.where(m1 == 5, 3.0, tl.where(m1 == 6, 4.0, 6.0)))),
                ),
            ),
        )

        out0 = (s0 * v0 * scale).to(Out_ptr.dtype.element_ty)
        out1 = (s1 * v1 * scale).to(Out_ptr.dtype.element_ty)

        tl.store(Out_ptr + b_offs * 2, out0, mask=mask)
        tl.store(Out_ptr + b_offs * 2 + 1, out1, mask=mask)


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
            w_deq = nvfp4_dequantize(packed_weight, scale, dtype=x.dtype)
        elif weight_master.size(1) % NVFP4_GROUP_SIZE == 0:
            p, s = nvfp4_quantize(weight_master.data)
            w_deq = nvfp4_dequantize(p, s, dtype=x.dtype)
        else:
            w_deq = weight_master.to(dtype=x.dtype)
        return torch.nn.functional.linear(x, w_deq, bias.to(dtype=x.dtype) if bias is not None else None)

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
    0.5 B packed E2M1 + 0.125 B BF16 scale per 16 elements), with fast Triton
    hardware quantization and dequantization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

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
        return nvfp4_dequantize(self.packed_weight, self.scale, dtype=self.dtype)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> NVFP4Linear:
        """Constructs an NVFP4Linear from an existing nn.Linear module."""
        w = linear.weight.data if hasattr(linear, "weight") else linear.linear.weight.data
        b = getattr(linear, "bias", None)
        if b is None and hasattr(linear, "linear"):
            b = getattr(linear.linear, "bias", None)
        mod = cls(
            w.size(1),
            w.size(0),
            bias=b is not None,
            dtype=w.dtype,
        )
        packed, scale = nvfp4_quantize(w)
        mod.packed_weight = packed.clone()
        mod.scale = scale.clone()
        if b is not None:
            mod.bias = nn.Parameter(b.data.clone())
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = nvfp4_dequantize(self.packed_weight, self.scale, dtype=x.dtype)
        return torch.nn.functional.linear(x, w, self.bias.to(dtype=x.dtype) if self.bias is not None else None)
