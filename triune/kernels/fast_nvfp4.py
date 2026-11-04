"""NVFP4 (NVIDIA Blackwell FP4) Format Implementation.

Provides microscopic block-scaled E2M1 quantization, Triton dequantization,
and memory-saving linear layers for Blackwell (SM100+ / SM120) architectures.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .dispatcher import is_triton_available

NVFP4_GROUP_SIZE = 16
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def nvfp4_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantizes a 2D tensor [rows, cols] into NVFP4 format with group size 16 block scaling.

    Returns:
        packed: uint8 tensor of shape [rows, cols // 2]
        group_scale: bfloat16 tensor of shape [rows, cols // 16]
    """
    assert tensor.dim() == 2, "Tensor must be 2D"
    rows, cols = tensor.shape
    assert cols % NVFP4_GROUP_SIZE == 0, f"cols must be a multiple of {NVFP4_GROUP_SIZE}"

    device = tensor.device
    t_f32 = tensor.float()

    # Reshape into groups of 16
    groups = t_f32.view(rows, cols // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    group_max = groups.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    group_scale = group_max / 6.0  # [rows, cols // 16, 1]

    # Normalize values to [-6.0, 6.0]
    normalized = (groups / group_scale).clamp(-6.0, 6.0)

    # Nearest neighbor in E2M1
    lut = torch.tensor(_E2M1_VALUES, device=device, dtype=torch.float32)
    sign = (normalized < 0).to(torch.uint8)
    abs_norm = normalized.abs().unsqueeze(-1)
    diffs = (abs_norm - lut).abs()
    best_idx = diffs.argmin(dim=-1).to(torch.uint8)

    # Nibble layout: bit 3 is sign, bits 0-2 is index
    nibbles = (sign << 3) | (best_idx & 0x07)
    nibbles_flat = nibbles.view(rows, cols)

    # Pack 2 nibbles per uint8
    even_nibbles = nibbles_flat[:, 0::2]
    odd_nibbles = nibbles_flat[:, 1::2]
    packed = even_nibbles | (odd_nibbles << 4)

    return packed, group_scale.squeeze(-1).to(torch.bfloat16)


if is_triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _nvfp4_dequant_kernel(
        PACKED,
        SCALE,
        OUT,
        stride_p_row,
        stride_s_row,
        stride_o_row,
        ROWS,
        COLS,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= ROWS:
            return

        p_row = PACKED + row_idx * stride_p_row
        s_row = SCALE + row_idx * stride_s_row
        o_row = OUT + row_idx * stride_o_row

        n_groups = COLS // 16
        for g in range(0, n_groups):
            scale = tl.load(s_row + g).to(tl.float32)
            for i in range(8):
                b = tl.load(p_row + g * 8 + i).to(tl.int32)
                n0 = b & 0x0F
                n1 = (b >> 4) & 0x0F

                # Dequantize n0
                mag0 = n0 & 7
                sign0 = tl.where((n0 & 8) != 0, -1.0, 1.0)
                val0 = tl.where(
                    mag0 == 0,
                    0.0,
                    tl.where(
                        mag0 == 1,
                        0.5,
                        tl.where(
                            mag0 == 2,
                            1.0,
                            tl.where(
                                mag0 == 3,
                                1.5,
                                tl.where(mag0 == 4, 2.0, tl.where(mag0 == 5, 3.0, tl.where(mag0 == 6, 4.0, 6.0))),
                            ),
                        ),
                    ),
                )
                out0 = val0 * sign0 * scale

                # Dequantize n1
                mag1 = n1 & 7
                sign1 = tl.where((n1 & 8) != 0, -1.0, 1.0)
                val1 = tl.where(
                    mag1 == 0,
                    0.0,
                    tl.where(
                        mag1 == 1,
                        0.5,
                        tl.where(
                            mag1 == 2,
                            1.0,
                            tl.where(
                                mag1 == 3,
                                1.5,
                                tl.where(mag1 == 4, 2.0, tl.where(mag1 == 5, 3.0, tl.where(mag1 == 6, 4.0, 6.0))),
                            ),
                        ),
                    ),
                )
                out1 = val1 * sign1 * scale

                tl.store(o_row + g * 16 + i * 2, out0.to(OUT.dtype.element_ty))
                tl.store(o_row + g * 16 + i * 2 + 1, out1.to(OUT.dtype.element_ty))


def nvfp4_dequantize(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantizes NVFP4 packed weights with group size 16 scales back to BF16/FP32."""
    rows, half_cols = packed.shape
    cols = half_cols * 2

    if is_triton_available() and packed.is_cuda:
        out = torch.empty((rows, cols), device=packed.device, dtype=dtype)
        _nvfp4_dequant_kernel[(rows,)](
            packed,
            scale,
            out,
            packed.stride(0),
            scale.stride(0),
            out.stride(0),
            rows,
            cols,
        )
        return out

    # PyTorch Vectorized Fallback
    lut = torch.tensor(_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    low = (packed & 0x0F).to(torch.long)
    high = (packed >> 4).to(torch.long)

    # Interleave low and high nibbles
    nibbles = torch.empty((rows, cols), device=packed.device, dtype=torch.long)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high

    mag = nibbles & 7
    sign = torch.where((nibbles & 8) != 0, -1.0, 1.0)
    vals = lut[mag] * sign

    group_scale = scale.float().repeat_interleave(NVFP4_GROUP_SIZE, dim=1)[:, :cols]
    return (vals * group_scale).to(dtype)


class NVFP4Linear(nn.Module):
    """Linear layer storing weights in 4-bit NVFP4 block-scaled format.

    Achieves a ~3.2x physical VRAM reduction compared to BF16, with on-the-fly
    Triton hardware dequantization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Pre-allocate dummy packed weights
        packed_cols = in_features // 2
        scale_cols = in_features // NVFP4_GROUP_SIZE
        self.register_buffer("packed_weight", torch.zeros((out_features, packed_cols), dtype=torch.uint8))
        self.register_buffer("scale", torch.ones((out_features, scale_cols), dtype=torch.bfloat16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> NVFP4Linear:
        """Constructs an NVFP4Linear from an existing nn.Linear module."""
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
        )
        packed, scale = nvfp4_quantize(linear.weight.data)
        mod.packed_weight.copy_(packed)
        mod.scale.copy_(scale)
        if linear.bias is not None:
            mod.bias.data.copy_(linear.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = nvfp4_dequantize(self.packed_weight, self.scale, dtype=x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)
