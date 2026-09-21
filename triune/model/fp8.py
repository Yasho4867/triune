"""Native FP8 (E4M3) Hardware Scaled GEMM Linear Layer for Triune Transformer.

Uses torch.autograd.Function to enable FP8 forward + BF16 backward for training.
Falls back to standard F.linear when hardware FP8 is unavailable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from triune.runtime.capabilities import PrecisionCapabilities

# Check if hardware FP8 scaled_mm is available at import time
_HAS_SCALED_MM = hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn")



def _quantize_to_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize a BF16 tensor to float8_e4m3fn with per-tensor scaling.

    Returns (fp8_tensor, inverse_scale) where inverse_scale is used to dequantize.
    Avoids intermediate FP32 allocations to prevent VRAM spikes.
    """
    amax = tensor.abs().amax().clamp_min(1e-12)
    scale = (448.0 / amax).to(dtype=tensor.dtype)
    tensor_fp8 = (tensor * scale).to(torch.float8_e4m3fn)
    scale_inv = (1.0 / scale.float())
    return tensor_fp8, scale_inv


class _FP8MatmulFn(torch.autograd.Function):
    """Custom autograd function: FP8 forward pass, BF16 backward pass."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        # Save originals for backward (in BF16)
        ctx.save_for_backward(x, weight, bias)

        # Flatten input for matmul: [*, in] -> [M, in]
        orig_shape = x.shape
        flat_x = x.reshape(-1, x.shape[-1])

        # Quantize to FP8 (or consume pre-quantized FP8 representation from FP8StagingBuffer)
        x_fp8, scale_x = _quantize_to_fp8(flat_x)
        if hasattr(weight, "_fp8_tensor") and weight._fp8_tensor is not None:
            w_fp8 = weight._fp8_tensor
            scale_w = getattr(weight, "_fp8_inv_scale", None)
            if scale_w is None and hasattr(weight, "_fp8_scale") and weight._fp8_scale is not None:
                scale_w = 1.0 / weight._fp8_scale.float()
            if not isinstance(scale_w, torch.Tensor):
                scale_w = torch.tensor(scale_w, device=weight.device, dtype=torch.float32)
        else:
            w_fp8, scale_w = _quantize_to_fp8(weight)

        # Execute hardware FP8 GEMM via torch._scaled_mm
        res = torch._scaled_mm(
            x_fp8,
            w_fp8.t().contiguous(),
            scale_a=scale_x,
            scale_b=scale_w,
            out_dtype=x.dtype,
        )
        out = res[0] if isinstance(res, (tuple, list)) else res

        if bias is not None:
            out = out + bias

        return out.reshape(*orig_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass runs in weight dtype (BF16) for gradient stability
        x, weight, bias = ctx.saved_tensors
        grad_out = grad_output.to(dtype=weight.dtype, device=weight.device)
        grad_output_flat = grad_out.reshape(-1, grad_out.shape[-1])
        x_flat = x.to(dtype=weight.dtype).reshape(-1, x.shape[-1])

        # If weight is in FP8 format on GPU, dequantize for backward reduction
        w = weight
        if w.dtype == torch.float8_e4m3fn:
            inv = getattr(w, "_fp8_inv_scale", None)
            if inv is None and hasattr(w, "_fp8_scale") and w._fp8_scale is not None:
                inv = 1.0 / w._fp8_scale.float()
            w = w.to(grad_out.dtype) if inv is None else (w.to(grad_out.dtype) * inv.to(grad_out.dtype))

        grad_x = grad_output_flat @ w  # [M, out] @ [out, in] = [M, in]
        grad_weight = grad_output_flat.t() @ x_flat  # [out, M] @ [M, in] = [out, in]
        grad_bias = grad_output_flat.sum(dim=0) if bias is not None else None

        grad_x = grad_x.reshape(x.shape).to(dtype=x.dtype)
        return grad_x, grad_weight, grad_bias


class FP8Linear(nn.Module):
    """Hardware-accelerated FP8 Linear layer.
    
    - Weights stored in BF16 (for gradient updates and checkpoint compatibility)
    - Forward pass executes matrix multiply in FP8 E4M3 on Tensor Cores
    - Backward pass runs in BF16 for numerical stability
    - Falls back to standard F.linear when FP8 hardware is unavailable
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_SCALED_MM and x.is_cuda:
            caps = PrecisionCapabilities.detect(x.device)
            if caps.native_scaled_mm:
                try:
                    return _FP8MatmulFn.apply(x, self.weight, self.bias)
                except (RuntimeError, NotImplementedError):
                    pass
        # Fallback path: ensure weight is in compute dtype (dequantizing if needed)
        w = self.weight
        if w.dtype == torch.float8_e4m3fn:
            inv = getattr(w, "_fp8_inv_scale", None)
            if inv is None and hasattr(w, "_fp8_scale") and w._fp8_scale is not None:
                inv = 1.0 / w._fp8_scale.float()
            w = w.to(x.dtype) if inv is None else (w.to(x.dtype) * inv.to(x.dtype))
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, fp8={_HAS_SCALED_MM}"


def convert_to_fp8(module: nn.Module, target_classes=(nn.Linear,)) -> nn.Module:
    """Recursively replaces target linear layers with FP8Linear."""
    for name, child in list(module.named_children()):
        if isinstance(child, target_classes) and not isinstance(child, FP8Linear):
            has_bias = child.bias is not None
            fp8_layer = FP8Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=has_bias,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                fp8_layer.weight.copy_(child.weight)
                if has_bias:
                    fp8_layer.bias.copy_(child.bias)
            setattr(module, name, fp8_layer)
        else:
            convert_to_fp8(child, target_classes=target_classes)
    return module

