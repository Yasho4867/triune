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
    """Custom autograd function: FP8 forward pass, BF16 backward pass.

    Explicitly separates the master weight Parameter (for autograd identity) from
    the temporary FP8 compute representation and scale metadata.
    """

    @staticmethod
    def forward(ctx, x, weight_master, bias, weight_fp8=None, weight_inv_scale=None):
        # Save originals for backward (in master precision)
        ctx.save_for_backward(x, weight_master, bias)

        # Check if hardware native scaled_mm is available
        use_native = False
        if _HAS_SCALED_MM and x.is_cuda:
            caps = PrecisionCapabilities.detect(x.device)
            if caps.native_scaled_mm:
                use_native = True

        orig_shape = x.shape
        flat_x = x.reshape(-1, x.shape[-1])

        if use_native:
            # Quantize activations to FP8
            x_fp8, scale_x = _quantize_to_fp8(flat_x)

            # Use pre-quantized weight if provided, otherwise quantize dynamically
            if weight_fp8 is not None and weight_inv_scale is not None:
                w_fp8 = weight_fp8
                scale_w = weight_inv_scale
                if not isinstance(scale_w, torch.Tensor):
                    scale_w = torch.tensor(scale_w, device=x.device, dtype=torch.float32)
            else:
                w_fp8, scale_w = _quantize_to_fp8(weight_master)

            try:
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
                return out.reshape(*orig_shape[:-1], weight_master.shape[0])
            except (RuntimeError, NotImplementedError):
                pass

        # Fallback path: FP8-weight compute with dequantization
        if weight_fp8 is not None and weight_inv_scale is not None:
            inv = weight_inv_scale.to(x.dtype) if isinstance(weight_inv_scale, torch.Tensor) else float(weight_inv_scale)
            w_compute = (weight_fp8.to(x.dtype) * inv)
        else:
            w_fp8, inv = _quantize_to_fp8(weight_master)
            w_compute = (w_fp8.to(x.dtype) * inv.to(x.dtype))

        out = flat_x @ w_compute.t()
        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], weight_master.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass runs in master weight dtype for numerical stability
        x, weight_master, bias = ctx.saved_tensors
        master_dtype = weight_master.dtype

        grad_out = grad_output.to(dtype=master_dtype, device=weight_master.device)
        grad_output_flat = grad_out.reshape(-1, grad_out.shape[-1])
        x_flat = x.to(dtype=master_dtype).reshape(-1, x.shape[-1])

        w = weight_master.to(master_dtype)

        grad_x = grad_output_flat @ w  # [M, out] @ [out, in] = [M, in]
        grad_weight = grad_output_flat.t() @ x_flat  # [out, M] @ [M, in] = [out, in]
        grad_bias = grad_output_flat.sum(dim=0).to(dtype=bias.dtype) if bias is not None else None

        grad_x = grad_x.reshape(x.shape).to(dtype=x.dtype)
        grad_weight = grad_weight.to(dtype=master_dtype)

        # Return gradients matching forward signature: (x, weight_master, bias, weight_fp8, weight_inv_scale)
        return grad_x, grad_weight, grad_bias, None, None


class FP8Linear(nn.Module):
    """Hardware-accelerated FP8 Linear layer.

    - Explicitly marked as `_triune_fp8_aware = True` for runtime staging opt-in.
    - Weights stored in BF16/FP32 master representation (for gradient updates and optimizer).
    - Forward pass executes native FP8 GEMM on Tensor Cores when supported by hardware/kernel,
      or dequantized FP8 compute in compute dtype when native _scaled_mm is unavailable.
    - Backward pass strictly returns gradients in master weight dtype.
    """

    _triune_fp8_aware: bool = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self._triune_fp8_aware = True
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
        weight_master = self.weight
        weight_fp8 = getattr(self.weight, "_fp8_tensor", None)
        weight_inv_scale = getattr(self.weight, "_fp8_inv_scale", None)

        return _FP8MatmulFn.apply(x, weight_master, self.bias, weight_fp8, weight_inv_scale)

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

