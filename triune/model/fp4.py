"""NVFP4 E2M1 Linear layer with Transformer Engine fast-path and native Triton block-scaled E2M1 support."""

from __future__ import annotations

import torch
import torch.nn as nn

from triune.kernels.fast_nvfp4 import NVFP4_GROUP_SIZE, NVFP4Linear, _NVFP4MatmulFn
from triune.runtime.capabilities import PrecisionCapabilities

try:
    import transformer_engine.pytorch as te

    HAS_TE = True
except ImportError:
    te = None
    HAS_TE = False


def _is_te_supported() -> bool:
    caps = PrecisionCapabilities.detect()
    return caps.transformer_engine and (caps.compute_capability[0] >= 8)


class FP4Linear(nn.Module):
    """NVFP4 E2M1 linear layer with Transformer Engine or native Triton block-scaled E2M1 execution.

    When Transformer Engine is available on Hopper/Blackwell hardware, delegates to
    ``te.Linear`` under the active precision recipe. Otherwise executes real
    block-scaled E2M1 (group size 16) quantization/dequantization via Triton kernels
    with Straight-Through Estimator (STE) backward gradients into master weights
    (preserving ``self.linear`` for checkpoint and optimizer compatibility), or uses
    packed 4-bit ``NVFP4Linear`` inference weights when converted via
    :func:`convert_to_nvfp4`.
    """

    _triune_fp4_aware: bool = True

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self._triune_fp4_aware = True
        self.in_features = in_features
        self.out_features = out_features
        self._use_te = _is_te_supported()
        if self._use_te:
            try:
                self.linear = te.Linear(in_features, out_features, bias=bias, params_dtype=torch.bfloat16)
                self.linear._triune_fp4_aware = True
                return
            except Exception:
                self._use_te = False
        self.linear = nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)
        self.linear._triune_fp4_aware = True

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.linear, "bias", None)

    def to_nvfp4_inference(self) -> NVFP4Linear:
        """Compresses this layer into packed 4-bit NVFP4 E2M1 inference storage (0.625 B/param)."""
        if self.in_features % NVFP4_GROUP_SIZE != 0:
            raise ValueError(
                f"NVFP4 packed format requires in_features % {NVFP4_GROUP_SIZE} == 0, got {self.in_features}"
            )
        return NVFP4Linear.from_linear(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_te:
            return self.linear(x)
        w = self.linear.weight
        if w.size(1) % NVFP4_GROUP_SIZE == 0:
            packed = getattr(w, "_nvfp4_packed", None)
            scale = getattr(w, "_nvfp4_scale", None)
            return _NVFP4MatmulFn.apply(x, w, self.linear.bias, packed, scale)
        return self.linear(x)


_SENSITIVE_NVFP4_EXACT_NAMES = frozenset({"router", "exit_head", "final_head", "lm_head", "embed", "embed_tokens"})


def convert_to_nvfp4(
    module: nn.Module,
    target_classes=(nn.Linear, FP4Linear),
    hadamard: bool = True,
    protect_sensitive: bool = True,
) -> nn.Module:
    """Recursively converts compatible Linear/FP4Linear layers into packed 4-bit NVFP4Linear.

    When ``protect_sensitive=True`` (default), small routing gates and vocabulary
    output heads (``router``, ``exit_head``, ``final_head``, ``lm_head``, small ``gate``)
    remain in full precision to preserve softmax/routing calibration.
    When ``hadamard=True`` (default), applies orthogonal block-16 Walsh-Hadamard
    preconditioning (FWHT-16) prior to E2M1 packing to suppress weight outliers.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, target_classes) and not isinstance(child, NVFP4Linear):
            out_f = child.out_features if hasattr(child, "out_features") else child.weight.size(0)
            if protect_sensitive and (name in _SENSITIVE_NVFP4_EXACT_NAMES or (name == "gate" and out_f <= 64)):
                continue
            in_f = child.in_features if hasattr(child, "in_features") else child.weight.size(1)
            if in_f % NVFP4_GROUP_SIZE == 0:
                setattr(module, name, NVFP4Linear.from_linear(child, hadamard=hadamard))
        else:
            convert_to_nvfp4(
                child,
                target_classes=target_classes,
                hadamard=hadamard,
                protect_sensitive=protect_sensitive,
            )
    return module
