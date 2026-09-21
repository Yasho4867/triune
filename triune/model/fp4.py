"""Transformer Engine-backed linear layer used by the existing FP4 path."""

from __future__ import annotations

import torch
import torch.nn as nn

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
    """Use Transformer Engine when available and hardware-supported, otherwise retain a BF16 fallback.

    The NVFP4 recipe itself is selected by the trainer's precision context. This
    module deliberately does not enter an autocast region on its own.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if _is_te_supported():
            try:
                self.linear = te.Linear(in_features, out_features, bias=bias, params_dtype=torch.bfloat16)
            except Exception:
                self.linear = nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
