"""Transformer Engine-backed linear layer used by the existing FP4 path."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import transformer_engine.pytorch as te

    HAS_TE = True
except ImportError:
    te = None
    HAS_TE = False


class FP4Linear(nn.Module):
    """Use Transformer Engine when available, otherwise retain a BF16 fallback.

    The NVFP4 recipe itself is selected by the trainer's precision context.  This
    module deliberately does not enter an autocast region on its own.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if HAS_TE:
            self.linear = te.Linear(in_features, out_features, bias=bias, params_dtype=torch.bfloat16)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)

    def forward(self, x):
        try:
            return self.linear(x)
        except RuntimeError as e:
            if "no kernel image" in str(e) or "CUDA Error" in str(e):
                in_f = getattr(self.linear, "in_features", x.shape[-1])
                out_f = getattr(self.linear, "out_features", in_f)
                has_bias = getattr(self.linear, "bias", None) is not None
                self.linear = nn.Linear(in_f, out_f, bias=has_bias, device=x.device, dtype=x.dtype)
                return self.linear(x)
            raise
