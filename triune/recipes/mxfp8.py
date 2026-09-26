"""Transformer Engine MXFP8 (Microscaling FP8) precision context support."""

from __future__ import annotations

import logging
import torch

from .bf16 import bf16_autocast

logger = logging.getLogger(__name__)

__all__ = ["build_mxfp8_precision_context"]


def build_mxfp8_precision_context(*, device) -> callable:
    """Build the MXFP8 precision context using Transformer Engine v2+ on Blackwell (SM100+).

    Falls back to BF16 autocast if MXFP8BlockScaling is not supported or device is not SM100+.
    """
    if device.type != "cuda":
        raise RuntimeError("MXFP8 requires CUDA")

    major, minor = torch.cuda.get_device_capability(device)
    if major < 10:
        msg = f"MXFP8 requires a Blackwell-class GPU (SM100+); found SM{major}.{minor}. Falling back to BF16."
        logger.warning(msg)
        ctx_fn = lambda: bf16_autocast(device.type)
        ctx_fn.description = msg
        return ctx_fn

    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, MXFP8BlockScaling

        recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
        autocast = getattr(te, "autocast", None) or getattr(te, "fp8_autocast", None)
        if autocast is not None:
            ctx_fn = lambda: autocast(enabled=True, recipe=recipe)
            ctx_fn.description = "Transformer Engine MXFP8 (BlockScaling 32) active"
            return ctx_fn
    except (ImportError, Exception) as exc:
        logger.warning(f"Transformer Engine MXFP8 unavailable ({exc}); falling back to BF16 autocast.")

    ctx_fn = lambda: bf16_autocast(device.type)
    ctx_fn.description = "Fallback BF16 autocast"
    return ctx_fn
