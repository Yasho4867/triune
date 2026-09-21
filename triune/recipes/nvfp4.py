"""Transformer Engine NVFP4 precision context."""

from __future__ import annotations

import torch

from .bf16 import bf16_autocast


def build_precision_context(*, use_fp4: bool, device) -> callable:
    """Build the training-forward precision context without entering it yet."""
    if not use_fp4:
        return lambda: bf16_autocast(device.type)

    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, NVFP4BlockScaling
    except ImportError as error:
        raise RuntimeError("NVFP4 requires Transformer Engine with NVFP4BlockScaling support") from error

    if device.type != "cuda":
        raise RuntimeError("NVFP4 requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    if major < 10:
        raise RuntimeError(f"NVFP4 requires a Blackwell-class GPU (SM100+); found {major}.{minor}")

    recipe = NVFP4BlockScaling(fp4_format=Format.E2M1)
    autocast = getattr(te, "autocast", None) or getattr(te, "fp8_autocast", None)
    if autocast is None:
        raise RuntimeError("Transformer Engine exposes no autocast context manager")
    return lambda: autocast(enabled=True, recipe=recipe)
