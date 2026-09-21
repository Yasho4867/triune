"""FP8 (E4M3 / Delayed Scaling) precision context support."""

from __future__ import annotations

import logging
import torch

from .bf16 import bf16_autocast

logger = logging.getLogger(__name__)


def build_fp8_precision_context(*, device, use_te: bool = True) -> callable:
    """Build FP8 precision autocast context.

    Tries Transformer Engine FP8 (DelayedScaling) first when requested and available,
    falling back to PyTorch native float8_e4m3fn autocast if TE is unavailable,
    or BF16 autocast if the GPU capability is below SM89.
    """
    if device.type != "cuda":
        raise RuntimeError("FP8 requires CUDA")

    major, minor = torch.cuda.get_device_capability(device)
    is_sm89_plus = (major > 8) or (major == 8 and minor >= 9)

    # 1. Try Transformer Engine if requested and on SM89+
    if use_te and is_sm89_plus:
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Format

            recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
            autocast = getattr(te, "autocast", None) or getattr(te, "fp8_autocast", None)
            if autocast is not None:
                ctx_fn = lambda: autocast(enabled=True, recipe=recipe)
                ctx_fn.description = "Transformer Engine FP8 (DelayedScaling HYBRID) active"
                return ctx_fn
        except (ImportError, Exception):
            pass

    # 2. Try PyTorch native float8 autocast on SM89+
    if is_sm89_plus and hasattr(torch, "float8_e4m3fn"):
        try:
            with torch.amp.autocast("cuda", dtype=torch.float8_e4m3fn):
                pass
            ctx_fn = lambda: torch.amp.autocast("cuda", dtype=torch.float8_e4m3fn)
            ctx_fn.description = "Native PyTorch FP8 (float8_e4m3fn) autocast active"
            return ctx_fn
        except Exception:
            pass

    # 3. Fallback to BF16 with accurate description
    if not is_sm89_plus:
        msg = f"GPU SM{major}.{minor} lacks FP8 Tensor Cores (requires SM89+ / Ada, Hopper, Blackwell); falling back to BF16 autocast."
    else:
        msg = "FP8 execution unavailable; falling back to BF16 autocast."

    logger.warning(msg)
    ctx_fn = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
    ctx_fn.description = msg
    return ctx_fn
