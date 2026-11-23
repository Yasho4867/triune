"""3-Tier Dynamic Router Operator (Reflex, Limbic, Cortex)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

TIER_REFLEX = 0
TIER_LIMBIC = 1
TIER_CORTEX = 2

TIER_NAMES = {
    TIER_REFLEX: "Reflex",
    TIER_LIMBIC: "Limbic",
    TIER_CORTEX: "Cortex",
}


def route_tiers(
    logits: torch.Tensor,
    reflex_threshold: float = 0.70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate 3-tier routing probabilities and selected execution tier.

    Args:
        logits: [B, 3] routing logits (0: Reflex, 1: Limbic, 2: Cortex)
        reflex_threshold: Confidence threshold for early exit at Reflex tier

    Returns:
        (probs, selected_tiers)
    """
    from ._c_bridge import c_route_tiers_f32

    c_res = c_route_tiers_f32(logits, reflex_threshold=reflex_threshold)
    if c_res is not None:
        return c_res

    probs = F.softmax(logits, dim=-1)
    p_reflex = probs[..., 0]
    p_limbic = probs[..., 1]
    p_cortex = probs[..., 2]

    selected = torch.where(
        p_reflex >= reflex_threshold,
        torch.tensor(TIER_REFLEX, device=logits.device),
        torch.where(
            p_limbic >= p_cortex,
            torch.tensor(TIER_LIMBIC, device=logits.device),
            torch.tensor(TIER_CORTEX, device=logits.device),
        ),
    )
    return probs, selected
