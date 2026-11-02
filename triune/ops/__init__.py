"""Hardware-Accelerated Native Operators Namespace."""

from __future__ import annotations

from .centroid_steer import (
    batched_subspace_projection,
    cholesky_qr,
    subspace_projection,
    triangular_steer_update,
)
from .gla import gla
from .muon import muon_orthogonalize
from .norm import norm
from .router import TIER_CORTEX, TIER_LIMBIC, TIER_NAMES, TIER_REFLEX, route_tiers
from .swiglu import swiglu

__all__ = [
    "cholesky_qr",
    "subspace_projection",
    "batched_subspace_projection",
    "triangular_steer_update",
    "gla",
    "muon_orthogonalize",
    "norm",
    "swiglu",
    "route_tiers",
    "TIER_REFLEX",
    "TIER_LIMBIC",
    "TIER_CORTEX",
    "TIER_NAMES",
]
