"""Hardware-Accelerated Native Operators Namespace."""

from __future__ import annotations

from ._c_bridge import (
    NativeTriuneEngine,
    c_centroid_steer_update_f32,
    c_compute_neuron_heatmap_f32,
    c_gla_recurrent_step_f32,
    c_route_tiers_f32,
    get_native_c_info,
    is_native_c_available,
)
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
    "NativeTriuneEngine",
    "is_native_c_available",
    "get_native_c_info",
    "c_centroid_steer_update_f32",
    "c_route_tiers_f32",
    "c_compute_neuron_heatmap_f32",
    "c_gla_recurrent_step_f32",
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
