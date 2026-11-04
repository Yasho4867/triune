"""Hardware-Accelerated Native Operators Namespace."""

from __future__ import annotations

from triune.kernels.fast_cross_entropy import (
    FastChunkedCrossEntropyLoss,
)
from triune.kernels.fast_cross_entropy import (
    chunked_cross_entropy_from_hidden as chunked_cross_entropy,
)
from triune.kernels.fast_cross_entropy import (
    fast_cross_entropy as cross_entropy,
)
from triune.kernels.fast_lora import fast_lora_forward as lora_forward
from triune.kernels.fast_lora import fast_lora_linear as lora_linear
from triune.kernels.fast_rope import fast_rope as rope
from triune.kernels.fast_swiglu import fast_gelu as gelu

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

rmsnorm = norm
muon_ns = muon_orthogonalize


def upgrade(model, **kwargs):
    """Universal 1-line Triune AI drop-in accelerator for any PyTorch or Hugging Face model."""
    from triune.universal import upgrade_model

    return upgrade_model(model, **kwargs)


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
    "muon_ns",
    "norm",
    "rmsnorm",
    "rope",
    "swiglu",
    "gelu",
    "cross_entropy",
    "chunked_cross_entropy",
    "FastChunkedCrossEntropyLoss",
    "lora_forward",
    "lora_linear",
    "upgrade",
    "route_tiers",
    "TIER_REFLEX",
    "TIER_LIMBIC",
    "TIER_CORTEX",
    "TIER_NAMES",
]
