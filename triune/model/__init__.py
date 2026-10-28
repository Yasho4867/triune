"""Triune Transformer neural network modules and architectures."""

from __future__ import annotations

from .adapters import WeightTiedExitProjection, attach_early_exits, generate_adaptive
from .attention import HybridAttention, VectorisedGLA
from .base import (
    CANONICAL_MODEL_SPECS,
    MODEL_REGISTRY,
    ModelSpec,
    TriuneModel,
    build_model_from_spec,
    get_model_spec,
    register_model,
)
from .block import TransformerBlock
from .factory import build_model
from .fp4 import FP4Linear
from .fp8 import FP8Linear
from .moe import MoE_FFN, RoutingResult
from .norms import RMSNorm
from .rotary import RotaryEmbedding, apply_rotary
from .router import GumbelSoftmaxRouter
from .transformer import TriuneTransformer
from .zoo import load_model

__all__ = [
    "CANONICAL_MODEL_SPECS",
    "MODEL_REGISTRY",
    "ModelSpec",
    "TriuneModel",
    "TriuneTransformer",
    "TransformerBlock",
    "HybridAttention",
    "VectorisedGLA",
    "MoE_FFN",
    "RoutingResult",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    "GumbelSoftmaxRouter",
    "FP4Linear",
    "FP8Linear",
    "build_model",
    "build_model_from_spec",
    "get_model_spec",
    "load_model",
    "register_model",
    "WeightTiedExitProjection",
    "attach_early_exits",
    "generate_adaptive",
]
