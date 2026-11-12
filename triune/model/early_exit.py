"""Early exit adapters and continuous latent routing modules."""

from __future__ import annotations

from triune.model.adapters import (
    EarlyExitManager,
    WeightTiedExitProjection,
    attach_early_exits,
    generate_adaptive,
)
from triune.model.jev import (
    ContinuousLatentExitRouter,
    JEVExitRouter,
    JEVLatentPredictor,
    JEVObjective,
    JEVRoutingResult,
    JointEmbeddingObjective,
    LatentPredictor,
    TriuneContinuousExitRouter,
    TriuneLatentObjective,
    TriuneLatentPredictor,
)

__all__ = [
    "EarlyExitManager",
    "WeightTiedExitProjection",
    "attach_early_exits",
    "generate_adaptive",
    "JEVLatentPredictor",
    "JEVObjective",
    "JEVExitRouter",
    "JEVRoutingResult",
    "LatentPredictor",
    "JointEmbeddingObjective",
    "ContinuousLatentExitRouter",
    "TriuneLatentPredictor",
    "TriuneLatentObjective",
    "TriuneContinuousExitRouter",
]
