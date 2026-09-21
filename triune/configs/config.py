"""Runtime configuration helpers for the callable training framework.

The established top-level ``config.py`` remains the single source of default
research settings.  This module turns those defaults into per-run dictionaries
so importing the package never parses arguments or starts training.
"""

from __future__ import annotations

from typing import Any, Mapping

import triune.model.config as defaults



def default_config() -> dict[str, Any]:
    """Return an independent mutable configuration for one training run."""
    return {
        "vocab_size": defaults.VOCAB_SIZE,
        "hidden_dim": defaults.HIDDEN_DIM,
        "num_layers": defaults.NUM_LAYERS,
        "num_heads": defaults.NUM_HEADS,
        "head_dim": defaults.GLA_HEAD_DIM,
        "num_experts": defaults.NUM_EXPERTS,
        "router_prefix_layers": defaults.ROUTER_PREFIX_LAYERS,
        "reflex_exit_layer": defaults.REFLEX_EXIT_LAYER,
        "limbic_exit_layer": defaults.LIMBIC_EXIT_LAYER,
        "batch_size": defaults.BATCH_SIZE,
        "grad_accum_steps": defaults.GRAD_ACCUM_STEPS,
        "seq_len": defaults.SEQ_LEN,
        "total_steps": defaults.TOTAL_STEPS,
        "save_every": defaults.SAVE_EVERY,
        "log_every": defaults.LOG_EVERY,
        "eval_every": defaults.EVAL_EVERY,
        "eval_batches": defaults.EVAL_BATCHES,
        "target_depth_dist": list(defaults.TARGET_DEPTH_DIST),
        "usage_ema_decay": defaults.DEPTH_USAGE_EMA,
        "bias_strength": defaults.DEPTH_BIAS_STRENGTH,
        "balance_coef": defaults.DEPTH_BALANCE_COEF,
        "lr": defaults.LR,
        "min_lr": defaults.MIN_LR,
        "warmup_steps": defaults.WARMUP_STEPS,
        "weight_decay": defaults.WEIGHT_DECAY,
        "betas": defaults.BETAS,
        "grad_clip": defaults.GRAD_CLIP,
        "checkpoint_dir": defaults.CHECKPOINT_DIR,
        "exploration": "linear",
        "exploration_steps": 5_000,
        "use_fp4": False,
        "use_fp8": False,
        "steer_scale": defaults.STEER_SCALE,
        "use_muon": defaults.USE_MUON,
        "muon_lr": defaults.MUON_LR,
        "muon_momentum": defaults.MUON_MOMENTUM,
        "muon_weight_decay": defaults.MUON_WEIGHT_DECAY,
        "shuffle_buffer": defaults.SHUFFLE_BUFFER,
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "num_workers": 0,
    }


def build_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = default_config()
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config["seq_len"] <= 0 or config["seq_len"] > defaults.ROPE_MAX_SEQ_LEN:
        raise ValueError(f"seq_len must be in [1, {defaults.ROPE_MAX_SEQ_LEN}]")
    if config["batch_size"] <= 0 or config["grad_accum_steps"] <= 0:
        raise ValueError("batch_size and grad_accum_steps must be positive")
    if config["total_steps"] < 0 or config["warmup_steps"] < 0:
        raise ValueError("total_steps and warmup_steps must be non-negative")
    if config["eval_every"] <= 0 or config["eval_batches"] <= 0:
        raise ValueError("eval_every and eval_batches must be positive")
    if config["save_every"] <= 0 or config["log_every"] <= 0:
        raise ValueError("save_every and log_every must be positive")
    target = config["target_depth_dist"]
    if len(target) != 3 or any(value < 0 for value in target):
        raise ValueError("target_depth_dist must contain three non-negative values")
    if abs(sum(target) - 1.0) > 1e-6:
        raise ValueError("target_depth_dist must sum to 1.0")
