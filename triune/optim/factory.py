"""Optimizer construction without module-level training state."""

from __future__ import annotations

import torch

import triune.model.config as defaults


from .centroid import AdamW8bit, CentroidSteerOptimizer, HAS_8BIT


def build_optimizer(model, config: dict):
    """Build the configured optimizer while preserving the centroid/GaLore path."""
    if config.get("galore", defaults.GALORE):
        print("[Optim] CentroidSteerOptimizer active")
        return CentroidSteerOptimizer(
            model,
            lr=config["lr"],
            betas=config["betas"],
            weight_decay=config["weight_decay"],
            rank=config.get("galore_rank", defaults.GALORE_RANK),
            update_gap=config.get("galore_update_gap", defaults.GALORE_UPDATE_GAP),
            steer_scale=config.get("steer_scale", defaults.STEER_SCALE),
            expert_lr=config.get("galore_lr", defaults.GALORE_LR),
            expert_betas=config.get("galore_betas", defaults.GALORE_BETAS),
            expert_wd=config.get("galore_weight_decay", defaults.GALORE_WEIGHT_DECAY),
            use_muon=config.get("use_muon", defaults.USE_MUON),
            muon_lr=config.get("muon_lr", defaults.MUON_LR),
            muon_momentum=config.get("muon_momentum", defaults.MUON_MOMENTUM),
            muon_weight_decay=config.get("muon_weight_decay", defaults.MUON_WEIGHT_DECAY),
        )
    if HAS_8BIT and AdamW8bit is not None:
        return AdamW8bit(model.parameters(), lr=config["lr"], betas=config["betas"], weight_decay=config["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=config["betas"], weight_decay=config["weight_decay"])
