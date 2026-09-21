"""Reproducibility helpers."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
