"""Checkpoint persistence and restoration for :class:`triune.Trainer`."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

# Schema-v2 canonical fingerprint specification (Safeguard 6)
FINGERPRINT_SPEC = {
    "schema_version": 2,
    "architecture": ["hidden_dim", "num_layers", "num_heads", "head_dim", "vocab_size"],
    "attention": ["head_dim", "use_rope", "rope_max_seq_len"],
    "moe": ["num_experts", "top_k", "shared_expert", "shared_scale", "capacity_multiplier"],
    "hierarchy": ["router_prefix_layers", "reflex_exit_layer", "limbic_exit_layer", "target_depth_dist"],
    "precision": ["use_fp4", "use_fp8"],
    "optimizer": ["use_muon", "muon_lr", "galore", "galore_rank"],
}


def extract_canonical_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extracts canonical architecture and training semantics according to Schema-v2 specification."""
    canonical: dict[str, Any] = {"schema_version": 2}
    for category, keys in FINGERPRINT_SPEC.items():
        if category == "schema_version":
            continue
        cat_dict: dict[str, Any] = {}
        for k in keys:
            val = config.get(k)
            if isinstance(val, tuple):
                val = list(val)
            cat_dict[k] = val
        canonical[category] = cat_dict
    return canonical


def compute_checkpoint_fingerprint(config: Mapping[str, Any]) -> str:
    """Computes deterministic SHA-256 fingerprint over Schema-v2 canonical semantics."""
    canonical = extract_canonical_semantics(config)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _checkpoint_payload(trainer, engine, step: int, loss: float) -> dict:
    run_id = None
    if hasattr(trainer, "logger") and trainer.logger is not None:
        run_id = getattr(trainer.logger, "run_id", None)
    best_loss = getattr(engine, "best_eval_loss", None) if engine is not None else None
    depth_ema = getattr(engine, "depth_usage_ema", None) if engine is not None else None
    cfg = getattr(trainer, "config", {}) or {}
    return {
        "step": step,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict() if trainer.optimizer is not None else {},
        "loss": loss,
        "best_eval_loss": best_loss,
        "config": cfg,
        "schema_version": 2,
        "fingerprint": compute_checkpoint_fingerprint(cfg),
        "canonical_semantics": extract_canonical_semantics(cfg),
        "depth_usage_ema": depth_ema,
        "wandb_run_id": run_id,
    }


def _atomic_save_checkpoint(payload: dict, target_path: Path) -> Path:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # In WSL2, saving >4GB zip files directly to /mnt/c (drvfs 9P mount) causes
    # PyTorch's zip writer to fail with "unexpected pos" on write_end_of_file().
    # Writing to native ext4 disk storage (~/.triune_cache/tmp) and stream-copying resolves this
    # without consuming host RAM (unlike /tmp which is tmpfs RAM).
    is_mnt = str(target_path.resolve()).startswith("/mnt/")
    if is_mnt:
        disk_tmp_dir = Path.home() / ".triune_cache" / "tmp"
        disk_tmp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = disk_tmp_dir / f"{target_path.stem}_{os.getpid()}_{time.time_ns()}.tmp"
        target_tmp = target_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            torch.save(payload, temp_file)
            shutil.copyfile(temp_file, target_tmp)
            target_tmp.replace(target_path)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            if target_tmp.exists():
                try:
                    target_tmp.unlink()
                except Exception:
                    pass
    else:
        temp_file = target_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(payload, temp_file)
        temp_file.replace(target_path)

    return target_path


def save_latest(trainer, engine, step: int, loss: float) -> Path:
    path = Path(trainer.config["checkpoint_dir"]) / "latest.pt"
    try:
        _atomic_save_checkpoint(_checkpoint_payload(trainer, engine, step, loss), path)
    except Exception as exc:
        logger.error("Failed to save latest checkpoint to %s: %s", path, exc, exc_info=True)
    return path


def save_best(trainer, engine, step: int, loss: float) -> Path:
    path = Path(trainer.config["checkpoint_dir"]) / "best.pt"
    try:
        _atomic_save_checkpoint(_checkpoint_payload(trainer, engine, step, loss), path)
    except Exception as exc:
        logger.error("Failed to save best checkpoint to %s: %s", path, exc, exc_info=True)
    return path


def load_checkpoint(trainer, path: str | Path, *, load_optimizer: bool) -> dict:
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)

    # Schema-v2 Architecture & Semantic fingerprint verification
    saved_config = checkpoint.get("config", {})
    if saved_config and hasattr(trainer, "config") and isinstance(trainer.config, dict):
        mismatches = []
        critical_checks = [
            ("hidden_dim", "architecture"),
            ("num_layers", "architecture"),
            ("num_heads", "architecture"),
            ("head_dim", "architecture"),
            ("vocab_size", "architecture"),
            ("num_experts", "moe"),
            ("top_k", "moe"),
            ("shared_expert", "moe"),
            ("router_prefix_layers", "hierarchy"),
            ("reflex_exit_layer", "hierarchy"),
            ("limbic_exit_layer", "hierarchy"),
        ]
        for key, cat in critical_checks:
            saved_val = saved_config.get(key)
            current_val = trainer.config.get(key)
            if saved_val is not None and current_val is not None and saved_val != current_val:
                mismatches.append(f"{key} [{cat}]: saved={saved_val} vs current={current_val}")

        if mismatches:
            raise ValueError(
                f"Checkpoint architecture mismatch for {path}:\n"
                + "\n".join(f"  - {m}" for m in mismatches)
                + f"\nCannot restore weights into an incompatible model configuration."
            )

    state_dict = checkpoint["model_state"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    trainer.model.load_state_dict(state_dict)
    if load_optimizer:
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
