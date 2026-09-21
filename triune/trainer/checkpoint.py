"""Checkpoint persistence and restoration for :class:`triune.Trainer`."""

from __future__ import annotations

from pathlib import Path

import torch


import logging
import os
import shutil
import tempfile
import time

logger = logging.getLogger(__name__)


def _checkpoint_payload(trainer, engine, step: int, loss: float) -> dict:
    return {
        "step": step,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "loss": loss,
        "best_eval_loss": engine.best_eval_loss,
        "config": trainer.config,
        "depth_usage_ema": engine.depth_usage_ema,
        "wandb_run_id": trainer.logger.run_id,
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
    state_dict = checkpoint["model_state"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    trainer.model.load_state_dict(state_dict)
    if load_optimizer:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
