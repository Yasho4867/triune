"""Telemetry Callback for Real-time Streaming & W&B / Studio Monitoring."""

from __future__ import annotations

from typing import Any, Dict

import torch

from .base import Callback


class TelemetryCallback(Callback):
    """Collects real-time step telemetry (Loss, Exit Head usage, VRAM) for WebUI streaming."""

    def __init__(self, broadcast_fn: Any | None = None) -> None:
        self.broadcast_fn = broadcast_fn
        self.history: list[Dict[str, Any]] = []

    def on_step_end(self, trainer: Any, step: int, logs: Dict[str, Any]) -> None:
        vram_allocated_gb = (
            torch.cuda.memory_allocated(trainer.device) / (1024**3)
            if torch.cuda.is_available() and trainer.device.type == "cuda"
            else 0.0
        )
        payload = {
            "step": step,
            "loss": logs.get("loss"),
            "lm_loss": logs.get("lm_loss"),
            "router_loss": logs.get("router_loss"),
            "target_router": logs.get("target_router"),
            "vram_gb": round(vram_allocated_gb, 3),
        }
        self.history.append(payload)
        if self.broadcast_fn:
            self.broadcast_fn(payload)
