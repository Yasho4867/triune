"""Real-Time Telemetry & Event Emitter Hooks for Triune Engine."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List


class EventEmitter:
    """Abstract Event Hook system for streaming live logs, loss curves, and token outputs."""

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event_name: str, listener: Callable) -> None:
        """Register an event listener callback."""
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(listener)

    def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event to all registered listeners."""
        if event_name in self._listeners:
            for listener in self._listeners[event_name]:
                try:
                    listener(*args, **kwargs)
                except Exception as err:
                    print(f"[Warning] EventEmitter listener error for '{event_name}': {err}")

    # Standard Hooks
    def emit_node_start(self, node_id: str, node_type: str, params: Dict[str, Any]) -> None:
        self.emit("on_node_start", {"node_id": node_id, "node_type": node_type, "params": params, "timestamp": time.time()})

    def emit_token_generate(self, token: str, step: int) -> None:
        self.emit("on_token_generate", {"token": token, "step": step, "timestamp": time.time()})

    def emit_loss_update(self, step: int, loss: float, lm_loss: float = 0.0, router_loss: float = 0.0) -> None:
        self.emit("on_loss_update", {
            "step": step,
            "loss": loss,
            "lm_loss": lm_loss,
            "router_loss": router_loss,
            "timestamp": time.time()
        })

    def emit_vram_update(self, allocated_gb: float, reserved_gb: float, total_gb: float) -> None:
        self.emit("on_vram_update", {
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "total_gb": total_gb,
            "timestamp": time.time()
        })


# Global Engine Telemetry Emitter Instance
global_emitter = EventEmitter()
