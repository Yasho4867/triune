"""Modular Callback System for Trainer Lifecycle Events."""

from __future__ import annotations

from typing import Any, Dict, List


class Callback:
    """Base class for all training callbacks."""

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_step_begin(self, trainer: Any, step: int) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, logs: Dict[str, Any]) -> None:
        pass

    def on_eval_begin(self, trainer: Any) -> None:
        pass

    def on_eval_end(self, trainer: Any, eval_loss: float) -> None:
        pass


class CallbackList(Callback):
    """Container for managing and dispatching events to multiple callbacks."""

    def __init__(self, callbacks: List[Callback] | None = None) -> None:
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_step_begin(self, trainer: Any, step: int) -> None:
        for cb in self.callbacks:
            cb.on_step_begin(trainer, step)

    def on_step_end(self, trainer: Any, step: int, logs: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_step_end(trainer, step, logs)

    def on_eval_begin(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_eval_begin(trainer)

    def on_eval_end(self, trainer: Any, eval_loss: float) -> None:
        for cb in self.callbacks:
            cb.on_eval_end(trainer, eval_loss)
