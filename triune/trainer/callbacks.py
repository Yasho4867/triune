"""Optional process-signal integration for a :class:`Trainer`."""

from __future__ import annotations

import signal


def install_checkpoint_signal_handlers(trainer) -> None:
    """Save a recoverable checkpoint when the process receives SIGINT or SIGTERM."""
    def save_and_reraise(signum, _frame):
        print(f"\n⚠️ Received signal {signum}; saving checkpoint...")
        trainer.save_latest(trainer.engine.step, 0.0)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, save_and_reraise)
    signal.signal(signal.SIGTERM, save_and_reraise)
