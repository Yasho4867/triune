"""Optional process-signal integration for a :class:`Trainer`."""

from __future__ import annotations

import signal


import threading


def install_checkpoint_signal_handlers(trainer) -> None:
    """Save a recoverable checkpoint when the process receives SIGINT or SIGTERM."""
    if threading.current_thread() is not threading.main_thread():
        return

    def save_and_reraise(signum, _frame):
        print(f"\n⚠️ Received signal {signum}; saving checkpoint...")
        try:
            trainer.save_latest(trainer.engine.step, 0.0)
        except Exception:
            pass
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, save_and_reraise)
        signal.signal(signal.SIGTERM, save_and_reraise)
    except (ValueError, AttributeError):
        pass
