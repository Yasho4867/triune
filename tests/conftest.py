"""Global pytest fixtures for preserving workspace artifacts across test runs."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def preserve_session_state():
    """Snapshot and restore checkpoints/session_state.json around the full pytest session."""
    session_file = Path(__file__).resolve().parents[1] / "checkpoints" / "session_state.json"
    backup = session_file.read_bytes() if session_file.is_file() else None
    yield
    if backup is not None:
        session_file.write_bytes(backup)
