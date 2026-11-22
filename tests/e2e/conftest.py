"""Shared E2E test configuration and dynamic package resolution fixtures for Lucid Machines v1.0.0."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: Requirement-driven opaque-box E2E tests for the Lucid Machines v1.0.0 release pipeline (R1-R5)",
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def active_pkg_name() -> str:
    assert importlib.util.find_spec("lucid_machines") is not None
    return "lucid_machines"
