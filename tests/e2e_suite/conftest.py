"""Pytest configuration and non-colliding E2E fixtures for tests/e2e_suite/."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import pytest

from .helpers import (
    ChromeBrowserController,
    StudioServerHandle,
    assert_r8_snapshot_intact,
    capture_r8_snapshot,
    launch_chrome_controller,
    run_website_tool,
    start_studio_server_process,
    terminate_process,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "tier1: Tier 1 Category-Partition core requirement tests")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary value, negative, and error-handling tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-feature pairwise integration tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-world end-to-end user/engineer scenarios")


@pytest.fixture(scope="session")
def e2e_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def e2e_python_exe() -> str:
    return sys.executable


@pytest.fixture(scope="session", autouse=True)
def r8_preservation_snapshot(e2e_repo_root: Path) -> Iterator[Dict[str, object]]:
    """Snapshot R8 preserved repo artifacts before the session and verify byte-identity after."""
    snapshot = capture_r8_snapshot(e2e_repo_root)
    yield snapshot
    assert_r8_snapshot_intact(e2e_repo_root, snapshot)


@pytest.fixture(scope="function")
def isolated_cli_workspace(tmp_path: Path, e2e_repo_root: Path) -> Path:
    """Isolated temporary CLI workspace with empty checkpoints/ and copied sample data/tokenizer."""
    ws = tmp_path / "cli_ws"
    (ws / "checkpoints").mkdir(parents=True, exist_ok=True)
    (ws / "checkpoints_full").mkdir(parents=True, exist_ok=True)
    (ws / "data").mkdir(parents=True, exist_ok=True)

    src_sample = e2e_repo_root / "data" / "fineweb_sample.jsonl"
    if src_sample.is_file():
        shutil.copy2(src_sample, ws / "data" / "fineweb_sample.jsonl")

    src_tok = e2e_repo_root / "triune_tokenizer.json"
    if src_tok.is_file():
        shutil.copy2(src_tok, ws / "triune_tokenizer.json")

    return ws


@pytest.fixture(scope="function")
def isolated_plugin_dir(tmp_path: Path) -> Path:
    """Isolated temporary config/plugin directory for ModuleManager without touching host Studio dirs."""
    plugin_root = tmp_path / "studio_config"
    (plugin_root / "modules").mkdir(parents=True, exist_ok=True)
    return plugin_root


@pytest.fixture(scope="module")
def studio_server(e2e_repo_root: Path, e2e_python_exe: str) -> Iterator[StudioServerHandle]:
    """Ephemeral Polysom Studio FastAPI server on 127.0.0.1:<free_port> with isolated HOME."""
    tmp_home = Path(tempfile.mkdtemp(prefix="e2e_studio_home_"))
    handle = start_studio_server_process(
        python_exe=e2e_python_exe,
        repo_root=e2e_repo_root,
        isolated_home=tmp_home,
        mask_windows_studio_dirs=True,
    )
    try:
        yield handle
    finally:
        terminate_process(handle.proc)
        shutil.rmtree(tmp_home, ignore_errors=True)


@pytest.fixture(scope="function")
def isolated_plugin_studio_server(
    tmp_path: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> Iterator[Callable[[Optional[Path], Optional[Dict[str, str]]], StudioServerHandle]]:
    """Factory fixture starting an ephemeral Studio server with masked C:\\PolysomStudio and custom HOME."""
    started: list[StudioServerHandle] = []

    def _start(
        home_dir: Optional[Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> StudioServerHandle:
        target_home = home_dir or (tmp_path / f"home_{len(started)}")
        handle = start_studio_server_process(
            python_exe=e2e_python_exe,
            repo_root=e2e_repo_root,
            isolated_home=target_home,
            mask_windows_studio_dirs=True,
            extra_env=extra_env,
        )
        started.append(handle)
        return handle

    try:
        yield _start
    finally:
        for handle in started:
            terminate_process(handle.proc)


@pytest.fixture(scope="module")
def cdp_browser() -> Iterator[ChromeBrowserController]:
    """Headless Chrome CDP controller using websockets (playwright-free)."""
    profile_dir = Path(tempfile.mkdtemp(prefix="e2e_chrome_profile_"))
    controller = launch_chrome_controller(profile_dir)
    try:
        yield controller
    finally:
        terminate_process(controller.proc)
        shutil.rmtree(profile_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def scratch_website_dist(e2e_repo_root: Path) -> Iterator[Path]:
    """Scratch Vite build output directory (never overwrites website/dist/)."""
    scratch_dir = Path(tempfile.mkdtemp(prefix="e2e_web_dist_"))
    try:
        proc = run_website_tool(
            e2e_repo_root,
            "vite",
            ["build", "--outDir", str(scratch_dir)],
            timeout=120.0,
        )
        assert proc.returncode == 0, (
            f"vite build --outDir failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        yield scratch_dir
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
