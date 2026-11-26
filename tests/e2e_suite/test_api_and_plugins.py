"""Feature #3 (Agent API Local Triune), Feature #4 (Scoped Plugin Discovery), and Feature #5 (Fast /v1/models, Workspace & Benchmarks API)."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import pytest

from .helpers import StudioServerHandle, start_studio_server_process, terminate_process

# ---------------------------------------------------------------------------
# Feature #3 — Agent API Local Triune Completion (5 T1 + 5 T2)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_f3_agents_run_explicit_triune_base(studio_server: StudioServerHandle) -> None:
    """F3-T1-01: POST /v1/agents/run with model='triune-base' completes via local PyTorch completion."""
    status, data, _ = studio_server.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "triune-base", "goal": "List workspace files", "max_steps": 1},
        timeout=60.0,
    )
    assert status == 200, f"Expected HTTP 200, got {status}: {data}"
    assert data.get("status") == "success", f"Agent run failed: {data}"
    assert str(data.get("run_id", "")).startswith("agent-"), f"Unexpected run_id: {data}"
    assert float(data.get("elapsed_ms", -1)) >= 0
    assert "Unsupported agent model reference" not in json.dumps(data)


@pytest.mark.tier1
def test_t1_f3_agents_run_triune_small_and_junkie_mini(studio_server: StudioServerHandle) -> None:
    """F3-T1-02: POST /v1/agents/run accepts both 'triune-small' and 'junkie-mini' model IDs."""
    for model_id in ("triune-small", "junkie-mini"):
        status, data, _ = studio_server.request_json(
            "POST",
            "/v1/agents/run",
            {"model": model_id, "goal": "Inspect repo summary", "max_steps": 1},
            timeout=60.0,
        )
        assert status == 200, f"{model_id} returned HTTP {status}: {data}"
        assert data.get("status") == "success", f"{model_id} agent run failed: {data}"


@pytest.mark.tier1
def test_t1_f3_agents_orchestrate_triune_base(studio_server: StudioServerHandle) -> None:
    """F3-T1-03: POST /v1/agents/orchestrate completes with local model='triune-base'."""
    status, data, _ = studio_server.request_json(
        "POST",
        "/v1/agents/orchestrate",
        {
            "model": "triune-base",
            "goal": "Check system layout",
            "personas": ["architect", "sentinel"],
            "mode": "parallel",
            "max_workers": 2,
            "max_steps": 1,
        },
        timeout=90.0,
    )
    assert status == 200, f"Expected HTTP 200, got {status}: {data}"
    assert data.get("status") == "success", f"Agent orchestrate failed: {data}"
    assert str(data.get("run_id", "")).startswith("orch-"), f"Unexpected run_id: {data}"
    assert data.get("worker_results") or data.get("results") or data.get("synthesis") is not None


@pytest.mark.tier1
def test_t1_f3_agents_loop_stream_triune_base(studio_server: StudioServerHandle) -> None:
    """F3-T1-04: POST /v1/agents/loop/stream emits start/done SSE events with model='triune-base'."""
    req = urllib.request.Request(
        f"{studio_server.url}/v1/agents/loop/stream",
        data=json.dumps({"model": "triune-base", "goal": "Inspect README.md", "mode": "react", "max_steps": 2}).encode(
            "utf-8"
        ),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        assert resp.status == 200
        ctype = resp.headers.get("content-type", "")
        assert "text/event-stream" in ctype.lower(), f"Expected SSE content-type, got {ctype}"
        raw = resp.read().decode("utf-8", errors="replace")

    assert "Unsupported agent model reference" not in raw, f"SSE stream failed with unsupported model:\n{raw}"
    assert "event: done" in raw or '"done"' in raw or "data:" in raw, f"Missing SSE events:\n{raw}"


@pytest.mark.tier1
def test_t1_f3_agents_run_auto_initializes_unloaded_model_and_records_history(
    isolated_plugin_studio_server: Any,
) -> None:
    """F3-T1-05: Fresh Studio server with unloaded pytorch_state auto-initializes 'triune-tiny' and logs history."""
    srv = isolated_plugin_studio_server()
    status, data, _ = srv.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "triune-tiny", "goal": "Ping workspace", "max_steps": 1},
        timeout=60.0,
    )
    assert status == 200, f"Auto-init agent run returned {status}: {data}"
    assert data.get("status") == "success", f"Auto-init agent run failed: {data}"

    h_status, history, _ = srv.request_json("GET", "/v1/agents/history", timeout=15.0)
    assert h_status == 200
    items = history if isinstance(history, list) else history.get("history", history.get("runs", []))
    assert any(item.get("run_id") == data.get("run_id") for item in items), (
        f"Run {data.get('run_id')} missing from /v1/agents/history: {history}"
    )


@pytest.mark.tier2
def test_t2_f3_unknown_model_returns_clean_error(studio_server: StudioServerHandle) -> None:
    """F3-T2-01: Unknown model reference returns clean error status without HTTP 500."""
    status, data, _ = studio_server.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "nonexistent-provider-xyz-999", "goal": "Test", "max_steps": 1},
        timeout=30.0,
    )
    assert status in (200, 400, 422), f"Unexpected server status {status}: {data}"
    assert isinstance(data, dict) and data.get("status") == "error", f"Expected clean error status: {data}"


@pytest.mark.tier2
def test_t2_f3_whitespace_model_handled_gracefully_without_500(isolated_plugin_studio_server: Any) -> None:
    """F3-T2-02: Whitespace model string on unloaded state returns structured JSON without HTTP 500."""
    srv = isolated_plugin_studio_server()
    status, data, _ = srv.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "   ", "goal": "Test", "max_steps": 1},
        timeout=30.0,
    )
    assert status < 500, f"Server crashed with {status}: {data}"
    assert isinstance(data, dict) and data.get("status") in {"success", "error"}


@pytest.mark.tier2
def test_t2_f3_allow_writes_blocked_by_default_403(studio_server: StudioServerHandle) -> None:
    """F3-T2-03: allow_writes=True is rejected with HTTP 403 unless LUCID_MACHINES_ALLOW_AGENT_WRITES=1."""
    for endpoint, body in (
        ("/v1/agents/run", {"model": "triune-base", "goal": "Write file", "allow_writes": True}),
        (
            "/v1/agents/orchestrate",
            {"model": "triune-base", "goal": "Write file", "personas": ["architect"], "allow_writes": True},
        ),
    ):
        status, data, _ = studio_server.request_json("POST", endpoint, body, timeout=20.0)
        assert status == 403, f"Expected HTTP 403 on {endpoint}, got {status}: {data}"
        assert data.get("status") == "error"
        assert "Security restriction" in str(data.get("detail", ""))


@pytest.mark.tier2
def test_t2_f3_max_steps_clamped_and_orchestrate_empty_personas_safe(
    studio_server: StudioServerHandle,
) -> None:
    """F3-T2-04: Extreme max_steps/max_workers and empty personas do not hang or HTTP 500."""
    s1, d1, _ = studio_server.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "triune-small", "goal": "Quick check", "max_steps": 999},
        timeout=60.0,
    )
    assert s1 < 500 and isinstance(d1, dict)

    s2, d2, _ = studio_server.request_json(
        "POST",
        "/v1/agents/orchestrate",
        {"model": "triune-small", "goal": "Quick check", "personas": [], "max_workers": 99, "max_steps": 1},
        timeout=60.0,
    )
    assert s2 < 500 and isinstance(d2, dict)


@pytest.mark.tier2
def test_t2_f3_remote_forwarded_client_rejected_without_token(studio_server: StudioServerHandle) -> None:
    """F3-T2-05: Proxied/non-loopback request without bearer token is rejected at auth boundary (401/503)."""
    status, data, _ = studio_server.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "triune-base", "goal": "Test auth", "max_steps": 1},
        headers={"X-Forwarded-For": "203.0.113.5"},
        timeout=20.0,
    )
    assert status in (401, 403, 503), f"Expected auth rejection (401/403/503), got {status}: {data}"


# ---------------------------------------------------------------------------
# Feature #4 — Scoped Plugin Discovery Compatibility (5 T1 + 5 T2)
# ---------------------------------------------------------------------------


_LEGACY_PLUGIN_PY = '''"""Legacy triune-namespaced plugin module."""
from triune.plugins.registry import register_node


@register_node(
    name="LegacyTriuneTestNode",
    category="Custom",
    description="Legacy plugin importing triune.plugins.registry",
    inputs={"x": "tensor"},
    outputs={"y": "tensor"},
)
def legacy_triune_test_node(x):
    return x
'''


@pytest.mark.tier1
def test_t1_f4_legacy_triune_plugin_discovers_cleanly(
    isolated_plugin_dir: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F4-T1-01: Plugin importing `from triune.plugins.registry import register_node` discovers and registers nodes."""
    mod_dir = isolated_plugin_dir / "modules" / "legacy-triune-node"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "node.py").write_text(_LEGACY_PLUGIN_PY, encoding="utf-8")

    probe = f"""
import importlib.util, json
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager
from lucid_machines.plugins.registry import global_registry

mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=True)
catalog_names = [n.get("name") for n in global_registry.get_catalog()]
print(json.dumps({{
    "nodes": nodes,
    "in_catalog": "LegacyTriuneTestNode" in catalog_names,
    "triune_spec_after": importlib.util.find_spec("triune") is not None,
}}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", probe], env=env, capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == 0, f"Probe failed:\n{proc.stdout}\n{proc.stderr}"
    assert "No module named 'triune'" not in (proc.stdout + proc.stderr)
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "LegacyTriuneTestNode" in info["nodes"], f"Legacy node not discovered: {info}"
    assert info["in_catalog"] is True


@pytest.mark.tier1
def test_t1_f4_global_triune_package_remains_absent_after_discovery(
    isolated_plugin_dir: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F4-T1-02: After scoped discovery, importlib.util.find_spec('triune') is None and repo triune/ is absent."""
    mod_dir = isolated_plugin_dir / "modules" / "legacy-triune-scope"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "node.py").write_text(_LEGACY_PLUGIN_PY, encoding="utf-8")

    probe = f"""
import importlib.util, json, sys
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager

mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=True)
print(json.dumps({{
    "nodes": nodes,
    "find_spec": importlib.util.find_spec("triune") is not None,
    "in_sys_modules": "triune" in sys.modules,
}}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", probe], env=env, capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert info["nodes"], "Expected legacy plugin nodes to load"
    assert info["find_spec"] is False, "Global triune spec leaked after discovery"
    assert info["in_sys_modules"] is False, "sys.modules['triune'] leaked after discovery"
    assert not (e2e_repo_root / "triune").exists()


@pytest.mark.tier1
def test_t1_f4_lucid_machines_plugin_json_preferred_over_triune_plugin_json(
    isolated_plugin_dir: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F4-T1-03: Manifest discovery reads lucid_machines_plugin.json before triune_plugin.json."""
    mod_dir = isolated_plugin_dir / "modules" / "dual-manifest-mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "preferred.py").write_text(
        """from lucid_machines.plugins.registry import register_node
@register_node(name="PreferredManifestNode", category="Custom", description="pref", inputs={"x": "tensor"}, outputs={"y": "tensor"})
def preferred_fn(x):
    return x
""",
        encoding="utf-8",
    )
    (mod_dir / "fallback_legacy.py").write_text(
        """from lucid_machines.plugins.registry import register_node
@register_node(name="FallbackTriuneManifestNode", category="Custom", description="fb", inputs={"x": "tensor"}, outputs={"y": "tensor"})
def fallback_fn(x):
    return x
""",
        encoding="utf-8",
    )
    (mod_dir / "lucid_machines_plugin.json").write_text(
        json.dumps({"manifest_version": 1, "version": "1.0.0", "entrypoints": ["preferred.py"]}),
        encoding="utf-8",
    )
    (mod_dir / "triune_plugin.json").write_text(
        json.dumps({"manifest_version": 1, "version": "1.0.0", "entrypoints": ["fallback_legacy.py"]}),
        encoding="utf-8",
    )

    probe = f"""
import json
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager

mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=True)
print(json.dumps(nodes))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", probe], env=env, capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == 0, proc.stderr
    nodes = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "PreferredManifestNode" in nodes, f"Expected lucid_machines_plugin.json entrypoint first, got: {nodes}"
    assert "FallbackTriuneManifestNode" not in nodes, f"Loaded triune_plugin.json instead of lucid_machines_plugin.json: {nodes}"


@pytest.mark.tier1
def test_t1_f4_modules_installed_populated_registered_nodes_and_clean_author(
    isolated_plugin_studio_server: Any,
) -> None:
    """F4-T1-04: Fresh isolated Studio server /v1/modules/installed populates registered_nodes and LucidMachines Research."""
    srv = isolated_plugin_studio_server()
    status, data, _ = srv.request_json("GET", "/v1/modules/installed", timeout=20.0)
    assert status == 200
    assert isinstance(data, list) and len(data) > 0, f"Expected starter installed modules, got: {data}"

    for mod in data:
        assert mod.get("author") != "Triune Research", f"Stale author 'Triune Research' in {mod}"

    custom_loss = next((m for m in data if m.get("id") == "custom-loss-nodes"), None)
    assert custom_loss is not None, f"Missing custom-loss-nodes in {data}"
    assert custom_loss.get("author") == "LucidMachines Research", custom_loss
    reg_nodes = custom_loss.get("registered_nodes", [])
    for expected in ("FocalCrossEntropyLoss", "CentroidBalanceLoss", "TemperatureAnnealer"):
        assert expected in reg_nodes, f"Missing {expected} in registered_nodes: {reg_nodes}"


@pytest.mark.tier1
def test_t1_f4_nodes_catalog_and_plugins_nodes_include_curated_nodes(
    isolated_plugin_studio_server: Any,
) -> None:
    """F4-T1-05: /v1/nodes/catalog and /v1/plugins/nodes return discovered curated plugin nodes."""
    srv = isolated_plugin_studio_server()
    for path in ("/v1/nodes/catalog", "/v1/plugins/nodes"):
        status, data, _ = srv.request_json("GET", path, timeout=20.0)
        assert status == 200, f"{path} failed: {status} {data}"
        items = data if isinstance(data, list) else data.get("nodes", data.get("catalog", []))
        names = {item.get("name") for item in items if isinstance(item, dict)}
        for expected in ("FocalCrossEntropyLoss", "CentroidBalanceLoss", "TemperatureAnnealer"):
            assert expected in names, f"{expected} missing from {path}: {sorted(names)}"


@pytest.mark.tier2
def test_t2_f4_untrusted_unsigned_plugin_rejected(
    isolated_plugin_dir: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F4-T2-01: Untrusted plugin without signed manifest returns 0 nodes and is not executed."""
    mod_dir = isolated_plugin_dir / "modules" / "unsigned-mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    marker = isolated_plugin_dir / "executed.flag"
    (mod_dir / "node.py").write_text(
        f"""from lucid_machines.plugins.registry import register_node
@register_node(name="UnsignedNode", category="Custom", description="x", inputs={{"x": "tensor"}}, outputs={{"y": "tensor"}})
def fn(x):
    return x
""",
        encoding="utf-8",
    )

    probe = f"""
import json
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager

mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=False)
print(json.dumps(nodes))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", probe], env=env, capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == 0, proc.stderr
    nodes = json.loads(proc.stdout.strip().splitlines()[-1])
    assert nodes == [], f"Untrusted unsigned plugin should be rejected, got: {nodes}"
    assert not marker.exists()


@pytest.mark.tier2
def test_t2_f4_ast_safety_gate_blocks_dangerous_imports_even_when_trusted(
    isolated_plugin_dir: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F4-T2-02: AST gate blocks subprocess/eval in plugins even when trusted=True."""
    mod_dir = isolated_plugin_dir / "modules" / "unsafe-ast-mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "evil_ast.py").write_text(
        """import subprocess
from lucid_machines.plugins.registry import register_node

@register_node(name="EvilAstNode", category="Custom", description="bad", inputs={"x": "tensor"}, outputs={"y": "tensor"})
def evil_fn(x):
    return eval("x")
""",
        encoding="utf-8",
    )

    probe = f"""
import json
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager

mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=True)
print(json.dumps(nodes))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", probe], env=env, capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == 0, proc.stderr
    nodes = json.loads(proc.stdout.strip().splitlines()[-1])
    assert nodes == [], f"Dangerous AST plugin should not load: {nodes}"


@pytest.mark.tier2
def test_t2_f4_symlink_or_junction_escape_inside_plugin_blocked(
    isolated_plugin_dir: Path, tmp_path: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F4-T2-03: Symlink/junction escape or path traversal inside plugin directory is blocked without skip."""
    outside_dir = tmp_path / "outside_plugin_target"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_py = outside_dir / "escaped_node.py"
    outside_py.write_text(
        """from lucid_machines.plugins.registry import register_node
@register_node(name="EscapedOutsideNode", category="Custom", description="esc", inputs={"x": "tensor"}, outputs={"y": "tensor"})
def esc_fn(x):
    return x
""",
        encoding="utf-8",
    )

    mod_dir = isolated_plugin_dir / "modules" / "escape-mod"
    mod_dir.mkdir(parents=True, exist_ok=True)

    link_created = False
    try:
        os.symlink(outside_py, mod_dir / "link_node.py")
        link_created = True
    except OSError:
        # Windows without Developer Mode: fall back to NTFS directory junction (no elevation needed)
        junction_path = mod_dir / "linked_sub"
        j_res = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction_path), str(outside_dir)],
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        link_created = j_res.returncode == 0 and junction_path.exists()

    if not link_created:
        # Traversal fallback in manifest
        (mod_dir / "lucid_machines_plugin.json").write_text(
            json.dumps({"manifest_version": 1, "version": "1.0.0", "entrypoints": ["../../outside_plugin_target/escaped_node.py"]}),
            encoding="utf-8",
        )

    probe = f"""
import json
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager

mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=True)
print(json.dumps(nodes))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", probe], env=env, capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == 0, proc.stderr
    nodes = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "EscapedOutsideNode" not in nodes, f"Plugin symlink/junction escape succeeded: {nodes}"


@pytest.mark.tier2
def test_t2_f4_manifest_sha256_tamper_and_traversal_rejected(
    isolated_plugin_dir: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F4-T2-04: Untrusted manifest with traversal entrypoint or wrong SHA-256 hash is rejected."""
    mod_dir = isolated_plugin_dir / "modules" / "tampered-manifest-mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    ep = mod_dir / "node.py"
    ep.write_text(
        """from lucid_machines.plugins.registry import register_node
@register_node(name="TamperedNode", category="Custom", description="t", inputs={"x": "tensor"}, outputs={"y": "tensor"})
def fn(x):
    return x
""",
        encoding="utf-8",
    )
    (mod_dir / "lucid_machines_plugin.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "version": "1.0.0",
                "entrypoints": ["../evil.py", "node.py"],
                "files": {"node.py": "0" * 64},
                "signature": "invalid",
            }
        ),
        encoding="utf-8",
    )

    probe = f"""
import json
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager, _verified_plugin_entrypoints

files, err = _verified_plugin_entrypoints(Path({str(mod_dir)!r}))
mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=False)
print(json.dumps({{"files": [str(f) for f in files], "err": err, "nodes": nodes}}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", probe], env=env, capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert info["err"], "Expected manifest verification error"
    assert info["nodes"] == [], f"Tampered manifest registered nodes: {info}"


@pytest.mark.tier2
def test_t2_f4_fallback_registry_clean_author_and_finally_cleanup_on_plugin_crash(
    isolated_plugin_dir: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F4-T2-05: manager.py fallback curated registry omits 'Triune Research' and cleans triune alias on crash."""
    from lucid_machines.modules import manager as mgr_mod

    manager_src = Path(mgr_mod.__file__).read_text(encoding="utf-8")
    assert '"author": "Triune Research"' not in manager_src, (
        "Stale 'Triune Research' author remains in lucid_machines/modules/manager.py"
    )

    mod_dir = isolated_plugin_dir / "modules" / "crashing-legacy-mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "crash_node.py").write_text(
        """from triune.plugins.registry import register_node

@register_node(name="WillCrash", category="Custom", description="c", inputs={"x": "tensor"}, outputs={"y": "tensor"})
def ok(x):
    return x

raise RuntimeError("boom during plugin import")
""",
        encoding="utf-8",
    )

    probe = f"""
import importlib.util, json, sys
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager

mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=True)
print(json.dumps({{
    "nodes": nodes,
    "find_spec": importlib.util.find_spec("triune") is not None,
    "in_sys_modules": "triune" in sys.modules,
}}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", probe], env=env, capture_output=True, text=True, timeout=30.0)
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    assert info["nodes"] == []
    assert info["find_spec"] is False, "Global triune alias leaked after plugin RuntimeError!"
    assert info["in_sys_modules"] is False, "sys.modules['triune'] leaked after plugin RuntimeError!"


# ---------------------------------------------------------------------------
# Feature #5 — Fast /v1/models, Non-Blocking Workspace State & Benchmarks API (5 T1 + 5 T2)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_f5_get_models_fast_default_under_100ms(studio_server: StudioServerHandle) -> None:
    """F5-T1-01: Default GET /v1/models returns presets & checkpoints in <100ms (no blocking PC/WSL scan)."""
    # Warm call (if initial TTL cache populates lazily on first request, test steady-state default)
    s0, d0, ms0 = studio_server.request_json("GET", "/v1/models", timeout=15.0)
    assert s0 == 200
    s1, d1, ms1 = studio_server.request_json("GET", "/v1/models", timeout=15.0)
    assert s1 == 200
    assert ms1 < 100.0, f"GET /v1/models took {ms1:.1f}ms (must be <100ms without PC/WSL scan)"
    ids = {m.get("id") for m in d1.get("data", [])}
    for required in ("triune-small", "triune-base", "junkie-mini"):
        assert required in ids, f"Missing preset {required} in /v1/models: {sorted(ids)}"


@pytest.mark.tier1
def test_t1_f5_post_models_scan_performs_explicit_scan(studio_server: StudioServerHandle) -> None:
    """F5-T1-02: POST /v1/models/scan executes explicit scan and returns structured scan payload."""
    status, data, _ = studio_server.request_json(
        "POST",
        "/v1/models/scan",
        {"include_wsl": False, "max_depth": 2},
        timeout=30.0,
    )
    assert status == 200, f"/v1/models/scan failed: {status} {data}"
    assert data.get("status") == "success"
    assert "models" in data and isinstance(data["models"], list)
    assert "counts_by_source" in data
    assert "scanned_roots" in data


@pytest.mark.tier1
def test_t1_f5_workspace_state_and_tree_non_blocking_fast(
    isolated_plugin_studio_server: Any, tmp_path: Path
) -> None:
    """F5-T1-03: /v1/workspace/state and /v1/workspace/tree?depth=2 respond quickly without blocking on git diff."""
    ws = tmp_path / "ws_repo"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (ws / "sub").mkdir()
    (ws / "sub" / "beta.md").write_text("# Beta\n", encoding="utf-8")

    srv = isolated_plugin_studio_server()
    op_s, op_d, _ = srv.request_json("POST", "/v1/workspace/open", {"path": str(ws)}, timeout=15.0)
    assert op_s == 200, op_d

    st_s, st_d, st_ms = srv.request_json("GET", "/v1/workspace/state", timeout=10.0)
    tr_s, tr_d, tr_ms = srv.request_json("GET", "/v1/workspace/tree?depth=2", timeout=10.0)
    assert st_s == 200 and tr_s == 200
    assert st_ms < 200.0, f"/v1/workspace/state took {st_ms:.1f}ms"
    assert tr_ms < 200.0, f"/v1/workspace/tree took {tr_ms:.1f}ms"
    assert "diff" not in st_d, "/v1/workspace/state should stay decoupled from git diff"
    names = {e.get("name") for e in tr_d.get("entries", [])}
    assert "alpha.py" in names and "sub" in names, f"Unexpected tree entries: {tr_d}"


@pytest.mark.tier1
def test_t1_f5_workspace_git_returns_dirty_metadata_and_diff(
    isolated_plugin_studio_server: Any, tmp_path: Path
) -> None:
    """F5-T1-04: GET /v1/workspace/git returns branch, dirty_count, and diff for a dirty git repo."""
    repo = tmp_path / "dirty_git_repo"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "e2e@lucidmachines.local"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "E2E Tester"], cwd=str(repo), check=True, capture_output=True)
    tracked = repo / "tracked.py"
    tracked.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True, capture_output=True)
    tracked.write_text("x = 2\n", encoding="utf-8")

    srv = isolated_plugin_studio_server()
    op_s, op_d, _ = srv.request_json("POST", "/v1/workspace/open", {"path": str(repo)}, timeout=15.0)
    assert op_s == 200, op_d

    gs, gd, _ = srv.request_json("GET", "/v1/workspace/git", timeout=15.0)
    assert gs == 200
    assert gd.get("is_git") is True
    assert gd.get("branch"), gd
    assert int(gd.get("dirty_count", 0)) >= 1, gd
    assert "+x = 2" in str(gd.get("diff", "")), gd


@pytest.mark.tier1
def test_t1_f5_engine_benchmarks_and_run_structured_payload(
    isolated_plugin_studio_server: Any,
) -> None:
    """F5-T1-05: POST /v1/engine/benchmarks/run and GET /v1/engine/benchmarks return structured comparison metrics."""
    srv = isolated_plugin_studio_server()
    run_status, run_data, _ = srv.request_json("POST", "/v1/engine/benchmarks/run", {}, timeout=90.0)
    assert run_status == 200, f"POST /v1/engine/benchmarks/run returned {run_status}: {run_data}"

    get_status, get_data, _ = srv.request_json("GET", "/v1/engine/benchmarks", timeout=15.0)
    assert get_status == 200, get_data
    expected_keys = {
        "triune_kernel_path",
        "unsloth_baseline",
        "standard_pytorch",
        "per_kernel_timings",
        "cross_entropy",
        "results",
    }
    assert any(k in get_data for k in expected_keys), f"Missing structured benchmark keys in {get_data}"


@pytest.mark.tier2
def test_t2_f5_models_scan_rejects_traversal_and_unc_extra_paths(
    studio_server: StudioServerHandle,
) -> None:
    """F5-T2-01: POST /v1/models/scan rejects path traversal and UNC share paths with HTTP 403."""
    for bad in ("../secret", r"\\evil-host\share"):
        status, data, _ = studio_server.request_json(
            "POST",
            "/v1/models/scan",
            {"extra_paths": [bad], "include_wsl": False},
            timeout=15.0,
        )
        assert status == 403, f"Expected HTTP 403 for {bad!r}, got {status}: {data}"


@pytest.mark.tier2
def test_t2_f5_workspace_tree_invalid_depth_and_traversal(
    studio_server: StudioServerHandle,
) -> None:
    """F5-T2-02: GET /v1/workspace/tree rejects out-of-range depth (422) and traversal path (403)."""
    for depth in (0, 99):
        status, data, _ = studio_server.request_json("GET", f"/v1/workspace/tree?depth={depth}", timeout=10.0)
        assert status == 422, f"Expected 422 for depth={depth}, got {status}: {data}"

    status_trav, data_trav, _ = studio_server.request_json(
        "GET", "/v1/workspace/tree?path=../outside&depth=1", timeout=10.0
    )
    assert status_trav == 403, f"Expected 403 for traversal path, got {status_trav}: {data_trav}"


@pytest.mark.tier2
def test_t2_f5_get_models_concurrent_burst_stays_fast_and_consistent(
    studio_server: StudioServerHandle,
) -> None:
    """F5-T2-03: 10 concurrent GET /v1/models requests all stay <100ms with identical preset lists."""
    # Ensure warm cache first
    studio_server.request_json("GET", "/v1/models", timeout=15.0)

    def _hit(_i: int) -> tuple[int, list[str], float]:
        s, d, ms = studio_server.request_json("GET", "/v1/models", timeout=15.0)
        presets = sorted(m["id"] for m in d.get("data", []) if m.get("kind") == "preset")
        return s, presets, ms

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_hit, range(10)))

    baseline = results[0][1]
    for s, presets, ms in results:
        assert s == 200
        assert ms < 100.0, f"Concurrent GET /v1/models took {ms:.1f}ms (>=100ms)"
        assert presets == baseline


@pytest.mark.tier2
def test_t2_f5_models_scan_include_wsl_false_skips_unc_roots(
    studio_server: StudioServerHandle,
) -> None:
    """F5-T2-04: POST /v1/models/scan with include_wsl=False scans zero \\\\wsl.localhost UNC roots."""
    status, data, _ = studio_server.request_json(
        "POST",
        "/v1/models/scan",
        {"include_wsl": False, "max_depth": 1},
        timeout=30.0,
    )
    assert status == 200
    roots = [str(r).lower() for r in data.get("scanned_roots", [])]
    assert not any("wsl.localhost" in r for r in roots), f"WSL root scanned despite include_wsl=False: {roots}"


@pytest.mark.tier2
def test_t2_f5_engine_benchmarks_missing_or_malformed_file_no_500(
    tmp_path: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F5-T2-05: GET /v1/engine/benchmarks returns clean JSON (never 500) when results file is missing or corrupt."""
    # Start a lightweight FastAPI server with cwd=tmp_path where benchmarks/benchmark_results.json is missing/corrupt
    from tests.e2e_suite.helpers import free_port, wait_http_ok

    port = free_port()
    env = os.environ.copy()
    env["LUCID_MACHINES_TESTING"] = "1"
    env["HOME"] = str(tmp_path)
    env["USERPROFILE"] = str(tmp_path)
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    code = (
        "import uvicorn\n"
        "from lucid_machines.api.server import create_app\n"
        "app = create_app()\n"
        f"uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')\n"
    )
    proc = subprocess.Popen(
        [e2e_python_exe, "-c", code],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_http_ok(f"http://127.0.0.1:{port}/healthz", timeout=45.0)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/engine/benchmarks", timeout=10.0) as resp:
            assert resp.status == 200
            d_missing = json.loads(resp.read().decode("utf-8"))
        assert isinstance(d_missing, dict)

        (tmp_path / "benchmarks").mkdir(exist_ok=True)
        (tmp_path / "benchmarks" / "benchmark_results.json").write_text("{malformed_json", encoding="utf-8")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/engine/benchmarks", timeout=10.0) as resp:
            assert resp.status == 200
            d_bad = json.loads(resp.read().decode("utf-8"))
        assert isinstance(d_bad, dict)
    finally:
        terminate_process(proc)
