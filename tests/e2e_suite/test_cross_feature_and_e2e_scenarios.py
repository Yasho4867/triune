"""R8 Preservation & Hygiene (5 T1 + 5 T2), Tier 3 Pairwise (12 tests), and Tier 4 E2E Scenarios (6 tests)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pytest

from .helpers import (
    STUDIO_TABS,
    STUDIO_THEMES,
    ChromeBrowserController,
    StudioCDPClient,
    StudioServerHandle,
    assert_r8_snapshot_intact,
    build_tiny_checkpoint,
    capture_r8_snapshot,
    run_kcx_cli,
    run_tectonic_scratch,
    run_website_tool,
    serve_static_dir,
    sha256_file,
)

GRADIENT_RE = re.compile(r"(?:linear|radial|conic)-gradient\s*\(", re.IGNORECASE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2600-\u27BF]")


# ---------------------------------------------------------------------------
# R8 Dedicated — Preservation & 4/5-Layer Naming Invariants (5 T1 + 5 T2)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_r8_preserved_checkpoints_data_exports_tokenizer_intact(e2e_repo_root: Path) -> None:
    """R8-T1-01: Preserved checkpoints, session_state.json (step 40555), data/, exports/, tokenizer intact."""
    session_file = e2e_repo_root / "checkpoints" / "session_state.json"
    rescued_ckpt = e2e_repo_root / "checkpoints" / "triune_studio_step_36500_rescued.pt"
    tok_file = e2e_repo_root / "triune_tokenizer.json"

    assert session_file.is_file()
    session_data = json.loads(session_file.read_text(encoding="utf-8"))
    assert session_data.get("step") == 40555, f"Unexpected step in session_state.json: {session_data}"
    assert session_data.get("active_checkpoint") == "triune_studio_step_36500_rescued.pt"

    assert rescued_ckpt.is_file() and rescued_ckpt.stat().st_size == 206_243_128
    assert tok_file.is_file() and tok_file.stat().st_size == 2_279_511

    data_dir = e2e_repo_root / "data"
    exports_dir = e2e_repo_root / "exports"
    assert (data_dir / "fineweb_sample.jsonl").is_file()
    assert exports_dir.is_dir() and any(exports_dir.iterdir())


@pytest.mark.tier1
def test_t1_r8_no_root_triune_shim_or_hmar_py(
    e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """R8-T1-02: Root triune/ shim and hmar.py are absent, and find_spec('triune') is None."""
    assert not (e2e_repo_root / "triune").exists()
    assert not (e2e_repo_root / "hmar.py").exists()

    proc = subprocess.run(
        [e2e_python_exe, "-c", "import importlib.util; assert importlib.util.find_spec('triune') is None"],
        cwd=str(e2e_repo_root),
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert proc.returncode == 0, f"triune package still importable:\n{proc.stderr}"


@pytest.mark.tier1
def test_t1_r8_ignored_artifacts_remain_untracked(e2e_repo_root: Path) -> None:
    """R8-T1-03: WORKSPACE_RULES.md, website/dist, website/node_modules, and *.dll remain untracked/ignored."""
    ls_res = subprocess.run(
        ["git", "ls-files", "WORKSPACE_RULES.md", "website/dist", "website/node_modules", "*.dll"],
        cwd=str(e2e_repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert ls_res.stdout.strip() == "", f"Forbidden ignored artifacts tracked in git:\n{ls_res.stdout}"

    for rel in ("WORKSPACE_RULES.md", "website/dist/", "website/node_modules/", "lucid_machines/ops/lucid_machines_core.dll"):
        ign = subprocess.run(
            ["git", "check-ignore", rel],
            cwd=str(e2e_repo_root),
            capture_output=True,
            text=True,
        )
        assert ign.returncode == 0, f"{rel} is not ignored by .gitignore"


@pytest.mark.tier1
def test_t1_r8_four_layer_naming_boundary_preserved(e2e_repo_root: Path) -> None:
    """R8-T1-04: pyproject.toml, lucid_machines exports, and models/configs preserve the 5-layer naming boundary."""
    import tomllib

    pyproj = tomllib.loads((e2e_repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproj["project"]["name"] == "lucid-machines"
    scripts = pyproj["project"]["scripts"]
    for alias in ("kcx", "lucid-machines", "polysom"):
        assert alias in scripts, f"Missing CLI script {alias!r} in pyproject.toml"

    import lucid_machines
    import lucid_machines.model as lm_model

    assert hasattr(lucid_machines, "TriuneTransformer") and hasattr(lucid_machines, "TriuneConfig")
    assert hasattr(lm_model, "TriuneTransformer")
    assert not hasattr(lm_model, "LucidTransformer")
    assert not hasattr(lm_model, "JunkieTransformer")

    cfg_dir = e2e_repo_root / "models" / "configs"
    assert (cfg_dir / "junkie.json").is_file()
    assert any(cfg_dir.glob("triune-*.json"))


@pytest.mark.tier1
def test_t1_r8_studio_src_zero_emojis_and_zero_gradients(e2e_repo_root: Path) -> None:
    """R8-T1-05: studio/src/ HTML, JS, CSS, and SVGs contain zero CSS gradients and zero emojis."""
    src_dir = e2e_repo_root / "studio" / "src"
    files = [
        src_dir / "index.html",
        src_dir / "app.js",
        src_dir / "style.css",
        *src_dir.rglob("*.svg"),
    ]
    for path in files:
        if not path.is_file():
            continue
        txt = path.read_text(encoding="utf-8", errors="replace")
        assert GRADIENT_RE.search(txt) is None, f"CSS gradient in {path}"
        assert EMOJI_RE.search(txt) is None, f"Emoji in {path}"


@pytest.mark.tier2
def test_t2_r8_no_stale_triune_shim_after_plugin_discovery(
    isolated_plugin_dir: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """R8-T2-01: Scoped legacy plugin discovery never leaves a global 'triune' package or sys.modules entry."""
    mod_dir = isolated_plugin_dir / "modules" / "r8-legacy-mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "node.py").write_text(
        """from triune.plugins.registry import register_node
@register_node(name="R8ScopedNode", category="Custom", description="r8", inputs={"x": "tensor"}, outputs={"y": "tensor"})
def fn(x):
    return x
""",
        encoding="utf-8",
    )
    script = f"""
import importlib.util, sys
from pathlib import Path
from lucid_machines.modules.manager import ModuleManager

mm = ModuleManager(config_dir={str(isolated_plugin_dir)!r})
nodes = mm.discover_and_register_nodes(Path({str(mod_dir)!r}), trusted=True)
assert "R8ScopedNode" in nodes, nodes
assert importlib.util.find_spec("triune") is None
assert "triune" not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    proc = subprocess.run([e2e_python_exe, "-c", script], env=env, capture_output=True, text=True, timeout=60.0)
    assert proc.returncode == 0, f"Scoped cleanup failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.tier2
def test_t2_r8_no_stale_root_triune_studio_binary(e2e_repo_root: Path) -> None:
    """R8-T2-02: No TriuneStudio* binaries remain in repo root, dist/, or studio/publish/."""
    assert not (e2e_repo_root / "TriuneStudio.exe").exists()
    for d in (e2e_repo_root / "dist", e2e_repo_root / "studio" / "publish"):
        if d.is_dir():
            assert list(d.glob("TriuneStudio*")) == []


@pytest.mark.tier2
def test_t2_r8_session_state_not_mutated_by_cli_read_commands(
    e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """R8-T2-03: Read-only kcx commands (doctor, models, checkpoint list, plan-memory, infer) keep session_state.json byte-identical."""
    session_file = e2e_repo_root / "checkpoints" / "session_state.json"
    before = sha256_file(session_file)

    for args in (
        ["doctor", "--json"],
        ["models", "--json"],
        ["checkpoint", "list", "--json"],
        ["plan-memory", "--vram-gb", "24", "--model", "junkie-3b"],
        ["infer", "-p", "Hello", "--max-tokens", "4", "--device", "cpu", "--no-stream"],
    ):
        proc = run_kcx_cli(
            e2e_python_exe,
            e2e_repo_root,
            args,
            cwd=e2e_repo_root,
            env_overrides={"HF_HUB_OFFLINE": "1"},
            timeout=90.0,
        )
        assert proc.returncode == 0, f"{args} failed:\n{proc.stdout}\n{proc.stderr}"

    assert sha256_file(session_file) == before


@pytest.mark.tier2
def test_t2_r8_no_tracked_checkpoints_or_secrets_in_git_index(e2e_repo_root: Path) -> None:
    """R8-T2-04: Git index tracks zero model weights (.pt/.pth/.ckpt/.safetensors) or .env secret files."""
    res = subprocess.run(
        ["git", "ls-files", "*.pt", "*.pth", "*.ckpt", "*.safetensors", ".env", ".env.*"],
        cwd=str(e2e_repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "", f"Tracked weights/secrets found:\n{res.stdout}"


@pytest.mark.tier2
def test_t2_r8_existing_baseline_suites_still_import_cleanly(
    e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """R8-T2-05: Existing test suites and tests/e2e_suite/ collect together with zero fixture/marker collisions."""
    proc = subprocess.run(
        [
            e2e_python_exe,
            "-m",
            "pytest",
            "--collect-only",
            "tests/e2e_suite",
            "tests/test_backward_compat.py",
            "tests/e2e/test_release_pipeline_e2e.py",
            "-q",
        ],
        cwd=str(e2e_repo_root),
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    assert proc.returncode == 0, f"Collection collision:\n{proc.stdout}\n{proc.stderr}"


# ---------------------------------------------------------------------------
# Tier 3 — Cross-Feature Pairwise Integration Matrix (12 tests: T3-01 .. T3-12)
# ---------------------------------------------------------------------------


@pytest.mark.tier3
def test_t3_01_active_checkpoint_shared_cli_and_agent_api(
    e2e_repo_root: Path, e2e_python_exe: str, studio_server: StudioServerHandle
) -> None:
    """T3-01 (F1 <-> F3): kcx infer auto-resolves active checkpoint and POST /v1/agents/run runs local triune-base."""
    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "--prompt", "Hello", "--max-tokens", "6", "--device", "cpu", "--no-stream", "--json"],
        cwd=e2e_repo_root,
        timeout=120.0,
    )
    assert proc.returncode == 0, f"kcx infer --json failed:\n{proc.stdout}\n{proc.stderr}"
    cli_data = json.loads(proc.stdout.strip())
    assert "triune_studio_step_36500_rescued.pt" in json.dumps(cli_data)

    status, agent_data, _ = studio_server.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "triune-base", "goal": "Summarize Triune exits", "max_steps": 1},
        timeout=60.0,
    )
    assert status == 200 and agent_data.get("status") == "success", agent_data


@pytest.mark.tier3
def test_t3_02_legacy_triune_plugin_to_marketplace_and_node_canvas(
    tmp_path: Path,
    isolated_plugin_studio_server: Any,
    cdp_browser: ChromeBrowserController,
) -> None:
    """T3-02 (F4 <-> F7): Legacy ~/.triune_studio metadata migrates cleanly, populates Marketplace DAG Nodes & Node Canvas."""
    temp_home = tmp_path / "legacy_home"
    legacy_studio = temp_home / ".triune_studio"
    (legacy_studio / "modules").mkdir(parents=True, exist_ok=True)
    # Seed legacy installed_modules.json with stale author to test migration + auto-discovery
    (legacy_studio / "installed_modules.json").write_text(
        json.dumps(
            [
                {
                    "id": "custom-loss-nodes",
                    "name": "Custom Loss DAG Nodes",
                    "author": "Triune Research",
                    "type": "plugin",
                    "version": "1.0.0",
                    "download_url": "builtin://custom-loss-nodes",
                    "installed_at": str(legacy_studio / "modules" / "custom-loss-nodes"),
                }
            ]
        ),
        encoding="utf-8",
    )

    srv = isolated_plugin_studio_server(home_dir=temp_home)
    st, installed, _ = srv.request_json("GET", "/v1/modules/installed", timeout=20.0)
    assert st == 200 and isinstance(installed, list)
    custom = next((m for m in installed if m.get("id") == "custom-loss-nodes"), None)
    assert custom is not None and custom.get("author") == "LucidMachines Research", custom
    assert "FocalCrossEntropyLoss" in custom.get("registered_nodes", []), custom

    cat_s, cat_d, _ = srv.request_json("GET", "/v1/nodes/catalog", timeout=15.0)
    assert cat_s == 200
    items = cat_d if isinstance(cat_d, list) else cat_d.get("nodes", cat_d.get("catalog", []))
    assert any(n.get("name") == "FocalCrossEntropyLoss" for n in items if isinstance(n, dict))

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await client.set_viewport(1440, 900, mobile=False)
        await client.navigate(f"{srv.url}/")
        await client.wait_for_studio_ready()
        await client.click_tab(6)  # modules
        await asyncio.sleep(0.4)
        mkt_txt = await client.eval_js("document.body.innerText || ''")
        await client.click_tab(2)  # nodegraph
        await asyncio.sleep(0.3)
        dag = await client.eval_js(
            """(() => {
              const minimap = document.querySelector('.ps-dag-minimap');
              const nodes = Array.from(document.querySelectorAll('.ps-dag-node, .dag-node'));
              const auto = nodes.find(n => (n.innerText || '').includes('Autograd Training Step'));
              if (!minimap || !auto) return { ok: false };
              const mr = minimap.getBoundingClientRect();
              const ar = auto.getBoundingClientRect();
              const ix = Math.max(0, Math.min(mr.right, ar.right) - Math.max(mr.left, ar.left));
              const iy = Math.max(0, Math.min(mr.bottom, ar.bottom) - Math.max(mr.top, ar.top));
              return { ok: true, overlap: ix * iy };
            })()"""
        )
        return {"mkt_txt": mkt_txt, "dag": dag}

    res = cdp_browser.run(_run)
    assert "Triune Research" not in res["mkt_txt"]
    assert "FocalCrossEntropyLoss" in res["mkt_txt"]
    assert res["dag"]["ok"] and res["dag"]["overlap"] == 0


@pytest.mark.tier3
def test_t3_03_six_themes_across_critical_tabs_1440x900(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """T3-03 (F6 <-> F7): All 6 themes across 8 critical tabs at 1440x900 satisfy contrast, ambient, & zero-overlap."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await client.set_viewport(1440, 900, mobile=False)
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        sigs = set()
        failures: List[str] = []
        for tid in STUDIO_THEMES:
            await client.select_theme(tid)
            amb = await client.inspect_ambient()
            if not amb.get("exists") or amb.get("opacity", 0) < 0.12 or not amb.get("notOccluded"):
                failures.append(f"Ambient visibility issue on {tid}: {amb}")
            sigs.add(amb.get("signature", ""))
            for tab_idx in (0, 1, 2, 4, 6, 7, 8, 9):
                await client.click_tab(tab_idx)
                audit = await client.audit_dom()
                if audit["overflowPx"] > 2 or audit["gradientHit"] or audit["emojiHit"] or audit["runtimeErrorBanner"]:
                    failures.append(f"{tid}/tab{tab_idx}: {audit}")
            if tid in ("mirage", "euphoria"):
                await client.click_tab(9)
                c_mz = await client.check_wcag_contrast([".transmutation-wizard-card .transmutation-title"])
                for it in c_mz:
                    if it["cr"] < 4.5:
                        failures.append(f"Light contrast on {tid} Model Zoo: {it}")
        return {"failures": failures, "unique_sigs": len(sigs)}

    res = cdp_browser.run(_run)
    assert res["failures"] == [], f"6-theme x critical-tab failures: {res['failures']}"
    assert res["unique_sigs"] == 6


@pytest.mark.tier3
def test_t3_04_fast_boot_waterfall_and_explicit_pc_scan(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """T3-04 (F5 <-> F7 <-> F8): Cold boot avoids waterfall/external calls; explicit Scan PC updates Model Zoo."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await client.set_viewport(1440, 900, mobile=False)
        await client.call("Network.clearBrowserCache")
        await client.navigate(f"{studio_server.url}/healthz")
        await client.eval_js("window.localStorage.clear()")
        client.reset_tracking()
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        await asyncio.sleep(1.5)
        boot_urls = [rec.url for rec in client.network_events.values()]

        await client.click_tab(9)  # models
        await client.wait_for_js("Boolean(document.querySelector('.transmutation-wizard-card'))", timeout=8.0)
        return {"boot_urls": boot_urls}

    res = cdp_browser.run(_run)
    urls = res["boot_urls"]
    assert sum(1 for u in urls if "polysom-logo.svg" in u) <= 1
    assert not any("huggingface.co" in u or "github.com" in u for u in urls)
    assert len(urls) < 24

    st, d, ms = studio_server.request_json("GET", "/v1/models", timeout=10.0)
    assert st == 200 and ms < 100.0
    st2, d2, _ = studio_server.request_json("POST", "/v1/models/scan", {"include_wsl": False, "max_depth": 1}, timeout=30.0)
    assert st2 == 200 and d2.get("status") == "success"


@pytest.mark.tier3
def test_t3_05_workspace_ide_autoload_and_agent_orchestrate(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """T3-05 (F3 <-> F5 <-> F7): Workspace IDE auto-loads tree rapidly & /v1/agents/orchestrate succeeds."""

    async def _run(client: StudioCDPClient) -> int:
        await client.set_viewport(1440, 900, mobile=False)
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        await client.click_tab(1)
        await client.wait_for_js("document.querySelectorAll('.ps-ide-tree-item').length >= 3", timeout=2.5)
        return int(await client.eval_js("document.querySelectorAll('.ps-ide-tree-item').length"))

    count = cdp_browser.run(_run)
    assert count >= 3

    st, data, _ = studio_server.request_json(
        "POST",
        "/v1/agents/orchestrate",
        {
            "model": "triune-base",
            "goal": "Inspect repo layout",
            "personas": ["architect", "reviewer"],
            "mode": "parallel",
            "max_steps": 1,
        },
        timeout=90.0,
    )
    assert st == 200 and data.get("status") == "success", data


@pytest.mark.tier3
def test_t3_06_system_hardware_benchmark_run_and_card_update(
    isolated_plugin_studio_server: Any, cdp_browser: ChromeBrowserController
) -> None:
    """T3-06 (F5 <-> F7): POST /v1/engine/benchmarks/run updates System & Hardware kernel cards cleanly."""
    srv = isolated_plugin_studio_server()
    rs, rd, _ = srv.request_json("POST", "/v1/engine/benchmarks/run", {}, timeout=90.0)
    assert rs == 200, rd

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await client.set_viewport(1440, 900, mobile=False)
        await client.navigate(f"{srv.url}/")
        await client.wait_for_studio_ready()
        await client.click_tab(8)  # environment
        await client.wait_for_js("Boolean(document.querySelector('.ps-env-summary-grid'))", timeout=8.0)
        return await client.eval_js(
            """(() => ({
              wordBreaks: Array.from(document.querySelectorAll('.ps-env-summary-grid .metric-value')).map(
                el => window.getComputedStyle(el).wordBreak
              ),
              hasCards: Boolean(document.querySelector('.kernel-deck-card, .benchmark-comparison-grid')),
            }))()"""
        )

    res = cdp_browser.run(_run)
    assert res["hasCards"] is True
    assert all(wb != "break-all" for wb in res["wordBreaks"])


@pytest.mark.tier3
def test_t3_07_packaging_checksums_match_website_download_ux(
    e2e_repo_root: Path, scratch_website_dist: Path, cdp_browser: ChromeBrowserController
) -> None:
    """T3-07 (F9 <-> F11): dist/SHA256SUMS.txt artifacts match Website download UX and CLI install commands."""
    sums = (e2e_repo_root / "dist" / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert "PolysomStudio-1.0.0-win-x64.zip" in sums

    with serve_static_dir(scratch_website_dist) as base_url:

        async def _run(client: StudioCDPClient) -> str:
            await client.set_viewport(1440, 900, mobile=False)
            await client.navigate(f"{base_url}/")
            await client.wait_for_js(
                "document.readyState === 'complete' && ((document.body && document.body.innerText) || '').length > 100",
                timeout=10.0,
            )
            return await client.eval_js("document.body.innerText || ''")

        txt = cdp_browser.run(_run)

    assert "PolysomStudio-1.0.0-win-x64.zip" in txt
    assert "kcx" in txt and "pip install lucid-machines" in txt


@pytest.mark.tier3
def test_t3_08_papers_pdf_sync_and_website_research_serving(
    e2e_repo_root: Path, scratch_website_dist: Path, tmp_path: Path, cdp_browser: ChromeBrowserController
) -> None:
    """T3-08 (F10 <-> F11): Scratch Tectonic builds sync with website/public/papers and serve HTTP 200 in #research."""
    for subdir, fname in (("paper1_centroidsteer", "centroidsteer.pdf"), ("paper2_triune_systems", "triune_architecture.pdf")):
        proc, _, _, pages = run_tectonic_scratch(e2e_repo_root, subdir, tmp_path)
        assert proc.returncode == 0 and pages >= 10
        assert sha256_file(e2e_repo_root / "papers" / subdir / fname) == sha256_file(scratch_website_dist / "papers" / fname)

    with serve_static_dir(scratch_website_dist) as base_url:

        async def _run(client: StudioCDPClient) -> List[str]:
            await client.set_viewport(1440, 900, mobile=False)
            await client.navigate(f"{base_url}/#research")
            await client.wait_for_js("(document.body.innerText || '').length > 100", timeout=10.0)
            return await client.eval_js("Array.from(document.querySelectorAll('a[href]')).map(a => a.getAttribute('href'))")

        hrefs = cdp_browser.run(_run)
        joined = " ".join(hrefs)
        assert "centroidsteer.pdf" in joined and "triune_architecture.pdf" in joined


@pytest.mark.tier3
def test_t3_09_cli_json_parity_with_studio_api(
    e2e_repo_root: Path, e2e_python_exe: str, studio_server: StudioServerHandle
) -> None:
    """T3-09 (F2 <-> F5): `kcx models --json` & `kcx checkpoint list --json` agree with Studio REST API."""
    p_models = run_kcx_cli(e2e_python_exe, e2e_repo_root, ["models", "--json"], cwd=e2e_repo_root, timeout=60.0)
    p_ckpts = run_kcx_cli(e2e_python_exe, e2e_repo_root, ["checkpoint", "list", "--json"], cwd=e2e_repo_root, timeout=60.0)
    assert p_models.returncode == 0, p_models.stderr
    assert p_ckpts.returncode == 0, p_ckpts.stderr

    cli_models = json.loads(p_models.stdout.strip())
    cli_ckpts = json.loads(p_ckpts.stdout.strip())

    s1, rest_models, _ = studio_server.request_json("GET", "/v1/models", timeout=15.0)
    s2, rest_ckpts, _ = studio_server.request_json("GET", "/v1/checkpoints", timeout=15.0)
    assert s1 == 200 and s2 == 200

    rest_ids = {m["id"] for m in rest_models.get("data", [])}
    assert "junkie-mini" in rest_ids
    assert "triune-base" in json.dumps(cli_models)
    assert "triune_studio_step_36500_rescued.pt" in json.dumps(cli_ckpts)
    assert "triune_studio_step_36500_rescued.pt" in json.dumps(rest_ckpts)


@pytest.mark.tier3
def test_t3_10_cli_autoresolve_with_stdin_eval_and_mqar(
    e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """T3-10 (F1 <-> F2): Checkpoint auto-resolve works together with stdin infer, offline eval --json, and mqar."""
    p_inf = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "--max-tokens", "4", "--seed", "42", "--system", "Be concise", "--device", "cpu", "--no-stream", "--json"],
        cwd=e2e_repo_root,
        input_text="Explain Triune\n",
        timeout=120.0,
    )
    assert p_inf.returncode == 0, p_inf.stderr
    json.loads(p_inf.stdout.strip())

    p_eval = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["eval", "--max-steps", "1", "--device", "cpu", "--json"],
        cwd=e2e_repo_root,
        env_overrides={"HF_HUB_OFFLINE": "1"},
        timeout=90.0,
    )
    assert p_eval.returncode == 0, p_eval.stderr
    json.loads(p_eval.stdout.strip())

    p_mqar = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["benchmark", "--suite", "mqar"],
        cwd=e2e_repo_root,
        timeout=120.0,
    )
    assert p_mqar.returncode == 0, p_mqar.stderr


@pytest.mark.tier3
def test_t3_11_packaging_and_cli_entrypoints_preserve_r8_no_triune_shim(
    e2e_repo_root: Path, tmp_path: Path
) -> None:
    """T3-11 (F2 <-> F4 <-> F9): Packaging bundles, installer strings, and 30+ plugin catalog preserve R8 (no triune shim)."""
    from lucid_machines.packaging import build_linux_native, build_macos_native
    from lucid_machines.plugins.registry import global_registry

    assert importlib.util.find_spec("triune") is None
    assert len(global_registry.get_catalog()) >= 30

    origin = tmp_path / "reloc_sentinel"
    origin.mkdir()
    lin = build_linux_native(workspace=origin)
    mac = build_macos_native(workspace=origin)
    assert "reloc_sentinel" not in Path(lin["launcher"]).read_text(encoding="utf-8")
    assert "reloc_sentinel" not in Path(mac["executable"]).read_text(encoding="utf-8")


@pytest.mark.tier3
def test_t3_12_full_stack_r8_preservation_invariant(
    e2e_repo_root: Path,
    e2e_python_exe: str,
    studio_server: StudioServerHandle,
    cdp_browser: ChromeBrowserController,
) -> None:
    """T3-12 (F1 <-> F3 <-> F8 <-> R8): Cross-layer CLI + Agent API + Chrome walk preserves all R8 hashes & git hygiene."""
    snap = capture_r8_snapshot(e2e_repo_root)

    p = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "-p", "Hi", "--max-tokens", "3", "--device", "cpu", "--no-stream"],
        cwd=e2e_repo_root,
        timeout=120.0,
    )
    assert p.returncode == 0

    st, ag, _ = studio_server.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "triune-small", "goal": "Ping", "max_steps": 1},
        timeout=60.0,
    )
    assert st == 200 and ag.get("status") == "success"

    async def _run(client: StudioCDPClient) -> None:
        await client.set_viewport(1440, 900, mobile=False)
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        for idx in (0, 6, 7, 8, 9):
            await client.click_tab(idx)

    cdp_browser.run(_run)
    assert_r8_snapshot_intact(e2e_repo_root, snap)
    assert not (e2e_repo_root / "triune").exists()
    assert not (e2e_repo_root / "hmar.py").exists()
    assert not (e2e_repo_root / "TriuneStudio.exe").exists()


@pytest.mark.tier3
def test_t3_13_plan_memory_junkie3b_colab_preset_and_model_config_consistency(
    e2e_repo_root: Path, e2e_python_exe: str, studio_server: StudioServerHandle
) -> None:
    """T3-13 (F2 <-> F5): MemoryPlanner / kcx plan-memory on junkie-3b agrees with /v1/models and Colab L4 preset."""
    from lucid_machines.runtime import MemoryPlanner

    st, models_payload, _ = studio_server.request_json("GET", "/v1/models", timeout=15.0)
    assert st == 200
    rest_ids = {m["id"] for m in models_payload.get("data", [])}
    assert "junkie-3b" in rest_ids, f"junkie-3b missing from GET /v1/models: {sorted(rest_ids)}"

    plan_24 = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=24.0)
    assert plan_24["total_params"] == 2_990_355_155
    assert plan_24["centroid_steer_static_vram_gb"] > 15.0
    assert plan_24["recommended_train_mode"] != "resident_bf16_train"

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["plan-memory", "--vram-gb", "24", "--model", "junkie-3b"],
        cwd=e2e_repo_root,
        timeout=45.0,
    )
    assert proc.returncode == 0, proc.stderr
    assert "resident_bf16_train" not in proc.stdout and "10.545 GB" not in proc.stdout

    nb = json.loads((e2e_repo_root / "cloud" / "lucid_machines_colab_runner.ipynb").read_text(encoding="utf-8"))
    nb_text = "\n".join(
        "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
        for c in nb.get("cells", [])
    )
    assert "expandable_segments:True" in nb_text
    assert "10.10 GiB" not in nb_text and "10.55 GB" not in nb_text


# ---------------------------------------------------------------------------
# Tier 4 — Real-World End-to-End Application Scenarios (6 tests: T4-S1 .. T4-S6)
# ---------------------------------------------------------------------------


@pytest.mark.tier4
def test_t4_s1_cold_boot_studio_theme_exploration_and_atlas_workflow(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """T4-S1: ML researcher cold-boots Studio, uses Neural Atlas, explores themes/effects, Model Zoo & Checkpoints."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await client.set_viewport(1440, 900, mobile=False)
        await client.call("Network.clearBrowserCache")
        await client.navigate(f"{studio_server.url}/healthz")
        await client.eval_js("window.localStorage.clear()")
        client.reset_tracking()
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        await asyncio.sleep(1.0)

        boot_urls = [rec.url for rec in client.network_events.values()]
        chat_w = await client.eval_js("document.querySelector('.chat-main-pane')?.getBoundingClientRect().width || 0")
        await client.eval_js("document.querySelector('.btn-stimulate-synapse')?.click()")

        # Appearance -> cycle themes & effects -> switch to Euphoria light theme
        await client.click_tab(7)
        await client.wait_for_js("document.querySelectorAll('.ps-effects-btn').length === 3", timeout=8.0)
        await client.select_theme("euphoria")
        amb = await client.inspect_ambient()

        # Model Zoo on Euphoria light theme
        await client.click_tab(9)
        await client.wait_for_js("Boolean(document.querySelector('.transmutation-wizard-card'))", timeout=8.0)
        mz_contrast = await client.check_wcag_contrast([".transmutation-wizard-card .transmutation-title"])

        # Checkpoints tab
        await client.click_tab(4)
        await client.wait_for_js("Boolean(document.querySelector('.ckpt-table-wrap'))", timeout=8.0)
        audit = await client.audit_dom()
        return {
            "boot_urls": boot_urls,
            "chat_w": chat_w,
            "amb": amb,
            "mz_contrast": mz_contrast,
            "audit": audit,
            "errors": client.console_errors + client.page_errors + client.http_failures,
        }

    res = cdp_browser.run(_run)
    assert res["errors"] == [], f"Runtime errors during T4-S1: {res['errors']}"
    assert sum(1 for u in res["boot_urls"] if "polysom-logo.svg" in u) <= 1
    assert not any("huggingface.co" in u or "github.com" in u for u in res["boot_urls"])
    assert res["chat_w"] >= 520.0
    assert res["amb"]["exists"] and res["amb"]["opacity"] >= 0.12
    assert all(c["cr"] >= 4.5 for c in res["mz_contrast"])
    assert res["audit"]["overflowPx"] <= 2 and not res["audit"]["gradientHit"] and not res["audit"]["emojiHit"]


@pytest.mark.tier4
def test_t4_s2_developer_workspace_ide_plugin_and_local_agent_orchestration(
    tmp_path: Path,
    isolated_plugin_studio_server: Any,
    cdp_browser: ChromeBrowserController,
) -> None:
    """T4-S2: Developer loads legacy ~/.triune_studio config, opens Workspace IDE, Marketplace, Node Canvas, & Agents."""
    temp_home = tmp_path / "dev_home"
    (temp_home / ".triune_studio" / "modules").mkdir(parents=True, exist_ok=True)
    srv = isolated_plugin_studio_server(home_dir=temp_home)

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await client.set_viewport(1440, 900, mobile=False)
        await client.navigate(f"{srv.url}/")
        await client.wait_for_studio_ready()
        await client.click_tab(1)  # workspace
        await client.wait_for_js("document.querySelectorAll('.ps-ide-tree-item').length >= 3", timeout=2.5)
        await client.click_tab(6)  # modules
        await asyncio.sleep(0.3)
        mkt_txt = await client.eval_js("document.body.innerText || ''")
        await client.click_tab(2)  # nodegraph
        await asyncio.sleep(0.3)
        dag_ok = await client.eval_js(
            """(() => {
              const mm = document.querySelector('.ps-dag-minimap');
              const auto = Array.from(document.querySelectorAll('.ps-dag-node, .dag-node')).find(n => (n.innerText || '').includes('Autograd Training Step'));
              if (!mm || !auto) return false;
              const mr = mm.getBoundingClientRect(), ar = auto.getBoundingClientRect();
              return Math.max(0, Math.min(mr.right, ar.right) - Math.max(mr.left, ar.left)) * Math.max(0, Math.min(mr.bottom, ar.bottom) - Math.max(mr.top, ar.top)) === 0;
            })()"""
        )
        return {"mkt_txt": mkt_txt, "dag_ok": dag_ok}

    res = cdp_browser.run(_run)
    assert "Triune Research" not in res["mkt_txt"]
    assert "FocalCrossEntropyLoss" in res["mkt_txt"]
    assert res["dag_ok"] is True

    s1, d1, _ = srv.request_json("POST", "/v1/agents/run", {"model": "triune-base", "goal": "Inspect repo", "max_steps": 1}, timeout=60.0)
    s2, d2, _ = srv.request_json(
        "POST",
        "/v1/agents/orchestrate",
        {"model": "triune-base", "goal": "Inspect repo", "personas": ["architect", "sentinel"], "max_steps": 1},
        timeout=90.0,
    )
    assert s1 == 200 and d1.get("status") == "success"
    assert s2 == 200 and d2.get("status") == "success"
    assert importlib.util.find_spec("triune") is None


@pytest.mark.tier4
def test_t4_s3_performance_engineer_benchmarks_and_kcx_cli_automation(
    e2e_repo_root: Path,
    e2e_python_exe: str,
    isolated_plugin_studio_server: Any,
    cdp_browser: ChromeBrowserController,
) -> None:
    """T4-S3: Performance engineer runs /v1/engine/benchmarks/run, inspects System & Hardware UI, & runs kcx CLI pipeline."""
    srv = isolated_plugin_studio_server()
    bs, bd, _ = srv.request_json("POST", "/v1/engine/benchmarks/run", {}, timeout=90.0)
    assert bs == 200, bd

    async def _run(client: StudioCDPClient) -> bool:
        await client.set_viewport(1440, 900, mobile=False)
        await client.navigate(f"{srv.url}/")
        await client.wait_for_studio_ready()
        await client.click_tab(8)
        await client.wait_for_js("Boolean(document.querySelector('.ps-env-summary-grid'))", timeout=8.0)
        return bool(await client.eval_js("Boolean(document.querySelector('.kernel-deck-card, .benchmark-comparison-grid'))"))

    assert cdp_browser.run(_run) is True

    for args in (
        ["doctor", "--json"],
        ["models", "--json"],
        ["checkpoint", "list", "--json"],
        ["plan-memory", "--vram-gb", "24", "--model", "junkie-3b"],
        ["infer", "--prompt", "Hello", "--max-tokens", "6", "--seed", "7", "--device", "cpu", "--no-stream", "--json"],
        ["eval", "--max-steps", "1", "--device", "cpu", "--json"],
        ["benchmark", "--suite", "mqar"],
    ):
        proc = run_kcx_cli(
            e2e_python_exe,
            e2e_repo_root,
            args,
            cwd=e2e_repo_root,
            env_overrides={"HF_HUB_OFFLINE": "1"},
            timeout=120.0,
        )
        assert proc.returncode == 0, f"{args} failed:\n{proc.stdout}\n{proc.stderr}"
        if args[0] == "plan-memory":
            assert "10.545 GB" not in proc.stdout and "resident_bf16_train" not in proc.stdout


@pytest.mark.tier4
def test_t4_s4_release_engineer_packaging_trust_and_os_aware_website(
    e2e_repo_root: Path, tmp_path: Path, cdp_browser: ChromeBrowserController
) -> None:
    """T4-S4: Release engineer checks dist/SHA256SUMS.txt, portable bundles, tsc/vite/oxfmt, and OS-aware Website UI."""
    from lucid_machines.packaging import build_linux_native, build_macos_native

    sums = (e2e_repo_root / "dist" / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for line in sums.splitlines():
        if not line.strip():
            continue
        digest, fname = line.split()[0].upper(), line.split()[-1].lstrip("*")
        assert sha256_file(e2e_repo_root / "dist" / fname) == digest

    build_linux_native(workspace=tmp_path / "pkg")
    build_macos_native(workspace=tmp_path / "pkg")

    assert run_website_tool(e2e_repo_root, "tsc", ["--noEmit"], timeout=60.0).returncode == 0
    web_out = tmp_path / "web_out"
    assert run_website_tool(e2e_repo_root, "vite", ["build", "--outDir", str(web_out)], timeout=120.0).returncode == 0
    assert run_website_tool(e2e_repo_root, "oxfmt", ["--check", "src"], timeout=60.0).returncode == 0

    with serve_static_dir(web_out) as base_url:

        async def _run(client: StudioCDPClient) -> Dict[str, Any]:
            await client.set_viewport(1440, 900, mobile=False)
            await client.set_user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130.0", "Win32")
            await client.navigate(f"{base_url}/")
            await client.wait_for_js(
                "document.readyState === 'complete' && ((document.body && document.body.innerText) || '').length > 100",
                timeout=10.0,
            )
            txt = await client.eval_js("document.body.innerText || ''")
            await client.set_viewport(390, 844, mobile=True)
            overflow = await client.eval_js("document.documentElement.scrollWidth - window.innerWidth")
            return {"txt": txt, "overflow": overflow, "errors": client.console_errors + client.page_errors + client.http_failures}

        res = cdp_browser.run(_run)

    assert res["errors"] == [] and res["overflow"] <= 2
    for term in ("Lucid Machines", "Triune", "Junkie", "Polysom Studio", "kcx", "SmartScreen", "contact@lucidmachines.co"):
        assert term in res["txt"]
    assert "research@lucidmachines.co" not in res["txt"].lower()


@pytest.mark.tier4
def test_t4_s5_academic_reviewer_papers_compilation_and_website_hosting(
    e2e_repo_root: Path, scratch_website_dist: Path, tmp_path: Path, cdp_browser: ChromeBrowserController
) -> None:
    """T4-S5: Academic reviewer compiles Papers 1 & 2 in scratch, checks #research UI, and downloads matching PDFs."""
    box_re = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\)")
    for subdir, fname in (("paper1_centroidsteer", "centroidsteer.pdf"), ("paper2_triune_systems", "triune_architecture.pdf")):
        proc, _, txt, pages = run_tectonic_scratch(e2e_repo_root, subdir, tmp_path)
        assert proc.returncode == 0 and pages >= 10
        assert all(float(m.group(1)) <= 2.0 for m in box_re.finditer(proc.stdout + "\n" + proc.stderr))
        assert "**" not in txt and "`" not in txt
        assert "founder@lucidmachines.co" in txt and "research@lucidmachines.co" not in txt.lower()

    with serve_static_dir(scratch_website_dist) as base_url:

        async def _run(client: StudioCDPClient) -> str:
            await client.set_viewport(1440, 900, mobile=False)
            await client.navigate(f"{base_url}/#research")
            await client.wait_for_js("(document.body.innerText || '').length > 100", timeout=10.0)
            return await client.eval_js("document.body.innerText || ''")

        sec_txt = cdp_browser.run(_run)
        assert "CentroidSteer" in sec_txt and "Triune" in sec_txt
        for subdir, fname in (("paper1_centroidsteer", "centroidsteer.pdf"), ("paper2_triune_systems", "triune_architecture.pdf")):
            with urllib.request.urlopen(f"{base_url}/papers/{fname}", timeout=10.0) as resp:
                assert resp.status == 200
                body = resp.read()
                assert body.startswith(b"%PDF-")


@pytest.mark.tier4
def test_t4_s6_full_preservation_and_repo_hygiene_audit(
    e2e_repo_root: Path,
    e2e_python_exe: str,
    studio_server: StudioServerHandle,
    cdp_browser: ChromeBrowserController,
    tmp_path: Path,
) -> None:
    """T4-S6: Full cross-layer smoke sweep confirms R8 byte-identity, zero shims/EXEs, and clean git hygiene."""
    snap = capture_r8_snapshot(e2e_repo_root)

    assert run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "-p", "Audit", "--max-tokens", "3", "--device", "cpu", "--no-stream"],
        cwd=e2e_repo_root,
        timeout=120.0,
    ).returncode == 0

    st, ag, _ = studio_server.request_json(
        "POST",
        "/v1/agents/run",
        {"model": "triune-base", "goal": "Audit", "max_steps": 1},
        timeout=60.0,
    )
    assert st == 200 and ag.get("status") == "success"

    async def _run(client: StudioCDPClient) -> None:
        await client.set_viewport(1440, 900, mobile=False)
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        for idx in range(len(STUDIO_TABS)):
            await client.click_tab(idx)

    cdp_browser.run(_run)
    run_tectonic_scratch(e2e_repo_root, "paper1_centroidsteer", tmp_path)
    run_website_tool(e2e_repo_root, "vite", ["build", "--outDir", str(tmp_path / "web_audit")], timeout=120.0)

    assert_r8_snapshot_intact(e2e_repo_root, snap)
    assert not (e2e_repo_root / "triune").exists()
    assert not (e2e_repo_root / "hmar.py").exists()
    assert not (e2e_repo_root / "TriuneStudio.exe").exists()


@pytest.mark.tier3
def test_t3_13_plan_memory_junkie3b_colab_preset_and_model_config_consistency(
    e2e_repo_root: Path, e2e_python_exe: str, studio_server: StudioServerHandle
) -> None:
    """T3-13 (F2 <-> F5): MemoryPlanner / kcx plan-memory on junkie-3b agrees with /v1/models and Colab L4 preset."""
    from lucid_machines.runtime import MemoryPlanner

    st, models_payload, _ = studio_server.request_json("GET", "/v1/models", timeout=15.0)
    assert st == 200
    rest_ids = {m["id"] for m in models_payload.get("data", [])}
    assert "junkie-3b" in rest_ids, f"junkie-3b missing from GET /v1/models: {sorted(rest_ids)}"

    plan_24 = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=24.0)
    assert plan_24["total_params"] == 2_990_355_155
    assert plan_24["centroid_steer_static_vram_gb"] > 15.0
    assert plan_24["recommended_train_mode"] != "resident_bf16_train"

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["plan-memory", "--vram-gb", "24", "--model", "junkie-3b"],
        cwd=e2e_repo_root,
        timeout=45.0,
    )
    assert proc.returncode == 0, proc.stderr
    assert "resident_bf16_train" not in proc.stdout and "10.545 GB" not in proc.stdout

    nb = json.loads((e2e_repo_root / "cloud" / "lucid_machines_colab_runner.ipynb").read_text(encoding="utf-8"))
    nb_text = "\n".join(
        "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
        for c in nb.get("cells", [])
    )
    assert "expandable_segments:True" in nb_text
    assert "10.10 GiB" not in nb_text and "10.55 GB" not in nb_text
    assert importlib.util.find_spec("triune") is None
    ls_ignored = subprocess.run(
        ["git", "ls-files", "WORKSPACE_RULES.md", "website/dist", "website/node_modules", "*.dll"],
        cwd=str(e2e_repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert ls_ignored.stdout.strip() == ""
