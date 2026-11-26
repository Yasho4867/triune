"""M2 Backend Contract Tests: Agent Local Triune Completion, Scoped Plugin Discovery, Fast /v1/models, Non-Blocking Workspace State & Genuine Benchmarks."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from lucid_machines.api import routes as routes_mod
from lucid_machines.api import workspace_routes as ws_mod
from lucid_machines.api.server import create_app
from lucid_machines.modules.manager import ModuleManager, _scoped_triune_import_alias
from lucid_machines.plugins.registry import global_registry, register_node
from lucid_machines.runtime import model_scanner


def _snapshot_pytorch_state() -> dict:
    st = routes_mod.pytorch_state
    return {
        "model": st.model,
        "optimizer": getattr(st, "optimizer", None),
        "tokenizer": getattr(st, "tokenizer", None),
        "weights_loaded": st.weights_loaded,
        "active_model_id": st.active_model_id,
        "active_checkpoint": st.active_checkpoint,
        "active_project": st.active_project,
        "active_runtime": getattr(st, "active_runtime", "pytorch"),
        "ollama_model_tag": getattr(st, "ollama_model_tag", None),
        "architecture_spec": dict(st.architecture_spec),
        "explicit_architecture_active": st.explicit_architecture_active,
        "history": list(getattr(st, "history", [])),
        "last_exit_usage": getattr(st, "last_exit_usage", None),
        "last_loss": getattr(st, "last_loss", None),
        "gpu_layers": getattr(st, "gpu_layers", None),
        "gpu_offload_pct": getattr(st, "gpu_offload_pct", 100),
        "batch_size": getattr(st, "batch_size", 4),
        "seq_len": getattr(st, "seq_len", 512),
        "dag_run_history": deque(getattr(st, "dag_run_history", []), maxlen=50),
        "agent_run_history": deque(getattr(st, "agent_run_history", []), maxlen=50),
    }


def _restore_pytorch_state(snap: dict) -> None:
    st = routes_mod.pytorch_state
    if hasattr(st, "_detach_prev_streaming_engine"):
        st._detach_prev_streaming_engine()
    for key, val in snap.items():
        setattr(st, key, val)


class TestM2BackendContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls._session_file = cls.repo_root / "checkpoints" / "session_state.json"
        cls._session_bytes = cls._session_file.read_bytes() if cls._session_file.is_file() else None
        cls._bench_file = cls.repo_root / "benchmarks" / "benchmark_results.json"
        cls._bench_bytes = cls._bench_file.read_bytes() if cls._bench_file.is_file() else None
        cls.client = TestClient(create_app(), base_url="http://127.0.0.1", client=("127.0.0.1", 43210))

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._session_bytes is not None:
            cls._session_file.write_bytes(cls._session_bytes)
        if cls._bench_bytes is not None:
            cls._bench_file.write_bytes(cls._bench_bytes)

    def setUp(self) -> None:
        self._pt_snap = _snapshot_pytorch_state()
        self._nodes_snap = dict(global_registry._nodes)
        self._orig_ws_root = ws_mod.workspace_manager.root
        routes_mod._invalidate_model_scan_cache()
        with routes_mod._latest_live_benchmark_lock:
            self._live_bench_snap = routes_mod._latest_live_benchmark
            routes_mod._latest_live_benchmark = None

    def tearDown(self) -> None:
        _restore_pytorch_state(self._pt_snap)
        global_registry._nodes.clear()
        global_registry._nodes.update(self._nodes_snap)
        if ws_mod.workspace_manager.root != self._orig_ws_root:
            ws_mod.workspace_manager.open_workspace(str(self._orig_ws_root))
        with routes_mod._latest_live_benchmark_lock:
            routes_mod._latest_live_benchmark = self._live_bench_snap
        routes_mod._invalidate_model_scan_cache()
        if self._session_bytes is not None:
            self.assertEqual(self._session_file.read_bytes(), self._session_bytes)
        if self._bench_bytes is not None:
            self.assertEqual(self._bench_file.read_bytes(), self._bench_bytes)

    # ------------------------------------------------------------------
    # D1: Agent API Local Triune Completion & Resolver Parity
    # ------------------------------------------------------------------
    def test_d1_agent_backend_resolver_external_unknown_and_local_parity(self) -> None:
        for ext in (
            "gpt-4o",
            "o1-mini",
            "o3-mini",
            "openai/gpt-4o",
            "openrouter/auto",
            "claude-3-5-sonnet-latest",
            "anthropic/claude-3-5",
            "gemini-2.5-flash",
            "gemini/gemini-2.5",
            "ollama:gemma3:4b",
            "unknown-provider-xyz",
            "llama3:8b",
        ):
            m1, fn1 = routes_mod.resolve_agent_backend(ext)
            m2, fn2 = ws_mod._resolve_loop_backend(ext)
            self.assertEqual((m1, fn1), (ext, None))
            self.assertEqual((m2, fn2), (ext, None))

        # Reset to unloaded state -> omitted model returns (None, None)
        routes_mod.pytorch_state.model = None
        routes_mod.pytorch_state.weights_loaded = False
        self.assertEqual(routes_mod.resolve_agent_backend(None), (None, None))
        self.assertEqual(ws_mod._resolve_loop_backend(""), (None, None))

        # Local Triune preset auto-initializes without mutating session_state.json
        chosen, local_fn = routes_mod.resolve_agent_backend("triune-tiny")
        self.assertEqual(chosen, "triune-tiny")
        self.assertTrue(callable(local_fn))
        self.assertIsNotNone(routes_mod.pytorch_state.model)
        self.assertIsNotNone(routes_mod.pytorch_state.tokenizer)
        self.assertTrue(routes_mod.pytorch_state.explicit_architecture_active)

        # >1B preset falls back safely when unloaded
        routes_mod.pytorch_state.model = None
        routes_mod.pytorch_state.weights_loaded = False
        chosen_big, fn_big = routes_mod.resolve_agent_backend("junkie")
        self.assertEqual(chosen_big, "junkie")
        self.assertTrue(callable(fn_big))
        self.assertIsNotNone(routes_mod.pytorch_state.model)
        self.assertEqual(routes_mod.pytorch_state.active_model_id, "triune-small")

    def test_d1_agent_run_orchestrate_and_loop_stream_local_triune(self) -> None:
        fake_gen = lambda *a, **kw: {"text": "Local Triune analysis complete.", "route": "CORTEX"}
        with patch.object(routes_mod.pytorch_state, "generate_text", side_effect=fake_gen):
            r_run = self.client.post(
                "/v1/agents/run",
                json={"goal": "Inspect local readiness.", "model": "triune-base", "max_steps": 2},
            )
            self.assertEqual(r_run.status_code, 200, r_run.text)
            j_run = r_run.json()
            self.assertEqual(j_run["status"], "success")
            self.assertIn("Local Triune analysis complete.", str(j_run.get("answer", "")))

            r_orch = self.client.post(
                "/v1/agents/orchestrate",
                json={
                    "goal": "Team check.",
                    "model": "lucid/triune-base",
                    "personas": ["architect", "sentinel"],
                    "mode": "parallel",
                    "max_workers": 2,
                    "max_steps": 1,
                },
            )
            self.assertEqual(r_orch.status_code, 200, r_orch.text)
            j_orch = r_orch.json()
            self.assertEqual(j_orch["status"], "success")
            self.assertEqual(len(j_orch.get("results", [])), 2)
            self.assertTrue(all(w.get("ok") for w in j_orch["results"]))

            r_pipe = self.client.post(
                "/v1/agents/orchestrate",
                json={
                    "goal": "Pipeline check.",
                    "model": "junkie-mini",
                    "personas": ["architect", "sentinel"],
                    "mode": "pipeline",
                    "max_steps": 1,
                },
            )
            self.assertEqual(r_pipe.status_code, 200, r_pipe.text)
            self.assertEqual(r_pipe.json()["status"], "success")

            r_loop = self.client.post(
                "/v1/agents/loop/stream",
                json={"goal": "Loop check.", "model": "triune-base", "mode": "react", "max_steps": 2},
            )
            self.assertEqual(r_loop.status_code, 200)
            evs = [
                json.loads(line[len("data:") :].strip())
                for line in r_loop.text.splitlines()
                if line.strip().startswith("data:")
            ]
            self.assertEqual(evs[0]["type"], "start")
            self.assertEqual(evs[-1]["type"], "done")
            self.assertIn("Local Triune analysis complete.", str(evs[-1].get("answer", "")))

        # Unmocked real forward pass smoke on lightweight triune-tiny
        r_real = self.client.post(
            "/v1/agents/run",
            json={"goal": "Reply briefly.", "model": "triune-tiny", "max_steps": 1},
        )
        self.assertEqual(r_real.status_code, 200, r_real.text)
        self.assertEqual(r_real.json()["status"], "success")

    # ------------------------------------------------------------------
    # D2: Scoped Plugin Discovery Compatibility & Curated Metadata
    # ------------------------------------------------------------------
    def test_d2_scoped_triune_alias_cleans_up_and_preserves_global_absence(self) -> None:
        self.assertFalse((self.repo_root / "triune").exists())
        self.assertIsNone(importlib.util.find_spec("triune"))

        @register_node("DualSchemaCheckNode", category="Test")
        def _dual_node(x):
            return x

        self.assertTrue(hasattr(_dual_node, "_lucid_machines_node_schema"))
        self.assertTrue(hasattr(_dual_node, "_triune_node_schema"))

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "studio_cfg"
            mgr = ModuleManager(config_dir=cfg)
            legacy_dir = mgr.modules_dir / "custom-loss-nodes"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            # Legacy plugin importing from triune.* with stale metadata & dormant evil-pkg
            (legacy_dir / "custom_loss_nodes.py").write_text(
                "from triune.plugins.registry import register_node\n"
                "import triune.plugins.registry as reg_mod\n"
                "from triune import TriuneConfig\n"
                "@register_node('LegacyAliasTestNode', category='Loss')\n"
                "def _run(x): return (x, reg_mod, TriuneConfig)\n",
                encoding="utf-8",
            )
            (mgr.modules_dir / "evil-pkg").mkdir()
            mgr._save_installed_modules(
                [
                    {
                        "id": "custom-loss-nodes",
                        "name": "Old Name",
                        "author": "Triune Research",
                        "repo_url": "https://github.com/Yasho4867/triune",
                        "download_url": "builtin://custom-loss-nodes",
                        "type": "plugin",
                        "version": "1.2.0",
                        "installed_at": str(legacy_dir),
                        "registered_nodes": [],
                    }
                ]
            )
            # First test direct legacy triune import before curated sync
            nodes_legacy = mgr.discover_and_register_nodes(legacy_dir, trusted=True)
            self.assertIn("LegacyAliasTestNode", nodes_legacy)
            self.assertIsNone(importlib.util.find_spec("triune"))
            self.assertEqual(importlib.import_module("lucid_machines").__spec__.name, "lucid_machines")

            # Now test _auto_register_installed_nodes refreshes builtin:// copy + metadata + registered_nodes
            mgr._auto_register_installed_nodes()
            mods = {m["id"]: m for m in mgr.get_installed_modules()}
            cln = mods["custom-loss-nodes"]
            self.assertEqual(cln["author"], "LucidMachines Research")
            self.assertTrue(
                {"FocalCrossEntropyLoss", "CentroidBalanceLoss", "TemperatureAnnealer"}.issubset(
                    set(cln["registered_nodes"])
                )
            )
            self.assertIn(
                "from lucid_machines.plugins.registry import register_node",
                (legacy_dir / "custom_loss_nodes.py").read_text(encoding="utf-8"),
            )
            src_dummy = Path(tmp) / "src_curated"
            dest_dummy = Path(tmp) / "dest_curated"
            (src_dummy / "__pycache__").mkdir(parents=True)
            (src_dummy / "__pycache__" / "x.cpython-314.pyc").write_bytes(b"pyc")
            (src_dummy / "stale.pyc").write_bytes(b"pyc")
            (src_dummy / "node.py").write_text("X = 1\n", encoding="utf-8")
            dest_dummy.mkdir(parents=True)
            ModuleManager._copy_curated_dir_if_changed(src_dummy, dest_dummy)
            self.assertTrue((dest_dummy / "node.py").is_file())
            self.assertFalse((dest_dummy / "__pycache__").exists())
            self.assertFalse((dest_dummy / "stale.pyc").exists())
            self.assertEqual(list((mgr.modules_dir / "evil-pkg").iterdir()), [])
            self.assertIsNone(importlib.util.find_spec("triune"))

    def test_d2_scoped_alias_error_tampering_and_thread_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModuleManager(config_dir=Path(tmp) / "cfg")
            bad_dir = mgr.modules_dir / "bad-plugin"
            bad_dir.mkdir(parents=True)
            (bad_dir / "broken.py").write_text(
                "import sys\n"
                "sys.modules['triune'] = object()\n"
                "import triune.bogus_missing_submod\n",
                encoding="utf-8",
            )
            (bad_dir / "lucid_machines_plugin.json").write_text(
                json.dumps({"entrypoints": ["broken.py"]}), encoding="utf-8"
            )
            (bad_dir / "triune_plugin.json").write_text(
                json.dumps({"entrypoints": ["other.py"]}), encoding="utf-8"
            )
            nodes = mgr.discover_and_register_nodes(bad_dir, trusted=True)
            self.assertEqual(nodes, [])
            self.assertIsNone(importlib.util.find_spec("triune"))
            self.assertNotIn("triune", sys.modules)

        # Concurrent threads must still see find_spec('triune') is None while alias lock is held
        seen_other_thread = []
        entered = threading.Event()
        release = threading.Event()

        def _holder():
            with _scoped_triune_import_alias():
                entered.set()
                release.wait(timeout=3.0)

        t = threading.Thread(target=_holder)
        t.start()
        entered.wait(timeout=3.0)
        try:
            seen_other_thread.append(importlib.util.find_spec("triune"))
        finally:
            release.set()
            t.join(timeout=3.0)
        self.assertEqual(seen_other_thread, [None])
        self.assertIsNone(importlib.util.find_spec("triune"))

        # Check fallback curated registry author
        with tempfile.TemporaryDirectory() as tmp_fb:
            mgr_fb = ModuleManager(config_dir=Path(tmp_fb) / "cfg")
            with patch("pathlib.Path.is_file", return_value=False):
                fallback = mgr_fb._load_curated_registry()
        self.assertEqual(fallback[0]["author"], "LucidMachines Research")

    # ------------------------------------------------------------------
    # D3: Fast /v1/models, Non-Blocking /v1/workspace/state & Genuine Benchmarks
    # ------------------------------------------------------------------
    def test_d3_wsl_single_pass_and_fast_models_cache(self) -> None:
        with (
            patch.object(model_scanner, "query_ollama_running_tags", return_value=set()) as ol_mock,
            patch.object(model_scanner, "_collect_wsl_user_homes", return_value=[]) as wsl_mock,
        ):
            model_scanner.scan_pc_models(include_wsl=True, include_pc_scan=False)
            ol_mock.assert_not_called()
            wsl_mock.assert_not_called()

            model_scanner.scan_pc_models(include_wsl=True, include_pc_scan=True)
            self.assertEqual(ol_mock.call_count, 1)
            self.assertEqual(wsl_mock.call_count, 1)

        with patch.object(model_scanner, "query_ollama_running_tags", return_value=set()) as ol_mock:
            t0 = time.perf_counter()
            r_fast = self.client.get("/v1/models")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.assertEqual(r_fast.status_code, 200)
            j_fast = r_fast.json()
            self.assertFalse(j_fast["pc_scanned"])
            self.assertLess(elapsed_ms, 100.0)
            self.assertGreaterEqual(len(j_fast["data"]), 17)
            ol_mock.assert_not_called()

            r_scan = self.client.post("/v1/models/scan", json={"include_wsl": False})
            self.assertEqual(r_scan.status_code, 200)
            self.assertTrue(r_scan.json()["pc_scanned"])
            self.assertEqual(ol_mock.call_count, 1)

            r_warm = self.client.get("/v1/models")
            self.assertEqual(r_warm.status_code, 200)
            self.assertTrue(r_warm.json()["cached"])
            self.assertTrue(r_warm.json()["pc_scanned"])
            self.assertEqual(ol_mock.call_count, 1)

    def test_d3_workspace_state_non_blocking_zero_git_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "repo"
            (repo_dir / ".git").mkdir(parents=True)
            (repo_dir / ".git" / "HEAD").write_text("ref: refs/heads/feature/fast-state\n", encoding="utf-8")
            with patch("lucid_machines.api.workspace_routes.subprocess.run") as sub_mock:
                info = ws_mod._fast_git_head_info(repo_dir)
                self.assertEqual(info, {"is_git": True, "branch": "feature/fast-state"})
                sub_mock.assert_not_called()

            # Test .git file (worktree/submodule gitdir pointer)
            wt_dir = Path(tmp) / "worktree"
            wt_gitdir = Path(tmp) / "actual_gitdir"
            wt_dir.mkdir()
            wt_gitdir.mkdir()
            (wt_gitdir / "HEAD").write_text("ref: refs/heads/wt-branch\n", encoding="utf-8")
            (wt_dir / ".git").write_text(f"gitdir: {wt_gitdir}\n", encoding="utf-8")
            self.assertEqual(ws_mod._fast_git_head_info(wt_dir), {"is_git": True, "branch": "wt-branch"})

        with patch("lucid_machines.api.workspace_routes.subprocess.run") as sub_mock:
            t0 = time.perf_counter()
            r_st = self.client.get("/v1/workspace/state")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.assertEqual(r_st.status_code, 200)
            self.assertTrue(r_st.json()["git_deferred"])
            self.assertLess(elapsed_ms, 50.0)
            sub_mock.assert_not_called()

    def test_d3_benchmarks_normalized_live_run_and_cpu_honesty(self) -> None:
        # 1. Recorded artifact normalization
        r_art = self.client.get("/v1/engine/benchmarks")
        self.assertEqual(r_art.status_code, 200)
        j_art = r_art.json()
        self.assertEqual(j_art["source"], "benchmarks/benchmark_results.json")
        self.assertIn("cross_entropy", j_art)
        self.assertIn("triune_kernel_path", j_art)
        self.assertIn("per_kernel_timings", j_art)
        self.assertEqual(j_art["per_kernel_timings"]["fast_rope"]["status"], "not_in_artifact")

        # 2. CPU honesty path (mocked CUDA unavailable -> never fabricates Unsloth or VRAM savings)
        with patch.object(torch.cuda, "is_available", return_value=False):
            r_cpu = self.client.post("/v1/engine/benchmarks/run", json={"mode": "quick"})
        self.assertEqual(r_cpu.status_code, 200)
        j_cpu = r_cpu.json()
        self.assertEqual(j_cpu["status"], "measured")
        self.assertEqual(j_cpu["source"], "live_api_runner")
        self.assertEqual(j_cpu["device_type"], "cpu")
        self.assertFalse(j_cpu["vram_measured"])
        self.assertEqual(j_cpu["unsloth_baseline"]["status"], "unavailable")
        self.assertIsNone(j_cpu["triune_kernel_path"]["vram_reduction_pct"])
        self.assertEqual(
            set(j_cpu["per_kernel_timings"].keys()),
            {
                "fast_cross_entropy",
                "fast_rmsnorm",
                "fast_swiglu",
                "fast_gla",
                "fast_rope",
                "fast_muon",
                "fast_lora",
            },
        )

        # GET /v1/engine/benchmarks & /v1/engine/kernels reflect the live run
        r_get = self.client.get("/v1/engine/benchmarks")
        self.assertEqual(r_get.status_code, 200)
        self.assertEqual(r_get.json()["source"], "live_api_runner")

        r_kern = self.client.get("/v1/engine/kernels")
        self.assertEqual(r_kern.status_code, 200)
        self.assertEqual(r_kern.json()["benchmarks_vs_baselines"]["status"], "measured")


if __name__ == "__main__":
    unittest.main()
