"""Comprehensive R4 Security Regression Test Suite for Lucid Machines / Triune Transformer.

Covers all R4 security boundaries:
1. Checkpoint & Adapter Deserialization (`weights_only=True` AST audit & crafted pickle rejection).
2. Strict Path Confinement across `/v1/models/load`, `/v1/checkpoints/load`, `/v1/datasets/*`,
   `/v1/models/export`, `/v1/finetune/*`, and DAG file nodes (`LocalFileReader`, `CheckpointManager`).
3. Reverse-Proxy Auth Enforcement (`Forwarded`, `X-Forwarded-For`, `X-Real-IP`, `LUCID_MACHINES_REQUIRE_AUTH=1`),
   CSRF/Origin & Host validation on HTTP & `/ws/telemetry`, security headers, payload caps, and rate limits.
4. SSRF Guards on dataset URL streaming & module downloads, and curated allowlist gating on `pip install`.
5. External plugin `trusted=False` default (Ed25519 signature required), `PythonSandbox` resource limits,
   agent `allow_writes` API gating, vLLM `trust_remote_code: False`, and Gemini header-only API key transmission.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _pkg_name() -> str:
    if importlib.util.find_spec("lucid_machines") is None:
        raise RuntimeError("lucid_machines package is required")
    return "lucid_machines"


def _import_mod(subpath: str) -> Any:
    return importlib.import_module(f"{_pkg_name()}.{subpath}")


class _MaliciousPicklePayload:
    def __reduce__(self):
        return (os.system, ("echo PICKLE_RCE_BLOCKED",))


class TestR4SecurityRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls._session_file = cls.repo_root / "checkpoints" / "session_state.json"
        cls._session_backup = cls._session_file.read_text(encoding="utf-8") if cls._session_file.is_file() else None
        server_mod = _import_mod("api.server")
        cls.app = server_mod.create_app()
        cls.client = TestClient(cls.app, client=("127.0.0.1", 43210))

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_session_backup", None) is not None:
            cls._session_file.write_text(cls._session_backup, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 1. Deserialization Hardening (weights_only=True)
    # -------------------------------------------------------------------------
    def test_01_ast_audit_all_torch_load_enforce_weights_only_true(self) -> None:
        pkg = _pkg_name()
        files_to_check = [
            self.repo_root / pkg / "trainer" / "finetune.py",
            self.repo_root / "scripts" / "rescue_checkpoint.py",
            self.repo_root / "benchmarks" / "verify_triangular_qr_residuals.py",
            self.repo_root / "scripts" / "benchmark_full_model_e2e.py",
        ]
        for fpath in files_to_check:
            self.assertTrue(fpath.is_file(), f"Missing expected file: {fpath}")
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("torch.load"):
                    kw_map = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
                    self.assertIn(
                        "weights_only",
                        kw_map,
                        f"torch.load missing weights_only=True at {fpath}:{node.lineno}",
                    )
                    val_node = kw_map["weights_only"]
                    self.assertTrue(
                        isinstance(val_node, ast.Constant) and val_node.value is True,
                        f"torch.load must set weights_only=True at {fpath}:{node.lineno}",
                    )

    def test_02_finetuner_load_adapter_rejects_malicious_pickle(self) -> None:
        finetune_mod = _import_mod("trainer.finetune")
        TriuneFineTuner = finetune_mod.TriuneFineTuner
        LoRAConfig = finetune_mod.LoRAConfig

        class _TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(16, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.q_proj(x)

        tuner = TriuneFineTuner(_TinyModel(), LoRAConfig(rank=4, alpha=8.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_pt = Path(tmpdir) / "adapter_model.pt"
            bad_pt.write_bytes(pickle.dumps({"q_proj.lora_A": _MaliciousPicklePayload()}))
            with self.assertRaises(Exception):
                tuner.load_adapter(bad_pt)

    # -------------------------------------------------------------------------
    # 2. Strict Path Confinement across API & DAG
    # -------------------------------------------------------------------------
    def test_03_api_path_confinement_blocks_traversal_and_outside_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outside_pt = tmp_path / "outside.pt"
            outside_pt.write_bytes(b"untrusted_weights")
            outside_jsonl = tmp_path / "outside.jsonl"
            outside_jsonl.write_text('{"text": "secret"}\n', encoding="utf-8")

            cases = [
                ("/v1/models/load", {"checkpoint_path": "../../outside.pt"}),
                ("/v1/models/load", {"model": "../../outside.pt"}),
                ("/v1/models/load", {"checkpoint_path": str(outside_pt)}),
                ("/v1/models/load", {"checkpoint_path": r"\\evil-server\share\model.pt"}),
                ("/v1/checkpoints/load", {"checkpoint_path": "../../outside.pt"}),
                ("/v1/checkpoints/load", {"filename": str(outside_pt)}),
                ("/v1/datasets/local/register", {"path": "../../outside.jsonl", "name": "escape"}),
                ("/v1/datasets/local/register", {"path": str(outside_jsonl), "name": "outside"}),
                ("/v1/datasets/select", {"dataset_path": "../../outside.jsonl"}),
                ("/v1/datasets/select", {"dataset_path": str(outside_jsonl)}),
                (
                    "/v1/models/export",
                    {"model_id": "triune-small", "format": "safetensors", "output_dir": str(tmp_path)},
                ),
                ("/v1/models/export", {"model_id": "../../escape", "format": "safetensors", "output_dir": "exports"}),
                ("/v1/finetune/start", {"adapter_name": "../../escape", "dataset_path": "data/fineweb_sample.jsonl"}),
                ("/v1/finetune/start", {"adapter_name": "valid_lora", "dataset_path": str(outside_jsonl)}),
                ("/v1/finetune/adapter", {"action": "attach", "path": str(tmp_path)}),
                ("/v1/finetune/adapter", {"action": "attach", "path": "../../outside_adapter"}),
                ("/v1/finetune/export", {"adapter_name": "../../outside_escape"}),
                ("/v1/finetune/export", {"adapter_path": str(tmp_path), "export_name": "ok_name"}),
                ("/v1/finetune/export", {"adapter_path": "checkpoints/adapters/test", "export_name": "../escape"}),
                ("/v1/training/config", {"dataset_path": "../../outside.jsonl"}),
                ("/v1/training/config", {"dataset_path": str(outside_jsonl)}),
                ("/v1/models/scan", {"extra_paths": ["../../outside"]}),
                ("/v1/models/scan", {"extra_paths": [r"\\evil-server\share\models"]}),
                ("/v1/models/upgrade_to_triune", {"model_name": "../../outside.pt"}),
                ("/v1/models/upgrade_to_triune", {"model_name": r"\\evil-server\share\model.pt"}),
                ("/v1/models/distill", {"teacher_source": "../../outside_teacher"}),
            ]
            for endpoint, payload in cases:
                with self.subTest(endpoint=endpoint, payload=payload):
                    resp = self.client.post(endpoint, json=payload)
                    self.assertIn(
                        resp.status_code,
                        (400, 403, 404, 422),
                        f"Expected 4xx rejection for {endpoint} with {payload}, got {resp.status_code}: {resp.text}",
                    )

    def test_04_dag_file_nodes_enforce_workspace_confinement(self) -> None:
        dag_mod = _import_mod("execution.dag")
        ExecutionEngine = dag_mod.ExecutionEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            outside_file = Path(tmpdir) / "leak.jsonl"
            outside_file.write_text('{"text": "secret"}\n', encoding="utf-8")

            res_traversal = ExecutionEngine._handle_data_node(
                {"id": "d1", "title": "Local File Reader", "params": {"file_path": "../../etc/passwd"}},
                {},
            )
            self.assertEqual(res_traversal["status"], "failed")
            self.assertIn("Security restriction", res_traversal["error"])

            res_outside = ExecutionEngine._handle_data_node(
                {"id": "d2", "title": "Local File Reader", "params": {"file_path": str(outside_file)}},
                {},
            )
            self.assertEqual(res_outside["status"], "failed")
            self.assertIn("Security restriction", res_outside["error"])

            dummy_model = nn.Linear(4, 4)
            res_ckpt = ExecutionEngine._handle_checkpoint_manager_node(
                {"id": "c1", "title": "CheckpointManager", "params": {"save_dir": str(tmpdir)}},
                {"model": dummy_model},
            )
            self.assertEqual(res_ckpt["status"], "failed")
            self.assertIn("Security restriction", res_ckpt["error"])

            res_ckpt_trav = ExecutionEngine._handle_checkpoint_manager_node(
                {"id": "c2", "title": "CheckpointManager", "params": {"save_dir": "../../escape"}},
                {"model": dummy_model},
            )
            self.assertEqual(res_ckpt_trav["status"], "failed")
            self.assertIn("Security restriction", res_ckpt_trav["error"])

            res_export = ExecutionEngine._handle_export_node(
                {"id": "e1", "title": "SafeTensorsExport", "params": {"filename": "../escape.safetensors"}},
                {"model": dummy_model},
            )
            self.assertEqual(res_export["status"], "failed")

    # -------------------------------------------------------------------------
    # 3. Reverse-Proxy Auth, CSRF/Origin, Host, WS Auth & Security Headers
    # -------------------------------------------------------------------------
    def test_05_reverse_proxy_headers_and_require_auth_enforce_bearer_token(self) -> None:
        token = "s" * 40
        for proxy_hdr in (
            {"X-Forwarded-For": "203.0.113.42"},
            {"X-Real-IP": "198.51.100.17"},
            {"Forwarded": "for=203.0.113.42;proto=https"},
        ):
            with patch.dict(os.environ, {"LUCID_MACHINES_API_TOKEN": "", "TRIUNE_API_TOKEN": ""}):
                unconf = self.client.get("/v1/system/config", headers=proxy_hdr)
                self.assertEqual(unconf.status_code, 503)

            with patch.dict(os.environ, {"LUCID_MACHINES_API_TOKEN": token, "TRIUNE_API_TOKEN": token}):
                unauth = self.client.get("/v1/system/config", headers=proxy_hdr)
                self.assertEqual(unauth.status_code, 401)

                wrong = self.client.get(
                    "/v1/system/config",
                    headers={**proxy_hdr, "Authorization": "Bearer wrong_token"},
                )
                self.assertEqual(wrong.status_code, 401)

                ok = self.client.get(
                    "/v1/system/config",
                    headers={**proxy_hdr, "Authorization": f"Bearer {token}"},
                )
                self.assertEqual(ok.status_code, 200)

        # LUCID_MACHINES_REQUIRE_AUTH=1 forces bearer auth even on loopback without proxy headers
        with patch.dict(
            os.environ,
            {"LUCID_MACHINES_REQUIRE_AUTH": "1", "LUCID_MACHINES_API_TOKEN": token, "TRIUNE_API_TOKEN": token},
        ):
            unauth_local = self.client.get("/v1/system/config")
            self.assertEqual(unauth_local.status_code, 401)
            auth_local = self.client.get("/v1/system/config", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(auth_local.status_code, 200)

    def test_06_csrf_origin_host_and_security_headers_and_rate_limit(self) -> None:
        # 1. Security headers present
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        self.assertEqual(hdrs.get("x-content-type-options"), "nosniff")
        self.assertEqual(hdrs.get("x-frame-options"), "DENY")
        self.assertEqual(hdrs.get("referrer-policy"), "no-referrer")
        self.assertIn("default-src 'self'", hdrs.get("content-security-policy", ""))

        # 2. Cross-origin POST blocked (403)
        csrf_resp = self.client.post(
            "/v1/chat/cancel",
            headers={"Origin": "https://evil.example.com"},
        )
        self.assertEqual(csrf_resp.status_code, 403)

        null_origin_resp = self.client.post(
            "/v1/chat/cancel",
            headers={"Origin": "null"},
        )
        self.assertEqual(null_origin_resp.status_code, 403)

        # Same-origin loopback POST allowed
        local_origin_resp = self.client.post(
            "/v1/chat/cancel",
            headers={"Origin": "http://127.0.0.1:8000"},
        )
        self.assertEqual(local_origin_resp.status_code, 200)

        # 3. Untrusted Host header blocked (DNS rebinding defense)
        rebind_resp = self.client.get(
            "/v1/system/config",
            headers={"Host": "evil-rebind.attacker.example"},
        )
        self.assertEqual(rebind_resp.status_code, 403)

        # 4. Payload size limit (413)
        with patch.dict(os.environ, {"LUCID_MACHINES_MAX_REQUEST_BYTES": "1024"}):
            big_resp = self.client.post(
                "/v1/tokenizer/tokenize",
                json={"text": "A" * 2048},
            )
            self.assertEqual(big_resp.status_code, 413)

        # 5. Rate limit enforcement (429)
        with patch.dict(os.environ, {"LUCID_MACHINES_RATE_LIMIT_PER_MIN": "3"}):
            statuses = [
                self.client.get("/healthz", headers={"X-Real-IP": "198.51.100.99"}).status_code for _ in range(5)
            ]
            self.assertIn(429, statuses)

    def test_07_websocket_telemetry_rejects_bad_origin_and_proxied_unauth(self) -> None:
        # Cross-origin WebSocket handshake must close with 1008
        with self.assertRaises(WebSocketDisconnect) as ctx_origin:
            with self.client.websocket_connect(
                "/ws/telemetry",
                headers={"Origin": "https://evil.example.com"},
            ):
                pass
        self.assertEqual(ctx_origin.exception.code, 1008)

        # Reverse-proxied WebSocket without valid token must close with 1008
        token = "w" * 40
        with patch.dict(os.environ, {"LUCID_MACHINES_API_TOKEN": token, "TRIUNE_API_TOKEN": token}):
            with self.assertRaises(WebSocketDisconnect) as ctx_auth:
                with self.client.websocket_connect(
                    "/ws/telemetry",
                    headers={"X-Forwarded-For": "203.0.113.77"},
                ) as ws:
                    ws.send_text("wrong-token")
                    ws.receive_text()
            self.assertEqual(ctx_auth.exception.code, 1008)

    # -------------------------------------------------------------------------
    # 4. SSRF Guards & Curated Pip Allowlist
    # -------------------------------------------------------------------------
    def test_08_ssrf_blocked_on_dataset_streaming_and_module_downloads(self) -> None:
        for ssrf_url in (
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:9999/secret.jsonl",
            "http://10.0.0.1/internal.jsonl",
            "http://192.168.1.1/admin.jsonl",
        ):
            r1 = self.client.post("/v1/datasets/select", json={"dataset_path": ssrf_url})
            self.assertEqual(r1.status_code, 403, f"Expected 403 SSRF block on /v1/datasets/select for {ssrf_url}")

            r2 = self.client.post("/v1/datasets/stream/connect", json={"dataset_name": ssrf_url})
            self.assertEqual(
                r2.status_code, 403, f"Expected 403 SSRF block on /v1/datasets/stream/connect for {ssrf_url}"
            )

        mgr_mod = _import_mod("modules.manager")
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = mgr_mod.ModuleManager(config_dir=Path(tmpdir) / "cfg")
            dl_res = mgr.install_module(
                {
                    "id": "ssrf-weights",
                    "name": "SSRF Weights",
                    "download_url": "http://169.254.169.254/latest/model.safetensors",
                    "type": "model",
                }
            )
            self.assertFalse(dl_res["success"])

            clone_res = mgr.clone_repository("http://169.254.169.254/repo.git", custom_name="ssrf-repo")
            self.assertFalse(clone_res["success"])

            # Arbitrary pip package not in allowlist must be blocked unless LUCID_MACHINES_ALLOW_PIP_INSTALL=1
            with patch("subprocess.check_call") as mock_pip:
                pip_res = mgr.install_module({"id": "unlisted-pkg", "package_name": "arbitrary-unlisted-pkg"})
                self.assertFalse(pip_res["success"])
                self.assertIn("allowlist", pip_res["message"])
                mock_pip.assert_not_called()

    # -------------------------------------------------------------------------
    # 5. Plugin Trust, Sandbox Limits, Agent Writes, vLLM & Gemini Header
    # -------------------------------------------------------------------------
    def test_09_external_plugins_default_untrusted_and_require_ed25519(self) -> None:
        mgr_mod = _import_mod("modules.manager")
        loader_mod = _import_mod("plugins.loader")
        mm = mgr_mod.ModuleManager()

        self.assertIs(inspect.signature(mm.discover_and_register_nodes).parameters["trusted"].default, False)
        self.assertIs(inspect.signature(mm.scan_workspace_plugins).parameters["trusted"].default, False)
        self.assertIs(inspect.signature(loader_mod.load_plugin_nodes).parameters["trusted"].default, False)

        with tempfile.TemporaryDirectory() as tmpdir:
            ext_dir = Path(tmpdir) / "unsigned_plugin"
            ext_dir.mkdir()
            (ext_dir / "node.py").write_text(
                f"from {_pkg_name()}.plugins.registry import register_node\n"
                "@register_node(name='UnsignedTestNode', category='Custom')\n"
                "def run(x): return x\n",
                encoding="utf-8",
            )
            discovered = mm.discover_and_register_nodes(ext_dir)
            self.assertEqual(discovered, [])

            # Spoofed .../lucid_machines/plugins path outside the real package root must remain untrusted
            spoof_dir = Path(tmpdir) / _pkg_name() / "plugins"
            spoof_dir.mkdir(parents=True)
            (spoof_dir / "spoof_node.py").write_text(
                f"from {_pkg_name()}.plugins.registry import register_node\n"
                "@register_node(name='SpoofedInternalNode', category='Custom')\n"
                "def run(x): return x\n",
                encoding="utf-8",
            )
            self.assertEqual(mm.discover_and_register_nodes(spoof_dir), [])

    def test_10_sandbox_limits_agent_writes_vllm_and_gemini_header(self) -> None:
        # 1. Agent allow_writes=True blocked via API by default
        with patch.dict(os.environ, {"LUCID_MACHINES_ALLOW_AGENT_WRITES": "", "TRIUNE_ALLOW_AGENT_WRITES": ""}):
            agent_resp = self.client.post(
                "/v1/agents/run",
                json={"goal": "Write a file", "allow_writes": True},
            )
            self.assertEqual(agent_resp.status_code, 403)

            orch_resp = self.client.post(
                "/v1/agents/orchestrate",
                json={"goal": "Write a file", "allow_writes": True},
            )
            self.assertEqual(orch_resp.status_code, 403)

        # 2. Remote / proxied sandbox execution blocked by default
        token = "x" * 40
        with patch.dict(
            os.environ,
            {
                "LUCID_MACHINES_API_TOKEN": token,
                "TRIUNE_API_TOKEN": token,
                "LUCID_MACHINES_ALLOW_REMOTE_SANDBOX": "",
                "TRIUNE_ALLOW_REMOTE_SANDBOX": "",
            },
        ):
            sb_resp = self.client.post(
                "/v1/sandbox/run",
                json={"code": "x = 1 + 1"},
                headers={"X-Forwarded-For": "203.0.113.55", "Authorization": f"Bearer {token}"},
            )
            self.assertEqual(sb_resp.status_code, 403)

        # 3. PythonSandbox enforces memory cap (e.g. 64 MB cap rejects 256 MB allocation)
        sandbox_mod = _import_mod("runtime.sandbox")
        sb = sandbox_mod.PythonSandbox(timeout=5, max_memory_mb=64)
        mem_res = sb.execute_code("big = 'A' * (256 * 1024 * 1024)\nx = len(big)")
        self.assertFalse(
            mem_res.get("success", False), f"Expected sandbox memory limit to reject 256MB alloc: {mem_res}"
        )

        # 4. vLLM bridge defaults trust_remote_code to False
        vllm_mod = _import_mod("export.vllm_bridge")
        vllm_cfg = vllm_mod.build_vllm_serving_config("triune-small")
        self.assertIs(vllm_cfg["trust_remote_code"], False)

        # 5. Gemini ProviderManager sends API key via x-goog-api-key header, never in URL query string
        providers_mod = _import_mod("agents.providers")
        pm = providers_mod.ProviderManager()
        pm.register_provider("gemini", api_key="gemini-secret-header-only-key")

        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}).encode(
            "utf-8"
        )
        fake_resp.__enter__.return_value = fake_resp

        with patch("urllib.request.urlopen", return_value=fake_resp) as mock_urlopen:
            pm.complete_chat("gemini-2.5-flash", [{"role": "user", "content": "ping"}])
            req_obj = mock_urlopen.call_args.args[0]
            self.assertNotIn("key=", req_obj.full_url)
            self.assertNotIn("gemini-secret-header-only-key", req_obj.full_url)
            headers_lower = {k.lower(): v for k, v in req_obj.header_items()}
            self.assertEqual(headers_lower.get("x-goog-api-key"), "gemini-secret-header-only-key")

        # 6. Agent tools.py never invokes subprocess with shell=True
        tools_path = self.repo_root / _pkg_name() / "agents" / "tools.py"
        tools_tree = ast.parse(tools_path.read_text(encoding="utf-8"), filename=str(tools_path))
        for node in ast.walk(tools_tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell":
                        self.assertFalse(
                            isinstance(kw.value, ast.Constant) and kw.value.value is True,
                            f"shell=True must not be used in {tools_path}:{node.lineno}",
                        )


if __name__ == "__main__":
    unittest.main()
