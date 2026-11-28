"""Feature and regression test suite for Polysom Studio Phase 2 upgrades.

Covers:
1. Telemetry endpoints (/v1/telemetry/gpu, /v1/telemetry/mind)
2. Token Logprob & Shannon Entropy Inspector (/v1/tokenizer/inspect, /v1/tokenizer/tokenize, logprobs on /v1/chat/completions)
3. Model Architecture Configs & Security confinement (/v1/configs, /v1/configs/{config_name})
4. Junkie Flagship Launchpad (/v1/training/launch_junkie)
5. Confined Dataset Preview (/v1/datasets/preview)
6. DAG History & Replay (/v1/dag/history, /v1/dag/replay/{run_id})
7. Agent Studio Run, Orchestrate, and History (/v1/agents/state, /v1/agents/run, /v1/agents/orchestrate, /v1/agents/history)
8. Frontend UI wiring, 6-theme catalog (including Euphoria), 34 .ps-* classes, and hygiene invariants
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import re
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from lucid_machines.api.routes import pytorch_state
from lucid_machines.api.server import create_app

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "studio/src/app.js").read_text(encoding="utf-8")
STYLE_CSS = (ROOT / "studio/src/style.css").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "studio/src/index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def studio_client():
    session_file = ROOT / "checkpoints/session_state.json"
    session_backup = session_file.read_text(encoding="utf-8") if session_file.is_file() else None
    client_kwargs = {}
    if "client" in inspect.signature(TestClient).parameters:
        client_kwargs["client"] = ("127.0.0.1", 43211)
    client = TestClient(create_app(), **client_kwargs)
    yield client
    if session_backup is not None:
        session_file.write_text(session_backup, encoding="utf-8")


def _snapshot_pytorch_state():
    return {
        "model": pytorch_state.model,
        "optimizer": getattr(pytorch_state, "optimizer", None),
        "weights_loaded": pytorch_state.weights_loaded,
        "active_model_id": pytorch_state.active_model_id,
        "active_checkpoint": pytorch_state.active_checkpoint,
        "active_project": pytorch_state.active_project,
        "architecture_spec": dict(pytorch_state.architecture_spec),
        "explicit_architecture_active": pytorch_state.explicit_architecture_active,
        "grad_accum_steps": pytorch_state.grad_accum_steps,
        "depth_mode": pytorch_state.depth_mode,
        "precision": getattr(pytorch_state, "precision", "bf16"),
        "gpu_layers": getattr(pytorch_state, "gpu_layers", None),
        "gpu_offload_pct": getattr(pytorch_state, "gpu_offload_pct", 100),
        "is_training": pytorch_state.is_training,
        "dag_run_history": deque(getattr(pytorch_state, "dag_run_history", []), maxlen=50),
        "agent_run_history": deque(getattr(pytorch_state, "agent_run_history", []), maxlen=50),
    }


def _restore_pytorch_state(snap):
    if hasattr(pytorch_state, "_detach_prev_streaming_engine"):
        pytorch_state._detach_prev_streaming_engine()
    pytorch_state.model = snap["model"]
    pytorch_state.optimizer = snap["optimizer"]
    pytorch_state.weights_loaded = snap["weights_loaded"]
    pytorch_state.active_model_id = snap["active_model_id"]
    pytorch_state.active_checkpoint = snap["active_checkpoint"]
    pytorch_state.active_project = snap["active_project"]
    pytorch_state.architecture_spec = snap["architecture_spec"]
    pytorch_state.explicit_architecture_active = snap["explicit_architecture_active"]
    pytorch_state.grad_accum_steps = snap["grad_accum_steps"]
    pytorch_state.depth_mode = snap["depth_mode"]
    pytorch_state.precision = snap["precision"]
    pytorch_state.gpu_layers = snap["gpu_layers"]
    pytorch_state.gpu_offload_pct = snap["gpu_offload_pct"]
    pytorch_state.is_training = snap["is_training"]
    pytorch_state.dag_run_history = deque(snap["dag_run_history"], maxlen=50)
    pytorch_state.agent_run_history = deque(snap["agent_run_history"], maxlen=50)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_gpu_and_mind_telemetry_endpoints(studio_client):
    res_gpu = studio_client.get("/v1/telemetry/gpu")
    assert res_gpu.status_code == 200
    gpu = res_gpu.json()
    assert gpu["status"] == "ok"
    assert "device_name" in gpu
    assert {"allocated_gb", "reserved_gb", "total_gb", "oom_risk"} <= set(gpu["vram"].keys())

    res_mind = studio_client.get("/v1/telemetry/mind")
    assert res_mind.status_code == 200
    mind = res_mind.json()
    assert mind["status"] in {"unloaded", "idle", "active"}
    assert "triune_tiers" in mind
    assert {"depth_mode", "active_tier", "exit_usage", "target_exit_distribution", "router_history"} <= set(
        mind["triune_tiers"].keys()
    )
    assert "centroid_steer" in mind
    assert "moe_routing" in mind
    assert "neuron_scope" in mind


def test_tokenizer_inspect_unloaded_and_loaded(studio_client):
    snap = _snapshot_pytorch_state()
    try:
        pytorch_state.model = None
        res_unloaded = studio_client.post("/v1/tokenizer/inspect", json={"text": "Hello Polysom", "top_k": 4})
        assert res_unloaded.status_code == 200
        body_unloaded = res_unloaded.json()
        assert body_unloaded["available"] is False
        assert body_unloaded["tokens"] == []

        # Instantiate real lightweight PyTorch model and inspect real logprobs + entropy
        pytorch_state.reconfigure_architecture(
            preset_id="triune-small",
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            head_dim=32,
            num_experts=2,
            vocab_size=32000,
        )
        res_loaded = studio_client.post(
            "/v1/tokenizer/inspect",
            json={"text": "Triune architecture routes reflex and cortex", "top_k": 3, "max_tokens": 32},
        )
        assert res_loaded.status_code == 200
        data = res_loaded.json()
        assert data["available"] is True
        tokens = data["tokens"]
        assert len(tokens) >= 2
        assert tokens[0]["logprob"] is None
        for tok in tokens[1:]:
            assert isinstance(tok["logprob"], float)
            assert 0.0 <= tok["prob"] <= 1.0
            assert tok["entropy"] >= 0.0
            assert len(tok["top_candidates"]) <= 3

        res_tok = studio_client.post("/v1/tokenizer/tokenize", json={"text": "Polysom Studio"})
        assert res_tok.status_code == 200
        assert len(res_tok.json()["tokens"]) >= 1
    finally:
        _restore_pytorch_state(snap)


def test_chat_completions_supports_sampling_and_logprobs(studio_client):
    snap = _snapshot_pytorch_state()
    try:
        pytorch_state.reconfigure_architecture(
            preset_id="triune-small",
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            head_dim=32,
            num_experts=2,
            vocab_size=32000,
        )
        res = studio_client.post(
            "/v1/chat/completions",
            json={
                "model": "triune-small",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 6,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 25,
                "repetition_penalty": 1.1,
                "logprobs": True,
                "stream": False,
            },
        )
        assert res.status_code == 200
        payload = res.json()
        assert "choices" in payload and len(payload["choices"]) == 1
        assert "telemetry" in payload
        assert "ttft_ms" in payload["telemetry"]
        assert "tokens_per_sec" in payload["telemetry"]
    finally:
        _restore_pytorch_state(snap)


def test_model_configs_listing_security_and_validation(studio_client):
    res_list = studio_client.get("/v1/configs")
    assert res_list.status_code == 200
    configs = res_list.json()["configs"]
    filenames = {item["filename"] for item in configs}
    assert "junkie.json" in filenames

    res_junkie = studio_client.get("/v1/configs/junkie.json")
    assert res_junkie.status_code == 200
    assert res_junkie.json()["filename"] == "junkie.json"
    assert res_junkie.json()["config"]["num_layers"] >= 4

    # Security: path traversal and dotdot rejection
    res_trav = studio_client.get("/v1/configs/..%5Cpyproject.toml")
    assert res_trav.status_code in {403, 404}
    res_dot = studio_client.get("/v1/configs/..secret.json")
    assert res_dot.status_code == 403

    # Validation + roundtrip save in isolated LUCID_MACHINES_ALLOWED_CONFIG_ROOTS
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cfg = Path(tmp_dir) / "sandbox_model.json"
        tmp_cfg.write_text(
            json.dumps(
                {
                    "name": "Sandbox",
                    "num_layers": 4,
                    "hidden_dim": 128,
                    "num_heads": 4,
                    "num_experts": 2,
                    "vocab_size": 32000,
                }
            ),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"LUCID_MACHINES_ALLOWED_CONFIG_ROOTS": tmp_dir}):
            res_invalid = studio_client.post(
                "/v1/configs/sandbox_model.json",
                json={
                    "config": {
                        "num_layers": 2,
                        "hidden_dim": 128,
                        "num_heads": 4,
                        "num_experts": 2,
                        "vocab_size": 32000,
                    }
                },
            )
            assert res_invalid.status_code == 422

            res_ok = studio_client.post(
                "/v1/configs/sandbox_model.json",
                json={
                    "config": {
                        "name": "Sandbox",
                        "num_layers": 6,
                        "hidden_dim": 256,
                        "num_heads": 4,
                        "num_experts": 4,
                        "vocab_size": 32000,
                    }
                },
            )
            assert res_ok.status_code == 200
            assert res_ok.json()["status"] == "saved"
            saved_json = json.loads(tmp_cfg.read_text(encoding="utf-8"))
            assert saved_json["num_layers"] == 6
            assert saved_json["hidden_dim"] == 256


def test_launch_junkie_configures_flagship_and_blocks_traversal(studio_client):
    snap = _snapshot_pytorch_state()
    try:
        res_bad = studio_client.post(
            "/v1/training/launch_junkie",
            json={"dataset_path": "../secret.jsonl", "start": False},
        )
        assert res_bad.status_code == 403

        res = studio_client.post(
            "/v1/training/launch_junkie",
            json={
                "batch_size": 2,
                "seq_len": 512,
                "grad_accum_steps": 4,
                "precision": "bf16",
                "depth_mode": "dynamic",
                "start": False,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "configured"
        assert body["model_id"] == "junkie"
        assert body["is_training"] is False
        assert pytorch_state.explicit_architecture_active is True
        assert pytorch_state.grad_accum_steps == 4
    finally:
        _restore_pytorch_state(snap)


def test_dataset_preview_reads_confined_jsonl_and_blocks_traversal(studio_client):
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = Path(tmp_dir) / "sample.jsonl"
        sample.write_text('{"text": "first"}\n{"text": "second"}\n', encoding="utf-8")
        with patch.dict(os.environ, {"LUCID_MACHINES_ALLOWED_DATA_ROOTS": tmp_dir}):
            res_ok = studio_client.get("/v1/datasets/preview", params={"path": str(sample), "limit": 5})
            assert res_ok.status_code == 200
            data = res_ok.json()
            assert data["record_count"] == 2
            assert len(data["records"]) == 2

            res_trav = studio_client.get("/v1/datasets/preview", params={"path": "../pyproject.toml"})
            assert res_trav.status_code == 403

            res_missing = studio_client.get(
                "/v1/datasets/preview", params={"path": str(Path(tmp_dir) / "missing.jsonl")}
            )
            assert res_missing.status_code == 404


def test_dag_history_and_replay_endpoints(studio_client):
    snap = _snapshot_pytorch_state()
    try:
        dag_payload = {
            "nodes": [
                {"id": "n1", "type": "Text Input", "data": {"config": {"text": "Polysom DAG test"}}},
                {"id": "n2", "type": "Console Output", "data": {"config": {}}},
            ],
            "edges": [{"source": "n1", "target": "n2"}],
        }
        res_exec = studio_client.post("/v1/dag/execute", json=dag_payload)
        assert res_exec.status_code == 200

        res_hist = studio_client.get("/v1/dag/history")
        assert res_hist.status_code == 200
        runs = res_hist.json()["runs"]
        assert len(runs) >= 1
        latest_id = runs[0]["run_id"]

        res_replay = studio_client.post(f"/v1/dag/replay/{latest_id}")
        assert res_replay.status_code == 200

        res_missing = studio_client.post("/v1/dag/replay/nonexistent-run-id")
        assert res_missing.status_code == 404
    finally:
        _restore_pytorch_state(snap)


def test_agent_run_orchestrate_and_history_endpoints(studio_client):
    snap = _snapshot_pytorch_state()
    try:
        res_state = studio_client.get("/v1/agents/state")
        assert res_state.status_code == 200

        res_run = studio_client.post(
            "/v1/agents/run",
            json={"goal": "Verify tool listing", "max_steps": 1},
        )
        assert res_run.status_code == 200
        run_data = res_run.json()
        assert run_data["status"] in {"success", "error"}

        res_orch = studio_client.post(
            "/v1/agents/orchestrate",
            json={
                "goal": "Inspect system status",
                "personas": ["architect", "sentinel"],
                "max_workers": 2,
                "max_steps": 1,
            },
        )
        assert res_orch.status_code == 200

        res_hist = studio_client.get("/v1/agents/history")
        assert res_hist.status_code == 200
        assert len(res_hist.json()["runs"]) >= 2
    finally:
        _restore_pytorch_state(snap)


def test_frontend_themes_ps_classes_and_hygiene_invariants():
    # Entry function and render target
    assert "function PolysomStudio()" in APP_JS
    assert "root.render(e(PolysomStudio))" in APP_JS

    # Logo and Windows path fixes
    assert "assets/polysom-logo.svg" in APP_JS
    assert "polysom-logo.svg" in INDEX_HTML
    assert "C:\\\\PolysomStudio" in APP_JS
    assert "TriuneStudio" not in APP_JS
    assert "SYSTEM_PROMPT_PRESETS" not in APP_JS
    assert "SYSTEM_TEMPLATES" in APP_JS

    # Six curated Polysom themes including Euphoria
    for theme_id in ("hypnagogic", "phosphene", "pareidolia", "afterimage", "mirage", "euphoria"):
        assert f"id: '{theme_id}'" in APP_JS
        assert f'data-theme="{theme_id}"' in STYLE_CSS or f"data-theme='{theme_id}'" in STYLE_CSS
    assert "#f6efe2" in APP_JS and "#f6efe2" in STYLE_CSS
    assert "#b53810" in APP_JS and "#b53810" in STYLE_CSS

    # All 34 .ps-* classes defined in CSS and wired in app.js
    ps_classes = sorted(set(re.findall(r"\.(ps-[a-z0-9-]+)", STYLE_CSS)))
    assert len(ps_classes) >= 34
    for cls_name in ps_classes:
        assert cls_name in APP_JS, f"CSS class {cls_name} is not wired in studio/src/app.js"

    # Inaugural v1.0.0 framing (no stale v3.0.0 or CentroidSteer v3 copy)
    assert "Polysom Studio v1.0.0" in APP_JS
    assert "v=1.0.0" in INDEX_HTML
    assert "3.0.0" not in APP_JS
    assert "3.0.0" not in INDEX_HTML
    assert "CentroidSteer v3" not in APP_JS

    # Zero CSS gradients, zero unsafe DOM sinks, and zero emoji/mojibake regressions
    assert "linear-gradient(" not in STYLE_CSS
    assert "radial-gradient(" not in STYLE_CSS
    assert "conic-gradient(" not in STYLE_CSS
    assert "\ufffd" not in APP_JS
    assert "'? " not in APP_JS
    assert "dangerouslySetInnerHTML" not in APP_JS
    assert "document.write(" not in APP_JS
    assert "innerHTML" not in APP_JS

    # R3 & R4 invariants: cleanText Unicode markers, SplitHandle, AmbientPerceptualLayer, Mind Probe
    assert "replace(/\\u0120/g" in APP_JS
    assert "replace(/\\u010A/g" in APP_JS
    assert not re.search(r"replace\(/\[?G", APP_JS)
    assert "function SplitHandle" in APP_JS or "const SplitHandle" in APP_JS
    assert "AmbientPerceptualLayer" in APP_JS
    assert "Polysom Mind Probe & Lucid Steer" in APP_JS


def test_configs_endpoint_lists_all_junkie_and_triune_variants(studio_client):
    res = studio_client.get("/v1/configs")
    assert res.status_code == 200
    filenames = {item["filename"] for item in res.json().get("configs", [])}
    for expected in ("junkie.json", "junkie-mini.json", "junkie-1b.json", "junkie-3b.json", "junkie-7b.json"):
        assert expected in filenames, f"Missing {expected} from /v1/configs"
    assert any(name.startswith("triune-") for name in filenames)


def test_launch_junkie_supports_presets_and_steer_mode(studio_client):
    snap = _snapshot_pytorch_state()
    try:
        res_bad_preset = studio_client.post("/v1/training/launch_junkie", json={"preset": "nonexistent-preset"})
        assert res_bad_preset.status_code == 400

        res_mini = studio_client.post(
            "/v1/training/launch_junkie",
            json={
                "preset": "junkie-mini",
                "steer_mode": "centroid",
                "target_depth_dist": [0.4, 0.35, 0.25],
                "top_k_experts": 2,
                "start": False,
            },
        )
        assert res_mini.status_code == 200
        body = res_mini.json()
        assert body["model_id"] == "junkie-mini"
        assert body["preset"] == "junkie-mini"
        assert body["architecture"]["num_layers"] == 12
    finally:
        _restore_pytorch_state(snap)


def test_training_optimizer_api_parity_and_no_stale_comparison(studio_client):
    snap = _snapshot_pytorch_state()
    try:
        get_res = studio_client.get("/v1/training/optimizer")
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert "comparison_vs_unsloth" not in get_data
        assert "triune/kernels/fast_muon.py" not in json.dumps(get_data)
        for field in ("steer_mode", "num_steer_dirs", "steer_update_gap", "steer_scale", "use_rms_steer"):
            assert field in get_data

        post_res = studio_client.post(
            "/v1/training/optimizer",
            json={
                "steer_mode": "random_same_mem",
                "num_steer_dirs": 6,
                "steer_update_gap": 4,
                "steer_scale": 0.42,
                "use_rms_steer": True,
            },
        )
        assert post_res.status_code == 200
        post_data = post_res.json()
        assert post_data["steer_mode"] == "random_same_mem"
        assert post_data["num_steer_dirs"] == 6
        assert post_data["steer_update_gap"] == 4
        assert abs(post_data["steer_scale"] - 0.42) < 1e-6
        assert post_data["use_rms_steer"] is True

        bad_res = studio_client.post("/v1/training/optimizer", json={"steer_mode": "bogus_mode"})
        assert bad_res.status_code == 400
    finally:
        _restore_pytorch_state(snap)


def test_chat_completions_forwards_workspace_context_and_csp_aligned(studio_client):
    res_health = studio_client.get("/healthz")
    assert res_health.status_code == 200
    csp = res_health.headers.get("Content-Security-Policy", "")
    assert "font-src 'self' data:" in csp
    assert "fonts.googleapis.com" not in csp
    assert "fonts.gstatic.com" not in csp

    captured = {}

    def _fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {
            "text": "Verified workspace answer.",
            "route": "REFLEX",
            "latency_ms": 12.0,
            "tokens_per_sec": 80,
            "tokens_count": 3,
            "vram_gb": 0.0,
            "weights_loaded": False,
        }

    snap = _snapshot_pytorch_state()
    try:
        with patch.object(pytorch_state, "generate_text", side_effect=_fake_generate):
            res = studio_client.post(
                "/v1/chat/completions",
                json={
                    "model": "triune-small",
                    "session_id": "test_ws_ctx_session",
                    "messages": [{"role": "user", "content": "Explain kernel.py"}],
                    "workspace_context": {
                        "workspace_root": "/tmp/demo_ws",
                        "attached_files": [
                            {
                                "path": "src/kernel.py",
                                "line_range": "1-10",
                                "content_excerpt": "def fast_op(x): return x * 2",
                            }
                        ],
                    },
                    "stream": False,
                },
            )
            assert res.status_code == 200
            assert "src/kernel.py" in captured["prompt"]
            assert "def fast_op(x): return x * 2" in captured["prompt"]
            assert "/tmp/demo_ws" in captured["prompt"]
    finally:
        _restore_pytorch_state(snap)
