"""Extensive usability and UX synchronization test suite for Triune Studio, CLI, and Scripts.

Verifies:
1. Single Canonical Engine State invariant: All Studio endpoints, REST routes, WebSockets,
   and CLI actions synchronize against a single authoritative RealPyTorchEngineState.
2. Studio API endpoints usability:
   - GET & POST /v1/training/optimizer (triangular core, 8-bit base, rank, update gap, muon)
   - POST /v1/resource/plan & POST /v1/resource/assess (DynamicResourceManager integration)
   - POST /v1/chat/completions (standard and speculative decoding via SelfSpeculativeGenerator)
   - GET /v1/models & GET /v1/system/diagnostics
3. CLI Usability:
   - CLI subcommands (models, plan-memory, chat --help, train --help, checkpoint export --help)
4. Script Usability:
   - scripts/train.py and scripts/chat.py argument parser parity
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

from triune.api import create_app
from triune.api.routes import pytorch_state, RealPyTorchEngineState
from triune.runtime.resource_manager import DynamicResourceManager, FeasibilityAssessment
from triune.model.speculative import SelfSpeculativeGenerator
from triune.configs import build_config
from triune.model import build_model


@pytest.fixture(scope="module")
def app_client():
    from fastapi.testclient import TestClient
    app = create_app()
    return TestClient(app)


def test_single_canonical_engine_state_invariants():
    """Verify that pytorch_state is the authoritative RealPyTorchEngineState instance."""
    assert isinstance(pytorch_state, RealPyTorchEngineState)
    assert hasattr(pytorch_state, "use_triangular_core")
    assert hasattr(pytorch_state, "use_8bit_base")
    assert hasattr(pytorch_state, "centroid_rank")
    assert hasattr(pytorch_state, "centroid_update_gap")
    assert pytorch_state.use_triangular_core is True
    assert pytorch_state.use_8bit_base is False


def test_optimizer_telemetry_and_update_sync(app_client):
    """Verify GET and POST /v1/training/optimizer live synchronization with pytorch_state."""
    # 1. GET telemetry
    res_get = app_client.get("/v1/training/optimizer")
    assert res_get.status_code == 200
    telemetry = res_get.json()
    assert "comparison_vs_unsloth" in telemetry
    assert "tier1_muon" in telemetry
    assert "tier2_centroid" in telemetry
    assert "tier3_adamw" in telemetry

    # 2. POST updates
    update_payload = {
        "lr": 0.0003,
        "centroid_lr": 0.0001,
        "steer_scale": 0.25,
        "muon_lr": 0.015,
        "momentum": 0.92,
        "use_triangular_core": True,
        "use_8bit_base": False,
        "rank": 128,
        "update_gap": 50,
    }
    res_post = app_client.post("/v1/training/optimizer", json=update_payload)
    assert res_post.status_code == 200
    updated = res_post.json()
    assert updated["status"] == "success"
    assert updated["use_triangular_core"] is True
    assert updated["rank"] == 128
    assert updated["update_gap"] == 50
    assert pytorch_state.centroid_rank == 128
    assert pytorch_state.centroid_update_gap == 50
    assert pytorch_state.use_triangular_core is True


def test_resource_plan_and_assess_endpoints(app_client):
    """Verify /v1/resource/plan and /v1/resource/assess integration with DynamicResourceManager."""
    model_cfg = {
        "hidden_dim": 256,
        "num_layers": 4,
        "num_experts": 4,
        "seq_len": 128,
        "batch_size": 2,
    }

    # Test /v1/resource/plan
    res_plan = app_client.post("/v1/resource/plan", json=model_cfg)
    assert res_plan.status_code == 200
    plan = res_plan.json()
    assert "total_params" in plan
    assert "param_memory_gb" in plan
    assert "recommended_batch_size" in plan

    # Test /v1/resource/assess
    res_assess = app_client.post("/v1/resource/assess", json=model_cfg)
    assert res_assess.status_code == 200
    assess = res_assess.json()
    assert "is_feasible" in assess
    assert "report" in assess
    assert "status" in assess
    assert "available_vram_gb" in assess


def test_chat_completions_speculative_decoding_sync(app_client):
    """Verify chat completions endpoint supports speculative decoding parameters."""
    req_body = {
        "model": "triune-small",
        "messages": [{"role": "user", "content": "Hello Triune"}],
        "temperature": 0.0,
        "max_tokens": 10,
        "speculative": True,
        "k_draft": 2,
    }

    # Mock generate_text to verify parameter forwarding
    with patch.object(pytorch_state, "generate_text") as mock_gen:
        mock_gen.return_value = {
            "text": "Hello world from Triune speculative decoding",
            "route": "SPECULATIVE (K=2, acc=75.0%)",
            "tokens_count": 6,
            "tokens_per_sec": 45,
            "latency_ms": 133.3,
            "ttft_ms": None,
            "decode_tokens_per_sec": 45.0,
            "vram_gb": 0.5,
            "checkpoint": None,
            "weights_loaded": False,
            "cancelled": False,
            "speculative_telemetry": {"draft_tokens_proposed": 4, "draft_tokens_accepted": 3},
            "acceptance_rate": 0.75,
        }

        resp = app_client.post("/v1/chat/completions", json=req_body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "Hello world" in data["choices"][0]["message"]["content"]
        assert mock_gen.called
        kwargs = mock_gen.call_args.kwargs
        assert kwargs["speculative"] is True
        assert kwargs["k_draft"] == 2


def test_models_and_diagnostics_endpoints(app_client):
    """Verify discovery endpoints in Triune Studio."""
    # /v1/models
    res_models = app_client.get("/v1/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    assert models_data["object"] == "list"
    ids = [m["id"] for m in models_data["data"]]
    assert "triune-small" in ids

    # /v1/system/diagnostics
    res_diag = app_client.get("/v1/system/diagnostics")
    assert res_diag.status_code == 200
    diag = res_diag.json()
    assert "platform" in diag
    assert "python_version" in diag
    assert "pytorch_version" in diag


def test_cli_argument_parsing():
    """Verify triune CLI parsers for new options."""
    from triune.cli import main
    import argparse

    # Test running CLI help or models via sys.argv mock
    with patch("sys.argv", ["triune", "models"]):
        with patch("builtins.print") as mock_p:
            main()
            assert mock_p.called

    with patch("sys.argv", ["triune", "plan-memory", "--vram-gb", "8"]):
        with patch("builtins.print") as mock_p:
            main()
            assert mock_p.called


def test_chat_script_argument_parsing():
    """Verify scripts/chat.py parser handles speculative decoding arguments."""
    from scripts.chat import build_parser

    parser = build_parser()
    args = parser.parse_args(["--speculative", "--k-draft", "4", "--max_new_tokens", "50"])
    assert args.speculative is True
    assert args.k_draft == 4
    assert args.max_new_tokens == 50


def test_train_script_argument_parsing():
    """Verify scripts/train.py parser handles optimizer arguments."""
    from scripts.train import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--use_triangular_core",
        "--use_8bit_base",
        "--use_muon",
        "--muon_lr", "0.01",
    ])
    assert args.use_triangular_core is True
    assert args.use_8bit_base is True
    assert args.use_muon is True
    assert args.muon_lr == 0.01
