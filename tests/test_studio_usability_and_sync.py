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

from unittest.mock import patch

import pytest

from triune.api import create_app
from triune.api.routes import RealPyTorchEngineState, pytorch_state


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
    args = parser.parse_args(
        [
            "--use_triangular_core",
            "--use_8bit_base",
            "--use_muon",
            "--muon_lr",
            "0.01",
        ]
    )
    assert args.use_triangular_core is True
    assert args.use_8bit_base is True
    assert args.use_muon is True
    assert args.muon_lr == 0.01


def test_upgrade_to_triune_architecture_endpoint(app_client):
    """Verify POST /v1/models/upgrade_to_triune transmutation logic and layer mapping."""
    payload = {
        "model_name": "llama-3-8b",
        "source_type": "dense_transformer",
        "reflex_ratio": 0.25,
        "limbic_experts": 8,
        "cortex_gla": True,
    }
    res = app_client.post("/v1/models/upgrade_to_triune", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["model_name"] == "llama-3-8b"
    assert data["architecture_family"] == "Llama 3 Dense"
    assert "compression_ratio" in data
    assert data["compression_ratio"] > 1.0
    assert "throughput_multiplier" in data
    assert data["throughput_multiplier"] >= 1.8
    assert "layer_mapping" in data
    assert len(data["layer_mapping"]) == 32

    tiers = {m["upgraded_tier"] for m in data["layer_mapping"]}
    assert tiers == {"reflex", "limbic", "cortex"}

    gpt2_payload = {
        "model_name": "gpt2",
        "source_type": "gpt2",
        "reflex_ratio": 0.33,
        "limbic_experts": 4,
        "cortex_gla": True,
        "apply_as_active": True,
    }
    res_gpt2 = app_client.post("/v1/models/upgrade_to_triune", json=gpt2_payload)
    assert res_gpt2.status_code == 200
    data_gpt2 = res_gpt2.json()
    assert data_gpt2["status"] == "success"
    assert data_gpt2["activated"] is True
    assert data_gpt2["active_model_id"] == "triune-gpt2"
    assert len(data_gpt2["layer_mapping"]) == 12
    assert data_gpt2["tier_summary"]["limbic_experts"] == 4

    # Verify playground neurons telemetry reports graphics interop and active model
    res_neurons = app_client.get("/v1/playground/neurons")
    assert res_neurons.status_code == 200
    n_data = res_neurons.json()
    assert "graphics_interop" in n_data
    assert "Zero-Copy 120 FPS" in n_data["graphics_interop"]
    assert "cuda_active" in n_data


def test_upgrade_all_seven_standard_models(app_client):
    """Verify all 7 standard models can be upgraded to Triune 3-Tier architecture."""
    models_to_test = [
        ("llama-3-8b", 32, "Llama 3 Dense"),
        ("mistral-7b", 32, "Mistral 7B Dense"),
        ("qwen-2.5-7b", 28, "Qwen 2.5 7B Dense"),
        ("gpt2", 12, "GPT-2 Dense"),
        ("llama-70b", 80, "Llama 3 70B Dense"),
        ("deepseek-v3", 61, "DeepSeek V3 Dense"),
        ("gemma-2-9b", 42, "Gemma 2 9B Dense"),
    ]

    for model_name, expected_layers, expected_family in models_to_test:
        payload = {
            "model_name": model_name,
            "source_type": "dense_transformer",
            "reflex_ratio": 0.25,
            "limbic_experts": 8,
            "cortex_gla": True,
            "apply_as_active": False,
        }
        res = app_client.post("/v1/models/upgrade_to_triune", json=payload)
        assert res.status_code == 200, f"Failed for {model_name}"
        data = res.json()
        assert data["status"] == "success"
        assert data["architecture_family"] == expected_family
        assert len(data["layer_mapping"]) == expected_layers
        assert data["compression_ratio"] > 1.0
        assert data["throughput_multiplier"] >= 1.8
        tiers = {m["upgraded_tier"] for m in data["layer_mapping"]}
        assert tiers == {"reflex", "limbic", "cortex"}

    # Also test apply_as_active for one of them
    res_active = app_client.post(
        "/v1/models/upgrade_to_triune",
        json={"model_name": "qwen-2.5-7b", "limbic_experts": 8, "apply_as_active": True},
    )
    assert res_active.status_code == 200
    data_active = res_active.json()
    assert data_active["status"] == "success"
    assert data_active["activated"] is True
    assert data_active["active_model_id"] == "triune-qwen-2.5-7b"

    # Verify studio/src/app.js exposes UPGRADEABLE_MODELS and the 7 standard models
    from pathlib import Path
    app_js = Path("studio/src/app.js").read_text(encoding="utf-8")
    assert "UPGRADEABLE_MODELS = [" in app_js
    for model_name, _, _ in models_to_test:
        assert f"id: '{model_name}'" in app_js
    assert "handleUpgradeModel(upgradeModelName, true)" in app_js


def test_neuron_scope_3d_synapse_simulator_integration():
    """Verify 3D Neural Particle & Wire Synapse Simulator in studio/src/app.js and style.css."""
    from pathlib import Path

    app_js_path = Path("studio/src/app.js")
    assert app_js_path.exists()
    app_js = app_js_path.read_text(encoding="utf-8")

    # 1. Component definition & integration
    assert "function NeuronSynapseCanvas3D(props)" in app_js
    assert "e(NeuronSynapseCanvas3D," in app_js
    assert "neuronScopeData" in app_js
    assert "activeTierIdx" in app_js

    # 2. 3 Distinct spatial clusters / tiers
    assert "tierName: 'Reflex'" in app_js
    assert "tierName: 'Limbic'" in app_js
    assert "tierName: 'Cortex'" in app_js
    assert "isExpertHub: true" in app_js

    # 3. Dynamic synapsing wires & perspective projection
    assert "wires.push({ from: rNode.id, to: targetHub.id" in app_js
    assert "wires.push({ from: lNode.id, to: targetCortex1.id" in app_js
    assert "const project = (x, y, z, cx, cy) =>" in app_js
    assert "camDistRef.current" in app_js
    assert "rotXRef.current" in app_js
    assert "rotYRef.current" in app_js

    # 4. Real-time pulse flow & live token velocity sync
    assert "streamVelocity.current" in app_js
    assert "pulsesRef.current" in app_js
    assert "triggerStimulate" in app_js

    # 5. Interactive controls & CSS styling
    css_path = Path("studio/src/style.css")
    assert css_path.exists()
    css_text = css_path.read_text(encoding="utf-8")
    assert ".neuron-theater-canvas-wrap" in css_text
    assert ".neuron-theater-canvas-wrap.expanded" in css_text
    assert ".neuron-tier-pill" in css_text
    assert ".btn-stimulate-synapse" in css_text


def test_appearance_themes_and_synchronization_invariants():
    """Verify all 9 themes in APPEARANCE_THEMES, default to obsidian-glass, topbar sync, and zero personas."""
    from pathlib import Path

    app_js_path = Path("studio/src/app.js")
    assert app_js_path.exists()
    app_js = app_js_path.read_text(encoding="utf-8")

    # 1. Verify all 9 themes in APPEARANCE_THEMES
    expected_themes = [
        "obsidian-glass",
        "cyberpunk-dark",
        "emerald-terminal",
        "modern-dark",
        "modern-forest",
        "classic-dark",
        "editorial-cream",
        "modern-light",
        "classic-light",
    ]
    for theme_id in expected_themes:
        assert f"id: '{theme_id}'" in app_js, f"Missing theme {theme_id} in APPEARANCE_THEMES"

    # 2. Verify obsidian-glass default & theme initialization sanitization
    assert "return 'obsidian-glass'" in app_js
    assert "triune.studio.theme" in app_js

    # 3. Verify topbar theme select synchronization
    assert "value: studioTheme" in app_js
    assert "Switch Studio Theme" in app_js
    assert "APPEARANCE_THEMES.map" in app_js

    # 4. Verify Current Theme card shows edition, mode, and Reset action
    assert "CURRENT THEME" in app_js
    assert "edition ·" in app_js
    assert "Reset to Studio Carbon (Default)" in app_js

    # 5. Verify zero persona chips, dropdowns, or system prompt presets
    assert "SYSTEM_PROMPT_PRESETS" not in app_js
    assert "DEFAULT_SYSTEM_PROMPT" in app_js


def test_neural_intelligence_console_layout_and_ergonomics():
    """Verify Unified Neural Intelligence Console, segmented tabs, topbar toggle, and height ergonomics."""
    from pathlib import Path

    app_js_path = Path("studio/src/app.js")
    assert app_js_path.exists()
    app_js = app_js_path.read_text(encoding="utf-8")

    css_path = Path("studio/src/style.css")
    assert css_path.exists()
    css = css_path.read_text(encoding="utf-8")

    # 1. Unified Neural Intelligence Console container & tabs in app.js
    assert "neural-intelligence-console" in app_js
    assert "neural-console-tabs" in app_js
    assert "3D Synapses" in app_js
    assert "Architecture Transmutation" in app_js
    assert "Split View" in app_js

    # 2. Topbar toggle for collapse/expand
    assert "neural-console-toggle-pill" in app_js
    assert "Collapse Console" in app_js
    assert "Neural Console" in app_js

    # 3. Compact default height (~220px) and split view containment
    assert "height: '165px'" in app_js
    assert "neural-console-body-split" in app_js
    assert "min-height: 165px" in css

    # 4. Chat scroll area flexibility & protection against squishing
    assert ".chat-scroll-area" in css
    assert "min-height: 0;" in css

    # 5. Crisp SVG icons and zero emoji glyphs in controls
    assert "'⛶ Compact'" not in app_js
    assert "'⛶ Expand'" not in app_js
    assert "neural-console-close-btn" in app_js


def test_appearance_theme_css_definitions_and_miniature_chrome_invariants():
    """Verify all 9 themes in style.css have rich miniature chrome preview cards, palette tokens, and ghost buttons."""
    from pathlib import Path

    css_path = Path("studio/src/style.css")
    assert css_path.exists()
    css = css_path.read_text(encoding="utf-8")

    all_9_themes = [
        "obsidian-glass",
        "cyberpunk-dark",
        "emerald-terminal",
        "modern-dark",
        "modern-forest",
        "classic-dark",
        "editorial-cream",
        "modern-light",
        "classic-light",
    ]

    # 1. Verify all 9 themes have actual theme class definitions
    for theme_id in all_9_themes:
        assert f".theme-{theme_id}" in css or f".react-studio-app.theme-{theme_id}" in css, (
            f"Missing theme definition for {theme_id}"
        )

    # 2. Verify all 9 preview card classes exist with theme-specific miniature tokens
    for theme_id in all_9_themes:
        assert f".preview-{theme_id}" in css, f"Missing preview card class for {theme_id}"

    # 3. Verify actual UI miniature chrome components in preview cards
    assert ".preview-sidebar" in css
    assert ".preview-topbar" in css
    assert ".preview-dot" in css
    assert ".preview-title" in css
    assert ".preview-card-a" in css
    assert ".preview-card-b" in css
    assert ".preview-metric-bar" in css
    assert ".preview-chart-line" in css
    assert ".preview-node-chip" in css

    # 4. Verify pseudo-element miniature UI rendering (metric fill, sparkline dot, node dot/labels)
    assert ".preview-metric-bar::before" in css
    assert ".preview-chart-line::after" in css
    assert ".preview-node-chip::before" in css
    assert ".preview-node-chip::after" in css

    # 5. Verify .appearance-apply high-contrast ghost button states
    assert ".appearance-apply" in css
    assert ".appearance-apply:hover:not(:disabled)" in css
    assert ".appearance-apply:disabled" in css or ".appearance-apply.applied" in css

    # 6. Verify zero dead persona CSS rules in style.css
    assert ".persona-ribbon-bar" not in css
    assert ".persona-chip-btn" not in css


def test_realtime_active_model_simulation_and_telemetry(app_client):
    """Verify live active model spec query and dynamic 3D simulation real-time wiring."""
    from pathlib import Path

    # 1. Verify /v1/models/active returns active model specification
    res = app_client.get("/v1/models/active")
    assert res.status_code == 200
    data = res.json()
    assert "active_model_spec" in data
    spec = data["active_model_spec"]
    assert spec["num_layers"] >= 4
    assert "tier_summary" in spec
    assert "layer_mapping" in spec
    assert len(spec["layer_mapping"]) == spec["num_layers"]

    # 2. Verify dynamic model selection in /v1/models/preset reflects in active_model_spec
    preset_res = app_client.post("/v1/models/preset", json={"preset_id": "triune-base", "num_layers": 8, "hidden_dim": 384, "num_heads": 6, "num_experts": 4})
    assert preset_res.status_code == 200

    neurons_res = app_client.get("/v1/playground/neurons")
    assert neurons_res.status_code == 200
    n_data = neurons_res.json()
    assert "active_model_spec" in n_data
    assert n_data["active_model_spec"]["num_layers"] == 8
    assert n_data["active_model_spec"]["num_experts"] == 4
    assert len(n_data["active_model_spec"]["layer_mapping"]) == 8

    # 3. Verify frontend studio/src/app.js uses dynamic active spec and real-time pulse emission
    app_js = Path("studio/src/app.js").read_text(encoding="utf-8")
    assert "activeSpec" in app_js
    assert "window.__triuneCanvasEmitPulse" in app_js
    assert "emitRealtimePulse" in app_js
    assert "upgradeModelName" in app_js
    assert "'active'" in app_js






