import torch
import torch.nn as nn

from triune.runtime.resource_manager import (
    DynamicResourceManager,
    LiveVRAMMonitor,
    HardwareProfile,
    FeasibilityAssessment,
    AllotmentPlan,
    VRAMBudgetExceededError,
)
from triune.model.transformer import TriuneTransformer


def test_hardware_probe():
    print("=== Testing Hardware Probe ===")
    hw = DynamicResourceManager.probe_hardware()
    print(f"Device: {hw.device_name}")
    print(f"Total VRAM: {hw.total_vram_gb} GB, Free VRAM: {hw.free_vram_gb} GB")
    print(f"Compute Capability: {hw.compute_capability}")
    print(f"Supports BF16: {hw.supports_bf16}, Supports FP8: {hw.supports_fp8}")
    print(f"Has Transformer Engine: {hw.has_transformer_engine}")
    assert hw.total_vram_gb > 0
    print("✅ Hardware probe passed!\n")


def test_feasibility_assessment():
    print("=== Testing Feasibility Assessment ===")
    # 1. triune-small in FP8
    config_small = {
        "model_name": "triune-small",
        "vocab_size": 32000,
        "hidden_dim": 1536,
        "num_layers": 18,
        "num_experts": 4,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "seq_len": 128,
        "use_fp8": True,
    }
    assessment = DynamicResourceManager.assess_feasibility(config_small)
    report = DynamicResourceManager.format_assessment_report(assessment)
    print(f"Report for triune-small FP8:\n{report}")
    assert assessment.total_params > 0
    assert assessment.precision == "FP8"
    assert len(assessment.recommendations) > 0

    # 2. triune-base in BF16 (should be HARD_LIMIT_EXCEEDED on 8GB GPU)
    config_base = {
        "model_name": "triune-base",
        "vocab_size": 32000,
        "hidden_dim": 1536,
        "num_layers": 24,
        "num_experts": 8,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "seq_len": 256,
        "use_fp8": False,
        "use_fp4": False,
    }
    assessment_base = DynamicResourceManager.assess_feasibility(config_base)
    print(f"\ntriune-base Status: {assessment_base.status}")
    print(f"  Weights: {assessment_base.param_memory_gb} GB (Free: {assessment_base.free_vram_gb} GB)")
    assert assessment_base.total_params > 4_000_000_000
    assert assessment_base.precision == "BF16"
    assert any("--force" in r for r in assessment_base.recommendations)
    print("✅ Feasibility Assessment passed!\n")


def test_architecture_immutability():
    print("=== Testing Architecture Immutability ===")
    # Ensure validate_and_allot NEVER alters num_layers or num_experts
    config = {
        "model_name": "triune-base",
        "vocab_size": 32000,
        "hidden_dim": 1536,
        "num_layers": 24,
        "num_experts": 8,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "seq_len": 256,
    }
    # Test with force=True so it doesn't raise
    plan = DynamicResourceManager.validate_and_allot(config, force=True, auto_fit=True)
    assert plan.config["num_layers"] == 24, f"Expected 24 layers, got {plan.config['num_layers']}"
    assert plan.config["num_experts"] == 8, f"Expected 8 experts, got {plan.config['num_experts']}"
    assert plan.config["hidden_dim"] == 1536, f"Expected 1536 hidden_dim, got {plan.config['hidden_dim']}"
    assert plan.forced_override is True
    print("✅ Architecture Immutability passed! (Layers & experts are never silently changed)\n")


def test_user_override():
    print("=== Testing User Override Authority ===")
    oversized_config = {
        "model_name": "triune-moe",
        "vocab_size": 32000,
        "hidden_dim": 1536,
        "num_layers": 32,
        "num_experts": 16,
        "batch_size": 8,
        "grad_accum_steps": 4,
        "seq_len": 256,
    }
    # Without force: should raise VRAMBudgetExceededError
    try:
        DynamicResourceManager.validate_and_allot(oversized_config, force=False)
        assert False, "Should have raised VRAMBudgetExceededError without --force"
    except VRAMBudgetExceededError as err:
        print(f"Caught expected VRAMBudgetExceededError (contains override instructions):")
        assert "--force" in str(err)
        print("  Verified error contains '--force' override instruction.")

    # With force: should succeed and mark forced_override=True
    plan = DynamicResourceManager.validate_and_allot(oversized_config, force=True)
    assert plan.forced_override is True
    assert "User override '--force' active" in plan.adjustment_reason
    print("✅ User Override Authority passed!\n")


def test_explicit_user_settings_preserved():
    print("=== Testing Explicit User Settings Preservation ===")
    config = {
        "model_name": "triune-small",
        "vocab_size": 32000,
        "hidden_dim": 1536,
        "num_layers": 18,
        "num_experts": 4,
        "batch_size": 3,
        "grad_accum_steps": 5,
        "seq_len": 128,
        "use_fp8": True,
    }
    user_overrides = {"batch_size": 3, "grad_accum_steps": 5}
    # Even if auto_fit=True, user explicitly gave batch_size=3 so it MUST NOT be altered
    plan = DynamicResourceManager.validate_and_allot(
        config, auto_fit=True, user_overrides=user_overrides
    )
    assert plan.batch_size == 3, f"Expected user batch_size 3, got {plan.batch_size}"
    assert plan.grad_accum_steps == 5, f"Expected user grad_accum 5, got {plan.grad_accum_steps}"
    print("✅ Explicit user settings preserved!\n")


def test_safe_to_device():
    print("=== Testing Safe Staged GPU Transfer ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriuneTransformer(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        head_dim=32,
        num_experts=2,
        router_prefix_layers=1,
        reflex_exit_layer=1,
        limbic_exit_layer=2,
    )
    loaded_model = DynamicResourceManager.safe_to_device(
        model, device=device, dtype=torch.bfloat16, force=True
    )
    for p in loaded_model.parameters():
        assert p.device.type == device.type
    print("✅ Safe staged transfer passed!\n")


def test_live_vram_monitor():
    print("=== Testing Live VRAM Monitor ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = LiveVRAMMonitor(device=device, defrag_threshold=0.80)
    telemetry = monitor.format_telemetry()
    print(f"Live Telemetry Output: {telemetry}")
    assert "VRAM:" in telemetry
    assert "Res:" in telemetry
    assert "Headroom:" in telemetry
    print("✅ Live VRAM monitor passed!\n")


if __name__ == "__main__":
    test_hardware_probe()
    test_feasibility_assessment()
    test_architecture_immutability()
    test_user_override()
    test_explicit_user_settings_preserved()
    test_safe_to_device()
    test_live_vram_monitor()
    print("🎉 ALL RESOURCE MANAGER & USER OVERRIDE TESTS PASSED!")
