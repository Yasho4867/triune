"""Targeted Correctness & Mathematical Parity Tests for Triune Optimizer:
1. 8-bit AdamW / Base Optimizer State Persistence.
2. Exact Checkpoint Round-Trip Parity for Triangular Core (Q_L, Q_R, Moments, Subspace Step).
3. Exact Checkpoint Round-Trip Parity for Standard GaLore Core.
4. Standard PyTorch LR Scheduler Synchronization across all 3 tiers.
5. Strict Kernel Input Immutability across CPU/CUDA and BF16/FP32.
"""

import copy
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.model.transformer import TriuneTransformer
from triune.optim.centroid import CentroidSteerOptimizer, HAS_8BIT
from triune.kernels.fast_muon import fast_zeropower_via_newtonschulz5, fast_zeropower_via_newtonschulz5_batched


def _create_mini_model(device="cpu", dtype=torch.float32):
    return TriuneTransformer(
        vocab_size=512,
        hidden_dim=64,
        num_layers=4,
        num_heads=2,
        head_dim=32,
        num_experts=2,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
        use_fp4=False,
        use_fp8=False,
    ).to(device=device, dtype=dtype)


def test_8bit_adamw_state_persistence():
    """Verify that base optimizer state is not cleared or corrupted during step_parameters."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _create_mini_model(device=device)

    opt = CentroidSteerOptimizer(
        model, lr=1e-3, rank=8, use_triangular_core=True, use_muon=True
    )
    assert opt.base_optimizer is not None

    x = torch.randint(0, 512, (2, 8), device=device)
    y = torch.randint(0, 512, (2, 8), device=device)

    # Track base optimizer parameter state progression
    first_base_param = opt.non_expert_params[0]

    for step in range(5):
        opt.zero_grad()
        out = model(x)[0]
        loss = F.cross_entropy(out.view(-1, 512), y.view(-1))
        loss.backward()
        opt.step()

        # Parameter state must exist and retain step count / history
        assert first_base_param in opt.base_optimizer.state, f"Base parameter state missing at step {step}"
        st = opt.base_optimizer.state[first_base_param]
        assert len(st) > 0, f"Base parameter state was cleared at step {step}!"
        if "step" in st:
            step_val = st["step"].item() if isinstance(st["step"], torch.Tensor) else st["step"]
            assert step_val == step + 1, f"Expected optimizer step {step + 1}, got {step_val}"


def test_checkpoint_roundtrip_triangular_core_parity():
    """Verify that checkpointing and restoring Triangular Core produces IDENTICAL subsequent updates."""
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_A = _create_mini_model(device=device)
    model_B = copy.deepcopy(model_A)

    opt_A = CentroidSteerOptimizer(
        model_A, lr=1e-3, rank=8, update_gap=5, use_triangular_core=True, use_muon=True, steer_scale=0.1
    )
    opt_B = CentroidSteerOptimizer(
        model_B, lr=1e-3, rank=8, update_gap=5, use_triangular_core=True, use_muon=True, steer_scale=0.1
    )

    # 1. Train Model A for 5 steps on deterministic batches
    batches = [
        (torch.randint(0, 512, (2, 8), device=device), torch.randint(0, 512, (2, 8), device=device))
        for _ in range(7)
    ]

    for step in range(5):
        opt_A.zero_grad()
        x, y = batches[step]
        loss_A = F.cross_entropy(model_A(x)[0].view(-1, 512), y.view(-1))
        loss_A.backward()
        opt_A.step()

    # 2. Checkpoint Model A and its Optimizer
    checkpoint = {
        "model_state": copy.deepcopy(model_A.state_dict()),
        "opt_state": copy.deepcopy(opt_A.state_dict()),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }

    # 3. Restore into Model B and Optimizer B
    model_B.load_state_dict(checkpoint["model_state"])
    opt_B.load_state_dict(checkpoint["opt_state"])
    torch.set_rng_state(checkpoint["rng_cpu"])
    if checkpoint["rng_cuda"] is not None:
        torch.cuda.set_rng_state(checkpoint["rng_cuda"])

    # Verify that bases and moments are present in Opt B and match Opt A
    assert opt_B.step_count == opt_A.step_count == 5
    for g_A, g_B in zip(opt_A.layer_groups, opt_B.layer_groups):
        assert g_B.get("Q_L") is not None, "Q_L must be restored!"
        assert g_B.get("Q_R") is not None, "Q_R must be restored!"
        assert torch.equal(g_A["Q_L"], g_B["Q_L"]), "Q_L basis mismatch after restore!"
        assert torch.equal(g_A["Q_R"], g_B["Q_R"]), "Q_R basis mismatch after restore!"
        assert torch.equal(g_A["state"]["momentum"], g_B["state"]["momentum"]), "Momentum mismatch after restore!"
        assert torch.equal(g_A["state"]["variance"], g_B["state"]["variance"]), "Variance mismatch after restore!"
        assert g_A["state"]["subspace_step"] == g_B["state"]["subspace_step"]

    # 4. Now run Step 6 on both models with the exact same input batch and RNG state
    x6, y6 = batches[5]

    rng_cpu_step = torch.get_rng_state()
    rng_cuda_step = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    opt_A.zero_grad()
    loss_A = F.cross_entropy(model_A(x6)[0].view(-1, 512), y6.view(-1))
    loss_A.backward()
    opt_A.step()

    # Reset RNG to exact step state so Model B executes with identical dropout and stochastic masks
    torch.set_rng_state(rng_cpu_step)
    if rng_cuda_step is not None:
        torch.cuda.set_rng_state(rng_cuda_step)

    opt_B.zero_grad()
    loss_B = F.cross_entropy(model_B(x6)[0].view(-1, 512), y6.view(-1))
    loss_B.backward()
    opt_B.step()

    # 5. Assert exact parameter equality across every single parameter
    for (name_A, p_A), (name_B, p_B) in zip(model_A.named_parameters(), model_B.named_parameters()):
        diff = (p_A - p_B).abs().max().item()
        assert diff < 1e-5, f"Parameter {name_A} diverged after checkpoint restore: max diff {diff}"


def test_checkpoint_roundtrip_standard_core_parity():
    """Verify that checkpointing and restoring Standard GaLore Core produces IDENTICAL subsequent updates."""
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_A = _create_mini_model(device=device)
    model_B = copy.deepcopy(model_A)

    opt_A = CentroidSteerOptimizer(
        model_A, lr=1e-3, rank=8, update_gap=5, use_triangular_core=False, use_muon=False
    )
    opt_B = CentroidSteerOptimizer(
        model_B, lr=1e-3, rank=8, update_gap=5, use_triangular_core=False, use_muon=False
    )

    batches = [
        (torch.randint(0, 512, (2, 8), device=device), torch.randint(0, 512, (2, 8), device=device))
        for _ in range(5)
    ]

    for step in range(3):
        opt_A.zero_grad()
        x, y = batches[step]
        loss_A = F.cross_entropy(model_A(x)[0].view(-1, 512), y.view(-1))
        loss_A.backward()
        opt_A.step()

    # Checkpoint & Restore
    checkpoint = {
        "model_state": copy.deepcopy(model_A.state_dict()),
        "opt_state": copy.deepcopy(opt_A.state_dict()),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    model_B.load_state_dict(checkpoint["model_state"])
    opt_B.load_state_dict(checkpoint["opt_state"])
    torch.set_rng_state(checkpoint["rng_cpu"])
    if checkpoint["rng_cuda"] is not None:
        torch.cuda.set_rng_state(checkpoint["rng_cuda"])

    # Step 4 on both with identical RNG state
    x4, y4 = batches[3]
    rng_cpu_step = torch.get_rng_state()
    rng_cuda_step = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    opt_A.zero_grad()
    F.cross_entropy(model_A(x4)[0].view(-1, 512), y4.view(-1)).backward()
    opt_A.step()

    # Reset RNG state so Model B gets identical forward/backward random state
    torch.set_rng_state(rng_cpu_step)
    if rng_cuda_step is not None:
        torch.cuda.set_rng_state(rng_cuda_step)

    opt_B.zero_grad()
    F.cross_entropy(model_B(x4)[0].view(-1, 512), y4.view(-1)).backward()
    opt_B.step()

    for (name_A, p_A), (name_B, p_B) in zip(model_A.named_parameters(), model_B.named_parameters()):
        diff = (p_A - p_B).abs().max().item()
        assert diff < 1e-5, f"Parameter {name_A} diverged after standard checkpoint restore: diff {diff}"


def test_pytorch_lr_scheduler_integration():
    """Verify that standard PyTorch lr_schedulers correctly cascade learning rate updates to all 3 tiers."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _create_mini_model(device=device)

    base_lr = 1e-3
    muon_lr = 0.02
    expert_lr = 5e-4

    opt = CentroidSteerOptimizer(
        model, lr=base_lr, expert_lr=expert_lr, muon_lr=muon_lr, use_muon=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)

    x = torch.randint(0, 512, (2, 8), device=device)
    y = torch.randint(0, 512, (2, 8), device=device)

    prev_base_lr = base_lr
    for step in range(10):
        opt.zero_grad()
        out = model(x)[0]
        loss = F.cross_entropy(out.view(-1, 512), y.view(-1))
        loss.backward()
        opt.step()
        scheduler.step()

        # Verify all tiers scaled down proportionally
        current_outer_lr = opt.param_groups[0]["lr"]
        expected_ratio = current_outer_lr / base_lr

        # Base optimizer pg lr
        cur_base_pg_lr = opt.base_optimizer.param_groups[0]["lr"]
        assert math.isclose(cur_base_pg_lr, current_outer_lr, rel_tol=1e-4)

        # Muon optimizer pg lr
        cur_muon_pg_lr = opt.muon_optimizer.param_groups[0]["lr"]
        assert math.isclose(cur_muon_pg_lr, muon_lr * expected_ratio, rel_tol=1e-4)

        # Expert lr
        assert math.isclose(opt.expert_lr, expert_lr * expected_ratio, rel_tol=1e-4)


def test_newtonschulz_input_tensor_strict_immutability():
    """Verify that Newton-Schulz orthogonalization never mutates caller input tensors in-place."""
    torch.manual_seed(42)
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for dev in devices:
        dtypes = [torch.float32]
        if dev == "cuda" and torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)

        for dt in dtypes:
            for shape in [(64, 64), (96, 32), (32, 96)]:
                G = torch.randn(*shape, device=dev, dtype=dt)
                G_clone = G.clone()
                out = fast_zeropower_via_newtonschulz5(G, steps=5)
                assert torch.equal(G, G_clone), f"Single-matrix input mutated in-place for {dev} {dt} {shape}!"

            for b_shape in [(4, 48, 48), (4, 80, 24), (4, 24, 80)]:
                G_b = torch.randn(*b_shape, device=dev, dtype=dt)
                G_b_clone = G_b.clone()
                out_b = fast_zeropower_via_newtonschulz5_batched(G_b, steps=5)
                assert torch.equal(G_b, G_b_clone), f"Batched input mutated in-place for {dev} {dt} {b_shape}!"
