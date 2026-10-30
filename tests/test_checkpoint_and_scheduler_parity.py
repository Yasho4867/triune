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


def test_repeated_step_parameters_base_state_survival():
    """Verify that repeated calls to step_parameters() preserve base optimizer state across layers and steps."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _create_mini_model(device=device)

    opt = CentroidSteerOptimizer(
        model, lr=1e-3, rank=8, use_triangular_core=True, use_muon=True
    )
    base_params = opt.non_expert_params
    assert len(base_params) > 0

    x = torch.randint(0, 512, (2, 8), device=device)
    y = torch.randint(0, 512, (2, 8), device=device)

    for global_step in range(10):
        opt.zero_grad()
        out = model(x)[0]
        loss = F.cross_entropy(out.view(-1, 512), y.view(-1))
        loss.backward()

        # Simulate layer-by-layer step_parameters calls
        half = max(1, len(base_params) // 2)
        p1 = base_params[:half]
        p2 = base_params[half:]

        # Step subset 1
        opt.step_parameters(p1, is_global_step_end=False)
        for p in p1:
            if p.grad is not None:
                assert p in opt.base_optimizer.state, f"State missing for p in p1 at step {global_step}"
                st = opt.base_optimizer.state[p]
                assert len(st) > 0, f"State dictionary was cleared for p in p1 at step {global_step}!"

        # Step subset 2
        opt.step_parameters(p2, is_global_step_end=True)
        for p in p2:
            if p.grad is not None:
                assert p in opt.base_optimizer.state, f"State missing for p in p2 at step {global_step}"
                st = opt.base_optimizer.state[p]
                assert len(st) > 0, f"State dictionary was cleared for p in p2 at step {global_step}!"

        # Crucially: assert subset 1 state was NOT wiped when subset 2 was stepped!
        for p in p1:
            if p.grad is not None:
                st1 = opt.base_optimizer.state[p]
                assert len(st1) > 0, f"p1 state was wiped during p2 step at global_step {global_step}!"
                if "step" in st1:
                    s_val = st1["step"].item() if isinstance(st1["step"], torch.Tensor) else st1["step"]
                    assert 1 <= s_val <= global_step + 1, f"Unexpected step count {s_val} at global_step {global_step}"


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


def test_triton_pytorch_triangular_kernel_parity():
    """Verify numerical parity between Triton fused triangular kernel and PyTorch reference."""
    from triune.kernels.fast_triangular import fast_triangular_adamw, HAS_TRITON

    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA or Triton unavailable for fused triangular kernel parity test.")

    device = torch.device("cuda")
    torch.manual_seed(42)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for N in [2, 8]:
        for DR in [9, 17, 33]:
            # Generate random upper-triangular gradient batches
            R = torch.randn(N, DR, DR, device=device, dtype=torch.float32)
            R = torch.triu(R)

            # PyTorch reference implementation
            mom_ref = torch.zeros(N, DR, DR, device=device, dtype=torch.float32)
            var_ref = torch.zeros(N, DR, DR, device=device, dtype=torch.float32)
            delta_ref = torch.zeros(N, DR, DR, device=device, dtype=torch.float32)

            # Triton implementation
            mom_triton = torch.zeros(N, DR, DR, device=device, dtype=torch.float32)
            var_triton = torch.zeros(N, DR, DR, device=device, dtype=torch.float32)
            delta_triton = torch.zeros(N, DR, DR, device=device, dtype=torch.float32)

            for step in range(1, 6):
                step_tensor = torch.full((N,), step, device=device, dtype=torch.long)

                # PyTorch Reference step
                mom_ref.mul_(beta1).add_(R, alpha=1.0 - beta1)
                var_ref.mul_(beta2).addcmul_(R, R, value=1.0 - beta2)
                step_view = step_tensor.view(N, 1, 1).to(torch.float32)
                bias_c1 = (1.0 - (beta1 ** step_view))
                bias_c2 = (1.0 - (beta2 ** step_view)).clamp_min(1e-7)
                m_hat = mom_ref / bias_c1
                v_hat = var_ref / bias_c2
                delta_ref.copy_(torch.triu(m_hat / (v_hat.sqrt() + eps)))

                # Triton kernel step
                fast_triangular_adamw(
                    R, mom_triton, var_triton, delta_triton, step_tensor, beta1, beta2, eps
                )

                # Assert exact parity between Triton and PyTorch
                diff_mom = (mom_ref - mom_triton).abs().max().item()
                diff_var = (var_ref - var_triton).abs().max().item()
                diff_delta = (delta_ref - delta_triton).abs().max().item()

                assert diff_mom < 1e-5, f"Mom diverged at step {step} (N={N}, DR={DR}): diff {diff_mom}"
                assert diff_var < 1e-5, f"Var diverged at step {step} (N={N}, DR={DR}): diff {diff_var}"
                assert diff_delta < 1e-5, f"Delta diverged at step {step} (N={N}, DR={DR}): diff {diff_delta}"


def test_trainer_checkpoint_roundtrip_equivalence(tmp_path):
    """Verify that saving and restoring through triune.trainer.checkpoint produces exact training continuation."""
    from triune.trainer.checkpoint import save_latest, load_checkpoint

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "checkpoint_dir": str(tmp_path),
        "hidden_dim": 64,
        "num_layers": 4,
        "num_heads": 2,
        "head_dim": 32,
        "vocab_size": 512,
        "num_experts": 2,
        "top_k": 1,
        "shared_expert": True,
        "shared_scale": 1.0,
        "capacity_multiplier": 1.25,
        "router_prefix_layers": 1,
        "reflex_exit_layer": 2,
        "limbic_exit_layer": 3,
        "target_depth_dist": (0.33, 0.33, 0.34),
        "use_fp4": False,
        "use_fp8": False,
        "use_muon": True,
        "muon_lr": 0.02,
        "galore": True,
        "galore_rank": 8,
        "use_rope": True,
        "rope_max_seq_len": 2048,
    }

    class MockTrainer:
        def __init__(self, model, optimizer, cfg):
            self.model = model
            self.optimizer = optimizer
            self.config = cfg

    model_A = _create_mini_model(device=device)
    opt_A = CentroidSteerOptimizer(
        model_A, lr=1e-3, rank=8, update_gap=5, use_triangular_core=True, use_muon=True
    )
    trainer_A = MockTrainer(model_A, opt_A, config)

    # Deterministic inputs
    batches = [
        (torch.randint(0, 512, (2, 8), device=device), torch.randint(0, 512, (2, 8), device=device))
        for _ in range(7)
    ]

    # Train for 5 steps
    for step in range(5):
        opt_A.zero_grad()
        x, y = batches[step]
        loss_A = F.cross_entropy(model_A(x)[0].view(-1, 512), y.view(-1))
        loss_A.backward()
        opt_A.step()

    # Save via checkpoint.py
    ckpt_path = save_latest(trainer_A, engine=None, step=5, loss=loss_A.item())
    assert ckpt_path.exists()

    # Create fresh model B and restore via checkpoint.py
    model_B = _create_mini_model(device=device)
    opt_B = CentroidSteerOptimizer(
        model_B, lr=1e-3, rank=8, update_gap=5, use_triangular_core=True, use_muon=True
    )
    trainer_B = MockTrainer(model_B, opt_B, copy.deepcopy(config))

    load_checkpoint(trainer_B, ckpt_path, load_optimizer=True)

    # Step 6 on both models with identical RNG
    x6, y6 = batches[5]
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    opt_A.zero_grad()
    loss_A = F.cross_entropy(model_A(x6)[0].view(-1, 512), y6.view(-1))
    loss_A.backward()
    opt_A.step()

    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state(rng_cuda)

    opt_B.zero_grad()
    loss_B = F.cross_entropy(model_B(x6)[0].view(-1, 512), y6.view(-1))
    loss_B.backward()
    opt_B.step()

    assert abs(loss_A.item() - loss_B.item()) < 1e-4
    for (name_A, p_A), (name_B, p_B) in zip(model_A.named_parameters(), model_B.named_parameters()):
        diff = (p_A - p_B).abs().max().item()
        assert diff < 1e-5, f"Parameter {name_A} diverged after load_checkpoint: diff {diff}"


def test_scheduler_and_resume_lr_continuity():
    """Verify learning-rate schedule continuity across checkpoint save and resume."""
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_A = _create_mini_model(device=device)
    base_lr = 1e-3
    muon_lr = 0.02
    expert_lr = 5e-4

    opt_A = CentroidSteerOptimizer(
        model_A, lr=base_lr, expert_lr=expert_lr, muon_lr=muon_lr, use_muon=True
    )
    sched_A = torch.optim.lr_scheduler.CosineAnnealingLR(opt_A, T_max=20, eta_min=1e-5)

    x = torch.randint(0, 512, (2, 8), device=device)
    y = torch.randint(0, 512, (2, 8), device=device)

    # Train for 7 steps
    for step in range(7):
        opt_A.zero_grad()
        loss = F.cross_entropy(model_A(x)[0].view(-1, 512), y.view(-1))
        loss.backward()
        opt_A.step()
        sched_A.step()

    # Checkpoint
    checkpoint = {
        "opt_state": copy.deepcopy(opt_A.state_dict()),
        "sched_state": copy.deepcopy(sched_A.state_dict()),
    }

    # New model, optimizer, scheduler
    model_B = _create_mini_model(device=device)
    model_B.load_state_dict(model_A.state_dict())
    opt_B = CentroidSteerOptimizer(
        model_B, lr=base_lr, expert_lr=expert_lr, muon_lr=muon_lr, use_muon=True
    )
    sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(opt_B, T_max=20, eta_min=1e-5)

    opt_B.load_state_dict(checkpoint["opt_state"])
    sched_B.load_state_dict(checkpoint["sched_state"])

    # Verify identical learning rates immediately upon restore
    assert math.isclose(opt_A.param_groups[0]["lr"], opt_B.param_groups[0]["lr"], rel_tol=1e-5)
    assert math.isclose(opt_A.expert_lr, opt_B.expert_lr, rel_tol=1e-5)
    assert math.isclose(opt_A.muon_optimizer.param_groups[0]["lr"], opt_B.muon_optimizer.param_groups[0]["lr"], rel_tol=1e-5)

    # Run 5 more steps on both and verify continued exact LR match at every step
    for step in range(8, 13):
        opt_A.zero_grad()
        F.cross_entropy(model_A(x)[0].view(-1, 512), y.view(-1)).backward()
        opt_A.step()
        sched_A.step()

        opt_B.zero_grad()
        F.cross_entropy(model_B(x)[0].view(-1, 512), y.view(-1)).backward()
        opt_B.step()
        sched_B.step()

        assert math.isclose(opt_A.param_groups[0]["lr"], opt_B.param_groups[0]["lr"], rel_tol=1e-5)
        assert math.isclose(opt_A.expert_lr, opt_B.expert_lr, rel_tol=1e-5)
        assert math.isclose(opt_A.muon_optimizer.param_groups[0]["lr"], opt_B.muon_optimizer.param_groups[0]["lr"], rel_tol=1e-5)

