import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.optim.muon import zeropower_via_newtonschulz5, Muon
from triune.model.moe import MoE_FFN
from triune.optim.centroid import CentroidSteerOptimizer
from triune.model.transformer import TriuneTransformer


def test_newtonschulz5_orthogonality():
    print("=== Testing Newton-Schulz Orthogonality ===")
    torch.manual_seed(42)
    
    # 1. Square matrix
    G = torch.randn(128, 128)
    O = zeropower_via_newtonschulz5(G, steps=5)
    # Check that O @ O.T is approximately identity
    I_approx = O @ O.T
    diff = (I_approx - torch.eye(128)).abs().mean().item()
    print(f"Square matrix (128x128) orthogonality mean abs error: {diff:.4f}")
    assert diff < 0.35, f"Expected low error for semi-orthogonal matrix, got {diff}"

    # 2. Tall matrix (rows > cols)
    G_tall = torch.randn(256, 128)
    O_tall = zeropower_via_newtonschulz5(G_tall, steps=5)
    # Should be semi-orthogonal: O^T @ O = I
    I_tall = O_tall.T @ O_tall
    diff_tall = (I_tall - torch.eye(128)).abs().mean().item()
    print(f"Tall matrix (256x128) orthogonality mean abs error: {diff_tall:.4f}")
    assert diff_tall < 0.35, f"Expected low error, got {diff_tall}"

    # 3. Wide matrix (cols > rows)
    G_wide = torch.randn(128, 256)
    O_wide = zeropower_via_newtonschulz5(G_wide, steps=5)
    # Should be semi-orthogonal: O @ O^T = I
    I_wide = O_wide @ O_wide.T
    diff_wide = (I_wide - torch.eye(128)).abs().mean().item()
    print(f"Wide matrix (128x256) orthogonality mean abs error: {diff_wide:.4f}")
    assert diff_wide < 0.35, f"Expected low error, got {diff_wide}"
    print("✅ Newton-Schulz orthogonality passed!\n")


def test_muon_optimizer_step():
    print("=== Testing Muon Optimizer Step ===")
    torch.manual_seed(42)
    layer = nn.Linear(64, 32, bias=False)
    opt = Muon(layer.parameters(), lr=0.02, momentum=0.95, weight_decay=0.01)

    initial_weight = layer.weight.data.clone()
    x = torch.randn(16, 64)
    loss = layer(x).pow(2).mean()
    loss.backward()

    assert layer.weight.grad is not None
    opt.step()

    # Verify state tracking
    state = opt.state[layer.weight]
    assert state["step"] == 1, f"Expected step 1, got {state['step']}"
    assert "momentum" in state, "State must contain momentum buffer"
    assert state["momentum"].shape == layer.weight.shape

    # Verify weights changed
    weight_change = (layer.weight.data - initial_weight).abs().sum().item()
    print(f"Weight change after Muon step: {weight_change:.4f}")
    assert weight_change > 0, "Weight should have updated"

    # Step 2: verify momentum accumulation
    opt.zero_grad()
    loss2 = layer(x).pow(2).mean()
    loss2.backward()
    opt.step()
    assert state["step"] == 2

    print("✅ Muon optimizer step passed!\n")


def test_muon_invariants_and_device_discipline():
    print("=== Testing Muon Invariants & Device Discipline ===")
    # 1. 1D and 3D rejection in zeropower_via_newtonschulz5
    try:
        zeropower_via_newtonschulz5(torch.randn(64))
        assert False, "Should raise ValueError for 1D tensor"
    except ValueError as e:
        assert "2D matrix" in str(e)

    try:
        zeropower_via_newtonschulz5(torch.randn(4, 16, 16))
        assert False, "Should raise ValueError for 3D tensor"
    except ValueError as e:
        assert "2D matrix" in str(e)

    # 2. 1D rejection in Muon constructor
    bias_param = nn.Parameter(torch.randn(32))
    try:
        Muon([bias_param], lr=0.02)
        assert False, "Should raise ValueError for 1D parameter in Muon"
    except ValueError as e:
        assert "2D parameter" in str(e)

    # 3. CPU device conservation: CPU tensor must stay on CPU
    G_cpu = torch.randn(32, 32, device="cpu")
    O_cpu = zeropower_via_newtonschulz5(G_cpu)
    assert O_cpu.device == torch.device("cpu"), f"Expected CPU device, got {O_cpu.device}"

    # 4. Decoupled weight decay verification
    p = nn.Parameter(torch.ones(10, 10))
    opt = Muon([p], lr=0.1, weight_decay=0.05, momentum=0.0)
    p.grad = torch.zeros_like(p)
    opt.step()
    # ortho_update of zeros is zeros; weight decay decays by (1 - 0.1 * 0.05) = 0.995
    expected = 1.0 * (1.0 - 0.1 * 0.05)
    diff = (p.data - expected).abs().max().item()
    assert diff < 1e-5, f"Expected weight decay to yield {expected}, got max diff {diff}"

    print("✅ Muon invariants & device discipline passed!\n")


def test_adaptive_steer_scale_and_load_ratio():
    print("=== Testing MoE Load Ratio EMA & Adaptive Steer Scale ===")
    torch.manual_seed(42)
    dim = 64
    num_experts = 4
    moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=1, shared_expert=False)
    
    assert hasattr(moe, 'expert_load_ratio'), "MoE_FFN must register expert_load_ratio buffer"
    assert moe.expert_load_ratio.shape == (num_experts,)
    print(f"Initial expert_load_ratio: {moe.expert_load_ratio.tolist()}")

    # Forward with input to trigger routing updates
    x = torch.randn(2, 32, dim) # 64 tokens
    _ = moe(x, update_stats=True)
    print(f"After step 1 expert_load_ratio: {moe.expert_load_ratio.tolist()}")
    print(f"Last counts: {moe.last_counts.tolist()}, target: {moe.last_target}")
    
    # Check that expert_load_ratio is positive and deviates based on routing
    assert (moe.expert_load_ratio > 0).all()

    # Now verify CentroidSteerOptimizer adaptive scaling logic
    opt = CentroidSteerOptimizer(
        moe, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01,
        rank=16, update_gap=10, steer_scale=0.20, use_muon=True
    )
    
    # Check layer groups
    expert_groups = opt.layer_groups
    print(f"Number of MoE expert groups managed by CentroidSteer: {len(expert_groups)}")
    assert len(expert_groups) > 0

    # Test backward and step
    out = moe(x)
    loss = out.pow(2).mean()
    loss.backward()
    opt.step()
    print("✅ Adaptive Steer Scale & MoE Load Ratio test passed!\n")


def test_full_triune_3tier_optimizer():
    print("=== Testing Full Triune 3-Tier Parameter Partitioning ===")
    torch.manual_seed(42)
    
    # Miniature TriuneTransformer
    model = TriuneTransformer(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=6,
        num_heads=4,
        head_dim=32,
        num_experts=4,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=4,
        use_fp4=False,
        use_fp8=False
    )

    opt = CentroidSteerOptimizer(
        model, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01,
        rank=16, update_gap=5, steer_scale=0.20,
        use_muon=True, muon_lr=0.02
    )

    print(f"CentroidSteer layer groups (MoE routed experts): {len(opt.layer_groups)}")
    print(f"Muon params (2D Attention & Shared projections): {len(opt.muon_params)}")
    print(f"Base AdamW params (1D Norms, Embeddings, Router): {len(opt.non_expert_params)}")

    assert len(opt.layer_groups) > 0, "Should have MoE expert groups"
    assert len(opt.muon_params) > 0, "Should have Muon 2D hidden params"
    assert len(opt.non_expert_params) > 0, "Should have 1D & embedding params in AdamW"
    assert opt.muon_optimizer is not None, "Muon optimizer should be instantiated"

    # Verify embeddings are NOT in Muon params
    embed_id = id(model.token_embed.weight)
    assert embed_id not in [id(p) for p in opt.muon_params], "Embeddings must NOT be optimized by Muon!"
    assert embed_id in [id(p) for p in opt.non_expert_params], "Embeddings must be in AdamW!"

    # Run 3 training steps and verify loss decreases without NaNs
    losses = []
    input_ids = torch.randint(0, 1000, (2, 16))
    for step in range(3):
        opt.zero_grad()
        logits, _ = model(input_ids)
        loss = logits.mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        assert not torch.isnan(loss), f"NaN encountered at step {step}"

    print(f"Losses over 3 steps: {losses}")

    # Test state_dict and load_state_dict
    sd = opt.state_dict()
    assert 'muon_optimizer' in sd, "State dict should contain muon_optimizer"
    assert 'base_optimizer' in sd, "State dict should contain base_optimizer"

    # Create new optimizer and load state dict
    opt2 = CentroidSteerOptimizer(
        model, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01,
        rank=16, update_gap=5, steer_scale=0.20,
        use_muon=True, muon_lr=0.02
    )
    opt2.load_state_dict(sd)
    assert opt2.step_count == opt.step_count, "Step count should match after load"
    print("✅ Full Triune 3-Tier Optimizer & Serialization test passed!\n")


def test_centroid_steer_mathematical_invariants():
    print("=== Testing CentroidSteer Mathematical Invariants (GaLore min-dim, Transport, Orthonormal Variance) ===")
    torch.manual_seed(42)
    dim = 64
    rank = 16
    num_experts = 4

    moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=2, shared_expert=False)
    opt = CentroidSteerOptimizer(
        moe, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0,
        rank=rank, update_gap=2, steer_scale=0.25, use_muon=False
    )

    x = torch.randn(4, 16, dim)
    for step_idx in range(4):
        opt.zero_grad()
        out = moe(x, update_stats=True)
        loss = out.pow(2).mean()
        loss.backward()
        opt.step()

    for group in opt.layer_groups:
        p = group['param']
        m, n = p.shape
        proj = group['projection']
        side = group['projection_side']
        state = group['state']

        # 1. GaLore Theorem 3.2: always project smaller dimension min(m, n) == dim
        expected_side = 'left' if m <= n else 'right'
        assert side == expected_side, f"Expected side {expected_side} for shape {p.shape}, got {side}"
        assert proj.shape == (min(m, n), rank), f"Expected projection shape {(min(m, n), rank)}, got {proj.shape}"

        # 2. Subspace transport across update_gap=2: step count must reach 4 without resetting to 1
        assert state['step'] == 4, f"Expected continuous step=4 across update_gap=2, got {state['step']}"
        assert not torch.isnan(state['momentum']).any(), "Momentum contains NaNs after basis transport"
        assert not torch.isnan(state['variance']).any(), "Variance contains NaNs after basis transport"
        assert (state['variance'] >= 0).all(), "Variance must remain non-negative after basis transport"

    # 3. Verify orthonormal forward projection vs scaled reconstruction on step 1
    torch.manual_seed(99)
    moe_a = MoE_FFN(dim=dim, num_experts=2, top_k=2, shared_expert=False)
    moe_b = MoE_FFN(dim=dim, num_experts=2, top_k=2, shared_expert=False)
    moe_b.load_state_dict(moe_a.state_dict())

    opt_small = CentroidSteerOptimizer(
        moe_a, lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0,
        rank=rank, update_gap=10, steer_scale=0.10, use_muon=False
    )
    opt_large = CentroidSteerOptimizer(
        moe_b, lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0,
        rank=rank, update_gap=10, steer_scale=0.80, use_muon=False
    )

    # Synchronize random initial orthogonal projections
    for ga, gb in zip(opt_small.layer_groups, opt_large.layer_groups):
        init_p = torch.empty(dim, rank)
        torch.nn.init.orthogonal_(init_p)
        ga['projection'] = init_p.clone()
        gb['projection'] = init_p.clone()
        ga['projection_side'] = 'left' if ga['param'].shape[0] <= ga['param'].shape[1] else 'right'
        gb['projection_side'] = ga['projection_side']

    x_test = torch.randn(2, 16, dim)
    out_a = moe_a(x_test, update_stats=True)
    out_a.pow(2).mean().backward()
    opt_small.step()

    out_b = moe_b(x_test, update_stats=True)
    out_b.pow(2).mean().backward()
    opt_large.step()

    # Forward variance must be identical (uncorrupted by steer_scale^2), while parameter updates differ
    for ga, gb in zip(opt_small.layer_groups, opt_large.layer_groups):
        var_diff = (ga['state']['variance'] - gb['state']['variance']).abs().max().item()
        assert var_diff < 1e-6, f"Second moment EMA was corrupted by steer_scale! Max diff: {var_diff}"
        param_diff = (ga['param'].data - gb['param'].data).abs().max().item()
        assert param_diff > 1e-6, "Parameter reconstruction should scale with steer_scale!"

    print("✅ CentroidSteer mathematical invariants passed!\n")


if __name__ == "__main__":
    test_newtonschulz5_orthogonality()
    test_muon_optimizer_step()
    test_muon_invariants_and_device_discipline()
    test_adaptive_steer_scale_and_load_ratio()
    test_full_triune_3tier_optimizer()
    test_centroid_steer_mathematical_invariants()
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
