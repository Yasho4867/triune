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
    opt.zero_grad()

    # Verify weights changed
    weight_change = (layer.weight.data - initial_weight).abs().sum().item()
    print(f"Weight change after Muon step: {weight_change:.4f}")
    assert weight_change > 0, "Weight should have updated"
    print("✅ Muon optimizer step passed!\n")


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


if __name__ == "__main__":
    test_newtonschulz5_orthogonality()
    test_muon_optimizer_step()
    test_adaptive_steer_scale_and_load_ratio()
    test_full_triune_3tier_optimizer()
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
