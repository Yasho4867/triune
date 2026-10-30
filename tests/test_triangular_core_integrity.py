import pytest
import torch
import torch.nn as nn
from triune.optim.centroid import CentroidSteerOptimizer
from triune.model import MoE_FFN


class DummyMoEModel(nn.Module):
    def __init__(self, d_model=256, num_experts=4, expert_dim=512):
        super().__init__()
        self.d_model = d_model
        self.ffn = MoE_FFN(
            dim=d_model,
            num_experts=num_experts,
            top_k=1,
            shared_expert=True,
            shared_scale=0.5
        )
        self.head = nn.Linear(d_model, 64, bias=False)

    def forward(self, x):
        h, aux = self.ffn(x)
        return self.head(h), aux


def test_triangular_core_state_persistence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyMoEModel(d_model=128, num_experts=4, expert_dim=256).to(device=device, dtype=torch.bfloat16)
    opt = CentroidSteerOptimizer(model, rank=16, use_triangular_core=True)

    # Step 1
    x = torch.randn(2, 16, 128, device=device, dtype=torch.bfloat16)
    out, _ = model(x)
    loss = out.sum()
    loss.backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    # Check that expert groups have non-zero momentum and variance
    active_expert_groups = [g for g in opt.layer_groups if g['module'] is not None]
    assert len(active_expert_groups) > 0, "Must have active expert groups"
    
    for g in active_expert_groups:
        mom1 = g['state']['momentum'].clone()
        var1 = g['state']['variance'].clone()
        step1 = g['state']['step']
        assert step1 == 1, f"Step count should be 1, got {step1}"
        assert mom1.abs().max().item() > 0.0, "Momentum must be non-zero after step 1"
        assert var1.abs().max().item() > 0.0, "Variance must be non-zero after step 1"

    # Step 2
    out, _ = model(x)
    loss = out.sum()
    loss.backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    for g in active_expert_groups:
        mom2 = g['state']['momentum']
        var2 = g['state']['variance']
        step2 = g['state']['step']
        assert step2 == 2, f"Step count should be 2, got {step2}"
        # Assert moments accumulated and changed
        assert not torch.equal(mom1, mom2), "Momentum must evolve across steps (persistence)"
        assert not torch.equal(var1, var2), "Variance must evolve across steps (persistence)"


def test_triangular_core_steering_survival():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyMoEModel(d_model=128, num_experts=4, expert_dim=256).to(device=device, dtype=torch.bfloat16)
    
    # Set simulated last_centroids on the MoE module
    c = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    model.ffn.last_centroids = c
    
    opt = CentroidSteerOptimizer(model, rank=16, steer_scale=0.5, use_triangular_core=True)

    x = torch.randn(2, 16, 128, device=device, dtype=torch.bfloat16)
    out, _ = model(x)
    loss = out.sum()
    loss.backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)

    active_expert_groups = [g for g in opt.layer_groups if g['module'] is not None]
    steered_groups = [g for g in active_expert_groups if g.get('steer_applied', False)]
    assert len(steered_groups) > 0, "At least one group should have centroid steering applied"
    
    for g in steered_groups:
        r = min(opt.rank, min(g['param'].shape))
        # Check that column r in Q_L and Q_R is non-zero
        ql_steer_norm = g['Q_L'][:, r].norm().item()
        qr_steer_norm = g['Q_R'][:, r].norm().item()
        assert ql_steer_norm > 0.9, f"Q_L steering vector should be unit norm, got {ql_steer_norm}"
        assert qr_steer_norm > 0.9, f"Q_R dual steering vector should be unit norm, got {qr_steer_norm}"


def test_triangular_core_mixed_ranks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Multi-head model with varied dimensions
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = MoE_FFN(dim=64, num_experts=2) # dim 64
            self.head_tiny = nn.Linear(64, 8, bias=False) # min dim 8 < rank 16!
            self.head_large = nn.Linear(64, 128, bias=False) # min dim 64 > rank 16!
        def forward(self, x):
            h, aux = self.ffn(x)
            return self.head_tiny(h) + self.head_large(h)[:, :8], aux

    model = MixedModel().to(device=device, dtype=torch.bfloat16)
    opt = CentroidSteerOptimizer(model, rank=16, use_triangular_core=True)

    x = torch.randn(2, 4, 64, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        opt.step_parameters(model.parameters(), is_global_step_end=True)
        opt.zero_grad()

    # Successful completion without shape error indicates mixed rank robustness
    assert opt.step_count == 3


def test_triangular_core_fp32_precision_and_subspace_step_invariants():
    """Verifies FP32 moment precision and continuous lifetime step vs basis-relative subspace step."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyMoEModel(d_model=128, num_experts=2, expert_dim=128).to(device=device, dtype=torch.bfloat16)
    opt = CentroidSteerOptimizer(model, rank=8, update_gap=2, use_triangular_core=True)

    x = torch.randn(2, 4, 128, device=device, dtype=torch.bfloat16)
    active_expert_groups = [g for g in opt.layer_groups if g['module'] is not None]
    assert len(active_expert_groups) > 0

    # Step 1: Initial subspace step 1, global step 1
    out, _ = model(x)
    out.sum().backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    for g in active_expert_groups:
        assert g['state']['momentum'].dtype == torch.float32, "Moments must be stored in FP32 to prevent BF16 underflow"
        assert g['state']['variance'].dtype == torch.float32, "Variance must be stored in FP32"
        assert g['state']['step'] == 1
        assert g['state']['subspace_step'] == 1

    # Step 2: Global step 2, subspace step 2 (hits update_gap=2)
    out, _ = model(x)
    out.sum().backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    for g in active_expert_groups:
        assert g['state']['step'] == 2
        assert g['state']['subspace_step'] == 2

    # Step 3: Projection basis refreshes, subspace_step resets to 1 while lifetime step continues to 3
    out, _ = model(x)
    out.sum().backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    for g in active_expert_groups:
        assert g['state']['step'] == 3, f"Lifetime step must be continuous (3), got {g['state']['step']}"
        assert g['state']['subspace_step'] == 1, f"Subspace step must reset to 1 on new basis, got {g['state']['subspace_step']}"

    print("✅ FP32 moment precision and subspace_step invariant verified!")
