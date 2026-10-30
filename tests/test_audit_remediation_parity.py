"""Audit remediation parity and regression test suite.

Proves:
1. MoE_FFN activation is GELU and positional constructor API is backwards-compatible.
2. Padding-aware LM loss normalization matches exact valid-token mean and prevents dilution.
3. Gradient checkpointing produces exact gradient parity and suppresses duplicate MoE state mutations.
4. Cholesky-QR orthogonalization is numerically stable across difficult scales and rank deficiencies.
5. Expert learning-rate scheduling scales proportionally with cosine decay.
6. Exit usage telemetry strictly distinguishes empirical measurements from target distributions.
"""

from __future__ import annotations

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.model import MoE_FFN, TriuneTransformer, get_model_spec, build_model_from_spec
from triune.optim.centroid import CentroidSteerOptimizer, fast_subspace_projection
from triune.trainer.engine import TrainingEngine
from triune.trainer.scheduler import lr_schedule, set_optimizer_lr


# ─────────────────────────────────────────────────────────────────────────────
# 1. MoE Architecture & Positional Constructor Compatibility
# ─────────────────────────────────────────────────────────────────────────────

def test_moe_activation_is_gelu_and_positional_api():
    """Verify MoE uses canonical GELU (not SiLU) and supports original positional arguments."""
    dim = 64
    num_experts = 4
    top_k = 2

    # Test positional call with 5 arguments: (dim, num_experts, top_k, use_fp4, use_fp8)
    moe_pos = MoE_FFN(dim, num_experts, top_k, False, False)
    assert moe_pos.dim == dim
    assert moe_pos.num_experts == num_experts
    assert moe_pos.top_k == top_k
    assert moe_pos.use_fp4 is False
    assert moe_pos.use_fp8 is False
    assert moe_pos.shared_expert is True

    # Test that each expert uses GELU activation
    for expert in moe_pos.experts:
        act = expert[1]
        assert isinstance(act, nn.GELU), f"Expected nn.GELU, got {type(act)}"

    # Test that shared expert uses GELU activation
    assert moe_pos.shared is not None
    assert isinstance(moe_pos.shared[1], nn.GELU)

    # Test keyword call with shared_expert=False
    moe_no_shared = MoE_FFN(dim=dim, num_experts=num_experts, top_k=top_k, shared_expert=False)
    assert moe_no_shared.shared is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Padding-Aware Loss Normalization
# ─────────────────────────────────────────────────────────────────────────────

class _MockTrainer:
    def __init__(self, vocab_size=100, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="none")
        self.router_loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cpu")
        self.config = {
            "target_depth_dist": [0.5, 0.3, 0.2],
            "exploration": "none",
            "exploration_steps": 0,
            "grad_clip": 1.0,
            "log_every": 1000,
            "total_steps": 1000,
            "lr": 0.001,
            "usage_ema_decay": 0.99,
        }
        self.model = None
        self.optimizer = None


def test_padding_loss_normalization():
    """Prove that _compute_lm_loss divides strictly by valid non-padding tokens."""
    trainer = _MockTrainer(vocab_size=16, pad_token_id=0)
    engine = TrainingEngine(trainer)

    B, T, V = 2, 8, 16
    torch.manual_seed(42)
    logits = torch.randn(B * T, V, requires_grad=True)

    # Case A: No padding (all targets valid)
    targets_no_pad = torch.randint(1, V, (B * T,))
    loss_no_pad = engine._compute_lm_loss(logits, targets_no_pad, V)
    manual_no_pad = F.cross_entropy(logits, targets_no_pad)
    assert torch.allclose(loss_no_pad, manual_no_pad, atol=1e-5)

    # Case B: 50% padding
    targets_half_pad = targets_no_pad.clone()
    targets_half_pad[: (B * T) // 2] = 0  # Pad first half
    loss_half_pad = engine._compute_lm_loss(logits, targets_half_pad, V)

    valid_mask = targets_half_pad != 0
    raw_losses = F.cross_entropy(logits, targets_half_pad, reduction="none")
    expected_masked_mean = raw_losses[valid_mask].mean()
    diluted_unmasked_mean = raw_losses.mean()  # Old buggy behavior

    assert torch.allclose(loss_half_pad, expected_masked_mean, atol=1e-5)
    # The old diluted loss is roughly half the magnitude of the correct loss
    assert not torch.allclose(loss_half_pad, diluted_unmasked_mean, atol=1e-2)

    # Case C: 100% padding (must not crash with ZeroDivisionError or NaN)
    targets_all_pad = torch.zeros(B * T, dtype=torch.long)
    loss_all_pad = engine._compute_lm_loss(logits, targets_all_pad, V)
    assert torch.isfinite(loss_all_pad)
    assert loss_all_pad.item() == 0.0

    # Case D: Gradient equivalence
    loss_half_pad.backward()
    grad_computed = logits.grad.clone()
    logits.grad.zero_()

    # Compute manual loss directly
    manual_loss = F.cross_entropy(logits[valid_mask], targets_half_pad[valid_mask])
    manual_loss.backward()
    grad_manual = logits.grad.clone()
    assert torch.allclose(grad_computed, grad_manual, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gradient Checkpointing Parity & State Invariance
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpointing_gradient_and_state_parity():
    """Prove that gradient checkpointing produces identical gradients and preserves MoE state."""
    dim = 64
    num_experts = 4
    top_k = 1
    B, T = 4, 16

    # Build model A without checkpointing
    torch.manual_seed(1234)
    moe_no_cp = MoE_FFN(dim=dim, num_experts=num_experts, top_k=top_k, shared_expert=True)
    moe_no_cp.train()

    # Build model B with checkpointing (identical weights)
    torch.manual_seed(1234)
    moe_cp = MoE_FFN(dim=dim, num_experts=num_experts, top_k=top_k, shared_expert=True)
    moe_cp.train()

    from triune.model.config import checkpoint_fwd_ctx, checkpoint_bwd_ctx

    x = torch.randn(B, T, dim)

    # Forward without checkpointing
    x_no_cp = x.clone().requires_grad_(True)
    out_no_cp = moe_no_cp(x_no_cp, update_stats=True)
    loss_no_cp = out_no_cp.sum()
    loss_no_cp.backward()

    # Forward with non-reentrant checkpointing
    x_cp = x.clone().requires_grad_(True)
    out_cp = torch.utils.checkpoint.checkpoint(
        lambda n: moe_cp(n, update_stats=True),
        x_cp,
        use_reentrant=False,
        context_fn=lambda: (checkpoint_fwd_ctx(), checkpoint_bwd_ctx()),
    )
    loss_cp = out_cp.sum()
    loss_cp.backward()

    # Output parity
    assert torch.allclose(out_no_cp, out_cp, atol=1e-5)
    assert torch.allclose(x_no_cp.grad, x_cp.grad, atol=1e-3)

    # Parameter gradient parity
    for p_no_cp, p_cp in zip(moe_no_cp.parameters(), moe_cp.parameters()):
        if p_no_cp.grad is not None:
            assert torch.allclose(p_no_cp.grad, p_cp.grad, atol=2e-3)

    # MoE state parity (no double mutation during backward recomputation!)
    assert moe_no_cp._global_step == moe_cp._global_step == 1
    assert moe_no_cp.overflow_counter == moe_cp.overflow_counter
    assert torch.allclose(moe_no_cp.last_centroids, moe_cp.last_centroids, atol=1e-5)
    assert torch.allclose(moe_no_cp.expert_load_ratio, moe_cp.expert_load_ratio, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cholesky-QR Subspace Projection Robustness
# ─────────────────────────────────────────────────────────────────────────────

def test_cholesky_qr_numerical_robustness():
    """Verify fast_subspace_projection stability across varied scales and rank-deficiencies."""
    torch.manual_seed(999)
    m, n = 128, 64
    rank = 8

    # Standard scale
    grad_normal = torch.randn(m, n)
    proj, side = fast_subspace_projection(grad_normal, rank=rank)
    assert proj.shape == (min(m, n), rank)
    gram = torch.mm(proj.t(), proj)
    eye = torch.eye(rank)
    assert torch.allclose(gram, eye, atol=1e-4)

    # Very small gradient scale (1e-7)
    grad_small = torch.randn(m, n) * 1e-7
    proj_small, _ = fast_subspace_projection(grad_small, rank=rank)
    assert torch.isfinite(proj_small).all()
    gram_small = torch.mm(proj_small.t(), proj_small)
    assert torch.allclose(gram_small, eye, atol=1e-4)

    # Very large gradient scale (1e4)
    grad_large = torch.randn(m, n) * 1e4
    proj_large, _ = fast_subspace_projection(grad_large, rank=rank)
    assert torch.isfinite(proj_large).all()
    gram_large = torch.mm(proj_large.t(), proj_large)
    assert torch.allclose(gram_large, eye, atol=1e-4)

    # Severely rank-deficient gradient (rank 1 matrix)
    u = torch.randn(m, 1)
    v = torch.randn(1, n)
    grad_rank1 = torch.mm(u, v)
    proj_rank1, _ = fast_subspace_projection(grad_rank1, rank=rank)
    assert torch.isfinite(proj_rank1).all()
    # Orthonormal columns preserved even under rank deficiency
    gram_r1 = torch.mm(proj_rank1.t(), proj_rank1)
    assert torch.allclose(gram_r1, eye, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Expert Learning-Rate Schedule Decay
# ─────────────────────────────────────────────────────────────────────────────

def test_expert_lr_schedule_decay():
    """Prove that expert learning rate decays in lockstep with base learning rate."""
    spec = get_model_spec("triune-nano", vocab_size=256)
    model = build_model_from_spec(spec)

    base_lr = 0.001
    expert_lr_peak = 0.002
    opt = CentroidSteerOptimizer(model, lr=base_lr, expert_lr=expert_lr_peak)

    # Verify initial rates
    assert opt.expert_lr == expert_lr_peak
    assert opt.base_optimizer.param_groups[0]["lr"] == base_lr

    # At warmup midpoint (50% base lr)
    half_lr = base_lr * 0.5
    set_optimizer_lr(opt, half_lr)
    assert math.isclose(opt.expert_lr, expert_lr_peak * 0.5, rel_tol=1e-5)
    assert math.isclose(opt.base_optimizer.param_groups[0]["lr"], half_lr, rel_tol=1e-5)

    # At minimum decay (10% base lr)
    min_lr = base_lr * 0.1
    set_optimizer_lr(opt, min_lr)
    assert math.isclose(opt.expert_lr, expert_lr_peak * 0.1, rel_tol=1e-5)
    assert math.isclose(opt.base_optimizer.param_groups[0]["lr"], min_lr, rel_tol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Telemetry Semantics: Measured Usage vs Target Distribution
# ─────────────────────────────────────────────────────────────────────────────

def test_telemetry_semantics_distinguishes_usage_from_target():
    """Verify that unmeasured models report exit_usage=None and expose target separately."""
    from triune.api.routes import RealPyTorchEngineState

    state = RealPyTorchEngineState()
    # Before training has produced actual measurements
    state.step = 0
    state.history = []
    state.last_exit_usage = None
    state.last_loss = None

    # Measured usage must be None
    measured_usage = state._get_current_exit_usage()
    assert measured_usage is None, f"Expected None for unmeasured exit usage, got {measured_usage}"

    # Target distribution must reflect configured target
    target_dist = state._get_target_exit_distribution()
    assert target_dist is not None
    assert "reflex" in target_dist and "limbic" in target_dist and "cortex" in target_dist
    assert sum(target_dist.values()) == 100.0
