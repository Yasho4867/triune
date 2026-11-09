import torch
import torch.nn as nn

from triune.model import MoE_FFN
from triune.optim.centroid import CentroidSteerOptimizer


class DummyMoEModel(nn.Module):
    def __init__(self, d_model=256, num_experts=4, expert_dim=512):
        super().__init__()
        self.d_model = d_model
        self.ffn = MoE_FFN(dim=d_model, num_experts=num_experts, top_k=1, shared_expert=True, shared_scale=0.5)
        self.head = nn.Linear(d_model, 64, bias=False)

    def forward(self, x):
        h = self.ffn(x)
        return self.head(h), None


def test_triangular_core_state_persistence():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyMoEModel(d_model=128, num_experts=4, expert_dim=256).to(device=device, dtype=torch.bfloat16)
    opt = CentroidSteerOptimizer(model, rank=16, use_triangular_core=True)

    # Step 1
    x = torch.randn(4, 32, 128, device=device, dtype=torch.bfloat16)
    out, _ = model(x)
    loss = out.sum()
    loss.backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    # Check that expert groups have non-zero momentum and variance
    active_expert_groups = [g for g in opt.layer_groups if g["module"] is not None]
    assert len(active_expert_groups) > 0, "Must have active expert groups"

    for g in active_expert_groups:
        mom1 = g["state"]["momentum"].clone()
        var1 = g["state"]["variance"].clone()
        step1 = g["state"]["step"]
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
        mom2 = g["state"]["momentum"]
        var2 = g["state"]["variance"]
        step2 = g["state"]["step"]
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

    active_expert_groups = [g for g in opt.layer_groups if g["module"] is not None]
    steered_groups = [g for g in active_expert_groups if g.get("steer_applied", False)]
    assert len(steered_groups) > 0, "At least one group should have centroid steering applied"

    for g in steered_groups:
        r = min(opt.rank, min(g["param"].shape))
        # Check that column r in Q_L and Q_R is non-zero
        ql_steer_norm = g["Q_L"][:, r].norm().item()
        qr_steer_norm = g["Q_R"][:, r].norm().item()
        assert ql_steer_norm > 0.9, f"Q_L steering vector should be unit norm, got {ql_steer_norm}"
        assert qr_steer_norm > 0.9, f"Q_R dual steering vector should be unit norm, got {qr_steer_norm}"


def test_triangular_core_mixed_ranks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multi-head model with varied dimensions
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = MoE_FFN(dim=64, num_experts=2)  # dim 64
            self.head_tiny = nn.Linear(64, 8, bias=False)  # min dim 8 < rank 16!
            self.head_large = nn.Linear(64, 128, bias=False)  # min dim 64 > rank 16!

        def forward(self, x):
            h = self.ffn(x)
            return self.head_tiny(h) + self.head_large(h)[..., :8], None

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
    active_expert_groups = [g for g in opt.layer_groups if g["module"] is not None]
    assert len(active_expert_groups) > 0

    # Step 1: Initial subspace step 1, global step 1
    out, _ = model(x)
    out.sum().backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    for g in active_expert_groups:
        assert g["state"]["momentum"].dtype == torch.float32, "Moments must be stored in FP32 to prevent BF16 underflow"
        assert g["state"]["variance"].dtype == torch.float32, "Variance must be stored in FP32"
        assert g["state"]["step"] == 1
        assert g["state"]["subspace_step"] == 1

    # Step 2: Global step 2, subspace step 2 (hits update_gap=2)
    out, _ = model(x)
    out.sum().backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    for g in active_expert_groups:
        assert g["state"]["step"] == 2
        assert g["state"]["subspace_step"] == 2

    # Step 3: Projection basis refreshes, subspace_step resets to 1 while lifetime step continues to 3
    out, _ = model(x)
    out.sum().backward()
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    for g in active_expert_groups:
        assert g["state"]["step"] == 3, f"Lifetime step must be continuous (3), got {g['state']['step']}"
        assert g["state"]["subspace_step"] == 1, (
            f"Subspace step must reset to 1 on new basis, got {g['state']['subspace_step']}"
        )

    print("✅ FP32 moment precision and subspace_step invariant verified!")


def test_basis_orthogonality_and_subdiagonal_residuals():
    """Validates empirical basis orthogonality and lower-triangular residual bounds in FP32 and BF16."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    m, n, r = 384, 1024, 16
    G_bf16 = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    G_fp32 = G_bf16.float()

    # Randomized Subspace Iteration
    Omega = torch.randn(n, r, device=device, dtype=torch.float32)
    Y = torch.mm(G_fp32, Omega)
    col_norms_y = torch.linalg.norm(Y, dim=0, keepdim=True)
    Y_norm = Y / col_norms_y.clamp_min(1e-12)
    L_y = torch.linalg.cholesky(torch.mm(Y_norm.t(), Y_norm) + 1e-7 * torch.eye(r, device=device))
    Q_L = torch.linalg.solve_triangular(L_y, Y_norm.t(), upper=False).t()

    Z = torch.mm(G_fp32.t(), Q_L)
    col_norms_z = torch.linalg.norm(Z, dim=0, keepdim=True)
    Z_norm = Z / col_norms_z.clamp_min(1e-12)
    L_z = torch.linalg.cholesky(torch.mm(Z_norm.t(), Z_norm) + 1e-7 * torch.eye(r, device=device))
    Q_R = torch.linalg.solve_triangular(L_z, Z_norm.t(), upper=False).t()

    Y2 = torch.mm(G_fp32, Q_R)
    col_norms_y2 = torch.linalg.norm(Y2, dim=0, keepdim=True)
    Y2_norm = Y2 / col_norms_y2.clamp_min(1e-12)
    L_y2 = torch.linalg.cholesky(torch.mm(Y2_norm.t(), Y2_norm) + 1e-7 * torch.eye(r, device=device))
    Q_L = torch.linalg.solve_triangular(L_y2, Y2_norm.t(), upper=False).t()

    # Steering augmentation
    c = torch.randn(m, 1, device=device, dtype=torch.float32)
    c_res = c - torch.mm(Q_L, torch.mm(Q_L.t(), c))
    c_orth = c_res / c_res.norm().clamp_min(1e-8)

    v = torch.mm(G_fp32.t(), c_orth)
    v_res = v - torch.mm(Q_R, torch.mm(Q_R.t(), v))
    v_orth = v_res / v_res.norm().clamp_min(1e-8)

    QL_bar_f32 = torch.cat([Q_L, c_orth], dim=1)
    QR_bar_f32 = torch.cat([Q_R, v_orth], dim=1)

    # 1. FP32 Orthogonality & Subdiagonal Residuals
    orth_l_f32 = (torch.mm(QL_bar_f32.t(), QL_bar_f32) - torch.eye(r + 1, device=device)).norm().item()
    orth_r_f32 = (torch.mm(QR_bar_f32.t(), QR_bar_f32) - torch.eye(r + 1, device=device)).norm().item()
    assert orth_l_f32 < 5e-5, f"FP32 Q_L orthogonality error must be < 5e-5, got {orth_l_f32:.3e}"
    assert orth_r_f32 < 5e-5, f"FP32 Q_R orthogonality error must be < 5e-5, got {orth_r_f32:.3e}"

    S_f32 = torch.mm(QL_bar_f32.t(), torch.mm(G_fp32, QR_bar_f32))
    lower_mask = torch.tril(torch.ones(r + 1, r + 1, device=device, dtype=torch.bool), diagonal=-1)
    subdiag_max_f32 = S_f32[lower_mask].abs().max().item()
    rel_subdiag_f32 = subdiag_max_f32 / G_fp32.norm().item()
    assert rel_subdiag_f32 < 1e-7, f"FP32 relative subdiagonal residual must be < 1e-7, got {rel_subdiag_f32:.3e}"

    # 2. BF16 Orthogonality & Subdiagonal Residuals (Accounting for 8-bit mantissa)
    QL_bar_bf16 = QL_bar_f32.bfloat16()
    QR_bar_bf16 = QR_bar_f32.bfloat16()
    orth_l_bf16 = (
        (torch.mm(QL_bar_bf16.t().float(), QL_bar_bf16.float()) - torch.eye(r + 1, device=device)).norm().item()
    )
    orth_r_bf16 = (
        (torch.mm(QR_bar_bf16.t().float(), QR_bar_bf16.float()) - torch.eye(r + 1, device=device)).norm().item()
    )
    assert orth_l_bf16 < 5e-3, f"BF16 Q_L orthogonality error must be < 5e-3, got {orth_l_bf16:.3e}"
    assert orth_r_bf16 < 5e-3, f"BF16 Q_R orthogonality error must be < 5e-3, got {orth_r_bf16:.3e}"

    S_bf16 = torch.mm(QL_bar_bf16.t(), torch.mm(G_bf16, QR_bar_bf16))
    subdiag_max_bf16 = S_bf16[lower_mask].float().abs().max().item()
    rel_subdiag_bf16 = subdiag_max_bf16 / G_bf16.float().norm().item()
    assert rel_subdiag_bf16 < 1e-4, f"BF16 relative subdiagonal residual must be < 1e-4, got {rel_subdiag_bf16:.3e}"
    print(f"✅ Orthogonality & subdiagonal bounds verified (FP32 rel={rel_subdiag_f32:.2e}, BF16 rel={rel_subdiag_bf16:.2e})")


def test_cholesky_householder_fallback_integrity():
    """Verifies that ill-conditioned or degenerate Gram matrices cleanly trigger Householder QR fallback in production batched_qr."""
    from triune.optim.centroid import CentroidSteerOptimizer, batched_qr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    K, m, r = 3, 128, 16
    Y = torch.randn(K, m, r, device=device)

    # Matrix 1: Artificially ill-conditioned (duplicate collinear columns, cond ratio >= 1e6)
    Y[1, :, 1] = Y[1, :, 0]

    # Matrix 2: Degenerate zero column
    Y[2, :, 0] = 0.0

    # Test the exact production batched_qr function
    Q, fallback_mask = batched_qr(Y, cond_threshold=1e4, eps_shift=1e-7, orth_tol=1e-4, return_fallback_mask=True)
    # Also verify diagnostics breakdown
    Q_diag, diag = batched_qr(Y, cond_threshold=1e4, eps_shift=1e-7, orth_tol=1e-4, return_diagnostics=True)
    assert torch.equal(Q, Q_diag)
    assert torch.equal(fallback_mask, diag["fallback_mask"])
    assert diag["cond_fail_mask"][1].item() or diag["orth_fail_mask"][1].item(), "Ill-conditioned matrix must trigger condition or orth fallback"
    assert diag["degen_mask"][2].item(), "Zero-column matrix must trigger degen fallback"

    # Verify mutually exclusive partition
    primary_sum = (
        diag["primary_degen_mask"].long()
        + diag["primary_cholesky_mask"].long()
        + diag["primary_cond_mask"].long()
        + diag["primary_orth_mask"].long()
    )
    assert torch.equal(primary_sum, fallback_mask.long()), "Primary fallback reasons must partition fallback_mask exactly"

    # Also test that batched_subspace_projection in triune.ops uses guarded QR
    from triune.ops import batched_subspace_projection

    test_grads = torch.randn(2, 64, 128, device=device)
    proj, side, diag_hist = batched_subspace_projection(test_grads, rank=8, niter=1, return_diagnostics=True)
    assert proj.shape == (2, 64, 8)
    assert len(diag_hist) > 0, "batched_subspace_projection must record QR diagnostics"

    # Also verify that CentroidSteerOptimizer.batched_qr matches
    Q_opt, fallback_opt = CentroidSteerOptimizer.batched_qr(
        Y, cond_threshold=1e4, eps_shift=1e-7, orth_tol=1e-4, return_fallback_mask=True
    )
    assert torch.equal(Q, Q_opt), "CentroidSteerOptimizer.batched_qr staticmethod must be identical"
    assert torch.equal(fallback_mask, fallback_opt)

    assert not fallback_mask[0].item(), "Well-conditioned matrix must use Cholesky path"
    assert fallback_mask[1].item(), "Ill-conditioned matrix must trigger Householder fallback"
    assert fallback_mask[2].item(), "Degenerate zero-column matrix must trigger Householder fallback"

    # All resulting Q matrices must be strictly orthonormal
    for i in range(K):
        orth_err = (torch.mm(Q[i].t(), Q[i]) - torch.eye(r, device=device)).norm().item()
        assert orth_err < 1e-4, f"Matrix {i} must be orthogonal after QR fallback, got {orth_err:.3e}"

    print("✅ Cholesky condition monitoring and Householder QR fallback integrity verified!")


def test_triangular_adamw_noise_filtering_and_update_equivalence():
    """Verifies that triangular AdamW matches unconstrained AdamW on exact cores and filters subdiagonal noise."""
    from triune.kernels.fast_triangular import fast_triangular_adamw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    r = 16

    # 1. Exact upper-triangular core: Bitwise equivalence between triangular AdamW and standard 2D AdamW
    S_exact = torch.triu(torch.randn(1, r + 1, r + 1, device=device, dtype=torch.float32))
    R_triu = S_exact.clone()
    mom_triu = torch.zeros(1, r + 1, r + 1, device=device, dtype=torch.float32)
    var_triu = torch.zeros(1, r + 1, r + 1, device=device, dtype=torch.float32)
    delta_triu = torch.zeros_like(R_triu)
    step_vals = torch.tensor([1.0], device=device)

    fast_triangular_adamw(
        R_triu, mom_triu, var_triu, delta_triu, step_vals, beta1=0.9, beta2=0.999, eps=1e-8
    )

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mom_full = (1.0 - beta1) * S_exact
    var_full = (1.0 - beta2) * (S_exact**2)
    bias_c1 = 1.0 - beta1
    bias_c2 = 1.0 - beta2
    delta_full = (mom_full / bias_c1) / ((var_full / bias_c2).sqrt() + eps)

    max_diff_exact = (delta_triu - delta_full).abs().max().item()
    assert max_diff_exact < 1e-6, f"Triangular AdamW must match 2D AdamW on exact cores, got diff {max_diff_exact:.3e}"

    # 2. Noise Filtering: Finite-precision subdiagonal noise is suppressed by triu(S)
    E_noise = torch.tril(torch.randn(1, r + 1, r + 1, device=device, dtype=torch.float32) * 1e-4, diagonal=-1)
    S_noisy = S_exact + E_noise

    # Unconstrained 2D Adam on noisy core: divides noise by noise standard deviation -> produces ~1.0 updates!
    mom_noisy = (1.0 - beta1) * S_noisy
    var_noisy = (1.0 - beta2) * (S_noisy**2)
    delta_noisy = (mom_noisy / bias_c1) / ((var_noisy / bias_c2).sqrt() + eps)
    lower_mask = torch.tril(torch.ones(r + 1, r + 1, device=device, dtype=torch.bool), diagonal=-1)
    unconstrained_noise_updates = delta_noisy[0][lower_mask].abs()
    assert unconstrained_noise_updates.mean().item() > 0.95, (
        "Unconstrained Adam must amplify coordinate noise into ~1.0 unit steps"
    )

    # Regularized Triu projection filter: zero subdiagonal noise before Adam variance normalization
    R_filtered = torch.triu(S_noisy)
    mom_filt = torch.zeros(1, r + 1, r + 1, device=device, dtype=torch.float32)
    var_filt = torch.zeros(1, r + 1, r + 1, device=device, dtype=torch.float32)
    delta_filt = torch.zeros_like(R_filtered)
    fast_triangular_adamw(
        R_filtered, mom_filt, var_filt, delta_filt, step_vals, beta1=0.9, beta2=0.999, eps=1e-8
    )
    subdiag_filt_max = delta_filt[0][lower_mask].abs().max().item()
    assert subdiag_filt_max == 0.0, f"Filtered subdiagonal update must be identically zero, got {subdiag_filt_max}"

    print("✅ AdamW exact core equivalence and noise suppression verified!")


def test_scholqr2_shift_bias_elimination_and_zero_grad_skip():
    """Verifies that Two-Pass Shifted Cholesky-QR (sCholQR2) eliminates shift bias on kappa~100 matrices without fallback, and skips unrouted zero-grad experts."""
    from triune.ops.centroid_steer import batched_qr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    # 1. Construct a full-rank matrix with condition number ~100 (where single-pass eps_shift=1e-7 has orth error > 1e-4)
    m, r = 384, 16
    U, _ = torch.linalg.qr(torch.randn(m, r, device=device, dtype=torch.float32), mode="reduced")
    V, _ = torch.linalg.qr(torch.randn(r, r, device=device, dtype=torch.float32), mode="reduced")
    # Singular values geometrically spaced from 1.0 down to 0.01 (kappa = 100)
    s = torch.logspace(0.0, -2.0, steps=r, device=device, dtype=torch.float32)
    Y_mod = (U * s.unsqueeze(0)) @ V.t()
    Y_batch = Y_mod.unsqueeze(0)

    Q, diag = batched_qr(Y_batch, cond_threshold=1e4, eps_shift=1e-7, orth_tol=1e-4, return_diagnostics=True)
    assert not diag["fallback_mask"][0].item(), (
        "sCholQR2 must survive on full-rank kappa=100 matrix without Householder fallback"
    )
    orth_err = (torch.mm(Q[0].t(), Q[0]) - torch.eye(r, device=device)).norm().item()
    assert orth_err < 1e-5, f"sCholQR2 orthogonality error must be < 1e-5, got {orth_err:.3e}"

    # 2. Verify that zero-gradient experts are skipped and non-refresh steps invoke 0 QR calls
    model = DummyMoEModel(d_model=128, num_experts=4, expert_dim=256).to(device=device, dtype=torch.bfloat16)
    opt = CentroidSteerOptimizer(
        model, rank=8, update_gap=5, use_triangular_core=True, adaptive_subspace_refresh=False
    )

    x = torch.randn(2, 8, 128, device=device, dtype=torch.bfloat16)
    out, _ = model(x)
    out.sum().backward()

    # Zero out one expert's gradients to simulate an unrouted expert
    expert_groups = [g for g in opt.layer_groups if g["module"] is not None]
    expert_groups[0]["param"].grad.zero_()

    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    telem_step1 = opt.get_qr_telemetry()
    assert telem_step1["primary_fallback_reasons"]["degenerate_columns"] == 0, (
        "Unrouted zero-gradient experts must be skipped before QR, causing 0 degenerate_columns fallbacks"
    )
    qr_calls_step1 = telem_step1["total_qr_calls"]
    assert qr_calls_step1 > 0

    # Step 2 (non-refresh step, since update_gap=5): all already-initialized experts should reuse bases with 0 new QR calls
    out, _ = model(x)
    out.sum().backward()
    expert_groups[0]["param"].grad.zero_()  # keep same expert unrouted
    opt.step_parameters(model.parameters(), is_global_step_end=True)
    opt.zero_grad()

    telem_step2 = opt.get_qr_telemetry()
    assert telem_step2["total_qr_calls"] == qr_calls_step1, (
        f"Non-refresh step must execute 0 new QR calls, expected {qr_calls_step1}, got {telem_step2['total_qr_calls']}"
    )
    print("✅ sCholQR2 shift-bias elimination and zero-grad/non-refresh QR skipping verified!")


def test_benchmark_config_validation_and_scientific_pvalues():
    """Verifies upfront parameter validation and non-zero scientific p-value preservation in verify_triangular_qr_residuals."""
    import sys
    from pathlib import Path

    import pytest

    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from benchmarks.verify_triangular_qr_residuals import (
        compute_paired_and_tost,
        format_p_value,
        run_numerical_qr_residual_verification,
        validate_experiment_configuration,
    )

    # 1. Upfront fail-fast validation: opt_steps > train_pool_batches must raise ValueError immediately
    with pytest.raises(ValueError, match="opt_steps.*cannot exceed train_pool_batches"):
        validate_experiment_configuration(opt_steps=500, train_pool_batches=200)

    with pytest.raises(ValueError, match="opt_steps.*cannot exceed train_pool_batches"):
        run_numerical_qr_residual_verification(opt_steps=250, train_pool_batches=100)

    # 2. Invalid scalar bounds, duplicate seeds, <2 eval seeds, unbalanced Latin square, and invalid synthetic shapes
    with pytest.raises(ValueError, match="seq_len"):
        validate_experiment_configuration(seq_len=1)

    with pytest.raises(ValueError, match="duplicate"):
        validate_experiment_configuration(seeds=[42, 1337, 42])

    with pytest.raises(ValueError, match="at least 2 integer seeds"):
        validate_experiment_configuration(opt_eval_seeds=[42])

    with pytest.raises(ValueError, match="duplicate"):
        validate_experiment_configuration(opt_eval_seeds=[42, 1337, 1337])

    with pytest.raises(ValueError, match="multiple of 3"):
        validate_experiment_configuration(opt_eval_seeds=[42, 1337, 2026, 7777], require_balanced_latin_square=True)

    with pytest.raises(ValueError, match="rank r=20 cannot exceed"):
        validate_experiment_configuration(synthetic_shapes=[(16, 32, 20)])

    # 3. Scientific p-value preservation (never rounded to 0.0)
    vals_tri = [4.1700 + 0.0002 * ((-1) ** i) for i in range(12)]
    vals_adm = [4.1663 + 0.0001 * ((-1) ** i) for i in range(12)]
    stats_out = compute_paired_and_tost(vals_tri, vals_adm, eq_margin=0.05)

    assert 0.0 < stats_out["p_value"] < 1e-4, f"Small p_value must retain non-zero float precision, got {stats_out['p_value']}"
    assert "<0.0001" in stats_out["p_value_formatted"]

    for margin_key, m_entry in stats_out["tost_sensitivity_profile"].items():
        assert 0.0 < m_entry["p_tost"] < 1e-4, f"{margin_key} p_tost must not be rounded to 0.0, got {m_entry['p_tost']}"
        assert 0.0 < m_entry["p_upper"] < 1e-4, f"{margin_key} p_upper must not be rounded to 0.0, got {m_entry['p_upper']}"
        assert "<0.0001" in m_entry["p_tost_formatted"]

    assert format_p_value(1.5129e-7) == "1.513e-07 (<0.0001)"
    assert format_p_value(0.0200) == "0.0200"
    print("✅ Benchmark configuration validation and scientific p-value formatting verified!")


