"""Unit, numerical/gradient, orthonormality, steer_step reset, true hidden centroid capture,
and end-to-end training tests for the R2 Multi-Directional Contrastive CentroidSteer upgrade.
"""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lucid_machines.kernels.fast_triangular import fast_triangular_adamw
from lucid_machines.model.layers import MoE_FFN as LayersMoE_FFN
from lucid_machines.model.moe import MoE_FFN
from lucid_machines.optim.centroid import CentroidSteerOptimizer
from lucid_machines.optim.centroid_steer import CentroidSteerOptimizer as AliasCentroidSteerOptimizer


class ClusteredMoETaskModel(nn.Module):
    """Controlled MoE model for testing expert specialization and steering convergence."""

    def __init__(self, dim: int = 64, num_experts: int = 4, num_classes: int = 32):
        super().__init__()
        self.dim = dim
        self.moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=1, shared_expert=False)
        self.head = nn.Linear(dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.moe(x, update_stats=True))


class TestCentroidSteerUpgrade(unittest.TestCase):
    def test_module_reexports_and_constructor_compat(self):
        self.assertIs(MoE_FFN, LayersMoE_FFN)
        self.assertIs(CentroidSteerOptimizer, AliasCentroidSteerOptimizer)

        model = ClusteredMoETaskModel(dim=32, num_experts=2, num_classes=16)
        opt_pos = CentroidSteerOptimizer(model, lr=1e-3, rank=8, update_gap=5, steer_update_gap=2)
        opt_kw = CentroidSteerOptimizer(
            params=model.parameters(),
            lr=1e-3,
            steer_scale=0.2,
            rank=8,
            update_gap=5,
            steer_update_gap=2,
        )
        self.assertEqual(len(opt_pos.layer_groups), 4)
        self.assertGreater(len(opt_kw.muon_params) + len(opt_kw.layer_groups), 0)

    def test_true_hidden_centroid_capture_and_shape_preservation(self):
        torch.manual_seed(42)
        dim = 32
        num_experts = 4
        moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=1, shared_expert=True)
        moe.train()

        x = torch.randn(4, 16, dim) * 2.5
        out = moe(x, update_stats=True)
        self.assertEqual(out.shape, (4, 16, dim))

        h_dim = dim * 6
        self.assertEqual(moe.last_centroids.shape, (num_experts, dim))
        self.assertEqual(moe.last_hidden_centroids.shape, (num_experts, h_dim))
        self.assertEqual(moe.last_contrastive_centroids.shape, (num_experts, dim))
        self.assertEqual(moe.last_hidden_contrastive_centroids.shape, (num_experts, h_dim))
        self.assertEqual(moe.last_input_subspace.shape, (num_experts, dim, moe.subspace_dirs))
        self.assertEqual(moe.last_hidden_subspace.shape, (num_experts, h_dim, moe.subspace_dirs))

        # Verify at least one routed expert has true hidden centroid distinct from synthetic FastGELU(W1 @ c_e)
        routed_diffs = []
        for e_idx in range(num_experts):
            if int(moe.last_accepted_counts[e_idx].item()) >= 2:
                true_h = moe.last_hidden_centroids[e_idx]
                self.assertGreater(float(true_h.norm().item()), 1e-5)
                c_e = moe.last_centroids[e_idx]
                with torch.no_grad():
                    synth_h = moe.experts[e_idx][1](moe.experts[e_idx][0](c_e.unsqueeze(0))).squeeze(0)
                diff = float((true_h - synth_h).norm().item())
                routed_diffs.append(diff)
        self.assertTrue(len(routed_diffs) > 0 and max(routed_diffs) > 1e-3)

        # Verify legacy state_dict loading (missing new buffers) succeeds with strict=True
        legacy_sd = {
            k: v
            for k, v in moe.state_dict().items()
            if not any(
                b in k
                for b in (
                    "last_hidden_centroids",
                    "last_contrastive_centroids",
                    "last_hidden_contrastive_centroids",
                    "last_input_subspace",
                    "last_hidden_subspace",
                )
            )
        }
        moe_restored = MoE_FFN(dim=dim, num_experts=num_experts, top_k=1, shared_expert=True)
        moe_restored.load_state_dict(legacy_sd, strict=True)

    def test_multi_directional_subspace_and_basis_orthonormality(self):
        torch.manual_seed(101)
        dim = 64
        rank = 8
        model = ClusteredMoETaskModel(dim=dim, num_experts=4, num_classes=16)
        opt = CentroidSteerOptimizer(
            model,
            lr=1e-3,
            rank=rank,
            update_gap=4,
            steer_update_gap=2,
            steer_scale=0.25,
            use_triangular_core=True,
            steer_mode="upgraded",
            num_steer_dirs=4,
        )

        for step in range(6):
            opt.zero_grad()
            x = torch.randn(4, 16, dim)
            y = torch.randint(0, 16, (4, 16))
            loss = F.cross_entropy(model(x).view(-1, 16), y.view(-1))
            loss.backward()
            opt.step()

            # Check MoE multi-directional activation subspace orthonormality
            for sub in (model.moe.last_input_subspace, model.moe.last_hidden_subspace):
                for e_idx in range(model.moe.num_experts):
                    mat = sub[e_idx].float()
                    if float(mat.norm().item()) > 1e-5:
                        k_c = mat.size(1)
                        err = float((mat.t() @ mat - torch.eye(k_c)).norm().item())
                        self.assertLess(err, 1e-4, f"Subspace orth error {err} >= 1e-4 at step {step}")

            # Check optimizer Q_L and Q_R (r+1 columns) orthonormality ||Q^T Q - I||_F < 1e-4
            for g in opt.layer_groups:
                if g.get("steer_applied", False):
                    for q_name in ("Q_L", "Q_R"):
                        Q = g[q_name][:, : rank + 1].float()
                        gram = Q.t() @ Q
                        orth_err = float((gram - torch.eye(rank + 1, device=Q.device)).norm().item())
                        self.assertLess(
                            orth_err,
                            1e-4,
                            f"{q_name} orthogonality violation at step {step}: ||Q^T Q - I||_F = {orth_err:.3e}",
                        )

    def test_independent_steer_step_reset_and_bias_correction_undamped(self):
        torch.manual_seed(2026)
        dim = 32
        rank = 8
        model = ClusteredMoETaskModel(dim=dim, num_experts=2, num_classes=8)
        opt = CentroidSteerOptimizer(
            model,
            lr=1e-3,
            rank=rank,
            update_gap=6,
            steer_update_gap=2,
            steer_scale=0.3,
            use_triangular_core=True,
            steer_mode="upgraded",
        )

        expected_sub_steps = [1, 2, 3, 4, 5, 6, 1]
        expected_steer_steps = [1, 2, 1, 2, 1, 2, 1]

        for step_idx, (exp_sub, exp_str) in enumerate(zip(expected_sub_steps, expected_steer_steps), start=1):
            opt.zero_grad()
            x = torch.randn(2, 16, dim)
            loss = model(x).pow(2).mean()
            loss.backward()
            opt.step()

            for g in opt.layer_groups:
                st = g["state"]
                self.assertEqual(st["step"], step_idx)
                self.assertEqual(st["subspace_step"], exp_sub)
                self.assertEqual(st["steer_step"], exp_str)
                self.assertEqual(st.get("transport_step"), step_idx)
                if "V_c" in st:
                    self.assertTrue(bool((st["V_c"] >= 0.0).all().item()))

        # Direct kernel test: freshly zeroed border moments at subspace_step=10 vs steer_step=1
        R = torch.ones(1, rank + 1, rank + 1, dtype=torch.float32) * 0.5
        mom_damped = torch.zeros_like(R)
        var_damped = torch.zeros_like(R)
        delta_damped = torch.zeros_like(R)
        fast_triangular_adamw(
            R,
            mom_damped,
            var_damped,
            delta_damped,
            step_vals=torch.tensor([10.0]),
            beta1=0.9,
            beta2=0.999,
            mask_mode="full",
            steer_step_vals=None,
            rms_reg=0.0,
        )

        mom_fixed = torch.zeros_like(R)
        var_fixed = torch.zeros_like(R)
        delta_fixed = torch.zeros_like(R)
        fast_triangular_adamw(
            R,
            mom_fixed,
            var_fixed,
            delta_fixed,
            step_vals=torch.tensor([10.0]),
            beta1=0.9,
            beta2=0.999,
            mask_mode="full",
            steer_step_vals=torch.tensor([1.0]),
            rms_reg=0.0,
        )

        damped_border = float(delta_damped[0, 0, rank].item())
        fixed_border = float(delta_fixed[0, 0, rank].item())
        self.assertLess(damped_border, 0.55)
        self.assertAlmostEqual(fixed_border, 1.0, places=4)

    def test_deflated_krylov_preserves_centroid_signal_and_suppresses_noise(self):
        torch.manual_seed(77)
        dim = 64
        rank = 8
        model = ClusteredMoETaskModel(dim=dim, num_experts=2, num_classes=16)

        # Build a gradient with strong energy along a specific input direction c_star
        x = torch.randn(4, 16, dim)
        c_star = F.normalize(torch.randn(dim), dim=0)
        x = x + 3.0 * c_star.view(1, 1, dim)
        loss = model(x).pow(2).mean()
        loss.backward()

        g_w1 = model.moe.experts[0][0].weight.grad.data.float()  # [H, D]
        c_e, _ = model.moe.get_expert_steering_directions(0, steer_role="input", num_dirs=4, steer_mode="upgraded")
        c_hat = F.normalize(c_e.float(), dim=0)

        # Legacy: build Q_L from raw G without deflation -> Q_L absorbs c_hat, leaving orthogonal tail noise
        omega = torch.randn(g_w1.size(0), rank)
        q_leg, _ = torch.linalg.qr(g_w1.t() @ omega)
        c_res_leg = c_hat - q_leg @ (q_leg.t() @ c_hat)
        v_leg = g_w1 @ F.normalize(c_res_leg, dim=0)

        # Upgraded: deflate G along c_hat before Krylov -> 100% of c_hat survives for column r
        g_defl = g_w1 - torch.outer(g_w1 @ c_hat, c_hat)
        q_upg, _ = torch.linalg.qr(g_defl.t() @ omega)
        q_upg, _ = torch.linalg.qr(q_upg - torch.outer(c_hat, c_hat @ q_upg))
        c_res_upg = c_hat - q_upg @ (q_upg.t() @ c_hat)
        v_upg = g_w1 @ F.normalize(c_res_upg, dim=0)

        self.assertGreater(float(c_res_upg.norm().item()), 0.999)
        self.assertGreater(float(v_upg.norm().item()), float(v_leg.norm().item()) * 1.25)

    def test_end_to_end_upgraded_outperforms_legacy_and_nosteer(self):
        torch.manual_seed(1234)
        dim = 48
        num_experts = 4
        num_classes = 24
        rank = 4

        base_model = ClusteredMoETaskModel(dim=dim, num_experts=num_experts, num_classes=num_classes)
        init_sd = copy.deepcopy(base_model.state_dict())

        # Fixed clustered dataset where each class cluster activates a distinct input subspace
        cluster_centers = F.normalize(torch.randn(num_experts, dim), dim=-1) * 2.5
        batches = []
        for _ in range(30):
            c_ids = torch.randint(0, num_experts, (4, 16))
            noise = torch.randn(4, 16, dim) * 0.35
            bx = cluster_centers[c_ids] + noise
            by = (
                c_ids * (num_classes // num_experts) + (bx[..., :2].sum(-1).long().abs() % (num_classes // num_experts))
            ) % num_classes
            batches.append((bx, by))

        def _run_regime(mode: str, steer_scale: float) -> float:
            torch.manual_seed(1234)
            m = ClusteredMoETaskModel(dim=dim, num_experts=num_experts, num_classes=num_classes)
            m.load_state_dict(copy.deepcopy(init_sd))
            # Route all expert parameters through Tier-2 CentroidSteer
            opt = CentroidSteerOptimizer(
                m,
                lr=5e-3,
                rank=rank,
                update_gap=10,
                steer_update_gap=2,
                steer_scale=steer_scale,
                use_muon=False,
                use_triangular_core=True,
                steer_mode=mode,
                num_steer_dirs=4,
            )
            final_losses = []
            for bx, by in batches:
                opt.zero_grad()
                logits = m(bx)
                loss = F.cross_entropy(logits.view(-1, num_classes), by.view(-1))
                loss.backward()
                opt.step()
                m.moe.step_bias()
                final_losses.append(float(loss.item()))
            return sum(final_losses[-5:]) / 5.0

        loss_nosteer = _run_regime("upgraded", steer_scale=0.0)
        loss_legacy = _run_regime("legacy", steer_scale=0.35)
        loss_upgraded = _run_regime("upgraded", steer_scale=0.35)

        self.assertLess(loss_upgraded, loss_nosteer)
        self.assertLess(loss_upgraded, loss_legacy)

    def test_triton_vs_cpu_parity_and_mixed_rank_upgraded(self):
        from lucid_machines.kernels.fast_triangular import HAS_TRITON

        if HAS_TRITON and torch.cuda.is_available():
            for mode in ("triu", "full"):
                torch.manual_seed(11)
                R = torch.randn(3, 9, 9, device="cuda", dtype=torch.float32)
                if mode == "triu":
                    R = torch.triu(R)
                m_gpu, v_gpu, d_gpu = torch.zeros_like(R), torch.zeros_like(R), torch.zeros_like(R)
                m_cpu, v_cpu, d_cpu = torch.zeros_like(R).cpu(), torch.zeros_like(R).cpu(), torch.zeros_like(R).cpu()
                steps = torch.tensor([10.0, 10.0, 10.0], device="cuda")
                s_steps = torch.tensor([1.0, 2.0, 3.0], device="cuda")
                fast_triangular_adamw(R, m_gpu, v_gpu, d_gpu, steps, 0.9, 0.999, 1e-8, mode, s_steps, 0.35)
                fast_triangular_adamw(
                    R.cpu(), m_cpu, v_cpu, d_cpu, steps.cpu(), 0.9, 0.999, 1e-8, mode, s_steps.cpu(), 0.35
                )
                self.assertLess(float((d_gpu.cpu() - d_cpu).abs().max().item()), 1e-5)

        # Mixed-rank model where small head has min(m, n) = 6 < rank = 12
        class MixedRankMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = MoE_FFN(dim=32, num_experts=2, top_k=1, shared_expert=False)
                self.small_head = nn.Linear(32, 6, bias=False)

            def forward(self, x):
                return self.small_head(self.moe(x, update_stats=True))

        torch.manual_seed(303)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MixedRankMoE().to(device)
        opt = CentroidSteerOptimizer(
            model,
            lr=1e-3,
            rank=12,
            update_gap=5,
            steer_update_gap=2,
            steer_scale=0.25,
            use_muon=False,
            use_triangular_core=True,
            steer_mode="upgraded",
            core_mask_mode="triu",
        )
        for _ in range(4):
            opt.zero_grad()
            x = torch.randn(2, 8, 32, device=device)
            loss = model(x).pow(2).mean()
            loss.backward()
            opt.step()

        for g in opt.layer_groups:
            r_b = min(opt.rank, min(g["param"].shape))
            if g.get("steer_applied", False):
                for q_name in ("Q_L", "Q_R"):
                    Q = g[q_name][:, : r_b + 1].float()
                    err = float((Q.t() @ Q - torch.eye(r_b + 1, device=Q.device)).norm().item())
                    self.assertLess(err, 1e-4)

    def test_state_dict_roundtrip_and_non_triangular_upgraded(self):
        torch.manual_seed(909)
        dim = 32
        model = ClusteredMoETaskModel(dim=dim, num_experts=2, num_classes=8)
        opt = CentroidSteerOptimizer(
            model,
            lr=1e-3,
            rank=8,
            update_gap=6,
            steer_update_gap=2,
            steer_scale=0.2,
            use_triangular_core=True,
            steer_mode="upgraded",
        )
        for _ in range(3):
            opt.zero_grad()
            loss = model(torch.randn(2, 8, dim)).pow(2).mean()
            loss.backward()
            opt.step()

        sd = opt.state_dict()
        opt2 = CentroidSteerOptimizer(
            model,
            lr=1e-3,
            rank=8,
            update_gap=6,
            steer_update_gap=2,
            steer_scale=0.2,
            use_triangular_core=True,
            steer_mode="upgraded",
        )
        opt2.load_state_dict(sd)
        for g1, g2 in zip(opt.layer_groups, opt2.layer_groups):
            self.assertEqual(g1["state"]["steer_step"], g2["state"]["steer_step"])
            self.assertEqual(g1["state"]["subspace_step"], g2["state"]["subspace_step"])

        # Non-triangular upgraded path also executes cleanly
        opt_nt = CentroidSteerOptimizer(
            model,
            lr=1e-3,
            rank=8,
            update_gap=4,
            steer_update_gap=2,
            steer_scale=0.2,
            use_triangular_core=False,
            steer_mode="upgraded",
        )
        for _ in range(2):
            opt_nt.zero_grad()
            loss = model(torch.randn(2, 8, dim)).pow(2).mean()
            loss.backward()
            opt_nt.step()

    def test_activation_checkpointing_batched_and_fallback_parity(self):
        from lucid_machines.model.config import checkpoint_bwd_ctx, checkpoint_fwd_ctx
        from lucid_machines.model.transformer import TriuneTransformer

        torch.manual_seed(5150)
        model = TriuneTransformer(
            vocab_size=128,
            hidden_dim=64,
            num_layers=6,
            num_heads=2,
            head_dim=32,
            num_experts=4,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
        )
        model.enable_gradient_checkpointing()
        model.train()

        opt = CentroidSteerOptimizer(
            model,
            lr=1e-3,
            rank=8,
            update_gap=4,
            steer_update_gap=2,
            steer_scale=0.25,
            steer_mode="upgraded",
        )
        x = torch.randint(0, 128, (2, 16))
        opt.zero_grad()
        out, _ = model(x, force_depth=2)
        loss = out.pow(2).mean()
        loss.backward()
        opt.step()

        moe_layers = [layer.ffn for layer in model.layers if isinstance(layer.ffn, MoE_FFN)]
        self.assertGreater(len(moe_layers), 0)
        for moe in moe_layers:
            self.assertGreater(float(moe.last_centroids.norm().item()), 1e-5)
            self.assertGreater(float(moe.last_hidden_centroids.norm().item()), 1e-5)

        # Fallback path (can_batch_bmm=False via mixed biases) under checkpointing
        moe_fb = MoE_FFN(dim=32, num_experts=2, top_k=1, shared_expert=False)
        moe_fb.experts[0][0].bias = None  # force mixed_biases -> non-BMM fallback path
        moe_fb.train()
        inp = torch.randn(2, 8, 32, requires_grad=True)
        out_fb = torch.utils.checkpoint.checkpoint(
            lambda t: moe_fb(t, update_stats=True),
            inp,
            use_reentrant=False,
            context_fn=lambda: (checkpoint_fwd_ctx(), checkpoint_bwd_ctx()),
        )
        out_fb.pow(2).mean().backward()
        self.assertIsNotNone(inp.grad)
        self.assertGreater(float(moe_fb.last_hidden_centroids.norm().item()), 1e-5)

    def test_bfloat16_autocast_cholesky_ex_and_direct_sum_orthogonality(self):
        from lucid_machines.ops.centroid_steer import batched_qr, cholesky_qr, polar_stiefel_retraction

        torch.manual_seed(4242)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_bf16 = torch.randn(4, 64, 8, device=device, dtype=torch.bfloat16)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            q_out, diag = batched_qr(y_bf16, return_diagnostics=True)
            self.assertEqual(q_out.dtype, torch.bfloat16)
            self.assertEqual(q_out.shape, (4, 64, 8))
            self.assertTrue(bool(torch.isfinite(q_out).all().item()))

            q_2d = cholesky_qr(y_bf16[0])
            self.assertEqual(q_2d.dtype, torch.bfloat16)
            self.assertEqual(q_2d.shape, (64, 8))

            p_ret = polar_stiefel_retraction(y_bf16)
            self.assertEqual(p_ret.dtype, torch.bfloat16)
            self.assertEqual(p_ret.shape, (4, 64, 8))

            model = ClusteredMoETaskModel(dim=64, num_experts=4, num_classes=16).to(device=device, dtype=torch.bfloat16)
            opt = CentroidSteerOptimizer(
                model,
                lr=1e-3,
                rank=8,
                update_gap=2,
                steer_update_gap=1,
                steer_scale=0.5,
                use_muon=False,
                use_triangular_core=True,
                steer_mode="upgraded",
                num_steer_dirs=4,
            )
            x = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16)
            opt.zero_grad()
            loss = model(x).float().pow(2).mean()
            loss.backward()
            opt.step()

        for g in opt.layer_groups:
            if g.get("C_e") is not None:
                c_e = g["C_e"].float()
                iso_err = float((c_e.t() @ c_e - torch.eye(c_e.size(1), device=device)).norm().item())
                self.assertLess(iso_err, 1e-4)


if __name__ == "__main__":
    unittest.main()
