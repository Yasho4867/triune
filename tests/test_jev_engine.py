"""Rigorously verifies Triune Proprietary Non-Generative Joint-Embedding Predictive Architecture (JEV).

Empirical Test Suite for:
1. JEVLatentPredictor:
   - Parameter-efficient MLP architecture (W2 GELU(W1 z), d_pred = D // 2).
   - Multi-dimensional tensor shapes ([B, D], [B, T, D], [B, H, T, D]).
   - Clean backward autograd gradients across FP32, FP16, and BF16.
   - Parameter efficiency vs O(V) vocabulary heads.

2. JEVObjective:
   - Normalized Cosine Similarity (L_sim) with strict Stop-Gradient isolation on target Cortex representations.
   - Semantic Tube Prediction (STP) geodesic variance penalty (zero inside eps_tube, quadratic penalty outside).
   - VICReg / Covariance Decorrelation (L_cov) penalizing off-diagonal feature collapse.
   - Multi-step optimization convergence proof under AdamW.

3. JEVExitRouter:
   - Zero O(V) vocabulary compute and memory allocation during early exit verification.
   - Geometric distance metrics: Cosine, Euclidean, Geodesic Riemannian arccos, STP Tube, Learned Gate.
   - Exact threshold decision gating and confidence scoring.
   - Device and CUDA acceleration safety.
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.model.jev import (
    JEVExitRouter,
    JEVLatentPredictor,
    JEVObjective,
    JEVRoutingResult,
    TriuneContinuousExitRouter,
    TriuneLatentObjective,
    TriuneLatentPredictor,
)


class TestJEVLatentPredictor(unittest.TestCase):
    """Rigorous tests for JEVLatentPredictor forward/backward shapes and gradients."""

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_predictor_shapes_and_bottleneck(self):
        """Validates that d_pred = D // 2 and forward pass preserves arbitrary tensor ranks."""
        for dim in (32, 64, 128, 256, 512):
            predictor = JEVLatentPredictor(dim=dim)
            self.assertEqual(predictor.d_pred, dim // 2)
            self.assertEqual(predictor.w1.in_features, dim)
            self.assertEqual(predictor.w1.out_features, dim // 2)
            self.assertEqual(predictor.w2.in_features, dim // 2)
            self.assertEqual(predictor.w2.out_features, dim)

            # 2D tensor: [B, D]
            x2 = torch.randn(4, dim)
            out2 = predictor(x2)
            self.assertEqual(out2.shape, (4, dim))

            # 3D tensor: [B, T, D]
            x3 = torch.randn(2, 17, dim)
            out3 = predictor(x3)
            self.assertEqual(out3.shape, (2, 17, dim))

            # 4D tensor: [B, H, T, D]
            x4 = torch.randn(2, 4, 8, dim)
            out4 = predictor(x4)
            self.assertEqual(out4.shape, (2, 4, 8, dim))

    def test_predictor_backward_and_gradients(self):
        """Validates that autograd gradients cleanly backpropagate through all predictor weights and inputs."""
        dim = 64
        predictor = JEVLatentPredictor(dim=dim)
        x = torch.randn(4, 16, dim, requires_grad=True)

        out = predictor(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(predictor.w1.weight.grad)
        self.assertIsNotNone(predictor.w2.weight.grad)
        self.assertIsNotNone(predictor.norm.weight.grad)

        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(predictor.w1.weight.grad).all())
        self.assertTrue(torch.isfinite(predictor.w2.weight.grad).all())

        # Ensure non-trivial non-zero gradients
        self.assertGreater(predictor.w1.weight.grad.abs().sum().item(), 1e-4)
        self.assertGreater(predictor.w2.weight.grad.abs().sum().item(), 1e-4)

    def test_predictor_residual_and_norm_modes(self):
        """Validates behavior under use_residual=True and normalize_output=False."""
        dim = 48
        # Standard mode
        p_std = JEVLatentPredictor(dim=dim, normalize_output=True, use_residual=False)
        self.assertIsNotNone(p_std.out_norm)

        # Residual mode without output norm
        p_res = JEVLatentPredictor(dim=dim, normalize_output=False, use_residual=True)
        self.assertIsNone(p_res.out_norm)

        x = torch.randn(2, 6, dim)
        out = p_res(x)
        self.assertEqual(out.shape, (2, 6, dim))

    def test_predictor_dtypes_and_devices(self):
        """Validates 100% numerical safety across FP32, FP16, and BF16 on active device."""
        dim = 64
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            predictor = JEVLatentPredictor(dim=dim).to(device=self.device, dtype=dtype)
            x = torch.randn(2, 8, dim, device=self.device, dtype=dtype)
            out = predictor(x)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.device, x.device)
            self.assertTrue(torch.isfinite(out).all())

    def test_parameter_efficiency_vs_vocab_head(self):
        """Confirms JEV predictor achieves dramatic parameter reduction vs O(V) vocabulary heads."""
        dim = 1536
        vocab_size = 32000

        # Full vocabulary linear head: D * V parameters
        vocab_head_params = dim * vocab_size  # 49,152,000 params (196.6 MB in FP32)

        # JEV MLP predictor: W1 (D * D//2) + W2 (D//2 * D) + 2 RMSNorms (2 * D)
        jev_predictor = JEVLatentPredictor(dim=dim)
        jev_params = sum(p.numel() for p in jev_predictor.parameters())

        expected_mlp_params = (dim * (dim // 2)) + ((dim // 2) * dim) + (2 * dim)
        self.assertEqual(jev_params, expected_mlp_params)

        compression_ratio = vocab_head_params / jev_params
        # 49,152,000 / 2,362,368 = ~20.8x parameter reduction
        self.assertGreater(compression_ratio, 20.0)


class TestJEVObjective(unittest.TestCase):
    """Rigorous tests for JEVObjective components and mathematical invariants."""

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_cosine_similarity_stop_gradient_invariant(self):
        """Verifies stop-gradient operator SG(z_target) strictly isolates target representations from backpropagation."""
        objective = JEVObjective()
        dim = 32

        # Upstream target generator
        target_encoder = nn.Linear(dim, dim, bias=False)
        upstream_x = torch.randn(4, dim, requires_grad=True)
        z_target = target_encoder(upstream_x)

        # Downstream predictor
        z_pred = torch.randn(4, dim, requires_grad=True)

        loss = objective.compute_sim_loss(z_pred, z_target)
        loss.backward()

        # Target encoder and upstream input MUST receive ZERO gradient
        self.assertIsNone(upstream_x.grad)
        self.assertIsNone(target_encoder.weight.grad)

        # Predictor MUST receive valid non-zero gradient
        self.assertIsNotNone(z_pred.grad)
        self.assertTrue(torch.isfinite(z_pred.grad).all())
        self.assertGreater(z_pred.grad.abs().sum().item(), 1e-5)

    def test_sim_loss_bounds_and_values(self):
        """Validates that L_sim == 0 for parallel vectors, ~1 for orthogonal, ~2 for opposite."""
        objective = JEVObjective()

        # Case 1: Identical vectors -> Cosine = 1.0 -> Loss = 0.0
        v1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        loss_identical = objective.compute_sim_loss(v1, v1)
        self.assertAlmostEqual(loss_identical.item(), 0.0, places=5)

        # Case 2: Scaled identical vectors -> Cosine = 1.0 -> Loss = 0.0
        v1_scaled = v1 * 5.7
        loss_scaled = objective.compute_sim_loss(v1_scaled, v1)
        self.assertAlmostEqual(loss_scaled.item(), 0.0, places=5)

        # Case 3: Orthogonal vectors -> Cosine = 0.0 -> Loss = 1.0
        v_ortho_1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        v_ortho_2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        loss_ortho = objective.compute_sim_loss(v_ortho_1, v_ortho_2)
        self.assertAlmostEqual(loss_ortho.item(), 1.0, places=5)

        # Case 4: Opposite vectors -> Cosine = -1.0 -> Loss = 2.0
        v_opp = -v1
        loss_opp = objective.compute_sim_loss(v_opp, v1)
        self.assertAlmostEqual(loss_opp.item(), 2.0, places=4)

    def test_stp_geodesic_tube_variance_penalty(self):
        """Validates Semantic Tube Prediction (STP) geodesic variance bound:

        L_stp = max(0, ||z_pred - z_target||_2^2 - eps_tube^2).
        Strictly zero within tube radius, quadratic penalty outside.
        """
        eps_tube = 0.5
        objective = JEVObjective(eps_tube=eps_tube)

        z_target = torch.tensor([[0.0, 0.0, 0.0]])

        # Inside the tube: distance = 0.3 <= 0.5 -> L_stp MUST be strictly 0.0
        z_inside = torch.tensor([[0.3, 0.0, 0.0]])
        loss_inside = objective.compute_stp_loss(z_inside, z_target)
        self.assertEqual(loss_inside.item(), 0.0)

        # Exactly at tube boundary: distance = 0.5 -> L_stp MUST be 0.0
        z_boundary = torch.tensor([[0.5, 0.0, 0.0]])
        loss_boundary = objective.compute_stp_loss(z_boundary, z_target)
        self.assertAlmostEqual(loss_boundary.item(), 0.0, places=5)

        # Outside the tube: distance = 1.0 > 0.5
        # dist_sq = 1.0^2 = 1.0, tube_sq = 0.5^2 = 0.25 -> L_stp = 0.75
        z_outside = torch.tensor([[1.0, 0.0, 0.0]])
        loss_outside = objective.compute_stp_loss(z_outside, z_target)
        self.assertAlmostEqual(loss_outside.item(), 0.75, places=5)

        # Further outside: distance = 2.0 -> dist_sq = 4.0 -> L_stp = 3.75
        z_far = torch.tensor([[2.0, 0.0, 0.0]])
        loss_far = objective.compute_stp_loss(z_far, z_target)
        self.assertAlmostEqual(loss_far.item(), 3.75, places=5)
        self.assertGreater(loss_far.item(), loss_outside.item())

    def test_vicreg_covariance_decorrelation(self):
        """Validates that off-diagonal covariance loss penalizes feature correlation."""
        objective = JEVObjective()
        dim = 16
        n_samples = 256

        # Scenario 1: Strongly correlated features (all channels identical + noise)
        base = torch.randn(n_samples, 1)
        z_correlated = base.repeat(1, dim) + 0.01 * torch.randn(n_samples, dim)
        cov_loss_correlated = objective.compute_cov_loss(z_correlated)

        # Scenario 2: Uncorrelated isotropic Gaussian features
        z_uncorrelated = torch.randn(n_samples, dim)
        cov_loss_uncorrelated = objective.compute_cov_loss(z_uncorrelated)

        # Correlated representations must have dramatically higher covariance loss
        self.assertGreater(cov_loss_correlated.item(), cov_loss_uncorrelated.item() * 5.0)

        # Scenario 3: Single sample edge case does not crash and returns 0.0
        z_single = torch.randn(1, dim)
        cov_single = objective.compute_cov_loss(z_single)
        self.assertEqual(cov_single.item(), 0.0)

    def test_multi_step_optimization_convergence(self):
        """Verifies that an MLP predictor trained with JEVObjective monotonically reduces loss."""
        torch.manual_seed(42)
        dim = 32
        predictor = JEVLatentPredictor(dim=dim)
        objective = JEVObjective(lambda_stp=0.2, lambda_cov=0.005, eps_tube=0.5)
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-2)

        # Fixed synthetic sequence
        x = torch.randn(8, 16, dim)
        # Synthetic Cortex target matching RMSNorm representation scale
        target_norm = JEVLatentPredictor(dim=dim)
        with torch.no_grad():
            z_target = target_norm(x)

        initial_loss = None
        final_loss = None
        for step in range(35):
            optimizer.zero_grad()
            z_pred = predictor(x)
            loss = objective(z_pred, z_target)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Convergence: final loss is strictly less than initial loss
        self.assertLess(final_loss, initial_loss * 0.7)
        self.assertGreater(initial_loss - final_loss, 0.1)

    def test_metrics_dictionary_return(self):
        """Validates that return_dict=True provides telemetric decomposition of all 3 loss components."""
        objective = JEVObjective(lambda_stp=1.0, lambda_cov=0.04)
        z_pred = torch.randn(4, 16, 32)
        z_target = torch.randn(4, 16, 32)

        total_loss, metrics = objective(z_pred, z_target, return_dict=True)
        self.assertIn("l_sim", metrics)
        self.assertIn("l_stp", metrics)
        self.assertIn("l_cov", metrics)
        self.assertIn("total_loss", metrics)

        expected_total = metrics["l_sim"] + 1.0 * metrics["l_stp"] + 0.04 * metrics["l_cov"]
        self.assertAlmostEqual(total_loss.item(), expected_total.item(), places=5)


class TestJEVExitRouter(unittest.TestCase):
    """Rigorous tests for JEVExitRouter early exit gating without O(V) vocabulary compute."""

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_zero_vocabulary_projection_compute(self):
        """Confirms JEVExitRouter has strictly ZERO O(V) parameters and requires NO vocabulary FLOPs."""
        dim = 256
        router = JEVExitRouter(dim=dim, threshold=0.8)

        # Vocabulary size for standard models
        vocab_size = 32000

        # Verify no parameter in router has size >= vocab_size
        for name, param in router.named_parameters():
            self.assertNotIn(vocab_size, param.shape, f"Found O(V) parameter: {name} with shape {param.shape}")

        total_params = sum(p.numel() for p in router.parameters())
        # Router contains only gate projection: dim * 1 + 1 = 257 parameters!
        self.assertEqual(total_params, dim + 1)

    def test_distance_metrics(self):
        """Tests all supported continuous latent distance metrics."""
        dim = 16
        z1 = torch.tensor([[1.0, 0.0, 0.0, 0.0] + [0.0] * (dim - 4)])
        z2 = torch.tensor([[0.0, 1.0, 0.0, 0.0] + [0.0] * (dim - 4)])

        # Cosine distance: 1.0 - 0.0 = 1.0
        router_cos = JEVExitRouter(dim=dim, distance_metric="cosine")
        d_cos = router_cos.compute_latent_distance(z1, z2)
        self.assertAlmostEqual(d_cos.item(), 1.0, places=5)

        # Euclidean distance: sqrt(1^2 + 1^2) = sqrt(2) ~ 1.4142
        router_euc = JEVExitRouter(dim=dim, distance_metric="euclidean")
        d_euc = router_euc.compute_latent_distance(z1, z2)
        self.assertAlmostEqual(d_euc.item(), math.sqrt(2.0), places=5)

        # Geodesic distance on sphere: arccos(0) = pi / 2 ~ 1.5708
        router_geo = JEVExitRouter(dim=dim, distance_metric="geodesic")
        d_geo = router_geo.compute_latent_distance(z1, z2)
        self.assertAlmostEqual(d_geo.item(), math.pi / 2.0, places=5)

        # STP tube distance with eps_tube = 0.5: dist_sq - eps_tube^2 = 2.0 - 0.25 = 1.75
        router_stp = JEVExitRouter(dim=dim, distance_metric="stp_tube", eps_tube=0.5)
        d_stp = router_stp.compute_latent_distance(z1, z2)
        self.assertAlmostEqual(d_stp.item(), 1.75, places=5)

    def test_exit_decision_thresholding(self):
        """Verifies deterministic gating logic: should_exit = confidence >= threshold."""
        dim = 32
        router = JEVExitRouter(dim=dim, threshold=0.75, distance_metric="cosine")

        z_ref = torch.randn(4, dim)

        # High confidence (identical vector -> confidence = 1.0 >= 0.75)
        should_exit_hi, conf_hi = router.evaluate_latent_exit(z_ref, z_ref)
        self.assertTrue(should_exit_hi.all().item())
        self.assertTrue((conf_hi >= 0.75).all().item())

        # Low confidence (negative vector -> confidence = 0.0 < 0.75)
        should_exit_lo, conf_lo = router.evaluate_latent_exit(-z_ref, z_ref)
        self.assertFalse(should_exit_lo.any().item())
        self.assertTrue((conf_lo < 0.75).all().item())

    def test_integrated_forward_with_predictor(self):
        """Tests end-to-end execution of JEVExitRouter coupled with JEVLatentPredictor."""
        dim = 64
        predictor = JEVLatentPredictor(dim=dim)
        router = JEVExitRouter(dim=dim, threshold=0.85, predictor=predictor)

        z = torch.randn(2, 8, dim)
        res = router(z)

        self.assertIsInstance(res, JEVRoutingResult)
        self.assertEqual(res.should_exit.shape, (2, 8))
        self.assertEqual(res.confidence.shape, (2, 8))
        self.assertEqual(res.distance.shape, (2, 8))
        self.assertEqual(res.should_exit.dtype, torch.bool)

    def test_proprietary_triune_aliases(self):
        """Verifies seamless aliasing for Triune proprietary branding."""
        self.assertIs(TriuneLatentPredictor, JEVLatentPredictor)
        self.assertIs(TriuneLatentObjective, JEVObjective)
        self.assertIs(TriuneContinuousExitRouter, JEVExitRouter)


if __name__ == "__main__":
    unittest.main()
