from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from triune.model import (
    ContinuousLatentExitRouter,
    JEVExitRouter,
    JEVLatentPredictor,
    JEVObjective,
    JEVRoutingResult,
    JointEmbeddingObjective,
    LatentPredictor,
    attach_early_exits,
)


class MockTransformerForJEV(nn.Module):
    def __init__(self, vocab_size: int = 200, hidden_dim: int = 64, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.final_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        return self.final_head(x)


class TestJEVLatentPredictor(unittest.TestCase):
    def test_architecture_and_shapes(self):
        hidden_dim = 128
        predictor = JEVLatentPredictor(hidden_dim)

        self.assertEqual(predictor.d_pred, hidden_dim // 2)
        self.assertEqual(predictor.w1.out_features, 64)
        self.assertEqual(predictor.w2.in_features, 64)
        self.assertEqual(predictor.w2.out_features, hidden_dim)

        x = torch.randn(2, 16, hidden_dim)
        pred = predictor(x)
        self.assertEqual(pred.shape, (2, 16, hidden_dim))

        x_2d = torch.randn(4, hidden_dim)
        pred_2d = predictor(x_2d)
        self.assertEqual(pred_2d.shape, (4, hidden_dim))

    def test_dtype_compatibility_and_autograd(self):
        hidden_dim = 64
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            p = JEVLatentPredictor(hidden_dim).to(dtype=dtype)
            x = torch.randn(2, 8, hidden_dim, dtype=dtype, requires_grad=True)
            out = p(x)
            self.assertEqual(out.dtype, dtype)

            if dtype in (torch.float32, torch.bfloat16):
                loss = out.float().sum()
                loss.backward()
                self.assertIsNotNone(x.grad)
                self.assertIsNotNone(p.w1.weight.grad)
                self.assertIsNotNone(p.w2.weight.grad)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_aliases(self):
        hidden_dim = 32
        p1 = JEVLatentPredictor(hidden_dim)
        p2 = LatentPredictor(hidden_dim)
        self.assertIsInstance(p2, JEVLatentPredictor)
        self.assertEqual(p1.d_pred, p2.d_pred)


class TestJEVObjective(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hidden_dim = 64
        self.objective = JEVObjective(lambda_stp=1.0, lambda_cov=0.04, eps_tube=0.1)

    def test_sim_loss_with_stop_gradient(self):
        z_pred = torch.randn(4, 16, self.hidden_dim, requires_grad=True)
        z_target = torch.randn(4, 16, self.hidden_dim, requires_grad=True)

        loss = self.objective.compute_sim_loss(z_pred, z_target)
        loss.backward()

        self.assertIsNone(z_target.grad)
        self.assertIsNotNone(z_pred.grad)
        self.assertTrue(torch.isfinite(z_pred.grad).all())
        self.assertGreater(z_pred.grad.abs().sum().item(), 0.0)

    def test_sim_loss_bounds(self):
        z = torch.randn(2, 8, self.hidden_dim)
        loss_identical = self.objective.compute_sim_loss(z, z)
        self.assertAlmostEqual(loss_identical.item(), 0.0, places=5)

        loss_opposite = self.objective.compute_sim_loss(z, -z)
        self.assertAlmostEqual(loss_opposite.item(), 2.0, places=5)

    def test_stp_geodesic_tube_variance_bound(self):
        eps_tube = 0.5
        obj = JEVObjective(eps_tube=eps_tube)

        z_target = torch.randn(4, self.hidden_dim)
        perturbation = torch.randn(4, self.hidden_dim)
        perturbation = perturbation / perturbation.norm(dim=-1, keepdim=True) * 0.2
        z_pred_inside = z_target + perturbation

        loss_inside = obj.compute_stp_loss(z_pred_inside, z_target)
        self.assertEqual(loss_inside.item(), 0.0)

        perturbation_outside = perturbation / perturbation.norm(dim=-1, keepdim=True) * 1.0
        z_pred_outside = z_target + perturbation_outside

        loss_outside = obj.compute_stp_loss(z_pred_outside, z_target)
        self.assertAlmostEqual(loss_outside.item(), 0.75, places=4)

    def test_cov_decorrelation_anti_collapse(self):
        batch_size = 64
        single_val = torch.randn(batch_size, 1)
        z_collapsed = single_val.repeat(1, self.hidden_dim)

        loss_collapsed = self.objective.compute_cov_loss(z_collapsed)
        self.assertGreater(loss_collapsed.item(), 1.0)

        z_decorrelated = torch.randn(1000, self.hidden_dim)
        loss_decorrelated = self.objective.compute_cov_loss(z_decorrelated)
        self.assertLess(loss_decorrelated.item(), 0.2)
        self.assertLess(loss_decorrelated.item(), loss_collapsed.item())

    def test_combined_forward_and_telemetry(self):
        z_pred = torch.randn(2, 8, self.hidden_dim, requires_grad=True)
        z_target = torch.randn(2, 8, self.hidden_dim)

        loss, metrics = self.objective(z_pred, z_target, return_dict=True)
        self.assertIn("l_sim", metrics)
        self.assertIn("l_stp", metrics)
        self.assertIn("l_cov", metrics)
        self.assertIn("total_loss", metrics)

        self.assertIsNotNone(self.objective.last_sim_loss)
        self.assertIsNotNone(self.objective.last_stp_loss)
        self.assertIsNotNone(self.objective.last_cov_loss)
        self.assertIsNotNone(self.objective.last_total_loss)

        loss.backward()
        self.assertIsNotNone(z_pred.grad)
        self.assertTrue(torch.isfinite(z_pred.grad).all())

    def test_aliases(self):
        obj = JointEmbeddingObjective()
        self.assertIsInstance(obj, JEVObjective)


class TestJEVExitRouter(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 64
        self.predictor = JEVLatentPredictor(self.hidden_dim)
        self.router = JEVExitRouter(
            dim=self.hidden_dim,
            threshold=0.80,
            distance_metric="cosine",
            predictor=self.predictor,
        )

    def test_zero_vocabulary_overhead(self):
        x = torch.randn(2, 8, self.hidden_dim)
        total_router_params = sum(p.numel() for p in self.router.parameters())
        self.assertLess(total_router_params, 10000)

        res = self.router(x)
        self.assertIsInstance(res, JEVRoutingResult)
        self.assertEqual(res.should_exit.shape, (2, 8))
        self.assertEqual(res.confidence.shape, (2, 8))
        self.assertEqual(res.distance.shape, (2, 8))

    def test_distance_metrics(self):
        for metric in ("cosine", "euclidean", "geodesic", "stp_tube", "learned"):
            router = JEVExitRouter(dim=self.hidden_dim, threshold=0.75, distance_metric=metric)
            z = torch.randn(2, 4, self.hidden_dim)
            z_ref = torch.randn(2, 4, self.hidden_dim)

            dist = router.compute_latent_distance(z, z_ref)
            conf = router.compute_latent_confidence(z, z_ref)
            self.assertEqual(dist.shape, (2, 4))
            self.assertEqual(conf.shape, (2, 4))
            self.assertTrue(torch.isfinite(dist).all())
            self.assertTrue(torch.isfinite(conf).all())

    def test_evaluate_latent_exit(self):
        z = torch.randn(1, 1, self.hidden_dim)
        should_exit, conf = self.router.evaluate_latent_exit(z, threshold=0.5)
        self.assertIsInstance(should_exit, torch.Tensor)
        self.assertIsInstance(conf, torch.Tensor)
        self.assertIn(should_exit.item(), (True, False))

    def test_aliases(self):
        r = ContinuousLatentExitRouter(dim=self.hidden_dim)
        self.assertIsInstance(r, JEVExitRouter)


class TestJEVEndToEndIntegration(unittest.TestCase):
    def test_attach_and_adaptive_generation_with_jev(self):
        model = MockTransformerForJEV(vocab_size=100, hidden_dim=64, num_layers=6)
        retrofitted = attach_early_exits(model, reflex_layer=2, limbic_layer=4, rank=16, use_jev=True)

        self.assertTrue(hasattr(retrofitted, "reflex_jev_router"))
        self.assertTrue(hasattr(retrofitted, "limbic_jev_router"))
        self.assertTrue(retrofitted.use_jev)

        input_ids = torch.tensor([[5, 10, 15]])
        res = retrofitted.generate_adaptive(
            input_ids=input_ids,
            confidence_threshold=0.75,
            max_new_tokens=8,
            temperature=0.7,
            use_jev=True,
        )

        self.assertIn("output_ids", res)
        self.assertIn("exit_counts", res)
        self.assertIn("flops_saved_pct", res)
        self.assertIn("effective_speedup", res)
        self.assertEqual(res["output_ids"].size(1), 11)
        self.assertGreaterEqual(res["effective_speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
