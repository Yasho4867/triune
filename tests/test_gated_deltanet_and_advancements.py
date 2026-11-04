"""Unit tests for Gated DeltaNet, DeepSeek-V3 router dynamic bias, and advanced optimizer features."""

import math
import unittest

import torch
import torch.nn as nn

from triune.kernels.fast_cross_entropy import chunked_cross_entropy_from_hidden
from triune.kernels.fast_gla import fast_chunk_gated_deltanet, fast_recurrent_gated_deltanet
from triune.kernels.fast_muon import compute_muon_update_scale
from triune.model.moe import MoE_FFN
from triune.model.router import GumbelSoftmaxRouter
from triune.optim.centroid import CentroidSteerOptimizer
from triune.optim.muon import Muon


class TestGatedDeltaNetAndAdvancements(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_gated_deltanet_shapes_and_finite_gradients(self):
        B, T, H, HD = 2, 16, 4, 32
        q = torch.randn(B, T, H, HD, requires_grad=True)
        k = torch.randn(B, T, H, HD, requires_grad=True)
        v = torch.randn(B, T, H, HD, requires_grad=True)
        g = (-torch.rand(B, T, H, HD)).detach().requires_grad_(True)

        out, state = fast_recurrent_gated_deltanet(q, k, v, g)
        self.assertEqual(out.shape, (B, T, H, HD))
        self.assertEqual(state.shape, (B, H, HD, HD))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        self.assertIsNotNone(g.grad)
        self.assertFalse(torch.isnan(q.grad).any())
        self.assertFalse(torch.isnan(k.grad).any())
        self.assertFalse(torch.isnan(v.grad).any())
        self.assertFalse(torch.isnan(g.grad).any())

    def test_fast_chunk_gated_deltanet_single_step(self):
        B, T, H, HD = 2, 1, 4, 32
        q = torch.randn(B, T, H, HD)
        k = torch.randn(B, T, H, HD)
        v = torch.randn(B, T, H, HD)
        g = -torch.rand(B, T, H, HD)

        out, state = fast_chunk_gated_deltanet(q, k, v, g)
        self.assertEqual(out.shape, (B, 1, H, HD))
        self.assertEqual(state.shape, (B, H, HD, HD))

    def test_compute_muon_update_scale(self):
        # Aspect ratio tests
        scale_square = compute_muon_update_scale(512, 512)
        self.assertAlmostEqual(scale_square, 1.0)

        scale_rect = compute_muon_update_scale(2048, 512)
        self.assertAlmostEqual(scale_rect, 2.0)  # sqrt(2048 / 512) = 2.0

        # RMS matching target
        scale_rms = compute_muon_update_scale(512, 512, rms_target=0.2)
        self.assertAlmostEqual(scale_rms, 0.2)

    def test_muon_with_rms_target(self):
        param = nn.Parameter(torch.randn(32, 64))
        opt = Muon([param], lr=0.01, rms_target=0.25)
        param.grad = torch.randn_like(param)
        opt.step()
        self.assertEqual(opt.param_groups[0]["rms_target"], 0.25)

    def test_deepseek_v3_moe_bias_steering(self):
        dim = 64
        num_experts = 4
        moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=2)

        x = torch.randn(8, 16, dim)
        _ = moe(x)

        # Pending bias update should be computed smoothly
        self.assertIsNotNone(moe._pending_bias_update)
        self.assertEqual(moe._pending_bias_update.shape, (num_experts,))

        # Apply bias update
        initial_bias = moe.expert_bias.clone()
        moe.step_bias()
        self.assertIsNone(moe._pending_bias_update)
        self.assertFalse(torch.equal(moe.expert_bias, initial_bias))

    def test_gumbel_softmax_router_decoupled_bias(self):
        router = GumbelSoftmaxRouter(hidden_dim=64)
        x = torch.randn(4, 16, 64)

        router.train()
        logits, y_route, balance_loss = router(x)
        self.assertEqual(logits.shape, (4, 3))
        self.assertEqual(y_route.shape, (4, 3))
        self.assertEqual(router.router_bias.shape, (3,))

    def test_chunked_cross_entropy_force_chunked(self):
        hidden = torch.randn(8, 64, requires_grad=True)
        head = nn.Linear(64, 128, bias=False)
        targets = torch.randint(0, 128, (8,))

        loss = chunked_cross_entropy_from_hidden(hidden, head, targets, chunk_size=4, force_chunked=True)
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertFalse(torch.isnan(hidden.grad).any())

    def test_centroid_steer_adaptive_energy_refresh(self):
        model = nn.Sequential(nn.Linear(64, 64))
        opt = CentroidSteerOptimizer(model.parameters(), lr=0.001, adaptive_subspace_refresh=True, energy_threshold=0.6)
        self.assertTrue(opt.adaptive_subspace_refresh)
        self.assertEqual(opt.energy_threshold, 0.6)

    def test_scale_residual_projections_initialization(self):
        from triune.model.transformer import TriuneTransformer

        model_standard = TriuneTransformer(
            vocab_size=256,
            hidden_dim=128,
            num_layers=18,
            num_heads=4,
            head_dim=32,
            num_experts=4,
            router_prefix_layers=2,
            reflex_exit_layer=4,
            limbic_exit_layer=10,
            scale_residual_projections=False,
        )
        model_scaled = TriuneTransformer(
            vocab_size=256,
            hidden_dim=128,
            num_layers=18,
            num_heads=4,
            head_dim=32,
            num_experts=4,
            router_prefix_layers=2,
            reflex_exit_layer=4,
            limbic_exit_layer=10,
            scale_residual_projections=True,
        )

        std_standard = model_standard.layers[0].attn.gla.out_proj.weight.std().item()
        std_scaled = model_scaled.layers[0].attn.gla.out_proj.weight.std().item()

        expected_scaled_std = 0.02 / math.sqrt(2 * 18)
        self.assertAlmostEqual(std_scaled, expected_scaled_std, delta=0.005)
        self.assertLess(std_scaled, std_standard)


if __name__ == "__main__":
    unittest.main()
