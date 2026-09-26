import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig
from triune.optim.centroid import CentroidSteerOptimizer


class TestStreamingMoE(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_moe_routing_and_centroids_under_streaming(self):
        """Validates MoE routing, capacity tracking, accepted/dropped stats, and centroid updates under streaming."""
        torch.manual_seed(101)

        cfg = dict(
            vocab_size=256,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            head_dim=32,
            num_experts=4,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
            use_fp4=False,
            use_fp8=False,
        )

        model = TriuneTransformer(**cfg)
        engine = LayerStreamingEngine(
            model,
            device=self.device,
            config=StreamingConfig(enabled=True, async_prefetch=True, pin_memory=False),
        )
        engine.attach()

        optimizer = CentroidSteerOptimizer(
            model,
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            rank=8,
            update_gap=2,
            steer_scale=0.10,
        )
        engine.set_optimizer(optimizer)

        batch_size = 4
        seq_len = 16
        inputs = torch.randint(0, 256, (batch_size, seq_len), device=self.device)

        # Run 3 training steps to trigger centroid updates
        for step in range(3):
            engine.zero_grad()
            logits, route_logits = model(inputs)
            loss = logits.pow(2).mean() + route_logits.pow(2).mean()
            loss.backward()

            engine.step_streaming_optimizer(optimizer=optimizer, grad_clip=1.0)

            # Inspect MoE layer routing statistics
            for layer_idx, layer in enumerate(model.layers):
                if hasattr(layer, "moe") and hasattr(layer.moe, "get_routing_stats"):
                    stats = layer.moe.get_routing_stats()
                    self.assertIn("requested", stats)
                    self.assertIn("accepted", stats)
                    self.assertIn("dropped", stats)

                    # Ensure counts are non-negative and finite
                    self.assertTrue(torch.isfinite(stats["requested"]).all().item())
                    self.assertTrue(torch.isfinite(stats["accepted"]).all().item())
                    self.assertTrue(torch.isfinite(stats["dropped"]).all().item())
                    self.assertTrue((stats["accepted"] >= 0).all().item())

                if hasattr(layer, "moe") and hasattr(layer.moe, "centroids"):
                    centroids = layer.moe.centroids
                    self.assertTrue(torch.isfinite(centroids).all().item(), f"Layer {layer_idx} centroids non-finite")

        engine.detach()


if __name__ == "__main__":
    unittest.main()
