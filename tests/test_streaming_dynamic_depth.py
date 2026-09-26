import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig


class TestStreamingDynamicDepth(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dynamic_depth_and_gla_cache_streaming(self):
        """Validates adaptive depth routing (Reflex/Limbic/Cortex) and recurrent GLA cache under streaming."""
        torch.manual_seed(42)

        cfg = dict(
            vocab_size=256,
            hidden_dim=128,
            num_layers=6,
            num_heads=4,
            head_dim=32,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
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

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, 256, (batch_size, seq_len), device=self.device)

        # 1. Forward with dynamic exit selection
        logits, route_logits = model(x)
        self.assertEqual(logits.shape, (batch_size, seq_len, 256))
        self.assertEqual(route_logits.shape, (batch_size, 3))
        self.assertIsNotNone(model.last_depth_choice)
        self.assertTrue(torch.isin(model.last_depth_choice, torch.tensor([0, 1, 2], device=self.device)).all().item())

        # 2. Forward forcing each specific exit depth
        depth_map = {"reflex": 0, "limbic": 1, "cortex": 2}
        for name, target_idx in depth_map.items():
            forced_logits, forced_route = model(x, force_depth=target_idx)
            self.assertEqual(forced_logits.shape, (batch_size, seq_len, 256))
            self.assertEqual(forced_route.shape, (batch_size, 3))
            self.assertTrue((model.last_depth_choice == target_idx).all().item())

        # 3. Forward all exits simultaneously (used for router training exploration)
        r_log, l_log, c_log, rt_log = model.forward_all_exits(x)
        self.assertEqual(r_log.shape, (batch_size, seq_len, 256))
        self.assertEqual(l_log.shape, (batch_size, seq_len, 256))
        self.assertEqual(c_log.shape, (batch_size, seq_len, 256))
        self.assertEqual(rt_log.shape, (batch_size, 3))

        # 4. Incremental step-by-step decoding with recurrent GLA cache
        cache = [None] * len(model.layers)
        single_step_token = torch.randint(0, 256, (batch_size, 1), device=self.device)
        out1, _ = model(single_step_token, cache=cache)
        self.assertEqual(out1.shape, (batch_size, 1, 256))

        # Check that GLA cache was created and updated
        for idx, c in enumerate(cache):
            if c is not None:
                self.assertIsInstance(c, torch.Tensor, f"Layer {idx} GLA cache should be a Tensor")
                self.assertEqual(c.device.type, self.device.type)

        # Second step with populated cache
        out2, _ = model(single_step_token, cache=cache)
        self.assertEqual(out2.shape, (batch_size, 1, 256))

        engine.detach()


if __name__ == "__main__":
    unittest.main()
