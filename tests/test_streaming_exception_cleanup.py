import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig


class TestStreamingExceptionCleanup(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_exception_cleanup(self):
        """Validates that exceptions during forward execution restore CPU master pointers and free GPU buffers."""
        cfg = dict(
            vocab_size=128,
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            head_dim=32,
            num_experts=2,
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

        # Inject an intentional exception into layer 2 forward
        def faulty_forward(*args, **kwargs):
            raise RuntimeError("Intentional forward injection test failure")

        orig_forward = model.layers[2].forward
        model.layers[2].forward = faulty_forward

        inputs = torch.randint(0, 128, (2, 8), device=self.device)

        with self.assertRaises(RuntimeError):
            try:
                model(inputs)
            except Exception:
                engine.cleanup_resident_layers()
                raise

        # Verify all layer parameters reverted to CPU storage
        for layer_idx, layer in enumerate(model.layers):
            self.assertFalse(getattr(layer, "_is_prefetched", False), f"Layer {layer_idx} left prefetched")
            for p in layer.parameters():
                self.assertEqual(p.device.type, "cpu", f"Layer {layer_idx} parameter stranded on GPU")
                self.assertEqual(p.data.data_ptr(), p._cpu_data.data_ptr(), f"Layer {layer_idx} parameter not restored to CPU master")
                self.assertIsNone(getattr(p, "_fp8_tensor", None))
                self.assertIsNone(getattr(p, "_fp8_scale", None))

        model.layers[2].forward = orig_forward
        engine.detach()

    def test_optimizer_exception_cleanup(self):
        """Validates that exceptions during streaming optimizer step call cleanup_resident_layers."""
        cfg = dict(
            vocab_size=128,
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            head_dim=32,
            num_experts=2,
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

        class FaultyOptimizer(torch.optim.Optimizer):
            def __init__(self, params):
                super().__init__(params, defaults={})

            def step_parameters(self, params, is_global_step_end=False):
                raise RuntimeError("Intentional optimizer step failure")

        faulty_opt = FaultyOptimizer(model.parameters())
        engine.set_optimizer(faulty_opt)

        inputs = torch.randint(0, 128, (2, 8), device=self.device)
        logits, route_logits = model(inputs)
        loss = logits.pow(2).mean()
        loss.backward()

        with self.assertRaises(RuntimeError):
            engine.step_streaming_optimizer(optimizer=faulty_opt)

        # Confirm cleanup_resident_layers ran during step_streaming_optimizer exception handler
        for layer_idx, layer in enumerate(model.layers):
            for p in layer.parameters():
                self.assertEqual(p.device.type, "cpu")
                self.assertEqual(p.data.data_ptr(), p._cpu_data.data_ptr())
                self.assertIsNone(getattr(p, "_fp8_tensor", None))

        engine.detach()


if __name__ == "__main__":
    unittest.main()
