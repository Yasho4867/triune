import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig


class TestStreamingPrefetchStress(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_prefetch_multi_iteration_stress_and_memory_stability(self):
        """Stress test async prefetch over 15 training iterations with strict memory bounds."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for prefetch stress testing")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

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

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        engine.set_optimizer(optimizer)

        initial_vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)

        num_iterations = 15
        batch_size = 2
        seq_len = 16

        vram_readings = []

        for it in range(num_iterations):
            inputs = torch.randint(0, 256, (batch_size, seq_len), device=self.device)

            engine.zero_grad()
            logits, route_logits = model(inputs)
            loss = logits.pow(2).mean() + route_logits.pow(2).mean()
            loss.backward()

            engine.step_streaming_optimizer(optimizer=optimizer, grad_clip=1.0)

            # Record memory after each complete iteration
            current_vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            vram_readings.append(current_vram_mb)

            # Invariant: all layer parameters must point to CPU master storage
            for layer_idx, layer in enumerate(model.layers):
                self.assertFalse(getattr(layer, "_is_prefetched", False), f"Layer {layer_idx} left with _is_prefetched=True")
                for p in layer.parameters():
                    self.assertEqual(p.device.type, "cpu", f"Layer {layer_idx} parameter stranded on GPU")
                    self.assertEqual(p.data.data_ptr(), p._cpu_data.data_ptr(), f"Layer {layer_idx} parameter did not revert to CPU master")
                    self.assertIsNone(p.grad, f"Layer {layer_idx} param.grad not cleared")
                    self.assertIsNone(getattr(p, "_cpu_grad", None), f"Layer {layer_idx} _cpu_grad not cleared")

        # Invariant: VRAM after iteration 15 should not exceed initial VRAM by more than 50MB (no memory leak)
        vram_growth = vram_readings[-1] - vram_readings[0]
        self.assertLess(vram_growth, 50.0, f"Memory leak detected: VRAM grew by {vram_growth:.2f} MB over iterations")

        engine.detach()


if __name__ == "__main__":
    unittest.main()
