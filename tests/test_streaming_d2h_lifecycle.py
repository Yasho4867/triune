import unittest
import copy
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig


class TestStreamingD2HLifecycle(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_d2h_async_accumulation_lifecycle(self):
        """Test 4 microbatches of async D2H accumulation without per-step sync."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for D2H stream tests")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

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

        model_ref = TriuneTransformer(**cfg).to(self.device).eval()
        model_str = TriuneTransformer(**cfg).eval()
        model_str.load_state_dict(copy.deepcopy(model_ref.state_dict()))

        engine = LayerStreamingEngine(
            model_str,
            device=self.device,
            config=StreamingConfig(enabled=True, async_prefetch=True, pin_memory=False),
        )
        engine.attach()

        opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=1e-3)
        opt_str = torch.optim.AdamW(model_str.parameters(), lr=1e-3)
        engine.set_optimizer(opt_str)

        num_accum_steps = 4
        batch_size = 2
        seq_len = 8

        # Create identical microbatches
        batches = [torch.randint(0, 128, (batch_size, seq_len), device=self.device) for _ in range(num_accum_steps)]

        # 1. Reference accumulation
        opt_ref.zero_grad()
        for i, batch in enumerate(batches):
            logits, _ = model_ref(batch)
            loss = logits.pow(2).mean() / num_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model_ref.parameters(), 1.0)
        opt_ref.step()

        # 2. Streaming accumulation
        engine.zero_grad()
        for i, batch in enumerate(batches):
            logits, _ = model_str(batch)
            loss = logits.pow(2).mean() / num_accum_steps
            loss.backward()

            # Verify that layer params offloaded grads to CPU and recorded d2h events
            has_layer_cpu_grad = False
            for layer in model_str.layers:
                for p in layer.parameters():
                    self.assertIsNone(p.grad, "GPU grad should be None after post_accum hook")
                    if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                        has_layer_cpu_grad = True
                        self.assertEqual(p._cpu_grad.device.type, "cpu")
            self.assertTrue(has_layer_cpu_grad, "Active layer parameters must offload CPU gradients")

        # Step streaming optimizer (should synchronize d2h events and apply updates)
        engine.step_streaming_optimizer(optimizer=opt_str, grad_clip=1.0)

        # Verify d2h events and cpu grads are retired
        for layer in model_str.layers:
            for p in layer.parameters():
                self.assertIsNone(getattr(p, "_d2h_event", None), "d2h event must be None after step")
                self.assertIsNone(getattr(p, "_cpu_grad", None), "cpu grad must be None after step")

        # Verify weight parity between ref and streaming models
        for (name, p_ref), (_, p_str) in zip(model_ref.named_parameters(), model_str.named_parameters()):
            ref_val = p_ref.data.cpu()
            str_val = getattr(p_str, "_cpu_data", p_str.data).cpu()
            torch.testing.assert_close(
                str_val,
                ref_val,
                rtol=1e-4,
                atol=1e-5,
                msg=f"D2H accumulated weight mismatch for {name}",
            )

        engine.detach()


if __name__ == "__main__":
    unittest.main()
