"""Unit tests for Zero-Overhead CUDAGraphEngine."""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from triune.runtime.cuda_graph import CUDAGraphEngine
from triune.universal import UniversalDenseTransformer


class SimpleModel(nn.Module):
    def __init__(self, dim: int = 64, vocab: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc = nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        logits = self.fc(h)
        return logits, h.mean(dim=1)


class TestCUDAGraphEngine(unittest.TestCase):
    def setUp(self):
        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.is_cuda else "cpu")

    def test_cpu_safe_fallback(self):
        """Engine gracefully returns False on CPU without crashing."""
        engine = CUDAGraphEngine("cpu")
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(2, 10)
        y = torch.randn(2, 10)
        captured = engine.capture_training_step(model, opt, x, y)
        self.assertFalse(captured)
        self.assertFalse(engine.is_captured)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for CUDA graph testing")
    def test_async_replay_no_host_synchronization(self):
        """step(sync=False) returns GPU tensor without forcing CPU synchronization."""
        model = SimpleModel().to(device=self.device, dtype=torch.bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        engine = CUDAGraphEngine(self.device)

        x = torch.randint(0, 256, (2, 32), device=self.device)
        y = torch.randint(0, 256, (2, 32), device=self.device)

        captured = engine.capture_training_step(model, opt, x, y, warmup_steps=3)
        self.assertTrue(captured)
        self.assertTrue(engine.is_captured)

        # Async step returns GPU-resident scalar tensor
        loss_tensor = engine.step(x, y, sync=False)
        self.assertIsInstance(loss_tensor, torch.Tensor)
        self.assertEqual(loss_tensor.device.type, "cuda")

        # Sync step returns float
        loss_float = engine.step(x, y, sync=True)
        self.assertIsInstance(loss_float, float)
        self.assertAlmostEqual(loss_float, float(loss_tensor.item()), places=3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for CUDA graph testing")
    def test_batch_variation_updates_output(self):
        """Replaying with different input batches produces different loss values."""
        model = SimpleModel().to(device=self.device, dtype=torch.bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        engine = CUDAGraphEngine(self.device)

        x1 = torch.zeros((2, 32), dtype=torch.long, device=self.device)
        y1 = torch.zeros((2, 32), dtype=torch.long, device=self.device)
        captured = engine.capture_training_step(model, opt, x1, y1, warmup_steps=3)
        self.assertTrue(captured)

        loss1 = engine.step(x1, y1, sync=True)

        x2 = torch.full((2, 32), 127, dtype=torch.long, device=self.device)
        y2 = torch.full((2, 32), 200, dtype=torch.long, device=self.device)
        loss2 = engine.step(x2, y2, sync=True)

        self.assertNotEqual(loss1, loss2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for CUDA graph testing")
    def test_dynamic_shape_mismatch_fallback(self):
        """Mismatched tensor dimensions safely execute fallback function."""
        model = SimpleModel().to(device=self.device, dtype=torch.bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        engine = CUDAGraphEngine(self.device)

        x = torch.randint(0, 256, (2, 32), device=self.device)
        y = torch.randint(0, 256, (2, 32), device=self.device)
        engine.capture_training_step(model, opt, x, y, warmup_steps=2)

        x_diff_shape = torch.randint(0, 256, (4, 16), device=self.device)
        self.assertFalse(engine.can_replay(x_diff_shape, y))

        fallback_called = False

        def my_fallback(bx, by):
            nonlocal fallback_called
            fallback_called = True
            return 999.0

        res = engine.step(x_diff_shape, y, sync=True, fallback_fn=my_fallback)
        self.assertTrue(fallback_called)
        self.assertEqual(res, 999.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for CUDA graph testing")
    def test_inference_capture_structured_output(self):
        """Inference capture cleanly returns structured tuple outputs without memory allocation."""
        model = SimpleModel().to(device=self.device, dtype=torch.bfloat16)
        engine = CUDAGraphEngine(self.device)
        x = torch.randint(0, 256, (2, 32), device=self.device)

        captured = engine.capture_inference_step(model, x, warmup_steps=2)
        self.assertTrue(captured)

        out = engine.inference_step(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (2, 32, 256))
        self.assertEqual(out[1].shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
