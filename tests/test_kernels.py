"""Unit tests for Triune High-Performance Kernel Acceleration Suite."""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.kernels import (
    is_triton_available,
    get_kernel_backend,
    fast_rmsnorm,
    FastRMSNorm,
    fast_rope,
    fast_cross_entropy,
    chunked_cross_entropy_from_hidden,
)


class TestKernelSuite(unittest.TestCase):
    def test_dispatcher_availability(self):
        avail = is_triton_available()
        self.assertIsInstance(avail, bool)
        backend = get_kernel_backend()
        self.assertIn(backend, ("triton", "pytorch"))

    def test_fast_rmsnorm_forward_and_backward(self):
        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D, requires_grad=True)
        weight = torch.ones(D, requires_grad=True)

        y = fast_rmsnorm(x, weight, eps=1e-6)
        self.assertEqual(y.shape, (B, T, D))

        # Check normalization scale: mean square should be approx 1.0
        ms = (y * y).mean(dim=-1)
        self.assertTrue(torch.allclose(ms, torch.ones_like(ms), atol=1e-3))

        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_fast_rmsnorm_module(self):
        norm = FastRMSNorm(128)
        x = torch.randn(4, 16, 128)
        out = norm(x)
        self.assertEqual(out.shape, (4, 16, 128))

    def test_fast_rope(self):
        B, H, T, D = 2, 4, 16, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        cos = torch.ones(T, D)
        sin = torch.zeros(T, D)

        q_rot, k_rot = fast_rope(q, k, cos, sin)
        # With cos=1 and sin=0, rotated tensors should equal original tensors
        self.assertTrue(torch.allclose(q_rot, q, atol=1e-5))
        self.assertTrue(torch.allclose(k_rot, k, atol=1e-5))

    def test_fast_cross_entropy(self):
        B, T, V = 2, 16, 500
        logits = torch.randn(B * T, V)
        targets = torch.randint(0, V, (B * T,))

        standard_loss = F.cross_entropy(logits, targets)
        fast_loss = fast_cross_entropy(logits, targets, chunk_size=8)
        self.assertTrue(torch.allclose(standard_loss, fast_loss, atol=1e-4))

    def test_chunked_cross_entropy_from_hidden(self):
        total_tokens = 32
        hidden_dim = 64
        vocab_size = 200
        hidden = torch.randn(total_tokens, hidden_dim)
        targets = torch.randint(0, vocab_size, (total_tokens,))
        lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Standard loss
        full_logits = lm_head(hidden)
        expected_loss = F.cross_entropy(full_logits, targets)

        # Chunked loss
        chunked_loss = chunked_cross_entropy_from_hidden(hidden, lm_head, targets, chunk_size=8)
        self.assertTrue(torch.allclose(expected_loss, chunked_loss, atol=1e-4))

    def test_fast_cross_entropy_grad_parity(self):
        """H-20: chunked CE gradient must match F.cross_entropy, including ignore_index rows."""
        torch.manual_seed(0)
        logits = torch.randn(64, 300, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(0, 300, (64,))
        targets[::5] = -100
        ref = F.cross_entropy(logits, targets, ignore_index=-100)
        (g_ref,) = torch.autograd.grad(ref * 2.5, logits)

        logits2 = logits.detach().clone().requires_grad_(True)
        out = fast_cross_entropy(logits2, targets, chunk_size=7)
        (out * 2.5).backward()
        self.assertTrue(torch.allclose(ref, out.to(ref.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(g_ref, logits2.grad, atol=1e-6))

    def test_chunked_from_hidden_grad_parity(self):
        """H-20: fused linear+CE must produce identical grads for hidden, weight and bias."""
        torch.manual_seed(1)
        hidden = torch.randn(2, 20, 48, dtype=torch.float64, requires_grad=True)
        head = nn.Linear(48, 150, bias=True).double()
        targets = torch.randint(0, 150, (2, 20))
        targets[0, :3] = -100

        ref = F.cross_entropy(head(hidden).reshape(-1, 150), targets.reshape(-1), ignore_index=-100)
        g_h, g_w, g_b = torch.autograd.grad(ref, (hidden, head.weight, head.bias))

        hidden2 = hidden.detach().clone().requires_grad_(True)
        head.zero_grad()
        out = chunked_cross_entropy_from_hidden(hidden2, head, targets, chunk_size=6)
        out.backward()
        self.assertTrue(torch.allclose(ref, out.to(ref.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(g_h, hidden2.grad, atol=1e-6))
        self.assertTrue(torch.allclose(g_w, head.weight.grad, atol=1e-6))
        self.assertTrue(torch.allclose(g_b, head.bias.grad, atol=1e-6))

    def test_chunked_ce_all_ignored(self):
        logits = torch.randn(4000, 50, requires_grad=True)
        targets = torch.full((4000,), -100)
        out = fast_cross_entropy(logits, targets, chunk_size=512)
        out.backward()
        self.assertEqual(out.item(), 0.0)
        self.assertTrue(torch.all(logits.grad == 0))

        hidden = torch.randn(10, 16, requires_grad=True)
        head = nn.Linear(16, 30, bias=False)
        out2 = chunked_cross_entropy_from_hidden(hidden, head, torch.full((10,), -100), chunk_size=4)
        out2.backward()
        self.assertEqual(out2.item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for memory measurement")
    def test_chunked_from_hidden_saves_memory(self):
        """H-20: peak memory must stay far below materializing the full [N, V] logits."""
        dev = torch.device("cuda")
        N, D, V, chunk = 8192, 256, 32000, 512
        hidden = torch.randn(N, D, device=dev, requires_grad=True)
        head = nn.Linear(D, V, bias=False).to(dev)
        targets = torch.randint(0, V, (N,), device=dev)
        full_logits_bytes = N * V * 4  # fp32 logits alone ~1 GB

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss = chunked_cross_entropy_from_hidden(hidden, head, targets, chunk_size=chunk)
        loss.backward()
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - base
        self.assertLess(peak_extra, full_logits_bytes * 0.25,
                        f"peak extra {peak_extra/1e6:.1f} MB vs full logits {full_logits_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    unittest.main()
