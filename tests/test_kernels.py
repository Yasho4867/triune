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


if __name__ == "__main__":
    unittest.main()
