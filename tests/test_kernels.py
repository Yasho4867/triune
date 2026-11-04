"""Unit tests for Triune High-Performance Kernel Acceleration Suite."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.kernels import (
    FastRMSNorm,
    chunked_cross_entropy_from_hidden,
    fast_cross_entropy,
    fast_rmsnorm,
    fast_rope,
    get_kernel_backend,
    is_triton_available,
)


class TestKernelSuite(unittest.TestCase):
    def test_dispatcher_availability(self):
        avail = is_triton_available()
        self.assertIsInstance(avail, bool)
        backend = get_kernel_backend()
        self.assertIn(
            backend, ("triton", "pytorch", "native-cuda-analytical", "triton-hardware-fused", "vectorized-cpu")
        )

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

    def test_fast_swiglu_and_gelu(self):
        from triune.kernels import FastGELU, fast_swiglu

        gate = torch.randn(4, 16, 64, requires_grad=True)
        up = torch.randn(4, 16, 64, requires_grad=True)

        # Fused SwiGLU forward & backward
        out = fast_swiglu(gate, up)
        self.assertEqual(out.shape, (4, 16, 64))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(gate.grad)
        self.assertIsNotNone(up.grad)

        # Compare with naive SwiGLU forward
        naive_out = F.silu(gate.detach()) * up.detach()
        self.assertTrue(torch.allclose(out.detach(), naive_out, atol=1e-5))

        # Fast GELU module
        gelu_mod = FastGELU()
        x = torch.randn(2, 8, 32, requires_grad=True)
        gx = gelu_mod(x)
        self.assertEqual(gx.shape, (2, 8, 32))
        gx.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_fast_muon_newton_schulz(self):
        from triune.kernels import fast_zeropower_via_newtonschulz5

        # 64x128 matrix
        G = torch.randn(64, 128)
        X = fast_zeropower_via_newtonschulz5(G, steps=5)
        self.assertEqual(X.shape, (64, 128))

        # Verify semi-orthogonality: X @ X.T should be close to identity
        gram = X @ X.t()
        eye = torch.eye(64, device=X.device, dtype=X.dtype)
        diff = (gram - eye).abs().mean().item()
        self.assertLess(diff, 0.35)

    def test_fast_lora_forward(self):
        from triune.kernels import fast_lora_forward

        B, T, D_in, D_out, rank = 2, 8, 32, 48, 8
        x = torch.randn(B, T, D_in, requires_grad=True)
        lora_A = torch.randn(rank, D_in, requires_grad=True)
        lora_B = torch.randn(D_out, rank, requires_grad=True)
        scaling = 2.0

        out = fast_lora_forward(x, lora_A, lora_B, scaling)
        self.assertEqual(out.shape, (B, T, D_out))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(lora_A.grad)
        self.assertIsNotNone(lora_B.grad)

    def test_fast_chunk_gla(self):
        from triune.kernels import fast_chunk_gla

        B, T, H, HD = 2, 64, 4, 32
        q = torch.randn(B, T, H, HD)
        k = torch.randn(B, T, H, HD)
        v = torch.randn(B, T, H, HD)
        g = -torch.rand(B, T, H, HD)  # Negative log decays

        out, state = fast_chunk_gla(q, k, v, g, chunk_size=32)
        self.assertEqual(out.shape, (B, T, H, HD))
        self.assertEqual(state.shape, (B, H, HD, HD))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton GLA scan parity")
    def test_fused_recurrent_scan_mathematical_parity(self):
        """Verify Triton FusedRecurrentScan against reference sequential recurrence in forward & backward."""
        from triune.kernels.fast_gla import FusedRecurrentScan

        if FusedRecurrentScan is None:
            self.skipTest("Triton not available")

        device = torch.device("cuda")
        torch.manual_seed(42)

        for N in (1, 4, 16, 32):  # num_chunks
            for with_init in (False, True):
                with self.subTest(num_chunks=N, with_init=with_init):
                    B, H, Dk, Dv = 2, 4, 32, 32
                    kv_ref = torch.randn(B, H, N, Dk, Dv, device=device, dtype=torch.float32, requires_grad=True)
                    decay_ref = torch.rand(B, H, N, Dk, 1, device=device, dtype=torch.float32, requires_grad=True)
                    init_ref = (
                        torch.randn(B, H, Dk, Dv, device=device, dtype=torch.float32, requires_grad=True)
                        if with_init
                        else torch.zeros(B, H, Dk, Dv, device=device, dtype=torch.float32, requires_grad=True)
                    )

                    kv_tri = kv_ref.detach().clone().requires_grad_(True)
                    decay_tri = decay_ref.detach().clone().requires_grad_(True)
                    init_tri = init_ref.detach().clone().requires_grad_(True)

                    # Reference PyTorch sequential loop
                    cur = init_ref
                    states_py = []
                    for i in range(N):
                        states_py.append(cur)
                        cur = torch.addcmul(kv_ref[:, :, i], cur, decay_ref[:, :, i])
                    s_py = torch.stack(states_py, dim=2)
                    fin_py = cur

                    # Triton FusedRecurrentScan
                    s_tri, fin_tri = FusedRecurrentScan.apply(kv_tri, decay_tri, init_tri)

                    # Forward parity
                    torch.testing.assert_close(s_tri, s_py, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(fin_tri, fin_py, atol=1e-5, rtol=1e-5)

                    # Backward gradient parity
                    grad_s = torch.randn_like(s_tri)
                    grad_fin = torch.randn_like(fin_tri)

                    loss_py = (s_py * grad_s).sum() + (fin_py * grad_fin).sum()
                    loss_py.backward()

                    loss_tri = (s_tri * grad_s).sum() + (fin_tri * grad_fin).sum()
                    loss_tri.backward()

                    torch.testing.assert_close(kv_tri.grad, kv_ref.grad, atol=5e-5, rtol=1e-4)
                    torch.testing.assert_close(decay_tri.grad, decay_ref.grad, atol=1e-4, rtol=1e-3)
                    torch.testing.assert_close(init_tri.grad, init_ref.grad, atol=5e-5, rtol=1e-4)

    def test_fast_chunk_gla_gradient_parity_and_lengths(self):
        """Assert fast_chunk_gla matches token-by-token recurrence across divisible and non-divisible lengths on CPU and CUDA."""
        from triune.kernels.fast_gla import fast_chunk_gla

        def _ref_gla(q, k, v, g, init_s=None):
            B, T, H, HD = q.shape
            cur = init_s.clone() if init_s is not None else torch.zeros(B, H, HD, HD, device=q.device, dtype=q.dtype)
            out = []
            for t in range(T):
                cur = cur * torch.exp(g[:, t]).unsqueeze(-1) + k[:, t].unsqueeze(-1) @ v[:, t].unsqueeze(-2)
                out.append((q[:, t].unsqueeze(-2) @ cur).squeeze(-2))
            return torch.stack(out, dim=1), cur

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        # 1. Exact mathematical parity in FP32 across CPU and CUDA (including irregular non-divisible sequence lengths)
        for dev in devices:
            device = torch.device(dev)
            for T in (1, 17, 32, 70, 128):
                for with_init in (False, True):
                    with self.subTest(device=dev, T=T, with_init=with_init, dtype="float32"):
                        torch.manual_seed(42 + T)
                        B, H, HD = 2, 4, 16
                        q1 = torch.randn(B, T, H, HD, device=device, dtype=torch.float32, requires_grad=True)
                        k1 = torch.randn_like(q1, requires_grad=True)
                        v1 = torch.randn_like(q1, requires_grad=True)
                        g1 = (-torch.rand_like(q1) * 2.0 - 0.05).requires_grad_(True)
                        s1 = (
                            torch.randn(B, H, HD, HD, device=device, dtype=torch.float32, requires_grad=True)
                            if with_init
                            else None
                        )

                        q2 = q1.detach().clone().requires_grad_(True)
                        k2 = k1.detach().clone().requires_grad_(True)
                        v2 = v1.detach().clone().requires_grad_(True)
                        g2 = g1.detach().clone().requires_grad_(True)
                        s2 = s1.detach().clone().requires_grad_(True) if s1 is not None else None

                        out_tri, fin_tri = fast_chunk_gla(q1, k1, v1, g1, initial_state=s1, chunk_size=32)
                        out_ref, fin_ref = _ref_gla(q2, k2, v2, g2, init_s=s2)

                        # Forward parity
                        torch.testing.assert_close(out_tri, out_ref, atol=2e-4, rtol=1e-3)
                        torch.testing.assert_close(fin_tri, fin_ref, atol=2e-4, rtol=1e-3)

                        # Backward gradient parity across all 5 inputs
                        grad_out = torch.randn_like(out_tri)
                        grad_fin = torch.randn_like(fin_tri)
                        loss_tri = (out_tri * grad_out).sum() + (fin_tri * grad_fin).sum()
                        loss_ref = (out_ref * grad_out).sum() + (fin_ref * grad_fin).sum()
                        loss_tri.backward()
                        loss_ref.backward()

                        torch.testing.assert_close(q1.grad, q2.grad, atol=3e-4, rtol=1e-3)
                        torch.testing.assert_close(k1.grad, k2.grad, atol=3e-4, rtol=1e-3)
                        torch.testing.assert_close(v1.grad, v2.grad, atol=3e-4, rtol=1e-3)
                        torch.testing.assert_close(g1.grad, g2.grad, atol=3e-4, rtol=1e-3)
                        if with_init:
                            torch.testing.assert_close(s1.grad, s2.grad, atol=3e-4, rtol=1e-3)

        # 2. CUDA BF16 numerical execution, reference parity, and gradient flow across divisible and irregular lengths
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            cuda_dev = torch.device("cuda")
            for T in (1, 8, 17, 32, 70):
                for with_init in (False, True):
                    with self.subTest(T=T, with_init=with_init, dtype="bfloat16"):
                        torch.manual_seed(100 + T)
                        B, H, HD = 2, 4, 16
                        q1 = torch.randn(B, T, H, HD, device=cuda_dev, dtype=torch.bfloat16, requires_grad=True)
                        k1 = torch.randn_like(q1, requires_grad=True)
                        v1 = torch.randn_like(q1, requires_grad=True)
                        g1 = (-torch.rand_like(q1) * 2.0 - 0.05).requires_grad_(True)
                        s1 = (
                            torch.randn(B, H, HD, HD, device=cuda_dev, dtype=torch.bfloat16, requires_grad=True)
                            if with_init
                            else None
                        )

                        out_tri, fin_tri = fast_chunk_gla(q1, k1, v1, g1, initial_state=s1, chunk_size=32)
                        self.assertEqual(out_tri.shape, (B, T, H, HD))
                        self.assertEqual(fin_tri.shape, (B, H, HD, HD))
                        self.assertFalse(torch.isnan(out_tri).any())
                        self.assertFalse(torch.isnan(fin_tri).any())

                        grad_out = torch.randn_like(out_tri)
                        grad_fin = torch.randn_like(fin_tri)
                        loss_tri = (out_tri * grad_out).sum() + (fin_tri * grad_fin).sum()
                        loss_tri.backward()

                        self.assertIsNotNone(q1.grad)
                        self.assertIsNotNone(k1.grad)
                        self.assertIsNotNone(v1.grad)
                        self.assertIsNotNone(g1.grad)
                        self.assertFalse(torch.isnan(q1.grad).any())
                        self.assertFalse(torch.isnan(k1.grad).any())
                        self.assertFalse(torch.isnan(v1.grad).any())
                        self.assertFalse(torch.isnan(g1.grad).any())
                        if with_init:
                            self.assertIsNotNone(s1.grad)
                            self.assertFalse(torch.isnan(s1.grad).any())

                        # Numerical parity verification against reference recurrence for T <= 8
                        if T <= 8:
                            q2 = q1.detach().clone().requires_grad_(True)
                            k2 = k1.detach().clone().requires_grad_(True)
                            v2 = v1.detach().clone().requires_grad_(True)
                            g2 = g1.detach().clone().requires_grad_(True)
                            s2 = s1.detach().clone().requires_grad_(True) if s1 is not None else None

                            out_ref, fin_ref = _ref_gla(q2, k2, v2, g2, init_s=s2)
                            torch.testing.assert_close(out_tri, out_ref, atol=0.25, rtol=0.25)
                            torch.testing.assert_close(fin_tri, fin_ref, atol=0.25, rtol=0.25)

                            loss_ref = (out_ref * grad_out).sum() + (fin_ref * grad_fin).sum()
                            loss_ref.backward()
                            torch.testing.assert_close(q1.grad, q2.grad, atol=0.35, rtol=0.35)
                            torch.testing.assert_close(k1.grad, k2.grad, atol=0.35, rtol=0.35)
                            torch.testing.assert_close(v1.grad, v2.grad, atol=0.35, rtol=0.35)

    def test_kernel_telemetry_and_turbo_mode(self):
        from triune.kernels import get_kernel_telemetry, is_turbo_mode_enabled, set_turbo_mode

        self.assertTrue(is_turbo_mode_enabled())
        telem = get_kernel_telemetry()
        self.assertIn("kernels", telem)
        self.assertIn("fast_swiglu", telem["kernels"])
        self.assertIn("fast_muon", telem["kernels"])
        self.assertIn("fast_gla", telem["kernels"])
        self.assertIn("fast_lora", telem["kernels"])
        self.assertIn("benchmarks_vs_baselines", telem)

        # Toggle test
        set_turbo_mode(False)
        self.assertFalse(is_turbo_mode_enabled())
        set_turbo_mode(True)
        self.assertTrue(is_turbo_mode_enabled())

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
        self.assertLess(
            peak_extra,
            full_logits_bytes * 0.25,
            f"peak extra {peak_extra / 1e6:.1f} MB vs full logits {full_logits_bytes / 1e6:.1f} MB",
        )

    def test_fast_fused_add_rmsnorm(self):
        from triune.kernels import fast_fused_add_rmsnorm

        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D, requires_grad=True)
        residual = torch.randn(B, T, D, requires_grad=True)
        weight = torch.ones(D, requires_grad=True)

        normed, res_out = fast_fused_add_rmsnorm(x, residual, weight, eps=1e-6)
        self.assertEqual(normed.shape, (B, T, D))
        self.assertEqual(res_out.shape, (B, T, D))

        loss = normed.sum() + res_out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(residual.grad)
        self.assertIsNotNone(weight.grad)

    def test_fast_fp8_linear(self):
        from triune.kernels import FastFP8Linear

        mod = FastFP8Linear(64, 32)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = mod(x)
        self.assertEqual(out.shape, (2, 8, 32))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(mod.weight.grad)

    def test_nvfp4_roundtrip(self):
        from triune.kernels import NVFP4Linear, nvfp4_dequantize, nvfp4_quantize

        rows, cols = 32, 64
        w = torch.randn(rows, cols, dtype=torch.bfloat16)
        packed, scale = nvfp4_quantize(w)
        self.assertEqual(packed.shape, (rows, cols // 2))
        self.assertEqual(scale.shape, (rows, cols // 16))

        recon = nvfp4_dequantize(packed, scale, dtype=torch.bfloat16)
        self.assertEqual(recon.shape, (rows, cols))
        cos_sim = torch.nn.functional.cosine_similarity(w.float().view(-1), recon.float().view(-1), dim=0).item()
        self.assertGreater(cos_sim, 0.95)

        # Test NVFP4Linear
        linear = nn.Linear(64, 32, bias=True, dtype=torch.bfloat16)
        nv_linear = NVFP4Linear.from_linear(linear)
        x = torch.randn(2, 4, 64, dtype=torch.bfloat16)
        out = nv_linear(x)
        self.assertEqual(out.shape, (2, 4, 32))


if __name__ == "__main__":
    unittest.main()
