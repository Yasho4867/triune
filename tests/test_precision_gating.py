"""Tests for Milestone 4 (Gate 4): Precision Capability Gating & Real FP8 Staging.

Verifies:
1. Dynamic capability probing on live hardware (no hardcoded GPU models).
2. Graceful resolution and strict rejection (fallback=False) across all precision tiers.
3. FP8StagingBuffer memory reduction and absolute CPU master weight immutability.
4. ParameterStager FP8 integration.
5. LayerStreamingEngine with fp8_weights=True forward/backward and detach lifecycle.
6. FP8Linear fallback and execution stability.
"""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from triune.runtime.capabilities import PrecisionCapabilities
from triune.runtime.staging import FP8StagingBuffer, ParameterStager, StagingBuffer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig
from triune.model.fp8 import FP8Linear
from triune.model.transformer import TriuneTransformer


class TestPrecisionCapabilityGating(unittest.TestCase):
    """Test dynamic precision capability probing and resolution."""

    def test_live_hardware_detection_dynamic(self):
        """Probes live hardware dynamically and asserts detected capabilities match environment."""
        caps = PrecisionCapabilities.detect()
        self.assertIsInstance(caps, PrecisionCapabilities)

        if torch.cuda.is_available():
            self.assertEqual(caps.device.type, "cuda")
            major, minor = torch.cuda.get_device_capability(caps.device)
            self.assertEqual(caps.compute_capability, (major, minor))
            self.assertEqual(caps.bf16_supported, major >= 8 and torch.cuda.is_bf16_supported())
            # Assert native scaled_mm is a boolean without throwing an unhandled exception
            self.assertIsInstance(caps.native_scaled_mm, bool)
            self.assertIsInstance(caps.fp8_gemm, bool)
            self.assertIsInstance(caps.fp4_gemm, bool)
            self.assertIsInstance(caps.transformer_engine, bool)
        else:
            self.assertEqual(caps.device.type, "cpu")
            self.assertEqual(caps.compute_capability, (0, 0))
            self.assertFalse(caps.fp8_gemm)
            self.assertFalse(caps.fp4_gemm)
            self.assertFalse(caps.native_scaled_mm)

    def test_simulated_capability_resolution(self):
        """Verifies resolve_precision across simulated hardware profiles."""
        # Profile 1: Modern Hopper SM90 with full FP8 support
        hopper_caps = PrecisionCapabilities(
            device=torch.device("cuda:0"),
            compute_capability=(9, 0),
            fp8_gemm=True,
            fp4_gemm=False,
            transformer_engine=True,
            native_scaled_mm=True,
            bf16_supported=True,
        )
        self.assertEqual(hopper_caps.resolve_precision("fp8"), "fp8")
        self.assertEqual(hopper_caps.resolve_precision("bf16"), "bf16")
        self.assertEqual(hopper_caps.resolve_precision("fp32"), "fp32")
        # FP4 on Hopper falls back to BF16
        self.assertEqual(hopper_caps.resolve_precision("fp4", fallback=True), "bf16")
        with self.assertRaises(RuntimeError):
            hopper_caps.resolve_precision("fp4", fallback=False)

        # Profile 2: Ampere SM86 (lacks FP8 hardware tensor cores)
        ampere_caps = PrecisionCapabilities(
            device=torch.device("cuda:0"),
            compute_capability=(8, 6),
            fp8_gemm=False,
            fp4_gemm=False,
            transformer_engine=False,
            native_scaled_mm=False,
            bf16_supported=True,
        )
        self.assertEqual(ampere_caps.resolve_precision("bf16"), "bf16")
        self.assertEqual(ampere_caps.resolve_precision("fp8", fallback=True), "bf16")
        with self.assertRaises(RuntimeError):
            ampere_caps.resolve_precision("fp8", fallback=False)
        with self.assertRaises(RuntimeError):
            ampere_caps.resolve_precision("fp4", fallback=False)

        # Profile 3: Legacy GPU / CPU (FP32 only)
        legacy_caps = PrecisionCapabilities(
            device=torch.device("cpu"),
            compute_capability=(0, 0),
            fp8_gemm=False,
            fp4_gemm=False,
            transformer_engine=False,
            native_scaled_mm=False,
            bf16_supported=False,
        )
        self.assertEqual(legacy_caps.resolve_precision("fp32"), "fp32")
        self.assertEqual(legacy_caps.resolve_precision("bf16", fallback=True), "fp32")
        self.assertEqual(legacy_caps.resolve_precision("fp8", fallback=True), "fp32")
        with self.assertRaises(RuntimeError):
            legacy_caps.resolve_precision("bf16", fallback=False)

        # Profile 4: Blackwell SM100 with full FP4 and FP8 support
        blackwell_caps = PrecisionCapabilities(
            device=torch.device("cuda:0"),
            compute_capability=(10, 0),
            fp8_gemm=True,
            fp4_gemm=True,
            transformer_engine=True,
            native_scaled_mm=True,
            bf16_supported=True,
        )
        self.assertEqual(blackwell_caps.resolve_precision("fp4"), "fp4")
        self.assertEqual(blackwell_caps.resolve_precision("fp8"), "fp8")


class TestFP8StagingBuffer(unittest.TestCase):
    """Test FP8 staging encapsulation, memory reduction, and master weight immutability."""

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_fp8_staging_lifecycle_and_immutability(self):
        """Asserts master weight immutability, 1-byte storage, and clean restoration."""
        device = torch.device("cuda:0")
        weight = nn.Parameter(torch.randn(128, 256, dtype=torch.bfloat16))
        master_copy = weight.data.clone()

        buf = FP8StagingBuffer(param=weight, cpu_master=weight.data)
        self.assertEqual(buf.cpu_master.dtype, torch.bfloat16)

        # 1. Stage to GPU
        gpu_tensor = buf.stage_to_gpu(device=device, use_hardware_fp8=True)
        self.assertIsNotNone(buf.fp8_tensor)
        self.assertEqual(buf.fp8_tensor.dtype, torch.float8_e4m3fn)
        self.assertEqual(buf.fp8_tensor.device.type, "cuda")

        # FP8 tensor is 1 byte per element; BF16 is 2 bytes per element
        self.assertEqual(buf.fp8_tensor.element_size(), 1)
        self.assertEqual(buf.cpu_master.element_size(), 2)
        self.assertEqual(buf.fp8_tensor.numel(), 128 * 256)

        # CRITICAL INVARIANT: CPU master must NOT be modified or overwritten into FP8
        self.assertEqual(buf.cpu_master.dtype, torch.bfloat16)
        self.assertEqual(buf.cpu_master.device.type, "cpu")
        torch.testing.assert_close(buf.cpu_master, master_copy)

        # Scale metadata exists
        self.assertIsNotNone(buf.scale)
        self.assertIsNotNone(buf.inv_scale)
        self.assertIsNotNone(buf.amax)

        # 2. Release GPU
        buf.release_gpu()
        self.assertIsNone(buf.gpu_tensor)
        self.assertIsNone(buf.fp8_tensor)
        self.assertIsNone(buf.scale)
        self.assertIsNone(buf.inv_scale)

        # Param data points directly back to permanent CPU master storage
        self.assertEqual(weight.data.data_ptr(), buf.cpu_master.data_ptr())
        self.assertEqual(weight.dtype, torch.bfloat16)
        torch.testing.assert_close(weight.data, master_copy)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_parameter_stager_with_fp8(self):
        """Asserts ParameterStager instantiates FP8StagingBuffer for 2D weights of FP8-aware modules."""
        device = torch.device("cuda:0")
        fp8_linear = FP8Linear(64, 128, bias=True).to(torch.bfloat16)
        ordinary_linear = nn.Linear(64, 128, bias=True).to(torch.bfloat16)

        stager = ParameterStager(device=device, use_fp8=True)
        stager.register_module(fp8_linear)
        stager.register_module(ordinary_linear)

        w_buf_fp8 = stager.buffers[id(fp8_linear.weight)]
        b_buf_fp8 = stager.buffers[id(fp8_linear.bias)]
        w_buf_ord = stager.buffers[id(ordinary_linear.weight)]

        self.assertIsInstance(w_buf_fp8, FP8StagingBuffer)
        # 1D bias is not quantized to FP8
        self.assertIsInstance(b_buf_fp8, StagingBuffer)
        self.assertNotIsInstance(b_buf_fp8, FP8StagingBuffer)
        # Ordinary nn.Linear is never quantized to FP8
        self.assertIsInstance(w_buf_ord, StagingBuffer)
        self.assertNotIsInstance(w_buf_ord, FP8StagingBuffer)

        # Stage and apply
        stager.stage_layer(fp8_linear, use_hardware_fp8=True)
        stager.apply_staged_tensors(fp8_linear)

        self.assertEqual(fp8_linear.weight.device.type, "cuda")
        self.assertEqual(fp8_linear.bias.device.type, "cuda")

        # Release layer
        stager.release_layer(fp8_linear)
        self.assertEqual(fp8_linear.weight.device.type, "cpu")
        self.assertEqual(fp8_linear.bias.device.type, "cpu")
        self.assertEqual(fp8_linear.weight.dtype, torch.bfloat16)


class TestStreamingFP8Engine(unittest.TestCase):
    """Test LayerStreamingEngine execution and lifecycle with fp8_weights=True."""

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_streaming_fp8_forward_backward_and_detach(self):
        """Validates that fp8_weights=True executes cleanly and detaches without memory leaks."""
        device = torch.device("cuda:0")
        model = TriuneTransformer(
            vocab_size=100,
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            head_dim=32,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
        ).to(device)

        engine = LayerStreamingEngine(
            model,
            device=device,
            config=StreamingConfig(
                enabled=True,
                async_prefetch=True,
                fp8_weights=True,
            ),
        )
        engine.attach()

        self.assertTrue(engine.is_attached)
        self.assertTrue(model._layer_streaming_active)

        # Verify layer parameters reside on CPU
        for layer in engine.layers:
            for p in layer.parameters():
                self.assertEqual(p._cpu_data.device.type, "cpu")
                self.assertIn(p._cpu_data.dtype, (torch.float32, torch.bfloat16))

        # Forward pass
        input_ids = torch.randint(0, 100, (2, 16), device=device)
        logits, route_logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 16, 100))

        # Backward pass
        loss = logits.sum()
        loss.backward()

        # Engine step
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        engine.step_streaming_optimizer(optimizer)

        # Detach
        engine.detach()
        self.assertFalse(engine.is_attached)
        self.assertFalse(getattr(model, "_layer_streaming_active", False))
        self.assertIsNone(getattr(model, "_streaming_engine", None))

        # Stale hooks, events, and gradient attributes must be completely removed
        for layer in model.layers:
            self.assertFalse(hasattr(layer, "_prefetched_cuda_params"))
            self.assertFalse(hasattr(layer, "_prefetched_cuda_buffers"))
            for p in layer.parameters():
                self.assertFalse(hasattr(p, "_d2h_event"))
                self.assertFalse(hasattr(p, "_cpu_grad"))


class TestFP8LinearExecution(unittest.TestCase):
    """Test FP8Linear fallback and forward/backward stability."""

    def test_fp8_linear_cpu_fallback(self):
        """FP8Linear on CPU falls back gracefully to standard F.linear."""
        layer = FP8Linear(32, 64, bias=True, dtype=torch.bfloat16)
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        out = layer(x)
        self.assertEqual(out.shape, (4, 64))

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_fp8_linear_cuda_execution(self):
        """FP8Linear on CUDA executes reliably regardless of native scaled_mm availability."""
        device = torch.device("cuda:0")
        layer = FP8Linear(32, 64, bias=True, dtype=torch.bfloat16).to(device)
        x = torch.randn(4, 32, dtype=torch.bfloat16, device=device, requires_grad=True)

        out = layer(x)
        self.assertEqual(out.shape, (4, 64))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertEqual(x.grad.shape, (4, 32))
        self.assertEqual(layer.weight.grad.shape, (64, 32))


if __name__ == "__main__":
    unittest.main()
