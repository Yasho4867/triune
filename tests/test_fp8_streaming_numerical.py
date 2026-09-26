import unittest
import torch
import torch.nn as nn

from triune.model.fp8 import FP8Linear, _FP8MatmulFn
from triune.runtime.staging import ParameterStager, StagingBuffer, FP8StagingBuffer
from triune.runtime.capabilities import PrecisionCapabilities
from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig


class TestFP8StreamingNumerical(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.caps = PrecisionCapabilities.detect()

    def test_ordinary_linear_not_staged_as_fp8(self):
        """Ordinary nn.Linear must NEVER receive an FP8 staging buffer or _fp8_tensor."""
        class MixedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.ordinary = nn.Linear(32, 32, bias=False)
                self.fp8_mod = FP8Linear(32, 32, bias=False)

        mod = MixedModule()
        stager = ParameterStager(
            device=self.device,
            prefetch_stream=None,
            d2h_stream=None,
            use_fp8=True,
        )
        stager.register_module(mod)

        # 1. Check registered buffer types
        ord_buf = stager.buffers[id(mod.ordinary.weight)]
        fp8_buf = stager.buffers[id(mod.fp8_mod.weight)]

        self.assertIs(type(ord_buf), StagingBuffer, "Ordinary nn.Linear must use standard StagingBuffer")
        self.assertIs(type(fp8_buf), FP8StagingBuffer, "FP8Linear must use FP8StagingBuffer")

        # 2. Stage the module
        stager.stage_layer(mod, stream=None, target_dtype=torch.bfloat16, use_hardware_fp8=True)
        stager.apply_staged_tensors(mod)

        # Ordinary linear must NOT have FP8 attributes
        self.assertFalse(hasattr(mod.ordinary.weight, "_fp8_tensor") and mod.ordinary.weight._fp8_tensor is not None,
                         "Ordinary linear must never receive _fp8_tensor")
        self.assertFalse(hasattr(mod.ordinary.weight, "_fp8_inv_scale") and mod.ordinary.weight._fp8_inv_scale is not None,
                         "Ordinary linear must never receive _fp8_inv_scale")
        self.assertEqual(mod.ordinary.weight.data.dtype, torch.bfloat16)

        # FP8 module receives _fp8_tensor and scale metadata if hardware fp8 is active
        self.assertEqual(mod.fp8_mod.weight.data.dtype, torch.bfloat16,
                         "Master weight data pointer on GPU must remain BF16")
        if self.caps.native_scaled_mm:
            self.assertIsNotNone(getattr(mod.fp8_mod.weight, "_fp8_tensor", None))
            self.assertIsNotNone(getattr(mod.fp8_mod.weight, "_fp8_inv_scale", None))

        # 3. Release layer
        stager.release_layer(mod)
        self.assertEqual(mod.ordinary.weight.data.data_ptr(), mod.ordinary.weight._cpu_data.data_ptr())
        self.assertEqual(mod.ordinary.weight.device.type, "cpu")
        self.assertEqual(mod.fp8_mod.weight.data.data_ptr(), mod.fp8_mod.weight._cpu_data.data_ptr())
        self.assertEqual(mod.fp8_mod.weight.device.type, "cpu")
        self.assertIsNone(getattr(mod.fp8_mod.weight, "_fp8_tensor", None))
        self.assertIsNone(getattr(mod.fp8_mod.weight, "_fp8_inv_scale", None))

        stager.clear()

    def test_fp8_parameter_identity_and_master_invariant(self):
        """CPU master weights remain BF16/FP32 and autograd tracks the master parameter."""
        torch.manual_seed(42)
        in_features, out_features = 64, 64
        fp8_layer = FP8Linear(in_features, out_features, bias=True).to(torch.bfloat16)
        
        cpu_master_weight = fp8_layer.weight.clone()
        cpu_master_bias = fp8_layer.bias.clone()

        stager = ParameterStager(
            device=self.device,
            prefetch_stream=None,
            d2h_stream=None,
            use_fp8=True,
        )
        stager.register_module(fp8_layer)

        # Stage with BF16 target
        stager.stage_layer(fp8_layer, stream=None, target_dtype=torch.bfloat16, use_hardware_fp8=True)
        stager.apply_staged_tensors(fp8_layer)

        # Invariant: p.data is GPU tensor in BF16
        self.assertEqual(fp8_layer.weight.data.dtype, torch.bfloat16)
        self.assertEqual(fp8_layer.weight.data.device.type, self.device.type)

        # Run forward + backward
        x = torch.randn(2, 4, in_features, dtype=torch.bfloat16, device=self.device)
        out = fp8_layer(x)
        loss = out.sum()
        loss.backward()

        # Invariant: grad_weight must be returned in weight_master.dtype (bfloat16)
        self.assertIsNotNone(fp8_layer.weight.grad)
        self.assertEqual(fp8_layer.weight.grad.dtype, torch.bfloat16)
        self.assertEqual(fp8_layer.bias.grad.dtype, torch.bfloat16)

        # Invariant: CPU master tensor was NOT corrupted
        self.assertTrue(torch.equal(fp8_layer.weight._cpu_data, cpu_master_weight))
        self.assertTrue(torch.equal(fp8_layer.bias._cpu_data, cpu_master_bias))

        stager.release_layer(fp8_layer)
        self.assertEqual(fp8_layer.weight.data.data_ptr(), fp8_layer.weight._cpu_data.data_ptr())
        self.assertEqual(fp8_layer.weight.device.type, "cpu")
        stager.clear()

    def test_fp8_linear_numerical_parity_vs_bf16(self):
        """Test A: FP8Linear output is numerically bounded relative to standard BF16 Linear."""
        torch.manual_seed(123)
        in_features, out_features = 128, 128
        ref_linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16, device=self.device)
        fp8_linear = FP8Linear(in_features, out_features, bias=False, dtype=torch.bfloat16, device=self.device)
        
        with torch.no_grad():
            fp8_linear.weight.copy_(ref_linear.weight)

        x = torch.randn(4, 16, in_features, dtype=torch.bfloat16, device=self.device)

        out_ref = ref_linear(x)
        out_fp8 = fp8_linear(x)

        # FP8 E4M3 quantization has ~8-bit resolution; cosine similarity should be >= 0.99
        cos_sim = torch.nn.functional.cosine_similarity(out_ref.flatten(), out_fp8.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.99, f"Cosine similarity {cos_sim} too low between FP8Linear and BF16 Linear")

    def test_fp8_linear_streaming_vs_non_streaming_parity(self):
        """Test B: FP8 streaming vs non-streaming forward/backward output and gradient parity."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        torch.manual_seed(777)
        in_dim = 64
        out_dim = 64

        class TinyFP8Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj1 = FP8Linear(in_dim, out_dim, bias=False, dtype=torch.bfloat16)
                self.proj2 = FP8Linear(out_dim, in_dim, bias=False, dtype=torch.bfloat16)

            def forward(self, x):
                return self.proj2(torch.relu(self.proj1(x)))

        mod_ref = TinyFP8Block().to(self.device)
        mod_str = TinyFP8Block()
        mod_str.load_state_dict(mod_ref.state_dict())

        stager = ParameterStager(device=self.device, prefetch_stream=None, d2h_stream=None, use_fp8=True)
        stager.register_module(mod_str)

        x = torch.randn(2, 8, in_dim, dtype=torch.bfloat16, device=self.device)

        # Non-streaming forward + backward
        mod_ref.zero_grad()
        out_ref = mod_ref(x)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Streaming forward + backward
        mod_str.zero_grad()
        stager.stage_layer(mod_str, stream=None, target_dtype=torch.bfloat16, use_hardware_fp8=True)
        stager.apply_staged_tensors(mod_str)
        out_str = mod_str(x)
        loss_str = out_str.sum()
        loss_str.backward()

        # Check forward output parity
        torch.testing.assert_close(out_str, out_ref, rtol=1e-3, atol=1e-3)

        # Check gradient parity
        for (n_ref, p_ref), (n_str, p_str) in zip(mod_ref.named_parameters(), mod_str.named_parameters()):
            self.assertIsNotNone(p_ref.grad)
            self.assertIsNotNone(p_str.grad)
            torch.testing.assert_close(p_str.grad, p_ref.grad, rtol=1e-3, atol=1e-3)

        stager.release_layer(mod_str)
        stager.clear()


if __name__ == "__main__":
    unittest.main()
