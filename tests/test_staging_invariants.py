"""Test ParameterStager invariants during layer streaming.

Verifies:
1. Parameter identity (id(p)) is strictly preserved through stage/release cycles.
2. CPU master tensor is restored upon release.
3. Forward and backward autograd passes function correctly with staged buffers.
4. FP8 temporary scaling attributes do not corrupt CPU master precision.
"""

import unittest
import torch
import torch.nn as nn

from triune.runtime.staging import ParameterStager, StagingBuffer, FP8StagingBuffer


class DummyLayer(nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class TestStagingInvariants(unittest.TestCase):
    def test_parameter_identity_and_restoration(self):
        """Verify that staging and releasing a layer preserves parameter id and restores CPU master."""
        layer = DummyLayer(8, 8)
        initial_params = list(layer.parameters())
        param_ids_before = [id(p) for p in initial_params]
        orig_weights = [p.clone() for p in initial_params]

        device = torch.device("cpu")
        stager = ParameterStager(device=device, use_fp8=False)
        stager.register_module(layer)

        # Stage layer
        stager.stage_layer(layer)
        stager.apply_staged_tensors(layer)

        # Ensure identities didn't change
        param_ids_during = [id(p) for p in layer.parameters()]
        self.assertEqual(param_ids_before, param_ids_during)

        # Release layer
        stager.release_layer(layer)
        param_ids_after = [id(p) for p in layer.parameters()]
        self.assertEqual(param_ids_before, param_ids_after)

        # Ensure data matches original values
        for orig, current in zip(orig_weights, layer.parameters()):
            self.assertTrue(torch.allclose(orig, current))

    def test_forward_backward_gradient_flow(self):
        """Verify forward and backward gradients flow properly through staged layer."""
        layer = DummyLayer(16, 16)
        device = torch.device("cpu")
        stager = ParameterStager(device=device, use_fp8=False)
        stager.register_module(layer)

        stager.stage_layer(layer)
        stager.apply_staged_tensors(layer)

        x = torch.randn(4, 16, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(layer.linear.weight.grad)
        self.assertEqual(layer.linear.weight.grad.shape, layer.linear.weight.shape)

        stager.release_layer(layer)

    def test_cpu_master_mutation_does_not_corrupt_backward(self):
        """Forward with weight A -> release to CPU -> mutate CPU master to B -> backward must use exact A."""
        torch.manual_seed(42)

        class TwoStageLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(16, 16)
                self.lin2 = nn.Linear(16, 16)

            def forward(self, x):
                return self.lin2(torch.relu(self.lin1(x)))

        ref_layer = TwoStageLayer()
        staged_layer = TwoStageLayer()
        staged_layer.load_state_dict(ref_layer.state_dict())

        x_ref = torch.randn(4, 16, requires_grad=True)
        x_staged = x_ref.detach().clone().requires_grad_(True)

        # Reference forward + backward with weight A
        out_ref = ref_layer(x_ref)
        (out_ref ** 2).sum().backward()

        # Staged forward with weight A
        stager = ParameterStager(device=torch.device("cpu"), use_fp8=False)
        stager.register_module(staged_layer)
        stager.stage_layer(staged_layer)
        stager.apply_staged_tensors(staged_layer)

        out_staged = staged_layer(x_staged)

        # Release to CPU and aggressively mutate CPU master weights in-place to B
        stager.release_layer(staged_layer)
        for p in staged_layer.parameters():
            p._cpu_data.mul_(-3.5).add_(42.0)

        # Backward must restage from forward snapshot A, not mutated B
        (out_staged ** 2).sum().backward()

        self.assertTrue(torch.allclose(x_ref.grad, x_staged.grad, atol=1e-6))
        for (name, p_ref), (_, p_staged) in zip(ref_layer.named_parameters(), staged_layer.named_parameters()):
            self.assertIsNotNone(p_staged.grad, f"Missing grad for {name}")
            self.assertTrue(
                torch.allclose(p_ref.grad, p_staged.grad, atol=1e-6),
                f"Gradient mismatch for {name}: mutated CPU master leaked into backward!",
            )
        stager.clear()

    def _assert_streaming_mutation_invariance(
        self,
        *,
        force_depth=None,
        use_checkpointing=False,
        use_fp8=False,
    ):
        """Helper verifying LayerStreamingEngine backward uses forward weight A even if CPU master is mutated to B."""
        from triune.model.transformer import TriuneTransformer

        torch.manual_seed(123)
        ref_model = TriuneTransformer(
            vocab_size=64,
            hidden_dim=256,
            num_layers=6,
            num_heads=2,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
            use_fp4=False,
            use_fp8=use_fp8,
        )
        stream_model = TriuneTransformer(
            vocab_size=64,
            hidden_dim=256,
            num_layers=6,
            num_heads=2,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
            use_fp4=False,
            use_fp8=use_fp8,
        )
        stream_model.load_state_dict(ref_model.state_dict())

        if use_checkpointing:
            ref_model.enable_gradient_checkpointing()
            stream_model.enable_gradient_checkpointing()

        stream_model.enable_layer_streaming(
            device=torch.device("cpu"),
            fp8_weights=use_fp8,
            pin_memory=False,
            async_prefetch=False,
        )

        input_ids = torch.randint(0, 64, (4, 16))

        # Reference forward + backward with weight A
        torch.manual_seed(999)
        logits_ref, route_ref, bal_ref = ref_model(
            input_ids, force_depth=force_depth, temperature=1.0, return_balance_loss=True
        )
        loss_ref = (logits_ref.float() ** 2).mean() + 0.1 * bal_ref.float()
        loss_ref.backward()

        # Streaming forward with weight A
        torch.manual_seed(999)
        logits_s, route_s, bal_s = stream_model(
            input_ids, force_depth=force_depth, temperature=1.0, return_balance_loss=True
        )
        self.assertTrue(torch.allclose(logits_ref.float(), logits_s.float(), atol=1e-5))

        # Mutate all released CPU master weights across all layers to B before backward!
        for layer in stream_model.layers:
            for p in layer.parameters():
                p._cpu_data.mul_(-2.0).add_(15.0)

        loss_s = (logits_s.float() ** 2).mean() + 0.1 * bal_s.float()
        loss_s.backward()

        # Verify every layer parameter's offloaded _cpu_grad matches reference grad from weight A
        for l_idx, (l_ref, l_s) in enumerate(zip(ref_model.layers, stream_model.layers)):
            for (p_name, p_ref), (_, p_s) in zip(l_ref.named_parameters(), l_s.named_parameters()):
                if p_ref.grad is None:
                    continue
                grad_s = getattr(p_s, "_cpu_grad", p_s.grad)
                self.assertIsNotNone(grad_s, f"Layer {l_idx}.{p_name} missing gradient")
                self.assertTrue(
                    torch.allclose(p_ref.grad.float(), grad_s.float(), atol=1e-4, rtol=1e-3),
                    f"Layer {l_idx}.{p_name} gradient corrupted by CPU master mutation!",
                )

        stream_model._streaming_engine.detach()

    def test_streaming_mutation_invariance_adaptive_early_exit(self):
        """Verify forward-snapshot isolation under adaptive early exit (force_depth=None)."""
        self._assert_streaming_mutation_invariance(
            force_depth=None, use_checkpointing=False, use_fp8=False
        )

    def test_streaming_mutation_invariance_with_checkpointing(self):
        """Verify forward-snapshot isolation with gradient checkpointing enabled."""
        self._assert_streaming_mutation_invariance(
            force_depth=2, use_checkpointing=True, use_fp8=False
        )

    def test_streaming_mutation_invariance_with_fp8(self):
        """Verify forward-snapshot isolation with FP8 linear layers and FP8 staging."""
        self._assert_streaming_mutation_invariance(
            force_depth=2, use_checkpointing=False, use_fp8=True
        )

    def test_deferred_mode_zero_snapshot_memory_savings(self):
        """Verify Mode 2 (capture_forward_snapshots=False) eliminates forward snapshot memory and matches gradients."""
        torch.manual_seed(42)
        dev = torch.device("cpu")

        layer1 = DummyLayer(64, 64)
        layer2 = DummyLayer(64, 64)
        layer2.load_state_dict(layer1.state_dict())

        # Mode 1: Immediate with forward snapshots
        stager1 = ParameterStager(device=dev, use_fp8=False, capture_forward_snapshots=True)
        stager1.register_module(layer1)
        stager1.stage_layer(layer1)
        stager1.apply_staged_tensors(layer1)

        x1 = torch.randn(8, 64, requires_grad=True)
        out1 = layer1(x1)
        sum1 = stager1.get_memory_summary()
        self.assertGreater(sum1["forward_snapshot_mb"], 0.0, "Mode 1 should allocate forward snapshots")

        out1.sum().backward()
        stager1.release_layer(layer1)

        # Mode 2: Deferred zero-snapshot streaming
        stager2 = ParameterStager(device=dev, use_fp8=False, capture_forward_snapshots=False)
        stager2.register_module(layer2)
        stager2.stage_layer(layer2)
        stager2.apply_staged_tensors(layer2)

        x2 = x1.detach().clone().requires_grad_(True)
        out2 = layer2(x2)
        sum2 = stager2.get_memory_summary()
        self.assertEqual(sum2["forward_snapshot_mb"], 0.0, "Mode 2 must have zero forward snapshot allocation")

        out2.sum().backward()
        stager2.release_layer(layer2)

        # Gradients must be numerically identical
        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=1e-6))
        self.assertTrue(torch.allclose(layer1.linear.weight.grad, layer2.linear.weight.grad, atol=1e-6))
        self.assertTrue(torch.allclose(layer1.linear.bias.grad, layer2.linear.bias.grad, atol=1e-6))
        stager1.clear()
        stager2.clear()

    def test_transfer_telemetry_tracking(self):
        """Verify that H2D and D2H transfer telemetry counts and byte counters are accurately maintained."""
        dev = torch.device("cpu")
        layer = DummyLayer(32, 32)
        stager = ParameterStager(device=dev, use_fp8=False)
        stager.register_module(layer)

        self.assertEqual(stager.telemetry["h2d_sync_transfers"], 0)
        self.assertEqual(stager.telemetry["h2d_bytes_transferred"], 0)

        # Stage layer (H2D)
        stager.stage_layer(layer)
        self.assertGreater(stager.telemetry["h2d_sync_transfers"], 0)
        self.assertGreater(stager.telemetry["h2d_bytes_transferred"], 0)

        # Sync layer back to CPU (D2H)
        stager.sync_layer_to_cpu(layer)
        self.assertGreater(stager.telemetry["d2h_sync_transfers"], 0)
        self.assertGreater(stager.telemetry["d2h_bytes_transferred"], 0)

        summary = stager.get_memory_summary()
        self.assertIn("telemetry", summary)
        self.assertEqual(summary["total_parameters"], 2)

        # Reset telemetry
        stager.reset_telemetry()
        self.assertEqual(stager.telemetry["h2d_sync_transfers"], 0)
        self.assertEqual(stager.telemetry["h2d_bytes_transferred"], 0)
        stager.clear()


if __name__ == "__main__":
    unittest.main()
