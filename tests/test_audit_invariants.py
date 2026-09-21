"""Comprehensive regression test suite verifying all architectural audit invariants."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.model.router import GumbelSoftmaxRouter
from triune.model.attention import VectorisedGLA
from triune.model.moe import MoE_FFN, RoutingResult
from triune.model.zoo import load_model
from triune.data.dataloader import CyclingDataLoader
from triune.trainer.checkpoint import save_latest, load_checkpoint
from triune.runtime.staging import ParameterStager, StagingBuffer


class ArchitecturalAuditInvariantsTest(unittest.TestCase):
    def test_transformer_structural_invariants(self):
        """P1.7 & P1.8: Invariant defense in TriuneTransformer __init__ and forward()."""
        # 1. Invalid layer ordering: reflex >= limbic
        with self.assertRaises(ValueError):
            TriuneTransformer(
                vocab_size=100, hidden_dim=64, num_layers=8, num_heads=2, head_dim=32,
                router_prefix_layers=2, reflex_exit_layer=5, limbic_exit_layer=4
            )

        # 2. Invalid layer ordering: prefix >= reflex
        with self.assertRaises(ValueError):
            TriuneTransformer(
                vocab_size=100, hidden_dim=64, num_layers=8, num_heads=2, head_dim=32,
                router_prefix_layers=4, reflex_exit_layer=4, limbic_exit_layer=6
            )

        # 3. Invalid target depth distribution (does not sum to 1.0)
        with self.assertRaises(ValueError):
            TriuneTransformer(
                vocab_size=100, hidden_dim=64, num_layers=8, num_heads=2, head_dim=32,
                target_depth_dist=(0.5, 0.5, 0.5)
            )

        # 4. Invalid force_depth in forward (early check)
        model = TriuneTransformer(
            vocab_size=100, hidden_dim=64, num_layers=6, num_heads=2, head_dim=32,
            router_prefix_layers=1, reflex_exit_layer=2, limbic_exit_layer=4
        )
        x = torch.randint(0, 100, (2, 8))
        with self.assertRaises(ValueError):
            model(x, force_depth=3)
        with self.assertRaises(ValueError):
            model(x, force_depth=-1)

    def test_router_temperature_and_target_dist(self):
        """P1.6 & P1.9: Gumbel temperature and distribution validation."""
        router = GumbelSoftmaxRouter(hidden_dim=64)
        x = torch.randn(2, 4, 64)

        # Temperature <= 0
        with self.assertRaises(ValueError):
            router(x, temperature=0.0)
        with self.assertRaises(ValueError):
            router(x, temperature=-0.5)

        # Non-finite temperature
        with self.assertRaises(ValueError):
            router(x, temperature=float('nan'))
        with self.assertRaises(ValueError):
            router(x, temperature=float('inf'))

        # Invalid target distribution on init
        with self.assertRaises(ValueError):
            GumbelSoftmaxRouter(hidden_dim=64, target_depth_dist=(0.1, 0.2))
        with self.assertRaises(ValueError):
            GumbelSoftmaxRouter(hidden_dim=64, target_depth_dist=(0.8, 0.8, 0.8))

    def test_moe_topk_renormalization_and_accounting(self):
        """P1.1, P1.2, P1.3: MoE top-k renormalization and routing statistics."""
        dim = 64
        num_experts = 4
        # Test top_k=2 renormalization
        moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=2, shared_expert=False)
        moe.train()
        x = torch.randn(2, 16, dim)
        out = moe(x, update_stats=True)
        self.assertEqual(out.shape, x.shape)

        # Verify load accounting
        self.assertTrue(hasattr(moe, "last_counts"))
        self.assertTrue(hasattr(moe, "last_requested_counts"))
        self.assertTrue(hasattr(moe, "last_accepted_counts"))
        self.assertEqual(moe.expert_load_ratio.shape, (num_experts,))
        self.assertTrue((moe.expert_load_ratio > 0).all())

    def test_gla_recurrent_cache_autoregressive_parity(self):
        """P2.1 & P2.2: VectorisedGLA recurrent state cache parity."""
        torch.manual_seed(42)
        dim = 64
        heads = 2
        head_dim = 32
        gla = VectorisedGLA(dim=dim, heads=heads, head_dim=head_dim)
        gla.eval()

        # Generate random sequence of length 4
        x = torch.randn(1, 4, dim)

        # 1. Full sequence forward pass (no cache)
        out_full, _ = gla(x)

        # 2. Step-by-step autoregressive generation using recurrent state cache (state, offset)
        cache = (None, 0)
        step_outputs = []
        for t in range(4):
            x_t = x[:, t : t + 1, :]
            out_t, cache = gla(x_t, cache=cache)
            step_outputs.append(out_t)

        out_recurrent = torch.cat(step_outputs, dim=1)

        # Compare outputs: step-by-step recurrent state matches full forward
        max_diff = (out_full - out_recurrent).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"Recurrent cache output drifted: max diff {max_diff}")

    def test_strict_model_zoo_loading(self):
        """P2.3: Strict load_model error handling for unknown models."""
        with self.assertRaises(ValueError) as ctx:
            load_model("totally_fake_architecture_xyz_404")
        self.assertIn("could not be resolved", str(ctx.exception))

        # Default model loads fine with valid layer hierarchy
        model_default = load_model(
            "default",
            num_layers=6,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
            hidden_dim=64,
            num_heads=2,
            head_dim=32,
            vocab_size=100,
        )
        self.assertIsNotNone(model_default)

    def test_cycling_dataloader_guards(self):
        """P2.4: Empty dataloader handling."""
        # Empty sized dataset
        with self.assertRaises(ValueError):
            CyclingDataLoader([])

        # Empty generator
        def empty_gen():
            if False:
                yield 1
        with self.assertRaises(RuntimeError):
            cdl = CyclingDataLoader(empty_gen())
            cdl.next()

    def test_checkpoint_fingerprint_validation(self):
        """P2.5: Checkpoint architecture fingerprint verification."""
        temp_dir = tempfile.mkdtemp()

        class DummyTrainer:
            def __init__(self, cfg):
                self.config = cfg
                self.device = torch.device("cpu")
                self.model = nn.Linear(cfg.get("hidden_dim", 32), 10)
                self.optimizer = torch.optim.AdamW(self.model.parameters())

        cfg_orig = {
            "checkpoint_dir": temp_dir,
            "hidden_dim": 32,
            "num_layers": 4,
            "num_heads": 2,
            "num_experts": 4,
            "vocab_size": 100,
        }
        trainer_orig = DummyTrainer(cfg_orig)

        class DummyEngine:
            best_eval_loss = 0.5
            depth_usage_ema = [0.33, 0.33, 0.34]

        ckpt_path = save_latest(trainer_orig, DummyEngine(), step=10, loss=0.5)

        # Successful load with matching config
        loaded = load_checkpoint(trainer_orig, ckpt_path, load_optimizer=True)
        self.assertEqual(loaded["step"], 10)

        # Incompatible architecture: mismatched hidden_dim
        cfg_bad = dict(cfg_orig)
        cfg_bad["hidden_dim"] = 64
        trainer_bad = DummyTrainer(cfg_bad)

        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(trainer_bad, ckpt_path, load_optimizer=True)
        self.assertIn("Checkpoint architecture mismatch", str(ctx.exception))

    def test_moe_routing_result_and_accepted_centroids(self):
        """Phase 7 & 8: RoutingResult container and accepted-only centroid statistics."""
        dim = 32
        num_experts = 2
        moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=1, shared_expert=False)
        moe.train()

        torch.manual_seed(42)
        # B*T = 100, capacity = math.ceil(100 / 2 * 1.5) = 75
        x = torch.randn(10, 10, dim)
        flat_x = x.reshape(100, dim)

        # Force all tokens to request expert 0 with varying confidence
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.bias.copy_(torch.tensor([100.0, 0.0]))
            moe.expert_bias.zero_()

        out = moe(x, update_stats=True)

        # 1. RoutingResult structure validation
        self.assertTrue(hasattr(moe, "last_requested_counts"))
        self.assertTrue(hasattr(moe, "last_accepted_counts"))
        self.assertTrue(hasattr(moe, "last_dropped_counts"))
        self.assertEqual(moe.last_requested_counts.tolist(), [100, 0])
        self.assertEqual(moe.last_accepted_counts.tolist(), [75, 0])
        self.assertEqual(moe.last_dropped_counts.tolist(), [25, 0])

        # 2. Phase 8 verification: centroid calculation must NOT equal the full 100 requested tokens
        unconstrained_mean = flat_x.mean(dim=0)
        centroid_0 = moe.last_centroids[0]
        centroid_diff_unconstrained = (centroid_0 - unconstrained_mean).abs().max().item()
        self.assertGreater(centroid_diff_unconstrained, 1e-4, "Centroid should not include dropped tokens")

        # 3. Direct update_routing_stats consistency
        moe.update_routing_stats(x)
        self.assertEqual(moe.last_requested_counts.tolist(), [100, 0])
        self.assertEqual(moe.last_accepted_counts.tolist(), [75, 0])
        self.assertEqual(moe.last_dropped_counts.tolist(), [25, 0])

    def test_parameter_stager_lifecycle(self):
        """Phase 3 & 14: ParameterStager isolation and lifecycle management."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        linear = nn.Linear(16, 16)
        orig_weight_cpu = linear.weight.clone()

        stager = ParameterStager(device=device)
        stager.register_module(linear)

        # 1. Stage to GPU
        stager.stage_layer(linear)
        self.assertEqual(linear.weight.data.device.type, device.type)

        # 2. Mutate staged GPU tensor (simulating optimizer or forward update)
        with torch.no_grad():
            linear.weight.data.add_(1.0)

        # 3. Sync back to permanent CPU master and release
        stager.sync_layer_to_cpu(linear)
        self.assertEqual(linear.weight.data.device.type, "cpu")
        torch.testing.assert_close(linear.weight.data, orig_weight_cpu + 1.0)

        # 4. Release and cleanup
        stager.release_layer(linear)
        stager.clear()
        self.assertEqual(len(stager.buffers), 0)


if __name__ == "__main__":
    unittest.main()
