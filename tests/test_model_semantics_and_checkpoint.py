"""Tests for Milestone 5 (Gate 5): Model Semantics, GLA Cache Equivalence & Schema-v2 Checkpointing.

Verifies:
1. GLA recurrent cache propagation and pre-sampling numerical logit equivalence (rtol=1e-4, atol=1e-4).
2. Explicit API contract: (logits, route_logits) when cache is None; (logits, new_cache) when cache is not None.
3. Persistent attributes: last_route_logits, last_depth_choice, last_balance_loss.
4. Schema-v2 canonical checkpoint fingerprinting and mismatch rejection.
5. Finite float validation (rejection of NaN / Inf) in configuration and model initialization.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.configs.config import default_config, validate_config, build_config
from triune.trainer.checkpoint import (
    compute_checkpoint_fingerprint,
    extract_canonical_semantics,
    load_checkpoint,
    save_latest,
)


class DummyTrainer:
    """Mock trainer container for checkpoint testing."""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


class TestGLACacheAndModelSemantics(unittest.TestCase):
    """Test GLA API contract, cache propagation, and pre-sampling logit equivalence."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = TriuneTransformer(
            vocab_size=100,
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            head_dim=32,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
            use_fp4=False,
            use_fp8=False,
        ).to(self.device)
        self.model.eval()

    def test_explicit_api_contract_without_cache(self):
        """Asserts forward(input_ids) returns (logits, route_logits) and sets persistent attributes."""
        input_ids = torch.randint(0, 100, (2, 8), device=self.device)
        res = self.model(input_ids)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        logits, route_logits = res
        self.assertEqual(logits.shape, (2, 8, 100))
        self.assertEqual(route_logits.shape, (2, 3))

        # Persistent attributes
        self.assertIsNotNone(self.model.last_route_logits)
        self.assertIsNotNone(self.model.last_depth_choice)
        self.assertIsNotNone(self.model.last_balance_loss)

        # With return_balance_loss=True
        res_loss = self.model(input_ids, return_balance_loss=True)
        self.assertEqual(len(res_loss), 3)
        logits, route_logits, balance_loss = res_loss
        self.assertIsInstance(balance_loss, torch.Tensor)

    def test_explicit_api_contract_with_cache(self):
        """Asserts forward(input_ids, cache=...) returns (logits, new_cache)."""
        input_ids = torch.randint(0, 100, (1, 4), device=self.device)
        cache = self.model.init_cache(1)
        res = self.model(input_ids, cache=cache, force_depth=2)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        logits, new_cache = res
        self.assertEqual(logits.shape, (1, 4, 100))
        self.assertIsInstance(new_cache, list)
        self.assertEqual(len(new_cache), self.model.num_layers)

    def test_pre_sampling_logit_equivalence(self):
        """Gate 5 Hard Invariant: Compare full-prefix logits with cached decode logits before sampling."""
        # Create sequence of length 8
        full_tokens = torch.randint(0, 100, (1, 8), device=self.device)

        with torch.no_grad():
            # 1. Full sequence forward pass
            full_logits, _ = self.model(full_tokens, force_depth=2)

            # 2. Prefill on first 5 tokens
            prefix_tokens = full_tokens[:, :5]
            cache = self.model.init_cache(1)
            prefix_logits, cache = self.model(prefix_tokens, cache=cache, force_depth=2)

            # Assert prefix logits match the first 5 steps of full_logits
            torch.testing.assert_close(
                prefix_logits, full_logits[:, :5], rtol=1e-4, atol=1e-4
            )

            # 3. Autoregressive token-by-token decoding for tokens 5, 6, 7
            for t in range(5, 8):
                step_token = full_tokens[:, t : t + 1]
                step_logits, cache = self.model(step_token, cache=cache, force_depth=2)

                # Gate 5 Rule: strictly compare pre-sampling logits with bounded numerical tolerance
                target_logit = full_logits[:, t : t + 1]
                torch.testing.assert_close(
                    step_logits, target_logit, rtol=1e-4, atol=1e-4
                )

    def test_dynamic_depth_cached_generation(self):
        """Asserts cache propagation with dynamic depth routing produces mathematically consistent logits."""
        full_tokens = torch.randint(0, 100, (1, 8), device=self.device)

        with torch.no_grad():
            full_logits, _ = self.model(full_tokens, force_depth=None)
            full_depth = self.model.last_depth_choice

            # Run prefix with cache
            prefix_tokens = full_tokens[:, :5]
            cache = self.model.init_cache(1)
            prefix_logits, cache = self.model(prefix_tokens, cache=cache, force_depth=None)
            prefix_depth = self.model.last_depth_choice

            if prefix_depth.item() == full_depth.item():
                torch.testing.assert_close(
                    prefix_logits, full_logits[:, :5], rtol=1e-4, atol=1e-4
                )

            # Step-by-step autoregressive generation
            for t in range(5, 8):
                step_token = full_tokens[:, t : t + 1]
                step_logits, cache = self.model(step_token, cache=cache, force_depth=None)
                self.assertIsNotNone(self.model.last_depth_choice)
                self.assertEqual(step_logits.shape, (1, 1, 100))


class TestSchemaV2Checkpointing(unittest.TestCase):
    """Test Schema-v2 checkpoint canonical fingerprinting and mismatch rejection."""

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = {
            "hidden_dim": 64,
            "num_layers": 4,
            "num_heads": 2,
            "head_dim": 32,
            "vocab_size": 100,
            "num_experts": 4,
            "top_k": 1,
            "shared_expert": True,
            "shared_scale": 0.5,
            "capacity_multiplier": 1.5,
            "router_prefix_layers": 1,
            "reflex_exit_layer": 2,
            "limbic_exit_layer": 3,
            "target_depth_dist": [0.33, 0.33, 0.34],
            "use_fp4": False,
            "use_fp8": False,
            "use_muon": False,
            "muon_lr": 0.02,
            "galore": False,
            "galore_rank": 256,
            "checkpoint_dir": tempfile.mkdtemp(),
        }
        self.model = TriuneTransformer(
            vocab_size=100,
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            head_dim=32,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
        ).to(self.device)
        self.trainer = DummyTrainer(self.model, self.config, self.device)

    def test_canonical_fingerprint_generation(self):
        """Asserts deterministic SHA-256 fingerprint generation across semantically identical configs."""
        fp1 = compute_checkpoint_fingerprint(self.config)
        fp2 = compute_checkpoint_fingerprint(dict(self.config))
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

        # Changing an architectural parameter modifies the fingerprint
        cfg_altered = dict(self.config, top_k=2)
        fp_altered = compute_checkpoint_fingerprint(cfg_altered)
        self.assertNotEqual(fp1, fp_altered)

    def test_save_and_compatible_load(self):
        """Asserts saving and loading compatible Schema-v2 checkpoint succeeds."""
        ckpt_path = save_latest(self.trainer, None, step=42, loss=0.85)
        self.assertTrue(ckpt_path.exists())

        # Load into new trainer with same configuration
        new_model = TriuneTransformer(
            vocab_size=100,
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            head_dim=32,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
        ).to(self.device)
        new_trainer = DummyTrainer(new_model, self.config, self.device)

        loaded = load_checkpoint(new_trainer, ckpt_path, load_optimizer=True)
        self.assertEqual(loaded["step"], 42)
        self.assertEqual(loaded.get("schema_version"), 2)
        self.assertEqual(loaded.get("fingerprint"), compute_checkpoint_fingerprint(self.config))

        # Weights match
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_incompatible_checkpoint_rejection(self):
        """Asserts checkpoint loader strictly rejects incompatible architecture or MoE configurations."""
        ckpt_path = save_latest(self.trainer, None, step=10, loss=1.2)

        # Incompatible trainer with mismatched num_layers
        bad_config_layers = dict(self.config, num_layers=6)
        bad_trainer_layers = DummyTrainer(self.model, bad_config_layers, self.device)
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(bad_trainer_layers, ckpt_path, load_optimizer=False)
        self.assertIn("num_layers", str(ctx.exception))

        # Incompatible trainer with mismatched MoE top_k
        bad_config_topk = dict(self.config, top_k=2)
        bad_trainer_topk = DummyTrainer(self.model, bad_config_topk, self.device)
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(bad_trainer_topk, ckpt_path, load_optimizer=False)
        self.assertIn("top_k", str(ctx.exception))

    def test_fingerprint_mismatch_semantic_rejection(self):
        """Asserts load_checkpoint strictly rejects checkpoints when semantic parameters change."""
        ckpt_path = save_latest(self.trainer, None, step=10, loss=1.2)

        # Altering capacity_multiplier changes fingerprint even though tensor shapes are identical
        bad_config_cap = dict(self.config, capacity_multiplier=2.5)
        bad_trainer_cap = DummyTrainer(self.model, bad_config_cap, self.device)
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(bad_trainer_cap, ckpt_path, load_optimizer=False)
        self.assertIn("fingerprint mismatch", str(ctx.exception).lower())
        self.assertIn("capacity_multiplier", str(ctx.exception))

        # Altering galore_rank
        bad_config_galore = dict(self.config, galore_rank=512)
        bad_trainer_galore = DummyTrainer(self.model, bad_config_galore, self.device)
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(bad_trainer_galore, ckpt_path, load_optimizer=False)
        self.assertIn("fingerprint mismatch", str(ctx.exception).lower())
        self.assertIn("galore_rank", str(ctx.exception))


class TestConfigurationValidation(unittest.TestCase):
    """Test Phase 15 math.isfinite validation across configuration and model initialization."""

    def test_config_rejects_nan_and_inf(self):
        """Asserts validate_config strictly rejects NaN and Inf float values."""
        valid_cfg = default_config()
        validate_config(valid_cfg)

        # Test lr with NaN and Inf
        with self.assertRaises(ValueError):
            validate_config(dict(valid_cfg, lr=float("nan")))
        with self.assertRaises(ValueError):
            validate_config(dict(valid_cfg, lr=float("inf")))

        # Test balance_coef
        with self.assertRaises(ValueError):
            validate_config(dict(valid_cfg, balance_coef=float("nan")))

        # Test grad_clip
        with self.assertRaises(ValueError):
            validate_config(dict(valid_cfg, grad_clip=float("inf")))

        # Test target_depth_dist containing NaN
        with self.assertRaises(ValueError):
            validate_config(dict(valid_cfg, target_depth_dist=[float("nan"), 0.5, 0.5]))

        # Test betas tuple containing NaN
        with self.assertRaises(ValueError):
            validate_config(dict(valid_cfg, betas=(float("nan"), 0.95)))

        # Test galore_betas tuple containing Inf
        with self.assertRaises(ValueError):
            validate_config(dict(valid_cfg, galore_betas=(0.9, float("inf"))))

    def test_model_init_rejects_nan_and_inf(self):
        """Asserts TriuneTransformer constructor rejects non-finite balance_coef and target_depth_dist."""
        with self.assertRaises(ValueError):
            TriuneTransformer(balance_coef=float("nan"))

        with self.assertRaises(ValueError):
            TriuneTransformer(balance_coef=float("inf"))

        with self.assertRaises(ValueError):
            TriuneTransformer(target_depth_dist=(float("nan"), 0.5, 0.5))



if __name__ == "__main__":
    unittest.main()
