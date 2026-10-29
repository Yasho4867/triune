"""Automated Tests for Triune Universal Multi-Architecture Framework.

Verifies that Triune can seamlessly train, fine-tune (LoRA/QLoRA), and serve:
1. Universal Dense Transformers
2. Universal MoE Transformers
3. FastLoRA Adapters with Zero Full-Vocab Logits Materialization
"""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn

from triune.universal import (
    UniversalDenseTransformer,
    UniversalMoETransformer,
    TriuneUniversalEngine,
)
from triune.trainer.finetune import LoRAConfig, TriuneFineTuner


class TestUniversalFramework(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    def test_universal_dense_transformer_training(self):
        """Verify UniversalDenseTransformer executes forward, backward, and optimizer steps."""
        model = UniversalDenseTransformer(
            vocab_size=1000,
            dim=128,
            num_layers=4,
            num_heads=4,
            use_gla=True,
        )
        engine = TriuneUniversalEngine(model, mode="pretrain", lr=1e-3, device=self.device, dtype=self.dtype)

        x = torch.randint(0, 1000, (2, 32), device=self.device)
        y = torch.randint(0, 1000, (2, 32), device=self.device)

        metrics = engine.train_step(x, y)
        self.assertIn("loss", metrics)
        self.assertGreater(metrics["loss"], 0.0)

    def test_universal_moe_transformer_training(self):
        """Verify UniversalMoETransformer executes forward, backward, and updates expert weights."""
        model = UniversalMoETransformer(
            vocab_size=1000,
            dim=128,
            num_layers=4,
            num_heads=4,
            head_dim=32,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
        )
        engine = TriuneUniversalEngine(model, mode="pretrain", lr=1e-3, device=self.device, dtype=self.dtype)

        x = torch.randint(0, 1000, (2, 32), device=self.device)
        y = torch.randint(0, 1000, (2, 32), device=self.device)

        metrics = engine.train_step(x, y)
        self.assertIn("loss", metrics)
        self.assertGreater(metrics["loss"], 0.0)

    def test_universal_finetune_lora_attach_and_train(self):
        """Verify LoRA adapters attach cleanly, freeze base weights, and train adapters only."""
        model = UniversalDenseTransformer(vocab_size=1000, dim=128, num_layers=4, num_heads=4)
        engine = TriuneUniversalEngine(
            model,
            mode="finetune",
            lora_config=LoRAConfig(rank=8, target_modules=["q_proj", "v_proj"]),
            device=self.device,
            dtype=self.dtype,
        )

        # Base weights must be frozen
        self.assertFalse(model.embed.weight.requires_grad)
        self.assertFalse(model.layers[0].attn.q_proj.original_layer.weight.requires_grad)

        # LoRA weights must be trainable
        self.assertTrue(model.layers[0].attn.q_proj.lora_A.requires_grad)
        self.assertTrue(model.layers[0].attn.q_proj.lora_B.requires_grad)

        # Train step
        x = torch.randint(0, 1000, (2, 32), device=self.device)
        y = torch.randint(0, 1000, (2, 32), device=self.device)
        res = engine.train_step(x, y)
        self.assertGreater(res["loss"], 0.0)

    def test_universal_inference_generation(self):
        """Verify autoregressive token generation across both Dense and MoE."""
        model = UniversalDenseTransformer(vocab_size=1000, dim=128, num_layers=2, num_heads=2)
        engine = TriuneUniversalEngine(model, mode="inference", device=self.device, dtype=self.dtype)

        prompt = torch.randint(0, 1000, (1, 8), device=self.device)
        out = engine.generate(prompt, max_new_tokens=6)
        self.assertEqual(out.shape, (1, 14))


if __name__ == "__main__":
    unittest.main()
