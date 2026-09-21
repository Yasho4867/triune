"""Unit tests for Triune Universal Early Exit Adapters and Dynamic Confidence Decoding."""

import unittest
import torch
import torch.nn as nn

from triune.configs import build_config
from triune.model import (
    TriuneTransformer,
    attach_early_exits,
    generate_adaptive,
    WeightTiedExitProjection,
)


class MockHFCausalLM(nn.Module):
    """Simulates a standard Hugging Face causal LM (like Llama or Mistral)."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.model = nn.Module()
        self.model.embed = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x = self.model.embed(input_ids)
        for layer in self.model.layers:
            x = x + layer(x)
        logits = self.lm_head(x)
        return logits


class TestEarlyExitAdapters(unittest.TestCase):
    def test_weight_tied_exit_projection(self):
        hidden_dim = 256
        vocab_size = 1000
        base_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        proj = WeightTiedExitProjection(hidden_dim, base_head=base_head, rank=32)

        x = torch.randn(2, 8, hidden_dim)
        logits = proj(x)
        self.assertEqual(logits.shape, (2, 8, vocab_size))

        # Check parameter count: down_proj (256*32) + up_proj (32*256) + norm (256) = 16640 params
        param_count = sum(p.numel() for p in proj.adapter_parameters() if p.requires_grad)
        self.assertEqual(param_count, 16640)
        self.assertLess(param_count, 50000)  # Under 50k parameters

    def test_attach_early_exits_to_hf_model(self):
        hf_model = MockHFCausalLM(vocab_size=500, hidden_dim=128, num_layers=8)
        retrofitted = attach_early_exits(hf_model, reflex_layer=2, limbic_layer=5, rank=16)

        self.assertTrue(hasattr(retrofitted, "reflex_head"))
        self.assertTrue(hasattr(retrofitted, "limbic_head"))
        self.assertEqual(retrofitted.reflex_layer_idx, 2)
        self.assertEqual(retrofitted.limbic_layer_idx, 5)

        # Run generate_adaptive
        input_ids = torch.tensor([[10, 20, 30]])
        res = retrofitted.generate_adaptive(
            input_ids=input_ids,
            confidence_threshold=0.80,
            max_new_tokens=10,
            temperature=0.7,
        )

        self.assertIn("output_ids", res)
        self.assertIn("exit_counts", res)
        self.assertIn("flops_saved_pct", res)
        self.assertIn("effective_speedup", res)
        self.assertEqual(res["output_ids"].size(1), 13)  # 3 prompt + 10 generated

    def test_attach_early_exits_to_triune_transformer(self):
        config = build_config({
            "vocab_size": 500,
            "hidden_dim": 256,
            "num_layers": 18,
            "num_heads": 2,
            "num_experts": 2,
            "router_prefix_layers": 3,
            "reflex_exit_layer": 5,
            "limbic_exit_layer": 13,
            "use_fp4": False,
            "use_fp8": False,
        })
        model = TriuneTransformer(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            num_experts=config["num_experts"],
            router_prefix_layers=config["router_prefix_layers"],
            reflex_exit_layer=config["reflex_exit_layer"],
            limbic_exit_layer=config["limbic_exit_layer"],
            use_fp4=False,
            use_fp8=False,
        )

        retrofitted = attach_early_exits(model, reflex_layer=5, limbic_layer=13, rank=16)
        input_ids = torch.tensor([[5, 12, 45]])
        res = retrofitted.generate_adaptive(
            input_ids=input_ids,
            confidence_threshold=0.75,
            max_new_tokens=6,
        )
        self.assertEqual(res["output_ids"].size(1), 9)
        self.assertGreaterEqual(res["effective_speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
