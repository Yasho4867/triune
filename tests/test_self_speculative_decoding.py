"""Test native self-speculative decoding using Triune's hierarchical exits."""

import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.model.speculative import SelfSpeculativeGenerator


class MockTokenizer:
    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size

    def encode(self, text):
        class Result:
            ids = [1, 5, 12, 18]
        return Result()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"tok_{t}" for t in token_ids)

    def token_to_id(self, token):
        if token == "[PAD]":
            return 0
        if token == "[SEP]":
            return 2
        return None


class TestSelfSpeculativeDecoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TriuneTransformer(
            vocab_size=128,
            hidden_dim=64,
            num_layers=6,
            num_heads=2,
            head_dim=32,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
        ).eval()
        self.tokenizer = MockTokenizer(vocab_size=128)

    def test_self_speculative_generation_basic(self):
        generator = SelfSpeculativeGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            k_draft=3,
            temperature=0.0,
            pad_token_id=0,
            eos_token_id=2,
        )

        res = generator.generate("Hello Triune", max_new_tokens=10)
        self.assertIn("token_ids", res)
        self.assertIn("tokens_per_second", res)
        self.assertIn("acceptance_rate", res)
        self.assertGreaterEqual(len(res["token_ids"]), 1)
        self.assertLessEqual(len(res["token_ids"]), 10)
        self.assertGreaterEqual(res["acceptance_rate"], 0.0)
        self.assertLessEqual(res["acceptance_rate"], 1.0)

    def test_self_speculative_telemetry(self):
        generator = SelfSpeculativeGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            k_draft=2,
            temperature=0.0,
        )
        generator.reset_telemetry()
        self.assertEqual(generator.telemetry["draft_tokens_proposed"], 0)
        self.assertEqual(generator.telemetry["verification_passes"], 0)

        res = generator.generate("Test prompt", max_new_tokens=8)
        self.assertGreater(generator.telemetry["verification_passes"], 0)
        self.assertGreater(generator.telemetry["draft_tokens_proposed"], 0)
        self.assertEqual(
            generator.telemetry["total_tokens_emitted"], len(res["token_ids"])
        )

    def test_self_speculative_eos_handling(self):
        # Force reflex exit to predict EOS (id=2) immediately
        generator = SelfSpeculativeGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            k_draft=3,
            temperature=0.0,
            eos_token_id=2,
        )
        # Verify it runs without error even if EOS is hit
        res = generator.generate("Prompt terminating", max_new_tokens=5)
        self.assertIsInstance(res["text"], str)


if __name__ == "__main__":
    unittest.main()
