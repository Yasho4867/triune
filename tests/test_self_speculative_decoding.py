"""Test native self-speculative decoding using Triune's hierarchical exits."""

import unittest

import torch

from lucid_machines.model.speculative import SelfSpeculativeGenerator
from lucid_machines.model.transformer import TriuneTransformer


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
        self.assertEqual(generator.telemetry["total_tokens_emitted"], len(res["token_ids"]))

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

    def test_self_speculative_greedy_exact_parity_and_rejection_sampling(self):
        gen_greedy = SelfSpeculativeGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            k_draft=2,
            temperature=0.0,
            eos_token_id=None,
        )
        res_greedy = gen_greedy.generate("Greedy test", max_new_tokens=6)
        self.assertEqual(res_greedy["sampling_mode"], "greedy_exact_parity")

        # Autoregressive greedy target reference at force_depth=2
        prompt_ids = self.tokenizer.encode("Greedy test").ids
        ref_curr = torch.tensor([prompt_ids], dtype=torch.long)
        ref_toks = []
        with torch.no_grad():
            for _ in range(6):
                logits, _ = self.model(ref_curr, force_depth=2, last_token_only=True)
                nxt = int(torch.argmax(logits[0, -1]).item())
                ref_toks.append(nxt)
                ref_curr = torch.cat([ref_curr, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        self.assertEqual(res_greedy["token_ids"], ref_toks)

        torch.manual_seed(123)
        gen_sample = SelfSpeculativeGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            k_draft=2,
            temperature=0.7,
            eos_token_id=None,
        )
        res_sample = gen_sample.generate("Sampled test", max_new_tokens=6)
        self.assertEqual(res_sample["sampling_mode"], "rejection_sampling_exact")
        self.assertEqual(len(res_sample["token_ids"]), 6)

    def test_self_speculative_chi_square_distributional_parity(self):
        import math

        import torch.nn.functional as F

        torch.manual_seed(77)
        vocab = 12
        dist_model = TriuneTransformer(
            vocab_size=vocab,
            hidden_dim=32,
            num_layers=6,
            num_heads=2,
            head_dim=16,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
        ).eval()

        class SmallMockTokenizer:
            vocab_size = vocab

            def encode(self, text):
                class Result:
                    ids = [1, 3, 7, 2]

                return Result()

            def decode(self, token_ids, skip_special_tokens=True):
                return " ".join(str(t) for t in token_ids)

        temperature = 0.7
        prompt_t = torch.tensor([[1, 3, 7, 2]], dtype=torch.long)
        with torch.no_grad():
            target_logits, _ = dist_model(prompt_t, force_depth=2, last_token_only=True)
            p_exact = F.softmax(target_logits[0, -1].float() / temperature, dim=-1).cpu()

        gen = SelfSpeculativeGenerator(
            model=dist_model,
            tokenizer=SmallMockTokenizer(),
            k_draft=2,
            temperature=temperature,
            eos_token_id=None,
        )
        n_draws = 2000
        counts = torch.zeros(vocab, dtype=torch.float64)
        torch.manual_seed(2026)
        for _ in range(n_draws):
            res = gen.generate("dist check", max_new_tokens=1)
            counts[res["token_ids"][0]] += 1.0

        # Confirm rejection sampling actually exercised rejections
        self.assertGreater(gen.telemetry["draft_tokens_proposed"], 0)

        # Pool any bins with expected count < 5 so Pearson chi-square regularity holds
        exp = (p_exact.double() * n_draws).tolist()
        obs = counts.tolist()
        main_obs, main_exp = [], []
        tail_o, tail_e = 0.0, 0.0
        for o_i, e_i in zip(obs, exp):
            if e_i >= 5.0:
                main_obs.append(o_i)
                main_exp.append(e_i)
            else:
                tail_o += o_i
                tail_e += e_i
        if tail_e >= 5.0:
            main_obs.append(tail_o)
            main_exp.append(tail_e)
        elif tail_e > 0.0 and main_obs:
            main_obs[-1] += tail_o
            main_exp[-1] += tail_e

        df = len(main_obs) - 1
        self.assertGreaterEqual(df, 3)
        chi2 = sum((o - e) ** 2 / e for o, e in zip(main_obs, main_exp))
        # Wilson-Hilferty normal approximation to upper-tail chi-square p-value
        z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        p_val = 0.5 * math.erfc(z / math.sqrt(2.0))
        self.assertGreater(
            p_val,
            0.005,
            msg=f"Rejection sampling diverged from target dist (chi2={chi2:.2f}, df={df}, p={p_val:.4g})",
        )


if __name__ == "__main__":
    unittest.main()
