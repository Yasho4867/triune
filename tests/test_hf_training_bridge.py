"""Tests verifying Hugging Face AutoModelForCausalLM / external CausalLM compatibility in Trainer & TrainingEngine."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.train import _HFTokenizerAdapter
from triune.configs import build_config
from triune.optim import CentroidSteerOptimizer
from triune.trainer.evaluation import forward_causal_lm, run_eval, run_sample
from triune.trainer.trainer import Trainer


class _MockHFTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 1

    def __len__(self):
        return 64

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(c) % 60) for c in text] or [3]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def convert_tokens_to_ids(self, tok):
        return None


class _MockHFCausalLM(nn.Module):
    """Mimics HF AutoModelForCausalLM: rejects force_depth and returns CausalLMOutput(logits=...)."""

    def __init__(self, vocab_size: int = 80, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        h = torch.tanh(self.proj(self.embed(input_ids)))
        return SimpleNamespace(logits=self.lm_head(h))


class TestHFTrainingBridge(unittest.TestCase):
    def test_hf_tokenizer_adapter_and_engine_step_eval_sample(self):
        # Simulate padded vocab_size (80 > len(tok)=64) common in Qwen checkpoints
        tok = _HFTokenizerAdapter(_MockHFTokenizer(), model_vocab_size=80)
        self.assertEqual(tok.get_vocab_size(), 80)
        self.assertEqual(tok.token_to_id("[PAD]"), 0)
        self.assertEqual(tok.token_to_id("[SEP]"), 2)

        model = _MockHFCausalLM(vocab_size=80, hidden_dim=32)
        logits, route = forward_causal_lm(model, torch.randint(0, 64, (2, 8)), force_depth=2)
        self.assertEqual(logits.shape, (2, 8, 80))
        self.assertIsNone(route)

        batches = [(torch.randint(0, 64, (2, 16)), torch.randint(0, 64, (2, 16))) for _ in range(2)]
        cfg = build_config(
            {
                "vocab_size": 80,
                "seq_len": 16,
                "batch_size": 2,
                "grad_accum_steps": 1,
                "eval_batches": 2,
                "depth_mode": "dynamic",  # Verify automatic fallback to cortex for non-Triune HF models
                "total_steps": 2,
                "lr": 1e-3,
            }
        )
        opt = CentroidSteerOptimizer(model, lr=1e-3, use_muon=True, steer_scale=0.0)
        trainer = Trainer(
            model=model,
            optimizer=opt,
            train_loader=batches,
            eval_loader=batches,
            tokenizer=tok,
            config=cfg,
            device=torch.device("cpu"),
        )
        stats = trainer.engine.step_once(advance_step=True)
        self.assertGreater(stats["lm_loss"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(stats["loss"])))

        eval_loss = run_eval(trainer, force_depth=2)
        self.assertGreater(eval_loss, 0.0)
        text = run_sample(trainer, prompt="hello", max_new_tokens=4)
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
