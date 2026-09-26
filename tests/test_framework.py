"""Small CPU smoke tests for the callable training framework."""

from __future__ import annotations

import torch
import torch.nn as nn
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from triune.configs import build_config
from triune.trainer import NullLogger, Trainer


class _Tokenizer:
    def get_vocab_size(self):
        return 11

    @staticmethod
    def token_to_id(token):
        return {"[PAD]": 0, "[SEP]": 1}.get(token)

    @staticmethod
    def encode(_text):
        return type("Encoding", (), {"ids": [2, 3]})()

    @staticmethod
    def decode(_ids):
        return ""


class _TinyTriune(nn.Module):
    def __init__(self, vocab_size=11, hidden_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.router = nn.Linear(hidden_dim, 3)

    def forward(self, input_ids, force_depth=None):
        hidden = self.embedding(input_ids)
        return self.head(hidden), self.router(hidden.mean(dim=1))

    def forward_all_exits(self, input_ids, update_stats=False):
        logits, route_logits = self(input_ids)
        return logits, logits, logits, route_logits


class FrameworkTest(unittest.TestCase):
    def test_one_cpu_training_step_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_training_and_resume(Path(directory))

    def _run_training_and_resume(self, checkpoint_dir: Path):
        config = build_config({
            "vocab_size": 11,
            "total_steps": 1,
            "warmup_steps": 0,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "seq_len": 3,
            "eval_batches": 1,
            "eval_every": 2,
            "log_every": 1,
            "save_every": 2,
            "checkpoint_dir": str(checkpoint_dir),
        })
        x = torch.tensor([[2, 3, 4], [5, 6, 7]])
        y = torch.tensor([[3, 4, 5], [6, 7, 8]])
        tokenizer = _Tokenizer()
        model = _TinyTriune()
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=config["lr"]),
            train_loader=[(x, y)],
            eval_loader=[(x, y)],
            tokenizer=tokenizer,
            config=config,
            device="cpu",
            logger=NullLogger(),
        )
        result = trainer.fit()
        self.assertEqual(result["step"], 1)

        restored = _TinyTriune()
        resumed = Trainer(
            model=restored,
            optimizer=torch.optim.AdamW(restored.parameters(), lr=config["lr"]),
            train_loader=[(x, y)],
            eval_loader=[(x, y)],
            tokenizer=tokenizer,
            config=config,
            device="cpu",
            logger=NullLogger(),
        )
        resumed.resume(checkpoint_dir / "latest.pt")
        self.assertEqual(resumed.engine.step, 1)

        # After resume, verify weights match
        for (n1, p1), (n2, p2) in zip(trainer.model.named_parameters(), resumed.model.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Weight mismatch after resume: {n1}")

        # Verify optimizer state was restored
        for (k1, v1), (k2, v2) in zip(
            trainer.optimizer.state_dict()['state'].items(),
            resumed.optimizer.state_dict()['state'].items()
        ):
            if isinstance(v1, dict):
                for sk in v1:
                    if isinstance(v1[sk], torch.Tensor):
                        torch.testing.assert_close(v1[sk], v2[sk], msg=f"Optimizer state mismatch: param {k1}, key {sk}")

    def test_fp8_linear_backward_dtype_mix(self):
        from triune.model.fp8 import FP8Linear
        layer = FP8Linear(16, 32, bias=True, dtype=torch.bfloat16)
        x = torch.randn(2, 4, 16, dtype=torch.bfloat16, requires_grad=True)
        out = layer(x)
        # Simulate cross entropy float32 loss
        loss = out.float().sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        self.assertEqual(layer.weight.grad.dtype, torch.bfloat16)

    def test_triune_model_with_fp8(self):
        from triune.model import build_model
        from triune.optim.factory import build_optimizer
        config = build_config({
            "vocab_size": 32,
            "hidden_dim": 64,
            "num_heads": 2,
            "head_dim": 32,
            "num_layers": 8,
            "num_experts": 2,
            "router_prefix_layers": 2,
            "reflex_exit_layer": 3,
            "limbic_exit_layer": 5,
            "use_fp8": True,
            "total_steps": 1,
            "checkpoint_dir": "test_ckpt"
        })
        model = build_model(config).to(dtype=torch.bfloat16)
        optimizer = build_optimizer(model, config)
        input_ids = torch.randint(0, 32, (2, 8))
        logits, route_logits = model(input_ids)
        loss = logits.float().sum() + route_logits.float().sum()
        loss.backward()
        optimizer.step()
        self.assertEqual(logits.shape, (2, 8, 32))


if __name__ == "__main__":
    unittest.main()
