"""Regression tests for large-batch chunked paths in the shared trainer."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import unittest

from triune.trainer.engine import TrainingEngine


class _ChunkTestModel(nn.Module):
    vocab_size = 16

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(self.vocab_size, 8)
        self.exits = nn.ModuleList(nn.Linear(8, self.vocab_size) for _ in range(3))
        self.router = nn.Linear(8, 3)
        self.last_depth_choice = None
        self.last_balance_loss = None

    def _outputs(self, input_ids):
        hidden = self.embedding(input_ids)
        logits = [head(hidden) for head in self.exits]
        route_logits = self.router(hidden.mean(dim=1))
        probabilities = route_logits.softmax(dim=-1)
        self.last_balance_loss = (probabilities.mean(dim=0) - torch.tensor(
            [0.5, 0.3, 0.2], device=input_ids.device
        )).square().mean()
        self.last_depth_choice = route_logits.argmax(dim=-1)
        return logits, route_logits

    def forward_all_exits(self, input_ids, update_stats=True):
        logits, route_logits = self._outputs(input_ids)
        return logits[0], logits[1], logits[2], route_logits

    def forward(self, input_ids, force_depth=None):
        logits, route_logits = self._outputs(input_ids)
        if force_depth is None:
            selected = logits[2]
        else:
            selected = logits[force_depth]
            self.last_depth_choice = torch.full(
                (input_ids.size(0),), force_depth, device=input_ids.device, dtype=torch.long
            )
        return selected, route_logits


class _ChunkTestTrainer:
    def __init__(self, depth_mode: str):
        self.model = _ChunkTestModel()
        self.device = torch.device("cpu")
        self.vocab_size = self.model.vocab_size
        self.pad_token_id = 0
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        self.router_loss_fn = nn.CrossEntropyLoss()
        self.grad_accum = 1
        self.z_loss_coef = 1e-3
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.config = {
            "target_depth_dist": [0.5, 0.3, 0.2],
            "exploration": "none",
            "exploration_steps": 0,
            "usage_ema_decay": 0.99,
            "depth_mode": depth_mode,
            "balance_coef": 0.3,
            "grad_accum_steps": 1,
            "grad_clip": 1.0,
            "log_every": 100,
            "total_steps": 10,
            "lr": 1e-3,
        }
        self._batch = torch.randint(1, self.vocab_size, (65, 65))

    def next_batch(self):
        return self._batch[:, :-1], self._batch[:, 1:]

    def model_autocast(self):
        return nullcontext()

    @staticmethod
    def lr_schedule(config, step):
        return config["lr"]

    @staticmethod
    def set_optimizer_lr(optimizer, lr, expert_lr=None):
        for group in optimizer.param_groups:
            group["lr"] = lr


class TestTrainingEngineChunking(unittest.TestCase):
    def test_large_microbatches_complete_all_depth_modes_with_component_metrics(self):
        # At 64 tokens/sequence the 2,048-token modes split above 32 rows and
        # Cortex splits above 64 rows, so B=65 exercises every chunked branch.
        for mode in ("cortex", "joint", "dynamic"):
            with self.subTest(depth_mode=mode):
                trainer = _ChunkTestTrainer(mode)
                engine = TrainingEngine(trainer)
                engine.step = 1  # avoid logging through unrelated trainer logger hooks

                result = engine.step_once()

                for metric in ("loss", "lm_loss", "router_loss", "balance_loss"):
                    self.assertTrue(torch.isfinite(torch.tensor(result[metric])), (mode, metric, result))
                if mode != "cortex":
                    self.assertGreater(result["router_loss"], 0, (mode, result))
                    self.assertGreaterEqual(result["balance_loss"], 0, (mode, result))
