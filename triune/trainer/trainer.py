"""Callable high-level training interface."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from triune.data import CyclingDataLoader
from triune.recipes import bf16_autocast

from .checkpoint import load_checkpoint, save_best, save_latest
from .engine import TrainingEngine
from .evaluation import run_eval, run_sample
from .logger import NullLogger
from .scheduler import lr_schedule, set_optimizer_lr


class Trainer:
    def __init__(
        self,
        *,
        model,
        optimizer,
        train_loader,
        eval_loader,
        tokenizer,
        config: dict,
        device,
        precision_context=None,
        logger=None,
        sample_prompt: str = "The history of",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device)
        self.logger = logger or NullLogger()
        self.sample_prompt = sample_prompt
        self._precision_context = precision_context or (lambda: bf16_autocast(self.device.type))

        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.sep_token_id = tokenizer.token_to_id("[SEP]")
        if self.pad_token_id is None or self.sep_token_id is None:
            raise ValueError("Tokenizer must define [PAD] and [SEP]")
        if self.config["vocab_size"] != self.vocab_size:
            raise ValueError("config vocab_size must match tokenizer vocabulary size")

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction="none")
        self.router_loss_fn = nn.CrossEntropyLoss()
        self.grad_accum = config["grad_accum_steps"]
        self.z_loss_coef = 1e-3
        self._train_batches = CyclingDataLoader(train_loader)
        print(f"⏳ Materializing {self.config.get('eval_batches', 1)} evaluation batch(es)...", flush=True)
        self.eval_batches = self._materialize_eval_batches()
        print(f"✅ Evaluation batch(es) ready.", flush=True)
        self.engine = TrainingEngine(self)

    def _materialize_eval_batches(self):
        iterator = iter(self.eval_loader)
        batches = []
        for _ in range(self.config["eval_batches"]):
            try:
                batches.append(next(iterator))
            except StopIteration as error:
                raise RuntimeError("Evaluation stream yielded fewer complete batches than requested") from error
        return batches

    def fit(self):
        return self.engine.train()

    def next_batch(self):
        return self._train_batches.next()

    def model_autocast(self):
        return self._precision_context()

    def bf16_autocast(self):
        return bf16_autocast(self.device.type)

    def run_eval(self, force_depth=None):
        return run_eval(self, force_depth)

    def run_sample(self, prompt: str | None = None, max_new_tokens: int = 40):
        return run_sample(self, prompt, max_new_tokens)

    def save_latest(self, step: int, loss: float):
        return save_latest(self, self.engine, step, loss)

    def save_best(self, step: int, loss: float):
        return save_best(self, self.engine, step, loss)

    def resume(self, path: str | Path, *, weights_only: bool = False) -> None:
        checkpoint = load_checkpoint(self, path, load_optimizer=not weights_only)
        self.engine.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        if "depth_usage_ema" in checkpoint:
            self.engine.depth_usage_ema = checkpoint["depth_usage_ema"].to(self.device)
        self.engine.step = 0 if weights_only else checkpoint["step"]

    def compile(self) -> None:
        self.model.forward = torch.compile(self.model.forward, fullgraph=False, dynamic=True)

    @staticmethod
    def lr_schedule(config: dict, step: int) -> float:
        return lr_schedule(config, step)

    @staticmethod
    def set_optimizer_lr(optimizer, lr: float) -> None:
        set_optimizer_lr(optimizer, lr)
