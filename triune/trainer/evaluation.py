"""Evaluation helpers operating exclusively on a Trainer instance."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


@torch.no_grad()
def run_eval(trainer, force_depth: int | None = None) -> float:
    was_training = trainer.model.training
    trainer.model.eval()
    losses = []
    for xb, yb in trainer.eval_batches:
        xb, yb = xb.to(trainer.device), yb.to(trainer.device)
        with trainer.bf16_autocast():
            logits, _ = trainer.model(xb, force_depth=force_depth)
            loss = nn.CrossEntropyLoss(ignore_index=trainer.pad_token_id)(
                logits.contiguous().view(-1, trainer.vocab_size), yb.contiguous().view(-1)
            )
        losses.append(loss.item())
    if was_training:
        trainer.model.train()
    if not losses:
        raise RuntimeError("Evaluation loader yielded no complete batches")
    return sum(losses) / len(losses)


@torch.no_grad()
def run_sample(trainer, prompt: str | None = None, max_new_tokens: int = 40) -> str:
    prompt = prompt or trainer.sample_prompt
    was_training = trainer.model.training
    trainer.model.eval()
    ids = torch.tensor(trainer.tokenizer.encode(prompt).ids, dtype=torch.long, device=trainer.device).unsqueeze(0)
    prompt_len = ids.size(1)
    for _ in range(max_new_tokens):
        with trainer.bf16_autocast():
            logits, _ = trainer.model(ids, force_depth=2)
        next_id = logits[0, -1].argmax().item()
        if next_id in (trainer.pad_token_id, trainer.sep_token_id):
            break
        ids = torch.cat((ids, torch.tensor([[next_id]], device=trainer.device)), dim=1)
    if was_training:
        trainer.model.train()
    return trainer.tokenizer.decode(ids[0, prompt_len:].tolist())
