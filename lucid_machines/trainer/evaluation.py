"""Evaluation helpers operating exclusively on a Trainer instance."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def forward_causal_lm(model, input_ids: torch.Tensor, force_depth: int | None = None):
    """Calls a Lucid or external (e.g. Hugging Face AutoModelForCausalLM) causal LM.

    Returns ``(logits, route_logits)`` regardless of whether ``model.forward`` accepts
    ``force_depth`` or returns a tuple, tensor, or ``CausalLMOutputWithPast`` object.
    """
    supports_depth = getattr(model, "_lucid_supports_force_depth", None)
    if supports_depth is None:
        base = getattr(model, "module", model)
        try:
            sig = inspect.signature(base.forward)
            supports_depth = "force_depth" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            supports_depth = False
        try:
            model._lucid_supports_force_depth = supports_depth
        except Exception:
            pass

    res = (
        model(input_ids, force_depth=force_depth) if (supports_depth and force_depth is not None) else model(input_ids)
    )
    if hasattr(res, "logits"):
        return res.logits, getattr(res, "route_logits", None)
    if isinstance(res, tuple):
        logits = res[0]
        route = (
            res[1]
            if len(res) > 1 and isinstance(res[1], torch.Tensor) and res[1].dim() == 2 and res[1].size(-1) == 3
            else None
        )
        return logits, route
    return res, None


@torch.no_grad()
def run_eval(trainer, force_depth: int | None = None) -> float:
    was_training = trainer.model.training
    trainer.model.eval()
    losses = []
    for xb, yb in trainer.eval_batches:
        xb, yb = xb.to(trainer.device), yb.to(trainer.device)
        with trainer.bf16_autocast():
            logits, _ = forward_causal_lm(trainer.model, xb, force_depth=force_depth)
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
            logits, _ = forward_causal_lm(trainer.model, ids, force_depth=2)
        next_id = logits[0, -1].argmax().item()
        if next_id in (trainer.pad_token_id, trainer.sep_token_id):
            break
        ids = torch.cat((ids, torch.tensor([[next_id]], device=trainer.device)), dim=1)
    if was_training:
        trainer.model.train()
    return trainer.tokenizer.decode(ids[0, prompt_len:].tolist())
