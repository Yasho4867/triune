"""Evaluation helpers operating exclusively on a Trainer instance."""

from __future__ import annotations

import importlib.util
import inspect
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    from lucid_machines.kernels.fast_cross_entropy import fast_cross_entropy

    was_training = trainer.model.training
    trainer.model.eval()
    raw_pad = getattr(trainer, "pad_token_id", -100)
    pad_id = -100 if raw_pad is None else int(raw_pad)
    losses = []
    for xb, yb in trainer.eval_batches:
        xb, yb = xb.to(trainer.device), yb.to(trainer.device)
        est_logits_bytes = int(xb.numel()) * int(trainer.vocab_size) * 4
        max_slice = max(1, min(xb.size(0), 4096 // max(1, xb.size(1))))
        if xb.device.type == "cpu" and est_logits_bytes > 512 * 1024 * 1024 and xb.size(0) > max_slice:
            batch_loss_sum = 0.0
            batch_valid = 0
            for sub_x, sub_y in zip(xb.split(max_slice), yb.split(max_slice)):
                sub_y_flat = sub_y.contiguous().view(-1)
                n_valid = int((sub_y_flat != pad_id).sum().item())
                if n_valid == 0:
                    continue
                with trainer.bf16_autocast():
                    sub_logits, _ = forward_causal_lm(trainer.model, sub_x, force_depth=force_depth)
                    sub_loss = fast_cross_entropy(
                        sub_logits.contiguous().view(-1, trainer.vocab_size),
                        sub_y_flat,
                        ignore_index=pad_id,
                    )
                batch_loss_sum += float(sub_loss.item()) * n_valid
                batch_valid += n_valid
            losses.append(batch_loss_sum / max(1, batch_valid))
        else:
            with trainer.bf16_autocast():
                logits, _ = forward_causal_lm(trainer.model, xb, force_depth=force_depth)
                loss = fast_cross_entropy(
                    logits.contiguous().view(-1, trainer.vocab_size),
                    yb.contiguous().view(-1),
                    ignore_index=pad_id,
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
    enc = trainer.tokenizer.encode(prompt)
    enc_ids = list(enc.ids if hasattr(enc, "ids") else enc)
    if not enc_ids:
        raise ValueError("Evaluation sample prompt produced zero tokens")
    was_training = trainer.model.training
    trainer.model.eval()
    ids = torch.tensor(enc_ids, dtype=torch.long, device=trainer.device).unsqueeze(0)
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


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    enc = tokenizer.encode(text)
    ids = enc.ids if hasattr(enc, "ids") else enc
    return [int(i) for i in ids]


@torch.no_grad()
def score_completion_loglikelihood(
    model: nn.Module,
    tokenizer: Any,
    prefix: str,
    continuation: str,
    *,
    device: str | torch.device,
    force_depth: int | None = None,
) -> tuple[float, int]:
    """Compute exact causal log-likelihood of ``continuation`` given ``prefix``."""
    prefix_ids = _encode_ids(tokenizer, prefix)
    cont_ids = _encode_ids(tokenizer, continuation)
    if not cont_ids:
        raise ValueError(f"Continuation produced zero tokens: {continuation!r}")
    if not prefix_ids:
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is None and hasattr(tokenizer, "token_to_id"):
            bos = tokenizer.token_to_id("[BOS]")
            if bos is None:
                bos = tokenizer.token_to_id("[PAD]")
        prefix_ids = [int(bos if bos is not None else 0)]

    max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if max_len is None:
        max_len = getattr(model, "rope_max_seq_len", 2048)
    max_len = max(16, int(max_len))
    if len(prefix_ids) + len(cont_ids) > max_len:
        keep_prefix = max(1, max_len - len(cont_ids))
        prefix_ids = prefix_ids[-keep_prefix:]

    dev = torch.device(device)
    full_ids = prefix_ids + cont_ids
    inp = torch.tensor(full_ids, dtype=torch.long, device=dev).unsqueeze(0)

    was_training = model.training
    model.eval()
    try:
        logits, _ = forward_causal_lm(model, inp, force_depth=force_depth)
    finally:
        if was_training:
            model.train()

    start_pos = len(prefix_ids) - 1
    end_pos = start_pos + len(cont_ids)
    target_logits = logits[0, start_pos:end_pos, :].float()
    log_probs = F.log_softmax(target_logits, dim=-1)
    tgt_tensor = torch.tensor(cont_ids, dtype=torch.long, device=logits.device).unsqueeze(-1)
    tok_lps = log_probs.gather(-1, tgt_tensor).squeeze(-1)
    return float(tok_lps.sum().item()), len(cont_ids)


_BUILTIN_ZEROSHOT_PROBES: dict[str, list[dict[str, Any]]] = {
    "arc_probe": [
        {
            "prompt": "Question: Which gas do plants primarily absorb from the atmosphere for photosynthesis?\nAnswer:",
            "choices": [" Carbon dioxide", " Nitrogen", " Helium", " Argon"],
            "answer": 0,
        },
        {
            "prompt": "Question: What is the boiling point of pure water at standard sea-level atmospheric pressure?\nAnswer:",
            "choices": [" 0 degrees Celsius", " 100 degrees Celsius", " 50 degrees Celsius", " 212 degrees Celsius"],
            "answer": 1,
        },
        {
            "prompt": "Question: Which organ in the human body pumps blood through the circulatory system?\nAnswer:",
            "choices": [" Liver", " Kidney", " Heart", " Stomach"],
            "answer": 2,
        },
        {
            "prompt": "Question: Which planet in the Solar System is closest to the Sun?\nAnswer:",
            "choices": [" Venus", " Earth", " Mars", " Mercury"],
            "answer": 3,
        },
    ],
    "hellaswag_probe": [
        {
            "prompt": "A person cracks two eggs into a hot greased skillet and stirs them gently until they set. Next, they",
            "choices": [
                " serve the cooked eggs onto a warm plate.",
                " bury the skillet in snow for three weeks.",
                " pour concrete into the refrigerator drawer.",
                " launch a satellite into geostationary orbit.",
            ],
            "answer": 0,
        },
        {
            "prompt": "Before crossing a busy city intersection, a pedestrian looks both ways and waits for",
            "choices": [
                " the ocean tide to reach low water.",
                " the crosswalk signal to turn to walk.",
                " a volcano to erupt nearby.",
                " winter solstice to arrive.",
            ],
            "answer": 1,
        },
        {
            "prompt": "After finishing a chapter in a paperback novel, the reader places a bookmark between the pages and",
            "choices": [
                " boils the book in saltwater.",
                " hammers nails into the spine.",
                " closes the cover for later.",
                " mails the bookmark to Mars.",
            ],
            "answer": 2,
        },
        {
            "prompt": "A software engineer writes a new unit test and runs pytest in the terminal to verify that",
            "choices": [
                " the office plants need sunlight.",
                " the coffee mug is ceramic.",
                " the sky remains blue outside.",
                " the function passes all assertions.",
            ],
            "answer": 3,
        },
    ],
    "mmlu_probe": [
        {
            "prompt": "Subject: Computer Science. Question: What is the worst-case time complexity of binary search on a sorted array of n elements?\nAnswer:",
            "choices": [" O(log n)", " O(n^2)", " O(2^n)", " O(n!)"],
            "answer": 0,
        },
        {
            "prompt": "Subject: Mathematics. Question: What is the derivative of f(x) = x^3 with respect to x?\nAnswer:",
            "choices": [" x^2", " 3x^2", " 3x", " x^4 / 4"],
            "answer": 1,
        },
        {
            "prompt": "Subject: Physics. Question: According to Newton's second law, force equals mass multiplied by\nAnswer:",
            "choices": [" volume", " density", " acceleration", " temperature"],
            "answer": 2,
        },
        {
            "prompt": "Subject: Chemistry. Question: What is the chemical symbol for gold?\nAnswer:",
            "choices": [" Ag", " Fe", " Pb", " Au"],
            "answer": 3,
        },
    ],
}


@torch.no_grad()
def evaluate_multiple_choice_tasks(
    model: nn.Module,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    device: str | torch.device,
    force_depth: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate genuine multiple-choice log-likelihood accuracy on a list of prompt/choice items."""
    selected = items[: max(1, int(limit))] if limit is not None else list(items)
    if not selected:
        raise ValueError("No multiple-choice evaluation items provided.")

    t0 = time.perf_counter()
    raw_correct = 0
    norm_correct = 0
    baseline_sum = 0.0
    target_lp_sum = 0.0
    evaluated = 0

    for item in selected:
        prompt = str(item.get("prompt") or item.get("question") or item.get("context") or "")
        choices = [str(c) for c in (item.get("choices") or [])]
        if len(choices) < 2:
            raise ValueError("Each multiple-choice evaluation item requires at least 2 choices.")
        raw_ans = item.get("answer", item.get("label", 0))
        if isinstance(raw_ans, str) and raw_ans in choices:
            gold_idx = choices.index(raw_ans)
        else:
            gold_idx = int(raw_ans)
        if not (0 <= gold_idx < len(choices)):
            raise ValueError(f"Answer index {gold_idx} out of bounds for {len(choices)} choices.")

        raw_scores: list[float] = []
        norm_scores: list[float] = []
        for choice in choices:
            lp, tok_len = score_completion_loglikelihood(
                model,
                tokenizer,
                prompt,
                choice,
                device=device,
                force_depth=force_depth,
            )
            raw_scores.append(lp)
            norm_scores.append(lp / max(1, tok_len))

        pred_raw = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
        pred_norm = max(range(len(norm_scores)), key=lambda i: norm_scores[i])
        if pred_raw == gold_idx:
            raw_correct += 1
        if pred_norm == gold_idx:
            norm_correct += 1
        baseline_sum += 1.0 / len(choices)
        target_lp_sum += raw_scores[gold_idx]
        evaluated += 1

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "evaluator": "native_loglikelihood",
        "num_examples": evaluated,
        "accuracy": round(raw_correct / evaluated, 4),
        "acc_norm": round(norm_correct / evaluated, 4),
        "random_baseline": round(baseline_sum / evaluated, 4),
        "mean_target_logprob": round(target_lp_sum / evaluated, 4),
        "latency_ms": round(latency_ms, 2),
    }


@torch.no_grad()
def evaluate_mqar(
    model: nn.Module,
    tokenizer: Any,
    *,
    device: str | torch.device,
    force_depth: int | None = None,
    num_trials: int = 16,
    num_kv_pairs: int = 4,
    distractor_tokens: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate genuine in-context Multi-Query Associative Recall (MQAR) via forward-pass log-likelihood."""
    n_trials = max(1, int(num_trials))
    n_pairs = max(2, int(num_kv_pairs))
    rng = random.Random(int(seed))
    key_pool = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "theta",
        "lambda",
        "omega",
        "sigma",
        "kappa",
        "tau",
    ]
    val_pool = [
        "417",
        "829",
        "193",
        "654",
        "308",
        "742",
        "581",
        "926",
        "265",
        "519",
        "843",
        "670",
    ]
    filler_words = [
        "note",
        "log",
        "record",
        "stream",
        "packet",
        "trace",
        "frame",
        "state",
        "cache",
        "index",
        "buffer",
        "token",
    ]

    t0 = time.perf_counter()
    choice_correct = 0
    exact_top1_correct = 0
    target_lp_sum = 0.0
    dev = torch.device(device)

    for trial_idx in range(n_trials):
        keys = rng.sample(key_pool, k=min(n_pairs, len(key_pool)))
        vals = rng.sample(val_pool, k=len(keys))
        kv_seg = "; ".join(f"{k}={v}" for k, v in zip(keys, vals))
        distractor_len = max(4, int(distractor_tokens) // 2)
        filler = " ".join(rng.choice(filler_words) for _ in range(distractor_len))
        q_idx = trial_idx % len(keys)
        q_key = keys[q_idx]
        target_val = vals[q_idx]
        prefix = f"Memory: {kv_seg}. Context: {filler}. Query: {q_key}="

        scores: list[float] = []
        for cand_val in vals:
            lp, _ = score_completion_loglikelihood(
                model,
                tokenizer,
                prefix,
                cand_val,
                device=dev,
                force_depth=force_depth,
            )
            scores.append(lp)

        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        if pred_idx == q_idx:
            choice_correct += 1
        target_lp_sum += scores[q_idx]

        # Exact next-token top-1 check
        prefix_ids = _encode_ids(tokenizer, prefix)
        target_ids = _encode_ids(tokenizer, target_val)
        if prefix_ids and target_ids:
            inp = torch.tensor(prefix_ids, dtype=torch.long, device=dev).unsqueeze(0)
            was_training = model.training
            model.eval()
            try:
                logits, _ = forward_causal_lm(model, inp, force_depth=force_depth)
            finally:
                if was_training:
                    model.train()
            top1_id = int(logits[0, -1, :].argmax().item())
            if top1_id == target_ids[0]:
                exact_top1_correct += 1

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "evaluator": "native_mqar_loglikelihood",
        "num_trials": n_trials,
        "num_kv_pairs": min(n_pairs, len(key_pool)),
        "choice_accuracy": round(choice_correct / n_trials, 4),
        "exact_top1_accuracy": round(exact_top1_correct / n_trials, 4),
        "random_baseline": round(1.0 / min(n_pairs, len(key_pool)), 4),
        "mean_target_logprob": round(target_lp_sum / n_trials, 4),
        "latency_ms": round(latency_ms, 2),
    }


def _load_mc_items_from_file(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        return parsed["items"]
    raise ValueError(f"Unsupported multiple-choice dataset structure in {path}")


@torch.no_grad()
def run_zero_shot_eval(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str] | str,
    *,
    device: str | torch.device,
    force_depth: int | None = None,
    limit: int | None = None,
    num_trials: int = 16,
    use_lm_eval: bool = True,
) -> dict[str, Any]:
    """Execute real zero-shot multiple-choice, MQAR, local JSONL, or optional lm_eval tasks."""
    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        task_list = [str(t).strip() for t in tasks if str(t).strip()]
    if not task_list:
        raise ValueError("At least one evaluation task must be specified.")

    results: dict[str, Any] = {}
    has_lm_eval = bool(use_lm_eval and importlib.util.find_spec("lm_eval") is not None)

    for task_name in task_list:
        low = task_name.lower()
        task_path = Path(task_name)
        if low == "mqar":
            results["mqar"] = evaluate_mqar(
                model,
                tokenizer,
                device=device,
                force_depth=force_depth,
                num_trials=num_trials,
            )
        elif low == "zeroshot":
            combined: list[dict[str, Any]] = []
            for probe_items in _BUILTIN_ZEROSHOT_PROBES.values():
                combined.extend(probe_items)
            results["zeroshot"] = evaluate_multiple_choice_tasks(
                model,
                tokenizer,
                combined,
                device=device,
                force_depth=force_depth,
                limit=limit,
            )
        elif low in _BUILTIN_ZEROSHOT_PROBES:
            results[low] = evaluate_multiple_choice_tasks(
                model,
                tokenizer,
                _BUILTIN_ZEROSHOT_PROBES[low],
                device=device,
                force_depth=force_depth,
                limit=limit,
            )
        elif task_path.is_file():
            items = _load_mc_items_from_file(task_path)
            results[task_name] = evaluate_multiple_choice_tasks(
                model,
                tokenizer,
                items,
                device=device,
                force_depth=force_depth,
                limit=limit,
            )
        elif has_lm_eval:
            import lm_eval
            from lm_eval.api.model import LM as _BaseLM

            class _TriuneLMEvalAdapter(_BaseLM):
                def __init__(self) -> None:
                    super().__init__()
                    self._dev = torch.device(device)

                @property
                def eot_token_id(self) -> int:
                    sep = tokenizer.token_to_id("[SEP]") if hasattr(tokenizer, "token_to_id") else None
                    return int(sep if sep is not None else getattr(tokenizer, "eos_token_id", 2) or 2)

                @property
                def max_length(self) -> int:
                    return int(getattr(model, "rope_max_seq_len", 2048))

                @property
                def max_gen_toks(self) -> int:
                    return 64

                @property
                def batch_size(self) -> int:
                    return 1

                @property
                def device(self) -> torch.device:
                    return self._dev

                def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
                    return _encode_ids(tokenizer, string)

                def tok_decode(self, tokens: list[int], **kwargs: Any) -> str:
                    return str(tokenizer.decode(tokens))

                def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
                    out: list[tuple[float, bool]] = []
                    for req in requests:
                        ctx, cont = req.args
                        lp, _ = score_completion_loglikelihood(
                            model,
                            tokenizer,
                            ctx,
                            cont,
                            device=self._dev,
                            force_depth=force_depth,
                        )
                        out.append((lp, False))
                    return out

                def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
                    out: list[float] = []
                    for req in requests:
                        (text,) = req.args
                        lp, _ = score_completion_loglikelihood(
                            model,
                            tokenizer,
                            "",
                            text,
                            device=self._dev,
                            force_depth=force_depth,
                        )
                        out.append(lp)
                    return out

                def generate_until(self, requests: list[Any]) -> list[str]:
                    return ["" for _ in requests]

            lm_out = lm_eval.simple_evaluate(
                model=_TriuneLMEvalAdapter(),
                tasks=[task_name],
                limit=limit,
            )
            results[task_name] = {
                "evaluator": "lm_eval",
                "results": (lm_out or {}).get("results", {}).get(task_name, {}),
            }
        else:
            raise ValueError(
                f"Unknown evaluation task '{task_name}'. Use 'mqar', 'zeroshot', "
                f"'arc_probe', 'hellaswag_probe', 'mmlu_probe', a local .json/.jsonl file, "
                f"or install `lm-eval`."
            )

    return {
        "tasks": task_list,
        "lm_eval_available": has_lm_eval,
        "results": results,
    }
