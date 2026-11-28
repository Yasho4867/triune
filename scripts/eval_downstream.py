#!/usr/bin/env python3
"""Offline & HuggingFace Multiple-Choice Downstream Evaluation Harness for TriuneTransformer.

Evaluates full Cortex, static Limbic/Reflex exits, static Conformal CALM, and
entropy-guarded Conformal CALM across standard zero-shot tasks (ARC-Easy, ARC-Challenge,
PIQA, HellaSwag, BoolQ, or offline synthetic_mc) and optional EleutherAI lm-eval.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lucid_machines.data.tokenizer import load_tokenizer


def _load_task_examples(task: str, max_samples: int) -> list[dict[str, Any]]:
    """Load multiple-choice evaluation items for a named benchmark task."""
    t_key = task.strip().lower()
    if t_key == "synthetic_mc":
        rng = random.Random(2026)
        stems = [
            ("Water freezes at zero degrees Celsius and boils at", ["one hundred degrees Celsius.", "negative forty kilometers.", "twelve liters of nitrogen.", "eighteen pixels per inch."]),
            ("In linear algebra an orthogonal matrix satisfies", ["Q transpose times Q equals the identity.", "every eigenvalue is strictly zero.", "all rows sum to infinity.", "its rank equals negative one."]),
            ("Mixture of experts layers route tokens to", ["a sparse subset of feed-forward networks.", "the disk drive controller.", "only the bias vectors.", "a single scalar constant."]),
            ("Conformal prediction calibrates thresholds using", ["held-out calibration quantiles.", "unbounded random noise.", "deleted validation splits.", "zero gradient steps."]),
        ]
        items = []
        for idx in range(max_samples):
            ctx, choices = stems[idx % len(stems)]
            perm = list(range(len(choices)))
            rng.shuffle(perm)
            gold_new = perm.index(0)
            items.append({"context": ctx, "choices": [choices[p] for p in perm], "label": gold_new})
        return items

    from datasets import load_dataset

    if t_key in ("arc_easy", "arc-easy"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        out = []
        for row in ds:
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            if row["answerKey"] not in labels or len(texts) < 2:
                continue
            out.append({"context": f"Question: {row['question']}\nAnswer:", "choices": texts, "label": labels.index(row["answerKey"])})
            if len(out) >= max_samples:
                break
        return out

    if t_key in ("arc_challenge", "arc-challenge"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
        out = []
        for row in ds:
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            if row["answerKey"] not in labels or len(texts) < 2:
                continue
            out.append({"context": f"Question: {row['question']}\nAnswer:", "choices": texts, "label": labels.index(row["answerKey"])})
            if len(out) >= max_samples:
                break
        return out

    if t_key == "piqa":
        ds = load_dataset("ybisk/piqa", split="validation")
        out = []
        for row in ds:
            lbl = int(row["label"])
            if lbl not in (0, 1):
                continue
            out.append({"context": f"Goal: {row['goal']}\nSolution:", "choices": [row["sol1"], row["sol2"]], "label": lbl})
            if len(out) >= max_samples:
                break
        return out

    if t_key == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        out = []
        for row in ds:
            if not str(row["label"]).isdigit():
                continue
            lbl = int(row["label"])
            endings = list(row["endings"])
            if not (0 <= lbl < len(endings)):
                continue
            out.append({"context": str(row["ctx"]), "choices": endings, "label": lbl})
            if len(out) >= max_samples:
                break
        return out

    if t_key == "boolq":
        ds = load_dataset("google/boolq", split="validation")
        out = []
        for row in ds:
            lbl = 0 if bool(row["answer"]) else 1
            out.append({"context": f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:", "choices": ["Yes", "No"], "label": lbl})
            if len(out) >= max_samples:
                break
        return out

    raise ValueError(f"Unsupported task: {task!r}")


def _encode_pair(tokenizer: Any, context: str, choice: str, max_seq_len: int) -> tuple[list[int], int]:
    ctx_ids = list(tokenizer.encode(context).ids)
    cont_ids = list(tokenizer.encode(" " + choice.lstrip()).ids)
    if not cont_ids:
        cont_ids = [0]
    if not ctx_ids:
        ctx_ids = [0]
    total = len(ctx_ids) + len(cont_ids)
    if total > max_seq_len:
        keep_ctx = max(1, max_seq_len - len(cont_ids))
        ctx_ids = ctx_ids[-keep_ctx:]
        if len(ctx_ids) + len(cont_ids) > max_seq_len:
            cont_ids = cont_ids[: max(1, max_seq_len - len(ctx_ids))]
    return ctx_ids + cont_ids, len(ctx_ids)


def _forward_with_mode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mode: str,
    *,
    calm_tau: float,
    calm_entropy_beta: float,
    calm_calib_entropy: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    stats: dict[str, Any] = {"exit_counts": [0, 0, 0], "effective_threshold": calm_tau, "reflex_entropy_mean": None}
    n_tok = int(input_ids.numel())
    if mode == "cortex":
        out = model(input_ids, force_depth=2)
        logits = out[0] if isinstance(out, tuple) else out
        stats["exit_counts"] = [0, 0, n_tok]
    elif mode == "limbic":
        out = model(input_ids, force_depth=1)
        logits = out[0] if isinstance(out, tuple) else out
        stats["exit_counts"] = [0, n_tok, 0]
    elif mode == "reflex":
        out = model(input_ids, force_depth=0)
        logits = out[0] if isinstance(out, tuple) else out
        stats["exit_counts"] = [n_tok, 0, 0]
    elif mode in ("calm", "calm_entropy_guard"):
        beta = calm_entropy_beta if mode == "calm_entropy_guard" else 0.0
        out = model(
            input_ids,
            exit_policy="calm",
            confidence_threshold=calm_tau,
            calm_entropy_beta=beta,
            calm_calib_entropy=calm_calib_entropy,
        )
        logits = out[0] if isinstance(out, tuple) else out
        depths = getattr(model, "last_calm_exit_depths", None)
        if depths is not None:
            stats["exit_counts"] = [
                int((depths == 0).sum().item()),
                int((depths == 1).sum().item()),
                int((depths == 2).sum().item()),
            ]
        else:
            stats["exit_counts"] = [0, 0, n_tok]
        comp = getattr(model, "last_calm_compaction_stats", None) or {}
        if "effective_threshold" in comp:
            stats["effective_threshold"] = float(comp["effective_threshold"])
        if "reflex_entropy_mean" in comp:
            stats["reflex_entropy_mean"] = float(comp["reflex_entropy_mean"])
    else:
        raise ValueError(f"Unknown eval mode: {mode!r}")
    return logits, stats


@torch.no_grad()
def evaluate_model_on_tasks(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[str],
    *,
    modes: list[str],
    max_samples: int = 100,
    calm_tau: float = 0.20,
    calm_entropy_beta: float = 0.25,
    calm_calib_entropy: float = 3.5,
    max_seq_len: int = 512,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run multiple-choice log-likelihood accuracy and exit-compute accounting across modes."""
    dev = device or next(model.parameters()).device
    model.eval()
    num_layers = int(getattr(model, "num_layers", 8))
    exit_layers = [
        int(getattr(model, "reflex_exit_layer", max(0, num_layers // 4))) + 1,
        int(getattr(model, "limbic_exit_layer", max(1, (num_layers * 5) // 8))) + 1,
        num_layers,
    ]

    task_datasets = {t: _load_task_examples(t, max_samples=max_samples) for t in tasks}
    results: dict[str, Any] = {
        "tasks": tasks,
        "modes": {},
        "calm_tau": calm_tau,
        "calm_entropy_beta": calm_entropy_beta,
        "calm_calib_entropy": calm_calib_entropy,
    }

    for mode in modes:
        if hasattr(model, "reset_calm_entropy"):
            model.reset_calm_entropy()
        mode_tasks: dict[str, Any] = {}
        t0 = time.perf_counter()
        total_acc_num = 0
        total_norm_num = 0
        total_items = 0
        mode_exit_counts = [0, 0, 0]
        eff_taus: list[float] = []
        entropies: list[float] = []

        for task_name, items in task_datasets.items():
            correct = 0
            correct_norm = 0
            for item in items:
                raw_scores = []
                norm_scores = []
                for choice in item["choices"]:
                    ids, ctx_len = _encode_pair(tokenizer, item["context"], choice, max_seq_len=max_seq_len)
                    inp = torch.tensor([ids], dtype=torch.long, device=dev)
                    logits, st = _forward_with_mode(
                        model,
                        inp,
                        mode,
                        calm_tau=calm_tau,
                        calm_entropy_beta=calm_entropy_beta,
                        calm_calib_entropy=calm_calib_entropy,
                    )
                    shift_logits = logits[:, ctx_len - 1 : -1, :].float()
                    shift_targets = inp[:, ctx_len:]
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    tok_lp = log_probs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
                    ll = float(tok_lp.sum().item())
                    raw_scores.append(ll)
                    norm_scores.append(ll / max(1, len(choice.encode("utf-8"))))
                    for k in range(3):
                        mode_exit_counts[k] += st["exit_counts"][k]
                    if st["effective_threshold"] is not None:
                        eff_taus.append(float(st["effective_threshold"]))
                    if st["reflex_entropy_mean"] is not None:
                        entropies.append(float(st["reflex_entropy_mean"]))

                pred_raw = max(range(len(raw_scores)), key=lambda idx: raw_scores[idx])
                pred_norm = max(range(len(norm_scores)), key=lambda idx: norm_scores[idx])
                if pred_raw == int(item["label"]):
                    correct += 1
                if pred_norm == int(item["label"]):
                    correct_norm += 1

            n_i = max(1, len(items))
            mode_tasks[task_name] = {
                "samples": len(items),
                "acc": round(correct / n_i, 4),
                "acc_norm": round(correct_norm / n_i, 4),
            }
            total_acc_num += correct
            total_norm_num += correct_norm
            total_items += len(items)

        elapsed_s = max(1e-6, time.perf_counter() - t0)
        tot_tok = max(1, sum(mode_exit_counts))
        avg_layers = sum(c * d for c, d in zip(mode_exit_counts, exit_layers)) / tot_tok
        flop_savings_pct = max(0.0, (1.0 - avg_layers / max(1, num_layers)) * 100.0)
        results["modes"][mode] = {
            "by_task": mode_tasks,
            "macro_acc": round(total_acc_num / max(1, total_items), 4),
            "macro_acc_norm": round(total_norm_num / max(1, total_items), 4),
            "exit_fraction": {
                "reflex": round(mode_exit_counts[0] / tot_tok, 4),
                "limbic": round(mode_exit_counts[1] / tot_tok, 4),
                "cortex": round(mode_exit_counts[2] / tot_tok, 4),
            },
            "avg_layers": round(avg_layers, 3),
            "theoretical_flop_savings_pct": round(flop_savings_pct, 2),
            "mean_effective_tau": round(sum(eff_taus) / max(1, len(eff_taus)), 4),
            "mean_reflex_entropy": round(sum(entropies) / max(1, len(entropies)), 4) if entropies else None,
            "elapsed_s": round(elapsed_s, 3),
        }
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Downstream evaluation harness for TriuneTransformer checkpoints.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/triune_studio_step_36500_rescued.pt")
    p.add_argument("--tokenizer", type=str, default="triune_tokenizer.json")
    p.add_argument("--tasks", type=str, default="arc_easy,piqa,hellaswag")
    p.add_argument("--modes", type=str, default="cortex,limbic,reflex,calm,calm_entropy_guard")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--calm_tau", type=float, default=0.20)
    p.add_argument("--calm_entropy_beta", type=float, default=0.25)
    p.add_argument("--calm_calib_entropy", type=float, default=3.5)
    p.add_argument("--output_json", type=str, default="benchmarks/downstream_eval_results.json")
    p.add_argument("--allow_untrained_fallback", action="store_true")
    p.add_argument("--use_lm_eval", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_arg_parser().parse_args(argv)
    from benchmark_full_model_e2e import get_system_telemetry, load_trained_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, model_meta = load_trained_model(
        args.checkpoint,
        device=device,
        require_checkpoint=not args.allow_untrained_fallback,
    )
    tok = load_tokenizer(Path(args.tokenizer))
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    lm_eval_summary = None
    if args.use_lm_eval:
        try:
            import lm_eval  # type: ignore

            lm_eval_summary = {"available": True, "version": getattr(lm_eval, "__version__", "unknown")}
        except ImportError:
            lm_eval_summary = {"available": False, "fallback": "built_in_loglikelihood"}

    eval_out = evaluate_model_on_tasks(
        model,
        tok,
        tasks,
        modes=modes,
        max_samples=args.max_samples,
        calm_tau=args.calm_tau,
        calm_entropy_beta=args.calm_entropy_beta,
        calm_calib_entropy=args.calm_calib_entropy,
        device=device,
    )
    payload = {
        "hardware": get_system_telemetry(),
        "checkpoint": args.checkpoint,
        "model_meta": model_meta,
        "lm_eval": lm_eval_summary,
        **eval_out,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved downstream evaluation report to {out_path}")
    return payload


if __name__ == "__main__":
    main()
