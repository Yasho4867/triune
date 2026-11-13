"""Calibrate CALM softmax-margin thresholds on one held-out split and evaluate agreement on a disjoint one."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from benchmark_full_model_e2e import load_trained_model  # noqa: E402
from triune.data.tokenizer import load_tokenizer  # noqa: E402

GRID = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
EXITS = ("reflex", "limbic")


def load_docs(source: str, count: int, skip: int) -> list[dict]:
    if source == "local":
        rows = [json.loads(line) for line in Path("data/fineweb_edu_large.jsonl").open(encoding="utf-8")]
        return [{"id": f"local:{i}", "text": r["text"]} for i, r in enumerate(rows)][skip : skip + count]
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    return [
        {"id": row["id"], "text": row["text"]} for row in stream.skip(skip).take(count)
    ]


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


@torch.no_grad()
def collect(model, tokenizer, docs: list[dict], seq_len: int, batch: int, device) -> dict[str, np.ndarray]:
    seqs = [tokenizer.encode(d["text"]).ids[:seq_len] for d in docs]
    seqs = [s for s in seqs if len(s) >= 32]
    margins = {e: [] for e in EXITS}
    agree = {e: [] for e in EXITS}
    for i in range(0, len(seqs), batch):
        group = seqs[i : i + batch]
        width = max(map(len, group))
        ids = torch.zeros(len(group), width, dtype=torch.long)
        valid = torch.zeros(len(group), width, dtype=torch.bool)
        for r, s in enumerate(group):
            ids[r, : len(s)] = torch.tensor(s)
            valid[r, : len(s)] = True
        reflex, limbic, cortex, _ = model.forward_all_exits(ids.to(device))
        target = cortex.argmax(-1)
        mask = valid.to(device)
        for name, logits in (("reflex", reflex), ("limbic", limbic)):
            top2 = torch.softmax(logits.float(), -1).topk(2, -1).values
            margins[name].append((top2[..., 0] - top2[..., 1])[mask].cpu())
            agree[name].append((logits.argmax(-1) == target)[mask].cpu())
    return {
        **{f"{e}_margin": torch.cat(margins[e]).numpy() for e in EXITS},
        **{f"{e}_agree": torch.cat(agree[e]).numpy() for e in EXITS},
    }


def curve(margin: np.ndarray, agree: np.ndarray) -> list[dict]:
    rows = []
    for t in GRID:
        sel = margin >= t
        n = int(sel.sum())
        hits = int(agree[sel].sum())
        rows.append(
            {
                "threshold": t,
                "coverage_pct": round(100 * n / len(margin), 2),
                "agreement_pct": round(100 * hits / n, 2) if n else None,
                "agreement_wilson95_lower_pct": round(100 * wilson_lower(hits, n), 2) if n else None,
                "positions": n,
            }
        )
    return rows


def calibrate(margin: np.ndarray, agree: np.ndarray, target: float, min_positions: int) -> float | None:
    for row in curve(margin, agree):
        if row["positions"] >= min_positions and row["agreement_wilson95_lower_pct"] >= 100 * target:
            return row["threshold"]
    return None


def evaluate(cal: dict, held: dict, exit_name: str, targets: list[float], min_positions: int) -> dict:
    m_cal, a_cal = cal[f"{exit_name}_margin"], cal[f"{exit_name}_agree"]
    m_held, a_held = held[f"{exit_name}_margin"], held[f"{exit_name}_agree"]
    result = {
        "unconditional_agreement_pct": {
            "calibration_split": round(100 * float(a_cal.mean()), 2),
            "evaluation_split": round(100 * float(a_held.mean()), 2),
        },
        "evaluation_curve": curve(m_held, a_held),
        "calibrated": {},
    }
    for target in targets:
        tau = calibrate(m_cal, a_cal, target, min_positions)
        entry = {"target_agreement_pct": 100 * target, "threshold": tau}
        if tau is not None:
            sel = m_held >= tau
            n, hits = int(sel.sum()), int(a_held[sel].sum())
            entry.update(
                evaluation_positions=n,
                evaluation_coverage_pct=round(100 * n / len(m_held), 2),
                evaluation_agreement_pct=round(100 * hits / n, 2) if n else None,
                evaluation_agreement_wilson95_lower_pct=round(100 * wilson_lower(hits, n), 2) if n else None,
                target_met_on_evaluation=bool(n >= min_positions and wilson_lower(hits, n) >= target),
            )
        else:
            entry["target_met_on_evaluation"] = False
        result["calibrated"][f"{int(100 * target)}"] = entry
    return result


def split_info(docs: list[dict], positions: int) -> dict:
    digest = hashlib.sha256("".join(d["id"] for d in docs).encode()).hexdigest()[:16]
    return {"documents": len(docs), "positions": positions, "first_id": docs[0]["id"], "last_id": docs[-1]["id"], "id_sha256_16": digest}


def render_report(result: dict) -> str:
    lines = [
        "# CALM Margin Calibration Report",
        "",
        f"- **Checkpoint**: `{result['checkpoint']}`",
        f"- **Source**: {result['source_description']}",
        f"- **Calibration split**: {json.dumps(result['splits']['calibration'])}",
        f"- **Evaluation split**: {json.dumps(result['splits']['evaluation'])}",
        f"- **Protocol**: smallest threshold on the calibration split whose Wilson 95% lower bound on exit-vs-cortex "
        f"top-1 agreement reaches the target (min {result['min_positions']} positions); that threshold is then applied "
        "unchanged to the disjoint evaluation split. Nothing is fit on the evaluation split.",
        f"- **Sequence length**: first {result['seq_len']} tokens of each document.",
        "",
    ]
    for exit_name, ev in result["exits"].items():
        lines += [
            f"## {exit_name.capitalize()} exit vs Cortex",
            "",
            f"Unconditional top-1 agreement: calibration {ev['unconditional_agreement_pct']['calibration_split']}%, "
            f"evaluation {ev['unconditional_agreement_pct']['evaluation_split']}%.",
            "",
            "| Target | Calibrated threshold | Eval coverage | Eval agreement | Eval Wilson-95 lower | Target met |",
            "|:---|:---:|:---:|:---:|:---:|:---:|",
        ]
        for key, c in ev["calibrated"].items():
            if c["threshold"] is None:
                lines.append(f"| {key}% | none reaches target on calibration split | - | - | - | no |")
            else:
                lines.append(
                    f"| {key}% | {c['threshold']} | {c['evaluation_coverage_pct']}% | {c['evaluation_agreement_pct']}% | "
                    f"{c['evaluation_agreement_wilson95_lower_pct']}% | {'yes' if c['target_met_on_evaluation'] else 'no'} |"
                )
        lines += [
            "",
            "Coverage vs agreement on the evaluation split:",
            "",
            "| Margin threshold | Coverage | Agreement | Wilson-95 lower | Positions |",
            "|:---:|:---:|:---:|:---:|:---:|",
        ]
        for r in ev["evaluation_curve"]:
            lines.append(
                f"| {r['threshold']} | {r['coverage_pct']}% | {r['agreement_pct']}% | {r['agreement_wilson95_lower_pct']}% | {r['positions']} |"
            )
        lines.append("")
    lines += ["## Verdict", "", result["verdict"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/triune_step_24501.pt")
    parser.add_argument("--source", choices=("hf", "local"), default="hf")
    parser.add_argument("--skip-docs", type=int, default=100_000)
    parser.add_argument("--docs-per-split", type=int, default=400)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--min-positions", type=int, default=500)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95])
    parser.add_argument("--output-stem", default="benchmarks/calm_calibration")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_trained_model(args.checkpoint, device=device)
    tokenizer = load_tokenizer("triune_tokenizer.json")

    n = args.docs_per_split
    docs = load_docs(args.source, 2 * n, args.skip_docs)
    cal_docs, eval_docs = docs[:n], docs[n:]
    cal = collect(model, tokenizer, cal_docs, args.seq_len, args.batch, device)
    held = collect(model, tokenizer, eval_docs, args.seq_len, args.batch, device)

    exits = {e: evaluate(cal, held, e, args.targets, args.min_positions) for e in EXITS}
    met = [e for e, ev in exits.items() if any(c["target_met_on_evaluation"] for c in ev["calibrated"].values())]
    if met:
        verdict = (
            f"Held-out agreement target reached for: {', '.join(met)} (see tables). Coverage at the calibrated "
            "threshold bounds the fraction of positions that may exit early; this benchmark measures agreement, "
            "not latency, so no speedup figure is claimed."
        )
    else:
        verdict = (
            "No calibrated threshold meets its agreement target on the disjoint evaluation split for either exit. "
            "CALM gating is not defensible for this checkpoint and no speedup claim is published."
        )
    source_description = (
        f"HuggingFaceFW/fineweb-edu `sample-10BT` (the training stream), documents {args.skip_docs} to "
        f"{args.skip_docs + 2 * n - 1} in stream order. Training consumed {meta['tokens_trained']:,} tokens "
        "(at ~1k tokens per document, well under the skipped prefix), so these documents were not seen in training."
        if args.source == "hf"
        else "local `data/fineweb_edu_large.jsonl`; the checkpoint was trained on the same corpus family, so overlap "
        "with training data cannot be ruled out."
    )
    result = {
        "checkpoint": args.checkpoint,
        "model": {k: meta[k] for k in ("step", "hidden_dim", "num_layers", "reflex_exit_layer", "limbic_exit_layer")},
        "source": args.source,
        "source_description": source_description,
        "seq_len": args.seq_len,
        "min_positions": args.min_positions,
        "splits": {
            "calibration": split_info(cal_docs, len(cal["reflex_margin"])),
            "evaluation": split_info(eval_docs, len(held["reflex_margin"])),
        },
        "exits": exits,
        "verdict": verdict,
    }
    stem = Path(args.output_stem)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(render_report(result), encoding="utf-8")
    print(verdict)
    for e, ev in exits.items():
        print(e, ev["unconditional_agreement_pct"], {k: (c["threshold"], c.get("evaluation_coverage_pct"), c.get("evaluation_agreement_pct")) for k, c in ev["calibrated"].items()})


if __name__ == "__main__":
    main()
