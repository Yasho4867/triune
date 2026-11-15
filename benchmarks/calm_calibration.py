"""Calibrate CALM softmax-margin thresholds on one held-out split and evaluate agreement, ECE, AUROC, Pareto curves, and live speedup on a disjoint split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from benchmark_full_model_e2e import load_trained_model  # noqa: E402

from triune.data.tokenizer import load_tokenizer  # noqa: E402

GRID = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
EXITS = ("reflex", "limbic")


def load_docs(source: str, count: int, skip: int, local_path: str = "data/fineweb_edu_large.jsonl") -> list[dict]:
    if source == "local":
        rows = [json.loads(line) for line in Path(local_path).open(encoding="utf-8")]
        return [{"id": f"local:{i}", "text": r["text"]} for i, r in enumerate(rows)][skip : skip + count]
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    return [{"id": row["id"], "text": row["text"]} for row in stream.skip(skip).take(count)]


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Exact Mann-Whitney U ROC-AUC with tie-aware average ranks."""
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return None
    from scipy.stats import rankdata

    ranks = rankdata(scores.astype(np.float64), method="average")
    u = float(ranks[labels == 1].sum()) - pos * (pos + 1) / 2.0
    return round(u / (pos * neg), 4)


def compute_reliability(scores: np.ndarray, agree: np.ndarray, n_bins: int = 10) -> dict:
    """Computes equal-width reliability diagram bins, ECE, MCE, and Brier score."""
    s = np.clip(scores.astype(np.float64), 0.0, 1.0)
    y = agree.astype(np.float64)
    n = max(1, len(s))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (s >= lo) & (s <= hi if i == n_bins - 1 else s < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append(
                {
                    "bin": f"[{lo:.1f}, {hi:.1f}{']' if i == n_bins - 1 else ')'}",
                    "count": 0,
                    "mean_score": None,
                    "empirical_agreement": None,
                    "wilson95_lower": None,
                    "gap": None,
                }
            )
            continue
        mean_s = float(s[mask].mean())
        hits = int(y[mask].sum())
        acc = hits / cnt
        gap = abs(acc - mean_s)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f}{']' if i == n_bins - 1 else ')'}",
                "count": cnt,
                "mean_score": round(mean_s, 4),
                "empirical_agreement": round(acc, 4),
                "wilson95_lower": round(wilson_lower(hits, cnt), 4),
                "gap": round(gap, 4),
            }
        )
    brier = float(np.mean((s - y) ** 2)) if len(s) else 0.0
    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "bins": bins,
    }


def _encode_ids(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, compress=False).ids
    except TypeError:
        return tokenizer.encode(text).ids


@torch.no_grad()
def collect(model, tokenizer, docs: list[dict], seq_len: int, batch: int, device) -> dict[str, np.ndarray]:
    seqs = [_encode_ids(tokenizer, d["text"])[:seq_len] for d in docs]
    seqs = [s for s in seqs if len(s) >= 32]
    margins = {e: [] for e in EXITS}
    confs = {e: [] for e in EXITS}
    agree = {e: [] for e in EXITS}
    nlls = {"reflex": [], "limbic": [], "cortex": []}
    trans_margins = {e: [] for e in EXITS}
    for i in range(0, len(seqs), batch):
        group = seqs[i : i + batch]
        width = max(map(len, group))
        ids = torch.zeros(len(group), width, dtype=torch.long)
        valid = torch.zeros(len(group), width, dtype=torch.bool)
        for r, s in enumerate(group):
            ids[r, : len(s)] = torch.tensor(s)
            valid[r, : len(s)] = True
        ids_dev = ids.to(device)
        reflex, limbic, cortex, _ = model.forward_all_exits(ids_dev, update_stats=False)
        target = cortex.argmax(-1)
        mask = valid.to(device)
        trans_mask = mask[:, 1:]
        next_targets = ids_dev[:, 1:][trans_mask]

        for name, logits in (("reflex", reflex), ("limbic", limbic)):
            probs = torch.softmax(logits.float(), -1)
            top2 = probs.topk(2, -1).values
            m_all = top2[..., 0] - top2[..., 1]
            c_all = top2[..., 0]
            margins[name].append(m_all[mask].cpu())
            confs[name].append(c_all[mask].cpu())
            agree[name].append((logits.argmax(-1) == target)[mask].cpu())
            trans_margins[name].append(m_all[:, :-1][trans_mask].cpu())
            ce = F.cross_entropy(logits[:, :-1].float()[trans_mask], next_targets, reduction="none")
            nlls[name].append(ce.cpu())

        ce_c = F.cross_entropy(cortex[:, :-1].float()[trans_mask], next_targets, reduction="none")
        nlls["cortex"].append(ce_c.cpu())

    return {
        **{f"{e}_margin": torch.cat(margins[e]).numpy() for e in EXITS},
        **{f"{e}_conf": torch.cat(confs[e]).numpy() for e in EXITS},
        **{f"{e}_agree": torch.cat(agree[e]).numpy() for e in EXITS},
        **{f"{e}_trans_margin": torch.cat(trans_margins[e]).numpy() for e in EXITS},
        **{f"{e}_nll": torch.cat(nlls[e]).numpy() for e in ("reflex", "limbic", "cortex")},
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
    c_held = held[f"{exit_name}_conf"]
    result = {
        "unconditional_agreement_pct": {
            "calibration_split": round(100 * float(a_cal.mean()), 2),
            "evaluation_split": round(100 * float(a_held.mean()), 2),
        },
        "margin_auroc": compute_auroc(m_held, a_held),
        "confidence_reliability": compute_reliability(c_held, a_held, n_bins=10),
        "margin_reliability": compute_reliability(m_held, a_held, n_bins=10),
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


def compute_pareto_sweep(held: dict, meta: dict) -> list[dict]:
    """Computes PPL-vs-FLOPs Pareto sweep across margin thresholds using hierarchical CALM routing."""
    rm = held["reflex_trans_margin"]
    lm = held["limbic_trans_margin"]
    r_nll = held["reflex_nll"].astype(np.float64)
    l_nll = held["limbic_nll"].astype(np.float64)
    c_nll = held["cortex_nll"].astype(np.float64)
    n_trans = max(1, len(c_nll))

    num_layers = max(1, int(meta["num_layers"]))
    r_cost = (int(meta["reflex_exit_layer"]) + 1) / num_layers
    l_cost = (int(meta["limbic_exit_layer"]) + 1) / num_layers
    cortex_ppl = math.exp(float(c_nll.mean()))

    sweep_thresholds = [0.0] + GRID + [1.01]
    rows = []
    for tau in sweep_thresholds:
        at_r = rm >= tau
        at_l = (~at_r) & (lm >= tau)
        at_c = ~(at_r | at_l)

        frac_r = float(at_r.mean())
        frac_l = float(at_l.mean())
        frac_c = float(at_c.mean())
        layer_compute = frac_r * r_cost + frac_l * l_cost + frac_c * 1.0

        comp_nll = np.where(at_r, r_nll, np.where(at_l, l_nll, c_nll))
        mean_nll = float(comp_nll.mean())
        calm_ppl = math.exp(min(mean_nll, 20.0))
        ppl_delta_pct = ((calm_ppl - cortex_ppl) / max(cortex_ppl, 1e-8)) * 100.0
        rows.append(
            {
                "threshold": "0.0 (All-Reflex)" if tau == 0.0 else ("1.01 (Dense-Cortex)" if tau > 1.0 else tau),
                "tau_value": tau,
                "reflex_exit_pct": round(100.0 * frac_r, 2),
                "limbic_exit_pct": round(100.0 * frac_l, 2),
                "cortex_exit_pct": round(100.0 * frac_c, 2),
                "layer_compute_pct": round(100.0 * layer_compute, 2),
                "compute_saved_pct": round(100.0 * (1.0 - layer_compute), 2),
                "calm_loss": round(mean_nll, 4),
                "calm_ppl": round(calm_ppl, 2),
                "cortex_ppl": round(cortex_ppl, 2),
                "ppl_delta_pct": round(ppl_delta_pct, 2),
                "transitions": n_trans,
            }
        )
    return rows


@torch.no_grad()
def benchmark_calm_speedup(
    model,
    tokenizer,
    eval_docs: list[dict],
    seq_len: int,
    thresholds: list[float],
    device: torch.device,
) -> dict:
    """Measures live end-to-end wall-clock throughput with true token-level CALM compaction vs dense Cortex."""
    seqs = []
    for d in eval_docs:
        ids = _encode_ids(tokenizer, d["text"])[:seq_len]
        if len(ids) >= min(seq_len, 128):
            seqs.append(ids[: min(seq_len, 256)])
        if len(seqs) >= 8:
            break
    if not seqs:
        return {"skipped": True, "reason": "insufficient held-out sequences"}
    width = min(len(s) for s in seqs)
    batch_ids = torch.tensor([s[:width] for s in seqs], dtype=torch.long, device=device)
    tokens_per_fwd = batch_ids.numel()

    def _time_mode(**fwd_kwargs) -> tuple[float, dict | None]:
        for _ in range(3):
            _ = model(batch_ids, **fwd_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = model(batch_ids, **fwd_kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        return times[len(times) // 2], getattr(model, "last_calm_compaction_stats", None)

    dense_ms, _ = _time_mode(force_depth=2)
    dense_tok_s = round(tokens_per_fwd / max(dense_ms / 1000.0, 1e-6), 1)

    runs = []
    for tau in thresholds:
        calm_ms, stats = _time_mode(exit_policy="calm", confidence_threshold=tau)
        calm_tok_s = round(tokens_per_fwd / max(calm_ms / 1000.0, 1e-6), 1)
        runs.append(
            {
                "threshold": tau,
                "median_ms": round(calm_ms, 2),
                "tokens_per_sec": calm_tok_s,
                "speedup_vs_dense": round(dense_ms / max(calm_ms, 1e-6), 3),
                "compaction_stats": stats,
            }
        )

    return {
        "batch_size": int(batch_ids.size(0)),
        "seq_len": int(batch_ids.size(1)),
        "tokens_per_forward": int(tokens_per_fwd),
        "dense_cortex_median_ms": round(dense_ms, 2),
        "dense_cortex_tokens_per_sec": dense_tok_s,
        "runs": runs,
    }


def split_info(docs: list[dict], positions: int) -> dict:
    digest = hashlib.sha256("".join(d["id"] for d in docs).encode()).hexdigest()[:16]
    return {
        "documents": len(docs),
        "positions": positions,
        "first_id": docs[0]["id"],
        "last_id": docs[-1]["id"],
        "id_sha256_16": digest,
    }


def render_report(result: dict) -> str:
    lines = [
        "# CALM Margin Calibration, Reliability & Pareto Report",
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
        conf_rel = ev["confidence_reliability"]
        lines += [
            f"## {exit_name.capitalize()} exit vs Cortex",
            "",
            f"- Unconditional top-1 agreement: calibration `{ev['unconditional_agreement_pct']['calibration_split']}%`, "
            f"evaluation `{ev['unconditional_agreement_pct']['evaluation_split']}%`",
            f"- Discrimination & Calibration Quality (eval split): "
            f"`Margin AUROC = {ev['margin_auroc']}`, `Confidence ECE (10-bin) = {conf_rel['ece']}`, "
            f"`MCE = {conf_rel['mce']}`, `Brier = {conf_rel['brier_score']}`",
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
            f"10-bin reliability diagram ({exit_name}, top-1 probability vs exit-cortex agreement):",
            "",
            "| Confidence Bin | Positions | Mean Confidence | Empirical Agreement | Wilson-95 Lower | |Acc - Conf| |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for b in conf_rel["bins"]:
            mc = f"{b['mean_score']:.4f}" if b["mean_score"] is not None else "-"
            acc = f"{b['empirical_agreement']:.4f}" if b["empirical_agreement"] is not None else "-"
            wl = f"{b['wilson95_lower']:.4f}" if b["wilson95_lower"] is not None else "-"
            gp = f"{b['gap']:.4f}" if b["gap"] is not None else "-"
            lines.append(f"| {b['bin']} | {b['count']} | {mc} | {acc} | {wl} | {gp} |")
        lines += [
            "",
            "Coverage vs agreement on the evaluation split (softmax probability margin):",
            "",
            "| Margin threshold | Coverage | Agreement | Wilson-95 lower | Positions |",
            "|:---:|:---:|:---:|:---:|:---:|",
        ]
        for r in ev["evaluation_curve"]:
            lines.append(
                f"| {r['threshold']} | {r['coverage_pct']}% | {r['agreement_pct']}% | {r['agreement_wilson95_lower_pct']}% | {r['positions']} |"
            )
        lines.append("")

    pareto = result.get("pareto_sweep", [])
    if pareto:
        lines += [
            "## Quality vs Layer-Compute Pareto Sweep (Hierarchical Reflex -> Limbic -> Cortex)",
            "",
            f"Evaluated over `{pareto[0]['transitions']:,}` held-out next-token transitions (Dense Cortex PPL = `{pareto[0]['cortex_ppl']}`):",
            "",
            "| Threshold ($\\tau$) | Reflex Exit | Limbic Exit | Cortex Exit | Active Layers % | Layer Compute Saved | CALM PPL | $\\Delta$ PPL vs Cortex |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for row in pareto:
            lines.append(
                f"| {row['threshold']} | {row['reflex_exit_pct']}% | {row['limbic_exit_pct']}% | {row['cortex_exit_pct']}% | "
                f"{row['layer_compute_pct']}% | {row['compute_saved_pct']}% | {row['calm_ppl']} | {row['ppl_delta_pct']:+.2f}% |"
            )
        lines.append("")

    sp = result.get("speedup_benchmark", {})
    if sp and not sp.get("skipped"):
        lines += [
            "## Live Token-Compacted CALM Wall-Clock Throughput",
            "",
            f"Batch `{sp['batch_size']} x {sp['seq_len']}` (`{sp['tokens_per_forward']:,}` tokens/forward) on held-out text. "
            f"Dense Cortex baseline (`force_depth=2`): `{sp['dense_cortex_median_ms']} ms` (`{sp['dense_cortex_tokens_per_sec']:,} tok/s`).",
            "",
            "| Margin Threshold ($\\tau$) | Reflex / Limbic / Cortex Tokens | Compacted SeqLens (L / C) | Median Latency | Throughput | Speedup vs Dense |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for r in sp.get("runs", []):
            st = r.get("compaction_stats") or {}
            dist = f"{st.get('reflex_tokens', 0)} / {st.get('limbic_tokens', 0)} / {st.get('cortex_tokens', 0)}"
            seqlens = f"{st.get('limbic_seq_len', 0)} / {st.get('cortex_seq_len', 0)}"
            lines.append(
                f"| {r['threshold']} | {dist} | {seqlens} | {r['median_ms']} ms | {r['tokens_per_sec']:,} tok/s | **{r['speedup_vs_dense']:.3f}x** |"
            )
        lines.append("")

    lines += ["## Verdict", "", result["verdict"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/triune_step_24501.pt")
    parser.add_argument("--source", choices=("hf", "local"), default="local")
    parser.add_argument("--local-path", default="data/fineweb_edu_large.jsonl")
    parser.add_argument("--skip-docs", type=int, default=7500)
    parser.add_argument("--docs-per-split", type=int, default=200)
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
    docs = load_docs(args.source, 2 * n, args.skip_docs, local_path=args.local_path)
    cal_docs, eval_docs = docs[:n], docs[n:]
    cal = collect(model, tokenizer, cal_docs, args.seq_len, args.batch, device)
    held = collect(model, tokenizer, eval_docs, args.seq_len, args.batch, device)

    exits = {e: evaluate(cal, held, e, args.targets, args.min_positions) for e in EXITS}
    pareto = compute_pareto_sweep(held, meta)

    calibrated_taus = [
        c["threshold"] for ev in exits.values() for c in ev["calibrated"].values() if c["threshold"] is not None
    ]
    speed_taus = sorted(set([0.20, 0.40, 0.60, 0.80] + calibrated_taus))
    speedup = benchmark_calm_speedup(model, tokenizer, eval_docs, args.seq_len, speed_taus, device)

    met = [e for e, ev in exits.items() if any(c["target_met_on_evaluation"] for c in ev["calibrated"].values())]
    if met:
        verdict = (
            f"Held-out agreement target reached on N={len(held['reflex_margin']):,} evaluation positions for: {', '.join(met)} "
            "(see reliability tables, Pareto sweep, and live token-compacted speedup benchmark)."
        )
    else:
        verdict = (
            f"Evaluated on N={len(held['reflex_margin']):,} held-out positions; no margin threshold met the requested "
            "Wilson-95 agreement lower bound on the disjoint evaluation split (see reliability & Pareto tables for diagnostic metrics)."
        )
    source_description = (
        f"HuggingFaceFW/fineweb-edu `sample-10BT` (the training stream), documents {args.skip_docs} to "
        f"{args.skip_docs + 2 * n - 1} in stream order."
        if args.source == "hf"
        else f"local `{args.local_path}`, disjoint held-out suffix documents {args.skip_docs} to {args.skip_docs + 2 * n - 1}."
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
        "pareto_sweep": pareto,
        "speedup_benchmark": speedup,
        "verdict": verdict,
    }
    stem = Path(args.output_stem)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(render_report(result), encoding="utf-8")
    print(verdict)
    for e, ev in exits.items():
        print(
            e,
            "auroc=",
            ev["margin_auroc"],
            "ece=",
            ev["confidence_reliability"]["ece"],
            ev["unconditional_agreement_pct"],
            {
                k: (c["threshold"], c.get("evaluation_coverage_pct"), c.get("evaluation_agreement_pct"))
                for k, c in ev["calibrated"].items()
            },
        )


if __name__ == "__main__":
    main()
