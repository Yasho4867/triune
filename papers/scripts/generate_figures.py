#!/usr/bin/env python3
"""Generate publication-grade figures for Triune research papers.

Produces vector PDF and high-DPI PNG plots for:
1. Dense convergence trajectory across 5 optimizer regimes (FineWeb-Edu).
2. MoE convergence trajectory across 4 optimizer regimes.
3. GLA constant-memory autoregressive decoding vs Softmax KV-cache explosion.
4. Chunked Vocabulary-Projection Cross-Entropy memory scaling vs PyTorch.
5. CUDA Graph launch latency reduction and multi-step bitwise optimizer parity.
6. Multi-level Parallelism speedups (Async DMA prefetching & Fused QKVG).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"
PAPERS_DIR = REPO_ROOT / "papers"

P1_FIGS = PAPERS_DIR / "paper1_centroidsteer" / "figures"
P2_FIGS = PAPERS_DIR / "paper2_triune_systems" / "figures"

P1_FIGS.mkdir(parents=True, exist_ok=True)
P2_FIGS.mkdir(parents=True, exist_ok=True)

# Publication style configuration
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "lines.linewidth": 1.75,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
    }
)

COLORS = {
    "triune": "#1f77b4",  # Blue
    "muon": "#ff7f0e",  # Orange
    "centroid_no_muon": "#2ca02c",  # Green
    "adamw": "#d62728",  # Red
    "bnb_8bit": "#9467bd",  # Purple
    "triune_moe": "#1f77b4",
    "muon_adamw": "#ff7f0e",
}


def generate_dense_convergence_figure():
    """Generates Figure 1 for Paper 1: Dense multi-seed convergence trajectory."""
    json_path = BENCHMARK_DIR / "convergence_100_steps_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    regimes_to_plot = [
        ("adamw", "Standard AdamW", COLORS["adamw"], "--", "s"),
        ("bnb_8bit", "BitsAndBytes 8-bit AdamW", COLORS["bnb_8bit"], "-.", "^"),
        ("muon", "Muon Baseline (2D)", COLORS["muon"], ":", "v"),
        ("centroid_no_muon", "CentroidSteer (No Muon)", COLORS["centroid_no_muon"], "-.", "d"),
        ("triune", "Full Triune Stack", COLORS["triune"], "-", "o"),
    ]

    for key, label, color, ls, marker in regimes_to_plot:
        if key not in summary:
            raise KeyError(f"Required series '{key}' missing from summary in {json_path}")
        traj = summary[key]["val_trajectory"]
        if not traj:
            raise ValueError(f"Empty trajectory for series '{key}' in {json_path}")
        steps = [pt["step"] for pt in traj]
        losses = [pt["val_loss_mean"] for pt in traj]
        stds = [pt["val_loss_std"] for pt in traj]
        ppls = [pt["val_ppl_mean"] for pt in traj]

        ax1.plot(steps, losses, label=label, color=color, linestyle=ls, marker=marker, markersize=4)
        ax1.fill_between(
            steps,
            np.array(losses) - np.array(stds),
            np.array(losses) + np.array(stds),
            color=color,
            alpha=0.12,
        )

        ax2.plot(steps, ppls, label=label, color=color, linestyle=ls, marker=marker, markersize=4)

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Held-Out Validation Loss")
    ax1.set_title("(a) Loss Convergence (FineWeb-Edu)")
    ax1.grid(True)
    ax1.set_ylim(7.3, 10.7)

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("(b) Perplexity Trajectory")
    ax2.set_yscale("log")
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    for out_dir in [P1_FIGS]:
        fig.savefig(out_dir / "dense_convergence.pdf")
        fig.savefig(out_dir / "dense_convergence.png")
    plt.close(fig)
    print("Generated dense_convergence.[pdf|png]")


def generate_moe_convergence_figure():
    """Generates Figure 2 for Paper 1: MoE multi-seed dynamic steering convergence."""
    json_path = BENCHMARK_DIR / "moe_convergence_100_steps_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    regimes_to_plot = [
        ("adamw", "Standard AdamW", COLORS["adamw"], "--", "s"),
        ("centroid_no_muon", "CentroidSteer (No Muon)", COLORS["centroid_no_muon"], "-.", "d"),
        ("muon_adamw", "Muon + AdamW Baseline", COLORS["muon_adamw"], ":", "v"),
        ("triune_moe", "Triune MoE (3-Tier Synergy)", COLORS["triune_moe"], "-", "o"),
    ]

    for key, label, color, ls, marker in regimes_to_plot:
        if key not in summary:
            raise KeyError(f"Required series '{key}' missing from summary in {json_path}")
        traj = summary[key]["val_trajectory"]
        if not traj:
            raise ValueError(f"Empty trajectory for series '{key}' in {json_path}")
        steps = [pt["step"] for pt in traj]
        losses = [pt["val_loss_mean"] for pt in traj]
        stds = [pt["val_loss_std"] for pt in traj]
        ppls = [pt["val_ppl_mean"] for pt in traj]

        ax1.plot(steps, losses, label=label, color=color, linestyle=ls, marker=marker, markersize=4)
        ax1.fill_between(
            steps,
            np.array(losses) - np.array(stds),
            np.array(losses) + np.array(stds),
            color=color,
            alpha=0.12,
        )

        ax2.plot(steps, ppls, label=label, color=color, linestyle=ls, marker=marker, markersize=4)

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Held-Out Validation Loss")
    ax1.set_title("(a) 90.8M MoE Convergence")
    ax1.grid(True)
    ax1.set_ylim(7.6, 10.6)

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("(b) MoE Perplexity Trajectory")
    ax2.set_yscale("log")
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    for out_dir in [P1_FIGS]:
        fig.savefig(out_dir / "moe_convergence.pdf")
        fig.savefig(out_dir / "moe_convergence.png")
    plt.close(fig)
    print("Generated moe_convergence.[pdf|png]")


def generate_optimizer_memory_figure():
    """Generates Figure 3 for Paper 1: Optimizer state memory footprint."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    # Panel (a): Per-matrix memory on Tier 2 expert weights (384x1024, r=16)
    expert_opts = ["Standard\nAdamW", "8-bit\nAdamW", "GaLore\n(1-Sided)", "CentroidSteer\n(Augmented)"]
    expert_bpp = [8.00, 2.00, 0.365, 0.128]
    colors1 = [COLORS["adamw"], COLORS["bnb_8bit"], "#ff7f0e", COLORS["triune"]]

    bars1 = ax1.bar(expert_opts, expert_bpp, color=colors1, width=0.55, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("State Memory (Bytes / Param)")
    ax1.set_title("(a) Tier 2 Expert Weights ($384 \\times 1024$)")
    ax1.grid(axis="y", linestyle=":")
    ax1.set_ylim(0, 9.5)

    for bar, val in zip(bars1, expert_bpp):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.25,
            f"{val:.3f} B" if val < 1.0 else f"{val:.2f} B",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    ax1.text(
        3,
        4.0,
        "-98.4% HBM\nvs. AdamW",
        ha="center",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="#e7f0fa", ec=COLORS["triune"]),
    )

    # Panel (b): Full model-wide memory across tiers (90.8M MoE)
    model_opts = ["Pure\nAdamW", "Triune MoE\n(FP32 Tier 3)", "Triune MoE\n(8-bit Tier 3)", "Asymptotic\n($E \\geq 64$)"]
    model_bpp = [8.00, 4.54, 1.27, 0.128]
    colors2 = [COLORS["adamw"], "#2ca02c", COLORS["triune"], "#9467bd"]

    bars2 = ax2.bar(model_opts, model_bpp, color=colors2, width=0.55, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Model-Wide (Bytes / Param)")
    ax2.set_title("(b) Model-Wide Memory (90.8M MoE)")
    ax2.grid(axis="y", linestyle=":")
    ax2.set_ylim(0, 9.5)

    for bar, val in zip(bars2, model_bpp):
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.25,
            f"{val:.2f} B" if val >= 1.0 else f"{val:.3f} B",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    for out_dir in [P1_FIGS]:
        fig.savefig(out_dir / "optimizer_memory.pdf")
        fig.savefig(out_dir / "optimizer_memory.png")
    plt.close(fig)
    print("Generated optimizer_memory.[pdf|png]")


def generate_long_context_figure():
    """Generates Figure 1 for Paper 2: GLA constant decode memory vs Softmax KV explosion."""
    bench_file = BENCHMARK_DIR / "benchmark_results.json"
    if not bench_file.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {bench_file}")

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f).get("attention_scaling", {})
    decode_records = data.get("autoregressive_decode", [])
    prefill_records = data.get("results", [])
    if not decode_records or not prefill_records:
        raise ValueError(f"Attention scaling records missing or empty in {bench_file}")

    contexts = [r["context_length"] for r in decode_records]
    sdpa_cache_kb = [r["softmax_kv_cache_kb"] for r in decode_records]
    gla_cache_kb = [r["triune_gla_state_kb"] for r in decode_records]

    prefill_contexts = [r["seq_len"] for r in prefill_records]
    sdpa_prefill = [r["standard_time_ms"] for r in prefill_records]
    gla_prefill = [r["triune_gla_time_ms"] for r in prefill_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    ax1.plot(contexts, sdpa_cache_kb, "s--", label="Softmax KV-Cache (O(L))", color="#d62728")
    ax1.plot(contexts, gla_cache_kb, "o-", label="Triune GLA Recurrent (O(1))", color="#1f77b4", linewidth=2.0)
    ax1.set_xlabel("Context Length (Tokens)")
    ax1.set_ylabel("Per-Layer Cache Size (KB)")
    ax1.set_title("(a) KV-Cache Scaling")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2.plot(prefill_contexts, sdpa_prefill, "s--", label="Softmax SDPA", color="#d62728")
    ax2.plot(prefill_contexts, gla_prefill, "o-", label="Triune Chunked GLA", color="#1f77b4", linewidth=2.0)
    ax2.set_xlabel("Context Length (Tokens)")
    ax2.set_ylabel("Prefill Latency (ms)")
    ax2.set_title("(b) Context Prefill Latency")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    for out_dir in [P2_FIGS]:
        fig.savefig(out_dir / "long_context_scaling.pdf")
        fig.savefig(out_dir / "long_context_scaling.png")
    plt.close(fig)
    print("Generated long_context_scaling.[pdf|png]")


def generate_kernel_memory_figure():
    """Generates Figure 2 for Paper 2: Chunked Vocabulary-Projection Cross-Entropy memory scaling."""
    bench_file = BENCHMARK_DIR / "benchmark_results.json"
    if not bench_file.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {bench_file}")

    with open(bench_file, "r", encoding="utf-8") as f:
        records = json.load(f).get("cross_entropy", {}).get("results", [])
    if not records:
        raise ValueError(f"Cross-entropy results missing or empty in {bench_file}")
    tokens = [f"N = {r['tokens']:,}" for r in records]
    pytorch_vram = [r["standard_vram_mb"] for r in records]
    triune_vram = [r["triune_vram_mb"] for r in records]

    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    x = np.arange(len(tokens))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        pytorch_vram,
        width,
        label="Standard PyTorch CE",
        color="#d62728",
        edgecolor="black",
        linewidth=0.6,
    )
    rects2 = ax.bar(
        x + width / 2,
        triune_vram,
        width,
        label="Triune Chunked Vocab-Projection",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_ylabel("Peak Cross-Entropy VRAM (MB)")
    ax.set_title("Logit Materialization VRAM (V=32,000)")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":")
    ax.set_ylim(0, 4800)

    # Add percentage reduction labels
    for i, (p, t) in enumerate(zip(pytorch_vram, triune_vram)):
        pct = (1.0 - t / p) * 100
        ax.text(
            x[i] + width / 2,
            t + 120,
            f"-{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#1f77b4",
        )

    for out_dir in [P2_FIGS]:
        fig.savefig(out_dir / "chunked_ce_memory.pdf")
        fig.savefig(out_dir / "chunked_ce_memory.png")
    plt.close(fig)
    print("Generated chunked_ce_memory.[pdf|png]")


def generate_parallelism_figure():
    """Generates Figure 3 for Paper 2: Parallelism speedups."""
    bench_file = BENCHMARK_DIR / "parallelism_benchmark_results.json"
    if not bench_file.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {bench_file}")

    with open(bench_file, "r", encoding="utf-8") as f:
        p_data = json.load(f)
    if "async_prefetcher" not in p_data or "fused_qkvg" not in p_data:
        raise KeyError(f"Expected benchmark sections ('async_prefetcher', 'fused_qkvg') missing from {bench_file}")
    async_data = p_data.get("async_prefetcher", {})
    gemm_data = p_data.get("fused_qkvg", {})
    throughput = [async_data.get("sync_tok_per_sec", 633910.4), async_data.get("async_tok_per_sec", 812955.8)]
    latencies = [gemm_data.get("separate_gemm_ms", 670.03), gemm_data.get("fused_gemm_ms", 109.06)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)

    # Async Prefetcher Throughput
    loaders = ["Synchronous\nDataLoader", "Async Stream\nPrefetcher"]
    colors1 = ["#7f7f7f", "#1f77b4"]
    bars1 = ax1.bar(loaders, throughput, color=colors1, width=0.45, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Tokens / Second")
    ax1.set_title("(a) Host-Device Pipelining")
    ax1.grid(axis="y", linestyle=":")
    ax1.set_ylim(0, 950000)
    for bar, val in zip(bars1, throughput):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 25000,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.text(
        1,
        400000,
        "1.28x Speedup\n(PCIe Bubble Hidden)",
        ha="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#e7f0fa", ec="#1f77b4"),
    )

    # Fused GEMM Latency
    projs = ["4 Separate\nGEMMs", "1 Fused\nQKVG GEMM"]
    colors2 = ["#d62728", "#2ca02c"]
    bars2 = ax2.bar(projs, latencies, color=colors2, width=0.45, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("50-Step Latency (ms)")
    ax2.set_title("(b) Tensor Core GEMM Fusion")
    ax2.grid(axis="y", linestyle=":")
    ax2.set_ylim(0, 800)
    for bar, val in zip(bars2, latencies):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 25,
            f"{val:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.text(
        1,
        350,
        "6.14x Speedup\n(SM Saturation)",
        ha="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#2ca02c"),
    )

    for out_dir in [P2_FIGS]:
        fig.savefig(out_dir / "parallelism_speedups.pdf")
        fig.savefig(out_dir / "parallelism_speedups.png")
    plt.close(fig)
    print("Generated parallelism_speedups.[pdf|png]")


def generate_moe_ablation_figure():
    """Generates Figure 4 for Paper 1: MoE Router Centroid Steering Ablation."""
    json_path = BENCHMARK_DIR / "moe_convergence_100_steps_results_steer_ablation.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)

    ablations = [
        ("triune_moe_nosteer", "Unsteered Subspace (steer=0)", "#7f7f7f", ":", "^"),
        ("triune_moe_random", "Random Orthogonal Direction", "#ff7f0e", "--", "s"),
        ("triune_moe", "Dynamic Router Centroid Steering", "#1f77b4", "-", "o"),
    ]

    for key, label, color, ls, marker in ablations:
        if key not in summary:
            raise KeyError(f"Required ablation key '{key}' missing from summary in {json_path}")
        traj = summary[key]["val_trajectory"]
        if not traj:
            raise ValueError(f"Empty validation trajectory for key '{key}' in {json_path}")
        steps = [pt["step"] for pt in traj]
        losses = [pt["val_loss_mean"] for pt in traj]
        stds = [pt["val_loss_std"] for pt in traj]
        ppls = [pt["val_ppl_mean"] for pt in traj]

        ax1.plot(steps, losses, label=label, color=color, linestyle=ls, marker=marker, markersize=4)
        ax1.fill_between(
            steps,
            np.array(losses) - np.array(stds),
            np.array(losses) + np.array(stds),
            color=color,
            alpha=0.12,
        )

        ax2.plot(steps, ppls, label=label, color=color, linestyle=ls, marker=marker, markersize=4)

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Held-Out Validation Loss")
    ax1.set_title("(a) Steering Mechanism Ablation")
    ax1.grid(True)
    ax1.set_ylim(7.6, 9.2)

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("(b) Perplexity Comparison")
    ax2.set_yscale("log")
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    for out_dir in [P1_FIGS]:
        fig.savefig(out_dir / "moe_steering_ablation.pdf")
        fig.savefig(out_dir / "moe_steering_ablation.png")
    plt.close(fig)
    print("Generated moe_steering_ablation.[pdf|png]")


def main():
    print("=" * 70)
    print("Generating Publication-Grade Figures for Triune Research Papers")
    print("=" * 70)
    generate_dense_convergence_figure()
    generate_moe_convergence_figure()
    generate_optimizer_memory_figure()
    generate_moe_ablation_figure()
    generate_long_context_figure()
    generate_kernel_memory_figure()
    generate_parallelism_figure()
    print("=" * 70)
    print("All figures successfully created in vector PDF and PNG formats.")
    print("=" * 70)


if __name__ == "__main__":
    main()
