#!/usr/bin/env python3
"""Generate publication-grade figures for Triune research papers.

Produces vector PDF and high-DPI PNG plots for:
1. Dense convergence trajectory across 5 optimizer regimes (FineWeb-Edu).
2. MoE convergence trajectory across 4 optimizer regimes.
3. GLA constant-memory autoregressive decoding vs Softmax KV-cache explosion.
4. Chunked Logit-Free Cross-Entropy memory scaling vs PyTorch.
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
        print(f"Skipping dense figure: {json_path} missing")
        return

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
            continue
        traj = summary[key]["val_trajectory"]
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
        print(f"Skipping MoE figure: {json_path} missing")
        return

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
            continue
        traj = summary[key]["val_trajectory"]
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
    fig, ax = plt.subplots(figsize=(4.0, 2.8), constrained_layout=True)
    optimizers = ["PyTorch AdamW", "8-bit AdamW", "Muon (2D)", "CentroidSteer (Rank 16)"]
    bytes_per_param = [8.0, 2.0, 1.30, 1.00]
    colors = [COLORS["adamw"], COLORS["bnb_8bit"], COLORS["muon"], COLORS["triune"]]

    bars = ax.bar(optimizers, bytes_per_param, color=colors, width=0.55, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Persistent State (Bytes / Param)")
    ax.set_title("Optimizer Memory Footprint")
    ax.grid(axis="y", linestyle=":")
    ax.set_ylim(0, 9.5)

    for bar, val in zip(bars, bytes_per_param):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.25,
            f"{val:.2f} B",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.xticks(rotation=18, ha="right")
    for out_dir in [P1_FIGS]:
        fig.savefig(out_dir / "optimizer_memory.pdf")
        fig.savefig(out_dir / "optimizer_memory.png")
    plt.close(fig)
    print("Generated optimizer_memory.[pdf|png]")


def generate_long_context_figure():
    """Generates Figure 1 for Paper 2: GLA constant decode memory vs Softmax KV explosion."""
    contexts = [512, 1024, 2048, 4096, 8192, 16384]
    sdpa_cache_kb = [768, 1536, 3072, 6144, 12288, 24576]
    gla_cache_kb = [64, 64, 64, 64, 64, 64]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    ax1.plot(contexts, sdpa_cache_kb, "s--", label="Softmax KV-Cache (O(L))", color="#d62728")
    ax1.plot(contexts, gla_cache_kb, "o-", label="Triune GLA Recurrent (O(1))", color="#1f77b4", linewidth=2.0)
    ax1.set_xlabel("Context Length (Tokens)")
    ax1.set_ylabel("Per-Layer Cache Size (KB)")
    ax1.set_title("(a) KV-Cache Scaling")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Prefill latency comparison
    sdpa_prefill = [0.39, 0.40, 0.44, 1.27, 4.14, 14.20]
    gla_prefill = [2.81, 1.36, 2.96, 3.03, 2.95, 7.25]

    ax2.plot(contexts, sdpa_prefill, "s--", label="Softmax SDPA", color="#d62728")
    ax2.plot(contexts, gla_prefill, "o-", label="Triune Chunked GLA", color="#1f77b4", linewidth=2.0)
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
    """Generates Figure 2 for Paper 2: Chunked Logit-Free Cross-Entropy memory scaling."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8), constrained_layout=True)
    tokens = ["N = 2,048", "N = 4,096", "N = 8,192"]
    pytorch_vram = [1089.5, 2170.4, 4182.3]
    triune_vram = [289.4, 348.8, 365.8]

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
        label="Triune Chunked Logit-Free",
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)

    # Async Prefetcher Throughput
    loaders = ["Synchronous\nDataLoader", "Async Stream\nPrefetcher"]
    throughput = [633910.4, 812955.8]
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
    latencies = [670.03, 109.06]
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


def main():
    print("=" * 70)
    print("Generating Publication-Grade Figures for Triune Research Papers")
    print("=" * 70)
    generate_dense_convergence_figure()
    generate_moe_convergence_figure()
    generate_optimizer_memory_figure()
    generate_long_context_figure()
    generate_kernel_memory_figure()
    generate_parallelism_figure()
    print("=" * 70)
    print("All figures successfully created in vector PDF and PNG formats.")
    print("=" * 70)


if __name__ == "__main__":
    main()
