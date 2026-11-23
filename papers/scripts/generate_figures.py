#!/usr/bin/env python3
"""Generate publication-grade figures for Triune research papers.

Produces vector PDF and high-DPI PNG plots for:
1. Dense convergence trajectory across 5 optimizer regimes (FineWeb-Edu, N=3).
2. MoE convergence trajectory across 6 optimizer regimes (FineWeb-Edu, N=8).
3. Measured optimizer state memory footprint across tiers (N=8 90.8M MoE + single-matrix).
4. MoE centroid steering ablation & ANOVA dispersion decomposition (N=8).
5. GLA constant-memory decoding & full-model 16k prefill scaling.
6. Chunked Vocabulary-Projection Cross-Entropy memory scaling vs PyTorch & Unsloth.
7. Multi-level Parallelism speedups (Async DMA prefetching & Fused QKVG).
8. Activation-checkpointed LayerStreamingEngine VRAM scaling (102M to 1.5B on 7.93 GiB GPU).
9. Held-out conformal CALM Pareto frontier (N=85,503) & 4-tier Hadamard NVFP4 quantization.
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
    "galore": "#17becf",  # Teal
    "badam": "#e377c2",  # Pink
    "triune_moe_nosteer": "#7f7f7f",  # Gray
    "triune_moe_random": "#bcbd22",  # Olive
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
    """Generates Figure 2 for Paper 1: MoE multi-seed convergence (N=8)."""
    json_path = BENCHMARK_DIR / "moe_convergence_100_steps_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    regimes_to_plot = [
        ("badam", "BAdam (Blockwise)", COLORS["badam"], ":", "^"),
        ("adamw", "Standard AdamW", COLORS["adamw"], "--", "s"),
        ("muon_adamw", "Muon + AdamW", COLORS["muon_adamw"], ":", "v"),
        ("galore", "GaLore (r=16)", COLORS["galore"], "-.", "P"),
        ("centroid_no_muon", "CentroidSteer (No Muon)", COLORS["centroid_no_muon"], "-.", "d"),
        ("triune_moe", "Triune MoE (3-Tier)", COLORS["triune_moe"], "-", "o"),
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
            alpha=0.10,
        )

        ax2.plot(steps, ppls, label=label, color=color, linestyle=ls, marker=marker, markersize=4)

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Held-Out Validation Loss")
    ax1.set_title("(a) 90.8M MoE Convergence (N=8)")
    ax1.grid(True)
    ax1.set_ylim(7.55, 10.6)

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("(b) MoE Perplexity Trajectory")
    ax2.set_yscale("log")
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.92)

    for out_dir in [P1_FIGS]:
        fig.savefig(out_dir / "moe_convergence.pdf")
        fig.savefig(out_dir / "moe_convergence.png")
    plt.close(fig)
    print("Generated moe_convergence.[pdf|png]")


def generate_optimizer_memory_figure():
    """Generates Figure 3 for Paper 1: Measured 90.8M MoE & analytical Tier-2 state memory."""
    json_path = BENCHMARK_DIR / "moe_convergence_100_steps_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)

    # Panel (a): Measured Tier 2 expert state bytes/param (vs analytical FP32 AdamW 8.0 B/p baseline)
    expert_opts = ["AdamW\n(BF16)", "BAdam\n(Block)", "GaLore\n(r=16)", "Centroid\n(No Muon)", "Triune MoE\n(3-Tier)"]
    expert_keys = ["adamw", "badam", "galore", "centroid_no_muon", "triune_moe"]
    expert_bpp = [summary[k]["expert_bytes_per_param"]["mean"] for k in expert_keys]
    colors1 = [COLORS["adamw"], COLORS["badam"], COLORS["galore"], COLORS["centroid_no_muon"], COLORS["triune_moe"]]

    bars1 = ax1.bar(expert_opts, expert_bpp, color=colors1, width=0.56, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Expert State (Bytes / Param)", fontsize=10)
    ax1.set_title("(a) Tier 2 Expert State (90.8M)", fontsize=10.5)
    ax1.tick_params(axis="x", labelsize=7.2)
    ax1.grid(axis="y", linestyle=":")
    ax1.set_ylim(0, 4.9)

    for bar, val in zip(bars1, expert_bpp):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.12,
            f"{val:.4f} B" if val < 0.5 else f"{val:.2f} B",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
        )
    ax1.text(
        2.9,
        2.3,
        "-97.2% vs BF16 AdamW\n(-98.6% vs FP32 8.0 B/p)",
        ha="center",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#e7f0fa", ec=COLORS["triune"]),
    )

    # Panel (b): Measured model-wide total state memory (MB) across 90.8M MoE regimes
    model_opts = [
        "AdamW\n(BF16)",
        "Muon+\nAdamW",
        "Triune\nMoE",
        "GaLore\n(8b Base)",
        "Centroid\n(8b Base)",
        "BAdam\n(Block)",
    ]
    model_keys = ["adamw", "muon_adamw", "triune_moe", "galore", "centroid_no_muon", "badam"]
    model_mb = [summary[k]["total_state_mb"]["mean"] for k in model_keys]
    colors2 = [
        COLORS["adamw"],
        COLORS["muon_adamw"],
        COLORS["triune_moe"],
        COLORS["galore"],
        COLORS["centroid_no_muon"],
        COLORS["badam"],
    ]

    bars2 = ax2.bar(model_opts, model_mb, color=colors2, width=0.56, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Total Optimizer State (MB)", fontsize=10)
    ax2.set_title("(b) Total Model State (90.8M)", fontsize=10.5)
    ax2.tick_params(axis="x", labelsize=7.0)
    ax2.grid(axis="y", linestyle=":")
    ax2.set_ylim(0, 305)

    for bar, val in zip(bars2, model_mb):
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 7,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    for out_dir in [P1_FIGS]:
        fig.savefig(out_dir / "optimizer_memory.pdf")
        fig.savefig(out_dir / "optimizer_memory.png")
    plt.close(fig)
    print("Generated optimizer_memory.[pdf|png]")


def generate_long_context_figure():
    """Generates Figure 1 for Paper 2: GLA kernel scaling & full-model 16k E2E prefill."""
    lc_file = BENCHMARK_DIR / "long_context_results.json"
    e2e_file = BENCHMARK_DIR / "full_model_e2e.json"
    if not lc_file.exists() or not e2e_file.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {lc_file} or {e2e_file}")

    with open(lc_file, "r", encoding="utf-8") as f:
        lc_data = json.load(f)
    with open(e2e_file, "r", encoding="utf-8") as f:
        e2e_data = json.load(f)

    decode_records = lc_data.get("decode_benchmarks", [])
    prefill_records = lc_data.get("prefill_benchmarks", [])
    e2e_prefill = e2e_data.get("prefill_benchmarks", [])
    if not decode_records or not prefill_records or not e2e_prefill:
        raise ValueError("Long-context or E2E prefill records missing or empty")

    contexts = [r["prompt_len"] for r in decode_records]
    sdpa_cache_kb = [r["sdpa_kv_cache_kb"] for r in decode_records]
    gla_cache_kb = [r["gla_state_kb"] for r in decode_records]

    prefill_contexts = [r["seq_len"] for r in prefill_records]
    sdpa_prefill = [r["sdpa"]["latency_ms"] for r in prefill_records]
    gla_prefill = [r["gla"]["latency_ms"] for r in prefill_records]

    b1_rows = [r for r in e2e_prefill if r["batch_size"] == 1]
    b4_rows = [r for r in e2e_prefill if r["batch_size"] == 4]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 2.8), constrained_layout=True)

    ax1.plot(contexts, sdpa_cache_kb, "s--", label="Softmax KV-Cache (O(L))", color="#d62728")
    ax1.plot(contexts, gla_cache_kb, "o-", label="Triune GLA State (O(1))", color="#1f77b4", linewidth=2.0)
    ax1.set_xlabel("Context Length (Tokens)")
    ax1.set_ylabel("Per-Layer Cache Size (KiB)")
    ax1.set_title("(a) KV-Cache Scaling (H=8, d=512)")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend(loc="center right", fontsize=7.5)

    ax2.plot(prefill_contexts, sdpa_prefill, "s--", label="Softmax SDPA", color="#d62728")
    ax2.plot(prefill_contexts, gla_prefill, "o-", label="Triune Chunked GLA", color="#1f77b4", linewidth=2.0)
    ax2.set_xlabel("Context Length (Tokens)")
    ax2.set_ylabel("Kernel Latency (ms)")
    ax2.set_title("(b) Multi-Head Prefill Latency")
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=7.5)

    ax3.plot(
        [r["seq_len"] for r in b1_rows],
        [r["tokens_per_sec"] / 1000.0 for r in b1_rows],
        "o-",
        label="Full 90.8M (B=1)",
        color="#1f77b4",
        linewidth=2.0,
    )
    ax3.plot(
        [r["seq_len"] for r in b4_rows],
        [r["tokens_per_sec"] / 1000.0 for r in b4_rows],
        "^-.",
        label="Full 90.8M (B=4)",
        color="#2ca02c",
        linewidth=1.8,
    )
    ax3.set_xlabel("Context Length (Tokens)")
    ax3.set_ylabel("E2E Throughput (k tok/s)")
    ax3.set_title("(c) 90.8M Full-Model E2E Prefill")
    ax3.set_ylim(0, 340)
    ax3.grid(True)
    ax3.legend(loc="lower right", fontsize=7.5)

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
    unsloth_vram = [r["unsloth_vram_mb"] for r in records]
    triune_vram = [r["triune_vram_mb"] for r in records]

    fig, ax = plt.subplots(figsize=(4.6, 2.8), constrained_layout=True)
    x = np.arange(len(tokens))
    width = 0.27

    ax.bar(
        x - width,
        pytorch_vram,
        width,
        label="Standard PyTorch CE",
        color="#d62728",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x,
        unsloth_vram,
        width,
        label="Unsloth (2026.10.10)",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        x + width,
        triune_vram,
        width,
        label="Triune Chunked CE",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_ylabel("Peak Cross-Entropy VRAM (MB)")
    ax.set_title("Logit Materialization VRAM (V=32,000)")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(axis="y", linestyle=":")
    ax.set_ylim(0, 4800)

    for i, (p, t) in enumerate(zip(pytorch_vram, triune_vram)):
        pct = (1.0 - t / p) * 100
        ax.text(
            x[i] + width,
            t + 120,
            f"-{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
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
    async_data = p_data["async_prefetcher"]
    gemm_data = p_data["fused_qkvg"]
    for k in ("sync_tok_per_sec", "async_tok_per_sec"):
        if k not in async_data:
            raise KeyError(f"Missing required key '{k}' in async_prefetcher of {bench_file}")
    for k in ("separate_gemm_ms", "fused_gemm_ms"):
        if k not in gemm_data:
            raise KeyError(f"Missing required key '{k}' in fused_qkvg of {bench_file}")

    throughput = [async_data["sync_tok_per_sec"], async_data["async_tok_per_sec"]]
    latencies = [gemm_data["separate_gemm_ms"], gemm_data["fused_gemm_ms"]]

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
    async_speedup = throughput[1] / max(throughput[0], 1e-12)
    gemm_speedup = latencies[0] / max(latencies[1], 1e-12)

    ax1.text(
        1,
        400000,
        f"{async_speedup:.2f}x Speedup\n(PCIe Bubble Hidden)",
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
        f"{gemm_speedup:.2f}x Speedup\n(SM Saturation)",
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
    """Generates Figure 4 for Paper 1: Steering ablation (N=8) & ANOVA energy decomposition."""
    json_path = BENCHMARK_DIR / "moe_convergence_100_steps_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)

    ablations = [
        ("triune_moe_nosteer", "Unsteered Subspace (steer=0)", COLORS["triune_moe_nosteer"], ":", "^"),
        ("triune_moe_random", "Random Orthogonal Steering", COLORS["triune_moe_random"], "--", "s"),
        ("triune_moe", "Dynamic Centroid Steering", COLORS["triune_moe"], "-", "o"),
    ]

    for key, label, color, ls, marker in ablations:
        if key not in summary:
            raise KeyError(f"Required ablation key '{key}' missing from summary in {json_path}")
        traj = summary[key]["val_trajectory"]
        if not traj:
            raise ValueError(f"Empty validation trajectory for key '{key}' in {json_path}")
        steps = [pt["step"] for pt in traj if pt["step"] >= 20]
        losses = [pt["val_loss_mean"] for pt in traj if pt["step"] >= 20]
        stds = [pt["val_loss_std"] for pt in traj if pt["step"] >= 20]

        ax1.plot(steps, losses, label=label, color=color, linestyle=ls, marker=marker, markersize=4)
        ax1.fill_between(
            steps,
            np.array(losses) - np.array(stds),
            np.array(losses) + np.array(stds),
            color=color,
            alpha=0.10,
        )

    ax1.set_xlabel("Training Steps (Post-Warmup View)")
    ax1.set_ylabel("Held-Out Validation Loss")
    ax1.set_title("(a) Steering Equivalence (N=8, p=0.12)")
    ax1.grid(True)
    ax1.set_ylim(7.55, 8.40)
    ax1.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.92)

    # Panel (b): Measured ANOVA Second-Moment Identity E[||c_e||^2] + E[tr(Sigma_e)] = d = 384
    geo_keys = ["adamw", "galore", "triune_moe_random", "triune_moe_nosteer", "triune_moe"]
    geo_labels = ["AdamW", "GaLore", "Random\nSteer", "Unsteered", "Centroid\nSteer"]
    cent_sq = [summary[k]["routing_concentration"]["mean_centroid_norm_sq"]["mean"] for k in geo_keys]
    within_tr = [summary[k]["routing_concentration"]["mean_within_cov_trace"]["mean"] for k in geo_keys]
    x = np.arange(len(geo_keys))
    ax2.bar(
        x, cent_sq, width=0.52, label=r"Between-Expert $\|\mu_e\|^2$", color="#1f77b4", edgecolor="black", linewidth=0.5
    )
    ax2.bar(
        x,
        within_tr,
        width=0.52,
        bottom=cent_sq,
        label=r"Within-Expert $\mathrm{tr}(\Sigma_e)$",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.axhline(384.0, color="black", linestyle="--", linewidth=0.9, label=r"RMSNorm Energy ($d=384$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(geo_labels, fontsize=7.5)
    ax2.set_ylabel("Mean Energy per Routed Token")
    ax2.set_title("(b) ANOVA Conservation (N=8, d=384)", fontsize=10.5)
    ax2.set_ylim(0, 445)
    ax2.grid(axis="y", linestyle=":")
    ax2.legend(loc="lower right", fontsize=7.0, framealpha=0.92)

    for out_dir in [P1_FIGS]:
        fig.savefig(out_dir / "moe_steering_ablation.pdf")
        fig.savefig(out_dir / "moe_steering_ablation.png")
    plt.close(fig)
    print("Generated moe_steering_ablation.[pdf|png]")


def generate_layer_streaming_figure():
    """Generates Figure 4 for Paper 2: Activation-checkpointed LayerStreamingEngine VRAM scaling."""
    json_path = BENCHMARK_DIR / "layer_streaming_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)["comparisons"]

    scales = [f"{r['params_m']:.0f}M" if r["params_m"] < 1000 else f"{r['params_m'] / 1000:.2f}B" for r in results]
    gpu_vram = [0.0 if r["gpu_resident"].get("oom") else r["gpu_resident"]["peak_allocated_mb"] for r in results]
    stream_vram = [r["layer_streaming"]["peak_allocated_mb"] for r in results]
    is_oom = [bool(r["gpu_resident"].get("oom")) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)
    x = np.arange(len(scales))
    width = 0.36

    ax1.bar(
        x - width / 2, gpu_vram, width, label="GPU-Resident Training", color="#d62728", edgecolor="black", linewidth=0.6
    )
    ax1.bar(
        x + width / 2,
        stream_vram,
        width,
        label="LayerStreaming + Act. Ckpt",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.axhline(8120.0, color="#333333", linestyle="--", linewidth=1.1, label="7.93 GiB Limit (RTX 5070)")

    for i, (oom, gval, sval) in enumerate(zip(is_oom, gpu_vram, stream_vram)):
        if oom:
            ax1.text(
                x[i] - width / 2,
                5600,
                "GPU\nOOM\n(>14G)",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#b00020",
            )
        else:
            lbl = f"{gval:,.0f}" if gval < 8120 else f"{gval:,.0f}\n(WDDM)"
            ax1.text(
                x[i] - width / 2,
                gval + 180,
                lbl,
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                color="#a61c1c",
            )
        ax1.text(
            x[i] + width / 2,
            sval + 180,
            f"{sval:,.0f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="#1f77b4",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(scales, fontsize=8.0)
    ax1.set_ylabel("Peak Allocated VRAM (MB)", fontsize=10)
    ax1.set_title("(a) Consumer GPU VRAM (B=2, T=512)", fontsize=10.5)
    ax1.set_ylim(0, 11800)
    ax1.grid(axis="y", linestyle=":")
    ax1.legend(loc="upper right", fontsize=6.8, framealpha=0.94)

    # Panel (b): Step Latency (271M WDDM spill crossover + 695M / 1.5B streaming speeds)
    lat_gpu = [
        0.0 if r["gpu_resident"].get("oom") else r["gpu_resident"]["median_latency_ms"] / 1000.0 for r in results
    ]
    lat_str = [r["layer_streaming"]["median_latency_ms"] / 1000.0 for r in results]
    ax2.bar(x - width / 2, lat_gpu, width, label="GPU-Resident", color="#d62728", edgecolor="black", linewidth=0.6)
    ax2.bar(
        x + width / 2,
        lat_str,
        width,
        label="LayerStreaming + Act. Ckpt",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
    )
    for i, (oom, gsec, ssec) in enumerate(zip(is_oom, lat_gpu, lat_str)):
        if oom:
            ax2.text(
                x[i] - width / 2,
                0.25,
                "OOM",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color="#b00020",
            )
        else:
            ax2.text(
                x[i] - width / 2, gsec + 0.12, f"{gsec:.2f}s", ha="center", va="bottom", fontsize=7.2, fontweight="bold"
            )
        ax2.text(
            x[i] + width / 2,
            ssec + 0.12,
            f"{ssec:.2f}s",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="#1f77b4",
        )
    ax2.annotate(
        "1.97x faster\n(avoids WDDM)",
        xy=(1 + width / 2, lat_str[1] + 0.45),
        xytext=(1.05, 4.35),
        ha="center",
        fontsize=7.0,
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="#e7f0fa", ec="#1f77b4"),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(scales, fontsize=8.0)
    ax2.set_ylabel("Median Training Step (s)", fontsize=10)
    ax2.set_title("(b) Step Latency & Paging Crossover", fontsize=10.5)
    ax2.set_ylim(0, 6.2)
    ax2.grid(axis="y", linestyle=":")
    ax2.legend(loc="upper left", fontsize=7.0, framealpha=0.92)

    for out_dir in [P2_FIGS]:
        fig.savefig(out_dir / "layer_streaming_scaling.pdf")
        fig.savefig(out_dir / "layer_streaming_scaling.png")
    plt.close(fig)
    print("Generated layer_streaming_scaling.[pdf|png]")


def generate_calm_quant_figure():
    """Generates Figure 5 for Paper 2: Conformal CALM Pareto (N=85,503) & 4-tier NVFP4 quantization."""
    calm_file = BENCHMARK_DIR / "calm_calibration.json"
    e2e_file = BENCHMARK_DIR / "full_model_e2e.json"
    if not calm_file.exists() or not e2e_file.exists():
        raise FileNotFoundError("Missing calm_calibration.json or full_model_e2e.json")

    with open(calm_file, "r", encoding="utf-8") as f:
        calm_data = json.load(f)
    with open(e2e_file, "r", encoding="utf-8") as f:
        e2e_data = json.load(f)

    pareto = calm_data["pareto_sweep"]
    saved_pct = [r["compute_saved_pct"] for r in pareto]
    calm_ppl = [r["calm_ppl"] for r in pareto]

    tiers = e2e_data["quantization_tiers"]["tiers"]
    tier_labels = ["BF16\n(16.0b)", "FP8 E4M3\n(8.0b)", "NVFP4\nHadamard", "NVFP4\nUnrot."]
    tier_mb = [t["layer_linear_mb"] for t in [tiers[0], tiers[1], tiers[3], tiers[2]]]
    tier_ppl = [t["fineweb_ppl"] for t in [tiers[0], tiers[1], tiers[3], tiers[2]]]
    tier_cos = [t["logit_cosine_similarity"] for t in [tiers[0], tiers[1], tiers[3], tiers[2]]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)

    ax1.plot(saved_pct, calm_ppl, "o-", color="#1f77b4", linewidth=2.0, markersize=4.5)
    for r in pareto:
        if r["tau_value"] in (0.05, 0.1, 0.2):
            offset = (-44, 9) if r["tau_value"] == 0.05 else (-32, 9)
            ax1.annotate(
                f"tau={r['tau_value']}",
                xy=(r["compute_saved_pct"], r["calm_ppl"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.3,
                color="#1f77b4",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=0.7),
            )
    ax1.set_xlabel("Transformer Layer FLOPs Saved (%)")
    ax1.set_ylabel("Held-Out Validation PPL")
    ax1.set_title("(a) Conformal CALM Pareto (N=85,503)", fontsize=10.5)
    ax1.grid(True)

    colors_q = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax2.bar(tier_labels, tier_mb, width=0.52, color=colors_q, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Linear Weight Memory (MB)")
    ax2.set_title("(b) 4-Tier Quantization vs. PPL", fontsize=10.5)
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.set_ylim(0, 192)
    ax2.grid(axis="y", linestyle=":")
    for bar, mb, ppl, cos in zip(bars, tier_mb, tier_ppl, tier_cos):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            mb + 4,
            f"{mb:.1f} MB\nPPL {ppl:.1f}\ncos={cos:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
        )

    for out_dir in [P2_FIGS]:
        fig.savefig(out_dir / "calm_quant_tradeoff.pdf")
        fig.savefig(out_dir / "calm_quant_tradeoff.png")
    plt.close(fig)
    print("Generated calm_quant_tradeoff.[pdf|png]")


def generate_centroidsteer_5seed_figure():
    """Generates Paper 1 Figure: 5-seed Direct-Sum + Block-Oja CentroidSteer validation (Suites 1-4)."""
    json_path = BENCHMARK_DIR / "centroidsteer_upgrade_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    s1 = data["suite1_mechanistic_diagnostics"]["block_oja_energy_capture"]
    s2 = data["suite2_multidomain_moe"]["summary"]
    s3 = data["suite3_fineweb_cold_start"]["summary"]
    s4 = data["suite4_fineweb_continuation"]["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)

    # Panel (a): Suite 3 FineWeb-Edu Cold-Start LM (5 seeds, 0.264 B/p vs baselines)
    regimes = [
        ("nosteer", "No-Steer\nLow-Rank\n(0.046 B/p)", COLORS["triune_moe_nosteer"]),
        ("random_same_mem", "Random\nSubspace\n(0.264 B/p)", COLORS["triune_moe_random"]),
        ("upgraded", "CentroidSteer\nDirect-Sum\n(0.264 B/p)", COLORS["triune"]),
        ("adamw", "Full-Rank\nAdamW\n(4.000 B/p)", COLORS["adamw"]),
    ]
    labels = [r[1] for r in regimes]
    means = [s3[r[0]]["final_val_loss"]["mean"] for r in regimes]
    stds = [s3[r[0]]["final_val_loss"]["std"] for r in regimes]
    colors = [r[2] for r in regimes]
    x = np.arange(len(regimes))

    bars = ax1.bar(x, means, yerr=stds, capsize=3, width=0.55, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.2)
    ax1.set_ylabel("Validation Loss (Lower is Better)", fontsize=9.5)
    ax1.set_title("(a) Suite 3 FineWeb-Edu LM (N=5 Seeds)", fontsize=10.2)
    ax1.set_ylim(2.05, 2.34)
    ax1.grid(axis="y", linestyle=":")
    for bar, m in zip(bars, means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            m + 0.022,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
        )
    ax1.text(
        1.5,
        2.305,
        "p=0.0112 vs No-Steer | p=0.0006 vs Random (5/5 seeds)",
        ha="center",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.22", fc="#e7f0fa", ec=COLORS["triune"]),
    )

    # Panel (b): Block-Oja Subspace Energy Capture across Suites 1-4
    suites_lbl = ["Suite 1\nGrad.", "Suite 2\nMulti-Dom.", "Suite 3\nFineWeb LM", "Suite 4\n75.8M Cont."]
    cs_energy = [
        s1["v3_block_oja_grad_capture_frac"] * 100.0,
        s2["upgraded"]["mean_border_energy_frac"]["mean"] * 100.0,
        s3["upgraded"]["mean_border_energy_frac"]["mean"] * 100.0,
        s4["upgraded"]["mean_border_energy_frac"]["mean"] * 100.0,
    ]
    rnd_energy = [
        s1["random_same_mem_grad_capture_frac"] * 100.0,
        s2["random_same_mem"]["mean_border_energy_frac"]["mean"] * 100.0,
        s3["random_same_mem"]["mean_border_energy_frac"]["mean"] * 100.0,
        s4["random_same_mem"]["mean_border_energy_frac"]["mean"] * 100.0,
    ]
    x2 = np.arange(len(suites_lbl))
    w = 0.36
    b_rnd = ax2.bar(
        x2 - w / 2,
        rnd_energy,
        w,
        label="Memory-Matched Random",
        color=COLORS["triune_moe_random"],
        edgecolor="black",
        linewidth=0.6,
    )
    b_cs = ax2.bar(
        x2 + w / 2,
        cs_energy,
        w,
        label="CentroidSteer Block-Oja",
        color=COLORS["triune"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(suites_lbl, fontsize=7.5)
    ax2.set_ylabel("Subspace Energy Capture (%)", fontsize=9.5)
    ax2.set_title("(b) Block-Oja Gradient Energy Capture", fontsize=10.2)
    ax2.set_ylim(0, 108)
    ax2.grid(axis="y", linestyle=":")
    ax2.legend(loc="upper right", fontsize=7.0, framealpha=0.92)
    for bar, v in zip(b_rnd, rnd_energy):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.6)
    for bar, v in zip(b_cs, cs_energy):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 1.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color=COLORS["triune"],
        )

    fig.savefig(P1_FIGS / "centroidsteer_5seed_validation.pdf")
    fig.savefig(P1_FIGS / "centroidsteer_5seed_validation.png")
    plt.close(fig)
    print("Generated centroidsteer_5seed_validation.[pdf|png]")


def main():
    print("=" * 70)
    print("Generating Publication-Grade Figures for Lucid Machines Research Papers")
    print("=" * 70)
    generate_dense_convergence_figure()
    generate_moe_convergence_figure()
    generate_optimizer_memory_figure()
    generate_moe_ablation_figure()
    generate_centroidsteer_5seed_figure()
    generate_long_context_figure()
    generate_kernel_memory_figure()
    generate_parallelism_figure()
    generate_layer_streaming_figure()
    generate_calm_quant_figure()
    print("=" * 70)
    print("All figures successfully created in vector PDF and PNG formats.")
    print("=" * 70)


if __name__ == "__main__":
    main()
