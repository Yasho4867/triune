#!/usr/bin/env python3
"""Measure CentroidSteer wall-clock refresh amortization and Grassmann chordal drift on 90.79M MoE."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lucid_machines.model.transformer import TriuneTransformer
from lucid_machines.optim.centroid import CentroidSteerOptimizer


def _hardware_info() -> dict[str, Any]:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "device_name": props.name,
            "total_vram_gb": round(props.total_memory / (1024**3), 2),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
        }
    return {"device_name": "CPU", "total_vram_gb": 0.0, "cuda_version": "none", "pytorch_version": torch.__version__}


def run_amortization_benchmark(
    gaps: tuple[int, ...] = (10, 20, 50, 100, 200, 500),
    batch_size: int = 4,
    seq_len: int = 128,
    n_non_refresh: int = 8,
    n_refresh_trials: int = 4,
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    torch.manual_seed(42)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = TriuneTransformer(
            vocab_size=32000,
            hidden_dim=384,
            num_layers=8,
            num_heads=6,
            head_dim=64,
            num_experts=4,
            top_k_experts=1,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=5,
        ).to(device=device, dtype=dtype)
    finally:
        torch.set_default_dtype(prev_dtype)

    model.train()
    x = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    y = torch.randint(0, 32000, (batch_size, seq_len), device=device).reshape(-1)
    loss_fn = nn.CrossEntropyLoss()

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Measure forward+backward time cleanly (shared across update_gap values)
    opt = CentroidSteerOptimizer(
        model=model,
        lr=3e-4,
        rank=16,
        update_gap=200,
        steer_scale=1.0,
        steer_update_gap=10,
        steer_mode="upgraded",
        num_steer_dirs=4,
        use_muon=True,
    )

    # Warmup (compile / Triton / allocators)
    for _ in range(3):
        opt.zero_grad()
        logits, _ = model(x, force_depth=2)
        loss = loss_fn(logits.reshape(-1, 32000), y)
        loss.backward()
        opt.step()
    _sync()

    # Measure forward+backward once per batch + steady non-refresh opt.step()
    fb_times = []
    non_refresh_opt_times = []
    for _ in range(n_non_refresh):
        opt.zero_grad()
        _sync()
        t0 = time.perf_counter()
        logits, _ = model(x, force_depth=2)
        loss = loss_fn(logits.reshape(-1, 32000), y)
        loss.backward()
        _sync()
        fb_times.append((time.perf_counter() - t0) * 1000.0)

        # Ensure next step stays non-refresh (and non-steer-refresh so pure steady step is isolated)
        opt.step_count = 15
        for g in opt.layer_groups:
            g["proj_step"] = 15
            g["last_steer_step"] = 15
        _sync()
        t1 = time.perf_counter()
        opt.step()
        _sync()
        non_refresh_opt_times.append((time.perf_counter() - t1) * 1000.0)

    # Measure Krylov sCholQR2 refresh step time + Grassmann steer transport step time
    refresh_opt_times = []
    for _ in range(n_refresh_trials):
        opt.zero_grad()
        logits, _ = model(x, force_depth=2)
        loss = loss_fn(logits.reshape(-1, 32000), y)
        loss.backward()
        opt.step_count = 200
        for g in opt.layer_groups:
            g["proj_step"] = 0
            g["last_steer_step"] = 0
        _sync()
        t2 = time.perf_counter()
        opt.step()
        _sync()
        refresh_opt_times.append((time.perf_counter() - t2) * 1000.0)

    # Run 30 consecutive real training steps at steer_update_gap=5 to measure steady Grassmann chordal drift
    opt_drift = CentroidSteerOptimizer(
        model=model,
        lr=3e-4,
        rank=16,
        update_gap=200,
        steer_scale=1.0,
        steer_update_gap=5,
        steer_mode="upgraded",
        num_steer_dirs=4,
        use_muon=True,
    )
    for _ in range(30):
        bx = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        by = torch.randint(0, 32000, (batch_size, seq_len), device=device).reshape(-1)
        opt_drift.zero_grad()
        logits, _ = model(bx, force_depth=2)
        loss = loss_fn(logits.reshape(-1, 32000), by)
        loss.backward()
        opt_drift.step()
    drift_stats = opt_drift.get_telemetry()["grassmann_transport"]

    fb_med = sorted(fb_times)[len(fb_times) // 2]
    non_ref_med = sorted(non_refresh_opt_times)[len(non_refresh_opt_times) // 2]
    ref_med = sorted(refresh_opt_times)[len(refresh_opt_times) // 2]
    refresh_penalty_ms = max(0.0, ref_med - non_ref_med)
    base_step_ms = fb_med + non_ref_med

    rows = []
    for gap in gaps:
        amortized_penalty = refresh_penalty_ms / float(gap)
        amortized_step_ms = base_step_ms + amortized_penalty
        overhead_pct = (amortized_penalty / amortized_step_ms) * 100.0
        rows.append(
            {
                "update_gap": int(gap),
                "fwd_bwd_ms": round(fb_med, 2),
                "non_refresh_opt_ms": round(non_ref_med, 2),
                "refresh_opt_ms": round(ref_med, 2),
                "amortized_refresh_ms": round(amortized_penalty, 3),
                "amortized_step_ms": round(amortized_step_ms, 2),
                "refresh_overhead_pct": round(overhead_pct, 2),
            }
        )

    payload = {
        "hardware": _hardware_info(),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "rank": 16,
        "num_steer_dirs": 4,
        "base_step_ms": round(base_step_ms, 2),
        "refresh_penalty_ms": round(refresh_penalty_ms, 2),
        "grassmann_drift_30steps": drift_stats,
        "sweep": rows,
    }
    out_p = REPO_ROOT / "benchmarks" / "centroidsteer_amortization_results.json"
    out_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved amortization benchmark to {out_p}")
    for row in rows:
        print(row)
    print("Grassmann drift:", drift_stats)
    return payload


if __name__ == "__main__":
    run_amortization_benchmark()
