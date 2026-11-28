#!/usr/bin/env python3
"""Measure PCIe async double-buffered prefetch overlap vs synchronous streaming on RTX 5070."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_layer_streaming import benchmark_step, create_model_variant, get_hardware_info

from lucid_machines.runtime.streaming import LayerStreamingEngine, StreamingConfig


def _measure_single_layer_h2d_ms(model: torch.nn.Module, device: torch.device, reps: int = 5) -> float:
    layer = model.layers[0]
    cpu_tensors = [p.detach().cpu().contiguous().pin_memory() for p in layer.parameters()] if device.type == "cuda" else [p.detach().cpu() for p in layer.parameters()]
    gpu_bufs = [torch.empty_like(t, device=device) for t in cpu_tensors]
    for c, g in zip(cpu_tensors, gpu_bufs):
        g.copy_(c)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for c, g in zip(cpu_tensors, gpu_bufs):
            g.copy_(c, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return round(times[len(times) // 2], 2)


def run_pcie_overlap_benchmark() -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    batch_size, seq_len, vocab_size = 2, 512, 4096

    results: dict[str, Any] = {
        "hardware": get_hardware_info(),
        "config": {"batch_size": batch_size, "seq_len": seq_len, "vocab_size": vocab_size, "dtype": "bfloat16"},
        "scales": [],
    }

    for scale in ("102M", "271M"):
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        m_probe, meta = create_model_variant(scale, vocab_size=vocab_size, dtype=dtype)
        layer_mb = sum(p.numel() * p.element_size() for p in m_probe.layers[0].parameters()) / (1024 * 1024)
        single_layer_h2d_ms = _measure_single_layer_h2d_ms(m_probe, device)
        del m_probe
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        mode_results: dict[str, Any] = {}
        for async_pf in (False, True):
            m_str, _ = create_model_variant(scale, vocab_size=vocab_size, dtype=dtype)
            opt_str = torch.optim.AdamW(m_str.parameters(), lr=1e-4)
            eng = LayerStreamingEngine(
                m_str,
                device=device,
                config=StreamingConfig(
                    enabled=True,
                    chunk_size=1,
                    pin_memory=(device.type == "cuda"),
                    async_prefetch=async_pf,
                    activation_checkpointing=True,
                    instant_optimizer=True,
                    layers_path="layers",
                ),
            )
            eng.attach(optimizer=opt_str)
            metrics = benchmark_step(m_str, opt_str, inputs, targets, engine=eng, warmup=2, steps=3)
            mode_results["async_prefetch_on" if async_pf else "async_prefetch_off"] = metrics
            del eng, opt_str, m_str
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        sync_ms = mode_results["async_prefetch_off"]["median_latency_ms"]
        async_ms = mode_results["async_prefetch_on"]["median_latency_ms"]
        saved_ms = round(sync_ms - async_ms, 2)
        speedup = round(sync_ms / max(1e-6, async_ms), 3)

        results["scales"].append(
            {
                **meta,
                "layer_params_mb": round(layer_mb, 2),
                "single_layer_h2d_ms": single_layer_h2d_ms,
                "num_layers_h2d_unhidden_est_ms": round(single_layer_h2d_ms * meta["num_layers"] * 3, 2),
                **mode_results,
                "prefetch_saved_ms": saved_ms,
                "prefetch_speedup": speedup,
            }
        )

    out_p = REPO_ROOT / "benchmarks" / "streaming_pcie_overlap_results.json"
    out_p.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved PCIe overlap benchmark to {out_p}")
    for row in results["scales"]:
        print(
            row["scale"],
            "h2d/layer=", row["single_layer_h2d_ms"], "ms |",
            "off=", row["async_prefetch_off"]["median_latency_ms"], "ms |",
            "on=", row["async_prefetch_on"]["median_latency_ms"], "ms |",
            "speedup=", row["prefetch_speedup"], "x",
        )
    return results


if __name__ == "__main__":
    run_pcie_overlap_benchmark()
