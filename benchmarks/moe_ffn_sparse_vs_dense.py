"""Measure the production MoE_FFN eval path (dense capacity-gather bmm) against a true top-k sparse path."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triune.model.moe import MoE_FFN  # noqa: E402

CONFIGS = {
    "default_1536d_8e_top1": dict(dim=1536, num_experts=8, top_k=1),
    "default_1536d_8e_top2": dict(dim=1536, num_experts=8, top_k=2),
    "checkpoint_384d_4e_top1": dict(dim=384, num_experts=4, top_k=1),
}


@torch.no_grad()
def sparse_forward(moe: MoE_FFN, x: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    flat_x = x.reshape(-1, D)
    logits = moe.router(flat_x)
    _, top_idx = torch.topk(logits + moe.expert_bias, moe.top_k, dim=-1)
    weights = torch.gather(F.softmax(logits, dim=-1), 1, top_idx)
    if moe.top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    expert_ids = top_idx.reshape(-1)
    order = expert_ids.argsort(stable=True)
    token_ids = torch.arange(flat_x.size(0), device=x.device).repeat_interleave(moe.top_k)[order]
    counts = torch.bincount(expert_ids, minlength=moe.num_experts).tolist()
    chunks = flat_x[token_ids].split(counts)
    y = torch.cat([moe.experts[e](c) for e, c in enumerate(chunks) if c.numel()])
    y = y * weights.reshape(-1)[order].unsqueeze(-1).to(y.dtype)
    out = torch.zeros_like(flat_x).index_add_(0, token_ids, y)
    if moe.shared_expert:
        out = out + moe.shared_scale * moe.shared(flat_x)
    return out.reshape(B, T, D)


def time_ms(fn, reps: int, device: torch.device) -> float:
    for _ in range(3):
        fn()
    samples = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def run(cfg: dict, token_counts: list[int], dtype: torch.dtype, device: torch.device, reps: int) -> list[dict]:
    torch.manual_seed(0)
    moe = MoE_FFN(**cfg).to(device=device, dtype=dtype).eval()
    moe.router.weight.data.normal_(0, 0.5)
    rows = []
    for n in token_counts:
        x = torch.randn(1, n, cfg["dim"], device=device, dtype=dtype)
        with torch.no_grad():
            dense_out, sparse_out = moe(x), sparse_forward(moe, x)
            max_diff = (dense_out.float() - sparse_out.float()).abs().max().item()
            dense_ms = time_ms(lambda: moe(x), reps, device)
            sparse_ms = time_ms(lambda: sparse_forward(moe, x), reps, device)
        rows.append(
            {
                "tokens": n,
                "dense_ms": round(dense_ms, 3),
                "sparse_ms": round(sparse_ms, 3),
                "sparse_speedup_x": round(dense_ms / sparse_ms, 3),
                "max_abs_diff": max_diff,
            }
        )
    del moe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 256, 1024, 4096])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output-json", default="benchmarks/moe_ffn_sparse_vs_dense.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    result = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "mode": "eval, MoE_FFN.forward (dense) vs sparse_forward (sort-by-expert, per-expert matmul)",
        "configs": {name: run(cfg, args.tokens, dtype, device, args.reps) for name, cfg in CONFIGS.items()},
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, rows in result["configs"].items():
        print(name)
        for r in rows:
            print(f"  N={r['tokens']:>6} dense={r['dense_ms']:>9.3f}ms sparse={r['sparse_ms']:>9.3f}ms x{r['sparse_speedup_x']}  diff={r['max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()
