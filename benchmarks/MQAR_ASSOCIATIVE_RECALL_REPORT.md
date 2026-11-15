# Triune Multi-Query Associative Recall (MQAR) Benchmark Report

- **Date**: `2026-11-14 12:22:27 UTC`
- **Device**: `cuda:0` | **Precision**: `torch.bfloat16`
- **Dimension**: `256` | **Heads**: `4`
- **Warmup**: 3 full iterations per `(seq_len, model_type)` prior to timing (`reps=15`)

## 1. Empirical Associative Retrieval Scaling (Standard & Overwrite-Key MQAR)

| Sequence Length | Pairs | Softmax Latency | Softmax Recall (Std / Overwrite) | GLA Latency | GLA Recall (Std / Overwrite) | Gated DeltaNet Latency | Gated DeltaNet Recall (Std / Overwrite) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 256 | 8 | 0.17 ms | 100.0% / 50.0% | 0.31 ms | 100.0% / 62.5% | 0.27 ms | 100.0% / 100.0% |
| 512 | 16 | 0.18 ms | 100.0% / 50.0% | 0.33 ms | 100.0% / 75.0% | 0.52 ms | 100.0% / 100.0% |
| 1,024 | 32 | 0.33 ms | 100.0% / 62.5% | 0.33 ms | 100.0% / 87.5% | 0.93 ms | 100.0% / 100.0% |
| 2,048 | 64 | 0.97 ms | 100.0% / 37.5% | 0.37 ms | 100.0% / 75.0% | 1.80 ms | 100.0% / 100.0% |

## 2. Key Insights
1. **Warmed-Up Scaling Integrity**: Pre-warming Triton and SDPA kernels for every sequence length eliminates first-call compilation spikes so latencies scale cleanly with sequence length.
2. **Delta-Rule Memory Overwrite Advantage**: While purely additive linear attention (GLA) superposes stale and updated key values on key overwrite, Gated DeltaNet applies the error-correcting delta update to erase stale associations and retrieve the latest value.
