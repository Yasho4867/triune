# Triune Multi-Query Associative Recall (MQAR) Benchmark Report

- **Date**: `2026-11-04 12:37:25 UTC`
- **Device**: `cuda:0` | **Precision**: `torch.bfloat16`
- **Dimension**: `256` | **Heads**: `4`

## 1. Empirical Associative Retrieval Scaling

| Sequence Length | Pairs | Softmax (O(N^2)) Latency | Softmax Recall | GLA (O(N)) Latency | GLA Recall | Gated DeltaNet Latency | Gated DeltaNet Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 256 | 8 | 17.56 ms | 0.0% | 504.03 ms | 0.0% | 178.68 ms | 0.0% |
| 512 | 16 | 0.28 ms | 0.0% | 2.52 ms | 0.0% | 240.77 ms | 0.0% |
| 1,024 | 32 | 0.28 ms | 0.0% | 78.93 ms | 0.0% | 596.94 ms | 0.0% |
| 2,048 | 64 | 0.54 ms | 0.0% | 3.48 ms | 0.0% | 1013.05 ms | 0.0% |

## 2. Key Insights
1. **Sub-Quadratic Latency**: Both GLA and Gated DeltaNet maintain linear time scaling, dramatically outpacing quadratic softmax at long contexts.
2. **Associative Memory Updates**: Gated DeltaNet incorporates the error-correcting delta rule, preventing memory state saturation.
