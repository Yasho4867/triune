# CALM Margin Calibration, Reliability & Pareto Report

- **Checkpoint**: `checkpoints/triune_studio_step_36500_rescued.pt`
- **Source**: local `data/fineweb_edu_large.jsonl`, disjoint held-out suffix documents 7500 to 7899.
- **Calibration split**: {"documents": 200, "positions": 87141, "first_id": "local:7500", "last_id": "local:7699", "id_sha256_16": "f26ebb16446fdfb9"}
- **Evaluation split**: {"documents": 200, "positions": 85503, "first_id": "local:7700", "last_id": "local:7899", "id_sha256_16": "100002c54e817dcb"}
- **Protocol**: smallest threshold on the calibration split whose Wilson 95% lower bound on exit-vs-cortex top-1 agreement reaches the target (min 500 positions); that threshold is then applied unchanged to the disjoint evaluation split. Nothing is fit on the evaluation split.
- **Sequence length**: first 512 tokens of each document.

## Reflex exit vs Cortex

- Unconditional top-1 agreement: calibration `73.67%`, evaluation `73.17%`
- Discrimination & Calibration Quality (eval split): `Margin AUROC = 0.8159`, `Confidence ECE (10-bin) = 0.4479`, `MCE = 0.5684`, `Brier = 0.3767`

| Target | Calibrated threshold | Eval coverage | Eval agreement | Eval Wilson-95 lower | Target met |
|:---|:---:|:---:|:---:|:---:|:---:|
| 90% | 0.1 | 47.89% | 92.62% | 92.37% | yes |
| 95% | 0.2 | 31.34% | 95.73% | 95.48% | yes |

10-bin reliability diagram (reflex, top-1 probability vs exit-cortex agreement):

| Confidence Bin | Positions | Mean Confidence | Empirical Agreement | Wilson-95 Lower | |Acc - Conf| |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [0.0, 0.1) | 19398 | 0.0598 | 0.4398 | 0.4328 | 0.3800 |
| [0.1, 0.2) | 22133 | 0.1479 | 0.6694 | 0.6631 | 0.5215 |
| [0.2, 0.3) | 15028 | 0.2460 | 0.8144 | 0.8081 | 0.5684 |
| [0.3, 0.4) | 9129 | 0.3458 | 0.8853 | 0.8786 | 0.5395 |
| [0.4, 0.5) | 5704 | 0.4462 | 0.9148 | 0.9073 | 0.4686 |
| [0.5, 0.6) | 3811 | 0.5472 | 0.9417 | 0.9339 | 0.3946 |
| [0.6, 0.7) | 2807 | 0.6473 | 0.9587 | 0.9507 | 0.3114 |
| [0.7, 0.8) | 2211 | 0.7489 | 0.9738 | 0.9662 | 0.2249 |
| [0.8, 0.9) | 1967 | 0.8491 | 0.9858 | 0.9795 | 0.1367 |
| [0.9, 1.0] | 3315 | 0.9679 | 0.9970 | 0.9945 | 0.0291 |

Coverage vs agreement on the evaluation split (softmax probability margin):

| Margin threshold | Coverage | Agreement | Wilson-95 lower | Positions |
|:---:|:---:|:---:|:---:|:---:|
| 0.05 | 63.24% | 88.15% | 87.88% | 54068 |
| 0.1 | 47.89% | 92.62% | 92.37% | 40950 |
| 0.15 | 38.34% | 94.71% | 94.46% | 32786 |
| 0.2 | 31.34% | 95.73% | 95.48% | 26795 |
| 0.25 | 26.15% | 96.56% | 96.31% | 22360 |
| 0.3 | 22.09% | 97.1% | 96.85% | 18887 |
| 0.35 | 18.79% | 97.62% | 97.37% | 16062 |
| 0.4 | 16.17% | 97.9% | 97.65% | 13829 |
| 0.45 | 14.03% | 98.26% | 98.01% | 11996 |
| 0.5 | 12.19% | 98.47% | 98.22% | 10419 |
| 0.55 | 10.69% | 98.67% | 98.41% | 9139 |
| 0.6 | 9.35% | 98.89% | 98.63% | 7991 |
| 0.65 | 8.19% | 99.03% | 98.77% | 7004 |
| 0.7 | 7.12% | 99.24% | 98.99% | 6089 |
| 0.75 | 6.17% | 99.41% | 99.17% | 5278 |
| 0.8 | 5.23% | 99.53% | 99.28% | 4476 |
| 0.85 | 4.29% | 99.73% | 99.5% | 3671 |
| 0.9 | 3.41% | 99.79% | 99.55% | 2914 |
| 0.95 | 2.45% | 99.86% | 99.58% | 2093 |

## Limbic exit vs Cortex

- Unconditional top-1 agreement: calibration `75.8%`, evaluation `75.25%`
- Discrimination & Calibration Quality (eval split): `Margin AUROC = 0.8319`, `Confidence ECE (10-bin) = 0.4802`, `MCE = 0.6063`, `Brier = 0.3987`

| Target | Calibrated threshold | Eval coverage | Eval agreement | Eval Wilson-95 lower | Target met |
|:---|:---:|:---:|:---:|:---:|:---:|
| 90% | 0.05 | 61.54% | 91.03% | 90.78% | yes |
| 95% | 0.15 | 37.08% | 96.52% | 96.31% | yes |

10-bin reliability diagram (limbic, top-1 probability vs exit-cortex agreement):

| Confidence Bin | Positions | Mean Confidence | Empirical Agreement | Wilson-95 Lower | |Acc - Conf| |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [0.0, 0.1) | 21548 | 0.0577 | 0.4661 | 0.4594 | 0.4083 |
| [0.1, 0.2) | 21705 | 0.1480 | 0.7184 | 0.7123 | 0.5703 |
| [0.2, 0.3) | 14805 | 0.2457 | 0.8520 | 0.8462 | 0.6063 |
| [0.3, 0.4) | 8860 | 0.3461 | 0.9159 | 0.9100 | 0.5698 |
| [0.4, 0.5) | 5512 | 0.4462 | 0.9399 | 0.9334 | 0.4938 |
| [0.5, 0.6) | 3574 | 0.5473 | 0.9572 | 0.9500 | 0.4099 |
| [0.6, 0.7) | 2563 | 0.6471 | 0.9723 | 0.9652 | 0.3252 |
| [0.7, 0.8) | 2100 | 0.7478 | 0.9857 | 0.9797 | 0.2380 |
| [0.8, 0.9) | 1735 | 0.8481 | 0.9885 | 0.9823 | 0.1403 |
| [0.9, 1.0] | 3101 | 0.9684 | 0.9981 | 0.9958 | 0.0296 |

Coverage vs agreement on the evaluation split (softmax probability margin):

| Margin threshold | Coverage | Agreement | Wilson-95 lower | Positions |
|:---:|:---:|:---:|:---:|:---:|
| 0.05 | 61.54% | 91.03% | 90.78% | 52620 |
| 0.1 | 46.65% | 94.88% | 94.66% | 39884 |
| 0.15 | 37.08% | 96.52% | 96.31% | 31706 |
| 0.2 | 30.2% | 97.25% | 97.05% | 25826 |
| 0.25 | 25.0% | 97.72% | 97.51% | 21372 |
| 0.3 | 20.99% | 98.1% | 97.89% | 17951 |
| 0.35 | 17.73% | 98.33% | 98.11% | 15160 |
| 0.4 | 15.18% | 98.55% | 98.33% | 12979 |
| 0.45 | 13.09% | 98.83% | 98.61% | 11194 |
| 0.5 | 11.42% | 98.97% | 98.75% | 9766 |
| 0.55 | 9.98% | 99.11% | 98.89% | 8529 |
| 0.6 | 8.72% | 99.37% | 99.16% | 7454 |
| 0.65 | 7.63% | 99.43% | 99.22% | 6521 |
| 0.7 | 6.64% | 99.47% | 99.25% | 5678 |
| 0.75 | 5.72% | 99.49% | 99.25% | 4887 |
| 0.8 | 4.8% | 99.59% | 99.34% | 4107 |
| 0.85 | 4.0% | 99.8% | 99.58% | 3418 |
| 0.9 | 3.22% | 99.93% | 99.74% | 2755 |
| 0.95 | 2.35% | 99.95% | 99.72% | 2008 |

## Quality vs Layer-Compute Pareto Sweep (Hierarchical Reflex -> Limbic -> Cortex)

Evaluated over `85,303` held-out next-token transitions (Dense Cortex PPL = `97.25`):

| Threshold ($\tau$) | Reflex Exit | Limbic Exit | Cortex Exit | Active Layers % | Layer Compute Saved | CALM PPL | $\Delta$ PPL vs Cortex |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 (All-Reflex) | 100.0% | 0.0% | 0.0% | 37.5% | 62.5% | 99.08 | +1.88% |
| 0.05 | 63.21% | 6.42% | 30.37% | 58.89% | 41.11% | 98.06 | +0.84% |
| 0.1 | 47.86% | 4.86% | 47.28% | 68.87% | 31.13% | 97.72 | +0.49% |
| 0.15 | 38.31% | 3.77% | 57.93% | 75.12% | 24.88% | 97.56 | +0.32% |
| 0.2 | 31.31% | 3.11% | 65.58% | 79.65% | 20.35% | 97.38 | +0.14% |
| 0.25 | 26.13% | 2.47% | 71.4% | 83.05% | 16.95% | 97.4 | +0.15% |
| 0.3 | 22.07% | 2.06% | 75.87% | 85.69% | 14.31% | 97.39 | +0.14% |
| 0.35 | 18.76% | 1.73% | 79.5% | 87.84% | 12.16% | 97.4 | +0.16% |
| 0.4 | 16.15% | 1.42% | 82.43% | 89.55% | 10.45% | 97.4 | +0.15% |
| 0.45 | 14.01% | 1.19% | 84.8% | 90.95% | 9.05% | 97.45 | +0.21% |
| 0.5 | 12.17% | 1.03% | 86.8% | 92.14% | 7.86% | 97.33 | +0.09% |
| 0.55 | 10.68% | 0.91% | 88.41% | 93.1% | 6.9% | 97.3 | +0.05% |
| 0.6 | 9.33% | 0.79% | 89.87% | 93.97% | 6.03% | 97.32 | +0.08% |
| 0.65 | 8.18% | 0.68% | 91.14% | 94.72% | 5.28% | 97.29 | +0.05% |
| 0.7 | 7.11% | 0.6% | 92.29% | 95.4% | 4.6% | 97.33 | +0.08% |
| 0.75 | 6.17% | 0.53% | 93.3% | 96.01% | 3.99% | 97.3 | +0.05% |
| 0.8 | 5.23% | 0.43% | 94.34% | 96.62% | 3.38% | 97.27 | +0.02% |
| 0.85 | 4.29% | 0.39% | 95.32% | 97.22% | 2.78% | 97.28 | +0.03% |
| 0.9 | 3.41% | 0.31% | 96.28% | 97.79% | 2.21% | 97.27 | +0.03% |
| 0.95 | 2.45% | 0.23% | 97.32% | 98.41% | 1.59% | 97.27 | +0.02% |
| 1.01 (Dense-Cortex) | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 97.25 | +0.00% |

## Live Token-Compacted CALM Wall-Clock Throughput

Batch `8 x 182` (`1,456` tokens/forward) on held-out text. Dense Cortex baseline (`force_depth=2`): `51.39 ms` (`28,330.9 tok/s`).

| Margin Threshold ($\tau$) | Reflex / Limbic / Cortex Tokens | Compacted SeqLens (L / C) | Median Latency | Throughput | Speedup vs Dense |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | 1456 / 539 / 410 | 182 / 182 | 78.96 ms | 18,440.2 tok/s | **0.651x** |
| 0.1 | 1456 / 785 / 652 | 182 / 182 | 78.37 ms | 18,577.4 tok/s | **0.656x** |
| 0.15 | 1456 / 922 / 819 | 182 / 182 | 72.12 ms | 20,188.4 tok/s | **0.713x** |
| 0.2 | 1456 / 1024 / 939 | 182 / 182 | 72.99 ms | 19,946.7 tok/s | **0.704x** |
| 0.4 | 1456 / 1258 / 1205 | 182 / 182 | 74.83 ms | 19,458.6 tok/s | **0.687x** |
| 0.6 | 1456 / 1353 / 1322 | 182 / 182 | 71.12 ms | 20,471.6 tok/s | **0.723x** |
| 0.8 | 1456 / 1395 / 1376 | 182 / 182 | 78.18 ms | 18,623.2 tok/s | **0.657x** |

## Verdict

Held-out agreement target reached on N=85,503 evaluation positions for: reflex, limbic (see reliability tables, Pareto sweep, and live token-compacted speedup benchmark).
