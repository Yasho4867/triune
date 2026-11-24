# Triune Multi-Seed 100-Step Mixture-of-Experts (MoE) Convergence Report
**Empirical Multi-Seed Optimizer Evaluation & 3-Tier Dynamic Steering Synergy on Real FineWeb-Edu**

- **Date / Timestamp**: `2026-11-15 07:30:40 UTC`
- **Git Commit**: [`569469e491`](https://github.com/lucid-machines/triune/commit/569469e491ea6186407ad53d10a63ffaa3582f76) (Dirty Working Tree)
- **Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU` (7.93 GB VRAM, CC 12.0)
- **Software**: PyTorch `2.14.1+cu132` | CUDA `13.2` | Python `3.14.7`
- **Dataset**: Real FineWeb-Edu educational text (`data\fineweb_edu_real.jsonl`, SHA-256 `aed10d4c98af...`)
- **Model Topology**: Universal Mixture-of-Experts Transformer (`90,788,895` parameters, 8 layers, 4 experts, 384 dim, 6 heads, 32,000 vocab)
- **Subspace & Optimizer Config**: `rank=16`, `update_gap=20`, `steer_update_gap=5`, `steer_scale=0.2`, `core_mask_mode=full`, `lr=0.001`, `muon_lr=0.002`
- **Experimental Protocol**:
  - **Random Seeds**: `42, 1337, 2026, 7, 99, 314, 777, 1024` ($N=8$ independent runs per regime)
  - **Step-0 Initial Weights**: Identical per seed (sampled once and cloned bit-for-bit into all regimes)
  - **Batch Sequences**: Identical causal batches fed in exact synchronous order across all regimes per seed
  - **Document Partitioning**: Strict document-level train/validation split with dynamic `[SEP]` separator packing
  - **Warm-Up Phase**: Common pre-warmup step executed before timing to eliminate CUDA JIT / memory allocation skew
  - **Execution Order**: Randomized regime order per seed to guarantee zero position or allocator bias
  - **Validation Set**: Multi-batch held-out split of 5 batches (2,560 tokens)

---

## 1. Active Parameter Tier Distribution & Measured Optimizer State Memory

| Regime | Total Params | Tier 1: Muon | Tier 2: Low-Rank | Tier 3: Base AdamW | Dynamic Steering | Total Opt State | Expert Opt State | Expert Cut vs BF16 / FP32 AdamW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard AdamW (`torch.optim.AdamW`) | 90,789,271 | 0 (0.0%) | 0 (0.0%) | 90,789,271 (100.0%) | No | 251.96 MB (2.910 B/p) | 135.21 MB (4.0000 B/p) | 0.0% / 50.0% |
| Muon + AdamW Baseline (Muon Attention + AdamW Experts) | 90,789,271 | 5,898,240 (6.5%) | 0 (0.0%) | 84,891,031 (93.5%) | No | 240.71 MB (2.780 B/p) | 135.21 MB (4.0000 B/p) | 0.0% / 50.0% |
| BAdam Baseline (Block-Coordinate AdamW, Luo et al. 2024) | 90,789,271 | 0 (0.0%) | 0 (0.0%) | 90,789,271 (100.0%) | No | 46.88 MB (0.541 B/p) | 27.04 MB (0.8000 B/p) | 80.0% / 90.0% |
| GaLore Baseline (Single-Sided SVD r=16, gap=20) | 90,789,271 | 0 (0.0%) | 41,287,680 (45.5%) | 49,501,591 (54.5%) | No | 102.43 MB (1.183 B/p) | 6.68 MB (0.1976 B/p) | 95.1% / 97.5% |
| CentroidSteer Only (No Muon: Triangular Subspace + Steering + AdamW Base) | 90,789,271 | 0 (0.0%) | 41,287,680 (45.5%) | 49,501,591 (54.5%) | Yes | 99.12 MB (1.145 B/p) | 3.87 MB (0.1144 B/p) | 97.1% / 98.6% |
| Triune Unsteered Ablation (Muon + Triangular Subspace, alpha=0) | 90,789,271 | 5,898,240 (6.5%) | 35,389,440 (39.0%) | 49,501,591 (54.5%) | No | 109.28 MB (1.262 B/p) | 3.78 MB (0.1118 B/p) | 97.2% / 98.6% |
| Triune Random-Steer Ablation (Muon + Triangular Subspace + Random Noise Dir) | 90,789,271 | 5,898,240 (6.5%) | 35,389,440 (39.0%) | 49,501,591 (54.5%) | No | 109.28 MB (1.262 B/p) | 3.78 MB (0.1118 B/p) | 97.2% / 98.6% |
| **Triune MoE Stack** (Muon Attention + CentroidSteer Experts + AdamW Base) | 90,789,271 | 5,898,240 (6.5%) | 35,389,440 (39.0%) | 49,501,591 (54.5%) | Yes | 109.28 MB (1.262 B/p) | 3.78 MB (0.1118 B/p) | 97.2% / 98.6% |

---

## 2. Multi-Seed MoE Optimization & Convergence Summary (Mean ± Std)

| Framework & Optimizer Regime | Initial Val Loss | Final Val Loss | Val Loss Drop | Final Val PPL | Steady Step | Total Time | Peak VRAM | Conv Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard AdamW (`torch.optim.AdamW`) | 10.4505 ± 0.0072 | 7.8035 ± 0.1301 | 2.6470 ± 0.1288 | 2467.70 ± 332.73 | 90.82 ± 3.18 ms | 9.61 ± 0.17 s | 4383.4 MB | 0.2755 loss/s |
| Muon + AdamW Baseline (Muon Attention + AdamW Experts) | 10.4505 ± 0.0072 | 7.6799 ± 0.0547 | 2.7706 ± 0.0536 | 2167.36 ± 122.76 | 90.42 ± 4.79 ms | 9.53 ± 0.30 s | 4376.1 MB | 0.2910 loss/s |
| BAdam Baseline (Block-Coordinate AdamW, Luo et al. 2024) | 10.4505 ± 0.0072 | 8.4076 ± 0.0408 | 2.0429 ± 0.0436 | 4484.29 ± 185.00 | 60.56 ± 3.14 ms | 5.97 ± 0.34 s | 4074.2 MB | 0.3429 loss/s |
| GaLore Baseline (Single-Sided SVD r=16, gap=20) | 10.4505 ± 0.0072 | 7.6561 ± 0.0584 | 2.7944 ± 0.0553 | 2116.70 ± 129.42 | 98.79 ± 5.67 ms | 10.70 ± 0.46 s | 4122.2 MB | 0.2617 loss/s |
| CentroidSteer Only (No Muon: Triangular Subspace + Steering + AdamW Base) | 10.4505 ± 0.0072 | **7.6459 ± 0.0625** | **2.8046 ± 0.0613** | **2095.76 ± 138.73** | 90.95 ± 3.78 ms | 9.65 ± 0.24 s | 4158.2 MB | 0.2907 loss/s |
| Triune Unsteered Ablation (Muon + Triangular Subspace, alpha=0) | 10.4505 ± 0.0072 | 7.6499 ± 0.0656 | 2.8006 ± 0.0624 | 2104.42 ± 144.31 | 91.71 ± 3.82 ms | 9.75 ± 0.43 s | 4218.4 MB | 0.2876 loss/s |
| Triune Random-Steer Ablation (Muon + Triangular Subspace + Random Noise Dir) | 10.4505 ± 0.0072 | 7.6505 ± 0.0658 | 2.8000 ± 0.0626 | 2105.76 ± 144.84 | 92.99 ± 3.95 ms | 9.80 ± 0.29 s | 4258.4 MB | 0.2858 loss/s |
| **Triune MoE Stack** (Muon Attention + CentroidSteer Experts + AdamW Base) | 10.4505 ± 0.0072 | 7.6521 ± 0.0654 | 2.7984 ± 0.0623 | 2109.04 ± 144.60 | 91.49 ± 2.79 ms | 9.64 ± 0.20 s | 4338.4 MB | 0.2902 loss/s |

---

## 3. Validation Loss & Perplexity Trajectory Across Training Steps

Evaluated over the document-isolated held-out validation set (5 batches, 2,560 tokens):

| Step | Metric | AdamW | Muon+AdamW | BAdam | GaLore | Centroid(NoMuon) | Triune(NoSteer) | Triune(RandSteer) | Triune MoE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 0** | Val Loss | **10.4505 ± 0.0072** | **10.4505 ± 0.0072** | **10.4505 ± 0.0072** | **10.4505 ± 0.0072** | **10.4505 ± 0.0072** | **10.4505 ± 0.0072** | **10.4505 ± 0.0072** | **10.4505 ± 0.0072** |
| | Val PPL | **34562.23 ± 248.95** | **34562.23 ± 248.95** | **34562.23 ± 248.95** | **34562.23 ± 248.95** | **34562.23 ± 248.95** | **34562.23 ± 248.95** | **34562.23 ± 248.95** | **34562.23 ± 248.95** |
| **Step 20** | Val Loss | 8.2881 ± 0.0689 | 8.2408 ± 0.0478 | 9.8338 ± 0.0423 | 8.2101 ± 0.0561 | 8.2091 ± 0.0566 | 8.2017 ± 0.0608 | **8.2016 ± 0.0608** | 8.2017 ± 0.0608 |
| | Val PPL | 3984.63 ± 276.61 | 3796.28 ± 182.15 | 18667.40 ± 804.89 | 3682.97 ± 207.00 | 3679.39 ± 208.71 | 3652.98 ± 221.95 | **3652.65 ± 221.90** | 3652.87 ± 221.86 |
| **Step 40** | Val Loss | 8.0766 ± 0.1003 | 8.0102 ± 0.0890 | 9.3619 ± 0.0430 | 8.0003 ± 0.0914 | 8.0062 ± 0.0977 | 8.0003 ± 0.0859 | **7.9996 ± 0.0861** | 8.0005 ± 0.0868 |
| | Val PPL | 3232.39 ± 323.56 | 3021.91 ± 265.80 | 11645.72 ± 491.99 | 2992.59 ± 271.19 | 3012.13 ± 292.93 | 2991.49 ± 258.54 | **2989.69 ± 259.03** | 2992.44 ± 261.26 |
| **Step 60** | Val Loss | 7.9601 ± 0.0573 | 7.8745 ± 0.0464 | 8.9237 ± 0.0281 | 7.8698 ± 0.0422 | 7.8728 ± 0.0491 | 7.8683 ± 0.0465 | 7.8693 ± 0.0439 | **7.8682 ± 0.0465** |
| | Val PPL | 2868.52 ± 162.54 | 2631.85 ± 123.99 | 7510.51 ± 212.03 | 2619.16 ± 111.08 | 2627.73 ± 128.33 | 2615.47 ± 122.11 | 2617.89 ± 115.75 | **2615.42 ± 122.22** |
| **Step 80** | Val Loss | 7.9052 ± 0.1041 | 7.8343 ± 0.0550 | 8.6411 ± 0.0764 | 7.8101 ± 0.0422 | 7.8141 ± 0.0494 | 7.7897 ± 0.0481 | 7.7898 ± 0.0477 | **7.7873 ± 0.0518** |
| | Val PPL | 2724.26 ± 285.88 | 2529.01 ± 138.88 | 5674.13 ± 444.88 | 2467.17 ± 104.78 | 2477.93 ± 123.10 | 2417.86 ± 114.41 | 2418.29 ± 113.46 | **2412.49 ± 122.03** |
| **Step 100** | Val Loss | 7.8035 ± 0.1301 | 7.6799 ± 0.0547 | 8.4076 ± 0.0408 | 7.6561 ± 0.0584 | **7.6459 ± 0.0625** | 7.6499 ± 0.0656 | 7.6505 ± 0.0658 | 7.6521 ± 0.0654 |
| | Val PPL | 2467.70 ± 332.73 | 2167.36 ± 122.76 | 4484.29 ± 185.00 | 2116.70 ± 129.42 | **2095.76 ± 138.73** | 2104.42 ± 144.31 | 2105.76 ± 144.84 | 2109.04 ± 144.60 |

---

## 4. Expert Routing, Gradient Activity & Empirical Variance Concentration

### Expert Routing Distribution (Tokens Accepted Per Expert Across All Layers)
- **AdamW**: Expert 0: 38,107 tokens, Expert 1: 29,873 tokens, Expert 2: 26,651 tokens, Expert 3: 41,493 tokens
- **Muon+AdamW**: Expert 0: 48,791 tokens, Expert 1: 30,294 tokens, Expert 2: 36,390 tokens, Expert 3: 37,543 tokens
- **BAdam**: Expert 0: 34,721 tokens, Expert 1: 30,992 tokens, Expert 2: 30,299 tokens, Expert 3: 37,369 tokens
- **GaLore**: Expert 0: 40,212 tokens, Expert 1: 36,309 tokens, Expert 2: 40,622 tokens, Expert 3: 34,643 tokens
- **Centroid(NoMuon)**: Expert 0: 41,590 tokens, Expert 1: 36,111 tokens, Expert 2: 38,106 tokens, Expert 3: 36,597 tokens
- **Triune(NoSteer)**: Expert 0: 41,681 tokens, Expert 1: 37,364 tokens, Expert 2: 39,149 tokens, Expert 3: 36,960 tokens
- **Triune(RandSteer)**: Expert 0: 44,887 tokens, Expert 1: 37,191 tokens, Expert 2: 38,272 tokens, Expert 3: 33,375 tokens
- **Triune MoE**: Expert 0: 40,300 tokens, Expert 1: 38,353 tokens, Expert 2: 38,829 tokens, Expert 3: 38,310 tokens

### Parameter Activity & Nonzero Gradients (`Triune MoE`)
- **Tier 1 (Muon)**: `5,898,240` / 5,898,240 (100.0% non-zero at sampled step)
- **Tier 2 (Subspace)**: `31,186,944` / 35,389,440 (88.1% non-zero at sampled step)
- **Tier 3 (Base AdamW)**: `24,755,108` / 49,501,591 (50.0% non-zero at sampled step)

### Empirical Routing Concentration Diagnostics (Held-Out Validation)
| Regime | Mean Centroid Norm² $\|c_e\|_2^2$ | Mean Intra-Expert Trace $\operatorname{tr}(\Sigma_e)$ | Dispersion Ratio $\operatorname{tr}(\Sigma_e)/\|c_e\|_2^2$ | Within / Global Var $\operatorname{tr}(\Sigma_e)/\operatorname{tr}(\Sigma_{\text{glob}})$ | Centroid Energy Frac | Router Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AdamW | 294.0880 | 89.9110 | 0.3731 | 0.6604 | 0.7659 | 0.9113 |
| Muon+AdamW | 254.0572 | 129.9425 | 0.6899 | 0.6250 | 0.6616 | 1.1527 |
| BAdam | 364.8197 | 19.1783 | 0.0560 | 0.9936 | 0.9500 | 1.2420 |
| GaLore | 274.8216 | 109.1780 | 0.5085 | 0.5177 | 0.7157 | 1.0861 |
| Centroid(NoMuon) | 265.2602 | 118.7370 | 0.5636 | 0.5579 | 0.6908 | 0.9216 |
| Triune(NoSteer) | 268.4333 | 115.5639 | 0.5284 | 0.5452 | 0.6990 | 0.8817 |
| Triune(RandSteer) | 260.4054 | 123.5941 | 0.5725 | 0.5906 | 0.6782 | 0.8973 |
| Triune MoE | 270.9937 | 113.0062 | 0.5080 | 0.5314 | 0.7057 | 0.8756 |

---

## 5. Statistical Comparisons (Paired & Welch Two-Sided t-Tests)

| Comparison | N | Mean A | Mean B | Mean Diff (A - B) | Paired t-test p | Welch t-test p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| triune_moe vs adamw | 8 | 7.6521 ± 0.0654 | 7.8035 ± 0.1301 | -0.1514 | 0.008211 | 0.014285 |
| triune_moe vs badam | 8 | 7.6521 ± 0.0654 | 8.4076 ± 0.0408 | -0.7555 | 0.000000 | 0.000000 |
| triune_moe vs galore | 8 | 7.6521 ± 0.0654 | 7.6561 ± 0.0584 | -0.0040 | 0.731608 | 0.898899 |
| triune_moe vs muon_adamw | 8 | 7.6521 ± 0.0654 | 7.6799 ± 0.0547 | -0.0278 | 0.060230 | 0.371904 |
| triune_moe vs centroid_no_muon | 8 | 7.6521 ± 0.0654 | 7.6459 ± 0.0625 | +0.0062 | 0.568951 | 0.849977 |
| triune_moe vs triune_moe_nosteer | 8 | 7.6521 ± 0.0654 | 7.6499 ± 0.0656 | +0.0022 | 0.118774 | 0.947109 |
| triune_moe vs triune_moe_random | 8 | 7.6521 ± 0.0654 | 7.6505 ± 0.0658 | +0.0016 | 0.143909 | 0.961783 |
| centroid_no_muon vs galore | 8 | 7.6459 ± 0.0625 | 7.6561 ± 0.0584 | -0.0102 | 0.247049 | 0.741640 |
| centroid_no_muon vs badam | 8 | 7.6459 ± 0.0625 | 8.4076 ± 0.0408 | -0.7617 | 0.000000 | 0.000000 |
| triune_moe_nosteer vs galore | 8 | 7.6499 ± 0.0656 | 7.6561 ± 0.0584 | -0.0062 | 0.618632 | 0.844147 |

---

## 6. Cryptographic Provenance & Reproduction

To reproduce these exact MoE results from a clean repository checkout:

```bash
python scripts/benchmark_moe_convergence_100_steps.py --steps 100 --batch-size 2 --seq-len 256 --val-interval 20 --seeds 42 1337 2026 7 99 314 777 1024 --num-val-batches 5 --data-path data/fineweb_edu_real.jsonl --required-tokens 120000 --num-val-docs 15 --lr 0.001 --muon-lr 0.002 --rank 16 --update-gap 20 --steer-update-gap 5 --core-mask-mode full --steer-scale 0.2 --regimes adamw muon_adamw badam galore centroid_no_muon triune_moe triune_moe_nosteer triune_moe_random
```

### Source File Cryptographic Verification Hashes (SHA-256)
| Relative Path | SHA-256 Checksum |
| :--- | :--- |
| `data/fineweb_edu_real.jsonl` | `aed10d4c98afb6da4f55ee3aa295981465b9fd49edb1e5886c36fb6fe3ae8e9c` |
| `scripts/benchmark_moe_convergence_100_steps.py` | `ec5c6ca99621dd1cb1ad9f7cc01407e83f9ba4019d7151370928edd0ecbf5168` |
| `triune/kernels/fast_gla.py` | `b5ff71c2ed2fa1f320041dc845191935997cd8f23bd5184f1b565815c8bbaf36` |
| `triune/kernels/fast_muon.py` | `4408f21b0d4f905510f18b6e22d1f9b1786f4146a7198f6397d99a469ee1dcd9` |
| `triune/kernels/fast_triangular.py` | `5e8dabd2245aa57ba85e6ec387b3f4b450e01c68a8706fa054e7bfad4b930412` |
| `triune/model/config.py` | `6648017152cab03081f0be66476610464a0a408249a7be147fc538037c61e615` |
| `triune/model/moe.py` | `d28800a893e917df0c9f74136d49adfbeb87fd0a8716da9c927553ad0aa15806` |
| `triune/model/transformer.py` | `698c7c195bdc332ae787e01fa15ae3fc7af5d6a502b6c84f7b7c8a294d5409fa` |
| `triune/optim/centroid.py` | `6874444e87b871339bf5281969ae0efe29473bd6e2653ffe983525beda2bbb59` |
| `triune/optim/muon.py` | `f6296eb681216fef02ae18815cf4dbb2c29c90703e770140988414319ce8035e` |
| `triune/universal.py` | `69085fe40f677cb47d4330ad9c14343ec70e4da960af1a2188925fd5c1a12a06` |
| `triune_tokenizer.json` | `df2c43cfab506acce484e8c45a12e497a7901622fa60058fd4cd5802f4e36451` |
