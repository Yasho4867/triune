# CentroidSteer Validation Report — Exact Orthogonal Direct-Sum & Grassmann Parallel Transport (Lucid Machines v1.0.0)

- **Timestamp (UTC)**: `2026-11-22T07:40:02Z`
- **Git Commit**: `57a14767647edc465bf7f221bdd260021cc561c7` (dirty: `True`)
- **Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU` (7.93 GB VRAM, CC `12.0`)
- **Software**: Python `3.14.7` | PyTorch `2.14.1+cu132` | CUDA `13.2`
- **Seeds**: `[42, 1337, 2026, 777, 1024]`

## Exact Reproduction Command

```bash
C:/Users/yashb_f1ls/OneDrive/Documents/TriuneTransformer/studio/studio_env/Scripts/python.exe benchmarks/validate_centroid_steer_upgrade.py --seeds 42 1337 2026 777 1024 --suites 1 2 3 4 --md-steps 120 --cold-steps 300 --cont-steps 100 --batch-size 16 --cont-batch-size 4 --seq-len 256 --cont-seq-len 128 --rank 4 --cont-rank 8 --md-rank 8 --steer-dirs 8 --md-steer-dirs 4 --steer-scale 1.0 --cont-steer-scale 0.75 --steer-update-gap 2 --lr 0.002 --expert-lr 0.0015 --cont-lr 0.0005 --cont-expert-lr 0.0005
```

---

## 1. Suite 1: CentroidSteer Architecture & Mechanistic Diagnostics

CentroidSteer decomposes MoE routed-expert gradients into an **Exact Orthogonal Direct-Sum** ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$), tracks expert-routed activation subspaces via **Matrix-Free Block-Oja EMA**, and transports preconditioned moments across subspace rotations via **Grassmann / Stiefel Parallel Transport** ($M_c \leftarrow M_c T$, $V_c \leftarrow V_c (T \odot T) + \bar{v}(1 - \text{col\_energy})$):

- **Direct-Sum Orthogonality**: `W_1` max relative error `4.50e-07`, `W_2` max relative error `7.90e-07`, step update `<Delta_W_parallel, Delta_W_perp>_F` error `0.00e+00`, Stiefel isometry `||C_e^T C_e - I||_F` max error `9.10e-07`.
- **Block-Oja Subspace Capture**: Captured `0.8736` of gradient energy vs `0.0427` (`random_same_mem`, **20.48x**); captured `0.6053` of activation variance vs `0.0333` (**18.16x**).
- **Grassmann Parallel Transport**: Continuous transport step reached `10..10` over 10 rotations (zero moment reset, `min(V_c) = 1.144e-14` >= 0).
- **Activation-Space Step Norm Ratio**: Direct-sum subspace steering restores `1.5446x` of full-rank AdamW along $C_e$ vs `0.1021x` for an unsteered rank-`8` low-rank core (**15.13x** amplification).

---

## 2. Suite 2: Controlled Multi-Seed Multi-Domain MoE Specialization

Configuration: `num_layers=2`, `dim=128`, `num_experts=4`, `rank=8`, `k_c=4`, `update_gap=10`, `steer_update_gap=2`, `steer_scale=1.0`, `steps=120`, `N=5` seeds.

| Regime | Val Loss (Mean +/- Std) | Val PPL | Val Accuracy | Domain Purity | Centroid Sep. | Subspace Energy | Expert State (B/p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `nosteer` (No-Steer Low-Rank) | 1.2243 +/- 0.0468 | 3.40 +/- 0.16 | 0.5675 +/- 0.0228 | 0.7830 +/- 0.1282 | 1.3434 +/- 0.0367 | 0.0000 +/- 0.0000 | 0.333 |
| `random_same_mem` (Memory-Matched Random Subspace) | 1.2044 +/- 0.0446 | 3.34 +/- 0.15 | 0.5757 +/- 0.0211 | 0.8190 +/- 0.1163 | 1.3277 +/- 0.0054 | 0.0188 +/- 0.0009 | 0.550 |
| `adamw` (Full-Rank AdamW) | 0.8307 +/- 0.0365 | 2.30 +/- 0.08 | 0.7007 +/- 0.0108 | 0.5821 +/- 0.0924 | 1.4215 +/- 0.1142 | 0.0000 +/- 0.0000 | 8.000 |
| `upgraded` (**CentroidSteer**) | 1.1992 +/- 0.0444 | 3.32 +/- 0.15 | 0.5729 +/- 0.0233 | 0.7113 +/- 0.1184 | 1.3278 +/- 0.0111 | 0.6188 +/- 0.0243 | 0.550 |

### Paired Statistical Comparisons (CentroidSteer vs. Scientific Baselines — Multi-Domain MoE)

| Comparison | Metric | Mean Delta (CentroidSteer - Base) | 95% CI (Paired) | Cohen's d | Paired t-test p | Wilcoxon 2-sided p | Wilcoxon 1-sided p |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Val Loss | -0.0250 | [-0.0503, +0.0002] | -1.2334 | 0.0510 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Val PPL | -0.0845 | [-0.1701, +0.0011] | -1.2263 | 0.0518 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Domain Purity | -0.0718 | [-0.1838, +0.0403] | -0.7954 | 0.1499 | 0.3125 | 0.9062 |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Subspace Energy | +0.6188 | [+0.5886, +0.6489] | +25.4743 | 5.687e-07 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Val Loss | -0.0052 | [-0.0295, +0.0192] | -0.2633 | 0.5877 | 0.6250 | 0.3125 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Val PPL | -0.0172 | [-0.0977, +0.0633] | -0.2654 | 0.5848 | 0.6250 | 0.3125 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Domain Purity | -0.1077 | [-0.1988, -0.0166] | -1.4681 | 0.0304 | 0.0625 | 1.0000 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Subspace Energy | +0.6000 | [+0.5699, +0.6300] | +24.8111 | 6.320e-07 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Val Loss | +0.3685 | [+0.2995, +0.4374] | +6.6373 | 0.0001 | 0.0625 | 1.0000 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Val PPL | +1.0239 | [+0.8185, +1.2293] | +6.1888 | 0.0002 | 0.0625 | 1.0000 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Domain Purity | +0.1292 | [+0.0404, +0.2180] | +1.8066 | 0.0156 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Subspace Energy | +0.6188 | [+0.5886, +0.6489] | +25.4743 | 5.687e-07 | 0.0625 | 0.0312 |

---

## 3. Suite 3: Real FineWeb-Edu Cold-Start LM Pretraining (Pure-Routed MoE Transformer Cold-Start (FineWeb-Edu))

Dataset: `data/fineweb_edu_large.jsonl` (SHA-256 `0432004653835d87...`), `dim=256`, `layers=4`, `experts=4`, `rank=4`, `k_c=8`, `update_gap=20`, `steer_update_gap=2`, `steer_scale=1.0`, `lr=0.002`, `expert_lr=0.0015`, `steps=300`, `B=16xT=256`, `N=5` seeds.

| Regime | Val Loss (Mean +/- Std) | Val PPL | Val Loss Drop | Centroid Sep. | Subspace Energy | Expert State (B/p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `nosteer` (No-Steer Low-Rank) | 2.2472 +/- 0.0069 | 9.46 +/- 0.07 | 3.3695 +/- 0.0069 | 1.3132 +/- 0.0159 | 0.0000 +/- 0.0000 | 0.046 |
| `random_same_mem` (Memory-Matched Random Subspace) | 2.2585 +/- 0.0131 | 9.57 +/- 0.13 | 3.3582 +/- 0.0131 | 1.3239 +/- 0.0191 | 0.0180 +/- 0.0003 | 0.264 |
| `adamw` (Full-Rank AdamW) | 2.1500 +/- 0.0417 | 8.59 +/- 0.36 | 3.4667 +/- 0.0417 | 1.3231 +/- 0.0187 | 0.0000 +/- 0.0000 | 4.000 |
| `upgraded` (**CentroidSteer**) | 2.2012 +/- 0.0178 | 9.04 +/- 0.16 | 3.4155 +/- 0.0178 | 1.3513 +/- 0.0264 | 0.9223 +/- 0.0158 | 0.264 |

### Paired Statistical Comparisons (CentroidSteer vs. Scientific Baselines — 3. Suite 3: Real FineWeb-Edu Cold-Start LM Pretraining)

| Comparison | Metric | Mean Delta (CentroidSteer - Base) | 95% CI (Paired) | Cohen's d | Paired t-test p | Wilcoxon 2-sided p | Wilcoxon 1-sided p |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Val Loss | -0.0460 | [-0.0746, -0.0173] | -1.9940 | 0.0112 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Val PPL | -0.4242 | [-0.6855, -0.1629] | -2.0161 | 0.0108 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Subspace Energy | +0.9223 | [+0.9026, +0.9419] | +58.2816 | 2.079e-08 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Val Loss | -0.0573 | [-0.0735, -0.0412] | -4.4191 | 0.0006 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Val PPL | -0.5328 | [-0.6802, -0.3855] | -4.4893 | 0.0006 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Subspace Energy | +0.9043 | [+0.8850, +0.9237] | +57.9598 | 2.126e-08 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Val Loss | +0.0512 | [-0.0115, +0.1139] | +1.0142 | 0.0859 | 0.1250 | 0.9688 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Val PPL | +0.4462 | [-0.1008, +0.9932] | +1.0127 | 0.0862 | 0.1250 | 0.9688 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Subspace Energy | +0.9223 | [+0.9026, +0.9419] | +58.2816 | 2.079e-08 | 0.0625 | 0.0312 |

---

## 4. Suite 4: Real FineWeb-Edu 75.8M Checkpoint Continuation (75.8M Triune-Base Routed-Expert Continuation (checkpoints/triune_studio_step_36500.pt))

Dataset: `data/fineweb_edu_large.jsonl` (SHA-256 `0432004653835d87...`), `dim=384`, `layers=8`, `experts=4`, `rank=8`, `k_c=8`, `update_gap=10`, `steer_update_gap=2`, `steer_scale=0.75`, `lr=0.0005`, `expert_lr=0.0005`, `steps=100`, `B=4xT=128`, `N=5` seeds.

| Regime | Val Loss (Mean +/- Std) | Val PPL | Val Loss Drop | Centroid Sep. | Subspace Energy | Expert State (B/p) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `nosteer` (No-Steer Low-Rank) | 4.9945 +/- 0.0001 | 147.60 +/- 0.02 | 0.0005 +/- 0.0001 | 1.4425 +/- 0.0027 | 0.0000 +/- 0.0000 | 0.055 |
| `random_same_mem` (Memory-Matched Random Subspace) | 4.9945 +/- 0.0001 | 147.61 +/- 0.01 | 0.0004 +/- 0.0001 | 1.4429 +/- 0.0007 | 0.0122 +/- 0.0004 | 0.201 |
| `adamw` (Full-Rank AdamW) | 4.9954 +/- 0.0008 | 147.73 +/- 0.12 | -0.0004 +/- 0.0008 | 1.4316 +/- 0.0002 | 0.0000 +/- 0.0000 | 3.994 |
| `upgraded` (**CentroidSteer**) | 4.9934 +/- 0.0003 | 147.43 +/- 0.04 | 0.0016 +/- 0.0003 | 1.4391 +/- 0.0006 | 0.7054 +/- 0.0060 | 0.201 |

### Paired Statistical Comparisons (CentroidSteer vs. Scientific Baselines — 4. Suite 4: Real FineWeb-Edu 75.8M Checkpoint Continuation)

| Comparison | Metric | Mean Delta (CentroidSteer - Base) | 95% CI (Paired) | Cohen's d | Paired t-test p | Wilcoxon 2-sided p | Wilcoxon 1-sided p |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Val Loss | -0.0011 | [-0.0014, -0.0008] | -4.9212 | 0.0004 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Val PPL | -0.1644 | [-0.2059, -0.1230] | -4.9290 | 0.0004 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `nosteer` (No-Steer Low-Rank) | Subspace Energy | +0.7054 | [+0.6980, +0.7128] | +118.3853 | 1.222e-09 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Val Loss | -0.0012 | [-0.0015, -0.0008] | -4.5466 | 0.0005 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Val PPL | -0.1724 | [-0.2194, -0.1254] | -4.5511 | 0.0005 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `random_same_mem` (Memory-Matched Random Subspace) | Subspace Energy | +0.6932 | [+0.6856, +0.7008] | +112.8785 | 1.478e-09 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Val Loss | -0.0020 | [-0.0029, -0.0011] | -2.6927 | 0.0038 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Val PPL | -0.2950 | [-0.4312, -0.1588] | -2.6898 | 0.0038 | 0.0625 | 0.0312 |
| **CentroidSteer** vs `adamw` (Full-Rank AdamW) | Subspace Energy | +0.7054 | [+0.6980, +0.7128] | +118.3853 | 1.222e-09 | 0.0625 | 0.0312 |

---

*Generated automatically by `benchmarks/validate_centroid_steer_upgrade.py`.*
