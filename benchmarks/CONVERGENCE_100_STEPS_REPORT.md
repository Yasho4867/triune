# Triune Multi-Seed 100-Step Convergence & Loss Trajectory Report
**Empirical Multi-Seed Optimizer Evaluation with Parity Initializations on Real FineWeb-Edu**

- **Date / Timestamp**: `2026-11-04 03:48:26 UTC`
- **Git Commit**: [`c879ed771c`](https://github.com/Yasho4867/triune/commit/c879ed771ce91c09761ad29060443d04821f98d3) (Dirty Working Tree)
- **Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU` (7.93 GB VRAM, CC 12.0)
- **Software**: PyTorch `2.14.1+cu132` | CUDA `13.2` | Python `3.14.7`
- **Dataset**: Real FineWeb-Edu educational text (`data\fineweb_edu_real.jsonl`, SHA-256 `aed10d4c98af...`)
- **Model Topology**: Universal Dense Transformer (43,456,896 parameters, 8 layers, 384 dim, 6 heads, 32,000 vocab)
- **Experimental Protocol**:
  - **Random Seeds**: `42, 1337, 2026` ($N=3$ independent runs per regime)
  - **Step-0 Initial Weights**: Identical per seed (sampled once and cloned bit-for-bit into all 3 regimes)
  - **Batch Sequences**: Identical causal batches fed in exact synchronous order across all regimes per seed
  - **Validation Set**: Multi-batch held-out split of 5 batches (2,560 tokens) evaluated every 20 steps
  - **Loss Function Configuration**: `standard` (`F.cross_entropy` applied symmetrically across all regimes to isolate pure optimizer performance)

---

## 1. Multi-Seed Optimization & Convergence Summary (Mean ± Std)

| Framework & Optimizer | Initial Val Loss | Final Val Loss (100 Steps) | Val Loss Drop | Final Val PPL | Median Step Latency | Total Wall-Clock Time | Peak VRAM Alloc | Convergence Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard AdamW (`adamw`) | 10.5199 ± 0.0196 | **7.4300 ± 0.0047** | **3.0899 ± 0.0153** | **1685.78 ± 7.88** | 19.80 ± 0.26 ms | 2.21 ± 0.33 s | **683.4 MB** | **1.4149 loss/s** |
| BitsAndBytes 8-bit AdamW (`bnb_8bit`) | 10.5199 ± 0.0196 | **7.4810 ± 0.0510** | **3.0389 ± 0.0516** | **1775.52 ± 91.86** | 24.16 ± 4.33 ms | 2.48 ± 0.42 s | **601.1 MB** | **1.2483 loss/s** |
| **Triune Stack** (`centroid_steer`) | 10.5199 ± 0.0196 | **7.3605 ± 0.0053** | **3.1594 ± 0.0232** | **1572.63 ± 8.30** | 20.93 ± 4.70 ms | 2.26 ± 0.62 s | **673.7 MB** | **1.4644 loss/s** |

---

## 2. Validation Loss & Perplexity Trajectory (Mean ± Std across Seeds)

Evaluated over the held-out validation set (2,560 tokens) at intervals of 20 steps:

| Step | Metric | Standard AdamW | BitsAndBytes 8-bit AdamW | Triune CentroidSteer |
| :---: | :--- | :---: | :---: | :---: |
| **Step 0** | Val Loss | 10.5199 ± 0.0196 | 10.5199 ± 0.0196 | **10.5199 ± 0.0196** |
| | Val PPL | 37050.82 ± 724.69 | 37050.82 ± 724.69 | **37050.82 ± 724.69** |
| **Step 20** | Val Loss | 8.0637 ± 0.0117 | 8.0881 ± 0.0212 | **8.0545 ± 0.0192** |
| | Val PPL | 3177.10 ± 37.20 | 3255.85 ± 69.51 | **3148.48 ± 60.18** |
| **Step 40** | Val Loss | 7.8382 ± 0.0495 | 7.9815 ± 0.2615 | **7.8051 ± 0.0443** |
| | Val PPL | 2537.63 ± 123.90 | 2996.35 ± 823.39 | **2454.72 ± 107.33** |
| **Step 60** | Val Loss | 7.8041 ± 0.0437 | 7.8834 ± 0.1216 | **7.7437 ± 0.0412** |
| | Val PPL | 2452.13 ± 108.22 | 2666.25 ± 332.77 | **2308.23 ± 95.72** |
| **Step 80** | Val Loss | 7.6643 ± 0.0311 | 7.6904 ± 0.0467 | **7.5987 ± 0.0533** |
| | Val PPL | 2131.47 ± 66.37 | 2188.71 ± 101.06 | **1997.51 ± 107.15** |
| **Step 100** | Val Loss | 7.4300 ± 0.0047 | 7.4810 ± 0.0510 | **7.3605 ± 0.0053** |
| | Val PPL | 1685.78 ± 7.88 | 1775.52 ± 91.86 | **1572.63 ± 8.30** |

---

## 3. Methodological Discussion & Empirical Insights

1. **Ablation of Confounding Factors**:
   - In earlier single-seed pilot iterations, the Triune training loop coupled the `CentroidSteerOptimizer` with `chunked_cross_entropy_from_hidden`, which conflated optimizer convergence with loss function gradient precision.
   - In this multi-seed benchmark, all regimes (`AdamW`, `AdamW8bit`, `CentroidSteer`) use the **exact same PyTorch cross-entropy loss function** on identical float representations.
   - All models begin at Step 0 from the **exact same model weights** for each random seed, and consume the **exact same batches** in identical order.
   - Consequently, the observed differences in loss trajectory isolate the algorithmic contribution of the optimizer stack.

2. **Mechanism of CentroidSteer**:
   - `CentroidSteerOptimizer` partitions model parameters into:
     - **2D Internal Weights (Attention projections & MLP layers)**: Optimized via Muon (Newton-Schulz orthogonalization) coupled with low-rank triangular subspace projection ($R$).
     - **1D / Non-Matrix Parameters (Norms, Biases)**: Handled by 8-bit AdamW base.
     - **Vocabulary & Embedding Tables**: Categorized via direct `nn.Embedding` module inspection and aspect-ratio checks ($\text{dim} \ge 1000, \text{ratio} \ge 4.0$), routing them to 8-bit AdamW base to avoid catastrophic singular value equalization on sparse token tables.
   - This orthogonal gradient steering enables larger stable effective updates along dominant singular directions without gradient explosion or dispersion.

3. **Scope and Calibration**:
   - These 100-step experiments represent an initial convergence calibration on FineWeb-Edu. While the multi-seed results demonstrate consistent loss reduction advantages during early pretraining phase, longer-horizon pretraining runs (10k-50k steps) will be performed to establish whether this differential persists across full pretraining budgets.
