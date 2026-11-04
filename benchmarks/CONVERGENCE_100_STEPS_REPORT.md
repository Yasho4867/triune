# Triune Multi-Seed 100-Step Convergence & Loss Trajectory Report
**Empirical Multi-Seed Optimizer Evaluation & Component Ablation on Real FineWeb-Edu**

- **Date / Timestamp**: `2026-11-04 10:21:52 UTC`
- **Git Commit**: [`e0effdc53e`](https://github.com/Yasho4867/triune/commit/e0effdc53e908a3ea01446818009778dd1730e53) (Clean Tree)
- **Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU` (7.93 GB VRAM, CC 12.0)
- **Software**: PyTorch `2.14.1+cu132` | CUDA `13.2` | Python `3.14.7`
- **Dataset**: Real FineWeb-Edu educational text (`data\fineweb_edu_real.jsonl`, SHA-256 `aed10d4c98af...`)
- **Model Topology**: Universal Dense Transformer (43,456,896 parameters, 8 layers, 384 dim, 6 heads, 32,000 vocab)
- **Experimental Protocol**:
  - **Random Seeds**: `42, 1337, 2026` ($N=3$ independent runs per regime)
  - **Step-0 Initial Weights**: Identical per seed (sampled once and cloned bit-for-bit into all regimes)
  - **Batch Sequences**: Identical causal batches fed in exact synchronous order across all regimes per seed
  - **Validation Set**: Multi-batch held-out split of 5 batches (2,560 tokens) evaluated every 20 steps
  - **Loss Function Configuration**: `standard` (`F.cross_entropy` applied symmetrically across all regimes to isolate pure optimizer performance)
  - **Base Optimizer Implementation**: `torch.optim.AdamW (fused=True)` (handles 1D normalization/biases and protected vocabulary embeddings)

---

## 1. Active Parameter Tier Distribution Across Regimes

The Triune optimization architecture dynamically partitions model weights into three specialized mathematical tiers:
- **Tier 1 (Muon)**: Newton-Schulz 5th-order orthogonalization for 2D hidden linear weights.
- **Tier 2 (CentroidSteer)**: Low-rank triangular subspace projection ($R$) along router activation centroids for MoE routed experts.
- **Tier 3 (Base AdamW)**: Fused AdamW (or 8-bit AdamW) for 1D parameters, layer norms, biases, and protected vocabulary embeddings.

On `UniversalDenseTransformer` (43,456,896 total parameters):

| Framework & Optimizer Regime | Total Trainable | Tier 1: Muon (2D Hidden) | Tier 2: CentroidSteer (Subspace) | Tier 3: Base AdamW (1D / Vocab) | Dynamic MoE Centroids? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard AdamW (`torch.optim.AdamW`) | 43,456,896 | 0 (0.0%) | 0 (0.0%) | 43,456,896 (100.0%) | No (Dense Model) |
| BitsAndBytes 8-bit AdamW (`AdamW8bit`) | 43,456,896 | 0 (0.0%) | 0 (0.0%) | 43,456,896 (100.0%) | No (Dense Model) |
| Muon Baseline (Muon 2D + Fused AdamW Base) | 43,456,896 | 4,718,592 (10.9%) | 0 (0.0%) | 38,738,304 (89.1%) | No (Dense Model) |
| CentroidSteer (No Muon: Subspace + AdamW) | 43,456,896 | 0 (0.0%) | 4,718,592 (10.9%) | 38,738,304 (89.1%) | No (Dense Model) |
| **Triune Stack** (Muon + CentroidSteer + AdamW Base) | 43,456,896 | 4,718,592 (10.9%) | 0 (0.0%) | 38,738,304 (89.1%) | No (Dense Model) |

---

## 2. Multi-Seed Optimization & Convergence Summary (Mean ± Std)

| Framework & Optimizer Regime | Initial Val Loss | Final Val Loss (100 Steps) | Val Loss Drop | Final Val PPL | Steady-State Step Latency | Setup Step 1 Latency | Total Wall-Clock Time | Peak VRAM Alloc | Convergence Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Standard AdamW (`torch.optim.AdamW`) | 10.5453 ± 0.0134 | **7.6621 ± 0.0577** | **2.8832 ± 0.0654** | **2128.65 ± 122.09** | 32.86 ± 0.33 ms | 45.71 ± 1.61 ms | 3.39 ± 0.09 s | **714.9 MB** | **0.8511 loss/s** |
| BitsAndBytes 8-bit AdamW (`AdamW8bit`) | 10.5453 ± 0.0134 | **7.9332 ± 0.0741** | **2.6121 ± 0.0617** | **2793.31 ± 202.37** | 37.07 ± 0.35 ms | 61.59 ± 4.37 ms | 3.81 ± 0.04 s | **632.5 MB** | **0.6851 loss/s** |
| Muon Baseline (Muon 2D + Fused AdamW Base) | 10.5453 ± 0.0134 | **7.5280 ± 0.0061** | **3.0173 ± 0.0123** | **1859.33 ± 11.34** | 31.00 ± 0.34 ms | 39.74 ± 2.77 ms | 3.18 ± 0.05 s | **863.2 MB** | **0.9485 loss/s** |
| CentroidSteer (No Muon: Subspace + AdamW) | 10.5453 ± 0.0134 | **7.4967 ± 0.0111** | **3.0485 ± 0.0029** | **1802.22 ± 19.85** | 33.39 ± 2.93 ms | 110.06 ± 11.31 ms | 3.55 ± 0.33 s | **924.8 MB** | **0.8642 loss/s** |
| **Triune Stack** (Muon + CentroidSteer + AdamW Base) | 10.5453 ± 0.0134 | **7.5280 ± 0.0061** | **3.0173 ± 0.0123** | **1859.33 ± 11.34** | 31.65 ± 1.37 ms | 40.77 ± 1.10 ms | 3.28 ± 0.17 s | **863.2 MB** | **0.9213 loss/s** |

---

## 3. Validation Loss & Perplexity Trajectory Across Training Steps

Evaluated over the held-out validation set (2,560 tokens) at intervals of 20 steps:

| Step | Metric | Standard AdamW | BitsAndBytes 8-bit | Muon Baseline | CentroidSteer (No Muon) | Triune Stack |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Step 0** | Val Loss | 10.5453 ± 0.0134 | 10.5453 ± 0.0134 | 10.5453 ± 0.0134 | 10.5453 ± 0.0134 | **10.5453 ± 0.0134** |
| | Val PPL | 38000.12 ± 507.14 | 38000.12 ± 507.14 | 38000.12 ± 507.14 | 38000.12 ± 507.14 | **38000.12 ± 507.14** |
| **Step 20** | Val Loss | 8.2845 ± 0.0275 | 8.3095 ± 0.1035 | 8.2634 ± 0.1263 | 8.2260 ± 0.1169 | **8.2634 ± 0.1263** |
| | Val PPL | 3962.94 ± 109.59 | 4076.73 ± 415.75 | 3899.79 ± 491.70 | 3754.08 ± 442.01 | **3899.79 ± 491.70** |
| **Step 40** | Val Loss | 8.0110 ± 0.0495 | 8.0763 ± 0.0990 | 7.9166 ± 0.0996 | 7.8911 ± 0.1323 | **7.9166 ± 0.0996** |
| | Val PPL | 3016.50 ± 150.98 | 3227.80 ± 318.35 | 2751.39 ± 272.28 | 2688.95 ± 352.26 | **2751.39 ± 272.28** |
| **Step 60** | Val Loss | 7.8857 ± 0.1456 | 8.0900 ± 0.1543 | 7.7432 ± 0.0805 | 7.7584 ± 0.0938 | **7.7432 ± 0.0805** |
| | Val PPL | 2677.44 ± 373.51 | 3287.33 ± 494.17 | 2310.97 ± 188.70 | 2348.15 ± 221.28 | **2310.97 ± 188.70** |
| **Step 80** | Val Loss | 7.8480 ± 0.0978 | 8.0755 ± 0.1188 | 7.6859 ± 0.0447 | 7.6863 ± 0.0434 | **7.6859 ± 0.0447** |
| | Val PPL | 2568.86 ± 256.81 | 3230.14 ± 394.31 | 2178.93 ± 98.56 | 2179.78 ± 94.42 | **2178.93 ± 98.56** |
| **Step 100** | Val Loss | 7.6621 ± 0.0577 | 7.9332 ± 0.0741 | 7.5280 ± 0.0061 | 7.4967 ± 0.0111 | **7.5280 ± 0.0061** |
| | Val PPL | 2128.65 ± 122.09 | 2793.31 ± 202.37 | 1859.33 ± 11.34 | 1802.22 ± 19.85 | **1859.33 ± 11.34** |

---

## 4. Component Ablation & Mechanistic Isolation

This benchmark rigorously evaluates the component tiers of the Triune optimization stack and yields four key empirical and architectural insights:

1. **Architecture-Specific Tier Partitioning & Dense Model Equivalence**:
   - `UniversalDenseTransformer` is a dense transformer architecture featuring standard multi-head attention and dense feed-forward blocks; it contains **no mixture-of-experts (`MoE_FFN`) modules**.
   - In `CentroidSteerOptimizer`, routed expert weights are assigned to Tier 2 (CentroidSteer subspace projection), while 2D hidden linear weights (attention projections and MLP up/down projections) are routed to Tier 1 (Muon orthogonalization) when `use_muon=True`.
   - Consequently, on dense models, Tier 2 contains **0 parameters** ($0.0\%$). The "Full Triune Stack" and the "Muon Baseline" execute the **exact same code paths** with identical parameter partitions: $4,718,592$ parameters ($10.86\%$) in Tier 1 (Muon) and $38,738,304$ parameters ($89.14\%$) in Tier 3 (Fused AdamW for embeddings, lm_head, layer norms, and biases).
   - This architectural reality fully accounts for the close convergence agreement observed between the Full Triune Stack (7.5280 ± 0.0061) and the Muon baseline (7.5280 ± 0.0061).

2. **CentroidSteer (No Muon) vs. Muon and Standard AdamW**:
   - When Muon is disabled (`use_muon=False`), all $4,718,592$ 2D hidden weights are routed to Tier 2 low-rank triangular subspace projection (GaLore fallback with steering scale $0.20$).
   - CentroidSteer without Muon achieved a final validation loss of **7.4967 ± 0.0111**, improving over standard AdamW (7.6621 ± 0.0577) and performing closely with pure Muon (7.5280 ± 0.0061).
   - **Latency Profile**: The initial step (Step 1) requires computing the initial SVD projection matrices across all 2D layer groups, which incurs a one-time setup latency. Once initialized, steady-state step latency remains fast and closely competitive with AdamW and Muon.

3. **Dynamic Router Centroid Steering in MoE Architectures**:
   - Full 3-tier synergy—simultaneous execution of Tier 1 Muon, Tier 2 dynamic centroid steering, and Tier 3 AdamW—is designed for and activates in Mixture-of-Experts models (`UniversalMoETransformer`).
   - In `UniversalMoETransformer` (8 layers, 4 experts per layer, 90.79M total parameters):
     - **Tier 1 (Muon)**: $5,898,240$ parameters ($6.50\%$)—attention projections and shared expert matrices.
     - **Tier 2 (CentroidSteer)**: $35,389,440$ parameters ($38.98\%$)—routed MoE expert weights dynamically steered along router cluster centroids.
     - **Tier 3 (Fused AdamW)**: $49,501,591$ parameters ($54.52\%$)—router gates, vocabulary embeddings, and layer norms.
   - For empirical multi-seed validation of all three tiers active on real FineWeb-Edu text, see [`benchmarks/MOE_CONVERGENCE_100_STEPS_REPORT.md`](MOE_CONVERGENCE_100_STEPS_REPORT.md).
   - Dynamic router centroid steering accelerates multi-expert convergence by projecting updates into the subspace aligned with expert token assignments.

4. **Base Optimizer & Vocabulary Safeguards**:
   - Both Muon and CentroidSteer protect 1D parameters (RMSNorm, biases) and vocabulary embeddings (`wte`, `lm_head`) from singular value equalization, routing them to **PyTorch Fused AdamW** by default (or BitsAndBytes 8-bit AdamW when `--use-8bit-base` is enabled).
   - Embedding matrices are safely detected and preserved via module type (`nn.Embedding`), tied weight pointer detection, and large vocabulary aspect-ratio checks ($\text{dim} \ge 1000, \text{ratio} \ge 4.0$).

---

## 5. Cryptographic Provenance & Reproduction

To reproduce these exact results from a clean repository checkout:

```bash
# Clean reproduction command
python scripts/benchmark_convergence_100_steps.py --steps 100 --seeds 42 1337 2026 --num-val-batches 5 --include-ablation
```

### Source File Cryptographic Verification Hashes (SHA-256)
| Relative Path | SHA-256 Checksum |
| :--- | :--- |
| `data/fineweb_edu_real.jsonl` | `aed10d4c98afb6da4f55ee3aa295981465b9fd49edb1e5886c36fb6fe3ae8e9c` |
| `scripts/benchmark_convergence_100_steps.py` | `8f84c70ceec0d2c2a718c178b2d8072d4d681ecc737786ca43df3b6a0843ea91` |
| `triune/kernels/fast_gla.py` | `e0b8cb1e519cd3703e14e2a0fd9ed900aca1f514b48110ce01a0e363a9028a27` |
| `triune/kernels/fast_muon.py` | `e321668ca1fa8099fb076a8c34bcf2adfe4743bb23e96678e4ef3e3ccf30d4ba` |
| `triune/kernels/fast_triangular.py` | `2e6f39a8812797ea5855f7e64fd78f7244db38c46525ad3049f05ecdbd0fa9ce` |
| `triune/model/config.py` | `df1ce36506111810dc73d6edc444eb18300b5d7a4bc8b277bd84e7ecac18f413` |
| `triune/optim/centroid.py` | `d778e1852356421026e942507c0ec8898f368e24a4095fe4ce654fc9f9fc26d0` |
| `triune/optim/muon.py` | `df0b12554b0c9d5b6b88dcab2244d008e370d6ab190d09b1417c9cbde6dbbbf9` |
| `triune/universal.py` | `d5bbe31b88809915535fba209fd7269e093d3b82cbb1662dc86c1fdaddbb2495` |
| `triune_tokenizer.json` | `df2c43cfab506acce484e8c45a12e497a7901622fa60058fd4cd5802f4e36451` |
