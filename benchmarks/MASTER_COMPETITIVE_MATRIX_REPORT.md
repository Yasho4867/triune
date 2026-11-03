# Triune vs Unsloth vs Standard AdamW — Master 3x2x2 Competitive Matrix

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU`  
**Precision**: `torch.bfloat16` with native CUDA Tensor Cores  
**Timestamp**: 2026-11-03 12:06:38 UTC  
**Architecture Scope**: Universal Dense Transformers (76M) & Triune Multi-Exit MoE (76M) across Pre-Training and LoRA Fine-Tuning.

---

## 1. Pre-Training from Scratch (Dense vs MoE)

| Architecture | Framework Stack | Step Latency | Throughput | Peak VRAM | VRAM Saved vs AdamW | Loss Drop (15 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Universal Dense (76M) | Standard AdamW | 50.48 ms | 20,284 tok/s | **882.6 MB** | **0.0%** | 5.8562 |
| Universal Dense (76M) | Unsloth Style | 61.13 ms | 16,752 tok/s | **665.6 MB** | **24.6%** | 5.8598 |
| Universal Dense (76M) | **Triune Stack** | 62.4 ms | 16,410 tok/s | **504.5 MB** | **42.8%** | 6.8131 |
| Triune MoE (76M) | Standard AdamW | 157.48 ms | 6,503 tok/s | **1409.9 MB** | **0.0%** | 3.5229 |
| Triune MoE (76M) | Unsloth Style | 171.89 ms | 5,957 tok/s | **1157.0 MB** | **17.9%** | 3.5348 |
| Triune MoE (76M) | **Triune Stack** | 173.89 ms | 5,889 tok/s | **1094.4 MB** | **22.4%** | 4.1021 |

---

## 2. LoRA Fine-Tuning (Dense vs MoE)

| Architecture | Framework Stack | Step Latency | Throughput | Peak VRAM | VRAM Saved vs AdamW | Loss Drop (15 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Universal Dense (76M) | Standard AdamW | 61.15 ms | 16,745 tok/s | **716.1 MB** | **0.0%** | 0.3794 |
| Universal Dense (76M) | Unsloth Style | 82.34 ms | 12,436 tok/s | **586.1 MB** | **18.2%** | 0.3805 |
| Universal Dense (76M) | **Triune Stack** | 63.17 ms | 16,211 tok/s | **402.2 MB** | **43.8%** | 0.3805 |
| Triune MoE (76M) | Standard AdamW | 163.54 ms | 6,262 tok/s | **1058.5 MB** | **0.0%** | 0.4108 |
| Triune MoE (76M) | Unsloth Style | 173.62 ms | 5,898 tok/s | **932.7 MB** | **11.9%** | 0.4122 |
| Triune MoE (76M) | **Triune Stack** | 164.51 ms | 6,224 tok/s | **933.6 MB** | **11.8%** | 0.412 |

---

## 3. Workload-by-Workload Empirical Trade-Offs

1. **Dense LoRA Fine-Tuning (Strongest Triune Win)**:
   - **Throughput**: Triune runs at **63.17 ms/step (16,211 tok/s)**, outperforming Unsloth 8-bit AdamW (**82.34 ms/step**, **12,436 tok/s**, **30.4% speedup**) while performing competitively with standard AdamW (61.15 ms/step).
   - **Memory**: Triune uses only **402.2 MB peak VRAM**, representing a **43.8% reduction vs standard AdamW (716.1 MB)** and a **31.4% reduction vs Unsloth 8-bit AdamW (586.1 MB)** due to fused linear-LoRA and chunked logit-free backprop.

2. **MoE LoRA Fine-Tuning**:
   - Triune runs at **164.51 ms/step** (vs Unsloth 173.62 ms/step and AdamW 163.54 ms/step) while matching Unsloth in peak VRAM (**933.6 MB vs 932.7 MB**, an 11.8% reduction vs standard AdamW's 1058.5 MB).

3. **Dense Pre-Training from Scratch (Memory & Convergence Win, Throughput Trade-off)**:
   - **Memory**: Triune slashes peak VRAM to **504.5 MB**, saving **42.8% vs standard AdamW (882.6 MB)** and **24.2% vs Unsloth (665.6 MB)**.
   - **Convergence**: CentroidSteer + Muon achieves **6.8131 loss drop** over 15 steps (+16.3% faster initial loss reduction vs 5.8562 for AdamW and 5.8598 for Unsloth).
   - **Throughput**: Triune pays a ~23% latency penalty (62.40 ms vs 50.48 ms for unquantized AdamW) due to online subspace projections.

4. **MoE Pre-Training from Scratch**:
   - Triune uses the lowest peak VRAM (**1,094.4 MB**, saving 22.4% vs AdamW's 1,409.9 MB and 5.4% vs Unsloth's 1,157.0 MB) while achieving the strongest loss reduction (**4.1021 vs 3.5229 AdamW and 3.5348 Unsloth**), running at comparable latency to Unsloth (173.89 ms vs 171.89 ms).
