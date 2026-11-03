# Triune vs BitsAndBytes 8-bit AdamW vs Standard AdamW — Master 3x2x2 Competitive Matrix

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU`  
**Precision**: `torch.bfloat16` with native CUDA Tensor Cores  
**Timestamp**: 2026-11-03 12:34:42 UTC  
**Git Provenance Commit**: `94601be9eb0d75da7e6c8fa3fff50497e66cf03a`  
**Architecture Scope**: Universal Dense Transformers (76M) & Triune Multi-Exit MoE (76M) across Pre-Training and LoRA Fine-Tuning.

> [!NOTE]
> **Competitor Scope & Naming**:
> - **Standard AdamW**: PyTorch's native `torch.optim.AdamW`.
> - **BitsAndBytes 8-bit AdamW**: Genuine native `bitsandbytes.optim.AdamW8bit` (the core 8-bit optimizer engine used by Unsloth) evaluated on identical Triune model architectures.
> - **Triune Stack**: `CentroidSteerOptimizer` (3-tier Muon + Triangular Core CentroidSteer + 8-bit AdamW base) during pre-training; fused `addmm_` LoRA + chunked logit-free cross-entropy during fine-tuning.
> - **Convergence Accounting**: Evaluation loss is measured on a **fixed held-out evaluation batch** (seed 999) before warmup and after the 15 training steps.

---

## 1. Pre-Training from Scratch (Dense vs MoE)

| Architecture | Framework & Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM Saved vs AdamW | Held-Out Eval Loss Drop |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Universal Dense (76M) | Standard AdamW (`AdamW`) | 42.43 ± 9.0 ms | 24,134 tok/s | **947.6 MB** | **0.0%** | -0.1739 |
| Universal Dense (76M) | BitsAndBytes 8-bit AdamW (`AdamW8bit`) | 49.17 ± 3.75 ms | 20,827 tok/s | **729.9 MB** | **23.0%** | -0.1793 |
| Universal Dense (76M) | **Triune Stack** (`CentroidSteerOptimizer`) | 53.78 ± 4.95 ms | 19,040 tok/s | **570.1 MB** | **39.8%** | -0.1648 |
| Triune MoE (76M) | Standard AdamW (`AdamW`) | 141.92 ± 9.16 ms | 7,216 tok/s | **1470.7 MB** | **0.0%** | -0.6916 |
| Triune MoE (76M) | BitsAndBytes 8-bit AdamW (`AdamW8bit`) | 196.15 ± 12.49 ms | 5,220 tok/s | **1215.6 MB** | **17.3%** | -0.6872 |
| Triune MoE (76M) | **Triune Stack** (`CentroidSteerOptimizer`) | 159.25 ± 8.1 ms | 6,430 tok/s | **1152.8 MB** | **21.6%** | -0.3041 |

---

## 2. LoRA Fine-Tuning (Dense vs MoE)

| Architecture | Framework & Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM Saved vs AdamW | Held-Out Eval Loss Drop |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Universal Dense (76M) | Standard AdamW (`AdamW`) | 62.88 ± 11.56 ms | 16,285 tok/s | **779.1 MB** | **0.0%** | 0.0007 |
| Universal Dense (76M) | BitsAndBytes 8-bit AdamW (`AdamW8bit`) | 69.56 ± 11.71 ms | 14,721 tok/s | **651.0 MB** | **16.4%** | 0.0009 |
| Universal Dense (76M) | **Triune Stack** (`AdamW`) | 78.98 ± 5.71 ms | 12,966 tok/s | **465.1 MB** | **40.3%** | 0.0008 |
| Triune MoE (76M) | Standard AdamW (`AdamW`) | 153.66 ± 8.9 ms | 6,664 tok/s | **1118.8 MB** | **0.0%** | -0.0022 |
| Triune MoE (76M) | BitsAndBytes 8-bit AdamW (`AdamW8bit`) | 156.59 ± 16.95 ms | 6,539 tok/s | **991.1 MB** | **11.4%** | -0.0017 |
| Triune MoE (76M) | **Triune Stack** (`AdamW`) | 154.17 ± 7.66 ms | 6,642 tok/s | **992.0 MB** | **11.3%** | -0.0017 |

---

## 3. Workload-by-Workload Empirical Trade-Offs

1. **Dense LoRA Fine-Tuning**:
   - **Memory**: Triune uses **465.1 MB peak VRAM**, achieving a **40.3% reduction vs Standard AdamW (779.1 MB)** and a **28.6% reduction vs BitsAndBytes 8-bit AdamW (651.0 MB)** due to chunked cross-entropy and fused adapter backprop.
   - **Throughput**: Triune runs at **78.98 ± 5.71 ms/step (12,966 tok/s)** vs **69.56 ± 11.71 ms/step** for BitsAndBytes 8-bit AdamW and **62.88 ± 11.56 ms/step** for standard AdamW.

2. **MoE LoRA Fine-Tuning**:
   - **Latency & Memory**: Triune runs at **154.17 ± 7.66 ms/step** (6,642 tok/s) and **992.0 MB peak VRAM** (11.3% lower VRAM than Standard AdamW's 1118.8 MB, operating at near parity with BitsAndBytes's 991.1 MB and 156.59 ms/step).

3. **Dense Pre-Training from Scratch (VRAM Efficiency Trade-Off)**:
   - **Memory**: Triune reduces peak VRAM to **570.1 MB**, saving **39.8% vs standard AdamW (947.6 MB)** and **21.9% vs BitsAndBytes 8-bit AdamW (729.9 MB)**.
   - **Throughput**: Latency is **53.78 ± 4.95 ms/step** (19,040 tok/s), trading ~25% raw speed vs unquantized AdamW (42.43 ms) in exchange for dramatic VRAM savings and subspace gradient tracking.

4. **MoE Pre-Training from Scratch (Speed & Memory Win over 8-Bit Baseline)**:
   - **Speed**: Triune runs at **159.25 ± 8.1 ms/step**, outperforming BitsAndBytes 8-bit AdamW (196.15 ms/step) by **18.8%**.
   - **Memory**: Triune uses the lowest peak VRAM in this category at **1152.8 MB**, saving **21.6% vs Standard AdamW (1470.7 MB)** and **5.2% vs BitsAndBytes 8-bit AdamW (1215.6 MB)**.
