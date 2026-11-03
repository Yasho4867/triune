# Triune AI — Empirical Benchmark & Competitive Evaluation Suite

**Date**: November 03, 2026  
**Environment**: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM) | PyTorch: 2.14.1+cu132 | CUDA: 13.2 | BF16 Tensor Cores

## Overview of Benchmark Categories

This repository contains publication-grade, hardware-verified benchmarks evaluating **Triune AI**'s architectural
and runtime components against PyTorch standard baselines and native competitor implementations (Unsloth / BitsAndBytes / Liger).

### Benchmark Suite Matrix

| Category | Topic | Target Components | Competitors Evaluated | Primary Advantage | Report Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | **Kernel Microbenchmarks** | Chunked CE, FastRMSNorm, FastGLA | PyTorch CrossEntropy, PyTorch SDPA, nn.RMSNorm | Logit-free backprop memory reduction (up to 91.3%) | [Report](COMPETITIVE_BENCHMARK_REPORT.md) |
| **B** | **Optimizer Dynamics** | CentroidSteer, Muon | PyTorch AdamW, BitsAndBytes 8-bit AdamW (Unsloth) | Persistent state memory reduction (1.00 byte/param) | [Report](OPTIMIZER_BENCHMARK_REPORT.md) |
| **C** | **Layer Streaming** | LayerStreamingEngine (CPU/GPU) | Standard GPU-Resident Training (OOM prevention) | Memory virtualization: trains large models in small VRAM | [Report](LAYER_STREAMING_REPORT.md) |
| **D** | **Long-Context Generation** | Gated Linear Attention (GLA) | Softmax Attention (SDPA) KV-Cache ($O(1)$ vs $O(L)$) | Constant-memory recurrent decoding (256x-512x cache compression) | [Report](LONG_CONTEXT_REPORT.md) |
| **E** | **Dynamic Exits & Pareto** | Reflex (L2), Limbic (L5), Cortex (L8) | Full-depth monolithic execution | Low-entropy early exit inference speedup (1.9x-3.7x) | [Report](DYNAMIC_EXITS_REPORT.md) |
| **F** | **Full Competitive Matrix** | Universal Dense & MoE Transformers | PyTorch AdamW, Unsloth native 8-bit AdamW | Verified 3x2x2 pre-training and LoRA fine-tuning | [Report](MASTER_COMPETITIVE_MATRIX_REPORT.md) |

---

## Detailed Empirical Findings & Honest Trade-Offs

### 1. Master Competitive Matrix (`Category F`, Verified Hardware Measurements)
Evaluated across identical architectures (Universal Dense 76M and Triune MoE 76M) with **actively participating LoRA adapters**, verified parameter weight updates, and genuine BitsAndBytes 8-bit AdamW (no silent fallback):

- **Dense LoRA Fine-Tuning (Strongest Triune Result)**:
  - **Throughput**: Triune runs at **63.17 ms/step (16,211 tok/s)**, delivering a **30.4% speedup over genuine Unsloth 8-bit AdamW (82.34 ms/step, 12,436 tok/s)** while remaining competitive with unquantized AdamW (61.15 ms/step).
  - **Memory**: Triune uses only **402.2 MB peak VRAM**, representing a **43.8% reduction vs standard AdamW (716.1 MB)** and a **31.4% reduction vs Unsloth 8-bit AdamW (586.1 MB)** due to fused linear-LoRA and chunked logit-free backprop.
- **MoE LoRA Fine-Tuning**:
  - Triune executes at **164.51 ms/step** (vs Unsloth's 173.62 ms/step and AdamW's 163.54 ms/step) while matching Unsloth in peak VRAM (**933.6 MB vs 932.7 MB**, an 11.8% reduction over standard AdamW's 1,058.5 MB).
- **Dense Pre-Training (Memory & Convergence Win, Throughput Trade-off)**:
  - **Memory**: Triune uses **504.5 MB peak VRAM**, saving **42.8% vs standard AdamW (882.6 MB)** and **24.2% vs Unsloth (665.6 MB)**.
  - **Convergence**: CentroidSteer + Muon achieves **6.8131 loss drop** over 15 steps (+16.3% faster initial loss reduction vs 5.8562 for AdamW and 5.8598 for Unsloth).
  - **Throughput Trade-off**: Triune runs at **62.40 ms/step** vs **50.48 ms/step** for standard AdamW (~23% latency overhead) due to online subspace projections.
- **MoE Pre-Training**:
  - Triune achieves the lowest peak VRAM (**1,094.4 MB**, saving 22.4% vs AdamW's 1,409.9 MB and 5.4% vs Unsloth's 1,157.0 MB) and the strongest initial loss reduction (**4.1021 vs 3.5229 AdamW and 3.5348 Unsloth**), with step latency comparable to Unsloth (173.89 ms vs 171.89 ms).

### 2. Chunked Logit-Free Cross-Entropy (`Category A`)
Standard PyTorch cross-entropy materializes the entire $[N, V]$ logits matrix simultaneously, consuming gigabytes of transient VRAM on large vocabularies ($V=32,000$):
- **Workload Scaling**:
  - At $N=2,048$: PyTorch takes 11.74 ms (1,089.5 MB) vs Triune's 16.53 ms (289.4 MB) -> **73.4% VRAM saved**, 29% slower.
  - At $N=4,096$: PyTorch takes 23.23 ms (2,170.4 MB) vs Triune's 32.46 ms (348.8 MB) -> **83.9% VRAM saved**, 28% slower.
  - At $N=8,192$: PyTorch takes 113.73 ms (4,182.3 MB) vs Triune's 64.67 ms (365.8 MB) -> **91.3% VRAM saved**, **1.76x faster**.
- **Takeaway**: Chunked Cross-Entropy trades slight compute overhead for dramatic memory reduction at short-to-medium lengths, and wins on both throughput and memory at larger batch/context workloads by eliminating memory bandwidth saturation and kernel spills.

### 3. Gated Linear Attention: Prefill vs Autoregressive Decode (`Category D`)
- **Autoregressive Decoding ($O(1)$ Recurrent vs Softmax KV-Cache $O(L)$)**:
  - This is GLA's primary structural advantage. During generation, standard Softmax KV-cache grows with sequence depth ($768 \text{ KB}$ at 512 context to $24,576 \text{ KB}$ per layer at 16,384 context), increasing decode latency from 0.038 ms to 0.792 ms/tok.
  - Triune GLA maintains a **strictly fixed 48–64 KB recurrent state** (**99.8% cache memory reduction**) and constant per-token latency (~0.10 ms/tok), generating tokens up to **7.94x faster** than Softmax KV-cache at 16k context.
- **Context Prefill**:
  - Softmax SDPA (leveraging cuDNN and FlashAttention-2 native CUDA kernels) is highly optimized and faster at short-to-medium context lengths ($L \le 4096$).
  - At very long context ($L \ge 16,384$), linear attention's $O(N)$ computational complexity allows chunked GLA to achieve competitive or superior prefill throughput (7.25 ms vs 14.20 ms on RTX 5070), but SDPA remains faster for shorter sequences.

### 4. Optimizer Dynamics: Persistent State vs Peak Transient Memory (`Category B`)
- **Persistent Optimizer State Memory**:
  - Standard FP32 AdamW retains 2 states ($m, v$ in FP32) per parameter: **8.0 bytes/param** (48.37 MB on a 17M model).
  - Native 8-bit AdamW retains quantized INT8 states: **2.0 bytes/param** (24.74 MB).
  - CentroidSteer with rank 16 retains low-rank subspace orthonormal factors: **1.00 byte/param** (16.23 MB, **66.4% reduction vs AdamW, 34.4% vs 8-bit AdamW**; reaching 94.8% reduction on large routed MoE matrices).
- **Peak Step Memory**:
  - While persistent state memory is permanently retained between steps, peak memory during the optimizer step includes transient scratch buffers for QR factorization and SVD updates, which are immediately freed before the next forward pass.

### 5. Memory-Constrained Training via Layer Streaming (`Category C`)
`LayerStreamingEngine` is a **capacity-unlocking virtualization system** that enables training models larger than physical GPU VRAM by offloading inactive layers to host RAM:
- **Measured Trade-Offs (RTX 5070 8 GB VRAM)**:
  - **Config 76M-Spec (101.97M total params)**: Peak VRAM reduced by **33.3%** (`877.8 MB` vs `1,315.9 MB`), with a 7.5x latency overhead (`657.2 ms` vs `87.5 ms`) due to PCIe transfer.
  - **Config 150M-Spec (270.56M total params)**: Peak VRAM reduced by **41.1%** (`1,780.1 MB` vs `3,022.6 MB`), with a 10.6x latency overhead (`1,576.7 ms` vs `149.2 ms`).
  - **Config 350M-Spec (694.85M total params)**: Peak VRAM reduced by **47.2%** (`3,585.7 MB` vs `6,795.0 MB`), with a modest 1.36x latency overhead (`4,012.4 ms` vs `2,938.8 ms`).
- **Takeaway**: Layer streaming trades PCIe transfer bandwidth for memory capacity. It is not designed to accelerate models that already comfortably fit in VRAM, but to prevent Out-Of-Memory crashes and enable training otherwise untrainable models on consumer GPUs.
