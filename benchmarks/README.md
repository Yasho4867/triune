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
| **F** | **Full Competitive Matrix** | Universal Dense & MoE Transformers | PyTorch AdamW, BitsAndBytes 8-bit AdamW | Verified 3x2x2 pre-training and LoRA fine-tuning | [Report](MASTER_COMPETITIVE_MATRIX_REPORT.md) |
| **G** | **Genuine Full-Stack Unsloth** | FastLlamaModel, FastLanguageModel, fast_lora, fast_cross_entropy_loss | Genuine installed Unsloth (2026.10.10) vs Standard PyTorch vs Triune AI | Pre-training and LoRA fine-tuning in isolated Python processes | [Report](GENUINE_UNSLOTH_VS_TRIUNE_REPORT.md) |

---

## Detailed Empirical Findings & Honest Trade-Offs

### 1. Master Competitive Matrix (`Category F`, Verified Hardware Measurements)
Evaluated across identical architectures (Universal Dense 76M and Triune MoE 76M) with **actively participating LoRA adapters**, verified parameter weight updates, and genuine BitsAndBytes 8-bit AdamW (no silent fallback):

- **Dense LoRA Fine-Tuning**:
  - **Memory**: Triune uses only **465.1 MB peak VRAM**, representing a **40.3% reduction vs Standard AdamW (779.1 MB)** and a **28.6% reduction vs BitsAndBytes 8-bit AdamW (651.0 MB)** due to fused linear-LoRA and chunked logit-free backprop.
  - **Throughput**: Triune runs at **78.98 ± 5.71 ms/step (12,966 tok/s)** vs **69.56 ± 11.71 ms/step** for BitsAndBytes 8-bit AdamW and **62.88 ± 11.56 ms/step** for standard AdamW.
- **MoE LoRA Fine-Tuning**:
  - Triune executes at **154.17 ± 7.66 ms/step** (6,642 tok/s) and **992.0 MB peak VRAM** (11.3% lower VRAM than Standard AdamW's 1,118.8 MB, operating at near parity with BitsAndBytes's 991.1 MB and 156.59 ms/step).
- **Dense Pre-Training (Memory & Convergence Win, Throughput Trade-off)**:
  - **Memory**: Triune uses **570.1 MB peak VRAM**, saving **39.8% vs standard AdamW (947.6 MB)** and **21.9% vs BitsAndBytes 8-bit AdamW (729.9 MB)**.
  - **Throughput Trade-off**: Triune runs at **53.78 ± 4.95 ms/step** (19,040 tok/s) vs **42.43 ± 9.00 ms/step** for standard AdamW (~25% latency overhead) due to online subspace projections.
- **MoE Pre-Training (Speed & Memory Win over 8-Bit Baseline)**:
  - Triune achieves the lowest peak VRAM in this category (**1,152.8 MB**, saving 21.6% vs AdamW's 1,470.7 MB and 5.2% vs BitsAndBytes's 1,215.6 MB) and outpaces BitsAndBytes 8-bit AdamW (**159.25 ms/step vs 196.15 ms/step**, an **18.8% speedup**).

### 1.1 Genuine Full-Stack Unsloth Head-to-Head (`Category G`, Isolated Subprocesses)
Benchmarked against the actual installed **`unsloth` 2026.10.10 framework** across Pre-Training and LoRA Fine-Tuning on 76M parameters (8 layers, 384 dim, 1024 MLP, 6 heads):
- **LoRA Fine-Tuning**:
  - **Throughput Parity**: Triune runs at **36.27 ± 1.00 ms/step (28,230 tok/s)** vs Genuine Unsloth's **35.66 ± 0.63 ms/step (28,714 tok/s)** — operating within **1.7% latency parity**, while delivering a **34.3% speedup over Standard PyTorch PEFT (55.24 ± 6.14 ms/step, 18,538 tok/s)**.
  - **VRAM**: Unsloth with 8-bit AdamW uses 340.6 MB; Triune with FP32 AdamW states uses 545.1 MB (-9.0% vs Standard PyTorch's 599.1 MB).
- **Pre-Training from Scratch**:
  - Unsloth's fused Triton kernels and 8-bit optimizer achieve **22.30 ± 0.50 ms/step (45,911 tok/s, 464.8 MB)** on standard dense Llama.
  - Triune's `CentroidSteerOptimizer` achieves **34.35 ± 0.95 ms/step (29,813 tok/s, 588.2 MB)**, running competitively with Standard PyTorch (`35.14 ms/step`) while saving 11.1% VRAM and providing 3-tier Muon orthogonalization.
- **Mixture of Experts Support**:
  - Triune natively supports and trains Multi-Exit MoE models (91.69 ms/step, 11,168 tok/s), whereas Unsloth is specialized exclusively for dense autoregressive architectures.

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
