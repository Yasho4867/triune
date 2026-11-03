# Triune AI — Empirical Benchmark & Competitive Evaluation Suite

**Date**: November 03, 2026
**Environment**: Device: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM) | PyTorch: 2.14.1+cu132 | CUDA: 13.2

## Overview of Benchmark Categories

This repository contains publication-grade, hardware-verified benchmarks evaluating **Triune AI**'s architectural
and runtime components against PyTorch standard baselines and native competitor implementations (Unsloth / BitsAndBytes / Liger).

### Benchmark Suite Matrix

| Category | Topic | Target Components | Competitors Evaluated | Report Link |
| :--- | :--- | :--- | :--- | :--- |
| **A** | **Kernel Microbenchmarks** | Chunked CE, FastRMSNorm, FastGLA | PyTorch CrossEntropy, PyTorch SDPA, nn.RMSNorm | [Report](COMPETITIVE_BENCHMARK_REPORT.md) |
| **B** | **Optimizer Dynamics** | CentroidSteer, Muon | PyTorch AdamW, BitsAndBytes 8-bit AdamW (Unsloth) | [Report](OPTIMIZER_BENCHMARK_REPORT.md) |
| **C** | **Layer Streaming** | LayerStreamingEngine (CPU/GPU) | Standard GPU-Resident Training (OOM prevention) | [Report](LAYER_STREAMING_REPORT.md) |
| **D** | **Long-Context Generation** | Gated Linear Attention (GLA) | Softmax Attention (SDPA) KV-Cache ($O(1)$ vs $O(L)$) | [Report](LONG_CONTEXT_REPORT.md) |
| **E** | **Dynamic Exits & Pareto** | Reflex (L2), Limbic (L5), Cortex (L8) | Full-depth monolithic execution | [Report](DYNAMIC_EXITS_REPORT.md) |
| **F** | **Full Competitive Matrix** | Universal Dense & MoE Transformers | PyTorch AdamW, Unsloth native 8-bit AdamW | [Report](MASTER_COMPETITIVE_MATRIX_REPORT.md) |

## Key Competitive Takeaways

1. **Optimizer State Memory, Speed & Convergence (`Category B`)**:
   - **CentroidSteer** achieves **1.00 byte/parameter** (`16.23 MB` state memory on a 17.02M parameter model), representing a **66.4% reduction** over PyTorch AdamW (`48.37 MB`) and a **34.4% reduction** over native BitsAndBytes 8-bit AdamW (`24.74 MB`).
   - Pure optimizer step latency is **4.76 ms/step (210.2 steps/s)** — **faster than native BitsAndBytes 8-bit AdamW** (`5.76 ms/step`, `173.6 steps/s`) — while achieving **-50.82% 50-step loss reduction** vs **-35.10%** for AdamW and 8-bit AdamW.
2. **Chunked Logit-Free Cross-Entropy (`Category A`)**:
   - At $N=8,192$ ($V=32,000$), Triune's BF16 Tensor Core chunked Cross-Entropy is **1.76x faster** than standard PyTorch (`64.67 ms` vs `113.73 ms`) while cutting peak VRAM by **91.3%** (`365.8 MB` vs `4,182.3 MB`).
3. **Long-Context Constant Memory & Prefill (`Category D`)**:
   - At $L=16,384$, Triune's GLA prefill reaches **1,199,876 tok/s (`13.65 ms`)**, outperforming FlashAttention/SDPA (`14.18 ms`, `1,155,036 tok/s`).
   - During autoregressive decoding, while Softmax KV-cache expands to **24,576 KB** per layer at $L=16,384$, Triune's GLA maintains a fixed **48–64 KB recurrent state** (**99.8% cache memory saved**) and decodes up to **7.94x–9.77x faster** (`0.085 ms/tok` vs `0.831 ms/tok`).
4. **Hierarchical Dynamic Exits (`Category E`)**:
   - **Reflex Exit (L2)** achieves a **3.67x forward speedup** (`7.29 ms` vs `26.76 ms`, `280,941 tok/s`) and **1.92x decode speedup** (`205.5 tok/s` vs `107.3 tok/s`) over full Cortex processing.
   - **Learned Dynamic Routing** executes at **6.84 ms (`299,506 tok/s`)**, saving **74.4% latency**.
5. **Memory-Constrained Training via Layer Streaming (`Category C`)**:
   - `LayerStreamingEngine` reduces peak GPU VRAM by **34.7% (76M)**, **42.3% (150M)**, and **47.9% (350M / 695M total params: `3,661 MB` vs `7,023 MB`)**, enabling large MoE training inside an 8 GB VRAM envelope.
6. **End-to-End 3x2x2 Competitive Matrix (`Category F`)**:
   - **Dense Pre-Training**: Triune uses **41.7% less VRAM** than AdamW (`528.7 MB` vs `907.2 MB`) and **23.6% less VRAM** than Unsloth (`692.4 MB`) with superior 15-step loss reduction (`6.81` vs `5.86`).
   - **MoE Pre-Training**: Triune uses **22.8% less VRAM** than AdamW (`1,089.5 MB` vs `1,411.4 MB`), beats Unsloth in VRAM (`1,157.1 MB`), and achieves the highest loss reduction (`4.10` vs `3.52`).
   - **LoRA Fine-Tuning**: Triune beats Unsloth in both speed and VRAM on Dense LoRA (`22.49 ms` / `45,525 tok/s` at `384.3 MB` vs Unsloth's `25.17 ms` / `40,691 tok/s` at `570.3 MB`) and on MoE LoRA (`91.37 ms` vs `94.01 ms`).

