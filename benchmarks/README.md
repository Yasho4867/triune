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
   - With native BF16 Tensor Core intra-chunk GEMM and logarithmic Hillis-Steele parallel prefix scan ($O(\log_2 N_c)$), Triune's GLA prefill **beats FlashAttention-2 / cuDNN SDPA starting at $L=8,192$** (`2.95 ms` / `2,775,190 tok/s` vs SDPA `4.14 ms` / `1,977,746 tok/s`, **1.40x faster**) and reaches **2,259,862 tok/s (`7.25 ms`) at $L=16,384$** vs SDPA `14.20 ms` (**1.96x faster**), while reducing GLA prefill activation VRAM by **75.7%** (`411.3 MB` vs `1,692.1 MB` previously).
   - During autoregressive decoding at $L=8,192$, while Softmax KV-cache expands to **16,384 KB** per layer (`0.47 ms/tok`), Triune's GLA maintains a fixed **64 KB recurrent state** (**256x smaller state**) and decodes faster (`0.42 ms/tok`).
4. **Hierarchical Dynamic Exits (`Category E`)**:
   - **Reflex Exit (L2)** achieves a **3.67x forward speedup** (`7.29 ms` vs `26.76 ms`, `280,941 tok/s`) and **1.92x decode speedup** (`205.5 tok/s` vs `107.3 tok/s`) over full Cortex processing.
   - **Learned Dynamic Routing** executes at **6.84 ms (`299,506 tok/s`)**, saving **74.4% latency**.
5. **Memory-Constrained Training via Layer Streaming (`Category C`)**:
   - With GPU-side gradient norm accumulation and double-buffered forward/backward CUDA prefetching, `LayerStreamingEngine` reduces peak GPU VRAM by **33.3% (`76M`: `877.8 MB` vs `1,315.9 MB`, `+52.9%` faster streaming throughput at `1,558.2 tok/s`)**, **41.1% (`150M`: `1,780.1 MB` vs `3,022.6 MB`)**, and **47.2% (`350M`: `3,585.7 MB` vs `6,795.0 MB`)**, enabling large MoE training inside an 8 GB VRAM envelope.
6. **End-to-End 3x2x2 Competitive Matrix (`Category F`)**:
   - **Dense Pre-Training**: Triune (`28,119 tok/s`, `+42.9%` faster than pre-fusion) uses **28.3% less VRAM** than AdamW (`632.5 MB` vs `882.6 MB`) and **20.3% less VRAM** than Unsloth (`793.6 MB`) while achieving **+16.3% stronger 15-step loss reduction** (`6.8133` vs `5.8567` AdamW and `5.8603` Unsloth).
   - **MoE Pre-Training**: Triune (`96.26 ms/step`, `10,637 tok/s`) **beats Unsloth in throughput** (`97.57 ms/step`, `10,495 tok/s`), uses **13.8% less VRAM** than AdamW (`1,163.2 MB` vs `1,350.0 MB`), beats Unsloth in VRAM (`1,223.6 MB`), and achieves **+16.5% stronger loss reduction** (`4.1028` vs `3.5202`).
   - **Dense LoRA Fine-Tuning**: With fused `addmm_` LoRA projections (`fast_lora_linear`) and zero-allocation chunked Cross-Entropy, Triune (`31.73 ms/step`, `32,272 tok/s`) **beats both AdamW (`32.82 ms/step`, `31,197 tok/s`) and Unsloth (`39.89 ms/step`, `25,672 tok/s`) in throughput** while cutting Peak VRAM by **25.2%** (`553.4 MB` vs `740.3 MB` AdamW / `738.3 MB` Unsloth).
