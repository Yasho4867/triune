# TriuneTransformer — Empirical Benchmark & Competitive Evaluation Suite

**Date**: October 30, 2026
**Environment**: Device: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM) | PyTorch: 2.14.1+cu132 | CUDA: 13.2

## Overview of Benchmark Categories

This repository contains publication-grade, hardware-verified benchmarks evaluating Triune's architectural
and runtime components against PyTorch standard baselines and native competitor implementations.

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

1. **Optimizer State Memory Reduction**:
   - **CentroidSteer** achieves **0.87 bytes/parameter** (14.14 MB state memory on a 17M parameter model), representing a **70.8% reduction** over PyTorch AdamW (48.37 MB) and a **42.8% reduction** over native BitsAndBytes 8-bit AdamW (24.74 MB).
2. **Long-Context Constant Memory ($O(1)$)**:
   - While Softmax KV-cache expands to **16,384 KB** per layer at $L=8,192$, Triune's GLA maintains a fixed **64 KB recurrent state**, preserving decode latency ($1.13$ ms/token).
3. **Hierarchical Dynamic Exits**:
   - **Reflex Exit (L2)** achieves a **3.20x speedup** (27.81 ms vs 88.86 ms, 73,640 tok/s) over full Cortex processing.
   - **Learned Dynamic Routing** executes at 29.55 ms (69,294 tok/s) saving **66.7% latency** while capturing over 90% of full-depth perplexity.
4. **Memory-Constrained Training via Layer Streaming**:
   - `LayerStreamingEngine` enables training models up to 350M+ parameters within an 8 GB VRAM envelope, reducing peak allocated memory by **up to 25.7%** compared to standard GPU-resident models that thrash system shared memory.
5. **LoRA Fine-Tuning Throughput**:
   - On MoE fine-tuning, Triune achieves **5,327 tokens/second** (192.23 ms/step), outperforming both native BitsAndBytes/Unsloth (5,027 tok/s) and standard AdamW (4,513 tok/s).

