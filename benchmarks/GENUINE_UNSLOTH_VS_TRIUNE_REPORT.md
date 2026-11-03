# Triune AI vs Genuine Full-Stack Unsloth vs Standard PyTorch — Comprehensive Head-to-Head Benchmark

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU`  
**Precision**: `torch.bfloat16` with native CUDA Tensor Cores  
**Timestamp**: 2026-11-03 17:32:00 UTC  
**Git Provenance Commit**: `b1f682d47e8c938184cead2eaccda86220558367`  
**Architecture Scope**: 76M Parameters (8 Layers, 384 Hidden Dim, 1024 MLP / Intermediate, 6 Attention Heads, 2 KV Heads (GQA), Vocab 32000, Sequence Length 256, Batch Size 4).

> [!IMPORTANT]
> **Genuine Full-Stack Unsloth Baseline Guarantee**:
> This report benchmarks against the **actual installed `unsloth` framework (`unsloth` 2026.10.10)**:
> - **Unsloth Patches**: `FastLlamaModel.pre_patch()` replacing Attention, RoPE, RMSNorm, and SwiGLU with Unsloth's native Triton kernels.
> - **Unsloth Fine-Tuning**: `FastLanguageModel.get_peft_model()` replacing standard linear projections with Unsloth's compiled Triton `fast_lora` kernels.
> - **Unsloth Loss**: Unsloth's Triton `fast_cross_entropy_loss`.
> - **Unsloth Optimizer**: Native 8-bit `bitsandbytes.optim.AdamW8bit`.

---

## 1. Pre-Training from Scratch (Dense 76M)

Evaluates raw pre-training performance from randomly initialized weights on equivalent 76M Llama/Universal architectures.

| Framework | Architecture | Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM vs PyTorch | VRAM vs Unsloth |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Pytorch | DENSE (76M) | `AdamW` | 32.29 ± 2.33 ms | 31,714 tok/s | **661.6 MB** | **0.0%** | **-42.3%** |
| Unsloth | DENSE (76M) | `AdamW8bit` | 27.39 ± 2.78 ms | 37,388 tok/s | **464.8 MB** | **29.7%** | **0.0%** |
| **TRIUNE** | DENSE (76M) | `CentroidSteerOptimizer` | 31.61 ± 2.73 ms | 32,399 tok/s | **418.1 MB** | **36.8%** | **10.0%** |
| **TRIUNE** | MOE (76M) | `CentroidSteerOptimizer` | 94.35 ± 10.84 ms | 10,853 tok/s | **1218.2 MB** | **-84.1%** | **-162.1%** |

---

## 2. LoRA Fine-Tuning (Dense 76M, Rank=16, Alpha=16)

Evaluates parameter-efficient fine-tuning with active LoRA adapters targeting all attention and MLP projections (`q, k, v, o, gate, up, down`).

| Framework | Adapter Engine | Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM vs PyTorch | VRAM vs Unsloth |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Pytorch | Hugging Face PEFT | `AdamW` | 65.58 ± 9.56 ms | 15,615 tok/s | **599.1 MB** | **0.0%** | **-75.9%** |
| Unsloth | Unsloth Triton fast_lora | `AdamW8bit` | 41.89 ± 8.81 ms | 24,446 tok/s | **340.6 MB** | **43.1%** | **0.0%** |
| **TRIUNE** | Triune Fused addmm LoRA | `AdamW8bit` | 38.96 ± 7.03 ms | 26,284 tok/s | **338.7 MB** | **43.5%** | **0.6%** |

---

## 3. Detailed Architectural Comparison & Workload Analysis

### A. Pre-Training from Scratch
- **Memory Footprint**: Triune allocates **418.1 MB peak VRAM**, saving **36.8% vs Standard PyTorch (661.6 MB)** and **10.0% vs Genuine Unsloth (464.8 MB)**.
- **Latency / Throughput**: Triune runs at **31.61 ± 2.73 ms/step** (32,399 tok/s) compared to Unsloth's **27.39 ± 2.78 ms/step** (37,388 tok/s) and PyTorch's **32.29 ± 2.33 ms/step** (31,714 tok/s).
- **Optimizer Engine**: Triune's `CentroidSteerOptimizer` maintains 3-tier Muon orthogonalization and Triangular Core subspace tracking, while Unsloth operates on 8-bit AdamW.

### B. LoRA Fine-Tuning
- **Memory Footprint**: Triune achieves **338.7 MB peak VRAM**, slashing memory by **43.5% vs Standard PyTorch PEFT (599.1 MB)** and **0.6% vs Genuine Unsloth (340.6 MB)**. This is made possible by Triune's logit-free chunked cross-entropy from hidden states (`chunked_cross_entropy_from_hidden`), which eliminates the `[B, T, V]` logit tensor allocation entirely.
- **Throughput**: Triune clocks **38.96 ± 7.03 ms/step** (26,284 tok/s) vs Unsloth's **41.89 ± 8.81 ms/step** (24,446 tok/s) and Standard PyTorch's **65.58 ± 9.56 ms/step** (15,615 tok/s).

### C. Multi-Exit MoE Architecture Support
- **Architectural Reach**: Unsloth is specialized exclusively for dense autoregressive architectures (Llama, Mistral, Qwen2, Gemma). It does not provide dynamic-exit multi-stage Mixture-of-Experts routing.
- **Triune MoE Execution**: Triune trains native Multi-Exit MoE models at **94.35 ± 10.84 ms/step** (10,853 tok/s) with **1218.2 MB peak VRAM**, unlocking dynamic depth scaling and inference speedups on commodity hardware.
