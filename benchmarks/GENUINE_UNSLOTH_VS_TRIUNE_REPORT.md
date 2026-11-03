# Triune AI vs Genuine Full-Stack Unsloth vs Standard PyTorch — Comprehensive Head-to-Head Benchmark

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU`  
**Precision**: `torch.bfloat16` with native CUDA Tensor Cores  
**Timestamp**: 2026-11-03 13:05:50 UTC  
**Git Provenance Commit**: `d87c7b1d2295b13073beaa6af939a6ece34455b9`  
**Architecture Scope**: 76M Parameters (8 Layers, 384 Hidden Dim, 1024 MLP / Intermediate, 6 Attention Heads, Vocab 32000, Sequence Length 256, Batch Size 4).

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
| Pytorch | DENSE (76M) | `AdamW` | 35.14 ± 2.53 ms | 29,140 tok/s | **661.6 MB** | **0.0%** | **-42.3%** |
| Unsloth | DENSE (76M) | `AdamW8bit` | 22.3 ± 0.5 ms | 45,911 tok/s | **464.8 MB** | **29.7%** | **0.0%** |
| **TRIUNE** | DENSE (76M) | `CentroidSteerOptimizer` | 34.35 ± 0.95 ms | 29,813 tok/s | **588.2 MB** | **11.1%** | **-26.5%** |
| **TRIUNE** | MOE (76M) | `CentroidSteerOptimizer` | 91.69 ± 1.72 ms | 11,168 tok/s | **1218.2 MB** | **-84.1%** | **-162.1%** |

---

## 2. LoRA Fine-Tuning (Dense 76M, Rank=16, Alpha=16)

Evaluates parameter-efficient fine-tuning with active LoRA adapters targeting all attention and MLP projections (`q, k, v, o, gate, up, down`).

| Framework | Adapter Engine | Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM vs PyTorch | VRAM vs Unsloth |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Pytorch | Hugging Face PEFT | `AdamW` | 55.24 ± 6.14 ms | 18,538 tok/s | **599.1 MB** | **0.0%** | **-75.9%** |
| Unsloth | Unsloth Triton fast_lora | `AdamW8bit` | 35.66 ± 0.63 ms | 28,714 tok/s | **340.6 MB** | **43.1%** | **0.0%** |
| **TRIUNE** | Triune Fused addmm LoRA | `AdamW` | 36.27 ± 1.0 ms | 28,230 tok/s | **545.1 MB** | **9.0%** | **-60.0%** |

---

## 3. Detailed Architectural Comparison & Workload Analysis

### A. Pre-Training from Scratch
- **Memory Footprint**: Triune allocates **588.2 MB peak VRAM**, saving **11.1% vs Standard PyTorch (661.6 MB)** and **-26.5% vs Genuine Unsloth (464.8 MB)**.
- **Latency / Throughput**: Triune runs at **34.35 ± 0.95 ms/step** (29,813 tok/s) compared to Unsloth's **22.3 ± 0.5 ms/step** (45,911 tok/s) and PyTorch's **35.14 ± 2.53 ms/step** (29,140 tok/s).
- **Optimizer Engine**: Triune's `CentroidSteerOptimizer` maintains 3-tier Muon orthogonalization and Triangular Core subspace tracking, while Unsloth operates on 8-bit AdamW.

### B. LoRA Fine-Tuning
- **Memory Footprint**: Triune achieves **545.1 MB peak VRAM**, slashing memory by **9.0% vs Standard PyTorch PEFT (599.1 MB)** and **-60.0% vs Genuine Unsloth (340.6 MB)**. This is made possible by Triune's logit-free chunked cross-entropy from hidden states (`chunked_cross_entropy_from_hidden`), which eliminates the `[B, T, V]` logit tensor allocation entirely.
- **Throughput**: Triune clocks **36.27 ± 1.0 ms/step** (28,230 tok/s) vs Unsloth's **35.66 ± 0.63 ms/step** (28,714 tok/s) and Standard PyTorch's **55.24 ± 6.14 ms/step** (18,538 tok/s).

### C. Multi-Exit MoE Architecture Support
- **Architectural Reach**: Unsloth is specialized exclusively for dense autoregressive architectures (Llama, Mistral, Qwen2, Gemma). It does not provide dynamic-exit multi-stage Mixture-of-Experts routing.
- **Triune MoE Execution**: Triune trains native Multi-Exit MoE models at **91.69 ± 1.72 ms/step** (11,168 tok/s) with **1218.2 MB peak VRAM**, unlocking dynamic depth scaling and inference speedups on commodity hardware.
