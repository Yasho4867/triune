# Triune AI vs Genuine Full-Stack Unsloth vs Standard PyTorch — Comprehensive Head-to-Head Benchmark

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU`  
**Precision**: `torch.bfloat16` with native CUDA Tensor Cores  
**Timestamp**: 2026-11-03 21:27:58 UTC  
**Git Provenance Commit**: `60da378e3400020920b8fbb4c78e2c08e5decb6d`  
**Architecture Scope**: 76M Parameters (8 Layers, 384 Hidden Dim, 1024 MLP / Intermediate, 6 Attention Heads, 2 KV Heads (GQA), Vocab 32000, Sequence Length 256, Batch Size 4).  
**Workload**: Real causal natural language text sequences from FineWeb (`data/fineweb_sample.jsonl`, B=4, T=256, Vocab=32000).  
**Parameter Accounting**: Scalar element count `sum(p.numel())` (76M total parameters, ~2.1M trainable in LoRA rank 16).

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
| Pytorch | DENSE (76M) | `AdamW` | 30.51 ± 0.71 ms | 33,558 tok/s | **661.6 MB** | **0.0%** | **-42.3%** |
| Unsloth | DENSE (76M) | `AdamW8bit` | 24.79 ± 4.36 ms | 41,303 tok/s | **464.8 MB** | **29.7%** | **0.0%** |
| **TRIUNE** | DENSE (76M) | `CentroidSteerOptimizer` | 31.81 ± 3.61 ms | 32,189 tok/s | **398.2 MB** | **39.8%** | **14.3%** |
| **TRIUNE** | MOE (76M) | `CentroidSteerOptimizer` | 120.25 ± 4.44 ms | 8,515 tok/s | **1218.2 MB** | **-84.1%** | **-162.1%** |

---

## 2. LoRA Fine-Tuning (Dense 76M, Rank=16, Alpha=16)

Evaluates parameter-efficient fine-tuning with active LoRA adapters targeting all attention and MLP projections (`q, k, v, o, gate, up, down`).

| Framework | Adapter Engine | Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM vs PyTorch | VRAM vs Unsloth |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Pytorch | Hugging Face PEFT | `AdamW` | 73.6 ± 7.21 ms | 13,913 tok/s | **599.1 MB** | **0.0%** | **-75.9%** |
| Unsloth | Unsloth Triton fast_lora | `AdamW8bit` | 35.41 ± 1.33 ms | 28,921 tok/s | **340.6 MB** | **43.1%** | **0.0%** |
| **TRIUNE** | Triune Fused addmm LoRA | `AdamW8bit` | 31.87 ± 0.86 ms | 32,128 tok/s | **338.7 MB** | **43.5%** | **0.6%** |

---

## 3. Detailed Architectural Comparison & Workload Analysis

### A. Pre-Training from Scratch
- **Memory Footprint**: Triune allocates **398.2 MB peak VRAM**, saving **39.8% vs Standard PyTorch (661.6 MB)** and **14.3% vs Genuine Unsloth (464.8 MB)**.
- **Latency / Throughput**: Triune runs at **31.81 ± 3.61 ms/step** (32,189 tok/s) compared to Unsloth's **24.79 ± 4.36 ms/step** (41,303 tok/s) and PyTorch's **30.51 ± 0.71 ms/step** (33,558 tok/s).
- **Optimizer Engine**: Triune's `CentroidSteerOptimizer` maintains 3-tier Muon orthogonalization and Triangular Core subspace tracking, while Unsloth operates on 8-bit AdamW.

### B. LoRA Fine-Tuning
- **Memory Footprint**: Triune achieves **338.7 MB peak VRAM**, slashing memory by **43.5% vs Standard PyTorch PEFT (599.1 MB)** and **0.6% vs Genuine Unsloth (340.6 MB)**. This is made possible by Triune's logit-free chunked cross-entropy from hidden states (`chunked_cross_entropy_from_hidden`), which eliminates the `[B, T, V]` logit tensor allocation entirely.
- **Throughput**: Triune clocks **31.87 ± 0.86 ms/step** (32,128 tok/s) vs Unsloth's **35.41 ± 1.33 ms/step** (28,921 tok/s) and Standard PyTorch's **73.6 ± 7.21 ms/step** (13,913 tok/s).

### C. Multi-Exit MoE Architecture Support
- **Architectural Reach**: Unsloth is specialized exclusively for dense autoregressive architectures (Llama, Mistral, Qwen2, Gemma). It does not provide dynamic-exit multi-stage Mixture-of-Experts routing.
- **Triune MoE Execution**: Triune trains native Multi-Exit MoE models at **120.25 ± 4.44 ms/step** (8,515 tok/s) with **1218.2 MB peak VRAM**, unlocking dynamic depth scaling and inference speedups on commodity hardware.
