# Triune AI vs Genuine Full-Stack Unsloth vs Standard PyTorch — Comprehensive Head-to-Head Benchmark

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU`  
**Precision**: `torch.bfloat16` with native CUDA Tensor Cores  
**Timestamp**: 2026-11-03 23:58:18 UTC  
**Git Provenance Commit**: `3fd3612ef803a0dc8f1376bac7856bf3c720136d` (Working-Tree Clean: `False`)  
**Script SHA-256**: `ad2803856269f4ff63b8cc3e1db682b4044558132e8d49f7f7d56004bf0fa1d0`  
**Dataset**: `data\fineweb_sample.jsonl` (Mode: `fineweb_real`, SHA-256: `8acbf30894156292e921fe4c6e11f634d0042b42f4150968c4d3023863a777c4`)  
**Tokenizer**: `triune_tokenizer.json` (SHA-256: `df2c43cfab506acce484e8c45a12e497a7901622fa60058fd4cd5802f4e36451`)  

### Verified Parameter Accounting & Model Presets
- **Dense Architecture**: **37,165,440 scalar parameters** (37.17M) across 8 layers, 384 hidden dim, 1024 MLP / intermediate dim, 6 attention heads, 2 KV heads (GQA), vocab 32,000, sequence length 256, batch size 4.
- **Dense LoRA Fine-Tuning**: **38,033,792 total parameters** (38.03M), with **868,352 trainable adapter parameters** (868.4K, **2.28%** trainable) targeting all attention and MLP projections (`q, k, v, o, gate, up, down`).
- **MoE Architecture**: **90,789,271 scalar parameters** (90.79M) across 8 layers, 384 dim, 4 routed experts + 1 shared expert.

> [!IMPORTANT]
> **Genuine Full-Stack Unsloth Baseline Guarantee**:
> This report benchmarks against the **actual installed `unsloth` framework (`unsloth` 2026.10.10)**:
> - **Unsloth Patches**: `FastLlamaModel.pre_patch()` replacing Attention, RoPE, RMSNorm, and SwiGLU with Unsloth's native Triton kernels.
> - **Unsloth Fine-Tuning**: `FastLanguageModel.get_peft_model()` replacing standard linear projections with Unsloth's compiled Triton `fast_lora` kernels.
> - **Unsloth Loss**: Unsloth's Triton `fast_cross_entropy_loss`.
> - **Unsloth Optimizer**: Native 8-bit `bitsandbytes.optim.AdamW8bit`.

---

## 1. Pre-Training from Scratch (Dense vs MoE)

Evaluates raw pre-training performance from randomly initialized weights on equivalent architectures.

| Framework | Architecture | Total Parameters | Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM vs PyTorch | VRAM vs Unsloth |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Pytorch | DENSE | 37,165,440 (37.17M) | `AdamW` | 34.58 ± 8.93 ms | 29,616 tok/s | **661.6 MB** | **0.0%** | **-42.3%** |
| Unsloth | DENSE | 37,165,440 (37.17M) | `AdamW8bit` | 25.92 ± 3.32 ms | 39,503 tok/s | **464.8 MB** | **29.7%** | **0.0%** |
| **TRIUNE** | DENSE | 37,165,440 (37.17M) | `CentroidSteerOptimizer` | 31.65 ± 1.91 ms | 32,349 tok/s | **398.2 MB** | **39.8%** | **14.3%** |
| **TRIUNE** | MOE | 90,789,271 (90.79M) | `CentroidSteerOptimizer` | 101.14 ± 12.93 ms | 10,125 tok/s | **1218.2 MB** | **-84.1%** | **-162.1%** |

---

## 2. LoRA Fine-Tuning (Rank=16, Alpha=16)

Evaluates parameter-efficient fine-tuning with active LoRA adapters targeting all attention and MLP projections (`q, k, v, o, gate, up, down`).

| Framework | Adapter Engine | Total Params | Trainable Params (% Trainable) | Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM vs PyTorch | VRAM vs Unsloth |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Pytorch | Hugging Face PEFT | 38,033,792 | 868,352 (2.28%) | `AdamW` | 56.94 ± 12.65 ms | 17,985 tok/s | **599.1 MB** | **0.0%** | **-75.9%** |
| Unsloth | Unsloth Triton fast_lora | 38,033,792 | 868,352 (2.28%) | `AdamW8bit` | 47.75 ± 8.53 ms | 21,445 tok/s | **340.6 MB** | **43.1%** | **0.0%** |
| **TRIUNE** | Triune Fused addmm LoRA | 38,033,792 | 868,352 (2.28%) | `AdamW8bit` | 38.32 ± 5.02 ms | 26,720 tok/s | **338.7 MB** | **43.5%** | **0.6%** |

---

## 3. Detailed Architectural Comparison & Workload Analysis

### A. Pre-Training from Scratch
- **Memory Footprint**: Triune allocates **398.2 MB peak VRAM**, saving **39.8% vs Standard PyTorch (661.6 MB)** and **14.3% vs Genuine Unsloth (464.8 MB)**.
- **Latency / Throughput**: Triune runs at **31.65 ± 1.91 ms/step** (32,349 tok/s) compared to Unsloth's **25.92 ± 3.32 ms/step** (39,503 tok/s) and PyTorch's **34.58 ± 8.93 ms/step** (29,616 tok/s).
- **Optimizer Engine**: Triune's `CentroidSteerOptimizer` maintains 3-tier Muon orthogonalization and Triangular Core subspace tracking, while Unsloth operates on 8-bit AdamW.

### B. LoRA Fine-Tuning
- **Memory Footprint**: Triune achieves **338.7 MB peak VRAM**, slashing memory by **43.5% vs Standard PyTorch PEFT (599.1 MB)** and **0.6% vs Genuine Unsloth (340.6 MB)**. This is made possible by Triune's logit-free chunked cross-entropy from hidden states (`chunked_cross_entropy_from_hidden`), which eliminates the `[B, T, V]` logit tensor allocation entirely.
- **Throughput**: Triune clocks **38.32 ± 5.02 ms/step** (26,720 tok/s) vs Unsloth's **47.75 ± 8.53 ms/step** (21,445 tok/s) and Standard PyTorch's **56.94 ± 12.65 ms/step** (17,985 tok/s).

### C. Multi-Exit MoE Architecture Support
- **Architectural Reach**: Unsloth is specialized exclusively for dense autoregressive architectures (Llama, Mistral, Qwen2, Gemma). It does not provide dynamic-exit multi-stage Mixture-of-Experts routing.
- **Triune MoE Execution**: Triune trains native Multi-Exit MoE models at **101.14 ± 12.93 ms/step** (10,125 tok/s) with **1218.2 MB peak VRAM**, unlocking dynamic depth scaling and inference speedups on commodity hardware.
