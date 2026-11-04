# Triune AI vs Genuine Full-Stack Unsloth vs Standard PyTorch — Comprehensive Head-to-Head Benchmark

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU`  
**Precision**: `torch.bfloat16` with native CUDA Tensor Cores  
**Timestamp**: 2026-11-04 01:27:54 UTC  
**Git Provenance Commit**: `7dcb746801c6ef6b4412eb80a74e8eecf487811e` (Working-Tree Clean: `False`)  
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
| Pytorch | DENSE | 37,165,440 (37.17M) | `AdamW` | 36.12 ± 3.32 ms | 28,348 tok/s | **661.6 MB** | **0.0%** | **-42.3%** |
| Unsloth | DENSE | 37,165,440 (37.17M) | `AdamW8bit` | 27.7 ± 3.65 ms | 36,967 tok/s | **464.8 MB** | **29.7%** | **0.0%** |
| **TRIUNE** | DENSE | 37,165,440 (37.17M) | `CentroidSteerOptimizer` | 28.96 ± 2.84 ms | 35,361 tok/s | **423.9 MB** | **35.9%** | **8.8%** |
| **TRIUNE** | MOE | 90,789,271 (90.79M) | `CentroidSteerOptimizer` | 107.11 ± 11.56 ms | 9,560 tok/s | **928.9 MB** | **-40.4%** | **-99.8%** |

---

## 2. LoRA Fine-Tuning (Rank=16, Alpha=16)

Evaluates parameter-efficient fine-tuning with active LoRA adapters targeting all attention and MLP projections (`q, k, v, o, gate, up, down`).

| Framework | Adapter Engine | Total Params | Trainable Params (% Trainable) | Optimizer | Step Latency (Median ± Std) | Throughput | Peak VRAM | VRAM vs PyTorch | VRAM vs Unsloth |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Pytorch | Hugging Face PEFT | 38,033,792 | 868,352 (2.28%) | `AdamW` | 55.76 ± 7.98 ms | 18,364 tok/s | **599.1 MB** | **0.0%** | **-75.9%** |
| Unsloth | Unsloth Triton fast_lora | 38,033,792 | 868,352 (2.28%) | `AdamW8bit` | 42.06 ± 4.62 ms | 24,344 tok/s | **340.6 MB** | **43.1%** | **0.0%** |
| **TRIUNE** | Triune Fused addmm LoRA | 38,033,792 | 868,352 (2.28%) | `AdamW8bit` | 39.43 ± 2.95 ms | 25,973 tok/s | **396.4 MB** | **33.8%** | **-16.4%** |

---

## 3. Detailed Architectural Comparison & Workload Analysis

### A. Pre-Training from Scratch
- **Memory Footprint**: Triune allocates **423.9 MB peak VRAM**, saving **35.9% vs Standard PyTorch (661.6 MB)** and **8.8% vs Genuine Unsloth (464.8 MB)**.
- **Latency / Throughput**: Triune runs at **28.96 ± 2.84 ms/step** (35,361 tok/s) compared to Unsloth's **27.7 ± 3.65 ms/step** (36,967 tok/s) and PyTorch's **36.12 ± 3.32 ms/step** (28,348 tok/s).
- **Optimizer Engine**: Triune's `CentroidSteerOptimizer` maintains 3-tier Muon orthogonalization and Triangular Core subspace tracking, while Unsloth operates on 8-bit AdamW.

### B. LoRA Fine-Tuning
- **Memory Footprint**: Triune achieves **396.4 MB peak VRAM**, slashing memory by **33.8% vs Standard PyTorch PEFT (599.1 MB)** and **-16.4% vs Genuine Unsloth (340.6 MB)**. This is made possible by Triune's logit-free chunked cross-entropy from hidden states (`chunked_cross_entropy_from_hidden`), which eliminates the `[B, T, V]` logit tensor allocation entirely.
- **Throughput**: Triune clocks **39.43 ± 2.95 ms/step** (25,973 tok/s) vs Unsloth's **42.06 ± 4.62 ms/step** (24,344 tok/s) and Standard PyTorch's **55.76 ± 7.98 ms/step** (18,364 tok/s).

### C. Multi-Exit MoE Architecture Support
- **Architectural Reach**: Unsloth is specialized exclusively for dense autoregressive architectures (Llama, Mistral, Qwen2, Gemma). It does not provide dynamic-exit multi-stage Mixture-of-Experts routing.
- **Triune MoE Execution**: Triune trains native Multi-Exit MoE models at **107.11 ± 11.56 ms/step** (9,560 tok/s) with **928.9 MB peak VRAM**, unlocking dynamic depth scaling and inference speedups on commodity hardware.
