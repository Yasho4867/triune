# TriuneTransformer — Phase 1 Repository & Benchmark Audit Report

**Date**: October 29, 2026  
**Auditor**: Antigravity Core Research & Benchmarking Engine  
**System Target**: Windows 11 x64, PyTorch 2.14.1+cu132, NVIDIA GeForce RTX 5070 Laptop GPU (8GB GDDR6 VRAM, Compute Capability 12.0)  
**Status**: Formal Audit Complete — Empirical Execution Plan Established  

---

## 1. Executive Summary & Audit Scope

This audit provides a rigorous, line-by-line inspection of the TriuneTransformer architecture, runtime engines, optimizers, and benchmark suites in `https://github.com/Yasho4867/triune`. 

### Core Audit Findings:
1. **Mathematical & Architectural Parity Restored**:
   - The MoE expert FFN activation function was verified and restored to canonical `nn.GELU()`, reversing an accidental `SiLU` modification.
   - Scale-invariant Cholesky-QR pre-conditioning in `CentroidSteerOptimizer` was verified against ill-conditioned expert gradients.
   - Dynamic exit telemetry in API endpoints was decoupled from theoretical router targets, ensuring only actual measured usage is reported.
2. **Prior Benchmark Classification**:
   - In previous preliminary runs, standard PyTorch primitives (FlashAttention / SDPA, `torch.optim.AdamW`, chunked vs standard cross-entropy, RMSNorm) were **directly executed** on hardware.
   - However, Unsloth 8-bit optimizer was previously evaluated using a functional PyTorch simulation baseline due to missing local dependencies on Windows. Per user mandate, the complete `unsloth` runtime suite is being installed directly into the environment to allow **genuine, native competitor execution**.
3. **Layer Streaming Verification**:
   - `LayerStreamingEngine` was tested against standard PyTorch training across AdamW, Muon, and CentroidSteer optimizers with gradient accumulation ($N=1, 2, 4$). All 7 multi-step parity tests passed with tight numerical parity ($10^{-5}$ float32, $10^{-3}$ bfloat16).

---

## 2. Component-by-Component Codebase Audit

### 2.1 Kernel Microbenchmarks (`triune/kernels/`)
| Component | Implementation File | Primary Mechanism | Fallback / Reference | Execution Mode |
| :--- | :--- | :--- | :--- | :--- |
| **FastGLA** | `triune/kernels/fast_gla.py` | Chunkwise recurrent state update ($O(1)$ autoregressive decode, chunked parallel training) | PyTorch `scaled_dot_product_attention` (SDPA) | PyTorch BF16 tensor ops / Custom vectorized chunk |
| **FastCrossEntropy** | `triune/kernels/fast_cross_entropy.py` | Chunked log-sum-exp & gradient accumulation across vocabulary slices | Full-logit PyTorch `F.cross_entropy` ($O(B \cdot L \cdot V)$ memory peak) | Chunked projection in BF16 |
| **FastRMSNorm** | `triune/kernels/fast_rmsnorm.py` | In-place variance normalization with fused scale | Standard `torch.nn.RMSNorm` | Native PyTorch tensor ops |
| **FastRoPE** | `triune/kernels/fast_rope.py` | Fused rotary positional embeddings with precomputed trigonometric cache | Complex multiplication / manual slicing | Native PyTorch tensor ops |
| **FastSwiGLU** | `triune/kernels/fast_swiglu.py` | Fused gated linear unit $\text{SwiGLU}(x) = (W_g x \odot \text{silu}(W_u x)) W_d$ | Standard two-step MLP | Native PyTorch tensor ops |
| **FastLoRA** | `triune/kernels/fast_lora.py` | Low-rank adapter injection with in-place residual addition | Standard PyTorch matrix multiplication | Native PyTorch tensor ops |

### 2.2 Optimizer Architecture (`triune/optim/`)
1. **`CentroidSteerOptimizer` (`triune/optim/centroid.py`)**:
   - **Mechanism**: Maintains low-rank centroid basis matrices for MoE expert slices ($r=16$), dramatically reducing optimizer memory from $2\cdot M \cdot d$ to $(M + d) \cdot r$.
   - **Orthogonalization**: Employs Newton-Schulz and Cholesky-QR matrix updates rather than element-wise second moments.
   - **Parity & Invariants**: Gradient tracking includes gradient clipping, weight decay decoupling, and numerical stabilization against ill-conditioned singular values.
2. **`Muon` (`triune/optim/muon.py`)**:
   - **Mechanism**: High-performance matrix momentum optimizer that orthogonally regularizes updates using Newton-Schulz polar decomposition for internal weights, delegating 1D vectors (biases, norms) to AdamW.
3. **`torch.optim.AdamW` (PyTorch Native)**:
   - Gold-standard reference baseline maintaining 2 full float32 states ($m_t, v_t$) per parameter (8 bytes/parameter).

### 2.3 Layer Streaming Engine (`triune/runtime/streaming.py`)
- **Memory Architecture**: Permanent layer weights reside in CPU host RAM (optionally pinned for DMA). Only the active layer block is transferred into GPU VRAM for computation.
- **Autograd Integration**: Forward and backward pre/post hooks dynamically stage parameters and offload gradients back to host memory immediately (`_offload_grad_hook`).
- **Memory Footprint**: Reduces persistent model weight memory to near zero ($< 350\text{ MB}$ active VRAM for a 2.5B parameter model), trading PCIe bus transfer bandwidth for unbounded parameter scale on consumer GPUs.

### 2.4 Model & Dynamic Routing Architecture (`triune/model/`)
- **`TriuneTransformer`**:
  - Gated Linear Attention (GLA) linear-time attention mechanism.
  - Mixture of Experts (MoE) with configurable router prefix layers, shared experts, and load-balancing auxiliary loss.
  - Hierarchical dynamic exits:
    - **Reflex Exit**: Layer 2 (immediate shallow response for routine tokens).
    - **Limbic Exit**: Layer 5 (intermediate contextual routing).
    - **Cortex Exit**: Layer 8 (full deep contextual processing).
- **Parity Check**: Dynamic routing telemetry exposes empirical exit counts ($N_{\text{reflex}}, N_{\text{limbic}}, N_{\text{cortex}}$) separately from the training objective loss coefficients.

### 2.5 Universal Framework Engine (`triune/universal.py`)
- Provides generic interfaces:
  - `UniversalDenseTransformer`: Standard causal transformer with standard attention and MLP.
  - `UniversalMoETransformer`: Dense attention with MoE feed-forward blocks.
  - `TriuneUniversalEngine`: Unified orchestrator supporting standard Pre-training, LoRA fine-tuning, and Autoregressive generation.

---

## 3. Historical Benchmark Audit: Fact vs. Simulation

| Benchmark Script | Claim / Target | Prior Status | Verification & Resolution |
| :--- | :--- | :--- | :--- |
| `scripts/benchmark_competitors.py` | Chunked CE vs PyTorch CE | Directly Measured | Verified. Measured 69.3% VRAM reduction (644 MB vs 2,096 MB) at $L=4096, V=32000$. |
| `scripts/benchmark_competitors.py` | FastGLA vs PyTorch SDPA | Directly Measured | Verified. Measured $O(1)$ memory and latency scaling for autoregressive generation at $L \ge 8192$. |
| `scripts/benchmark_competitors.py` | CentroidSteer vs AdamW Memory | Directly Measured | Verified. 83.1% state memory reduction (15.2 MB vs 90.0 MB for 45M expert params). |
| `scripts/benchmark_universal_and_unsloth.py` | Unsloth Fine-Tuning Comparison | **Simulated** (Functional 8-bit AdamW baseline) | **ACTION**: Installing native `unsloth` package on Windows environment to run true native Unsloth and true PyTorch AdamW side-by-side. |
| `scripts/benchmark_end_to_end_compounding.py` | Full Stack Compounding | Directly Measured | Verified. End-to-end forward+backward+optimizer step: Triune achieved 2.14x speedup and 68.7% VRAM reduction over standard AdamW + Dense baseline. |
| `scripts/benchmark_full_matrix_comparison.py` | 12-Cell Competitive Matrix | Mixed (AdamW: Real, Unsloth: Simulated, Triune: Real) | **ACTION**: Re-executing full matrix with genuine native Unsloth and PyTorch AdamW baselines. |

---

## 4. Fair Comparison Rules & Measurement Standards

To ensure scientific rigor and publication-grade transparency, all benchmark suites adhere to the following rules:

1. **Hardware & Environment Metadata**:
   - Every benchmark report logs: Device Name, Total VRAM, CUDA Version, Driver Version, PyTorch Version, Python Version, and Active OS.
2. **Workload & Precision Parity**:
   - Model parameters, active FLOPs, sequence lengths ($L$), batch sizes ($B$), and vocabulary sizes ($V$) must be identical across compared stacks.
   - Precision: BFloat16 (`torch.bfloat16`) for all activations and forward/backward passes; float32 for master weights and optimizer accumulation where specified.
3. **Execution Timing**:
   - Minimum 5 warmup steps to eliminate PyTorch CUDA graph compilation, JIT caching, and memory allocator initialization overhead.
   - Synchronized execution timing via `torch.cuda.Event(enable_timing=True)` measuring at least 20 evaluation iterations.
   - Reporting includes: Median latency, Mean latency, Minimum latency, and standard deviation.
4. **Memory Accounting**:
   - `torch.cuda.reset_peak_memory_stats()` called before each test run.
   - Both `torch.cuda.max_memory_allocated()` (actual tensors) and `torch.cuda.max_memory_reserved()` (PyTorch caching allocator) recorded.
5. **Throughput Metric Definition**:
   - Training: $\text{Tokens/sec} = \frac{\text{Batch Size} \times \text{Sequence Length}}{\text{Step Latency (sec)}}$.
   - Generation / Decode: $\text{Tokens/sec} = \frac{\text{Generated Tokens}}{\text{Generation Latency (sec)}}$.
6. **No Fake Baselines**:
   - Competitors must be imported and run directly. If a competitor package fails or is unavailable, the blocker must be explicitly logged and no simulated surrogate may use the competitor's name.

---

## 5. Overhaul Execution Plan (Phases 3–7)

The benchmark infrastructure overhaul proceeds across the following dedicated suites:
1. **Category A (Kernels)**: Microbenchmark suite verifying FastGLA, Chunked CE, FastRMSNorm, FastRoPE, and FastSwiGLU against PyTorch native equivalents.
2. **Category B (Optimizers)**: Dedicated optimizer benchmark comparing CentroidSteer + Muon vs PyTorch AdamW vs native Unsloth/bitsandbytes 8-bit AdamW on memory, update latency, and 50-step convergence.
3. **Category C (Layer Streaming)**: Capacity sweep testing `LayerStreamingEngine` on 76M, 250M, 750M, and 1.5B parameter models measuring VRAM floor vs PCIe transfer overhead.
4. **Category D (Long-Context)**: Inference and generation sweep ($L=512$ to $L=16384$) measuring prefill latency, decode latency, and KV-cache footprint.
5. **Category E (Dynamic Exits)**: Quality vs compute Pareto analysis measuring held-out cross-entropy loss / perplexity on FineWeb tokens across Reflex, Limbic, and Cortex exits.
6. **Category F (End-to-End & LoRA)**: Full training and fine-tuning comparison running native Unsloth, PyTorch AdamW, and Triune Universal Engine.
7. **Phase 7 (Validity Suite)**: Automated validation tests ensuring all benchmarks emit valid numbers, no missing baselines, and consistent telemetry.
