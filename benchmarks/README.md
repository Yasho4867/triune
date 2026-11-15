# Triune AI — Empirical Benchmark & Competitive Evaluation Suite

**Date**: November 15, 2026  
**Environment**: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM) | PyTorch: 2.14.1+cu132 | CUDA: 13.2 | BF16 Tensor Cores

## Overview of Benchmark Categories

This repository contains publication-grade, hardware-verified benchmarks evaluating **Triune AI**'s architectural and runtime components against PyTorch standard baselines and genuine native competitor implementations (`unsloth 2026.10.10` / `bitsandbytes 0.50.2` / PyTorch SDPA; `liger_kernel` is unavailable on Windows in our environment and is explicitly omitted rather than approximated). We explicitly distinguish **isolated kernel microbenchmarks** from **end-to-end 90.79M full-model evaluations** and **legacy block diagnostics** so single-layer kernel speedups are never conflated with full-stack latency.

### Benchmark Suite Matrix

| Category | Topic | Script / Driver | Competitors / Baselines | Primary Result (Nov 2026 Verified) | Report Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | **Kernel Microbenchmarks** | `scripts/benchmark_competitors.py` | PyTorch, Genuine Unsloth (`2026.10.10`), BnB `0.50.2` | CE VRAM `-98.5%` vs PT (`-94.1%` vs Unsloth); SwiGLU `1.95x`; isolated GLA `10.07x` prefill / `7.31x` decode; NVFP4 `3.2x` | [Report](COMPETITIVE_BENCHMARK_REPORT.md) |
| **B** | **Optimizer Dynamics** | `scripts/benchmark_optimizer_suite.py` | PyTorch AdamW, BitsAndBytes 8-bit AdamW | `CentroidSteer` Tier 2 `0.128 B/p` (-98.4%); `4.54 B/p` model-wide | [Report](OPTIMIZER_BENCHMARK_REPORT.md) |
| **C** | **Layer Streaming** | `scripts/benchmark_layer_streaming.py` | Standard GPU-Resident Training | Up to -47.2% peak VRAM virtualization trade-off | [Report](LAYER_STREAMING_REPORT.md) |
| **D** | **Long-Context Generation** | `scripts/benchmark_long_context.py` | Multi-layer GLA vs Softmax SDPA ($O(1)$ vs $O(L)$) | Constant `64.0 KiB` state (`99.8%` @ `16k`), `4.02x` prefill (`4.40M tok/s`) & `4.13x` decode @ `16k` | [Report](LONG_CONTEXT_REPORT.md) |
| **E** | **Dynamic Exits & Pareto** | `scripts/benchmark_dynamic_exits.py` | Reflex (L2), Limbic (L5), Cortex (L8) | Reflex `280,941 tok/s` (`7.29 ms`) vs Cortex `76,543 tok/s` (`26.76 ms`); Dynamic `299,506 tok/s` | [Report](DYNAMIC_EXITS_REPORT.md) |
| **F** | **Full Competitive Matrix** | `scripts/benchmark_full_matrix_comparison.py` | PyTorch AdamW, BitsAndBytes 8-bit AdamW | Verified 3x2x2 pre-training and LoRA fine-tuning | [Report](MASTER_COMPETITIVE_MATRIX_REPORT.md) |
| **G** | **Genuine Full-Stack Unsloth** | `scripts/benchmark_unsloth_real.py` | Genuine Unsloth (`2026.10.10`) vs PyTorch vs Triune | `+6.7%` LoRA throughput vs Unsloth (`25,973` vs `24,344 tok/s`); lowest dense VRAM (`423.9 MB`); native MoE | [Report](GENUINE_UNSLOTH_VS_TRIUNE_REPORT.md) |
| **H** | **CUDA Graph Replay & Parity** | `scripts/benchmark_cuda_graph.py` | PyTorch Eager, CUDA Graph Sync vs Async | -99.0% host launch latency, 1.60x wall-clock speedup, `0.00e+00` drift | [Report](CUDA_GRAPH_BENCHMARK_REPORT.md) |
| **I** | **Full 90.79M Model E2E** | `scripts/benchmark_full_model_e2e.py` | Complete `checkpoints/` (`90.79M` / `64.21M` active) | `444,530` (`1.55M` peak) tok/s prefill, `129 tok/s` decode, `77.07 tok/s` spec, `100.0%` post-distill CALM, NVFP4 `46.65 MB`, `2.65x` tok | [Report](FULL_MODEL_E2E_REPORT.md) |
| **J** | **Production Sparse `MoE_FFN`** | `benchmarks/moe_ffn_sparse_vs_dense.py` | Sparse Scatter-Gather vs Dense 3D BMM | `4.774x` (`top1`) & `5.450x` (`top2`) @ `1536d`; `7.458x` @ `384d` | [Report](FULL_MODEL_E2E_REPORT.md) |
| **K** | **MQAR Associative Recall** | `scripts/benchmark_associative_recall.py` | Softmax vs GLA vs Gated DeltaNet | `100.0%` std recall across all; `100.0%` overwrite recall on Gated DeltaNet | [Report](MQAR_ASSOCIATIVE_RECALL_REPORT.md) |

---

## Detailed Empirical Findings & Honest Trade-Offs

### 1. Master Competitive Matrix (`Category F`, Verified Hardware Measurements)
Evaluated across identical architectures (Universal Dense **43,456,896 scalar params** and Triune MoE **90,789,271 scalar params**) with **actively participating LoRA adapters**, verified parameter weight updates, exact scalar parameter accounting, and genuine BitsAndBytes 8-bit AdamW on real FineWeb text batches:

- **Dense LoRA Fine-Tuning** (44.59M total, 1.13M trainable params / 2.54%):
  - **Memory**: Triune uses **496.4 MB peak VRAM**, representing a **38.7% reduction vs Standard AdamW (810.0 MB)** and a **27.1% reduction vs BitsAndBytes 8-bit AdamW (681.2 MB)** due to fused linear-LoRA and chunked logit-free backprop.
  - **Throughput**: Triune runs at **28.61 ± 3.01 ms/step (35,786 tok/s)** vs **32.15 ± 4.94 ms/step (31,848 tok/s)** for BitsAndBytes 8-bit AdamW and **26.53 ± 3.47 ms/step (38,599 tok/s)** for standard AdamW.
- **MoE LoRA Fine-Tuning** (91.28M total, 491.5K trainable params / 0.54%):
  - Triune executes at **87.28 ± 14.74 ms/step** (11,732 tok/s) and **994.8 MB peak VRAM** (11.2% lower VRAM than Standard AdamW's 1,120.4 MB, operating at near parity with BitsAndBytes's 993.9 MB and 103.28 ms/step).
- **Dense Pre-Training (Memory & Convergence Win, Throughput Trade-off)**:
  - **Memory**: Triune uses **573.0 MB peak VRAM**, saving **39.2% vs standard AdamW (942.7 MB)** and **21.7% vs BitsAndBytes 8-bit AdamW (732.1 MB)**.
  - **Throughput Trade-off**: Triune runs at **32.79 ± 1.84 ms/step** (31,224 tok/s) vs **22.74 ± 1.93 ms/step** (45,036 tok/s) for standard AdamW due to online subspace projections and 3-tier Muon orthogonalization.
- **MoE Pre-Training (Memory Win)**:
  - Triune achieves the lowest peak VRAM in this category (**1,150.4 MB**, saving 22.0% vs AdamW's 1,474.2 MB and 5.5% vs BitsAndBytes's 1,217.8 MB) and executes at **97.32 ± 12.66 ms/step** (10,522 tok/s).

### 1.1 Genuine Full-Stack Unsloth Head-to-Head (`Category G`, Isolated Subprocesses)
Benchmarked against the actual installed **`unsloth` 2026.10.10 framework** (`bitsandbytes 0.50.2`, real causal FineWeb text batches) across Pre-Training and LoRA Fine-Tuning (`37,165,440` base params, 8 layers, 384 dim, 1024 MLP, 6 heads, 2 KV heads; `90,789,271` MoE params):
- **LoRA Fine-Tuning** (`38.03M` total / `868,352` trainable params / `2.28%`, Rank 16, Batch Size 2, Seq Len 256):
  - **Throughput**: Triune runs at **39.43 ± 2.95 ms/step (25,973 tok/s)** vs Genuine Unsloth's **42.06 ± 4.62 ms/step (24,344 tok/s)** — delivering **`+6.7%` higher throughput than Unsloth** and a **`1.41x` speedup over Standard PyTorch PEFT (`55.76 ± 7.98 ms/step`, `18,364 tok/s`)**.
  - **VRAM**: Triune uses **396.4 MB peak VRAM** (`-33.8%` vs Standard PyTorch's `599.1 MB`; Genuine Unsloth `340.6 MB`).
- **Pre-Training from Scratch** (`37.17M` params, Batch Size 4, Seq Len 256):
  - Standard PyTorch (AdamW) executes at **`36.12 ± 3.32 ms/step` (`28,348 tok/s`, `661.6 MB`)**.
  - Genuine Unsloth (`2026.10.10`, `AdamW8bit`) achieves **`27.70 ± 3.65 ms/step` (`36,967 tok/s`, `464.8 MB`)**.
  - Triune's `CentroidSteerOptimizer` achieves **`28.96 ± 2.84 ms/step` (`35,361 tok/s`, `423.9 MB`)**, delivering the **lowest peak VRAM** (**`-35.9%` vs Standard PyTorch**, **`-8.8%` vs Genuine Unsloth**) alongside 3-tier Muon/CentroidSteer updates.
- **Mixture of Experts Support**:
  - Triune natively supports and trains `90.79M` Multi-Exit MoE models (**`107.11 ± 11.56 ms/step`, `9,560 tok/s`, `928.9 MB`**), whereas Unsloth is specialized exclusively for dense autoregressive architectures.

### 1.2 CUDA Graph Replay & Multi-Step Mathematical Parity (`Category H`)
Evaluated using `scripts/benchmark_cuda_graph.py` on 37.17M parameters across eager execution, synchronous graph replay, and asynchronous graph replay, alongside a 10-step mathematical equivalence verification:
- **Host Launch Latency Elimination**:
  - Standard PyTorch Eager: **18,278.5 $\mu$s** CPU launch latency.
  - CUDA Graph with Host Sync (`sync=True` via `.item()`): **14,362.0 $\mu$s** (blocks host until GPU finishes).
  - CUDA Graph Asynchronous (`sync=False` default): **188.3 $\mu$s** (**-99.0% host launch latency**, zero host-device synchronization).
  - Total step latency drops from **22.90 ms** (eager) to **14.33 ms** (graph) — delivering a **1.60x end-to-end wall-clock speedup**.
- **10-Step Multi-Step Optimizer Parity**:
  - Tested over 10 consecutive training steps against eager PyTorch using identical seeds and real causal FineWeb sequences.
  - Maximum observed parameter weight drift: **`0.00e+00`**.
  - Maximum observed loss delta: **`0.00e+00`**.
  - Proves 100% mathematical and numerical equivalence with standard PyTorch eager training.

### 2. Kernel Primitives, Chunked CE & 4-Bit NVFP4 (`Category A`)
Benchmarked directly on hardware against PyTorch, genuine `unsloth 2026.10.10`, and `bitsandbytes 0.50.2` (`liger_kernel` is unavailable on Windows and explicitly omitted rather than approximated):
- **Chunked Logit-Free Cross-Entropy ($V=32,000$)**:
  - At $N=2,048$: PyTorch `14.45 ms (1,032.0 MB)` vs Genuine Unsloth `6.17 ms (275.9 MB)` vs Triune `5.48 ms (56.2 MB)`.
  - At $N=4,096$: PyTorch `25.93 ms (2,000.0 MB)` vs Genuine Unsloth `12.69 ms (526.4 MB)` vs Triune `11.44 ms (57.7 MB)`.
  - At $N=8,192$: PyTorch `59.04 ms (4,000.0 MB)` vs Genuine Unsloth `24.67 ms (1,029.4 MB)` vs Triune **`22.97 ms (60.7 MB)`** -> **`98.5%` VRAM saved vs PyTorch** (`94.1%` vs Unsloth), **`2.57x` faster than PyTorch** (`1.07x` vs Unsloth), `loss_diff=0.00e+00`, `grad_diff=4.47e-08`.
- **Fused Activations & 4-Bit NVFP4 (`[4096 x 4096]`)**:
  - **RMSNorm** (`4x1024, d=2048`): Triune `0.349 ms (33.03 MB)` vs Naive `4.185 ms (208.0 MB)` (**`11.99x`**, `-84.1%` VRAM), `F.rms_norm` `0.353 ms (32.02 MB)`, and Genuine Unsloth `0.280 ms (33.13 MB)`.
  - **SwiGLU** (`4x1024, h=4096`): Triune **`1.123 ms (96.0 MB)`** vs PyTorch `2.190 ms (128.0 MB)` (**`1.95x`**) and Genuine Unsloth `2.221 ms (96.0 MB)` (**`1.98x`**).
  - **Weight Quantization (`[4096x4096]`)**: BF16 `32.0 MB (1.0x, 0.855 ms, cos=1.0)`, FP8 `16.0 MB (2.0x, 1.915 ms, cos=0.99931)`, and packed **4-bit NVFP4 `10.0 MB (5.0 b/w, 3.2x, 0.99 ms [0.399 ms QDQ], cos=0.99503)`**.

### 3. Gated Linear Attention & Associative Recall (`Categories D, K`)
- **Isolated Kernel vs. Multi-Layer Sweeps ($O(1)$ Recurrent vs Softmax KV-Cache $O(L)$)**:
  - In isolated single-layer kernel profiling (`H=6, d_k=64`, BF16; `COMPETITIVE_BENCHMARK_REPORT.md`), GLA maintains a fixed `48.0 KiB` recurrent state (`99.8%` reduction vs `24,576.0 KiB` SDPA KV-cache at $L=16,384$) and achieves `0.1399 ms/tok` vs `1.0226 ms/tok` (**`7.31x` decode speedup**) alongside `0.60 ms` (`27,169,010 tok/s`) vs `6.07 ms` (`2,698,429 tok/s`) at `T=8,192` prefill (**`10.07x` prefill speedup**).
  - In full multi-head `VectorisedGLA` sweeps (`scripts/benchmark_long_context.py`, `H=8, d=512`, `LONG_CONTEXT_REPORT.md`), GLA crosses over SDPA at `L >= 4,096` (`1.02 ms / 56.69 MB` vs `1.33 ms / 68.50 MB`) and reaches `3.72 ms (4,404,384.0 tok/s, 94.69 MB)` vs SDPA `14.97 ms (1,094,700.6 tok/s, 152.50 MB)` at `L=16,384` (**`4.02x` prefill speedup, `-37.9%` VRAM**), while decode retains a strictly invariant `64.0 KiB` state (`99.8%` cache reduction vs `32,768 KiB` SDPA) and outperforms SDPA at every length (`0.151 -> 0.147 -> 0.152 -> 0.275 ms/tok` vs `0.166 -> 0.175 -> 0.483 -> 1.137 ms/tok`, **`4.13x` decode speedup** at `16k`).
- **Warmed-Up MQAR (`scripts/benchmark_associative_recall.py`, `MQAR_ASSOCIATIVE_RECALL_REPORT.md`)**:
  - On causal 1-step bigram induction key-value retrieval with 3-iteration per-length warmup ($T \in \{256..2048\}$), Softmax (`0.17 -> 0.97 ms`), GLA (**`0.31 -> 0.33 -> 0.33 -> 0.37 ms`**), and Gated DeltaNet all achieve **100.0% standard recall**. Under **50% Overwrite-Key MQAR**, Gated DeltaNet maintains **100.0% overwrite recall** across every length (`0.27 -> 0.52 -> 0.93 -> 1.80 ms`) vs. GLA (`62.5%–87.5%`) and Softmax (`37.5%–62.5%`).

### 4. Optimizer Dynamics: Persistent State vs Peak Transient Memory (`Category B`)
- **Persistent Optimizer State Memory (sCholQR2 Triangular Core)**:
  - Standard FP32 AdamW retains 2 states ($m, v$ in FP32) per parameter: **8.00 bytes/param**.
  - Native 8-bit AdamW retains quantized INT8 states: **2.00 bytes/param**.
  - Production `CentroidSteer` Tier 2 (routed expert weights $384 \times 1024, r=16$) maintains only an upper-triangular $(r+1)\times(r+1)$ FP32 core plus BF16 Stiefel bases: **0.128 bytes/param** (**98.4% reduction vs FP32 AdamW**), yielding **4.54 bytes/param model-wide** on 90.79M MoE with FP32 Tier 3 AdamW (**1.27 bytes/param** with 8-bit Tier 3). In earlier standalone 17M synthetic suite comparisons (`optimizer_benchmark_results.json`), the model-wide snapshot measured `1.00 B/p` (`-66.4%` vs AdamW).
- **Peak Step Memory & Refresh Cost**:
  - Non-refresh steps (95%–99.5% of steps) perform zero QR calls and zero host-device syncs. Refresh steps execute `sCholQR2` Krylov projections with sliced Householder fallback ($83.76\%$ fast-path survival across $11,832$ live calls), and scratch buffers are freed immediately before the next forward pass. In 36 fresh OS subprocesses (`N=12` seeds, $3\times3$ Latin squares), Triangular CentroidSteer cuts peak PyTorch tensor allocation by **126.8 MB (-11.9%)** at matched step latency (`89.54 ± 6.67 ms` vs `89.67 ± 3.74 ms`).

### 5. End-to-End 90.79M Checkpoint, NVFP4 & Production Sparse MoE (`Categories I, J`)
- **Complete `90.79M` (`64.21M` active) Checkpoint (`FULL_MODEL_E2E_REPORT.md`, `checkpoints/triune_step_24501.pt`)**:
  - Sustains **`444,530 tok/s`** prefill (`B=1, T=2048`, `4.61 ms`, `248.8 MB`), **`1,546,622 tok/s`** peak prefill (`B=32, T=512`, `10.59 ms`, `402.1 MB`), **`129 tokens/sec`** sequential decode (`7.723 ms/tok`), **`3.69x` Reflex (L2) vs Cortex (L8)** speedup (`2.44 ms` vs `9.00 ms` at `B=16, T=512`), **`77.07 tok/s`** greedy exact-match self-speculative decode (`T=0.0, 63.6%` acceptance), **`61.99 tok/s`** Leviathan et al. rejection-sampled decode (`T=0.7, 51.3%` acceptance), and **`2.65x`** held-out lossless text compression (`compress_mode="expand"` exact `0.0` max logit diff).
  - After 50 steps of Cortex-preserving exit-head KL self-distillation (`distill_exit_heads()`, KL loss `236.975 -> 46.5053`, Cortex logits bit-for-bit invariant), compacted per-token CALM routing at `tau=0.20` achieves **`100.0%` high-confidence Reflex-Cortex agreement** on held-out Python (`67.2%` overall) and held-out FineWeb (`79.7%` overall) vs. `8.8%` (`23.4%` overall) on the uncalibrated single-exit baseline at `tau=0.40`.
  - **Multi-Tier E2E Weight Quantization** (`78.26M` linear params across `103` linear layers): BF16 `149.26 MB (10.04 ms, cos=1.0)`, FP8 `74.63 MB (2.0x, 63.67 ms, cos=0.97377)`, and **4-bit NVFP4 `46.65 MB (3.2x, 17.33 ms, cos=0.92187)`** with `7` verified layer-0 `NVFP4StagingBuffer` host-device transfers.
  - Production `MoE_FFN` (`use_sparse_eval=True` by default, zero-sync `N_tok=1` BMM `1.358x` / `0.0` max diff) achieves **`7.458x` speedup** (`1.967 ms` vs `14.668 ms`, `75.0%` FLOPs saved) at checkpoint width (`d=384, E=8, k=2, N=4096`) and **`4.774x` (`k=1`) / `5.450x` (`k=2`)** at `d=1536, E=8, N=4096` (`benchmarks/moe_ffn_sparse_vs_dense.py`), reported separately from legacy Python-loop `MoERoutingBlock` (`9.56 ms` vs `9.428 ms`, `0.99x`).

### 6. Memory-Constrained Training via Layer Streaming (`Category C`)
`LayerStreamingEngine` is a **capacity-unlocking virtualization system** that enables training models larger than physical GPU VRAM by offloading inactive layers to host RAM while preserving write-once forward snapshots and non-parameter buffer mutations (`is_buffer=True`):
- **Measured Trade-Offs (RTX 5070 8 GB VRAM)**:
  - **Config 76M-Spec (101.97M total params)**: Peak VRAM reduced by **33.3%** (`877.8 MB` vs `1,315.9 MB`), with a 7.5x latency overhead (`657.2 ms` vs `87.5 ms`) due to PCIe transfer.
  - **Config 150M-Spec (270.56M total params)**: Peak VRAM reduced by **41.1%** (`1,780.1 MB` vs `3,022.6 MB`), with a 10.6x latency overhead (`1,576.7 ms` vs `149.2 ms`).
  - **Config 350M-Spec (694.85M total params)**: Peak VRAM reduced by **47.2%** (`3,585.7 MB` vs `6,795.0 MB`), with a modest 1.36x latency overhead (`4,012.4 ms` vs `2,938.8 ms`).
- **Takeaway**: Layer streaming trades PCIe transfer bandwidth for memory capacity. It is not designed to accelerate models that already comfortably fit in VRAM, but to prevent Out-Of-Memory crashes and enable training otherwise untrainable models on consumer GPUs.
