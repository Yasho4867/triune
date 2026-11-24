# Lucid Machines — Empirical Benchmark & Competitive Evaluation Suite

**Date**: November 15, 2026  
**Environment**: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM) | PyTorch: 2.14.1+cu132 | CUDA: 13.2 | BF16 Tensor Cores

## Overview of Benchmark Categories

This repository contains publication-grade, hardware-verified benchmarks evaluating **Lucid Machines**' (`lucid_machines`) **Triune** architecture and runtime components (powering the flagship **Junkie** model) against PyTorch standard baselines and genuine native competitor implementations (`unsloth 2026.10.10` / `bitsandbytes 0.50.2` / PyTorch SDPA; `liger_kernel` is unavailable on Windows in our environment and is explicitly omitted rather than approximated). We explicitly distinguish **isolated kernel microbenchmarks** from **end-to-end 90.79M full-model evaluations** and **legacy block diagnostics** so single-layer kernel speedups are never conflated with full-stack latency.

> **CLI Quickstart (`kcx` — Polysom KCX):** Install with `pip install -e .` and run `kcx train --model_name junkie`, `kcx chat`, `kcx studio` (compatibility aliases `polysom` / `lucid-machines` also supported), or execute the benchmark suite via `python scripts/run_all_benchmarks.py --all`.

### Benchmark Suite Matrix

| Category | Topic | Script / Driver | Competitors / Baselines | Primary Result (Nov 2026 Verified) | Report Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | **Kernel Microbenchmarks** | `scripts/benchmark_competitors.py` | PyTorch, Genuine Unsloth (`2026.10.10`), BnB `0.50.2` | CE VRAM `-98.5%` vs PT (`-94.1%` vs Unsloth); SwiGLU `1.95x`; isolated GLA `10.07x` prefill / `7.31x` decode; NVFP4 `3.2x` | [Report](COMPETITIVE_BENCHMARK_REPORT.md) |
| **B** | **Optimizer Dynamics & N=8 MoE** | `scripts/benchmark_moe_convergence_100_steps.py` | AdamW, Muon, GaLore, BAdam, CentroidSteer | `3.78 MB` (`0.1118 B/p`) measured Tier-2 state (`0.128 B/p` analytical); `p=0.0082` vs AdamW; ANOVA `46.86%` var. red. | [Report](MOE_CONVERGENCE_100_STEPS_REPORT.md) |
| **C** | **Layer Streaming Scaling** | `scripts/benchmark_layer_streaming.py` | GPU-Resident vs Checkpointed Streaming | `-76.4%` (`102M`), `-83.8%` & **`1.97x` faster** (`271M` WDDM); **`695M` (`2.12 GB`) & `1.52B` (`3.13 GB`)** in `8 GB` | [Report](LAYER_STREAMING_REPORT.md) |
| **D** | **Long-Context Generation** | `scripts/benchmark_long_context.py` | Multi-layer GLA vs Softmax SDPA ($O(1)$ vs $O(L)$) | Constant `64.0 KiB` state (`99.8%` @ `16k`), `4.02x` prefill (`4.40M tok/s`) & `4.13x` decode @ `16k` | [Report](LONG_CONTEXT_REPORT.md) |
| **E** | **Dynamic Exits & Pareto** | `scripts/benchmark_dynamic_exits.py` | Reflex (L2), Limbic (L5), Cortex (L8) | Reflex `280,941 tok/s` (`7.29 ms`) vs Cortex `76,543 tok/s` (`26.76 ms`); Dynamic `299,506 tok/s` | [Report](DYNAMIC_EXITS_REPORT.md) |
| **F** | **Full Competitive Matrix** | `scripts/benchmark_full_matrix_comparison.py` | PyTorch AdamW, BitsAndBytes 8-bit AdamW | Verified 3x2x2 pre-training and LoRA fine-tuning | [Report](MASTER_COMPETITIVE_MATRIX_REPORT.md) |
| **G** | **Genuine Full-Stack Unsloth** | `scripts/benchmark_unsloth_real.py` | Genuine Unsloth (`2026.10.10`) vs PyTorch vs Lucid Machines | `+6.7%` LoRA throughput vs Unsloth (`25,973` vs `24,344 tok/s`); lowest dense VRAM (`423.9 MB`); native MoE | [Report](GENUINE_UNSLOTH_VS_TRIUNE_REPORT.md) |
| **H** | **CUDA Graph Replay & Parity** | `scripts/benchmark_cuda_graph.py` | PyTorch Eager, CUDA Graph Sync vs Async | -99.0% host launch latency, 1.60x wall-clock speedup, `0.00e+00` drift | [Report](CUDA_GRAPH_BENCHMARK_REPORT.md) |
| **I** | **Full 90.79M Model E2E & CALM** | `scripts/benchmark_full_model_e2e.py`, `benchmarks/calm_calibration.py` | Rescued `step_40555` (`90.79M` / `64.21M` active) | `181.2k` (`B=1`) / `297.8k` (`B=4`) `16k` prefill, `157 tok/s` decode, `91.62/90.74` spec, `N=85,503` CALM, 4-tier Hadamard NVFP4, `2.65x` tok | [E2E](FULL_MODEL_E2E_REPORT.md) / [CALM](calm_calibration.md) |
| **J** | **Production Sparse `MoE_FFN`** | `benchmarks/moe_ffn_sparse_vs_dense.py` | Sparse Scatter-Gather vs Dense 3D BMM | `4.774x` (`top1`) & `5.450x` (`top2`) @ `1536d`; `7.518x` @ `384d` (`75.0%` FLOPs saved) | [Report](FULL_MODEL_E2E_REPORT.md) |
| **K** | **MQAR Associative Recall** | `scripts/benchmark_associative_recall.py` | Softmax vs GLA vs Gated DeltaNet | `100.0%` std recall across all; `100.0%` overwrite recall on Gated DeltaNet | [Report](MQAR_ASSOCIATIVE_RECALL_REPORT.md) |

---

## Detailed Empirical Findings & Honest Trade-Offs

### 1. Master Competitive Matrix (`Category F`, Verified Hardware Measurements)
Evaluated across identical architectures (Universal Dense **43,456,896 scalar params** and Triune MoE **90,789,271 scalar params**) with **actively participating LoRA adapters**, verified parameter weight updates, exact scalar parameter accounting, and genuine BitsAndBytes 8-bit AdamW on real FineWeb text batches:

- **Dense LoRA Fine-Tuning** (44.59M total, 1.13M trainable params / 2.54%):
  - **Memory**: Lucid Machines uses **496.4 MB peak VRAM**, representing a **38.7% reduction vs Standard AdamW (810.0 MB)** and a **27.1% reduction vs BitsAndBytes 8-bit AdamW (681.2 MB)** due to fused linear-LoRA and chunked logit-free backprop.
  - **Throughput**: Lucid Machines runs at **28.61 ± 3.01 ms/step (35,786 tok/s)** vs **32.15 ± 4.94 ms/step (31,848 tok/s)** for BitsAndBytes 8-bit AdamW and **26.53 ± 3.47 ms/step (38,599 tok/s)** for standard AdamW.
- **MoE LoRA Fine-Tuning** (91.28M total, 491.5K trainable params / 0.54%):
  - Lucid Machines executes at **87.28 ± 14.74 ms/step** (11,732 tok/s) and **994.8 MB peak VRAM** (11.2% lower VRAM than Standard AdamW's 1,120.4 MB, operating at near parity with BitsAndBytes's 993.9 MB and 103.28 ms/step).
- **Dense Pre-Training (Memory & Convergence Win, Throughput Trade-off)**:
  - **Memory**: Lucid Machines uses **573.0 MB peak VRAM**, saving **39.2% vs standard AdamW (942.7 MB)** and **21.7% vs BitsAndBytes 8-bit AdamW (732.1 MB)**.
  - **Throughput Trade-off**: Lucid Machines runs at **32.79 ± 1.84 ms/step** (31,224 tok/s) vs **22.74 ± 1.93 ms/step** (45,036 tok/s) for standard AdamW due to online subspace projections and 3-tier Muon orthogonalization.
- **MoE Pre-Training (Memory Win)**:
  - Lucid Machines achieves the lowest peak VRAM in this category (**1,150.4 MB**, saving 22.0% vs AdamW's 1,474.2 MB and 5.5% vs BitsAndBytes's 1,217.8 MB) and executes at **97.32 ± 12.66 ms/step** (10,522 tok/s).

### 1.1 Genuine Full-Stack Unsloth Head-to-Head (`Category G`, Isolated Subprocesses)
Benchmarked against the actual installed **`unsloth` 2026.10.10 framework** (`bitsandbytes 0.50.2`, real causal FineWeb text batches) across Pre-Training and LoRA Fine-Tuning (`37,165,440` base params, 8 layers, 384 dim, 1024 MLP, 6 heads, 2 KV heads; `90,789,271` MoE params):
- **LoRA Fine-Tuning** (`38.03M` total / `868,352` trainable params / `2.28%`, Rank 16, Batch Size 2, Seq Len 256):
  - **Throughput**: Lucid Machines runs at **39.43 ± 2.95 ms/step (25,973 tok/s)** vs Genuine Unsloth's **42.06 ± 4.62 ms/step (24,344 tok/s)** — delivering **`+6.7%` higher throughput than Unsloth** and a **`1.41x` speedup over Standard PyTorch PEFT (`55.76 ± 7.98 ms/step`, `18,364 tok/s`)**.
  - **VRAM**: Lucid Machines uses **396.4 MB peak VRAM** (`-33.8%` vs Standard PyTorch's `599.1 MB`; Genuine Unsloth `340.6 MB`).
- **Pre-Training from Scratch** (`37.17M` params, Batch Size 4, Seq Len 256):
  - Standard PyTorch (AdamW) executes at **`36.12 ± 3.32 ms/step` (`28,348 tok/s`, `661.6 MB`)**.
  - Genuine Unsloth (`2026.10.10`, `AdamW8bit`) achieves **`27.70 ± 3.65 ms/step` (`36,967 tok/s`, `464.8 MB`)**.
  - Lucid Machines' `CentroidSteerOptimizer` achieves **`28.96 ± 2.84 ms/step` (`35,361 tok/s`, `423.9 MB`)**, delivering the **lowest peak VRAM** (**`-35.9%` vs Standard PyTorch**, **`-8.8%` vs Genuine Unsloth**) alongside 3-tier Muon/CentroidSteer updates.
- **Mixture of Experts Support**:
  - Lucid Machines natively supports and trains `90.79M` Multi-Exit MoE models (**`107.11 ± 11.56 ms/step`, `9,560 tok/s`, `928.9 MB`**), whereas Unsloth is specialized exclusively for dense autoregressive architectures.

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
  - At $N=2,048$: PyTorch `14.45 ms (1,032.0 MB)` vs Genuine Unsloth `6.17 ms (275.9 MB)` vs Lucid Machines `5.48 ms (56.2 MB)`.
  - At $N=4,096$: PyTorch `25.93 ms (2,000.0 MB)` vs Genuine Unsloth `12.69 ms (526.4 MB)` vs Lucid Machines `11.44 ms (57.7 MB)`.
  - At $N=8,192$: PyTorch `59.04 ms (4,000.0 MB)` vs Genuine Unsloth `24.67 ms (1,029.4 MB)` vs Lucid Machines **`22.97 ms (60.7 MB)`** -> **`98.5%` VRAM saved vs PyTorch** (`94.1%` vs Unsloth), **`2.57x` faster than PyTorch** (`1.07x` vs Unsloth), `loss_diff=0.00e+00`, `grad_diff=4.47e-08`.
- **Fused Activations & 4-Bit NVFP4 (`[4096 x 4096]`)**:
  - **RMSNorm** (`4x1024, d=2048`): Lucid Machines `0.349 ms (33.03 MB)` vs Naive `4.185 ms (208.0 MB)` (**`11.99x`**, `-84.1%` VRAM), `F.rms_norm` `0.353 ms (32.02 MB)`, and Genuine Unsloth `0.280 ms (33.13 MB)`.
  - **SwiGLU** (`4x1024, h=4096`): Lucid Machines **`1.123 ms (96.0 MB)`** vs PyTorch `2.190 ms (128.0 MB)` (**`1.95x`**) and Genuine Unsloth `2.221 ms (96.0 MB)` (**`1.98x`**).
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
  - Production `CentroidSteer` Tier 2 (routed expert weights $384 \times 1024, r=16$) maintains only an upper-triangular $(r+1)\times(r+1)$ FP32 core plus BF16 Stiefel bases: **0.128 bytes/param** analytical (**3.78 MB / 0.1118 bytes/param** measured in $N=8$ 90.79M MoE pre-training vs. `135.21 MB` AdamW and `6.68 MB` GaLore; **98.4% reduction vs FP32 AdamW**), yielding **4.54 bytes/param model-wide** on 90.79M MoE with FP32 Tier 3 AdamW (**1.27 bytes/param** with 8-bit Tier 3). In $N=8$ FineWeb-Edu MoE runs, Triune MoE improves over Standard AdamW (`p=0.0082`, paired $\Delta = -0.1514$) with Second-Moment ANOVA variance conservation ($46.86\%$ within-expert dispersion reduction), while centroid steering vs. unsteered Stiefel ($p=0.1188$) establishes no statistically significant 100-step loss separation.
- **Peak Step Memory & Refresh Cost**:
  - Non-refresh steps (95%–99.5% of steps) perform zero QR calls and zero host-device syncs. Refresh steps execute `sCholQR2` Krylov projections with sliced Householder fallback ($83.76\%$ fast-path survival across $11,832$ live calls), and scratch buffers are freed immediately before the next forward pass. In $N=12$ fresh OS child processes, Triangular CentroidSteer cuts peak PyTorch tensor allocation by **122.4 MB (-11.5%)** (`942.09 ± 7.81 MB` vs `1,064.49 ± 6.83 MB`) at matched step latency (`114.36 ± 37.69 ms` vs `117.67 ± 37.37 ms`).

### 5. End-to-End 90.79M Checkpoint, Conformal CALM, NVFP4 & Sparse MoE (`Categories I, J`)
- **Complete `90.79M` (`64.21M` active) Rescued Checkpoint (`FULL_MODEL_E2E_REPORT.md`, `checkpoints/triune_studio_step_36500_rescued.pt`, `step_40555`)**:
  - Monotonic exit recovery (`scripts/rescue_checkpoint.py`, `+16.61M` tokens in `61.04 s` at `272,110 tok/s`): flips inverted raw `step_36500` validation perplexities (`112.55 / 114.08 / 132.67`) to monotonic **Reflex `111.80` > Limbic `111.18` > Cortex `110.92` PPL** over `65,536` held-out tokens.
  - Sustains **`200,463 tok/s`** (`B=1, T=2048`), **`242,029 tok/s`** (`B=1, T=8192`), **`181,150 tok/s` (`B=1`) and `297,753 tok/s` (`B=4`) at `L=16,384`** (`527.44 MB` / `853.14 MB` VRAM), **`383,135 tok/s`** peak prefill (`B=32, T=1024`), **`157 tokens/sec`** sequential decode (`6.363 ms/tok`), **`4.31x` Reflex (L2) vs Cortex (L8)** speedup (`2.764 ms` vs `11.90 ms` at `B=4, T=512`), **`91.62 tok/s`** greedy exact-match self-speculative decode (`T=0.0, 51.28%` acceptance), **`90.74 tok/s`** $\chi^2$-verified Leviathan et al. rejection-sampled decode (`T=0.7, 63.64%` acceptance), and **`2.65x`** held-out lossless text compression (`compress_mode="expand"` exact `0.0` max logit diff; `order_pool` and `mean_pool`).
  - **Conformal CALM Calibration (`benchmarks/calm_calibration.md`, `N=87,141` cal / `N=85,503` eval tokens)**: KL self-distillation aligns intermediate heads (`AUROC = 0.8159` Reflex, `0.8319` Limbic) and meets conformal guarantees on held-out FineWeb-Edu ($\tau=0.20$: `95.73%` Reflex agreement at `31.34%` coverage saving **`20.35%` transformer FLOPs** at `+0.14%` PPL delta; `95.5%` Python / `97.1%` FineWeb on `N=2048` E2E prompts). Note: while token confidence & exit-logit retention are per-token, `TriuneTransformer` compacts deeper layers at sample/row level (`~settled_r.all(dim=1)`), causing eager batched CALM on `D=384, V=32,000` to execute at `0.294x–0.723x` vs dense Cortex.
  - **4-Tier E2E Weight Quantization** (`78.26M` linear params across `96` `NVFP4Linear` blocks + `7 NVFP4StagingBuffer` transfers = `5.94 MB`): BF16 `149.26 MB (97.504 PPL)`, FP8 E4M3 `74.63 MB (2.00x, 101.560 PPL)`, Unrotated NVFP4 `46.65 MB (3.20x, 126.887 PPL, cos=0.80252)`, and **NVFP4 Hadamard Fused W4A16+Protected `78.88 MB (1.89x, 119.106 PPL, cos=0.81629)`**.
  - Production `MoE_FFN` (`use_sparse_eval=True` by default, zero-sync `N_tok=1` BMM `1.648x` / `0.0` max diff) achieves **`7.518x` speedup** (`1.990 ms` vs `14.957 ms`, `75.0%` FLOPs saved) at checkpoint width (`d=384, E=8, k=2, N=4096`) and **`4.774x` (`k=1`) / `5.450x` (`k=2`)** at `d=1536, E=8, N=4096` (`benchmarks/moe_ffn_sparse_vs_dense.py`), reported separately from legacy Python-loop `MoERoutingBlock` (`13.916 ms` vs `9.244 ms`).

### 6. Activation-Checkpointed Layer Streaming Scaling (`Category C`)
`LayerStreamingEngine` (`activation_checkpointing=True`, `_is_bwd_staged` reentry guard, cuBLAS workspace cleanup, generic HF layer discovery) virtualizes model layers between host RAM and GPU VRAM (`LAYER_STREAMING_REPORT.md`):
- **Measured Consumer-GPU Scaling (`B=2, T=512, V=4096`, BF16 on 8 GB RTX 5070)**:
  - **101.97M (`AdamW BF16`)**: Peak allocated VRAM cut by **`76.43%`** (`1,033.6 MB` vs `4,384.9 MB`), step `773.6 ms` vs `151.6 ms` (`1,323.7 tok/s`).
  - **270.56M (`AdamW BF16`)**: Peak allocated VRAM cut by **`83.76%`** (`1,498.1 MB` vs `9,224.7 MB` WDDM spill), running **`1.97x` faster** than WDDM paging (`1,576.5 ms` vs `3,111.6 ms`, `649.5 tok/s`).
  - **694.85M (`AdamW BF16`)**: Prevents GPU-resident OOM (`>13.93 GiB` requested), training at **`2,115.7 MB`** allocated (`4,076.0 MB` reserved, `4,026.9 ms/step`, `254.3 tok/s`).
  - **1,522.76M (`CentroidSteer r=16`, 8-bit base)**: Prevents GPU-resident OOM (`>13.32 GiB` requested), training inside `8 GB` VRAM at **`3,132.6 MB`** allocated (`6,248.0 MB` reserved, `5,033.4 ms/step`, `203.4 tok/s`).
- **Takeaway**: Checkpointed layer streaming trades host-device bandwidth for depth invariance ($\Theta(1)$ weight/optimizer VRAM), preventing OOM up to `1.52B` and outperforming Windows WDDM oversubscription once resident memory spills past physical VRAM.
