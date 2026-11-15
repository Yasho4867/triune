# Triune Model End-to-End Benchmark & Verification Report

**Date:** 2026-11-15 04:07:39 UTC  
**Target System:** NVIDIA GeForce RTX 5070 Laptop GPU (12.0) | PyTorch 2.14.1+cu132 | CUDA 13.2  
**Model Checkpoint:** `checkpoints/triune_step_24501.pt` (Trained to Step 24,501)  
**Architecture Spec:** 8 Layers, Dim 384, 6 Heads, 4 Experts, Vocab 32,000  
**Parameters:** 90,789,271 total (64,206,871 active per token under Top-1 MoE routing) | Precision: `bf16`

---

## Executive Summary & Boundary Distinction

> [!IMPORTANT]
> **Boundary Distinction (Audit Resolution)**:  
> Prior benchmarks reported **5.21M tokens/sec** on isolated batched GLA prefill kernel execution ($B=32, T=4096, D=256$).  
> This report benchmarks the **complete, end-to-end 90.8M-parameter (64.2M active per token) TriuneTransformer** including:
> 1. Token Embeddings
> 2. 8-layer Transformer pipeline (GLA linear attention + dynamic early exits + MoE)
> 3. True Sparse Top-$k$ scatter-gather MoE routing
> 4. Final RMSNorm
> 5. 32,000-class LM Head vocabulary projection
> 6. Autoregressive recurrent GLA generation (Decode throughput)

---

## 1. Full Model Prefill Throughput (End-to-End)

Measures full-stack forward pass through all 8 layers and the 32k vocabulary projection head.

| Batch Size | Sequence Length | Total Positions | Mean Latency (ms) | P50 Latency (ms) | Throughput (Tokens/s) | VRAM (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  1 |  128 |     128 |     2.22 |     2.20 | **     57,585** |   228.5 |
|  1 |  512 |     512 |     2.23 |     2.23 | **    230,153** |   232.6 |
|  1 | 1024 |   1,024 |     3.00 |     2.95 | **    341,415** |   237.8 |
|  1 | 2048 |   2,048 |     4.61 |     4.70 | **    444,530** |   248.8 |
|  4 |  128 |     512 |     2.30 |     2.29 | **    222,747** |   232.6 |
|  4 |  512 |   2,048 |     2.53 |     2.62 | **    810,774** |   248.8 |
|  4 | 1024 |   4,096 |     3.80 |     3.73 | **  1,078,008** |   269.4 |
|  4 | 2048 |   8,192 |     7.60 |     7.56 | **  1,077,481** |   311.3 |
| 16 |  128 |   2,048 |     2.23 |     2.19 | **    917,394** |   248.8 |
| 16 |  512 |   8,192 |    14.07 |    13.99 | **    582,055** |   313.6 |
| 16 | 1024 |  16,384 |    19.57 |    19.57 | **    837,138** |   394.6 |
| 16 | 2048 |  32,768 |    24.71 |    24.72 | **  1,325,868** |   570.3 |
| 32 |  128 |   4,096 |     3.11 |     3.14 | **  1,318,207** |   276.8 |
| 32 |  512 |  16,384 |    10.59 |    10.59 | **  1,546,622** |   402.1 |
| 32 | 1024 |  32,768 |    29.87 |    30.05 | **  1,096,963** |   543.3 |
| 32 | 2048 |  65,536 |    66.45 |    66.46 | **    986,164** |   852.5 |

---

## 2. Autoregressive Sequential Generation (Decode Throughput)

Measures token-by-token sequential autoregressive generation using recurrent constant-memory GLA states:

- **Prompt Length**: 64 tokens
- **Generated Tokens**: 64 tokens
- **Mean Single-Step Latency**: `7.723 ms`
- **Decode Generation Rate**: **`129 tokens/sec`**

---

## 3. Dynamic Early Exit Speedups

Measures hardware latency when early exit decisions are triggered at Reflex (L2) or Limbic (L5) vs full Cortex (L8):

| Exit Layer | Exit Depth | Mean Latency (ms) | Throughput (Tokens/s) | Speedup vs Cortex |
|:---|:---:|:---:|:---:|:---:|
| **Reflex (L2)** | L2 | 2.44 ms | 838,522 tok/s | **3.69x** |
| **Limbic (L5)** | L5 | 5.97 ms | 343,217 tok/s | **1.51x** |
| **Cortex (L8 / Full)** | L8 / Full | 9.00 ms | 227,521 tok/s | **1.0x** |

---

## 4. Self-Speculative Decoding (Meta AI LayerSkip, ACL 2024)

Eliminates auxiliary model overhead by using Reflex (L2) to draft $K$ tokens and Cortex (L8) to verify all $K$ tokens in a single parallel pass:

| Mode | $K$ | Tokens | Speed (tok/s) | Proposed / Accepted | Acceptance Rate | Parity Contract |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Greedy (`T=0.0`)** | 3 | 32 | **77.07** | 33 / 21 | **63.6%** | PASS (Exact Match) |
| **Sampled (`T=0.7`)** | 3 | 32 | **61.99** | 39 / 20 | **51.3%** | Rejection Sampling (`exact_distribution`) |

---

## 5. Dynamic Confidence Gating (Google Research CALM, NeurIPS)

Zero-overhead per-token Softmax Margin ($\Delta p = p_{(1)} - p_{(2)}$) routing with per-sample compacted early exits:

- **Exit-Head KL Self-Distillation (Backbone & Cortex Frozen)**: `50` steps (KL loss `236.975 -> 46.5053`), `checkpoint_calibrated=True`
- **Confidence Threshold ($\tau$)**: `0.2`
- **Held-Out Python Code (`sample_text`, $N=128$)**: Overall Agreement=`67.2%`, High-Confidence Eligible=`10` (7.8%), High-Confidence Agreement=**`100.0%`** (Batched `B=2`: Reflex=19, Limbic=5, Cortex=232)
- **Held-Out FineWeb-Edu ($N=128$)**: Overall Agreement=`79.7%`, High-Confidence Eligible=`24` (18.8%), High-Confidence Agreement=**`100.0%`** (Batched `B=2`: Reflex=40, Limbic=5, Cortex=211)
- **Uncalibrated Raw Checkpoint Baseline ($\tau=0.4$, Python)**: Overall=`23.4%`, High-Confidence=`8.8%` (because untied intermediate exit heads in `triune_step_24501.pt` were trained under single-exit `dynamic` mode without Cortex KL alignment; frozen-backbone KL self-distillation aligns exit heads while keeping Cortex `force_depth=2` logits bit-for-bit invariant).

---

## 6. MoE Execution Paths: Production `MoE_FFN` vs Legacy `MoERoutingBlock`

Compares the production `MoE_FFN` module (`use_sparse_eval=True` by default: zero-sync GPU `N_tok=1` BMM + grouped sparse dispatch for `N_tok>1`) against dense BMM (`use_sparse_eval=False`), separately from the legacy Python-loop `MoERoutingBlock` diagnostic:

| Module / Tokens ($N_{\text{tok}}$) | Sparse Latency | Dense Latency | Speedup | Max Diff |
|:---|:---:|:---:|:---:|:---:|
| **`MoE_FFN` (`N=1`)** | 0.647 ms | 0.878 ms | **1.358x** | `0.0` |
| **`MoE_FFN` (`N=256`)** | 1.656 ms | 1.735 ms | **1.047x** | `0.007812` |
| **`MoE_FFN` (`N=4096`)** | 1.967 ms | 14.668 ms | **7.458x** | `0.007812` |
| `MoERoutingBlock` (legacy loop, `N=4096`) | 9.56 ms | 9.428 ms | 0.99x | `0.000157` |
- **Theoretical Active FLOP Reduction**: **`75.0% FLOPs saved`** (`k=2/8` active experts; see also [`benchmarks/moe_ffn_sparse_vs_dense.json`](moe_ffn_sparse_vs_dense.json)).

---

## 7. Dynamic Tokenizer Disjoint Evaluation & Model Embedding Modes

Evaluated on held-out unseen corpora after fitting `learn_ngrams` on a disjoint training split (64 n-grams learned):

| Held-Out Corpus Domain | Uncompressed Tokens | Compressed Tokens | Compression Ratio | Lossless Roundtrip |
|:---|:---:|:---:|:---:|:---:|
| **Python Code (Held-Out)** | 679 | 259 | **2.62x** | 100% PASS |
| **Mathematical Prose (Held-Out)** | 219 | 83 | **2.64x** | 100% PASS |
| **JSON Structured Config (Held-Out)** | 662 | 248 | **2.67x** | 100% PASS |
| **Repetitive Log & Delimiters (Held-Out)** | 1360 | 512 | **2.66x** | 100% PASS |

**Aggregate Held-Out Compression Ratio**: **`2.65x`**

- **Embedding Mode Verification** (distinct from text reconstruction parity): `mean_pool` (averages constituent embeddings, compressed length=35); `expand` (decompression inside forward, exact logit diff vs base=`0.0`); `extend_vocabulary_for_tokenizer` (added 714 trainable composite rows).

---

## 8. Auxiliary Latent Manifold Probing (JEV)

Probing continuous latent geometry alignment from Reflex (L2) to Cortex (L8):

- **Source Exit**: Reflex (Layer 2)
- **Target Exit**: Cortex (Layer 8)
- **Cosine Similarity**: `0.7541`
- **Geodesic Arc Distance on $\mathbb{S}^{D-1}$**: `0.6697 rad` (38.37°)

---

## 9. Full-Model Weight Quantization Tiers (BF16 vs FP8 E4M3 vs NVFP4 4-Bit)

Evaluated across all `78,257,664` transformer-layer linear parameters (`103` `NVFP4Linear` modules converted, `7` layer-0 `NVFP4StagingBuffer` H2D/D2H transfers verified):

| Precision Tier | Effective Bits/Weight | Layer Weight Footprint | Compression | E2E Forward (`B=2,T=128`) | Output Logit Cosine Sim |
|:---|:---:|:---:|:---:|:---:|:---:|
| **BF16 (16-bit Baseline)** | 16.0 | 149.26 MB | **1.0x** | 10.04 ms | `1.0` |
| **FP8 (8-bit E4M3)** | 8.0 | 74.63 MB | **2.0x** | 63.67 ms | `0.97377` |
| **NVFP4 (4-bit E2M1 + E4M3 Scales)** | 5.0 | 46.65 MB | **3.2x** | 17.33 ms | `0.92187` |

---

## Conclusion & Defensibility

1. **Full-stack end-to-end execution**: 8-layer model with 32k vocabulary head measured across batch and sequence dimensions.
2. **Exact Speculative Parity Semantics**: Verified exact token-for-token parity at `T=0.0` (greedy) and Leviathan et al. (2023) rejection sampling at `T=0.7`.
3. **Default Production Sparse MoE**: `MoE_FFN` enables `use_sparse_eval=True` by default, benchmarked separately from the legacy `MoERoutingBlock` diagnostic.
4. **Lossless Text vs Equivalent Compute Distinction**: Explicitly separates held-out lossless text round-trip from `mean_pool` vs exact `expand` forward semantics.