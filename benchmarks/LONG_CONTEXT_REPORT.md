# Triune Category D: Long-Context Inference & Generation Report

**Date**: November 15, 2026
**Hardware**: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM)
**Software**: PyTorch 2.14.1+cu132, CUDA 13.2, Python 3.14.7

## 1. Executive Summary

Standard Softmax Attention exhibits quadratic $O(L^2)$ computational complexity and linear $O(L)$ KV-cache
memory consumption during autoregressive decoding. Triune's Gated Linear Attention (GLA) replaces quadratic
attention matrices with chunkwise linear recurrent state updates, yielding **$O(1)$ constant decode memory**
and sub-millisecond per-token latency even at context lengths exceeding 16,384 tokens.

## 2. Prefill Latency & VRAM Scaling Sweep

| Context Length ($L$) | Standard SDPA Latency | GLA Latency | SDPA Peak VRAM | GLA Peak VRAM | VRAM Reduction | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **512** | 0.22 ms | 0.32 ms | 42.0 MB | 43.5 MB | -3.6% | **0.69x** |
| **1024** | 0.23 ms | 0.34 ms | 47.5 MB | 46.5 MB | 2.1% | **0.68x** |
| **2048** | 0.42 ms | 0.6 ms | 54.5 MB | 50.69 MB | 7.0% | **0.7x** |
| **4096** | 1.33 ms | 1.02 ms | 68.5 MB | 56.69 MB | 17.2% | **1.3x** |
| **8192** | 4.31 ms | 2.01 ms | 96.5 MB | 68.69 MB | 28.8% | **2.14x** |
| **16384** | 14.97 ms | 3.72 ms | 152.5 MB | 94.69 MB | 37.9% | **4.02x** |

## 3. Autoregressive Decode Scaling ($O(1)$ vs $O(L)$)

| Prompt Context Length | SDPA KV-Cache Memory | GLA Recurrent State Memory | SDPA Decode Latency | GLA Decode Latency | Decode Latency Advantage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **512** | 1024.0 KB | **64.0 KB (Fixed)** | 0.166 ms/tok | **0.151 ms/tok** | **1.1x faster** |
| **2048** | 4096.0 KB | **64.0 KB (Fixed)** | 0.175 ms/tok | **0.147 ms/tok** | **1.18x faster** |
| **8192** | 16384.0 KB | **64.0 KB (Fixed)** | 0.483 ms/tok | **0.152 ms/tok** | **3.19x faster** |
| **16384** | 32768.0 KB | **64.0 KB (Fixed)** | 1.137 ms/tok | **0.275 ms/tok** | **4.13x faster** |

## 4. Key Takeaways

1. **Constant Memory Footprint**:
   - Standard Softmax KV-cache grows linearly with sequence length. At $L=8,192$, the KV-cache consumes megabytes per layer.
   - GLA maintains a fixed recurrent matrix of shape $(H, d_k, d_v)$ (64.0 KB total) regardless of context length.
2. **Decode Latency & Scaling**:
   - With fused Triton single-token recurrent state kernels, GLA decode latency stays bounded (0.147 - 0.275 ms/token across $L=512..16384$) with constant 64.0 KB state, eliminating $O(L)$ KV-cache memory growth.

