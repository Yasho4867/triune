# Triune Long-Context Scaling Benchmark Report
**Gold-Standard Context Length & Autoregressive Decoding Scaling Analysis**

- **Date / Timestamp**: `2026-11-04 02:48:28 UTC`
- **Git Commit**: [`7dcb746801`](https://github.com/Yasho4867/triune/commit/7dcb746801c6ef6b4412eb80a74e8eecf487811e) (Dirty Working Tree)
- **Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU` (7.93 GB VRAM, CC 12.0)
- **Software**: PyTorch `2.14.1+cu132` | CUDA `13.2` | Python `3.14.7`
- **Precision**: `torch.bfloat16` Native Blackwell Tensor Cores

---

## 1. Full Training Step Scaling ($T = 512 \to 8192$)

Comparison between standard quadratic attention (**Standard SDPA + PyTorch Cross-Entropy**) and linear recurrent attention (**Triune Fast GLA + Fused Cross-Entropy**). Both models feature identical parameter topologies (384 dim, 8 layers, 6 heads, 32,000 vocab).

| Context Length ($T$) | Stack | Step Latency (Median) | Throughput | MFU (%) | Peak Allocated VRAM | Peak Reserved VRAM | VRAM Savings vs SDPA | Speedup vs SDPA |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | Standard SDPA + PyTorch CE | **13.63 ms** (Fwd: 6.23 ms, Bwd: 7.33 ms) | 37,556 tok/s | 7.5% | **429.9 MB** | 486.0 MB | **0.0%** | 1.0x |
| **512** | **Triune Fast GLA + Fused CE** | **35.24 ms** (Fwd: 13.69 ms, Bwd: 21.47 ms) | 14,528 tok/s | 2.98% | **576.3 MB** | 610.0 MB | **-34.1%** | 0.39x |
| **1024** | Standard SDPA + PyTorch CE | **16.92 ms** (Fwd: 8.28 ms, Bwd: 8.66 ms) | 60,533 tok/s | 12.91% | **843.2 MB** | 1256.0 MB | **0.0%** | 1.0x |
| **1024** | **Triune Fast GLA + Fused CE** | **36.01 ms** (Fwd: 14.13 ms, Bwd: 21.65 ms) | 28,439 tok/s | 6.21% | **822.7 MB** | 1006.0 MB | **2.4%** | 0.47x |
| **2048** | Standard SDPA + PyTorch CE | **26.87 ms** (Fwd: 9.07 ms, Bwd: 17.68 ms) | 76,209 tok/s | 18.3% | **1415.7 MB** | 1988.0 MB | **0.0%** | 1.0x |
| **2048** | **Triune Fast GLA + Fused CE** | **37.25 ms** (Fwd: 14.63 ms, Bwd: 22.52 ms) | 54,983 tok/s | 13.49% | **1309.8 MB** | 1482.0 MB | **7.5%** | 0.72x |
| **4096** | Standard SDPA + PyTorch CE | **62.67 ms** (Fwd: 19.59 ms, Bwd: 42.89 ms) | 65,356 tok/s | 19.22% | **2533.7 MB** | 3440.0 MB | **0.0%** | 1.0x |
| **4096** | **Triune Fast GLA + Fused CE** | **50.31 ms** (Fwd: 17.36 ms, Bwd: 32.26 ms) | 81,416 tok/s | 24.36% | **2258.2 MB** | 2504.0 MB | **10.9%** | 1.25x |
| **8192** | Standard SDPA + PyTorch CE | **167.74 ms** (Fwd: 48.67 ms, Bwd: 118.65 ms) | 48,838 tok/s | 19.63% | **4774.2 MB** | 6294.0 MB | **0.0%** | 1.0x |
| **8192** | **Triune Fast GLA + Fused CE** | **155.45 ms** (Fwd: 97.96 ms, Bwd: 57.48 ms) | 52,700 tok/s | 21.45% | **3440.9 MB** | 3774.0 MB | **27.9%** | 1.08x |

---

## 2. Autoregressive Single-Token Generation Scaling

In production LLM serving, generation latency and memory are dominated by the KV-cache. As prompt length grows:
- **Standard SDPA Attention** must retain all past key/value tokens, causing memory to scale linearly $O(T)$ and decoding latency to degrade.
- **Triune Fast Gated Linear Attention (GLA)** maintains a fixed recurrent hidden state $S \in \mathbb{R}^{H \times d_k \times d_v}$, achieving **strict $O(1)$ memory complexity and constant decoding latency**.

| Prompt Length ($T$) | SDPA Decode Latency | SDPA KV-Cache Memory | Triune GLA Decode Latency | Triune Recurrent State | Decode Speedup | State Memory Saved |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **512 tokens** | 0.0675 ms | 6144.0 KB | **0.1357 ms** | **384.0 KB** | **0.5x** | **93.8%** |
| **1,024 tokens** | 0.0954 ms | 12288.0 KB | **0.1278 ms** | **384.0 KB** | **0.75x** | **96.9%** |
| **2,048 tokens** | 0.1434 ms | 24576.0 KB | **0.1309 ms** | **384.0 KB** | **1.1x** | **98.4%** |
| **4,096 tokens** | 0.2369 ms | 49152.0 KB | **0.1302 ms** | **384.0 KB** | **1.82x** | **99.2%** |
| **8,192 tokens** | 0.4157 ms | 98304.0 KB | **0.1266 ms** | **384.0 KB** | **3.28x** | **99.6%** |
| **16,384 tokens** | 0.7632 ms | 196608.0 KB | **0.1304 ms** | **384.0 KB** | **5.85x** | **99.8%** |

---

## 3. Asymptotic Scaling Analysis & Workload Trade-offs

1. **Architectural Comparison & Crossover Behavior**:
   - This experiment evaluates an architectural comparison between quadratic softmax attention (Standard SDPA) and linear recurrent attention (Triune Gated Linear Attention with Fused Chunk Recurrence).
   - At short sequence lengths ( \le 2048$), SDPA's highly optimized FlashAttention CUDA C++ implementation is faster (e.g. 13.63 ms vs 35.24 ms at =512$) due to lower constant overhead and register saturation.
   - Triune GLA demonstrates linear (T)$ complexity and crosses over around  \approx 4096$ (50.31 ms vs 62.67 ms, **1.25x faster**) and maintains lower step latency at  = 8192$ (155.45 ms vs 167.74 ms, **1.08x faster**) while slashing **1,333 MB (27.9%)** of peak training VRAM.

2. **(1)$ Constant Memory in Autoregressive Generation**:
   - While standard Transformer KV-cache footprints scale proportionally to context length (demanding 196.6 MB at 16,384 tokens), Triune's recurrent state remains exactly **48.0 KB per layer (384.0 KB total across all 8 layers)**.
   - This unlocks dramatic decoding speedups as context expands (**1.82x faster** at 4,096 tokens, **3.28x faster** at 8,192 tokens, and **5.85x faster** at 16,384 tokens) with **99.8% memory savings**.

3. **Logit-Free Loss Optimization**:
   - The integration of Triune's chunked cross-entropy from hidden states (chunked_cross_entropy_from_hidden) eliminates the intermediate [B, T, V] logit tensor, preventing GPU out-of-memory errors at long sequence lengths.
