# Triune Category C: Memory-Constrained Training & Layer Streaming Report

**Date**: November 15, 2026
**Hardware**: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM)
**Software**: PyTorch 2.14.1+cu132, CUDA 13.2, Python 3.14.7
**Workload**: Batch Size 2, Sequence Length 512, Vocab Size 4096, Precision BFloat16

## 1. Summary of Findings

The Triune `LayerStreamingEngine` decouples model parameter capacity from physical GPU VRAM by virtualizing transformer layer storage:
- Layer weights reside in host CPU RAM and are dynamically staged into GPU VRAM layer-by-layer during forward and backward passes.
- CUDA prefetching overlaps host-to-device transfers with compute on preceding layers.
- Immediate gradient offloading frees GPU VRAM as soon as autograd computes each layer's parameter derivatives.

## 2. Empirical Benchmark Results

| Model Scale | Total Parameters | Streaming Optimizer | Standard GPU Peak VRAM | LayerStreaming Peak VRAM | VRAM Reduction | Standard Latency | Streaming Latency | Throughput (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **102M** | 101.97M | AdamW (BF16) | 4384.9 MB | 1033.64 MB | 76.43% | 151.55 ms | 773.61 ms | 1323.7 tok/s |
| **271M** | 270.56M | AdamW (BF16) | 9224.72 MB | 1498.12 MB | 83.76% | 3111.6 ms | 1576.49 ms | 649.5 tok/s |
| **695M** | 694.85M | AdamW (BF16) | OOM (>8 GB) | 2115.73 MB | **OOM Prevented** | N/A | 4026.89 ms | 254.3 tok/s |
| **1.5B** | 1522.76M | CentroidSteerOptimizer (r=16, 8-bit base) | OOM (>8 GB) | 3132.62 MB | **OOM Prevented** | N/A | 5033.42 ms | 203.4 tok/s |

## 3. Analysis: Memory Footprint vs. PCIe Bandwidth Trade-off

1. **Peak Memory Floor**:
   - The `LayerStreamingEngine` keeps active layer parameter VRAM bounded by the size of the largest individual layer chunk rather than the aggregate model depth.
   - For large models (695M–1.5B+), layer streaming enables stable training within consumer VRAM envelopes (8 GB) where standard GPU-resident models experience severe memory pressure or OOM, and pairing with `CentroidSteerOptimizer` keeps host-side optimizer memory bounded within a 16 GB RAM budget.
2. **Throughput & PCIe Transfer Overhead**:
   - Because layer weights must traverse the PCIe bus twice per step (forward pass H2D and backward pass H2D + gradient D2H), step latency increases proportionally to PCIe bandwidth limits.
   - Asynchronous prefetching via secondary CUDA streams overlaps transfer with compute when compute intensity per layer is sufficient.

