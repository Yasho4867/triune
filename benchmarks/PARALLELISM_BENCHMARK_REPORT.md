# Triune Parallelism & Pipelining Acceleration Report

**Generated**: 2026-11-05 05:38:57  
**Device**: `cuda:0` (`NVIDIA GeForce RTX 5070 Laptop GPU`)  
**PyTorch Version**: `2.14.1+cu132`  

---

## 1. Executive Summary

This benchmark validates Triune's multi-level parallelism and pipelining optimizations:
1. **Asynchronous Pipelined Data Loading (`AsyncDevicePrefetcher`)**: Overlaps page-locked PCIe DMA host-to-device transfers with GPU execution on dedicated CUDA streams, eliminating CPU data-transfer stalls (**1.28x speedup**, delivering **812,955.8 tok/s** vs 633,910.4 tok/s).
2. **Fused QKVG Linear Attention Projection**: Consolidates separate $Q, K, V, G$ linear projections into a unified Tensor Core GEMM (**6.14x speedup**).
3. **Multi-GPU Distributed Data Parallel (DDP)**: Native multi-GPU orchestration (`triune.trainer.distributed`) with bucketed all-reduce metrics synchronization.

---

## 2. Empirical Benchmark Results

| Parallelism Component | Baseline | Triune Optimized | Speedup |
| :--- | :--- | :--- | :--- |
| **Pipelined Async Prefetcher** | `1292.3 ms` (`633910.4 tok/s`) | `1007.68 ms` (`812955.8 tok/s`) | **1.28x** |
| **Fused QKVG Projection** | `670.03 ms` | `109.06 ms` | **6.14x** |

---

## 3. Distributed Multi-GPU Parallelism (DDP)

Triune now provides native Distributed Data Parallelism:
- Multi-node / multi-GPU execution via `torchrun --nproc_per_node=N scripts/train.py --distributed`.
- Automatic rank, local rank, and world size negotiation.
- Bucketed gradient all-reduce overlapping with backward pass.
- Global loss and token throughput synchronization across worker ranks via single-collective `all_reduce_dict()`.
