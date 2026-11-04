# Triune AI — CUDA Graph vs Eager Execution & Optimizer Parity Report

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU` (8,123.8 MB VRAM)  
**Timestamp**: 2026-11-03 23:52:45 UTC  
**Git Provenance Commit**: `3fd3612ef803a0dc8f1376bac7856bf3c720136d` (Working-Tree Clean: `True`)  
**Script SHA-256**: `2ed3f7649996257a7f6a2efb19ce91019b474645b0dfbb8884621d881b12feb0`  
**Dataset**: `data\fineweb_sample.jsonl` (Mode: `fineweb_real`, SHA-256: `8acbf30894156292e921fe4c6e11f634d0042b42f4150968c4d3023863a777c4`)  
**Model Architecture**: `UniversalDenseTransformer` (37,165,440 scalar parameters | 8 layers, 384 dim, 6 heads, 2 KV heads, 1024 MLP dim)  
**Workload**: Batch Size = 4, Seq Len = 256, Steps = 15, Vocab = 32000

---

## 1. Executive Summary & Host-Device Separation

CUDA Graph execution eliminates the Python interpreter overhead and kernel launch latency by baking static execution DAGs directly into GPU silicon command buffers.

| Execution Mode | Host Launch Latency (CPU) | GPU Execution Time (Silicon) | Total Step Latency (Wall Clock) | Host Dispatch Speedup vs Eager |
| :--- | :---: | :---: | :---: | :---: |
| **Standard PyTorch Eager** | 18278.5 $\mu$s | 22.85 ms | 22.90 ms | Baseline (0%) |
| **CUDA Graph (Host Sync via `.item()`)** | 14362.0 $\mu$s | 14.32 ms | 14.37 ms | Blocks host on GPU |
| **CUDA Graph (Asynchronous Replay)** | **188.3 $\mu$s** | **14.29 ms** | **14.33 ms** | **-99.0% launch latency** |

> [!IMPORTANT]
> **Host Synchronization Elimination**:
> When `sync=False` (default in Triune's `CUDAGraphEngine`), `step()` executes a single non-blocking hardware pulse `graph.replay()` and returns the GPU-resident scalar loss tensor. The host CPU returns in **188.3 $\mu$s** without waiting for GPU execution, completely decoupling host dispatch from device completion.

---

## 2. Multi-Step Mathematical Optimizer Parity

To verify that CUDA Graph replay is mathematically identical to ordinary training over multiple steps, two identical models were initialized with identical seeds and trained over 10 consecutive steps with real causal FineWeb batches.

| Step | Eager Loss | CUDA Graph Loss | Loss Delta | Max Parameter Weight Drift | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4.84080 | 4.84080 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 2 | 4.28449 | 4.28449 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 3 | 3.49092 | 3.49092 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 4 | 2.14266 | 2.14266 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 5 | 1.67813 | 1.67813 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 6 | 1.39022 | 1.39022 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 7 | 0.96975 | 0.96975 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 8 | 0.85226 | 0.85226 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 9 | 0.74333 | 0.74333 | 0.00e+00 | 0.00e+00 | **PASSED** |
| 10 | 0.60012 | 0.60012 | 0.00e+00 | 0.00e+00 | **PASSED** |

- **Maximum Observed Weight Drift**: `0.00e+00`
- **Maximum Observed Loss Delta**: `0.00e+00`
- **Parity Verdict**: **100% Mathematically Equivalent**. In-place gradient zeroing (`set_to_none=False`) and static optimizer state buffers ensure bitwise parity with eager PyTorch updates.
