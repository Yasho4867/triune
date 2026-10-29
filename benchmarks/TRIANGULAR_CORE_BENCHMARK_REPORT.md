# Universal Triangular Core Benchmark Report

- **Device**: NVIDIA GeForce RTX 5070 Laptop GPU
- **PyTorch**: 2.14.1+cu132
- **CUDA**: 13.2

## 1. Pure MoE Expert Optimizer Step (32 Matrices Microbenchmark)

| Optimizer | Median Latency (ms) | State VRAM (MB) |
| :--- | :---: | :---: |
| PyTorch Standard AdamW | **1.77 ms** | **60.05 MB** |
| CentroidSteer (Legacy BMM) | **55.37 ms** | **12.82 MB** |
| CentroidSteer (Universal Triangular Core) | **26.5 ms** | **11.08 MB** |

## 2. End-to-End Training Step (Forward + Backward + Step)

| Optimizer | Step Latency (ms) | Peak VRAM (MB) | Loss (0 -> 20) | Final Grad Norm |
| :--- | :---: | :---: | :---: | :---: |
| PyTorch Standard AdamW | **179.52 ms** | **279.8 MB** | 7.25 -> 6.3125 | 1.7106 |
| CentroidSteer (Legacy BMM) | **293.94 ms** | **245.0 MB** | 5.7188 -> 5.0 | 1.2064 |
| CentroidSteer (Universal Triangular Core) | **291.81 ms** | **243.8 MB** | 5.7188 -> 5.0 | 1.2052 |
