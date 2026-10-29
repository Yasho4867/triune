# Universal Triangular Core Benchmark Report

- **Device**: NVIDIA GeForce RTX 5070 Laptop GPU
- **PyTorch**: 2.14.1+cu132
- **CUDA**: 13.2

## 1. Pure MoE Expert Optimizer Step (32 Matrices Microbenchmark)

| Optimizer | Median Latency (ms) | State VRAM (MB) |
| :--- | :---: | :---: |
| PyTorch Standard AdamW | **1.31 ms** | **45.12 MB** |
| CentroidSteer (Legacy BMM) | **39.65 ms** | **11.56 MB** |
| CentroidSteer (Universal Triangular Core) | **38.71 ms** | **10.33 MB** |

## 2. End-to-End Training Step (Forward + Backward + Step)

| Optimizer | Step Latency (ms) | Peak VRAM (MB) | Loss (0 -> 20) | Final Grad Norm |
| :--- | :---: | :---: | :---: | :---: |
| PyTorch Standard AdamW | **188.4 ms** | **247.4 MB** | 7.25 -> 6.3125 | 1.6849 |
| CentroidSteer (Legacy BMM) | **298.46 ms** | **218.0 MB** | 5.75 -> 4.9688 | 1.2115 |
| CentroidSteer (Universal Triangular Core) | **325.7 ms** | **216.7 MB** | 5.75 -> 4.9688 | 1.212 |
