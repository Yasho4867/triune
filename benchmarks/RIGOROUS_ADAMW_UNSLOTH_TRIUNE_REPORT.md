# Rigorous Competitive Benchmark Report: PyTorch AdamW vs Native Unsloth vs Triune

- **Device**: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM)
- **PyTorch**: 2.14.1+cu132 | **CUDA**: 13.2
- **Unsloth**: 2026.10.10 | **BitsAndBytes**: 0.50.2

## 1. MoE Pre-Training from Scratch

| Stack | Optimizer Details | GPU Step Latency (ms) | Wall-Clock Latency (ms) | GPU Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AdamW** | PyTorch Standard AdamW | **77.59 ms** | **77.71 ms** | **6599.0** | **271.4 MB** | Untrained 7.7004 -> 7.3701 -> 6.1044 (Δ +1.2657) |
| **Unsloth** | bitsandbytes 8-bit AdamW | **94.87 ms** | **95.12 ms** | **5397.0** | **239.8 MB** | Untrained 7.7004 -> 7.3699 -> 6.1044 (Δ +1.2655) |
| **Triune** | Triune CentroidSteer (Triangular Core + Batched Muon) | **120.36 ms** | **120.41 ms** | **4254.0** | **440.7 MB** | Untrained 7.7004 -> 6.4378 -> 4.4372 (Δ +2.0006) |

## 2. Dense Pre-Training

| Stack | Optimizer Details | GPU Step Latency (ms) | Wall-Clock Latency (ms) | GPU Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AdamW** | PyTorch Standard AdamW | **76.27 ms** | **76.61 ms** | **6713.0** | **201.6 MB** | Untrained 7.6472 -> 7.3545 -> 6.1605 (Δ +1.1940) |
| **Unsloth** | bitsandbytes 8-bit AdamW | **72.17 ms** | **72.19 ms** | **7094.0** | **182.4 MB** | Untrained 7.6472 -> 7.3545 -> 6.1578 (Δ +1.1967) |
| **Triune** | Triune CentroidSteer (Triangular Core + Batched Muon) | **90.9 ms** | **90.92 ms** | **5633.0** | **366.1 MB** | Untrained 7.6472 -> 5.7868 -> 3.4878 (Δ +2.2990) |

## 3. LoRA Fine-Tuning

| Stack | Optimizer Details | GPU Step Latency (ms) | Wall-Clock Latency (ms) | GPU Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AdamW** | PyTorch Standard AdamW | **97.15 ms** | **97.16 ms** | **5270.0** | **192.2 MB** | Untrained 7.6766 -> 7.6884 -> 7.6531 (Δ +0.0353) |
| **Unsloth** | bitsandbytes 8-bit AdamW | **101.24 ms** | **101.26 ms** | **5057.0** | **191.8 MB** | Untrained 7.6766 -> 7.6895 -> 7.6529 (Δ +0.0365) |
| **Triune** | CentroidSteerOptimizer (Batched Muon Newton-Schulz) | **110.8 ms** | **110.82 ms** | **4621.0** | **191.9 MB** | Untrained 7.6766 -> 7.6253 -> 6.8149 (Δ +0.8103) |

## 4. Pure Optimizer Microbenchmark (33 2D weight matrices (1536x256 & 256x1536))

| Optimizer | GPU Latency (ms) | Wall-Clock Latency (ms) | Persistent State VRAM (MB) | Peak Step Scratch (MB) | State Reduction vs AdamW |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Standard AdamW** | **1.92 ms** | **0.66 ms** | **48.13 MB** | **24.06 MB** | **Baseline** |
| **Unsloth Native 8-bit AdamW** | **4.24 ms** | **4.27 ms** | **24.6 MB** | **0.0 MB** | **-48.9%** |
| **Triune CentroidSteer (Triangular Core)** | **15.58 ms** | **15.57 ms** | **2.41 MB** | **48.02 MB** | **-95.0%** |
