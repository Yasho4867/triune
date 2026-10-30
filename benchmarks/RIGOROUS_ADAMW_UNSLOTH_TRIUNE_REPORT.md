# Rigorous Competitive Benchmark Report: PyTorch AdamW vs Native Unsloth vs Triune

- **Device**: NVIDIA GeForce RTX 5070 Laptop GPU (7.93 GB VRAM)
- **PyTorch**: 2.14.1+cu132 | **CUDA**: 13.2
- **Unsloth**: 2026.10.10 | **BitsAndBytes**: 0.50.2

## 1. MoE Pre-Training from Scratch

| Stack | Optimizer Details | Step Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AdamW** | PyTorch Standard AdamW | **158.29 ms** | **3235.0** | **271.3 MB** | Untrained 7.7004 -> 7.3701 -> 6.1057 (Δ +1.2644) |
| **Unsloth** | bitsandbytes 8-bit AdamW | **158.08 ms** | **3239.0** | **240.4 MB** | Untrained 7.7004 -> 7.3702 -> 6.1032 (Δ +1.2670) |
| **Triune** | Triune CentroidSteer (Triangular Core + Batched Muon) | **184.89 ms** | **2769.0** | **221.8 MB** | Untrained 7.7004 -> 6.4305 -> 4.2607 (Δ +2.1698) |

## 2. Dense Pre-Training

| Stack | Optimizer Details | Step Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AdamW** | PyTorch Standard AdamW | **110.51 ms** | **4633.0** | **201.6 MB** | Untrained 7.6472 -> 7.3545 -> 6.1605 (Δ +1.1940) |
| **Unsloth** | bitsandbytes 8-bit AdamW | **113.15 ms** | **4525.0** | **182.4 MB** | Untrained 7.6472 -> 7.3545 -> 6.1578 (Δ +1.1967) |
| **Triune** | Triune CentroidSteer (Triangular Core + Batched Muon) | **150.17 ms** | **3409.0** | **178.0 MB** | Untrained 7.6472 -> 5.7858 -> 4.3815 (Δ +1.4043) |

## 3. LoRA Fine-Tuning

| Stack | Optimizer Details | Step Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AdamW** | PyTorch Standard AdamW | **176.25 ms** | **2905.0** | **192.2 MB** | Untrained 7.6766 -> 7.6884 -> 7.6529 (Δ +0.0355) |
| **Unsloth** | bitsandbytes 8-bit AdamW | **192.09 ms** | **2665.0** | **191.8 MB** | Untrained 7.6766 -> 7.6895 -> 7.6528 (Δ +0.0366) |
| **Triune** | CentroidSteerOptimizer (Batched Muon Newton-Schulz) | **182.53 ms** | **2805.0** | **191.8 MB** | Untrained 7.6766 -> 7.6253 -> 6.8152 (Δ +0.8101) |

## 4. Pure Optimizer Microbenchmark (33 2D weight matrices (1536x256 & 256x1536))

| Optimizer | Step Latency (ms) | State VRAM Footprint (MB) | VRAM Reduction vs AdamW |
| :--- | :---: | :---: | :---: |
| **PyTorch Standard AdamW** | **1.54 ms** | **48.13 MB** | **Baseline** |
| **Unsloth Native 8-bit AdamW** | **6.59 ms** | **24.61 MB** | **-48.9%** |
| **Triune CentroidSteer (Triangular Core)** | **42.5 ms** | **2.29 MB** | **-95.2%** |
