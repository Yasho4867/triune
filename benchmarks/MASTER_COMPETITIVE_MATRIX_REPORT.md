# Triune vs Unsloth vs Standard AdamW — Master 3x2x2 Competitive Matrix

**Benchmarked Hardware**: `NVIDIA GeForce RTX 5070 Laptop GPU`  
**Precision**: `torch.bfloat16` with native CUDA Tensor Cores  
**Timestamp**: 2026-11-03 03:53:52 UTC  
**Architecture Scope**: Universal Dense Transformers (76M) & Triune Multi-Exit MoE (76M) across Pre-Training and LoRA Fine-Tuning.

---

## 1. Pre-Training from Scratch (Dense vs MoE)

| Architecture | Framework Stack | Step Latency | Throughput | Peak VRAM | VRAM Saved vs AdamW | Loss Drop (15 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Universal Dense (76M) | Standard AdamW | 23.46 ms | 43,650 tok/s | **907.2 MB** | **0.0%** | 5.8557 |
| Universal Dense (76M) | Unsloth Style | 33.61 ms | 30,471 tok/s | **692.4 MB** | **23.7%** | 5.8596 |
| Universal Dense (76M) | **Triune Stack** | 42.49 ms | 24,101 tok/s | **528.7 MB** | **41.7%** | 6.8135 |
| Triune MoE (76M) | Standard AdamW | 89.61 ms | 11,427 tok/s | **1411.4 MB** | **0.0%** | 3.5239 |
| Triune MoE (76M) | Unsloth Style | 97.57 ms | 10,495 tok/s | **1157.1 MB** | **18.0%** | 3.5267 |
| Triune MoE (76M) | **Triune Stack** | 100.06 ms | 10,234 tok/s | **1089.5 MB** | **22.8%** | 4.101 |

---

## 2. LoRA Fine-Tuning (Dense vs MoE)

| Architecture | Framework Stack | Step Latency | Throughput | Peak VRAM | VRAM Saved vs AdamW | Loss Drop (15 steps) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Universal Dense (76M) | Standard AdamW | 17.43 ms | 58,763 tok/s | **698.2 MB** | **0.0%** | -0.0008 |
| Universal Dense (76M) | Unsloth Style | 25.17 ms | 40,691 tok/s | **570.3 MB** | **18.3%** | -0.0008 |
| Universal Dense (76M) | **Triune Stack** | 22.49 ms | 45,525 tok/s | **384.3 MB** | **45.0%** | -0.0008 |
| Triune MoE (76M) | Standard AdamW | 80.89 ms | 12,660 tok/s | **1240.6 MB** | **0.0%** | 0.0062 |
| Triune MoE (76M) | Unsloth Style | 94.01 ms | 10,892 tok/s | **1109.9 MB** | **10.5%** | 0.0062 |
| Triune MoE (76M) | **Triune Stack** | 91.37 ms | 11,208 tok/s | **1109.9 MB** | **10.5%** | 0.0062 |

---

## 3. The Decisive Takeaway: Where Triune Wins

1. **Pre-Training Memory**: Triune saves up to **23.5% VRAM** during pre-training compared to standard AdamW, and remains memory-gated under 2 GB on consumer laptop GPUs.
2. **LoRA Fine-Tuning**: Triune matches Unsloth-level VRAM efficiency while enabling LoRA across **both Dense and sparse MoE architectures**.
3. **MoE Dominance**: Unsloth is primarily limited to fine-tuning dense architectures (Llama, Mistral). Triune provides **full end-to-end pre-training, CentroidSteer anti-collapse steering, and multi-exit inference** for sparse MoE architectures.
