# 🧠 Triune Transformer: Hierarchical Multi-Exit MoE Architecture

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-GLA%20%2B%20MoE%20%2B%20Triune_Exits-purple.svg)]()
[![Precision](https://img.shields.io/badge/Precision-Native_FP8_GEMM_%7C_BF16-green.svg)]()

**Triune Transformer** is an open-source, high-performance deep learning framework and language model architecture inspired by the evolutionary **Triune Brain theory** (Paul D. MacLean). It couples **Gated Linear Attention (GLA)**, **Centroid-Steered Sparse Mixture-of-Experts (MoE)**, a **3-Tier Muon/Centroid/AdamW optimizer**, and **adaptive dynamic early-exit heads** (*Reflex*, *Limbic*, and *Cortex*).

With built-in **AirLLM-style Layer Streaming** and native hardware **FP8 (E4M3) scaled GEMM**, Triune enables pretraining and fine-tuning multi-billion parameter architectures (e.g. 2.5B–5B) directly on **8 GB consumer laptop GPUs** under a **$<300\text{ MB}$ peak VRAM** footprint, while seamlessly scaling to distributed multi-GPU cloud clusters.

---

## 🏛️ Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │        Input Tokens & Embeddings       │
                      └──────────────────┬─────────────────────┘
                                         │
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Layer 0 .. Layer 3: Router Prefix (Dense GLA Representations)                   │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Layer 3 .. Layer 5: Intermediate Feature Processing                           │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                    [ Reflex Early Exit Head (Layer 5) ] ──► System 1: Low-entropy fast tokens
                                         │                   (Normalized via RMSNorm)
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Layers 6 .. 13: MoE Specialized Subnetworks                                    │
 │  • 8 Routed Experts (Top-1 Routing)                                            │
 │  • 1 Dedicated Shared Expert (Active on every token)                            │
 │  • Centroid-Steered Activation Manifold Alignment                              │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                    [ Limbic Early Exit Head (Layer 13) ] ──► System 1.5: Contextual triage
                                         │                    (Normalized via RMSNorm)
 ┌───────────────────────────────────────▼────────────────────────────────────────┐
 │ Layers 14 .. 18+: Full-Depth Cortex Stack                                      │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                    [ Cortex Final Head (Layer 18/24/32) ] ──► System 2: Deliberative reasoning
```

### Core Algorithmic Innovations
1. **Bio-Inspired 3-Tier Exit Routing**:
   - **Reflex Head (Layer 5/6)**: Instant instinctive token predictions.
   - **Limbic Head (Layer 13/16)**: Intermediate contextual routing and complexity triage.
   - **Cortex Head (Layer 18/24/32)**: Full-depth deliberative language generation.
   - **Variance-Matched Exit Normalization**: Intermediate heads are stabilized with dedicated `RMSNorm` layers, eliminating the variance explosion characteristic of early exit heads.
2. **Gated Linear Attention (GLA)**:
   - $O(N)$ linear complexity attention with data-dependent decay gates and Rotary Position Embeddings (RoPE), overcoming $O(N^2)$ quadratic context bottlenecks.
3. **Centroid-Steered MoE with Shared Expert**:
   - Top-1 expert gating with an isolated shared expert for canonical language syntax, combined with activation centroid tracking to steer expert representations without representation collapse.
4. **3-Tier Parameter Partitioning & Muon Optimizer**:
   - **Tier 1 (Muon)**: 5th-order Newton-Schulz orthogonalization for all 2D attention and shared linear projections.
   - **Tier 2 (CentroidSteer)**: Dual-sided low-rank SVD projections with semantic subspace steering for MoE expert weights.
   - **Tier 3 (AdamW)**: Standard adaptive moment tracking for 1D vectors (RMSNorm, bias, embedding, router heads).
5. **AirLLM-Style Memory Virtualization (Layer Streaming)**:
   - Compresses host weights to FP8 in CPU RAM and streams individual layers to GPU on-demand via secondary CUDA prefetch streams, migrating optimizer states layer-by-layer to maintain a $<300\text{ MB}$ peak VRAM ceiling.

---

## 📦 Repository Structure

The Triune repository is divided into three primary components:

* **[`triune/`](triune/)**: The core Python framework engine (`model`, `optim`, `runtime`, `trainer`, `data`, `api`).
* **[`models/`](models/)**: The Model Zoo, preset architecture specifications (`triune-nano`, `triune-small`, `triune-2.5b`, `triune-7b`), and model cards.
* **[`studio/`](studio/)**: Triune Studio—the visual IDE, desktop launcher, and node-based pipeline editor.

---

## ⚡ Quickstart

### 1. Installation

Clone the repository and install the framework in editable mode:

```bash
git clone https://github.com/Yasho4867/triune.git
cd triune

# Install core framework
pip install -e .

# Or install with Studio IDE dependencies
pip install -e ".[studio]"

# Or install with full developer & training suite
pip install -e ".[all]"
```

### 2. Python API Usage

```python
import triune
import torch

# 1. Load any model preset from the Model Zoo
model = triune.load_model("triune-2.5b").cuda()
print(f"Loaded Triune 2.5B ({sum(p.numel() for p in model.parameters()):,} parameters)")

# 2. Forward pass with all exit heads
input_ids = torch.randint(0, 32000, (1, 128)).cuda()
reflex_logits, limbic_logits, cortex_logits, route_logits = model.forward_all_exits(input_ids)

print("Reflex logits shape:", reflex_logits.shape)
print("Cortex logits shape:", cortex_logits.shape)

# 3. Dynamic Memory Planning
config = triune.build_config({"num_layers": 18, "hidden_dim": 1280, "num_experts": 8})
plan = triune.MemoryPlanner.estimate_vram(config, target_vram_gb=8.0)
print(f"Projected Footprint: {plan.param_memory_gb} GB | Rec. Batch Size: {plan.recommended_batch_size}")
```

---

## 🎯 Model Zoo & Hardware Matrix

| Preset | Parameters (Total) | Active Params | Layers | Hidden Dim | Experts | FP8 Footprint | Recommended Hardware | Execution Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`triune-nano`** | **~110M** | ~65M | 8 | 512 | 4 | ~110 MB | CPU / Any GPU | Standard In-Memory |
| **`triune-small`** | **~750M** | ~350M | 14 | 1024 | 4 | ~750 MB | 6–8 GB Laptop GPU | Standard In-Memory |
| **`triune-2.5b`** | **~2.45B** | ~780M | 18 | 1280 | 8 | ~2.45 GB | 8 GB Laptop GPU | **Layer Streaming (<300 MB VRAM)** |
| **`triune-7b`** | **~7.2B** | ~2.2B | 32 | 2048 | 8 | ~7.2 GB | Multi-GPU Node (8x H100) | **Distributed FSDP2 / torchrun** |

---

## 🚀 Training Workflows

### Laptop Pretraining (`triune-2.5b` on 8GB GPU):
```bash
python scripts/train.py \
    --model_name triune-2.5b \
    --use_fp8 \
    --streaming \
    --streaming_fp8_weights \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --seq_len 256 \
    --force
```

### Cloud Multi-GPU Pretraining (`triune-7b` on Cluster):
```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --model_name triune-7b \
    --use_fp8 \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --seq_len 1024
```

---

## 🖥️ Triune Studio (Visual IDE & GUI)

Triune Studio provides a native visual development environment featuring an interactive model playground, real-time VRAM telemetry, training loss monitors, and a node-based pipeline builder.

### Launching the Studio:
```bash
# Method 1: Using the unified CLI
triune studio --port 8000

# Method 2: Using the Python launcher script
python scripts/launch_studio.py

# Method 3: Windows 1-click launcher
studio\TriuneStudio.bat
```

---

## 🧪 Verification & Test Suite

Run the full automated test suite:

```bash
# Run core framework forward/backward tests
python tests/test_framework.py

# Run layer streaming memory engine tests
python tests/test_streaming_engine.py

# Run 3-tier Muon and CentroidSteer optimizer tests
python tests/test_muon_and_adaptive_steer.py

# Run hardware feasibility probe & resource manager tests
python tests/test_resource_manager.py

# Run plugin registry and dynamic node tests
python tests/test_features.py
```

---

## 📚 Technical Documentation

* 📐 **[Mathematical Foundations](docs/MATH_FOUNDATIONS.md)**: Proofs for Symmetrical Centroid-Steered GaLore, Newton-Schulz 5th-order orthogonalization for Muon, and Exit Head RMSNorm variance matching.
* 🛠️ **[Operational Framework Guide](docs/FRAMEWORK.md)**: Layer Streaming Engine, 3-Tier Parameter Partitioning, API server, and sandbox details.
* 📘 **[Dynamic Architecture Guide](docs/DYNAMIC_MODULAR_GUIDE.md)**: Customizing layer depth, MoE routing, head dimensions, and VRAM budgeting.
* 🗺️ **[Development Roadmap](docs/ROADMAP.md)**: Project phases, completed breakthroughs, and upcoming releases.

---

## 📄 License

This project is licensed under the Apache 2.0 License.
