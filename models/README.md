# Triune Model Zoo & Architecture Presets

This directory contains standard architectural specifications, configurations, and documentation for the **Triune Transformer** model family.

---

## Architectural Highlights

Triune models implement a 3-tier bio-inspired hierarchical architecture:
* **Gated Linear Attention (GLA)**: $O(N)$ linear complexity attention with data-dependent decay gates and Rotary Position Embeddings (RoPE).
* **Sparse Mixture of Experts (MoE)**: Centroid-steered routing with $E$ specialized experts plus a dedicated shared expert active on every token.
* **3-Tier Adaptive Early Exits**:
  * **Reflex Head**: Shallow early exit for low-entropy, instinctive tokens ($20\text{--}30\%$ depth).
  * **Limbic Head**: Intermediate exit for contextual routing and priority evaluations ($60\text{--}70\%$ depth).
  * **Cortex Head**: Full-depth deliberative generation ($100\%$ depth).
  * All intermediate exit heads are normalized with `RMSNorm` to match Cortex variance scale.

---

## Model Family Specifications

| Model | Total Params | Active Params | Layers | Hidden Dim | Heads | Experts (Top-1) | Reflex / Limbic Exit | FP8 Size | Target Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`triune-nano`** | **~110M** | ~65M | 8 | 512 | 4 | 4 + 1 shared | L3 / L6 | ~110 MB | CPU / Laptop GPU (Instant dev & tests) |
| **`triune-small`** | **~750M** | ~350M | 14 | 1024 | 8 | 4 + 1 shared | L5 / L10 | ~750 MB | 6–8 GB GPU (Ultra-fast laptop training) |
| **`triune-2.5b`** | **~2.45B** | ~780M | 18 | 1280 | 10 | 8 + 1 shared | L5 / L13 | ~2.45 GB | 8 GB GPU with Layer Streaming |
| **`triune-7b`** | **~7.2B** | ~2.2B | 32 | 2048 | 16 | 8 + 1 shared | L8 / L22 | ~7.2 GB | Multi-GPU Cloud Node (FSDP2 / torchrun) |

---

## Quick Loading via Python API

```python
import triune

# Load any model by preset name
model = triune.load_model("triune-2.5b")
print(f"Loaded {model.__class__.__name__} with {sum(p.numel() for p in model.parameters()):,} parameters")

# Or load from a specific configuration file
import json
from pathlib import Path
from triune.configs import build_config

config_path = Path("models/configs/triune-2.5b.json")
with open(config_path) as f:
    config_dict = json.load(f)

model = triune.build_model(build_config(config_dict))
```

---

## Training Presets

### Laptop Training (`triune-2.5b` with Layer Streaming & FP8):
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

### Cloud Distributed Training (`triune-7b` on multi-GPU):
```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --model_name triune-7b \
    --use_fp8 \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --seq_len 1024
```
