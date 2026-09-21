# Model Zoo

This directory contains architecture specifications and configuration files for the Triune model family.

## Architecture Overview

Triune models combine:
- **Gated Linear Attention (GLA)**: $O(N)$ linear attention with data-dependent decay gates and RoPE.
- **Sparse Mixture of Experts (MoE)**: Centroid-steered routing with $E$ experts and a dedicated shared expert.
- **Hierarchical Early Exits**:
  - **Reflex Head**: Intermediate exit at early depth (~25% depth).
  - **Limbic Head**: Intermediate exit at mid depth (~65% depth).
  - **Cortex Head**: Full-depth exit at final layer (100% depth).
  - Intermediate exit heads are normalized with RMSNorm to match Cortex representation scale.

## Model Specifications

| Model | Total Params | Active Params | Layers | Hidden Dim | Heads | Experts (Top-1) | Exit Layers | FP8 Weight Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `triune-nano` | 110M | 65M | 8 | 512 | 4 | 4 + 1 shared | L3, L6 | ~110 MB |
| `triune-small` | 750M | 350M | 14 | 1024 | 8 | 4 + 1 shared | L5, L10 | ~750 MB |
| `triune-2.5b` | 2.45B | 780M | 18 | 1280 | 10 | 8 + 1 shared | L5, L13 | ~2.45 GB |
| `triune-7b` | 7.2B | 2.2B | 32 | 2048 | 16 | 8 + 1 shared | L8, L22 | ~7.2 GB |

JSON configurations are located in `models/configs/`.

## Loading Models

```python
import triune

# Load preset
model = triune.load_model("triune-2.5b")
print(f"Loaded {model.__class__.__name__} ({sum(p.numel() for p in model.parameters()):,} parameters)")
```

Or load from a JSON configuration file:

```python
import json
from pathlib import Path
from triune.configs import build_config

with open("models/configs/triune-2.5b.json") as f:
    config_dict = json.load(f)

model = triune.build_model(build_config(config_dict))
```

## Training Examples

### Single-GPU with Layer Streaming (`triune-2.5b`)

```bash
python scripts/train.py \
    --model_name triune-2.5b \
    --use_fp8 \
    --streaming \
    --streaming_fp8_weights \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --seq_len 256
```

### Multi-GPU Distributed (`triune-7b`)

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --model_name triune-7b \
    --use_fp8 \
    --batch_size 8 \
    --grad_accum_steps 4 \
    --seq_len 1024
```
