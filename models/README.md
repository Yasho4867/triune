# Lucid Machines — Triune Architecture Presets & Inaugural `Junkie` Configurations (`v1.0.0`)

This directory contains architecture specifications and configuration files for the **Triune architecture by Lucid Machines** (`TriuneTransformer` / `TriuneModel` / `TriuneConfig` inside the `lucid_machines` package, plus `UniversalTransformer`), including the 12 `triune-*` architecture scale templates (`models/configs/triune-*.json`) and the dedicated configurations for **Junkie** (`models/configs/junkie-mini.json`, `junkie-1b.json`, `junkie-3b.json`, `junkie-7b.json`, and `junkie.json`), Lucid Machines' inaugural flagship model to be trained on the Triune architecture.

## Architecture Overview

`TriuneTransformer` (the Triune neural architecture by **Lucid Machines**) combines:
- **Gated Linear Attention (`VectorisedGLA`) & Gated DeltaNet**: $O(N)$ linear attention with data-dependent decay gates, fused QKVG projections, RoPE, and constant $O(1)$ recurrent decode memory (`48.0 KiB` isolated kernel at $H=6$; `64.0 KiB` multi-layer up to $L=16,384$).
- **Sparse Mixture of Experts (`MoE_FFN`)**: Default sparse scatter-gather evaluation (`use_sparse_eval=True`, `1.648x` single-token zero-sync speedup and `7.518x` batch prefill speedup at `d=384, E=8, k=2, N=4096`) with a dedicated shared expert, multi-directional contrastive input statistics (`last_centroids`, `[E, D]`), and true post-SwiGLU hidden centroids (`last_hidden_centroids`, `[E, H]`) for **CentroidSteer** down-projection steering.
- **3-Tier Hierarchical Early Exits**:
  - **Reflex Head**: Intermediate exit at early depth (~25% depth).
  - **Limbic Head**: Intermediate exit at mid depth (~65% depth).
  - **Cortex Head**: Full-depth exit at final layer (100% depth).
  - Intermediate exit heads are normalized with parameter-free `RMSNorm` to match Cortex representation scale and calibrated via conformal CALM margin gating (`exit_policy="calm"`).

## Inaugural Flagship Model To Be Trained (`models/configs/junkie*.json`)

| Config File | Model Name | Total Params (Exact `meta`) | Active Params (Reflex / Limbic / Cortex / Expected) | Layers | Hidden Dim | Heads | Experts | Exit Layers |
| :--- | :--- | ---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `models/configs/junkie-mini.json` | **`Junkie-Mini`** (`junkie-mini`) | **535,101,975** (`535.10M`) | `161.30M` / `252.59M` / `280.65M` (**230.81M** expected) | 12 | 1024 | 8 | 4 + 1 shared | L6, L9 |
| `models/configs/junkie-1b.json` | **`Junkie-1B`** (`junkie-1b`) | **1,162,535,483** (`1.16B`) | `174.94M` / `387.95M` / `568.15M` (**374.99M** expected) | 22 | 1024 | 8 | 4 + 1 shared | L7, L14 |
| `models/configs/junkie-3b.json` | **`Junkie-3B`** (`junkie-3b`) | **2,990,355,155** (`2.99B`) | `106.75M` / `198.07M` / `469.69M` (**395.74M** MoE-trunk / **206.73M** dynamic) | 16 | 1024 | 8 | 16 + 1 shared | L2, L5 |
| `models/configs/junkie-7b.json` & `junkie.json` | **`Junkie-7B` / `Junkie`** | **7,396,845,163** (`7.40B`) | `1.17B` / `2.02B` / `2.68B` (**1.95B** expected) | 32 | 2048 | 16 | 8 + 1 shared | L18, L25 |

## Triune Architecture Size Templates (`models/configs/triune-*.json`)

| Template Preset (`v1.0.0`) | Legacy Alias | Layers | Hidden Dim | Heads | Experts | Exit Layers | Target Tier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `triune-tiny` | `triune-micro` | 4 | 192 | 3 | 2 + 1 shared | L2, L3 | Ultra-Light Edge / CI |
| `triune-edge` | — | 6 | 256 | 4 | 4 + 1 shared | L2, L4 | Compact Edge / Laptop |
| `triune-nano` | — | 8 | 512 | 4 | 4 + 1 shared | L3, L6 | Fast Local Prototyping |
| `triune-base` | — | 8 | 384 | 6 | 4 + 1 shared | L2, L5 | Consumer GPU Pre-Training |
| `triune-small` | — | 14 | 1024 | 8 | 4 + 1 shared | L5, L10 | Workstation Baseline |
| `triune-pro` | — | 16 | 768 | 12 | 8 + 1 shared | L5, L10 | High-Capacity Workstation |
| `triune-1b` | `triune-1.5b` | 24 | 1024 | 16 | 8 + 1 shared | L7, L16 | 1.5B Layer-Streamable MoE |
| `triune-2.5b` | — | 18 | 1280 | 10 | 8 + 1 shared | L5, L13 | 2.45B Multi-Exit MoE |
| `triune-large` | — | 28 | 1536 | 12 | 8 + 1 shared | L8, L19 | Deep Workstation MoE |
| `triune-moe` | — | 24 | 1280 | 10 | 16 + 1 shared | L6, L16 | 16-Expert Sparse MoE |
| `triune-7b` | — | 32 | 2048 | 16 | 8 + 1 shared | L8, L22 | 7.2B Flagship MoE |
| `triune-frontier` | `triune-32b` | 48 | 2560 | 20 | 16 + 1 shared | L11, L30 | Frontier Sparse Scale |

JSON configurations live in `models/configs/` (`triune-*.json` + `junkie.json`, all with `"architecture": "TriuneTransformer"` and `"model_type": "triune"`).

> **Evaluated 90.79M Baseline Checkpoint**: All current end-to-end benchmarks evaluate the existing `90.79M` checkpoint (`checkpoints/triune_studio_step_36500_rescued.pt`, `step_40555`, **90,789,271** total / **64,206,871** active parameters per token), the 5-seed FineWeb-Edu cold-start LM benchmark (`Suite 3`, `0.264 B/p`, `2.2012` vs. `2.2472` `nosteer`, `p = 0.0112`, and `2.2585` `random_same_mem`, `p = 0.0006`), and the `75.8M` FineWeb-Edu continuation benchmark (`Suite 4`, `0.201 B/p`, `4.9934` vs. `4.9945` `nosteer`, `p = 0.0004`, `4.9945` `random_same_mem`, `p = 0.0005`, and `4.9954` `adamw`, `p = 0.0038`). **Junkie** (`models/configs/junkie*.json`) is Lucid Machines' inaugural flagship model prepared for training on the Triune architecture.

## Loading Models

```python
import lucid_machines  # (or `import triune` via compatibility shim)

# Load a Triune architecture template ("triune-2.5b") or first flagship target ("junkie")
model = lucid_machines.load_model("triune-2.5b")
print(f"Loaded {model.__class__.__name__} ({sum(p.numel() for p in model.parameters()):,} parameters)")
```

Or load directly from a JSON configuration file (including `junkie.json`):

```python
import json

with open("models/configs/junkie.json") as f:
    config_dict = json.load(f)
```
