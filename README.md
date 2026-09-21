# Triune

Triune is a PyTorch framework for training and serving multi-exit Mixture-of-Experts (MoE) language models.

## Key Components

- **Gated Linear Attention (GLA)**: $O(N)$ linear-time attention with data-dependent decay gates and Rotary Position Embeddings (RoPE).
- **Sparse Mixture of Experts**: Top-1 expert routing with an always-active shared expert and activation centroid tracking.
- **Hierarchical Exit Heads**: Intermediate exit heads (Reflex, Limbic) and full-depth exit (Cortex), normalized with RMSNorm to ensure uniform representation scale across depths.
- **3-Tier Partitioned Optimizer**:
  - **Muon**: 5th-order Newton-Schulz orthogonalization for 2D attention and shared linear projections.
  - **CentroidSteer**: Dual-sided low-rank SVD projections with semantic centroid steering for routed MoE experts.
  - **AdamW**: Standard adaptive moments for 1D vectors, norms, embeddings, and router heads.
- **Layer Streaming Runtime**: On-demand layer transfer and optimizer state migration between host RAM and GPU VRAM, enabling training of 2.5B+ models under 350 MB peak VRAM.
- **Native FP8 Execution**: Hardware-accelerated `float8_e4m3fn` scaled matrix multiplication with dynamic quantization.
- **Triune Studio**: Web and desktop interface for monitoring training, interactive inference, and workflow management.

## Repository Structure

```text
triune/                 Python package (model, optim, runtime, trainer, data, api)
models/                 Architecture configurations (nano, small, 2.5b, 7b)
studio/                 Triune Studio frontend and desktop launcher
scripts/                Training, chat, and utility scripts
tests/                  Unit and integration tests
docs/                   Mathematical foundations and technical guides
```

## Installation

```bash
git clone https://github.com/Yasho4867/triune.git
cd triune

# Core framework
pip install -e .

# With Studio desktop dependencies
pip install -e ".[studio]"

# Full installation (training, dev, studio)
pip install -e ".[all]"
```

## Quickstart

### Python API

```python
import torch
import triune

# Load model preset
model = triune.load_model("triune-2.5b")

# Forward pass returning all exit head logits
input_ids = torch.randint(0, 32000, (1, 128))
reflex_logits, limbic_logits, cortex_logits, route_logits = model.forward_all_exits(input_ids)

# Memory estimation
config = triune.build_config({"num_layers": 18, "hidden_dim": 1280, "num_experts": 8})
plan = triune.MemoryPlanner.estimate_vram(config, target_vram_gb=8.0)
print(f"Estimated parameters: {plan.param_memory_gb:.2f} GB")
```

### CLI

```bash
# Launch Triune Studio
triune studio --port 8000

# Start API server
triune serve --port 8000

# Interactive chat
triune chat --model triune-2.5b

# Check memory requirements
triune plan-memory --vram-gb 8.0
```

## Model Presets

| Model | Total Params | Active Params | Layers | Hidden Dim | Heads | Experts | Exit Layers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `triune-nano` | 110M | 65M | 8 | 512 | 4 | 4 + 1 shared | L3, L6 |
| `triune-small` | 750M | 350M | 14 | 1024 | 8 | 4 + 1 shared | L5, L10 |
| `triune-2.5b` | 2.45B | 780M | 18 | 1280 | 10 | 8 + 1 shared | L5, L13 |
| `triune-7b` | 7.2B | 2.2B | 32 | 2048 | 16 | 8 + 1 shared | L8, L22 |

Configuration files are located in `models/configs/`.

## Training

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

## Testing

```bash
pytest tests/
```

Individual test suites:
```bash
python tests/test_framework.py
python tests/test_streaming_engine.py
python tests/test_muon_and_adaptive_steer.py
python tests/test_resource_manager.py
python tests/test_features.py
python tests/test_ecosystem.py
python tests/test_integration.py
```

## Documentation

- [Mathematical Foundations](docs/MATH_FOUNDATIONS.md): Formulations and proofs for SVD subspace projections, centroid steering, Muon orthogonalization, and exit normalization.
- [Framework Guide](docs/FRAMEWORK.md): Subsystems, runtime layer streaming, and optimizer architecture.
- [Architecture Guide](docs/DYNAMIC_MODULAR_GUIDE.md): Configuration options, parameter constraints, and memory planning.
- [Roadmap](docs/ROADMAP.md): Project milestones and targets.

## License

Apache 2.0
