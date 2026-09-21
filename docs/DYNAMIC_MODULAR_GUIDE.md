# Architecture Configuration Guide

Triune models are configured via parameter dictionaries or JSON files. This guide details architectural parameters, structural invariants, and hardware sizing.

---

## 1. Model Instantiation

```python
from triune.model import TriuneTransformer

model = TriuneTransformer(
    vocab_size=32000,
    hidden_dim=1280,            # Must equal num_heads * head_dim
    num_layers=18,
    num_heads=10,
    head_dim=128,
    num_experts=8,              # MoE experts per routed layer
    router_prefix_layers=3,     # Prefix layers before routing decision
    reflex_exit_layer=5,        # Intermediate exit 1
    limbic_exit_layer=13,       # Intermediate exit 2
    use_fp8=True                # FP8 scaled GEMM execution
)
```

### Structural Invariants

1. **Hidden Dimension**:
   $$\text{hidden\_dim} = \text{num\_heads} \times \text{head\_dim}$$

2. **Layer Order**:
   $$\text{router\_prefix\_layers} < \text{reflex\_exit\_layer} < \text{limbic\_exit\_layer} < \text{num\_layers}$$

3. **Feed-Forward Layers**:
   - Layers $0 \le i \le \text{reflex\_exit\_layer}$: Dense feed-forward layers.
   - Layers $i > \text{reflex\_exit\_layer}$: Sparse MoE blocks with `num_experts` and a shared expert.

4. **Intermediate Exit Normalization**:
   - `reflex_exit_layer` and `limbic_exit_layer` apply dedicated parameter-free `RMSNorm` normalization before their linear projection heads to match the variance scale of the final Cortex exit.

---

## 2. Hardware Planning and Execution Modes

| Available VRAM | Preset | Parameters | Recommended Execution Flags |
| :--- | :--- | :---: | :--- |
| **8 GB** | `triune-2.5b` | 2.45B | `--streaming --streaming_fp8_weights --use_fp8 --batch_size 2 --grad_accum 8` |
| **12 GB** | `triune-small` | 750M | `--use_fp8 --batch_size 4 --grad_accum_steps 4` |
| **16–24 GB** | `triune-2.5b` | 2.45B | `--use_fp8 --batch_size 4 --grad_accum_steps 4` |
| **80 GB** | `triune-7b` | 7.2B | `--use_fp8 --batch_size 8 --grad_accum_steps 4 --seq_len 1024` |
| **Multi-GPU (8x)** | `triune-7b` | 7.2B | `torchrun --nproc_per_node=8 scripts/train.py --model_name triune-7b` |

---

## 3. Router Configuration

The model uses a Straight-Through (ST) Gumbel-Softmax router for discrete pathway execution during forward passes and continuous gradient propagation during backward passes.

```python
from triune.model.router import GumbelSoftmaxRouter

router = GumbelSoftmaxRouter(
    hidden_dim=1280,
    target_depth_dist=(0.40, 0.35, 0.25),  # Reflex, Limbic, Cortex target distribution
    balance_coef=0.30                      # Auxiliary load-balancing coefficient
)
```

During forward execution:
```python
logits, route_logits = model(input_ids, temperature=0.8)
balance_loss = model.last_balance_loss
```

---

## 4. Training CLI Flags

Hyperparameters can be overridden directly from the command line:

```bash
python scripts/train.py \
    --model_name triune-2.5b \
    --num_layers 18 \
    --num_experts 8 \
    --seq_len 256 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 1e-4 \
    --total_steps 50000 \
    --target_depth_dist 0.34,0.33,0.33 \
    --steer_scale 0.20 \
    --shuffle_buffer 64 \
    --save_every 1000 \
    --eval_every 500
```

### Parameter Reference

- `--model_name`: Architecture preset (`triune-nano`, `triune-small`, `triune-2.5b`, `triune-7b`).
- `--num_layers`: Total layer count (must exceed `--limbic_exit_layer`).
- `--num_experts`: Number of MoE experts per routed block.
- `--dataset_name`: Hugging Face dataset ID or local path (`.jsonl`, `.txt`, `.parquet`).
- `--target_depth_dist`: Comma-separated target probabilities for early exits (e.g. `0.4,0.3,0.3`).
- `--steer_scale`: Subspace steering coefficient for centroid-augmented GaLore optimizer.
- `--shuffle_buffer`: Token shuffle buffer size for streaming datasets.
- `--no_wandb`: Disables Weights & Biases logging for local training.
