# Framework Guide

Triune is a modular PyTorch framework for multi-exit Mixture-of-Experts language models.

## Repository Layout

```text
triune/
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependency fallback
├── triune/                     # Core framework
│   ├── __init__.py             # Public exports (load_model, TriuneTransformer, Muon, etc.)
│   ├── cli.py                  # CLI entrypoints (studio, train, chat, plan-memory)
│   ├── model/                  # Transformer, Attention, MoE, Norms, FP8, Zoo
│   ├── optim/                  # Optimizers: Muon, CentroidSteer, AdamW, Factory
│   ├── runtime/                # Layer Streaming, Resource Manager, Sandbox, Profiler
│   ├── trainer/                # Trainer, TrainingEngine, Checkpointing, Scheduler
│   ├── data/                   # Tokenizers, Streaming Datasets, CyclingDataLoader
│   ├── configs/                # Configuration builder and defaults
│   ├── modules/                # Modular extension and registry system
│   ├── api/                    # FastAPI and WebSocket server
│   └── desktop.py              # Desktop window launcher
├── studio/                     # Triune Studio frontend and 1-click launchers
│   ├── src/                    # Web frontend assets (HTML, CSS, React UI)
│   ├── TriuneStudio.bat        # 1-click desktop launcher (Windows)
│   ├── TriuneStudio.sh         # 1-click desktop launcher (Linux / macOS)
│   └── installer/              # Automated hardware-aware installers
│       ├── installer.py        # Cross-platform hardware probe and venv setup
│       ├── install_studio.bat  # 1-click installer (Windows)
│       └── install_studio.sh   # 1-click installer (Linux / macOS)
└── models/                     # Model zoo presets and configurations
    ├── README.md               # Model specifications
    └── configs/                # JSON architecture configurations
```

---

## Core Subsystems

### 1. Model Architecture (`triune.model`)

- **`TriuneTransformer`**: Core architecture combining Gated Linear Attention (GLA), Mixture-of-Experts (MoE) with a shared expert, and three variance-matched exit heads (`Reflex`, `Limbic`, `Cortex`).
- **`load_model(name_or_path, **kwargs)`**: Instantiates model presets (`triune-nano`, `triune-small`, `triune-2.5b`, `triune-7b`), loads from JSON architecture configs in `models/configs/`, or loads checkpoint weights.
- **`register_model(name)`**: Decorator to register custom model architectures.

```python
import triune

model = triune.load_model("triune-2.5b")
```

### 2. Partitioned Optimizer (`triune.optim`)

Parameters are partitioned into three optimization tiers:

- **Tier 1 (Muon)**: 5th-order Newton-Schulz matrix orthogonalization for non-expert 2D hidden projections (attention projections, shared expert projections, intermediate heads).
- **Tier 2 (CentroidSteer)**: Dual-sided low-rank SVD projections with semantic activation centroid steering for routed MoE experts.
- **Tier 3 (AdamW)**: Adaptive moment estimation for 1D vectors (normalization weights, biases, router heads, token embeddings).

```python
from triune.optim import build_optimizer

optimizer = build_optimizer(model, config)
```

### 3. Layer Streaming Engine (`triune.runtime.streaming`)

Enables training models that exceed physical GPU memory capacity:

- **Host Weight Compression**: Stores layer parameters in CPU host RAM compressed to `torch.float8_e4m3fn`.
- **On-Demand Micro-Transfers**: Uses a CUDA prefetch stream to transfer layer blocks immediately prior to forward and backward passes.
- **Optimizer State Migration**: Optimizer states (`exp_avg`, `exp_avg_sq`, `momentum`) migrate to GPU memory during each layer's step and are evicted back to host RAM immediately after, maintaining peak VRAM under 350 MB.

```python
from triune.runtime import LayerStreamingEngine, StreamingConfig

engine = LayerStreamingEngine(model, StreamingConfig(pin_memory=False, use_fp8=True))
engine.attach()
```

### 4. Hardware Resource Manager (`triune.runtime.resource_manager`)

- **`DynamicResourceManager`**: Inspects GPU memory, compute capability, BF16/FP8 support, and calculates allocation feasibility.
- **`LiveVRAMMonitor`**: Real-time VRAM telemetry tracking allocated, reserved, and peak memory.
- **Hardware Overrides**: The `--force` flag allows running configurations when the system warns of tight memory margins without modifying model parameters.

### 5. API Server (`triune.api`)

- **`create_app()`**: Configures a FastAPI application serving OpenAI-compatible endpoints (`/v1/chat/completions`), telemetry WebSockets (`/ws/telemetry`), and UI static files.
- **`run_server(host, port)`**: Starts the Uvicorn server hosting the backend.

---

## CLI Reference

```bash
# Launch Triune Studio desktop window
triune studio --port 8000

# Run headless FastAPI server
triune serve --host 0.0.0.0 --port 8000

# Interactive chat session
triune chat --model triune-2.5b

# Calculate memory plan for target hardware
triune plan-memory --vram-gb 8.0
```

---

## Testing

```bash
pytest tests/
```

Individual test modules:
```bash
python tests/test_framework.py
python tests/test_streaming_engine.py
python tests/test_muon_and_adaptive_steer.py
python tests/test_resource_manager.py
python tests/test_features.py
```
