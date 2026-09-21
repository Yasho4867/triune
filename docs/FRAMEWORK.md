# Triune Ecosystem Framework Guide

## Architecture Overview

Triune is an **All-in-One AI Research Suite** engineered around a single source of truth: the `triune` Python package. It combines high-performance model architectures, adaptive multi-exit routing, an advanced 3-tier optimizer suite, AirLLM-style layer streaming, hardware resource management, and an embedded web/desktop Studio IDE.

```
TriuneTransformer/
├── pyproject.toml              # PyPA package configuration (pip install -e .)
├── requirements.txt            # Lightweight pip requirements fallback
├── triune/                     # CORE FRAMEWORK ENGINE
│   ├── __init__.py             # Public exports (load_model, TriuneTransformer, Muon, etc.)
│   ├── cli.py                  # Unified CLI (studio, train, chat, plan-memory)
│   ├── model/                  # Transformer, Attention, MoE, Norms, FP8, Zoo
│   ├── optim/                  # 3-Tier Optimizer: Muon, CentroidSteer, AdamW, Factory
│   ├── runtime/                # Layer Streaming, Resource Manager, Sandbox, Profiler
│   ├── trainer/                # Trainer, TrainingEngine, Checkpointing, Scheduler
│   ├── data/                   # Tokenizers, Streaming Datasets, CyclingDataLoader
│   ├── configs/                # Dynamic configuration builder and defaults
│   ├── modules/                # Modular extension & registry system
│   ├── api/                    # FastAPI & WebSockets (OpenAI Spec & Telemetry)
│   └── desktop.py              # PyWebView desktop application launcher
├── studio/                     # TRIUNE STUDIO (Visual IDE & GUI)
│   ├── src/                    # Web UI Frontend (HTML, React, CSS, Vendor assets)
│   ├── launcher/               # Desktop runner (desktop.py)
│   └── installer/              # Windows installer build scripts
└── models/                     # TRIUNE MODEL ZOO & ARCHITECTURE PRESETS
    ├── README.md               # Model Zoo documentation & hardware matrix
    └── configs/                # Architecture preset JSON definitions
```

---

## 🔑 Core Subsystems & Public APIs

### 1. Model Architecture & Zoo (`triune.model`)
* **`TriuneTransformer`**: Primary model class featuring Gated Linear Attention (GLA), Mixture-of-Experts (MoE) with a dedicated shared expert, and three variance-matched exit heads (`Reflex`, `Limbic`, `Cortex`).
* **`load_model(name_or_path, **kwargs)`**: Instantiates native Triune presets (`triune-nano`, `triune-small`, `triune-2.5b`, `triune-7b`, `triune-base`), loads from JSON architecture configs in `models/configs/`, or loads weights from checkpoints.
* **`register_model(name)`**: Decorator allowing third-party and custom models to register with the unified loader.

```python
import triune

# Load a 2.5B parameter Triune model
model = triune.load_model("triune-2.5b")
```

### 2. 3-Tier Optimizer Suite (`triune.optim`)
Triune divides model parameters into three distinct mathematical tiers:
* **Tier 1 (Muon)**: Applies 5th-order Newton-Schulz matrix orthogonalization to all non-expert 2D hidden projections (attention projections, shared expert projections, and intermediate heads).
* **Tier 2 (CentroidSteer)**: Symmetrical dual-sided low-rank SVD projections with semantic activation centroid steering for routed MoE expert matrices.
* **Tier 3 (AdamW)**: Adaptive moment estimation for 1D vectors (RMSNorm scale weights, biases, router heads, token embeddings).

```python
from triune.optim import build_optimizer

optimizer = build_optimizer(model, config)
```

### 3. Layer Streaming Engine (`triune.runtime.streaming`)
Allows models far exceeding physical GPU VRAM (such as `triune-2.5b` with 4.95B–2.5B parameters on an 8 GB laptop GPU) to train without out-of-memory errors:
* **FP8 Host Weight Compression**: Weights are compressed to `torch.float8_e4m3fn` in CPU host RAM (~2.4 GB host memory).
* **On-Demand Micro-Transfers**: A secondary CUDA prefetch stream transfers blocks immediately prior to forward/backward execution.
* **Layer-by-Layer Optimizer State Migration**: Optimizer states (`exp_avg`, `exp_avg_sq`, `momentum`) migrate to GPU immediately before each layer step and are evicted back to host RAM immediately after, maintaining a peak VRAM footprint of strictly $<300\text{ MB}$.

```python
from triune.runtime import LayerStreamingEngine, StreamingConfig

engine = LayerStreamingEngine(model, StreamingConfig(pin_memory=False, use_fp8=True))
engine.attach()
```

### 4. Hardware Resource Manager (`triune.runtime.resource_manager`)
* **`DynamicResourceManager`**: Probes physical GPU VRAM, compute capability, BF16/FP8 support, and assesses feasibility.
* **`LiveVRAMMonitor`**: Provides real-time formatted telemetry (`VRAM: allocated/total | Res | Peak | Headroom`).
* **User Override Authority**: Protects architectural immutability. Hardware limits can be bypassed cleanly via the `--force` flag without silently truncating layers or expert counts.

### 5. Embedded API & Studio Server (`triune.api`)
* **`create_app()`**: Configures a FastAPI application serving OpenAI-compatible endpoints (`/v1/chat/completions`), training telemetry WebSockets (`/ws/telemetry`), and static UI assets under `/static`.
* **`run_server(host, port)`**: Launches the Uvicorn server hosting the Studio backend.

---

## 💻 CLI Commands

The unified CLI provides command-line control over all engine capabilities:

```bash
# Launch Triune Studio native desktop window
triune studio --port 8000

# Launch headless FastAPI server for Studio & remote clients
triune serve --host 0.0.0.0 --port 8000

# Interactive terminal chat with any Triune model
triune chat --model triune-2.5b

# Calculate theoretical VRAM memory plan for target hardware
triune plan-memory --vram-gb 8.0
```

---

## 🧪 Automated Testing

```bash
# Core forward, backward, and training cycle tests
python tests/test_framework.py

# Layer streaming engine & optimizer state migration tests
python tests/test_streaming_engine.py

# 3-tier Muon and CentroidSteer optimizer tests
python tests/test_muon_and_adaptive_steer.py

# Hardware probe & user override tests
python tests/test_resource_manager.py

# Dynamic node registry and plugin tests
python tests/test_features.py
```
