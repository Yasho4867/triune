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
├── studio/                     # Triune Studio frontend, launchers, and Windows app
│   ├── src/                    # Web frontend assets (HTML, CSS, React UI)
│   ├── TriuneStudio.bat        # 1-click desktop launcher (Windows)
│   ├── TriuneStudio.sh         # 1-click desktop launcher (Linux / macOS)
│   ├── windows/                # Visual Studio WPF + WebView2 native host
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

### 6. Training and Studio Graphs

The CLI `Trainer` and Studio training controls use `triune.trainer.engine.TrainingEngine`. The DAG's **Autograd Training Step** uses that same engine when connected to a live dataset stream without explicit tensor/model inputs. Its explicit `model_handle` + token-batch mode is a separate, lightweight one-step operator; it does not inherit every live trainer precision, routing, or accumulation setting.

Other DAG nodes are executable operators when they have the inputs their ports declare: GLA attention, MoE, depth routing, FP8/FP4-aware linear, RMSNorm, RoPE, loss, evaluation, checkpoint, and isolated LoRA operations. Missing inputs, unavailable hardware, and execution errors are reported as failures instead of fabricated success. Nodes explicitly catalogued as **specifications** remain design/configuration metadata and are not treated as a completed computation.

Load **CLI Training Graph** in Studio for the canonical shared-engine training step. For a custom tensor graph, connect actual tensors through the declared typed ports; use `use_node_config=true` on a model node only when you intentionally want an isolated graph model instead of the active Studio model.

#### Building custom nodes

The DAG is an acyclic, single-pass computation graph: each node executes once in topological order. It is suited to model/data preparation, tensor operations, one-step optimization, evaluation, export, and orchestration. It is not itself a recurrent training scheduler; use the shared `TrainingStep` node for an actual CLI-equivalent update, or put iteration/scheduling in a higher-level trainer/runtime. Graph **Execute** runs operators; **Compile to Training** applies supported configuration to the active Studio trainer and does not translate arbitrary graph topology into a training loop.

Plugins become composable operators by declaring typed ports and returning those output names. Function argument names should match port names; configuration belongs in ordinary keyword parameters. For example:

```python
from triune.plugins.registry import global_registry

@global_registry.register(
    name="ScaleTensor",
    category="Tensor",
    inputs=[{"name": "value", "type": "tensor"}],
    outputs=[{"name": "scaled", "type": "tensor"}],
)
def scale_tensor(value, factor=2.0):
    return {"scaled": value * factor}
```

Connect the upstream `value` output to `ScaleTensor.value`, set `factor` in the node parameters, and connect `ScaleTensor.scaled` downstream. Required declared outputs must be returned or the node fails; optional ports can be declared with `required=False`. `tensor`, `tensor_tuple`, `model_handle`, `optimizer_handle`, `dataset_stream`, `text`, `loss`, `metrics`, `config`, and `any` are supported port types. Plugins run in-process with the framework and must be trusted; the plugin registry is not a security sandbox.

In Studio, Ctrl-click nodes to select multiple nodes, then choose **Group** (or press Ctrl+G). Triune creates a typed subgraph with public boundary sockets and remaps wires around the group. **Ungroup** restores its internal nodes and connections; moving the group moves its contents when expanded. Graphs are autosaved in the current browser profile. **Save As** exports a portable `.triune.json` workflow and **Open** imports one. A workflow file contains architecture/configuration and graph data, not model weights, credentials, or installed plugin code. Groups are reusable within workflow files; they do not yet have a separate shared asset library or nested in-canvas editor.

---

## CLI Reference

Direct non-loopback binds require all three of `TRIUNE_API_TOKEN` (at least 32 characters), `TRIUNE_TLS_CERTFILE`, and `TRIUNE_TLS_KEYFILE`; the server refuses to start if any is missing. The token is sent as a bearer credential, so do not disable TLS. For Cloudflare, use a Cloudflare Tunnel with a Cloudflare Access policy and route it to the loopback-only Studio server (for example, `http://127.0.0.1:8000`) so the public host and port stay closed. Configure Access to protect the Studio, API, and WebSocket paths. The tunnel-to-origin connection is loopback, so this deployment relies on Cloudflare Access at the edge; the app does not independently validate Cloudflare Access JWTs. Direct remote access to the Python runner additionally requires `TRIUNE_ALLOW_REMOTE_SANDBOX=1`; keep that disabled unless authenticated operators need it.

```bash
# Launch Triune Studio desktop window
triune studio --port 8000

# Run headless FastAPI server
triune serve --port 8000

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
