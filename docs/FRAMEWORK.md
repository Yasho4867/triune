# Lucid Machines — Framework Guide

**Lucid Machines** (`lucid_machines` / `lucid_machines`) is a modular, hardware-aware deep learning ecosystem combining a native C/AVX2/CUDA/Triton operator engine (`lucid_machines_core`), **the Triune architecture by Lucid Machines** (**`TriuneTransformer` / `TriuneModel` / `TriuneConfig`**, plus `UniversalTransformer`) prepared to train our first flagship model **Junkie**, and the **Polysom Studio** desktop workstation.

## 3-Tier Refactored Architecture & Repository Layout

```text
Lucid Machines/
├── studio/publish/PolysomStudio.exe  # Self-contained native Windows desktop binary (.NET 10 + WebView2 + lucid_machines_core.dll)
├── pyproject.toml              # Package configuration (lucid-machines distribution, lucid_machines SDK, kcx primary CLI + polysom / lucid-machines / triune aliases)
├── csrc/                       # Tier 1: Native C/AVX2/OpenMP & CUDA Engine (lucid_machines_core)
│   ├── engine/                 # C ABI header & engine implementation for native kernels and telemetry
│   ├── kernels/                # Native CUDA kernels
│   └── build_native.py         # Cross-platform shared library builder (.dll / .so / .dylib)
├── lucid_machines/             # Tier 2: Core Python SDK & Runtime (with `triune/` compat shim)
│   ├── __init__.py             # Public API (TriuneTransformer, TriuneModel, Model, Config, CentroidSteer, Muon, ops)
│   ├── cli.py                  # Unified CLI entrypoints (studio, serve, train, finetune, chat, benchmark)
│   ├── universal.py            # UniversalTransformer (Dense & MoE with fused QKV/SwiGLU projections)
│   ├── ops/                    # Unified Operator API (dispatches to Triton/CUDA or native C/AVX2)
│   │   ├── _c_bridge.py        # Zero-copy ctypes bridge to lucid_machines_core (triune_core fallback)
│   │   ├── gla.py              # Gated Linear Attention operator
│   │   ├── norm.py             # Fused RMSNorm operator
│   │   ├── rope.py             # Rotary Position Embedding operator
│   │   ├── cross_entropy.py    # Chunked Logit-Free Cross-Entropy operator
│   │   ├── centroid_steer.py   # Batched Cholesky-QR & CentroidSteer operator
│   │   ├── muon_ns.py          # 5th-order Newton-Schulz polar orthogonalization operator
│   │   └── speculative.py      # Self-speculative Reflex-to-Cortex decoding operator
│   ├── kernels/                # Triton & PyTorch Tensor Core kernels (fast_gla, fast_muon, fast_triangular, fast_ce)
│   ├── model/                  # TriuneTransformer, TriuneModel, TriuneConfig, VectorisedGLA, MoE_FFN, GumbelSoftmaxRouter, FP8/FP4, Zoo
│   ├── optim/                  # 3-Tier Optimizers: Muon, CentroidSteer (Triangular Core), AdamW
│   ├── runtime/                # LayerStreamingEngine, ParameterStager, DynamicResourceManager, NeuronScope
│   ├── trainer/                # Trainer, TrainingEngine, LoRA/QLoRA, Checkpointing, Scheduler
│   ├── data/                   # Tokenizers, Streaming Datasets, CyclingDataLoader
│   ├── configs/                # Configuration builder and defaults
│   ├── plugins/                # Signed extension and DAG operator registry
│   ├── api/                    # FastAPI server, OpenAI-compatible endpoints, WebSocket telemetry
│   └── desktop.py              # Native desktop host & process lifecycle manager
├── studio/                     # Tier 3: Polysom Studio Frontend & Native Host (6 editorial themes)
│   ├── src/                    # Web frontend assets (DAG editor, Obsidian 3D Neuron Scope)
│   ├── windows/                # Visual Studio WPF + WebView2 native host & C# P/Invoke bridge
│   └── installer/              # Automated hardware-aware installers
├── models/                     # Triune architecture templates (`triune-*.json`) & `junkie.json` flagship model target
├── benchmarks/                 # Hardware-verified empirical benchmark reports
└── tests/                      # Mathematical invariant, parity, C-ABI, and integration test suites
```

---

## Core Subsystems

### 1. Model Architecture (`lucid_machines.model`)

- **`TriuneTransformer` / `TriuneModel` / `TriuneConfig`**: Core 3-tier **Triune** neural architecture by Lucid Machines combining Gated Linear Attention (GLA), Mixture-of-Experts (MoE) with a shared expert, and three variance-matched exit heads (`Reflex`, `Limbic`, `Cortex`), serving as the substrate for **Junkie** (our first flagship model to be trained).
- **`distill_exit_heads(...)` & `exit_distill_coef`**: Cortex-preserving KL self-distillation aligns intermediate `Reflex` and `Limbic` exit heads toward frozen `Cortex` distributions (`torch.equal` Cortex output invariance), integrated both as an inference calibration pass and in `TrainingEngine` `depth_mode="joint"`.
- **`convert_to_fp8(...)`, `convert_to_nvfp4(...)`, & Hadamard W4A16**: Drop-in post-training weight compression supporting E4M3 `FP8Linear` (`2.0x` compression, `%16`-aligned `torch._scaled_mm`), packed 4-bit E2M1 + 16-element E4M3 block-scaled `NVFP4Linear` (`5.0 b/w = 3.20x` compression; `46.65 MB` across `96` quantized blocks / `78.26M` params on the `90.79M` checkpoint), and Hadamard-rotated W4A16 + protected blocks (`78.88 MB`, `119.106` vs. `126.887` unrotated PPL).
- **`load_model(name_or_path, **kwargs)` & HuggingFace Bridge (`lucid_machines.export.hf_compat`, `lucid_machines.model.adapters`)**: Instantiates Triune architecture templates (`triune-tiny`, `triune-edge`, `triune-nano`, `triune-base`, `triune-small`, `triune-pro`, `triune-1b`, `triune-2.5b`, `triune-large`, `triune-moe`, `triune-7b`, `triune-frontier`, first flagship target `junkie`, plus legacy shorthand aliases), falls back transparently to `transformers.AutoModelForCausalLM.from_pretrained`, exports standard `TriuneForCausalLM` bundles (`export_to_huggingface`), and attaches `WeightTiedExitProjection` / `EarlyExitManager` early exits to pretrained models.
- **`register_model(name)`**: Decorator to register custom model architectures.

```python
import lucid_machines

model = lucid_machines.load_model("triune-2.5b")
```

### 2. Partitioned Optimizer (`lucid_machines.optim`)

Parameters are partitioned into three optimization tiers:

- **Tier 1 (Muon)**: 5th-order Newton-Schulz matrix orthogonalization for non-expert 2D hidden projections (attention projections, shared expert projections, intermediate heads).
- **Tier 2 (CentroidSteer)**: Exact Orthogonal Direct-Sum ($G = G_{\parallel C_e} \oplus G_{\perp C_e}$) and multi-directional ($k_c \in [4, 8]$) Block-Oja EMA activation subspace steering with Grassmann parallel transport, batched `sCholQR2` Upper-Triangular Krylov Core on $\operatorname{St}(r+1, d)$, centroid-deflated Krylov, true post-SwiGLU hidden centroids, independent `steer_step` reset, and border RMS regularization (`0.1118–0.3678 B/p` measured on routed MoE experts).
- **Tier 3 (AdamW)**: Adaptive moment estimation for 1D vectors (normalization weights, biases, router heads, token embeddings).

```python
from lucid_machines.optim import build_optimizer

optimizer = build_optimizer(model, config)
```

### 3. Layer Streaming Engine (`lucid_machines.runtime.streaming`)

Enables training models that exceed physical GPU memory capacity across both `TriuneTransformer` backbones and generic HuggingFace architectures (`_resolve_layers` / `layers_path`):

- **Host Weight & Activation Compression**: Stores layer parameters in pinned/host CPU RAM (`float8_e4m3fn` or packed 4-bit NVFP4 via `ParameterStager(use_fp4=True)`), with `activation_checkpointing=True` releasing saved intermediate activation graphs across blocks.
- **On-Demand Micro-Transfers & Reentry Guards**: Uses a high-priority CUDA prefetch stream to overlap transfers with compute and `_is_bwd_staged` reentry guards alongside non-parameter buffer copy-back (`is_buffer=True`).
- **Optimizer State Migration**: Optimizer states (`exp_avg`, `exp_avg_sq`, `CentroidSteer` triangular buffers) migrate to GPU memory during each layer's step and evict immediately after, cutting `102M`–`271M` VRAM by `76.4%`–`83.8%` (outperforming Windows WDDM spill by `1.97x` at `271M`) and fitting `695M` (`2.12 GB`) and `1.52B + CentroidSteer` (`3.13 GB`) inside `8 GB` consumer VRAM.

```python
from lucid_machines.runtime import LayerStreamingEngine, StreamingConfig

engine = LayerStreamingEngine(model, StreamingConfig(pin_memory=False, use_fp8=True))
engine.attach()
```

### 4. Hardware Resource Manager (`lucid_machines.runtime.resource_manager`)

- **`DynamicResourceManager`**: Inspects GPU memory, compute capability, BF16/FP8 support, and calculates allocation feasibility.
- **`LiveVRAMMonitor`**: Real-time VRAM telemetry tracking allocated, reserved, and peak memory.
- **Hardware Overrides**: The `--force` flag allows running configurations when the system warns of tight memory margins without modifying model parameters.

### 5. API Server (`lucid_machines.api`)

- **`create_app()`**: Configures a FastAPI application serving OpenAI-compatible endpoints (`/v1/chat/completions`), telemetry WebSockets (`/ws/telemetry`), and Polysom Studio UI static files.
- **`run_server(host, port)`**: Starts the Uvicorn server hosting the backend.

### 6. Training and Polysom Studio Graphs

The CLI `Trainer` and Polysom Studio training controls use `lucid_machines.trainer.engine.TrainingEngine`. The DAG's **Autograd Training Step** uses that same engine when connected to a live dataset stream without explicit tensor/model inputs. Its explicit `model_handle` + token-batch mode is a separate, lightweight one-step operator; it does not inherit every live trainer precision, routing, or accumulation setting.

Other DAG nodes are executable operators when they have the inputs their ports declare: GLA attention, MoE, depth routing, FP8/FP4-aware linear, RMSNorm, RoPE, loss, evaluation, checkpoint, and isolated LoRA operations. Missing inputs, unavailable hardware, and execution errors are reported as failures instead of fabricated success. Nodes explicitly catalogued as **specifications** remain design/configuration metadata and are not treated as a completed computation.

Load **CLI Training Graph** in Polysom Studio for the canonical shared-engine training step. For a custom tensor graph, connect actual tensors through the declared typed ports; use `use_node_config=true` on a model node only when you intentionally want an isolated graph model instead of the active Studio model.

#### Building custom nodes

The DAG is an acyclic, single-pass computation graph: each node executes once in topological order. It is suited to model/data preparation, tensor operations, one-step optimization, evaluation, export, and orchestration. It is not itself a recurrent training scheduler; use the shared `TrainingStep` node for an actual CLI-equivalent update, or put iteration/scheduling in a higher-level trainer/runtime. Graph **Execute** runs operators; **Compile to Training** applies supported configuration to the active Studio trainer and does not translate arbitrary graph topology into a training loop.

Polysom Studio serializes full-graph and single-node execution against the shared engine so concurrent tabs cannot mutate the same runtime at once. Full-graph requests accept an optional `run_id`; `run_queued`, `run_started`, per-node, and `run_finished` events include it, and `GET /v1/dag/queue` reports the active run and waiting count.

Plugins become composable operators by declaring typed ports and returning those output names. Function argument names should match port names; configuration belongs in ordinary keyword parameters. For example:

```python
from lucid_machines.plugins.registry import global_registry


@global_registry.register(
    name="ScaleTensor",
    category="Tensor",
    inputs=[{"name": "value", "type": "tensor"}],
    outputs=[{"name": "scaled", "type": "tensor"}],
)
def scale_tensor(value, factor=2.0):
    return {"scaled": value * factor}
```

Connect the upstream `value` output to `ScaleTensor.value`, set `factor` in the node parameters, and connect `ScaleTensor.scaled` downstream. Required declared outputs must be returned or the node fails; optional ports can be declared with `required=False`. `tensor`, `tensor_tuple`, `model_handle`, `optimizer_handle`, `dataset_stream`, `text`, `loss`, `metrics`, `config`, and `any` are supported port types. External plugins default to `trusted=False` and require Ed25519 signature verification (`LUCID_MACHINES_PLUGIN_TRUSTED_KEYS`).

In Polysom Studio, Ctrl-click nodes to select multiple nodes, then choose **Group** (or press Ctrl+G). Polysom Studio creates a typed subgraph with public boundary sockets and remaps wires around the group. **Ungroup** restores its internal nodes and connections; moving the group moves its contents when expanded. Graphs are autosaved in the current browser profile. **Save As** exports a portable workflow file and **Open** imports one.

---

## CLI Reference

For the native Windows setup and portable package workflow, see [`WINDOWS_INSTALLATION.md`](WINDOWS_INSTALLATION.md).

Direct non-loopback binds require all three of `LUCID_MACHINES_API_TOKEN` (at least 32 characters; legacy `TRIUNE_API_TOKEN` fallback), `LUCID_MACHINES_TLS_CERTFILE`, and `LUCID_MACHINES_TLS_KEYFILE`; the server refuses to start if any is missing. For Cloudflare, use a Cloudflare Tunnel with a Cloudflare Access policy and route it to the loopback-only Studio server (`http://127.0.0.1:8000`) so the public host and port stay closed. Direct remote access to the Python runner additionally requires `LUCID_MACHINES_ALLOW_REMOTE_SANDBOX=1`; keep that disabled unless authenticated operators need it.

```bash
# Launch Polysom Studio desktop window (`polysom` / `lucid-machines studio` compat aliases)
kcx studio --port 8000

# Run headless FastAPI server
kcx serve --port 8000

# Interactive chat session
kcx chat --model triune-2.5b

# Calculate memory plan for target hardware
kcx plan-memory --vram-gb 8.0
```

---

## Testing

```bash
pytest -q
python -m unittest discover tests
```

Individual invariant and integration modules:
```bash
pytest tests/test_pipeline_integration.py tests/test_self_speculative_decoding.py tests/test_staging_invariants.py tests/test_triangular_core_integrity.py tests/test_benchmark_validity.py
python tests/test_framework.py
python tests/test_streaming_engine.py
python tests/test_muon_and_adaptive_steer.py
python tests/test_resource_manager.py
python tests/test_features.py
```
