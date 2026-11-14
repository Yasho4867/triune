# Triune AI

**Triune AI** is a modular, hardware-aware deep learning ecosystem and native desktop IDE for training, fine-tuning, and serving multi-exit Mixture-of-Experts (`TriuneTransformer`) and dense (`UniversalTransformer`) language models on consumer hardware.

## Key Capabilities & Pillars

- **Gated Linear Attention (GLA)**: $O(N)$ linear-time complexity and strictly constant $O(1)$ memory (`48.0 KiB` for 6 heads, `64.0 KiB` for 8 heads) recurrent decoding with native BF16 Tensor Core intra-chunk GEMM, fused chunkwise recurrent Triton scan for autograd training (eliminating intermediate PyTorch graph node allocations), and logarithmic parallel prefix scan for inference, providing a **256x–512x smaller recurrent cache** and up to **7.94x faster autoregressive decoding** at long contexts ($L=16,384$) than standard Softmax KV-caching.
- **Sparse Mixture of Experts (MoE)**: Batched 3D BMM top-$k$ expert routing with an always-active shared expert, zero-sync GPU overflow counters, and activation centroid tracking.
- **Hierarchical Dynamic Exits**: Calibrated intermediate exit heads (**Reflex**, **Limbic**) and full-depth exit (**Cortex**), delivering a **1.56x forward latency speedup** (-35.8% latency) and **1.49x faster autoregressive decoding** ($6.24\text{ ms/tok}$ vs $9.32\text{ ms/tok}$) via entropy-gated token routing, with shallow Reflex exits executing at up to $22,709\text{ tok/s}$.
- **3-Tier Partitioned Optimizer (`CentroidSteerOptimizer`)**:
  - **Tier 1 (`Muon`)**: Batched 5th-order Newton-Schulz polar orthogonalization with fused `baddbmm` polynomial evaluation and multi-tensor `_foreach` updates for 2D hidden projections ($2.0\text{ bytes/param}$).
  - **Tier 2 (`CentroidSteer`)**: Universal Triangular Core with batched Cholesky-QR subspace iteration, condition-ratio monitoring, dual-orthonormal centroid steering, and Triton-fused upper-triangular AdamW for routed MoE experts (**98.4% state memory reduction to 0.128 bytes/param**, with highest convergence efficiency on 90.8M MoE).
  - **Tier 3 (`AdamW` / `8-bit AdamW`)**: Fused adaptive moments for 1D vectors, norms, router gates, and vocabulary embeddings ($4.54\text{ B/p}$ model-wide with standard FP32 AdamW, or $1.27\text{ B/p}$ with 8-bit AdamW).
- **Layer Streaming Runtime (`LayerStreamingEngine`)**: On-demand layer staging and immediate gradient offloading between host RAM and GPU VRAM with configurable `resident_gpu_layers`, reducing peak GPU VRAM by **up to 47.9%** without pegging system RAM.
- **Native C/AVX2/CUDA/Triton Operator Engine (`csrc/` & `triune.ops`)**: Zero-copy `ctypes` / P/Invoke C ABI (`triune_core.dll` / `libtriune_core.so`) and Triton/Tensor Core kernels for RMSNorm, RoPE, GLA recurrent decode, chunked vocabulary-projection Cross-Entropy (**up to 91.3% peak transient VRAM saved**), and Neuron Scope activation telemetry.
- **Triune AI Studio (`TriuneStudio.exe` / `TriuneStudioSetup.exe`)**: Self-contained native Windows desktop application (`.NET 10` + `WebView2` + `triune_core.dll`), WPF installer (`TriuneStudioSetup.exe`), and visual DAG IDE for interactive training, LoRA/QLoRA fine-tuning, live Neuron Scope inspection, and OpenAI-compatible serving.

## 3-Tier Refactored Repository Structure

```text
TriuneTransformer/
├── TriuneStudio.exe            # Self-contained native Windows desktop binary (.NET 10 + WebView2 + triune_core.dll)
├── csrc/                       # Tier 1: Native C/AVX2/OpenMP & CUDA Engine (libtriune_core)
│   ├── include/triune_core.h   # C ABI header for native operators & Neuron Scope telemetry
│   ├── src/                    # Native C/AVX2 kernels (RMSNorm, RoPE, GLA step, CentroidSteer, Router)
│   ├── cuda/                   # Native CUDA kernels
│   └── build_native.py         # Cross-platform shared library builder (.dll / .so / .dylib)
├── triune/                     # Tier 2: Core Python SDK, Kernels & Runtime
│   ├── __init__.py             # Minimalist Public API (Model, Config, CentroidSteer, Muon, ops)
│   ├── ops/                    # Unified Operator API (gla, rmsnorm, rope, cross_entropy, centroid_steer, muon_ns)
│   ├── kernels/                # Triton & PyTorch Tensor Core kernels (fast_gla, fast_muon, fast_triangular, fast_ce)
│   ├── model/                  # TriuneTransformer, VectorisedGLA, MoE_FFN, GumbelSoftmaxRouter, FP8/FP4, Zoo
│   ├── optim/                  # 3-Tier Optimizers: Muon, CentroidSteer (Triangular Core), AdamW
│   ├── runtime/                # LayerStreamingEngine, ParameterStager, DynamicResourceManager, NeuronScope
│   ├── trainer/                # Trainer, TrainingEngine, LoRA/QLoRA, Checkpointing, Scheduler
│   ├── api/                    # FastAPI server, OpenAI-compatible REST API, WebSocket telemetry
│   └── cli.py                  # Unified CLI (studio, serve, train, finetune, chat, benchmark, doctor)
├── studio/                     # Tier 3: Triune AI Studio Frontend, Native Host & Installers
├── models/                     # Architecture presets (triune-nano, triune-small, triune-2.5b, triune-7b)
├── benchmarks/                 # Hardware-verified empirical benchmark reports (Categories A–F)
├── scripts/                    # Training, benchmark, and build utilities
├── tests/                      # Mathematical invariant, parity, C-ABI, and integration test suites
└── docs/                       # Mathematical foundations and technical guides
```

## 5 Ways to Use Triune AI in Your Workflows

1. **1-Line Universal Booster (`triune.upgrade`)**:
   Wrap *any* PyTorch or Hugging Face model with `engine = triune.upgrade(model, lora_rank=16)` to automatically replace `RMSNorm`/`LayerNorm` with `FastRMSNorm`, inject fused `addmm_` LoRA adapters (`fast_lora_linear`), attach `CentroidSteerOptimizer`, and run zero-allocation chunked Cross-Entropy (`chunked_cross_entropy`).
2. **Minimalist Python SDK (`import triune`)**:
   Build dense or MoE transformers with `triune.Model(triune.Config(...))` or `triune.load_model("triune-2.5b")`, and train with `triune.CentroidSteer` or `triune.Muon`.
3. **Drop-In Low-VRAM Optimizers (`triune.CentroidSteer` & `triune.Muon`)**:
   Plug `CentroidSteerOptimizer` or `Muon` directly into any existing PyTorch or Hugging Face training loop to cut expert optimizer state VRAM by **up to 98.4%** ($0.128\text{ bytes/param}$) and model-wide VRAM by **43.3%–84.1%**.
4. **Standalone Fused Operators (`triune.ops`)**:
   Call `triune.ops.cross_entropy`, `triune.ops.chunked_cross_entropy` (saving up to **91.3% VRAM**), `triune.ops.gla`, `triune.ops.rmsnorm`, `triune.ops.rope`, `triune.ops.swiglu`, or `triune.ops.lora_linear` inside custom PyTorch models.
5. **Native Desktop IDE & Unified CLI (`TriuneStudio.exe` / `triune studio` / `triune serve` / `triune doctor`)**:
   Launch the self-contained Windows desktop application (`TriuneStudio.exe`) or serve models locally at `http://127.0.0.1:8000/v1/chat/completions` for drop-in integration with Cursor, Continue, LangChain, or OpenAI SDK clients.

## 1-Click Installation (Recommended)

Triune provides a native Windows installer (`TriuneStudioSetup.exe` built in `studio/publish/`) as well as automated installers in `studio/installer/` that inspect your hardware (NVIDIA CUDA, AMD ROCm, Apple Silicon Metal, or CPU), configure a virtual environment, install the optimal PyTorch distribution, and create desktop launchers.

### Windows (Installer, Double-Click, or PowerShell)
```cmd
:: Run the native Windows installer or automated setup script:
studio\publish\TriuneStudioSetup.exe
studio\installer\install_studio.bat
```

### Linux / WSL2 / macOS (Terminal)
```bash
chmod +x studio/installer/install_studio.sh
./studio/installer/install_studio.sh
```

### Universal Python Installer (Any OS)
```bash
python studio/installer/installer.py
```

---

## Launching Triune AI Studio

After installation, launch **Triune AI Studio** as a native desktop application:

- **Windows (Self-Contained Native Executable)**: Double-click `TriuneStudio.exe` at the repository root or `studio\publish\TriuneStudio.exe` (bundles `.NET 10`, `WebView2`, and `triune_core.dll` into a single native executable).
- **Windows (Batch Fallback)**: `studio\TriuneStudio.bat`
- **Linux / macOS**: `./studio/TriuneStudio.sh`
- **Python / CLI**: `triune studio --port 8000` or `python scripts/launch_studio.py --port 8000`

### Rebuilding the Native Windows App (`TriuneStudio.exe`) and Installer (`TriuneStudioSetup.exe`)

To build the native C/AVX2 shared library (`triune_core.dll`) and publish both `studio/publish/TriuneStudio.exe` and `studio/publish/TriuneStudioSetup.exe` (plus portable ZIP archives in `dist/`):

```powershell
python csrc/build_native.py
powershell -ExecutionPolicy Bypass -File scripts/publish_studio_windows.ps1
Copy-Item studio/publish/TriuneStudio.exe .\TriuneStudio.exe -Force
```

### Studio appearance and extensions

Studio opens in the **Modern Light** edition. Use **Appearance & Themes** in the sidebar to switch between Modern and Classic editions, choose light or dark mode, apply a curated theme, or set a personal accent color. Appearance preferences are saved locally in that desktop/browser profile. The **Extensions Marketplace** is separate: it lists curated modules and searches GitHub/Hugging Face; use **Add local or remote extensions** when you intentionally want to scan a workspace or clone a repository. Remote plugin code stays dormant after cloning and updating. Activation requires an explicit warning acknowledgement, a version-1 manifest, a trusted Ed25519 signature, matching SHA-256 entrypoint hashes, and the strict AST checks. Publishers can refresh the manifest signature with `python scripts/sign_plugin_manifest.py path/to/plugin --key-id publisher-key-1`; the tool prompts for the raw base64 private key without echoing it. Configure `TRIUNE_PLUGIN_TRUSTED_KEYS` as a JSON object mapping key IDs to base64-encoded Ed25519 public keys; do not put private signing keys in Triune. Even a valid signature does not sandbox plugin code, so activate only publishers you trust.

User-entered provider and hub credentials are stored in the operating system credential store, not `studio_config.json`. The Windows desktop setup uses Windows Credential Manager even when model computation is hosted in WSL; no separate WSL keyring package or daemon is required. Standalone Linux deployments use Secret Service/KWallet, and environment variables remain supported for headless setups. Legacy plaintext credentials are read for migration compatibility and are moved to the OS store on the next successful save.

---

## Manual Installation (Alternative)

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
# Launch Triune Studio Desktop App
triune studio --port 8000

# Start embedded API & UI server
triune serve --port 8000

# Interactive terminal chat
triune chat --checkpoint checkpoints_full/best.pt

# GPU & hardware diagnostics
triune gpucheck

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

