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
studio/                 Triune Studio frontend, installers, and Windows app solution
scripts/                Training, chat, and utility scripts
tests/                  Unit and integration tests
docs/                   Mathematical foundations and technical guides
```

## 🚀 1-Click Installation (Recommended)

Triune provides an automated installer in `studio/installer/` that inspects your hardware (NVIDIA CUDA, AMD ROCm, Apple Silicon Metal, or CPU), configures a virtual environment, installs the optimal PyTorch distribution, and creates desktop launchers.

### Windows (Double-Click or Command Prompt / PowerShell)
```cmd
:: Double-click install_studio.bat or run from terminal:
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

## 🖥️ Launching Triune Studio

After installation, launch Triune Studio as a desktop application:

- **Windows (native app)**: Open `studio\windows\TriuneStudio.sln` in Visual Studio and publish the `TriuneStudio.Windows` project (steps below). Run `studio\windows\publish\TriuneStudio.exe`; the installer points the Desktop shortcut to this executable when it is present.
- **Windows (Python fallback)**: `studio\TriuneStudio.bat` (used only if the native app has not been published)
- **Linux / macOS**: `./studio/TriuneStudio.sh`
- **Python / CLI**: `python scripts/launch_studio.py --port 8000`

### Build the native Windows app

Install Visual Studio with the **.NET desktop development** workload, the .NET 10 SDK, and the Microsoft Edge WebView2 Runtime. Open `studio/windows/TriuneStudio.sln`, restore NuGet packages, then build or publish `TriuneStudio.Windows`. The native WPF window hosts the existing Studio interface and starts the existing Python backend (including its WSL GPU detection); it does not replace or duplicate the model engine.

To publish a self-contained x64 app from PowerShell at the repository root:

```powershell
dotnet publish studio/windows/TriuneStudio.Windows/TriuneStudio.Windows.csproj `
  -c Release -r win-x64 --self-contained true `
  -o studio/windows/publish
```

The .NET runtime is bundled, but WebView2 Runtime and the installed Triune Python environment are still required. For a desktop shortcut, rerun `studio/installer/install_studio.bat` after publishing; it will target the `.exe` instead of the batch fallback.

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

