# Windows Studio Setup

Triune Studio runs its Python backend natively on Windows; WSL is not required.
The Windows desktop shell is published self-contained for .NET. Its first-launch
**Install / repair Studio environment** action provisions the managed Python
environment, PyTorch, CUDA-compatible dependencies, W&B, and Studio packages.
An NVIDIA driver is still required for CUDA acceleration. If Python is missing,
the setup launcher installs Python 3.12 through Windows Package Manager (`winget`)
for the current user.

## Build a portable Windows package

From a developer checkout with the .NET 10 SDK and Windows WebView2 Runtime:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/publish_studio_windows.ps1
```

This creates a versioned folder and ZIP under `dist/`; it does not delete or
overwrite earlier packages. Extract the ZIP and launch
`studio/windows/publish/TriuneStudio.exe`. If the Python/CUDA environment has
not yet been provisioned, use **Install / repair Studio environment** on the
startup screen. The package contains the framework and UI sources but not user
checkpoints, credentials, datasets, or model weights.

## Repair an existing checkout

Close Studio first, then run:

```powershell
studio\installer\install_studio.bat --skip-torch --yes --no-launch
```

This keeps the existing PyTorch installation and refreshes Studio dependencies.
To reinstall/probe PyTorch and CUDA too, omit `--skip-torch` (the default), then
complete the installer's hardware verification before launching Studio again.
Startup and setup logs are stored under `%LOCALAPPDATA%\Triune\logs`.

## Measure local inference

Load the same checkpoint in Studio before running its mode. Run the two modes
sequentially on the same machine and settings, rather than loading duplicate
model copies at once on a memory-limited GPU:

```powershell
python scripts/benchmark_inference.py --mode cli --checkpoint checkpoints\triune_studio_step_25500.pt --prompt "Explain the model's routing hierarchy." --max-tokens 128
python scripts/benchmark_inference.py --mode studio --checkpoint checkpoints\triune_studio_step_25500.pt --prompt "Explain the model's routing hierarchy." --max-tokens 128
```

The report includes first-token latency, decode tokens/second, total latency,
device, and checkpoint. Studio mode refuses to run if a different checkpoint is
active. The tokenizer, prompt, depth, temperature, and output budget should
match for a useful comparison.
