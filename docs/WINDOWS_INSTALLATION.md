# Polysom Studio (Lucid Machines) — Windows Setup & Portable Packaging (`v1.0.0`)

**Polysom Studio** (by **Lucid Machines**) runs its Python backend and native C/AVX2 engine (`lucid_machines_core.dll`, with `triune_core.dll` fallback) natively on Windows; WSL is not required.

The Windows desktop shell (`PolysomStudio.exe`) is published as a self-contained single-file executable bundling `.NET 10`, `WebView2`, all 6 Polysom Studio themes (`Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`), and the native C/AVX2 shared library. Its first-launch **Install / repair Studio environment** action (or the standalone `PolysomStudioSetup.exe` installer) provisions the managed Python environment (`studio/studio_env`), PyTorch, CUDA-compatible dependencies, W&B, and Studio packages. If Python is missing, the setup launcher can bootstrap Python 3.12 via Windows Package Manager (`winget`) for the current user.

---

## 1. Build a Portable Windows Package

From a developer checkout with the `.NET 10 SDK` and Windows WebView2 Runtime:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/publish_studio_windows.ps1
```

This publishes the self-contained executables to `studio/publish/` and creates the versioned portable ZIP archive (`dist/PolysomStudio-1.0.0-win-x64.zip`) plus a `.sha256` checksum file under `dist/`. Extract the ZIP and launch `PolysomStudio.exe` or run `PolysomStudioSetup.exe` to register Desktop/Start Menu shortcuts and the Windows `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall` entry.

---

## 2. Unattended Verification, Repair & Uninstall Flags

```powershell
# Verify existing environment health (exits with code 0 when all subsystems pass)
studio\publish\PolysomStudioSetup.exe --verify

# Unattended install/repair without auto-launching Polysom Studio
studio\publish\PolysomStudioSetup.exe --yes --no-launch

# Fast dependency repair keeping existing PyTorch/CUDA wheels
studio\installer\install_studio.bat --skip-torch --yes --no-launch

# Unattended removal of shortcuts and HKCU Uninstall registry entry (preserves checkpoints/data)
studio\publish\PolysomStudioSetup.exe --uninstall --yes
```

Startup and setup logs are stored under `%LOCALAPPDATA%\Lucid Machines\logs` (with legacy `%LOCALAPPDATA%\Triune\logs` fallback).

---

## 3. Measure Local Inference Parity (CLI vs. Polysom Studio Server)

Load the same checkpoint in Polysom Studio before running its mode. Run the two modes sequentially on the same machine and settings rather than loading duplicate model copies at once on a memory-limited GPU:

```powershell
python scripts/benchmark_inference.py --mode cli --checkpoint checkpoints\triune_studio_step_25500.pt --prompt "Explain the model's routing hierarchy." --max-tokens 128
python scripts/benchmark_inference.py --mode studio --checkpoint checkpoints\triune_studio_step_25500.pt --prompt "Explain the model's routing hierarchy." --max-tokens 128
```

The report includes the short kernel-warmup cost, first-token latency (TTFT, e.g., `71.3 ms` warm on RTX 5070), decode tokens/second (`40.8 tok/s` warm decode), total latency, device, and checkpoint.
