# Lucid Machines v1.0.0 — Installation & Setup Guide

**Lucid Machines** (`lucid-machines` / `lucid_machines`) supports Windows, Linux, WSL2, and macOS. It includes both a Python SDK, unified **`kcx`** (**Polysom KCX**) CLI (with `polysom` / `lucid-machines` compatibility aliases), and a self-contained native Windows desktop workbench (**Polysom Studio v1.0.0**).

---

## 1. Native Windows Desktop App & Installer (`studio/publish/`)

On Windows 10/11 (x64), the native WPF setup wizard (`PolysomStudioSetup.exe`) and automated installer (`studio\installer\install_studio.bat`) probe your NVIDIA CUDA driver, provision the isolated Python virtual environment (`studio/studio_env`), install hardware-matched PyTorch wheels, register the `kcx` (plus `polysom` / `lucid-machines`) CLIs on your User `PATH`, register the `HKCU` Windows `Uninstall` entry, and create Desktop and Start Menu shortcuts.

### Interactive GUI Setup
```powershell
studio\publish\PolysomStudioSetup.exe
```

### Headless / Unattended CLI Flags
```powershell
# Verify existing environment health (exits with code 0 on success)
studio\publish\PolysomStudioSetup.exe --verify

# Unattended install/repair without launching the Polysom Studio window
studio\publish\PolysomStudioSetup.exe --yes --no-launch

# Unattended fast repair keeping existing PyTorch installation
studio\installer\install_studio.bat --skip-torch --yes --no-launch

# Unattended clean removal of shortcuts and HKCU Uninstall registry key
studio\publish\PolysomStudioSetup.exe --uninstall --yes
```

### Rebuilding the Native Binaries & Portable Distribution ZIP
Requires the `.NET 10 SDK` and a C++17 compiler (MSVC / Clang / GCC) for the native C/AVX2 engine (`lucid_machines_core.dll`):
```powershell
.\studio\studio_env\Scripts\python.exe csrc/build_native.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/publish_studio_windows.ps1
```
This publishes self-contained single-file executables to `studio/publish/` and produces `dist/PolysomStudio-1.0.0-win-x64.zip` along with a `.sha256` checksum manifest.

---

## 2. Linux, WSL2 & macOS Automated Installation

```bash
chmod +x studio/installer/install_studio.sh
./studio/installer/install_studio.sh
```

Or run the cross-platform Python installer directly:
```bash
python3 studio/installer/installer.py --yes --no-launch
```

---

## 3. Manual Python SDK Installation (`pip`)

```bash
# Core framework
pip install -e .

# With Polysom Studio web/API server dependencies
pip install -e ".[studio]"

# Full developer & training installation
pip install -e ".[all]"
```

---

## 4. Post-Install Verification

Run the built-in hardware and subsystem diagnostic suite:
```bash
kcx doctor
# Launch Polysom Studio:
kcx studio --port 8000
# (compatibility aliases `polysom`, `lucid-machines`, and `triune` remain supported)
```
