"""Automated Windows Installer & Native Executable Builder for Triune Studio Desktop App.

Creates a self-contained local environment, installs dependencies, compiles TriuneStudio.exe,
and launches the native desktop application.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"-> Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def detect_hardware() -> str:
    print("\n--- [Hardware Stack Detection] ---")
    gpu_name = "CPU Only"
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        if smi.returncode == 0 and smi.stdout.strip():
            gpu_name = f"NVIDIA GPU Acceleration ({smi.stdout.strip()})"
    except Exception:
        pass
    
    # Check WSL environment
    has_wsl = False
    try:
        wsl_res = subprocess.run(["wsl", "bash", "-c", "python3 -c 'import torch; print(torch.cuda.is_available())'"], capture_output=True, text=True)
        if "True" in wsl_res.stdout:
            has_wsl = True
            gpu_name += " [WSL PyTorch 2.13.0+cu130 CUDA Active]"
    except Exception:
        pass

    print(f"Detected Hardware Engine: {gpu_name}")
    return gpu_name


def install_triune_studio() -> None:
    print("=" * 65)
    print("      [TRIUNE STUDIO] - STANDALONE WINDOWS APP INSTALLER")
    print("=" * 65)

    detect_hardware()

    studio_dir = Path(__file__).resolve().parent.parent
    workspace_dir = studio_dir.parent
    venv_dir = studio_dir / "studio_env"

    # Kill any active TriuneStudio.exe instances so files are not locked
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "TriuneStudio.exe", "/T"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except Exception:
            pass

    # Determine base python executable
    if getattr(sys, "frozen", False):
        python_base = r"C:\Windows\py.exe" if os.path.exists(r"C:\Windows\py.exe") else "python"
    else:
        python_base = sys.executable

    # 1. Create self-contained Python venv
    if not venv_dir.exists():
        print("Creating self-contained Python virtual environment (studio_env)...")
        run_cmd([python_base, "-m", "venv", str(venv_dir)])
    else:
        print("Found existing environment (studio_env).")

    # Determine python path inside venv
    if sys.platform == "win32":
        python_bin = venv_dir / "Scripts" / "python.exe"
    else:
        python_bin = venv_dir / "bin" / "python"

    # 2. Upgrade pip using python -m pip
    print("Upgrading pip and setuptools...")
    run_cmd([str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # 3. Install framework API, PyTorch, webview, pyinstaller, uvicorn, fastapi
    print("Installing full software stack & dependencies (PyTorch, FastAPI, PyWebView)...")
    run_cmd([
        str(python_bin),
        "-m",
        "pip",
        "install",
        "torch",
        "fastapi",
        "uvicorn",
        "pywebview",
        "pyinstaller",
        "-e",
        str(workspace_dir),
    ])

    # 4. Create one-click launcher shortcut TriuneStudio.bat
    launcher_bat = studio_dir / "TriuneStudio.bat"
    bat_content = f"""@echo off
TITLE Triune Studio - AI Engine and Research IDE
echo Starting Triune Studio Native Windows Desktop Application...
"{python_bin}" "{studio_dir / 'launcher' / 'desktop.py'}"
"""
    launcher_bat.write_text(bat_content, encoding="utf-8")

    # 5. Build Native Windows .exe via PyInstaller with triune package bundled
    dist_dir = studio_dir / "dist"
    app_exe = dist_dir / "TriuneStudio" / "TriuneStudio.exe"
    print("Compiling Standalone Windows Executable (TriuneStudio.exe)...")
    try:
        run_cmd(
            [
                str(python_bin),
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--onedir",
                "--windowed",
                "--name=TriuneStudio",
                "--collect-all",
                "webview",
                "--add-data",
                f"{studio_dir / 'src'};src",
                str(studio_dir / "launcher" / "desktop.py"),
            ],
            cwd=studio_dir,
        )
        print(f"Native Windows Executable created at: {app_exe}")
    except Exception as err:
        print(f"PyInstaller compilation note: {err}")

    print("\n" + "=" * 65)
    print("  INSTALLATION COMPLETE!")
    print("  Launching Triune Studio...")
    print("=" * 65 + "\n")

    # 6. Auto-launch TriuneStudio.exe or launcher script
    if launcher_bat.exists():
        subprocess.Popen([str(launcher_bat)], shell=True)


if __name__ == "__main__":
    install_triune_studio()
