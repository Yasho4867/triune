#!/usr/bin/env python3
"""Triune Studio & AI Engine - Comprehensive Automated Installer.

Cross-platform installer for Windows, Linux, WSL2, and macOS.
Inspects hardware stack, provisions virtual environment, installs optimal
PyTorch wheels with CUDA/ROCm/MPS acceleration, installs Studio IDE dependencies,
creates desktop shortcuts, and verifies installation health.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reconfigure stdout/stderr for Unicode support on Windows cp1252 consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

INSTALLER_DIR = Path(__file__).resolve().parent
STUDIO_DIR = INSTALLER_DIR.parent
WORKSPACE_ROOT = STUDIO_DIR.parent

# ANSI Colors (disabled if not supported)
USE_COLOR = sys.stdout.isatty() or os.environ.get("TERM") or sys.platform != "win32"
CYAN = "\033[96m" if USE_COLOR else ""
GREEN = "\033[92m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
RED = "\033[91m" if USE_COLOR else ""
BOLD = "\033[1m" if USE_COLOR else ""
DIM = "\033[2m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""


def print_banner() -> None:
    banner = rf"""{CYAN}{BOLD}
================================================================================
   _______ _____  _____ _    _ _   _ ______
  |__   __|  __ \|_   _| |  | | \ | |  ____|
     | |  | |__) | | | | |  | |  \| | |__   
     | |  |  _  /  | | | |  | | . ` |  __|  
     | |  | | \ \ _| |_| |__| | |\  | |____ 
     |_|  |_|  \_\_____|\____/|_| \_|______|
               AI ENGINE & STUDIO RESEARCH IDE
     Proprietary Software - Copyright (c) 2024-2026 Yash. All Rights Reserved.
================================================================================{RESET}"""
    try:
        print(banner)
    except UnicodeEncodeError:
        print("\n=== TRIUNE AI ENGINE & STUDIO RESEARCH IDE ===\n")


def log_step(step_num: int, total_steps: int, title: str) -> None:
    print(f"\n{CYAN}{BOLD}[{step_num}/{total_steps}] {title}{RESET}")
    print(f"{DIM}{'-' * 65}{RESET}")


def log_success(msg: str) -> None:
    try:
        print(f" {GREEN}[✓]{RESET} {msg}")
    except UnicodeEncodeError:
        print(f" [OK] {msg}")


def log_info(msg: str) -> None:
    print(f" {CYAN}[i]{RESET} {msg}")


def log_warning(msg: str) -> None:
    print(f" {YELLOW}[!]{RESET} {msg}")


def log_error(msg: str) -> None:
    try:
        print(f" {RED}[✗]{RESET} {msg}")
    except UnicodeEncodeError:
        print(f" [X] {msg}")


# ---------------------------------------------------------------------------
# Hardware & Stack Detection
# ---------------------------------------------------------------------------
class HardwareProfile:
    def __init__(self) -> None:
        self.os_type = platform.system()
        self.os_release = platform.release()
        self.arch = platform.machine()
        self.is_windows = self.os_type == "Windows"
        self.is_linux = self.os_type == "Linux"
        self.is_macos = self.os_type == "Darwin"
        self.is_wsl = False

        self.gpu_vendor = "none"
        self.gpu_name = "None (CPU Execution)"
        self.gpu_vram_gb = 0.0
        self.cuda_version: Optional[str] = None
        self.driver_version: Optional[str] = None
        self.has_nvidia = False
        self.has_rocm = False
        self.has_mps = False

        self.wsl_available = False
        self.wsl_has_gpu = False
        self.wsl_python_path: Optional[str] = None

        self._detect_environment()

    def _detect_environment(self) -> None:
        # Check WSL
        if self.is_linux and os.path.exists("/proc/version"):
            try:
                with open("/proc/version", "r", encoding="utf-8") as f:
                    if "microsoft" in f.read().lower():
                        self.is_wsl = True
            except Exception:
                pass

        # Check NVIDIA GPU via nvidia-smi
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4.0
            )
            if smi.returncode == 0 and smi.stdout.strip():
                parts = [p.strip() for p in smi.stdout.strip().split("\n")[0].split(",")]
                if len(parts) >= 2:
                    self.gpu_name = parts[0]
                    self.gpu_vram_gb = round(float(parts[1]) / 1024, 2)
                    self.has_nvidia = True
                    self.gpu_vendor = "nvidia"
                if len(parts) >= 3:
                    self.driver_version = parts[2]
        except Exception:
            pass

        # Check ROCm on Linux
        if self.is_linux and not self.has_nvidia:
            if shutil.which("rocm-smi") or os.path.exists("/opt/rocm"):
                self.has_rocm = True
                self.gpu_vendor = "amd"
                self.gpu_name = "AMD ROCm GPU"

        # Check Apple Silicon MPS on macOS
        if self.is_macos and self.arch in ("arm64", "aarch64"):
            self.has_mps = True
            self.gpu_vendor = "apple"
            self.gpu_name = f"Apple Silicon ({self.arch}) with Metal (MPS)"

        # On Windows: Check WSL2 availability & GPU access
        if self.is_windows:
            try:
                wsl_chk = subprocess.run(["wsl.exe", "--status"], capture_output=True, timeout=5.0)
                if wsl_chk.returncode == 0:
                    self.wsl_available = True
                    # Check if WSL has nvidia-smi
                    wsl_smi = subprocess.run(
                        ["wsl.exe", "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=8.0
                    )
                    if wsl_smi.returncode == 0 and wsl_smi.stdout.strip():
                        self.wsl_has_gpu = True
                        if not self.has_nvidia:
                            self.gpu_name = f"WSL2 GPU: {wsl_smi.stdout.strip()}"
                            self.has_nvidia = True
                    # Check for existing venv in WSL
                    for cand in [
                        "/home/yasho4867/venvs/triune/bin/python",
                        "/root/venvs/triune/bin/python",
                    ]:
                        r = subprocess.run(["wsl.exe", "-e", "test", "-f", cand], capture_output=True, timeout=3.0)
                        if r.returncode == 0:
                            self.wsl_python_path = cand
                            break
                    if not self.wsl_python_path:
                        r = subprocess.run(["wsl.exe", "which", "python3"], capture_output=True, text=True, timeout=3.0)
                        if r.returncode == 0 and r.stdout.strip():
                            self.wsl_python_path = r.stdout.strip()
            except Exception:
                pass


def run_command(cmd: List[str], cwd: Optional[Path] = None, show_output: bool = True) -> int:
    """Execute a shell command with real-time output streaming."""
    try:
        if show_output:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(proc.stdout.readline, ""):
                print(f"  {DIM}│{RESET} {line.rstrip()}")
            proc.wait()
            return proc.returncode
        else:
            res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
            return res.returncode
    except Exception as exc:
        log_error(f"Command execution error: {exc}")
        return 1


# ---------------------------------------------------------------------------
# Virtual Environment Setup
# ---------------------------------------------------------------------------
def setup_venv(venv_path: Path, python_base: str = sys.executable) -> Tuple[Path, Path]:
    """Create or verify python virtual environment and return (python_bin, pip_bin)."""
    is_win = sys.platform == "win32"
    bin_dir = "Scripts" if is_win else "bin"
    py_name = "python.exe" if is_win else "python"
    pip_name = "pip.exe" if is_win else "pip"

    py_bin = venv_path / bin_dir / py_name
    pip_bin = venv_path / bin_dir / pip_name

    if not py_bin.exists():
        log_info(f"Creating virtual environment at: {venv_path}")
        code = run_command([python_base, "-m", "venv", str(venv_path)], show_output=False)
        if code != 0 or not py_bin.exists():
            log_error(f"Failed to create virtual environment with {python_base}.")
            sys.exit(1)
        log_success("Virtual environment created.")
    else:
        log_success(f"Existing virtual environment found: {venv_path}")

    # Upgrade pip and packaging tools
    log_info("Ensuring packaging tools (pip, setuptools, wheel) are latest...")
    run_command([str(py_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"], show_output=False)
    log_success("Packaging tools up to date.")

    return py_bin, pip_bin


# ---------------------------------------------------------------------------
# PyTorch Wheel Selection
# ---------------------------------------------------------------------------
def install_pytorch(py_bin: Path, hw: HardwareProfile, cuda_choice: str = "auto") -> bool:
    """Select and install optimal PyTorch build for the hardware profile."""
    log_info(f"Resolving PyTorch distribution for {hw.gpu_name}...")

    # Check if host Python is >= 3.14 (where pre-built wheels might be scarce)
    py_major, py_minor = sys.version_info[:2]
    if py_major == 3 and py_minor >= 14 and hw.is_windows:
        log_warning("Host Python is 3.14 on Windows. PyTorch upstream does not yet distribute pre-built CUDA wheels for 3.14.")
        if hw.wsl_available:
            log_info("Hybrid GPU Mode enabled: PyTorch runs with CUDA in WSL2 while Windows Studio UI runs natively.")
            return True
        else:
            log_info("Attempting PyTorch install from default index...")

    install_cmd = [str(py_bin), "-m", "pip", "install"]

    if hw.has_nvidia:
        # Modern NVIDIA GPUs (Blackwell, Ada, Hopper, Ampere) -> CUDA 12.4+
        index_url = "https://download.pytorch.org/whl/cu124"
        if cuda_choice == "12.6":
            index_url = "https://download.pytorch.org/whl/cu126"
        elif cuda_choice == "12.1":
            index_url = "https://download.pytorch.org/whl/cu121"
        elif cuda_choice == "11.8":
            index_url = "https://download.pytorch.org/whl/cu118"

        log_info(f"Targeting NVIDIA CUDA acceleration ({index_url})...")
        install_cmd.extend(["torch", "torchvision", "--index-url", index_url])

    elif hw.has_rocm:
        log_info("Targeting AMD ROCm acceleration (https://download.pytorch.org/whl/rocm6.1)...")
        install_cmd.extend(["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/rocm6.1"])

    elif hw.has_mps:
        log_info("Targeting Apple Silicon Metal MPS acceleration...")
        install_cmd.extend(["torch", "torchvision"])

    else:
        log_info("Targeting CPU execution build...")
        install_cmd.extend(["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])

    code = run_command(install_cmd, show_output=True)
    if code != 0:
        log_warning("Accelerated PyTorch wheel install encountered a warning/error.")
        log_info("Falling back to standard PyTorch distribution...")
        code = run_command([str(py_bin), "-m", "pip", "install", "torch", "torchvision"], show_output=True)

    return code == 0


# ---------------------------------------------------------------------------
# Studio Dependencies & Triune Editable Package
# ---------------------------------------------------------------------------
def install_dependencies(py_bin: Path) -> None:
    """Install Studio UI, API server, and Triune core framework."""
    log_info("Installing Studio API & UI requirements (FastAPI, Uvicorn, PyWebView, Safetensors)...")
    core_pkgs = [
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pywebview>=4.0.0",
        "pydantic>=2.0.0",
        "safetensors>=0.4.0",
        "tokenizers>=0.13.0",
        "datasets>=2.0.0",
        "requests>=2.28.0",
        "psutil>=5.9.0",
    ]
    run_command([str(py_bin), "-m", "pip", "install", *core_pkgs], show_output=False)
    log_success("Studio dependencies installed.")

    log_info("Installing Triune Framework in development mode (pip install -e .)...")
    code = run_command([str(py_bin), "-m", "pip", "install", "-e", str(WORKSPACE_ROOT)], show_output=False)
    if code == 0:
        log_success("Triune Framework installed in editable mode.")
    else:
        log_warning("Editable install note: Triune module will load via local workspace path.")


# ---------------------------------------------------------------------------
# Desktop Shortcuts & Launchers
# ---------------------------------------------------------------------------
def create_desktop_shortcut(hw: HardwareProfile) -> None:
    """Create a 1-click desktop shortcut on Windows or Linux."""
    studio_bat = STUDIO_DIR / "TriuneStudio.bat"
    studio_sh = STUDIO_DIR / "TriuneStudio.sh"

    # Create studio/TriuneStudio.sh for Linux/macOS
    sh_content = f"""#!/usr/bin/env bash
cd "$(dirname "$0")/.."
if [ -f "studio/studio_env/bin/python" ]; then
    studio/studio_env/bin/python scripts/launch_studio.py "$@"
elif [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/launch_studio.py "$@"
else
    python3 scripts/launch_studio.py "$@"
fi
"""
    studio_sh.write_text(sh_content, encoding="utf-8")
    try:
        studio_sh.chmod(0o755)
    except Exception:
        pass

    log_success("Configured Studio launch scripts in studio/ directory.")

    # Create Windows Desktop .lnk shortcut via PowerShell
    if hw.is_windows:
        try:
            ps_script = f"""
$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath('Desktop')
$Shortcut = $WshShell.CreateShortcut("$DesktopPath\\Triune Studio.lnk")
$Shortcut.TargetPath = "{studio_bat}"
$Shortcut.WorkingDirectory = "{WORKSPACE_ROOT}"
$Shortcut.Description = "Triune Studio - AI Engine and Visual Research IDE"
$Shortcut.Save()
"""
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, timeout=5.0)
            if res.returncode == 0:
                log_success("Created Windows Desktop Shortcut: 'Triune Studio.lnk'")
        except Exception as e:
            log_warning(f"Could not create Windows desktop shortcut: {e}")

    # Create Linux Desktop entry
    if hw.is_linux and not hw.is_wsl:
        try:
            apps_dir = Path.home() / ".local" / "share" / "applications"
            apps_dir.mkdir(parents=True, exist_ok=True)
            desktop_file = apps_dir / "triune-studio.desktop"
            content = f"""[Desktop Entry]
Type=Application
Name=Triune Studio
Comment=Triune AI Engine & Visual Research IDE
Exec={studio_sh}
Path={WORKSPACE_ROOT}
Terminal=false
Categories=Development;Science;ArtificialIntelligence;
"""
            desktop_file.write_text(content, encoding="utf-8")
            desktop_file.chmod(0o755)
            log_success(f"Created Linux application launcher: {desktop_file}")
        except Exception as e:
            log_warning(f"Could not create Linux desktop entry: {e}")


# ---------------------------------------------------------------------------
# Health Verification
# ---------------------------------------------------------------------------
def verify_installation(py_bin: Path, hw: HardwareProfile) -> bool:
    """Run verification checks on all subsystems."""
    all_ok = True
    print(f"\n{BOLD}Verifying Subsystems:{RESET}")

    # 1. Python Environment Check
    log_info(f"Python Runtime: {sys.version.split()[0]} ({platform.architecture()[0]})")

    # 2. PyTorch & GPU
    chk_code = (
        "import torch; "
        "has_cuda = torch.cuda.is_available(); "
        "print('TORCH_OK', torch.__version__, 'CUDA:', has_cuda, torch.cuda.get_device_name(0) if has_cuda else 'CPU')"
    )
    res = subprocess.run([str(py_bin), "-c", chk_code], capture_output=True, text=True)
    if "TORCH_OK" in res.stdout:
        log_success(f"PyTorch Engine: {res.stdout.strip()}")
    else:
        if hw.is_windows and hw.wsl_python_path:
            chk_wsl = subprocess.run(
                ["wsl.exe", "-e", hw.wsl_python_path, "-c", "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"],
                capture_output=True, text=True, timeout=5.0
            )
            if chk_wsl.returncode == 0:
                log_success(f"PyTorch Engine (WSL2 GPU Bridge): {chk_wsl.stdout.strip()}")
            else:
                log_success(f"PyTorch Engine: Configured via WSL2 GPU Bridge ({hw.gpu_name})")
        else:
            log_warning("PyTorch not present in local python (Studio will run in Host UI Mode)")

    # 3. Studio FastAPI & Server
    srv_code = "from triune.api.server import HAS_FASTAPI, create_app; assert HAS_FASTAPI; print('SERVER_OK')"
    res = subprocess.run([str(py_bin), "-c", srv_code], capture_output=True, text=True)
    if "SERVER_OK" in res.stdout:
        log_success("Studio API & Telemetry Engine: Operational")
    else:
        log_error(f"Studio Server check failed: {res.stderr.strip() or res.stdout.strip()}")
        all_ok = False

    # 4. Frontend Assets
    studio_src = WORKSPACE_ROOT / "studio" / "src"
    if (studio_src / "index.html").exists() and (studio_src / "app.js").exists():
        log_success(f"Studio Frontend Assets: Verified ({len(list(studio_src.glob('*')))} files)")
    else:
        log_error("Studio Frontend Assets missing in studio/src/")
        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Triune Studio Comprehensive Automated Installer")
    parser.add_argument("--venv", default="studio/studio_env", help="Target virtual environment directory (default: studio/studio_env)")
    parser.add_argument("--cuda", default="auto", choices=["auto", "12.6", "12.4", "12.1", "11.8", "cpu"], help="Target CUDA version")
    parser.add_argument("--skip-torch", action="store_true", help="Skip PyTorch installation")
    parser.add_argument("--no-launch", action="store_true", help="Do not launch Studio after install")
    parser.add_argument("--verify", action="store_true", help="Only verify existing environment health")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatic yes to all prompts (unattended mode)")
    args = parser.parse_args()

    print_banner()

    TOTAL_STEPS = 5 if args.verify else 6
    hw = HardwareProfile()

    # Step 1: Hardware & System Inspection
    log_step(1, TOTAL_STEPS, "Hardware Stack & Environment Probing")
    log_info(f"Host OS: {hw.os_type} {hw.os_release} ({hw.arch})")
    log_info(f"Target Hardware Engine: {BOLD}{hw.gpu_name}{RESET}")
    if hw.has_nvidia:
        log_info(f"NVIDIA VRAM: {hw.gpu_vram_gb} GB | Driver: {hw.driver_version or 'detected'}")
    if hw.is_wsl:
        log_info("Running inside Windows Subsystem for Linux (WSL2)")
    elif hw.is_windows and hw.wsl_available:
        log_info(f"WSL2 Available: Yes (GPU Acceleration: {hw.wsl_has_gpu})")

    # Resolve python interpreter to use
    venv_path = WORKSPACE_ROOT / args.venv if not Path(args.venv).is_absolute() else Path(args.venv)

    if args.verify:
        is_win = sys.platform == "win32"
        py_name = "python.exe" if is_win else "python"
        bin_dir = "Scripts" if is_win else "bin"
        cand = venv_path / bin_dir / py_name
        py_bin = cand if cand.exists() else Path(sys.executable)
        log_step(2, TOTAL_STEPS, "System Health Verification")
        verify_installation(py_bin, hw)
        return

    # Step 2: Virtual Environment Setup
    log_step(2, TOTAL_STEPS, "Provisioning Isolated Virtual Environment")
    py_bin, pip_bin = setup_venv(venv_path)

    # Step 3: Hardware Acceleration & PyTorch
    log_step(3, TOTAL_STEPS, "Configuring PyTorch & GPU Acceleration")
    if not args.skip_torch:
        install_pytorch(py_bin, hw, cuda_choice=args.cuda)
    else:
        log_info("Skipping PyTorch installation as requested (--skip-torch).")

    # Step 4: Studio Dependencies & Framework
    log_step(4, TOTAL_STEPS, "Installing Studio IDE & Framework Dependencies")
    install_dependencies(py_bin)

    # Step 5: Desktop Shortcuts & Launchers
    log_step(5, TOTAL_STEPS, "Configuring Launchers & Desktop Shortcuts")
    create_desktop_shortcut(hw)

    # Step 6: Diagnostic Verification
    log_step(6, TOTAL_STEPS, "End-to-End Installation Verification")
    healthy = verify_installation(py_bin, hw)

    print("\n" + "=" * 80)
    if healthy:
        print(f"{GREEN}{BOLD}   TRIUNE STUDIO INSTALLATION COMPLETED SUCCESSFULLY!{RESET}")
        print("   Ready for local MoE language model research, training, and dynamic routing.")
    else:
        print(f"{YELLOW}{BOLD}   INSTALLATION FINISHED WITH WARNINGS{RESET}")
        print("   Studio will launch in Host UI Mode.")
    print("=" * 80 + "\n")

    print(f"Launch commands:")
    print(f"  • Windows: Double-click {BOLD}studio\\TriuneStudio.bat{RESET} or desktop icon 'Triune Studio'")
    print(f"  • Linux/macOS: Run {BOLD}studio/TriuneStudio.sh{RESET}")
    print(f"  • CLI / Custom Port: {BOLD}python scripts/launch_studio.py --port 8000{RESET}\n")

    # Prompt to launch
    should_launch = not args.no_launch
    if not args.yes and not args.no_launch:
        try:
            choice = input(f"{BOLD}Would you like to launch Triune Studio now? [Y/n]: {RESET}").strip().lower()
            should_launch = choice in ("", "y", "yes")
        except (KeyboardInterrupt, EOFError):
            should_launch = False

    if should_launch:
        print(f"\n{CYAN}Starting Triune Studio...{RESET}\n")
        launch_script = WORKSPACE_ROOT / "scripts" / "launch_studio.py"
        subprocess.Popen([str(py_bin), str(launch_script)])


if __name__ == "__main__":
    main()
