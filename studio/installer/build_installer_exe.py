"""Compiles installer.py into a single Standalone Windows Setup Executable (TriuneStudioSetup.exe)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def build_installer_exe() -> None:
    print("=" * 65)
    print("   COMPILING STANDALONE WINDOWS SETUP EXECUTABLE (TriuneStudioSetup.exe)")
    print("=" * 65)

    installer_dir = Path(__file__).resolve().parent
    studio_dir = installer_dir.parent
    venv_python = studio_dir / "studio_env" / "Scripts" / "python.exe"
    python_bin = venv_python if venv_python.exists() else Path(sys.executable)

    # Run PyInstaller --onefile to produce TriuneStudioSetup.exe
    cmd = [
        str(python_bin),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--name=TriuneStudioSetup",
        str(installer_dir / "installer.py"),
    ]

    try:
        subprocess.check_call(cmd, cwd=installer_dir)
        setup_exe = installer_dir / "dist" / "TriuneStudioSetup.exe"
        print("\n" + "=" * 65)
        print("  STANDALONE WINDOWS INSTALLER CREATED SUCCESSFULLY!")
        print(f"  Path: {setup_exe}")
        print("  Double-clicking this .exe on Windows 11 installs Triune Studio")
        print("  and launches TriuneStudio.exe!")
        print("=" * 65 + "\n")
    except Exception as error:
        print(f"PyInstaller setup build error: {error}")


if __name__ == "__main__":
    build_installer_exe()
