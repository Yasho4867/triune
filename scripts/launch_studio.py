"""Native Desktop Launcher for Triune Studio."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(_stream, "reconfigure"):
                _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triune.desktop import launch_desktop_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triune Studio Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind Studio on (default: 8000)")
    parser.add_argument("--browser", action="store_true", help="Launch directly in default web browser instead of desktop window")
    parser.add_argument("--native-host", action="store_true", help="Start the backend for the native Windows desktop host without opening a browser window")
    parser.add_argument("--use-wsl", action="store_true", help="Explicitly allow the non-native Windows launcher to use a CUDA-enabled WSL backend")
    args = parser.parse_args()

    launch_desktop_app(port=args.port, use_browser=args.browser, native_host=args.native_host, allow_wsl=args.use_wsl)
