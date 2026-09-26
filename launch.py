#!/usr/bin/env python3
"""Root launcher for Triune Studio."""

import argparse
import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from triune.desktop import launch_desktop_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triune Studio Desktop Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Local port for Studio API & UI (default: 8000)")
    args = parser.parse_args()

    launch_desktop_app(port=args.port)
