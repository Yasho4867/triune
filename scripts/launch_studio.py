"""Native Desktop Launcher for Triune Studio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triune.desktop import launch_desktop_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triune Studio Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind Studio on (default: 8000)")
    parser.add_argument("--browser", action="store_true", help="Launch directly in default web browser instead of desktop window")
    args = parser.parse_args()

    launch_desktop_app(port=args.port, use_browser=args.browser)
