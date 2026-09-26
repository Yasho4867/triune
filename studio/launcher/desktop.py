"""Desktop launcher forwarder for Triune Studio.

Delegates directly to canonical desktop launcher engine in :mod:`triune.desktop`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
workspace = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace))

from triune.desktop import launch_desktop_app

if __name__ == "__main__":
    launch_desktop_app()
