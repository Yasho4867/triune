"""Desktop Application Launcher for Triune Studio."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser


def launch_desktop_app(port: int = 8000) -> None:
    """Launch embedded Triune Studio API server and open as a native desktop application window."""
    from triune.api import run_server

    print(f"[Studio] Starting Triune Framework Engine on http://127.0.0.1:{port}")
    server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": "127.0.0.1", "port": port},
        daemon=True,
    )
    server_thread.start()

    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}/"

    # 1. Try PyWebView for a desktop window (Windows, macOS, Linux)
    try:
        import webview

        print("[Studio] Launching native window via PyWebView...")
        webview.create_window(
            title="Triune Studio",
            url=url,
            width=1340,
            height=880,
            resizable=True,
            min_size=(900, 600),
        )
        webview.start()
        return
    except ImportError:
        pass

    # 2. Windows: Try MS Edge App Mode
    if sys.platform == "win32":
        try:
            edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
            if not os.path.exists(edge_path):
                edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

            if os.path.exists(edge_path):
                print("[Studio] Launching MS Edge application mode...")
                subprocess.Popen([edge_path, f"--app={url}", "--name=Triune Studio"])
                return
        except Exception:
            pass

    # 3. Fallback to default system browser
    print(f"[Studio] Opening in browser: {url}")
    webbrowser.open(url)
