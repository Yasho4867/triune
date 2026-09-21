"""Standalone Desktop Window Launcher for Triune Studio App.

Architecture:
  1. Start a static HTTP file server IMMEDIATELY for instant UI
  2. Spawn studio_env\\Scripts\\python.exe to run the Triune API server
     as a SUBPROCESS (torch works fine in the venv, just not inside PyInstaller)
  3. PyWebView opens the UI window
"""

from __future__ import annotations

import http.server
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


# ---------------------------------------------------------------------------
# Log file redirection (frozen exe only)
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    try:
        log_file = Path(sys.executable).parent / "triune_studio.log"
        log_fp = open(log_file, "a", encoding="utf-8", buffering=1)
        sys.stdout = log_fp
        sys.stderr = log_fp
        print(f"\n--- [Triune Studio Session Started at {time.ctime()}] ---")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def find_free_port(start: int = 8005) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def wait_for_server(url: str, timeout: float = 10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def find_workspace() -> Path:
    """Resolve true workspace root directory."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        for cand in [exe_dir.parent.parent, exe_dir.parent, exe_dir, Path.cwd()]:
            if (cand / "triune").exists():
                return cand
    workspace = Path(__file__).resolve().parent.parent.parent
    if (workspace / "triune").exists():
        return workspace
    return Path.cwd()


# ---------------------------------------------------------------------------
# Static HTTP file server (serves index.html / style.css / app.js)
# ---------------------------------------------------------------------------
def start_static_server(directory: Path, port: int) -> None:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, fmt, *args):
            pass

    def serve():
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", port), QuietHandler) as httpd:
            print(f"[Static Server] Serving UI from {directory} on http://127.0.0.1:{port}")
            httpd.serve_forever()

    threading.Thread(target=serve, daemon=True).start()


# ---------------------------------------------------------------------------
# Find the venv python executable (studio_env or active interpreter)
# ---------------------------------------------------------------------------
def find_venv_python() -> Path | None:
    """Find a Python interpreter with torch installed (Windows or Linux)."""
    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    workspace = find_workspace()

    is_win = sys.platform == "win32"
    bin_dir = "Scripts" if is_win else "bin"
    py_name = "python.exe" if is_win else "python"

    candidates = [
        exe_dir / "studio_env" / bin_dir / py_name,
        exe_dir.parent / "studio_env" / bin_dir / py_name,
        workspace / "studio_env" / bin_dir / py_name,
        workspace / ".venv" / bin_dir / py_name,
        Path(sys.executable),
    ]
    for p in candidates:
        if p.exists():
            print(f"[Launcher] Found venv python: {p}")
            return p

    print("[Launcher] WARNING: Could not find venv python executable")
    for p in candidates:
        print(f"  Checked: {p} (exists={p.exists()})")
    return None


# ---------------------------------------------------------------------------
# Start the API backend server as a subprocess
# ---------------------------------------------------------------------------
def start_api_server(venv_python: Path, port: int) -> subprocess.Popen | None:
    """Spawn the Triune API server using the venv python (where torch works)."""
    workspace = find_workspace()
    ws_path = str(workspace)

    env = os.environ.copy()
    env["PYTHONPATH"] = ws_path + os.pathsep + env.get("PYTHONPATH", "")

    server_script = (
        f"import sys; "
        f"sys.path.insert(0, r'{ws_path}'); "
        f"from triune.api import run_server; run_server(port={port})"
    )

    try:
        proc = subprocess.Popen(
            [str(venv_python), "-c", server_script],
            cwd=str(workspace),
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        print(f"[API Server] Spawned venv python (PID {proc.pid}) on port {port}")
        return proc
    except Exception as e:
        print(f"[API Server] Failed to start: {e}")
        return None


# ---------------------------------------------------------------------------
# Resolve static UI assets directory
# ---------------------------------------------------------------------------
def find_studio_src() -> Path | None:
    workspace = find_workspace()

    candidates = []
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidates += [
            meipass / "src",
            meipass / "_internal" / "src",
            Path(sys.executable).parent / "src",
            Path(sys.executable).parent / "_internal" / "src",
        ]

    candidates += [
        workspace / "studio" / "src",
        workspace / "studio",
        Path.cwd() / "studio" / "src",
        Path.cwd() / "studio",
    ]

    for d in candidates:
        if (d / "index.html").exists():
            print(f"[Launcher] Found UI assets at: {d}")
            return d

    print("[Launcher] WARNING: Could not find index.html!")
    for d in candidates:
        print(f"  Checked: {d} (exists={d.exists()})")
    return None


# ---------------------------------------------------------------------------
# Main launch
# ---------------------------------------------------------------------------
def launch() -> None:
    ui_port = find_free_port(8005)
    api_port = find_free_port(ui_port + 1)
    studio_src = find_studio_src()

    if studio_src is None:
        print("[FATAL] No UI assets found. Cannot launch.")
        return

    # STEP 1: Start the static HTTP server IMMEDIATELY — UI loads in <1s
    start_static_server(studio_src, ui_port)
    target_url = f"http://127.0.0.1:{ui_port}/"
    wait_for_server(target_url, timeout=3.0)
    print(f"[Launcher] UI ready at {target_url}")

    # STEP 2: Start the API backend via venv python subprocess
    api_proc = None
    venv_py = find_venv_python()
    if venv_py:
        api_proc = start_api_server(venv_py, api_port)
        if api_proc:
            api_url = f"http://127.0.0.1:{api_port}/v1/system/diagnostics"
            if wait_for_server(api_url, timeout=15.0):
                print(f"[Launcher] API PyTorch Engine server ready on port {api_port}")
            else:
                print(f"[Launcher] API server starting on port {api_port} (UI will auto-discover)")
    else:
        print("[Launcher] No venv python found — UI-only mode")

    # STEP 3: Open PyWebView window
    try:
        import webview

        print(f"[Launcher] Opening PyWebView → {target_url}")
        window = webview.create_window(
            title="Triune Studio - AI Engine & Research IDE",
            url=target_url,
            width=1360,
            height=880,
            resizable=True,
            min_size=(900, 600),
        )

        def on_closed():
            print("[Launcher] Window closed. Cleaning up...")
            if api_proc and api_proc.poll() is None:
                api_proc.terminate()
            os._exit(0)

        window.events.closed += on_closed
        webview.start()
        os._exit(0)
    except Exception as e:
        print(f"[Launcher] PyWebView unavailable ({e}), opening browser...")
        webbrowser.open(target_url)


if __name__ == "__main__":
    try:
        launch()
    finally:
        os._exit(0)
