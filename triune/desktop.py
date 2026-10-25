"""Desktop Application Launcher for Triune Studio.

Seamlessly launches the embedded Triune API & UI server and opens a native desktop
application window across Windows, Linux, WSL2, and macOS.
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


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Find the next available TCP port."""
    for p in range(start_port, start_port + max_attempts):
        if not _is_port_in_use(p):
            return p
    return start_port


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    """Poll a URL until it responds with HTTP 200 or timeout occurs."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=0.8) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def _is_triune_server_alive(port: int) -> bool:
    """Check if an active Triune Studio server is responding on this port."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    return False


def _find_wsl_python() -> str | None:
    """Find a WSL Python with CUDA only when a caller explicitly opts into WSL."""
    if sys.platform != "win32":
        return None
    try:
        script = (
            'for p in "$HOME/venvs/triune/bin/python" "$HOME/.venv/bin/python"; do '
            'if [ -x "$p" ] && "$p" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" '
            '>/dev/null 2>&1; then echo "$p"; exit 0; fi; done; '
            'if command -v python3 >/dev/null 2>&1 && python3 -c '
            '"import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; '
            'then command -v python3; fi'
        )
        result = subprocess.run(
            ["wsl.exe", "bash", "-lc", script], capture_output=True, text=True, timeout=15.0
        )
        candidate = result.stdout.strip().splitlines()
        if result.returncode == 0 and candidate:
            return candidate[-1]
    except Exception:
        pass
    return None


def _to_wsl_path(path: Path) -> str:
    """Convert a Windows Path to a WSL POSIX mount path."""
    p_posix = path.resolve().as_posix()
    drive = p_posix[0].lower()
    rest = p_posix[2:]  # Strip 'C:'
    return f"/mnt/{drive}{rest}"


def _start_fallback_http_server(directory: Path, port: int) -> None:
    """Run built-in Python HTTP server if FastAPI is completely unavailable."""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, fmt, *args):
            pass

    def serve():
        socketserver.TCPServer.allow_reuse_address = True
        try:
            with socketserver.TCPServer(("127.0.0.1", port), QuietHandler) as httpd:
                print(f"[Studio Fallback] Serving static UI on http://127.0.0.1:{port}")
                httpd.serve_forever()
        except Exception as err:
            print(f"[Studio Fallback Error] {err}")

    t = threading.Thread(target=serve, daemon=True)
    t.start()


def launch_desktop_app(
    port: int = 8000,
    use_browser: bool = False,
    native_host: bool = False,
    allow_wsl: bool = False,
) -> None:
    """Launch embedded Triune Studio API server and open as a native desktop application window."""
    workspace_root = Path(__file__).resolve().parent.parent
    studio_src = workspace_root / "studio" / "src"

    # Check if a Triune Studio server is already alive on this port
    server_already_running = False
    if _is_port_in_use(port):
        if native_host:
            # The native shell must start the backend with its current runtime
            # environment (including the Windows credential-store bridge).
            # Reusing an old API process here makes UI updates appear active
            # while operations such as BYOK save still run stale code.
            raise RuntimeError(
                f"Studio port {port} is already in use. Close the existing Studio window or choose another port, then retry."
            )
        if _is_triune_server_alive(port):
            print(f"[Studio] Connected to existing Triune Studio backend on http://127.0.0.1:{port}/")
            server_already_running = True
        else:
            # Port is occupied by a foreign application; allocate the next free port
            free_p = _find_free_port(start_port=port)
            if free_p != port:
                print(f"[Studio] Port {port} is occupied by another application. Reallocating to port {free_p}...")
                port = free_p

    # 1. Determine optimal backend execution mode
    local_has_torch = False
    local_has_cuda = False
    try:
        import torch
        local_has_torch = True
        local_has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # Windows Studio is self-contained around the native managed runtime by
    # default. WSL is an explicit opt-in, never an invisible fallback.
    wsl_py = None
    if allow_wsl and not native_host and sys.platform == "win32" and not local_has_cuda:
        wsl_py = _find_wsl_python()

    api_proc = None

    if not server_already_running:
        if local_has_cuda:
            # Native Python with CUDA PyTorch
            print(f"[Studio] Starting native PyTorch CUDA backend on http://127.0.0.1:{port}...")
            from triune.api import run_server
            server_thread = threading.Thread(
                target=run_server,
                kwargs={"host": "127.0.0.1", "port": port},
                daemon=True,
            )
            server_thread.start()

        elif wsl_py:
            # Windows host bridging to high-performance WSL2 GPU engine
            wsl_ws = _to_wsl_path(workspace_root)
            windows_python = _to_wsl_path(Path(sys.executable))
            print(f"[Studio] Hardware Bridge: Launching backend in WSL2 with GPU acceleration ({wsl_py})...")
            # The API and model engine run in WSL, but user credentials belong in
            # the Windows user's Credential Manager. Pass the already-provisioned
            # Windows Studio interpreter to the WSL backend as a vault bridge;
            # no Linux keyring package or desktop keyring daemon is required.
            wsl_startup = (
                "import os,sys; "
                f"os.environ['TRIUNE_WINDOWS_PYTHON']={windows_python!r}; "
                f"sys.path.insert(0, {wsl_ws!r}); "
                f"from triune.api import run_server; run_server(host='127.0.0.1', port={port})"
            )
            wsl_cmd = [
                "wsl.exe", "-e", wsl_py, "-c",
                # WSL localhost forwarding exposes this guest-loopback service
                # to the Windows host without opening it on the LAN. A wildcard
                # bind is intentionally rejected by run_server unless TLS and
                # an API token are configured.
                wsl_startup
            ]
            try:
                api_proc = subprocess.Popen(
                    wsl_cmd,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )
            except Exception as e:
                print(f"[Studio] Failed to launch WSL backend: {e}")

        elif local_has_torch:
            # Native host Python with CPU PyTorch
            print(f"[Studio] Starting native PyTorch CPU backend on http://127.0.0.1:{port}...")
            from triune.api import run_server
            server_thread = threading.Thread(
                target=run_server,
                kwargs={"host": "127.0.0.1", "port": port},
                daemon=True,
            )
            server_thread.start()

        else:
            # Native host Python in lightweight UI Mode or Fallback
            try:
                from triune.api import run_server
                print(f"[Studio] Starting Triune Studio in Host UI Mode on http://127.0.0.1:{port}...")
                server_thread = threading.Thread(
                    target=run_server,
                    kwargs={"host": "127.0.0.1", "port": port},
                    daemon=True,
                )
                server_thread.start()
            except Exception:
                print(f"[Studio] Starting built-in fallback HTTP server on port {port}...")
                _start_fallback_http_server(studio_src, port)

    url = f"http://127.0.0.1:{port}/"

    # 2. Wait for server to be responsive BEFORE opening browser/webview
    if not server_already_running:
        print(f"[Studio] Waiting for server to initialize on {url}...")
        # The root path can return HTTP 200 from an unrelated local service if
        # another process wins a port-allocation race. Require Triune's own API
        # health route before opening the desktop window.
        health_url = f"http://127.0.0.1:{port}/healthz"
        # WSL may need to wake its VM and import the CUDA/PyTorch stack on a
        # cold start. The native Windows shell allows longer than this same
        # deadline so it can display the backend's real startup error if it
        # exits, instead of killing a healthy but still initializing process.
        startup_timeout = 90.0 if wsl_py else 30.0
        server_ready = _wait_for_server(health_url, timeout=startup_timeout)
        if server_ready:
            print(f"[Studio] Server ready and responsive on {url}!")
        else:
            if api_proc and api_proc.poll() is None:
                api_proc.terminate()
                try:
                    api_proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    api_proc.kill()
            raise RuntimeError(
                f"Triune Studio backend did not become ready at {url} within {startup_timeout:g} seconds. "
                "The desktop window was not opened; check the backend startup error above."
            )

    # A native Windows WPF host embeds the Studio UI in WebView2 and owns the
    # backend lifecycle; it never delegates the backend to WSL.
    if native_host:
        print(f"[Studio] Backend ready for native desktop host at {url}.")
        consecutive_health_failures = 0
        try:
            while True:
                if api_proc and api_proc.poll() is not None:
                    raise RuntimeError("Triune backend process exited unexpectedly.")
                if not _is_triune_server_alive(port):
                    consecutive_health_failures += 1
                    if consecutive_health_failures >= 5:
                        raise RuntimeError(
                            "Triune backend failed five consecutive health checks; restarting the desktop backend supervisor."
                        )
                else:
                    consecutive_health_failures = 0
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            if api_proc and api_proc.poll() is None:
                api_proc.terminate()
                try:
                    api_proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    api_proc.kill()
        return

    # 3. Launch native desktop window via PyWebView
    if not use_browser:
        try:
            import webview

            print("[Studio] Opening native desktop window via PyWebView...")
            window = webview.create_window(
                title="Triune Studio - AI Engine & Research IDE",
                url=url,
                width=1380,
                height=900,
                resizable=True,
                min_size=(960, 640),
            )

            def on_closed():
                print("[Studio] Window closed. Shutting down background processes...")
                if api_proc and api_proc.poll() is None:
                    api_proc.terminate()
                os._exit(0)

            window.events.closed += on_closed
            webview.start()
            if api_proc and api_proc.poll() is None:
                api_proc.terminate()
            return
        except Exception as err:
            print(f"[Studio] PyWebView note: {err}")

        # 4. Windows: Try MS Edge App Mode
        if sys.platform == "win32":
            try:
                edge_paths = [
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                ]
                for edge_path in edge_paths:
                    if os.path.exists(edge_path):
                        print("[Studio] Opening in Microsoft Edge App Mode...")
                        proc = subprocess.Popen([edge_path, f"--app={url}", "--name=Triune Studio"])
                        time.sleep(1.0)
                        if proc.poll() is not None:
                            # Edge delegated to existing browser instance; keep server running
                            print(f"[Studio] Triune Studio is running at {url}. Press Ctrl+C to stop.")
                            try:
                                if api_proc:
                                    api_proc.wait()
                                else:
                                    while True:
                                        time.sleep(1.0)
                            except KeyboardInterrupt:
                                print("\n[Studio] Shutting down...")
                            finally:
                                if api_proc and api_proc.poll() is None:
                                    api_proc.terminate()
                            return
                        else:
                            try:
                                proc.wait()
                            finally:
                                if api_proc and api_proc.poll() is None:
                                    api_proc.terminate()
                            return
            except Exception:
                pass

        # 5. Linux / macOS: Try Chrome / Chromium App Mode
        if sys.platform != "win32":
            for browser in ["google-chrome", "chromium", "chromium-browser"]:
                try:
                    proc = subprocess.Popen([browser, f"--app={url}"])
                    try:
                        proc.wait()
                    finally:
                        if api_proc and api_proc.poll() is None:
                            api_proc.terminate()
                    return
                except FileNotFoundError:
                    continue

    # 6. Fallback or explicit browser mode: Default web browser
    print(f"[Studio] Opening in default web browser: {url}")
    webbrowser.open(url)
    print(f"[Studio] Triune Studio is running at {url}. Press Ctrl+C to stop.")
    try:
        if api_proc:
            api_proc.wait()
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Studio] Shutting down...")
    finally:
        if api_proc and api_proc.poll() is None:
            api_proc.terminate()
