"""FastAPI Embedded Server for Triune Studio & Remote Clients."""

from __future__ import annotations

import sys
import ipaddress
import os
import hmac
from pathlib import Path

try:
    import fastapi
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from .routes import router

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    uvicorn = None
    router = None


def _resolve_studio_src() -> "Path | None":
    """Find the studio static assets directory."""
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        for cand in [meipass / "src", meipass / "_internal" / "src", Path(sys.executable).parent / "src"]:
            if cand.is_dir():
                return cand

    workspace_root = Path(__file__).resolve().parent.parent.parent
    for cand in [
        workspace_root / "studio" / "src",
        workspace_root / "studio",
        Path.cwd() / "studio" / "src",
        Path.cwd() / "studio",
    ]:
        if cand.is_dir():
            return cand
    return None


def create_app():
    """Create and configure FastAPI application for Triune Engine.

    IMPORTANT: The API router is included on the app directly so all
    ``/v1/*``, ``/api/*``, and ``/ws/*`` routes are registered as
    first-class path operations.  Static files are mounted under a
    ``/static`` sub-path so they can never shadow API endpoints.
    A final catch-all GET route serves ``index.html`` for any
    unmatched path so the single-page app still works.
    """
    if not HAS_FASTAPI:
        raise RuntimeError(
            "fastapi is required to run the API server. "
            "Install with `pip install fastapi uvicorn`"
        )

    from fastapi.responses import FileResponse, JSONResponse

    app = fastapi.FastAPI(
        title="Triune Framework Server",
        description="Embedded API & Telemetry Server powering Triune Studio",
        version="2.0.0",
    )

    @app.get("/healthz", include_in_schema=False)
    async def health_check():
        """Cheap process/liveness check for desktop supervisors and startup probes."""
        return {"status": "ok"}

    @app.middleware("http")
    async def restrict_remote_api(request, call_next):
        """Require a configured bearer token for direct remote API access."""
        if request.url.path.startswith(("/v1/", "/api/")):
            peer = request.client.host if request.client else None
            try:
                is_loopback = peer is not None and ipaddress.ip_address(peer).is_loopback
            except ValueError:
                is_loopback = False
            if not is_loopback:
                api_token = os.environ.get("TRIUNE_API_TOKEN", "")
                if len(api_token) < 32:
                    return JSONResponse(status_code=503, content={"detail": "Remote API is unavailable: configure TRIUNE_API_TOKEN with at least 32 characters."})
                supplied = request.headers.get("authorization", "")
                scheme, _, credential = supplied.partition(" ")
                if scheme.lower() != "bearer" or not hmac.compare_digest(credential, api_token):
                    return JSONResponse(
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                        content={"detail": "A valid bearer token is required for remote API access."},
                    )
        return await call_next(request)

    # 1. Include all API routes FIRST – these take priority.
    app.include_router(router)

    # 2. Resolve static assets directory.
    studio_src = _resolve_studio_src()

    if studio_src and studio_src.is_dir():
        print(f"[Studio UI] Mounting static assets from: {studio_src}")
        # Mount under /static so JS/CSS/images are accessible without
        # colliding with /v1 or /api paths.
        app.mount("/static", StaticFiles(directory=str(studio_src)), name="studio_static")

        # Catch-all: serve index.html for the root and any unmatched
        # path (SPA fallback).  Because this is registered AFTER the
        # router, API routes always win.
        index_path = studio_src / "index.html"

        @app.get("/")
        async def serve_root():
            return FileResponse(str(index_path))

        @app.get("/{full_path:path}")
        async def serve_spa_fallback(full_path: str):
            # Try to serve the exact file first (e.g. /style.css)
            try:
                candidate = (studio_src / full_path).resolve(strict=True)
                candidate.relative_to(studio_src.resolve())
                if candidate.is_file():
                    return FileResponse(str(candidate))
            except (OSError, RuntimeError, ValueError):
                pass
            # Otherwise fall back to index.html
            return FileResponse(str(index_path))

    return app


if HAS_FASTAPI and uvicorn is not None:
    class SafeServer(uvicorn.Server):
        """Custom Uvicorn Server that safely bypasses signal handler registration when run in background threads."""
        def install_signal_handlers(self) -> None:
            import threading
            if threading.current_thread() is threading.main_thread():
                try:
                    super().install_signal_handlers()
                except Exception:
                    pass
else:
    SafeServer = object


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch embedded API server using Uvicorn safely across main or background threads."""
    if not HAS_FASTAPI:
        raise RuntimeError("fastapi and uvicorn are required to run the API server. Install with `pip install fastapi uvicorn`")
    import uvicorn

    try:
        is_loopback_host = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback_host = False
    tls_cert = os.environ.get("TRIUNE_TLS_CERTFILE", "")
    tls_key = os.environ.get("TRIUNE_TLS_KEYFILE", "")
    if not is_loopback_host:
        if len(os.environ.get("TRIUNE_API_TOKEN", "")) < 32:
            raise RuntimeError("Binding to a remote interface requires TRIUNE_API_TOKEN with at least 32 characters.")
        if not tls_cert or not tls_key or not Path(tls_cert).is_file() or not Path(tls_key).is_file():
            raise RuntimeError(
                "Binding to a remote interface requires valid TRIUNE_TLS_CERTFILE and TRIUNE_TLS_KEYFILE files. "
                "Alternatively, keep Triune on loopback and expose it through an authenticated HTTPS proxy or Cloudflare Tunnel."
            )

    try:
        app = create_app()
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            ssl_certfile=tls_cert or None,
            ssl_keyfile=tls_key or None,
        )
        server = SafeServer(config)
        print(f"[Triune API Server] Starting uvicorn server on http://{host}:{port}...")
        server.run()
    except Exception as err:
        print(f"[Triune API Server ERROR] Failed to start uvicorn server: {err}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_server()

