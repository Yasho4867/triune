"""FastAPI Embedded Server for Polysom Studio & Remote Clients."""

from __future__ import annotations

import collections
import hmac
import ipaddress
import os
import sys
import time
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    import fastapi
    import uvicorn
    from fastapi.staticfiles import StaticFiles

    from .routes import router
    from .workspace_routes import workspace_router

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    uvicorn = None
    router = None
    workspace_router = None


def _get_env(name: str, default: str = "") -> str:
    """Read LUCID_MACHINES_<name> first, falling back to TRIUNE_<name>."""
    for prefix in ("LUCID_MACHINES_", "TRIUNE_"):
        val = os.environ.get(f"{prefix}{name}")
        if val is not None and val != "":
            return val
    return default


def _has_proxy_headers(headers: Any) -> bool:
    """Return True if any reverse-proxy forwarding header is present."""
    if not headers:
        return False
    for h in ("forwarded", "x-forwarded-for", "x-real-ip"):
        val = headers.get(h)
        if val is not None and str(val).strip():
            return True
    return False


def _is_test_runtime() -> bool:
    """Return True only when executing inside an active test harness (never in production)."""
    if (
        os.environ.get("LUCID_MACHINES_TESTING") == "0"
        or os.environ.get("TRIUNE_TESTING") == "0"
        or _get_env("ENV").strip().lower() in {"prod", "production"}
    ):
        return False
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or _get_env("TESTING") == "1"


def _is_trusted_loopback_peer(peer: str | None, headers: Any = None) -> bool:
    """Determine if a request originates from a direct local loopback client without proxy headers."""
    if _has_proxy_headers(headers):
        return False
    if _get_env("REQUIRE_AUTH").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if not peer:
        return False
    if peer == "testclient":
        return _is_test_runtime()
    if peer.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(peer).is_loopback
    except ValueError:
        return False


def _is_allowed_host(host_header: str) -> bool:
    """Validate Host header against loopback defaults and LUCID_MACHINES_ALLOWED_HOSTS / TRIUNE_ALLOWED_HOSTS."""
    raw = str(host_header or "").strip()
    if not raw:
        return True
    try:
        parsed_host = urllib.parse.urlsplit(f"//{raw}").hostname or raw.split(":")[0]
    except Exception:
        parsed_host = raw.split(":")[0]
    host_low = parsed_host.strip("[]").lower()
    allowed = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
    if _is_test_runtime():
        allowed.update({"testserver", "testclient"})
    extra = _get_env("ALLOWED_HOSTS")
    if extra:
        for item in extra.split(","):
            clean = item.strip().lower()
            if clean:
                allowed.add(clean.split(":")[0].strip("[]"))
    return host_low in allowed


def _is_allowed_origin(origin_header: str, host_header: str = "") -> bool:
    """Validate Origin header on state-changing HTTP requests and WebSocket handshakes."""
    raw = str(origin_header or "").strip()
    if not raw:
        return True
    if raw.lower() == "null":
        return False
    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False
    origin_host = parsed.hostname.strip("[]").lower()
    allowed_hosts = {"127.0.0.1", "localhost", "::1"}
    if _is_test_runtime():
        allowed_hosts.update({"testserver", "testclient"})
    allowed_full: set[str] = set()
    extra = _get_env("ALLOWED_ORIGINS")
    if extra:
        for item in extra.split(","):
            clean = item.strip()
            if not clean:
                continue
            allowed_full.add(clean.rstrip("/").lower())
            try:
                p_extra = urllib.parse.urlparse(clean)
                if p_extra.hostname:
                    allowed_hosts.add(p_extra.hostname.strip("[]").lower())
                else:
                    allowed_hosts.add(clean.split(":")[0].strip("[]").lower())
            except Exception:
                pass
    return origin_host in allowed_hosts or raw.rstrip("/").lower() in allowed_full


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' data: https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' ws: wss: http://127.0.0.1:* http://localhost:*; "
        "object-src 'none'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ),
}


def _apply_security_headers(response: Any) -> Any:
    for hdr_k, hdr_v in _SECURITY_HEADERS.items():
        if hdr_k not in response.headers:
            response.headers[hdr_k] = hdr_v
    return response


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


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    from .routes import pytorch_state

    if pytorch_state is not None:
        pytorch_state.schedule_cuda_inference_warmup()
    yield
    if pytorch_state is not None:
        pytorch_state.stop_prefetch_worker()


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
        raise RuntimeError("fastapi is required to run the API server. Install with `pip install fastapi uvicorn`")

    from fastapi.responses import FileResponse, JSONResponse

    app = fastapi.FastAPI(
        title="Lucid Machines AI Server",
        description="Embedded API & Telemetry Server powering Polysom Studio",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/healthz", include_in_schema=False)
    async def health_check():
        """Cheap process/liveness check for desktop supervisors and startup probes."""
        # Keep this route deliberately independent of PyTorch, CUDA, model
        # residency, and configuration I/O.  The desktop client uses it to
        # decide whether a localhost process is Lucid Machines before issuing richer
        # diagnostics requests.
        return {"status": "ok", "service": "polysom-studio"}

    _rate_buckets: dict[str, collections.deque] = collections.defaultdict(collections.deque)

    @app.middleware("http")
    async def restrict_remote_api(request, call_next):
        """Enforce Origin/Host validation, payload limits, bearer auth on remote/proxied requests, and security headers."""
        # 1. Request payload size cap
        max_bytes_str = _get_env("MAX_REQUEST_BYTES", str(10 * 1024 * 1024))
        try:
            max_bytes = max(1024, int(max_bytes_str))
        except ValueError:
            max_bytes = 10 * 1024 * 1024
        content_len = request.headers.get("content-length")
        if content_len:
            try:
                if int(content_len) > max_bytes:
                    return _apply_security_headers(
                        JSONResponse(
                            status_code=413,
                            content={"status": "error", "detail": "Request payload exceeds maximum allowed size."},
                        )
                    )
            except ValueError:
                pass

        # 2. Optional configurable per-client rate limit
        rate_limit_str = _get_env("RATE_LIMIT_PER_MIN", "")
        if rate_limit_str:
            try:
                rate_limit = int(rate_limit_str)
            except ValueError:
                rate_limit = 0
            if rate_limit > 0:
                xff = request.headers.get("x-forwarded-for", "")
                first_xff = xff.split(",")[0].strip() if xff else ""
                peer_key = (
                    first_xff
                    or request.headers.get("x-real-ip")
                    or (request.client.host if request.client else "unknown")
                )[:128]
                now = time.monotonic()
                if len(_rate_buckets) >= 4096 and peer_key not in _rate_buckets:
                    stale_keys = [k for k, b in _rate_buckets.items() if not b or now - b[-1] > 60.0]
                    for k in stale_keys[:1024]:
                        _rate_buckets.pop(k, None)
                    if len(_rate_buckets) >= 4096:
                        _rate_buckets.pop(next(iter(_rate_buckets)), None)
                bucket = _rate_buckets[str(peer_key)]
                while bucket and now - bucket[0] > 60.0:
                    bucket.popleft()
                if len(bucket) >= rate_limit:
                    return _apply_security_headers(
                        JSONResponse(
                            status_code=429,
                            content={"status": "error", "detail": "Rate limit exceeded. Try again later."},
                        )
                    )
                bucket.append(now)

        # 3. Validate Host header
        host_hdr = request.headers.get("host", "")
        if host_hdr and not _is_allowed_host(host_hdr):
            return _apply_security_headers(
                JSONResponse(
                    status_code=403,
                    content={"status": "error", "detail": "Untrusted Host header rejected."},
                )
            )

        # 4. Validate Origin header on state-changing HTTP methods
        if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            origin_hdr = request.headers.get("origin", "")
            if origin_hdr and not _is_allowed_origin(origin_hdr, host_hdr):
                return _apply_security_headers(
                    JSONResponse(
                        status_code=403,
                        content={"status": "error", "detail": "Cross-origin request rejected by Origin policy."},
                    )
                )

        # 5. Require bearer token when non-loopback, reverse-proxied, or REQUIRE_AUTH=1
        if request.url.path.startswith(("/v1/", "/api/")):
            peer = request.client.host if request.client else None
            is_loopback = _is_trusted_loopback_peer(peer, request.headers)
            if not is_loopback:
                api_token = _get_env("API_TOKEN", "")
                if len(api_token) < 32:
                    return _apply_security_headers(
                        JSONResponse(
                            status_code=503,
                            content={
                                "detail": (
                                    "Remote API is unavailable: configure LUCID_MACHINES_API_TOKEN (or TRIUNE_API_TOKEN) "
                                    "with at least 32 characters."
                                )
                            },
                        )
                    )
                supplied = request.headers.get("authorization", "")
                scheme, _, credential = supplied.partition(" ")
                if scheme.lower() != "bearer" or not hmac.compare_digest(credential, api_token):
                    return _apply_security_headers(
                        JSONResponse(
                            status_code=401,
                            headers={"WWW-Authenticate": "Bearer"},
                            content={"detail": "A valid bearer token is required for remote or proxied API access."},
                        )
                    )
        response = await call_next(request)
        return _apply_security_headers(response)

    # 1. Include all API routes FIRST – these take priority.
    app.include_router(router)
    if workspace_router is not None:
        app.include_router(workspace_router)

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
        no_cache_headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        @app.get("/")
        async def serve_root():
            return FileResponse(str(index_path), headers=no_cache_headers)

        @app.get("/{full_path:path}")
        async def serve_spa_fallback(full_path: str):
            # Try to serve the exact file first (e.g. /style.css)
            try:
                candidate = (studio_src / full_path).resolve(strict=True)
                candidate.relative_to(studio_src.resolve())
                if candidate.is_file():
                    return FileResponse(str(candidate), headers=no_cache_headers)
            except (OSError, RuntimeError, ValueError):
                pass
            # Otherwise fall back to index.html
            return FileResponse(str(index_path), headers=no_cache_headers)

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
        raise RuntimeError(
            "fastapi and uvicorn are required to run the API server. Install with `pip install fastapi uvicorn`"
        )
    import uvicorn

    try:
        is_loopback_host = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback_host = False
    tls_cert = _get_env("TLS_CERTFILE", "")
    tls_key = _get_env("TLS_KEYFILE", "")
    if not is_loopback_host:
        if len(_get_env("API_TOKEN", "")) < 32:
            raise RuntimeError(
                "Binding to a remote interface requires LUCID_MACHINES_API_TOKEN (or TRIUNE_API_TOKEN) with at least 32 characters."
            )
        if not tls_cert or not tls_key or not Path(tls_cert).is_file() or not Path(tls_key).is_file():
            raise RuntimeError(
                "Binding to a remote interface requires valid LUCID_MACHINES_TLS_CERTFILE and LUCID_MACHINES_TLS_KEYFILE files. "
                "Alternatively, keep the server on loopback and expose it through an authenticated HTTPS proxy with LUCID_MACHINES_API_TOKEN set."
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
