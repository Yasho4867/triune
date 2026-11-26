"""Shared harness, CDP browser client, CLI/API/Tectonic/Website helpers for tests/e2e_suite/."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple

CHROME_CANDIDATES = [
    os.environ.get("CHROME_PATH", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

STUDIO_TABS: List[Tuple[int, str, str]] = [
    (0, "chat", "Chat & Playground"),
    (1, "workspace", "Workspace & Agent IDE"),
    (2, "nodegraph", "Visual Node Canvas"),
    (3, "training", "Training & Telemetry"),
    (4, "checkpoints", "Checkpoints & Weights"),
    (5, "finetune", "LoRA / QLoRA Tuner"),
    (6, "modules", "Modules & Repos"),
    (7, "customize", "Appearance & Themes"),
    (8, "environment", "System & Hardware"),
    (9, "models", "Model Zoo & Exporters"),
    (10, "datasets", "Dataset & Tokenizer"),
    (11, "notebook", "Python Sandbox"),
    (12, "byok", "BYOK Subscriptions"),
]

STUDIO_THEMES: List[str] = [
    "hypnagogic",
    "phosphene",
    "pareidolia",
    "afterimage",
    "mirage",
    "euphoria",
]


def find_chrome_binary() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return candidate
    which_chrome = shutil.which("chrome") or shutil.which("google-chrome")
    if which_chrome:
        return which_chrome
    raise FileNotFoundError(
        "Google Chrome executable not found. Set CHROME_PATH or install Chrome at "
        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    )


def find_tectonic_binary() -> str:
    candidates = [
        os.environ.get("TECTONIC_PATH", ""),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "tectonic", "tectonic.exe"),
        shutil.which("tectonic") or "",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "tectonic.exe not found. Expected at %LOCALAPPDATA%\\tectonic\\tectonic.exe or on PATH."
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def wait_http_ok(url: str, timeout: float = 45.0) -> Any:
    deadline = time.monotonic() + timeout
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    raw = resp.read().decode("utf-8")
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return raw
        except Exception as exc:
            last_err = exc
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for HTTP 200 at {url}: {last_err}")


def terminate_process(proc: Optional[subprocess.Popen[Any]], timeout: float = 5.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


# ---------------------------------------------------------------------------
# R8 Preservation Snapshot Helpers
# ---------------------------------------------------------------------------


def capture_r8_snapshot(repo_root: Path) -> Dict[str, Any]:
    session_file = repo_root / "checkpoints" / "session_state.json"
    rescued_ckpt = repo_root / "checkpoints" / "triune_studio_step_36500_rescued.pt"
    tokenizer_file = repo_root / "triune_tokenizer.json"

    data_entries = {}
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        for p in sorted(data_dir.iterdir()):
            if p.is_file():
                data_entries[p.name] = p.stat().st_size

    export_entries = {}
    exports_dir = repo_root / "exports"
    if exports_dir.is_dir():
        for p in sorted(exports_dir.iterdir()):
            if p.is_file():
                export_entries[p.name] = p.stat().st_size

    return {
        "session_sha256": sha256_file(session_file) if session_file.is_file() else None,
        "session_bytes": session_file.stat().st_size if session_file.is_file() else None,
        "rescued_sha256": sha256_file(rescued_ckpt) if rescued_ckpt.is_file() else None,
        "rescued_bytes": rescued_ckpt.stat().st_size if rescued_ckpt.is_file() else None,
        "tokenizer_sha256": sha256_file(tokenizer_file) if tokenizer_file.is_file() else None,
        "tokenizer_bytes": tokenizer_file.stat().st_size if tokenizer_file.is_file() else None,
        "data_entries": data_entries,
        "export_entries": export_entries,
    }


def assert_r8_snapshot_intact(repo_root: Path, snapshot: Dict[str, Any]) -> None:
    after = capture_r8_snapshot(repo_root)
    assert after["session_sha256"] == snapshot["session_sha256"], (
        "R8 violation: checkpoints/session_state.json was modified!"
    )
    assert after["rescued_sha256"] == snapshot["rescued_sha256"], (
        "R8 violation: checkpoints/triune_studio_step_36500_rescued.pt was modified!"
    )
    assert after["tokenizer_sha256"] == snapshot["tokenizer_sha256"], (
        "R8 violation: triune_tokenizer.json was modified!"
    )
    assert after["data_entries"] == snapshot["data_entries"], (
        f"R8 violation: data/ files changed: before={snapshot['data_entries']} after={after['data_entries']}"
    )
    assert after["export_entries"] == snapshot["export_entries"], (
        f"R8 violation: exports/ files changed: before={snapshot['export_entries']} after={after['export_entries']}"
    )


# ---------------------------------------------------------------------------
# Synthetic Tiny Checkpoint & CLI Subprocess Helpers
# ---------------------------------------------------------------------------


def build_tiny_checkpoint(
    dest_path: Path,
    repo_root: Path,
    step: int = 10,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Path:
    import torch
    from lucid_machines.data import load_tokenizer
    from lucid_machines.model import TriuneTransformer

    tok = load_tokenizer(repo_root / "triune_tokenizer.json")
    vocab_size = tok.get_vocab_size()
    cfg: Dict[str, Any] = {
        "vocab_size": vocab_size,
        "hidden_dim": 64,
        "num_layers": 5,
        "num_heads": 2,
        "head_dim": 32,
        "num_experts": 2,
        "top_k_experts": 1,
        "shared_expert": True,
        "shared_expert_scale": 0.5,
        "expert_hidden_multiplier": 6,
        "rope_max_seq_len": 4096,
        "router_prefix_layers": 1,
        "reflex_exit_layer": 2,
        "limbic_exit_layer": 3,
        "seq_len": 16,
        "batch_size": 1,
    }
    if extra_config:
        cfg.update(extra_config)

    model = TriuneTransformer(
        vocab_size=vocab_size,
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        head_dim=int(cfg["head_dim"]),
        num_experts=int(cfg["num_experts"]),
        top_k_experts=int(cfg["top_k_experts"]),
        router_prefix_layers=int(cfg["router_prefix_layers"]),
        reflex_exit_layer=int(cfg["reflex_exit_layer"]),
        limbic_exit_layer=int(cfg["limbic_exit_layer"]),
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "step": step}, dest_path)
    return dest_path


def run_kcx_cli(
    python_exe: str,
    repo_root: Path,
    args: List[str],
    cwd: Optional[Path] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    input_text: Optional[str] = None,
    timeout: float = 90.0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    repo_str = str(repo_root)
    env["PYTHONPATH"] = f"{repo_str}{os.pathsep}{existing_pp}" if existing_pp else repo_str
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("WANDB_MODE", "disabled")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [python_exe, "-m", "lucid_machines", *args],
        cwd=str(cwd or repo_root),
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Ephemeral Studio Server Process Helper
# ---------------------------------------------------------------------------


@dataclass
class StudioServerHandle:
    url: str
    port: int
    proc: subprocess.Popen[str]
    home_dir: Optional[Path] = None
    config_dir: Optional[Path] = None

    def request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> Tuple[int, Any, float]:
        url = f"{self.url}{path}"
        hdrs = {"Accept": "application/json"}
        data: Optional[bytes] = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                try:
                    return resp.status, json.loads(body), elapsed_ms
                except json.JSONDecodeError:
                    return resp.status, body, elapsed_ms
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            body = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(body), elapsed_ms
            except json.JSONDecodeError:
                return exc.code, body, elapsed_ms


def start_studio_server_process(
    python_exe: str,
    repo_root: Path,
    isolated_home: Optional[Path] = None,
    mask_windows_studio_dirs: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> StudioServerHandle:
    port = free_port()
    env = os.environ.copy()
    env["LUCID_MACHINES_TESTING"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else str(repo_root)
    if isolated_home is not None:
        isolated_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(isolated_home)
        env["USERPROFILE"] = str(isolated_home)
    if extra_env:
        env.update(extra_env)

    bootstrap_lines = [
        "import os, uvicorn",
    ]
    if mask_windows_studio_dirs:
        bootstrap_lines.extend(
            [
                "_orig_exists = os.path.exists",
                "def _masked_exists(p):",
                "    s = str(p).rstrip('\\\\/').lower()",
                "    if s in {'c:\\\\polysomstudio', 'c:\\\\lucidmachinesstudio'}:",
                "        return False",
                "    return _orig_exists(p)",
                "os.path.exists = _masked_exists",
            ]
        )
    bootstrap_lines.extend(
        [
            "from lucid_machines.api.server import create_app",
            "app = create_app()",
            f"uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')",
        ]
    )
    proc = subprocess.Popen(
        [python_exe, "-c", "\n".join(bootstrap_lines)],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        wait_http_ok(f"{url}/healthz", timeout=90.0)
    except Exception:
        terminate_process(proc)
        stderr = proc.stderr.read() if proc.stderr else ""
        stdout = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(
            f"Studio server failed to become healthy at {url}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    return StudioServerHandle(url=url, port=port, proc=proc, home_dir=isolated_home)


# ---------------------------------------------------------------------------
# Chrome CDP Over WebSockets Client & Probes
# ---------------------------------------------------------------------------


DOM_AUDIT_JS = r"""
(() => {
  const scrollWidth = document.documentElement.scrollWidth;
  const innerWidth = window.innerWidth;
  const bodyText = document.body ? (document.body.innerText || '') : '';
  const hasObjectObject = bodyText.includes('[object Object]');
  const emojiRegex = /[\u{1F300}-\u{1F5FF}\u{1F600}-\u{1F64F}\u{1F680}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{1FA70}-\u{1FAFF}\u{2600}-\u{27BF}]/u;
  const emojiMatch = bodyText.match(emojiRegex);
  const gradRegex = /(?:linear|radial|conic)-gradient\s*\(/i;
  let gradientHit = null;
  const allEls = document.querySelectorAll('*');
  for (let i = 0; i < allEls.length; i++) {
    const bg = window.getComputedStyle(allEls[i]).backgroundImage;
    if (bg && bg !== 'none' && gradRegex.test(bg)) {
      gradientHit = allEls[i].tagName + '.' + (allEls[i].className || '') + ': ' + bg;
      break;
    }
  }
  const activeTabBtn = document.querySelector('.react-nav .nav-item.active');
  return {
    scrollWidth,
    innerWidth,
    overflowPx: scrollWidth - innerWidth,
    hasObjectObject,
    emojiHit: emojiMatch ? emojiMatch[0] : null,
    gradientHit,
    textLen: bodyText.trim().length,
    activeNavTitle: activeTabBtn ? activeTabBtn.getAttribute('title') : null,
    theme: document.documentElement.getAttribute('data-theme'),
    runtimeErrorBanner: Boolean(document.getElementById('polysom-runtime-error')),
  };
})()
"""


def build_wcag_contrast_js(selectors: List[str]) -> str:
    sel_json = json.dumps(selectors)
    return f"""
(() => {{
  function parseColor(str) {{
    if (!str) return null;
    const m = str.match(/rgba?\\(\\s*(\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?)(?:,\\s*(\\d+(?:\\.\\d+)?))?\\s*\\)/i);
    if (!m) return null;
    return {{
      r: parseFloat(m[1]),
      g: parseFloat(m[2]),
      b: parseFloat(m[3]),
      a: m[4] !== undefined ? parseFloat(m[4]) : 1.0,
    }};
  }}
  function blendOver(fg, bg) {{
    const a = fg.a;
    return {{
      r: fg.r * a + bg.r * (1 - a),
      g: fg.g * a + bg.g * (1 - a),
      b: fg.b * a + bg.b * (1 - a),
      a: 1.0,
    }};
  }}
  function effectiveBg(el) {{
    const layers = [];
    let cur = el;
    while (cur && cur.nodeType === 1) {{
      const bg = parseColor(window.getComputedStyle(cur).backgroundColor);
      if (bg && bg.a > 0.01) layers.push(bg);
      if (bg && bg.a >= 0.99) break;
      cur = cur.parentElement;
    }}
    const rootApp = document.querySelector('.react-studio-app') || document.body;
    const base = parseColor(window.getComputedStyle(rootApp).backgroundColor) || {{ r: 11, g: 10, b: 20, a: 1 }};
    let acc = {{ r: base.r, g: base.g, b: base.b, a: 1 }};
    for (let i = layers.length - 1; i >= 0; i--) {{
      acc = blendOver(layers[i], acc);
    }}
    return acc;
  }}
  function relLum(c) {{
    const lin = [c.r, c.g, c.b].map(v => {{
      const s = v / 255.0;
      return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    }});
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
  }}
  function contrastRatio(c1, c2) {{
    const l1 = relLum(c1);
    const l2 = relLum(c2);
    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
  }}
  const results = [];
  for (const sel of {sel_json}) {{
    const nodes = Array.from(document.querySelectorAll(sel));
    for (const el of nodes) {{
      const r = el.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) continue;
      const cs = window.getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity) === 0) continue;
      const fgRaw = parseColor(cs.color);
      if (!fgRaw) continue;
      const bg = effectiveBg(el);
      const fg = fgRaw.a < 0.99 ? blendOver(fgRaw, bg) : fgRaw;
      const cr = contrastRatio(fg, bg);
      results.push({{
        selector: sel,
        text: (el.innerText || el.value || '').trim().slice(0, 60),
        cr: Math.round(cr * 100) / 100,
        fg: cs.color,
        bg: `rgb(${{Math.round(bg.r)}}, ${{Math.round(bg.g)}}, ${{Math.round(bg.b)}})`,
        fontSize: parseFloat(cs.fontSize || '14'),
      }});
    }}
  }}
  return results;
}})()
"""


AMBIENT_INSPECT_JS = r"""
(() => {
  const el = document.querySelector('.ps-ambient-perceptual');
  if (!el) return { exists: false };
  const cs = window.getComputedStyle(el);
  const primitives = el.querySelectorAll('svg path, svg circle, svg rect, svg polygon, svg ellipse, svg line, canvas').length;
  const svg = el.querySelector('svg');
  const canvas = el.querySelector('canvas');
  let sig = '';
  if (svg) {
    sig += svg.innerHTML.replace(/\s+/g, ' ').trim();
  }
  if (canvas) {
    sig += `|canvas:${canvas.width}x${canvas.height}`;
  }
  const pane = document.querySelector('.react-main') || document.querySelector('.react-view-container');
  const paneCs = pane ? window.getComputedStyle(pane) : null;
  let paneBgAlpha = 1.0;
  if (paneCs && paneCs.backgroundColor) {
    const m = paneCs.backgroundColor.match(/rgba\([^,]+,[^,]+,[^,]+,\s*([0-9.]+)\)/i);
    if (m) paneBgAlpha = parseFloat(m[1]);
    else if (paneCs.backgroundColor === 'transparent') paneBgAlpha = 0.0;
  }
  const ambientZ = parseInt(cs.zIndex || '0', 10) || 0;
  const paneZ = paneCs ? (parseInt(paneCs.zIndex || '0', 10) || 0) : 0;
  return {
    exists: true,
    className: el.className,
    display: cs.display,
    visibility: cs.visibility,
    opacity: parseFloat(cs.opacity || '0'),
    pointerEvents: cs.pointerEvents,
    primitives,
    signature: sig,
    ambientZ,
    paneZ,
    paneBgAlpha,
    notOccluded: (ambientZ > paneZ) || (paneBgAlpha < 0.96),
  };
})()
"""


@dataclass
class NetworkEventRecord:
    request_id: str
    url: str
    method: str
    resource_type: str
    t0: float
    status: Optional[int] = None
    t_resp: Optional[float] = None
    failed: bool = False
    error_text: str = ""


class StudioCDPClient:
    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self._id = 0
        self._pending: Dict[int, asyncio.Future[Any]] = {}
        self.console_errors: List[str] = []
        self.page_errors: List[str] = []
        self.log_errors: List[str] = []
        self.http_failures: List[str] = []
        self.ws_failures: List[str] = []
        self.network_events: Dict[str, NetworkEventRecord] = {}
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if "id" in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError(f"CDP error: {msg['error']}"))
                        else:
                            fut.set_result(msg.get("result", {}))
                    continue

                method = msg.get("method", "")
                params = msg.get("params", {})
                if method == "Runtime.consoleAPICalled":
                    if params.get("type") == "error":
                        args = [
                            str(a.get("value") if "value" in a else a.get("description", ""))
                            for a in params.get("args", [])
                        ]
                        self.console_errors.append(" ".join(args))
                elif method == "Runtime.exceptionThrown":
                    details = params.get("exceptionDetails", {})
                    exc_obj = details.get("exception", {})
                    self.page_errors.append(
                        exc_obj.get("description") or details.get("text") or json.dumps(details)
                    )
                elif method == "Log.entryAdded":
                    entry = params.get("entry", {})
                    if entry.get("level") == "error":
                        self.log_errors.append(f"{entry.get('source', '')}: {entry.get('text', '')}")
                elif method == "Network.requestWillBeSent":
                    rid = str(params.get("requestId", ""))
                    req = params.get("request", {})
                    self.network_events[rid] = NetworkEventRecord(
                        request_id=rid,
                        url=str(req.get("url", "")),
                        method=str(req.get("method", "GET")),
                        resource_type=str(params.get("type", "")),
                        t0=float(params.get("timestamp", time.monotonic())),
                    )
                elif method == "Network.responseReceived":
                    rid = str(params.get("requestId", ""))
                    resp = params.get("response", {})
                    status = int(resp.get("status", 0) or 0)
                    rec = self.network_events.get(rid)
                    if rec is not None:
                        rec.status = status
                        rec.t_resp = float(params.get("timestamp", time.monotonic()))
                    if status >= 400:
                        self.http_failures.append(f"HTTP {status} {resp.get('url', '')}")
                elif method == "Network.loadingFailed":
                    rid = str(params.get("requestId", ""))
                    rec = self.network_events.get(rid)
                    if rec is not None:
                        rec.failed = True
                        rec.error_text = str(params.get("errorText", ""))
                    if not params.get("canceled", False):
                        self.http_failures.append(
                            f"LOAD_FAILED {params.get('type', '')}: {params.get('errorText', '')}"
                        )
                elif method == "Network.webSocketHandshakeResponseReceived":
                    resp = params.get("response", {})
                    status = int(resp.get("status", 101) or 101)
                    if status >= 400:
                        self.ws_failures.append(f"WS HTTP {status}")
                elif method == "Network.webSocketFrameError":
                    self.ws_failures.append(f"WS frame error: {params.get('errorMessage', '')}")
        except asyncio.CancelledError:
            pass

    def reset_tracking(self) -> None:
        self.console_errors.clear()
        self.page_errors.clear()
        self.log_errors.clear()
        self.http_failures.clear()
        self.ws_failures.clear()
        self.network_events.clear()

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._id += 1
        req_id = self._id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self.ws.send(json.dumps({"id": req_id, "method": method, "params": params or {}}))
        return await asyncio.wait_for(fut, timeout=20.0)

    async def enable_all(self) -> None:
        await self.call("Page.enable")
        await self.call("Runtime.enable")
        await self.call("Log.enable")
        await self.call("Network.enable")

    async def set_viewport(self, width: int, height: int, mobile: bool = False) -> None:
        await self.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": mobile},
        )

    async def set_user_agent(self, user_agent: str, platform: str) -> None:
        await self.call(
            "Emulation.setUserAgentOverride",
            {"userAgent": user_agent, "platform": platform},
        )

    async def navigate(self, url: str) -> None:
        await self.call("Page.navigate", {"url": url})

    async def eval_js(self, expr: str) -> Any:
        res = await self.call(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
        )
        if "exceptionDetails" in res:
            raise AssertionError(f"JS evaluation error: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    async def wait_for_js(self, expr: str, timeout: float = 10.0, interval: float = 0.1) -> Any:
        deadline = time.monotonic() + timeout
        last_val: Any = None
        while time.monotonic() < deadline:
            last_val = await self.eval_js(expr)
            if last_val:
                return last_val
            await asyncio.sleep(interval)
        raise AssertionError(f"Timed out after {timeout}s waiting for JS condition ({expr}); last={last_val!r}")

    async def wait_for_studio_ready(self, timeout: float = 15.0) -> None:
        await self.wait_for_js(
            "document.querySelectorAll('.react-nav button.nav-item').length === 13",
            timeout=timeout,
        )

    async def click_tab(self, index: int) -> None:
        ok = await self.eval_js(
            f"""(() => {{
              const btn = document.querySelectorAll('.react-nav button.nav-item')[{index}];
              if (!btn) return false;
              btn.click();
              return true;
            }})()"""
        )
        assert ok, f"Failed to click nav tab index {index}"
        await asyncio.sleep(0.2)

    async def select_theme(self, theme_id: str) -> None:
        ok = await self.eval_js(
            f"""(() => {{
              const sel = document.querySelector('select[aria-label="Switch Studio Theme"]');
              if (!sel) return false;
              sel.value = {json.dumps(theme_id)};
              sel.dispatchEvent(new Event('change', {{ bubbles: true }}));
              return true;
            }})()"""
        )
        assert ok, f"Failed to select theme {theme_id} via topbar select"
        await self.wait_for_js(
            f"document.documentElement.getAttribute('data-theme') === {json.dumps(theme_id)}",
            timeout=5.0,
        )

    async def audit_dom(self) -> Dict[str, Any]:
        return await self.eval_js(DOM_AUDIT_JS)

    async def inspect_ambient(self) -> Dict[str, Any]:
        return await self.eval_js(AMBIENT_INSPECT_JS)

    async def check_wcag_contrast(self, selectors: List[str]) -> List[Dict[str, Any]]:
        return await self.eval_js(build_wcag_contrast_js(selectors))

    async def close(self) -> None:
        self._reader_task.cancel()
        try:
            await self._reader_task
        except asyncio.CancelledError:
            pass


@dataclass
class ChromeBrowserController:
    chrome_bin: str
    port: int
    proc: subprocess.Popen[str]
    profile_dir: Path
    default_ws_url: str

    def run(self, coro_factory: Callable[[StudioCDPClient], Awaitable[Any]]) -> Any:
        async def _runner() -> Any:
            import websockets

            ws_url = self.default_ws_url
            target_id: Optional[str] = None
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.port}/json/new?about:blank",
                    method="PUT",
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    info = json.loads(resp.read().decode("utf-8"))
                    if info.get("webSocketDebuggerUrl"):
                        ws_url = info["webSocketDebuggerUrl"]
                        target_id = info.get("id")
            except Exception:
                pass

            async with websockets.connect(ws_url, max_size=32 * 1024 * 1024) as ws:
                client = StudioCDPClient(ws)
                try:
                    await client.enable_all()
                    return await coro_factory(client)
                finally:
                    await client.close()
                    if target_id:
                        try:
                            urllib.request.urlopen(
                                f"http://127.0.0.1:{self.port}/json/close/{target_id}",
                                timeout=2.0,
                            ).read()
                        except Exception:
                            pass

        return asyncio.run(_runner())


def launch_chrome_controller(profile_dir: Path) -> ChromeBrowserController:
    chrome_bin = find_chrome_binary()
    cdp_port = free_port()
    profile_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={profile_dir}",
            "--window-size=1440,900",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        targets = wait_http_ok(f"http://127.0.0.1:{cdp_port}/json/list", timeout=15.0)
        page_target = next((t for t in targets if t.get("type") == "page"), targets[0])
        return ChromeBrowserController(
            chrome_bin=chrome_bin,
            port=cdp_port,
            proc=proc,
            profile_dir=profile_dir,
            default_ws_url=page_target["webSocketDebuggerUrl"],
        )
    except Exception:
        terminate_process(proc)
        raise


# ---------------------------------------------------------------------------
# Papers (Tectonic) & Website (Vite / tsc / oxfmt / Static HTTP) Helpers
# ---------------------------------------------------------------------------


def run_tectonic_scratch(
    repo_root: Path,
    paper_subdir: str,
    scratch_root: Path,
    timeout: float = 120.0,
) -> Tuple[subprocess.CompletedProcess[str], Path, str, int]:
    from pypdf import PdfReader

    src_dir = repo_root / "papers" / paper_subdir
    dst_dir = scratch_root / paper_subdir
    if dst_dir.exists():
        shutil.rmtree(dst_dir, ignore_errors=True)
    shutil.copytree(src_dir, dst_dir)

    tectonic_bin = find_tectonic_binary()
    proc = subprocess.run(
        [tectonic_bin, "main.tex"],
        cwd=str(dst_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    pdf_path = dst_dir / "main.pdf"
    extracted = ""
    page_count = 0
    if pdf_path.is_file():
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    return proc, pdf_path, extracted, page_count


def run_website_tool(
    repo_root: Path,
    tool_name: str,
    args: List[str],
    timeout: float = 90.0,
) -> subprocess.CompletedProcess[str]:
    website_dir = repo_root / "website"
    bin_path = website_dir / "node_modules" / ".bin" / f"{tool_name}.cmd"
    if not bin_path.is_file():
        bin_path = website_dir / "node_modules" / ".bin" / tool_name
    assert bin_path.is_file(), f"Missing website tool binary: {bin_path}"
    return subprocess.run(
        [str(bin_path), *args],
        cwd=str(website_dir),
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


class _QuietStaticHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


@contextmanager
def serve_static_dir(directory: Path) -> Iterator[str]:
    port = free_port()
    handler = functools.partial(_QuietStaticHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    try:
        wait_http_ok(f"{url}/", timeout=10.0)
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
