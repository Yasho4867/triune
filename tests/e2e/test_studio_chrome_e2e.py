"""Live Headless Chrome E2E Verification for Polysom Studio (R3 & R4).

Launches the real FastAPI Studio server on an ephemeral localhost port, drives
headless Chrome via CDP over WebSocket, and verifies:
- All 13 sidebar tabs (including Chat Arena, Token Lab, Mind Probe, Neural Atlas 2D/3D, Context Bridge)
- All 6 perceptual themes (hypnagogic, phosphene, pareidolia, afterimage, mirage, euphoria)
- Visual Effects toggle (Full / Subtle / Off)
- Responsive viewports (1440x900 and 960x640)
- Zero console errors, uncaught page errors, failed HTTP/WS requests, [object Object], emojis, CSS gradients, or horizontal overflow
- /v1/configs includes junkie-mini and junkie-3b
- checkpoints/session_state.json remains untouched (step == 40555)
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

CHROME_CANDIDATES = [
    os.environ.get("CHROME_PATH", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def _find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_ok(url: str, timeout: float = 45.0) -> Any:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}")


class _CDPClient:
    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self._id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self.console_errors: list[str] = []
        self.page_errors: list[str] = []
        self.log_errors: list[str] = []
        self.http_failures: list[str] = []
        self.ws_failures: list[str] = []
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
                    self.page_errors.append(exc_obj.get("description") or details.get("text") or json.dumps(details))
                elif method == "Log.entryAdded":
                    entry = params.get("entry", {})
                    if entry.get("level") == "error":
                        self.log_errors.append(f"{entry.get('source', '')}: {entry.get('text', '')}")
                elif method == "Network.responseReceived":
                    resp = params.get("response", {})
                    status = int(resp.get("status", 0) or 0)
                    if status >= 400:
                        self.http_failures.append(f"HTTP {status} {resp.get('url', '')}")
                elif method == "Network.loadingFailed":
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

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        req_id = self._id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self.ws.send(json.dumps({"id": req_id, "method": method, "params": params or {}}))
        return await asyncio.wait_for(fut, timeout=15.0)

    async def eval_js(self, expr: str) -> Any:
        res = await self.call(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
        )
        if "exceptionDetails" in res:
            raise AssertionError(f"JS evaluation error: {res['exceptionDetails']}")
        return res.get("result", {}).get("value")

    async def close(self) -> None:
        self._reader_task.cancel()
        try:
            await self._reader_task
        except asyncio.CancelledError:
            pass


_AUDIT_DOM_JS = r"""
(() => {
  const scrollWidth = document.documentElement.scrollWidth;
  const innerWidth = window.innerWidth;
  const bodyText = document.body ? (document.body.innerText || '') : '';
  const hasObjectObject = bodyText.includes('[object Object]');
  const emojiRegex = /[\u{1F300}-\u{1F5FF}\u{1F600}-\u{1F64F}\u{1F680}-\u{1F6FF}\u{1F900}-\u{1F9FF}\u{1FA70}-\u{1FAFF}]/u;
  const emojiMatch = bodyText.match(emojiRegex);
  const gradRegex = /(?:linear|radial|conic)-gradient\s*\(/i;
  let gradientHit = null;
  const allEls = document.querySelectorAll('*');
  for (let i = 0; i < allEls.length; i++) {
    const bg = window.getComputedStyle(allEls[i]).backgroundImage;
    if (bg && bg !== 'none' && gradRegex.test(bg)) {
      gradientHit = allEls[i].tagName + '.' + allEls[i].className + ': ' + bg;
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


async def _run_studio_chrome_audit(base_url: str, cdp_ws_url: str) -> dict[str, Any]:
    import websockets

    report: dict[str, Any] = {"tabs_1440": [], "tabs_960": [], "themes": [], "effects": [], "chat": {}}
    async with websockets.connect(cdp_ws_url, max_size=16 * 1024 * 1024) as ws:
        cdp = _CDPClient(ws)
        try:
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Log.enable")
            await cdp.call("Network.enable")
            await cdp.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
            )
            await cdp.call("Page.navigate", {"url": f"{base_url}/"})

            # Wait for React mount + 13 sidebar tabs
            for _ in range(60):
                nav_count = await cdp.eval_js("document.querySelectorAll('.react-nav button.nav-item').length")
                if nav_count == 13:
                    break
                await asyncio.sleep(0.25)
            else:
                raise AssertionError(f"Expected 13 sidebar tabs, saw {nav_count}")

            # Wait briefly for initial REST/WS hydration
            await asyncio.sleep(0.8)

            # 1. Verify Chat sub-views, Context Bridge, and Neural Atlas 2D/3D
            has_context_bridge = await cdp.eval_js("Boolean(document.querySelector('.ps-context-bridge'))")
            has_atlas_canvas = await cdp.eval_js(
                "Boolean(document.querySelector('.neuron-theater-canvas-wrap canvas'))"
            )
            atlas_modes_seen = []
            for _ in range(3):
                label = await cdp.eval_js(
                    "(() => { const b = document.querySelectorAll('.neuron-theater-controls .neuron-ctrl-btn')[0]; return b ? b.innerText.trim() : null; })()"
                )
                atlas_modes_seen.append(label)
                await cdp.eval_js(
                    "(() => { const b = document.querySelectorAll('.neuron-theater-controls .neuron-ctrl-btn')[0]; if (b) b.click(); })()"
                )
                await asyncio.sleep(0.15)
            await cdp.eval_js(
                "(() => { const s = document.querySelector('.btn-stimulate-synapse'); if (s) s.click(); })()"
            )

            chat_submodes = {}
            for mode_label in ("Dual Arena", "Token Lab", "Mind Probe", "Chat & Reason"):
                clicked = await cdp.eval_js(
                    f"""(() => {{
                      const btns = Array.from(document.querySelectorAll('.playground-mode-btn'));
                      const target = btns.find(b => b.innerText.includes({json.dumps(mode_label)}));
                      if (!target) return false;
                      target.click();
                      return true;
                    }})()"""
                )
                await asyncio.sleep(0.25)
                audit = await cdp.eval_js(_AUDIT_DOM_JS)
                chat_submodes[mode_label] = {
                    "clicked": clicked,
                    "overflowPx": audit["overflowPx"],
                    "hasObjectObject": audit["hasObjectObject"],
                    "emojiHit": audit["emojiHit"],
                    "gradientHit": audit["gradientHit"],
                    "hasProbeView": await cdp.eval_js("Boolean(document.querySelector('.ps-probe-view'))"),
                    "hasArenaView": await cdp.eval_js(
                        "document.body.innerText.includes('Dual Model') || document.body.innerText.includes('Arena')"
                    ),
                }
            report["chat"] = {
                "has_context_bridge": has_context_bridge,
                "has_atlas_canvas": has_atlas_canvas,
                "atlas_modes_seen": atlas_modes_seen,
                "submodes": chat_submodes,
            }

            # 2. Verify all 13 sidebar tabs at 1440x900
            nav_len_after_chat = await cdp.eval_js("document.querySelectorAll('.react-nav button.nav-item').length")
            if nav_len_after_chat != 13:
                err_banner = await cdp.eval_js(
                    "(() => { const el = document.getElementById('polysom-runtime-error'); return el ? el.innerText : document.body.innerText.slice(0, 600); })()"
                )
                raise AssertionError(
                    f"Sidebar unmounted after Chat submodes! nav_len={nav_len_after_chat}, "
                    f"console_errors={cdp.console_errors}, page_errors={cdp.page_errors}, "
                    f"http_failures={cdp.http_failures}, banner={err_banner!r}"
                )
            for idx in range(13):
                await cdp.eval_js(f"document.querySelectorAll('.react-nav button.nav-item')[{idx}].click()")
                await asyncio.sleep(0.25)
                audit = await cdp.eval_js(_AUDIT_DOM_JS)
                audit["tab_index"] = idx
                report["tabs_1440"].append(audit)

            # 3. Verify all 6 perceptual themes + Effects toggle on customize tab (index 7)
            await cdp.eval_js("document.querySelectorAll('.react-nav button.nav-item')[7].click()")
            await asyncio.sleep(0.25)

            for theme_id in ("hypnagogic", "phosphene", "pareidolia", "afterimage", "mirage", "euphoria"):
                changed = await cdp.eval_js(
                    f"""(() => {{
                      const sel = document.querySelector('select[aria-label="Switch Studio Theme"]');
                      if (!sel) return false;
                      sel.value = {json.dumps(theme_id)};
                      sel.dispatchEvent(new Event('change', {{ bubbles: true }}));
                      return true;
                    }})()"""
                )
                await asyncio.sleep(0.2)
                audit = await cdp.eval_js(_AUDIT_DOM_JS)
                has_ambient = await cdp.eval_js("Boolean(document.querySelector('.ps-ambient-perceptual'))")
                report["themes"].append(
                    {
                        "expected": theme_id,
                        "changed": changed,
                        "applied": audit["theme"],
                        "has_ambient": has_ambient,
                        "overflowPx": audit["overflowPx"],
                        "gradientHit": audit["gradientHit"],
                    }
                )

            # Effects toggle: Full / Subtle / Off
            for eff_label, expected_cls in (
                ("Effects: Full", "full"),
                ("Subtle", "subtle"),
                ("Off / Reduced", "off"),
                ("Effects: Full", "full"),
            ):
                clicked = await cdp.eval_js(
                    f"""(() => {{
                      const btns = Array.from(document.querySelectorAll('.ps-effects-toggle .ps-effects-btn'));
                      const b = btns.find(el => el.innerText.trim() === {json.dumps(eff_label)});
                      if (!b) return false;
                      b.click();
                      return true;
                    }})()"""
                )
                await asyncio.sleep(0.15)
                ambient_cls = await cdp.eval_js(
                    "(() => { const el = document.querySelector('.ps-ambient-perceptual'); return el ? el.className : null; })()"
                )
                report["effects"].append(
                    {"label": eff_label, "expected": expected_cls, "clicked": clicked, "ambient_cls": ambient_cls}
                )

            # Restore hypnagogic theme
            await cdp.eval_js(
                """(() => {
                  const sel = document.querySelector('select[aria-label="Switch Studio Theme"]');
                  if (sel) {
                    sel.value = 'hypnagogic';
                    sel.dispatchEvent(new Event('change', { bubbles: true }));
                  }
                })()"""
            )

            # 4. Verify all 13 tabs at 960x640 viewport
            await cdp.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": 960, "height": 640, "deviceScaleFactor": 1, "mobile": False},
            )
            await asyncio.sleep(0.2)
            for idx in range(13):
                await cdp.eval_js(f"document.querySelectorAll('.react-nav button.nav-item')[{idx}].click()")
                await asyncio.sleep(0.22)
                audit = await cdp.eval_js(_AUDIT_DOM_JS)
                audit["tab_index"] = idx
                report["tabs_960"].append(audit)

            report["console_errors"] = list(cdp.console_errors)
            report["page_errors"] = list(cdp.page_errors)
            report["log_errors"] = list(cdp.log_errors)
            report["http_failures"] = list(cdp.http_failures)
            report["ws_failures"] = list(cdp.ws_failures)
        finally:
            await cdp.close()
    return report


@pytest.mark.e2e
def test_studio_headless_chrome_e2e(repo_root: Path) -> None:
    chrome_bin = _find_chrome()
    if not chrome_bin:
        pytest.skip("Google Chrome executable not installed; skipping live headless Chrome E2E.")

    session_file = repo_root / "checkpoints" / "session_state.json"
    session_before = session_file.read_bytes() if session_file.is_file() else None
    if session_before is not None:
        assert json.loads(session_before.decode("utf-8")).get("step") == 40555

    studio_port = _free_port()
    cdp_port = _free_port()
    base_url = f"http://127.0.0.1:{studio_port}"

    env = os.environ.copy()
    env["LUCID_MACHINES_TESTING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    server_cmd = [
        sys.executable,
        "-c",
        (
            "import uvicorn; from lucid_machines.api import create_app; "
            f"uvicorn.run(create_app(), host='127.0.0.1', port={studio_port}, log_level='warning')"
        ),
    ]
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    chrome_proc: subprocess.Popen[str] | None = None
    chrome_profile = tempfile.mkdtemp(prefix="polysom-chrome-")
    try:
        _wait_http_ok(f"{base_url}/healthz", timeout=45.0)
        configs_payload = _wait_http_ok(f"{base_url}/v1/configs", timeout=10.0)
        config_ids = {
            c.get("id") or Path(str(c.get("path", ""))).stem
            for c in configs_payload.get("configs", [])
            if isinstance(c, dict)
        }
        assert "junkie-mini" in config_ids, f"junkie-mini missing from /v1/configs: {config_ids}"
        assert "junkie-3b" in config_ids, f"junkie-3b missing from /v1/configs: {config_ids}"

        chrome_proc = subprocess.Popen(
            [
                chrome_bin,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                f"--remote-debugging-port={cdp_port}",
                f"--user-data-dir={chrome_profile}",
                "--window-size=1440,900",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        targets = _wait_http_ok(f"http://127.0.0.1:{cdp_port}/json/list", timeout=15.0)
        page_target = next((t for t in targets if t.get("type") == "page"), targets[0])
        cdp_ws_url = page_target["webSocketDebuggerUrl"]

        report = asyncio.run(_run_studio_chrome_audit(base_url, cdp_ws_url))

        assert report["console_errors"] == [], f"Console errors: {report['console_errors']}"
        assert report["page_errors"] == [], f"Uncaught page errors: {report['page_errors']}"
        assert report["log_errors"] == [], f"Browser log errors: {report['log_errors']}"
        assert report["http_failures"] == [], f"Failed HTTP requests: {report['http_failures']}"
        assert report["ws_failures"] == [], f"Failed WS requests: {report['ws_failures']}"

        chat_info = report["chat"]
        assert chat_info["has_context_bridge"] is True, "Context Bridge missing in Chat"
        assert chat_info["has_atlas_canvas"] is True, "Neural Atlas canvas missing in Chat"
        assert any("Obsidian" in str(m) for m in chat_info["atlas_modes_seen"]), chat_info["atlas_modes_seen"]
        assert any("3D Manifold" in str(m) for m in chat_info["atlas_modes_seen"]), chat_info["atlas_modes_seen"]
        for mode_label, info in chat_info["submodes"].items():
            assert info["clicked"] is True, f"Failed to click Chat mode {mode_label}"
            assert info["overflowPx"] <= 2, f"Chat mode {mode_label} overflowed by {info['overflowPx']}px"
            assert not info["hasObjectObject"], f"[object Object] in Chat mode {mode_label}"
            assert info["emojiHit"] is None, f"Emoji in Chat mode {mode_label}: {info['emojiHit']}"
            assert info["gradientHit"] is None, f"CSS gradient in Chat mode {mode_label}: {info['gradientHit']}"
        assert chat_info["submodes"]["Mind Probe"]["hasProbeView"] is True

        for label, tabs in (("1440x900", report["tabs_1440"]), ("960x640", report["tabs_960"])):
            assert len(tabs) == 13
            for t in tabs:
                assert not t["runtimeErrorBanner"], f"Runtime error banner on {label} tab {t}"
                assert t["textLen"] > 40, f"Empty tab on {label}: {t}"
                assert t["overflowPx"] <= 2, (
                    f"Horizontal overflow on {label} tab {t['activeNavTitle']}: {t['overflowPx']}px"
                )
                assert not t["hasObjectObject"], f"[object Object] on {label} tab {t['activeNavTitle']}"
                assert t["emojiHit"] is None, f"Emoji on {label} tab {t['activeNavTitle']}: {t['emojiHit']}"
                assert t["gradientHit"] is None, (
                    f"CSS gradient on {label} tab {t['activeNavTitle']}: {t['gradientHit']}"
                )

        for th in report["themes"]:
            assert th["changed"] is True
            assert th["applied"] == th["expected"], f"Theme mismatch: {th}"
            assert th["has_ambient"] is True, f"Ambient perceptual layer missing for {th['expected']}"
            assert th["overflowPx"] <= 2, f"Overflow under theme {th['expected']}: {th['overflowPx']}px"
            assert th["gradientHit"] is None, f"CSS gradient under theme {th['expected']}: {th['gradientHit']}"

        assert report["effects"][0]["ambient_cls"] and "full" in report["effects"][0]["ambient_cls"]
        assert report["effects"][1]["ambient_cls"] and "subtle" in report["effects"][1]["ambient_cls"]
        assert report["effects"][2]["ambient_cls"] is None

        if session_before is not None:
            session_after = session_file.read_bytes()
            assert session_after == session_before, "checkpoints/session_state.json was modified!"
            assert json.loads(session_after.decode("utf-8")).get("step") == 40555
    finally:
        if chrome_proc is not None:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome_proc.kill()
                chrome_proc.wait(timeout=5)
        import shutil

        shutil.rmtree(chrome_profile, ignore_errors=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
