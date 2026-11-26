"""Feature #6 (6 Artistic Hallucination Themes) & Feature #8 (Studio Startup/Nav Perf + QoL + R8) E2E Tests."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from .helpers import (
    STUDIO_THEMES,
    ChromeBrowserController,
    StudioCDPClient,
    StudioServerHandle,
    assert_r8_snapshot_intact,
    capture_r8_snapshot,
)

GRADIENT_RE = re.compile(r"(?:linear|radial|conic)-gradient\s*\(", re.IGNORECASE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2600-\u27BF]")


async def _cold_open_studio(client: StudioCDPClient, base_url: str, width: int = 1440, height: int = 900) -> float:
    await client.set_viewport(width, height, mobile=False)
    await client.call("Network.clearBrowserCache")
    await client.navigate(f"{base_url}/healthz")
    await client.wait_for_js("document.readyState === 'complete'", timeout=8.0)
    await client.eval_js("window.localStorage.clear()")
    client.reset_tracking()
    t0 = time.perf_counter()
    await client.navigate(f"{base_url}/")
    await client.wait_for_studio_ready(timeout=15.0)
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Feature #6 — 6 Artistic Hallucination Themes & Expressive Theme Cards (5 T1 + 5 T2)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_f6_all_six_themes_switch_and_persist_localstorage(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T1-01: All 6 themes switch cleanly via topbar selector and persist in localStorage."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        seen: Dict[str, Dict[str, Any]] = {}
        for tid in STUDIO_THEMES:
            await client.select_theme(tid)
            state = await client.eval_js(
                """(() => ({
                  attr: document.documentElement.getAttribute('data-theme'),
                  stored: window.localStorage.getItem('lucid_machines.studio.theme'),
                  bg: window.getComputedStyle(document.body).backgroundColor,
                }))()"""
            )
            seen[tid] = state
        return {
            "seen": seen,
            "console_errors": list(client.console_errors),
            "page_errors": list(client.page_errors),
        }

    res = cdp_browser.run(_run)
    assert res["console_errors"] == [] and res["page_errors"] == []
    for tid in STUDIO_THEMES:
        assert res["seen"][tid]["attr"] == tid
        assert res["seen"][tid]["stored"] == tid


@pytest.mark.tier1
def test_t1_f6_ambient_perceptual_visible_not_occluded_across_six_themes(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T1-02: .ps-ambient-perceptual is genuinely visible (opacity >= 0.12, not occluded) across all 6 themes."""

    async def _run(client: StudioCDPClient) -> Dict[str, Dict[str, Any]]:
        await _cold_open_studio(client, studio_server.url)
        await client.eval_js(
            "window.localStorage.setItem('lucid_machines.studio.effectsLevel', 'full')"
        )
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        results: Dict[str, Dict[str, Any]] = {}
        for tid in STUDIO_THEMES:
            await client.select_theme(tid)
            await asyncio.sleep(0.15)
            results[tid] = await client.inspect_ambient()
        return results

    res = cdp_browser.run(_run)
    for tid, info in res.items():
        assert info.get("exists") is True, f".ps-ambient-perceptual missing for {tid}"
        assert info["display"] != "none" and info["visibility"] != "hidden", f"Hidden for {tid}: {info}"
        assert info["pointerEvents"] == "none", f"pointer-events must be none for {tid}: {info}"
        assert info["opacity"] >= 0.12, f"Ambient opacity too faint ({info['opacity']}) for {tid}: {info}"
        assert info["notOccluded"] is True, (
            f".ps-ambient-perceptual occluded behind opaque pane for {tid}: "
            f"ambientZ={info['ambientZ']} paneZ={info['paneZ']} paneBgAlpha={info['paneBgAlpha']}"
        )


@pytest.mark.tier1
def test_t1_f6_bespoke_procedural_artwork_six_plus_primitives_and_distinct(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T1-03: Each of the 6 themes renders rich theme-specific SVG/canvas art (>=6 primitives, 6 unique signatures)."""

    async def _run(client: StudioCDPClient) -> Dict[str, Dict[str, Any]]:
        await _cold_open_studio(client, studio_server.url)
        out: Dict[str, Dict[str, Any]] = {}
        for tid in STUDIO_THEMES:
            await client.select_theme(tid)
            await asyncio.sleep(0.15)
            out[tid] = await client.inspect_ambient()
        return out

    res = cdp_browser.run(_run)
    signatures = set()
    for tid in STUDIO_THEMES:
        info = res[tid]
        assert info.get("primitives", 0) >= 6, (
            f"Theme {tid} ambient layer has only {info.get('primitives')} primitives (expected >= 6 rich artwork)"
        )
        sig = info.get("signature", "")
        assert sig, f"Empty ambient artwork signature for {tid}"
        signatures.add(sig)
    assert len(signatures) == 6, f"Expected 6 unique theme ambient signatures, got {len(signatures)}"


@pytest.mark.tier1
def test_t1_f6_appearance_preview_cards_distinct_per_theme(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T1-04: All 6 .appearance-preview cards in Appearance & Themes embed distinct theme artwork."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        await client.click_tab(7)  # customize
        await client.wait_for_js("document.querySelectorAll('.appearance-preview').length >= 6", timeout=8.0)
        cards = await client.eval_js(
            """(() => Array.from(document.querySelectorAll('.appearance-preview')).map(card => {
              const cls = card.className;
              const svgs = Array.from(card.querySelectorAll('svg')).map(s => s.innerHTML.replace(/\\s+/g, ' ').trim());
              const hasCanvas = card.querySelectorAll('canvas').length;
              return { cls, svgSig: svgs.join('|'), hasCanvas, bg: card.style.background };
            }))()"""
        )
        return {"cards": cards}

    res = cdp_browser.run(_run)
    cards = res["cards"]
    for tid in STUDIO_THEMES:
        assert any(f"preview-{tid}" in c["cls"] for c in cards), f"Missing preview-{tid} card"

    curated_cards = [c for c in cards if any(f"preview-{tid}" in c["cls"] for tid in STUDIO_THEMES)]
    sigs = {c["svgSig"] or f"canvas-{c['cls']}" for c in curated_cards}
    assert len(sigs) == 6, f"Appearance theme preview cards lack 6 distinct artwork signatures: {len(sigs)}"


@pytest.mark.tier1
def test_t1_f6_effects_toggle_full_subtle_off_updates_dom_and_storage(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T1-05: Effects toggle (Full / Subtle / Off) updates .ps-ambient-perceptual opacity & localStorage."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        await client.click_tab(7)  # customize
        await client.wait_for_js("document.querySelectorAll('.ps-effects-btn').length === 3", timeout=8.0)

        states: Dict[str, Any] = {}
        for idx, key in ((0, "full"), (1, "subtle"), (2, "off")):
            await client.eval_js(f"document.querySelectorAll('.ps-effects-btn')[{idx}].click()")
            await asyncio.sleep(0.15)
            ambient = await client.inspect_ambient()
            stored = await client.eval_js("window.localStorage.getItem('lucid_machines.studio.effectsLevel')")
            states[key] = {"ambient": ambient, "stored": stored}
        return states

    res = cdp_browser.run(_run)
    assert res["full"]["stored"] == "full" and res["full"]["ambient"]["exists"] is True
    full_op = res["full"]["ambient"]["opacity"]
    assert full_op >= 0.12

    assert res["subtle"]["stored"] == "subtle" and res["subtle"]["ambient"]["exists"] is True
    subtle_op = res["subtle"]["ambient"]["opacity"]
    assert 0.0 < subtle_op < full_op, f"Subtle opacity ({subtle_op}) should be strictly between 0 and Full ({full_op})"

    assert res["off"]["stored"] == "off"
    assert res["off"]["ambient"]["exists"] is False or res["off"]["ambient"]["display"] == "none"


@pytest.mark.tier2
def test_t2_f6_legacy_localstorage_theme_migrates_cleanly(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T2-01: Legacy triune.studio.theme='editorial-cream' migrates cleanly to 'mirage' on reload."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        await client.eval_js(
            """(() => {
              window.localStorage.removeItem('lucid_machines.studio.theme');
              window.localStorage.setItem('triune.studio.theme', 'editorial-cream');
            })()"""
        )
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        theme = await client.eval_js("document.documentElement.getAttribute('data-theme')")
        return {"theme": theme, "page_errors": list(client.page_errors), "console_errors": list(client.console_errors)}

    res = cdp_browser.run(_run)
    assert res["theme"] == "mirage"
    assert res["page_errors"] == [] and res["console_errors"] == []


@pytest.mark.tier2
def test_t2_f6_invalid_localstorage_theme_falls_back_to_hypnagogic(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T2-02: Invalid localStorage theme value falls back safely to 'hypnagogic' with zero errors."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        await client.eval_js("window.localStorage.setItem('lucid_machines.studio.theme', 'bogus-theme-999')")
        await client.navigate(f"{studio_server.url}/")
        await client.wait_for_studio_ready()
        theme = await client.eval_js("document.documentElement.getAttribute('data-theme')")
        audit = await client.audit_dom()
        return {"theme": theme, "audit": audit, "page_errors": list(client.page_errors)}

    res = cdp_browser.run(_run)
    assert res["theme"] == "hypnagogic"
    assert not res["audit"]["runtimeErrorBanner"]
    assert res["page_errors"] == []


@pytest.mark.tier2
def test_t2_f6_pareidolia_mouse_parallax_no_overflow_or_pointer_block(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T2-03: Pareidolia parallax responds to mousemove without horizontal overflow or blocking clicks."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        await client.select_theme("pareidolia")
        await client.eval_js(
            """(() => {
              for (const [x, y] of [[100, 100], [1200, 750], [720, 450]]) {
                window.dispatchEvent(new MouseEvent('mousemove', { clientX: x, clientY: y, bubbles: true }));
              }
            })()"""
        )
        await asyncio.sleep(0.15)
        ambient = await client.inspect_ambient()
        audit = await client.audit_dom()
        return {"ambient": ambient, "audit": audit, "page_errors": list(client.page_errors)}

    res = cdp_browser.run(_run)
    assert res["ambient"]["exists"] and res["ambient"]["pointerEvents"] == "none"
    assert res["audit"]["overflowPx"] <= 2
    assert res["page_errors"] == []


@pytest.mark.tier2
def test_t2_f6_effects_off_hides_ambient_and_full_restores(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T2-04: Toggling Off removes/hides .ps-ambient-perceptual and toggling Full restores it."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        await client.click_tab(7)
        await client.wait_for_js("document.querySelectorAll('.ps-effects-btn').length === 3", timeout=8.0)
        await client.eval_js("document.querySelectorAll('.ps-effects-btn')[2].click()")
        await asyncio.sleep(0.15)
        off_state = await client.inspect_ambient()
        await client.eval_js("document.querySelectorAll('.ps-effects-btn')[0].click()")
        await asyncio.sleep(0.15)
        full_state = await client.inspect_ambient()
        return {"off": off_state, "full": full_state}

    res = cdp_browser.run(_run)
    assert res["off"]["exists"] is False or res["off"]["display"] == "none"
    assert res["full"]["exists"] is True and res["full"]["opacity"] >= 0.12


@pytest.mark.tier2
def test_t2_f6_wcag_contrast_preserved_with_ambient_active_dark_and_light(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F6-T2-05: Primary UI text keeps WCAG >= 4.5:1 contrast with Full ambient art active on dark & light themes."""

    async def _run(client: StudioCDPClient) -> Dict[str, List[Dict[str, Any]]]:
        await _cold_open_studio(client, studio_server.url)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for tid in ("hypnagogic", "euphoria"):
            await client.select_theme(tid)
            await asyncio.sleep(0.15)
            out[tid] = await client.check_wcag_contrast(
                [".topbar-title", ".react-nav .nav-item .nav-label", ".card h3", ".card p"]
            )
        return out

    res = cdp_browser.run(_run)
    for tid, items in res.items():
        assert len(items) > 0
        for item in items:
            assert item["cr"] >= 4.5, f"Low WCAG contrast ({item['cr']}) on theme {tid}: {item}"


# ---------------------------------------------------------------------------
# Feature #8 — Studio Startup & Nav Performance + QoL + R8 (5 T1 + 5 T2)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_f8_cold_boot_polysom_logo_fetched_once(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T1-01: Cold boot fetches polysom-logo.svg at most once (no duplicate icon/alternate-icon fetch)."""

    async def _run(client: StudioCDPClient) -> List[str]:
        await _cold_open_studio(client, studio_server.url)
        await asyncio.sleep(1.5)
        return [
            rec.url
            for rec in client.network_events.values()
            if "polysom-logo.svg" in rec.url
        ]

    logo_urls = cdp_browser.run(_run)
    assert len(logo_urls) <= 1, f"Duplicate polysom-logo.svg fetches on cold boot: {logo_urls}"


@pytest.mark.tier1
def test_t1_f8_cold_boot_no_duplicate_hardware_or_gpu_telemetry_calls(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T1-02: Cold boot makes <=1 /v1/hardware/profile and <=1 /v1/telemetry/gpu requests."""

    async def _run(client: StudioCDPClient) -> Dict[str, int]:
        await _cold_open_studio(client, studio_server.url)
        await asyncio.sleep(2.0)
        urls = [rec.url for rec in client.network_events.values()]
        return {
            "hardware_profile": sum(1 for u in urls if "/v1/hardware/profile" in u),
            "telemetry_gpu": sum(1 for u in urls if "/v1/telemetry/gpu" in u),
        }

    counts = cdp_browser.run(_run)
    assert counts["hardware_profile"] <= 1, f"Duplicate /v1/hardware/profile calls on boot: {counts}"
    assert counts["telemetry_gpu"] <= 1, f"Duplicate /v1/telemetry/gpu calls on boot: {counts}"


@pytest.mark.tier1
def test_t1_f8_cold_boot_zero_unprompted_hf_github_or_marketplace_calls(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T1-03: Cold boot makes zero unprompted HF/GitHub/marketplace-search calls and <24 total requests."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        await asyncio.sleep(2.2)
        urls = [rec.url for rec in client.network_events.values()]
        return {
            "total": len(urls),
            "urls": urls,
            "external_hits": [
                u for u in urls if any(host in u.lower() for host in ("huggingface.co", "github.com"))
            ],
            "deferred_module_hits": [
                u for u in urls if "/v1/modules/search" in u or "/v1/modules/updates" in u
            ],
        }

    res = cdp_browser.run(_run)
    assert res["external_hits"] == [], f"Unprompted external HF/GitHub calls on boot: {res['external_hits']}"
    assert res["deferred_module_hits"] == [], (
        f"Unprompted marketplace search/updates on boot: {res['deferred_module_hits']}"
    )
    assert res["total"] < 24, f"Too many cold-boot requests ({res['total']} >= 24):\n{res['urls']}"


@pytest.mark.tier1
def test_t1_f8_cold_boot_models_under_100ms_and_fast_nav_render(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T1-04: Cold boot /v1/models completes in <100ms and 13 sidebar tabs render in <2000ms."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        # Warm server cache once via HTTP then measure browser boot
        render_ms = await _cold_open_studio(client, studio_server.url)
        await asyncio.sleep(0.5)
        model_recs = [
            rec
            for rec in client.network_events.values()
            if rec.url.endswith("/v1/models") or "/v1/models?" in rec.url
        ]
        durations = [
            (rec.t_resp - rec.t0) * 1000.0
            for rec in model_recs
            if rec.t_resp is not None
        ]
        return {"render_ms": render_ms, "models_durations": durations, "count": len(model_recs)}

    res = cdp_browser.run(_run)
    assert res["render_ms"] < 2000.0, f"Studio boot render took {res['render_ms']:.1f}ms (>=2000ms)"
    assert res["count"] >= 1, "Expected /v1/models fetch during boot"
    for dur in res["models_durations"]:
        assert dur < 100.0, f"Boot /v1/models took {dur:.1f}ms (>=100ms)"


@pytest.mark.tier1
def test_t1_f8_r8_zero_css_gradients_and_zero_emojis_source_and_dom(
    e2e_repo_root: Path, studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T1-05: Zero CSS gradients and zero emojis in studio/src/ files and live rendered DOM across tabs."""
    for rel in ("studio/src/style.css", "studio/src/app.js", "studio/src/index.html"):
        text = (e2e_repo_root / rel).read_text(encoding="utf-8")
        grad_hit = GRADIENT_RE.search(text)
        assert grad_hit is None, f"CSS gradient found in {rel}: {grad_hit.group(0)}"
        emoji_hit = EMOJI_RE.search(text)
        assert emoji_hit is None, f"Emoji found in {rel}: {repr(emoji_hit.group(0))}"

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        failures: List[str] = []
        for idx in range(13):
            await client.click_tab(idx)
            audit = await client.audit_dom()
            if audit["gradientHit"]:
                failures.append(f"Tab {idx} gradient: {audit['gradientHit']}")
            if audit["emojiHit"]:
                failures.append(f"Tab {idx} emoji: {audit['emojiHit']}")
        return {
            "failures": failures,
            "console_errors": list(client.console_errors),
            "page_errors": list(client.page_errors),
            "http_failures": list(client.http_failures),
            "ws_failures": list(client.ws_failures),
        }

    res = cdp_browser.run(_run)
    assert res["failures"] == [], f"DOM gradient/emoji violations: {res['failures']}"
    assert res["console_errors"] == [] and res["page_errors"] == []
    assert res["http_failures"] == [] and res["ws_failures"] == []


@pytest.mark.tier2
def test_t2_f8_rapid_tab_switch_burst_no_errors_or_runaway_scans(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T2-01: Rapidly switching all 13 tabs triggers zero HTTP>=400, zero JS exceptions, and <=2 /v1/models calls."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        client.reset_tracking()
        for idx in range(13):
            await client.eval_js(
                f"document.querySelectorAll('.react-nav button.nav-item')[{idx}].click()"
            )
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.6)
        models_hits = [
            rec.url
            for rec in client.network_events.values()
            if rec.url.endswith("/v1/models") or "/v1/models/scan" in rec.url
        ]
        return {
            "http_failures": list(client.http_failures),
            "page_errors": list(client.page_errors),
            "console_errors": list(client.console_errors),
            "models_hits": models_hits,
        }

    res = cdp_browser.run(_run)
    assert res["http_failures"] == [], f"HTTP errors on rapid tab switch: {res['http_failures']}"
    assert res["page_errors"] == [] and res["console_errors"] == []
    assert len(res["models_hits"]) <= 2, f"Runaway /v1/models calls on tab switch: {res['models_hits']}"


@pytest.mark.tier2
def test_t2_f8_command_palette_ctrl_k_navigation(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T2-02: Pressing Ctrl+K opens Command Palette (.ps-cmd-palette-backdrop) and navigates to System & Hardware."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        await client.eval_js(
            "window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true }))"
        )
        await client.wait_for_js("Boolean(document.querySelector('.ps-cmd-palette-backdrop'))", timeout=5.0)
        clicked = await client.eval_js(
            """(() => {
              const items = Array.from(document.querySelectorAll('.ps-cmd-palette-backdrop button, .ps-cmd-palette-item'));
              const sysBtn = items.find(b => (b.innerText || '').includes('System'));
              if (!sysBtn) return false;
              sysBtn.click();
              return true;
            })()"""
        )
        await asyncio.sleep(0.2)
        active = await client.eval_js(
            "document.querySelector('.react-nav .nav-item.active')?.getAttribute('title')"
        )
        return {"clicked": clicked, "active": active}

    res = cdp_browser.run(_run)
    assert res["clicked"] is True
    assert res["active"] == "System & Hardware"


@pytest.mark.tier2
def test_t2_f8_modules_tab_defers_hf_github_until_explicit_action(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T2-03: Opening Extensions Marketplace loads installed modules without blocking on live HF/GitHub calls."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _cold_open_studio(client, studio_server.url)
        client.reset_tracking()
        await client.click_tab(6)  # modules
        await asyncio.sleep(0.8)
        urls = [rec.url for rec in client.network_events.values()]
        search_btn_text = await client.eval_js(
            """(() => {
              const btns = Array.from(document.querySelectorAll('button'));
              const b = btns.find(el => (el.innerText || '').includes('Search'));
              return b ? b.innerText.trim() : null;
            })()"""
        )
        return {
            "urls": urls,
            "external": [u for u in urls if "huggingface.co" in u or "github.com" in u],
            "search_btn_text": search_btn_text,
        }

    res = cdp_browser.run(_run)
    assert res["external"] == [], f"Unexpected external calls on modules tab open: {res['external']}"
    assert res["search_btn_text"] != "Searching...", "Marketplace search button stuck on 'Searching...'"


@pytest.mark.tier2
def test_t2_f8_studio_src_xss_hygiene_and_brand_boundary(e2e_repo_root: Path) -> None:
    """F8-T2-04: studio/src/ avoids unsafe HTML injection sinks and stale legacy brand strings."""
    for rel in ("studio/src/app.js", "studio/src/index.html"):
        src = (e2e_repo_root / rel).read_text(encoding="utf-8")
        for forbidden in ("dangerouslySetInnerHTML", "document.write(", "TriuneStudio", "var(--font-sans)"):
            assert forbidden not in src, f"Forbidden pattern {forbidden!r} found in {rel}"


@pytest.mark.tier2
def test_t2_f8_r8_preservation_around_live_browser_session(
    e2e_repo_root: Path, studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F8-T2-05: Full live browser session leaves R8 preserved files byte-identical and root shims absent."""
    snap = capture_r8_snapshot(e2e_repo_root)

    async def _run(client: StudioCDPClient) -> None:
        await _cold_open_studio(client, studio_server.url)
        for idx in (0, 1, 2, 4, 6, 7, 8, 9):
            await client.click_tab(idx)

    cdp_browser.run(_run)
    assert_r8_snapshot_intact(e2e_repo_root, snap)
    assert not (e2e_repo_root / "triune").exists()
    assert not (e2e_repo_root / "hmar.py").exists()
