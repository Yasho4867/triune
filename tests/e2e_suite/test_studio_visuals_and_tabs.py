"""Feature #7 — Studio Visual, Contrast & Layout Fixes Across 6 Themes & 13 Tabs E2E Tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from .helpers import (
    STUDIO_TABS,
    ChromeBrowserController,
    StudioCDPClient,
    StudioServerHandle,
)


async def _open_clean_studio(
    client: StudioCDPClient, base_url: str, width: int = 1440, height: int = 900
) -> None:
    await client.set_viewport(width, height, mobile=False)
    await client.navigate(f"{base_url}/healthz")
    await client.wait_for_js("document.readyState === 'complete'", timeout=8.0)
    await client.eval_js("window.localStorage.clear()")
    client.reset_tracking()
    await client.navigate(f"{base_url}/")
    await client.wait_for_studio_ready(timeout=15.0)
    await asyncio.sleep(0.3)


# ---------------------------------------------------------------------------
# Feature #7 — Studio Visual, Contrast & Layout Fixes (5 T1 + 5 T2)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_f7_model_zoo_transmutation_card_light_contrast_and_scan_pc_nowrap(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T1-01: Model Zoo .transmutation-wizard-card renders readable contrast on light themes & single-line Scan PC."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)
        await client.click_tab(9)  # models
        await client.wait_for_js("Boolean(document.querySelector('.transmutation-wizard-card'))", timeout=8.0)

        results: Dict[str, Any] = {}
        for tid in ("euphoria", "mirage"):
            await client.select_theme(tid)
            await asyncio.sleep(0.15)
            contrasts = await client.check_wcag_contrast(
                [
                    ".transmutation-wizard-card .transmutation-title",
                    ".transmutation-wizard-card label",
                    ".transmutation-wizard-card .transmutation-select",
                    ".transmutation-wizard-card p",
                ]
            )
            scan_btn = await client.eval_js(
                """(() => {
                  const btns = Array.from(document.querySelectorAll('.transmutation-wizard-card button, button'));
                  const b = btns.find(el => (el.innerText || '').includes('Scan PC'));
                  if (!b) return null;
                  const r = b.getBoundingClientRect();
                  const cs = window.getComputedStyle(b);
                  const lh = parseFloat(cs.lineHeight) || (parseFloat(cs.fontSize || '13') * 1.3);
                  return { text: b.innerText.trim(), height: r.height, lineHeight: lh, whiteSpace: cs.whiteSpace };
                })()"""
            )
            results[tid] = {"contrasts": contrasts, "scan_btn": scan_btn}
        return results

    res = cdp_browser.run(_run)
    for tid in ("euphoria", "mirage"):
        contrasts = res[tid]["contrasts"]
        assert len(contrasts) >= 1, f"No sampled elements inside .transmutation-wizard-card on {tid}"
        for item in contrasts:
            assert item["cr"] >= 4.5, f"Light-theme contrast failure on {tid} Model Zoo: {item}"
        sb = res[tid]["scan_btn"]
        assert sb is not None, "Scan PC button missing in Model Zoo"
        assert sb["height"] <= max(44.0, sb["lineHeight"] * 1.9), f"Scan PC button wrapped multi-line: {sb}"


@pytest.mark.tier1
def test_t1_f7_chat_three_pane_split_telemetry_pill_and_composer_pill(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T1-02: Chat center pane >= 520px at 1440x900, telemetry pill uses inline-flex gap, composer pill stays contained."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)
        await client.click_tab(0)  # chat
        await client.wait_for_js("Boolean(document.querySelector('.chat-main-pane'))", timeout=8.0)
        return await client.eval_js(
            """(() => {
              const mainPane = document.querySelector('.chat-main-pane');
              const mainRect = mainPane ? mainPane.getBoundingClientRect() : null;

              const actionsBar = document.querySelector('.chat-input-actions-bar') || document.querySelector('.chat-input-area');
              const barRect = actionsBar ? actionsBar.getBoundingClientRect() : null;
              const pills = Array.from(document.querySelectorAll('.chat-input-meta-left .pill'));
              const pillCheck = pills.map(p => {
                const r = p.getBoundingClientRect();
                const cs = window.getComputedStyle(p);
                return {
                  text: (p.innerText || '').trim(),
                  right: r.right,
                  left: r.left,
                  barRight: barRect ? barRect.right : 9999,
                  overflowingBar: barRect ? (r.right > barRect.right + 2) : false,
                  whiteSpace: cs.whiteSpace,
                };
              });

              let pillRule = null;
              for (const sheet of Array.from(document.styleSheets || [])) {
                let rules;
                try { rules = sheet.cssRules; } catch (_) { continue; }
                for (const rule of Array.from(rules || [])) {
                  if (rule.selectorText && rule.selectorText.includes('.bubble-telemetry-pill')) {
                    pillRule = {
                      display: rule.style.display,
                      gap: rule.style.gap || rule.style.columnGap,
                    };
                  }
                }
              }

              return {
                mainWidth: mainRect ? mainRect.width : 0,
                pillCheck,
                pillRule,
              };
            })()"""
        )

    res = cdp_browser.run(_run)
    assert res["mainWidth"] >= 520.0, f"Chat 3-pane center column too narrow: {res['mainWidth']}px"
    assert res["pillRule"] is not None, "Missing .bubble-telemetry-pill CSS rule"
    assert "flex" in str(res["pillRule"]["display"]), f"Expected inline-flex on .bubble-telemetry-pill: {res['pillRule']}"
    assert res["pillRule"]["gap"] and res["pillRule"]["gap"] != "0px", (
        f"Expected positive gap on .bubble-telemetry-pill: {res['pillRule']}"
    )
    for p in res["pillCheck"]:
        assert not p["overflowingBar"], f"Composer pill overflows bar: {p}"


@pytest.mark.tier1
def test_t1_f7_neural_atlas_controls_contrast_and_non_overlap(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T1-03: Neural Atlas HUD & controls have WCAG >= 4.5 contrast on light themes and zero HUD/controls collision."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)
        await client.click_tab(0)
        await client.wait_for_js("Boolean(document.querySelector('.neuron-theater-canvas-wrap canvas'))", timeout=8.0)

        out: Dict[str, Any] = {}
        for tid in ("mirage", "euphoria"):
            await client.select_theme(tid)
            await asyncio.sleep(0.15)
            contrasts = await client.check_wcag_contrast([".neuron-hud-item", ".neuron-ctrl-btn"])
            geom = await client.eval_js(
                """(() => {
                  const wrap = document.querySelector('.neuron-theater-canvas-wrap');
                  const canvas = wrap ? wrap.querySelector('canvas') : null;
                  const hud = document.querySelector('.neuron-theater-hud');
                  const ctrls = document.querySelector('.neuron-theater-controls');
                  if (!wrap || !canvas || !ctrls) return { ok: false };
                  const wr = wrap.getBoundingClientRect();
                  const cr = canvas.getBoundingClientRect();
                  const ctr = ctrls.getBoundingClientRect();
                  const hr = hud ? hud.getBoundingClientRect() : null;
                  let hudOverlap = 0;
                  if (hr) {
                    const ix = Math.max(0, Math.min(hr.right, ctr.right) - Math.max(hr.left, ctr.left));
                    const iy = Math.max(0, Math.min(hr.bottom, ctr.bottom) - Math.max(hr.top, ctr.top));
                    hudOverlap = ix * iy;
                  }
                  return {
                    ok: true,
                    canvasInside: cr.right <= wr.right + 2 && cr.left >= wr.left - 2,
                    controlsInside: ctr.right <= wr.right + 2 && ctr.bottom <= wr.bottom + 2,
                    hudOverlap,
                  };
                })()"""
            )
            out[tid] = {"contrasts": contrasts, "geom": geom}
        return out

    res = cdp_browser.run(_run)
    for tid in ("mirage", "euphoria"):
        assert len(res[tid]["contrasts"]) >= 1
        for item in res[tid]["contrasts"]:
            assert item["cr"] >= 4.5, f"Neural Atlas contrast failure on {tid}: {item}"
        g = res[tid]["geom"]
        assert g["ok"] and g["canvasInside"] and g["controlsInside"]
        assert g["hudOverlap"] == 0, f"Neural Atlas HUD overlaps controls on {tid}: {g}"


@pytest.mark.tier1
def test_t1_f7_workspace_ide_autoload_and_node_canvas_minimap_autofit(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T1-04: Workspace IDE auto-loads tree on activation; Visual Node Canvas auto-fits without minimap overlap."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)

        # 1. Workspace IDE auto-load without clicking Refresh
        await client.click_tab(1)
        await client.wait_for_js("document.querySelectorAll('.ps-ide-tree-item').length >= 3", timeout=3.0)
        ws_state = await client.eval_js(
            """(() => ({
              count: document.querySelectorAll('.ps-ide-tree-item').length,
              emptyText: (document.body.innerText || '').includes('No files loaded. Click Refresh or Open Folder.'),
            }))()"""
        )

        # 2. Visual Node Canvas auto-fit & minimap non-overlap
        await client.click_tab(2)
        await client.wait_for_js("Boolean(document.querySelector('.ps-dag-canvas'))", timeout=8.0)
        await asyncio.sleep(0.3)
        dag_geom = await client.eval_js(
            """(() => {
              const canvas = document.querySelector('.ps-dag-canvas');
              const minimap = document.querySelector('.ps-dag-minimap');
              const nodes = Array.from(document.querySelectorAll('.ps-dag-node, .dag-node'));
              const autograd = nodes.find(n => (n.innerText || '').includes('Autograd Training Step'));
              if (!canvas || !minimap || !autograd) {
                return { found: false, hasCanvas: Boolean(canvas), hasMinimap: Boolean(minimap), nodeCount: nodes.length };
              }
              const cr = canvas.getBoundingClientRect();
              const mr = minimap.getBoundingClientRect();
              const ar = autograd.getBoundingClientRect();
              const ix = Math.max(0, Math.min(mr.right, ar.right) - Math.max(mr.left, ar.left));
              const iy = Math.max(0, Math.min(mr.bottom, ar.bottom) - Math.max(mr.top, ar.top));
              return {
                found: true,
                overlap: ix * iy,
                autogradClippedRight: ar.right > cr.right + 2,
                autogradClippedLeft: ar.left < cr.left - 2,
              };
            })()"""
        )
        return {"ws": ws_state, "dag": dag_geom}

    res = cdp_browser.run(_run)
    assert res["ws"]["count"] >= 3 and res["ws"]["emptyText"] is False
    dag = res["dag"]
    assert dag["found"] is True, f"DAG nodes or minimap missing: {dag}"
    assert dag["overlap"] == 0, f".ps-dag-minimap overlaps Autograd Training Step card: {dag}"
    assert not dag["autogradClippedRight"] and not dag["autogradClippedLeft"], (
        f"Autograd Training Step clipped outside DAG canvas on auto-fit: {dag}"
    )


@pytest.mark.tier1
def test_t1_f7_checkpoints_hardware_marketplace_and_toast_placement(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T1-05: Checkpoints badges/dropdown, System & Hardware benchmarks/wrapping, Marketplace badges/author, Toast non-overlap."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)

        # 1. Checkpoints (tab 4)
        await client.click_tab(4)
        await client.wait_for_js("Boolean(document.querySelector('.ckpt-table-wrap'))", timeout=8.0)
        ckpt = await client.eval_js(
            """(() => {
              const wrap = document.querySelector('.ckpt-table-wrap');
              const wr = wrap ? wrap.getBoundingClientRect() : null;
              const badges = Array.from(document.querySelectorAll('.badge-ckpt-idle')).map(b => {
                const r = b.getBoundingClientRect();
                const cs = window.getComputedStyle(b);
                return { text: b.innerText.trim(), h: r.height, ws: cs.whiteSpace };
              });
              const selects = Array.from(document.querySelectorAll('.ckpt-table select')).map(s => {
                const r = s.getBoundingClientRect();
                return { right: r.right, wrapRight: wr ? wr.right : 9999, clipped: wr ? (r.right > wr.right + 2) : false };
              });
              return { badges, selects };
            })()"""
        )

        # 2. System & Hardware (tab 8)
        await client.click_tab(8)
        await client.wait_for_js("Boolean(document.querySelector('.ps-env-summary-grid'))", timeout=8.0)
        hw = await client.eval_js(
            """(() => {
              const vals = Array.from(document.querySelectorAll('.ps-env-summary-grid .metric-value')).map(el => ({
                text: el.innerText,
                wordBreak: window.getComputedStyle(el).wordBreak,
                overflowWrap: window.getComputedStyle(el).overflowWrap,
              }));
              const hasBenchBtn = Array.from(document.querySelectorAll('button')).some(
                b => /run benchmark/i.test(b.innerText || '')
              );
              const hasBenchCard = Boolean(document.querySelector('.kernel-deck-card, .benchmark-comparison-grid'));
              return { vals, hasBenchBtn, hasBenchCard };
            })()"""
        )

        # 3. Extensions Marketplace (tab 6)
        await client.click_tab(6)
        await asyncio.sleep(0.5)
        mkt = await client.eval_js(
            """(() => {
              const txt = document.body.innerText || '';
              const searchBtn = Array.from(document.querySelectorAll('button')).find(
                b => (b.innerText || '').includes('Search')
              );
              const dagBadges = Array.from(document.querySelectorAll('.repo-artifact-badge.plugin')).map(
                el => el.innerText.trim()
              );
              return {
                searchLabel: searchBtn ? searchBtn.innerText.trim() : null,
                hasStaleAuthor: txt.includes('Triune Research'),
                hasLegacyHomeDot: txt.includes('.triune_studio'),
                dagBadges,
              };
            })()"""
        )

        # 4. Trigger toast on Appearance tab and check .ps-toast-stack position/non-overlap
        await client.click_tab(7)
        await client.wait_for_js("document.querySelectorAll('.ps-effects-btn').length === 3", timeout=8.0)
        await client.eval_js("document.querySelectorAll('.ps-effects-btn')[0].click()")
        await asyncio.sleep(0.15)
        toast = await client.eval_js(
            """(() => {
              const stack = document.querySelector('.ps-toast-stack');
              if (!stack) return { exists: false };
              const r = stack.getBoundingClientRect();
              return { exists: true, top: r.top, bottom: r.bottom, right: r.right, inTopHalf: r.top < window.innerHeight / 2 };
            })()"""
        )
        return {"ckpt": ckpt, "hw": hw, "mkt": mkt, "toast": toast}

    res = cdp_browser.run(_run)
    for b in res["ckpt"]["badges"]:
        assert b["h"] < 36, f".badge-ckpt-idle wrapped multi-line: {b}"
    for s in res["ckpt"]["selects"]:
        assert not s["clipped"], f"Checkpoint dropdown clipped beyond table wrap: {s}"

    for v in res["hw"]["vals"]:
        assert v["wordBreak"] != "break-all", f"Metric value allows mid-word break-all: {v}"
    assert res["hw"]["hasBenchBtn"] is True, "Missing 1-click Run Benchmarks button on System & Hardware tab"
    assert res["hw"]["hasBenchCard"] is True, "Missing kernel benchmark cards on System & Hardware tab"

    assert res["mkt"]["searchLabel"] != "Searching...", "Marketplace search button stuck on 'Searching...'"
    assert not res["mkt"]["hasStaleAuthor"], "Stale 'Triune Research' visible in Marketplace"
    assert not res["mkt"]["hasLegacyHomeDot"], "Legacy '.triune_studio' path displayed in Marketplace"
    assert any(b != "DAG Nodes:" and len(b) > len("DAG Nodes:") for b in res["mkt"]["dagBadges"]), (
        f"Empty 'DAG Nodes:' badge in Marketplace: {res['mkt']['dagBadges']}"
    )

    if res["toast"]["exists"]:
        assert res["toast"]["inTopHalf"] is True, f"Toast stack should stay in top-right, not bottom controls: {res['toast']}"


@pytest.mark.tier2
def test_t2_f7_all_13_tabs_two_viewports_zero_horizontal_overflow(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T2-01: All 13 tabs at 1440x900 and 960x640 render cleanly with <=2px horizontal overflow."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        audits: Dict[str, List[Dict[str, Any]]] = {}
        for vw, vh in ((1440, 900), (960, 640)):
            label = f"{vw}x{vh}"
            await _open_clean_studio(client, studio_server.url, vw, vh)
            rows: List[Dict[str, Any]] = []
            for idx, _tab_id, _title in STUDIO_TABS:
                await client.click_tab(idx)
                rows.append(await client.audit_dom())
            audits[label] = rows
        return {
            "audits": audits,
            "console_errors": list(client.console_errors),
            "page_errors": list(client.page_errors),
        }

    res = cdp_browser.run(_run)
    assert res["console_errors"] == [] and res["page_errors"] == []
    for label, rows in res["audits"].items():
        assert len(rows) == 13
        for a in rows:
            assert not a["runtimeErrorBanner"], f"Runtime error banner on {label}: {a}"
            assert a["textLen"] > 40, f"Empty tab on {label}: {a}"
            assert a["overflowPx"] <= 2, f"Horizontal overflow on {label} ({a['activeNavTitle']}): {a['overflowPx']}px"
            assert not a["hasObjectObject"], f"[object Object] leaked on {label} ({a['activeNavTitle']})"


@pytest.mark.tier2
def test_t2_f7_dark_and_light_themes_contrast_matrix_key_tabs(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T2-02: Dark & light themes maintain >= 4.5:1 text contrast on chat, models, environment, and modules."""

    async def _run(client: StudioCDPClient) -> Dict[str, List[Dict[str, Any]]]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)
        failures: Dict[str, List[Dict[str, Any]]] = {}
        for tid in ("hypnagogic", "phosphene", "mirage", "euphoria"):
            await client.select_theme(tid)
            bad: List[Dict[str, Any]] = []
            for tab_idx in (0, 6, 8, 9):
                await client.click_tab(tab_idx)
                items = await client.check_wcag_contrast(
                    [".card h3", ".card p", ".btn-sec", ".metric-value", ".repo-card-installed"]
                )
                for it in items:
                    if it["cr"] < 4.5 and it["text"]:
                        bad.append({"tab": tab_idx, **it})
            failures[tid] = bad
        return failures

    res = cdp_browser.run(_run)
    for tid, bad in res.items():
        assert bad == [], f"Contrast failures under theme {tid}: {bad[:5]}"


@pytest.mark.tier2
def test_t2_f7_chat_long_checkpoint_pill_truncates_with_ellipsis(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T2-03: Composer checkpoint pill enforces nowrap + overflow hidden + ellipsis truncation."""

    async def _run(client: StudioCDPClient) -> List[Dict[str, Any]]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)
        await client.click_tab(0)
        await client.wait_for_js("Boolean(document.querySelector('.chat-input-meta-left'))", timeout=8.0)
        return await client.eval_js(
            """(() => {
              const pills = Array.from(document.querySelectorAll('.chat-input-meta-left .pill'));
              return pills.map(p => {
                const cs = window.getComputedStyle(p);
                return {
                  text: p.innerText.trim(),
                  overflow: cs.overflow,
                  textOverflow: cs.textOverflow,
                  whiteSpace: cs.whiteSpace,
                };
              });
            })()"""
        )

    pills = cdp_browser.run(_run)
    assert len(pills) >= 1
    assert any(
        p["whiteSpace"] == "nowrap" and p["overflow"] == "hidden" and p["textOverflow"] == "ellipsis"
        for p in pills
    ), f"Expected ellipsis truncation CSS on composer checkpoint pill, got: {pills}"


@pytest.mark.tier2
def test_t2_f7_neural_atlas_mode_cycling_and_stimulate_clean(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T2-04: Cycling Neural Atlas 2D/3D modes and clicking Stimulate produces zero errors & valid canvas."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)
        await client.click_tab(0)
        await client.wait_for_js("Boolean(document.querySelector('.neuron-theater-canvas-wrap canvas'))", timeout=8.0)
        labels: List[str] = []
        for _ in range(3):
            lbl = await client.eval_js(
                "document.querySelectorAll('.neuron-theater-controls .neuron-ctrl-btn')[0]?.innerText?.trim()"
            )
            labels.append(str(lbl))
            await client.eval_js(
                "document.querySelectorAll('.neuron-theater-controls .neuron-ctrl-btn')[0]?.click()"
            )
            await asyncio.sleep(0.15)
        await client.eval_js("document.querySelector('.btn-stimulate-synapse')?.click()")
        await asyncio.sleep(0.2)
        dims = await client.eval_js(
            """(() => {
              const c = document.querySelector('.neuron-theater-canvas-wrap canvas');
              return c ? { w: c.width, h: c.height } : null;
            })()"""
        )
        return {
            "labels": labels,
            "dims": dims,
            "page_errors": list(client.page_errors),
            "console_errors": list(client.console_errors),
        }

    res = cdp_browser.run(_run)
    assert res["dims"] and res["dims"]["w"] > 0 and res["dims"]["h"] > 0
    assert len(set(res["labels"])) >= 2, f"Atlas mode cycling did not change modes: {res['labels']}"
    assert res["page_errors"] == [] and res["console_errors"] == []


@pytest.mark.tier2
def test_t2_f7_node_canvas_fit_button_and_checkpoints_narrow_viewport(
    studio_server: StudioServerHandle, cdp_browser: ChromeBrowserController
) -> None:
    """F7-T2-05: Fit DAG button keeps Autograd Training Step unclipped; Checkpoints fits cleanly at 960x640."""

    async def _run(client: StudioCDPClient) -> Dict[str, Any]:
        await _open_clean_studio(client, studio_server.url, 1440, 900)
        await client.click_tab(2)
        await client.wait_for_js("Boolean(document.querySelector('button[title=\"Fit DAG to Viewport\"]'))", timeout=8.0)
        await client.eval_js("document.querySelector('button[title=\"Fit DAG to Viewport\"]')?.click()")
        await asyncio.sleep(0.2)
        fit_check = await client.eval_js(
            """(() => {
              const canvas = document.querySelector('.ps-dag-canvas');
              const nodes = Array.from(document.querySelectorAll('.ps-dag-node, .dag-node'));
              const autograd = nodes.find(n => (n.innerText || '').includes('Autograd Training Step'));
              if (!canvas || !autograd) return { ok: false };
              const cr = canvas.getBoundingClientRect();
              const ar = autograd.getBoundingClientRect();
              return { ok: true, inside: ar.right <= cr.right + 2 && ar.left >= cr.left - 2 };
            })()"""
        )

        await client.set_viewport(960, 640, mobile=False)
        await client.click_tab(4)
        await asyncio.sleep(0.25)
        audit_960 = await client.audit_dom()
        return {"fit": fit_check, "audit_960": audit_960, "page_errors": list(client.page_errors)}

    res = cdp_browser.run(_run)
    assert res["fit"]["ok"] and res["fit"]["inside"], f"Autograd clipped after Fit DAG: {res['fit']}"
    assert res["audit_960"]["overflowPx"] <= 2
    assert res["page_errors"] == []
