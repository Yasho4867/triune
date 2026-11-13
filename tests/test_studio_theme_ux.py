from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "studio/src/app.js").read_text(encoding="utf-8")
CSS = (ROOT / "studio/src/style.css").read_text(encoding="utf-8")


def test_neural_inspector_surfaces_follow_active_theme():
    """The shared Playground/Model Zoo inspector must remain legible in light themes."""
    assert ".react-studio-app .neural-intelligence-console" in CSS
    assert ".react-studio-app .transmutation-layer-inspector" in CSS
    assert "background: var(--bg-card) !important;" in CSS
    assert "color: var(--text-main);" in CSS
    assert "color: 'var(--text-main)'" in APP


def test_inspector_accents_and_controls_are_theme_aware():
    assert "--inspector-reflex" in CSS
    assert "--inspector-limbic" in CSS
    assert "--inspector-cortex" in CSS
    assert ".react-studio-app .neural-console-tab-btn.active" in CSS
    assert "background: 'var(--bg-card)'" in APP
    assert "color: 'var(--text-main)'" in APP


def test_neural_console_has_compact_viewport_fallbacks():
    assert "@media (max-width: 640px)" in CSS
    assert ".react-studio-app .neural-intelligence-console.dock-right" in CSS
    assert "min-width: 0 !important;" in CSS


def test_styles_do_not_reference_missing_font_token():
    assert "var(--font-sans)" not in CSS


def test_live_active_states_and_primary_actions_follow_selected_palette():
    nav_active = CSS.split(".nav-item.active {", 1)[1].split("}", 1)[0]
    assert "background: var(--primary-glow)" in nav_active
    assert "border-color: var(--border-glow)" in nav_active
    assert "color: var(--primary)" in nav_active
    assert "rgba(139, 92, 246" not in nav_active
    assert "background: var(--primary);" in CSS.split(".btn-send {", 1)[1].split("}", 1)[0]
    assert "color: var(--primary-contrast);" in CSS.split(".btn-send {", 1)[1].split("}", 1)[0]
    assert "--primary-contrast': contrastRatio" in APP
    assert "borderColor: 'var(--border-glow)'" in APP


def test_neural_graph_scales_to_canvas_and_avoids_overlapping_labels():
    assert "const sceneScale = Math.min(2, Math.max(0.8, Math.min(width / 600, height / 150)))" in APP
    assert "const labelRects = [];" in APP
    assert "labelRects.some(other =>" in APP
    assert "labelRects.length >= 12" in APP
    assert "customHeight: dockMode === 'dock-top' ? '100%' : bodyHeight" in APP
