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
