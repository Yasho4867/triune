"""Opaque-Box End-to-End Release Pipeline Test Suite (R1-R5) for Lucid Machines v1.0.0.

Runs against the active `lucid_machines` package (with `triune` backward-compatibility
shim verification) via dynamic package resolution.
"""

from __future__ import annotations

import ast
import functools
import http.server
import importlib
import importlib.util
import inspect
import json
import os
import pickle
import re
import socketserver
import subprocess
import threading
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest


def _resolve_pkg_name() -> str:
    if importlib.util.find_spec("lucid_machines") is None:
        raise RuntimeError("lucid_machines package is required")
    return "lucid_machines"


def _resolve_pkg() -> Any:
    if importlib.util.find_spec("lucid_machines") is None:
        raise RuntimeError("lucid_machines package is required")
    return importlib.import_module("lucid_machines")


def _import_submod(subpath: str) -> Any:
    pkg_name = _resolve_pkg_name()
    return importlib.import_module(f"{pkg_name}.{subpath}")


# ============================================================================
# R1: Release Name Report & Backward-Compatibility Aliases
# ============================================================================


@pytest.mark.e2e
def test_r1_release_name_report_exists_and_complete(repo_root: Path) -> None:
    report_path = repo_root / "docs" / "RELEASE_NAME_REPORT.md"
    assert report_path.is_file(), "Missing docs/RELEASE_NAME_REPORT.md"
    text = report_path.read_text(encoding="utf-8")
    assert len(text.strip()) > 500, "docs/RELEASE_NAME_REPORT.md is unexpectedly short"

    # Must document 4-layer brand separation: Lucid Machines, Triune architecture, Junkie model, Polysom Studio
    assert "Lucid Machines" in text
    assert "lucid_machines" in text or "lucid-machines" in text
    assert "Triune" in text
    assert "Junkie" in text
    assert "Polysom Studio" in text
    assert "LUCID_MACHINES_" in text
    assert "1.0.0" in text

    # Rejected previous draft name must be completely purged
    lower = text.lower()
    rejected_draft = "axo" + "steer"
    assert rejected_draft not in lower, "docs/RELEASE_NAME_REPORT.md must contain zero rejected draft name hits"

    # Must cover public registries & web/DNS checks
    for registry in ("pypi", "npm", "github"):
        assert registry in lower, f"Conflict report missing registry check for {registry}"
    assert "lucidmachines.github.io" in lower or "dns" in lower or "domain" in lower, (
        "RELEASE_NAME_REPORT.md must document domain/GitHub Pages conflict findings"
    )

    # Explicit non-legal / trademark disclaimer
    assert "trademark" in lower and ("legal" in lower or "clearance" in lower), (
        "RELEASE_NAME_REPORT.md must include an explicit non-legal/trademark clearance disclaimer"
    )


@pytest.mark.e2e
def test_r1_dynamic_package_and_compat_aliases(repo_root: Path) -> None:
    pkg = _resolve_pkg()
    assert pkg is not None
    base_mod = _import_submod("model.base")
    assert hasattr(base_mod, "get_model_spec")
    assert hasattr(base_mod, "_LEGACY_PRESET_ALIASES")
    assert hasattr(base_mod, "CANONICAL_MODEL_SPECS")

    # Canonical Triune architecture presets & legacy aliases must resolve cleanly
    for preset_id in (
        "triune-micro",
        "triune-1.5b",
        "triune-moe",
        "triune-small",
        "triune-nano",
        "triune-base",
    ):
        spec = base_mod.get_model_spec(preset_id)
        assert spec is not None
        assert spec.num_layers >= 4
        assert spec.hidden_dim > 0

    # Verify models/configs/triune-*.json + junkie.json exist and zero rejected draft configs remain
    cfg_dir = repo_root / "models" / "configs"
    if cfg_dir.is_dir():
        triune_cfgs = list(cfg_dir.glob("triune-*.json"))
        assert len(triune_cfgs) >= 4, f"Expected >=4 models/configs/triune-*.json presets, found {len(triune_cfgs)}"
        assert (cfg_dir / "junkie.json").is_file(), (
            "Missing dedicated models/configs/junkie.json inaugural model config"
        )
        rejected_draft = "axo" + "steer"
        assert [p for p in cfg_dir.glob("*.json") if rejected_draft in p.name.lower()] == [], (
            "Rejected draft model configs must be removed"
        )

    # Verify 4-layer separation: zero disallowed class names in package exports
    disallowed_cls = "Lucid" + "Transformer"
    disallowed_junkie_cls = "Junkie" + "Transformer"
    if importlib.util.find_spec("lucid_machines"):
        lm_pkg = importlib.import_module("lucid_machines")
        assert not hasattr(lm_pkg, disallowed_cls), f"{disallowed_cls} must not exist (core architecture is Triune)"
        assert not hasattr(lm_pkg, disallowed_junkie_cls), f"{disallowed_junkie_cls} must not exist (Junkie is a model)"


# ============================================================================
# R2: CentroidSteer Multi-Directional Upgrade, Orthonormality & Validation
# ============================================================================


@pytest.mark.e2e
def test_r2_moe_ffn_true_hidden_centroid_capture_and_shapes() -> None:
    import torch

    moe_mod = _import_submod("model.moe")
    MoE_FFN = moe_mod.MoE_FFN

    torch.manual_seed(42)
    dim = 32
    num_experts = 4
    moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=2)
    moe.train()

    x = torch.randn(2, 16, dim, requires_grad=True)
    out = moe(x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 16, dim), f"Expected output shape (2, 16, {dim}), got {out.shape}"

    # Verify input centroids and true hidden post-activation centroids
    assert hasattr(moe, "last_centroids"), "MoE_FFN must expose last_centroids"
    assert moe.last_centroids is not None
    assert moe.last_centroids.shape[0] == num_experts
    assert moe.last_centroids.shape[-1] == dim
    assert float(moe.last_centroids.abs().sum().item()) > 0.0

    assert hasattr(moe, "last_hidden_centroids"), (
        "MoE_FFN must capture true hidden post-activation centroids in last_hidden_centroids ([E, H])"
    )
    assert moe.last_hidden_centroids is not None
    hidden_dim = moe.experts[0][0].out_features
    assert moe.last_hidden_centroids.shape == (num_experts, hidden_dim), (
        f"Expected last_hidden_centroids shape {(num_experts, hidden_dim)}, got {moe.last_hidden_centroids.shape}"
    )
    assert float(moe.last_hidden_centroids.abs().sum().item()) > 0.0, (
        "last_hidden_centroids remained all zeros after training forward pass"
    )


@pytest.mark.e2e
def test_r2_centroid_steer_orthonormality_and_independent_steer_step_reset() -> None:
    import torch
    import torch.nn as nn

    moe_mod = _import_submod("model.moe")
    optim_mod = _import_submod("optim")
    MoE_FFN = moe_mod.MoE_FFN
    CentroidSteerOptimizer = optim_mod.CentroidSteerOptimizer

    class _ToyMoEModel(nn.Module):
        def __init__(self, dim: int = 32, num_experts: int = 4) -> None:
            super().__init__()
            self.embed = nn.Embedding(64, dim)
            self.moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=2)
            self.head = nn.Linear(dim, 64, bias=False)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            h = self.embed(idx)
            h = self.moe(h)
            if isinstance(h, tuple):
                h = h[0]
            return self.head(h)

    torch.manual_seed(1337)
    model = _ToyMoEModel(dim=32, num_experts=4)
    model.train()

    opt = CentroidSteerOptimizer(
        model,
        lr=1e-3,
        rank=4,
        update_gap=6,
        steer_update_gap=2,
        steer_scale=0.25,
    )

    steer_steps_observed: list[int] = []
    subspace_steps_observed: list[int] = []

    for step_idx in range(6):
        tokens = torch.randint(0, 64, (2, 16))
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 64), tokens.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Check active expert layer_groups for orthonormality ||Q^T Q - I||_F < 1e-4
        assert hasattr(opt, "layer_groups") and len(opt.layer_groups) > 0
        for g in opt.layer_groups:
            for key in ("Q_L", "Q_R"):
                q_mat = g.get(key)
                if q_mat is None:
                    continue
                # Extract non-zero active columns
                col_norms = q_mat.float().norm(dim=0)
                active_cols = col_norms > 1e-6
                if int(active_cols.sum().item()) > 0:
                    q_act = q_mat[:, active_cols].float()
                    k_cols = q_act.size(1)
                    gram = q_act.T @ q_act
                    eye = torch.eye(k_cols, device=gram.device, dtype=gram.dtype)
                    frob_err = float(torch.linalg.norm(gram - eye, ord="fro").item())
                    assert frob_err < 1e-4, (
                        f"Orthonormality violation on {key} at step {step_idx}: ||Q^T Q - I||_F = {frob_err:.6e} >= 1e-4"
                    )

        g0 = opt.layer_groups[0]
        st0 = g0.get("state", {})
        s_step = st0.get("steer_step", g0.get("steer_step"))
        sub_step = st0.get("subspace_step")
        if s_step is not None:
            steer_steps_observed.append(int(s_step))
        if sub_step is not None:
            subspace_steps_observed.append(int(sub_step))

    # Verify independent steer_step tracking and reset on steering refresh (steer_update_gap=2 < update_gap=6)
    assert len(steer_steps_observed) > 0, (
        "CentroidSteerOptimizer must track an independent steer_step counter in group state"
    )
    assert max(subspace_steps_observed) > max(steer_steps_observed) or steer_steps_observed.count(1) >= 2, (
        f"Expected steer_step to reset independently on steer refresh (steer_update_gap=2, update_gap=6); "
        f"got steer_steps={steer_steps_observed}, subspace_steps={subspace_steps_observed}"
    )


@pytest.mark.e2e
def test_r2_validation_report_and_metrics_exist(repo_root: Path) -> None:
    report_path = repo_root / "benchmarks" / "CENTROIDSTEER_UPGRADE_VALIDATION_REPORT.md"
    assert report_path.is_file(), "Missing benchmarks/CENTROIDSTEER_UPGRADE_VALIDATION_REPORT.md"
    report_text = report_path.read_text(encoding="utf-8")
    assert len(report_text.strip()) > 500, "CENTROIDSTEER_UPGRADE_VALIDATION_REPORT.md is too short"

    lower = report_text.lower()
    # Must mention scientific baseline comparisons & multi-seed validation
    for term in ("adamw", "seed"):
        assert term in lower, f"Validation report missing expected term {term!r}"
    assert "no_steer" in lower or "nosteer" in lower or "no-steer" in lower, (
        "Validation report must compare against no-steer baseline"
    )

    # Verify JSON validation artifact exists in benchmarks/
    bench_dir = repo_root / "benchmarks"
    json_candidates = [
        p
        for p in bench_dir.glob("*.json")
        if "centroid" in p.name.lower() and ("upgrade" in p.name.lower() or "validate" in p.name.lower())
    ]
    assert len(json_candidates) >= 1, (
        "Missing JSON benchmark artifact for CentroidSteer upgrade validation in benchmarks/"
    )
    data = json.loads(json_candidates[0].read_text(encoding="utf-8"))
    assert isinstance(data, dict) and len(data) > 0


# ============================================================================
# R3: Preservation Guardrails & Safe Branch/Baseline
# ============================================================================


@pytest.mark.e2e
def test_r3_preservation_of_checkpoints_data_papers_and_tokenizer(repo_root: Path) -> None:
    # 1. Checkpoints preserved
    ckpt_dir = repo_root / "checkpoints"
    assert ckpt_dir.is_dir(), "checkpoints/ directory missing"
    pt_files = list(ckpt_dir.glob("*.pt"))
    assert len(pt_files) >= 20, f"Expected >=20 preserved .pt checkpoints in checkpoints/, found {len(pt_files)}"
    assert (ckpt_dir / "triune_step_0.pt").is_file(), "Preserved checkpoint triune_step_0.pt missing"
    assert (ckpt_dir / "triune_studio_step_36500.pt").is_file(), (
        "Preserved checkpoint triune_studio_step_36500.pt missing"
    )

    # 2. Data preserved
    data_dir = repo_root / "data"
    assert data_dir.is_dir(), "data/ directory missing"
    for fname in ("fineweb_edu_real.jsonl", "fineweb_edu_large.jsonl", "finetune.jsonl"):
        fpath = data_dir / fname
        assert fpath.is_file() and fpath.stat().st_size > 0, f"Preserved dataset data/{fname} missing or empty"

    # 3. Papers preserved
    papers_dir = repo_root / "papers"
    assert (papers_dir / "paper1_centroidsteer" / "main.tex").is_file()
    assert (papers_dir / "paper1_centroidsteer" / "centroidsteer.pdf").is_file()
    assert (papers_dir / "paper2_triune_systems" / "main.tex").is_file()
    assert (papers_dir / "paper2_triune_systems" / "triune_architecture.pdf").is_file()

    # 4. Tokenizer preserved
    tok_path = repo_root / "triune_tokenizer.json"
    assert tok_path.is_file() and tok_path.stat().st_size > 100, "triune_tokenizer.json must remain intact"


@pytest.mark.e2e
def test_r3_workspace_rules_untracked_and_release_branch(repo_root: Path) -> None:
    assert (repo_root / "WORKSPACE_RULES.md").is_file(), "Local WORKSPACE_RULES.md must exist on disk"
    res = subprocess.run(
        ["git", "ls-files", "WORKSPACE_RULES.md"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "", "git ls-files WORKSPACE_RULES.md MUST be empty (untracked policy)"

    branch_res = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert branch_res.stdout.strip() in ("release/lucid-machines-v1.0.0", "HEAD"), (
        f"Expected branch release/lucid-machines-v1.0.0, got {branch_res.stdout.strip()}"
    )


# ============================================================================
# R4: Rock-Solid Security Regression Checks
# ============================================================================


class _ExploitPayload:
    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (os.system, ("echo SHOULD_NEVER_RUN",))


@pytest.mark.e2e
def test_r4_weights_only_enforced_in_finetune_and_scripts(repo_root: Path, tmp_path: Path) -> None:
    pkg_name = _resolve_pkg_name()
    target_files = [
        repo_root / pkg_name / "trainer" / "finetune.py",
        repo_root / "scripts" / "rescue_checkpoint.py",
        repo_root / "benchmarks" / "verify_triangular_qr_residuals.py",
    ]
    for fpath in target_files:
        assert fpath.is_file(), f"Expected file {fpath} to exist"
        source = fpath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_str = ast.unparse(node.func)
                if func_str.endswith("torch.load"):
                    kw_map = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
                    assert "weights_only" in kw_map, f"Missing weights_only=True in torch.load at {fpath}:{node.lineno}"
                    val_node = kw_map["weights_only"]
                    assert not (isinstance(val_node, ast.Constant) and val_node.value is False), (
                        f"Unsafe weights_only=False in {fpath}:{node.lineno}"
                    )

    # Verify untrusted pickle payload fails to deserialize when loading adapter weights
    malicious_pt = tmp_path / "adapter_weights.pt"
    malicious_pt.write_bytes(pickle.dumps({"malicious": _ExploitPayload()}))
    import torch

    with pytest.raises(Exception):
        torch.load(malicious_pt, map_location="cpu", weights_only=True)


@pytest.mark.e2e
def test_r4_api_path_confinement_and_proxy_auth_and_headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    server_mod = _import_submod("api.server")
    app = server_mod.create_app()
    client = TestClient(app)

    # 1. Proxy headers (X-Forwarded-For / X-Real-IP / Forwarded) MUST require valid bearer token
    test_token = "a" * 40
    monkeypatch.setenv("TRIUNE_API_TOKEN", test_token)
    monkeypatch.setenv("LUCID_MACHINES_API_TOKEN", test_token)

    unauth_resp = client.get(
        "/v1/system/config",
        headers={"X-Forwarded-For": "203.0.113.195"},
    )
    assert unauth_resp.status_code in (401, 403, 503), (
        f"Reverse-proxied request without bearer token must be rejected, got {unauth_resp.status_code}"
    )

    auth_resp = client.get(
        "/v1/system/config",
        headers={
            "X-Forwarded-For": "203.0.113.195",
            "Authorization": f"Bearer {test_token}",
        },
    )
    assert auth_resp.status_code == 200, (
        f"Reverse-proxied request with valid bearer token should succeed, got {auth_resp.status_code}"
    )

    # 2. Security headers must be present on responses
    hdrs_lower = {k.lower(): v for k, v in auth_resp.headers.items()}
    assert hdrs_lower.get("x-content-type-options", "").lower() == "nosniff"
    assert hdrs_lower.get("x-frame-options", "").upper() == "DENY"
    assert "content-security-policy" in hdrs_lower

    # 3. Cross-origin state-changing request must be rejected (CSRF / Origin check)
    csrf_resp = client.post(
        "/v1/models/load",
        json={"model": "triune-small"},
        headers={"Origin": "https://evil-attacker.example.com"},
    )
    assert csrf_resp.status_code in (400, 403), (
        f"Cross-origin POST with untrusted Origin must be rejected (400/403), got {csrf_resp.status_code}"
    )

    # 4. Path confinement: outside-workspace traversal paths must be rejected across endpoints
    outside_file = tmp_path / "outside_secret.pt"
    outside_file.write_bytes(b"not_a_valid_checkpoint")

    for endpoint, payload in [
        ("/v1/models/load", {"model": "../../outside_secret.pt"}),
        ("/v1/models/load", {"model": str(outside_file)}),
        ("/v1/checkpoints/load", {"filename": "../../outside_secret.pt"}),
        ("/v1/datasets/local/register", {"name": "traverse", "path": str(outside_file)}),
        ("/v1/models/export", {"model_id": "triune-small", "format": "safetensors", "output_dir": str(tmp_path)}),
        ("/v1/finetune/export", {"adapter_name": "../../outside_escape"}),
    ]:
        resp = client.post(endpoint, json=payload)
        assert resp.status_code in (400, 403, 404, 422), (
            f"Path confinement check failed on {endpoint} with payload {payload}: got HTTP {resp.status_code}"
        )


@pytest.mark.e2e
def test_r4_gemini_header_only_key_and_untrusted_plugins_and_vllm(repo_root: Path, tmp_path: Path) -> None:
    pkg_name = _resolve_pkg_name()

    # 1. Gemini API key must be sent via x-goog-api-key header, never in ?key= URL query string
    routes_text = (repo_root / pkg_name / "api" / "routes.py").read_text(encoding="utf-8")
    providers_path = repo_root / pkg_name / "agents" / "providers.py"
    providers_text = providers_path.read_text(encoding="utf-8") if providers_path.is_file() else ""
    for label, content in [("routes.py", routes_text), ("providers.py", providers_text)]:
        assert "?key=" not in content and "&key=" not in content, (
            f"Gemini API key must not be passed in URL query string (?key=) in {label}"
        )

    # 2. External workspace plugin scanning must default to trusted=False (refusing unsigned scripts)
    mgr_mod = _import_submod("modules.manager")
    mm = mgr_mod.ModuleManager()
    custom_dir = tmp_path / "unsigned_custom_nodes"
    custom_dir.mkdir(parents=True, exist_ok=True)
    unsigned_plugin = custom_dir / "unsigned_evil_node.py"
    unsigned_plugin.write_text(
        f"from {pkg_name}.plugins.registry import register_node\n"
        "@register_node(name='UnsignedEvilNode', category='Custom')\n"
        "def unsigned_evil_node(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    # Calling discover_and_register_nodes without trusted=True on an external dir must not load unsigned code
    sig = inspect.signature(mm.discover_and_register_nodes)
    assert sig.parameters["trusted"].default is False, (
        f"discover_and_register_nodes default for 'trusted' must be False, got {sig.parameters['trusted'].default}"
    )
    discovered = mm.discover_and_register_nodes(custom_dir)
    assert "UnsignedEvilNode" not in discovered, "Unsigned external plugin must not be loaded when trusted=False"

    # 3. vLLM trust_remote_code must be False
    assert '"trust_remote_code": True' not in routes_text, (
        "routes.py must not export vLLM config with trust_remote_code: True"
    )
    vllm_bridge = repo_root / pkg_name / "export" / "vllm_bridge.py"
    if vllm_bridge.is_file():
        vllm_text = vllm_bridge.read_text(encoding="utf-8")
        assert '"trust_remote_code": True' not in vllm_text and "'trust_remote_code': True" not in vllm_text

    # 4. Security regression test file must exist
    assert (repo_root / "tests" / "test_security_regression.py").is_file(), "Missing tests/test_security_regression.py"


# ============================================================================
# R5: Website, Local Launch Kit, Release Notes, Deployment Guide & Studio UI
# ============================================================================


class _RelativeAssetHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.asset_refs: list[tuple[str, str]] = []
        self.anchor_refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag in ("link", "script", "img", "source"):
            if tag == "link" and (attr_map.get("rel") or "").lower() == "canonical":
                return
            ref = attr_map.get("href") or attr_map.get("src")
            if ref:
                self.asset_refs.append((tag, ref))
        elif tag == "a":
            href = attr_map.get("href")
            if href:
                self.anchor_refs.append(href)


@pytest.mark.e2e
def test_r5_website_integrity_relative_links_and_live_http(repo_root: Path) -> None:
    website_dir = repo_root / "website"
    index_html = website_dir / "index.html"
    assert index_html.is_file(), "Missing website/index.html"
    assert (website_dir / "src" / "main.tsx").is_file(), "Missing website/src/main.tsx"
    assert (website_dir / "src" / "App.tsx").is_file(), "Missing website/src/App.tsx"
    svg_files = list(website_dir.rglob("*.svg"))
    assert len(svg_files) >= 1, "Expected SVG visual diagrams in website/"

    html_text = index_html.read_text(encoding="utf-8")
    lower_html = html_text.lower()
    rejected_draft = "axo" + "steer"
    assert rejected_draft not in lower_html, "website/index.html must contain zero rejected draft name hits"
    for required_topic in (
        "lucid machines",
        "triune",
        "junkie",
        "polysom",
        "reflex",
        "limbic",
        "cortex",
        "centroidsteer",
        "security",
    ):
        assert required_topic in lower_html, f"website/index.html missing required topic {required_topic!r}"

    parser = _RelativeAssetHTMLParser()
    parser.feed(html_text)
    assert len(parser.asset_refs) >= 2, "website/index.html should reference the favicon and React entry module"

    for tag, ref in parser.asset_refs:
        if ref.startswith("data:"):
            continue
        assert not ref.startswith(("http://", "https://", "//")), (
            f"Website asset <{tag}='{ref}'> must use local paths only (zero external CDN URLs)"
        )
        clean_ref = ref.split("#", 1)[0].split("?", 1)[0]
        if not clean_ref:
            continue
        if clean_ref.startswith("/"):
            public_root = (website_dir / "public").resolve()
            target = (public_root / clean_ref.lstrip("/")).resolve()
            assert target.is_relative_to(public_root), f"Asset {ref} escapes website/public/"
            assert target.is_file(), f"Referenced Vite public asset {clean_ref} does not exist in website/public/"
        else:
            target = (website_dir / clean_ref).resolve()
            assert target.is_relative_to(website_dir.resolve()), f"Asset {ref} escapes website/"
            assert target.is_file(), f"Referenced website asset {clean_ref} does not exist on disk"

    dist_dir = website_dir / "dist"
    dist_index = dist_dir / "index.html"
    assert dist_index.is_file(), "Missing built website/dist/index.html"
    dist_html = dist_index.read_text(encoding="utf-8")
    assert "noindex" not in dist_html.lower(), "website/dist/index.html must not contain noindex"
    dist_parser = _RelativeAssetHTMLParser()
    dist_parser.feed(dist_html)
    assert len(dist_parser.asset_refs) >= 2, "website/dist/index.html should reference built assets"
    for tag, ref in dist_parser.asset_refs:
        if ref.startswith("data:"):
            continue
        assert not ref.startswith(("http://", "https://", "//")), (
            f"Built website asset <{tag}='{ref}'> must use local paths only (zero external CDN URLs)"
        )
        clean_ref = ref.split("#", 1)[0].split("?", 1)[0]
        if clean_ref:
            target = (dist_dir / clean_ref.lstrip("/")).resolve()
            assert target.is_relative_to(dist_dir.resolve()), f"Built asset {ref} escapes website/dist/"
            assert target.is_file(), f"Built website asset {clean_ref} does not exist in website/dist/"

    # Live static HTTP server smoke check (source website/ + built website/dist/)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(website_dir))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            for rel_path in ("index.html", "src/main.tsx", "src/App.tsx", "public/lucid-symbol.svg"):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/{rel_path}", timeout=5) as resp:
                    assert resp.status == 200
                    body = resp.read()
                    assert len(body) > 100
        finally:
            httpd.shutdown()

    dist_handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(dist_dir))
    with socketserver.TCPServer(("127.0.0.1", 0), dist_handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            dist_paths = ["index.html", "lucid-symbol.svg", "favicon.png", "robots.txt"]
            for _, ref in dist_parser.asset_refs:
                clean_ref = ref.split("#", 1)[0].split("?", 1)[0].lstrip("/")
                if clean_ref and clean_ref not in dist_paths:
                    dist_paths.append(clean_ref)
            for rel_path in dist_paths:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/{rel_path}", timeout=5) as resp:
                    assert resp.status == 200
                    body = resp.read()
                    assert len(body) > 10
        finally:
            httpd.shutdown()


@pytest.mark.e2e
def test_r5_launch_kit_release_notes_and_deployment_guide(repo_root: Path) -> None:
    launch_dir = repo_root / "launch_kit"
    assert launch_dir.is_dir(), "Missing launch_kit/ directory"
    required_kit_files = (
        "README.md",
        "HERO_MESSAGING.md",
        "LAUNCH_BLOG.md",
        "SOCIAL_POSTS.md",
        "AD_COPY_VARIANTS.md",
    )
    for fname in required_kit_files:
        fpath = launch_dir / fname
        assert fpath.is_file() and fpath.stat().st_size > 200, f"Missing or empty launch_kit/{fname}"

    social_text = (launch_dir / "SOCIAL_POSTS.md").read_text(encoding="utf-8").lower()
    for platform in ("linkedin", "hacker news", "reddit"):
        assert platform in social_text, f"launch_kit/SOCIAL_POSTS.md missing draft for {platform}"
    assert "x " in social_text or "twitter" in social_text or "x/" in social_text

    kit_svgs = list(launch_dir.rglob("*.svg"))
    assert len(kit_svgs) >= 1, "Expected SVG benchmark/architecture visual cards in launch_kit/"

    # RELEASE_NOTES.md and DEPLOYMENT_GUIDE.md
    rel_notes = repo_root / "RELEASE_NOTES.md"
    dep_guide = repo_root / "DEPLOYMENT_GUIDE.md"
    assert rel_notes.is_file() and rel_notes.stat().st_size > 400, "Missing or incomplete RELEASE_NOTES.md"
    assert dep_guide.is_file() and dep_guide.stat().st_size > 400, "Missing or incomplete DEPLOYMENT_GUIDE.md"

    rel_text = rel_notes.read_text(encoding="utf-8")
    rejected_draft = "axo" + "steer"
    assert "Lucid Machines" in rel_text and "1.0.0" in rel_text
    assert "Junkie" in rel_text and "Polysom Studio" in rel_text
    assert "Triune" in rel_text
    assert rejected_draft not in rel_text.lower()

    dep_text = dep_guide.read_text(encoding="utf-8").lower()
    assert rejected_draft not in dep_text
    for target in ("github releases", "github pages", "pypi", "twine", "hugging face", "polysomstudiosetup.exe"):
        assert target in dep_text, f"DEPLOYMENT_GUIDE.md missing instructions for {target}"


@pytest.mark.e2e
def test_r5_studio_ui_brand_assets_zero_emojis_and_zero_gradients(repo_root: Path) -> None:
    studio_src = repo_root / "studio" / "src"
    logo_paths = [
        studio_src / "assets" / "polysom-logo.svg",
        studio_src / "polysom-logo.svg",
    ]
    existing_logos = [p for p in logo_paths if p.is_file()]
    assert len(existing_logos) >= 1, "Missing polysom-logo.svg in studio/src/assets/ or studio/src/"
    for lp in existing_logos:
        svg_content = lp.read_text(encoding="utf-8")
        assert "<svg" in svg_content and "</svg>" in svg_content
        assert "linearGradient" not in svg_content and "radialGradient" not in svg_content, (
            f"Studio SVG asset {lp.name} must not use SVG gradients"
        )

    # Verify all 5 locked Polysom Studio hallucination/dream themes are wired into studio/src/
    app_js_text = (studio_src / "app.js").read_text(encoding="utf-8")
    css_text = (studio_src / "style.css").read_text(encoding="utf-8")
    for theme_name in ("Hypnagogic", "Phosphene", "Pareidolia", "Afterimage", "Mirage"):
        assert theme_name in app_js_text, f"Missing locked theme {theme_name!r} in studio/src/app.js"
        assert theme_name.lower() in css_text.lower(), (
            f"Missing theme CSS rules for {theme_name!r} in studio/src/style.css"
        )

    # Zero CSS gradients and zero emojis in Studio UI surfaces
    gradient_re = re.compile(r"(?:linear|radial|conic)-gradient\s*\(", re.IGNORECASE)
    emoji_re = re.compile(
        "[\U0001f300-\U0001f5ff\U0001f600-\U0001f64f\U0001f680-\U0001f6ff\U0001f900-\U0001f9ff\U0001fa70-\U0001faff]"
    )
    rejected_draft = "axo" + "steer"
    for ui_file in [studio_src / "index.html", studio_src / "style.css", studio_src / "app.js", *existing_logos]:
        text = ui_file.read_text(encoding="utf-8")
        assert rejected_draft not in text.lower(), f"Rejected draft name found in Studio UI file {ui_file.name}"
        assert not gradient_re.search(text), f"Prohibited CSS gradient found in Studio UI file {ui_file.name}"
        assert not emoji_re.search(text), f"Prohibited emoji pictograph found in Studio UI file {ui_file.name}"
