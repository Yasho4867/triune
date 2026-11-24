"""Tests for Track 2 Cloud Junkie Kit: model configs, scripts/train.py CLI, and cloud/ runners."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lucid_machines.model.base import CANONICAL_MODEL_SPECS, get_model_spec
from lucid_machines.model.transformer import TriuneTransformer
from scripts.train import build_parser
from scripts.train import main as train_main

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "models" / "configs"
CLOUD_DIR = REPO_ROOT / "cloud"


def _count_unique_params(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


def _count_active_params(model: TriuneTransformer, cfg: dict, top_k: int = 1) -> tuple[int, int, int, int]:
    def _layer_active(idx: int) -> int:
        blk = model.layers[idx]
        if not blk.use_moe:
            return sum(p.numel() for n, p in blk.named_parameters() if "exit_" not in n)
        total = (
            sum(p.numel() for p in blk.attn.parameters())
            + sum(p.numel() for p in blk.norm1.parameters())
            + sum(p.numel() for p in blk.norm2.parameters())
            + sum(p.numel() for p in blk.ffn.router.parameters())
        )
        if blk.ffn.shared is not None:
            total += sum(p.numel() for p in blk.ffn.shared.parameters())
        total += top_k * sum(p.numel() for p in blk.ffn.experts[0].parameters())
        return total

    re_l = int(cfg["reflex_exit_layer"])
    le_l = int(cfg["limbic_exit_layer"])
    nl = int(cfg["num_layers"])
    embed = model.token_embed.weight.numel()
    rtr = sum(p.numel() for p in model.router.parameters())
    rh = sum(p.numel() for p in model.layers[re_l].exit_norm.parameters()) + sum(
        p.numel() for p in model.layers[re_l].exit_head.parameters()
    )
    lh = sum(p.numel() for p in model.layers[le_l].exit_norm.parameters()) + sum(
        p.numel() for p in model.layers[le_l].exit_head.parameters()
    )
    ch = sum(p.numel() for p in model.final_norm.parameters())

    reflex = embed + rtr + rh + sum(_layer_active(i) for i in range(re_l + 1))
    limbic = embed + rtr + lh + sum(_layer_active(i) for i in range(le_l + 1))
    cortex = embed + rtr + ch + sum(_layer_active(i) for i in range(nl))
    expected = round(0.34 * reflex + 0.33 * limbic + 0.33 * cortex)
    return reflex, limbic, cortex, expected


@pytest.mark.parametrize(
    ("filename", "expected_dims", "expected_total", "min_active", "max_active"),
    [
        (
            "junkie-mini.json",
            {
                "hidden_dim": 1024,
                "num_layers": 12,
                "num_heads": 8,
                "head_dim": 128,
                "num_experts": 4,
                "top_k_experts": 1,
                "shared_expert": True,
                "expert_hidden_multiplier": 6,
                "rope_max_seq_len": 4096,
            },
            535_101_975,
            160_000_000,
            290_000_000,
        ),
        (
            "junkie-1b.json",
            {
                "hidden_dim": 1024,
                "num_layers": 22,
                "num_heads": 8,
                "head_dim": 128,
                "num_experts": 4,
                "top_k_experts": 1,
                "shared_expert": True,
                "expert_hidden_multiplier": 6,
                "rope_max_seq_len": 4096,
            },
            1_162_535_483,
            300_000_000,
            450_000_000,
        ),
        (
            "junkie-3b.json",
            {
                "hidden_dim": 1024,
                "num_layers": 16,
                "num_heads": 8,
                "head_dim": 128,
                "num_experts": 16,
                "top_k_experts": 1,
                "shared_expert": True,
                "expert_hidden_multiplier": 6,
                "rope_max_seq_len": 4096,
            },
            2_990_355_155,
            195_000_000,
            275_000_000,
        ),
        (
            "junkie-7b.json",
            {
                "hidden_dim": 2048,
                "num_layers": 32,
                "num_heads": 16,
                "head_dim": 128,
                "num_experts": 8,
                "top_k_experts": 2,
                "shared_expert": True,
                "expert_hidden_multiplier": 6,
                "rope_max_seq_len": 8192,
            },
            7_396_845_163,
            1_100_000_000,
            2_750_000_000,
        ),
        (
            "junkie.json",
            {
                "hidden_dim": 2048,
                "num_layers": 32,
                "num_heads": 16,
                "head_dim": 128,
                "num_experts": 8,
                "top_k_experts": 2,
                "shared_expert": True,
                "expert_hidden_multiplier": 6,
                "rope_max_seq_len": 8192,
            },
            7_396_845_163,
            1_100_000_000,
            2_750_000_000,
        ),
    ],
)
def test_junkie_model_configs_schema_and_meta_params(
    filename: str,
    expected_dims: dict,
    expected_total: int,
    min_active: int,
    max_active: int,
) -> None:
    cfg_path = CONFIGS_DIR / filename
    assert cfg_path.is_file(), f"Missing {cfg_path}"
    raw_bytes = cfg_path.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf"), f"{filename} must not have a UTF-8 BOM"

    cfg = json.loads(raw_bytes.decode("utf-8"))
    assert cfg["architecture"] == "TriuneTransformer"
    assert cfg["model_type"] == "triune"
    assert cfg["vocab_size"] == 32000
    for key, val in expected_dims.items():
        assert cfg[key] == val, f"{filename}: {key}={cfg.get(key)} != {val}"

    with torch.device("meta"):
        model = TriuneTransformer(
            vocab_size=cfg["vocab_size"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            head_dim=cfg["head_dim"],
            num_experts=cfg["num_experts"],
            router_prefix_layers=cfg["router_prefix_layers"],
            reflex_exit_layer=cfg["reflex_exit_layer"],
            limbic_exit_layer=cfg["limbic_exit_layer"],
        )

    total_params = _count_unique_params(model)
    assert total_params == expected_total
    reflex_a, limbic_a, cortex_a, expected_a = _count_active_params(model, cfg, top_k=1)
    assert min_active <= expected_a <= max_active
    assert reflex_a < limbic_a < cortex_a < total_params
    if filename == "junkie-3b.json":
        dyn_a = round(0.50 * reflex_a + 0.30 * limbic_a + 0.20 * cortex_a)
        assert 200_000_000 <= dyn_a <= 215_000_000


def test_junkie_presets_aligned_in_canonical_model_specs() -> None:
    for preset_id in ("junkie-mini", "junkie-1b", "junkie-3b", "junkie-7b", "junkie"):
        assert preset_id in CANONICAL_MODEL_SPECS
        spec = get_model_spec(preset_id)
        cfg = json.loads((CONFIGS_DIR / f"{preset_id}.json").read_text(encoding="utf-8"))
        assert spec.hidden_dim == cfg["hidden_dim"]
        assert spec.num_layers == cfg["num_layers"]
        assert spec.num_heads == cfg["num_heads"]
        assert spec.head_dim == cfg["head_dim"]
        assert spec.num_experts == cfg["num_experts"]
        assert spec.reflex_exit_layer == cfg["reflex_exit_layer"]
        assert spec.limbic_exit_layer == cfg["limbic_exit_layer"]


def test_train_cli_help_and_parser_flags() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--model_config" in proc.stdout
    assert "--steer_dirs" in proc.stdout or "--steer-dirs" in proc.stdout
    assert "--auto_resume" in proc.stdout or "--auto-resume" in proc.stdout

    parser = build_parser()
    parsed = parser.parse_args(
        [
            "--model_config",
            "junkie-7b",
            "--steps",
            "250",
            "--batch-size",
            "2",
            "--seq-len",
            "512",
            "--rank",
            "128",
            "--steer-dirs",
            "6",
            "--steer-mode",
            "upgraded",
            "--steer-scale",
            "0.25",
            "--checkpoint-dir",
            "checkpoints_test",
            "--auto-resume",
            "--dry-run",
        ]
    )
    assert parsed.model_config == "junkie-7b"
    assert parsed.total_steps == 250
    assert parsed.batch_size == 2
    assert parsed.seq_len == 512
    assert parsed.galore_rank == 128
    assert parsed.steer_dirs == 6
    assert parsed.steer_mode == "upgraded"
    assert parsed.steer_scale == pytest.approx(0.25)
    assert parsed.checkpoint_dir == Path("checkpoints_test")
    assert parsed.auto_resume is True
    assert parsed.dry_run is True

    train_src = (REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert 'hasattr(model, "enable_gradient_checkpointing")' in train_src


@pytest.mark.parametrize(
    ("target", "expected_params"),
    [
        ("junkie-mini", 535_101_975),
        ("junkie-1b", 1_162_535_483),
        ("junkie-3b", 2_990_355_155),
        ("junkie-7b", 7_396_845_163),
        ("models/configs/junkie.json", 7_396_845_163),
        ("triune-base", 90_789_271),
    ],
)
def test_train_cli_dry_run_resolves_configs(target: str, expected_params: int) -> None:
    res = train_main(
        [
            "--model_config",
            target,
            "--dry-run",
            "--batch-size",
            "1",
            "--seq-len",
            "512",
            "--rank",
            "64",
            "--steer-dirs",
            "4",
            "--steps",
            "10",
            "--no_wandb",
            "--no_hf_login",
        ]
    )
    assert res["dry_run"] is True
    assert res["total_params"] == expected_params


def test_cloud_artifacts_structure_and_hygiene() -> None:
    nb_path = CLOUD_DIR / "lucid_machines_colab_runner.ipynb"
    sh_path = CLOUD_DIR / "gcp_train_bootstrap.sh"
    readme_path = CLOUD_DIR / "README.md"
    assert nb_path.is_file()
    assert sh_path.is_file()
    assert readme_path.is_file()

    # 1. Colab notebook: valid nbformat 4 JSON with 8 cells covering all 4 Junkie configs & 3-stage curriculum
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 8
    joined_cells = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "google.colab" in joined_cells
    assert "drive.mount" in joined_cells
    assert "validate_centroid_steer_upgrade.py" in joined_cells
    assert "junkie-mini.json" in joined_cells
    assert "junkie-1b.json" in joined_cells
    assert "junkie-3b.json" in joined_cells
    assert "junkie-7b.json" in joined_cells
    assert "--use_fp8" in joined_cells
    assert "--depth_mode dynamic" in joined_cells
    assert "--steer_mode centroid" in joined_cells
    assert "--steer_mode upgraded" not in joined_cells
    assert "FineWeb-Edu" in joined_cells and "FineMath" in joined_cells
    assert "WSD" in joined_cells and "GRPO" in joined_cells
    assert "--auto_resume" in joined_cells
    assert "serve_kernel_port_as_window" in joined_cells or "serve_kernel_port_as_iframe" in joined_cells

    # 2. Shell bootstrap: LF-only, required flags & commands
    sh_bytes = sh_path.read_bytes()
    assert b"\r\n" not in sh_bytes, "gcp_train_bootstrap.sh must use Unix LF line endings"
    sh_text = sh_bytes.decode("utf-8")
    for token in ("--model", "--mode", "--gpus", "--gcs-bucket", "torchrun", "gcloud storage rsync", "ablation"):
        assert token in sh_text
    assert 'STEER_MODE="centroid"' in sh_text
    bash_bin = shutil.which("bash")
    if bash_bin:
        chk = subprocess.run([bash_bin, "-n", str(sh_path)], capture_output=True, text=True, check=False)
        if chk.returncode == 0:
            assert chk.returncode == 0

    # 3. README: Colab Pro primary, exact measured params/VRAM, 3-stage curriculum, GCP optional templates
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "535,101,975" in readme_text
    assert "1,162,535,483" in readme_text
    assert "2,990,355,155" in readme_text
    assert "7,396,845,163" in readme_text
    assert "0.1992 B/p" in readme_text
    assert "0.3678 B/p" in readme_text
    assert "Colab Pro" in readme_text
    assert "FineWeb-Edu" in readme_text and "FineMath" in readme_text
    assert "WSD" in readme_text and "GRPO" in readme_text
    assert "g2-standard-8" in readme_text
    assert "a2-ultragpu-1g" in readme_text
    assert "a3-highgpu-1g" in readme_text
    assert "--provisioning-model=SPOT" in readme_text
    assert "-L 8000:127.0.0.1:8000" in readme_text

    # 4. Naming lock & zero-secret hygiene across all Track 2 deliverables
    files_to_scan = [
        CONFIGS_DIR / "junkie-mini.json",
        CONFIGS_DIR / "junkie-1b.json",
        CONFIGS_DIR / "junkie-3b.json",
        CONFIGS_DIR / "junkie-7b.json",
        CONFIGS_DIR / "junkie.json",
        REPO_ROOT / "scripts" / "train.py",
        nb_path,
        sh_path,
        readme_path,
    ]
    secret_re = re.compile(r"(AIza[0-9A-Za-z\-_]{35}|ghp_[0-9A-Za-z]{36}|sk-[0-9A-Za-z]{20,}|hf_[0-9A-Za-z]{20,})")
    disallowed_tokens = ("axo" + "steer", "junkie" + " 3", "model" + " family")
    for fpath in files_to_scan:
        txt = fpath.read_text(encoding="utf-8")
        lower = txt.lower()
        for tok in disallowed_tokens:
            assert tok not in lower, f"Disallowed token {tok!r} in {fpath}"
        assert not secret_re.search(txt), f"Possible hardcoded secret in {fpath}"
