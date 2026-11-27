"""Tests for fused BF16 stochastic rounding, MoE subspace gating, and Colab A100-40GB selection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lucid_machines.kernels.fast_bf16_sr import bf16_stochastic_round_update_
from lucid_machines.model.moe import MoE_FFN
from lucid_machines.model.transformer import TriuneTransformer
from lucid_machines.optim.centroid import CentroidSteerOptimizer
from lucid_machines.trainer.engine import TrainingEngine


def _check_sr_unbiased(device: torch.device) -> None:
    param = torch.ones((4096,), device=device, dtype=torch.bfloat16)
    delta = torch.full((4096,), 1e-4, device=device, dtype=torch.float32)
    bf16_stochastic_round_update_(param, delta, beta_wd=1.0, alpha_upd=1.0, seed=1234)
    mean_val = float(param.float().mean().item())
    assert 1.00003 <= mean_val <= 1.00025

    param_wd = torch.full((128,), 2.0, device=device, dtype=torch.bfloat16)
    delta_wd = torch.full((128,), -0.5, device=device, dtype=torch.float32)
    bf16_stochastic_round_update_(param_wd, delta_wd, beta_wd=0.5, alpha_upd=0.5, seed=55)
    assert torch.allclose(param_wd.float(), torch.full((128,), 0.75, device=device), atol=1e-2)


def test_bf16_stochastic_round_update_cpu_and_cuda() -> None:
    _check_sr_unbiased(torch.device("cpu"))
    if torch.cuda.is_available():
        _check_sr_unbiased(torch.device("cuda"))


def test_moe_skip_subspace_stats_updates_bias_but_defers_subspace() -> None:
    moe = MoE_FFN(dim=32, num_experts=4, top_k=1, use_fp8=False, shared_expert=True)
    moe.train()
    x = torch.randn(2, 16, 32, dtype=torch.float32)

    moe._skip_subspace_stats = True
    _ = moe(x, update_stats=True)
    assert moe.last_counts is not None
    assert moe._pending_bias_update is not None
    assert float(moe.last_centroids.abs().sum().item()) == 0.0
    assert float(moe.last_input_subspace.abs().sum().item()) == 0.0

    moe._skip_subspace_stats = False
    _ = moe(x, update_stats=True)
    assert float(moe.last_centroids.abs().sum().item()) > 0.0
    assert float(moe.last_input_subspace.abs().sum().item()) > 0.0


def test_engine_gates_subspace_stats_and_restores_flag() -> None:
    model = TriuneTransformer(
        vocab_size=64,
        hidden_dim=32,
        num_layers=5,
        num_heads=4,
        head_dim=8,
        num_experts=4,
        top_k_experts=1,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
    )
    opt = CentroidSteerOptimizer(model, lr=1e-3, rank=4, steer_update_gap=5)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))

    trainer = SimpleNamespace(
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        grad_accum=2,
        vocab_size=64,
        z_loss_coef=1e-3,
        pad_token_id=-100,
        config={
            "target_depth_dist": [0.5, 0.3, 0.2],
            "depth_mode": "cortex",
            "exploration": "none",
            "exploration_steps": 0,
            "lr": 1e-3,
            "grad_clip": 1.0,
            "usage_ema_decay": 0.99,
            "balance_coef": 0.3,
        },
        next_batch=lambda: (x, y),
        model_autocast=lambda: torch.amp.autocast("cpu", enabled=False),
        lr_schedule=lambda cfg, step: 1e-3,
        set_optimizer_lr=lambda o, lr, expert_lr=None: None,
    )
    engine = TrainingEngine(trainer)
    moe_mods = [m for m in model.modules() if isinstance(m, MoE_FFN)]
    assert moe_mods

    seen_skip_flags: list[bool] = []
    orig_fwd = moe_mods[0].forward

    def _spy_fwd(*args, **kwargs):
        seen_skip_flags.append(bool(moe_mods[0]._skip_subspace_stats))
        return orig_fwd(*args, **kwargs)

    moe_mods[0].forward = _spy_fwd

    # step=0 (<2): only last microbatch (index 1) updates subspace
    engine.step_once(advance_step=True)
    assert seen_skip_flags == [True, False]
    assert all(not m._skip_subspace_stats for m in moe_mods)

    # step=2 ((2+1)%5 != 0): both microbatches skip subspace stats, and flag is restored
    engine.step = 2
    seen_skip_flags.clear()
    engine.step_once(advance_step=True)
    assert seen_skip_flags == [True, True]
    assert all(not m._skip_subspace_stats for m in moe_mods)


def test_colab_cell3_a100_40gb_threshold_and_resume_rank(tmp_path: Path) -> None:
    nb_path = Path(__file__).resolve().parents[1] / "cloud" / "lucid_machines_colab_runner.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell3_src = next(
        "".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).lstrip().startswith("# Cell 3:")
    )
    assert "elif vram_gb >= 35.0:" in cell3_src
    assert 'extra_flags = "--no-grad_checkpoint"' in cell3_src

    ckpt_dir = tmp_path / "junkie-3b"
    ckpt_dir.mkdir()
    torch.save({"config": {"galore_rank": 64, "use_fp8": True}}, ckpt_dir / "latest.pt")

    # Extract tier-selection & checkpoint-rank logic without executing `!python3` or `%cd`
    snippet = []
    for line in cell3_src.splitlines():
        if line.startswith("if vram_gb >= 70.0:") or snippet:
            if line.startswith("fp8_flag ="):
                break
            snippet.append(line)
    ns = {"vram_gb": 39.49, "ckpt_3b": ckpt_dir, "torch": torch, "os": __import__("os"), "supports_fp8": False}
    exec("\n".join(snippet), ns)
    assert ns["batch_val"] == 8
    assert ns["accum_val"] == 2
    assert ns["seq_val"] == 1024
    assert ns["steps_val"] == 100000
    assert ns["extra_flags"] == "--no-grad_checkpoint"
    assert ns["rank_val"] == 64  # inherited from mocked checkpoint on resume
    assert ns["use_fp8_val"] is True
