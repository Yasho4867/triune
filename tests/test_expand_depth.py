"""Unit tests for Triune depth-upscaling / best-layer looping (scripts/expand_depth.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lucid_machines.configs.config import build_config
from lucid_machines.model import TriuneTransformer
from lucid_machines.trainer.checkpoint import compute_checkpoint_fingerprint, extract_canonical_semantics
from scripts.expand_depth import expand_checkpoint_file


def _make_tiny_checkpoint(path: Path) -> tuple[TriuneTransformer, dict]:
    cfg = build_config(
        {
            "model_name": "triune-tiny-expand-test",
            "vocab_size": 128,
            "hidden_dim": 64,
            "num_layers": 6,
            "num_heads": 2,
            "head_dim": 32,
            "num_experts": 2,
            "top_k_experts": 1,
            "shared_expert": True,
            "shared_expert_scale": 0.5,
            "expert_hidden_multiplier": 2,
            "router_prefix_layers": 1,
            "reflex_exit_layer": 2,
            "limbic_exit_layer": 4,
            "rope_max_seq_len": 256,
            "galore_rank": 16,
        }
    )
    model = TriuneTransformer(
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        head_dim=cfg["head_dim"],
        num_experts=cfg["num_experts"],
        top_k_experts=cfg["top_k_experts"],
        shared_expert=cfg["shared_expert"],
        shared_expert_scale=cfg["shared_expert_scale"],
        expert_hidden_multiplier=cfg["expert_hidden_multiplier"],
        rope_max_seq_len=cfg["rope_max_seq_len"],
        router_prefix_layers=cfg["router_prefix_layers"],
        reflex_exit_layer=cfg["reflex_exit_layer"],
        limbic_exit_layer=cfg["limbic_exit_layer"],
    )
    payload = {
        "step": 42,
        "model_state": model.state_dict(),
        "optimizer_state": {"dummy": True},
        "config": cfg,
        "schema_version": 2,
        "fingerprint": compute_checkpoint_fingerprint(cfg),
        "canonical_semantics": extract_canonical_semantics(cfg),
    }
    torch.save(payload, path)
    return model, cfg


@pytest.mark.parametrize("strategy", ["solar_loop", "best_loop"])
def test_expand_depth_strict_load_reflex_parity_and_recursive(tmp_path: Path, strategy: str) -> None:
    src_ckpt = tmp_path / "src_6l.pt"
    exp_10_ckpt = tmp_path / f"exp_10l_{strategy}.pt"
    exp_14_ckpt = tmp_path / f"exp_14l_{strategy}.pt"

    src_model, _ = _make_tiny_checkpoint(src_ckpt)
    src_model.eval()
    tokens = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        src_reflex, _, _, _ = src_model.forward_all_exits(tokens)

    res_10 = expand_checkpoint_file(
        input_path=src_ckpt,
        output_path=exp_10_ckpt,
        target_layers=10,
        limbic_exit_layer=6,
        strategy=strategy,
        smooth_scale=0.25,
    )
    assert res_10["source_layers"] == 6
    assert res_10["target_layers"] == 10
    assert len(res_10["layer_map"]) == 10
    assert res_10["layer_map"][:3] == [0, 1, 2]

    ckpt_10 = torch.load(exp_10_ckpt, map_location="cpu", weights_only=True)
    cfg_10 = ckpt_10["config"]
    assert ckpt_10["fingerprint"] == compute_checkpoint_fingerprint(cfg_10)

    model_10 = TriuneTransformer(
        vocab_size=cfg_10["vocab_size"],
        hidden_dim=cfg_10["hidden_dim"],
        num_layers=cfg_10["num_layers"],
        num_heads=cfg_10["num_heads"],
        head_dim=cfg_10["head_dim"],
        num_experts=cfg_10["num_experts"],
        top_k_experts=cfg_10["top_k_experts"],
        shared_expert=cfg_10["shared_expert"],
        shared_expert_scale=cfg_10["shared_expert_scale"],
        expert_hidden_multiplier=cfg_10["expert_hidden_multiplier"],
        rope_max_seq_len=cfg_10["rope_max_seq_len"],
        router_prefix_layers=cfg_10["router_prefix_layers"],
        reflex_exit_layer=cfg_10["reflex_exit_layer"],
        limbic_exit_layer=cfg_10["limbic_exit_layer"],
    )
    model_10.load_state_dict(ckpt_10["model_state"], strict=True)
    model_10.eval()
    with torch.no_grad():
        r10, l10, c10, route10 = model_10.forward_all_exits(tokens)

    for t in (r10, l10, c10, route10):
        assert torch.isfinite(t).all()
    assert torch.allclose(r10, src_reflex, atol=1e-5)

    # Recursive expansion 10L -> 14L
    res_14 = expand_checkpoint_file(
        input_path=exp_10_ckpt,
        output_path=exp_14_ckpt,
        target_layers=14,
        limbic_exit_layer=8,
        strategy=strategy,
        smooth_scale=0.25,
    )
    ckpt_14 = torch.load(exp_14_ckpt, map_location="cpu", weights_only=True)
    cfg_14 = ckpt_14["config"]
    assert res_14["target_layers"] == 14
    model_14 = TriuneTransformer(
        vocab_size=cfg_14["vocab_size"],
        hidden_dim=cfg_14["hidden_dim"],
        num_layers=cfg_14["num_layers"],
        num_heads=cfg_14["num_heads"],
        head_dim=cfg_14["head_dim"],
        num_experts=cfg_14["num_experts"],
        top_k_experts=cfg_14["top_k_experts"],
        shared_expert=cfg_14["shared_expert"],
        shared_expert_scale=cfg_14["shared_expert_scale"],
        expert_hidden_multiplier=cfg_14["expert_hidden_multiplier"],
        rope_max_seq_len=cfg_14["rope_max_seq_len"],
        router_prefix_layers=cfg_14["router_prefix_layers"],
        reflex_exit_layer=cfg_14["reflex_exit_layer"],
        limbic_exit_layer=cfg_14["limbic_exit_layer"],
    )
    model_14.load_state_dict(ckpt_14["model_state"], strict=True)
