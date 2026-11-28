"""Unit and regression tests for gap-closure hardening across CALM, C ABI streams, Grassmann transport, and downstream eval."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch

from lucid_machines.data.tokenizer import load_tokenizer
from lucid_machines.model.transformer import TriuneTransformer
from lucid_machines.ops._c_bridge import (
    NativeLucidMachinesEngine,
    _sync_pytorch_cuda_stream,
    get_current_cuda_stream_ptr,
)
from lucid_machines.optim.centroid import CentroidSteerOptimizer
from scripts.eval_downstream import evaluate_model_on_tasks


def _tiny_triune(vocab_size: int = 128) -> TriuneTransformer:
    torch.manual_seed(7)
    return TriuneTransformer(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=5,
        num_heads=4,
        head_dim=16,
        num_experts=4,
        top_k_experts=1,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
    )


def test_calm_online_entropy_guard_preserves_static_and_raises_tau_on_ood() -> None:
    model = _tiny_triune(vocab_size=128)
    model.eval()
    x = torch.randint(0, 128, (2, 16))

    # 1. Static CALM (beta=0.0) leaves threshold unchanged at 0.05 even under high entropy
    logits_static, _ = model(x, exit_policy="calm", confidence_threshold=0.05, calm_entropy_beta=0.0)
    stats_static = model.last_calm_compaction_stats
    assert stats_static["effective_threshold"] == 0.05
    assert logits_static.shape == (2, 16, 128)

    # 2. Online entropy guard (beta=0.5) against low calib entropy threshold forces eff_threshold strictly upward
    model.reset_calm_entropy()
    logits_guarded, _ = model(
        x,
        exit_policy="calm",
        confidence_threshold=0.05,
        calm_entropy_beta=0.5,
        calm_calib_entropy=0.10,
        calm_entropy_ema_decay=0.5,
    )
    stats_guarded = model.last_calm_compaction_stats
    assert stats_guarded["reflex_entropy_mean"] > 1.0
    assert stats_guarded["effective_threshold"] > 0.05
    assert logits_guarded.shape == (2, 16, 128)


def test_c_bridge_stream_helpers_and_sync() -> None:
    ptr = get_current_cuda_stream_ptr(0)
    assert isinstance(ptr, int)
    assert ptr >= 0
    _sync_pytorch_cuda_stream(0)

    # Instantiate native engine wrapper if binary present (safe no-op otherwise) and call step() without crash
    eng = NativeLucidMachinesEngine(vocab_size=128, hidden_dim=64, num_layers=4, num_heads=4, head_dim=16)
    res = eng.step(batch_size=2, seq_len=8)
    assert isinstance(res, dict)
    assert "ok" in res
    eng.close()


def test_centroidsteer_grassmann_drift_telemetry_and_trust_region_damping() -> None:
    torch.manual_seed(11)
    model = _tiny_triune(vocab_size=128)

    def _run_rotation(clip_val: float | None):
        torch.manual_seed(11)
        m = _tiny_triune(vocab_size=128)
        opt = CentroidSteerOptimizer(
            model=m,
            lr=1e-3,
            rank=8,
            update_gap=50,
            steer_scale=1.0,
            steer_update_gap=1,
            steer_mode="upgraded",
            num_steer_dirs=2,
            use_muon=True,
            transport_drift_clip=clip_val,
        )
        for step in range(4):
            # Shift input token distribution sharply on step 2 to drive subspace rotation
            lo = 0 if step < 2 else 64
            hi = 64 if step < 2 else 128
            bx = torch.randint(lo, hi, (3, 16))
            by = torch.randint(lo, hi, (3, 16)).reshape(-1)
            opt.zero_grad()
            logits, _ = m(bx, force_depth=2)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 128), by)
            loss.backward()
            opt.step()
        return opt.get_telemetry()["grassmann_transport"], m

    tel_noclip, m_noclip = _run_rotation(None)
    assert tel_noclip["events"] > 0
    assert tel_noclip["drift_max"] >= tel_noclip["drift_mean"] >= 0.0
    assert tel_noclip["damped_events"] == 0
    assert tel_noclip["drift_clip"] is None

    tel_clipped, m_clipped = _run_rotation(0.05)
    assert tel_clipped["events"] > 0
    assert tel_clipped["damped_events"] > 0
    assert tel_clipped["drift_clip"] == 0.05
    for p in m_clipped.parameters():
        assert torch.isfinite(p).all()


def test_eval_downstream_offline_harness(tmp_path: Path) -> None:
    tok = load_tokenizer(Path("triune_tokenizer.json"))
    vocab = tok.get_vocab_size()
    model = _tiny_triune(vocab_size=vocab)
    report = evaluate_model_on_tasks(
        model,
        tok,
        tasks=["synthetic_mc"],
        modes=["cortex", "limbic", "reflex", "calm", "calm_entropy_guard"],
        max_samples=4,
        calm_tau=0.20,
        calm_entropy_beta=0.25,
        calm_calib_entropy=1.0,
    )
    assert report["tasks"] == ["synthetic_mc"]
    for mode in ("cortex", "limbic", "reflex", "calm", "calm_entropy_guard"):
        m = report["modes"][mode]
        assert 0.0 <= m["macro_acc"] <= 1.0
        assert not math.isnan(m["theoretical_flop_savings_pct"])
    assert report["modes"]["calm_entropy_guard"]["mean_effective_tau"] >= report["modes"]["calm"]["mean_effective_tau"]
