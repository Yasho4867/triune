"""Feature #1 (kcx Checkpoint Auto-Resolution) & Feature #2 (kcx/lucid_machines Competitor Parity) E2E Tests."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn.functional as F

from .helpers import (
    build_tiny_checkpoint,
    free_port,
    run_kcx_cli,
    terminate_process,
    wait_http_ok,
)

# ---------------------------------------------------------------------------
# Feature #1 — kcx Checkpoint Auto-Resolution & Offline Eval Fallback (6 T1 + 5 T2)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_f1_infer_auto_resolves_real_session_active_checkpoint(
    e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F1-T1-01: `kcx infer` without --checkpoint resolves checkpoints/session_state.json active_checkpoint."""
    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "--prompt", "Hello", "--max-tokens", "4", "--device", "cpu", "--no-stream"],
        cwd=e2e_repo_root,
        timeout=120.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"kcx infer failed without --checkpoint (rc={proc.returncode}):\n{combined}"
    assert "FileNotFoundError" not in combined
    assert "triune_studio_step_36500_rescued.pt" in combined or len(proc.stdout.strip()) > 0


@pytest.mark.tier1
def test_t1_f1_infer_prefers_session_state_active_over_newer_pt(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F1-T1-02: session_state.json active_checkpoint takes precedence over a newer .pt by mtime."""
    older_ckpt = isolated_cli_workspace / "checkpoints" / "older_active.pt"
    newer_ckpt = isolated_cli_workspace / "checkpoints" / "newer_other.pt"
    build_tiny_checkpoint(older_ckpt, e2e_repo_root, step=11)
    build_tiny_checkpoint(newer_ckpt, e2e_repo_root, step=22)

    now = time.time()
    os.utime(older_ckpt, (now - 300, now - 300))
    os.utime(newer_ckpt, (now, now))

    (isolated_cli_workspace / "checkpoints" / "session_state.json").write_text(
        json.dumps({"active_checkpoint": "older_active.pt", "step": 11}),
        encoding="utf-8",
    )

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "--prompt", "Hi", "--max-tokens", "3", "--device", "cpu", "--no-stream"],
        cwd=isolated_cli_workspace,
        timeout=90.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"kcx infer failed (rc={proc.returncode}):\n{combined}"
    assert "older_active.pt" in combined, f"Expected active_checkpoint older_active.pt, got:\n{combined}"


@pytest.mark.tier1
def test_t1_f1_newest_pt_fallback_across_checkpoints_and_checkpoints_full(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F1-T1-03: Without session_state.json, auto-resolution picks newest .pt across checkpoints/ and checkpoints_full/."""
    old_ckpt = isolated_cli_workspace / "checkpoints" / "old_ckpt.pt"
    newest_ckpt = isolated_cli_workspace / "checkpoints_full" / "newest_full.pt"
    build_tiny_checkpoint(old_ckpt, e2e_repo_root, step=5)
    build_tiny_checkpoint(newest_ckpt, e2e_repo_root, step=50)

    now = time.time()
    os.utime(old_ckpt, (now - 600, now - 600))
    os.utime(newest_ckpt, (now, now))

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "--prompt", "Hi", "--max-tokens", "3", "--device", "cpu", "--no-stream"],
        cwd=isolated_cli_workspace,
        timeout=90.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"kcx infer fallback failed (rc={proc.returncode}):\n{combined}"
    assert "newest_full.pt" in combined, f"Expected newest_full.pt fallback, got:\n{combined}"


@pytest.mark.tier1
def test_t1_f1_eval_auto_resolves_checkpoint_and_offline_sample_dataset(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F1-T1-04: `kcx eval --max-steps 1` auto-resolves checkpoint and falls back offline to data/fineweb_sample.jsonl."""
    ckpt = isolated_cli_workspace / "checkpoints" / "eval_active.pt"
    build_tiny_checkpoint(ckpt, e2e_repo_root, step=15)
    (isolated_cli_workspace / "checkpoints" / "session_state.json").write_text(
        json.dumps({"active_checkpoint": "eval_active.pt", "step": 15}),
        encoding="utf-8",
    )

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["eval", "--max-steps", "1", "--device", "cpu"],
        cwd=isolated_cli_workspace,
        env_overrides={"HF_HUB_OFFLINE": "1"},
        timeout=60.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"kcx eval offline fallback failed (rc={proc.returncode}):\n{combined}"
    assert "EVALUATION REPORT" in combined or "Perplexity" in combined, combined


@pytest.mark.tier1
def test_t1_f1_chat_benchmark_profile_default_resolution(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str, tmp_path: Path
) -> None:
    """F1-T1-05: `kcx benchmark`, `kcx profile`, and `kcx chat` auto-resolve active checkpoint cleanly."""
    ckpt = isolated_cli_workspace / "checkpoints" / "suite_active.pt"
    build_tiny_checkpoint(ckpt, e2e_repo_root, step=20)
    (isolated_cli_workspace / "checkpoints" / "session_state.json").write_text(
        json.dumps({"active_checkpoint": "suite_active.pt", "step": 20}),
        encoding="utf-8",
    )

    bench_proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["benchmark", "--max-tokens", "3", "--device", "cpu"],
        cwd=isolated_cli_workspace,
        timeout=90.0,
    )
    assert bench_proc.returncode == 0, f"kcx benchmark failed:\n{bench_proc.stdout}\n{bench_proc.stderr}"
    assert "suite_active.pt" in (bench_proc.stdout + bench_proc.stderr)

    prof_out = tmp_path / "prof_out.txt"
    prof_proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        [
            "profile",
            "--output",
            str(prof_out),
            "--batch-size",
            "1",
            "--seq-len",
            "16",
            "--warmup-steps",
            "1",
            "--active-steps",
            "1",
        ],
        cwd=isolated_cli_workspace,
        timeout=90.0,
    )
    assert prof_proc.returncode == 0, f"kcx profile failed:\n{prof_proc.stdout}\n{prof_proc.stderr}"
    assert prof_out.is_file() and prof_out.stat().st_size > 0

    chat_proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["chat", "--max-tokens", "3", "--device", "cpu"],
        cwd=isolated_cli_workspace,
        input_text="/exit\n",
        timeout=90.0,
    )
    chat_out = chat_proc.stdout + "\n" + chat_proc.stderr
    assert chat_proc.returncode == 0, f"kcx chat failed:\n{chat_out}"
    assert "Goodbye." in chat_out or "Session ended." in chat_out


@pytest.mark.tier1
def test_t1_f1_scripts_chat_and_profile_model_auto_resolve(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str, tmp_path: Path
) -> None:
    """F1-T1-06: Standalone scripts/chat.py and scripts/profile_model.py auto-resolve active checkpoint."""
    ckpt = isolated_cli_workspace / "checkpoints" / "script_active.pt"
    build_tiny_checkpoint(ckpt, e2e_repo_root, step=25)
    (isolated_cli_workspace / "checkpoints" / "session_state.json").write_text(
        json.dumps({"active_checkpoint": "script_active.pt", "step": 25}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    env["PYTHONIOENCODING"] = "utf-8"

    chat_res = subprocess.run(
        [e2e_python_exe, str(e2e_repo_root / "scripts" / "chat.py"), "--device", "cpu"],
        cwd=str(isolated_cli_workspace),
        env=env,
        input="/exit\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90.0,
    )
    assert chat_res.returncode == 0, f"scripts/chat.py failed:\n{chat_res.stdout}\n{chat_res.stderr}"

    prof_file = tmp_path / "script_prof.txt"
    prof_res = subprocess.run(
        [
            e2e_python_exe,
            str(e2e_repo_root / "scripts" / "profile_model.py"),
            "--output",
            str(prof_file),
            "--batch_size",
            "1",
            "--seq_len",
            "16",
            "--warmup_steps",
            "1",
            "--active_steps",
            "1",
        ],
        cwd=str(isolated_cli_workspace),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90.0,
    )
    assert prof_res.returncode == 0, f"scripts/profile_model.py failed:\n{prof_res.stdout}\n{prof_res.stderr}"
    assert prof_file.is_file() and prof_file.stat().st_size > 0


@pytest.mark.tier2
def test_t2_f1_no_checkpoints_anywhere_clean_notice_no_traceback(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str, tmp_path: Path
) -> None:
    """F1-T2-01: When checkpoints/ and checkpoints_full/ are empty, commands exit non-zero with clean notice."""
    cmds = [
        (["infer", "-p", "Hi", "--device", "cpu"], None),
        (["eval", "--max-steps", "1", "--device", "cpu"], None),
        (["benchmark", "--device", "cpu"], None),
        (["profile", "--output", str(tmp_path / "none.txt")], None),
        (["chat", "--device", "cpu"], "/exit\n"),
    ]
    for args, stdin_txt in cmds:
        proc = run_kcx_cli(
            e2e_python_exe,
            e2e_repo_root,
            args,
            cwd=isolated_cli_workspace,
            input_text=stdin_txt,
            timeout=60.0,
        )
        combined = proc.stdout + "\n" + proc.stderr
        assert proc.returncode != 0, f"{args} unexpectedly exited 0 with no checkpoints:\n{combined}"
        assert "Traceback (most recent call last)" not in combined, (
            f"{args} leaked raw Python traceback:\n{combined}"
        )
        assert any(
            k in combined.lower() for k in ("checkpoint", "not found", "no checkpoint", "missing")
        ), f"{args} missing user-friendly notice:\n{combined}"


@pytest.mark.tier2
def test_t2_f1_explicit_nonexistent_checkpoint_clean_error_no_fallback(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F1-T2-02: Explicit --checkpoint pointing to missing file must fail cleanly without falling back."""
    valid = isolated_cli_workspace / "checkpoints" / "valid.pt"
    build_tiny_checkpoint(valid, e2e_repo_root, step=1)
    (isolated_cli_workspace / "checkpoints" / "session_state.json").write_text(
        json.dumps({"active_checkpoint": "valid.pt", "step": 1}),
        encoding="utf-8",
    )

    for args in (
        ["infer", "--checkpoint", "does_not_exist_123.pt", "-p", "Hi", "--device", "cpu"],
        ["eval", "--checkpoint", "does_not_exist_123.pt", "--max-steps", "1", "--device", "cpu"],
    ):
        proc = run_kcx_cli(e2e_python_exe, e2e_repo_root, args, cwd=isolated_cli_workspace, timeout=60.0)
        combined = proc.stdout + "\n" + proc.stderr
        assert proc.returncode != 0, f"{args} silently fell back when explicit checkpoint was missing:\n{combined}"
        assert "Traceback (most recent call last)" not in combined, f"Raw traceback leaked:\n{combined}"
        assert "does_not_exist_123.pt" in combined or "not found" in combined.lower()


@pytest.mark.tier2
def test_t2_f1_corrupt_or_stale_session_state_falls_back_to_newest_pt(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F1-T2-03: Corrupt JSON or stale active_checkpoint in session_state.json recovers via newest .pt."""
    valid = isolated_cli_workspace / "checkpoints" / "valid_recovery.pt"
    build_tiny_checkpoint(valid, e2e_repo_root, step=7)
    state_file = isolated_cli_workspace / "checkpoints" / "session_state.json"

    for payload in ("{corrupt_json_bytes", json.dumps({"active_checkpoint": "deleted_ckpt.pt", "step": 99})):
        state_file.write_text(payload, encoding="utf-8")
        proc = run_kcx_cli(
            e2e_python_exe,
            e2e_repo_root,
            ["infer", "-p", "Hi", "--max-tokens", "2", "--device", "cpu", "--no-stream"],
            cwd=isolated_cli_workspace,
            timeout=90.0,
        )
        combined = proc.stdout + "\n" + proc.stderr
        assert proc.returncode == 0, f"Recovery failed for state={payload!r}:\n{combined}"
        assert "valid_recovery.pt" in combined, f"Did not select valid_recovery.pt:\n{combined}"


@pytest.mark.tier2
def test_t2_f1_corrupt_checkpoint_file_clean_error(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F1-T2-04: Corrupt .pt archive produces clean non-zero error without unhandled traceback."""
    broken = isolated_cli_workspace / "checkpoints" / "broken.pt"
    broken.write_bytes(b"not a valid pytorch archive at all")

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "--checkpoint", str(broken), "-p", "Hi", "--device", "cpu"],
        cwd=isolated_cli_workspace,
        timeout=60.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode != 0
    assert "Traceback (most recent call last)" not in combined, f"Raw traceback leaked:\n{combined}"


@pytest.mark.tier2
def test_t2_f1_eval_missing_explicit_dataset_clean_error(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F1-T2-05: Explicit nonexistent --dataset on `kcx eval` must fail cleanly rather than silent sample fallback."""
    valid = isolated_cli_workspace / "checkpoints" / "valid_eval.pt"
    build_tiny_checkpoint(valid, e2e_repo_root, step=1)

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        [
            "eval",
            "--checkpoint",
            str(valid),
            "--dataset",
            "missing_local_file_xyz_999.jsonl",
            "--max-steps",
            "1",
            "--device",
            "cpu",
        ],
        cwd=isolated_cli_workspace,
        env_overrides={"HF_HUB_OFFLINE": "1"},
        timeout=60.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode != 0, f"kcx eval silently ignored missing explicit --dataset:\n{combined}"
    assert "Traceback (most recent call last)" not in combined, f"Raw traceback leaked:\n{combined}"


# ---------------------------------------------------------------------------
# Feature #2 — kcx & lucid_machines Competitor Parity (6 T1 + 6 T2)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_f2_polar_express_4step_muon_orthogonality_and_optimizer_step() -> None:
    """F2-T1-01: Polar Express 4-step orthogonalization and Muon(ns_steps=4) step produce semi-orthogonal updates."""
    from lucid_machines.kernels import fast_muon
    from lucid_machines.kernels.fast_muon import (
        fast_zeropower_via_newtonschulz5,
        fast_zeropower_via_newtonschulz5_batched,
    )
    from lucid_machines.optim.muon import Muon

    # Verify Polar Express 4-step schedule exists / is active in fast_muon
    has_polar_express = any(
        "polar" in name.lower() for name in dir(fast_muon)
    )
    assert has_polar_express, "Expected Polar Express 4-step schedule/implementation in lucid_machines.kernels.fast_muon"

    torch.manual_seed(7)
    for shape in ((64, 64), (96, 48), (48, 96)):
        g = torch.randn(*shape, dtype=torch.float32)
        g_orig = g.clone()
        ortho = fast_zeropower_via_newtonschulz5(g, steps=4)
        assert torch.equal(g, g_orig), "Input gradient tensor was mutated in-place"
        assert ortho.shape == g.shape
        assert torch.isfinite(ortho).all()
        gram = ortho @ ortho.T if shape[0] <= shape[1] else ortho.T @ ortho
        eye = torch.eye(gram.shape[0], dtype=gram.dtype)
        err = (gram.float() - eye).abs().mean().item()
        assert err < 0.25, f"Poor 4-step semi-orthogonality for shape {shape}: mean abs error={err:.4f}"

    batched = torch.randn(4, 48, 64, dtype=torch.float32)
    b_out = fast_zeropower_via_newtonschulz5_batched(batched, steps=4)
    assert b_out.shape == batched.shape and torch.isfinite(b_out).all()

    param = torch.nn.Parameter(torch.randn(32, 48))
    before = param.detach().clone()
    opt = Muon([param], lr=1e-2, ns_steps=4)
    param.grad = torch.randn_like(param)
    opt.step()
    assert not torch.equal(param.detach(), before)
    assert torch.isfinite(param).all()


@pytest.mark.tier1
def test_t1_f2_training_engine_chunked_hidden_ce_parity_and_step(e2e_repo_root: Path) -> None:
    """F2-T1-02: TrainingEngine integrates chunked_cross_entropy_from_hidden with exact F.cross_entropy parity."""
    import inspect
    from lucid_machines.kernels.fast_cross_entropy import chunked_cross_entropy_from_hidden
    from lucid_machines.model import TriuneTransformer
    from lucid_machines.trainer import engine as engine_mod
    from lucid_machines.trainer.engine import TrainingEngine

    engine_src = inspect.getsource(engine_mod)
    assert "chunked_cross_entropy_from_hidden" in engine_src, (
        "TrainingEngine must wire chunked_cross_entropy_from_hidden to avoid materializing full [B,T,V] logits"
    )

    # Verify chunked_cross_entropy_from_hidden numerical parity vs reference F.cross_entropy
    torch.manual_seed(42)
    hidden = torch.randn(32, 24, requires_grad=True)
    weight = torch.randn(128, 24, requires_grad=True)
    targets = torch.randint(0, 128, (32,))
    targets[0] = 0

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    ref_loss = F.cross_entropy(h_ref @ w_ref.T, targets, ignore_index=0)
    ref_loss.backward()

    chunk_loss = chunked_cross_entropy_from_hidden(hidden, weight, targets, ignore_index=0, chunk_size=8)
    chunk_loss.backward()
    assert torch.allclose(chunk_loss.detach(), ref_loss.detach(), atol=1e-4)
    assert torch.allclose(hidden.grad, h_ref.grad, atol=1e-4)
    assert torch.allclose(weight.grad, w_ref.grad, atol=1e-4)

    # Real TriuneTransformer step across cortex/joint/dynamic depth modes
    for mode in ("cortex", "joint", "dynamic"):
        model = TriuneTransformer(
            vocab_size=128,
            hidden_dim=32,
            num_layers=5,
            num_heads=2,
            head_dim=16,
            num_experts=2,
            top_k_experts=1,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
        )

        class _TrainerAdapter:
            def __init__(self, m: TriuneTransformer, dmode: str) -> None:
                self.model = m
                self.device = torch.device("cpu")
                self.vocab_size = 128
                self.pad_token_id = 0
                self.grad_accum = 1
                self.z_loss_coef = 1e-3
                self.optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
                self.config = {
                    "target_depth_dist": [0.5, 0.3, 0.2],
                    "exploration": "none",
                    "exploration_steps": 0,
                    "usage_ema_decay": 0.99,
                    "depth_mode": dmode,
                    "balance_coef": 0.3,
                    "grad_accum_steps": 1,
                    "grad_clip": 1.0,
                    "log_every": 100,
                    "total_steps": 5,
                    "lr": 1e-3,
                }
                self._batch = torch.randint(1, 128, (2, 17))

            def next_batch(self):
                return self._batch[:, :-1], self._batch[:, 1:]

            def model_autocast(self):
                return nullcontext()

            @staticmethod
            def lr_schedule(cfg, step):
                return cfg["lr"]

            @staticmethod
            def set_optimizer_lr(opt, lr, expert_lr=None):
                for g in opt.param_groups:
                    g["lr"] = lr

        eng = TrainingEngine(_TrainerAdapter(model, mode))
        eng.step = 1
        metrics = eng.step_once()
        assert torch.isfinite(torch.tensor(metrics["loss"])), f"Non-finite loss in mode {mode}: {metrics}"


@pytest.mark.tier1
def test_t1_f2_benchmark_suite_mqar_runs_cleanly(e2e_repo_root: Path, e2e_python_exe: str) -> None:
    """F2-T1-03: `kcx benchmark --help` exposes `mqar` and `kcx benchmark --suite mqar` runs cleanly."""
    help_proc = run_kcx_cli(e2e_python_exe, e2e_repo_root, ["benchmark", "--help"], timeout=30.0)
    assert help_proc.returncode == 0
    assert "mqar" in help_proc.stdout.lower(), f"`mqar` missing from `kcx benchmark --help`:\n{help_proc.stdout}"

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["benchmark", "--suite", "mqar"],
        cwd=e2e_repo_root,
        timeout=120.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"kcx benchmark --suite mqar failed:\n{combined}"
    assert any(k in combined.lower() for k in ("mqar", "recall", "gla", "deltanet")), combined


@pytest.mark.tier1
def test_t1_f2_universal_json_output_across_cli_commands(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F2-T1-04: `--json` emits valid machine-readable JSON on scan, models, doctor, gpucheck, checkpoint list, eval, infer."""
    ckpt = isolated_cli_workspace / "checkpoints" / "json_test.pt"
    build_tiny_checkpoint(ckpt, e2e_repo_root, step=10)
    (isolated_cli_workspace / "checkpoints" / "session_state.json").write_text(
        json.dumps({"active_checkpoint": "json_test.pt", "step": 10}),
        encoding="utf-8",
    )

    commands: List[List[str]] = [
        ["models", "--json"],
        ["doctor", "--json"],
        ["gpucheck", "--json"],
        ["checkpoint", "list", "--json"],
        ["scan", "--no-wsl", "--json"],
        ["infer", "--checkpoint", str(ckpt), "-p", "Hi", "--max-tokens", "3", "--device", "cpu", "--no-stream", "--json"],
        [
            "eval",
            "--checkpoint",
            str(ckpt),
            "--dataset",
            str(isolated_cli_workspace / "data" / "fineweb_sample.jsonl"),
            "--max-steps",
            "1",
            "--device",
            "cpu",
            "--json",
        ],
    ]
    for args in commands:
        proc = run_kcx_cli(
            e2e_python_exe,
            e2e_repo_root,
            args,
            cwd=isolated_cli_workspace,
            env_overrides={"HF_HUB_OFFLINE": "1"},
            timeout=90.0,
        )
        assert proc.returncode == 0, f"{args} failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        parsed = json.loads(proc.stdout.strip())
        assert isinstance(parsed, (dict, list)), f"{args} JSON output not dict/list: {type(parsed)}"


@pytest.mark.tier1
def test_t1_f2_infer_stdin_system_and_deterministic_seed(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F2-T1-05: `kcx infer` accepts stdin prompt, --system, and --seed for deterministic generation."""
    ckpt = isolated_cli_workspace / "checkpoints" / "seed_test.pt"
    build_tiny_checkpoint(ckpt, e2e_repo_root, step=8)

    outputs: List[str] = []
    for _ in range(2):
        proc = run_kcx_cli(
            e2e_python_exe,
            e2e_repo_root,
            [
                "infer",
                "--checkpoint",
                str(ckpt),
                "--system",
                "Be concise.",
                "--seed",
                "1234",
                "--max-tokens",
                "6",
                "--device",
                "cpu",
                "--no-stream",
                "--json",
            ],
            cwd=isolated_cli_workspace,
            input_text="Hello from stdin\n",
            timeout=90.0,
        )
        assert proc.returncode == 0, f"kcx infer stdin/seed failed:\n{proc.stdout}\n{proc.stderr}"
        data = json.loads(proc.stdout.strip())
        text_val = data.get("text") or data.get("output") or data.get("generated_text") or json.dumps(data)
        outputs.append(str(text_val))

    assert outputs[0] == outputs[1], f"Deterministic --seed 1234 produced different outputs: {outputs}"


@pytest.mark.tier1
def test_t1_f2_finetune_recipe_yaml_json_and_agent_parity_flags(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str, tmp_path: Path
) -> None:
    """F2-T1-06: `kcx finetune --recipe` (YAML/JSON + W&B flag), `kcx agent` parity flags, and `kcx serve --model`."""
    ckpt = isolated_cli_workspace / "checkpoints" / "ft_base.pt"
    build_tiny_checkpoint(ckpt, e2e_repo_root, step=10)

    # 1. Finetune --recipe (YAML) & --no-wandb / --wandb parser check
    ft_help = run_kcx_cli(e2e_python_exe, e2e_repo_root, ["finetune", "--help"], timeout=30.0)
    assert ft_help.returncode == 0
    assert "--recipe" in ft_help.stdout
    assert "--wandb" in ft_help.stdout

    adapter_out = tmp_path / "adapter_out"
    recipe_yaml = tmp_path / "recipe.yaml"
    recipe_yaml.write_text(
        "\n".join(
            [
                f"checkpoint: {ckpt.as_posix()}",
                f"dataset: {(isolated_cli_workspace / 'data' / 'fineweb_sample.jsonl').as_posix()}",
                "rank: 4",
                "max_steps: 1",
                "batch_size: 1",
                "seq_len: 16",
                "device: cpu",
                f"output_dir: {adapter_out.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    ft_proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["finetune", "--recipe", str(recipe_yaml)],
        cwd=isolated_cli_workspace,
        env_overrides={"HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"},
        timeout=120.0,
    )
    assert ft_proc.returncode == 0, f"kcx finetune --recipe failed:\n{ft_proc.stdout}\n{ft_proc.stderr}"
    assert adapter_out.is_dir() and any(adapter_out.iterdir()), "Finetune did not write adapter output"

    # 2. Agent --output-format json / stream-json, stdin goal, --allowed-tools, --resume
    store_dir = tmp_path / "agent_store"
    agent_ws = tmp_path / "agent_ws"
    agent_ws.mkdir(parents=True, exist_ok=True)
    (agent_ws / "sample.txt").write_text("triune sample", encoding="utf-8")

    ag_proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        [
            "agent",
            "--workspace",
            str(agent_ws),
            "--store-dir",
            str(store_dir),
            "--output-format",
            "json",
            "--allowed-tools",
            "read_file,list_dir",
        ],
        cwd=isolated_cli_workspace,
        input_text="list workspace files\n",
        timeout=90.0,
    )
    assert ag_proc.returncode == 0, f"kcx agent --output-format json failed:\n{ag_proc.stdout}\n{ag_proc.stderr}"
    ag_json = json.loads(ag_proc.stdout.strip())
    assert isinstance(ag_json, dict)
    session_id = ag_json.get("session_id") or ag_json.get("id")
    assert session_id, f"Expected session_id in agent JSON output: {ag_json}"

    ag_resume = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        [
            "agent",
            "--workspace",
            str(agent_ws),
            "--store-dir",
            str(store_dir),
            "--resume",
            str(session_id),
            "--output-format",
            "stream-json",
            "summarize sample.txt",
        ],
        cwd=isolated_cli_workspace,
        timeout=90.0,
    )
    assert ag_resume.returncode == 0, f"kcx agent --resume failed:\n{ag_resume.stdout}\n{ag_resume.stderr}"
    ndjson_lines = [ln.strip() for ln in ag_resume.stdout.splitlines() if ln.strip()]
    assert len(ndjson_lines) >= 1 and all(isinstance(json.loads(ln), dict) for ln in ndjson_lines)

    # 3. Serve --model preload check
    serve_help = run_kcx_cli(e2e_python_exe, e2e_repo_root, ["serve", "--help"], timeout=30.0)
    assert serve_help.returncode == 0 and "--model" in serve_help.stdout

    port = free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    env["LUCID_MACHINES_TESTING"] = "1"
    srv_proc = subprocess.Popen(
        [
            e2e_python_exe,
            "-m",
            "lucid_machines",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            "triune-nano",
        ],
        cwd=str(e2e_repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        health = wait_http_ok(f"http://127.0.0.1:{port}/healthz", timeout=45.0)
        assert isinstance(health, dict)
    finally:
        terminate_process(srv_proc)


@pytest.mark.tier2
def test_t2_f2_infer_empty_stdin_without_prompt_exits_2(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F2-T2-01: Empty stdin with no prompt argument exits cleanly with code 2 without hanging."""
    ckpt = isolated_cli_workspace / "checkpoints" / "empty_stdin.pt"
    build_tiny_checkpoint(ckpt, e2e_repo_root, step=1)

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["infer", "--checkpoint", str(ckpt), "--device", "cpu"],
        cwd=isolated_cli_workspace,
        input_text="",
        timeout=30.0,
    )
    assert proc.returncode == 2, f"Expected exit code 2 on empty stdin + no prompt, got {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.tier2
def test_t2_f2_polar_express_invalid_inputs_raise_cleanly() -> None:
    """F2-T2-02: Invalid tensor ranks raise ValueError; zero tensor avoids NaN and mutation."""
    from lucid_machines.kernels.fast_muon import fast_zeropower_via_newtonschulz5

    with pytest.raises(ValueError):
        fast_zeropower_via_newtonschulz5(torch.randn(16), steps=4)
    with pytest.raises(ValueError):
        fast_zeropower_via_newtonschulz5(torch.randn(2, 2, 4, 4), steps=4)

    zeros = torch.zeros(16, 16)
    z_out = fast_zeropower_via_newtonschulz5(zeros, steps=4)
    assert torch.isfinite(z_out).all(), f"Zero matrix produced NaN/Inf in 4-step NS: {z_out}"


@pytest.mark.tier2
def test_t2_f2_finetune_invalid_or_missing_recipe_clean_error(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str, tmp_path: Path
) -> None:
    """F2-T2-03: Missing or malformed --recipe exits non-zero cleanly without raw traceback."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":::bad_yaml[\n  broken: [", encoding="utf-8")

    for recipe_path in (tmp_path / "missing_recipe.yaml", bad_yaml):
        proc = run_kcx_cli(
            e2e_python_exe,
            e2e_repo_root,
            ["finetune", "--recipe", str(recipe_path)],
            cwd=isolated_cli_workspace,
            env_overrides={"WANDB_MODE": "disabled"},
            timeout=45.0,
        )
        combined = proc.stdout + "\n" + proc.stderr
        assert proc.returncode != 0
        assert "Traceback (most recent call last)" not in combined, f"Raw traceback leaked:\n{combined}"


@pytest.mark.tier2
def test_t2_f2_agent_disallowed_tool_and_invalid_resume_id(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str, tmp_path: Path
) -> None:
    """F2-T2-04: --allowed-tools restricts disallowed tools; invalid --resume exits cleanly non-zero."""
    store_dir = tmp_path / "agent_store"
    agent_ws = tmp_path / "agent_ws"
    agent_ws.mkdir(parents=True, exist_ok=True)

    proc_missing_resume = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        [
            "agent",
            "--workspace",
            str(agent_ws),
            "--store-dir",
            str(store_dir),
            "--resume",
            "nonexistent_session_999",
            "--output-format",
            "json",
            "status",
        ],
        cwd=isolated_cli_workspace,
        timeout=45.0,
    )
    combined = proc_missing_resume.stdout + "\n" + proc_missing_resume.stderr
    assert proc_missing_resume.returncode != 0
    assert "Traceback (most recent call last)" not in combined

    proc_tools = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        [
            "agent",
            "--workspace",
            str(agent_ws),
            "--store-dir",
            str(store_dir),
            "--allowed-tools",
            "read_file",
            "--output-format",
            "json",
            "write a file named forbidden.txt and run a shell command",
        ],
        cwd=isolated_cli_workspace,
        timeout=60.0,
    )
    assert not (agent_ws / "forbidden.txt").exists(), "Disallowed write tool executed despite --allowed-tools read_file"


@pytest.mark.tier2
def test_t2_f2_json_mode_errors_emit_clean_failure(
    isolated_cli_workspace: Path, e2e_repo_root: Path, e2e_python_exe: str, tmp_path: Path
) -> None:
    """F2-T2-05: Missing checkpoint under --json emits non-zero exit and clean JSON/notice without traceback."""
    missing = tmp_path / "missing_ckpt.pt"
    for args in (
        ["infer", "--checkpoint", str(missing), "-p", "Hi", "--json"],
        ["eval", "--checkpoint", str(missing), "--json"],
    ):
        proc = run_kcx_cli(e2e_python_exe, e2e_repo_root, args, cwd=isolated_cli_workspace, timeout=45.0)
        combined = proc.stdout + "\n" + proc.stderr
        assert proc.returncode != 0
        assert "Traceback (most recent call last)" not in combined


@pytest.mark.tier2
def test_t2_f2_serve_invalid_preload_model_fails_cleanly(
    e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F2-T2-06: `kcx serve --model nonexistent_model_xyz_999` fails cleanly non-zero without hanging."""
    port = free_port()
    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["serve", "--host", "127.0.0.1", "--port", str(port), "--model", "nonexistent_model_xyz_999"],
        cwd=e2e_repo_root,
        timeout=30.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode != 0, f"kcx serve unexpectedly succeeded with invalid --model:\n{combined}"
    assert "Traceback (most recent call last)" not in combined


@pytest.mark.tier1
def test_t1_f2_plan_memory_junkie_3b_realistic_l4_budget(
    e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F2-T1-07: MemoryPlanner & `kcx plan-memory` on junkie-3b accurately account for weights, grads & rank-64 opt state."""
    from lucid_machines.runtime import MemoryPlanner

    junkie_3b_cfg = json.loads((e2e_repo_root / "models" / "configs" / "junkie-3b.json").read_text(encoding="utf-8"))
    est_64 = MemoryPlanner.estimate_vram(
        {
            **junkie_3b_cfg,
            "galore_rank": 64,
            "steer_mode": "upgraded",
            "use_muon": True,
        },
        target_vram_gb=24.0,
        available_vram_gb=24.0,
    )
    # Real 2.99B BF16 weights (~5.57 GB), BF16 grads (~5.43-5.57 GB, NOT 0.5x = 2.78 GB),
    # and realistic CentroidSteer rank-64 optimizer buffers (NOT 0.20 B/p = 1.66 GB).
    assert est_64.param_memory_gb >= 5.5, f"Unrealistic BF16 weights for junkie-3b: {est_64}"
    assert est_64.gradient_memory_gb >= 5.0, (
        f"Unrealistic BF16 gradient memory for junkie-3b (must reflect ~1.95-2.0 B/p): {est_64}"
    )
    assert est_64.optimizer_memory_gb > 3.5, (
        f"Unrealistic rank-64 CentroidSteer optimizer state ({est_64.optimizer_memory_gb} GB): {est_64}"
    )
    assert est_64.total_vram_gb > 15.0, f"Unrealistic total VRAM estimate ({est_64.total_vram_gb} GB): {est_64}"

    # Lowering galore_rank (16..32) must measurably reduce optimizer & total VRAM footprint.
    est_16 = MemoryPlanner.estimate_vram(
        {**junkie_3b_cfg, "galore_rank": 16, "steer_mode": "upgraded", "use_muon": True},
        target_vram_gb=24.0,
        available_vram_gb=24.0,
    )
    assert est_16.optimizer_memory_gb < est_64.optimizer_memory_gb, (
        f"Lower galore_rank=16 ({est_16.optimizer_memory_gb} GB) did not reduce optimizer state vs 64 ({est_64.optimizer_memory_gb} GB)"
    )

    plan_24 = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=24.0)
    assert plan_24["bf16_weights_gb"] >= 5.5
    assert plan_24["centroid_steer_static_vram_gb"] > 15.0, (
        f"junkie-3b static VRAM estimate unrealistically low: {plan_24}"
    )
    train_mode = str(plan_24.get("recommended_train_mode", ""))
    assert train_mode != "resident_bf16_train", (
        f"Must not falsely report resident_bf16_train fits rank-64 junkie-3b on 24GB L4: {plan_24}"
    )
    assert "streaming" in train_mode.lower() or "rank" in train_mode.lower(), (
        f"Expected 24GB junkie-3b training verdict to require --streaming or reduced rank, got {train_mode!r}"
    )

    proc = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["plan-memory", "--vram-gb", "24", "--model", "junkie-3b"],
        cwd=e2e_repo_root,
        timeout=45.0,
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"kcx plan-memory failed:\n{combined}"
    assert "2,990,355,155" in combined
    assert "10.545 GB" not in combined and "10.55 GB" not in combined and "opt=1.659 GB" not in combined, (
        f"Stale unrealistic 10.55 GB / 1.659 GB estimate reported by kcx plan-memory:\n{combined}"
    )
    assert "resident_bf16_train" not in combined, (
        f"kcx plan-memory falsely recommended resident_bf16_train for 24GB L4:\n{combined}"
    )
    assert "--streaming" in combined or "rank" in combined.lower(), (
        f"kcx plan-memory 24GB output should surface --streaming or reduced rank guidance:\n{combined}"
    )


@pytest.mark.tier1
def test_t1_f2_colab_runner_notebook_hardened_presets_and_syntax(e2e_repo_root: Path) -> None:
    """F2-T1-08: cloud/lucid_machines_colab_runner.ipynb is valid nbformat, sets expandable_segments, and has safe A100/G4/L4 presets."""
    import ast

    nb_path = e2e_repo_root / "cloud" / "lucid_machines_colab_runner.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4, f"Unexpected nbformat: {nb.get('nbformat')}"
    cells = nb.get("cells")
    assert isinstance(cells, list) and len(cells) >= 6

    all_sources: List[str] = []
    for idx, cell in enumerate(cells):
        src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        all_sources.append(src)
        if cell.get("cell_type") != "code":
            continue
        py_lines: List[str] = []
        in_shell_cont = False
        for line in src.splitlines():
            stripped = line.strip()
            if in_shell_cont:
                in_shell_cont = stripped.endswith("\\")
                continue
            if stripped.startswith("!") or stripped.startswith("%"):
                in_shell_cont = stripped.endswith("\\")
                continue
            py_lines.append(line)
        ast.parse("\n".join(py_lines), filename=f"{nb_path.name}:cell[{idx}]")

    full_nb_text = "\n".join(all_sources)
    assert "PYTORCH_CUDA_ALLOC_CONF" in full_nb_text, "Missing PYTORCH_CUDA_ALLOC_CONF in Colab notebook"
    assert "expandable_segments:True" in full_nb_text, "Missing expandable_segments:True in Colab notebook"
    assert "10.55 GB" not in full_nb_text and "10.10 GiB" not in full_nb_text and "12.89 GiB" not in full_nb_text, (
        "Colab notebook still contains false 10.10 GiB / 12.89 GiB / 10.55 GB resident junkie-3b claims"
    )
    assert "<20 GiB total resident on L4 24GB" not in full_nb_text, (
        "Colab notebook still falsely claims junkie-3b fits <20 GiB resident on L4 24GB"
    )

    lower_nb = full_nb_text.lower()
    for gpu_tier in ("a100", "g4", "l4"):
        assert gpu_tier in lower_nb, f"Missing {gpu_tier.upper()} preset/guidance in Colab notebook"

    # L4 24GB junkie-3b cell must NOT run resident --rank 64 without --streaming
    junkie_3b_cells = [s for s in all_sources if "junkie-3b" in s and "scripts/train.py" in s and "--dry-run" not in s]
    assert len(junkie_3b_cells) >= 1, "Missing junkie-3b training cell in Colab notebook"
    for cell_src in junkie_3b_cells:
        has_streaming = "--streaming" in cell_src or "--stream" in cell_src
        has_reduced_rank = bool(re.search(r"--rank\s+(?:16|24|32)\b", cell_src))
        is_a100_only = "a100" in cell_src.lower() and "l4" not in cell_src.lower()
        if not is_a100_only:
            assert has_streaming or has_reduced_rank, (
                f"Colab junkie-3b L4 preset must enable --streaming or reduced rank (16..32):\n{cell_src}"
            )


@pytest.mark.tier2
def test_t2_f2_plan_memory_vram_boundaries_and_rank_monotonicity(e2e_repo_root: Path) -> None:
    """F2-T2-07: MemoryPlanner enforces 8GB/24GB/80GB VRAM verdicts and strict galore_rank monotonicity."""
    from lucid_machines.runtime import MemoryPlanner

    junkie_3b_cfg = json.loads((e2e_repo_root / "models" / "configs" / "junkie-3b.json").read_text(encoding="utf-8"))

    e16 = MemoryPlanner.estimate_vram({**junkie_3b_cfg, "galore_rank": 16, "steer_mode": "upgraded", "use_muon": True}, target_vram_gb=24.0, available_vram_gb=24.0)
    e32 = MemoryPlanner.estimate_vram({**junkie_3b_cfg, "galore_rank": 32, "steer_mode": "upgraded", "use_muon": True}, target_vram_gb=24.0, available_vram_gb=24.0)
    e64 = MemoryPlanner.estimate_vram({**junkie_3b_cfg, "galore_rank": 64, "steer_mode": "upgraded", "use_muon": True}, target_vram_gb=24.0, available_vram_gb=24.0)
    assert e16.optimizer_memory_gb < e32.optimizer_memory_gb < e64.optimizer_memory_gb, (
        f"Expected strict rank monotonicity 16 < 32 < 64: {(e16.optimizer_memory_gb, e32.optimizer_memory_gb, e64.optimizer_memory_gb)}"
    )
    assert e16.total_vram_gb < e32.total_vram_gb < e64.total_vram_gb

    p8 = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=8.0)
    p24 = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=24.0)
    p80 = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=80.0)

    assert "streaming" in str(p8["recommended_train_mode"]).lower(), f"8GB must require streaming: {p8}"
    assert p24["recommended_train_mode"] != "resident_bf16_train", f"24GB must not claim resident_bf16_train: {p24}"
    assert "resident" in str(p80["recommended_train_mode"]).lower(), f"80GB A100 should fit junkie-3b resident: {p80}"


@pytest.mark.tier2
def test_t2_f2_plan_memory_invalid_vram_and_missing_model_errors(
    e2e_repo_root: Path, e2e_python_exe: str
) -> None:
    """F2-T2-08: `kcx plan-memory` rejects invalid --vram-gb and missing offline models cleanly without traceback."""
    bad_vram = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["plan-memory", "--vram-gb", "not_a_number"],
        cwd=e2e_repo_root,
        timeout=30.0,
    )
    assert bad_vram.returncode == 2

    for non_pos in ("0", "-4"):
        p_zero = run_kcx_cli(
            e2e_python_exe,
            e2e_repo_root,
            ["plan-memory", "--vram-gb", non_pos, "--model", "junkie-3b"],
            cwd=e2e_repo_root,
            timeout=30.0,
        )
        assert "Traceback (most recent call last)" not in (p_zero.stdout + p_zero.stderr)

    p_missing = run_kcx_cli(
        e2e_python_exe,
        e2e_repo_root,
        ["plan-memory", "--model", "nonexistent_org_xyz/missing_model_999", "--vram-gb", "24"],
        cwd=e2e_repo_root,
        env_overrides={"HF_HUB_OFFLINE": "1"},
        timeout=60.0,
    )
    combined = p_missing.stdout + "\n" + p_missing.stderr
    assert p_missing.returncode != 0, f"Expected non-zero exit for missing offline model:\n{combined}"
    assert "Traceback (most recent call last)" not in combined, f"Raw traceback leaked:\n{combined}"
