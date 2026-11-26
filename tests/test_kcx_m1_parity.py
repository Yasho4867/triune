"""Comprehensive parity tests for Milestone M1: R5 checkpoint resolution, R1 framework upgrades, and CLI ergonomics."""

from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lucid_machines.cli import main as cli_main
from lucid_machines.cli import resolve_default_checkpoint
from lucid_machines.export.exporter import export_model
from lucid_machines.export.vllm_bridge import build_vllm_serving_config, export_vllm_bundle
from lucid_machines.inference import generate_response
from lucid_machines.inference import resolve_default_checkpoint as inference_resolve_default_checkpoint
from lucid_machines.kernels.fast_muon import (
    POLAR_EXPRESS_COEFFS_4,
    compute_polar_express_coeffs,
    fast_zeropower_via_newtonschulz5,
    fast_zeropower_via_newtonschulz5_batched,
    fast_zeropower_via_polar_express,
    fast_zeropower_via_polar_express_batched,
)
from lucid_machines.model.transformer import TriuneTransformer
from lucid_machines.trainer.engine import TrainingEngine
from lucid_machines.trainer.evaluation import (
    evaluate_mqar,
    evaluate_multiple_choice_tasks,
    run_sample,
    run_zero_shot_eval,
)
from lucid_machines.universal import UniversalDenseTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(argv: list[str], *, stdin_text: str | None = None, prog: str = "kcx") -> tuple[int, str]:
    buf = io.StringIO()
    code = 0
    ctx_stdin = patch.object(sys, "stdin", io.StringIO(stdin_text)) if stdin_text is not None else nullcontext()
    with patch.object(sys, "argv", [prog, *argv]), redirect_stdout(buf), ctx_stdin:
        try:
            cli_main()
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return code, buf.getvalue()


def _create_tiny_checkpoint(tmp_path: Path, *, batch_size: int = 1, seq_len: int = 16) -> tuple[Path, Path, int]:
    tok_path = REPO_ROOT / "triune_tokenizer.json"
    from lucid_machines.data import load_tokenizer

    tok = load_tokenizer(tok_path)
    vocab_size = tok.get_vocab_size()
    cfg = {
        "vocab_size": vocab_size,
        "hidden_dim": 64,
        "num_layers": 5,
        "num_heads": 2,
        "head_dim": 32,
        "num_experts": 2,
        "top_k_experts": 1,
        "shared_expert": True,
        "shared_expert_scale": 0.5,
        "expert_hidden_multiplier": 6,
        "rope_max_seq_len": 4096,
        "router_prefix_layers": 1,
        "reflex_exit_layer": 2,
        "limbic_exit_layer": 3,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }
    torch.manual_seed(7)
    model = TriuneTransformer(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=5,
        num_heads=2,
        head_dim=32,
        num_experts=2,
        top_k_experts=1,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
    )
    ckpt_path = tmp_path / "tiny_ckpt.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg, "step": 12}, ckpt_path)
    return ckpt_path, tok_path, vocab_size


def test_resolve_default_checkpoint_precedence_and_torch_free(tmp_path: Path) -> None:
    assert resolve_default_checkpoint is inference_resolve_default_checkpoint

    d1 = tmp_path / "checkpoints"
    d2 = tmp_path / "checkpoints_full"
    d1.mkdir()
    d2.mkdir()
    sess = d1 / "session_state.json"

    # Empty -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        resolve_default_checkpoint(None, search_dirs=(d1, d2), session_file=sess)

    older = d1 / "step_100.pt"
    newer = d1 / "step_200.pt"
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    # Step C: newest .pt when no session_state or legacy defaults exist
    assert resolve_default_checkpoint(None, search_dirs=(d1, d2), session_file=sess) == newer

    # Step B: legacy defaults respect prefer="active" vs "latest"
    best = d1 / "best.pt"
    latest = d1 / "latest.pt"
    best.write_bytes(b"best")
    latest.write_bytes(b"latest")
    assert resolve_default_checkpoint(None, prefer="active", search_dirs=(d1, d2), session_file=sess) == best
    assert resolve_default_checkpoint(None, prefer="latest", search_dirs=(d1, d2), session_file=sess) == latest

    # Step A: session_state active_checkpoint takes precedence over legacy/mtime
    sess.write_text(json.dumps({"active_checkpoint": "step_100.pt"}), encoding="utf-8")
    assert resolve_default_checkpoint(None, search_dirs=(d1, d2), session_file=sess) == older

    # Corrupt session_state falls back gracefully
    sess.write_text("{corrupt json", encoding="utf-8")
    assert resolve_default_checkpoint(None, prefer="active", search_dirs=(d1, d2), session_file=sess) == best

    # Explicit existing & explicit missing never silently fall back
    explicit_ok = tmp_path / "custom.pt"
    explicit_ok.write_bytes(b"ok")
    assert resolve_default_checkpoint(explicit_ok, search_dirs=(d1, d2), session_file=sess) == explicit_ok
    with pytest.raises(FileNotFoundError):
        resolve_default_checkpoint(tmp_path / "missing_explicit.pt", search_dirs=(d1, d2), session_file=sess)

    # Lazy torch invariant: importing cli + resolve_default_checkpoint must NOT import torch
    probe = (
        "import sys; "
        "from lucid_machines.cli import build_cli_parser, resolve_default_checkpoint; "
        "build_cli_parser('kcx'); "
        "assert 'torch' not in sys.modules, 'torch eagerly imported'"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_cli_missing_and_corrupt_checkpoint_errors_and_offline_eval(tmp_path: Path) -> None:
    missing = str(tmp_path / "does_not_exist.pt")
    for argv in (
        ["infer", "--prompt", "hi", "--checkpoint", missing],
        ["eval", "--checkpoint", missing],
        ["benchmark", "--checkpoint", missing],
        ["profile", "--checkpoint", missing],
        ["export", "--checkpoint", missing, "--format", "safetensors"],
        ["adapter", "merge", "--adapter", missing, "--checkpoint", missing],
    ):
        code, out = _run_cli(argv)
        assert code == 1, (argv, out)
        assert "Traceback" not in out

    # JSON error output on corrupt checkpoint
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a valid torch zip")
    code_inf, out_inf = _run_cli(["infer", "--prompt", "hi", "--checkpoint", str(corrupt), "--json"])
    assert code_inf == 1
    err_inf = json.loads(out_inf)
    assert err_inf["status"] == "error"
    assert err_inf["ok"] is False

    code_ev, out_ev = _run_cli(["eval", "--checkpoint", str(corrupt), "--json"])
    assert code_ev == 1
    err_ev = json.loads(out_ev)
    assert err_ev["status"] == "error"

    # Offline kcx eval fallback + small-sample batch_size fallback + shuffle_buffer cap
    ckpt_path, tok_path, _ = _create_tiny_checkpoint(tmp_path, batch_size=64, seq_len=16)
    code_ok, out_ok = _run_cli(
        [
            "eval",
            "--checkpoint",
            str(ckpt_path),
            "--tokenizer",
            str(tok_path),
            "--max-steps",
            "1",
            "--device",
            "cpu",
            "--prompt",
            "Hello",
            "--max-tokens",
            "3",
            "--json",
        ]
    )
    assert code_ok == 0, out_ok
    ev_json = json.loads(out_ok)
    assert ev_json["status"] == "ok"
    assert ev_json["batches_evaluated"] == 1
    assert "fineweb_sample.jsonl" in str(ev_json["dataset"])
    assert ev_json["perplexity"] > 0

    # run_sample empty encoded prompt guard
    fake_tok = SimpleNamespace(encode=lambda _text: SimpleNamespace(ids=[]))
    fake_trainer = SimpleNamespace(tokenizer=fake_tok, sample_prompt="hi")
    with pytest.raises(ValueError, match="zero tokens"):
        run_sample(fake_trainer, prompt="")


def test_polar_express_orthogonality_and_batched_parity() -> None:
    assert len(POLAR_EXPRESS_COEFFS_4) == 4
    for s in range(1, 8):
        sched = compute_polar_express_coeffs(s)
        assert len(sched) == s

    torch.manual_seed(42)
    cases = [
        torch.randn(128, 128),
        torch.randn(256, 128),
        torch.randn(128, 256),
        torch.diag(torch.linspace(1.0, 0.01, 64)),
    ]
    for g in cases:
        g_orig = g.clone()
        pe = fast_zeropower_via_polar_express(g, steps=4)
        ns = fast_zeropower_via_newtonschulz5(g, steps=5, method="newtonschulz5")
        assert torch.equal(g, g_orig)
        assert torch.isfinite(pe).all()

        # Exact polar factor U @ V^T via SVD
        u, _, vh = torch.linalg.svd(g, full_matrices=False)
        polar_exact = u @ vh
        cos_pe = torch.nn.functional.cosine_similarity(pe.flatten(), polar_exact.flatten(), dim=0).item()
        cos_ns = torch.nn.functional.cosine_similarity(ns.flatten(), polar_exact.flatten(), dim=0).item()
        assert cos_pe >= cos_ns - 1e-4

        eye = torch.eye(min(g.shape))
        gram_pe = pe @ pe.T if pe.size(0) <= pe.size(1) else pe.T @ pe
        gram_ns = ns @ ns.T if ns.size(0) <= ns.size(1) else ns.T @ ns
        err_pe = torch.linalg.matrix_norm(gram_pe - eye).item()
        err_ns = torch.linalg.matrix_norm(gram_ns - eye).item()
        assert err_pe < err_ns

    # Zero-matrix invariance
    z = torch.zeros(32, 32)
    assert torch.equal(fast_zeropower_via_polar_express(z), z)

    # Batched vs sequential parity
    batch = torch.randn(4, 32, 48)
    b_out = fast_zeropower_via_polar_express_batched(batch, steps=4)
    seq_out = torch.stack([fast_zeropower_via_polar_express(batch[i], steps=4) for i in range(4)], dim=0)
    assert (b_out - seq_out).abs().max().item() < 1e-4

    # Default method on fast_zeropower_via_newtonschulz5_batched matches polar_express
    b_def = fast_zeropower_via_newtonschulz5_batched(batch)
    assert torch.equal(b_def, b_out)

    # BF16 finite & accurate
    g_bf16 = torch.randn(128, 64, dtype=torch.bfloat16)
    pe_bf16 = fast_zeropower_via_polar_express(g_bf16, steps=4)
    assert pe_bf16.dtype == torch.bfloat16
    assert torch.isfinite(pe_bf16).all()


def test_training_engine_chunked_hidden_ce_parity_and_fallbacks() -> None:
    def _make_trainer(model, depth_mode: str, *, use_chunked_ce: bool, exploration: str = "none"):
        vocab_size = int(getattr(model, "vocab_size", 64))
        torch.manual_seed(99)
        batch = torch.randint(1, vocab_size, (4, 33), dtype=torch.long)

        class _StubTrainer:
            def __init__(self) -> None:
                self.model = model
                self.device = torch.device("cpu")
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.router_loss_fn = nn.CrossEntropyLoss()
                self.grad_accum = 1
                self.z_loss_coef = 1e-3
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                self.config = {
                    "target_depth_dist": [0.34, 0.33, 0.33],
                    "exploration": exploration,
                    "exploration_steps": 10 if exploration != "none" else 0,
                    "usage_ema_decay": 0.99,
                    "depth_mode": depth_mode,
                    "balance_coef": 0.3,
                    "grad_accum_steps": 1,
                    "grad_clip": 1.0,
                    "log_every": 100,
                    "total_steps": 10,
                    "lr": 1e-3,
                    "use_chunked_ce": use_chunked_ce,
                    "ce_chunk_size": 32,
                }

            def next_batch(self):
                return batch[:, :-1], batch[:, 1:].contiguous()

            @staticmethod
            def model_autocast():
                return nullcontext()

            @staticmethod
            def lr_schedule(cfg, _step):
                return cfg["lr"]

            @staticmethod
            def set_optimizer_lr(opt, lr, expert_lr=None):
                for g in opt.param_groups:
                    g["lr"] = lr

        return _StubTrainer()

    # 1. TriuneTransformer cortex & dynamic parity
    for mode in ("cortex", "dynamic"):
        torch.manual_seed(17)
        m_ref = TriuneTransformer(
            vocab_size=64,
            hidden_dim=32,
            num_layers=5,
            num_heads=2,
            head_dim=16,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
        )
        m_chk = copy.deepcopy(m_ref)
        t_ref = _make_trainer(m_ref, mode, use_chunked_ce=False)
        t_chk = _make_trainer(m_chk, mode, use_chunked_ce=True)
        e_ref = TrainingEngine(t_ref)
        e_chk = TrainingEngine(t_chk)
        e_ref.step = 1
        e_chk.step = 1

        torch.manual_seed(777)
        s_ref = e_ref.step_once()
        torch.manual_seed(777)
        s_chk = e_chk.step_once()
        assert abs(s_ref["lm_loss"] - s_chk["lm_loss"]) < 1e-5
        assert abs(s_ref["loss"] - s_chk["loss"]) < 1e-5
        for (_, p1), (_, p2) in zip(m_ref.named_parameters(), m_chk.named_parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)

    # 2. UniversalDenseTransformer return_hidden=True parity
    torch.manual_seed(23)
    d_ref = UniversalDenseTransformer(vocab_size=64, dim=32, num_layers=2, num_heads=2)
    d_chk = copy.deepcopy(d_ref)
    t_d_ref = _make_trainer(d_ref, "cortex", use_chunked_ce=False)
    t_d_chk = _make_trainer(d_chk, "cortex", use_chunked_ce=True)
    torch.manual_seed(777)
    sd_ref = TrainingEngine(t_d_ref).step_once()
    torch.manual_seed(777)
    sd_chk = TrainingEngine(t_d_chk).step_once()
    assert abs(sd_ref["lm_loss"] - sd_chk["lm_loss"]) < 1e-5
    for (_, p1), (_, p2) in zip(d_ref.named_parameters(), d_chk.named_parameters()):
        torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)

    # 3. Verify chunked_cross_entropy_from_hidden is actually called on TriuneTransformer cortex
    with patch(
        "lucid_machines.trainer.engine.chunked_cross_entropy_from_hidden",
        side_effect=AssertionError("chunked_ce_invoked"),
    ):
        with pytest.raises(AssertionError, match="chunked_ce_invoked"):
            TrainingEngine(_make_trainer(copy.deepcopy(m_ref), "cortex", use_chunked_ce=True)).step_once()

    # 4. Logits-only dummy model falls back cleanly without hitting chunked_cross_entropy_from_hidden
    class _LogitsOnly(nn.Module):
        vocab_size = 64

        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(64, 16)
            self.head = nn.Linear(16, 64)

        def forward(self, input_ids: torch.Tensor):
            return self.head(self.emb(input_ids))

    lo_stats = TrainingEngine(_make_trainer(_LogitsOnly(), "cortex", use_chunked_ce=True)).step_once()
    assert torch.isfinite(torch.tensor(lo_stats["lm_loss"]))

    # 3. Mixed-dtype guard (bf16 hidden + fp32 final_head) must not crash
    torch.manual_seed(31)
    m_mix = TriuneTransformer(
        vocab_size=64,
        hidden_dim=32,
        num_layers=5,
        num_heads=2,
        head_dim=16,
        num_experts=2,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
    )
    m_mix.final_norm.weight.data = m_mix.final_norm.weight.data.bfloat16()
    t_mix = _make_trainer(m_mix, "cortex", use_chunked_ce=True)
    e_mix = TrainingEngine(t_mix)
    x_mix, y_mix = t_mix.next_batch()
    loss_mix, _ = e_mix._forward_lm_loss(x_mix, y_mix.reshape(-1), 64, force_depth=2)
    assert torch.isfinite(loss_mix)


def test_mqar_and_zeroshot_eval_and_cli_hooks(tmp_path: Path) -> None:
    ckpt_path, tok_path, _ = _create_tiny_checkpoint(tmp_path)
    from lucid_machines.inference import load_checkpoint_model

    model, tok, _, dev, _, _ = load_checkpoint_model(ckpt_path, tok_path, device="cpu")
    mqar_res = evaluate_mqar(model, tok, device=dev, num_trials=3, num_kv_pairs=3)
    assert mqar_res["evaluator"] == "native_mqar_loglikelihood"
    assert mqar_res["num_trials"] == 3
    assert 0.0 <= mqar_res["choice_accuracy"] <= 1.0
    assert 0.0 <= mqar_res["exact_top1_accuracy"] <= 1.0

    # Direct multiple-choice evaluation check
    mc_direct = evaluate_multiple_choice_tasks(
        model,
        tok,
        [{"prompt": "Water freezes at", "choices": [" 0 C", " 100 C"], "answer": 0}],
        device=dev,
    )
    assert mc_direct["evaluator"] == "native_loglikelihood"
    assert mc_direct["num_examples"] == 1

    # Local JSONL MC task
    mc_file = tmp_path / "mc.jsonl"
    mc_file.write_text(
        json.dumps({"prompt": "Water freezes at", "choices": [" 0 C", " 100 C"], "answer": 0}) + "\n",
        encoding="utf-8",
    )
    zs = run_zero_shot_eval(model, tok, ["mqar", "zeroshot", str(mc_file)], device=dev, limit=2, num_trials=2)
    assert set(zs["results"].keys()) == {"mqar", "zeroshot", str(mc_file)}
    assert zs["results"]["zeroshot"]["num_examples"] == 2
    assert zs["results"][str(mc_file)]["num_examples"] == 1

    # CLI eval --tasks --json
    code_ev, out_ev = _run_cli(
        [
            "eval",
            "--checkpoint",
            str(ckpt_path),
            "--tokenizer",
            str(tok_path),
            "--tasks",
            "mqar,zeroshot",
            "--num-trials",
            "2",
            "--limit",
            "2",
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert code_ev == 0, out_ev
    parsed_ev = json.loads(out_ev)
    assert parsed_ev["status"] == "ok"
    assert "mqar" in parsed_ev["results"]
    assert "zeroshot" in parsed_ev["results"]

    # CLI benchmark --suite mqar --quick --json
    code_bm, out_bm = _run_cli(
        [
            "benchmark",
            "--suite",
            "mqar",
            "--quick",
            "--seq-lens",
            "64",
            "--checkpoint",
            str(ckpt_path),
            "--tokenizer",
            str(tok_path),
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert code_bm == 0, out_bm
    parsed_bm = json.loads(out_bm)
    assert parsed_bm["status"] == "ok"
    assert parsed_bm["suite"] == "mqar"
    assert len(parsed_bm["kernel_results"]) == 3
    assert parsed_bm["checkpoint_eval"] is not None


def test_universal_json_and_infer_ergonomics(tmp_path: Path) -> None:
    ckpt_path, tok_path, _ = _create_tiny_checkpoint(tmp_path)

    for cmd in (["models", "--json"], ["gpucheck", "--json"], ["doctor", "--json"], ["checkpoint", "list", "--json"]):
        code, out = _run_cli(cmd)
        assert code == 0, (cmd, out)
        payload = json.loads(out)
        assert "status" in payload

    fake_scan = {
        "models": [
            {
                "id": "triune-test",
                "name": "triune-test",
                "source": "checkpoint",
                "kind": "checkpoint",
                "architecture": "TriuneTransformer",
                "quantization": "BF16",
                "size_str": "1.0 MB",
            }
        ]
    }
    with patch("lucid_machines.runtime.model_scanner.scan_pc_models", return_value=fake_scan):
        code_sc, out_sc = _run_cli(["scan", "--json"])
        assert code_sc == 0
        assert json.loads(out_sc)["count"] == 1

    code_cfg_err, out_cfg_err = _run_cli(["config", "show", "missing-cfg-xyz", "--json"])
    assert code_cfg_err == 1
    assert json.loads(out_cfg_err)["status"] == "error"

    # Direct generate_response system_prompt + seed check
    from lucid_machines.inference import load_checkpoint_model

    m_inf, tok_inf, _, dev_inf, pad_inf, eos_inf = load_checkpoint_model(ckpt_path, tok_path, device="cpu")
    g1 = generate_response(
        m_inf,
        tok_inf,
        "Hello",
        device=dev_inf,
        pad_token_id=pad_inf,
        eos_token_id=eos_inf,
        system_prompt="Reply briefly.",
        seed=7,
        max_new_tokens=4,
    )
    g2 = generate_response(
        m_inf,
        tok_inf,
        "Hello",
        device=dev_inf,
        pad_token_id=pad_inf,
        eos_token_id=eos_inf,
        system_prompt="Reply briefly.",
        seed=7,
        max_new_tokens=4,
    )
    assert g1 == g2

    # infer --system --seed --json + stdin piping + deterministic seed
    code1, out1 = _run_cli(
        [
            "infer",
            "--checkpoint",
            str(ckpt_path),
            "--tokenizer",
            str(tok_path),
            "--stdin",
            "--system",
            "Be concise.",
            "--seed",
            "123",
            "--max-tokens",
            "4",
            "--device",
            "cpu",
            "--json",
        ],
        stdin_text="Hello world",
    )
    code2, out2 = _run_cli(
        [
            "infer",
            "--checkpoint",
            str(ckpt_path),
            "--tokenizer",
            str(tok_path),
            "--prompt",
            "Hello world",
            "--system",
            "Be concise.",
            "--seed",
            "123",
            "--max-tokens",
            "4",
            "--device",
            "cpu",
            "--json",
        ]
    )
    assert code1 == 0, out1
    assert code2 == 0, out2
    d1, d2 = json.loads(out1), json.loads(out2)
    assert d1["status"] == "ok" and d1["seed"] == 123
    assert d1["text"] == d2["text"]


def test_serve_model_preload_and_finetune_recipe_and_wandb(tmp_path: Path) -> None:
    # 1. serve --model dispatch without mutating checkpoints/session_state.json
    mock_state = MagicMock()
    with (
        patch("lucid_machines.api.routes.pytorch_state", mock_state),
        patch("lucid_machines.api.run_server") as mock_run_server,
    ):
        c1, _ = _run_cli(["serve", "--model", "ollama:gemma3:4b", "--port", "8123"])
        assert c1 == 0
        mock_state.load_ollama_model.assert_called_once_with("ollama:gemma3:4b")
        mock_run_server.assert_called_once_with(host="127.0.0.1", port=8123)

        c2, _ = _run_cli(["serve", "--model", "triune-small"])
        assert c2 == 0
        mock_state.reconfigure_architecture.assert_called_once_with("triune-small", instantiate_model=True)

        fake_pt = tmp_path / "m.pt"
        fake_pt.write_bytes(b"pt")
        c3, _ = _run_cli(["serve", "--model", str(fake_pt)])
        assert c3 == 0
        mock_state.load_model_checkpoint.assert_called_once_with(fake_pt)

        c4, _ = _run_cli(["serve", "--model", "Qwen/Qwen2.5-0.5B", "--nvfp4", "--disk-stream"])
        assert c4 == 0
        mock_state.load_hf_model.assert_called_once_with("Qwen/Qwen2.5-0.5B", nvfp4=True, disk_stream=True)

    # 2. finetune --recipe (YAML & JSON) precedence + mocked WandbLogger
    recipe_yaml = tmp_path / "recipe.yaml"
    recipe_yaml.write_text(
        "\n".join(
            [
                "lora:",
                "  rank: 8",
                "  alpha: 16.0",
                "data:",
                "  dataset: data/fineweb_sample.jsonl",
                "  seq_len: 32",
                "training:",
                "  epochs: 2",
                "  batch_size: 2",
                "  lr: 0.0001",
                f"  output_dir: {(tmp_path / 'out_yaml').as_posix()}",
                "logging:",
                "  wandb: true",
                "  project: test_proj",
                "  run_id: yaml_run",
            ]
        ),
        encoding="utf-8",
    )

    ckpt_path, _, _ = _create_tiny_checkpoint(tmp_path)
    fake_wb = MagicMock()
    fake_wb.run_id = "yaml_run"
    captured_fit: dict = {}

    def _fake_fit(self, **kwargs):
        captured_fit.update(kwargs)
        kwargs["on_step"](1, 1, 1.25, 1.15)
        return {"final_loss": 1.25, "output_dir": kwargs["output_dir"]}

    with (
        patch("lucid_machines.trainer.logger.WandbLogger", return_value=fake_wb) as mock_wb_cls,
        patch("lucid_machines.trainer.finetune.TriuneFineTuner.fit", new=_fake_fit),
    ):
        # CLI --epochs 5 overrides recipe epochs=2; recipe rank=8 & seq_len=32 override defaults
        code_ft, out_ft = _run_cli(
            [
                "finetune",
                "--checkpoint",
                str(ckpt_path),
                "--recipe",
                str(recipe_yaml),
                "--epochs",
                "5",
                "--max-steps",
                "1",
            ]
        )
    assert code_ft == 0, out_ft
    assert captured_fit["epochs"] == 5
    assert captured_fit["seq_len"] == 32
    assert captured_fit["batch_size"] == 2
    mock_wb_cls.assert_called_once()
    assert mock_wb_cls.call_args.kwargs["project"] == "test_proj"
    assert mock_wb_cls.call_args.kwargs["run_id"] == "yaml_run"
    fake_wb.log.assert_called_once_with({"loss": 1.25, "lm_loss": 1.15}, step=1)
    fake_wb.finish.assert_called_once()

    # Verify JSON recipe + finally finish() even when fit() raises
    recipe_json = tmp_path / "recipe.json"
    recipe_json.write_text(
        json.dumps(
            {
                "rank": 4,
                "epochs": 3,
                "seq_len": 24,
                "batch_size": 2,
                "output_dir": (tmp_path / "out_json").as_posix(),
                "wandb": True,
                "wandb_project": "json_proj",
                "wandb_run": "json_run",
            }
        ),
        encoding="utf-8",
    )
    fake_wb2 = MagicMock()
    fake_wb2.run_id = "json_run"
    with (
        patch("lucid_machines.trainer.logger.WandbLogger", return_value=fake_wb2),
        patch("lucid_machines.trainer.finetune.TriuneFineTuner.fit", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            _run_cli(["finetune", "--checkpoint", str(ckpt_path), "--recipe", str(recipe_json)])
    fake_wb2.finish.assert_called_once()


def test_agent_claude_code_parity_and_vllm_export(tmp_path: Path) -> None:
    # 1. agent --output-format json / stream-json / --session-id / --resume / --stdin / --allowed-tools
    store_dir = tmp_path / "agent_store"
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    captured_tools: list[list[str]] = []

    def _fake_run_goal(self, goal, **kwargs):
        captured_tools.append(sorted(self.tools.names()))
        on_ev = kwargs.get("on_event")
        if on_ev:
            on_ev("thought", {"step": 1, "thought": f"Working on: {goal}"})
        return {
            "answer": f"done:{goal}:hist={len(kwargs.get('conversation_history') or [])}",
            "persona": "engineer",
            "reasoning_depth": "reflex",
            "steps": 1,
            "created_skills": [],
            "learned_lessons": [],
        }

    with patch("lucid_machines.agents.AgentHarness.run_goal", new=_fake_run_goal):
        # First turn via stdin + stream-json + session-id + allowed-tools
        c1, out1 = _run_cli(
            [
                "agent",
                "--workspace",
                str(ws_dir),
                "--store-dir",
                str(store_dir),
                "--session-id",
                "sess_m1",
                "--allowed-tools",
                "read_file,list_dir",
                "--output-format",
                "stream-json",
                "--stdin",
            ],
            stdin_text="first goal",
        )
        assert c1 == 0, out1
        assert captured_tools[-1] == ["list_dir", "read_file"]
        lines = [json.loads(ln) for ln in out1.strip().splitlines() if ln.strip()]
        assert lines[0]["type"] == "thought"
        assert lines[-1]["type"] == "result"
        assert "done:first goal:hist=0" in lines[-1]["answer"]

        # Second turn resumes sess_m1 and gets history=2
        c2, out2 = _run_cli(
            [
                "agent",
                "--workspace",
                str(ws_dir),
                "--store-dir",
                str(store_dir),
                "--resume",
                "sess_m1",
                "--output-format",
                "json",
                "--goal",
                "second goal",
            ]
        )
        assert c2 == 0, out2
        res2 = json.loads(out2)
        assert res2["status"] == "ok"
        assert res2["session_id"] == "sess_m1"
        assert "hist=2" in res2["answer"]

        # Unknown-only allowed-tools -> exit 2
        c_bad_tool, _ = _run_cli(
            [
                "agent",
                "--workspace",
                str(ws_dir),
                "--store-dir",
                str(store_dir),
                "--allowed-tools",
                "nonexistent_tool_xyz",
                "--goal",
                "hi",
            ]
        )
        assert c_bad_tool == 2

        # Missing --resume session -> exit 1
        c_no_sess, _ = _run_cli(
            [
                "agent",
                "--workspace",
                str(ws_dir),
                "--store-dir",
                str(store_dir),
                "--resume",
                "missing_session_xyz",
                "--goal",
                "hi",
            ]
        )
        assert c_no_sess == 1

    # 2. export --format vllm + GGUF early exit 2
    ckpt_path, _, _ = _create_tiny_checkpoint(tmp_path)
    vllm_out = tmp_path / "vllm_bundle"
    code_exp, out_exp = _run_cli(["export", "--checkpoint", str(ckpt_path), "--format", "vllm", "--output", str(vllm_out)])
    assert code_exp == 0, out_exp
    assert (vllm_out / "config.json").is_file()
    assert (vllm_out / "model.safetensors").is_file()
    assert (vllm_out / "vllm_serving.json").is_file()
    vcfg = json.loads((vllm_out / "vllm_serving.json").read_text(encoding="utf-8"))
    assert vcfg["trust_remote_code"] is False
    assert "vllm.entrypoints.openai.api_server" in vcfg["launch_command"]

    # Direct exporter Python API check
    tiny_m = TriuneTransformer(
        vocab_size=64,
        hidden_dim=32,
        num_layers=5,
        num_heads=2,
        head_dim=16,
        num_experts=2,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
    )
    api_dir = export_model(tiny_m, tmp_path / "vllm_api", fmt="vllm")
    assert (Path(api_dir) / "vllm_serving.json").is_file()
    direct_dir = export_vllm_bundle(tiny_m, tmp_path / "vllm_direct", model_id="tiny-direct")
    assert (direct_dir / "vllm_serving.json").is_file()
    assert build_vllm_serving_config("triune-small")["trust_remote_code"] is False
