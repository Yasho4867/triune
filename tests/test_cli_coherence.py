"""Tests for Milestone M1: CLI coherence, subcommands, exit codes, ModelSpec & Triune config threading."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from lucid_machines import __version__
from lucid_machines.cli import build_cli_parser
from lucid_machines.cli import main as cli_main
from lucid_machines.model.base import CANONICAL_MODEL_SPECS, build_model_from_spec, get_model_spec
from lucid_machines.model.transformer import TriuneTransformer
from lucid_machines.packaging import POLYSOM_SVG_ICON
from lucid_machines.runtime.memory_planner import MemoryPlanner
from scripts.chat import main as chat_main
from scripts.train import main as train_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli_inproc(argv: list[str], prog: str = "lucid-machines") -> tuple[int, str]:
    buf = io.StringIO()
    code = 0
    with patch.object(sys, "argv", [prog, *argv]), redirect_stdout(buf):
        try:
            cli_main()
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return code, buf.getvalue()


def test_cli_version_flag_and_dynamic_prog() -> None:
    code, out = _run_cli_inproc(["--version"], prog="lucid-machines")
    assert code == 0
    assert f"lucid-machines {__version__}" in out

    code_short, out_short = _run_cli_inproc(["-V"], prog="polysom")
    assert code_short == 0
    assert f"polysom {__version__}" in out_short

    code_help, out_help = _run_cli_inproc(["--help"], prog="polysom")
    assert code_help == 0
    assert "polysom scan" in out_help
    assert "polysom eval" in out_help
    assert "polysom config" in out_help
    assert "polysom workspace" in out_help


def test_cli_train_help_and_dry_run_forwarding() -> None:
    proc_help = subprocess.run(
        [sys.executable, "-m", "lucid_machines", "train", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_help.returncode == 0, proc_help.stderr
    assert "--model_config" in proc_help.stdout
    assert "--steer_mode" in proc_help.stdout

    proc_dry = subprocess.run(
        [sys.executable, "-m", "lucid_machines", "train", "--model_config", "junkie-3b", "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_dry.returncode == 0, proc_dry.stderr
    assert "2,990,355,155" in proc_dry.stdout


def test_cli_error_exit_codes_and_lazy_subactions() -> None:
    code_infer, _ = _run_cli_inproc(["infer"])
    assert code_infer == 2

    with patch("lucid_machines.runtime.model_scanner.query_ollama_running_tags", return_value=set()):
        code_ol, _ = _run_cli_inproc(["infer", "-p", "hi", "--model", "ollama:missing:1b"])
        assert code_ol == 1

        with pytest.raises(SystemExit) as exc_chat:
            chat_main(["--model", "ollama:missing:1b"])
        assert exc_chat.value.code == 1

    code_persona, _ = _run_cli_inproc(["agent", "--persona", "nonexistent_persona_xyz"])
    assert code_persona == 2

    code_ws_missing, _ = _run_cli_inproc(["agent", "--workspace", str(REPO_ROOT / "no_such_workspace_dir_xyz")])
    assert code_ws_missing == 1

    with patch("lucid_machines.agents.AgentHarness.run_goal", side_effect=RuntimeError("boom")):
        code_goal_err, _ = _run_cli_inproc(["agent", "--goal", "test fail"])
        assert code_goal_err == 1

    with patch("lucid_machines.agents.MultiAgentOrchestrator.orchestrate_team", side_effect=RuntimeError("team boom")):
        code_team_err, _ = _run_cli_inproc(["agent", "--team", "architect,engineer", "--goal", "test fail"])
        assert code_team_err == 1

    code_gguf_ckpt, out_gguf_ckpt = _run_cli_inproc(
        ["checkpoint", "export", "--checkpoint", "missing.pt", "--format", "gguf"]
    )
    assert code_gguf_ckpt == 2
    assert "not yet supported" in out_gguf_ckpt.lower() or "gguf" in out_gguf_ckpt.lower()

    code_gguf_exp, _ = _run_cli_inproc(["export", "--checkpoint", "missing.pt", "--format", "gguf"])
    assert code_gguf_exp == 2

    # Adapter & checkpoint with no subaction must exit 2 and NOT eagerly import torch
    for sub in ("adapter", "checkpoint"):
        probe = (
            "import sys; sys.argv=['lucid-machines', '" + sub + "']; "
            "from lucid_machines.cli import main; "
            "code = 0\n"
            "try:\n main()\n"
            "except SystemExit as e: code = e.code\n"
            "assert code == 2, f'got {code}'; "
            "assert 'torch' not in sys.modules, 'torch eagerly imported'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout


def test_cli_plan_memory_local_presets_and_models_active_column() -> None:
    est = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=24.0)
    assert est["is_triune"] is True
    assert est["total_params"] == 2_990_355_155
    assert est["top_k_experts"] == 1
    assert est["active_params"] < est["total_params"]

    code_pm, out_pm = _run_cli_inproc(["plan-memory", "--model", "junkie-3b", "--vram-gb", "24"])
    assert code_pm == 0
    assert "2,990,355,155" in out_pm
    assert "CentroidSteer" in out_pm

    code_m, out_m = _run_cli_inproc(["models"])
    assert code_m == 0
    assert "Active" in out_m
    assert "junkie-3b" in out_m


def test_cli_config_and_workspace_subcommands(tmp_path: Path) -> None:
    code_list, out_list = _run_cli_inproc(["config", "list"])
    assert code_list == 0
    assert "junkie-3b" in out_list

    code_show, out_show = _run_cli_inproc(["config", "show", "junkie-3b", "--json"])
    assert code_show == 0
    parsed = json.loads(out_show)
    assert parsed["top_k_experts"] == 1
    assert parsed["num_experts"] == 16

    code_val, _ = _run_cli_inproc(["config", "validate", "--all"])
    assert code_val == 0

    code_missing, _ = _run_cli_inproc(["config", "show", "nonexistent-preset-xyz"])
    assert code_missing == 1

    # Workspace subcommand
    ws_dir = tmp_path / "ws_project"
    code_init, _ = _run_cli_inproc(["workspace", "init", "--path", str(ws_dir)])
    assert code_init == 0
    assert (ws_dir / "models" / "configs").is_dir()

    sample_file = ws_dir / "hello.txt"
    sample_file.write_text("line one\nline two\n", encoding="utf-8")

    code_st, out_st = _run_cli_inproc(["workspace", "status", "--path", str(ws_dir)])
    assert code_st == 0
    assert "ws_project" in out_st

    code_tr, out_tr = _run_cli_inproc(["workspace", "tree", "--path", str(ws_dir), "--depth", "2"])
    assert code_tr == 0
    assert "hello.txt" in out_tr

    code_rd, out_rd = _run_cli_inproc(["workspace", "read", "hello.txt", "--path", str(ws_dir)])
    assert code_rd == 0
    assert "line two" in out_rd

    code_trav, _ = _run_cli_inproc(["workspace", "read", "../outside.txt", "--path", str(ws_dir)])
    assert code_trav == 2


def test_cli_eval_and_agent_workspace_mode(tmp_path: Path) -> None:
    # Missing checkpoint -> non-zero exit
    code_missing, _ = _run_cli_inproc(["eval", "--checkpoint", str(tmp_path / "nope.pt")])
    assert code_missing == 1

    # Real eval smoke test with synthetic tiny checkpoint & fake eval_loader
    tok_path = REPO_ROOT / "triune_tokenizer.json"
    from lucid_machines.data import load_tokenizer

    tok = load_tokenizer(tok_path)
    vocab_size = tok.get_vocab_size()
    tiny_cfg = {
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
        "seq_len": 16,
        "batch_size": 1,
    }
    tiny_model = TriuneTransformer(
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
    ckpt_path = tmp_path / "tiny.pt"
    torch.save({"model_state": tiny_model.state_dict(), "config": tiny_cfg, "step": 1}, ckpt_path)

    xb = torch.randint(0, vocab_size, (1, 16), dtype=torch.long)
    yb = torch.randint(0, vocab_size, (1, 16), dtype=torch.long)
    with patch("lucid_machines.data.build_dataloader", return_value=iter([(xb, yb)])):
        code_eval, out_eval = _run_cli_inproc(
            [
                "eval",
                "--checkpoint",
                str(ckpt_path),
                "--tokenizer",
                str(tok_path),
                "--max-steps",
                "1",
                "--prompt",
                "hello",
                "--max-tokens",
                "3",
                "--device",
                "cpu",
            ]
        )
    assert code_eval == 0, out_eval
    assert "EVALUATION REPORT" in out_eval.upper()
    assert "Perplexity" in out_eval

    # Agent --workspace & --mode pev
    parser = build_cli_parser()
    parsed = parser.parse_args(["agent", "--workspace", str(tmp_path), "--mode", "pev", "-g", "check"])
    assert parsed.workspace == str(tmp_path)
    assert parsed.mode == "pev"

    with patch(
        "lucid_machines.agents.loops.run_agent_loop", return_value={"answer": "pev done", "steps": 1}
    ) as mock_loop:
        code_pev, out_pev = _run_cli_inproc(
            ["agent", "--workspace", str(tmp_path), "--mode", "pev", "--goal", "inspect"]
        )
    assert code_pev == 0
    assert "pev done" in out_pev
    assert mock_loop.call_args.kwargs["mode"] == "pev"


def test_modelspec_topk_and_transformer_config_threading() -> None:
    s1 = get_model_spec("junkie-3b")
    assert s1.top_k_experts == 1
    assert s1.rope_max_seq_len == 4096
    tot1, act1 = s1.compute_exact_params()
    assert tot1 == 2_990_355_155

    s2 = get_model_spec("junkie-3b", top_k_experts=2)
    tot2, act2 = s2.compute_exact_params()
    assert tot2 == tot1
    assert act2 > act1

    # Verify ModelSpec exact match across CANONICAL_MODEL_SPECS
    spec_small = CANONICAL_MODEL_SPECS["triune-small"]
    tot_s, act_s = spec_small.compute_exact_params()
    m_small = build_model_from_spec(spec_small)
    seen: set[int] = set()
    actual_small = 0
    for p in m_small.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            actual_small += p.numel()
    assert tot_s == actual_small
    assert act_s < tot_s

    # Thread top_k_experts, shared_expert, shared_expert_scale, expert_hidden_multiplier, rope_max_seq_len
    m = TriuneTransformer(
        vocab_size=128,
        hidden_dim=64,
        num_layers=5,
        num_heads=2,
        head_dim=32,
        num_experts=4,
        top_k_experts=2,
        shared_expert=False,
        shared_expert_scale=0.25,
        expert_hidden_multiplier=4,
        rope_max_seq_len=16384,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
    )
    moe_blk = m.layers[3]
    assert moe_blk.ffn.top_k == 2
    assert moe_blk.ffn.shared is None
    assert moe_blk.ffn.shared_scale == pytest.approx(0.25)
    assert moe_blk.ffn.hidden_dim == 64 * 4
    assert moe_blk.attn.gla.rope.max_seq_len == 16384


def test_train_steer_mode_and_packaging_eye_svg() -> None:
    res_rand = train_main(
        [
            "--model_config",
            "junkie-3b",
            "--steer_mode",
            "random_same_mem",
            "--dry-run",
            "--no_wandb",
            "--no_hf_login",
        ]
    )
    assert res_rand["config"]["steer_mode"] == "random_same_mem"
    assert res_rand["config"]["top_k_experts"] == 1
    assert res_rand["config"]["rope_max_seq_len"] == 4096

    res_nosteer = train_main(
        [
            "--model_config",
            "junkie-7b",
            "--steer_mode",
            "nosteer",
            "--dry-run",
            "--no_wandb",
            "--no_hf_login",
        ]
    )
    assert res_nosteer["config"]["steer_mode"] == "nosteer"
    assert res_nosteer["config"]["steer_scale"] == 0.0
    assert res_nosteer["config"]["top_k_experts"] == 2

    res_cent = train_main(
        [
            "--model_config",
            "junkie-mini",
            "--steer_mode",
            "centroid",
            "--dry-run",
            "--no_wandb",
            "--no_hf_login",
        ]
    )
    assert res_cent["config"]["steer_mode"] == "centroid"

    assert "#7b3f3c" in POLYSOM_SVG_ICON
    assert "#f4ecda" in POLYSOM_SVG_ICON
    assert "gradient" not in POLYSOM_SVG_ICON.lower()


def test_plan_memory_train_modes_and_installer_no_personal_wsl_path() -> None:
    est_8gb = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=8.0)
    assert "streaming" in est_8gb["recommended_train_mode"]
    assert "resident_bf16" not in est_8gb["recommended_train_mode"]

    est_12gb = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=12.0)
    assert "streaming" in est_12gb["recommended_train_mode"]

    est_24gb = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=24.0)
    assert est_24gb["recommended_train_mode"] != "resident_bf16_train"
    assert "streaming" in est_24gb["recommended_train_mode"] or "rank" in est_24gb["recommended_train_mode"]

    est_80gb = MemoryPlanner.estimate_hf_model("junkie-3b", available_vram_gb=80.0)
    assert est_80gb["recommended_train_mode"] == "resident_bf16_train"

    code_pm8, out_pm8 = _run_cli_inproc(["plan-memory", "--model", "junkie-3b", "--vram-gb", "8"], prog="kcx")
    assert code_pm8 == 0
    assert "Inference Mode" in out_pm8
    assert "Training Mode" in out_pm8
    assert "streaming" in out_pm8

    installer_text = (REPO_ROOT / "studio" / "installer" / "installer.py").read_text(encoding="utf-8")
    assert ("yasho" + "4867") not in installer_text
