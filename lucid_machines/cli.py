"""Unified Command-Line Interface for Lucid Machines.

Unified CLI and backend engine powering terminal workflows, weight
transmutation, local Ollama execution, fine-tuning, and Polysom Studio.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

from lucid_machines.cli_ui import (
    ensure_utf8_stdio,
    is_tty,
    render_banner,
    render_help_overview,
    render_kv_panel,
    render_notice,
    render_rule,
    render_table,
    render_telemetry_bar,
    resolve_cli_prog,
    route_badge_ansi,
    status_spinner,
)

_FINETUNE_DEFAULTS: dict[str, Any] = {
    "dataset": "data/fineweb_sample.jsonl",
    "dataset_split": "train",
    "max_samples": 2000,
    "rank": 16,
    "alpha": 32.0,
    "epochs": 3,
    "lr": 2e-4,
    "batch_size": 4,
    "seq_len": 256,
    "grad_accum": 1,
    "output_dir": "checkpoints/adapters/lora_finetuned",
    "quantization": "4-bit NF4",
    "stream_chunk": 1,
    "wandb_project": "lucid_machines",
}


def resolve_default_checkpoint(
    explicit: str | Path | None = None,
    *,
    prefer: str = "active",
    search_dirs: tuple[str | Path, ...] = ("checkpoints", "checkpoints_full"),
    session_file: str | Path = "checkpoints/session_state.json",
) -> Path:
    """Resolve explicit checkpoint path, or auto-resolve active/legacy/newest .pt without importing torch."""
    if explicit is not None and str(explicit).strip() != "":
        cand = Path(explicit)
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"Checkpoint not found: {explicit}")

    sess_p = Path(session_file)
    if sess_p.is_file():
        try:
            data = json.loads(sess_p.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, ValueError):
            data = None
        if isinstance(data, dict):
            active = data.get("active_checkpoint")
            if isinstance(active, str) and active.strip():
                active_s = active.strip()
                raw_candidates: list[Path] = [Path(active_s)]
                for d in search_dirs:
                    raw_candidates.append(Path(d) / active_s)
                    raw_candidates.append(Path(d) / Path(active_s).name)
                seen: set[str] = set()
                for cand in raw_candidates:
                    key = str(cand)
                    if key in seen:
                        continue
                    seen.add(key)
                    if cand.is_file():
                        return cand

    if tuple(str(d) for d in search_dirs) == ("checkpoints", "checkpoints_full"):
        if prefer == "latest":
            legacy_paths = (
                Path("checkpoints_full/latest.pt"),
                Path("checkpoints_full/best.pt"),
                Path("checkpoints/latest.pt"),
                Path("checkpoints/best.pt"),
            )
        else:
            legacy_paths = (
                Path("checkpoints_full/best.pt"),
                Path("checkpoints_full/latest.pt"),
                Path("checkpoints/best.pt"),
                Path("checkpoints/latest.pt"),
            )
    else:
        names = ("latest.pt", "best.pt") if prefer == "latest" else ("best.pt", "latest.pt")
        legacy_paths = tuple(Path(d) / name for d in search_dirs for name in names)

    for leg in legacy_paths:
        if leg.is_file():
            return leg

    candidates = [f for d in search_dirs if Path(d).is_dir() for f in Path(d).glob("*.pt") if f.is_file()]
    if candidates:
        return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))

    raise FileNotFoundError(
        "No checkpoint found in checkpoints/ or checkpoints_full/. "
        "Pass --checkpoint <path> or --model <hf_or_ollama_id>."
    )


def _emit_json(payload: dict[str, Any] | list[Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _emit_json_error(
    command: str,
    message: str,
    *,
    code: int = 1,
    details: dict[str, Any] | None = None,
) -> NoReturn:
    payload: dict[str, Any] = {
        "ok": False,
        "status": "error",
        "command": command,
        "error": str(message),
        "exit_code": int(code),
    }
    if details:
        payload["details"] = details
    print(json.dumps(payload, indent=2, default=str))
    raise SystemExit(code)


def _cli_fail(
    message: str,
    *,
    code: int = 1,
    use_json: bool = False,
    command: str = "cli",
    extra: dict[str, Any] | None = None,
) -> NoReturn:
    if use_json:
        _emit_json_error(command, message, code=code, details=extra)
    render_notice(message, level="error")
    raise SystemExit(code)


@contextlib.contextmanager
def _quiet_json_stdout(enabled: bool):
    if not enabled:
        yield
        return
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        yield


def _maybe_spinner(message: str, quiet: bool):
    return contextlib.nullcontext() if quiet else status_spinner(message)


def _read_piped_stdin(*, explicit_stdin: bool = False) -> str | None:
    """Safely read piped stdin without tripping pytest's DontReadFromInput capture."""
    if sys.stdin is None:
        return None
    if explicit_stdin:
        try:
            return sys.stdin.read()
        except (OSError, ValueError):
            return None
    if type(sys.stdin).__name__ == "DontReadFromInput":
        return None
    try:
        if sys.stdin.isatty():
            return None
        return sys.stdin.read()
    except (OSError, ValueError):
        return None


def _sanitize_session_id(raw_id: str) -> str:
    raw = str(raw_id or "").strip()
    if not raw or ".." in raw or "/" in raw or "\\" in raw:
        raise ValueError(f"Invalid session ID: {raw_id!r}")
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw).strip("._-")
    if not safe:
        raise ValueError(f"Invalid session ID: {raw_id!r}")
    return safe


def _load_finetune_recipe(recipe_path: str) -> dict[str, Any]:
    path = Path(recipe_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuning recipe file not found: {recipe_path}")
    raw_text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(raw_text)
    else:
        data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise ValueError(f"Recipe file {recipe_path} must decode to a JSON/YAML mapping.")

    flat: dict[str, Any] = {}
    for k, v in data.items():
        if k not in ("model", "lora", "data", "training", "logging"):
            flat[k.replace("-", "_")] = v

    if isinstance(data.get("model"), str):
        flat["model"] = data["model"]
    elif isinstance(data.get("model"), dict):
        msec = data["model"]
        for alias in ("name", "model_name_or_path", "id", "model"):
            if msec.get(alias) is not None:
                flat["model"] = msec[alias]
                break
        for key in ("checkpoint", "quantization", "qlora", "disk_stream", "disk_cache", "quantize_lm_head"):
            if key in msec and msec[key] is not None:
                flat[key] = msec[key]

    if isinstance(data.get("lora"), dict):
        lsec = data["lora"]
        if lsec.get("rank", lsec.get("r")) is not None:
            flat["rank"] = lsec.get("rank", lsec.get("r"))
        if lsec.get("alpha", lsec.get("lora_alpha")) is not None:
            flat["alpha"] = lsec.get("alpha", lsec.get("lora_alpha"))

    if isinstance(data.get("data"), dict):
        dsec = data["data"]
        if dsec.get("dataset", dsec.get("path")) is not None:
            flat["dataset"] = dsec.get("dataset", dsec.get("path"))
        if dsec.get("dataset_split", dsec.get("split")) is not None:
            flat["dataset_split"] = dsec.get("dataset_split", dsec.get("split"))
        if dsec.get("max_samples") is not None:
            flat["max_samples"] = dsec["max_samples"]
        if dsec.get("seq_len") is not None:
            flat["seq_len"] = dsec["seq_len"]

    if isinstance(data.get("training"), dict):
        tsec = data["training"]
        key_map = {
            "epochs": "epochs",
            "lr": "lr",
            "learning_rate": "lr",
            "batch_size": "batch_size",
            "grad_accum": "grad_accum",
            "grad_accum_steps": "grad_accum",
            "max_steps": "max_steps",
            "output_dir": "output_dir",
            "checkpointing": "checkpointing",
            "gradient_checkpointing": "checkpointing",
            "stream": "stream",
            "stream_layers": "stream",
            "stream_chunk": "stream_chunk",
            "stream_chunk_size": "stream_chunk",
            "disk_stream": "disk_stream",
            "disk_cache": "disk_cache",
            "quantize_lm_head": "quantize_lm_head",
            "seq_len": "seq_len",
        }
        for src_k, dst_k in key_map.items():
            if src_k in tsec and tsec[src_k] is not None:
                flat[dst_k] = tsec[src_k]

    if isinstance(data.get("logging"), dict):
        lg = data["logging"]
        if lg.get("wandb", lg.get("enabled")) is not None:
            flat["wandb"] = bool(lg.get("wandb", lg.get("enabled")))
        if lg.get("wandb_project", lg.get("project")) is not None:
            flat["wandb_project"] = lg.get("wandb_project", lg.get("project"))
        if lg.get("wandb_run", lg.get("run_id", lg.get("run_name"))) is not None:
            flat["wandb_run"] = lg.get("wandb_run", lg.get("run_id", lg.get("run_name")))

    return flat


def _format_param_count(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def _get_subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) and name in action.choices:
            return action.choices[name]
    return parser


def build_cli_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Build the unified Lucid Machines CLI argument parser."""
    from lucid_machines import __version__

    cli_prog = prog or resolve_cli_prog()
    parser = argparse.ArgumentParser(
        prog=cli_prog,
        description="Polysom KCX (Lucid Machines CLI) — Triune 3-Tier Mixture-of-Experts, Weight Transmuter & Local Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available CLI Commands")

    # 1. Studio command
    studio_parser = subparsers.add_parser("studio", help="Launch Polysom Studio Native Desktop / Browser Application")
    studio_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    studio_parser.add_argument(
        "--browser", action="store_true", help="Launch directly in default browser instead of native desktop host"
    )
    studio_parser.add_argument(
        "--native-host", action="store_true", help="Start backend for native Windows desktop host"
    )
    studio_parser.add_argument("--use-wsl", action="store_true", help="Allow CUDA-enabled WSL backend on Windows")

    # 2. Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Launch embedded FastAPI server for Studio & OpenAI-compatible REST API"
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    serve_parser.add_argument("--concurrency", type=int, default=16, help="Maximum concurrent requests (default: 16)")
    serve_parser.add_argument("--api-key", default=None, help="Bearer token for remote API authentication")
    serve_parser.add_argument(
        "--model",
        default=None,
        help="Preload a Triune/Junkie preset, .pt/.safetensors checkpoint, HF model/dir, or ollama:tag at startup",
    )
    serve_parser.add_argument("--nvfp4", action="store_true", help="Quantize preloaded HF model to NVFP4")
    serve_parser.add_argument("--disk-stream", action="store_true", help="Stream preloaded NVFP4 shards from disk")

    # 3. Chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive terminal chat session with Triune, HF/transmuted models, or local Ollama"
    )
    chat_parser.add_argument("--model", default=None, help="HF model repo ID, local HF dir, NVFP4 cache, or ollama:tag")
    chat_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path (auto-resolves active/latest checkpoint when omitted)",
    )
    chat_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    chat_parser.add_argument(
        "--adapter", default=None, help="LoRA adapter directory or safetensors path to attach for chat"
    )
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    chat_parser.add_argument(
        "--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=256, help="Maximum generated tokens"
    )
    chat_parser.add_argument(
        "--speculative", action="store_true", help="Enable Reflex-to-Cortex self-speculative decoding"
    )
    chat_parser.add_argument(
        "--k-draft", type=int, default=3, help="Draft tokens per cycle for speculative decoding (default: 3)"
    )
    chat_parser.add_argument("--nvfp4", action="store_true", help="Quantize linear weights to Hadamard NVFP4")
    chat_parser.add_argument("--disk-stream", action="store_true", help="Stream NVFP4 layers directly from NVMe disk")
    chat_parser.add_argument("--disk-cache", default=None, help="NVFP4 disk cache directory")
    chat_parser.add_argument(
        "--transmute", action="store_true", help="Attach 3-tier Reflex/Limbic/Cortex early-exit heads to HF weights"
    )
    chat_parser.add_argument(
        "--reflex-ratio", type=float, default=0.25, help="Reflex exit ratio when --transmute is set"
    )
    chat_parser.add_argument(
        "--depth",
        default="auto",
        choices=("auto", "reflex", "limbic", "cortex"),
        help="Initial exit depth routing (default: auto)",
    )
    chat_parser.add_argument("--device", default=None, help="Target device (cuda / cpu)")

    # 4. Infer / Generate command
    infer_parser = subparsers.add_parser(
        "infer",
        aliases=["inference", "generate"],
        help="Run high-performance streaming inference with Triune, HF/transmuted models, or local Ollama",
    )
    infer_parser.add_argument("prompt_pos", nargs="?", default=None, help="Inference prompt text (positional)")
    infer_parser.add_argument("-p", "--prompt", default=None, help="Inference prompt text")
    infer_parser.add_argument("--stdin", action="store_true", help="Read inference prompt from standard input")
    infer_parser.add_argument("--system", default=None, help="Optional system prompt prepended to conversation")
    infer_parser.add_argument("--seed", type=int, default=None, help="Deterministic RNG seed for sampling")
    infer_parser.add_argument(
        "--model", default=None, help="HF model repo ID, local HF dir, NVFP4 cache, or ollama:tag"
    )
    infer_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path (auto-resolves active/latest checkpoint when omitted)",
    )
    infer_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    infer_parser.add_argument("--adapter", default=None, help="LoRA adapter directory or safetensors path (optional)")
    infer_parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=128,
        help="Maximum generated tokens (default: 128)",
    )
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    infer_parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering (default: 50)")
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")
    infer_parser.add_argument(
        "--depth",
        default="auto",
        choices=("auto", "reflex", "limbic", "cortex"),
        help="Exit depth routing (default: auto)",
    )
    infer_parser.add_argument("--nvfp4", action="store_true", help="Quantize linear weights to Hadamard NVFP4")
    infer_parser.add_argument("--disk-stream", action="store_true", help="Stream NVFP4 layers directly from NVMe disk")
    infer_parser.add_argument("--disk-cache", default=None, help="NVFP4 disk cache directory")
    infer_parser.add_argument(
        "--quantize-lm-head", action="store_true", help="Also pack lm_head into NVFP4 for tight VRAM"
    )
    infer_parser.add_argument(
        "--transmute", action="store_true", help="Attach 3-tier early-exit heads when loading HF/SafeTensors"
    )
    infer_parser.add_argument("--device", default=None, help="Target device (cuda / cpu)")
    infer_parser.add_argument("--no-stream", action="store_true", help="Disable live token streaming")
    infer_parser.add_argument("--json", action="store_true", help="Emit structured JSON response and telemetry")

    # 4b. Scan local models command
    scan_parser = subparsers.add_parser(
        "scan",
        aliases=["discover"],
        help="Scan this PC for live Ollama, Hugging Face, vLLM, Hermes, LM Studio, NVFP4 & checkpoint models",
    )
    scan_parser.add_argument(
        "--source", default=None, help="Filter results by source (e.g. ollama, huggingface, checkpoint)"
    )
    scan_parser.add_argument(
        "--path", action="append", default=None, help="Additional folder path to scan (repeatable)"
    )
    scan_parser.add_argument("--no-wsl", action="store_true", help="Skip scanning WSL filesystem model caches")
    scan_parser.add_argument("--json", action="store_true", help="Emit JSON scan results without Rich tables")

    # 4c. Real Weight Transmutation command
    transmute_parser = subparsers.add_parser(
        "transmute",
        help="Convert a Hugging Face / SafeTensors model into 3-tier Triune with real weight transfer",
    )
    transmute_parser.add_argument("--model", required=True, help="HF repo ID, local HF dir, or NVFP4 disk cache")
    transmute_parser.add_argument(
        "--mode",
        default="early-exit",
        choices=("early-exit", "native-distill"),
        help="early-exit (zero-loss 3-tier heads) or native-distill (surgical GLA+MoE transfer)",
    )
    transmute_parser.add_argument("--reflex-ratio", type=float, default=0.25, help="Reflex exit layer fraction")
    transmute_parser.add_argument("--experts", type=int, default=8, help="MoE expert count for native-distill mode")
    transmute_parser.add_argument("--rank", type=int, default=64, help="Early-exit low-rank projection dimension")
    transmute_parser.add_argument("--nvfp4", action="store_true", help="Quantize linear blocks to NVFP4")
    transmute_parser.add_argument("--disk-stream", action="store_true", help="Stream NVFP4 shards directly from disk")
    transmute_parser.add_argument("--disk-cache", default=None, help="NVFP4 disk cache directory")
    transmute_parser.add_argument(
        "-p", "--prompt", default=None, help="Optional test prompt to generate after transmuting"
    )
    transmute_parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens when --prompt is given")
    transmute_parser.add_argument("--device", default=None, help="Target device (cuda / cpu)")

    # 5. Fine-Tuning command
    finetune_parser = subparsers.add_parser(
        "finetune", help="Fine-tune Triune or Hugging Face models with LoRA / NVFP4 QLoRA & layer streaming"
    )
    finetune_parser.add_argument(
        "--model", default=None, help="HF model repo ID, local HF dir, NVFP4 disk cache, or Triune preset"
    )
    finetune_parser.add_argument("--recipe", default=None, help="Path to a YAML or JSON fine-tuning recipe file")
    finetune_parser.add_argument(
        "--dataset", default=None, help="Dataset path or HF dataset ID (default: data/fineweb_sample.jsonl)"
    )
    finetune_parser.add_argument(
        "--dataset-split", default=None, help="Dataset split when using HF Hub datasets (default: train)"
    )
    finetune_parser.add_argument(
        "--max-samples", type=int, default=None, help="Max streaming samples from HF dataset (default: 2000)"
    )
    finetune_parser.add_argument("--checkpoint", default=None, help="Base model checkpoint path (optional)")
    finetune_parser.add_argument("--rank", type=int, default=None, help="LoRA rank dimension r (default: 16)")
    finetune_parser.add_argument("--alpha", type=float, default=None, help="LoRA alpha scaling (default: 32.0)")
    finetune_parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: 3)")
    finetune_parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 2e-4)")
    finetune_parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch size per step (default: 4)")
    finetune_parser.add_argument("--seq-len", type=int, default=None, help="Sequence length (default: 256)")
    finetune_parser.add_argument(
        "--grad-accum", type=int, default=None, help="Gradient accumulation steps (default: 1)"
    )
    finetune_parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (optional)")
    finetune_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for adapter weights (default: checkpoints/adapters/lora_finetuned)",
    )
    finetune_parser.add_argument(
        "--quantization", default=None, help="Quantization precision (e.g. '4-bit NVFP4', 'BF16', default: 4-bit NF4)"
    )
    finetune_parser.add_argument("--qlora", action="store_true", help="Enable true packed 4-bit NVFP4 base QLoRA")
    finetune_parser.add_argument("--stream", action="store_true", help="Enable RAM-to-VRAM LayerStreamingEngine")
    finetune_parser.add_argument(
        "--stream-chunk", type=int, default=None, help="Layer chunk size for LayerStreamingEngine (default: 1)"
    )
    finetune_parser.add_argument(
        "--disk-stream", action="store_true", help="Enable NVMe NVFP4 shard streaming (14B-27B+)"
    )
    finetune_parser.add_argument("--disk-cache", default=None, help="NVFP4 disk cache directory")
    finetune_parser.add_argument(
        "--quantize-lm-head", action="store_true", help="Pack lm_head into NVFP4 in disk cache"
    )
    finetune_parser.add_argument("--checkpointing", action="store_true", help="Enable activation checkpointing")
    finetune_parser.add_argument("--wandb", action="store_true", help="Log fine-tuning metrics to Weights & Biases")
    finetune_parser.add_argument(
        "--wandb-project", default=None, help="Weights & Biases project name (default: lucid_machines)"
    )
    finetune_parser.add_argument("--wandb-run", default=None, help="Weights & Biases run ID or name")

    # 6. Adapter Management command
    adapter_parser = subparsers.add_parser("adapter", help="Manage LoRA/QLoRA adapters (list, merge, export)")
    adapter_sub = adapter_parser.add_subparsers(dest="adapter_action", help="Adapter action")

    adapter_sub.add_parser("list", help="List all discovered adapters on disk")

    merge_parser = adapter_sub.add_parser("merge", help="Merge LoRA adapter permanently into a base checkpoint")
    merge_parser.add_argument("--adapter", required=True, help="Adapter directory or safetensors path")
    merge_parser.add_argument("--checkpoint", required=True, help="Base model checkpoint path")
    merge_parser.add_argument("--output", default=None, help="Output path for merged checkpoint")
    merge_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")

    export_adapter_parser = adapter_sub.add_parser("export", help="Export LoRA adapter weights as standalone file")
    export_adapter_parser.add_argument("--adapter", required=True, help="Adapter path to export")
    export_adapter_parser.add_argument("--output", default=None, help="Destination export path")
    export_adapter_parser.add_argument(
        "--format", default="safetensors", choices=("safetensors", "pt"), help="Export format"
    )

    # 7. Checkpoint Management command
    ckpt_parser = subparsers.add_parser("checkpoint", help="Manage model checkpoints (list, export)")
    ckpt_sub = ckpt_parser.add_subparsers(dest="ckpt_action", help="Checkpoint action")

    ckpt_list_parser = ckpt_sub.add_parser("list", help="List local checkpoints")
    ckpt_list_parser.add_argument("--json", action="store_true", help="Emit JSON list of checkpoints")

    ckpt_export_parser = ckpt_sub.add_parser("export", help="Export checkpoint to SafeTensors, ONNX, HF, or vLLM")
    ckpt_export_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Source checkpoint path (auto-resolves active/latest checkpoint when omitted)",
    )
    ckpt_export_parser.add_argument("--output", default=None, help="Destination file or bundle path")
    ckpt_export_parser.add_argument(
        "--format",
        default="safetensors",
        choices=("safetensors", "gguf", "onnx", "hf", "vllm"),
        help="Target format",
    )

    # 8. Top-Level Export / Convert command
    export_parser = subparsers.add_parser(
        "export",
        aliases=["convert"],
        help="Export model checkpoint directly to SafeTensors, ONNX, Hugging Face, or vLLM bundle format",
    )
    export_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Source checkpoint path (auto-resolves active/latest checkpoint when omitted)",
    )
    export_parser.add_argument("--output", default=None, help="Destination output path")
    export_parser.add_argument(
        "--format",
        default="safetensors",
        choices=("safetensors", "gguf", "onnx", "hf", "vllm"),
        help="Target format",
    )

    # 9. Model Zoo command
    models_parser = subparsers.add_parser("models", help="Display architecture presets from the Triune Model Zoo")
    models_parser.add_argument("--json", action="store_true", help="Emit JSON model catalog")

    # 10. Train command
    train_parser = subparsers.add_parser("train", help="Train Triune Transformer models (pre-training or fine-tuning)")
    train_parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passed to training script")

    # 11. Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark", aliases=["bench"], help="Run inference or comprehensive optimization benchmarks"
    )
    bench_parser.add_argument(
        "--suite",
        default=None,
        choices=("all", "kernels", "optimizers", "streaming", "context", "routing", "matrix", "mqar"),
        help="Optimization benchmark suite",
    )
    bench_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (auto-resolves active/latest checkpoint when omitted)",
    )
    bench_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    bench_parser.add_argument("--prompt", default="The meaning of life is", help="Prompt text")
    bench_parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    bench_parser.add_argument("--device", default=None, help="Device (cuda / cpu)")
    bench_parser.add_argument(
        "--depth", default="auto", choices=("auto", "reflex", "limbic", "cortex"), help="Exit depth routing"
    )
    bench_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    bench_parser.add_argument("--quick", action="store_true", help="Run reduced sequence/trial budget for fast checks")
    bench_parser.add_argument(
        "--seq-lens", default=None, help="Comma-separated sequence lengths for --suite mqar (e.g. 128,256)"
    )
    bench_parser.add_argument("--json", action="store_true", help="Emit JSON benchmark results")

    # 12. GPU check command
    gpu_parser = subparsers.add_parser("gpucheck", help="Print CUDA hardware and driver diagnostics")
    gpu_parser.add_argument("--json", action="store_true", help="Emit JSON GPU diagnostics")

    # 13. Memory Plan command
    mem_parser = subparsers.add_parser("plan-memory", help="Estimate VRAM/RAM budget for Triune or Hugging Face models")
    mem_parser.add_argument("--vram-gb", type=float, default=None, help="Target VRAM in GB (default: auto-detect)")
    mem_parser.add_argument("--model", default=None, help="Optional HF model ID or local HF directory to plan for")

    # 14. Cache NVFP4 shards for NVMe streaming
    cache_parser = subparsers.add_parser(
        "cache-nvfp4", help="Pack an HF Safetensors model layer-by-layer into an NVMe NVFP4 cache (~0.625 B/param)"
    )
    cache_parser.add_argument("--model", required=True, help="HF repo ID or local model directory with safetensors")
    cache_parser.add_argument(
        "--out", "--cache-dir", "--output", dest="out", default=None, help="Output cache directory"
    )
    cache_parser.add_argument("--no-hadamard", action="store_true", help="Disable FWHT-16 outlier rotation")
    cache_parser.add_argument("--quantize-lm-head", action="store_true", help="Also pack lm_head into NVFP4")
    cache_parser.add_argument("--force", action="store_true", help="Rebuild cache even if manifest exists")

    # 15. Install / Setup command
    install_parser = subparsers.add_parser(
        "install", help="Launch Polysom Studio Setup Wizard or automated environment provisioner"
    )
    install_parser.add_argument(
        "--cuda",
        default="auto",
        choices=("auto", "13.2", "13.0", "12.6", "12.4", "12.1", "11.8", "cpu"),
        help="CUDA wheel target",
    )
    install_parser.add_argument("--verify", action="store_true", help="Verify existing installation health only")
    install_parser.add_argument("--skip-torch", action="store_true", help="Skip PyTorch wheel installation")
    install_parser.add_argument("--no-launch", action="store_true", help="Do not launch Studio after installation")
    install_parser.add_argument("-y", "--yes", action="store_true", help="Run unattended without GUI prompt")

    # 16. Doctor / Info command
    doctor_parser = subparsers.add_parser(
        "doctor",
        aliases=["info"],
        help="Run comprehensive end-to-end diagnostics on GPU, C/AVX2 engine, optimizers, Ollama & Studio backend",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Emit JSON diagnostics payload")

    # 17. Build Native Apps command
    native_parser = subparsers.add_parser(
        "build-native",
        help="Build native desktop apps & installers for Windows (.exe), Linux (AppDir), and macOS (.app)",
    )
    native_parser.add_argument(
        "--platform",
        default="all",
        choices=("all", "windows", "linux", "macos"),
        help="Target OS platform (default: all)",
    )

    # 18. Profile command
    prof_parser = subparsers.add_parser("profile", help="Profile Triune training steps")
    prof_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (auto-resolves latest/active checkpoint when omitted)",
    )
    prof_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    prof_parser.add_argument("--output", default="profiler_output.txt", help="Profiler output report path")
    prof_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    prof_parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    prof_parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps (default: 5)")
    prof_parser.add_argument(
        "--active-steps", type=int, default=10, help="Number of active profiling steps (default: 10)"
    )

    # 19. Tokenizer command
    tok_parser = subparsers.add_parser("tokenizer", help="Build or train a BPE tokenizer")
    tok_parser.add_argument("--output", default="triune_tokenizer.json", help="Output tokenizer JSON path")
    tok_parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")

    # 20. Self-Learning Agent Harness command
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run the self-learning cognitive Agent Harness (skills, dynamic personas, user memory, tool mastery)",
    )
    agent_parser.add_argument("goal_pos", nargs="?", default=None, help="Optional single-shot autonomous goal")
    agent_parser.add_argument("-g", "--goal", default=None, help="Run one autonomous multi-step goal and exit")
    agent_parser.add_argument(
        "--model",
        default=None,
        help="Agent model backend (e.g. ollama:gemma3:4b, gpt-4o-mini, claude-3-5-sonnet-latest)",
    )
    agent_parser.add_argument(
        "--persona",
        default="auto",
        help="Initial persona (auto, architect, engineer, researcher, debugger, optimizer, tutor, sentinel, or custom)",
    )
    agent_parser.add_argument(
        "--depth",
        default="auto",
        choices=("auto", "reflex", "limbic", "cortex"),
        help="Reasoning depth tier (default: auto)",
    )
    agent_parser.add_argument(
        "--max-steps", type=int, default=10, help="Max reasoning/tool steps per goal (default: 10)"
    )
    agent_parser.add_argument("--allow-shell", action="store_true", help="Permit workspace shell execution tool")
    agent_parser.add_argument("--read-only", action="store_true", help="Disable workspace write/edit file tools")
    agent_parser.add_argument(
        "--store-dir", default="checkpoints/agent_harness", help="Persistent memory/skills directory"
    )
    agent_parser.add_argument(
        "--no-auto-skill", action="store_true", help="Disable automatic post-task skill synthesis"
    )
    agent_parser.add_argument(
        "--team",
        default=None,
        help="Comma-separated specialist personas for multi-agent orchestration (e.g. architect,engineer,sentinel)",
    )
    agent_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run --team specialists concurrently via MultiAgentOrchestrator.execute_parallel",
    )
    agent_parser.add_argument(
        "--max-workers", type=int, default=4, help="Max concurrent workers for --parallel team orchestration"
    )
    agent_parser.add_argument(
        "--workspace", default=".", help="Workspace root directory for agent file and terminal tools (default: .)"
    )
    agent_parser.add_argument(
        "--mode",
        default="react",
        choices=("react", "pev", "pipeline"),
        help="Agent execution loop mode: react, pev (plan-execute-verify), or pipeline (multi-agent)",
    )
    agent_parser.add_argument(
        "--output-format",
        default="text",
        choices=("text", "json", "stream-json"),
        help="Output format: text (Rich), json (single object), or stream-json (NDJSON events)",
    )
    agent_parser.add_argument(
        "--resume",
        default=None,
        dest="resume_session",
        help="Resume an agent conversation session by ID from <store_dir>/sessions/",
    )
    agent_parser.add_argument(
        "--session-id",
        default=None,
        help="Explicit session ID to save conversation history under",
    )
    agent_parser.add_argument(
        "--allowed-tools",
        default=None,
        help="Comma-separated allowlist of permitted agent tools (e.g. read_file,search_code,list_dir)",
    )
    agent_parser.add_argument("--stdin", action="store_true", help="Read autonomous goal from standard input")

    # 21. Evaluate command
    eval_parser = subparsers.add_parser(
        "eval",
        aliases=["evaluate"],
        help="Evaluate a Triune checkpoint on a validation dataset, MQAR/zero-shot tasks, and optional sample prompt",
    )
    eval_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path (auto-resolves active/latest checkpoint when omitted)",
    )
    eval_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer JSON path")
    eval_parser.add_argument("--dataset", default=None, help="Validation dataset name or local file path")
    eval_parser.add_argument("--dataset-config", default=None, help="Dataset config/subset name")
    eval_parser.add_argument("-p", "--prompt", default=None, help="Optional sample prompt to generate after eval")
    eval_parser.add_argument(
        "--max-steps",
        "--eval-batches",
        dest="max_steps",
        type=int,
        default=5,
        help="Max evaluation batches (default: 5)",
    )
    eval_parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens when --prompt is given")
    eval_parser.add_argument(
        "--depth",
        default="auto",
        choices=("auto", "reflex", "limbic", "cortex"),
        help="Exit depth routing during evaluation (default: auto)",
    )
    eval_parser.add_argument("--device", default=None, help="Target device (cuda / cpu)")
    eval_parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated evaluation tasks (e.g. mqar,zeroshot,arc_probe,hellaswag_probe,mmlu_probe or JSONL path)",
    )
    eval_parser.add_argument(
        "--num-trials", type=int, default=16, help="Number of trials for MQAR task evaluation (default: 16)"
    )
    eval_parser.add_argument("--limit", type=int, default=None, help="Optional max examples per zero-shot task")
    eval_parser.add_argument("--json", action="store_true", help="Emit JSON evaluation metrics")

    # 22. Config management command
    config_parser = subparsers.add_parser(
        "config",
        help="List, inspect, and validate Triune/Junkie model configs and session state",
    )
    config_sub = config_parser.add_subparsers(dest="config_action", help="Config action")
    config_sub.add_parser("list", help="List available model configs in models/configs/ and canonical presets")
    config_show = config_sub.add_parser("show", help="Inspect a model config, preset, or session_state.json")
    config_show.add_argument("target", nargs="?", default=None, help="Preset name, 'session', or JSON path")
    config_show.add_argument("--file", default=None, help="Explicit JSON config file path")
    config_show.add_argument("--json", action="store_true", help="Print raw JSON output")
    config_val = config_sub.add_parser("validate", help="Validate model config(s) against ModelSpec & build_config")
    config_val.add_argument("target", nargs="?", default=None, help="Preset name or JSON path to validate")
    config_val.add_argument("--all", action="store_true", help="Validate all models/configs/*.json presets")

    # 23. Workspace operations command
    ws_parser = subparsers.add_parser(
        "workspace",
        help="Inspect status, file tree, read files, or initialize a project workspace",
    )
    ws_sub = ws_parser.add_subparsers(dest="ws_action", help="Workspace action")
    ws_status = ws_sub.add_parser("status", help="Show workspace root, git status, and checkpoint/config summary")
    ws_status.add_argument("--path", default=".", help="Workspace root directory (default: .)")
    ws_tree = ws_sub.add_parser("tree", help="Display workspace directory tree")
    ws_tree.add_argument("--path", default=".", help="Workspace root directory (default: .)")
    ws_tree.add_argument("--depth", type=int, default=2, help="Max directory depth (default: 2)")
    ws_read = ws_sub.add_parser("read", help="Read a text file safely confined inside the workspace")
    ws_read.add_argument("file_pos", nargs="?", default=None, help="Relative file path inside workspace")
    ws_read.add_argument("--file", default=None, help="Relative file path inside workspace")
    ws_read.add_argument("--path", default=".", help="Workspace root directory (default: .)")
    ws_read.add_argument("--start", type=int, default=1, help="Start line (1-indexed)")
    ws_read.add_argument("--end", type=int, default=200, help="End line (inclusive)")
    ws_init = ws_sub.add_parser(
        "init", help="Initialize standard workspace directories safely without overwriting files"
    )
    ws_init.add_argument("--path", default=".", help="Workspace root directory (default: .)")

    return parser


def main() -> None:
    """Dispatch CLI commands with Rich formatting and unified engine orchestration."""
    ensure_utf8_stdio()
    argv = sys.argv[1:]
    prog = resolve_cli_prog()
    prog_stem = Path(sys.argv[0]).stem.lower() if sys.argv else ""
    if prog_stem in ("polysom", "polysom-script") and (
        not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help", "-V", "--version"))
    ):
        argv = ["studio", *argv]
    elif not argv or argv[0] in ("-h", "--help"):
        render_help_overview(prog=prog)
        return
    elif argv[0] in ("-V", "--version"):
        from lucid_machines import __version__

        print(f"{prog} {__version__}")
        return

    if argv[0] == "train":
        from scripts.train import main as train_main

        train_argv = argv[1:]
        sys.argv = [f"{prog} train", *train_argv]
        train_main(train_argv)
        return

    parser = build_cli_parser(prog=prog)
    args = parser.parse_args(argv)

    if args.command == "studio":
        from lucid_machines.desktop import launch_desktop_app

        mode_name = "Browser UI" if args.browser else ("Native Host Backend" if args.native_host else "Desktop IDE")
        render_banner("POLYSOM STUDIO", subtitle=f"Launching {mode_name}", badge=f"port {args.port}")
        launch_desktop_app(
            port=args.port,
            use_browser=args.browser,
            native_host=getattr(args, "native_host", False),
            allow_wsl=getattr(args, "use_wsl", False),
        )

    elif args.command == "serve":
        import os

        if args.api_key:
            os.environ["LUCID_MACHINES_API_TOKEN"] = args.api_key

        preload_model = getattr(args, "model", None)
        if preload_model:
            raw_m = str(preload_model).strip()
            m_low = raw_m.lower()
            m_path = Path(raw_m)

            if (
                "lucid_machines.api.routes" not in sys.modules
                and not m_low.startswith("ollama:")
                and not m_path.exists()
            ):
                from lucid_machines.model.base import _LEGACY_PRESET_ALIASES, CANONICAL_MODEL_SPECS

                cfg_candidate = Path(__file__).resolve().parents[1] / "models" / "configs" / f"{raw_m}.json"
                if (
                    m_low not in CANONICAL_MODEL_SPECS
                    and m_low not in _LEGACY_PRESET_ALIASES
                    and not cfg_candidate.is_file()
                ):
                    if m_path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".gguf"}:
                        _cli_fail(f"Failed to preload model '{preload_model}': file not found.", code=1)
                    try:
                        from huggingface_hub import HfApi, try_to_load_from_cache

                        cached_cfg = try_to_load_from_cache(repo_id=raw_m, filename="config.json")
                        if not isinstance(cached_cfg, str):
                            if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
                                raise FileNotFoundError(
                                    f"'{raw_m}' not found locally and HF_HUB_OFFLINE is enabled."
                                )
                            HfApi().model_info(raw_m, timeout=3.0)
                    except Exception as exc:
                        _cli_fail(f"Failed to preload model '{preload_model}': {exc}", code=1)

            from lucid_machines.api.routes import pytorch_state
            from lucid_machines.model.base import _LEGACY_PRESET_ALIASES, CANONICAL_MODEL_SPECS

            try:
                with status_spinner(f"Preloading model '{raw_m}'..."):
                    if m_low.startswith("ollama:"):
                        pytorch_state.load_ollama_model(raw_m)
                    elif m_path.is_file() and (
                        m_path.suffix.lower() in {".pt", ".pth", ".ckpt"}
                        or (
                            m_path.suffix.lower() == ".safetensors"
                            and not (m_path.parent / "config.json").is_file()
                            and not (m_path.parent / "manifest.json").is_file()
                        )
                    ):
                        pytorch_state.load_model_checkpoint(m_path)
                    elif m_low in CANONICAL_MODEL_SPECS or m_low in _LEGACY_PRESET_ALIASES:
                        pytorch_state.reconfigure_architecture(m_low, instantiate_model=True)
                    else:
                        target_src = (
                            str(m_path.parent)
                            if (m_path.is_file() and (m_path.parent / "config.json").is_file())
                            else raw_m
                        )
                        pytorch_state.load_hf_model(
                            target_src,
                            nvfp4=bool(getattr(args, "nvfp4", False)),
                            disk_stream=bool(getattr(args, "disk_stream", False)),
                        )
            except Exception as exc:
                _cli_fail(f"Failed to preload model '{preload_model}': {exc}", code=1)

        from lucid_machines.api import run_server

        render_banner(
            "LUCID MACHINES API SERVER",
            subtitle="OpenAI-Compatible & Polysom Studio Backend",
            badge=f"http://{args.host}:{args.port}",
        )
        serve_rows: list[tuple[str, Any]] = [
            ("Endpoint URL", f"http://{args.host}:{args.port}"),
            ("OpenAI Base", f"http://{args.host}:{args.port}/v1"),
            ("API Docs", f"http://{args.host}:{args.port}/docs"),
            ("Concurrency", args.concurrency),
            ("Bearer Auth", "Enabled" if args.api_key else "Loopback-only (default)"),
        ]
        if preload_model:
            serve_rows.append(("Preloaded Model", preload_model))
        render_kv_panel("Server Configuration", serve_rows, border_style="cyan")
        run_server(host=args.host, port=args.port)

    elif args.command == "chat":
        from scripts.chat import main as chat_main

        chat_args: list[str] = []
        if not getattr(args, "model", None):
            try:
                ckpt_path = resolve_default_checkpoint(args.checkpoint)
            except FileNotFoundError as exc:
                _cli_fail(str(exc), code=1)
            chat_args.extend(["--checkpoint", str(ckpt_path)])
        else:
            chat_args.extend(["--model", args.model])
            if args.checkpoint:
                chat_args.extend(["--checkpoint", str(args.checkpoint)])

        chat_args.extend(
            [
                "--tokenizer_path",
                args.tokenizer,
                "--temperature",
                str(args.temperature),
                "--repetition_penalty",
                str(args.repetition_penalty),
                "--max_new_tokens",
                str(args.max_tokens),
                "--depth",
                args.depth,
                "--reflex-ratio",
                str(getattr(args, "reflex_ratio", 0.25)),
            ]
        )
        if args.adapter:
            chat_args.extend(["--adapter", args.adapter])
        if getattr(args, "speculative", False):
            chat_args.append("--speculative")
        if getattr(args, "k_draft", None):
            chat_args.extend(["--k-draft", str(args.k_draft)])
        if getattr(args, "nvfp4", False):
            chat_args.append("--nvfp4")
        if getattr(args, "disk_stream", False):
            chat_args.append("--disk-stream")
        if getattr(args, "disk_cache", None):
            chat_args.extend(["--disk-cache", args.disk_cache])
        if getattr(args, "transmute", False):
            chat_args.append("--transmute")
        if getattr(args, "device", None):
            chat_args.extend(["--device", args.device])
        chat_main(chat_args)

    elif args.command in ("infer", "inference", "generate"):
        use_json = bool(getattr(args, "json", False))
        prompt = args.prompt or args.prompt_pos
        if prompt is None or prompt == "-" or getattr(args, "stdin", False):
            piped = _read_piped_stdin(explicit_stdin=bool(getattr(args, "stdin", False)))
            if piped is not None:
                prompt = piped.strip()
            elif prompt == "-":
                prompt = None
        if not prompt or not str(prompt).strip():
            _cli_fail(
                "Please provide a prompt via positional argument, --prompt, or piped stdin.",
                code=2,
                use_json=use_json,
                command="infer",
            )
        prompt = str(prompt).strip()

        if getattr(args, "model", None):
            source = str(args.model)
            resolved_ckpt: Path | None = None
        else:
            try:
                resolved_ckpt = resolve_default_checkpoint(args.checkpoint)
            except FileNotFoundError as exc:
                _cli_fail(str(exc), code=1, use_json=use_json, command="infer")
            source = str(resolved_ckpt)

        is_ollama = bool(args.model and str(args.model).strip().lower().startswith("ollama:"))
        system_prompt = getattr(args, "system", None)
        seed_val = getattr(args, "seed", None)

        if not use_json:
            render_banner("TRIUNE INFERENCE ENGINE", subtitle=str(source), badge=args.depth.upper())
            cfg_rows: list[tuple[str, Any]] = [
                ("Source", source),
                ("Route", "OLLAMA LOCAL" if is_ollama else args.depth.upper()),
                ("Sampling", f"temp={args.temperature}  ·  top_k={args.top_k}  ·  max_tokens={args.max_tokens}"),
            ]
            if system_prompt:
                cfg_rows.append(("System", system_prompt[:80]))
            if seed_val is not None:
                cfg_rows.append(("Seed", seed_val))
            if args.adapter:
                cfg_rows.append(("LoRA Adapter", args.adapter))
            render_kv_panel("Run Configuration", cfg_rows, border_style="bright_black")

        if is_ollama:
            from lucid_machines.runtime.model_scanner import (
                _is_tag_live_in_ollama,
                query_ollama_running_tags,
                stream_ollama_chat,
            )

            if not _is_tag_live_in_ollama(source, query_ollama_running_tags()):
                _cli_fail(
                    f"'{source}' is not live in Ollama. Start it via `ollama run {str(source).removeprefix('ollama:')}`.",
                    code=1,
                    use_json=use_json,
                    command="infer",
                )

            ol_messages: list[dict[str, str]] = []
            if system_prompt:
                ol_messages.append({"role": "system", "content": system_prompt})
            ol_messages.append({"role": "user", "content": prompt})

            if not use_json:
                print(f"\n  Prompt: {prompt}")
                sys.stdout.write(f"  {route_badge_ansi('OLLAMA')} ")
                sys.stdout.flush()
            try:
                with _quiet_json_stdout(use_json):
                    ol_res = stream_ollama_chat(
                        source,
                        ol_messages,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        on_text=(
                            (lambda d: (sys.stdout.write(d), sys.stdout.flush()))
                            if (not args.no_stream and not use_json)
                            else None
                        ),
                    )
            except Exception as exc:
                _cli_fail(f"Ollama inference failed: {exc}", code=1, use_json=use_json, command="infer")

            if use_json:
                _emit_json(
                    {
                        "status": "ok",
                        "source": source,
                        "route": "OLLAMA",
                        "text": ol_res.get("text", ""),
                        "tokens_count": int(ol_res.get("tokens_count", 0)),
                        "tokens_per_sec": float(ol_res.get("tokens_per_sec", 0.0)),
                        "ttft_ms": ol_res.get("ttft_ms"),
                        "latency_ms": float(ol_res.get("latency_ms", 0.0)),
                        "vram_gb": 0.0,
                        "load_seconds": 0.0,
                        "seed": seed_val,
                    }
                )
                return

            if args.no_stream:
                print(ol_res["text"])
            else:
                print()
            render_telemetry_bar(
                "OLLAMA",
                tok_sec=float(ol_res.get("tokens_per_sec", 0.0)),
                ttft_ms=ol_res.get("ttft_ms"),
                tokens=int(ol_res.get("tokens_count", 0)),
                latency_ms=float(ol_res.get("latency_ms", 0.0)),
                vram_gb=0.0,
            )
            return

        import random

        import torch

        from lucid_machines.inference import generate_response, load_checkpoint_model, load_model_for_inference

        if seed_val is not None:
            random.seed(int(seed_val))
            torch.manual_seed(int(seed_val))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed_val))

        depth_map = {"auto": None, "reflex": 0, "limbic": 1, "cortex": 2}
        force_depth = depth_map.get(args.depth)
        load_t0 = time.perf_counter()
        try:
            with _quiet_json_stdout(use_json), _maybe_spinner(f"Loading weights for {source}...", use_json):
                if getattr(args, "transmute", False) and (getattr(args, "model", None) or Path(source).is_dir()):
                    from lucid_machines.model.transmuter import transmute_hf_to_triune

                    model, tokenizer, _, device, pad_id, eos_id = transmute_hf_to_triune(
                        source,
                        quantize="nvfp4" if getattr(args, "nvfp4", False) else None,
                        disk_stream=getattr(args, "disk_stream", False),
                        disk_cache_dir=getattr(args, "disk_cache", None),
                        device=args.device,
                        tokenizer_path=args.tokenizer,
                    )
                else:
                    use_unified = bool(
                        getattr(args, "model", None)
                        or getattr(args, "nvfp4", False)
                        or getattr(args, "disk_stream", False)
                        or Path(source).is_dir()
                    )
                    if use_unified:
                        model, tokenizer, _, device, pad_id, eos_id = load_model_for_inference(
                            source,
                            args.tokenizer,
                            adapter_path=args.adapter,
                            device=args.device,
                            quantize="nvfp4" if getattr(args, "nvfp4", False) else None,
                            disk_stream=getattr(args, "disk_stream", False),
                            disk_cache_dir=getattr(args, "disk_cache", None),
                            quantize_lm_head=getattr(args, "quantize_lm_head", False),
                        )
                    else:
                        model, tokenizer, _, device, pad_id, eos_id = load_checkpoint_model(
                            resolved_ckpt or source,
                            args.tokenizer,
                            adapter_path=args.adapter,
                            device=args.device,
                        )
        except Exception as exc:
            _cli_fail(f"Failed to load model '{source}': {exc}", code=1, use_json=use_json, command="infer")
        load_time = time.perf_counter() - load_t0

        stream = (not args.no_stream) and (not use_json)
        if not use_json:
            print(f"\n  Prompt: {prompt}")
            sys.stdout.write(f"  {route_badge_ansi(args.depth)} ")
            sys.stdout.flush()

        last_route = [args.depth.upper()]
        printed_len = [0]

        def _on_text(cum_text, **kwargs):
            if kwargs.get("route"):
                last_route[0] = kwargs["route"]
            new_text = str(cum_text)[printed_len[0] :]
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                printed_len[0] = len(str(cum_text))

        try:
            with _quiet_json_stdout(use_json):
                res = generate_response(
                    model,
                    tokenizer,
                    prompt,
                    device=device,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    force_depth=force_depth,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_tokens,
                    system_prompt=system_prompt,
                    seed=seed_val,
                    on_text=_on_text if stream else None,
                    return_info=True,
                )
        except Exception as exc:
            _cli_fail(f"Inference failed: {exc}", code=1, use_json=use_json, command="infer")
        finally:
            if getattr(model, "_streaming_engine", None) is not None:
                model._streaming_engine.detach()
            if getattr(model, "_exit_manager", None) is not None and hasattr(model._exit_manager, "remove_hooks"):
                model._exit_manager.remove_hooks()

        route_used = res.get("route", last_route[0])
        vram_gb = round(torch.cuda.memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
        if use_json:
            _emit_json(
                {
                    "status": "ok",
                    "source": source,
                    "route": str(route_used),
                    "text": res.get("text", ""),
                    "tokens_count": int(res.get("tokens_count", 0)),
                    "tokens_per_sec": round(float(res.get("tokens_per_sec", 0.0)), 3),
                    "ttft_ms": res.get("ttft_ms"),
                    "latency_ms": float(res.get("latency_ms", 0.0)),
                    "vram_gb": vram_gb,
                    "load_seconds": round(load_time, 3),
                    "seed": seed_val,
                }
            )
            return

        if not stream:
            print(res["text"])
        else:
            print()

        render_telemetry_bar(
            str(route_used),
            tok_sec=float(res.get("tokens_per_sec", 0.0)),
            ttft_ms=res.get("ttft_ms"),
            tokens=int(res.get("tokens_count", 0)),
            latency_ms=float(res.get("latency_ms", 0.0)),
            vram_gb=vram_gb,
            load_s=load_time,
        )

    elif args.command in ("scan", "discover"):
        from lucid_machines.runtime.model_scanner import scan_pc_models

        use_json = bool(getattr(args, "json", False))
        if not use_json:
            render_banner("PC MODEL SCANNER", subtitle="Ollama · Hugging Face · vLLM · Hermes · Checkpoints")
        with _quiet_json_stdout(use_json), _maybe_spinner("Scanning local runtimes and model caches...", use_json):
            scan_data = scan_pc_models(
                extra_paths=getattr(args, "path", None),
                include_wsl=not getattr(args, "no_wsl", False),
            )
        models_list = list(scan_data.get("models") or [])
        if getattr(args, "source", None):
            flt = str(args.source).strip().lower()
            models_list = [
                m
                for m in models_list
                if flt in str(m.get("source", "")).lower() or flt in str(m.get("kind", "")).lower()
            ]

        if use_json:
            _emit_json({"status": "ok", "count": len(models_list), "models": models_list})
            return

        if not models_list:
            render_notice("No local models matched the scan criteria.", level="warn")
            return

        rows = []
        for m in models_list:
            src = str(m.get("source") or m.get("kind") or "local").upper()
            mid = str(m.get("id") or m.get("name") or "")
            arch = str(m.get("architecture") or m.get("kind") or "-")
            quant = str(m.get("quantization") or "-")
            sz = str(m.get("size_str") or "-")
            if m.get("ollama_running"):
                status = "● LIVE (Ollama)"
            elif m.get("can_load_direct"):
                status = "Direct PyTorch"
            elif m.get("can_transmute"):
                status = "Transmute"
            else:
                status = "Inspect"
            rows.append([src, mid, arch, quant, sz, status])

        render_table(
            f"Discovered Models ({len(models_list)})",
            columns=[
                ("Source", "left", "cyan"),
                ("Model / Path", "left", "bold white"),
                ("Arch / Kind", "left", "white"),
                ("Quant", "left", "yellow"),
                ("Size", "right", "white"),
                ("Action", "left", "green"),
            ],
            rows=rows,
            caption=f"Tip: run `{prog} chat --model ollama:<tag>` for live Ollama or `{prog} transmute --model <id>` for 3-tier HF transfer.",
        )

    elif args.command == "transmute":
        from lucid_machines.inference import generate_response, load_model_for_inference
        from lucid_machines.model.transmuter import transfer_weights_to_native_triune, transmute_hf_to_triune

        mode_label = "Zero-Loss 3-Tier Early-Exit" if args.mode == "early-exit" else "Native Surgical GLA + MoE"
        render_banner("TRIUNE WEIGHT TRANSMUTER", subtitle=str(args.model), badge=mode_label)
        t0 = time.perf_counter()
        quant = "nvfp4" if args.nvfp4 else None

        with status_spinner(f"Loading and transmuting weights ({args.mode})..."):
            if args.mode == "native-distill":
                teacher, tokenizer, _, dev, pad_id, eos_id = load_model_for_inference(
                    args.model, device=args.device, quantize=None, disk_stream=False
                )
                model, report = transfer_weights_to_native_triune(
                    teacher,
                    tokenizer,
                    reflex_ratio=args.reflex_ratio,
                    num_experts=args.experts,
                )
                metadata = report
            else:
                model, tokenizer, metadata, dev, pad_id, eos_id = transmute_hf_to_triune(
                    args.model,
                    reflex_ratio=args.reflex_ratio,
                    quantize=quant,
                    disk_stream=args.disk_stream,
                    disk_cache_dir=args.disk_cache,
                    rank=args.rank,
                    use_jev=True,
                    device=args.device,
                )
        elapsed = time.perf_counter() - t0
        rows: list[tuple[str, object]] = [
            ("Source Model", args.model),
            ("Transmute Mode", metadata.get("transmute_mode", args.mode)),
            ("Weights Transferred", "Yes (real weights intact)" if metadata.get("weights_transferred") else "No"),
            (
                "Exit Layers",
                f"Reflex L{metadata.get('reflex_exit_layer')}  ·  Limbic L{metadata.get('limbic_exit_layer')}  ·  Total {metadata.get('num_layers', metadata.get('student_layers', '?'))}L",
            ),
            ("Device", str(dev)),
            ("Load + Surgery Time", f"{elapsed:.2f}s"),
        ]
        if metadata.get("transferred_params"):
            cov = float(metadata.get("coverage_ratio", 0.0)) * 100.0
            rows.append(
                (
                    "Transferred Tensors",
                    f"{metadata['transferred_tensors']} tensors ({_format_param_count(int(metadata['transferred_params']))} params · {cov:.1f}% coverage)",
                )
            )
        render_kv_panel("Transmutation Report", rows, border_style="green")

        if args.prompt:
            print(f"\n  Prompt: {args.prompt}")
            sys.stdout.write(f"  {route_badge_ansi('TRIUNE')} ")
            sys.stdout.flush()
            printed = [0]
            res = generate_response(
                model,
                tokenizer,
                args.prompt,
                device=dev,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                max_new_tokens=args.max_tokens,
                on_text=lambda txt, **_: (
                    sys.stdout.write(str(txt)[printed[0] :]),
                    sys.stdout.flush(),
                    printed.__setitem__(0, len(str(txt))),
                ),
                return_info=True,
            )
            print()
            render_telemetry_bar(
                str(res.get("route", "CORTEX")),
                tok_sec=float(res.get("tokens_per_sec", 0.0)),
                ttft_ms=res.get("ttft_ms"),
                tokens=int(res.get("tokens_count", 0)),
                latency_ms=float(res.get("latency_ms", 0.0)),
                vram_gb=0.0,
            )

        if getattr(model, "_streaming_engine", None) is not None:
            model._streaming_engine.detach()
        if getattr(model, "_exit_manager", None) is not None and hasattr(model._exit_manager, "remove_hooks"):
            model._exit_manager.remove_hooks()

    elif args.command == "finetune":
        recipe: dict[str, Any] = {}
        if getattr(args, "recipe", None):
            try:
                recipe = _load_finetune_recipe(args.recipe)
            except Exception as exc:
                _cli_fail(f"Failed to load recipe '{args.recipe}': {exc}", code=2)

        def _pick(cli_val: Any, key: str) -> Any:
            if cli_val is not None:
                return cli_val
            if key in recipe and recipe[key] is not None:
                return recipe[key]
            return _FINETUNE_DEFAULTS.get(key)

        eff_model = args.model if args.model is not None else recipe.get("model")
        eff_checkpoint = args.checkpoint if args.checkpoint is not None else recipe.get("checkpoint")
        eff_dataset = str(_pick(args.dataset, "dataset"))
        eff_dataset_split = str(_pick(args.dataset_split, "dataset_split"))
        eff_max_samples = int(_pick(args.max_samples, "max_samples"))
        eff_rank = int(_pick(args.rank, "rank"))
        eff_alpha = float(_pick(args.alpha, "alpha"))
        eff_epochs = int(_pick(args.epochs, "epochs"))
        eff_lr = float(_pick(args.lr, "lr"))
        eff_batch_size = int(_pick(args.batch_size, "batch_size"))
        eff_seq_len = int(_pick(args.seq_len, "seq_len"))
        eff_grad_accum = int(_pick(args.grad_accum, "grad_accum"))
        eff_max_steps = args.max_steps if args.max_steps is not None else recipe.get("max_steps")
        if eff_max_steps is not None:
            eff_max_steps = int(eff_max_steps)
        eff_output_dir = str(_pick(args.output_dir, "output_dir"))
        eff_quantization = str(_pick(args.quantization, "quantization"))
        eff_qlora = bool(args.qlora or recipe.get("qlora", False))
        eff_stream = bool(args.stream or recipe.get("stream", False))
        eff_stream_chunk = int(_pick(args.stream_chunk, "stream_chunk"))
        eff_disk_stream = bool(args.disk_stream or recipe.get("disk_stream", False))
        eff_disk_cache = args.disk_cache if args.disk_cache is not None else recipe.get("disk_cache")
        eff_quant_lm_head = bool(getattr(args, "quantize_lm_head", False) or recipe.get("quantize_lm_head", False))
        eff_checkpointing = bool(args.checkpointing or recipe.get("checkpointing", False))

        eff_wandb_project = str(_pick(getattr(args, "wandb_project", None), "wandb_project"))
        eff_wandb_run = getattr(args, "wandb_run", None) or recipe.get("wandb_run")
        use_wandb = bool(getattr(args, "wandb", False) or recipe.get("wandb", False) or eff_wandb_run)

        if eff_rank <= 0 or eff_alpha <= 0 or eff_epochs <= 0 or eff_lr <= 0 or eff_batch_size <= 0 or eff_seq_len <= 0 or eff_grad_accum <= 0 or eff_stream_chunk <= 0:
            _cli_fail(
                "Invalid finetune hyperparameters: rank, alpha, epochs, lr, batch-size, seq-len, grad-accum, and stream-chunk must be positive.",
                code=2,
            )

        if eff_checkpoint and not eff_model and not Path(str(eff_checkpoint)).is_file():
            _cli_fail(f"Checkpoint not found: {eff_checkpoint}", code=1)

        import torch

        from lucid_machines.configs import build_config
        from lucid_machines.inference import load_checkpoint_model, load_model_for_inference
        from lucid_machines.model import build_model
        from lucid_machines.trainer.finetune import LoRAConfig, TriuneFineTuner
        from lucid_machines.trainer.logger import NullLogger, WandbLogger

        render_banner(
            "TRIUNE FINE-TUNING",
            subtitle=eff_model or eff_checkpoint or "Triune Default",
            badge=f"r={eff_rank} · {'NVFP4 QLoRA' if eff_qlora else eff_quantization}",
        )
        tokenizer = None
        try:
            if eff_model:
                quant = "nvfp4" if eff_qlora else None
                with status_spinner(f"Loading base model {eff_model}..."):
                    model, tokenizer, _, device, _, _ = load_model_for_inference(
                        eff_model,
                        device=None,
                        quantize=quant,
                        disk_stream=eff_disk_stream,
                        disk_cache_dir=eff_disk_cache,
                        quantize_lm_head=eff_quant_lm_head,
                    )
                if hasattr(tokenizer, "hf"):
                    tokenizer = tokenizer.hf
            elif eff_checkpoint:
                with status_spinner(f"Loading checkpoint {eff_checkpoint}..."):
                    model, tokenizer, _, device, _, _ = load_checkpoint_model(eff_checkpoint)
            else:
                render_notice("Instantiating default Triune architecture for fine-tuning...", level="info")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cfg = build_config({})
                model = build_model(cfg).to(device)
        except Exception as exc:
            _cli_fail(f"Failed to load base model/checkpoint for fine-tuning: {exc}", code=1)

        if use_wandb:
            try:
                wb_logger: Any = WandbLogger(
                    project=eff_wandb_project,
                    config={
                        "model": eff_model or eff_checkpoint or "triune-default",
                        "dataset": eff_dataset,
                        "rank": eff_rank,
                        "alpha": eff_alpha,
                        "lr": eff_lr,
                        "epochs": eff_epochs,
                        "batch_size": eff_batch_size,
                        "seq_len": eff_seq_len,
                        "grad_accum": eff_grad_accum,
                        "qlora": eff_qlora,
                    },
                    run_id=eff_wandb_run,
                )
            except RuntimeError as exc:
                _cli_fail(str(exc), code=1)
        else:
            wb_logger = NullLogger()

        lora_cfg = LoRAConfig(
            rank=eff_rank,
            alpha=eff_alpha,
            use_qlora=eff_qlora,
            quantization=("4-bit NVFP4" if eff_qlora else eff_quantization),
            quantize_base=(True if eff_qlora else None),
        )
        tuner = TriuneFineTuner(
            model,
            lora_cfg,
            tokenizer=tokenizer,
            model_name_or_path=eff_model,
            adapter_device=device,
        )
        trainable = tuner.count_trainable_parameters()
        total_p = sum(p.numel() for p in model.parameters())
        fin_rows: list[tuple[str, Any]] = [
            ("Trainable Params", f"{trainable:,} / {total_p:,} ({100.0 * trainable / max(1, total_p):.2f}%)"),
            ("Dataset", f"{eff_dataset} (split={eff_dataset_split})"),
            (
                "Schedule",
                f"epochs={eff_epochs} · batch={eff_batch_size} · accum={eff_grad_accum} · lr={eff_lr}",
            ),
            ("Output Dir", eff_output_dir),
        ]
        if getattr(args, "recipe", None):
            fin_rows.append(("Recipe", args.recipe))
        if use_wandb:
            fin_rows.append(("W&B Run", f"{eff_wandb_project} ({wb_logger.run_id or 'auto'})"))
        render_kv_panel("LoRA Configuration", fin_rows, border_style="cyan")

        def _cli_on_step(step: int, total_s: int, loss_val: float, lm_loss_val: float):
            wb_logger.log({"loss": loss_val, "lm_loss": lm_loss_val}, step=step)
            interval = max(1, total_s // 20)
            if step % interval == 0 or step == total_s:
                pct = round((step / max(1, total_s)) * 100)
                print(
                    f"  [Step {step:>4}/{total_s} ({pct:>3}%)] Loss: {loss_val:.4f} | LM Loss: {lm_loss_val:.4f}",
                    flush=True,
                )

        try:
            res = tuner.fit(
                dataset_path=eff_dataset,
                output_dir=eff_output_dir,
                epochs=eff_epochs,
                batch_size=eff_batch_size,
                seq_len=eff_seq_len,
                lr=eff_lr,
                max_steps=eff_max_steps,
                on_step=_cli_on_step,
                tokenizer=tokenizer,
                dataset_split=eff_dataset_split,
                max_samples=eff_max_samples,
                grad_accum_steps=eff_grad_accum,
                stream_layers=bool(eff_stream and not eff_disk_stream),
                stream_chunk_size=eff_stream_chunk,
                gradient_checkpointing=eff_checkpointing,
            )
            render_notice(
                f"Fine-tuning finished! Final Loss: {res['final_loss']:.4f} -> Saved to: {res['output_dir']}",
                level="success",
            )
        finally:
            wb_logger.finish()
            if getattr(model, "_streaming_engine", None) is not None:
                model._streaming_engine.detach()

    elif args.command == "adapter":
        if args.adapter_action == "list":
            from lucid_machines.trainer.finetune import list_adapters

            ads = list_adapters()
            if not ads:
                render_notice(
                    "No adapters found on disk (scanned checkpoints/, checkpoints/adapters/, exports/).",
                    level="info",
                )
            else:
                render_table(
                    f"LoRA Adapters on Disk ({len(ads)})",
                    columns=[
                        ("Name", "left", "bold cyan"),
                        ("Rank / Alpha", "left", "white"),
                        ("Precision", "left", "yellow"),
                        ("Size", "right", "white"),
                        ("Path", "left", "dim"),
                    ],
                    rows=[
                        [
                            a["name"],
                            f"r={a['rank']}, α={a['alpha']}",
                            a["quantization"],
                            f"{a['size_mb']} MB",
                            a["path"],
                        ]
                        for a in ads
                    ],
                )
        elif args.adapter_action == "merge":
            from lucid_machines.trainer.finetune import merge_adapter_checkpoint

            if not Path(args.checkpoint).is_file():
                _cli_fail(f"Checkpoint not found: {args.checkpoint}", code=1)
            try:
                out = merge_adapter_checkpoint(
                    args.adapter, args.checkpoint, output_path=args.output, tokenizer_path=args.tokenizer
                )
            except Exception as exc:
                _cli_fail(f"Failed to merge adapter '{args.adapter}': {exc}", code=1)
            render_notice(f"Merged adapter '{args.adapter}' into '{args.checkpoint}' -> {out}", level="success")
        elif args.adapter_action == "export":
            from lucid_machines.trainer.finetune import export_adapter

            try:
                out = export_adapter(args.adapter, output_path=args.output, fmt=args.format)
            except Exception as exc:
                _cli_fail(f"Failed to export adapter '{args.adapter}': {exc}", code=1)
            render_notice(f"Exported adapter to: {out}", level="success")
        else:
            _get_subparser(parser, "adapter").print_help()
            raise SystemExit(2)

    elif args.command == "checkpoint":
        if args.ckpt_action == "list":
            import json as _json

            use_json = bool(getattr(args, "json", False))
            ckpts = []
            ckpt_items: list[dict[str, Any]] = []
            with _quiet_json_stdout(use_json):
                for d in ("checkpoints", "checkpoints_full"):
                    p = Path(d)
                    if p.is_dir():
                        for f in sorted(p.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
                            step_str = "-"
                            model_str = "-"
                            loss_str = "-"
                            step_num = None
                            loss_num = None
                            try:
                                import torch

                                meta = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
                                if isinstance(meta, dict):
                                    if meta.get("step") is not None:
                                        step_num = meta["step"]
                                        step_str = str(step_num)
                                    cfg_meta = meta.get("config") or {}
                                    if isinstance(cfg_meta, dict) and cfg_meta.get("model_name"):
                                        model_str = str(cfg_meta["model_name"])
                                    loss_val = meta.get("best_eval_loss")
                                    if loss_val is None:
                                        loss_val = meta.get("loss")
                                    if isinstance(loss_val, (int, float)):
                                        loss_num = float(loss_val)
                                        loss_str = f"{loss_val:.4f}"
                            except Exception:
                                pass
                            st_stat = f.stat()
                            size_mb = round(st_stat.st_size / (1024 * 1024), 2)
                            modified_str = time.ctime(st_stat.st_mtime)
                            ckpts.append(
                                (
                                    str(f),
                                    step_str,
                                    model_str,
                                    loss_str,
                                    f"{size_mb:.1f} MB",
                                    modified_str,
                                )
                            )
                            ckpt_items.append(
                                {
                                    "path": str(f),
                                    "step": step_num,
                                    "model": model_str if model_str != "-" else None,
                                    "loss": loss_num,
                                    "size_mb": size_mb,
                                    "modified_epoch": st_stat.st_mtime,
                                    "modified": modified_str,
                                }
                            )
            session_p = Path("checkpoints/session_state.json")
            session_caption = None
            session_dict = None
            if session_p.is_file():
                try:
                    st = _json.loads(session_p.read_text(encoding="utf-8"))
                    session_dict = st
                    session_caption = f"Session state: step={st.get('step')} · active={st.get('active_checkpoint') or st.get('active_model_id')}"
                except Exception:
                    pass
            if use_json:
                _emit_json(
                    {
                        "status": "ok",
                        "count": len(ckpt_items),
                        "checkpoints": ckpt_items,
                        "session_state": session_dict,
                    }
                )
            elif not ckpts:
                render_notice("No checkpoints found in checkpoints/ or checkpoints_full/.", level="info")
            else:
                render_table(
                    f"Local Training Checkpoints ({len(ckpts)})",
                    columns=[
                        ("Path", "left", "bold cyan"),
                        ("Step", "right", "white"),
                        ("Model", "left", "green"),
                        ("Loss", "right", "yellow"),
                        ("Size", "right", "yellow"),
                        ("Modified", "left", "dim"),
                    ],
                    rows=[list(row) for row in ckpts],
                    caption=session_caption,
                )
        elif args.ckpt_action == "export":
            if str(args.format).lower() == "gguf":
                render_notice(
                    "GGUF export is not yet supported for Triune's GLA, early-exit heads, and MoE architecture "
                    "(no weights were exported). Use `--format vllm`, `--format hf`, `--format safetensors`, or `--format onnx` "
                    "(or run existing GGUFs via `--model ollama:<tag>`).",
                    level="error",
                )
                raise SystemExit(2)

            try:
                ckpt_path = resolve_default_checkpoint(args.checkpoint)
            except FileNotFoundError as exc:
                _cli_fail(str(exc), code=1)

            from lucid_machines.export import export_model
            from lucid_machines.inference import load_checkpoint_model

            try:
                with status_spinner(f"Loading checkpoint {ckpt_path}..."):
                    model, tokenizer, _, _, _, _ = load_checkpoint_model(ckpt_path)
            except Exception as exc:
                _cli_fail(f"Failed to load checkpoint '{ckpt_path}': {exc}", code=1)
            try:
                if args.format == "hf":
                    from lucid_machines.export.hf_compat import export_to_huggingface

                    out = args.output or f"exports/{ckpt_path.stem}_hf"
                    out_p = export_to_huggingface(model, out, tokenizer=tokenizer)
                    render_notice(f"Exported Hugging Face bundle to: {out_p}", level="success")
                elif args.format == "vllm":
                    from lucid_machines.export.vllm_bridge import export_vllm_bundle

                    out = args.output or f"exports/{ckpt_path.stem}_vllm"
                    out_p = export_vllm_bundle(model, out, model_id=ckpt_path.stem, tokenizer=tokenizer)
                    vllm_cfg = json.loads((Path(out_p) / "vllm_serving.json").read_text(encoding="utf-8"))
                    render_kv_panel(
                        "vLLM Serving Bundle Ready",
                        [
                            ("Bundle Dir", str(out_p)),
                            ("Serving Config", str(Path(out_p) / "vllm_serving.json")),
                            ("Trust Remote Code", str(vllm_cfg.get("trust_remote_code"))),
                            ("Launch Command", str(vllm_cfg.get("launch_command"))),
                        ],
                        border_style="green",
                    )
                    render_notice(f"Exported vLLM serving bundle to: {out_p}", level="success")
                else:
                    out = args.output or f"exports/{ckpt_path.stem}.{args.format}"
                    out_p = export_model(model, out, fmt=args.format)
                    render_notice(f"Exported checkpoint to: {out_p}", level="success")
            except NotImplementedError as exc:
                render_notice(str(exc), level="error")
                raise SystemExit(2)
        else:
            _get_subparser(parser, "checkpoint").print_help()
            raise SystemExit(2)

    elif args.command in ("export", "convert"):
        if str(args.format).lower() == "gguf":
            render_notice(
                "GGUF export is not yet supported for Triune's GLA, early-exit heads, and MoE architecture "
                "(no weights were exported). Use `--format vllm`, `--format hf`, `--format safetensors`, or `--format onnx` "
                "(or run existing GGUFs via `--model ollama:<tag>`).",
                level="error",
            )
            raise SystemExit(2)

        try:
            ckpt_path = resolve_default_checkpoint(args.checkpoint)
        except FileNotFoundError as exc:
            _cli_fail(str(exc), code=1)

        from lucid_machines.export import export_model
        from lucid_machines.inference import load_checkpoint_model

        try:
            with status_spinner(f"Loading checkpoint {ckpt_path}..."):
                model, tokenizer, _, _, _, _ = load_checkpoint_model(ckpt_path)
        except Exception as exc:
            _cli_fail(f"Failed to load checkpoint '{ckpt_path}': {exc}", code=1)
        try:
            if args.format == "hf":
                from lucid_machines.export.hf_compat import export_to_huggingface

                out = args.output or f"exports/{ckpt_path.stem}_hf"
                out_p = export_to_huggingface(model, out, tokenizer=tokenizer)
                render_notice(f"Exported Hugging Face bundle to: {out_p}", level="success")
            elif args.format == "vllm":
                from lucid_machines.export.vllm_bridge import export_vllm_bundle

                out = args.output or f"exports/{ckpt_path.stem}_vllm"
                out_p = export_vllm_bundle(model, out, model_id=ckpt_path.stem, tokenizer=tokenizer)
                vllm_cfg = json.loads((Path(out_p) / "vllm_serving.json").read_text(encoding="utf-8"))
                render_kv_panel(
                    "vLLM Serving Bundle Ready",
                    [
                        ("Bundle Dir", str(out_p)),
                        ("Serving Config", str(Path(out_p) / "vllm_serving.json")),
                        ("Trust Remote Code", str(vllm_cfg.get("trust_remote_code"))),
                        ("Launch Command", str(vllm_cfg.get("launch_command"))),
                    ],
                    border_style="green",
                )
                render_notice(f"Exported vLLM serving bundle to: {out_p}", level="success")
            else:
                out = args.output or f"exports/{ckpt_path.stem}.{args.format}"
                out_p = export_model(model, out, fmt=args.format)
                render_notice(f"Exported checkpoint to: {out_p}", level="success")
        except NotImplementedError as exc:
            render_notice(str(exc), level="error")
            raise SystemExit(2)

    elif args.command == "models":
        import json as _json

        from lucid_machines.model.base import CANONICAL_MODEL_SPECS, ModelSpec

        use_json = bool(getattr(args, "json", False))
        merged_specs: dict[str, ModelSpec] = dict(CANONICAL_MODEL_SPECS)
        cfg_dir = Path("models/configs")
        if not cfg_dir.is_dir():
            cfg_dir = Path(__file__).resolve().parent.parent / "models" / "configs"
        if cfg_dir.is_dir():
            for cfg_file in sorted(cfg_dir.glob("*.json")):
                try:
                    raw = _json.loads(cfg_file.read_text(encoding="utf-8-sig"))
                    if isinstance(raw, dict) and cfg_file.stem not in merged_specs:
                        merged_specs[cfg_file.stem] = ModelSpec.from_dict({"model_id": cfg_file.stem, **raw})
                except Exception:
                    pass

        if use_json:
            model_records = []
            for k, v in merged_specs.items():
                total_p, active_p = v.compute_exact_params()
                model_records.append(
                    {
                        "model_id": k,
                        "name": v.name,
                        "num_layers": v.num_layers,
                        "hidden_dim": v.hidden_dim,
                        "num_experts": v.num_experts,
                        "experts_shared": v.experts_shared,
                        "top_k_experts": v.top_k_experts,
                        "total_params": total_p,
                        "active_params": active_p,
                        "tier": v.tier,
                    }
                )
            _emit_json({"status": "ok", "count": len(model_records), "models": model_records})
            return

        rows = []
        for k, v in merged_specs.items():
            total_p, active_p = v.compute_exact_params()
            rows.append(
                [
                    k,
                    v.name,
                    v.num_layers,
                    v.hidden_dim,
                    f"{v.num_experts}R/{v.experts_shared}S (k={v.top_k_experts})",
                    _format_param_count(total_p),
                    _format_param_count(active_p),
                    v.tier,
                ]
            )
        render_table(
            f"Triune & Junkie Model Architecture Zoo ({len(merged_specs)} Presets)",
            columns=[
                ("Preset ID", "left", "bold cyan"),
                ("Architecture", "left", "white"),
                ("Layers", "right", "yellow"),
                ("Hidden", "right", "white"),
                ("Experts", "left", "green"),
                ("Total", "right", "cyan"),
                ("Active", "right", "green"),
                ("Tier", "left", "dim"),
            ],
            rows=rows,
        )

    elif args.command == "train":
        from scripts.train import main as train_main

        train_argv = list(args.train_args or [])
        sys.argv = [f"{prog} train", *train_argv]
        train_main(train_argv)

    elif args.command in ("benchmark", "bench"):
        use_json = bool(getattr(args, "json", False))
        suite = getattr(args, "suite", None)
        if suite == "mqar":
            from scripts.benchmark_inference import benchmark_mqar

            mqar_args = argparse.Namespace(
                checkpoint=args.checkpoint,
                tokenizer=args.tokenizer,
                device=args.device,
                depth=args.depth,
                quick=bool(getattr(args, "quick", False)),
                seq_lens=getattr(args, "seq_lens", None),
            )
            try:
                with _quiet_json_stdout(use_json), _maybe_spinner("Running MQAR associative recall suite...", use_json):
                    res = benchmark_mqar(mqar_args)
            except Exception as exc:
                _cli_fail(f"MQAR benchmark failed: {exc}", code=1, use_json=use_json, command="benchmark")
            if use_json:
                _emit_json(res)
                return
            render_banner("TRIUNE MQAR BENCHMARK SUITE", subtitle=f"{res['device']} · {res['dtype']}")
            k_rows = [
                [
                    r["model_type"],
                    r["seq_len"],
                    r["num_kv_pairs"],
                    f"{r['accuracy'] * 100:.1f}%",
                    f"{r['overwrite_accuracy'] * 100:.1f}%",
                    f"{r['latency_ms']:.2f} ms",
                ]
                for r in res["kernel_results"]
            ]
            render_table(
                "MQAR Kernel Scaling",
                columns=[
                    ("Kernel", "left", "bold cyan"),
                    ("Seq Len", "right", "white"),
                    ("KV Pairs", "right", "yellow"),
                    ("Recall Acc", "right", "green"),
                    ("Overwrite Acc", "right", "cyan"),
                    ("Latency", "right", "white"),
                ],
                rows=k_rows,
            )
            if res.get("checkpoint_eval"):
                ce = res["checkpoint_eval"]
                render_kv_panel(
                    "Checkpoint MQAR Recall",
                    [
                        ("Checkpoint", ce["checkpoint"]),
                        ("Trials", ce["num_trials"]),
                        ("Choice Accuracy", f"{ce['choice_accuracy'] * 100:.1f}% (baseline {ce['random_baseline'] * 100:.1f}%)"),
                        ("Exact Top-1", f"{ce['exact_top1_accuracy'] * 100:.1f}%"),
                        ("Mean Target LogProb", f"{ce['mean_target_logprob']:.4f}"),
                    ],
                    border_style="green",
                )
            return

        if suite:
            from scripts.run_all_benchmarks import run_category

            suite_map = {
                "all": ["a", "b", "c", "d", "e", "f"],
                "kernels": ["a"],
                "optimizers": ["b"],
                "streaming": ["c"],
                "context": ["d"],
                "routing": ["e"],
                "matrix": ["f"],
            }
            cats = suite_map.get(suite, ["a"])
            if not use_json:
                render_banner("TRIUNE BENCHMARK SUITE", subtitle=f"Running suite '{suite}'")
            with _quiet_json_stdout(use_json):
                for cat in cats:
                    run_category(cat, sys.executable)
            if use_json:
                _emit_json({"status": "ok", "suite": suite, "categories": cats})
            return

        try:
            ckpt_path = resolve_default_checkpoint(args.checkpoint)
        except FileNotFoundError as exc:
            _cli_fail(str(exc), code=1, use_json=use_json, command="benchmark")

        from scripts.benchmark_inference import benchmark_cli

        b_args = argparse.Namespace(
            checkpoint=str(ckpt_path),
            tokenizer=args.tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            device=args.device,
            depth=args.depth,
            temperature=args.temperature,
            warmup_tokens=8,
        )
        try:
            with _quiet_json_stdout(use_json), _maybe_spinner(f"Benchmarking inference on {ckpt_path}...", use_json):
                res = benchmark_cli(b_args)
        except Exception as exc:
            _cli_fail(f"Inference benchmark failed: {exc}", code=1, use_json=use_json, command="benchmark")

        if use_json:
            _emit_json({"status": "ok", **res})
            return

        render_kv_panel(
            "Triune Inference Benchmark",
            [
                ("Checkpoint", res["checkpoint"]),
                ("Device", res["device"]),
                ("Load Time", f"{res['model_load_seconds']} s"),
                ("TTFT", f"{res.get('ttft_ms', res.get('first_token_latency_ms', 0)):.2f} ms"),
                ("Decode Speed", f"{res.get('tokens_per_second', 0):.2f} tokens/sec"),
                ("Tokens Generated", res.get("tokens_count", res.get("tokens_generated", 0))),
                ("CUDA VRAM", f"{res.get('cuda_allocated_gb', 0)} GB"),
                ("Response Preview", str(res.get("text", res.get("response", "")))[:120]),
            ],
            border_style="cyan",
        )

    elif args.command == "gpucheck":
        from lucid_machines.utils.device import device_diagnostic

        use_json = bool(getattr(args, "json", False))
        with _quiet_json_stdout(use_json):
            diag = device_diagnostic()
        if use_json:
            _emit_json({"status": "ok", **diag})
            return

        render_banner(
            "TRIUNE GPU & RUNTIME CHECK",
            subtitle=f"PyTorch {diag['torch']}",
            badge="CUDA" if diag["cuda_available"] else "CPU ONLY",
        )
        gpu_list = diag.get("gpus") or []
        if gpu_list:
            render_table(
                "Detected CUDA GPUs",
                columns=[
                    ("GPU", "right", "cyan"),
                    ("Device Name", "left", "bold white"),
                    ("Memory", "right", "green"),
                    ("Compute Cap", "left", "yellow"),
                ],
                rows=[
                    [f"[{i}]", g["name"], f"{g['memory_gib']:.2f} GiB", f"SM {g['capability'][0]}.{g['capability'][1]}"]
                    for i, g in enumerate(gpu_list)
                ],
            )
        else:
            render_notice("No CUDA GPUs detected — running on CPU fallback.", level="warn")

    elif args.command == "plan-memory":
        from lucid_machines.configs import build_config
        from lucid_machines.runtime import MemoryPlanner

        if getattr(args, "vram_gb", None) is not None and args.vram_gb <= 0:
            render_notice(
                f"Invalid --vram-gb value ({args.vram_gb}): VRAM budget must be a positive number.",
                level="error",
            )
            sys.exit(2)

        if getattr(args, "model", None):
            try:
                hf_plan = MemoryPlanner.estimate_hf_model(args.model, available_vram_gb=args.vram_gb)
            except Exception as exc:
                render_notice(f"Could not inspect model '{args.model}': {exc}", level="error")
                sys.exit(1)
            plan_rows = [
                (
                    "Parameters",
                    f"{hf_plan['total_params']:,} ({hf_plan['layers']}L, hidden={hf_plan['hidden_size']}, vocab={hf_plan['vocab_size']:,})",
                ),
            ]
            if hf_plan.get("active_params"):
                plan_rows.append(
                    (
                        "Active / Token",
                        f"{hf_plan['active_params']:,} (top_k={hf_plan.get('top_k_experts', 1)})",
                    )
                )
            plan_rows.extend(
                [
                    ("BF16 Weights", f"{hf_plan['bf16_weights_gb']} GB"),
                    (
                        "NVFP4 Cache (SSD/VRAM)",
                        f"{hf_plan['nvfp4_disk_cache_gb']} GB (~{hf_plan['per_layer_nvfp4_mb']} MB/layer; quant lm_head={hf_plan['nvfp4_quant_lmhead_cache_gb']} GB)",
                    ),
                ]
            )
            if hf_plan.get("centroid_steer_static_vram_gb") is not None:
                fp8_static = hf_plan.get("centroid_steer_fp8_static_vram_gb")
                if fp8_static is not None:
                    train_str = f"{hf_plan['centroid_steer_static_vram_gb']} GB BF16 / {fp8_static} GB FP8 static (opt={hf_plan['centroid_steer_opt_gb']} GB)"
                else:
                    train_str = f"{hf_plan['centroid_steer_static_vram_gb']} GB static (opt={hf_plan['centroid_steer_opt_gb']} GB)"
                plan_rows.append(("CentroidSteer Train", train_str))
            plan_rows.append(
                ("Hardware", f"{hf_plan['available_vram_gb']} GB VRAM  ·  {hf_plan['host_ram_gb']} GB Host RAM")
            )
            if hf_plan.get("recommended_train_mode"):
                plan_rows.extend(
                    [
                        ("Inference Mode", hf_plan["recommended_mode"]),
                        ("Training Mode", hf_plan["recommended_train_mode"]),
                    ]
                )
            else:
                plan_rows.append(("Recommended Mode", hf_plan["recommended_mode"]))
            render_kv_panel(
                f"Hardware Memory Plan · {hf_plan['model']}",
                plan_rows,
                border_style="cyan",
            )
        else:
            config = build_config({})
            plan = MemoryPlanner.estimate_vram(config, target_vram_gb=args.vram_gb or 8.0)
            render_kv_panel(
                "Triune VRAM Memory Plan Estimate",
                [
                    ("Parameters", f"{plan.total_params:,} ({plan.param_memory_gb} GB)"),
                    ("Optimizer State", f"{plan.optimizer_memory_gb} GB"),
                    ("Activation Memory", f"{plan.activation_memory_gb} GB"),
                    ("Batch Size", plan.recommended_batch_size),
                    ("Grad Accum", plan.recommended_grad_accum),
                    ("Precision", plan.recommended_precision),
                ],
                border_style="cyan",
            )

    elif args.command == "cache-nvfp4":
        from lucid_machines.runtime.disk_streaming import build_nvfp4_disk_cache

        render_banner("NVFP4 DISK CACHE PACKER", subtitle=str(args.model))
        with status_spinner("Packing SafeTensors layers into NVFP4 shards..."):
            cache_path = build_nvfp4_disk_cache(
                args.model,
                cache_dir=args.out,
                hadamard=(not args.no_hadamard),
                quantize_lm_head=bool(args.quantize_lm_head),
                force=bool(args.force),
            )
        manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
        raw_mb = manifest["raw_bytes"] / (1024 * 1024)
        packed_mb = manifest["packed_bytes"] / (1024 * 1024)
        render_kv_panel(
            "NVFP4 Cache Ready",
            [
                ("Cache Path", str(cache_path)),
                ("Layers Packed", manifest["num_layers"]),
                ("Footprint", f"{raw_mb:.1f} MB -> {packed_mb:.1f} MB ({manifest['compression_ratio']}x compression)"),
            ],
            border_style="green",
        )

    elif args.command == "install":
        import subprocess

        roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
        setup_exe = next(
            (
                cand
                for root in roots
                for cand in (
                    root / "PolysomStudioSetup.exe",
                    root / "studio" / "windows" / "publish" / "PolysomStudioSetup.exe",
                    root / "studio" / "publish" / "PolysomStudioSetup.exe",
                )
                if cand.is_file()
            ),
            None,
        )
        installer_script = next(
            (
                root / "studio" / "installer" / "installer.py"
                for root in roots
                if (root / "studio" / "installer" / "installer.py").is_file()
            ),
            None,
        )
        workspace = (
            setup_exe.parent
            if setup_exe is not None
            else (installer_script.parents[2] if installer_script else Path.cwd())
        )
        if sys.platform == "win32" and setup_exe is not None and not args.yes and not args.verify:
            render_notice(f"Launching Native Windows Setup Wizard: {setup_exe}", level="info")
            subprocess.Popen([str(setup_exe)], cwd=str(workspace))
        elif installer_script is not None:
            cmd = [sys.executable, str(installer_script), "--cuda", args.cuda]
            if args.verify:
                cmd.append("--verify")
            if args.skip_torch:
                cmd.append("--skip-torch")
            if args.no_launch:
                cmd.append("--no-launch")
            if args.yes:
                cmd.append("--yes")
            raise SystemExit(subprocess.call(cmd, cwd=str(workspace)))
        elif args.verify:
            from lucid_machines.utils.device import device_diagnostic

            diag = device_diagnostic()
            render_notice(
                f"Environment verified: Python {diag['python']} · PyTorch {diag['torch']} · CUDA={diag['cuda_available']}",
                level="success",
            )
        else:
            render_notice(
                "Studio installer script (studio/installer/installer.py) is not bundled in this wheel install. "
                "Install extras via `pip install lucid-machines[studio]` or run from a source checkout.",
                level="warn",
            )
            raise SystemExit(1)

    elif args.command in ("doctor", "info"):
        import importlib.util
        import json as _json

        from lucid_machines.model.base import CANONICAL_MODEL_SPECS
        from lucid_machines.ops import get_native_c_info
        from lucid_machines.runtime.model_scanner import query_ollama_running_tags
        from lucid_machines.utils.device import device_diagnostic

        use_json = bool(getattr(args, "json", False))
        with _quiet_json_stdout(use_json):
            diag = device_diagnostic()
            c_info = get_native_c_info()
            has_fastapi = importlib.util.find_spec("fastapi") is not None
            ollama_tags = sorted(t for t in query_ollama_running_tags() if ":" in t)

        session_p = Path("checkpoints/session_state.json")
        st_dict = None
        if session_p.is_file():
            try:
                st_dict = _json.loads(session_p.read_text(encoding="utf-8"))
            except Exception:
                st_dict = None

        if use_json:
            _emit_json(
                {
                    "status": "healthy",
                    "python": diag["python"],
                    "torch": diag["torch"],
                    "cuda_available": diag["cuda_available"],
                    "gpus": diag.get("gpus") or [],
                    "native_c": c_info,
                    "fastapi_available": has_fastapi,
                    "ollama_running_tags": ollama_tags,
                    "canonical_presets_count": len(CANONICAL_MODEL_SPECS),
                    "session_state": st_dict,
                }
            )
            return

        render_banner("TRIUNE SYSTEM DOCTOR", subtitle="Runtime, Kernels, Local Runtimes & State", badge="HEALTHY")
        rows: list[tuple[str, object]] = [
            ("Python Runtime", diag["python"]),
            ("PyTorch Version", diag["torch"]),
            ("CUDA Available", "Yes" if diag["cuda_available"] else "No (CPU mode)"),
        ]
        for idx, gpu in enumerate(diag.get("gpus", [])):
            maj, min_v = gpu["capability"]
            rows.append((f"GPU [{idx}]", f"{gpu['name']} ({gpu['memory_gib']:.2f} GiB, SM {maj}.{min_v})"))
        rows.extend(
            [
                (
                    "Native C Engine",
                    f"{'Loaded' if c_info.get('available') else 'Fallback'} (v{c_info.get('version', '2.0')}, CUDA-Bound={c_info.get('cuda_runtime_bound', False)}, SIMD=AVX2)",
                ),
                ("Optimizers", "CentroidSteer (Triangular Core), Muon (Polar Express 4-Step NS), AdamW / 8-bit"),
                ("Fused Operators", "GLA, Chunked CE, RMSNorm, RoPE, SwiGLU, Fused LoRA, 3-Tier Transmuter"),
                ("Model Zoo", f"{len(CANONICAL_MODEL_SPECS)} Canonical Architecture Presets"),
                ("Studio FastAPI", "Operational" if has_fastapi else "Missing (install [studio] extras)"),
                (
                    "Local Ollama",
                    f"Running ({', '.join(ollama_tags[:6])})" if ollama_tags else "Not detected on 127.0.0.1:11434",
                ),
            ]
        )
        if st_dict is not None:
            rows.append(
                (
                    "Active Session",
                    f"{st_dict.get('active_checkpoint') or st_dict.get('active_model_id')} (step {st_dict.get('step')}, {st_dict.get('total_tokens_trained', 0):,} tokens)",
                )
            )
            rows.append(("Dataset Stream", f"{st_dict.get('dataset_name')} (offset {st_dict.get('hf_stream_offset', 0)})"))
        render_kv_panel("System Diagnostics", rows, border_style="green")

    elif args.command == "build-native":
        from lucid_machines.packaging import build_native_targets

        render_banner("TRIUNE NATIVE PACKAGER", subtitle=f"Platform: {args.platform}")
        with status_spinner("Building native packaging targets..."):
            res = build_native_targets(args.platform)
        if isinstance(res, dict):
            render_kv_panel(
                "Build Summary",
                [(k, json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in res.items()],
                border_style="green",
            )
        else:
            print(json.dumps(res, indent=2))

    elif args.command == "profile":
        from scripts.profile_model import main as prof_main

        try:
            ckpt_path = resolve_default_checkpoint(args.checkpoint, prefer="latest")
            prof_main(
                [
                    "--checkpoint",
                    str(ckpt_path),
                    "--tokenizer_path",
                    args.tokenizer,
                    "--output",
                    args.output,
                    "--batch_size",
                    str(args.batch_size),
                    "--seq_len",
                    str(args.seq_len),
                    "--warmup_steps",
                    str(args.warmup_steps),
                    "--active_steps",
                    str(args.active_steps),
                ]
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            _cli_fail(f"Profiling failed: {exc}", code=1)

    elif args.command == "tokenizer":
        from scripts.tokenizer import main as tok_main

        tok_main(
            [
                "--output",
                args.output,
                "--vocab_size",
                str(args.vocab_size),
            ]
        )

    elif args.command == "agent":
        from lucid_machines.agents import AgentHarness
        from lucid_machines.agents._storage import atomic_write_json, load_validated_json_dict

        output_format = str(getattr(args, "output_format", "text") or "text").lower()
        is_machine_fmt = output_format in ("json", "stream-json")

        ws_path = Path(args.workspace).expanduser().resolve(strict=False)
        if not ws_path.is_dir():
            _cli_fail(
                f"Workspace directory does not exist: {args.workspace}",
                code=1,
                use_json=is_machine_fmt,
                command="agent",
            )

        harness = AgentHarness(
            workspace=ws_path,
            store_dir=args.store_dir,
            model=args.model,
            allow_shell=bool(args.allow_shell),
            allow_writes=not bool(args.read_only),
        )
        if not args.model and harness.model is None and harness.completion_fn is None:
            def _offline_local_completion(messages, **_kw):
                tool_msgs = [m for m in messages if m.get("role") == "tool"]
                if not tool_msgs and "list_dir" in harness.tools._tools:
                    return {
                        "content": json.dumps(
                            {"thought": "Inspect workspace files via list_dir", "action": "list_dir", "arguments": {"path": "."}}
                        )
                    }
                last_obs = str(tool_msgs[-1].get("content", "") if tool_msgs else "Workspace inspected.")
                return {
                    "content": json.dumps(
                        {"thought": "Summarize observed workspace output", "action": "final", "answer": last_obs[:1200]}
                    )
                }

            harness.completion_fn = _offline_local_completion
            harness.model = "triune-local-fallback"

        if getattr(args, "allowed_tools", None):
            allowed_set = {t.strip() for t in str(args.allowed_tools).split(",") if t.strip()}
            known_tools = set(harness.tools.names())
            if not (allowed_set & known_tools):
                _cli_fail(
                    f"No known tools matched --allowed-tools={args.allowed_tools!r}. Available: {sorted(known_tools)}",
                    code=2,
                    use_json=is_machine_fmt,
                    command="agent",
                )
            for tname in list(harness.tools._tools):
                if tname not in allowed_set:
                    harness.tools._tools.pop(tname, None)
            if args.mode == "pev" and not is_machine_fmt:
                render_notice(
                    "Note: --allowed-tools enforces strict tool scoping in 'react' mode; "
                    "'pev' loop initializes its own default workspace toolbelt.",
                    level="warn",
                )

        auto_persona = str(args.persona or "auto").lower() == "auto"
        auto_depth = str(args.depth or "auto").lower() == "auto"
        if not auto_persona:
            try:
                harness.state_controller.shift(
                    persona=args.persona,
                    reasoning_depth=None if auto_depth else args.depth,
                    reason="CLI flag override",
                )
            except ValueError as exc:
                _cli_fail(str(exc), code=2, use_json=is_machine_fmt, command="agent")
        elif not auto_depth:
            harness.state_controller.shift(reasoning_depth=args.depth, reason="CLI depth override")

        sessions_dir = Path(args.store_dir) / "sessions"
        resumed_history: list[dict[str, str]] = []
        persist_session = bool(getattr(args, "resume_session", None) or getattr(args, "session_id", None) or is_machine_fmt)
        if getattr(args, "resume_session", None):
            try:
                sid = _sanitize_session_id(args.resume_session)
            except ValueError as exc:
                _cli_fail(str(exc), code=2, use_json=is_machine_fmt, command="agent")
            sess_path = sessions_dir / f"{sid}.json"
            if not sess_path.is_file():
                _cli_fail(
                    f"Agent session '{sid}' not found at {sess_path}",
                    code=1,
                    use_json=is_machine_fmt,
                    command="agent",
                )
            try:
                sess_data = load_validated_json_dict(sess_path)
            except Exception as exc:
                _cli_fail(
                    f"Failed to load agent session '{sid}': {exc}",
                    code=1,
                    use_json=is_machine_fmt,
                    command="agent",
                )
            resumed_history = list(sess_data.get("history") or [])
        else:
            if getattr(args, "session_id", None):
                try:
                    sid = _sanitize_session_id(args.session_id)
                except ValueError as exc:
                    _cli_fail(str(exc), code=2, use_json=is_machine_fmt, command="agent")
            else:
                sid = f"session_{int(time.time())}"
            sess_path = sessions_dir / f"{sid}.json"

        def _save_session(hist: list[dict[str, str]]) -> None:
            if not persist_session:
                return
            sessions_dir.mkdir(parents=True, exist_ok=True)
            st_obj = harness.state_controller.state
            atomic_write_json(
                sess_path,
                {
                    "session_id": sid,
                    "persona": st_obj.persona,
                    "reasoning_depth": st_obj.reasoning_depth,
                    "model": harness.model,
                    "updated_at": time.time(),
                    "history": hist,
                },
            )

        model_badge = harness.model or "NO BACKEND (configure Ollama or BYOK)"
        if not is_machine_fmt:
            render_banner(
                "TRIUNE AGENT HARNESS",
                subtitle=f"{harness.state_controller.active_profile.title} · {harness.state_controller.state.reasoning_depth.upper()} · mode={args.mode}",
                badge=model_badge,
            )

        collected_events: list[dict[str, Any]] = []

        def _render_agent_event(ev_type: str, data: dict) -> None:
            if output_format == "stream-json":
                print(json.dumps({"type": ev_type, "session_id": sid, **data}, default=str), flush=True)
                return
            if output_format == "json":
                collected_events.append({"type": ev_type, **data})
                return
            if ev_type == "state_selected":
                sk = ", ".join(data.get("matched_skills") or []) or "none"
                render_notice(
                    f"State: {data['persona']}/{str(data['reasoning_depth']).upper()} ({data['title']}) · Active Skills: {sk}",
                    level="info",
                )
            elif ev_type == "state_shifted":
                render_notice(
                    f"Adaptive State Shift: {data['from']} -> {data['to']} ({data.get('reason', '')})",
                    level="warn",
                )
            elif ev_type == "thought":
                print(f"  [thought #{data['step']}] {str(data['thought'])[:220]}")
            elif ev_type == "tool_start":
                print(f"  -> {data['tool']}({data.get('arguments') or {}})")
            elif ev_type == "tool_result":
                status_icon = "✓" if data.get("ok") else "✗"
                detail = data.get("result_preview") if data.get("ok") else data.get("error")
                print(f"     {status_icon} [{data.get('latency_ms', 0):.1f}ms] {str(detail)[:180]}")
            elif ev_type == "skill_synthesized":
                render_notice(f"Auto-created reusable skill: {data['name']} (`{data['slug']}`)", level="success")
            elif ev_type == "memory_updated":
                render_notice(f"Remembered {data['category']}: {data['value']}", level="info")

        def _print_agent_status() -> None:
            st = harness.state_controller.state
            render_kv_panel(
                "Harness Cognitive Status",
                [
                    ("Workspace", str(ws_path)),
                    ("Execution Mode", args.mode.upper()),
                    ("Backend Model", harness.model or "None configured"),
                    ("Active Persona", f"{st.persona} ({harness.state_controller.active_profile.title})"),
                    ("Reasoning Tier", st.reasoning_depth.upper()),
                    ("Saved Preferences", len(harness.memory.preferences)),
                    ("Learned Lessons", len(harness.memory.lessons)),
                    ("Synthesized Skills", len(harness.skills.skills)),
                    ("Registered Tools", len(harness.tools.names())),
                ],
                border_style="cyan",
            )

        single_goal = args.goal or args.goal_pos
        if single_goal is None or single_goal == "-" or getattr(args, "stdin", False):
            piped_goal = _read_piped_stdin(explicit_stdin=bool(getattr(args, "stdin", False)))
            if piped_goal and piped_goal.strip():
                single_goal = piped_goal.strip()
            elif getattr(args, "stdin", False) or single_goal == "-":
                _cli_fail("No autonomous goal provided on stdin.", code=2, use_json=is_machine_fmt, command="agent")

        if single_goal and (getattr(args, "team", None) or getattr(args, "parallel", False) or args.mode == "pipeline"):
            from lucid_machines.agents import MultiAgentOrchestrator

            raw_team = getattr(args, "team", None) or "architect,engineer,sentinel"
            team_list = [p.strip() for p in str(raw_team).split(",") if p.strip()]
            mode = "parallel" if getattr(args, "parallel", False) else "pipeline"
            try:
                orch = MultiAgentOrchestrator(harness=harness)
                with _quiet_json_stdout(is_machine_fmt):
                    summary = orch.orchestrate_team(
                        single_goal,
                        team_list,
                        mode=mode,
                        model=harness.model,
                        max_workers=max(1, int(getattr(args, "max_workers", 4))),
                        max_steps=max(1, int(args.max_steps)),
                    )
                if output_format == "stream-json":
                    print(
                        json.dumps({"type": "result", "status": "ok", "session_id": sid, **summary}, default=str),
                        flush=True,
                    )
                elif output_format == "json":
                    _emit_json({"status": "ok", "session_id": sid, **summary})
                else:
                    render_table(
                        f"Multi-Agent Team Orchestration ({summary['mode'].upper()})",
                        [
                            ("Agent", "left", "cyan"),
                            ("Role", "left", "white"),
                            ("OK", "center", "green"),
                            ("Output", "left", "white"),
                        ],
                        [
                            [r["agent"], r["role"], "yes" if r["ok"] else "no", str(r["output"])[:160]]
                            for r in summary["results"]
                        ],
                    )
            except Exception as exc:
                _cli_fail(f"Team orchestration failed: {exc}", code=1, use_json=is_machine_fmt, command="agent")
            return
        if single_goal and args.mode == "pev":
            from lucid_machines.agents.loops import run_agent_loop

            def _emit_loop(ev: dict) -> None:
                if output_format == "stream-json":
                    print(json.dumps({"session_id": sid, **ev}, default=str), flush=True)
                    return
                if output_format == "json":
                    collected_events.append(dict(ev))
                    return
                etype = ev.get("type")
                if etype == "plan":
                    for idx, step_txt in enumerate(ev.get("steps") or [], 1):
                        print(f"  [plan #{idx}] {step_txt}")
                elif etype == "thought":
                    print(f"  [thought #{ev.get('step', '?')}] {str(ev.get('thought', ''))[:220]}")
                elif etype == "tool_call":
                    print(f"  -> {ev.get('tool')}({ev.get('arguments') or {}})")
                elif etype == "tool_result":
                    icon = "✓" if ev.get("ok") else "✗"
                    detail = ev.get("result_preview") if ev.get("ok") else ev.get("error")
                    print(f"     {icon} {str(detail)[:180]}")

            try:
                res = run_agent_loop(
                    harness=harness,
                    goal=single_goal,
                    mode="pev",
                    autonomy="autonomous_sandbox" if not args.read_only else "supervised",
                    max_steps=args.max_steps,
                    emit=_emit_loop,
                )
                if res.get("error"):
                    _cli_fail(
                        f"Agent PEV execution failed: {res['error']}",
                        code=1,
                        use_json=is_machine_fmt,
                        command="agent",
                    )
                resumed_history.append({"role": "user", "content": single_goal})
                resumed_history.append({"role": "assistant", "content": str(res.get("answer") or "")})
                _save_session(resumed_history)
                if output_format == "stream-json":
                    print(
                        json.dumps({"type": "result", "status": "ok", "session_id": sid, **res}, default=str),
                        flush=True,
                    )
                elif output_format == "json":
                    _emit_json({"status": "ok", "session_id": sid, "events": collected_events, **res})
                else:
                    render_kv_panel(
                        f"PEV Result ({res.get('steps', 0)} steps)",
                        [
                            ("Workspace", str(ws_path)),
                            ("Response", str(res.get("answer") or "")),
                        ],
                        border_style="green",
                    )
            except SystemExit:
                raise
            except Exception as exc:
                _cli_fail(f"Agent execution failed: {exc}", code=1, use_json=is_machine_fmt, command="agent")
            return
        if single_goal:
            try:
                res = harness.run_goal(
                    single_goal,
                    max_steps=args.max_steps,
                    on_event=_render_agent_event,
                    auto_persona=auto_persona,
                    auto_depth=auto_depth,
                    auto_crystallize=not bool(args.no_auto_skill),
                    conversation_history=resumed_history if resumed_history else None,
                )
                resumed_history.append({"role": "user", "content": single_goal})
                resumed_history.append({"role": "assistant", "content": str(res.get("answer") or "")})
                _save_session(resumed_history)
                if output_format == "stream-json":
                    print(
                        json.dumps({"type": "result", "status": "ok", "session_id": sid, **res}, default=str),
                        flush=True,
                    )
                elif output_format == "json":
                    _emit_json({"status": "ok", "session_id": sid, "events": collected_events, **res})
                else:
                    render_kv_panel(
                        f"Result ({res['persona']} · {str(res['reasoning_depth']).upper()} · {res['steps']} tool calls)",
                        [
                            ("Response", res["answer"]),
                            ("Created Skills", ", ".join(res["created_skills"]) or "none"),
                            ("Lessons Learned", len(res["learned_lessons"])),
                        ],
                        border_style="green",
                    )
            except Exception as exc:
                _cli_fail(f"Agent execution failed: {exc}", code=1, use_json=is_machine_fmt, command="agent")
            return

        _print_agent_status()
        render_notice(
            "Interactive Agent REPL ready. Commands: /status, /persona <name>, /state <reflex|limbic|cortex>, /skills, /memory, /memory delete <key>, /memory clear, /tools, /learn <note>, /exit",
            level="info",
        )
        history: list[dict[str, str]] = list(resumed_history)
        tty = is_tty()
        prompt_str = "\n\033[1;33magent ❯ \033[0m" if tty else "\nGoal: "

        while True:
            try:
                raw_goal = input(prompt_str)
            except (EOFError, KeyboardInterrupt):
                print()
                render_notice("Agent session ended.", level="info")
                return

            text = raw_goal.strip()
            if not text:
                continue
            low = text.lower()
            if low in {"exit", "quit", "/exit", "/quit"}:
                render_notice("Goodbye.", level="info")
                return
            if low == "/help":
                render_table(
                    "Agent Harness Slash Commands",
                    [("Command", "left", "bold cyan"), ("Effect", "left", "white")],
                    [
                        ["/status", "Inspect active persona, state tier, memory & skills"],
                        ["/persona <name>", f"Switch persona ({', '.join(sorted(harness.state_controller.personas))})"],
                        ["/state <tier>", "Set reasoning tier: reflex | limbic | cortex"],
                        ["/skills", "List authored and auto-learned skills"],
                        ["/memory", "Show stored user wants/needs and learned lessons"],
                        ["/memory delete <key>", "Delete a specific stored user preference"],
                        ["/memory clear", "Clear all stored preferences and lessons"],
                        ["/tools", "Inspect learned tool reliability, avg latency & tips"],
                        ["/learn <text>", "Pin a preference or constraint into user memory"],
                        ["/clear", "Clear current turn conversation context"],
                        ["/exit", "Exit interactive harness"],
                    ],
                )
                continue
            if low == "/status":
                _print_agent_status()
                continue
            if low == "/clear":
                history.clear()
                render_notice("Conversation context cleared (long-term memory/skills preserved).", level="success")
                continue
            if low == "/skills":
                sk_list = harness.skills.summarize_catalog()
                if not sk_list:
                    render_notice(
                        "No synthesized skills yet. Give the agent multi-step goals or call create_skill.", level="info"
                    )
                else:
                    render_table(
                        f"Skill Library ({len(sk_list)})",
                        [
                            ("Slug", "left", "bold cyan"),
                            ("Uses", "right", "white"),
                            ("Success", "right", "green"),
                            ("Description", "left", "white"),
                        ],
                        [
                            [s["slug"], s["uses"], f"{int(s['success_rate'] * 100)}%", s["description"][:70]]
                            for s in sk_list
                        ],
                    )
                continue
            if low == "/memory clear":
                removed = harness.memory.clear_all()
                render_notice(
                    f"Cleared agent memory (preferences={removed['preferences']}, lessons={removed['lessons']}, tool_stats={removed['tool_stats']}).",
                    level="success",
                )
                continue
            if low.startswith("/memory delete "):
                target_key = text.split(None, 2)[2].strip() if len(text.split(None, 2)) >= 3 else ""
                if target_key and harness.memory.delete_preference(target_key):
                    render_notice(f"Deleted preference '{target_key}'.", level="success")
                else:
                    render_notice(f"Preference key '{target_key}' not found.", level="warn")
                continue
            if low == "/memory":
                pref_rows = [[p.category, p.key, p.value] for p in harness.memory.preferences.values()]
                les_rows = [[l.lesson_id, f"{l.confidence:.2f}", l.lesson[:75]] for l in harness.memory.lessons[-12:]]
                if pref_rows:
                    render_table(
                        "User Wants & Constraints",
                        [("Cat", "left", "cyan"), ("Key", "left", "white"), ("Value", "left", "green")],
                        pref_rows,
                    )
                if les_rows:
                    render_table(
                        "Self-Learned Lessons",
                        [("ID", "left", "cyan"), ("Conf", "right", "yellow"), ("Lesson", "left", "white")],
                        les_rows,
                    )
                if not pref_rows and not les_rows:
                    render_notice("No user preferences or lessons stored yet.", level="info")
                continue
            if low == "/tools":
                t_rows = []
                for tname in harness.tools.names():
                    st = harness.memory.tool_stats.get(tname)
                    if st and st.calls > 0:
                        t_rows.append(
                            [
                                tname,
                                st.calls,
                                f"{int(st.success_rate * 100)}%",
                                f"{st.avg_latency_ms:.0f}ms",
                                (st.learned_tips[-1][:55] if st.learned_tips else "-"),
                            ]
                        )
                    else:
                        t_rows.append([tname, 0, "-", "-", "unmeasured"])
                render_table(
                    f"Tool Proficiency Matrix ({len(t_rows)} tools)",
                    [
                        ("Tool", "left", "bold cyan"),
                        ("Calls", "right", "white"),
                        ("Reliability", "right", "green"),
                        ("Avg ms", "right", "yellow"),
                        ("Learned Tip", "left", "dim"),
                    ],
                    t_rows,
                )
                continue
            if low.startswith("/persona "):
                target_p = text.split(None, 1)[1].strip()
                try:
                    st = harness.state_controller.shift(persona=target_p, reason="user slash command")
                    auto_persona = False
                    render_notice(f"Persona shifted to {st.persona} ({st.reasoning_depth.upper()})", level="success")
                except ValueError as exc:
                    render_notice(str(exc), level="error")
                continue
            if low.startswith("/state "):
                target_d = low.split(None, 1)[1].strip()
                if target_d in {"reflex", "limbic", "cortex"}:
                    st = harness.state_controller.shift(reasoning_depth=target_d, reason="user slash command")
                    auto_depth = False
                    render_notice(f"Reasoning depth set to {st.reasoning_depth.upper()}", level="success")
                else:
                    render_notice("Usage: /state <reflex|limbic|cortex>", level="warn")
                continue
            if low.startswith("/learn "):
                note = text.split(None, 1)[1].strip()
                pref = harness.memory.remember_preference(
                    f"user_note_{len(harness.memory.preferences) + 1}", note, category="preference", priority=9
                )
                render_notice(f"Pinned user preference [{pref.key}]: {pref.value}", level="success")
                continue

            try:
                res = harness.run_goal(
                    text,
                    max_steps=args.max_steps,
                    on_event=_render_agent_event,
                    auto_persona=auto_persona,
                    auto_depth=auto_depth,
                    auto_crystallize=not bool(args.no_auto_skill),
                    conversation_history=history,
                )
                history.append({"role": "user", "content": text})
                history.append({"role": "assistant", "content": res["answer"]})
                _save_session(history)
                render_rule(f"{res['persona']} · {res['reasoning_depth'].upper()}")
                print(res["answer"])
            except Exception as exc:
                render_notice(f"Agent error: {exc}", level="error")

    elif args.command in ("eval", "evaluate"):
        import math

        use_json = bool(getattr(args, "json", False))
        try:
            ckpt_path = resolve_default_checkpoint(args.checkpoint)
        except FileNotFoundError as exc:
            _cli_fail(str(exc), code=1, use_json=use_json, command="eval")

        tok_path = Path(args.tokenizer)
        if not tok_path.is_file() and args.tokenizer == "triune_tokenizer.json":
            repo_tok = Path(__file__).resolve().parent.parent / "triune_tokenizer.json"
            if repo_tok.is_file():
                tok_path = repo_tok
        if not tok_path.is_file():
            _cli_fail(f"Tokenizer not found: {args.tokenizer}", code=1, use_json=use_json, command="eval")

        import torch

        from lucid_machines.configs.config import build_config
        from lucid_machines.data import build_dataloader, load_tokenizer
        from lucid_machines.model import build_model
        from lucid_machines.model.transformer import TriuneTransformer
        from lucid_machines.recipes import bf16_autocast
        from lucid_machines.trainer.evaluation import run_eval, run_sample, run_zero_shot_eval

        dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            tokenizer = load_tokenizer(tok_path)
        except Exception as exc:
            _cli_fail(f"Failed to load tokenizer '{tok_path}': {exc}", code=1, use_json=use_json, command="eval")
        vocab_size = tokenizer.get_vocab_size()
        pad_id = tokenizer.token_to_id("[PAD]")
        sep_id = tokenizer.token_to_id("[SEP]")
        if pad_id is None:
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        if sep_id is None:
            sep_id = getattr(tokenizer, "eos_token_id", None) or pad_id

        try:
            with _quiet_json_stdout(use_json), _maybe_spinner(f"Loading checkpoint {ckpt_path} on {dev}...", use_json):
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                ckpt_cfg = payload.get("config") if isinstance(payload, dict) else None
                cfg_overrides = dict(ckpt_cfg) if isinstance(ckpt_cfg, dict) else {}
                cfg_overrides["vocab_size"] = vocab_size

                if any(
                    k in cfg_overrides
                    for k in (
                        "top_k_experts",
                        "shared_expert",
                        "shared_expert_scale",
                        "expert_hidden_multiplier",
                        "rope_max_seq_len",
                    )
                ):
                    tmp_cfg = build_config(cfg_overrides)
                    model = TriuneTransformer(
                        vocab_size=int(tmp_cfg.get("vocab_size", vocab_size)),
                        hidden_dim=int(tmp_cfg.get("hidden_dim", 1536)),
                        num_layers=int(tmp_cfg.get("num_layers", 24)),
                        num_heads=int(tmp_cfg.get("num_heads", 12)),
                        head_dim=int(tmp_cfg.get("head_dim", 128)),
                        num_experts=int(tmp_cfg.get("num_experts", 8)),
                        top_k_experts=int(tmp_cfg.get("top_k_experts", 1)),
                        shared_expert=bool(tmp_cfg.get("shared_expert", True)),
                        shared_expert_scale=float(tmp_cfg.get("shared_expert_scale", 0.5)),
                        expert_hidden_multiplier=int(tmp_cfg.get("expert_hidden_multiplier", 6)),
                        rope_max_seq_len=int(tmp_cfg.get("rope_max_seq_len", 8192)),
                        router_prefix_layers=int(tmp_cfg.get("router_prefix_layers", 3)),
                        reflex_exit_layer=int(tmp_cfg.get("reflex_exit_layer", 6)),
                        limbic_exit_layer=int(tmp_cfg.get("limbic_exit_layer", 16)),
                        target_depth_dist=tuple(tmp_cfg.get("target_depth_dist", (0.34, 0.33, 0.33))),
                        balance_coef=float(tmp_cfg.get("balance_coef", 0.30)),
                        use_fp4=bool(tmp_cfg.get("use_fp4", False)),
                        use_fp8=bool(tmp_cfg.get("use_fp8", False)),
                    )
                else:
                    tmp_cfg = build_config(cfg_overrides)
                    model = build_model(tmp_cfg)

                state_dict = (
                    payload.get("model_state") or payload.get("model_state_dict") or payload
                    if isinstance(payload, dict)
                    else payload
                )
                if not isinstance(state_dict, dict):
                    raise ValueError(f"Unsupported checkpoint payload type: {type(state_dict).__name__}")
                model.load_state_dict(state_dict, strict=False)
                model = model.to(dev)
        except Exception as exc:
            _cli_fail(f"Failed to load checkpoint '{ckpt_path}': {exc}", code=1, use_json=use_json, command="eval")

        depth_map = {"auto": None, "reflex": 0, "limbic": 1, "cortex": 2}
        force_depth = depth_map.get(args.depth)

        if getattr(args, "tasks", None):
            try:
                with _quiet_json_stdout(use_json), _maybe_spinner(f"Evaluating tasks ({args.tasks})...", use_json):
                    task_eval = run_zero_shot_eval(
                        model,
                        tokenizer,
                        args.tasks,
                        device=dev,
                        force_depth=force_depth,
                        limit=getattr(args, "limit", None),
                        num_trials=int(getattr(args, "num_trials", 16)),
                    )
            except Exception as exc:
                _cli_fail(f"Task evaluation failed: {exc}", code=1, use_json=use_json, command="eval")

            if use_json:
                _emit_json(
                    {
                        "status": "ok",
                        "checkpoint": str(ckpt_path),
                        "device": str(dev),
                        "depth": args.depth,
                        **task_eval,
                    }
                )
                return

            render_banner("TRIUNE TASK EVALUATION REPORT", subtitle=str(ckpt_path), badge=args.depth.upper())
            task_rows: list[list[Any]] = []
            for tname, tmetrics in task_eval["results"].items():
                acc = tmetrics.get("choice_accuracy", tmetrics.get("accuracy", "-"))
                acc_str = f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else str(acc)
                base_v = tmetrics.get("random_baseline", "-")
                base_str = f"{base_v * 100:.1f}%" if isinstance(base_v, (int, float)) else str(base_v)
                n_ex = tmetrics.get("num_trials", tmetrics.get("num_examples", "-"))
                lp_v = tmetrics.get("mean_target_logprob", "-")
                lp_str = f"{lp_v:.4f}" if isinstance(lp_v, (int, float)) else str(lp_v)
                task_rows.append([tname, tmetrics.get("evaluator", "-"), n_ex, acc_str, base_str, lp_str])
            render_table(
                "Zero-Shot & MQAR Task Evaluation",
                columns=[
                    ("Task", "left", "bold cyan"),
                    ("Evaluator", "left", "dim"),
                    ("Examples", "right", "white"),
                    ("Accuracy", "right", "green"),
                    ("Baseline", "right", "yellow"),
                    ("Mean LogProb", "right", "cyan"),
                ],
                rows=task_rows,
            )
            return

        max_batches = max(1, int(args.max_steps))
        cfg_overrides["eval_batches"] = max_batches
        local_sample = Path("data/fineweb_sample.jsonl")
        if not local_sample.is_file():
            repo_sample = Path(__file__).resolve().parent.parent / "data" / "fineweb_sample.jsonl"
            if repo_sample.is_file():
                local_sample = repo_sample

        using_local_fallback = False
        if args.dataset:
            cfg_overrides["dataset_name"] = args.dataset
        else:
            ckpt_ds = cfg_overrides.get("dataset_name")
            if (not ckpt_ds or not Path(str(ckpt_ds)).exists()) and local_sample.is_file():
                cfg_overrides["dataset_name"] = str(local_sample)
                cfg_overrides["dataset_config"] = None
                using_local_fallback = True
        if getattr(args, "dataset_config", None):
            cfg_overrides["dataset_config"] = args.dataset_config

        raw_buf = int(cfg_overrides.get("shuffle_buffer") or 4096)
        eval_bs = max(1, int(cfg_overrides.get("batch_size") or 8))
        cfg_overrides["shuffle_buffer"] = min(raw_buf, max(16, max_batches * eval_bs))
        cfg = build_config(cfg_overrides)

        def _collect_eval_batches(eval_cfg: dict[str, Any]) -> list[Any]:
            eval_loader = build_dataloader(tokenizer, eval_cfg, sep_id, is_holdout=True)
            collected: list[Any] = []
            for idx, batch in enumerate(eval_loader):
                collected.append(batch)
                if idx + 1 >= max_batches:
                    break
            return collected

        try:
            with _quiet_json_stdout(use_json), _maybe_spinner("Loading evaluation batches...", use_json):
                eval_batches = _collect_eval_batches(cfg)
                if not eval_batches and using_local_fallback and int(cfg.get("batch_size", 1)) > 1:
                    retry_bs = min(4, int(cfg["batch_size"]))
                    retry_cfg = build_config(
                        {
                            **cfg,
                            "batch_size": retry_bs,
                            "shuffle_buffer": min(16, max_batches * retry_bs),
                        }
                    )
                    eval_batches = _collect_eval_batches(retry_cfg)
        except Exception as exc:
            _cli_fail(f"Failed to load evaluation dataset: {exc}", code=1, use_json=use_json, command="eval")

        if not eval_batches:
            _cli_fail("Evaluation dataset yielded zero batches.", code=1, use_json=use_json, command="eval")

        eval_ctx = type(
            "_EvalContext",
            (),
            {
                "model": model,
                "eval_batches": eval_batches,
                "device": dev,
                "bf16_autocast": lambda _self=None: bf16_autocast(dev.type),
                "pad_token_id": pad_id,
                "sep_token_id": sep_id,
                "vocab_size": vocab_size,
                "tokenizer": tokenizer,
                "sample_prompt": args.prompt or "The meaning of life is",
            },
        )()

        try:
            with _quiet_json_stdout(use_json):
                avg_loss = run_eval(eval_ctx, force_depth=force_depth)
                ppl = math.exp(min(20.0, avg_loss))
                sample_text = (
                    run_sample(eval_ctx, prompt=args.prompt, max_new_tokens=args.max_tokens) if args.prompt else None
                )
        except Exception as exc:
            _cli_fail(f"Evaluation failed: {exc}", code=1, use_json=use_json, command="eval")

        if use_json:
            payload_out: dict[str, Any] = {
                "status": "ok",
                "checkpoint": str(ckpt_path),
                "device": str(dev),
                "depth": args.depth,
                "dataset": cfg.get("dataset_name"),
                "batches_evaluated": len(eval_batches),
                "eval_loss": round(avg_loss, 6),
                "perplexity": round(ppl, 4),
            }
            if args.prompt:
                payload_out["prompt"] = args.prompt
                payload_out["sample_output"] = sample_text
            _emit_json(payload_out)
            return

        rows: list[tuple[str, object]] = [
            ("Checkpoint", str(ckpt_path)),
            ("Device", str(dev)),
            ("Depth Route", args.depth.upper()),
            ("Batches Evaluated", len(eval_batches)),
            ("Eval Loss", f"{avg_loss:.4f}"),
            ("Perplexity", f"{ppl:.2f}"),
        ]
        if args.prompt:
            rows.append(("Prompt", args.prompt))
            rows.append(("Sample Output", sample_text))
        render_banner("TRIUNE EVALUATION REPORT", subtitle=str(ckpt_path), badge=args.depth.upper())
        render_kv_panel("Evaluation Metrics", rows, border_style="green")

    elif args.command == "config":
        import json as _json

        from lucid_machines.configs.config import build_config, validate_config
        from lucid_machines.model.base import _LEGACY_PRESET_ALIASES, CANONICAL_MODEL_SPECS, ModelSpec, get_model_spec

        cfg_dir = Path("models/configs")
        if not cfg_dir.is_dir():
            cfg_dir = Path(__file__).resolve().parent.parent / "models" / "configs"

        def _load_cfg_target(raw_target: str) -> tuple[str, dict]:
            t = str(raw_target or "").strip()
            if not t:
                raise FileNotFoundError("No config name or file path provided.")
            if t.lower() in ("session", "session_state", "session_state.json"):
                sp = Path("checkpoints/session_state.json")
                if not sp.is_file():
                    raise FileNotFoundError("checkpoints/session_state.json not found.")
                return "session_state", _json.loads(sp.read_text(encoding="utf-8-sig"))
            cand = Path(t)
            if cand.is_file():
                return cand.stem, _json.loads(cand.read_text(encoding="utf-8-sig"))
            for stem in (t, t.lower()):
                f = cfg_dir / (stem if stem.endswith(".json") else f"{stem}.json")
                if f.is_file():
                    return f.stem, _json.loads(f.read_text(encoding="utf-8-sig"))
            low = t.lower()
            if low in CANONICAL_MODEL_SPECS or low in _LEGACY_PRESET_ALIASES:
                spec = get_model_spec(low)
                return spec.model_id, spec.to_dict()
            raise FileNotFoundError(f"Config '{raw_target}' not found in models/configs/ or CANONICAL_MODEL_SPECS.")

        if args.config_action == "list":
            rows = []
            if cfg_dir.is_dir():
                for f in sorted(cfg_dir.glob("*.json")):
                    try:
                        d = _json.loads(f.read_text(encoding="utf-8-sig"))
                        rows.append(
                            [
                                f.stem,
                                d.get("architecture", "TriuneTransformer"),
                                f"{d.get('num_layers', '?')}L / {d.get('hidden_dim', '?')}D",
                                f"{d.get('num_experts', '?')}E (k={d.get('top_k_experts', 1)})",
                                d.get("rope_max_seq_len", 8192),
                                str(f),
                            ]
                        )
                    except Exception:
                        rows.append([f.stem, "invalid-json", "-", "-", "-", str(f)])
            render_table(
                f"Model Configurations ({len(rows)})",
                columns=[
                    ("Config ID", "left", "bold cyan"),
                    ("Architecture", "left", "white"),
                    ("Shape", "left", "yellow"),
                    ("MoE", "left", "green"),
                    ("RoPE Max", "right", "white"),
                    ("Path", "left", "dim"),
                ],
                rows=rows,
                caption="Tip: run `config validate --all` or `config show junkie-3b`.",
            )
        elif args.config_action == "show":
            use_json = bool(getattr(args, "json", False))
            target = args.file or args.target
            if not target:
                _cli_fail(
                    "Provide a config preset name, 'session', or --file <path>.",
                    code=2,
                    use_json=use_json,
                    command="config show",
                )
            try:
                name, raw_cfg = _load_cfg_target(target)
            except Exception as exc:
                _cli_fail(str(exc), code=1, use_json=use_json, command="config show")
            if use_json:
                print(_json.dumps(raw_cfg, indent=2))
            else:
                render_kv_panel(
                    f"Config · {name}",
                    [(k, _json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in raw_cfg.items()],
                    border_style="cyan",
                )
        elif args.config_action == "validate":
            targets: list[str] = []
            if args.target:
                targets = [args.target]
            elif cfg_dir.is_dir():
                targets = [str(f) for f in sorted(cfg_dir.glob("*.json"))]
            else:
                targets = list(CANONICAL_MODEL_SPECS.keys())

            failures = 0
            rows = []
            for t in targets:
                try:
                    name, raw_cfg = _load_cfg_target(t)
                    merged = build_config(
                        {
                            k: raw_cfg[k]
                            for k in (
                                "vocab_size",
                                "hidden_dim",
                                "num_layers",
                                "num_heads",
                                "head_dim",
                                "num_experts",
                                "router_prefix_layers",
                                "reflex_exit_layer",
                                "limbic_exit_layer",
                                "target_depth_dist",
                                "balance_coef",
                                "use_fp4",
                                "use_fp8",
                            )
                            if k in raw_cfg and raw_cfg[k] is not None
                        }
                    )
                    validate_config(merged)
                    spec = ModelSpec.from_dict({"model_id": name, **raw_cfg})
                    if not (
                        0
                        < spec.router_prefix_layers
                        < spec.reflex_exit_layer
                        < spec.limbic_exit_layer
                        < spec.num_layers
                    ):
                        raise ValueError("Invalid Triune layer hierarchy")
                    if spec.hidden_dim != spec.num_heads * spec.head_dim:
                        raise ValueError("hidden_dim != num_heads * head_dim")
                    tot, act = spec.compute_exact_params()
                    rows.append(
                        [
                            name,
                            "VALID",
                            _format_param_count(tot),
                            _format_param_count(act),
                            f"top_k={spec.top_k_experts}",
                        ]
                    )
                except Exception as exc:
                    failures += 1
                    rows.append([str(t), "INVALID", "-", "-", str(exc)])
            render_table(
                f"Config Validation ({len(rows)} checked, {failures} failed)",
                columns=[
                    ("Target", "left", "bold cyan"),
                    ("Status", "left", "green" if failures == 0 else "red"),
                    ("Total", "right", "white"),
                    ("Active", "right", "cyan"),
                    ("Notes", "left", "dim"),
                ],
                rows=rows,
            )
            if failures:
                raise SystemExit(1)
        else:
            _get_subparser(parser, "config").print_help()
            raise SystemExit(2)

    elif args.command == "workspace":
        import json as _json

        from lucid_machines.agents.tools import _has_traversal_syntax

        def _ws_root(raw_root: str) -> Path:
            p = Path(raw_root or ".").expanduser().resolve(strict=False)
            if not p.is_dir():
                render_notice(f"Workspace directory not found: {raw_root}", level="error")
                raise SystemExit(1)
            return p

        def _ws_confine(ws_root: Path, rel_path: str) -> Path:
            clean = str(rel_path or "").strip()
            if not clean or _has_traversal_syntax(clean):
                render_notice(f"Rejected invalid or traversal path: {rel_path!r}", level="error")
                raise SystemExit(2)
            cand = (ws_root / clean).resolve(strict=False)
            if not (cand == ws_root or cand.is_relative_to(ws_root)):
                render_notice(f"Path escapes workspace root: {rel_path!r}", level="error")
                raise SystemExit(2)
            return cand

        if args.ws_action == "status":
            import subprocess

            ws_root = _ws_root(args.path)
            branch = "-"
            dirty_cnt = 0
            if (ws_root / ".git").exists():
                try:
                    br = subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=str(ws_root),
                        capture_output=True,
                        text=True,
                        timeout=4,
                    )
                    if br.returncode == 0:
                        branch = br.stdout.strip()
                    st = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=str(ws_root),
                        capture_output=True,
                        text=True,
                        timeout=4,
                    )
                    if st.returncode == 0:
                        dirty_cnt = len([ln for ln in st.stdout.splitlines() if ln.strip()])
                except Exception:
                    pass
            cfg_count = (
                len(list((ws_root / "models" / "configs").glob("*.json")))
                if (ws_root / "models" / "configs").is_dir()
                else 0
            )
            ckpt_count = len(list((ws_root / "checkpoints").glob("*.pt"))) if (ws_root / "checkpoints").is_dir() else 0
            ws_rows: list[tuple[str, object]] = [
                ("Root", str(ws_root)),
                ("Git Branch", branch),
                ("Git Dirty Entries", dirty_cnt),
                ("Model Configs", cfg_count),
                ("Checkpoints (.pt)", ckpt_count),
            ]
            sess = ws_root / "checkpoints" / "session_state.json"
            if sess.is_file():
                try:
                    sd = _json.loads(sess.read_text(encoding="utf-8"))
                    ws_rows.append(("Session Step", sd.get("step", "?")))
                except Exception:
                    pass
            render_kv_panel("Workspace Status", ws_rows, border_style="cyan")
        elif args.ws_action == "tree":
            ws_root = _ws_root(args.path)
            max_depth = max(1, int(args.depth))
            skip_dirs = {
                ".git",
                "node_modules",
                "__pycache__",
                "studio_env",
                ".venv",
                "venv",
                ".pytest_cache",
                ".ruff_cache",
            }
            lines: list[str] = [str(ws_root)]

            def _walk(cur: Path, depth: int, prefix: str) -> None:
                if depth > max_depth:
                    return
                try:
                    entries = sorted(cur.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
                except OSError:
                    return
                entries = [e for e in entries if e.name not in skip_dirs]
                for idx, entry in enumerate(entries[:80]):
                    connector = "└── " if idx == len(entries) - 1 else "├── "
                    suffix = "/" if entry.is_dir() else ""
                    lines.append(f"{prefix}{connector}{entry.name}{suffix}")
                    if entry.is_dir() and depth < max_depth:
                        ext = "    " if idx == len(entries) - 1 else "│   "
                        _walk(entry, depth + 1, prefix + ext)

            _walk(ws_root, 1, "")
            print("\n".join(lines))
        elif args.ws_action == "read":
            ws_root = _ws_root(args.path)
            rel = args.file or args.file_pos
            if not rel:
                render_notice("Provide a relative file path to read.", level="error")
                raise SystemExit(2)
            target_file = _ws_confine(ws_root, rel)
            if not target_file.is_file():
                render_notice(f"File not found inside workspace: {rel}", level="error")
                raise SystemExit(1)
            raw_bytes = target_file.read_bytes()
            if b"\x00" in raw_bytes[:4096]:
                render_notice(f"Refusing to display binary file: {rel}", level="error")
                raise SystemExit(2)
            text_lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
            start_idx = max(1, int(args.start))
            end_idx = min(len(text_lines), max(start_idx, int(args.end)))
            for ln_no in range(start_idx, end_idx + 1):
                print(f"{ln_no:>5} | {text_lines[ln_no - 1]}")
        elif args.ws_action == "init":
            ws_root = Path(args.path or ".").expanduser().resolve(strict=False)
            ws_root.mkdir(parents=True, exist_ok=True)
            created: list[str] = []
            for sub in ("models/configs", "checkpoints", "exports", "data"):
                d = ws_root / sub
                if not d.exists():
                    d.mkdir(parents=True, exist_ok=True)
                    created.append(sub)
            render_notice(
                f"Workspace ready at {ws_root}"
                + (f" (created: {', '.join(created)})" if created else " (all directories present)"),
                level="success",
            )
        else:
            _get_subparser(parser, "workspace").print_help()
            raise SystemExit(2)

    else:
        render_help_overview(prog=prog)


if __name__ == "__main__":
    main()
