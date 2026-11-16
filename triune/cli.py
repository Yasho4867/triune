"""Unified Command-Line Interface for Triune Engine.

Unified CLI and backend engine powering terminal workflows, weight
transmutation, local Ollama execution, fine-tuning, and Triune Studio.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from triune.cli_ui import (
    ensure_utf8_stdio,
    is_tty,
    render_banner,
    render_help_overview,
    render_kv_panel,
    render_notice,
    render_rule,
    render_table,
    render_telemetry_bar,
    route_badge_ansi,
    status_spinner,
)


def _format_param_count(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the unified Triune CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="triune",
        description="Triune AI Unified CLI — 3-Tier Mixture-of-Experts, Weight Transmuter & Local Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available CLI Commands")

    # 1. Studio command
    studio_parser = subparsers.add_parser("studio", help="Launch Triune Studio Native Desktop / Browser Application")
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

    # 3. Chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive terminal chat session with Triune, HF/transmuted models, or local Ollama"
    )
    chat_parser.add_argument("--model", default=None, help="HF model repo ID, local HF dir, NVFP4 cache, or ollama:tag")
    chat_parser.add_argument("--checkpoint", default="checkpoints_full/best.pt", help="Model checkpoint path")
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
    infer_parser.add_argument(
        "--model", default=None, help="HF model repo ID, local HF dir, NVFP4 cache, or ollama:tag"
    )
    infer_parser.add_argument("--checkpoint", default="checkpoints_full/best.pt", help="Model checkpoint path")
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
    finetune_parser.add_argument("--dataset", default="data/fineweb_sample.jsonl", help="Dataset path or HF dataset ID")
    finetune_parser.add_argument("--dataset-split", default="train", help="Dataset split when using HF Hub datasets")
    finetune_parser.add_argument("--max-samples", type=int, default=2000, help="Max streaming samples from HF dataset")
    finetune_parser.add_argument("--checkpoint", default=None, help="Base model checkpoint path (optional)")
    finetune_parser.add_argument("--rank", type=int, default=16, help="LoRA rank dimension r (default: 16)")
    finetune_parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha scaling (default: 32.0)")
    finetune_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    finetune_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    finetune_parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per step (default: 4)")
    finetune_parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    finetune_parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    finetune_parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (optional)")
    finetune_parser.add_argument(
        "--output-dir", default="checkpoints/adapters/lora_finetuned", help="Output directory for adapter weights"
    )
    finetune_parser.add_argument(
        "--quantization", default="4-bit NF4", help="Quantization precision (e.g. '4-bit NVFP4', 'BF16')"
    )
    finetune_parser.add_argument("--qlora", action="store_true", help="Enable true packed 4-bit NVFP4 base QLoRA")
    finetune_parser.add_argument("--stream", action="store_true", help="Enable RAM-to-VRAM LayerStreamingEngine")
    finetune_parser.add_argument(
        "--stream-chunk", type=int, default=1, help="Layer chunk size for LayerStreamingEngine"
    )
    finetune_parser.add_argument(
        "--disk-stream", action="store_true", help="Enable NVMe NVFP4 shard streaming (14B-27B+)"
    )
    finetune_parser.add_argument("--disk-cache", default=None, help="NVFP4 disk cache directory")
    finetune_parser.add_argument(
        "--quantize-lm-head", action="store_true", help="Pack lm_head into NVFP4 in disk cache"
    )
    finetune_parser.add_argument("--checkpointing", action="store_true", help="Enable activation checkpointing")

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

    ckpt_sub.add_parser("list", help="List local checkpoints")

    ckpt_export_parser = ckpt_sub.add_parser("export", help="Export checkpoint to SafeTensors, GGUF, or ONNX")
    ckpt_export_parser.add_argument("--checkpoint", required=True, help="Source checkpoint path")
    ckpt_export_parser.add_argument("--output", default=None, help="Destination file path")
    ckpt_export_parser.add_argument(
        "--format", default="safetensors", choices=("safetensors", "gguf", "onnx", "hf"), help="Target format"
    )

    # 8. Top-Level Export / Convert command
    export_parser = subparsers.add_parser(
        "export",
        aliases=["convert"],
        help="Export model checkpoint directly to SafeTensors, GGUF, ONNX, or Hugging Face format",
    )
    export_parser.add_argument("--checkpoint", required=True, help="Source checkpoint path")
    export_parser.add_argument("--output", default=None, help="Destination output path")
    export_parser.add_argument(
        "--format", default="safetensors", choices=("safetensors", "gguf", "onnx", "hf"), help="Target format"
    )

    # 9. Model Zoo command
    subparsers.add_parser("models", help="Display architecture presets from the Triune Model Zoo")

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
        choices=("all", "kernels", "optimizers", "streaming", "context", "routing", "matrix"),
        help="Optimization benchmark suite",
    )
    bench_parser.add_argument("--checkpoint", default="checkpoints_full/best.pt", help="Checkpoint path")
    bench_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    bench_parser.add_argument("--prompt", default="The meaning of life is", help="Prompt text")
    bench_parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    bench_parser.add_argument("--device", default=None, help="Device (cuda / cpu)")
    bench_parser.add_argument(
        "--depth", default="auto", choices=("auto", "reflex", "limbic", "cortex"), help="Exit depth routing"
    )
    bench_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")

    # 12. GPU check command
    subparsers.add_parser("gpucheck", help="Print CUDA hardware and driver diagnostics")

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
        "install", help="Launch Triune Studio Setup Wizard or automated environment provisioner"
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
    subparsers.add_parser(
        "doctor",
        aliases=["info"],
        help="Run comprehensive end-to-end diagnostics on GPU, C/AVX2 engine, optimizers, Ollama & Studio backend",
    )

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
    prof_parser.add_argument("--checkpoint", default="checkpoints_full/latest.pt", help="Checkpoint path")
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

    parser._triune_subparsers = {  # type: ignore[attr-defined]
        "adapter": adapter_parser,
        "checkpoint": ckpt_parser,
    }
    return parser


def main() -> None:
    """Dispatch CLI commands with Rich formatting and unified engine orchestration."""
    ensure_utf8_stdio()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        render_help_overview()
        return

    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "studio":
        from triune.desktop import launch_desktop_app

        mode_name = "Browser UI" if args.browser else ("Native Host Backend" if args.native_host else "Desktop IDE")
        render_banner("TRIUNE STUDIO", subtitle=f"Launching {mode_name}", badge=f"port {args.port}")
        launch_desktop_app(
            port=args.port,
            use_browser=args.browser,
            native_host=getattr(args, "native_host", False),
            allow_wsl=getattr(args, "use_wsl", False),
        )

    elif args.command == "serve":
        import os

        from triune.api import run_server

        if args.api_key:
            os.environ["TRIUNE_API_TOKEN"] = args.api_key
        render_banner(
            "TRIUNE API SERVER", subtitle="OpenAI-Compatible & Studio Backend", badge=f"http://{args.host}:{args.port}"
        )
        render_kv_panel(
            "Server Configuration",
            [
                ("Endpoint URL", f"http://{args.host}:{args.port}"),
                ("OpenAI Base", f"http://{args.host}:{args.port}/v1"),
                ("API Docs", f"http://{args.host}:{args.port}/docs"),
                ("Concurrency", args.concurrency),
                ("Bearer Auth", "Enabled" if args.api_key else "Loopback-only (default)"),
            ],
            border_style="cyan",
        )
        run_server(host=args.host, port=args.port)

    elif args.command == "chat":
        from scripts.chat import main as chat_main

        chat_args = [
            "--checkpoint",
            args.checkpoint,
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
        if getattr(args, "model", None):
            chat_args.extend(["--model", args.model])
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
        prompt = args.prompt or args.prompt_pos
        if not prompt:
            render_notice("Please provide a prompt via positional argument or --prompt.", level="error")
            return

        source = getattr(args, "model", None) or args.checkpoint
        is_ollama = bool(args.model and str(args.model).strip().lower().startswith("ollama:"))

        render_banner("TRIUNE INFERENCE ENGINE", subtitle=str(source), badge=args.depth.upper())
        cfg_rows = [
            ("Source", source),
            ("Route", "OLLAMA LOCAL" if is_ollama else args.depth.upper()),
            ("Sampling", f"temp={args.temperature}  ·  top_k={args.top_k}  ·  max_tokens={args.max_tokens}"),
        ]
        if args.adapter:
            cfg_rows.append(("LoRA Adapter", args.adapter))
        render_kv_panel("Run Configuration", cfg_rows, border_style="bright_black")

        if is_ollama:
            from triune.runtime.model_scanner import (
                _is_tag_live_in_ollama,
                query_ollama_running_tags,
                stream_ollama_chat,
            )

            if not _is_tag_live_in_ollama(source, query_ollama_running_tags()):
                render_notice(
                    f"'{source}' is not live in Ollama. Start it via `ollama run {str(source).removeprefix('ollama:')}`.",
                    level="error",
                )
                return

            print(f"\n  Prompt: {prompt}")
            sys.stdout.write(f"  {route_badge_ansi('OLLAMA')} ")
            sys.stdout.flush()
            ol_res = stream_ollama_chat(
                source,
                [{"role": "user", "content": prompt}],
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                on_text=(lambda d: (sys.stdout.write(d), sys.stdout.flush())) if not args.no_stream else None,
            )
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

        import torch

        from triune.inference import generate_response, load_checkpoint_model, load_model_for_inference

        depth_map = {"auto": None, "reflex": 0, "limbic": 1, "cortex": 2}
        force_depth = depth_map.get(args.depth)
        load_t0 = time.perf_counter()
        with status_spinner(f"Loading weights for {source}..."):
            if getattr(args, "transmute", False) and (getattr(args, "model", None) or Path(source).is_dir()):
                from triune.model.transmuter import transmute_hf_to_triune

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
                        args.checkpoint,
                        args.tokenizer,
                        adapter_path=args.adapter,
                        device=args.device,
                    )
        load_time = time.perf_counter() - load_t0

        stream = not args.no_stream
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
            on_text=_on_text if stream else None,
            return_info=True,
        )
        if getattr(model, "_streaming_engine", None) is not None:
            model._streaming_engine.detach()
        if getattr(model, "_exit_manager", None) is not None and hasattr(model._exit_manager, "remove_hooks"):
            model._exit_manager.remove_hooks()

        if not stream:
            print(res["text"])
        else:
            print()

        route_used = res.get("route", last_route[0])
        vram_gb = round(torch.cuda.memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
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
        from triune.runtime.model_scanner import scan_pc_models

        render_banner("PC MODEL SCANNER", subtitle="Ollama · Hugging Face · vLLM · Hermes · Checkpoints")
        with status_spinner("Scanning local runtimes and model caches..."):
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
            caption="Tip: run `triune chat --model ollama:<tag>` for live Ollama or `triune transmute --model <id>` for 3-tier HF transfer.",
        )

    elif args.command == "transmute":
        from triune.inference import generate_response, load_model_for_inference
        from triune.model.transmuter import transfer_weights_to_native_triune, transmute_hf_to_triune

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
        import torch

        from triune.configs import build_config
        from triune.inference import load_checkpoint_model, load_model_for_inference
        from triune.model import build_model
        from triune.trainer.finetune import LoRAConfig, TriuneFineTuner

        render_banner(
            "TRIUNE FINE-TUNING",
            subtitle=args.model or args.checkpoint or "Triune Default",
            badge=f"r={args.rank} · {'NVFP4 QLoRA' if args.qlora else args.quantization}",
        )
        tokenizer = None
        if getattr(args, "model", None):
            quant = "nvfp4" if args.qlora else None
            with status_spinner(f"Loading base model {args.model}..."):
                model, tokenizer, _, device, _, _ = load_model_for_inference(
                    args.model,
                    device=None,
                    quantize=quant,
                    disk_stream=args.disk_stream,
                    disk_cache_dir=args.disk_cache,
                    quantize_lm_head=getattr(args, "quantize_lm_head", False),
                )
            if hasattr(tokenizer, "hf"):
                tokenizer = tokenizer.hf
        elif args.checkpoint and Path(args.checkpoint).is_file():
            with status_spinner(f"Loading checkpoint {args.checkpoint}..."):
                model, tokenizer, _, device, _, _ = load_checkpoint_model(args.checkpoint)
        else:
            render_notice("Instantiating default Triune architecture for fine-tuning...", level="info")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cfg = build_config({})
            model = build_model(cfg).to(device)

        lora_cfg = LoRAConfig(
            rank=args.rank,
            alpha=args.alpha,
            use_qlora=bool(args.qlora),
            quantization=("4-bit NVFP4" if args.qlora else args.quantization),
            quantize_base=(True if args.qlora else None),
        )
        tuner = TriuneFineTuner(
            model,
            lora_cfg,
            tokenizer=tokenizer,
            model_name_or_path=getattr(args, "model", None),
            adapter_device=device,
        )
        trainable = tuner.count_trainable_parameters()
        total_p = sum(p.numel() for p in model.parameters())
        render_kv_panel(
            "LoRA Configuration",
            [
                ("Trainable Params", f"{trainable:,} / {total_p:,} ({100.0 * trainable / max(1, total_p):.2f}%)"),
                ("Dataset", f"{args.dataset} (split={args.dataset_split})"),
                (
                    "Schedule",
                    f"epochs={args.epochs} · batch={args.batch_size} · accum={args.grad_accum} · lr={args.lr}",
                ),
                ("Output Dir", args.output_dir),
            ],
            border_style="cyan",
        )

        def _cli_on_step(step: int, total_s: int, loss_val: float, lm_loss_val: float):
            interval = max(1, total_s // 20)
            if step % interval == 0 or step == total_s:
                pct = round((step / max(1, total_s)) * 100)
                print(
                    f"  [Step {step:>4}/{total_s} ({pct:>3}%)] Loss: {loss_val:.4f} | LM Loss: {lm_loss_val:.4f}",
                    flush=True,
                )

        try:
            res = tuner.fit(
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                lr=args.lr,
                max_steps=args.max_steps,
                on_step=_cli_on_step,
                tokenizer=tokenizer,
                dataset_split=args.dataset_split,
                max_samples=args.max_samples,
                grad_accum_steps=args.grad_accum,
                stream_layers=bool(args.stream and not args.disk_stream),
                stream_chunk_size=args.stream_chunk,
                gradient_checkpointing=bool(args.checkpointing),
            )
            render_notice(
                f"Fine-tuning finished! Final Loss: {res['final_loss']:.4f} -> Saved to: {res['output_dir']}",
                level="success",
            )
        finally:
            if getattr(model, "_streaming_engine", None) is not None:
                model._streaming_engine.detach()

    elif args.command == "adapter":
        from triune.trainer.finetune import export_adapter, list_adapters, merge_adapter_checkpoint

        if args.adapter_action == "list":
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
            out = merge_adapter_checkpoint(
                args.adapter, args.checkpoint, output_path=args.output, tokenizer_path=args.tokenizer
            )
            render_notice(f"Merged adapter '{args.adapter}' into '{args.checkpoint}' -> {out}", level="success")
        elif args.adapter_action == "export":
            out = export_adapter(args.adapter, output_path=args.output, fmt=args.format)
            render_notice(f"Exported adapter to: {out}", level="success")
        else:
            parser._triune_subparsers["adapter"].print_help()  # type: ignore[attr-defined]

    elif args.command == "checkpoint":
        from triune.export import export_model
        from triune.inference import load_checkpoint_model

        if args.ckpt_action == "list":
            ckpts = []
            for d in ("checkpoints", "checkpoints_full"):
                p = Path(d)
                if p.is_dir():
                    for f in sorted(p.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
                        ckpts.append(
                            (str(f), f"{f.stat().st_size / (1024 * 1024):.1f} MB", time.ctime(f.stat().st_mtime))
                        )
            if not ckpts:
                render_notice("No checkpoints found in checkpoints/ or checkpoints_full/.", level="info")
            else:
                render_table(
                    f"Local Training Checkpoints ({len(ckpts)})",
                    columns=[
                        ("Path", "left", "bold cyan"),
                        ("Size", "right", "yellow"),
                        ("Modified", "left", "dim"),
                    ],
                    rows=[[p, sz, mt] for p, sz, mt in ckpts],
                )
        elif args.ckpt_action == "export":
            with status_spinner(f"Loading checkpoint {args.checkpoint}..."):
                model, tokenizer, _, _, _, _ = load_checkpoint_model(args.checkpoint)
            if args.format == "hf":
                from triune.export.hf_compat import export_to_huggingface

                out = args.output or f"exports/{Path(args.checkpoint).stem}_hf"
                out_p = export_to_huggingface(model, out, tokenizer=tokenizer)
                render_notice(f"Exported Hugging Face bundle to: {out_p}", level="success")
            else:
                out = args.output or f"exports/{Path(args.checkpoint).stem}.{args.format}"
                out_p = export_model(model, out, fmt=args.format)
                render_notice(f"Exported checkpoint to: {out_p}", level="success")
        else:
            parser._triune_subparsers["checkpoint"].print_help()  # type: ignore[attr-defined]

    elif args.command in ("export", "convert"):
        from triune.export import export_model
        from triune.inference import load_checkpoint_model

        with status_spinner(f"Loading checkpoint {args.checkpoint}..."):
            model, tokenizer, _, _, _, _ = load_checkpoint_model(args.checkpoint)
        if args.format == "hf":
            from triune.export.hf_compat import export_to_huggingface

            out = args.output or f"exports/{Path(args.checkpoint).stem}_hf"
            out_p = export_to_huggingface(model, out, tokenizer=tokenizer)
            render_notice(f"Exported Hugging Face bundle to: {out_p}", level="success")
        else:
            out = args.output or f"exports/{Path(args.checkpoint).stem}.{args.format}"
            out_p = export_model(model, out, fmt=args.format)
            render_notice(f"Exported checkpoint to: {out_p}", level="success")

    elif args.command == "models":
        from triune.model.base import CANONICAL_MODEL_SPECS

        rows = [
            [
                k,
                v.name,
                v.num_layers,
                v.hidden_dim,
                f"{v.num_experts}R/{v.experts_shared}S",
                v.tier,
            ]
            for k, v in CANONICAL_MODEL_SPECS.items()
        ]
        render_table(
            f"Triune Model Architecture Zoo ({len(CANONICAL_MODEL_SPECS)} Presets)",
            columns=[
                ("Preset ID", "left", "bold cyan"),
                ("Architecture", "left", "white"),
                ("Layers", "right", "yellow"),
                ("Hidden", "right", "white"),
                ("Experts", "left", "green"),
                ("Tier", "left", "dim"),
            ],
            rows=rows,
        )

    elif args.command == "train":
        from scripts.train import main as train_main

        sys.argv = [sys.argv[0]] + (args.train_args or [])
        train_main()

    elif args.command in ("benchmark", "bench"):
        if getattr(args, "suite", None):
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
            cats = suite_map.get(args.suite, ["a"])
            render_banner("TRIUNE BENCHMARK SUITE", subtitle=f"Running suite '{args.suite}'")
            for cat in cats:
                run_category(cat, sys.executable)
            return

        from scripts.benchmark_inference import benchmark_cli

        b_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            tokenizer=args.tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            device=args.device,
            depth=args.depth,
            temperature=args.temperature,
            warmup_tokens=8,
        )
        with status_spinner(f"Benchmarking inference on {args.checkpoint}..."):
            res = benchmark_cli(b_args)
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
        from triune.utils.device import device_diagnostic

        diag = device_diagnostic()
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
        from triune.configs import build_config
        from triune.runtime import MemoryPlanner

        if getattr(args, "model", None):
            hf_plan = MemoryPlanner.estimate_hf_model(args.model, available_vram_gb=args.vram_gb)
            render_kv_panel(
                f"Hardware Memory Plan · {hf_plan['model']}",
                [
                    (
                        "Parameters",
                        f"{hf_plan['total_params']:,} ({hf_plan['layers']}L, hidden={hf_plan['hidden_size']}, vocab={hf_plan['vocab_size']:,})",
                    ),
                    ("BF16 Weights", f"{hf_plan['bf16_weights_gb']} GB"),
                    (
                        "NVFP4 Cache (SSD/VRAM)",
                        f"{hf_plan['nvfp4_disk_cache_gb']} GB (~{hf_plan['per_layer_nvfp4_mb']} MB/layer; quant lm_head={hf_plan['nvfp4_quant_lmhead_cache_gb']} GB)",
                    ),
                    ("Hardware", f"{hf_plan['available_vram_gb']} GB VRAM  ·  {hf_plan['host_ram_gb']} GB Host RAM"),
                    ("Recommended Mode", hf_plan["recommended_mode"]),
                ],
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
        import json

        from triune.runtime.disk_streaming import build_nvfp4_disk_cache

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

        workspace = Path(__file__).resolve().parent.parent
        setup_exe = workspace / "TriuneStudioSetup.exe"
        if not setup_exe.is_file():
            setup_exe = workspace / "studio" / "windows" / "publish" / "TriuneStudioSetup.exe"
        if sys.platform == "win32" and setup_exe.is_file() and not args.yes and not args.verify:
            render_notice(f"Launching Native Windows Setup Wizard: {setup_exe}", level="info")
            subprocess.Popen([str(setup_exe)], cwd=str(workspace))
        else:
            installer_script = workspace / "studio" / "installer" / "installer.py"
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

    elif args.command in ("doctor", "info"):
        import importlib.util
        import json

        from triune.model.base import CANONICAL_MODEL_SPECS
        from triune.ops import get_native_c_info
        from triune.runtime.model_scanner import query_ollama_running_tags
        from triune.utils.device import device_diagnostic

        diag = device_diagnostic()
        c_info = get_native_c_info()
        has_fastapi = importlib.util.find_spec("fastapi") is not None
        ollama_tags = sorted(t for t in query_ollama_running_tags() if ":" in t)

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
                ("Optimizers", "CentroidSteer (Triangular Core), Muon (5th-Order NS), AdamW / 8-bit"),
                ("Fused Operators", "GLA, Chunked CE, RMSNorm, RoPE, SwiGLU, Fused LoRA, 3-Tier Transmuter"),
                ("Model Zoo", f"{len(CANONICAL_MODEL_SPECS)} Canonical Architecture Presets"),
                ("Studio FastAPI", "Operational" if has_fastapi else "Missing (install [studio] extras)"),
                (
                    "Local Ollama",
                    f"Running ({', '.join(ollama_tags[:6])})" if ollama_tags else "Not detected on 127.0.0.1:11434",
                ),
            ]
        )
        session_p = Path("checkpoints/session_state.json")
        if session_p.is_file():
            try:
                st = json.loads(session_p.read_text(encoding="utf-8"))
                rows.append(
                    (
                        "Active Session",
                        f"{st.get('active_checkpoint') or st.get('active_model_id')} (step {st.get('step')}, {st.get('total_tokens_trained', 0):,} tokens)",
                    )
                )
                rows.append(("Dataset Stream", f"{st.get('dataset_name')} (offset {st.get('hf_stream_offset', 0)})"))
            except Exception:
                pass
        render_kv_panel("System Diagnostics", rows, border_style="green")

    elif args.command == "build-native":
        import json

        from triune.packaging import build_native_targets

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

        prof_main(
            [
                "--checkpoint",
                args.checkpoint,
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
        from triune.agents import AgentHarness

        harness = AgentHarness(
            workspace=".",
            store_dir=args.store_dir,
            model=args.model,
            allow_shell=bool(args.allow_shell),
            allow_writes=not bool(args.read_only),
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
                render_notice(str(exc), level="error")
                return
        elif not auto_depth:
            harness.state_controller.shift(reasoning_depth=args.depth, reason="CLI depth override")

        model_badge = harness.model or "NO BACKEND (configure Ollama or BYOK)"
        render_banner(
            "TRIUNE AGENT HARNESS",
            subtitle=f"{harness.state_controller.active_profile.title} · {harness.state_controller.state.reasoning_depth.upper()}",
            badge=model_badge,
        )

        def _render_agent_event(ev_type: str, data: dict) -> None:
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
        if single_goal:
            try:
                res = harness.run_goal(
                    single_goal,
                    max_steps=args.max_steps,
                    on_event=_render_agent_event,
                    auto_persona=auto_persona,
                    auto_depth=auto_depth,
                    auto_crystallize=not bool(args.no_auto_skill),
                )
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
                render_notice(f"Agent execution failed: {exc}", level="error")
            return

        _print_agent_status()
        render_notice(
            "Interactive Agent REPL ready. Commands: /status, /persona <name>, /state <reflex|limbic|cortex>, /skills, /memory, /memory delete <key>, /memory clear, /tools, /learn <note>, /exit",
            level="info",
        )
        history: list[dict[str, str]] = []
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
                render_rule(f"{res['persona']} · {res['reasoning_depth'].upper()}")
                print(res["answer"])
            except Exception as exc:
                render_notice(f"Agent error: {exc}", level="error")

    else:
        render_help_overview()


if __name__ == "__main__":
    main()
