"""Unified Command-Line Interface for Triune Engine.

Unified CLI and backend engine powering both terminal workflows and Triune Studio.
All commands share identical model definitions, runtime memory management,
LoRA/QLoRA fine-tuning abstractions, inference routines, and export tools.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_IS_TTY = sys.stdout.isatty()


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _IS_TTY else str(text)


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _IS_TTY else str(text)


def _cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m" if _IS_TTY else str(text)


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m" if _IS_TTY else str(text)


def _magenta(text: str) -> str:
    return f"\033[35m{text}\033[0m" if _IS_TTY else str(text)


def _route_badge(route: str) -> str:
    r = str(route).upper()
    if "REFLEX" in r:
        return "\033[1;36m[REFLEX]\033[0m" if _IS_TTY else "[REFLEX]"
    elif "LIMBIC" in r:
        return "\033[1;33m[LIMBIC]\033[0m" if _IS_TTY else "[LIMBIC]"
    elif "CORTEX" in r:
        return "\033[1;35m[CORTEX]\033[0m" if _IS_TTY else "[CORTEX]"
    return f"\033[1;32m[{r}]\033[0m" if _IS_TTY else f"[{r}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="triune",
        description="Triune AI Unified CLI — High-Performance Mixture-of-Experts & Native Operator Engine",
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
        "chat", help="Interactive terminal chat session with Triune or Hugging Face models"
    )
    chat_parser.add_argument("--model", default=None, help="HF model repo ID, local HF dir, or NVFP4 disk cache")
    chat_parser.add_argument("--checkpoint", default="checkpoints_full/best.pt", help="Model checkpoint path")
    chat_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    chat_parser.add_argument(
        "--adapter", default=None, help="LoRA adapter directory or safetensors path to attach for chat"
    )
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    chat_parser.add_argument(
        "--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=100, help="Maximum generated tokens"
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
    chat_parser.add_argument("--device", default=None, help="Target device (cuda / cpu)")

    # 4. Infer / Generate command
    infer_parser = subparsers.add_parser(
        "infer",
        aliases=["inference", "generate"],
        help="Run high-performance streaming inference with Triune or Hugging Face models",
    )
    infer_parser.add_argument("prompt_pos", nargs="?", default=None, help="Inference prompt text (positional)")
    infer_parser.add_argument("-p", "--prompt", default=None, help="Inference prompt text")
    infer_parser.add_argument("--model", default=None, help="HF model repo ID, local HF dir, or NVFP4 disk cache")
    infer_parser.add_argument("--checkpoint", default="checkpoints_full/best.pt", help="Model checkpoint path")
    infer_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    infer_parser.add_argument("--adapter", default=None, help="LoRA adapter directory or safetensors path (optional)")
    infer_parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=100,
        help="Maximum generated tokens (default: 100)",
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
    infer_parser.add_argument("--device", default=None, help="Target device (cuda / cpu)")
    infer_parser.add_argument("--no-stream", action="store_true", help="Disable live token streaming")

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

    # 5. Adapter Management command
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

    # 6. Checkpoint Management command
    ckpt_parser = subparsers.add_parser("checkpoint", help="Manage model checkpoints (list, export)")
    ckpt_sub = ckpt_parser.add_subparsers(dest="ckpt_action", help="Checkpoint action")

    ckpt_sub.add_parser("list", help="List local checkpoints")

    ckpt_export_parser = ckpt_sub.add_parser("export", help="Export checkpoint to SafeTensors, GGUF, or ONNX")
    ckpt_export_parser.add_argument("--checkpoint", required=True, help="Source checkpoint path")
    ckpt_export_parser.add_argument("--output", default=None, help="Destination file path")
    ckpt_export_parser.add_argument(
        "--format", default="safetensors", choices=("safetensors", "gguf", "onnx", "hf"), help="Target format"
    )

    # 7. Top-Level Export / Convert command
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

    # 8. Model Zoo command
    subparsers.add_parser("models", help="Display architecture presets from the Triune Model Zoo")

    # 8. Train command
    train_parser = subparsers.add_parser("train", help="Train Triune Transformer models (pre-training or fine-tuning)")
    train_parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passed to training script")

    # 9. Benchmark command
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

    # 10. GPU check command
    subparsers.add_parser("gpucheck", help="Print CUDA hardware and driver diagnostics")

    # 11. Memory Plan command
    mem_parser = subparsers.add_parser("plan-memory", help="Estimate VRAM/RAM budget for Triune or Hugging Face models")
    mem_parser.add_argument("--vram-gb", type=float, default=None, help="Target VRAM in GB (default: auto-detect)")
    mem_parser.add_argument("--model", default=None, help="Optional HF model ID or local HF directory to plan for")

    # 11b. Cache NVFP4 shards for NVMe streaming
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

    # 12. Install / Setup command
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

    # 13. Doctor / Info command
    subparsers.add_parser(
        "doctor",
        aliases=["info"],
        help="Run comprehensive end-to-end diagnostics on GPU, C/AVX2 engine, optimizers, and Studio backend",
    )

    # 14. Build Native Apps command
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

    # 15. Profile command
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

    # 16. Tokenizer command
    tok_parser = subparsers.add_parser("tokenizer", help="Build or train a BPE tokenizer")
    tok_parser.add_argument("--output", default="triune_tokenizer.json", help="Output tokenizer JSON path")
    tok_parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")

    args = parser.parse_args()

    if args.command == "studio":
        from triune.desktop import launch_desktop_app

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
        print(
            f"[Triune] Starting Triune API Server on http://{args.host}:{args.port} (concurrency: {args.concurrency})"
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
        if getattr(args, "device", None):
            chat_args.extend(["--device", args.device])
        chat_main(chat_args)

    elif args.command in ("infer", "inference", "generate"):
        prompt = args.prompt or args.prompt_pos
        if not prompt:
            print("[Triune CLI] Error: Please provide a prompt via positional argument or --prompt.")
            return

        import torch

        from triune.inference import generate_response, load_checkpoint_model, load_model_for_inference

        depth_map = {"auto": None, "reflex": 0, "limbic": 1, "cortex": 2}
        force_depth = depth_map.get(args.depth)
        source = getattr(args, "model", None) or args.checkpoint

        print(f"\n{_bold('Triune High-Performance Inference Engine')}")
        print(f"  Model Source: {source}")
        print(f"  Target Route: {args.depth.upper()} | Temp: {args.temperature} | Max Tokens: {args.max_tokens}")
        if args.adapter:
            print(f"  LoRA Adapter: {args.adapter}")

        load_t0 = time.perf_counter()
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
        print(f"\n{_bold('Prompt:')} {prompt}")
        print(f"{_bold('Output:')} ", end="", flush=True)

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

        if not stream:
            print(res["text"])
        else:
            print()

        route_used = res.get("route", last_route[0])
        vram_gb = round(torch.cuda.memory_allocated(device) / (1024**3), 2) if device.type == "cuda" else 0.0
        tok_count = res.get("tokens_count", 0)
        tok_sec = res.get("tokens_per_sec", 0.0)
        ttft_ms = res.get("ttft_ms", 0.0)
        lat_ms = res.get("latency_ms", 0.0)

        print(f"\n{'-' * 70}")
        print(
            f"Telemetry: {_route_badge(route_used)} | "
            f"{_cyan(f'{tok_sec:.1f} tok/s')} | "
            f"TTFT: {_yellow(f'{ttft_ms:.1f}ms' if ttft_ms else 'N/A')} | "
            f"Total: {tok_count} tokens in {lat_ms:.0f}ms | "
            f"VRAM: {vram_gb:.2f} GB | "
            f"Load: {load_time:.2f}s"
        )
        print(f"{'-' * 70}\n")

    elif args.command == "finetune":
        import torch

        from triune.configs import build_config
        from triune.inference import load_checkpoint_model, load_model_for_inference
        from triune.model import build_model
        from triune.trainer.finetune import LoRAConfig, TriuneFineTuner

        print(
            f"\n[Triune CLI] Initializing Fine-Tuning Session (rank={args.rank}, alpha={args.alpha}, epochs={args.epochs})"
        )
        tokenizer = None
        if getattr(args, "model", None):
            print(f"[Triune CLI] Loading model source: {args.model}")
            quant = "nvfp4" if args.qlora else None
            model, tokenizer, _, device, _, _ = load_model_for_inference(
                args.model,
                device=None,
                quantize=quant,
                disk_stream=args.disk_stream,
                disk_cache_dir=args.disk_cache,
                quantize_lm_head=getattr(args, "quantize_lm_head", False),
            )
            # Load raw HF tokenizer if wrapped
            if hasattr(tokenizer, "hf"):
                tokenizer = tokenizer.hf
        elif args.checkpoint and Path(args.checkpoint).is_file():
            print(f"[Triune CLI] Loading base weights from checkpoint: {args.checkpoint}")
            model, tokenizer, _, device, _, _ = load_checkpoint_model(args.checkpoint)
        else:
            print("[Triune CLI] Instantiating default Triune architecture for fine-tuning...")
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
        print(f"[Triune CLI] Model ready: {trainable:,} trainable adapter params ({total_p:,} total)")

        def _cli_on_step(step: int, total_s: int, loss_val: float, lm_loss_val: float):
            interval = max(1, total_s // 20)
            if step % interval == 0 or step == total_s:
                pct = round((step / max(1, total_s)) * 100)
                print(
                    f"  [Step {step}/{total_s} ({pct}%)] Loss: {loss_val:.4f} | LM Loss: {lm_loss_val:.4f}", flush=True
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
            print(
                f"[Triune CLI] Fine-tuning finished! Final Loss: {res['final_loss']} -> Saved to: {res['output_dir']}\n"
            )
        finally:
            if getattr(model, "_streaming_engine", None) is not None:
                model._streaming_engine.detach()

    elif args.command == "adapter":
        from triune.trainer.finetune import export_adapter, list_adapters, merge_adapter_checkpoint

        if args.adapter_action == "list":
            ads = list_adapters()
            if not ads:
                print("\nNo adapters found on disk (scanned checkpoints/, checkpoints/adapters/, exports/).\n")
            else:
                print(f"\nDiscovered {len(ads)} LoRA Adapters on Disk:")
                print(f"{'NAME':<24} {'RANK/ALPHA':<14} {'PRECISION':<12} {'SIZE':<10} {'PATH'}")
                print("-" * 80)
                for a in ads:
                    print(
                        f"{a['name']:<24} r={a['rank']}, α={a['alpha']:<6} {a['quantization']:<12} {a['size_mb']} MB    {a['path']}"
                    )
                print()
        elif args.adapter_action == "merge":
            out = merge_adapter_checkpoint(
                args.adapter, args.checkpoint, output_path=args.output, tokenizer_path=args.tokenizer
            )
            print(f"\nSuccessfully merged adapter '{args.adapter}' into '{args.checkpoint}' -> Saved to: {out}\n")
        elif args.adapter_action == "export":
            out = export_adapter(args.adapter, output_path=args.output, fmt=args.format)
            print(f"\nSuccessfully exported adapter to: {out}\n")
        else:
            adapter_parser.print_help()

    elif args.command == "checkpoint":
        from triune.export import export_model
        from triune.inference import load_checkpoint_model

        if args.ckpt_action == "list":
            ckpts = []
            for d in ("checkpoints", "checkpoints_full"):
                p = Path(d)
                if p.is_dir():
                    for f in p.glob("*.pt"):
                        ckpts.append((f, f.stat().st_size / (1024 * 1024), time.ctime(f.stat().st_mtime)))
            if not ckpts:
                print("\nNo checkpoints found in checkpoints/ or checkpoints_full/.\n")
            else:
                print(f"\nDiscovered {len(ckpts)} Checkpoints:")
                print(f"{'FILENAME':<35} {'SIZE':<12} {'MODIFIED'}")
                print("-" * 75)
                for f, sz, mt in ckpts:
                    print(f"{str(f):<35} {sz:.1f} MB     {mt}")
                print()
        elif args.ckpt_action == "export":
            print(f"[Triune CLI] Loading checkpoint: {args.checkpoint}")
            model, tokenizer, _, _, _, _ = load_checkpoint_model(args.checkpoint)
            if args.format == "hf":
                from triune.export.hf_compat import export_to_huggingface

                out = args.output or f"exports/{Path(args.checkpoint).stem}_hf"
                out_p = export_to_huggingface(model, out, tokenizer=tokenizer)
                print(f"[Triune CLI] Successfully exported Hugging Face bundle to: {out_p}\n")
            else:
                out = args.output or f"exports/{Path(args.checkpoint).stem}.{args.format}"
                out_p = export_model(model, out, fmt=args.format)
                print(f"[Triune CLI] Successfully exported to: {out_p}\n")
        else:
            ckpt_parser.print_help()

    elif args.command in ("export", "convert"):
        from triune.export import export_model
        from triune.inference import load_checkpoint_model

        print(f"[Triune CLI] Loading checkpoint: {args.checkpoint}")
        model, tokenizer, _, _, _, _ = load_checkpoint_model(args.checkpoint)
        if args.format == "hf":
            from triune.export.hf_compat import export_to_huggingface

            out = args.output or f"exports/{Path(args.checkpoint).stem}_hf"
            out_p = export_to_huggingface(model, out, tokenizer=tokenizer)
            print(f"[Triune CLI] Successfully exported Hugging Face bundle to: {out_p}\n")
        else:
            out = args.output or f"exports/{Path(args.checkpoint).stem}.{args.format}"
            out_p = export_model(model, out, fmt=args.format)
            print(f"[Triune CLI] Successfully exported to: {out_p}\n")

    elif args.command == "models":
        from triune.model.base import CANONICAL_MODEL_SPECS

        print(f"\nTriune Model Architecture Zoo ({len(CANONICAL_MODEL_SPECS)} Verified Presets):")
        print(f"{'PRESET ID':<16} {'NAME':<36} {'LAYERS':<8} {'DIM':<8} {'EXPERTS':<9} {'TIER'}")
        print("-" * 95)
        for k, v in CANONICAL_MODEL_SPECS.items():
            print(
                f"{k:<16} {v.name:<36} {v.num_layers:<8} {v.hidden_dim:<8} {v.num_experts}R/{v.experts_shared}S      {v.tier}"
            )
        print()

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
        )
        res = benchmark_cli(b_args)
        print("\nTriune Inference Benchmark Results:")
        print(f"  Checkpoint: {res['checkpoint']}")
        print(f"  Device: {res['device']}")
        print(f"  Model Load Time: {res['model_load_seconds']} s")
        print(f"  Time to First Token (TTFT): {res.get('first_token_latency_ms', 0):.2f} ms")
        print(f"  Decode Throughput: {res.get('tokens_per_second', 0):.2f} tokens/sec")
        print(f"  Tokens Generated: {res.get('tokens_generated', 0)}")
        print(f"  CUDA VRAM Allocated: {res.get('cuda_allocated_gb', 0)} GB")
        print(f"  Response: {res.get('response', '')}\n")

    elif args.command == "gpucheck":
        from scripts.gpucheck import main as gpucheck_main

        gpucheck_main()

    elif args.command == "plan-memory":
        from triune.configs import build_config
        from triune.runtime import MemoryPlanner

        if getattr(args, "model", None):
            hf_plan = MemoryPlanner.estimate_hf_model(args.model, available_vram_gb=args.vram_gb)
            print(f"\n[Triune] Hardware Plan for {hf_plan['model']}:")
            print(
                f"   Parameters            : {hf_plan['total_params']:,} ({hf_plan['layers']}L, hidden={hf_plan['hidden_size']}, vocab={hf_plan['vocab_size']:,})"
            )
            print(f"   BF16 Weights          : {hf_plan['bf16_weights_gb']} GB")
            print(
                f"   NVFP4 Cache (SSD/VRAM): {hf_plan['nvfp4_disk_cache_gb']} GB (~{hf_plan['per_layer_nvfp4_mb']} MB/layer; lm_head-quantized={hf_plan['nvfp4_quant_lmhead_cache_gb']} GB)"
            )
            print(
                f"   System Hardware       : {hf_plan['available_vram_gb']} GB VRAM | {hf_plan['host_ram_gb']} GB Host RAM"
            )
            print(f"   Recommended Mode      : {hf_plan['recommended_mode']}\n")
        else:
            config = build_config({})
            plan = MemoryPlanner.estimate_vram(config, target_vram_gb=args.vram_gb or 8.0)
            print("[Triune] VRAM Memory Plan Estimate:")
            print(f"   Parameters: {plan.total_params:,} ({plan.param_memory_gb} GB)")
            print(f"   Optimizer State: {plan.optimizer_memory_gb} GB")
            print(f"   Activation Memory: {plan.activation_memory_gb} GB")
            print(f"   Recommended Batch Size: {plan.recommended_batch_size}")
            print(f"   Recommended Grad Accum: {plan.recommended_grad_accum}")
            print(f"   Recommended Precision: {plan.recommended_precision}")

    elif args.command == "cache-nvfp4":
        import json

        from triune.runtime.disk_streaming import build_nvfp4_disk_cache

        print(f"[Triune CLI] Packing HF Safetensors -> NVFP4 disk cache for: {args.model}")
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
        print(
            f"[Triune CLI] Ready at {cache_path}\n"
            f"   Layers: {manifest['num_layers']} | Size: {raw_mb:.1f} MB -> {packed_mb:.1f} MB ({manifest['compression_ratio']}x compression)\n"
        )

    elif args.command == "install":
        import subprocess

        workspace = Path(__file__).resolve().parent.parent
        setup_exe = workspace / "TriuneStudioSetup.exe"
        if not setup_exe.is_file():
            setup_exe = workspace / "studio" / "windows" / "publish" / "TriuneStudioSetup.exe"
        if sys.platform == "win32" and setup_exe.is_file() and not args.yes and not args.verify:
            print(f"[Triune CLI] Launching Native Windows Setup Wizard: {setup_exe}")
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

        import torch

        from triune.utils.device import device_diagnostic

        has_fastapi = importlib.util.find_spec("fastapi") is not None
        from triune.model.base import CANONICAL_MODEL_SPECS
        from triune.ops import get_native_c_info

        diag = device_diagnostic()
        c_info = get_native_c_info()
        print("\n=== Triune AI System & Engine Doctor ===")
        print(f"  Python Runtime   : {diag['python']}")
        print(f"  PyTorch Version  : {diag['torch']}")
        print(f"  CUDA Available   : {diag['cuda_available']}")
        for idx, gpu in enumerate(diag.get("gpus", [])):
            maj, min_v = gpu["capability"]
            print(f"  GPU [{idx}]          : {gpu['name']} ({gpu['memory_gib']:.2f} GiB, SM{maj}.{min_v})")
        print(
            f"  Native C Engine  : {'Loaded' if c_info.get('available') else 'Fallback'} (v{c_info.get('version', '2.0')}, CUDA-Bound={c_info.get('cuda_runtime_bound', False)}, SIMD=AVX2)"
        )
        print("  Optimizers       : CentroidSteer (Triangular Core), Muon (5th-Order NS), AdamW / 8-bit")
        print("  Fused Operators  : GLA, Chunked CE, RMSNorm, RoPE, SwiGLU, Fused LoRA, Universal Upgrade")
        print(f"  Model Zoo        : {len(CANONICAL_MODEL_SPECS)} Verified Architecture Presets")
        print(f"  Studio FastAPI   : {'Operational' if has_fastapi else 'Missing'}")
        session_p = Path("checkpoints/session_state.json")
        if session_p.is_file():
            import json

            st = json.loads(session_p.read_text(encoding="utf-8"))
            print(
                f"  Active Checkpoint: {st.get('active_checkpoint')} (step {st.get('step')}, {st.get('total_tokens_trained', 0):,} tokens)"
            )
            print(f"  Streaming Dataset: {st.get('dataset_name')} (offset {st.get('hf_stream_offset', 0)})")
        print("  Status           : HEALTHY\n")

    elif args.command == "build-native":
        import json

        from triune.packaging import build_native_targets

        res = build_native_targets(args.platform)
        print("\n[Triune Native Packager] Build Summary:")
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

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
