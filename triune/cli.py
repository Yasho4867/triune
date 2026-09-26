"""Unified Command-Line Interface for Triune Engine."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="triune", description="Triune Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available CLI Commands")

    # Studio command
    studio_parser = subparsers.add_parser("studio", help="Launch Triune Studio Native Windows Application")
    studio_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch embedded FastAPI server for Studio & API")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat session")
    chat_parser.add_argument("--checkpoint", default="checkpoints_full/best.pt", help="Model checkpoint path")
    chat_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum generated tokens")

    # GPU check command
    subparsers.add_parser("gpucheck", help="Print CUDA and hardware diagnostics")

    # Memory Plan command
    mem_parser = subparsers.add_parser("plan-memory", help="Estimate VRAM budget for RTX 5070 or target GPU")
    mem_parser.add_argument("--vram-gb", type=float, default=8.0, help="Target VRAM in GB")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train Triune Transformer models")
    train_parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passed to training script")

    args = parser.parse_args()

    if args.command == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0]] + (args.train_args or [])
        train_main()

    elif args.command == "studio":
        from triune.desktop import launch_desktop_app

        launch_desktop_app(port=args.port)
    elif args.command == "serve":
        from triune.api import run_server

        print(f"🚀 Starting Triune API Server on http://{args.host}:{args.port}")
        run_server(host=args.host, port=args.port)
    elif args.command == "chat":
        from scripts.chat import main as chat_main
        chat_args = [
            "--checkpoint", args.checkpoint,
            "--tokenizer_path", args.tokenizer,
            "--temperature", str(args.temperature),
            "--max_new_tokens", str(args.max_tokens),
        ]
        chat_main(chat_args)
    elif args.command == "gpucheck":
        from scripts.gpucheck import main as gpucheck_main
        gpucheck_main()
    elif args.command == "plan-memory":
        from triune.configs import build_config
        from triune.runtime import MemoryPlanner

        config = build_config({})
        plan = MemoryPlanner.estimate_vram(config, target_vram_gb=args.vram_gb)
        print("🧠 VRAM Memory Plan Estimate:")
        print(f"   Parameters: {plan.total_params:,} ({plan.param_memory_gb} GB)")
        print(f"   Optimizer State: {plan.optimizer_memory_gb} GB")
        print(f"   Activation Memory: {plan.activation_memory_gb} GB")
        print(f"   Recommended Batch Size: {plan.recommended_batch_size}")
        print(f"   Recommended Grad Accum: {plan.recommended_grad_accum}")
        print(f"   Recommended Precision: {plan.recommended_precision}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
