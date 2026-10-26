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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="triune",
        description="Triune Framework Unified CLI — High-Performance Mixture-of-Experts Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available CLI Commands")

    # 1. Studio command
    studio_parser = subparsers.add_parser("studio", help="Launch Triune Studio Native Desktop / Browser Application")
    studio_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    studio_parser.add_argument("--browser", action="store_true", help="Launch directly in default browser instead of native desktop host")
    studio_parser.add_argument("--native-host", action="store_true", help="Start backend for native Windows desktop host")
    studio_parser.add_argument("--use-wsl", action="store_true", help="Allow CUDA-enabled WSL backend on Windows")

    # 2. Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch embedded FastAPI server for Studio & OpenAI-compatible REST API")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")

    # 3. Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat session with dynamic exit depth routing")
    chat_parser.add_argument("--checkpoint", default="checkpoints_full/best.pt", help="Model checkpoint path")
    chat_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    chat_parser.add_argument("--adapter", default=None, help="LoRA adapter directory or safetensors path to attach for chat")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    chat_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum generated tokens")

    # 4. Fine-Tuning command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune Triune models with LoRA or QLoRA parameter-efficient adaptation")
    finetune_parser.add_argument("--dataset", default="data/fineweb_sample.jsonl", help="Dataset path (.jsonl or .csv)")
    finetune_parser.add_argument("--checkpoint", default=None, help="Base model checkpoint path (optional)")
    finetune_parser.add_argument("--rank", type=int, default=16, help="LoRA rank dimension r (default: 16)")
    finetune_parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha scaling (default: 32.0)")
    finetune_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    finetune_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    finetune_parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step (default: 4)")
    finetune_parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (optional)")
    finetune_parser.add_argument("--output-dir", default="checkpoints/adapters/lora_finetuned", help="Output directory for adapter weights")
    finetune_parser.add_argument("--quantization", default="4-bit NF4", help="Quantization precision (default: 4-bit NF4)")

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
    export_adapter_parser.add_argument("--format", default="safetensors", choices=("safetensors", "pt"), help="Export format")

    # 6. Checkpoint Management command
    ckpt_parser = subparsers.add_parser("checkpoint", help="Manage model checkpoints (list, export)")
    ckpt_sub = ckpt_parser.add_subparsers(dest="ckpt_action", help="Checkpoint action")

    ckpt_sub.add_parser("list", help="List local checkpoints")

    ckpt_export_parser = ckpt_sub.add_parser("export", help="Export checkpoint to SafeTensors, GGUF, or ONNX")
    ckpt_export_parser.add_argument("--checkpoint", required=True, help="Source checkpoint path")
    ckpt_export_parser.add_argument("--output", default=None, help="Destination file path")
    ckpt_export_parser.add_argument("--format", default="safetensors", choices=("safetensors", "gguf", "onnx"), help="Target format")

    # 7. Model Zoo command
    subparsers.add_parser("models", help="Display architecture presets from the Triune Model Zoo")

    # 8. Train command
    train_parser = subparsers.add_parser("train", help="Train Triune Transformer models (pre-training or fine-tuning)")
    train_parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passed to training script")

    # 9. Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Measure steady-state inference decode speed and latency")
    bench_parser.add_argument("--checkpoint", default="checkpoints_full/best.pt", help="Checkpoint path")
    bench_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    bench_parser.add_argument("--prompt", default="The meaning of life is", help="Prompt text")
    bench_parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    bench_parser.add_argument("--device", default=None, help="Device (cuda / cpu)")
    bench_parser.add_argument("--depth", default="auto", choices=("auto", "reflex", "limbic", "cortex"), help="Exit depth routing")
    bench_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")

    # 10. GPU check command
    subparsers.add_parser("gpucheck", help="Print CUDA hardware and driver diagnostics")

    # 11. Memory Plan command
    mem_parser = subparsers.add_parser("plan-memory", help="Estimate VRAM budget for target GPU")
    mem_parser.add_argument("--vram-gb", type=float, default=8.0, help="Target VRAM in GB")

    # 12. Install / Setup command
    install_parser = subparsers.add_parser("install", help="Launch Triune Studio Setup Wizard or automated environment provisioner")
    install_parser.add_argument("--cuda", default="auto", choices=("auto", "13.2", "13.0", "12.6", "12.4", "12.1", "11.8", "cpu"), help="CUDA wheel target")
    install_parser.add_argument("--verify", action="store_true", help="Verify existing installation health only")
    install_parser.add_argument("--skip-torch", action="store_true", help="Skip PyTorch wheel installation")
    install_parser.add_argument("--no-launch", action="store_true", help="Do not launch Studio after installation")
    install_parser.add_argument("-y", "--yes", action="store_true", help="Run unattended without GUI prompt")

    # 13. Doctor command
    subparsers.add_parser("doctor", help="Run comprehensive end-to-end diagnostics on GPU, optimizer, router, and Studio backend")

    # 14. Build Native Apps command
    native_parser = subparsers.add_parser("build-native", help="Build native desktop apps & installers for Windows (.exe), Linux (AppDir), and macOS (.app)")
    native_parser.add_argument("--platform", default="all", choices=("all", "windows", "linux", "macos"), help="Target OS platform (default: all)")

    # 15. Profile command
    prof_parser = subparsers.add_parser("profile", help="Profile Triune training steps")
    prof_parser.add_argument("--checkpoint", default="checkpoints_full/latest.pt", help="Checkpoint path")
    prof_parser.add_argument("--tokenizer", default="triune_tokenizer.json", help="Tokenizer path")
    prof_parser.add_argument("--output", default="profiler_output.txt", help="Profiler output report path")
    prof_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    prof_parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")

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
        from triune.api import run_server
        print(f"[Triune] Starting Triune API Server on http://{args.host}:{args.port}")
        run_server(host=args.host, port=args.port)

    elif args.command == "chat":
        from scripts.chat import main as chat_main
        chat_args = [
            "--checkpoint", args.checkpoint,
            "--tokenizer_path", args.tokenizer,
            "--temperature", str(args.temperature),
            "--repetition_penalty", str(args.repetition_penalty),
            "--max_new_tokens", str(args.max_tokens),
        ]
        if args.adapter:
            chat_args.extend(["--adapter", args.adapter])
        chat_main(chat_args)

    elif args.command == "finetune":
        from triune.trainer.finetune import LoRAConfig, TriuneFineTuner
        from triune.inference import load_checkpoint_model
        from triune.model import build_model
        from triune.configs import build_config
        import torch

        print(f"\n[Triune CLI] Initializing Fine-Tuning Session (rank={args.rank}, alpha={args.alpha}, epochs={args.epochs})")
        if args.checkpoint and Path(args.checkpoint).is_file():
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
            quantization=args.quantization,
        )
        tuner = TriuneFineTuner(model, lora_cfg)
        trainable = tuner.count_trainable_parameters()
        total_p = sum(p.numel() for p in model.parameters())
        print(f"[Triune CLI] Model ready: {trainable:,} trainable adapter params ({total_p:,} total)")

        def _cli_on_step(step: int, total_s: int, loss_val: float, lm_loss_val: float):
            interval = max(1, total_s // 20)
            if step % interval == 0 or step == total_s:
                pct = round((step / max(1, total_s)) * 100)
                print(f"  [Step {step}/{total_s} ({pct}%)] Loss: {loss_val:.4f} | LM Loss: {lm_loss_val:.4f}", flush=True)

        res = tuner.fit(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_steps=args.max_steps,
            on_step=_cli_on_step,
        )
        print(f"[Triune CLI] Fine-tuning finished! Final Loss: {res['final_loss']} -> Saved to: {res['output_dir']}\n")

    elif args.command == "adapter":
        from triune.trainer.finetune import list_adapters, export_adapter, merge_adapter_checkpoint
        if args.adapter_action == "list":
            ads = list_adapters()
            if not ads:
                print("\nNo adapters found on disk (scanned checkpoints/, checkpoints/adapters/, exports/).\n")
            else:
                print(f"\nDiscovered {len(ads)} LoRA Adapters on Disk:")
                print(f"{'NAME':<24} {'RANK/ALPHA':<14} {'PRECISION':<12} {'SIZE':<10} {'PATH'}")
                print("-" * 80)
                for a in ads:
                    print(f"{a['name']:<24} r={a['rank']}, α={a['alpha']:<6} {a['quantization']:<12} {a['size_mb']} MB    {a['path']}")
                print()
        elif args.adapter_action == "merge":
            out = merge_adapter_checkpoint(args.adapter, args.checkpoint, output_path=args.output, tokenizer_path=args.tokenizer)
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
            model, _, _, _, _, _ = load_checkpoint_model(args.checkpoint)
            out = args.output or f"exports/{Path(args.checkpoint).stem}.{args.format}"
            out_p = export_model(model, out, fmt=args.format)
            print(f"[Triune CLI] Successfully exported to: {out_p}\n")
        else:
            ckpt_parser.print_help()

    elif args.command == "models":
        from triune.model.base import CANONICAL_MODEL_SPECS
        print(f"\nTriune Model Architecture Zoo ({len(CANONICAL_MODEL_SPECS)} Verified Presets):")
        print(f"{'PRESET ID':<16} {'NAME':<36} {'LAYERS':<8} {'DIM':<8} {'EXPERTS':<9} {'TIER'}")
        print("-" * 95)
        for k, v in CANONICAL_MODEL_SPECS.items():
            print(f"{k:<16} {v.name:<36} {v.num_layers:<8} {v.hidden_dim:<8} {v.num_experts}R/{v.experts_shared}S      {v.tier}")
        print()

    elif args.command == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0]] + (args.train_args or [])
        train_main()

    elif args.command == "benchmark":
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

        config = build_config({})
        plan = MemoryPlanner.estimate_vram(config, target_vram_gb=args.vram_gb)
        print("[Triune] VRAM Memory Plan Estimate:")
        print(f"   Parameters: {plan.total_params:,} ({plan.param_memory_gb} GB)")
        print(f"   Optimizer State: {plan.optimizer_memory_gb} GB")
        print(f"   Activation Memory: {plan.activation_memory_gb} GB")
        print(f"   Recommended Batch Size: {plan.recommended_batch_size}")
        print(f"   Recommended Grad Accum: {plan.recommended_grad_accum}")
        print(f"   Recommended Precision: {plan.recommended_precision}")

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

    elif args.command == "doctor":
        import torch
        from triune.utils.device import device_diagnostic
        from triune.api.server import HAS_FASTAPI

        diag = device_diagnostic()
        print("\n=== Triune System & Engine Doctor ===")
        print(f"  Python Runtime   : {diag['python']}")
        print(f"  PyTorch Version  : {diag['torch']}")
        print(f"  CUDA Available   : {diag['cuda_available']}")
        for idx, gpu in enumerate(diag.get("gpus", [])):
            maj, min_v = gpu["capability"]
            print(f"  GPU [{idx}]          : {gpu['name']} ({gpu['memory_gib']:.2f} GiB, SM{maj}.{min_v})")
        print(f"  Studio FastAPI   : {'Operational' if HAS_FASTAPI else 'Missing'}")
        session_p = Path("checkpoints/session_state.json")
        if session_p.is_file():
            import json
            st = json.loads(session_p.read_text(encoding="utf-8"))
            print(f"  Active Checkpoint: {st.get('active_checkpoint')} (step {st.get('step')}, {st.get('total_tokens_trained', 0):,} tokens)")
            print(f"  Streaming Dataset: {st.get('dataset_name')} (offset {st.get('hf_stream_offset', 0)})")
        print("  Status           : HEALTHY\n")

    elif args.command == "build-native":
        from triune.packaging import build_native_targets
        import json
        res = build_native_targets(args.platform)
        print("\n[Triune Native Packager] Build Summary:")
        print(json.dumps(res, indent=2))

    elif args.command == "profile":
        from scripts.profile_model import main as prof_main
        prof_main([
            "--checkpoint", args.checkpoint,
            "--tokenizer_path", args.tokenizer,
            "--output", args.output,
            "--batch_size", str(args.batch_size),
            "--seq_len", str(args.seq_len),
        ])

    elif args.command == "tokenizer":
        from scripts.tokenizer import main as tok_main
        tok_main([
            "--output", args.output,
            "--vocab_size", str(args.vocab_size),
        ])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
