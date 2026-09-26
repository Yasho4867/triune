"""Command-line entry point for callable Triune training."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

# ── Disable HuggingFace XET protocol (MUST be before any HF imports) ─────
# huggingface_hub ≥1.31 ships hf-xet which routes ALL downloads through
# us.aws.cdn.hf.co/xet-bridge-us/.  From non-US locations the SSL
# handshake to that endpoint frequently times out.  Disabling XET
# forces standard HTTPS streaming via CloudFront edge servers.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triune.configs.config import build_config
from triune.data import build_dataloader, load_tokenizer
from triune.model import build_model
from triune.optim.factory import build_optimizer
from triune.recipes import bf16_autocast, build_fp8_precision_context, build_precision_context
from triune.trainer import NullLogger, Trainer, WandbLogger


def _depth_distribution(value: str) -> list[float]:
    try:
        return [float(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Triune Transformer")
    parser.add_argument("--resume_best", action="store_true")
    parser.add_argument("--resume_latest", type=Path)
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--eval_batches", type=int)
    parser.add_argument("--target_depth_dist", type=_depth_distribution)
    parser.add_argument("--usage_ema_decay", type=float)
    parser.add_argument("--bias_strength", type=float)
    parser.add_argument("--balance_coef", type=float)
    parser.add_argument("--exploration", choices=("linear", "cosine", "none"))
    parser.add_argument("--exploration_steps", type=int)
    parser.add_argument("--steer_scale", type=float)
    parser.add_argument("--use_fp4", "--use_nvfp4", dest="use_fp4", action="store_true")
    parser.add_argument("--use_fp8", action="store_true", help="Enable native FP8 (E4M3) precision context")
    parser.add_argument("--use_muon", action=argparse.BooleanOptionalAction, default=True, help="Enable Muon for non-expert 2D hidden weights")
    parser.add_argument("--muon_lr", type=float, help="Muon learning rate (default: 0.02)")
    parser.add_argument("--galore", action=argparse.BooleanOptionalAction, default=None, help="Enable GaLore low-rank gradient projection for expert weights")
    parser.add_argument("--galore_rank", type=int, help="GaLore projection subspace rank (default: 256)")
    parser.add_argument("--galore_update_gap", type=int, help="GaLore subspace update frequency in steps (default: 200)")
    parser.add_argument("--galore_lr", type=float, help="GaLore learning rate (default: 1e-4)")
    parser.add_argument("--galore_weight_decay", type=float, help="GaLore weight decay (default: 0.05)")
    parser.add_argument("--force", action="store_true", help="Override resource manager feasibility recommendations and proceed regardless of VRAM budget")
    parser.add_argument("--auto_fit", action="store_true", default=False, help="Automatically adopt recommended microbatch and grad_accum settings if not explicitly specified")
    parser.add_argument("--streaming", action="store_true", help="Enable AirLLM-style layer streaming to train large models on consumer GPUs with <1.5 GB VRAM")
    parser.add_argument("--streaming_chunk_size", type=int, default=1, help="Number of layers to stream/cache simultaneously (default: 1)")
    parser.add_argument("--streaming_pin_memory", action=argparse.BooleanOptionalAction, default=None, help="Pin host memory for PCIe DMA in streaming mode (auto-detected based on RAM)")
    parser.add_argument("--streaming_prefetch", action=argparse.BooleanOptionalAction, default=True, help="Prefetch next layer on secondary CUDA stream during layer streaming")
    parser.add_argument("--streaming_fp8_weights", action=argparse.BooleanOptionalAction, default=True, help="Compress CPU host resident weights to FP8 to halve host RAM from 9.2 GB to 4.6 GB (default: True)")
    parser.add_argument("--strict_vram", action="store_true", help="Deprecated alias; use default behavior without --force")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_hf_login", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="triune-2.5b",
        help="Architecture preset (triune-nano, triune-small, triune-2.5b, triune-7b) or Hugging Face model ID",
    )
    parser.add_argument("--lora", action="store_true", help="Enable LoRA parameter-efficient fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank dimension (default: 16)")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA alpha scaling (default: 32.0)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora_targets", type=str, default=None, help="Comma-separated target module names for LoRA")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_experts", type=int)
    parser.add_argument("--tokenizer_path", type=Path, default=Path("triune_tokenizer.json"))
    parser.add_argument("--shuffle_buffer", type=int)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_config")
    parser.add_argument("--num_workers", type=int)
    return parser


def _request_hf_token() -> None:
    if "HF_TOKEN" in os.environ:
        return
    if not sys.stdin.isatty():
        print("⚠️ Non-interactive session: set HF_TOKEN to avoid dataset rate limits")
        return
    token = getpass.getpass("🔑 Enter your Hugging Face API token: ")
    if token.strip():
        os.environ["HF_TOKEN"] = token.strip()
        print("✅ HF_TOKEN set")
    else:
        print("⚠️ No HF token provided; dataset rate limits may apply")


def _checkpoint_run_id(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("wandb_run_id")
    except (OSError, RuntimeError, KeyError):
        return None


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    overrides = {
        key: getattr(args, key)
        for key in (
            "checkpoint_dir", "seq_len", "lr", "total_steps", "batch_size", "grad_accum_steps",
            "save_every", "log_every", "eval_every", "eval_batches", "target_depth_dist", "usage_ema_decay", "bias_strength",
            "balance_coef", "exploration", "exploration_steps", "steer_scale", "use_muon", "muon_lr",
            "galore", "galore_rank", "galore_update_gap", "galore_lr", "galore_weight_decay",
            "dataset_name",
            "dataset_config", "num_workers", "num_layers", "num_experts", "shuffle_buffer",
        )
        if getattr(args, key) is not None
    }
    model_name_str = (args.model_name or "triune-2.5b").lower()
    preset_configs = {
        "triune-nano": {"num_layers": 8, "hidden_dim": 512, "num_heads": 4, "head_dim": 128, "num_experts": 4, "router_prefix_layers": 2, "reflex_exit_layer": 3, "limbic_exit_layer": 6},
        "triune-small": {"num_layers": 14, "hidden_dim": 1024, "num_heads": 8, "head_dim": 128, "num_experts": 4, "router_prefix_layers": 3, "reflex_exit_layer": 5, "limbic_exit_layer": 10},
        "triune-2.5b": {"num_layers": 18, "hidden_dim": 1280, "num_heads": 10, "head_dim": 128, "num_experts": 8, "router_prefix_layers": 3, "reflex_exit_layer": 5, "limbic_exit_layer": 13},
        "triune-7b": {"num_layers": 32, "hidden_dim": 2048, "num_heads": 16, "head_dim": 128, "num_experts": 8, "router_prefix_layers": 4, "reflex_exit_layer": 8, "limbic_exit_layer": 22},
        "triune-base": {"num_layers": 24, "hidden_dim": 1536, "num_heads": 12, "head_dim": 128, "num_experts": 8, "router_prefix_layers": 3, "reflex_exit_layer": 6, "limbic_exit_layer": 16},
        "triune-moe": {"num_layers": 32, "hidden_dim": 1536, "num_heads": 12, "head_dim": 128, "num_experts": 16, "router_prefix_layers": 3, "reflex_exit_layer": 6, "limbic_exit_layer": 16},
    }
    if model_name_str in preset_configs:
        for k, v in preset_configs[model_name_str].items():
            overrides.setdefault(k, v)

    if overrides.get("checkpoint_dir"):
        overrides["checkpoint_dir"] = str(overrides["checkpoint_dir"])
    overrides["use_fp4"] = args.use_fp4
    overrides["use_fp8"] = args.use_fp8
    overrides["streaming"] = args.streaming
    overrides["streaming_chunk_size"] = args.streaming_chunk_size
    overrides["streaming_pin_memory"] = args.streaming_pin_memory
    overrides["streaming_prefetch"] = args.streaming_prefetch
    overrides["streaming_fp8_weights"] = getattr(args, "streaming_fp8_weights", False)
    config = build_config(overrides)
    if args.model_name:
        config["model_name"] = args.model_name
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    else:
        device = torch.device("cpu")
        print("⚠️ [Hardware] CUDA not detected. Training on CPU.", flush=True)
    from triune.runtime.resource_manager import DynamicResourceManager


    try:
        tokenizer = load_tokenizer(args.tokenizer_path)
        config["vocab_size"] = tokenizer.get_vocab_size()
        sep_token_id = tokenizer.token_to_id("[SEP]")
        if sep_token_id is None:
            raise ValueError("Tokenizer must define a [SEP] special token")

        # Run feasibility assessment and display transparent breakdown
        assessment = DynamicResourceManager.assess_feasibility(
            config, device=device, user_overrides=overrides, safety_ceiling=0.85
        )
        print(DynamicResourceManager.format_assessment_report(assessment), flush=True)

        if args.streaming:
            print(f"🌊 [Layer Streaming Engine] Active: chunk size {args.streaming_chunk_size}. Layers reside in CPU RAM and stream to GPU on-demand (<1.5 GB VRAM target).", flush=True)

        # Validate and allot resources; respects user --force override and explicit flags
        allotment = DynamicResourceManager.validate_and_allot(
            config,
            device=device,
            force=args.force or args.streaming,
            auto_fit=args.auto_fit,
            user_overrides=overrides,
            safety_ceiling=0.85,
        )
        config = allotment.config

        if allotment.adjustment_reason:
            print(f"⚙️ [Resource Manager] Status: {allotment.adjustment_reason}", flush=True)

        if not args.no_hf_login:
            _request_hf_token()

        resume_path = args.resume_latest or (Path(config["checkpoint_dir"]) / "best.pt" if args.resume_best else None)
        logger = NullLogger() if args.no_wandb else WandbLogger(
            project="triune-transformer", config=config, run_id=_checkpoint_run_id(resume_path)
        )

        # Build model with BF16 default dtype
        if model_name_str not in preset_configs and ("/" in args.model_name or not Path(args.model_name).exists()):
            try:
                from transformers import AutoModelForCausalLM
                print(f"⏳ Loading Hugging Face model '{args.model_name}'...", flush=True)
                raw_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
            except Exception as e:
                print(f"⚠️ Could not load via transformers ({e}); building Triune model...", flush=True)
                torch.set_default_dtype(torch.bfloat16)
                try:
                    raw_model = build_model(config)
                finally:
                    torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.bfloat16)
            try:
                raw_model = build_model(config)
            finally:
                torch.set_default_dtype(torch.float32)

        if args.streaming:
            from triune.runtime.streaming import enable_layer_streaming
            enable_layer_streaming(
                raw_model,
                device=device,
                chunk_size=args.streaming_chunk_size,
                pin_memory=args.streaming_pin_memory,
                async_prefetch=args.streaming_prefetch,
                fp8_weights=args.streaming_fp8_weights,
            )
            model = raw_model
        else:
            model = DynamicResourceManager.safe_to_device(
                raw_model, device=device, dtype=torch.bfloat16, force=args.force
            )

        if args.lora:
            from triune.trainer.finetune import LoRAConfig, TriuneFineTuner
            targets = [t.strip() for t in args.lora_targets.split(",")] if args.lora_targets else None
            lora_cfg = LoRAConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=targets,
            )
            TriuneFineTuner(model, lora_cfg)
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ LoRA attached: rank={args.lora_rank}, alpha={args.lora_alpha}, trainable={trainable_p:,}", flush=True)

        if args.grad_checkpoint is not False and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("✅ Selective gradient checkpointing enabled", flush=True)

        print(f"Params: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
        optimizer = build_optimizer(model, config)
        print(f"⏳ Connecting dataset stream: '{config.get('dataset_name')}' ({config.get('dataset_config')})...", flush=True)
        train_loader = build_dataloader(tokenizer, config, sep_token_id, is_holdout=False)
        eval_loader = build_dataloader(tokenizer, config, sep_token_id, is_holdout=True)
        print("✅ Dataset stream configured.", flush=True)
        if config.get("use_fp8", False):
            fp8_count = sum(1 for m in model.modules() if getattr(m, '_triune_fp8_aware', False))
            if fp8_count == 0:
                print("⚠️ [FP8] use_fp8=True but no FP8Linear modules found in model. Check model factory.", flush=True)

            use_te = config.get("use_te", True)
            precision_context = build_fp8_precision_context(device=device, use_te=use_te)
            desc = getattr(precision_context, "description", "FP8 precision context active")
            if "falling back" in desc.lower():
                print(f"⚠️ {desc}", flush=True)
            else:
                print(f"✅ {desc}", flush=True)
        elif config.get("use_fp4"):
            precision_context = build_precision_context(use_fp4=True, device=device)
        else:
            precision_context = lambda: bf16_autocast(device.type)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            config=config,
            device=device,
            precision_context=precision_context,
            logger=logger,
        )
        if args.compile:
            trainer.compile()
            print("✅ torch.compile enabled", flush=True)
        if resume_path and not args.fresh:
            trainer.resume(resume_path, weights_only=args.resume_best)
            print(f"✅ Resumed from {resume_path}", flush=True)
        elif args.fresh:
            print("🆕 Fresh start", flush=True)
        print(f"🚀 Training from step {trainer.engine.step} to {config['total_steps']}", flush=True)
        return trainer.fit()
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
