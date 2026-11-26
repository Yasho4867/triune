"""CLI entry point for verifiable Stage-3 Tool-Use, Agent-Harness & Self-Improvement GRPO RL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lucid_machines.configs.config import build_config
from lucid_machines.data import load_tokenizer
from lucid_machines.model.base import get_model_spec
from lucid_machines.model.transformer import TriuneTransformer
from lucid_machines.trainer.finetune import LoRAConfig, TriuneFineTuner
from lucid_machines.trainer.grpo import GRPOHarnessTrainer, load_verifiable_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3: Verifiable Tool/Harness GRPO & Self-Improvement RL")
    parser.add_argument("--model_config", "--model-config", dest="model_config", type=str, default="junkie-3b")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional trained Triune checkpoint to tune")
    parser.add_argument("--tokenizer_path", type=Path, default=Path("triune_tokenizer.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/grpo_harness"))
    parser.add_argument("--tasks_jsonl", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--use_fp8", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _load_model_config(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    cfg_candidate = Path(__file__).resolve().parents[1] / "models" / "configs" / f"{name_or_path}.json"
    if cfg_candidate.is_file():
        return json.loads(cfg_candidate.read_text(encoding="utf-8"))
    spec = get_model_spec(name_or_path)
    return {
        "model_name": name_or_path,
        "vocab_size": 32000,
        "hidden_dim": spec.hidden_dim,
        "num_layers": spec.num_layers,
        "num_heads": spec.num_heads,
        "head_dim": spec.head_dim,
        "num_experts": spec.num_experts,
        "router_prefix_layers": spec.router_prefix_layers,
        "reflex_exit_layer": spec.reflex_exit_layer,
        "limbic_exit_layer": spec.limbic_exit_layer,
    }


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    tasks = load_verifiable_tasks(args.tasks_jsonl)
    raw_cfg = _load_model_config(args.model_config)
    raw_cfg["use_fp8"] = bool(args.use_fp8 or raw_cfg.get("use_fp8", False))
    cfg = build_config(raw_cfg)

    if args.dry_run:
        out = {
            "dry_run": True,
            "model": args.model_config,
            "tasks": len(tasks),
            "categories": sorted({t.category for t in tasks}),
            "group_size": args.group_size,
            "use_lora": bool(args.use_lora),
        }
        print(f"[GRPO DryRun] {json.dumps(out)}", flush=True)
        return out

    tokenizer = load_tokenizer(args.tokenizer_path)
    cfg["vocab_size"] = tokenizer.get_vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_default_dtype(torch.bfloat16)
    try:
        build_dev = device if device.type == "cuda" else None
        if build_dev is not None:
            with torch.device(build_dev):
                model = TriuneTransformer(
                    vocab_size=int(cfg["vocab_size"]),
                    hidden_dim=int(cfg["hidden_dim"]),
                    num_layers=int(cfg["num_layers"]),
                    num_heads=int(cfg["num_heads"]),
                    head_dim=int(cfg["head_dim"]),
                    num_experts=int(cfg["num_experts"]),
                    top_k_experts=int(raw_cfg.get("top_k_experts", 1)),
                    shared_expert=bool(raw_cfg.get("shared_expert", True)),
                    shared_expert_scale=float(raw_cfg.get("shared_expert_scale", 0.5)),
                    expert_hidden_multiplier=int(raw_cfg.get("expert_hidden_multiplier", 6)),
                    router_prefix_layers=int(cfg["router_prefix_layers"]),
                    reflex_exit_layer=int(cfg["reflex_exit_layer"]),
                    limbic_exit_layer=int(cfg["limbic_exit_layer"]),
                    use_fp8=bool(cfg.get("use_fp8", False)),
                )
        else:
            model = TriuneTransformer(
                vocab_size=int(cfg["vocab_size"]),
                hidden_dim=int(cfg["hidden_dim"]),
                num_layers=int(cfg["num_layers"]),
                num_heads=int(cfg["num_heads"]),
                head_dim=int(cfg["head_dim"]),
                num_experts=int(cfg["num_experts"]),
                top_k_experts=int(raw_cfg.get("top_k_experts", 1)),
                shared_expert=bool(raw_cfg.get("shared_expert", True)),
                shared_expert_scale=float(raw_cfg.get("shared_expert_scale", 0.5)),
                expert_hidden_multiplier=int(raw_cfg.get("expert_hidden_multiplier", 6)),
                router_prefix_layers=int(cfg["router_prefix_layers"]),
                reflex_exit_layer=int(cfg["reflex_exit_layer"]),
                limbic_exit_layer=int(cfg["limbic_exit_layer"]),
                use_fp8=bool(cfg.get("use_fp8", False)),
            )
    finally:
        torch.set_default_dtype(torch.float32)

    if args.checkpoint is not None and args.checkpoint.is_file():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        sd = ckpt.get("model_state", ckpt)
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"✅ Loaded base weights from {args.checkpoint}", flush=True)

    fine_tuner = None
    if args.use_lora:
        lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=float(args.lora_rank * 2), quantize_base=False)
        fine_tuner = TriuneFineTuner(model, lora_cfg)
        print(f"✅ GRPO LoRA attached (rank={args.lora_rank})", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    trainer = GRPOHarnessTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        tasks=tasks,
        device=device,
        output_dir=args.output_dir,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        kl_coef=args.kl_coef,
    )
    summary = trainer.train(args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if fine_tuner is not None:
        fine_tuner.save_adapter(args.output_dir / "adapter")
    torch.save({"model_state": model.state_dict(), "grpo_summary": summary}, args.output_dir / "grpo_latest.pt")
    return summary


if __name__ == "__main__":
    main()
