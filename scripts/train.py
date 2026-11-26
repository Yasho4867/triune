"""Command-line entry point for callable Lucid Machines training."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
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

from lucid_machines.configs.config import build_config
from lucid_machines.data import build_dataloader, load_tokenizer
from lucid_machines.model import build_model
from lucid_machines.optim.factory import build_optimizer
from lucid_machines.recipes import bf16_autocast, build_fp8_precision_context, build_precision_context
from lucid_machines.trainer import NullLogger, Trainer, WandbLogger


def _depth_distribution(value: str) -> list[float]:
    try:
        return [float(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Triune Transformer")
    parser.add_argument("--resume_best", "--resume-best", dest="resume_best", action="store_true")
    parser.add_argument("--resume_latest", "--resume-latest", dest="resume_latest", type=Path)
    parser.add_argument(
        "--auto_resume",
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="Automatically resume from <checkpoint_dir>/latest.pt if present",
    )
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", dest="checkpoint_dir", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seq_len", "--seq-len", dest="seq_len", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_steps", "--warmup-steps", dest="warmup_steps", type=int)
    parser.add_argument("--total_steps", "--steps", dest="total_steps", type=int)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("--grad_accum_steps", "--grad-accum-steps", dest="grad_accum_steps", type=int)
    parser.add_argument(
        "--depth_mode",
        "--depth-mode",
        dest="depth_mode",
        choices=("dynamic", "joint", "cortex"),
        help="Depth routing supervision mode",
    )
    parser.add_argument(
        "--grad_checkpoint",
        "--grad-checkpoint",
        dest="grad_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save_every", "--save-every", dest="save_every", type=int)
    parser.add_argument("--log_every", "--log-every", dest="log_every", type=int)
    parser.add_argument("--eval_every", "--eval-every", dest="eval_every", type=int)
    parser.add_argument("--eval_batches", "--eval-batches", dest="eval_batches", type=int)
    parser.add_argument("--target_depth_dist", type=_depth_distribution)
    parser.add_argument("--usage_ema_decay", type=float)
    parser.add_argument("--bias_strength", type=float)
    parser.add_argument("--balance_coef", type=float)
    parser.add_argument("--exploration", choices=("linear", "cosine", "none"))
    parser.add_argument("--exploration_steps", type=int)
    parser.add_argument("--steer_scale", "--steer-scale", dest="steer_scale", type=float)
    parser.add_argument(
        "--steer_dirs",
        "--steer-dirs",
        dest="steer_dirs",
        type=int,
        help="Number of orthogonal activation subspace directions for CentroidSteer",
    )
    parser.add_argument(
        "--steer_update_gap",
        "--steer-update-gap",
        dest="steer_update_gap",
        type=int,
        default=None,
        help="Steps between CentroidSteer C_e activation-subspace refreshes (default: config value)",
    )
    parser.add_argument(
        "--steer_mode",
        "--steer-mode",
        dest="steer_mode",
        choices=("centroid", "upgraded", "legacy", "nosteer", "random_same_mem"),
        default=None,
        help="CentroidSteer mode",
    )
    parser.add_argument("--use_fp4", "--use_nvfp4", dest="use_fp4", action="store_true")
    parser.add_argument("--use_fp8", action="store_true", help="Enable native FP8 (E4M3) precision context")
    parser.add_argument(
        "--use_muon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Muon for non-expert 2D hidden weights",
    )
    parser.add_argument(
        "--use_triangular_core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fused triangular block update kernel for CentroidSteer",
    )
    parser.add_argument(
        "--use_8bit_base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use 8-bit AdamW base optimizer if bitsandbytes is available",
    )
    parser.add_argument("--muon_lr", type=float, help="Muon learning rate (default: 0.02)")
    parser.add_argument(
        "--galore",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable GaLore low-rank gradient projection for expert weights",
    )
    parser.add_argument(
        "--galore_rank", "--rank", dest="galore_rank", type=int, help="GaLore projection subspace rank (default: 256)"
    )
    parser.add_argument(
        "--galore_update_gap", type=int, help="GaLore subspace update frequency in steps (default: 200)"
    )
    parser.add_argument("--galore_lr", type=float, help="GaLore learning rate (default: 1e-4)")
    parser.add_argument("--galore_weight_decay", type=float, help="GaLore weight decay (default: 0.05)")
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable multi-GPU Distributed Data Parallel (DDP) training (auto-detected under torchrun)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override resource manager feasibility recommendations and proceed regardless of VRAM budget",
    )
    parser.add_argument(
        "--auto_fit",
        action="store_true",
        default=False,
        help="Automatically adopt recommended microbatch and grad_accum settings if not explicitly specified",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable AirLLM-style layer streaming to train large models on consumer GPUs with <1.5 GB VRAM",
    )
    parser.add_argument(
        "--streaming_chunk_size",
        type=int,
        default=1,
        help="Number of layers to stream/cache simultaneously (default: 1)",
    )
    parser.add_argument(
        "--streaming_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pin host memory for PCIe DMA in streaming mode (auto-detected based on RAM)",
    )
    parser.add_argument(
        "--streaming_prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefetch next layer on secondary CUDA stream during layer streaming",
    )
    parser.add_argument(
        "--streaming_fp8_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress CPU host resident weights to FP8 to halve host RAM from 9.2 GB to 4.6 GB (default: True)",
    )
    parser.add_argument(
        "--strict_vram", action="store_true", help="Deprecated alias; use default behavior without --force"
    )
    parser.add_argument("--no_wandb", "--no-wandb", dest="no_wandb", action="store_true")
    parser.add_argument("--no_hf_login", "--no-hf-login", dest="no_hf_login", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="triune-2.5b",
        help="Architecture preset (junkie-mini, junkie-1b, junkie-3b, junkie-7b, junkie, triune-small, triune-base, etc.) or Hugging Face model ID",
    )
    parser.add_argument(
        "--model_config",
        "--model-config",
        "--config",
        dest="model_config",
        type=str,
        default=None,
        help="Model config preset name (e.g. junkie-mini, junkie-1b, junkie-3b, junkie-7b, junkie, triune-base) or path to a models/configs/*.json file",
    )
    parser.add_argument("--lora", action="store_true", help="Enable LoRA parameter-efficient fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank dimension (default: 16)")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA alpha scaling (default: 32.0)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora_targets", type=str, default=None, help="Comma-separated target module names for LoRA")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_experts", type=int)
    parser.add_argument(
        "--tokenizer_path", "--tokenizer-path", dest="tokenizer_path", type=Path, default=Path("triune_tokenizer.json")
    )
    parser.add_argument("--shuffle_buffer", type=int)
    parser.add_argument("--dataset_name", "--dataset-name", dest="dataset_name")
    parser.add_argument("--dataset_config", "--dataset-config", dest="dataset_config")
    parser.add_argument(
        "--dataset_mix",
        "--dataset-mix",
        dest="dataset_mix",
        type=str,
        default=None,
        help="Weighted curriculum preset (junkie_stage1, junkie_stage2, junkie_tool_sft) or repo:cfg:weight list",
    )
    parser.add_argument(
        "--lr_schedule",
        "--lr-schedule",
        dest="lr_schedule",
        choices=("cosine", "wsd"),
        default=None,
        help="Learning rate schedule: cosine (default) or wsd (Warmup-Stable-Decay)",
    )
    parser.add_argument(
        "--wsd_decay_frac",
        "--wsd-decay-frac",
        dest="wsd_decay_frac",
        type=float,
        default=None,
        help="Fraction of post-warmup training allocated to WSD cooldown decay (default: 0.15)",
    )
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int)
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


class _HFTokenizerAdapter:
    """Adapts a Hugging Face tokenizer to Lucid Machines' [PAD]/[SEP] + .encode().ids interface."""

    def __init__(self, hf_tokenizer, model_vocab_size: int | None = None):
        self._tok = hf_tokenizer
        eos = getattr(hf_tokenizer, "eos_token_id", None)
        pad = getattr(hf_tokenizer, "pad_token_id", None)
        self.pad_token_id = int(pad if pad is not None else (eos if eos is not None else 0))
        self.sep_token_id = int(eos if eos is not None else self.pad_token_id)
        tok_len = int(len(hf_tokenizer))
        self._vocab_size = max(tok_len, int(model_vocab_size)) if model_vocab_size else tok_len

    def encode(self, text: str):
        ids = self._tok.encode(text, add_special_tokens=False)
        return type("HFEncoded", (), {"ids": ids})()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str):
        if token == "[PAD]":
            return self.pad_token_id
        if token == "[SEP]":
            return self.sep_token_id
        tid = self._tok.convert_tokens_to_ids(token)
        unk = getattr(self._tok, "unk_token_id", None)
        return None if (tid is None or (unk is not None and tid == unk)) else int(tid)

    def get_vocab_size(self) -> int:
        return self._vocab_size


def _checkpoint_run_id(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True).get("wandb_run_id")
    except (OSError, RuntimeError, KeyError):
        return None


def _resolve_local_model_config(raw_spec: str) -> tuple[dict | None, str | None]:
    """Resolve a JSON path or preset stem under models/configs/<name>.json using pathlib."""
    spec_str = raw_spec.strip()
    if not spec_str:
        return None, None
    candidate = Path(spec_str)
    if candidate.is_file() and candidate.suffix.lower() == ".json":
        return json.loads(candidate.read_text(encoding="utf-8-sig")), candidate.stem
    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "models" / "configs"
    for stem in (spec_str, spec_str.lower()):
        cfg_file = configs_dir / (stem if stem.endswith(".json") else f"{stem}.json")
        if cfg_file.is_file():
            return json.loads(cfg_file.read_text(encoding="utf-8-sig")), cfg_file.stem
    return None, None


def _count_meta_triune_params(config: dict) -> int:
    """Measure exact unique parameters on torch.device('meta') without allocating RAM/VRAM."""
    from lucid_machines.model.config import (
        EXPERT_HIDDEN_MULTIPLIER,
        ROPE_MAX_SEQ_LEN,
        SHARED_EXPERT,
        SHARED_EXPERT_SCALE,
        TOP_K_EXPERTS,
    )
    from lucid_machines.model.transformer import TriuneTransformer

    with torch.device("meta"):
        meta_model = TriuneTransformer(
            vocab_size=int(config.get("vocab_size", 32000)),
            hidden_dim=int(config.get("hidden_dim", 1536)),
            num_layers=int(config.get("num_layers", 24)),
            num_heads=int(config.get("num_heads", 12)),
            head_dim=int(config.get("head_dim", 128)),
            num_experts=int(config.get("num_experts", 8)),
            top_k_experts=int(config.get("top_k_experts", TOP_K_EXPERTS)),
            shared_expert=bool(config.get("shared_expert", SHARED_EXPERT)),
            shared_expert_scale=float(config.get("shared_expert_scale", SHARED_EXPERT_SCALE)),
            expert_hidden_multiplier=int(config.get("expert_hidden_multiplier", EXPERT_HIDDEN_MULTIPLIER)),
            rope_max_seq_len=int(config.get("rope_max_seq_len", ROPE_MAX_SEQ_LEN)),
            router_prefix_layers=int(config.get("router_prefix_layers", 3)),
            reflex_exit_layer=int(config.get("reflex_exit_layer", 6)),
            limbic_exit_layer=int(config.get("limbic_exit_layer", 16)),
            target_depth_dist=tuple(config.get("target_depth_dist", (0.34, 0.33, 0.33))),
            balance_coef=float(config.get("balance_coef", 0.30)),
            use_fp4=bool(config.get("use_fp4", False)),
            use_fp8=bool(config.get("use_fp8", False)),
            streaming_fp8_weights=bool(config.get("streaming_fp8_weights", False)),
        )
    seen: set[int] = set()
    total = 0
    for p in meta_model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


def _build_triune_from_config(config: dict):
    """Instantiate TriuneTransformer directly when extra config fields are present, else via build_model."""
    from lucid_machines.model.config import (
        DEPTH_BALANCE_COEF,
        EXPERT_HIDDEN_MULTIPLIER,
        ROPE_MAX_SEQ_LEN,
        SHARED_EXPERT,
        SHARED_EXPERT_SCALE,
        TARGET_DEPTH_DIST,
        TOP_K_EXPERTS,
    )
    from lucid_machines.model.transformer import TriuneTransformer

    if any(
        k in config
        for k in (
            "top_k_experts",
            "shared_expert",
            "shared_expert_scale",
            "expert_hidden_multiplier",
            "rope_max_seq_len",
        )
    ):
        return TriuneTransformer(
            vocab_size=int(config.get("vocab_size", 32000)),
            hidden_dim=int(config.get("hidden_dim", 1536)),
            num_layers=int(config.get("num_layers", 24)),
            num_heads=int(config.get("num_heads", 12)),
            head_dim=int(config.get("head_dim", 128)),
            num_experts=int(config.get("num_experts", 8)),
            top_k_experts=int(config.get("top_k_experts", TOP_K_EXPERTS)),
            shared_expert=bool(config.get("shared_expert", SHARED_EXPERT)),
            shared_expert_scale=float(config.get("shared_expert_scale", SHARED_EXPERT_SCALE)),
            expert_hidden_multiplier=int(config.get("expert_hidden_multiplier", EXPERT_HIDDEN_MULTIPLIER)),
            rope_max_seq_len=int(config.get("rope_max_seq_len", ROPE_MAX_SEQ_LEN)),
            router_prefix_layers=int(config.get("router_prefix_layers", 3)),
            reflex_exit_layer=int(config.get("reflex_exit_layer", 6)),
            limbic_exit_layer=int(config.get("limbic_exit_layer", 16)),
            target_depth_dist=tuple(config.get("target_depth_dist", TARGET_DEPTH_DIST)),
            balance_coef=float(config.get("balance_coef", DEPTH_BALANCE_COEF)),
            use_fp4=bool(config.get("use_fp4", False)),
            use_fp8=bool(config.get("use_fp8", False)),
            streaming_fp8_weights=bool(config.get("streaming_fp8_weights", False)),
        )
    return build_model(config)


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    overrides = {
        key: getattr(args, key)
        for key in (
            "checkpoint_dir",
            "seq_len",
            "lr",
            "warmup_steps",
            "total_steps",
            "batch_size",
            "grad_accum_steps",
            "depth_mode",
            "save_every",
            "log_every",
            "eval_every",
            "eval_batches",
            "target_depth_dist",
            "usage_ema_decay",
            "bias_strength",
            "balance_coef",
            "exploration",
            "exploration_steps",
            "steer_scale",
            "steer_update_gap",
            "use_muon",
            "muon_lr",
            "use_triangular_core",
            "use_8bit_base",
            "galore",
            "galore_rank",
            "galore_update_gap",
            "galore_lr",
            "galore_weight_decay",
            "dataset_name",
            "dataset_config",
            "dataset_mix",
            "lr_schedule",
            "wsd_decay_frac",
            "num_workers",
            "num_layers",
            "num_experts",
            "shuffle_buffer",
        )
        if getattr(args, key) is not None
    }
    requested_steer_mode = getattr(args, "steer_mode", None)
    if requested_steer_mode == "nosteer":
        overrides["steer_scale"] = 0.0
        overrides["steer_mode"] = "legacy"
    elif requested_steer_mode == "centroid":
        overrides["steer_mode"] = "upgraded"
    elif requested_steer_mode is not None:
        overrides["steer_mode"] = requested_steer_mode
    if args.steer_dirs is not None:
        overrides["num_steer_dirs"] = int(args.steer_dirs)

    from lucid_machines.model.base import _LEGACY_PRESET_ALIASES, CANONICAL_MODEL_SPECS, get_model_spec

    requested_model = (args.model_config or args.model_name or "triune-2.5b").strip()
    local_json_cfg, local_json_stem = _resolve_local_model_config(requested_model)
    model_name_str = requested_model.lower()
    is_canonical_preset = model_name_str in CANONICAL_MODEL_SPECS or model_name_str in _LEGACY_PRESET_ALIASES
    is_local_triune_config = local_json_cfg is not None

    if local_json_cfg is not None:
        for k in (
            "vocab_size",
            "num_layers",
            "hidden_dim",
            "num_heads",
            "head_dim",
            "num_experts",
            "top_k_experts",
            "shared_expert",
            "shared_expert_scale",
            "expert_hidden_multiplier",
            "rope_max_seq_len",
            "router_prefix_layers",
            "reflex_exit_layer",
            "limbic_exit_layer",
            "target_depth_dist",
            "balance_coef",
            "use_fp4",
            "use_fp8",
        ):
            if k in local_json_cfg and local_json_cfg[k] is not None:
                overrides.setdefault(k, local_json_cfg[k])
    elif is_canonical_preset:
        spec = get_model_spec(model_name_str)
        for k in (
            "num_layers",
            "hidden_dim",
            "num_heads",
            "head_dim",
            "num_experts",
            "top_k_experts",
            "shared_expert",
            "shared_expert_scale",
            "expert_hidden_multiplier",
            "rope_max_seq_len",
            "router_prefix_layers",
            "reflex_exit_layer",
            "limbic_exit_layer",
        ):
            overrides.setdefault(k, getattr(spec, k))

    if overrides.get("checkpoint_dir"):
        overrides["checkpoint_dir"] = str(overrides["checkpoint_dir"])
    overrides["use_fp4"] = args.use_fp4 or bool(overrides.get("use_fp4", False))
    overrides["use_fp8"] = args.use_fp8 or bool(overrides.get("use_fp8", False))
    overrides["streaming"] = args.streaming
    overrides["streaming_chunk_size"] = args.streaming_chunk_size
    overrides["streaming_pin_memory"] = args.streaming_pin_memory
    overrides["streaming_prefetch"] = args.streaming_prefetch
    overrides["streaming_fp8_weights"] = bool(args.streaming and getattr(args, "streaming_fp8_weights", False))
    config = build_config(overrides)
    if requested_steer_mode == "nosteer":
        config["steer_mode"] = "nosteer"
    elif requested_steer_mode == "centroid":
        config["steer_mode"] = "centroid"
    config["model_name"] = local_json_stem or requested_model
    if not args.dry_run:
        Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    dist_requested = args.distributed
    if dist_requested is None and not args.dry_run:
        dist_requested = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1

    from lucid_machines.trainer.distributed import (
        cleanup_distributed,
        setup_distributed,
        wrap_distributed_model,
    )

    dist_ctx = setup_distributed() if dist_requested else None
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if dist_ctx and dist_ctx.is_distributed:
        device = torch.device(f"cuda:{dist_ctx.local_rank}" if torch.cuda.is_available() else "cpu")
        if dist_ctx.is_main_process:
            print(
                f"🌐 [Distributed DDP] Initialized process {dist_ctx.rank}/{dist_ctx.world_size} on {device}",
                flush=True,
            )
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if not args.dry_run:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if sys.platform == "win32":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        else:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    else:
        device = torch.device("cpu")
        if not args.dry_run:
            print("⚠️ [Hardware] CUDA not detected. Training on CPU.", flush=True)
    from lucid_machines.runtime.resource_manager import DynamicResourceManager

    logger = None
    try:
        is_hf_candidate = bool(
            not is_canonical_preset
            and not is_local_triune_config
            and requested_model
            and (
                "/" in requested_model
                or (Path(requested_model).is_dir() and (Path(requested_model) / "config.json").exists())
            )
        )
        hf_cfg = None
        tokenizer = None
        if is_hf_candidate and not args.dry_run:
            try:
                from transformers import AutoConfig, AutoTokenizer

                hf_cfg = AutoConfig.from_pretrained(requested_model)
                config["hidden_dim"] = int(getattr(hf_cfg, "hidden_size", config["hidden_dim"]))
                config["num_layers"] = int(getattr(hf_cfg, "num_hidden_layers", config["num_layers"]))
                config["num_heads"] = int(getattr(hf_cfg, "num_attention_heads", config["num_heads"]))
                if args.depth_mode is None:
                    config["depth_mode"] = "cortex"
                if args.tokenizer_path == Path("triune_tokenizer.json"):
                    hf_tok = AutoTokenizer.from_pretrained(requested_model)
                    tokenizer = _HFTokenizerAdapter(hf_tok, model_vocab_size=getattr(hf_cfg, "vocab_size", None))
            except Exception as exc:
                print(f"⚠️ Hugging Face metadata/tokenizer load skipped ({exc}); using local defaults.", flush=True)

        if args.dry_run:
            if args.tokenizer_path.is_file():
                tokenizer = load_tokenizer(args.tokenizer_path)
                config["vocab_size"] = tokenizer.get_vocab_size()
            meta_params = _count_meta_triune_params(config)
            assessment = DynamicResourceManager.assess_feasibility(
                config, device=device, user_overrides=overrides, safety_ceiling=0.85
            )
            print(DynamicResourceManager.format_assessment_report(assessment), flush=True)
            print(f"[DryRun] model={config['model_name']} meta_params={meta_params:,}", flush=True)
            return {
                "dry_run": True,
                "model_name": config["model_name"],
                "total_params": meta_params,
                "config": config,
            }

        if tokenizer is None:
            tokenizer = load_tokenizer(args.tokenizer_path)

        config["vocab_size"] = tokenizer.get_vocab_size()
        sep_token_id = tokenizer.token_to_id("[SEP]")
        if sep_token_id is None:
            sep_token_id = getattr(tokenizer, "eos_token_id", None)
        if sep_token_id is None:
            raise ValueError("Tokenizer must define a [SEP] or EOS special token")

        # Run feasibility assessment and display transparent breakdown
        assessment = DynamicResourceManager.assess_feasibility(
            config, device=device, user_overrides=overrides, safety_ceiling=0.85
        )
        print(DynamicResourceManager.format_assessment_report(assessment), flush=True)

        if args.streaming:
            print(
                f"🌊 [Layer Streaming Engine] Active: chunk size {args.streaming_chunk_size}. Layers reside in CPU RAM and stream to GPU on-demand (<1.5 GB VRAM target).",
                flush=True,
            )

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

        resume_path = args.resume_latest
        if resume_path is not None and resume_path.is_dir():
            resume_path = resume_path / "latest.pt"
        if resume_path is None and args.auto_resume:
            auto_candidate = Path(config["checkpoint_dir"]) / "latest.pt"
            if auto_candidate.is_file():
                resume_path = auto_candidate
        if resume_path is None and args.resume_best:
            resume_path = Path(config["checkpoint_dir"]) / "best.pt"

        logger = (
            NullLogger()
            if args.no_wandb or (dist_ctx and not dist_ctx.is_main_process)
            else WandbLogger(project="lucid_machines", config=config, run_id=_checkpoint_run_id(resume_path))
        )

        import lucid_machines.model.config as model_defaults

        prev_steer_dirs = model_defaults.CENTROID_SUBSPACE_DIRS
        if args.steer_dirs is not None:
            model_defaults.CENTROID_SUBSPACE_DIRS = max(1, int(args.steer_dirs))

        try:
            # Build model with BF16 default dtype (directly on CUDA when not streaming to avoid CPU RAM spike)
            build_ctx = torch.device(device) if (not args.streaming and device.type == "cuda") else contextlib.nullcontext()
            if is_hf_candidate:
                try:
                    from transformers import AutoModelForCausalLM

                    print(f"⏳ Loading Hugging Face model '{requested_model}'...", flush=True)
                    raw_model = AutoModelForCausalLM.from_pretrained(requested_model, torch_dtype=torch.bfloat16)
                except Exception as e:
                    print(f"⚠️ Could not load via transformers ({e}); building Triune model...", flush=True)
                    torch.set_default_dtype(torch.bfloat16)
                    try:
                        with build_ctx:
                            raw_model = _build_triune_from_config(config)
                    finally:
                        torch.set_default_dtype(torch.float32)
            else:
                torch.set_default_dtype(torch.bfloat16)
                try:
                    with build_ctx:
                        raw_model = _build_triune_from_config(config)
                finally:
                    torch.set_default_dtype(torch.float32)
        finally:
            model_defaults.CENTROID_SUBSPACE_DIRS = prev_steer_dirs

        if args.streaming:
            from lucid_machines.runtime.streaming import enable_layer_streaming

            stream_layers_path = None
            if hasattr(raw_model, "model") and hasattr(raw_model.model, "layers"):
                stream_layers_path = "model.layers"
            elif hasattr(raw_model, "transformer") and hasattr(raw_model.transformer, "h"):
                stream_layers_path = "transformer.h"

            enable_layer_streaming(
                raw_model,
                device=device,
                chunk_size=args.streaming_chunk_size,
                pin_memory=args.streaming_pin_memory,
                async_prefetch=args.streaming_prefetch,
                fp8_weights=args.streaming_fp8_weights,
                layers_path=stream_layers_path,
            )
            model = raw_model
        else:
            model = DynamicResourceManager.safe_to_device(
                raw_model, device=device, dtype=torch.bfloat16, force=args.force
            )

        if args.lora:
            from lucid_machines.trainer import finetune as _finetune_mod

            _FineTuner = (
                getattr(_finetune_mod, "LucidFineTuner", None)
                or getattr(_finetune_mod, "Lucid MachinesFineTuner", None)
                or _finetune_mod.TriuneFineTuner
            )
            targets = [t.strip() for t in args.lora_targets.split(",")] if args.lora_targets else None
            lora_cfg = _finetune_mod.LoRAConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=targets,
            )
            _FineTuner(model, lora_cfg)
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"✅ LoRA attached: rank={args.lora_rank}, alpha={args.lora_alpha}, trainable={trainable_p:,}",
                flush=True,
            )

        if args.grad_checkpoint is not False:
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
                print("✅ Selective gradient checkpointing enabled", flush=True)
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                print("✅ Selective gradient checkpointing enabled", flush=True)

        print(f"Params: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
        opt_config = dict(config)
        if opt_config.get("steer_mode") == "nosteer":
            opt_config["steer_mode"] = "legacy"
            opt_config["steer_scale"] = 0.0
        elif opt_config.get("steer_mode") == "centroid":
            opt_config["steer_mode"] = "upgraded"
        optimizer = build_optimizer(model, opt_config)
        if args.steer_dirs is not None and hasattr(optimizer, "num_steer_dirs"):
            optimizer.num_steer_dirs = max(1, int(args.steer_dirs))
        if requested_steer_mode in ("upgraded", "legacy", "random_same_mem") and hasattr(optimizer, "steer_mode"):
            optimizer.steer_mode = requested_steer_mode
        elif requested_steer_mode == "centroid" and hasattr(optimizer, "steer_mode"):
            optimizer.steer_mode = "upgraded"
        elif requested_steer_mode == "nosteer" and hasattr(optimizer, "steer_scale"):
            optimizer.steer_scale = 0.0
        print(
            f"⏳ Connecting dataset stream: '{config.get('dataset_name')}' ({config.get('dataset_config')})...",
            flush=True,
        )
        train_loader = build_dataloader(tokenizer, config, sep_token_id, is_holdout=False)
        eval_loader = build_dataloader(tokenizer, config, sep_token_id, is_holdout=True)
        print("✅ Dataset stream configured.", flush=True)
        if config.get("use_fp8", False):
            fp8_count = sum(
                1
                for m in model.modules()
                if getattr(m, "_lm_fp8_aware", False) or getattr(m, "_triune_fp8_aware", False)
            )
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
        if dist_ctx and dist_ctx.is_distributed:
            model = wrap_distributed_model(
                model,
                device_ids=[dist_ctx.local_rank] if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
            print(f"🌐 [Distributed DDP] Model wrapped for rank {dist_ctx.rank}", flush=True)

        if args.compile:
            trainer.compile()
            print("✅ torch.compile enabled", flush=True)
        if resume_path and not args.fresh and resume_path.is_file():
            trainer.resume(resume_path, weights_only=args.resume_best)
            print(f"✅ Resumed from {resume_path}", flush=True)
        elif resume_path and not args.fresh and not args.auto_resume:
            raise FileNotFoundError(f"Checkpoint not found for resume: {resume_path}")
        elif args.fresh:
            print("🆕 Fresh start", flush=True)
        print(f"🚀 Training from step {trainer.engine.step} to {config['total_steps']}", flush=True)
        return trainer.fit()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        if dist_ctx and dist_ctx.is_distributed:
            cleanup_distributed()
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    main()
