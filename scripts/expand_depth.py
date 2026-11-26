"""Reusable depth-upscaling / best-layer looping expander for Triune checkpoints.

Expands any TriuneTransformer checkpoint (e.g., Junkie-3B 16L -> Junkie-7B-DUS 36L,
or recursively Junkie-7B-DUS 36L -> 52L+) while strictly preserving Triune structural
invariants:
  - Prefix + Reflex exit layers (0 .. reflex_exit_layer) stay dense & bit-exact.
  - Only MoE layers (reflex_exit_layer + 1 .. num_layers - 1) are looped/expanded.
  - Limbic exit head/norm is transplanted from the source Limbic layer to the
    target Limbic layer, and stripped from intermediate cloned blocks.
  - Residual output projections on newly inserted duplicate blocks can be damped
    by ``--smooth-scale`` (default 0.25) to avoid initial perplexity spikes during
    expansion healing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lucid_machines.configs.config import build_config
from lucid_machines.trainer.checkpoint import (
    _atomic_save_checkpoint,
    compute_checkpoint_fingerprint,
    extract_canonical_semantics,
)

_LAYER_PREFIX_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_EXPERT_WEIGHT_RE = re.compile(r"^ffn\.experts\.(\d+)\.(0|2)\.weight$")
_RESIDUAL_OUT_SUFFIX_RE = re.compile(
    r"^(?:attn\.gla\.out_proj\.(?:weight|bias)|attn\.out_proj\.(?:weight|bias)|ffn\.(?:experts\.\d+|shared)\.2\.(?:weight|bias))$"
)
_ARCH_KEYS = (
    "vocab_size",
    "hidden_dim",
    "num_layers",
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
)


def _load_target_json(target_spec: str) -> dict[str, Any]:
    cand = Path(target_spec)
    if cand.is_file():
        return json.loads(cand.read_text(encoding="utf-8-sig"))
    repo_root = Path(__file__).resolve().parents[1]
    stem = target_spec if target_spec.endswith(".json") else f"{target_spec}.json"
    cfg_path = repo_root / "models" / "configs" / stem
    if cfg_path.is_file():
        return json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    raise FileNotFoundError(f"Target config JSON not found: {target_spec}")


def _parse_loop_range(raw: str | None, valid_indices: Sequence[int]) -> list[int] | None:
    if not raw:
        return None
    parts = raw.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"--loop-range must be START:END (inclusive), got {raw!r}")
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise ValueError(f"Invalid --loop-range {raw!r}: start > end")
    valid_set = set(valid_indices)
    chosen = [idx for idx in range(start, end + 1) if idx in valid_set]
    if not chosen:
        raise ValueError(f"--loop-range {raw!r} contains no valid MoE layer indices from {list(valid_indices)}")
    return chosen


def score_moe_layers(state_dict: Mapping[str, torch.Tensor], moe_indices: Sequence[int]) -> dict[int, float]:
    """Score MoE layers by inter-expert specialization and projection signal energy."""
    scores: dict[int, float] = {}
    for idx in moe_indices:
        prefix = f"layers.{idx}."
        w_in: list[torch.Tensor] = []
        w_out: list[torch.Tensor] = []
        attn_rms = 0.0
        for key, tensor in state_dict.items():
            if not key.startswith(prefix) or not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                continue
            rel = key[len(prefix) :]
            m = _EXPERT_WEIGHT_RE.match(rel)
            if m:
                t_f = tensor.detach().float()
                if m.group(2) == "0":
                    w_in.append(t_f)
                else:
                    w_out.append(t_f)
            elif rel.startswith("attn.") and rel.endswith(".weight"):
                attn_rms += float(tensor.detach().float().pow(2).mean().sqrt().item())

        dispersion = 0.0
        for group in (w_in, w_out):
            if len(group) >= 2:
                stacked = torch.stack(group, dim=0)
                dispersion += float(stacked.std(dim=0, unbiased=False).mean().item())
            elif len(group) == 1:
                dispersion += float(group[0].std(unbiased=False).item())
        scores[int(idx)] = dispersion + (0.1 * attn_rms)
    return scores


def _tile_contiguous(span: Sequence[int], count: int) -> list[int]:
    if count <= 0:
        return []
    out: list[int] = []
    span_list = list(span)
    while len(out) < count:
        take = min(len(span_list), count - len(out))
        out.extend(span_list[:take])
    return out


def compute_layer_map(
    *,
    src_layers: int,
    target_layers: int,
    reflex_exit_layer: int,
    strategy: str = "best_loop",
    state_dict: Mapping[str, torch.Tensor] | None = None,
    loop_range: str | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Build deterministic src_layer -> dst_layer mapping preserving dense prefix + Reflex."""
    if target_layers <= src_layers:
        raise ValueError(f"target_layers ({target_layers}) must be > src_layers ({src_layers})")
    if reflex_exit_layer + 1 >= src_layers:
        raise ValueError(
            f"Source checkpoint has no MoE layers after reflex_exit_layer={reflex_exit_layer} (src_layers={src_layers})"
        )

    prefix = list(range(0, reflex_exit_layer + 1))
    moe_indices = list(range(reflex_exit_layer + 1, src_layers))
    needed_moe = target_layers - len(prefix)

    explicit_span = _parse_loop_range(loop_range, moe_indices)
    layer_scores = score_moe_layers(state_dict or {}, moe_indices) if state_dict is not None else {}

    if explicit_span is not None:
        if len(moe_indices) >= 2 and needed_moe >= 2:
            head_moe = [moe_indices[0]]
            tail_moe = [moe_indices[-1]]
            middle_needed = needed_moe - 2
            middle_seq = _tile_contiguous(explicit_span, middle_needed)
            moe_map = head_moe + middle_seq + tail_moe
        else:
            moe_map = _tile_contiguous(explicit_span, needed_moe)
        meta: dict[str, Any] = {"loop_span": explicit_span, "layer_scores": layer_scores}
        return prefix + moe_map, meta

    if len(moe_indices) == 1:
        moe_map = [moe_indices[0]] * needed_moe
        return prefix + moe_map, {"loop_span": list(moe_indices), "layer_scores": layer_scores}

    head_moe = [moe_indices[0]]
    tail_moe = [moe_indices[-1]]
    core = moe_indices[1:-1] if len(moe_indices) >= 3 else moe_indices
    middle_needed = needed_moe - 2

    if strategy == "solar_loop":
        # Preserve original interior pass once, then extend from core centered/full passes cleanly.
        if middle_needed <= len(core):
            chosen_span = list(core[:middle_needed])
        else:
            chosen_span = list(core)
        moe_map = head_moe + _tile_contiguous(chosen_span, middle_needed) + tail_moe
        return prefix + moe_map, {"loop_span": chosen_span, "layer_scores": layer_scores}

    if strategy == "best_loop":
        # Choose top-scoring contiguous core window (at least half of core when core >= 2) to loop.
        win_len = max(1, min(len(core), max((len(core) + 1) // 2, len(core) - 2)))
        best_start = 0
        best_score = float("-inf")
        for start in range(0, len(core) - win_len + 1):
            w = core[start : start + win_len]
            s = sum(layer_scores.get(i, 1.0) for i in w)
            if s > best_score:
                best_score = s
                best_start = start
        best_span = list(core[best_start : best_start + win_len])
        # Emit full original core once when room permits, then repeat highest-scoring span.
        if middle_needed >= len(core):
            extra_needed = middle_needed - len(core)
            middle_seq = list(core) + _tile_contiguous(best_span, extra_needed)
        else:
            middle_seq = list(best_span[:middle_needed])
        moe_map = head_moe + middle_seq + tail_moe
        return prefix + moe_map, {"loop_span": best_span, "layer_scores": layer_scores}

    raise ValueError(f"Unsupported expansion strategy {strategy!r}; expected 'solar_loop' or 'best_loop'")


def expand_state_dict_and_config(
    *,
    src_state: Mapping[str, torch.Tensor],
    src_config: Mapping[str, Any],
    target_layers: int | None = None,
    target_config: Mapping[str, Any] | None = None,
    limbic_exit_layer: int | None = None,
    strategy: str = "best_loop",
    loop_range: str | None = None,
    smooth_scale: float = 0.25,
    use_fp8: bool | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Expand a Triune model state_dict and config cleanly into a deeper Triune model."""
    merged_cfg: dict[str, Any] = dict(src_config)
    if target_config:
        for k in _ARCH_KEYS:
            if k in target_config and target_config[k] is not None:
                merged_cfg[k] = target_config[k]
        if target_config.get("model_name"):
            merged_cfg["model_name"] = target_config["model_name"]

    src_layers = int(src_config.get("num_layers", 0))
    if src_layers <= 0:
        inferred = [int(m.group(1)) for k in src_state if (m := _LAYER_PREFIX_RE.match(k))]
        if not inferred:
            raise ValueError("Cannot infer num_layers from source state_dict/config")
        src_layers = max(inferred) + 1

    reflex_exit = int(merged_cfg.get("reflex_exit_layer", src_config.get("reflex_exit_layer", 2)))
    src_limbic_exit = int(src_config.get("limbic_exit_layer", max(reflex_exit + 1, src_layers // 2)))

    if target_layers is not None:
        final_target_layers = int(target_layers)
    elif target_config and target_config.get("num_layers") is not None:
        final_target_layers = int(target_config["num_layers"])
    else:
        raise ValueError("Must provide --target-layers or --target-config with num_layers")

    if limbic_exit_layer is not None:
        final_limbic_exit = int(limbic_exit_layer)
    elif target_config and target_config.get("limbic_exit_layer") is not None:
        final_limbic_exit = int(target_config["limbic_exit_layer"])
    else:
        final_limbic_exit = reflex_exit + max(2, (final_target_layers - reflex_exit) // 3)

    if not (reflex_exit < final_limbic_exit < final_target_layers):
        raise ValueError(
            f"Target hierarchy invalid: must satisfy reflex ({reflex_exit}) < limbic ({final_limbic_exit}) < num_layers ({final_target_layers})"
        )

    merged_cfg["num_layers"] = final_target_layers
    merged_cfg["reflex_exit_layer"] = reflex_exit
    merged_cfg["limbic_exit_layer"] = final_limbic_exit
    if use_fp8 is not None:
        merged_cfg["use_fp8"] = bool(use_fp8)
        if merged_cfg["use_fp8"]:
            merged_cfg["use_fp4"] = False

    new_cfg = build_config(merged_cfg)

    clean_src = {k.removeprefix("_orig_mod."): v for k, v in src_state.items()}
    layer_map, map_meta = compute_layer_map(
        src_layers=src_layers,
        target_layers=final_target_layers,
        reflex_exit_layer=reflex_exit,
        strategy=strategy,
        state_dict=clean_src,
        loop_range=loop_range,
    )

    new_state: dict[str, torch.Tensor] = {}
    for k, v in clean_src.items():
        if not _LAYER_PREFIX_RE.match(k):
            new_state[k] = v.detach().clone()

    per_src_keys: dict[int, list[tuple[str, torch.Tensor]]] = {i: [] for i in range(src_layers)}
    for k, v in clean_src.items():
        m = _LAYER_PREFIX_RE.match(k)
        if m:
            s_idx = int(m.group(1))
            if s_idx in per_src_keys:
                per_src_keys[s_idx].append((m.group(2), v))

    seen_sources: set[int] = set()
    for dst_idx, src_idx in enumerate(layer_map):
        is_duplicate = src_idx in seen_sources
        seen_sources.add(src_idx)
        for suffix, tensor in per_src_keys[src_idx]:
            if suffix.startswith("exit_norm.") or suffix.startswith("exit_head."):
                if dst_idx != reflex_exit:
                    continue
            out_t = tensor.detach().clone()
            if is_duplicate and dst_idx > reflex_exit and smooth_scale != 1.0 and _RESIDUAL_OUT_SUFFIX_RE.match(suffix):
                out_t = out_t * float(smooth_scale)
            new_state[f"layers.{dst_idx}.{suffix}"] = out_t

    limbic_pairs = [
        (suffix, tensor)
        for suffix, tensor in per_src_keys[src_limbic_exit]
        if suffix.startswith("exit_norm.") or suffix.startswith("exit_head.")
    ]
    if not limbic_pairs:
        raise ValueError(
            f"Source Limbic layer layers.{src_limbic_exit} has no exit_norm/exit_head weights to transplant"
        )
    for suffix, tensor in limbic_pairs:
        new_state[f"layers.{final_limbic_exit}.{suffix}"] = tensor.detach().clone()

    expansion_meta = {
        "source_layers": src_layers,
        "target_layers": final_target_layers,
        "reflex_exit_layer": reflex_exit,
        "source_limbic_exit_layer": src_limbic_exit,
        "target_limbic_exit_layer": final_limbic_exit,
        "strategy": strategy,
        "smooth_scale": float(smooth_scale),
        "layer_map": layer_map,
        **map_meta,
    }
    return new_state, new_cfg, expansion_meta


def expand_checkpoint_file(
    *,
    input_path: Path,
    output_path: Path,
    target_layers: int | None = None,
    target_config_spec: str | None = None,
    limbic_exit_layer: int | None = None,
    strategy: str = "best_loop",
    loop_range: str | None = None,
    smooth_scale: float = 0.25,
    use_fp8: bool | None = None,
) -> dict[str, Any]:
    """Load a source Triune checkpoint, depth-expand it, and atomically write Schema-v2 output."""
    ckpt = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, Mapping) or "model_state" not in ckpt:
        raise ValueError(f"Expected checkpoint dict containing 'model_state' in {input_path}")

    src_state = ckpt["model_state"]
    src_config = ckpt.get("config") or {}
    target_cfg_json = _load_target_json(target_config_spec) if target_config_spec else None

    new_state, new_cfg, exp_meta = expand_state_dict_and_config(
        src_state=src_state,
        src_config=src_config,
        target_layers=target_layers,
        target_config=target_cfg_json,
        limbic_exit_layer=limbic_exit_layer,
        strategy=strategy,
        loop_range=loop_range,
        smooth_scale=smooth_scale,
        use_fp8=use_fp8,
    )
    exp_meta["source_checkpoint"] = str(input_path)
    exp_meta["source_step"] = ckpt.get("step", 0)

    payload = {
        "step": 0,
        "model_state": new_state,
        "optimizer_state": {},
        "loss": None,
        "best_eval_loss": None,
        "config": new_cfg,
        "schema_version": 2,
        "fingerprint": compute_checkpoint_fingerprint(new_cfg),
        "canonical_semantics": extract_canonical_semantics(new_cfg),
        "depth_usage_ema": ckpt.get("depth_usage_ema"),
        "wandb_run_id": None,
        "expansion_metadata": exp_meta,
    }
    saved = _atomic_save_checkpoint(payload, output_path)
    return {
        "output_path": str(saved),
        "source_layers": exp_meta["source_layers"],
        "target_layers": exp_meta["target_layers"],
        "strategy": strategy,
        "layer_map": exp_meta["layer_map"],
        "fingerprint": payload["fingerprint"],
        "param_count": sum(t.numel() for t in new_state.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand a Triune checkpoint via best-layer looping / Solar-DUS")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Source .pt checkpoint")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Destination expanded .pt checkpoint")
    parser.add_argument("--target-config", type=str, default=None, help="Target config JSON or preset stem")
    parser.add_argument("--target-layers", type=int, default=None, help="Target layer count (if no --target-config)")
    parser.add_argument("--limbic-exit-layer", type=int, default=None, help="Override target limbic_exit_layer")
    parser.add_argument("--strategy", choices=("best_loop", "solar_loop"), default="best_loop")
    parser.add_argument("--loop-range", type=str, default=None, help="Optional explicit inclusive START:END MoE range")
    parser.add_argument("--smooth-scale", type=float, default=0.25, help="Residual scale on duplicate blocks (1.0=off)")
    parser.add_argument("--use-fp8", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)

    summary = expand_checkpoint_file(
        input_path=args.input,
        output_path=args.output,
        target_layers=args.target_layers,
        target_config_spec=args.target_config,
        limbic_exit_layer=args.limbic_exit_layer,
        strategy=args.strategy,
        loop_range=args.loop_range,
        smooth_scale=args.smooth_scale,
        use_fp8=args.use_fp8,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
