"""Real telemetry, logprob/entropy inspection, mind-montage, and config helpers for Polysom Studio."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


_SAFE_CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ALLOWED_PREVIEW_EXTS = {".jsonl", ".json", ".txt", ".csv", ".tsv"}


def compute_token_logprobs_and_entropy(
    model: Any,
    tokenizer: Any,
    text: str,
    device: Any,
    *,
    top_k: int = 5,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """Compute real token-level logprobs, Shannon entropy (bits), and top-k candidates via model forward pass."""
    if not HAS_TORCH or torch is None or model is None or tokenizer is None:
        return {
            "available": False,
            "reason": "PyTorch model and tokenizer must be initialized to compute token logprobs.",
            "tokens": [],
        }
    clean_text = str(text or "")
    if not clean_text.strip():
        return {"available": False, "reason": "Input text is empty.", "tokens": []}

    encoded = tokenizer.encode(clean_text)
    ids = list(getattr(encoded, "ids", []) or [])
    if not ids:
        return {"available": False, "reason": "Tokenizer produced zero tokens.", "tokens": []}

    ids = ids[: max(2, min(int(max_tokens), 512))]
    k = max(1, min(int(top_k), 20))

    try:
        model_dev = next((p.device for p in model.parameters()), None)
        if model_dev is None:
            model_dev = next((b.device for b in model.buffers()), device)
    except Exception:
        model_dev = device

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
        with torch.inference_mode():
            inp = torch.tensor(ids, dtype=torch.long, device=model_dev).unsqueeze(0)
            out = model(inp)
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out

            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                return {
                    "available": False,
                    "reason": "Active model output does not expose 3D token logits.",
                    "tokens": [],
                }

            seq_len = int(inp.shape[1])
            tokens_out: List[Dict[str, Any]] = []
            # First token has no conditioning prefix logits; emit structural entry
            first_tok_text = tokenizer.decode([ids[0]], skip_special_tokens=False)
            tokens_out.append(
                {
                    "id": int(ids[0]),
                    "text": first_tok_text,
                    "logprob": None,
                    "prob": None,
                    "entropy": None,
                    "top_candidates": [],
                }
            )
            if seq_len >= 2:
                step_logits = logits[0, : seq_len - 1, :].float()
                log_probs = torch.log_softmax(step_logits, dim=-1)
                probs = log_probs.exp()
                # H = -sum(p * log2(p))
                entropy_bits = -(probs * (log_probs / math.log(2.0))).sum(dim=-1)
                top_vals, top_idxs = torch.topk(log_probs, k=min(k, log_probs.shape[-1]), dim=-1)

                for pos in range(seq_len - 1):
                    target_id = int(ids[pos + 1])
                    tok_str = tokenizer.decode([target_id], skip_special_tokens=False)
                    if target_id < log_probs.shape[-1]:
                        tok_lp = float(log_probs[pos, target_id].item())
                        tok_p = float(math.exp(tok_lp))
                    else:
                        tok_lp = -20.0
                        tok_p = 0.0
                    ent = float(entropy_bits[pos].item())
                    cands: List[Dict[str, Any]] = []
                    for rank_i in range(top_vals.shape[-1]):
                        cid = int(top_idxs[pos, rank_i].item())
                        clp = float(top_vals[pos, rank_i].item())
                        cands.append(
                            {
                                "id": cid,
                                "text": tokenizer.decode([cid], skip_special_tokens=False),
                                "logprob": round(clp, 4),
                                "prob": round(math.exp(clp) * 100.0, 2),
                            }
                        )
                    tokens_out.append(
                        {
                            "id": target_id,
                            "text": tok_str,
                            "logprob": round(tok_lp, 4),
                            "prob": round(tok_p * 100.0, 2),
                            "entropy": round(ent, 3),
                            "top_candidates": cands,
                        }
                    )
            return {
                "available": True,
                "token_count": len(tokens_out),
                "tokens": tokens_out,
            }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Logprob inspection failed: {exc}",
            "tokens": [],
        }
    finally:
        if was_training:
            try:
                model.train()
            except Exception:
                pass


def collect_gpu_telemetry(pytorch_state: Any, vram_profiler: Any) -> Dict[str, Any]:
    """Return real live GPU/VRAM and latest inference timing telemetry."""
    has_cuda = bool(HAS_TORCH and torch is not None and torch.cuda.is_available())
    vram = vram_profiler.get_vram_stats()
    vram_out = {
        "allocated_gb": float(vram.get("allocated_gb", 0.0)),
        "reserved_gb": float(vram.get("reserved_gb", 0.0)),
        "total_gb": float(vram.get("total_gb", 0.0)),
        "oom_risk": bool(vram_profiler.check_oom_risk(0.90)),
    }
    last_inf = getattr(pytorch_state, "last_inference_telemetry", None)
    return {
        "status": "ok",
        "cuda_available": has_cuda,
        "device_name": getattr(pytorch_state, "device_name", "CPU"),
        "vram": vram_out,
        "active_model": getattr(pytorch_state, "active_model_id", None),
        "weights_loaded": bool(getattr(pytorch_state, "weights_loaded", False)),
        "active_checkpoint": getattr(pytorch_state, "active_checkpoint", None)
        if getattr(pytorch_state, "weights_loaded", False)
        else None,
        "is_training": bool(getattr(pytorch_state, "is_training", False)),
        "step": int(getattr(pytorch_state, "step", 0) or 0),
        "training_throughput": float(pytorch_state._get_current_throughput()),
        "last_inference": dict(last_inf) if isinstance(last_inf, dict) else None,
    }


def collect_mind_telemetry(pytorch_state: Any) -> Dict[str, Any]:
    """Collect real Polysom 'Look Into The Mind' telemetry from the active model, optimizer, and NeuronScope."""
    model = getattr(pytorch_state, "model", None)
    model_loaded = model is not None
    weights_loaded = bool(getattr(pytorch_state, "weights_loaded", False))
    active_model_id = str(getattr(pytorch_state, "active_model_id", "triune-small") or "triune-small")
    arch_spec = dict(getattr(pytorch_state, "architecture_spec", {}) or {})

    exit_usage = pytorch_state._get_current_exit_usage()
    target_dist = pytorch_state._get_target_exit_distribution()
    depth_mode = str(getattr(pytorch_state, "depth_mode", "dynamic") or "dynamic")

    scope = getattr(pytorch_state, "neuron_scope", None)
    scope_snap = scope.get_snapshot() if scope is not None else {}

    active_tier = int(scope_snap.get("active_tier", 0) or 0)
    active_tier_name = str(scope_snap.get("active_tier_name", "Reflex"))
    router_history = list(scope_snap.get("router_history", []) or [])
    layer_bins = dict(scope_snap.get("layer_bins", {}) or {})

    # Inspect real MoE layers + CentroidSteer metrics if model is instantiated
    moe_layers: List[Dict[str, Any]] = []
    agg_requested: List[int] = []
    agg_accepted: List[int] = []
    agg_dropped: List[int] = []
    centroid_energies: List[float] = []
    centroid_drifts: List[float] = []

    if HAS_TORCH and torch is not None and model is not None:
        for mod_name, mod in model.named_modules():
            if not (hasattr(mod, "last_centroids") and hasattr(mod, "num_experts")):
                continue
            num_exp = int(getattr(mod, "num_experts", 0) or 0)
            req_c = getattr(mod, "last_requested_counts", None)
            acc_c = getattr(mod, "last_accepted_counts", None)
            drp_c = getattr(mod, "last_dropped_counts", None)
            req_list = [int(x) for x in req_c.detach().cpu().tolist()] if isinstance(req_c, torch.Tensor) else []
            acc_list = [int(x) for x in acc_c.detach().cpu().tolist()] if isinstance(acc_c, torch.Tensor) else []
            drp_list = [int(x) for x in drp_c.detach().cpu().tolist()] if isinstance(drp_c, torch.Tensor) else []

            if req_list:
                if not agg_requested:
                    agg_requested = [0] * len(req_list)
                for idx_e, val in enumerate(req_list):
                    if idx_e < len(agg_requested):
                        agg_requested[idx_e] += val
            if acc_list:
                if not agg_accepted:
                    agg_accepted = [0] * len(acc_list)
                for idx_e, val in enumerate(acc_list):
                    if idx_e < len(agg_accepted):
                        agg_accepted[idx_e] += val
            if drp_list:
                if not agg_dropped:
                    agg_dropped = [0] * len(drp_list)
                for idx_e, val in enumerate(drp_list):
                    if idx_e < len(agg_dropped):
                        agg_dropped[idx_e] += val

            c_tensor = getattr(mod, "last_centroids", None)
            c_contr = getattr(mod, "last_contrastive_centroids", None)
            c_snap = getattr(mod, "_recorded_centroids_snapshot", None)

            c_norm = None
            energy_val = None
            drift_val = None
            if isinstance(c_tensor, torch.Tensor) and c_tensor.numel() > 0:
                c_f = c_tensor.detach().float()
                c_norm = round(float(c_f.norm().item()), 5)
                if c_norm > 1e-8 and isinstance(c_contr, torch.Tensor) and c_contr.shape == c_f.shape:
                    contr_f = c_contr.detach().float()
                    # Alignment energy: normalized contrastive separation energy across experts
                    energy_val = round(float((contr_f.pow(2).sum() / max(1, num_exp)).item()), 5)
                    centroid_energies.append(energy_val)
                if c_norm > 1e-8 and isinstance(c_snap, torch.Tensor) and c_snap.shape == c_f.shape:
                    drift_val = round(float((c_f - c_snap.detach().float()).norm().item()), 5)
                    centroid_drifts.append(drift_val)

            moe_layers.append(
                {
                    "layer_name": mod_name or "moe",
                    "num_experts": num_exp,
                    "requested_counts": req_list,
                    "accepted_counts": acc_list,
                    "dropped_counts": drp_list,
                    "centroid_norm": c_norm,
                    "alignment_energy": energy_val,
                    "centroid_drift": drift_val,
                }
            )

    opt = getattr(pytorch_state, "optimizer", None)
    opt_telemetry = opt.get_telemetry() if (opt is not None and hasattr(opt, "get_telemetry")) else None
    centroid_active = bool(
        (opt_telemetry and opt_telemetry.get("tier2_centroid", {}).get("param_count", 0) > 0)
        or len(centroid_energies) > 0
    )

    total_acc = sum(agg_accepted)
    utilization_pct = [round((c / total_acc) * 100.0, 1) for c in agg_accepted] if total_acc > 0 else []

    if not model_loaded:
        mind_status = "unloaded"
    elif (
        int(getattr(pytorch_state, "step", 0) or 0) > 0
        or bool(getattr(pytorch_state, "is_training", False))
        or total_acc > 0
        or bool(layer_bins)
    ):
        mind_status = "active"
    else:
        mind_status = "idle"

    return {
        "status": mind_status,
        "model_loaded": model_loaded,
        "weights_loaded": weights_loaded,
        "active_model_id": active_model_id,
        "active_checkpoint": getattr(pytorch_state, "active_checkpoint", None) if weights_loaded else None,
        "step": int(getattr(pytorch_state, "step", 0) or 0),
        "is_training": bool(getattr(pytorch_state, "is_training", False)),
        "architecture": arch_spec,
        "triune_tiers": {
            "depth_mode": depth_mode,
            "active_tier": active_tier,
            "active_tier_name": active_tier_name,
            "exit_usage": exit_usage,
            "target_exit_distribution": target_dist,
            "router_history": router_history,
        },
        "centroid_steer": {
            "active": centroid_active,
            "use_triangular_core": bool(getattr(pytorch_state, "use_triangular_core", True)),
            "subspace_rank": int(getattr(pytorch_state, "centroid_rank", 64) or 64),
            "steer_scale": float(getattr(pytorch_state, "steer_scale", 0.2) or 0.2),
            "update_gap": int(getattr(pytorch_state, "centroid_update_gap", 100) or 100),
            "alignment_energy": round(sum(centroid_energies) / len(centroid_energies), 5)
            if centroid_energies
            else None,
            "centroid_drift": round(sum(centroid_drifts) / len(centroid_drifts), 5) if centroid_drifts else None,
            "optimizer_tier2": opt_telemetry.get("tier2_centroid") if isinstance(opt_telemetry, dict) else None,
        },
        "moe_routing": {
            "has_routing_data": total_acc > 0 or bool(agg_requested),
            "num_moe_layers": len(moe_layers),
            "aggregated_requested": agg_requested,
            "aggregated_accepted": agg_accepted,
            "aggregated_dropped": agg_dropped,
            "utilization_pct": utilization_pct,
            "layers": moe_layers,
        },
        "neuron_scope": {
            "has_activations": bool(layer_bins),
            "layer_bins": layer_bins,
        },
    }


def resolve_confined_config_path(
    config_name: str,
    allowed_roots: List[Path],
    has_traversal_fn: Callable[[str], bool],
    is_confined_fn: Callable[[Path, List[Path]], bool],
) -> Optional[Path]:
    """Resolve a JSON config filename strictly within models/configs/."""
    raw = str(config_name or "").strip()
    if not raw or has_traversal_fn(raw) or "/" in raw or "\\" in raw:
        return None
    stem_or_file = raw if raw.endswith(".json") else f"{raw}.json"
    if not _SAFE_CONFIG_NAME_RE.fullmatch(stem_or_file):
        return None
    first_confined: Optional[Path] = None
    for root in allowed_roots:
        candidate = (root / stem_or_file).resolve(strict=False)
        if is_confined_fn(candidate, allowed_roots):
            if candidate.is_file():
                return candidate
            if first_confined is None:
                first_confined = candidate
    return first_confined


def list_model_configs(allowed_roots: List[Path]) -> List[Dict[str, Any]]:
    """List valid JSON architecture configs from models/configs/ (junkie*.json first, then triune-*.json)."""
    seen: set[str] = set()
    items: List[Dict[str, Any]] = []
    for root in allowed_roots:
        if not root.is_dir():
            continue
        for f in sorted(root.glob("*.json")):
            if f.name in seen:
                continue
            if not f.name.startswith(("junkie", "triune-")):
                continue
            seen.add(f.name)
            try:
                data = json.loads(f.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            items.append(
                {
                    "filename": f.name,
                    "id": f.stem,
                    "name": data.get("name", f.stem),
                    "architecture": data.get("architecture", "TriuneTransformer"),
                    "status": data.get("status", "architecture_preset"),
                    "description": data.get("description", ""),
                    "num_layers": data.get("num_layers"),
                    "hidden_dim": data.get("hidden_dim"),
                    "num_heads": data.get("num_heads"),
                    "num_experts": data.get("num_experts"),
                    "vocab_size": data.get("vocab_size"),
                    "is_flagship_junkie": f.name == "junkie.json",
                }
            )
    items.sort(
        key=lambda x: (
            0 if x["is_flagship_junkie"] else (1 if x["filename"].startswith("junkie") else 2),
            x["filename"],
        )
    )
    return items


def preview_dataset_file(
    raw_path: str,
    allowed_roots: List[Path],
    has_traversal_fn: Callable[[str], bool],
    is_confined_fn: Callable[[Path, List[Path]], bool],
    *,
    limit: int = 10,
) -> tuple[int, Dict[str, Any]]:
    """Safely read up to `limit` sample records from a confined local dataset file."""
    clean = str(raw_path or "").strip()
    if not clean or has_traversal_fn(clean) or not is_confined_fn(Path(clean), allowed_roots):
        return 403, {
            "status": "error",
            "detail": "Security restriction: dataset preview path must reside inside data/ or datasets/.",
        }
    target = Path(clean).resolve(strict=False)
    if target.suffix.lower() not in _ALLOWED_PREVIEW_EXTS:
        return 403, {
            "status": "error",
            "detail": f"Security restriction: disallowed dataset preview extension {target.suffix}.",
        }
    if not target.is_file():
        return 404, {"status": "error", "detail": f"Dataset file not found: {clean}"}

    max_rows = max(1, min(int(limit), 50))
    records: List[Dict[str, Any]] = []
    ext = target.suffix.lower()
    total_bytes = target.stat().st_size

    try:
        if ext == ".json":
            raw_text = target.read_text(encoding="utf-8-sig", errors="replace")[:262144]
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                for idx, row in enumerate(parsed[:max_rows]):
                    records.append(
                        {"index": idx, "preview": row if isinstance(row, dict) else {"value": str(row)[:500]}}
                    )
            elif isinstance(parsed, dict):
                records.append({"index": 0, "preview": parsed})
        else:
            with target.open("r", encoding="utf-8-sig", errors="replace") as fh:
                for idx, line in enumerate(fh):
                    if idx >= max_rows:
                        break
                    s = line.strip()
                    if not s:
                        continue
                    if ext == ".jsonl":
                        try:
                            obj = json.loads(s[:16384])
                            records.append(
                                {"index": idx, "preview": obj if isinstance(obj, dict) else {"text": str(obj)[:500]}}
                            )
                        except Exception:
                            records.append({"index": idx, "preview": {"raw": s[:500]}})
                    else:
                        records.append({"index": idx, "preview": {"text": s[:500]}})
    except Exception as exc:
        return 400, {"status": "error", "detail": f"Could not preview dataset: {exc}"}

    return 200, {
        "status": "success",
        "path": str(target).replace("\\", "/"),
        "filename": target.name,
        "total_bytes": total_bytes,
        "record_count": len(records),
        "records": records,
    }
