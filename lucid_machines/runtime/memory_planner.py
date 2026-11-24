"""VRAM Memory Planner & Automated Hardware Budgeting Engine.

Calculates memory footprints across parameters, optimizer states, activations,
and gradient accumulation buffers to fit models cleanly on consumer GPUs (e.g. RTX 5070 8GB/12GB VRAM).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class MemoryEstimate:
    total_params: int
    param_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    gradient_memory_gb: float
    total_vram_gb: float
    recommended_batch_size: int
    recommended_grad_accum: int
    recommended_checkpointing: bool
    recommended_precision: str


def _resolve_local_triune_spec(model_id_or_dir: str) -> dict | None:
    """Resolve a Triune/Junkie preset name or local models/configs/*.json before hitting Hugging Face."""
    raw = str(model_id_or_dir or "").strip()
    if not raw:
        return None

    cand = Path(raw)
    if cand.is_file() and cand.suffix.lower() == ".json":
        try:
            data = json.loads(cand.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict) and ("hidden_dim" in data or data.get("model_type") == "triune"):
                return {**data, "_source_name": cand.stem}
        except Exception:
            pass

    repo_root = Path(__file__).resolve().parents[2]
    configs_dir = repo_root / "models" / "configs"
    for stem in (raw, raw.lower()):
        cfg_file = configs_dir / (stem if stem.endswith(".json") else f"{stem}.json")
        if cfg_file.is_file():
            try:
                data = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    return {**data, "_source_name": cfg_file.stem}
            except Exception:
                pass

    try:
        from lucid_machines.model.base import _LEGACY_PRESET_ALIASES, CANONICAL_MODEL_SPECS, get_model_spec

        key = raw.lower()
        if key in CANONICAL_MODEL_SPECS or key in _LEGACY_PRESET_ALIASES:
            spec = get_model_spec(key)
            return {**spec.to_dict(), "_source_name": spec.model_id}
    except Exception:
        pass
    return None


class MemoryPlanner:
    """Hardware-aware VRAM Memory Planner.

    Accurately models the Triune architecture & CentroidSteer + Muon + Base 3-tier optimizer:
    - Embedding + 2 early-exit heads: 3 * vocab_size * hidden_dim + 2 * (vocab_size + hidden_dim) + hidden_dim
    - Per-layer attention: 4 projections (Q,K,V,G) + output = 5 * hidden_dim^2
    - Per-layer MoE FFN: (num_experts + shared_expert) * 2 * hidden_dim * (hidden_dim * expert_multiplier)
    - Per-layer dense FFN (prefix layers): 2 * hidden_dim * (hidden_dim * 4)
    - Depth router: GumbelSoftmaxRouter(hidden_dim -> 256 -> 3)
    - Optimizer: CentroidSteer low-rank expert subspace (~0.20-0.37 B/p) + Muon (4 B/p) + Base (8 B/p),
      with optional legacy GaLore fallback.
    """

    @staticmethod
    def estimate_vram(
        config: dict,
        target_vram_gb: float = 8.0,
        available_vram_gb: float | None = None,
    ) -> MemoryEstimate:
        """Estimate VRAM allocation breakdown and recommend optimal training hyperparameters."""
        if available_vram_gb is None and torch.cuda.is_available():
            available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_vram_gb = available_vram_gb or target_vram_gb

        vocab_size = int(config.get("vocab_size", 32000))
        hidden_dim = int(config.get("hidden_dim", 1536))
        num_layers = max(4, int(config.get("num_layers", 24)))
        num_experts = max(1, int(config.get("num_experts", 8)))
        seq_len = int(config.get("seq_len", 256))
        use_fp4 = bool(config.get("use_fp4", False))
        use_fp8 = bool(config.get("use_fp8", False))
        expert_multiplier = max(1, int(config.get("expert_hidden_multiplier", 6)))
        has_shared = bool(config.get("shared_expert", True)) and int(config.get("experts_shared", 1)) >= 1
        shared_cnt = 1 if has_shared else 0
        reflex_exit_layer = max(2, min(int(config.get("reflex_exit_layer", 6)), num_layers - 2))
        galore_rank = int(config.get("galore_rank", 64))
        num_steer_dirs = max(1, int(config.get("num_steer_dirs", 4)))
        use_muon = bool(config.get("use_muon", True))
        optimizer_mode = str(config.get("optimizer_mode", "centroid_steer")).lower()

        # ── Exact Parameter Count (aligned with ModelSpec & TriuneTransformer) ──
        vocab_and_exits = (3 * vocab_size * hidden_dim) + (2 * (vocab_size + hidden_dim)) + hidden_dim
        router_params = (hidden_dim * 256) + 256 + (256 * 3) + 3
        dense_layers = reflex_exit_layer + 1
        moe_layers = max(0, num_layers - dense_layers)

        per_dense = (13 * hidden_dim * hidden_dim) + (7 * hidden_dim)
        expert_ffn_dim = hidden_dim * expert_multiplier
        per_moe_expert_params = (
            moe_layers
            * (num_experts + shared_cnt)
            * (2 * hidden_dim * expert_ffn_dim + (expert_multiplier + 1) * hidden_dim)
        )
        per_moe_non_expert = (5 * hidden_dim * hidden_dim) + ((num_experts + 2) * hidden_dim) + num_experts
        per_moe_total = per_moe_non_expert + (
            (num_experts + shared_cnt) * (2 * hidden_dim * expert_ffn_dim + (expert_multiplier + 1) * hidden_dim)
        )

        total_params = vocab_and_exits + router_params + (dense_layers * per_dense) + (moe_layers * per_moe_total)

        # ── Bytes Per Parameter ──────────────────────────────
        if use_fp4:
            bytes_per_param = 0.5  # 4-bit (0.5 bytes per parameter)
        elif use_fp8:
            bytes_per_param = 1.0  # 8-bit (1.0 byte per parameter)
        else:
            bytes_per_param = 2.0  # 16-bit BF16 (2.0 bytes per parameter)

        param_memory_gb = (total_params * bytes_per_param) / (1024**3)

        # ── Optimizer Memory (CentroidSteer + Muon + Base vs GaLore fallback) ──
        expert_2d_params = moe_layers * (num_experts + shared_cnt) * (2 * hidden_dim * expert_ffn_dim)
        non_expert_2d_params = dense_layers * (
            5 * hidden_dim * hidden_dim + 8 * hidden_dim * hidden_dim
        ) + moe_layers * (5 * hidden_dim * hidden_dim)
        base_1d_and_embed_params = max(0, total_params - expert_2d_params - non_expert_2d_params)

        if optimizer_mode == "galore" or not use_muon:
            # Legacy GaLore low-rank projection fallback
            rank_ratio = min(1.0, galore_rank / max(1, hidden_dim))
            optimizer_bytes_per_param = 10.0 * rank_ratio
            norm_params = num_layers * 2 * hidden_dim
            optimizer_memory_gb = ((total_params - norm_params) * optimizer_bytes_per_param + norm_params * 8.0) / (
                1024**3
            )
        else:
            # CentroidSteer + Muon + Base 3-tier footprint:
            # - Expert matrices: low-rank subspace (rank) + CentroidSteer directional buffers (~0.199-0.368 B/p)
            # - Non-expert 2D hidden matrices: Muon BF16/FP32 single momentum buffer (~4.0 B/p)
            # - Embeddings / exit heads / norms / routers: Base AdamW (8.0 B/p)
            rank_ratio = min(1.0, galore_rank / max(1, hidden_dim))
            # Empirical rank=64/d=1024 (~0.199 B/p) to rank=256/d=2048 (~0.368 B/p)
            expert_bpp = min(2.0, max(0.18, (2.6 * rank_ratio) + (0.008 * num_steer_dirs)))
            muon_bpp = 4.0
            base_bpp = 8.0
            opt_bytes = (
                (expert_2d_params * expert_bpp)
                + (non_expert_2d_params * muon_bpp)
                + (base_1d_and_embed_params * base_bpp)
            )
            optimizer_memory_gb = opt_bytes / (1024**3)

        # ── Activation Memory (with Gradient Checkpointing) ───
        checkpointing = True
        activation_per_sample_bytes = num_layers * seq_len * hidden_dim * 2

        # ── Gradient Memory ──────────────────────────────────
        gradient_memory_gb = (total_params * bytes_per_param * 0.5) / (1024**3)

        # ── Target Safety Threshold ──────────────────────────
        cuda_context_gb = 0.5  # PyTorch CUDA context + cuBLAS workspace
        base_overhead_gb = param_memory_gb + optimizer_memory_gb + gradient_memory_gb + cuda_context_gb
        target_limit_gb = available_vram_gb * 0.90  # 90% safety margin

        remaining_vram_gb = max(0.1, target_limit_gb - base_overhead_gb)

        # ── Optimal Batch Size & Grad Accumulation ───────────
        activation_per_sample_gb = activation_per_sample_bytes / (1024**3)

        recommended_batch_size = 1
        for bs in [4, 2, 1]:
            if bs * activation_per_sample_gb <= remaining_vram_gb:
                recommended_batch_size = bs
                break

        effective_target = 8
        recommended_grad_accum = max(1, effective_target // recommended_batch_size)

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                recommended_precision = "fp8"
            else:
                recommended_precision = "bf16"
        else:
            recommended_precision = "bf16"

        activation_memory_gb = recommended_batch_size * activation_per_sample_gb
        total_vram_gb = (
            param_memory_gb + optimizer_memory_gb + activation_memory_gb + gradient_memory_gb + cuda_context_gb
        )

        return MemoryEstimate(
            total_params=total_params,
            param_memory_gb=round(param_memory_gb, 3),
            optimizer_memory_gb=round(optimizer_memory_gb, 3),
            activation_memory_gb=round(activation_memory_gb, 3),
            gradient_memory_gb=round(gradient_memory_gb, 3),
            total_vram_gb=round(total_vram_gb, 3),
            recommended_batch_size=recommended_batch_size,
            recommended_grad_accum=recommended_grad_accum,
            recommended_checkpointing=checkpointing,
            recommended_precision=recommended_precision,
        )

    @staticmethod
    def estimate_hf_model(
        model_id_or_dir: str,
        *,
        available_vram_gb: float | None = None,
        available_ram_gb: float | None = None,
        local_files_only: bool = False,
    ) -> dict:
        """Inspect a local Triune/Junkie preset or HF AutoConfig and recommend execution mode."""
        if available_vram_gb is None and torch.cuda.is_available():
            available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_gb = float(available_vram_gb or 8.0)

        if available_ram_gb is None:
            try:
                import psutil

                available_ram_gb = psutil.virtual_memory().total / (1024**3)
            except Exception:
                available_ram_gb = 16.0
        ram_gb = float(available_ram_gb)

        local_spec = _resolve_local_triune_spec(model_id_or_dir)
        if local_spec is not None:
            from lucid_machines.model.base import get_model_spec

            spec = get_model_spec(
                local_spec.get("_source_name") or local_spec.get("model_id") or str(model_id_or_dir),
                **{
                    k: local_spec[k]
                    for k in (
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
                    )
                    if k in local_spec and local_spec[k] is not None
                },
            )
            total_params, active_params = spec.compute_exact_params()
            layers = int(spec.num_layers)
            hidden = int(spec.hidden_dim)
            vocab = int(spec.vocab_size)
            tied = True
            root_p = (3 * vocab * hidden) + (2 * (vocab + hidden)) + hidden
            layer_p = max(0, total_params - root_p)
            per_layer_p = layer_p / max(1, layers)

            bf16_gb = (total_params * 2.0) / (1024**3)
            nvfp4_gb = (root_p * 2.0 + layer_p * 0.625) / (1024**3)
            nvfp4_quant_head_gb = (vocab * hidden * 2.0 + layer_p * 0.625) / (1024**3)
            per_layer_nvfp4_mb = (per_layer_p * 0.625) / (1024**2)

            if bf16_gb + 1.5 <= vram_gb:
                mode = "resident_bf16"
            elif nvfp4_gb + 1.2 <= vram_gb:
                mode = "resident_nvfp4"
            elif bf16_gb <= ram_gb * 0.65:
                mode = "ram_layer_streaming"
            else:
                mode = "disk_nvfp4_streaming"

            train_est = MemoryPlanner.estimate_vram(
                {**spec.to_dict(), "galore_rank": 256 if hidden >= 2048 else 64},
                target_vram_gb=vram_gb,
                available_vram_gb=vram_gb,
            )
            train_est_fp8 = MemoryPlanner.estimate_vram(
                {**spec.to_dict(), "galore_rank": 256 if hidden >= 2048 else 64, "use_fp8": True},
                target_vram_gb=vram_gb,
                available_vram_gb=vram_gb,
            )
            fp8_train_static_gb = round(train_est_fp8.total_vram_gb + (total_params * 0.5) / (1024**3), 3)
            safe_train_vram_gb = vram_gb * 0.85
            if train_est.total_vram_gb <= safe_train_vram_gb:
                train_mode = "resident_bf16_train"
            elif fp8_train_static_gb <= safe_train_vram_gb:
                train_mode = "resident_fp8_train (--use_fp8)"
            elif bf16_gb <= ram_gb * 0.65:
                train_mode = "layer_streaming_train (--streaming --use_fp8)"
            else:
                train_mode = "disk_nvfp4_streaming_train (--streaming --use_fp8)"
            return {
                "model": str(model_id_or_dir),
                "preset_id": spec.model_id,
                "is_triune": True,
                "total_params": total_params,
                "active_params": active_params,
                "top_k_experts": spec.top_k_experts,
                "layers": layers,
                "hidden_size": hidden,
                "vocab_size": vocab,
                "tied_embeddings": tied,
                "bf16_weights_gb": round(bf16_gb, 2),
                "nvfp4_disk_cache_gb": round(nvfp4_gb, 2),
                "nvfp4_quant_lmhead_cache_gb": round(nvfp4_quant_head_gb, 2),
                "per_layer_nvfp4_mb": round(per_layer_nvfp4_mb, 1),
                "centroid_steer_opt_gb": train_est.optimizer_memory_gb,
                "centroid_steer_static_vram_gb": train_est.total_vram_gb,
                "centroid_steer_fp8_static_vram_gb": fp8_train_static_gb,
                "available_vram_gb": round(vram_gb, 2),
                "host_ram_gb": round(ram_gb, 2),
                "recommended_mode": mode,
                "recommended_train_mode": train_mode,
            }

        from transformers import AutoConfig

        if local_files_only:
            cfg = AutoConfig.from_pretrained(str(model_id_or_dir), local_files_only=True)
        else:
            try:
                cfg = AutoConfig.from_pretrained(str(model_id_or_dir), local_files_only=True)
            except Exception:
                cfg = AutoConfig.from_pretrained(str(model_id_or_dir))

        vocab = int(getattr(cfg, "vocab_size", 32000))
        hidden = int(getattr(cfg, "hidden_size", 1024))
        layers = int(getattr(cfg, "num_hidden_layers", 24))
        inter = int(getattr(cfg, "intermediate_size", hidden * 4))
        n_heads = int(getattr(cfg, "num_attention_heads", max(1, hidden // 64)))
        n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
        head_dim = int(getattr(cfg, "head_dim", hidden // max(1, n_heads)))
        tied = bool(getattr(cfg, "tie_word_embeddings", False))

        # Per-layer param formula for standard GQA + SwiGLU Transformers
        attn_p = hidden * (n_heads * head_dim) + 2 * hidden * (n_kv * head_dim) + (n_heads * head_dim) * hidden
        mlp_p = 3 * hidden * inter  # gate, up, down (or 2x non-gated)
        per_layer_p = attn_p + mlp_p + 2 * hidden
        root_p = vocab * hidden * (1 if tied else 2) + hidden
        total_params = root_p + layers * per_layer_p

        bf16_gb = (total_params * 2.0) / (1024**3)
        nvfp4_gb = (root_p * 2.0 + layers * per_layer_p * 0.625) / (1024**3)
        nvfp4_quant_head_gb = (
            vocab * hidden * 2.0 + (0 if tied else vocab * hidden * 0.625) + layers * per_layer_p * 0.625
        ) / (1024**3)
        per_layer_nvfp4_mb = (per_layer_p * 0.625) / (1024**2)

        if bf16_gb + 1.5 <= vram_gb:
            mode = "resident_bf16"
        elif nvfp4_gb + 1.2 <= vram_gb:
            mode = "resident_nvfp4"
        elif bf16_gb <= ram_gb * 0.65:
            mode = "ram_layer_streaming"
        else:
            mode = "disk_nvfp4_streaming"

        return {
            "model": str(model_id_or_dir),
            "is_triune": False,
            "total_params": total_params,
            "layers": layers,
            "hidden_size": hidden,
            "vocab_size": vocab,
            "tied_embeddings": tied,
            "bf16_weights_gb": round(bf16_gb, 2),
            "nvfp4_disk_cache_gb": round(nvfp4_gb, 2),
            "nvfp4_quant_lmhead_cache_gb": round(nvfp4_quant_head_gb, 2),
            "per_layer_nvfp4_mb": round(per_layer_nvfp4_mb, 1),
            "available_vram_gb": round(vram_gb, 2),
            "host_ram_gb": round(ram_gb, 2),
            "recommended_mode": mode,
        }
