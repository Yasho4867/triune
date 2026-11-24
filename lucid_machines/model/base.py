"""Universal Model Abstraction Base Class & Registry.

Provides support for Dense Transformers, MoEs, SSM/Mamba, and User-Defined Custom Architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


@dataclass
class ModelSpec:
    """Authoritative, single source of truth for Triune model architecture & runtime execution (LucidMachines)."""

    model_id: str = "triune-small"
    name: str = "Triune-Small (Canonical 6L MoE)"
    vocab_size: int = 32000
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    head_dim: int = 64
    num_experts: int = 4
    experts_routed: int = 4
    experts_shared: int = 1
    top_k_experts: int = 1
    shared_expert: bool = True
    shared_expert_scale: float = 0.5
    expert_hidden_multiplier: int = 6
    rope_max_seq_len: int = 8192
    router_prefix_layers: int = 1
    reflex_exit_layer: int = 2
    limbic_exit_layer: int = 4
    target_depth_dist: Tuple[float, float, float] = (0.50, 0.30, 0.20)
    balance_coef: float = 0.3
    depth_mode: str = "dynamic"
    precision: str = "bf16"
    use_fp4: bool = False
    use_fp8: bool = False
    streaming_fp8_weights: bool = False
    use_layer_streaming: bool = False
    chunk_size: int = 1
    recommended_batch: int = 4
    recommended_seq: int = 64
    tier: str = "Production Baseline"
    params_label: str = "33.8M params"
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def compute_exact_params(self) -> Tuple[int, int]:
        """Compute exact mathematical (total_params, active_params_per_token) for this TriuneTransformer spec."""
        v = int(self.vocab_size)
        d = int(self.hidden_dim)
        l = max(4, int(self.num_layers))
        r = max(2, min(int(self.reflex_exit_layer), l - 2))
        e = max(1, int(self.num_experts))
        has_shared = bool(self.shared_expert) and int(self.experts_shared) >= 1
        s = 1 if has_shared else 0
        mult = max(1, int(self.expert_hidden_multiplier))
        top_k = max(1, min(int(self.top_k_experts), e))

        vocab_and_exits = (3 * v * d) + (2 * (v + d)) + d
        router_params = (d * 256) + 256 + (256 * 3) + 3
        dense_layers = r + 1
        moe_layers = max(0, l - dense_layers)

        per_dense = (13 * d * d) + (7 * d)
        per_moe_total = (5 + 2 * mult * (e + s)) * d * d + ((mult + 1) * (e + s) + e + 2) * d + e
        active_experts = top_k + s
        per_moe_active = (5 + 2 * mult * active_experts) * d * d + ((mult + 1) * active_experts + e + 2) * d + e

        total_params = vocab_and_exits + router_params + (dense_layers * per_dense) + (moe_layers * per_moe_total)
        active_params = vocab_and_exits + router_params + (dense_layers * per_dense) + (moe_layers * per_moe_active)
        return total_params, active_params

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        total_p, active_p = self.compute_exact_params()
        d["preset_id"] = self.model_id
        d["params"] = self.params_label
        d["total_params"] = total_p
        d["active_params"] = active_p
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSpec":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        if "model_id" not in filtered and "preset_id" in data:
            filtered["model_id"] = str(data["preset_id"])
        if "model_id" not in filtered and "model_name" in data:
            filtered["model_id"] = str(data["model_name"])
        if "target_depth_dist" in filtered and isinstance(filtered["target_depth_dist"], list):
            filtered["target_depth_dist"] = tuple(float(x) for x in filtered["target_depth_dist"])
        extra = {k: v for k, v in data.items() if k not in valid_keys}
        if "extra_config" not in filtered:
            filtered["extra_config"] = extra
        return cls(**filtered)


CANONICAL_MODEL_SPECS: Dict[str, ModelSpec] = {
    "triune-tiny": ModelSpec(
        model_id="triune-tiny",
        name="Triune-Tiny (13M, 4L / 128D / 2E)",
        tier="Tier 0 - Ultra-Light Prototyping",
        num_layers=4,
        hidden_dim=128,
        num_heads=2,
        head_dim=64,
        num_experts=2,
        experts_routed=2,
        experts_shared=1,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
        recommended_batch=4,
        recommended_seq=64,
        params_label="13.2M params (13.2M active)",
    ),
    "triune-nano": ModelSpec(
        model_id="triune-nano",
        name="Triune-Nano (21M, 4L / 192D / 2E)",
        tier="Tier 1 - Edge & CPU Prototyping",
        num_layers=4,
        hidden_dim=192,
        num_heads=3,
        head_dim=64,
        num_experts=2,
        experts_routed=2,
        experts_shared=1,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
        recommended_batch=4,
        recommended_seq=64,
        params_label="21.0M params (21.0M active)",
    ),
    "triune-small": ModelSpec(
        model_id="triune-small",
        name="Triune-Small (36M, 6L / 256D / 4E)",
        tier="Tier 2 - Canonical Baseline (Default)",
        num_layers=6,
        hidden_dim=256,
        num_heads=4,
        head_dim=64,
        num_experts=4,
        experts_routed=4,
        experts_shared=1,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=4,
        recommended_batch=4,
        recommended_seq=64,
        params_label="36.0M params (32.9M active)",
    ),
    "triune-base": ModelSpec(
        model_id="triune-base",
        name="Triune-Base (76M, 8L / 384D / 4E)",
        tier="Tier 3 - Consumer GPU Pre-Training",
        num_layers=8,
        hidden_dim=384,
        num_heads=6,
        head_dim=64,
        num_experts=4,
        experts_routed=4,
        experts_shared=1,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=5,
        recommended_batch=4,
        recommended_seq=128,
        params_label="75.8M params (64.0M active)",
    ),
    "triune-edge": ModelSpec(
        model_id="triune-edge",
        name="Triune-Edge (152M, 8L / 512D / 4E)",
        tier="Tier 3B - Local Workstation Edge MoE",
        num_layers=8,
        hidden_dim=512,
        num_heads=4,
        head_dim=128,
        num_experts=4,
        experts_routed=4,
        experts_shared=1,
        router_prefix_layers=2,
        reflex_exit_layer=3,
        limbic_exit_layer=6,
        recommended_batch=2,
        recommended_seq=128,
        params_label="152.0M params (126.8M active)",
    ),
    "triune-pro": ModelSpec(
        model_id="triune-pro",
        name="Triune-Pro (208M, 12L / 512D / 8E)",
        tier="Tier 4 - Mid-Scale 8-Expert MoE",
        num_layers=12,
        hidden_dim=512,
        num_heads=8,
        head_dim=64,
        num_experts=8,
        experts_routed=8,
        experts_shared=1,
        top_k_experts=2,
        rope_max_seq_len=8192,
        router_prefix_layers=2,
        reflex_exit_layer=4,
        limbic_exit_layer=8,
        recommended_batch=2,
        recommended_seq=128,
        params_label="207.5M params (119.4M active)",
    ),
    "triune-large": ModelSpec(
        model_id="triune-large",
        name="Triune-Large (574M, 16L / 768D / 8E)",
        tier="Tier 5 - Sub-Billion High-Capacity MoE",
        num_layers=16,
        hidden_dim=768,
        num_heads=12,
        head_dim=64,
        num_experts=8,
        experts_routed=8,
        experts_shared=1,
        top_k_experts=2,
        rope_max_seq_len=8192,
        router_prefix_layers=2,
        reflex_exit_layer=5,
        limbic_exit_layer=10,
        recommended_batch=1,
        recommended_seq=128,
        params_label="573.9M params (290.8M active)",
    ),
    "triune-1b": ModelSpec(
        model_id="triune-1b",
        name="Triune-1B (1.50B, 24L / 1024D / 8E)",
        tier="Billion-Scale - Entry Frontier MoE",
        num_layers=24,
        hidden_dim=1024,
        num_heads=16,
        head_dim=64,
        num_experts=8,
        experts_routed=8,
        experts_shared=1,
        top_k_experts=2,
        rope_max_seq_len=8192,
        router_prefix_layers=3,
        reflex_exit_layer=7,
        limbic_exit_layer=16,
        recommended_batch=1,
        recommended_seq=128,
        params_label="1.50B params (696M active)",
    ),
    "triune-2.5b": ModelSpec(
        model_id="triune-2.5b",
        name="Triune-2.5B (2.37B, 18L / 1280D / 8E)",
        tier="Multi-Billion - 2.5B Workstation MoE",
        num_layers=18,
        hidden_dim=1280,
        num_heads=10,
        head_dim=128,
        num_experts=8,
        experts_routed=8,
        experts_shared=1,
        top_k_experts=1,
        rope_max_seq_len=4096,
        router_prefix_layers=3,
        reflex_exit_layer=5,
        limbic_exit_layer=13,
        recommended_batch=1,
        recommended_seq=128,
        params_label="2.37B params (1.01B active)",
    ),
    "triune-moe": ModelSpec(
        model_id="triune-moe",
        name="Triune-MoE (3.35B, 24L / 1536D / 8E)",
        tier="Multi-Billion - Canonical 24L MoE",
        num_layers=24,
        hidden_dim=1536,
        num_heads=12,
        head_dim=128,
        num_experts=8,
        experts_routed=8,
        experts_shared=1,
        top_k_experts=2,
        rope_max_seq_len=8192,
        router_prefix_layers=3,
        reflex_exit_layer=6,
        limbic_exit_layer=16,
        recommended_batch=1,
        recommended_seq=128,
        params_label="3.35B params (1.50B active)",
    ),
    "triune-7b": ModelSpec(
        model_id="triune-7b",
        name="Triune-7B (7.38B, 32L / 1920D / 8E)",
        tier="Multi-Billion - 7B Flagship MoE",
        num_layers=32,
        hidden_dim=1920,
        num_heads=30,
        head_dim=64,
        num_experts=8,
        experts_routed=8,
        experts_shared=1,
        top_k_experts=1,
        rope_max_seq_len=4096,
        router_prefix_layers=4,
        reflex_exit_layer=7,
        limbic_exit_layer=20,
        recommended_batch=1,
        recommended_seq=128,
        params_label="7.38B params (3.13B active)",
    ),
    "triune-frontier": ModelSpec(
        model_id="triune-frontier",
        name="Triune-Frontier (34.5B, 48L / 2560D / 16E)",
        tier="Multi-Billion - Frontier Sparse MoE",
        num_layers=48,
        hidden_dim=2560,
        num_heads=20,
        head_dim=128,
        num_experts=16,
        experts_routed=16,
        experts_shared=1,
        top_k_experts=2,
        rope_max_seq_len=16384,
        router_prefix_layers=5,
        reflex_exit_layer=11,
        limbic_exit_layer=30,
        recommended_batch=1,
        recommended_seq=128,
        params_label="34.5B params (8.11B active)",
    ),
    "junkie-mini": ModelSpec(
        model_id="junkie-mini",
        name="Junkie-Mini (535.1M, 12L / 1024D / 4E)",
        tier="Fast Single-GPU Target - Junkie-Mini on Triune Architecture",
        num_layers=12,
        hidden_dim=1024,
        num_heads=8,
        head_dim=128,
        num_experts=4,
        experts_routed=4,
        experts_shared=1,
        top_k_experts=1,
        rope_max_seq_len=4096,
        router_prefix_layers=2,
        reflex_exit_layer=6,
        limbic_exit_layer=9,
        recommended_batch=2,
        recommended_seq=512,
        params_label="535.1M params (230.8M expected active)",
    ),
    "junkie-1b": ModelSpec(
        model_id="junkie-1b",
        name="Junkie-1B (1.16B, 22L / 1024D / 4E)",
        tier="1B Single-GPU Target - Junkie-1B on Triune Architecture",
        num_layers=22,
        hidden_dim=1024,
        num_heads=8,
        head_dim=128,
        num_experts=4,
        experts_routed=4,
        experts_shared=1,
        top_k_experts=1,
        rope_max_seq_len=4096,
        router_prefix_layers=2,
        reflex_exit_layer=7,
        limbic_exit_layer=14,
        recommended_batch=2,
        recommended_seq=512,
        params_label="1.16B params (375.0M expected active)",
    ),
    "junkie-3b": ModelSpec(
        model_id="junkie-3b",
        name="Junkie-3B (2.99B, 16L / 1024D / 16E)",
        tier="Colab L4 Sweet Spot - 16-Expert Sparse Junkie-3B on Triune Architecture",
        num_layers=16,
        hidden_dim=1024,
        num_heads=8,
        head_dim=128,
        num_experts=16,
        experts_routed=16,
        experts_shared=1,
        top_k_experts=1,
        rope_max_seq_len=4096,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=5,
        recommended_batch=2,
        recommended_seq=512,
        params_label="2.99B params (206.7M dynamic / 256.7M uniform active)",
    ),
    "junkie-7b": ModelSpec(
        model_id="junkie-7b",
        name="Junkie-7B (7.40B, 32L / 2048D / 8E)",
        tier="Flagship Release - Junkie-7B on Triune Architecture",
        num_layers=32,
        hidden_dim=2048,
        num_heads=16,
        head_dim=128,
        num_experts=8,
        experts_routed=8,
        experts_shared=1,
        top_k_experts=2,
        rope_max_seq_len=8192,
        router_prefix_layers=4,
        reflex_exit_layer=18,
        limbic_exit_layer=25,
        recommended_batch=1,
        recommended_seq=512,
        params_label="7.40B params (1.95B expected active)",
    ),
    "junkie": ModelSpec(
        model_id="junkie",
        name="Junkie (7.40B Flagship, 32L / 2048D / 8E)",
        tier="Flagship Release - First Model on Triune Architecture",
        num_layers=32,
        hidden_dim=2048,
        num_heads=16,
        head_dim=128,
        num_experts=8,
        experts_routed=8,
        experts_shared=1,
        top_k_experts=2,
        rope_max_seq_len=8192,
        router_prefix_layers=4,
        reflex_exit_layer=18,
        limbic_exit_layer=25,
        recommended_batch=1,
        recommended_seq=512,
        params_label="7.40B params (1.95B expected active)",
    ),
}

_LEGACY_PRESET_ALIASES: Dict[str, str] = {
    "triune-micro": "triune-nano",
    "triune-drafter": "triune-nano",
    "triune-code-moe": "triune-base",
    "triune-math-gla": "triune-base",
    "triune-multimodal": "triune-base",
    "triune-medium": "triune-pro",
    "triune-1.5b": "triune-1b",
    "triune-2.5b": "triune-3b",
    "triune-3b": "triune-2.5b",
    "triune-3.0b": "triune-2.5b",
    "triune-moe": "triune-7b",
    "triune-7.0b": "triune-7b",
    "triune-14b": "triune-frontier",
    "triune-14.5b": "triune-frontier",
    "triune-32b": "triune-frontier",
    "triune-32.0b": "triune-frontier",
    "triune-34b": "triune-frontier",
    "triune-34.5b": "triune-frontier",
    "triune-70b": "triune-frontier",
    "triune-70.0b": "triune-frontier",
    "triune-71b": "triune-frontier",
    "triune-71.0b": "triune-frontier",
}


def get_model_spec(preset_id: Optional[str] = None, **overrides: Any) -> ModelSpec:
    """Resolve a canonical ModelSpec by preset_id and apply any overrides while enforcing layer hierarchy."""
    raw_id = (preset_id or "triune-small").lower()
    p_id = raw_id
    seen = set()
    while p_id in _LEGACY_PRESET_ALIASES and p_id not in seen:
        seen.add(p_id)
        p_id = _LEGACY_PRESET_ALIASES[p_id]
        if p_id in CANONICAL_MODEL_SPECS:
            break
    base_spec = CANONICAL_MODEL_SPECS.get(p_id, CANONICAL_MODEL_SPECS["triune-small"])
    spec_dict = base_spec.to_dict()
    if raw_id in CANONICAL_MODEL_SPECS:
        spec_dict["model_id"] = raw_id
    elif preset_id:
        spec_dict["model_id"] = preset_id
        if raw_id not in _LEGACY_PRESET_ALIASES:
            spec_dict["name"] = f"Triune-Custom ({preset_id})"

    for k, v in overrides.items():
        if v is not None:
            spec_dict[k] = v

    L = max(4, int(spec_dict.get("num_layers", 6)))
    r_prefix = max(1, min(int(spec_dict.get("router_prefix_layers", 1)), max(1, L - 3)))
    r_reflex = max(r_prefix + 1, min(int(spec_dict.get("reflex_exit_layer", max(2, L // 3))), max(r_prefix + 1, L - 2)))
    r_limbic = max(
        r_reflex + 1, min(int(spec_dict.get("limbic_exit_layer", max(3, (2 * L) // 3))), max(r_reflex + 1, L - 1))
    )

    num_heads = max(1, int(spec_dict.get("num_heads", 4)))
    head_dim = max(1, int(spec_dict.get("head_dim", 64)))
    hidden_dim = int(spec_dict.get("hidden_dim", num_heads * head_dim))
    if hidden_dim != num_heads * head_dim:
        if "hidden_dim" in overrides and "head_dim" not in overrides and hidden_dim % num_heads == 0:
            head_dim = hidden_dim // num_heads
        else:
            hidden_dim = num_heads * head_dim

    num_experts = max(1, int(spec_dict.get("num_experts", 4)))

    req_prec = str(spec_dict.get("precision", "bf16")).lower()
    if "fp8" in req_prec or "e4m3" in req_prec or "e5m2" in req_prec:
        req_prec = "fp8"
    elif "fp4" in req_prec or "nf4" in req_prec or "int4" in req_prec:
        req_prec = "fp4"
    elif "32" in req_prec:
        req_prec = "fp32"
    elif "16" in req_prec and "bf" not in req_prec:
        req_prec = "fp16"

    if "precision" in overrides and overrides["precision"] is not None:
        use_fp8 = bool(overrides.get("use_fp8", req_prec == "fp8"))
        use_fp4 = bool(overrides.get("use_fp4", req_prec == "fp4"))
    else:
        use_fp4 = bool(spec_dict.get("use_fp4", False) or req_prec == "fp4")
        use_fp8 = bool(spec_dict.get("use_fp8", False) or req_prec == "fp8")
    precision = (
        "fp8" if use_fp8 else ("fp4" if use_fp4 else (req_prec if req_prec in ("bf16", "fp16", "fp32") else "bf16"))
    )

    spec_dict.update(
        {
            "num_layers": L,
            "router_prefix_layers": r_prefix,
            "reflex_exit_layer": r_reflex,
            "limbic_exit_layer": r_limbic,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "use_fp4": use_fp4,
            "use_fp8": use_fp8,
            "precision": precision,
        }
    )
    return ModelSpec.from_dict(spec_dict)


def build_model_from_spec(spec: ModelSpec) -> nn.Module:
    """Builds a real executable TriuneTransformer strictly conforming to the ModelSpec."""
    from lucid_machines.model.transformer import TriuneTransformer

    # Validate layer hierarchy invariants: 0 < router_prefix < reflex < limbic < num_layers (minimum 4 layers)
    L = max(4, spec.num_layers)
    r_prefix = max(1, min(spec.router_prefix_layers, max(1, L - 3)))
    r_reflex = max(r_prefix + 1, min(spec.reflex_exit_layer, max(r_prefix + 1, L - 2)))
    r_limbic = max(r_reflex + 1, min(spec.limbic_exit_layer, max(r_reflex + 1, L - 1)))
    h_dim = spec.num_heads * spec.head_dim

    total_p, _ = spec.compute_exact_params()
    use_bf16_alloc = total_p >= 150_000_000 and str(getattr(spec, "precision", "bf16")).lower() != "fp32"
    orig_dtype = torch.get_default_dtype()
    orig_linear_reset = nn.Linear.reset_parameters
    orig_embed_reset = nn.Embedding.reset_parameters

    try:
        if use_bf16_alloc:
            torch.set_default_dtype(torch.bfloat16)
            nn.Linear.reset_parameters = lambda self: None
            nn.Embedding.reset_parameters = lambda self: None

        model = TriuneTransformer(
            vocab_size=spec.vocab_size,
            hidden_dim=h_dim,
            num_layers=L,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            num_experts=spec.num_experts,
            top_k_experts=spec.top_k_experts,
            shared_expert=spec.shared_expert,
            shared_expert_scale=spec.shared_expert_scale,
            expert_hidden_multiplier=spec.expert_hidden_multiplier,
            rope_max_seq_len=spec.rope_max_seq_len,
            router_prefix_layers=r_prefix,
            reflex_exit_layer=r_reflex,
            limbic_exit_layer=r_limbic,
            target_depth_dist=spec.target_depth_dist,
            balance_coef=spec.balance_coef,
            use_fp4=spec.use_fp4,
            use_fp8=spec.use_fp8,
            streaming_fp8_weights=spec.streaming_fp8_weights,
        )
    finally:
        if use_bf16_alloc:
            torch.set_default_dtype(orig_dtype)
            nn.Linear.reset_parameters = orig_linear_reset
            nn.Embedding.reset_parameters = orig_embed_reset

    if use_bf16_alloc:
        with torch.no_grad():
            has_cuda = torch.cuda.is_available()
            gpu_buf = None
            max_numel = 0
            for name, p in model.named_parameters():
                if "norm" in name or p.ndim == 1:
                    if "bias" in name:
                        p.zero_()
                    else:
                        p.fill_(1.0)
                elif has_cuda and p.device.type == "cpu" and p.numel() >= 65536:
                    if gpu_buf is None or max_numel < p.numel():
                        gpu_buf = torch.empty(p.numel(), dtype=p.dtype, device="cuda:0")
                        max_numel = p.numel()
                    view = gpu_buf[: p.numel()].view_as(p)
                    view.normal_(0.0, 0.02)
                    p.copy_(view, non_blocking=False)
                else:
                    p.normal_(0.0, 0.02)
            del gpu_buf

    model.model_spec = spec
    return model


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator to register a custom model class into the LucidMachines Model Zoo."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class TriuneModel(nn.Module, ABC):
    """Abstract Base Class for all Triune-compatible model architectures (LucidMachines)."""

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, force_depth: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (logits, aux_router_loss)."""
        pass

    @abstractmethod
    def get_num_params(self) -> int:
        """Return total parameter count."""
        pass
