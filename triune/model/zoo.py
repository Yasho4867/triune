"""Unified Model Zoo & Model Loading API.

Provides seamless loading for native Triune models ("triune-small", "triune-base", "triune-moe"),
Hugging Face models ("llama-3", "qwen-2.5"), and user-registered custom architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import torch

from .base import MODEL_REGISTRY
from .factory import build_model


def load_model(model_name_or_path: str | Path, **kwargs: Any) -> torch.nn.Module:
    """Unified Model Loader.

    Loads native Triune models, registered custom architectures, or local checkpoints.
    """
    from triune.configs import build_config

    model_name_str = str(model_name_or_path).lower()


    # Check local checkpoint path
    if Path(model_name_or_path).is_file():
        checkpoint = torch.load(model_name_or_path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", {})
        config = build_config({**saved_config, **kwargs})
        model = build_model(config)
        state_dict = checkpoint["model_state"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model

    # Check registered model zoo
    if model_name_str in MODEL_REGISTRY:
        cls = MODEL_REGISTRY[model_name_str]
        return cls(**kwargs)

    # Check JSON config file directly or inside models/configs/
    json_path = Path(model_name_or_path)
    if not json_path.is_file() and not json_path.suffix:
        # Check standard config directory
        repo_root = Path(__file__).resolve().parent.parent.parent
        cand = repo_root / "models" / "configs" / f"{model_name_str}.json"
        if cand.is_file():
            json_path = cand

    if json_path.is_file() and json_path.suffix.lower() == ".json":
        import json
        with open(json_path, encoding="utf-8") as f:
            file_cfg = json.load(f)
        config = build_config({**file_cfg, **kwargs})
        return build_model(config)

    # Built-in Triune presets
    presets = {
        "triune-nano": {
            "num_layers": 8,
            "hidden_dim": 512,
            "num_heads": 4,
            "head_dim": 128,
            "num_experts": 4,
            "router_prefix_layers": 2,
            "reflex_exit_layer": 3,
            "limbic_exit_layer": 6,
        },
        "triune-small": {
            "num_layers": 14,
            "hidden_dim": 1024,
            "num_heads": 8,
            "head_dim": 128,
            "num_experts": 4,
            "router_prefix_layers": 3,
            "reflex_exit_layer": 5,
            "limbic_exit_layer": 10,
        },
        "triune-2.5b": {
            "num_layers": 18,
            "hidden_dim": 1280,
            "num_heads": 10,
            "head_dim": 128,
            "num_experts": 8,
            "router_prefix_layers": 3,
            "reflex_exit_layer": 5,
            "limbic_exit_layer": 13,
        },
        "triune-base": {
            "num_layers": 24,
            "hidden_dim": 1536,
            "num_heads": 12,
            "head_dim": 128,
            "num_experts": 8,
            "router_prefix_layers": 3,
            "reflex_exit_layer": 6,
            "limbic_exit_layer": 16,
        },
        "triune-7b": {
            "num_layers": 32,
            "hidden_dim": 2048,
            "num_heads": 16,
            "head_dim": 128,
            "num_experts": 8,
            "router_prefix_layers": 4,
            "reflex_exit_layer": 8,
            "limbic_exit_layer": 22,
        },
        "triune-moe": {
            "num_layers": 32,
            "hidden_dim": 1536,
            "num_heads": 12,
            "head_dim": 128,
            "num_experts": 16,
            "router_prefix_layers": 3,
            "reflex_exit_layer": 6,
            "limbic_exit_layer": 16,
        },
        "triune-large": {
            "num_layers": 32,
            "hidden_dim": 2048,
            "num_heads": 16,
            "head_dim": 128,
            "num_experts": 16,
            "router_prefix_layers": 4,
            "reflex_exit_layer": 8,
            "limbic_exit_layer": 22,
        },
    }

    if model_name_str in presets:
        config = build_config({**presets[model_name_str], **kwargs})
        return build_model(config)

    # Fallback to default build_model with kwargs as config overrides
    config = build_config(kwargs)
    return build_model(config)
