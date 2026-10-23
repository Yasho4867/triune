"""Unified Model Zoo & Model Loading API.

Provides seamless loading for native Triune models ("triune-small", "triune-base", "triune-moe"),
Hugging Face models ("llama-3", "qwen-2.5"), and user-registered custom architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch

from .base import CANONICAL_MODEL_SPECS, MODEL_REGISTRY, build_model_from_spec, get_model_spec
from .factory import build_model


def load_model(model_name_or_path: str | Path, **kwargs: Any) -> torch.nn.Module:
    """Unified Model Loader.

    Loads native Triune models, registered custom architectures, or local checkpoints.
    """
    from triune.configs import build_config

    model_name_str = str(model_name_or_path).lower()
    path_obj = Path(model_name_or_path)

    # Check local JSON config file
    if path_obj.is_file() and path_obj.suffix.lower() == ".json":
        import json
        with open(path_obj, encoding="utf-8") as f:
            file_cfg = json.load(f)
        config = build_config({**file_cfg, **kwargs})
        return build_model(config)

    # Check local checkpoint path
    if path_obj.is_file():
        checkpoint = torch.load(model_name_or_path, map_location="cpu", weights_only=True)
        saved_config = checkpoint.get("config", {})
        config = build_config({**saved_config, **kwargs})
        model = build_model(config)
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model

    # Check registered model zoo
    if model_name_str in MODEL_REGISTRY:
        cls = MODEL_REGISTRY[model_name_str]
        return cls(**kwargs)

    # Canonical built-in Triune presets (single source of truth in CANONICAL_MODEL_SPECS)
    if model_name_str in CANONICAL_MODEL_SPECS:
        spec = get_model_spec(model_name_str, **kwargs)
        return build_model_from_spec(spec)

    # Check JSON config file directly or inside models/configs/
    json_path = Path(model_name_or_path)
    if not json_path.is_file() and not json_path.suffix:
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

    if model_name_str in ("default", "none", ""):
        config = build_config(kwargs)
        return build_model(config)

    # Check Hugging Face hub / local directory
    hf_err = None
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(str(model_name_or_path), **kwargs)
    except Exception as exc:
        hf_err = exc

    raise ValueError(
        f"Model '{model_name_or_path}' could not be resolved. "
        f"It is not a valid checkpoint path, registered architecture in MODEL_REGISTRY, "
        f"known JSON config, or built-in preset ({list(CANONICAL_MODEL_SPECS.keys())}). "
        f"Attempted Hugging Face AutoModelForCausalLM loading failed with: {hf_err}"
    )
