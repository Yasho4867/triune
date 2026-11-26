"""vLLM Serving Configuration Bridge with Safe Remote Code Defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def build_vllm_serving_config(
    model_id: str,
    output_dir: str | Path = "exports",
    *,
    architecture: str = "TriuneTransformer",
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
    enable_chunked_prefill: bool = True,
    trust_remote_code: bool = False,
    port: int = 8000,
) -> Dict[str, Any]:
    """Generate a hardened vLLM serving configuration with trust_remote_code disabled by default."""
    out_dir = Path(output_dir)
    weights_path = (out_dir / f"{model_id}.safetensors").as_posix()
    return {
        "model_id": model_id,
        "architecture": architecture,
        "weights_path": str(weights_path),
        "tensor_parallel_size": int(tensor_parallel_size),
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "enforce_eager": bool(enforce_eager),
        "enable_chunked_prefill": bool(enable_chunked_prefill),
        "trust_remote_code": bool(trust_remote_code),
        "launch_command": (
            f"python -m vllm.entrypoints.openai.api_server --model ./{model_id}.safetensors --port {int(port)}"
        ),
    }


def export_vllm_bundle(
    model: Any,
    output_dir: str | Path,
    *,
    model_id: str | None = None,
    tokenizer: Any = None,
    tokenizer_path: str | Path | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = False,
    enable_chunked_prefill: bool = True,
    trust_remote_code: bool = False,
    port: int = 8000,
) -> Path:
    """Export an HF-compatible Triune directory plus vllm_serving.json deployment manifest."""
    from .hf_compat import export_to_huggingface

    out_dir = Path(output_dir)
    export_to_huggingface(model, out_dir, tokenizer=tokenizer, tokenizer_path=tokenizer_path)
    resolved_id = model_id or out_dir.name or "triune"
    cfg = build_vllm_serving_config(
        resolved_id,
        out_dir,
        architecture=type(getattr(model, "transformer", model)).__name__,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=enable_chunked_prefill,
        trust_remote_code=trust_remote_code,
        port=port,
    )
    cfg["bundle_dir"] = out_dir.as_posix()
    cfg["weights_path"] = (out_dir / "model.safetensors").as_posix()
    cfg["launch_command"] = (
        f"python -m vllm.entrypoints.openai.api_server --model {out_dir.as_posix()} --port {int(port)}"
    )
    cfg["note"] = (
        "TriuneForCausalLM uses HF auto_map; pass --trust-remote-code explicitly if loading custom modeling_triune.py."
    )
    (out_dir / "vllm_serving.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out_dir


__all__ = ["build_vllm_serving_config", "export_vllm_bundle"]
