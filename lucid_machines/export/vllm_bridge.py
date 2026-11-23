"""vLLM Serving Configuration Bridge with Safe Remote Code Defaults."""

from __future__ import annotations

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


__all__ = ["build_vllm_serving_config"]
