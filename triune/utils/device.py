"""Non-invasive device diagnostics."""

from __future__ import annotations

import sys

import torch


def device_diagnostic() -> dict:
    result = {"python": sys.executable, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if result["cuda_available"]:
        result["gpus"] = [
            {"name": torch.cuda.get_device_name(index), "memory_gib": torch.cuda.get_device_properties(index).total_memory / 1024**3,
             "capability": torch.cuda.get_device_capability(index)}
            for index in range(torch.cuda.device_count())
        ]
    else:
        result["gpus"] = []
    return result
