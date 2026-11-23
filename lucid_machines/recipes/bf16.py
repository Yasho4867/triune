from contextlib import nullcontext

import torch


def bf16_autocast(device_type: str = "cuda"):
    return torch.amp.autocast(device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
