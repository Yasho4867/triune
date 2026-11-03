"""Rotary Position Embedding (RoPE) implementation with fast kernel dispatch."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .config import ROPE_MAX_SEQ_LEN


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = ROPE_MAX_SEQ_LEN, base: float = 10000.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError(f"RoPE dimension must be a positive even number, got {dim}.")
        base = float(base)
        if max_seq_len <= 0 or not math.isfinite(base) or base <= 0:
            raise ValueError("RoPE max_seq_len and base must be positive, with a finite base.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        total_len = offset + seq_len
        if total_len > self.cos_cached.size(0):
            new_max = max(total_len, self.cos_cached.size(0) * 2)
            self.max_seq_len = new_max
            t = torch.arange(new_max, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            emb = torch.cat((freqs, freqs), dim=-1)
            target_dtype = self.cos_cached.dtype
            self.register_buffer("cos_cached", emb.cos().to(device=device, dtype=target_dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin().to(device=device, dtype=target_dtype), persistent=False)
        elif self.cos_cached.device != device:
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)
        return (
            self.cos_cached[offset:total_len],
            self.sin_cached[offset:total_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from triune.kernels import fast_rope
    return fast_rope(q, k, cos, sin)
