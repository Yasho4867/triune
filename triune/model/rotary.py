import math
import torch
import torch.nn as nn

from .config import ROPE_MAX_SEQ_LEN

from .config import *
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=ROPE_MAX_SEQ_LEN):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len, device):
        # Return in fp32 to preserve precision; caller casts q/k to bf16
        if seq_len <= self.max_seq_len:
            return self.cos_cached[:seq_len].to(device, torch.float32), self.sin_cached[:seq_len].to(device, torch.float32)

        # Generation can grow past the training sequence length.  Do not silently
        # return a too-short cache; build the required positions on demand.
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(q, k, cos, sin):
    # q, k are bf16; cos, sin are fp32 – we cast inside
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

