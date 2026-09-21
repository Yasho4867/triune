import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import *
from .rotary import *


try:
    from fla.ops.gla import chunk_gla
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    chunk_gla = None


def _pytorch_chunk_gla(q, k, v, g):
    """Vectorized PyTorch fallback for chunk_gla when flash-linear-attention is absent."""
    B, T, H, HD = q.shape
    decay = torch.exp(g)
    S = torch.zeros(B, H, HD, HD, device=q.device, dtype=q.dtype)
    out = torch.empty(B, T, H, HD, device=q.device, dtype=q.dtype)
    for t in range(T):
        decay_t = decay[:, t, :, :].unsqueeze(-1)
        k_t = k[:, t, :, :].unsqueeze(-1)
        v_t = v[:, t, :, :].unsqueeze(-2)
        S = S * decay_t + torch.matmul(k_t, v_t)
        q_t = q[:, t, :, :].unsqueeze(-2)
        out[:, t, :, :] = torch.matmul(q_t, S).squeeze(-2)
    return out, None


class VectorisedGLA(nn.Module):
    def __init__(self, dim, heads, head_dim=GLA_HEAD_DIM):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.hidden_dim = heads * head_dim
        self.q_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.g_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.use_rope = USE_ROPE
        if self.use_rope:
            self.rope = RotaryEmbedding(head_dim, max_seq_len=ROPE_MAX_SEQ_LEN)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        H, HD = self.heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, HD)
        k = self.k_proj(x).view(B, T, H, HD)
        v = self.v_proj(x).view(B, T, H, HD)
        g = F.logsigmoid(self.g_proj(x).view(B, T, H, HD))

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        if self.use_rope:
            cos, sin = self.rope(T, x.device)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q, k = apply_rotary(q, k, cos, sin)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        if HAS_FLA and x.is_cuda:
            # Convert tensors to bfloat16/float16 for FLA Triton kernel compatibility & memory efficiency
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q_fla = q.to(dtype=target_dtype)
            k_fla = k.to(dtype=target_dtype)
            v_fla = v.to(dtype=target_dtype)
            g_fla = g.to(dtype=target_dtype)
            out, _ = chunk_gla(q_fla, k_fla, v_fla, g_fla, scale=None)
            out = out.to(dtype=x.dtype)
        else:
            out, _ = _pytorch_chunk_gla(q, k, v, g)

        out = out.reshape(B, T, self.hidden_dim)
        return self.out_proj(out), None



class HybridAttention(nn.Module):
    def __init__(self, dim, heads, use_fp4=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.gla = VectorisedGLA(dim, heads, GLA_HEAD_DIM)

    def forward(self, x, cache=None):
        return self.gla(x, cache)

