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


def _pytorch_chunk_gla(q, k, v, g, initial_state=None):
    """Reference recurrent PyTorch implementation of Gated Linear Attention (GLA).
    
    Used as fallback when flash-linear-attention (FLA) is absent, on non-CUDA devices,
    or during state-cached autoregressive inference.
    Warning: This sequential loop runs in O(T) steps and is intended for correctness and
    single-token autoregressive generation, rather than high-throughput chunkwise training.
    """
    B, T, H, HD = q.shape
    decay = torch.exp(g)
    if initial_state is not None:
        S = initial_state.to(device=q.device, dtype=q.dtype).clone()
    else:
        S = torch.zeros(B, H, HD, HD, device=q.device, dtype=q.dtype)
    out = torch.empty(B, T, H, HD, device=q.device, dtype=q.dtype)
    for t in range(T):
        decay_t = decay[:, t, :, :].unsqueeze(-1)
        k_t = k[:, t, :, :].unsqueeze(-1)
        v_t = v[:, t, :, :].unsqueeze(-2)
        S = S * decay_t + torch.matmul(k_t, v_t)
        q_t = q[:, t, :, :].unsqueeze(-2)
        out[:, t, :, :] = torch.matmul(q_t, S).squeeze(-2)
    return out, S


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

        initial_state = None
        offset = 0
        cache_format = "none"
        if isinstance(cache, dict):
            initial_state = cache.get("state")
            offset = cache.get("offset", 0)
            cache_format = "dict"
        elif isinstance(cache, tuple):
            initial_state, offset = cache
            cache_format = "tuple"
        elif torch.is_tensor(cache):
            initial_state = cache
            cache_format = "tensor"

        if self.use_rope:
            cos, sin = self.rope(T, x.device, offset=offset)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q, k = apply_rotary(q, k, cos, sin)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        if HAS_FLA and x.is_cuda and initial_state is None:
            # Convert tensors to bfloat16/float16 for FLA Triton kernel compatibility & memory efficiency
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q_fla = q.to(dtype=target_dtype)
            k_fla = k.to(dtype=target_dtype)
            v_fla = v.to(dtype=target_dtype)
            g_fla = g.to(dtype=target_dtype)
            out, final_state = chunk_gla(q_fla, k_fla, v_fla, g_fla, scale=None)
            out = out.to(dtype=x.dtype)
        else:
            out, final_state = _pytorch_chunk_gla(q, k, v, g, initial_state=initial_state)

        out = out.reshape(B, T, self.hidden_dim)
        if cache_format == "dict":
            new_cache = {"state": final_state, "offset": offset + T}
        elif cache_format == "tuple":
            new_cache = (final_state, offset + T)
        elif cache_format == "tensor":
            new_cache = final_state
        else:
            new_cache = (final_state, offset + T) if cache is not None else None
        return self.out_proj(out), new_cache



class HybridAttention(nn.Module):
    def __init__(self, dim, heads, use_fp4=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.gla = VectorisedGLA(dim, heads, GLA_HEAD_DIM)

    def forward(self, x, cache=None):
        return self.gla(x, cache)

