import torch
import torch.nn as nn

from .attention import *
from .router import *
from .moe import *
from .norms import *
from .fp8 import FP8Linear

from .config import *
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, layer_idx, vocab_size, exit_layers, num_experts=NUM_EXPERTS, use_moe=True, use_fp4=False, use_fp8=False):
        super().__init__()
        self.attn = HybridAttention(dim, heads, use_fp4=use_fp4)
        if use_moe:
            self.ffn = MoE_FFN(dim, num_experts=num_experts, use_fp4=use_fp4, use_fp8=use_fp8)
        else:
            DenseLinear = FP8Linear if use_fp8 else (FP4Linear if use_fp4 else nn.Linear)
            self.ffn = nn.Sequential(
                DenseLinear(dim, dim*4), nn.GELU(), DenseLinear(dim*4, dim)
            )
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.layer_idx = layer_idx
        if layer_idx in exit_layers:
            self.exit_norm = RMSNorm(dim)
            self.exit_head = nn.Linear(dim, vocab_size)
        else:
            self.exit_norm = None
            self.exit_head = None
        self.use_moe = use_moe
        self._use_gradient_checkpointing = False

    def forward(self, x, return_exit=False, cache=None, update_stats=True):
        if self._use_gradient_checkpointing and self.training:
            attn_out, new_cache = torch.utils.checkpoint.checkpoint(
                self.attn, self.norm1(x), use_reentrant=False
            )
            x = x + attn_out
            norm2_x = self.norm2(x)
            if isinstance(self.ffn, MoE_FFN):
                ffn_out = torch.utils.checkpoint.checkpoint(
                    lambda n: self.ffn(n, update_stats=update_stats), norm2_x, use_reentrant=False
                )
            else:
                ffn_out = torch.utils.checkpoint.checkpoint(
                    self.ffn, norm2_x, use_reentrant=False
                )
            x = x + ffn_out
        else:
            attn_out, new_cache = self.attn(self.norm1(x), cache=cache)
            x = x + attn_out
            if isinstance(self.ffn, MoE_FFN):
                x = x + self.ffn(self.norm2(x), update_stats=update_stats)
            else:
                x = x + self.ffn(self.norm2(x))
        if self.exit_head and return_exit:
            exit_logits = self.exit_head(self.exit_norm(x))
            return x, exit_logits, new_cache
        return x, None, new_cache

# ─── TriuneTransformer ──────────────────────────────────────────
