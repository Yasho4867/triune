"""Transformer block containing GLA attention, MoE FFN, and early exit heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import HybridAttention
from .config import NUM_EXPERTS
from .fp4 import FP4Linear
from .fp8 import FP8Linear
from .moe import MoE_FFN
from .norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        layer_idx: int,
        vocab_size: int,
        exit_layers: tuple[int, ...] | list[int],
        num_experts: int = NUM_EXPERTS,
        use_moe: bool = True,
        use_fp4: bool = False,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.attn = HybridAttention(dim, heads, use_fp4=use_fp4, use_fp8=use_fp8)
        if use_moe:
            self.ffn = MoE_FFN(dim, num_experts=num_experts, use_fp4=use_fp4, use_fp8=use_fp8)
        else:
            DenseLinear = FP8Linear if use_fp8 else (FP4Linear if use_fp4 else nn.Linear)
            self.ffn = nn.Sequential(
                DenseLinear(dim, dim * 4), nn.GELU(), DenseLinear(dim * 4, dim)
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

    def forward(self, x: torch.Tensor, return_exit: bool = False, cache=None, update_stats: bool = True):
        if self._use_gradient_checkpointing and self.training:
            attn_out, new_cache = torch.utils.checkpoint.checkpoint(
                lambda n: self.attn(n, cache=cache), self.norm1(x), use_reentrant=False
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
