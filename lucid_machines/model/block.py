"""Transformer block containing GLA attention, MoE FFN, and early exit heads."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from lucid_machines.kernels.fast_swiglu import FastGELU

from .attention import HybridAttention
from .config import (
    EXPERT_HIDDEN_MULTIPLIER,
    NUM_EXPERTS,
    ROPE_MAX_SEQ_LEN,
    SHARED_EXPERT,
    SHARED_EXPERT_SCALE,
    TOP_K_EXPERTS,
    checkpoint_bwd_ctx,
    checkpoint_fwd_ctx,
)
from .fp4 import FP4Linear
from .fp8 import FP8Linear
from .moe import MoE_FFN
from .norms import RMSNorm
from .rotary import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        layer_idx: int,
        vocab_size: int,
        exit_layers: tuple[int, ...] | list[int],
        num_experts: int = NUM_EXPERTS,
        top_k_experts: int = TOP_K_EXPERTS,
        shared_expert: bool = SHARED_EXPERT,
        shared_expert_scale: float = SHARED_EXPERT_SCALE,
        expert_hidden_multiplier: int = EXPERT_HIDDEN_MULTIPLIER,
        rope_max_seq_len: int = ROPE_MAX_SEQ_LEN,
        use_moe: bool = True,
        use_fp4: bool = False,
        use_fp8: bool = False,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.top_k_experts = max(1, int(top_k_experts))
        self.shared_expert = bool(shared_expert)
        self.shared_expert_scale = float(shared_expert_scale)
        self.expert_hidden_multiplier = max(1, int(expert_hidden_multiplier))
        self.rope_max_seq_len = max(1, int(rope_max_seq_len))
        self.attn = HybridAttention(
            dim,
            heads,
            use_fp4=use_fp4,
            use_fp8=use_fp8,
            use_sdpa=use_sdpa,
            layer_idx=layer_idx,
            reflex_exit_layer=min(exit_layers) if exit_layers else None,
        )
        if getattr(self.attn.gla, "use_rope", False) and self.rope_max_seq_len != ROPE_MAX_SEQ_LEN:
            self.attn.gla.rope = RotaryEmbedding(self.attn.head_dim, max_seq_len=self.rope_max_seq_len)
        if use_moe:
            self.ffn = MoE_FFN(
                dim,
                num_experts=num_experts,
                top_k=self.top_k_experts,
                use_fp4=use_fp4,
                use_fp8=use_fp8,
                shared_expert=self.shared_expert,
                shared_scale=self.shared_expert_scale,
            )
            custom_hidden = dim * self.expert_hidden_multiplier
            if custom_hidden != self.ffn.hidden_dim:
                Linear = FP8Linear if use_fp8 else (FP4Linear if use_fp4 else nn.Linear)
                self.ffn.hidden_dim = custom_hidden
                self.ffn.experts = nn.ModuleList(
                    [
                        nn.Sequential(Linear(dim, custom_hidden), FastGELU(), Linear(custom_hidden, dim))
                        for _ in range(num_experts)
                    ]
                )
                if self.shared_expert:
                    self.ffn.shared = nn.Sequential(
                        Linear(dim, custom_hidden),
                        FastGELU(),
                        Linear(custom_hidden, dim),
                    )
                self.ffn.register_buffer("last_hidden_centroids", torch.zeros(num_experts, custom_hidden))
                self.ffn.register_buffer("last_hidden_contrastive_centroids", torch.zeros(num_experts, custom_hidden))
                self.ffn.register_buffer(
                    "last_hidden_subspace", torch.zeros(num_experts, custom_hidden, self.ffn.subspace_dirs)
                )
        else:
            DenseLinear = FP8Linear if use_fp8 else (FP4Linear if use_fp4 else nn.Linear)
            self.ffn = nn.Sequential(DenseLinear(dim, dim * 4), FastGELU(), DenseLinear(dim * 4, dim))
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

    def forward(
        self,
        x: torch.Tensor,
        return_exit: bool = False,
        cache=None,
        update_stats: bool = True,
        exit_decider: Callable[[torch.Tensor], bool] | None = None,
        exit_last_token_only: bool = False,
        active_mask: torch.Tensor | None = None,
    ):

        use_token_compaction = (
            active_mask is not None and active_mask.shape == x.shape[:2] and not bool(active_mask.all())
        )
        if self._use_gradient_checkpointing and self.training and torch.is_grad_enabled() and not use_token_compaction:
            attn_out, new_cache = torch.utils.checkpoint.checkpoint(
                lambda n: self.attn(n, cache=cache),
                self.norm1(x),
                use_reentrant=False,
                context_fn=lambda: (checkpoint_fwd_ctx(), checkpoint_bwd_ctx()),
            )
            x = x + attn_out
            norm2_x = self.norm2(x)
            if isinstance(self.ffn, MoE_FFN):
                ffn_out = torch.utils.checkpoint.checkpoint(
                    lambda n: self.ffn(n, update_stats=update_stats),
                    norm2_x,
                    use_reentrant=False,
                    context_fn=lambda: (checkpoint_fwd_ctx(), checkpoint_bwd_ctx()),
                )
            else:
                ffn_out = torch.utils.checkpoint.checkpoint(
                    self.ffn,
                    norm2_x,
                    use_reentrant=False,
                    context_fn=lambda: (checkpoint_fwd_ctx(), checkpoint_bwd_ctx()),
                )
            x = x + ffn_out
        else:
            attn_out, new_cache = self.attn(self.norm1(x), cache=cache)
            x = x + attn_out
            if use_token_compaction:
                x_active = x[active_mask]
                if x_active.numel() > 0:
                    x_norm = self.norm2(x_active.unsqueeze(0))
                    if isinstance(self.ffn, MoE_FFN):
                        ffn_active = self.ffn(x_norm, update_stats=update_stats).squeeze(0)
                    else:
                        ffn_active = self.ffn(x_norm).squeeze(0)
                    x = x.clone()
                    x[active_mask] = x_active + ffn_active.to(x.dtype)
            elif isinstance(self.ffn, MoE_FFN):
                x = x + self.ffn(self.norm2(x), update_stats=update_stats)
            else:
                x = x + self.ffn(self.norm2(x))
        if self.exit_head and (return_exit or (exit_decider is not None and exit_decider(x))):
            x_exit = x[:, -1:, :] if exit_last_token_only else x
            exit_logits = self.exit_head(self.exit_norm(x_exit))
            return x, exit_logits, new_cache
        return x, None, new_cache
