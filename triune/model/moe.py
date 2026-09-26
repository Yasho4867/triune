from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fp4 import *
from .fp8 import FP8Linear
from .norms import *
from .config import *


@dataclass
class RoutingResult:
    """Unified routing statistics and assignment tensor container."""
    indices: torch.Tensor             # [B * T, top_k] expert indices
    weights: torch.Tensor             # [B * T, top_k] normalized routing weights
    accepted_mask: torch.Tensor       # [B * T, top_k] boolean mask of tokens accepted by expert capacity
    requested_counts: torch.Tensor    # [num_experts] total tokens requesting each expert
    accepted_counts: torch.Tensor     # [num_experts] tokens actually accepted into capacity
    dropped_counts: torch.Tensor      # [num_experts] tokens dropped due to capacity overflow


class MoE_FFN(nn.Module):
    def __init__(self, dim, num_experts=NUM_EXPERTS, top_k=TOP_K_EXPERTS, use_fp4=False, use_fp8=False, shared_expert=SHARED_EXPERT):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        self.shared_expert = shared_expert
        self.shared_scale = SHARED_EXPERT_SCALE
        if use_fp8:
            LinearCls = FP8Linear
        elif use_fp4:
            LinearCls = FP4Linear
        else:
            LinearCls = nn.Linear

        self.experts = nn.ModuleList([
            nn.Sequential(
                LinearCls(dim, dim * EXPERT_HIDDEN_MULTIPLIER),
                nn.GELU(),
                LinearCls(dim * EXPERT_HIDDEN_MULTIPLIER, dim)
            )
            for _ in range(num_experts)
        ])

        if shared_expert:
            self.shared = nn.Sequential(
                LinearCls(dim, dim * EXPERT_HIDDEN_MULTIPLIER),
                nn.GELU(),
                LinearCls(dim * EXPERT_HIDDEN_MULTIPLIER, dim)
            )

        self.router = nn.Linear(dim, num_experts)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('expert_load_ratio', torch.ones(num_experts))
        self.last_centroids = None
        self.last_counts = None
        self.last_requested_counts = None
        self.last_accepted_counts = None
        self.last_dropped_counts = None
        self.last_target = None
        self.overflow_counter = 0
        self._global_step = 0

    def forward(self, x, update_stats=True):
        """If update_stats=False, skip bias/centroid updates (used during label generation)."""
        B, T, D = x.shape
        flat_x = x.reshape(B * T, D)

        bias_free_logits = self.router(x)
        biased_logits = bias_free_logits + self.expert_bias

        if self.training and update_stats:
            temp = max(0.1, 1.0 - self._global_step / 5000)
            gate_weights = F.softmax(bias_free_logits / temp, dim=-1)
        else:
            gate_weights = F.softmax(bias_free_logits, dim=-1)

        # DeepSeek-V3 style load-balancing: biased_logits selects top-k experts for balanced
        # assignment, while unbiased bias_free_logits provides pure token mixing weights.
        _, top_idx = torch.topk(biased_logits, self.top_k, dim=-1)
        flat_idx = top_idx.reshape(B * T, self.top_k)
        flat_vals = torch.gather(gate_weights, dim=-1, index=top_idx).reshape(B * T, self.top_k)

        # Renormalize top-k weights so selected expert contributions sum to 1.0
        if self.top_k > 1:
            flat_vals = flat_vals / (flat_vals.sum(dim=-1, keepdim=True) + 1e-20)

        if self.training:
            capacity = int(math.ceil((B * T) / self.num_experts * MOE_CAPACITY_MULTIPLIER))
        else:
            capacity = B * T
        out = torch.zeros_like(flat_x)
        accepted_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        accepted_mask = torch.zeros((B * T, self.top_k), dtype=torch.bool, device=x.device)

        if self.shared_expert:
            shared_out = self.shared(flat_x)
            out = out + self.shared_scale * shared_out

        for k in range(self.top_k):
            idx_k = flat_idx[:, k]
            val_k = flat_vals[:, k]
            for e in range(self.num_experts):
                mask = (idx_k == e)
                indices = mask.nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                dropped = indices.numel() - capacity
                if dropped > 0:
                    scores = val_k[indices]
                    keep = torch.argsort(scores, descending=True)[:capacity]
                    self.overflow_counter += dropped
                    indices = indices[keep]
                    val_k_keep = val_k[indices]
                else:
                    val_k_keep = val_k[indices]

                accepted_counts[e] += indices.numel()
                accepted_mask[indices, k] = True
                expert_input = flat_x[indices]

                orig_tokens = expert_input.size(0)
                pad = (-orig_tokens) % 16

                if pad:
                     expert_input = F.pad(expert_input, (0, 0, 0, pad))

                expert_output = self.experts[e](expert_input)

                if pad:
                    expert_output = expert_output[:orig_tokens]

                # Fix C-1: out-of-place to preserve autograd graph
                out = out.index_add(0, indices, expert_output * val_k_keep.unsqueeze(-1))

        requested_counts = torch.bincount(flat_idx.flatten(), minlength=self.num_experts)
        dropped_counts = (requested_counts - accepted_counts).clamp_min(0)
        routing_result = RoutingResult(
            indices=flat_idx,
            weights=flat_vals,
            accepted_mask=accepted_mask,
            requested_counts=requested_counts,
            accepted_counts=accepted_counts,
            dropped_counts=dropped_counts,
        )

        if self.training and update_stats:
            self._update_routing_stats(flat_x, routing_result)

        if self.training:
            self._global_step += 1

        return out.reshape(B, T, D)

    @torch.no_grad()
    def _update_routing_stats(self, flat_x, result: RoutingResult):
        """Update non-gradient routing state without retaining an activation graph."""
        target = max(1.0, float((flat_x.size(0) * self.top_k) / self.num_experts))
        
        # Bias steering equalizes requested demand across experts
        self._pending_bias_update = -(result.requested_counts - target).sign() * MOE_BIAS_UPDATE_RATE

        # Expert utilization ratio reflects actual accepted tokens processed
        batch_ratio = result.accepted_counts.float() / target
        self.expert_load_ratio.mul_(0.9).add_(batch_ratio.to(self.expert_load_ratio.device), alpha=0.1)
        self.last_counts = result.requested_counts.detach()
        self.last_requested_counts = result.requested_counts.detach()
        self.last_accepted_counts = result.accepted_counts.detach()
        self.last_dropped_counts = result.dropped_counts.detach()
        self.last_target = target

        # Centroids must use accepted assignments only (Phase 8)
        centroids = []
        for expert_idx in range(self.num_experts):
            mask = (result.indices == expert_idx) & result.accepted_mask
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
            if token_indices.numel() > 0:
                centroids.append(flat_x[token_indices].mean(dim=0))
            else:
                centroids.append(torch.zeros(self.dim, device=flat_x.device, dtype=flat_x.dtype))
        self.last_centroids = torch.stack(centroids)

    @torch.no_grad()
    def step_bias(self):
        """Apply pending bias updates after backward is complete."""
        if hasattr(self, "_pending_bias_update") and self._pending_bias_update is not None:
            self.expert_bias.add_(self._pending_bias_update.to(self.expert_bias.device)).clamp_(-5.0, 5.0)
            self._pending_bias_update = None

    @torch.no_grad()
    def update_routing_stats(self, x):
        """Update routing state for checkpointed forwards exactly once."""
        B, T, D = x.shape
        flat_x = x.detach().reshape(B * T, D)

        bias_free_logits = self.router(x)
        biased_logits = bias_free_logits + self.expert_bias

        if self.training:
            temp = max(0.1, 1.0 - self._global_step / 5000)
            gate_weights = F.softmax(bias_free_logits / temp, dim=-1)
        else:
            gate_weights = F.softmax(bias_free_logits, dim=-1)

        _, top_idx = torch.topk(biased_logits, self.top_k, dim=-1)
        flat_idx = top_idx.reshape(B * T, self.top_k)
        flat_vals = torch.gather(gate_weights, dim=-1, index=top_idx).reshape(B * T, self.top_k)
        if self.top_k > 1:
            flat_vals = flat_vals / (flat_vals.sum(dim=-1, keepdim=True) + 1e-20)

        capacity = int(math.ceil((B * T) / self.num_experts * MOE_CAPACITY_MULTIPLIER))
        accepted_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        accepted_mask = torch.zeros((B * T, self.top_k), dtype=torch.bool, device=x.device)

        for k in range(self.top_k):
            idx_k = flat_idx[:, k]
            val_k = flat_vals[:, k]
            for e in range(self.num_experts):
                mask = (idx_k == e)
                indices = mask.nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                dropped = indices.numel() - capacity
                if dropped > 0:
                    scores = val_k[indices]
                    keep = torch.argsort(scores, descending=True)[:capacity]
                    indices = indices[keep]
                accepted_counts[e] += indices.numel()
                accepted_mask[indices, k] = True

        requested_counts = torch.bincount(flat_idx.flatten(), minlength=self.num_experts)
        dropped_counts = (requested_counts - accepted_counts).clamp_min(0)
        result = RoutingResult(
            indices=flat_idx,
            weights=flat_vals,
            accepted_mask=accepted_mask,
            requested_counts=requested_counts,
            accepted_counts=accepted_counts,
            dropped_counts=dropped_counts,
        )
        self._update_routing_stats(flat_x, result)

# ─── Transformer Block ──────────────────────────────────────────
