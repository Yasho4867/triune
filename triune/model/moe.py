from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    CENTROID_EMA,
    EXPERT_HIDDEN_MULTIPLIER,
    MOE_BIAS_UPDATE_RATE,
    MOE_CAPACITY_MULTIPLIER,
    NUM_EXPERTS,
    SHARED_EXPERT,
    SHARED_EXPERT_SCALE,
    TOP_K_EXPERTS,
    is_checkpoint_recomputing,
)
from .fp4 import FP4Linear
from .fp8 import FP8Linear
from triune.kernels.fast_swiglu import FastGELU


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
    def __init__(
        self,
        dim: int,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K_EXPERTS,
        use_fp4: bool = False,
        use_fp8: bool = False,
        shared_expert: bool = SHARED_EXPERT,
        shared_scale: float = SHARED_EXPERT_SCALE,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_fp4 = use_fp4
        self.use_fp8 = use_fp8
        self.shared_expert = shared_expert
        self.shared_scale = shared_scale

        Linear = FP8Linear if use_fp8 else (FP4Linear if use_fp4 else nn.Linear)
        hidden_dim = dim * EXPERT_HIDDEN_MULTIPLIER

        # Standard expert MLP: canonical GELU activation matching pre-trained checkpoints
        self.experts = nn.ModuleList([
            nn.Sequential(
                Linear(dim, hidden_dim),
                FastGELU(),
                Linear(hidden_dim, dim),
            ) for _ in range(num_experts)
        ])

        # DeepSeek-V3 inspired Shared Expert: always processes every token with fixed scale
        if self.shared_expert:
            self.shared = nn.Sequential(
                Linear(dim, hidden_dim),
                FastGELU(),
                Linear(hidden_dim, dim),
            )
        else:
            self.shared = None

        self.router = nn.Linear(dim, num_experts)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.register_buffer("expert_load_ratio", torch.ones(num_experts))
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

        if self.training:
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

        # Fast path for single-token autoregressive decoding: evaluate only the selected top-k
        # experts directly without per-expert nonzero() CUDA stream syncs, padding, or bincount.
        if not self.training and B * T == 1 and not any(isinstance(m, FP4Linear) for m in self.experts.modules()):
            if self.shared_expert:
                out = (self.shared_scale * self.shared(flat_x)).to(flat_x.dtype)
            else:
                out = torch.zeros_like(flat_x)
            chosen_experts = flat_idx[0].tolist()
            for k_idx, e_idx in enumerate(chosen_experts):
                w = flat_vals[0, k_idx]
                out = out + (self.experts[e_idx](flat_x) * w).to(out.dtype)
            return out.reshape(B, T, D)

        if self.training:
            capacity = int(math.ceil((B * T) / self.num_experts * MOE_CAPACITY_MULTIPLIER))
        else:
            capacity = B * T
        out = torch.zeros_like(flat_x)
        need_stats = bool(self.training and update_stats and not is_checkpoint_recomputing())
        accepted_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device) if need_stats else None
        accepted_mask = torch.zeros((B * T, self.top_k), dtype=torch.bool, device=x.device) if need_stats else None

        if self.shared_expert:
            shared_out = self.shared(flat_x)
            out = out + (self.shared_scale * shared_out).to(out.dtype)

        any_b1 = any(e[0].bias is not None for e in self.experts)
        all_b1 = all(e[0].bias is not None for e in self.experts)
        any_b2 = any(e[2].bias is not None for e in self.experts)
        all_b2 = all(e[2].bias is not None for e in self.experts)
        mixed_biases = (any_b1 != all_b1) or (any_b2 != all_b2)

        can_batch_bmm = (
            not self.use_fp4
            and not self.use_fp8
            and not mixed_biases
            and all(isinstance(e[0], nn.Linear) and isinstance(e[2], nn.Linear) for e in self.experts)
        )

        if can_batch_bmm:
            W1_stack = torch.stack([e[0].weight for e in self.experts], dim=0)
            W2_stack = torch.stack([e[2].weight for e in self.experts], dim=0)
            has_b1 = all_b1
            has_b2 = all_b2
            b1_stack = torch.stack([e[0].bias for e in self.experts], dim=0).unsqueeze(1) if has_b1 else None
            b2_stack = torch.stack([e[2].bias for e in self.experts], dim=0).unsqueeze(1) if has_b2 else None

        for k in range(self.top_k):
            idx_k = flat_idx[:, k]
            val_k = flat_vals[:, k]

            if can_batch_bmm:
                expert_token_indices = []
                expert_val_keeps = []
                max_tokens = 0
                for e in range(self.num_experts):
                    mask = (idx_k == e)
                    indices = mask.nonzero(as_tuple=True)[0]
                    if indices.numel() == 0:
                        expert_token_indices.append(indices)
                        expert_val_keeps.append(indices)
                        continue
                    dropped = indices.numel() - capacity
                    if dropped > 0:
                        scores = val_k[indices]
                        keep = torch.argsort(scores, descending=True)[:capacity]
                        if need_stats:
                            self.overflow_counter += dropped
                        indices = indices[keep]
                        val_k_keep = val_k[indices]
                    else:
                        val_k_keep = val_k[indices]

                    if need_stats:
                        accepted_counts[e] += indices.numel()
                        accepted_mask[indices, k] = True

                    expert_token_indices.append(indices)
                    expert_val_keeps.append(val_k_keep)
                    if indices.numel() > max_tokens:
                        max_tokens = indices.numel()

                if max_tokens > 0:
                    pad_m = (-max_tokens) % 16
                    aligned_m = max_tokens + pad_m
                    padded_input = torch.zeros(self.num_experts, aligned_m, D, device=flat_x.device, dtype=flat_x.dtype)
                    for e in range(self.num_experts):
                        idx = expert_token_indices[e]
                        if idx.numel() > 0:
                            padded_input[e, :idx.numel()] = flat_x[idx]

                    h_pre = torch.bmm(padded_input, W1_stack.transpose(1, 2))
                    if b1_stack is not None:
                        h_pre = h_pre + b1_stack
                    h = self.experts[0][1](h_pre)

                    y = torch.bmm(h, W2_stack.transpose(1, 2))
                    if b2_stack is not None:
                        y = y + b2_stack

                    for e in range(self.num_experts):
                        idx = expert_token_indices[e]
                        if idx.numel() > 0:
                            val_keep = expert_val_keeps[e]
                            out = out.index_add(0, idx, (y[e, :idx.numel()] * val_keep.unsqueeze(-1)).to(out.dtype))
            else:
                for e in range(self.num_experts):
                    mask = (idx_k == e)
                    indices = mask.nonzero(as_tuple=True)[0]
                    if indices.numel() == 0:
                        continue
                    dropped = indices.numel() - capacity
                    if dropped > 0:
                        scores = val_k[indices]
                        keep = torch.argsort(scores, descending=True)[:capacity]
                        if need_stats:
                            self.overflow_counter += dropped
                        indices = indices[keep]
                        val_k_keep = val_k[indices]
                    else:
                        val_k_keep = val_k[indices]

                    if need_stats:
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
                    out = out.index_add(0, indices, (expert_output * val_k_keep.unsqueeze(-1)).to(out.dtype))

        if need_stats:
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
            self._update_routing_stats(flat_x, routing_result)

        if self.training and not getattr(self, "_step_managed_by_engine", False) and not is_checkpoint_recomputing():
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

        # Centroids tracking with exponential moving average (CENTROID_EMA)
        ema_decay = getattr(self, "centroid_ema", CENTROID_EMA)
        if self.last_centroids is None or self.last_centroids.device != flat_x.device or self.last_centroids.shape != (self.num_experts, self.dim):
            self.last_centroids = torch.zeros(self.num_experts, self.dim, device=flat_x.device, dtype=flat_x.dtype)

        for expert_idx in range(self.num_experts):
            mask = (result.indices == expert_idx) & result.accepted_mask
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
            if token_indices.numel() > 0:
                batch_centroid = flat_x[token_indices].mean(dim=0).detach()
                curr_centroid = self.last_centroids[expert_idx]
                if curr_centroid.norm() < 1e-6:
                    self.last_centroids[expert_idx] = batch_centroid
                else:
                    self.last_centroids[expert_idx] = ema_decay * curr_centroid + (1.0 - ema_decay) * batch_centroid
            # If no tokens accepted this batch, retain the previous EMA centroid (do not zero out)

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
