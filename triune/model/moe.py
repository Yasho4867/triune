import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.kernels.fast_swiglu import FastGELU

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


@dataclass
class RoutingResult:
    """Unified routing statistics and assignment tensor container."""

    indices: torch.Tensor  # [B * T, top_k] expert indices
    weights: torch.Tensor  # [B * T, top_k] normalized routing weights
    accepted_mask: torch.Tensor  # [B * T, top_k] boolean mask of tokens accepted by expert capacity
    requested_counts: torch.Tensor  # [num_experts] total tokens requesting each expert
    accepted_counts: torch.Tensor  # [num_experts] tokens actually accepted into capacity
    dropped_counts: torch.Tensor  # [num_experts] tokens dropped due to capacity overflow


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
        self.use_sparse_eval = False

        Linear = FP8Linear if use_fp8 else (FP4Linear if use_fp4 else nn.Linear)
        hidden_dim = dim * EXPERT_HIDDEN_MULTIPLIER

        # Standard expert MLP: canonical GELU activation matching pre-trained checkpoints
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    Linear(dim, hidden_dim),
                    FastGELU(),
                    Linear(hidden_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )

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
        self.register_buffer("last_centroids", torch.zeros(num_experts, dim))
        self.last_counts = None
        self.last_requested_counts = None
        self.last_accepted_counts = None
        self.last_dropped_counts = None
        self.last_target = None
        self.register_buffer("_overflow_counter", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long))

    @property
    def overflow_counter(self) -> int:
        buffers = self.__dict__.get("_buffers")
        if buffers is not None and "_overflow_counter" in buffers and buffers["_overflow_counter"] is not None:
            return int(buffers["_overflow_counter"].item())
        return 0

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "overflow_counter":
            buffers = self.__dict__.get("_buffers")
            if buffers is not None and "_overflow_counter" in buffers and buffers["_overflow_counter"] is not None:
                if isinstance(value, torch.Tensor):
                    buffers["_overflow_counter"].copy_(
                        value.to(dtype=torch.long, device=buffers["_overflow_counter"].device)
                    )
                else:
                    buffers["_overflow_counter"].fill_(int(value))
                return
            value = torch.tensor(int(value), dtype=torch.long)
            super().__setattr__("_overflow_counter", value)
            return
        if name == "_global_step" and not isinstance(value, torch.Tensor):
            buffers = self.__dict__.get("_buffers")
            if buffers is not None and "_global_step" in buffers and buffers["_global_step"] is not None:
                buffers["_global_step"].fill_(int(value))
                return
            value = torch.tensor(int(value), dtype=torch.long)
        super().__setattr__(name, value)

    def sparse_forward(self, x: torch.Tensor) -> torch.Tensor:
        """True Top-k sparse scatter-gather forward pass grouped by selected expert."""
        B, T, D = x.shape
        flat_x = x.reshape(-1, D)
        logits = self.router(flat_x)
        _, top_idx = torch.topk(logits + self.expert_bias, self.top_k, dim=-1)
        weights = torch.gather(F.softmax(logits, dim=-1), 1, top_idx)
        if self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        expert_ids = top_idx.reshape(-1)
        order = expert_ids.argsort(stable=True)
        token_ids = torch.arange(flat_x.size(0), device=x.device).repeat_interleave(self.top_k)[order]
        counts = torch.bincount(expert_ids, minlength=self.num_experts).tolist()
        chunks = flat_x[token_ids].split(counts)
        ys = [self.experts[e](c) for e, c in enumerate(chunks) if c.numel()]
        out = torch.zeros_like(flat_x)
        if ys:
            y = torch.cat(ys)
            y = y * weights.reshape(-1)[order].unsqueeze(-1).to(y.dtype)
            out.index_add_(0, token_ids, y.to(out.dtype))
        if self.shared_expert:
            out = out + (self.shared_scale * self.shared(flat_x)).to(out.dtype)
        return out.reshape(B, T, D)

    def forward(self, x, update_stats=True):
        """If update_stats=False, skip bias/centroid updates (used during label generation)."""
        if not self.training and getattr(self, "use_sparse_eval", False):
            return self.sparse_forward(x)
        B, T, D = x.shape
        N_tok = B * T
        flat_x = x.reshape(N_tok, D)

        bias_free_logits = self.router(x)
        biased_logits = bias_free_logits + self.expert_bias

        if self.training:
            temp = (1.0 - self._global_step.float() / 5000.0).clamp(min=0.1)
            gate_weights = F.softmax(bias_free_logits / temp, dim=-1)
        else:
            gate_weights = F.softmax(bias_free_logits, dim=-1)

        # DeepSeek-V3 style load-balancing: biased_logits selects top-k experts for balanced
        # assignment, while unbiased bias_free_logits provides pure token mixing weights.
        _, top_idx = torch.topk(biased_logits, self.top_k, dim=-1)
        flat_idx = top_idx.reshape(N_tok, self.top_k)
        flat_vals = torch.gather(gate_weights, dim=-1, index=top_idx).reshape(N_tok, self.top_k)

        # Renormalize top-k weights so selected expert contributions sum to 1.0
        if self.top_k > 1:
            flat_vals = flat_vals / (flat_vals.sum(dim=-1, keepdim=True) + 1e-20)

        def _get_submod_weight_bias(submod):
            if hasattr(submod, "original_layer"):
                orig = submod.original_layer
                return getattr(orig, "weight", None), getattr(orig, "bias", None)
            return getattr(submod, "weight", None), getattr(submod, "bias", None)

        def _get_submod_lora(submod):
            if (
                hasattr(submod, "lora_A")
                and submod.lora_A is not None
                and hasattr(submod, "lora_B")
                and submod.lora_B is not None
            ):
                return (
                    submod.lora_A,
                    submod.lora_B,
                    float(getattr(submod, "scaling", getattr(submod, "lora_scaling", 2.0))),
                )
            return None

        any_b1 = any(_get_submod_weight_bias(e[0])[1] is not None for e in self.experts)
        all_b1 = all(_get_submod_weight_bias(e[0])[1] is not None for e in self.experts)
        any_b2 = any(_get_submod_weight_bias(e[2])[1] is not None for e in self.experts)
        all_b2 = all(_get_submod_weight_bias(e[2])[1] is not None for e in self.experts)
        mixed_biases = (any_b1 != all_b1) or (any_b2 != all_b2)

        has_lora_e0 = all(_get_submod_lora(e[0]) is not None for e in self.experts)
        has_lora_e2 = all(_get_submod_lora(e[2]) is not None for e in self.experts)
        no_lora_e0 = all(_get_submod_lora(e[0]) is None for e in self.experts)
        no_lora_e2 = all(_get_submod_lora(e[2]) is None for e in self.experts)

        can_batch_bmm = (
            not self.use_fp4
            and not self.use_fp8
            and not mixed_biases
            and (has_lora_e0 or no_lora_e0)
            and (has_lora_e2 or no_lora_e2)
            and all(
                _get_submod_weight_bias(e[0])[0] is not None and _get_submod_weight_bias(e[2])[0] is not None
                for e in self.experts
            )
        )

        is_capturing = bool(x.is_cuda and torch.cuda.is_current_stream_capturing())
        use_shared_stream = bool(x.is_cuda and self.shared_expert and not is_capturing)
        shared_stream = None
        shared_event = None
        shared_out = None

        if use_shared_stream:
            if not hasattr(self, "_shared_stream") or self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream(device=x.device)
            shared_stream = self._shared_stream
            shared_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(shared_stream):
                shared_out = self.shared(flat_x)
                shared_event = torch.cuda.Event()
                shared_event.record(shared_stream)

        # Fast path for single-token autoregressive decoding: evaluate only the selected top-k
        # experts directly on GPU without .tolist() or nonzero() CUDA stream syncs.
        if not self.training and N_tok == 1 and not any(isinstance(m, FP4Linear) for m in self.experts.modules()):
            out = torch.zeros_like(flat_x)
            if can_batch_bmm:
                W1_stack = torch.stack([_get_submod_weight_bias(e[0])[0] for e in self.experts], dim=0)
                W2_stack = torch.stack([_get_submod_weight_bias(e[2])[0] for e in self.experts], dim=0)
                sel_idx = flat_idx[0]  # [top_k]
                w1_sel = W1_stack[sel_idx]  # [top_k, H_ffn, D]
                w2_sel = W2_stack[sel_idx]  # [top_k, D, H_ffn]
                x_sel = flat_x.unsqueeze(0).expand(self.top_k, 1, D)  # [top_k, 1, D]
                h_pre = torch.bmm(x_sel, w1_sel.transpose(1, 2))
                if has_lora_e0:
                    A1_stack = torch.stack([e[0].lora_A for e in self.experts], dim=0)[sel_idx]
                    B1_stack = torch.stack([e[0].lora_B for e in self.experts], dim=0)[sel_idx]
                    s1 = _get_submod_lora(self.experts[0][0])[2]
                    h_pre = h_pre + torch.bmm(torch.bmm(x_sel, A1_stack.transpose(1, 2)) * s1, B1_stack.transpose(1, 2))
                if all_b1:
                    b1_stack = torch.stack([_get_submod_weight_bias(e[0])[1] for e in self.experts], dim=0)
                    h_pre = h_pre + b1_stack[sel_idx].unsqueeze(1)
                h = self.experts[0][1](h_pre)
                y_sel = torch.bmm(h, w2_sel.transpose(1, 2))
                if has_lora_e2:
                    A2_stack = torch.stack([e[2].lora_A for e in self.experts], dim=0)[sel_idx]
                    B2_stack = torch.stack([e[2].lora_B for e in self.experts], dim=0)[sel_idx]
                    s2 = _get_submod_lora(self.experts[0][2])[2]
                    y_sel = y_sel + torch.bmm(torch.bmm(h, A2_stack.transpose(1, 2)) * s2, B2_stack.transpose(1, 2))
                if all_b2:
                    b2_stack = torch.stack([_get_submod_weight_bias(e[2])[1] for e in self.experts], dim=0)
                    y_sel = y_sel + b2_stack[sel_idx].unsqueeze(1)
                weights_sel = flat_vals[0].view(self.top_k, 1, 1).to(y_sel.dtype)
                out = out + (y_sel * weights_sel).sum(dim=0).to(out.dtype)
            else:
                chosen_experts = flat_idx[0].tolist()
                for k_idx, e_idx in enumerate(chosen_experts):
                    w = flat_vals[0, k_idx]
                    out = out + (self.experts[e_idx](flat_x) * w).to(out.dtype)

            if use_shared_stream and shared_out is not None:
                torch.cuda.current_stream().wait_event(shared_event)
                out = out + (self.shared_scale * shared_out).to(out.dtype)
            elif self.shared_expert:
                out = out + (self.shared_scale * self.shared(flat_x)).to(out.dtype)
            return out.reshape(B, T, D)

        if self.training:
            capacity = int(math.ceil(N_tok / self.num_experts * MOE_CAPACITY_MULTIPLIER))
        else:
            capacity = N_tok
        out = torch.zeros_like(flat_x)
        need_stats = bool(self.training and update_stats and not is_checkpoint_recomputing() and not is_capturing)
        accepted_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device) if need_stats else None
        accepted_mask = torch.zeros((N_tok, self.top_k), dtype=torch.bool, device=x.device) if need_stats else None

        if not use_shared_stream and self.shared_expert:
            shared_out = self.shared(flat_x)
            out = out + (self.shared_scale * shared_out).to(out.dtype)

        if can_batch_bmm:
            W1_stack = torch.stack([_get_submod_weight_bias(e[0])[0] for e in self.experts], dim=0)
            W2_stack = torch.stack([_get_submod_weight_bias(e[2])[0] for e in self.experts], dim=0)
            b1_stack = (
                torch.stack([_get_submod_weight_bias(e[0])[1] for e in self.experts], dim=0).unsqueeze(1)
                if all_b1
                else None
            )
            b2_stack = (
                torch.stack([_get_submod_weight_bias(e[2])[1] for e in self.experts], dim=0).unsqueeze(1)
                if all_b2
                else None
            )
            if has_lora_e0:
                A1_stack = torch.stack([e[0].lora_A for e in self.experts], dim=0)
                B1_stack = torch.stack([e[0].lora_B for e in self.experts], dim=0)
                s1 = _get_submod_lora(self.experts[0][0])[2]
            else:
                A1_stack = B1_stack = None
                s1 = 1.0

            if has_lora_e2:
                A2_stack = torch.stack([e[2].lora_A for e in self.experts], dim=0)
                B2_stack = torch.stack([e[2].lora_B for e in self.experts], dim=0)
                s2 = _get_submod_lora(self.experts[0][2])[2]
            else:
                A2_stack = B2_stack = None
                s2 = 1.0

            if is_capturing:
                x_exp = flat_x.unsqueeze(0).expand(self.num_experts, -1, -1)
                h_pre = torch.bmm(x_exp, W1_stack.transpose(1, 2))
                if has_lora_e0:
                    h_pre = h_pre + torch.bmm(torch.bmm(x_exp, A1_stack.transpose(1, 2)) * s1, B1_stack.transpose(1, 2))
                if b1_stack is not None:
                    h_pre = h_pre + b1_stack
                h = self.experts[0][1](h_pre)
                y_all = torch.bmm(h, W2_stack.transpose(1, 2))
                if has_lora_e2:
                    y_all = y_all + torch.bmm(torch.bmm(h, A2_stack.transpose(1, 2)) * s2, B2_stack.transpose(1, 2))
                if b2_stack is not None:
                    y_all = y_all + b2_stack
                routing_weights = torch.zeros(N_tok, self.num_experts, device=x.device, dtype=y_all.dtype)
                routing_weights.scatter_add_(1, flat_idx, flat_vals.to(y_all.dtype))
                out = out + (y_all * routing_weights.t().unsqueeze(-1)).sum(dim=0).to(out.dtype)
                return out.view(B, T, D)

            # Vectorized zero-sync Top-K Capacity Gather + Batched BMM + Scatter:
            # Eliminates all per-expert mask.nonzero() GPU-to-CPU synchronization stalls.
            cap = max(1, min(capacity, N_tok))
            pad_m = (-cap) % 16
            tie_break = torch.arange(N_tok, device=x.device, dtype=torch.float32).unsqueeze(1) * 1e-7
            overflow_total = torch.zeros((), dtype=torch.long, device=x.device) if need_stats else None

            for k in range(self.top_k):
                idx_k = flat_idx[:, k]
                val_k = flat_vals[:, k]
                one_hot_k = F.one_hot(idx_k, num_classes=self.num_experts).bool()  # [N_tok, E]

                scores_ne = torch.where(one_hot_k, val_k.unsqueeze(1).float() + 1.0 - tie_break, float("-inf"))
                _, top_cap_idx = torch.topk(scores_ne, k=cap, dim=0)  # [cap, E]
                top_idx_e = top_cap_idx.t().contiguous()  # [E, cap]
                valid_e = one_hot_k.t().gather(1, top_idx_e)  # [E, cap]

                if need_stats:
                    req_k = one_hot_k.sum(dim=0)
                    acc_k = valid_e.sum(dim=1)
                    accepted_counts = accepted_counts + acc_k
                    overflow_total = overflow_total + (req_k - acc_k).clamp_min(0).sum()
                    acc_ne = torch.zeros(self.num_experts, N_tok, dtype=torch.bool, device=x.device)
                    acc_ne.scatter_(1, top_idx_e, valid_e)
                    accepted_mask[:, k] = acc_ne.any(dim=0)

                gathered_x = flat_x[top_idx_e]  # [E, cap, D]
                if pad_m > 0:
                    gathered_x = F.pad(gathered_x, (0, 0, 0, pad_m))
                h_pre = torch.bmm(gathered_x, W1_stack.transpose(1, 2))
                if has_lora_e0:
                    h_pre = h_pre + torch.bmm(
                        torch.bmm(gathered_x, A1_stack.transpose(1, 2)) * s1, B1_stack.transpose(1, 2)
                    )
                if b1_stack is not None:
                    h_pre = h_pre + b1_stack
                h = self.experts[0][1](h_pre)
                y = torch.bmm(h, W2_stack.transpose(1, 2))
                if has_lora_e2:
                    y = y + torch.bmm(torch.bmm(h, A2_stack.transpose(1, 2)) * s2, B2_stack.transpose(1, 2))
                if b2_stack is not None:
                    y = y + b2_stack
                if pad_m > 0:
                    y = y[:, :cap, :]

                weights_e = (val_k[top_idx_e] * valid_e.to(val_k.dtype)).unsqueeze(-1)
                weighted_y = (y * weights_e).to(out.dtype)
                out = out.index_add(0, top_idx_e.reshape(-1), weighted_y.reshape(-1, D))

            if need_stats and overflow_total is not None:
                if self._overflow_counter.device != overflow_total.device:
                    self._overflow_counter = self._overflow_counter.to(overflow_total.device)
                self._overflow_counter.add_(overflow_total)
        else:
            for k in range(self.top_k):
                idx_k = flat_idx[:, k]
                val_k = flat_vals[:, k]
                for e in range(self.num_experts):
                    mask = idx_k == e
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

        if use_shared_stream and shared_out is not None:
            torch.cuda.current_stream().wait_event(shared_event)
            out = out + (self.shared_scale * shared_out).to(out.dtype)

        return out.reshape(B, T, D)

    @torch.no_grad()
    def _update_routing_stats(self, flat_x, result: RoutingResult):
        """Update non-gradient routing state without retaining an activation graph or syncing GPU to CPU."""
        target = max(1.0, float((flat_x.size(0) * self.top_k) / self.num_experts))

        # DeepSeek-V3 style continuous proportional bias steering:
        # Decouples load-balancing from LM loss gradients while eliminating bang-bang chattering
        counts_f = result.requested_counts.float().to(device=self.expert_bias.device)
        error_ratio = (counts_f - target) / target
        self._pending_bias_update = -error_ratio * MOE_BIAS_UPDATE_RATE

        # Expert utilization ratio reflects actual accepted tokens processed
        batch_ratio = result.accepted_counts.float() / target
        self.expert_load_ratio.mul_(0.9).add_(batch_ratio.to(self.expert_load_ratio.device), alpha=0.1)
        self.last_counts = result.requested_counts.detach()
        self.last_requested_counts = result.requested_counts.detach()
        self.last_accepted_counts = result.accepted_counts.detach()
        self.last_dropped_counts = result.dropped_counts.detach()
        self.last_target = target

        # Vectorized Centroid EMA update across all experts simultaneously (zero nonzero() or .norm() CPU syncs)
        ema_decay = float(getattr(self, "centroid_ema", CENTROID_EMA))
        if (
            self.last_centroids is None
            or self.last_centroids.device != flat_x.device
            or self.last_centroids.shape != (self.num_experts, self.dim)
        ):
            self.last_centroids = torch.zeros(self.num_experts, self.dim, device=flat_x.device, dtype=flat_x.dtype)

        expert_token_mask = torch.zeros(flat_x.size(0), self.num_experts, dtype=torch.bool, device=flat_x.device)
        expert_token_mask.scatter_(1, result.indices, result.accepted_mask)
        mask_f = expert_token_mask.t().to(dtype=flat_x.dtype)  # [E, N_tok]
        counts = mask_f.sum(dim=1, keepdim=True)  # [E, 1]
        batch_centroids = torch.mm(mask_f, flat_x.detach()) / counts.clamp_min(1.0)
        has_tokens = counts > 0
        is_uninit = self.last_centroids.float().norm(dim=-1, keepdim=True) < 1e-6
        blended = torch.where(
            is_uninit,
            batch_centroids,
            self.last_centroids * ema_decay + batch_centroids * (1.0 - ema_decay),
        )
        self.last_centroids.copy_(torch.where(has_tokens, blended, self.last_centroids))

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
        N_tok = B * T
        flat_x = x.detach().reshape(N_tok, D)

        bias_free_logits = self.router(x)
        biased_logits = bias_free_logits + self.expert_bias

        if self.training:
            temp = (1.0 - self._global_step.float() / 5000.0).clamp(min=0.1)
            gate_weights = F.softmax(bias_free_logits / temp, dim=-1)
        else:
            gate_weights = F.softmax(bias_free_logits, dim=-1)

        _, top_idx = torch.topk(biased_logits, self.top_k, dim=-1)
        flat_idx = top_idx.reshape(N_tok, self.top_k)
        flat_vals = torch.gather(gate_weights, dim=-1, index=top_idx).reshape(N_tok, self.top_k)
        if self.top_k > 1:
            flat_vals = flat_vals / (flat_vals.sum(dim=-1, keepdim=True) + 1e-20)

        capacity = int(math.ceil(N_tok / self.num_experts * MOE_CAPACITY_MULTIPLIER))
        cap = max(1, min(capacity, N_tok))
        tie_break = torch.arange(N_tok, device=x.device, dtype=torch.float32).unsqueeze(1) * 1e-7
        accepted_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        accepted_mask = torch.zeros((N_tok, self.top_k), dtype=torch.bool, device=x.device)

        for k in range(self.top_k):
            idx_k = flat_idx[:, k]
            val_k = flat_vals[:, k]
            one_hot_k = F.one_hot(idx_k, num_classes=self.num_experts).bool()
            scores_ne = torch.where(one_hot_k, val_k.unsqueeze(1).float() + 1.0 - tie_break, float("-inf"))
            _, top_cap_idx = torch.topk(scores_ne, k=cap, dim=0)
            top_idx_e = top_cap_idx.t().contiguous()
            valid_e = one_hot_k.t().gather(1, top_idx_e)
            accepted_counts = accepted_counts + valid_e.sum(dim=1)
            acc_ne = torch.zeros(self.num_experts, N_tok, dtype=torch.bool, device=x.device)
            acc_ne.scatter_(1, top_idx_e, valid_e)
            accepted_mask[:, k] = acc_ne.any(dim=0)

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
