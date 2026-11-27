import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lucid_machines.kernels.fast_swiglu import FastGELU
from lucid_machines.ops.centroid_steer import batched_qr

from .config import (
    CENTROID_EMA,
    CENTROID_SUBSPACE_DIRS,
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


def _orthonormalize_subspace_batch(
    raw_dirs: torch.Tensor,
    prev_dirs: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """FP32 sign-preserving sCholQR2 + Householder fallback orthonormalization over [E, D, k_c]."""
    orig_dtype = raw_dirs.dtype
    with torch.amp.autocast(device_type=raw_dirs.device.type, enabled=False):
        V = raw_dirs.float()
        E, D, K = V.shape
        active_expert = (V.norm(dim=(1, 2)) > eps).view(E, 1, 1)

        if K > D:
            # Degenerate tiny-dimension fallback when k_c > D
            Q, _ = torch.linalg.qr(V, mode="reduced")
            if prev_dirs is not None and prev_dirs.shape == Q.shape:
                prev_f = prev_dirs.float()
                prev_act = prev_f.norm(dim=1, keepdim=True) > eps
                dots = (Q * prev_f).sum(dim=1, keepdim=True)
                Q = Q * torch.where(prev_act & (dots < 0.0), -1.0, 1.0)
            Q = torch.where(active_expert, Q, torch.zeros_like(Q))
            return Q.to(dtype=orig_dtype)

        # Seed any inactive experts with identity columns before batched_qr so QR never fails on all-zeros,
        # then mask inactive experts back to zero afterwards.
        eye_k = torch.eye(D, K, device=V.device, dtype=torch.float32).unsqueeze(0).expand(E, -1, -1)
        V_safe = torch.where(active_expert, V, eye_k)
        Q = batched_qr(V_safe, cond_threshold=1e4, eps_shift=1e-7, orth_tol=1e-4, out_dtype=torch.float32)

        if prev_dirs is not None and prev_dirs.shape == Q.shape:
            prev_f = prev_dirs.float()
            prev_act = prev_f.norm(dim=1, keepdim=True) > eps
            dots = (Q * prev_f).sum(dim=1, keepdim=True)
            Q = Q * torch.where(prev_act & (dots < 0.0), -1.0, 1.0)

        Q = torch.where(active_expert, Q, torch.zeros_like(Q))
        return Q.to(dtype=orig_dtype)


class _ZeroCopyPackedStack(torch.autograd.Function):
    """Zero-copy view of a packed [E, d_out, d_in] buffer linked cleanly to individual Parameter grads."""

    @staticmethod
    def forward(ctx, packed_buf: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
        return packed_buf.view_as(packed_buf)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (None, *grad_output.unbind(0))


class _GroupedLinearFn(torch.autograd.Function):
    """Zero-padding grouped GEMM autograd wrapper over torch._grouped_mm."""

    @staticmethod
    def forward(
        ctx,
        x_2d: torch.Tensor,
        weight_3d: torch.Tensor,
        offs: torch.Tensor,
        drop_mask_col: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
        out = torch._grouped_mm(x_c, weight_3d.transpose(1, 2), offs=offs)
        if drop_mask_col is not None:
            out.masked_fill_(drop_mask_col, 0.0)
        ctx.save_for_backward(x_c, weight_3d, offs, drop_mask_col)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_c, weight_3d, offs, drop_mask_col = ctx.saved_tensors
        go = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        if drop_mask_col is not None and go is grad_out:
            go = go.clone()
        if drop_mask_col is not None:
            go.masked_fill_(drop_mask_col, 0.0)
        if ctx.needs_input_grad[0]:
            gx = torch._grouped_mm(go, weight_3d, offs=offs)
            if drop_mask_col is not None:
                gx.masked_fill_(drop_mask_col, 0.0)
        else:
            gx = None
        gw = torch._grouped_mm(go.transpose(0, 1), x_c, offs=offs) if ctx.needs_input_grad[1] else None
        return gx, gw, None, None


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
        use_sparse_eval: bool = True,
        subspace_dirs: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_fp4 = use_fp4
        self.use_fp8 = use_fp8
        self.shared_expert = shared_expert
        self.shared_scale = shared_scale
        self.use_sparse_eval = use_sparse_eval

        Linear = FP8Linear if use_fp8 else (FP4Linear if use_fp4 else nn.Linear)
        hidden_dim = dim * EXPERT_HIDDEN_MULTIPLIER
        self.hidden_dim = hidden_dim
        self.subspace_dirs = max(1, int(CENTROID_SUBSPACE_DIRS if subspace_dirs is None else subspace_dirs))

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
        self.register_buffer("last_hidden_centroids", torch.zeros(num_experts, hidden_dim))
        self.register_buffer("last_contrastive_centroids", torch.zeros(num_experts, dim))
        self.register_buffer("last_hidden_contrastive_centroids", torch.zeros(num_experts, hidden_dim))
        self.register_buffer("last_input_subspace", torch.zeros(num_experts, dim, self.subspace_dirs))
        self.register_buffer("last_hidden_subspace", torch.zeros(num_experts, hidden_dim, self.subspace_dirs))
        self.last_counts = None
        self.last_requested_counts = None
        self.last_accepted_counts = None
        self.last_dropped_counts = None
        self.last_target = None
        self._recorded_centroids_snapshot: torch.Tensor | None = None
        self._recorded_hidden_snapshot: torch.Tensor | None = None
        self._oja_probe_cache: dict[tuple[int, int, int, str, int | None], torch.Tensor] = {}
        self._skip_subspace_stats: bool = False
        self.register_buffer("_overflow_counter", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long))

    def _get_packed_expert_weights(self, w_list: list[torch.Tensor], cache_attr: str) -> torch.Tensor:
        """Return a zero-copy [E, d_out, d_in] view backed by unified contiguous expert Parameter storage."""
        if not w_list:
            raise ValueError("Expected non-empty expert weight list")
        w0 = w_list[0]
        is_compiling = hasattr(torch, "compiler") and torch.compiler.is_compiling()
        if (
            is_compiling
            or w0.is_meta
            or any(
                not isinstance(w, torch.Tensor)
                or type(w) is not nn.Parameter
                or w.is_meta
                or w.is_inference()
                or w.device != w0.device
                or w.dtype != w0.dtype
                or w.shape != w0.shape
                or not w.is_contiguous()
                or hasattr(w, "_fp8_tensor")
                or hasattr(w, "packed_weight")
                for w in w_list
            )
        ):
            return torch.stack(w_list, dim=0)

        num_e = len(w_list)
        target_shape = (num_e, *w0.shape)
        buf: torch.Tensor | None = getattr(self, cache_attr, None)
        step_bytes = w0.numel() * w0.element_size()
        packed_ok = (
            buf is not None
            and buf.shape == target_shape
            and buf.device == w0.device
            and buf.dtype == w0.dtype
            and buf.is_contiguous()
            and step_bytes > 0
            and all(w.data_ptr() == buf.data_ptr() + i * step_bytes for i, w in enumerate(w_list))
        )
        if not packed_ok:
            with torch.no_grad():
                buf = torch.empty(target_shape, device=w0.device, dtype=w0.dtype)
                for i, w in enumerate(w_list):
                    buf[i].copy_(w.data)
                    w.data = buf[i]
                setattr(self, cache_attr, buf)
        if torch.is_grad_enabled() and any(w.requires_grad for w in w_list):
            return _ZeroCopyPackedStack.apply(buf, *w_list)
        return buf

    def _get_oja_probe(self, d_in: int, k_dirs: int, device: torch.device) -> torch.Tensor:
        """Return deterministic orthonormal seed probe [E, d_in, k_dirs] in FP32 without per-step overhead."""
        cache = getattr(self, "_oja_probe_cache", None)
        if cache is None:
            cache = {}
            self._oja_probe_cache = cache
        key = (self.num_experts, int(d_in), int(k_dirs), device.type, device.index)
        probe = cache.get(key)
        if probe is None:
            gen = torch.Generator(device="cpu").manual_seed(1337 + int(d_in) * 17 + int(k_dirs))
            raw = torch.randn(self.num_experts, int(d_in), int(k_dirs), generator=gen, dtype=torch.float32)
            probe = _orthonormalize_subspace_batch(raw).to(device=device)
            cache[key] = probe
        return probe

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for buf_name in (
            "last_hidden_centroids",
            "last_contrastive_centroids",
            "last_hidden_contrastive_centroids",
            "last_input_subspace",
            "last_hidden_subspace",
        ):
            key = prefix + buf_name
            if key not in state_dict and buf_name in self._buffers and self._buffers[buf_name] is not None:
                state_dict[key] = self._buffers[buf_name].clone()
            elif key in state_dict and buf_name in ("last_input_subspace", "last_hidden_subspace"):
                loaded = state_dict[key]
                target = self._buffers.get(buf_name)
                if target is not None and loaded.ndim == 3 and loaded.shape[:2] == target.shape[:2]:
                    if loaded.shape[2] > target.shape[2]:
                        state_dict[key] = loaded[:, :, : target.shape[2]].contiguous()
                    elif loaded.shape[2] < target.shape[2]:
                        pad_k = target.shape[2] - loaded.shape[2]
                        padded = torch.cat(
                            [
                                loaded,
                                torch.zeros(
                                    loaded.shape[0], loaded.shape[1], pad_k, dtype=loaded.dtype, device=loaded.device
                                ),
                            ],
                            dim=-1,
                        )
                        state_dict[key] = _orthonormalize_subspace_batch(padded)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if getattr(self, "last_centroids", None) is not None:
            self._recorded_centroids_snapshot = self.last_centroids.detach().clone()
        if getattr(self, "last_hidden_centroids", None) is not None:
            self._recorded_hidden_snapshot = self.last_hidden_centroids.detach().clone()

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
        N_tok = flat_x.size(0)
        logits = self.router(flat_x)
        _, top_idx = torch.topk(logits + self.expert_bias, self.top_k, dim=-1)
        weights = torch.gather(F.softmax(logits, dim=-1), 1, top_idx)
        if self.top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

        if (
            N_tok == 1
            and not self.use_fp4
            and not self.use_fp8
            and all(
                not hasattr(e[0], "lora_A")
                and not hasattr(e[2], "lora_A")
                and not hasattr(e[0], "packed_weight")
                and not hasattr(e[2], "packed_weight")
                and getattr(e[0], "weight", None) is not None
                and getattr(e[2], "weight", None) is not None
                for e in self.experts
            )
        ):
            sel_idx = top_idx[0]
            w1_sel = torch.stack([e[0].weight for e in self.experts], dim=0)[sel_idx]
            w2_sel = torch.stack([e[2].weight for e in self.experts], dim=0)[sel_idx]
            x_sel = flat_x.unsqueeze(0).expand(self.top_k, 1, D)
            h_pre = torch.bmm(x_sel, w1_sel.transpose(1, 2))
            if self.experts[0][0].bias is not None and all(e[0].bias is not None for e in self.experts):
                b1_sel = torch.stack([e[0].bias for e in self.experts], dim=0)[sel_idx].unsqueeze(1)
                h_pre = h_pre + b1_sel
            h = self.experts[0][1](h_pre)
            y_sel = torch.bmm(h, w2_sel.transpose(1, 2))
            if self.experts[0][2].bias is not None and all(e[2].bias is not None for e in self.experts):
                b2_sel = torch.stack([e[2].bias for e in self.experts], dim=0)[sel_idx].unsqueeze(1)
                y_sel = y_sel + b2_sel
            out = (y_sel * weights[0].view(self.top_k, 1, 1).to(y_sel.dtype)).sum(dim=0).to(flat_x.dtype)
            if self.shared_expert:
                out = out + (self.shared_scale * self.shared(flat_x)).to(out.dtype)
            return out.reshape(B, T, D)

        expert_ids = top_idx.reshape(-1)
        order = expert_ids.argsort(stable=True)
        token_ids = torch.arange(N_tok, device=x.device).repeat_interleave(self.top_k)[order]
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
        is_capturing = bool(x.is_cuda and torch.cuda.is_current_stream_capturing())
        if (
            not self.training
            and not is_capturing
            and getattr(self, "use_sparse_eval", True)
            and (x.shape[0] * x.shape[1] > 1)
        ):
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
            and not any(hasattr(getattr(m, "weight", None), "_fp8_tensor") for m in self.experts.modules())
            and not any(hasattr(m, "packed_weight") for m in self.experts.modules())
            and not mixed_biases
            and (has_lora_e0 or no_lora_e0)
            and (has_lora_e2 or no_lora_e2)
            and all(
                _get_submod_weight_bias(e[0])[0] is not None and _get_submod_weight_bias(e[2])[0] is not None
                for e in self.experts
            )
        )

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
        if (
            not self.training
            and N_tok == 1
            and not any(isinstance(m, FP4Linear) or hasattr(m, "packed_weight") for m in self.experts.modules())
        ):
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
        need_subspace = bool(need_stats and not getattr(self, "_skip_subspace_stats", False))
        accepted_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device) if need_stats else None
        accepted_mask = torch.zeros((N_tok, self.top_k), dtype=torch.bool, device=x.device) if need_stats else None
        k_dirs = int(getattr(self, "subspace_dirs", CENTROID_SUBSPACE_DIRS))
        h_dim = int(getattr(self, "hidden_dim", D * EXPERT_HIDDEN_MULTIPLIER))
        batch_in_oja = None
        batch_hid_oja = None
        batch_hid_sum = None
        Q_in_cur = None
        Q_hid_cur = None
        if need_subspace:
            with torch.no_grad(), torch.amp.autocast(device_type=x.device.type, enabled=False):
                batch_in_oja = torch.zeros(self.num_experts, D, k_dirs, device=x.device, dtype=torch.float32)
                batch_hid_oja = torch.zeros(self.num_experts, h_dim, k_dirs, device=x.device, dtype=torch.float32)
                batch_hid_sum = torch.zeros(self.num_experts, h_dim, device=x.device, dtype=torch.float32)

                probe_in = self._get_oja_probe(D, k_dirs, x.device)
                if (
                    getattr(self, "last_input_subspace", None) is not None
                    and self.last_input_subspace.device == x.device
                    and self.last_input_subspace.shape == (self.num_experts, D, k_dirs)
                ):
                    sub_in_f = self.last_input_subspace.float()
                    uninit_in = (sub_in_f.norm(dim=(1, 2)) < 1e-6).view(self.num_experts, 1, 1)
                    Q_in_cur = torch.where(uninit_in, probe_in, sub_in_f)
                else:
                    Q_in_cur = probe_in

                probe_hid = self._get_oja_probe(h_dim, k_dirs, x.device)
                if (
                    getattr(self, "last_hidden_subspace", None) is not None
                    and self.last_hidden_subspace.device == x.device
                    and self.last_hidden_subspace.shape == (self.num_experts, h_dim, k_dirs)
                ):
                    sub_hid_f = self.last_hidden_subspace.float()
                    uninit_hid = (sub_hid_f.norm(dim=(1, 2)) < 1e-6).view(self.num_experts, 1, 1)
                    Q_hid_cur = torch.where(uninit_hid, probe_hid, sub_hid_f)
                else:
                    Q_hid_cur = probe_hid

        if not use_shared_stream and self.shared_expert:
            shared_out = self.shared(flat_x)
            out = out + (self.shared_scale * shared_out).to(out.dtype)

        if can_batch_bmm:
            nograd_stacks = None
            if not torch.is_grad_enabled() and no_lora_e0 and no_lora_e2:
                w_pairs = [(_get_submod_weight_bias(e[0])[0], _get_submod_weight_bias(e[2])[0]) for e in self.experts]
                cache_sig = (
                    x.device,
                    w_pairs[0][0].dtype,
                    tuple((w1.data_ptr(), w1._version, w2.data_ptr(), w2._version) for w1, w2 in w_pairs),
                )
                cached = getattr(self, "_nograd_weight_stack_cache", None)
                if cached is not None and cached[0] == cache_sig:
                    nograd_stacks = cached[1]
                else:
                    W1_c = torch.stack([w1 for w1, _ in w_pairs], dim=0)
                    W2_c = torch.stack([w2 for _, w2 in w_pairs], dim=0)
                    b1_c = (
                        torch.stack([_get_submod_weight_bias(e[0])[1] for e in self.experts], dim=0).unsqueeze(1)
                        if all_b1
                        else None
                    )
                    b2_c = (
                        torch.stack([_get_submod_weight_bias(e[2])[1] for e in self.experts], dim=0).unsqueeze(1)
                        if all_b2
                        else None
                    )
                    nograd_stacks = (W1_c, W2_c, b1_c, b2_c)
                    self._nograd_weight_stack_cache = (cache_sig, nograd_stacks)
            elif getattr(self, "_nograd_weight_stack_cache", None) is not None:
                self._nograd_weight_stack_cache = None

            if nograd_stacks is not None:
                W1_stack, W2_stack, b1_stack, b2_stack = nograd_stacks
            else:
                w1_list = [_get_submod_weight_bias(e[0])[0] for e in self.experts]
                w2_list = [_get_submod_weight_bias(e[2])[0] for e in self.experts]
                W1_stack = self._get_packed_expert_weights(w1_list, "_w1_packed_storage")
                W2_stack = self._get_packed_expert_weights(w2_list, "_w2_packed_storage")
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

            # Vectorized zero-sync Top-K Capacity Gather + Grouped/Batched GEMM + Scatter:
            # Uses torch._grouped_mm when eligible (zero padding overhead) with automatic packed BMM fallback.
            cap = max(1, min(capacity, N_tok))
            is_compiling_moe = hasattr(torch, "compiler") and torch.compiler.is_compiling()
            use_grouped_mm = bool(
                hasattr(torch, "_grouped_mm")
                and x.is_cuda
                and x.dtype in (torch.bfloat16, torch.float16)
                and W1_stack.dtype == x.dtype
                and W2_stack.dtype == x.dtype
                and (D % 16 == 0)
                and (h_dim % 16 == 0)
                and no_lora_e0
                and no_lora_e2
                and not is_compiling_moe
            )
            overflow_total = torch.zeros((), dtype=torch.long, device=x.device) if need_stats else None

            if use_grouped_mm:
                try:
                    b1_2d = b1_stack.squeeze(1).to(x.dtype) if b1_stack is not None else None
                    b2_2d = b2_stack.squeeze(1).to(x.dtype) if b2_stack is not None else None
                    act_fn = self.experts[0][1]
                    for k in range(self.top_k):
                        idx_k = flat_idx[:, k]
                        val_k = flat_vals[:, k]
                        order_score = torch.argsort(val_k.float(), descending=True, stable=True)
                        exp_sort = torch.argsort(idx_k[order_score], stable=True)
                        perm = order_score[exp_sort]
                        sorted_exp = idx_k[perm]
                        counts = torch.bincount(sorted_exp, minlength=self.num_experts)
                        starts = counts.cumsum(0) - counts
                        pos_in_exp = torch.arange(N_tok, device=x.device) - starts[sorted_exp]
                        keep = pos_in_exp < cap
                        keep_sort = (~keep).to(torch.int8).argsort(stable=True)
                        perm_k = perm[keep_sort]
                        exp_k = idx_k[perm_k]
                        kept_mask = keep[keep_sort]
                        drop_mask_col = (~kept_mask).unsqueeze(1)
                        kept_counts = counts.clamp(max=cap)
                        offs = kept_counts.cumsum(0).to(torch.int32)

                        if need_stats:
                            accepted_counts = accepted_counts + kept_counts
                            overflow_total = overflow_total + (counts - kept_counts).clamp_min(0).sum()
                            accepted_mask[perm_k, k] = kept_mask

                        x_g = flat_x[perm_k]
                        c_list = None
                        if need_subspace and batch_in_oja is not None:
                            with torch.no_grad(), torch.amp.autocast(device_type=x.device.type, enabled=False):
                                c_list = kept_counts.tolist()
                                x_f = x_g.detach().float()
                                pos = 0
                                for e_i, cnt in enumerate(c_list):
                                    if cnt > 0:
                                        x_e = x_f[pos : pos + cnt]
                                        batch_in_oja[e_i].add_(torch.mm(x_e.t(), torch.mm(x_e, Q_in_cur[e_i])))
                                        pos += cnt

                        h_pre = _GroupedLinearFn.apply(x_g, W1_stack, offs, drop_mask_col)
                        if b1_2d is not None:
                            h_pre = h_pre + b1_2d[exp_k]
                        h = act_fn(h_pre)
                        if need_subspace and batch_hid_oja is not None and batch_hid_sum is not None:
                            with torch.no_grad(), torch.amp.autocast(device_type=x.device.type, enabled=False):
                                if c_list is None:
                                    c_list = kept_counts.tolist()
                                h_f = h.detach().float()
                                pos = 0
                                for e_i, cnt in enumerate(c_list):
                                    if cnt > 0:
                                        h_e = h_f[pos : pos + cnt]
                                        batch_hid_sum[e_i].add_(h_e.sum(dim=0))
                                        batch_hid_oja[e_i].add_(torch.mm(h_e.t(), torch.mm(h_e, Q_hid_cur[e_i])))
                                        pos += cnt

                        y = _GroupedLinearFn.apply(h, W2_stack, offs, drop_mask_col)
                        if b2_2d is not None:
                            y = y + b2_2d[exp_k]

                        w_k = torch.where(kept_mask, val_k[perm_k], val_k.new_zeros(()))
                        weighted_y = (y * w_k.unsqueeze(-1)).to(out.dtype)
                        out = out.index_add(0, perm_k, weighted_y)
                except (RuntimeError, NotImplementedError):
                    use_grouped_mm = False
                    out = torch.zeros_like(flat_x)
                    if need_shared_out := (not use_shared_stream and shared_out is not None):
                        out = out + (self.shared_scale * shared_out).to(out.dtype)
                    if need_stats:
                        accepted_counts.zero_()
                        accepted_mask.zero_()
                        overflow_total.zero_()

            if not use_grouped_mm:
                pad_m = (-cap) % 16
                tie_break = torch.arange(N_tok, device=x.device, dtype=torch.float32).unsqueeze(1) * 1e-7

                for k in range(self.top_k):
                    idx_k = flat_idx[:, k]
                    val_k = flat_vals[:, k]
                    one_hot_k = F.one_hot(idx_k, num_classes=self.num_experts).bool()  # [N_tok, E]

                    scores_ne = torch.where(one_hot_k, val_k.unsqueeze(1).float() + 1.0 - tie_break, float("-inf"))
                    _, top_cap_idx = torch.topk(scores_ne, k=cap, dim=0)  # [cap, E]
                    top_idx_e = top_cap_idx.t().contiguous()  # [E, cap]
                    valid_e = one_hot_k.t().gather(1, top_idx_e)  # [E, cap]

                    if need_stats:
                        with torch.no_grad():
                            req_k = one_hot_k.sum(dim=0)
                            acc_k = valid_e.sum(dim=1)
                            accepted_counts = accepted_counts + acc_k
                            overflow_total = overflow_total + (req_k - acc_k).clamp_min(0).sum()
                            acc_ne = torch.zeros(self.num_experts, N_tok, dtype=torch.bool, device=x.device)
                            acc_ne.scatter_(1, top_idx_e, valid_e)
                            accepted_mask[:, k] = acc_ne.any(dim=0)

                    gathered_x = flat_x[top_idx_e]  # [E, cap, D]
                    if need_subspace and batch_in_oja is not None:
                        with torch.no_grad(), torch.amp.autocast(device_type=x.device.type, enabled=False):
                            m0 = valid_e.to(dtype=torch.float32)  # [E, cap]
                            x_m = gathered_x.detach().float() * m0.unsqueeze(-1)
                            z_in = torch.bmm(x_m, Q_in_cur)  # [E, cap, k_dirs]
                            batch_in_oja.add_(torch.bmm(x_m.transpose(1, 2), z_in))

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
                    if need_subspace and batch_hid_oja is not None and batch_hid_sum is not None:
                        with torch.no_grad(), torch.amp.autocast(device_type=x.device.type, enabled=False):
                            h_cap = (h[:, :cap, :] if pad_m > 0 else h).detach().float()
                            m0 = valid_e.to(dtype=torch.float32)
                            h_m = h_cap * m0.unsqueeze(-1)
                            batch_hid_sum.add_(h_m.sum(dim=1))
                            z_hid = torch.bmm(h_m, Q_hid_cur)  # [E, cap, k_dirs]
                            batch_hid_oja.add_(torch.bmm(h_m.transpose(1, 2), z_hid))
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

                    if need_subspace and batch_in_oja is not None:
                        with torch.no_grad(), torch.amp.autocast(device_type=x.device.type, enabled=False):
                            x_m_e = expert_input.detach().float()
                            z_in_e = torch.mm(x_m_e, Q_in_cur[e])
                            batch_in_oja[e].add_(torch.mm(x_m_e.t(), z_in_e))

                    if pad:
                        expert_input = F.pad(expert_input, (0, 0, 0, pad))

                    if (
                        need_subspace
                        and batch_hid_oja is not None
                        and batch_hid_sum is not None
                        and len(self.experts[e]) >= 3
                    ):
                        h_e = self.experts[e][1](self.experts[e][0](expert_input))
                        with torch.no_grad(), torch.amp.autocast(device_type=x.device.type, enabled=False):
                            h_valid = (h_e[:orig_tokens] if pad else h_e).detach().float()
                            batch_hid_sum[e].add_(h_valid.sum(dim=0))
                            z_hid_e = torch.mm(h_valid, Q_hid_cur[e])
                            batch_hid_oja[e].add_(torch.mm(h_valid.t(), z_hid_e))
                        expert_output = self.experts[e][2](h_e)
                    else:
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
            self._update_routing_stats(
                flat_x,
                routing_result,
                batch_in_oja=batch_in_oja,
                batch_hid_oja=batch_hid_oja,
                batch_hid_sum=batch_hid_sum,
                q_in_cur=Q_in_cur,
                q_hid_cur=Q_hid_cur,
                update_subspace=need_subspace,
            )

        if self.training and not getattr(self, "_step_managed_by_engine", False) and not is_checkpoint_recomputing():
            self._global_step += 1

        if use_shared_stream and shared_out is not None:
            torch.cuda.current_stream().wait_event(shared_event)
            out = out + (self.shared_scale * shared_out).to(out.dtype)

        return out.reshape(B, T, D)

    @torch.no_grad()
    def _update_routing_stats(
        self,
        flat_x,
        result: RoutingResult,
        batch_in_oja: torch.Tensor | None = None,
        batch_hid_oja: torch.Tensor | None = None,
        batch_hid_sum: torch.Tensor | None = None,
        q_in_cur: torch.Tensor | None = None,
        q_hid_cur: torch.Tensor | None = None,
        update_subspace: bool = True,
    ):
        """Update non-gradient routing state and Block-Oja activation subspaces without CPU-GPU syncs."""
        with torch.amp.autocast(device_type=flat_x.device.type, enabled=False):
            return self._update_routing_stats_impl(
                flat_x,
                result,
                batch_in_oja=batch_in_oja,
                batch_hid_oja=batch_hid_oja,
                batch_hid_sum=batch_hid_sum,
                q_in_cur=q_in_cur,
                q_hid_cur=q_hid_cur,
                update_subspace=update_subspace,
            )

    @torch.no_grad()
    def _update_routing_stats_impl(
        self,
        flat_x,
        result: RoutingResult,
        batch_in_oja: torch.Tensor | None = None,
        batch_hid_oja: torch.Tensor | None = None,
        batch_hid_sum: torch.Tensor | None = None,
        q_in_cur: torch.Tensor | None = None,
        q_hid_cur: torch.Tensor | None = None,
        update_subspace: bool = True,
    ):
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
        if not update_subspace:
            return

        # Vectorized Centroid & Block-Oja Subspace EMA update across all experts simultaneously
        ema_decay = float(getattr(self, "centroid_ema", CENTROID_EMA))
        k_dirs = int(getattr(self, "subspace_dirs", CENTROID_SUBSPACE_DIRS))
        h_dim = int(getattr(self, "hidden_dim", self.dim * EXPERT_HIDDEN_MULTIPLIER))
        dev = flat_x.device
        dt = flat_x.dtype

        if (
            self.last_centroids is None
            or self.last_centroids.device != dev
            or self.last_centroids.shape != (self.num_experts, self.dim)
        ):
            self.last_centroids = torch.zeros(self.num_experts, self.dim, device=dev, dtype=dt)
        if (
            getattr(self, "last_contrastive_centroids", None) is None
            or self.last_contrastive_centroids.device != dev
            or self.last_contrastive_centroids.shape != (self.num_experts, self.dim)
        ):
            self.last_contrastive_centroids = torch.zeros(self.num_experts, self.dim, device=dev, dtype=dt)
        if (
            getattr(self, "last_input_subspace", None) is None
            or self.last_input_subspace.device != dev
            or self.last_input_subspace.shape != (self.num_experts, self.dim, k_dirs)
        ):
            self.last_input_subspace = torch.zeros(self.num_experts, self.dim, k_dirs, device=dev, dtype=dt)
        if (
            getattr(self, "last_hidden_centroids", None) is None
            or self.last_hidden_centroids.device != dev
            or self.last_hidden_centroids.shape != (self.num_experts, h_dim)
        ):
            self.last_hidden_centroids = torch.zeros(self.num_experts, h_dim, device=dev, dtype=dt)
        if (
            getattr(self, "last_hidden_contrastive_centroids", None) is None
            or self.last_hidden_contrastive_centroids.device != dev
            or self.last_hidden_contrastive_centroids.shape != (self.num_experts, h_dim)
        ):
            self.last_hidden_contrastive_centroids = torch.zeros(self.num_experts, h_dim, device=dev, dtype=dt)
        if (
            getattr(self, "last_hidden_subspace", None) is None
            or self.last_hidden_subspace.device != dev
            or self.last_hidden_subspace.shape != (self.num_experts, h_dim, k_dirs)
        ):
            self.last_hidden_subspace = torch.zeros(self.num_experts, h_dim, k_dirs, device=dev, dtype=dt)

        n_tok = flat_x.size(0)
        x_det = flat_x.detach()
        expert_token_mask = torch.zeros(n_tok, self.num_experts, dtype=torch.bool, device=dev)
        expert_token_mask.scatter_(1, result.indices, result.accepted_mask)
        mask_f = expert_token_mask.t().to(dtype=torch.float32)  # [E, N_tok]
        counts = mask_f.sum(dim=1, keepdim=True)  # [E, 1]
        safe_counts = counts.clamp_min(1.0)
        has_tokens = counts > 0

        # Pre-update global reference means (for standalone update_routing_stats fallback)
        if q_in_cur is None:
            sub_in_f = self.last_input_subspace.float()
            uninit_in = (sub_in_f.norm(dim=(1, 2)) < 1e-6).view(self.num_experts, 1, 1)
            q_in_cur = torch.where(uninit_in, self._get_oja_probe(self.dim, k_dirs, dev), sub_in_f)
        if q_hid_cur is None:
            sub_hid_f = self.last_hidden_subspace.float()
            uninit_hid = (sub_hid_f.norm(dim=(1, 2)) < 1e-6).view(self.num_experts, 1, 1)
            q_hid_cur = torch.where(uninit_hid, self._get_oja_probe(h_dim, k_dirs, dev), sub_hid_f)

        if batch_in_oja is None:
            batch_in_oja = torch.zeros(self.num_experts, self.dim, k_dirs, device=dev, dtype=torch.float32)
            x_det_f = x_det.float()
            for e_i in range(self.num_experts):
                w_row = mask_f[e_i].unsqueeze(-1)
                z_e = torch.mm(x_det_f, q_in_cur[e_i])
                batch_in_oja[e_i] = torch.mm((x_det_f * w_row).t(), z_e)

        batch_centroids = (torch.mm(mask_f, x_det.float()) / safe_counts).to(dtype=dt)
        is_uninit = self.last_centroids.float().norm(dim=-1, keepdim=True) < 1e-6
        blended = torch.where(
            is_uninit,
            batch_centroids,
            self.last_centroids * ema_decay + batch_centroids * (1.0 - ema_decay),
        )
        self.last_centroids.copy_(torch.where(has_tokens, blended, self.last_centroids))

        # Contrastive input centroids: delta_mu_e = mu_e - mean_{j in active}(mu_j)
        act_in = (self.last_centroids.float().norm(dim=-1, keepdim=True) > 1e-6).to(dt)
        mean_in = (self.last_centroids * act_in).sum(dim=0, keepdim=True) / act_in.sum().clamp_min(1.0)
        contr_in = (self.last_centroids - mean_in) if self.num_experts > 1 else self.last_centroids
        contr_in = torch.where(contr_in.float().norm(dim=-1, keepdim=True) > 1e-6, contr_in, self.last_centroids)
        self.last_contrastive_centroids.copy_(contr_in)

        # Matrix-free Block-Oja + Contrastive Anchor for input subspace [E, D, k_dirs]
        sub_ema = min(ema_decay, 0.65)
        y_cov_in = batch_in_oja / safe_counts.unsqueeze(-1)
        delta_in = self.last_contrastive_centroids.float().unsqueeze(-1)  # [E, D, 1]
        delta_in_hat = delta_in / delta_in.norm(dim=1, keepdim=True).clamp_min(1e-8)
        s_in = (y_cov_in.norm(dim=(1, 2), keepdim=True) / math.sqrt(float(k_dirs))).clamp_min(1e-8)
        y_hat_in = y_cov_in / s_in
        y_batch_in = y_hat_in + 0.15 * torch.bmm(delta_in_hat, torch.bmm(delta_in_hat.transpose(1, 2), q_in_cur))
        is_uninit_sub_in = (self.last_input_subspace.float().norm(dim=(1, 2)) < 1e-6).view(self.num_experts, 1, 1)
        y_init_in = y_batch_in.clone()
        y_init_in[:, :, 0] = y_init_in[:, :, 0] + 0.25 * delta_in_hat.squeeze(-1)
        u_in = torch.where(
            is_uninit_sub_in,
            y_init_in + 1e-4 * q_in_cur,
            sub_ema * q_in_cur + (1.0 - sub_ema) * y_batch_in,
        )
        ortho_in = _orthonormalize_subspace_batch(u_in, prev_dirs=q_in_cur).to(dtype=dt)
        self.last_input_subspace.copy_(torch.where(has_tokens.unsqueeze(-1), ortho_in, self.last_input_subspace))

        # True post-activation hidden centroids and Block-Oja hidden subspace [E, H, k_dirs]
        if batch_hid_sum is None or batch_hid_oja is None:
            batch_hid_sum = torch.zeros(self.num_experts, h_dim, device=dev, dtype=torch.float32)
            batch_hid_oja = torch.zeros(self.num_experts, h_dim, k_dirs, device=dev, dtype=torch.float32)
            can_stack_w1 = all(
                getattr(e[0], "weight", None) is not None and not hasattr(e[0], "packed_weight") for e in self.experts
            )
            if can_stack_w1:
                W1_s = torch.stack([e[0].weight for e in self.experts], dim=0)
                b1_s = (
                    torch.stack([e[0].bias for e in self.experts], dim=0).unsqueeze(1)
                    if all(getattr(e[0], "bias", None) is not None for e in self.experts)
                    else None
                )
                cap = max(1, min(int(math.ceil(n_tok / self.num_experts * MOE_CAPACITY_MULTIPLIER)), n_tok))
                tie_break = torch.arange(n_tok, device=dev, dtype=torch.float32).unsqueeze(1) * 1e-7
                for k in range(self.top_k):
                    idx_k = result.indices[:, k]
                    val_k = result.weights[:, k]
                    one_hot_k = F.one_hot(idx_k, num_classes=self.num_experts).bool()
                    scores_ne = torch.where(one_hot_k, val_k.unsqueeze(1).float() + 1.0 - tie_break, float("-inf"))
                    _, top_cap_idx = torch.topk(scores_ne, k=cap, dim=0)
                    top_idx_e = top_cap_idx.t().contiguous()
                    valid_e = one_hot_k.t().gather(1, top_idx_e)
                    gx = x_det[top_idx_e]
                    hp = torch.bmm(gx, W1_s.transpose(1, 2))
                    if b1_s is not None:
                        hp = hp + b1_s
                    h_c_raw = self.experts[0][1](hp).detach().float()
                    m0 = valid_e.to(dtype=torch.float32)
                    h_m = h_c_raw * m0.unsqueeze(-1)
                    batch_hid_sum.add_(h_m.sum(dim=1))
                    z_hid = torch.bmm(h_m, q_hid_cur)
                    batch_hid_oja.add_(torch.bmm(h_m.transpose(1, 2), z_hid))

        batch_h_centroids = (batch_hid_sum / safe_counts).to(dtype=dt)
        h_uninit = self.last_hidden_centroids.float().norm(dim=-1, keepdim=True) < 1e-6
        h_blended = torch.where(
            h_uninit,
            batch_h_centroids,
            self.last_hidden_centroids * ema_decay + batch_h_centroids * (1.0 - ema_decay),
        )
        self.last_hidden_centroids.copy_(torch.where(has_tokens, h_blended, self.last_hidden_centroids))

        act_hid = (self.last_hidden_centroids.float().norm(dim=-1, keepdim=True) > 1e-6).to(dt)
        mean_hid = (self.last_hidden_centroids * act_hid).sum(dim=0, keepdim=True) / act_hid.sum().clamp_min(1.0)
        contr_hid = (self.last_hidden_centroids - mean_hid) if self.num_experts > 1 else self.last_hidden_centroids
        contr_hid = torch.where(
            contr_hid.float().norm(dim=-1, keepdim=True) > 1e-6, contr_hid, self.last_hidden_centroids
        )
        self.last_hidden_contrastive_centroids.copy_(contr_hid)

        # Matrix-free Block-Oja + Contrastive Anchor for hidden subspace [E, H, k_dirs]
        y_cov_hid = batch_hid_oja / safe_counts.unsqueeze(-1)
        delta_hid = self.last_hidden_contrastive_centroids.float().unsqueeze(-1)  # [E, H, 1]
        delta_hid_hat = delta_hid / delta_hid.norm(dim=1, keepdim=True).clamp_min(1e-8)
        s_hid = (y_cov_hid.norm(dim=(1, 2), keepdim=True) / math.sqrt(float(k_dirs))).clamp_min(1e-8)
        y_hat_hid = y_cov_hid / s_hid
        y_batch_hid = y_hat_hid + 0.15 * torch.bmm(delta_hid_hat, torch.bmm(delta_hid_hat.transpose(1, 2), q_hid_cur))
        is_uninit_sub_hid = (self.last_hidden_subspace.float().norm(dim=(1, 2)) < 1e-6).view(self.num_experts, 1, 1)
        y_init_hid = y_batch_hid.clone()
        y_init_hid[:, :, 0] = y_init_hid[:, :, 0] + 0.25 * delta_hid_hat.squeeze(-1)
        u_hid = torch.where(
            is_uninit_sub_hid,
            y_init_hid + 1e-4 * q_hid_cur,
            sub_ema * q_hid_cur + (1.0 - sub_ema) * y_batch_hid,
        )
        ortho_hid = _orthonormalize_subspace_batch(u_hid, prev_dirs=q_hid_cur).to(dtype=dt)
        self.last_hidden_subspace.copy_(torch.where(has_tokens.unsqueeze(-1), ortho_hid, self.last_hidden_subspace))
        self._recorded_centroids_snapshot = self.last_centroids.detach().clone()
        self._recorded_hidden_snapshot = self.last_hidden_centroids.detach().clone()

    @torch.no_grad()
    def get_batch_steering_subspaces(
        self,
        steer_role: str = "input",
        num_dirs: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, list[bool]] | None:
        """Return batched subspaces [E, D, k_eff] and per-expert validity flags without per-expert .item() syncs."""
        if self.last_centroids is None or num_dirs < 1:
            return None
        centroids_overridden = (
            self._recorded_centroids_snapshot is None
            or self._recorded_centroids_snapshot.shape != self.last_centroids.shape
            or self._recorded_centroids_snapshot.device != self.last_centroids.device
            or not torch.equal(self.last_centroids, self._recorded_centroids_snapshot)
        )
        if centroids_overridden:
            return None
        if steer_role == "hidden":
            h_cents = getattr(self, "last_hidden_centroids", None)
            if h_cents is None:
                return None
            hidden_overridden = (
                self._recorded_hidden_snapshot is None
                or self._recorded_hidden_snapshot.shape != h_cents.shape
                or self._recorded_hidden_snapshot.device != h_cents.device
                or not torch.equal(h_cents, self._recorded_hidden_snapshot)
            )
            if hidden_overridden:
                return None
            sub = getattr(self, "last_hidden_subspace", None)
        else:
            sub = getattr(self, "last_input_subspace", None)
        if sub is None:
            return None

        k_eff = min(num_dirs, sub.size(-1))
        sub_k = sub[:, :, :k_eff]
        valid_flags = (sub_k.float().norm(dim=(1, 2)) > 1e-8).tolist()
        dev = device or sub_k.device
        dt = dtype or sub_k.dtype
        return sub_k.to(device=dev, dtype=dt), valid_flags

    @torch.no_grad()
    def get_expert_steering_directions(
        self,
        expert_idx: int,
        steer_role: str = "input",
        num_dirs: int = 4,
        steer_mode: str = "upgraded",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (primary_centroid [in_dim], multi_dir_subspace [in_dim, k_c] | None) for expert_idx."""
        if self.last_centroids is None or expert_idx is None or expert_idx >= self.last_centroids.size(0):
            return None, None
        dev = device or self.last_centroids.device
        dt = dtype or self.last_centroids.dtype

        if steer_mode == "legacy":
            c_e = self.last_centroids[expert_idx].to(device=dev, dtype=dt)
            if steer_role == "hidden":
                if expert_idx < len(self.experts) and len(self.experts[expert_idx]) >= 2:
                    exp = self.experts[expert_idx]
                    h = exp[0](c_e.unsqueeze(0).to(device=self.last_centroids.device, dtype=self.last_centroids.dtype))
                    if len(exp) > 2:
                        h = exp[1](h)
                    return h.squeeze(0).to(device=dev, dtype=dt), None
                return None, None
            return c_e, None

        centroids_overridden = (
            self._recorded_centroids_snapshot is None
            or self._recorded_centroids_snapshot.shape != self.last_centroids.shape
            or self._recorded_centroids_snapshot.device != self.last_centroids.device
            or not torch.equal(self.last_centroids, self._recorded_centroids_snapshot)
        )
        hidden_overridden = (
            getattr(self, "last_hidden_centroids", None) is not None
            and (
                self._recorded_hidden_snapshot is None
                or self._recorded_hidden_snapshot.shape != self.last_hidden_centroids.shape
                or self._recorded_hidden_snapshot.device != self.last_hidden_centroids.device
                or not torch.equal(self.last_hidden_centroids, self._recorded_hidden_snapshot)
            )
            and float(self.last_hidden_centroids[expert_idx].float().norm().item()) > 1e-8
        )

        if steer_role == "hidden":
            if hidden_overridden:
                h_c = self.last_hidden_centroids[expert_idx].to(device=dev, dtype=dt)
                h_u = F.normalize(h_c.float(), dim=0).to(dtype=dt).unsqueeze(-1)
                return h_c, h_u
            if not centroids_overridden and getattr(self, "last_hidden_centroids", None) is not None:
                h_raw = self.last_hidden_centroids[expert_idx]
                if float(h_raw.float().norm().item()) > 1e-8:
                    h_contr = getattr(self, "last_hidden_contrastive_centroids", None)
                    c_primary = (
                        h_contr[expert_idx]
                        if (h_contr is not None and float(h_contr[expert_idx].float().norm().item()) > 1e-8)
                        else h_raw
                    ).to(device=dev, dtype=dt)
                    h_sub = getattr(self, "last_hidden_subspace", None)
                    if h_sub is not None and num_dirs >= 1:
                        k_eff = min(num_dirs, h_sub.size(-1))
                        sub_mat = h_sub[expert_idx, :, :k_eff].to(device=dev, dtype=dt)
                        if float(sub_mat.float().norm().item()) > 1e-8:
                            return c_primary, sub_mat
                    return c_primary, None
            # Fallback when last_centroids was externally overridden or hidden centroids not yet populated
            c_e = self.last_centroids[expert_idx]
            if (
                float(c_e.float().norm().item()) > 1e-8
                and expert_idx < len(self.experts)
                and len(self.experts[expert_idx]) >= 2
            ):
                exp = self.experts[expert_idx]
                h = exp[0](c_e.unsqueeze(0))
                if len(exp) > 2:
                    h = exp[1](h)
                h_vec = h.squeeze(0).to(device=dev, dtype=dt)
                if float(h_vec.float().norm().item()) > 1e-8:
                    return h_vec, F.normalize(h_vec.float(), dim=0).to(dtype=dt).unsqueeze(-1)
            return None, None

        # steer_role == "input"
        if centroids_overridden:
            c_in = self.last_centroids[expert_idx].to(device=dev, dtype=dt)
            if float(c_in.float().norm().item()) > 1e-8:
                return c_in, F.normalize(c_in.float(), dim=0).to(dtype=dt).unsqueeze(-1)
            return None, None

        in_contr = getattr(self, "last_contrastive_centroids", None)
        c_raw = self.last_centroids[expert_idx]
        c_primary = (
            in_contr[expert_idx]
            if (in_contr is not None and float(in_contr[expert_idx].float().norm().item()) > 1e-8)
            else c_raw
        ).to(device=dev, dtype=dt)
        in_sub = getattr(self, "last_input_subspace", None)
        if in_sub is not None and num_dirs >= 1:
            k_eff = min(num_dirs, in_sub.size(-1))
            sub_mat = in_sub[expert_idx, :, :k_eff].to(device=dev, dtype=dt)
            if float(sub_mat.float().norm().item()) > 1e-8:
                return c_primary, sub_mat
        if float(c_primary.float().norm().item()) > 1e-8:
            return c_primary, F.normalize(c_primary.float(), dim=0).to(dtype=dt).unsqueeze(-1)
        return None, None

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
