"""Chunked Fast Cross-Entropy Loss.

Eliminates the multi-gigabyte logits VRAM explosion on large vocabularies (e.g. Llama 3 128k vocab)
by computing log-sum-exp and negative log-likelihood in chunked streaming blocks.

Fix H-20: Both entry points are implemented as custom ``torch.autograd.Function``s so that only a
single ``[chunk, V]`` logits block is ever alive at once. The previous implementation summed per-chunk
losses through autograd, which retained every chunk's softmax graph simultaneously and therefore
provided no memory savings.

* ``chunked_cross_entropy_from_hidden`` (fused linear + CE, Liger/Unsloth style): gradients w.r.t.
  hidden states, LM-head weight and bias are computed chunk-by-chunk during the forward pass. The
  full ``[N, V]`` logits tensor is never materialized; peak extra memory is ``O(chunk * V)``.
* ``fast_cross_entropy`` (logits already materialized): saves only the logits/targets and recomputes
  each chunk's softmax during backward, so no additional ``[N, V]`` activation is retained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dispatcher import is_triton_available  # noqa: F401  (re-exported for backward compatibility)


def _chunk_ce_grad(chunk_logits_fp32: torch.Tensor, chunk_y: torch.Tensor, ignore_index: int):
    """Returns (sum_loss, grad_logits_unscaled) for one chunk in-place on chunk_logits_fp32 (zero CPU sync)."""
    row_idx = torch.arange(chunk_y.size(0), device=chunk_y.device)
    valid = chunk_y != ignore_index
    safe_y = torch.where(valid, chunk_y, torch.zeros_like(chunk_y)) if ignore_index >= 0 else chunk_y.clamp_min(0)
    lse = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)
    target_logit = chunk_logits_fp32[row_idx, safe_y].unsqueeze(1)
    valid_col = valid.unsqueeze(1)
    loss_sum = ((lse - target_logit) * valid_col).sum()

    grad = chunk_logits_fp32.sub_(lse).exp_()
    grad[row_idx, safe_y] -= 1.0
    grad.mul_(valid_col)
    return loss_sum, grad


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, chunk_size: int, ignore_index: int):
        flat_h = hidden.reshape(-1, hidden.size(-1))
        flat_y = targets.reshape(-1)
        n_tokens = flat_h.size(0)
        # Adaptive chunk size: saturate Tensor Cores when vocab <= 32768 while capping transient memory
        eff_chunk = max(int(chunk_size), 1024) if weight.size(0) <= 32768 else int(chunk_size)

        # Device-side valid count without host synchronization
        n_valid = (flat_y != ignore_index).sum().clamp_min(1)
        inv_n = 1.0 / n_valid.to(torch.float32)
        acc = torch.promote_types(weight.dtype, torch.float32)

        need_h, need_w, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], (
            bias is not None and ctx.needs_input_grad[2]
        )
        grad_h = torch.empty_like(flat_h) if need_h else None
        grad_w = torch.zeros_like(weight) if need_w else None
        grad_b = torch.zeros(bias.shape, device=bias.device, dtype=acc) if need_b else None

        total = torch.zeros((), device=hidden.device, dtype=acc)
        for i in range(0, n_tokens, eff_chunk):
            h = flat_h[i : i + eff_chunk]
            y = flat_y[i : i + eff_chunk]
            logits = F.linear(h, weight, bias).to(acc)
            loss_sum, g = _chunk_ce_grad(logits, y, ignore_index)
            total = total + loss_sum
            del logits
            g.mul_(inv_n)
            if need_b:
                grad_b = grad_b + g.sum(0)
            g_w = g.to(weight.dtype)
            del g
            if need_h:
                torch.matmul(g_w, weight, out=grad_h[i : i + eff_chunk])
            if need_w:
                grad_w.addmm_(g_w.t(), h.to(weight.dtype))
            del g_w
        total = total * inv_n

        ctx.save_for_backward(
            grad_h.reshape(hidden.shape) if grad_h is not None else None,
            grad_w if grad_w is not None else None,
            grad_b.to(bias.dtype) if grad_b is not None else None,
        )
        return total

    @staticmethod
    def backward(ctx, grad_out):
        grad_h, grad_w, grad_b = ctx.saved_tensors
        scale = grad_out
        return (
            grad_h * scale.to(grad_h.dtype) if grad_h is not None else None,
            grad_w * scale.to(grad_w.dtype) if grad_w is not None else None,
            grad_b * scale.to(grad_b.dtype) if grad_b is not None else None,
            None,
            None,
            None,
        )


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, chunk_size: int, ignore_index: int):
        flat_l = logits.reshape(-1, logits.size(-1))
        flat_y = targets.reshape(-1)
        # Device-side valid count without host synchronization
        n_valid = (flat_y != ignore_index).sum().clamp_min(1)
        acc = torch.promote_types(logits.dtype, torch.float32)
        total = torch.zeros((), device=logits.device, dtype=acc)
        with torch.no_grad():
            for i in range(0, flat_l.size(0), chunk_size):
                y = flat_y[i : i + chunk_size]
                total = total + F.cross_entropy(
                    flat_l[i : i + chunk_size].to(acc), y, ignore_index=ignore_index, reduction="sum"
                )
        total = total / n_valid.to(acc)
        ctx.save_for_backward(logits, targets, n_valid)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        return total

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets, n_valid = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        grad = torch.empty_like(logits)
        flat_l = logits.reshape(-1, logits.size(-1))
        flat_g = grad.view(-1, logits.size(-1))
        flat_y = targets.reshape(-1)
        acc = torch.promote_types(logits.dtype, torch.float32)
        scale = grad_out.to(acc) / n_valid.to(acc)
        for i in range(0, flat_l.size(0), ctx.chunk_size):
            _, g = _chunk_ce_grad(flat_l[i : i + ctx.chunk_size].to(acc), flat_y[i : i + ctx.chunk_size], ctx.ignore_index)
            flat_g[i : i + ctx.chunk_size] = (g * scale).to(grad.dtype)
        return grad, None, None, None


def chunked_cross_entropy_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear | torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Computes cross-entropy directly from hidden states without materializing full [B, T, V] logits.

    Saves 4+ GB of VRAM on large vocabulary models (e.g., Llama-3 128k). Gradients for the hidden
    states and LM head are produced chunk-by-chunk, so peak extra memory is O(chunk_size * V).
    """
    weight = lm_head.weight if isinstance(lm_head, nn.Linear) else lm_head
    bias = lm_head.bias if isinstance(lm_head, nn.Linear) else None
    return _FusedLinearCrossEntropy.apply(hidden_states, weight, bias, targets, int(chunk_size), int(ignore_index))


def fast_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 2048,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Computes cross-entropy loss in chunks to minimize intermediate activation memory."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    eff_chunk = max(int(chunk_size), 1024) if flat_logits.size(-1) <= 32768 else int(chunk_size)
    if flat_logits.size(0) <= eff_chunk:
        acc = torch.promote_types(flat_logits.dtype, torch.float32)
        valid_count = (flat_targets != ignore_index).sum().clamp_min(1)
        return F.cross_entropy(flat_logits.to(acc), flat_targets, ignore_index=ignore_index, reduction="sum") / valid_count
    return _ChunkedCrossEntropy.apply(logits, targets, eff_chunk, int(ignore_index))


class FastChunkedCrossEntropyLoss(nn.Module):
    """Drop-in nn.Module wrapper for chunked cross-entropy (accepts either logits or (hidden, lm_head))."""

    def __init__(self, chunk_size: int = 1024, ignore_index: int = -100):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(
        self,
        logits_or_hidden: torch.Tensor,
        targets: torch.Tensor,
        lm_head: nn.Linear | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lm_head is not None:
            return chunked_cross_entropy_from_hidden(
                logits_or_hidden, lm_head, targets, chunk_size=self.chunk_size, ignore_index=self.ignore_index
            )
        return fast_cross_entropy(
            logits_or_hidden, targets, chunk_size=self.chunk_size, ignore_index=self.ignore_index
        )
