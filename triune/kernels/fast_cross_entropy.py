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
    """Returns (sum_loss, grad_logits_unscaled) for one chunk. grad = softmax - onehot, zeroed on ignored rows."""
    valid = chunk_y != ignore_index
    safe_y = torch.where(valid, chunk_y, torch.zeros_like(chunk_y))
    lse = torch.logsumexp(chunk_logits_fp32, dim=-1)
    target_logit = chunk_logits_fp32.gather(1, safe_y.unsqueeze(1)).squeeze(1)
    loss_sum = ((lse - target_logit) * valid).sum()

    grad = torch.softmax(chunk_logits_fp32, dim=-1)
    grad.scatter_add_(1, safe_y.unsqueeze(1), -torch.ones_like(target_logit).unsqueeze(1))
    grad.mul_(valid.unsqueeze(1).to(grad.dtype))
    return loss_sum, grad


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, chunk_size: int, ignore_index: int):
        flat_h = hidden.reshape(-1, hidden.size(-1))
        flat_y = targets.reshape(-1)
        n_tokens = flat_h.size(0)
        n_valid = int((flat_y != ignore_index).sum().item())
        acc = torch.promote_types(weight.dtype, torch.float32)

        need_h, need_w, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], (
            bias is not None and ctx.needs_input_grad[2]
        )
        grad_h = torch.zeros_like(flat_h) if need_h else None
        grad_w = torch.zeros(weight.shape, device=weight.device, dtype=acc) if need_w else None
        grad_b = torch.zeros(bias.shape, device=bias.device, dtype=acc) if need_b else None

        total = torch.zeros((), device=hidden.device, dtype=acc)
        if n_valid > 0:
            inv_n = 1.0 / n_valid
            for i in range(0, n_tokens, chunk_size):
                h = flat_h[i : i + chunk_size]
                y = flat_y[i : i + chunk_size]
                if not bool((y != ignore_index).any()):
                    continue
                logits = F.linear(h, weight, bias).to(acc)
                loss_sum, g = _chunk_ce_grad(logits, y, ignore_index)
                total += loss_sum
                del logits
                g.mul_(inv_n)
                if need_h:
                    grad_h[i : i + chunk_size] = (g.to(weight.dtype) @ weight).to(grad_h.dtype)
                if need_w:
                    grad_w.addmm_(g.t(), h.to(acc))
                if need_b:
                    grad_b += g.sum(0)
                del g
            total = total * inv_n

        ctx.save_for_backward(
            grad_h.reshape(hidden.shape) if grad_h is not None else None,
            grad_w.to(weight.dtype) if grad_w is not None else None,
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
        n_valid = int((flat_y != ignore_index).sum().item())
        acc = torch.promote_types(logits.dtype, torch.float32)
        total = torch.zeros((), device=logits.device, dtype=acc)
        if n_valid > 0:
            with torch.no_grad():
                for i in range(0, flat_l.size(0), chunk_size):
                    y = flat_y[i : i + chunk_size]
                    total += F.cross_entropy(
                        flat_l[i : i + chunk_size].to(acc), y, ignore_index=ignore_index, reduction="sum"
                    )
            total = total / n_valid
        ctx.save_for_backward(logits, targets)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.n_valid = n_valid
        return total

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        grad = torch.zeros_like(logits)
        if ctx.n_valid == 0:
            return grad, None, None, None
        flat_l = logits.reshape(-1, logits.size(-1))
        flat_g = grad.view(-1, logits.size(-1))
        flat_y = targets.reshape(-1)
        acc = torch.promote_types(logits.dtype, torch.float32)
        scale = grad_out.to(acc) / ctx.n_valid
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
    if flat_logits.size(0) <= chunk_size:
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)
    return _ChunkedCrossEntropy.apply(logits, targets, int(chunk_size), int(ignore_index))
