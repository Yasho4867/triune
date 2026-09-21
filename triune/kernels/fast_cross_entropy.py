"""Chunked Fast Cross-Entropy Loss.

Eliminates the multi-gigabyte logits VRAM explosion on large vocabularies (e.g. Llama 3 128k vocab)
by computing log-sum-exp and negative log-likelihood in chunked streaming blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dispatcher import is_triton_available


def chunked_cross_entropy_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear | torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Computes cross-entropy directly from hidden states without materializing full [B, T, V] logits.

    Saves 4+ GB of VRAM on large vocabulary models (e.g., Llama-3 128k).
    """
    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_targets = targets.reshape(-1)

    total_tokens = flat_hidden.size(0)
    total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
    valid_tokens = 0

    weight = lm_head.weight if isinstance(lm_head, nn.Linear) else lm_head
    bias = lm_head.bias if isinstance(lm_head, nn.Linear) else None

    for i in range(0, total_tokens, chunk_size):
        chunk_h = flat_hidden[i : i + chunk_size]
        chunk_y = flat_targets[i : i + chunk_size]

        valid_mask = chunk_y != ignore_index
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            continue

        # Project only this chunk into logits
        chunk_logits = F.linear(chunk_h, weight, bias)
        chunk_loss = F.cross_entropy(chunk_logits, chunk_y, ignore_index=ignore_index, reduction="sum")

        total_loss = total_loss + chunk_loss
        valid_tokens += n_valid

    if valid_tokens > 0:
        return total_loss / valid_tokens
    return total_loss


def fast_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 2048,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Computes cross-entropy loss in chunks to minimize intermediate activation memory."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)

    total_tokens = flat_logits.size(0)
    if total_tokens <= chunk_size:
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)

    total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    valid_tokens = 0

    for i in range(0, total_tokens, chunk_size):
        chunk_l = flat_logits[i : i + chunk_size]
        chunk_y = flat_targets[i : i + chunk_size]

        valid_mask = chunk_y != ignore_index
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            continue

        loss_chunk = F.cross_entropy(chunk_l, chunk_y, ignore_index=ignore_index, reduction="sum")
        total_loss = total_loss + loss_chunk
        valid_tokens += n_valid

    if valid_tokens > 0:
        return total_loss / valid_tokens
    return total_loss
