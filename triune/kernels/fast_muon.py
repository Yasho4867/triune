"""High-Performance Fused Muon Newton-Schulz Optimizer Kernel.

Optimizes 5th-order matrix orthogonalization (zeropower_via_newtonschulz5) by:
1. Complete buffer reuse: Reusing (A, A2, B) across all parameter updates.
2. Fused polynomial evaluation via in-place addmm / baddbmm.
3. Automatic Tensor Core precision routing (BF16 on Ampere/Ada/Blackwell, FP32 fallback).
"""

from __future__ import annotations

import torch


def fast_zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    scratch_buffer: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    return_scratch: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fast, memory-efficient Newton-Schulz orthogonalization.

    Uses the optimized quintic polynomial coefficients:
      a = 3.4445, b = -4.7750, c = 2.0315
    for fast convergence to the polar decomposition (semi-orthogonal matrix).
    """
    if G.ndim != 2:
        raise ValueError(f"Matrix orthogonalization requires 2D tensor, got ndim={G.ndim}")

    orig_device = G.device
    orig_dtype = G.dtype

    # Determine execution dtype
    if orig_device.type == "cuda" and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    X = G.to(device=orig_device, dtype=compute_dtype).clone()

    # In-place normalization on owned scratch tensor
    norm = torch.linalg.vector_norm(X)
    X = X.div_(norm + eps)

    # Ensure rows <= cols for efficiency
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True

    m, n = X.shape
    a, b, c = 3.4445, -4.7750, 2.0315

    # Allocate or reuse scratch buffers for A, A2, B [m, m]
    if (
        isinstance(scratch_buffer, (tuple, list))
        and len(scratch_buffer) == 3
        and scratch_buffer[0].shape == (m, m)
        and scratch_buffer[0].device == orig_device
        and scratch_buffer[0].dtype == compute_dtype
    ):
        A, A2, B = scratch_buffer
    elif (
        isinstance(scratch_buffer, torch.Tensor)
        and scratch_buffer.shape == (m, m)
        and scratch_buffer.device == orig_device
        and scratch_buffer.dtype == compute_dtype
    ):
        A = scratch_buffer
        A2 = torch.empty((m, m), device=orig_device, dtype=compute_dtype)
        B = torch.empty((m, m), device=orig_device, dtype=compute_dtype)
    else:
        A = torch.empty((m, m), device=orig_device, dtype=compute_dtype)
        A2 = torch.empty((m, m), device=orig_device, dtype=compute_dtype)
        B = torch.empty((m, m), device=orig_device, dtype=compute_dtype)

    for _ in range(steps):
        # A = X @ X.T
        torch.mm(X, X.t(), out=A)

        # A2 = A @ A
        torch.mm(A, A, out=A2)

        # B = b * A + c * A2
        B.copy_(A).mul_(b).add_(A2, alpha=c)

        # X = a * X + B @ X
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if transposed:
        X = X.t()

    res = X.to(device=orig_device, dtype=orig_dtype)
    if return_scratch:
        return res, (A, A2, B)
    return res


def fast_zeropower_via_newtonschulz5_batched(
    G_batch: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    scratch_buffer: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_scratch: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Batched Newton-Schulz orthogonalization across a batch of matching-shape 2D matrices.

    Args:
        G_batch: 3D tensor of shape [B, M, N].
        steps: Number of Newton-Schulz iterations (default: 5).
        eps: Small constant for vector norm stability.
        scratch_buffer: Optional tuple of (A, A2, B) preallocated scratch tensors.
        return_scratch: Whether to return scratch buffer for zero-allocation reuse.

    Returns:
        Orthogonalized tensor of shape [B, M, N] in original device and dtype.
    """
    if G_batch.ndim != 3:
        raise ValueError(f"Batched orthogonalization requires 3D tensor [B, M, N], got ndim={G_batch.ndim}")

    orig_device = G_batch.device
    orig_dtype = G_batch.dtype

    if orig_device.type == "cuda" and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    X = G_batch.to(device=orig_device, dtype=compute_dtype).clone()

    # In-place normalization across rows and cols on owned scratch batch
    norm = torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True)
    X = X.div_(norm + eps)

    # Ensure rows <= cols for efficiency
    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True

    B_sz, M, N = X.shape
    a, b, c = 3.4445, -4.7750, 2.0315

    if (
        isinstance(scratch_buffer, (tuple, list))
        and len(scratch_buffer) == 3
        and scratch_buffer[0].shape == (B_sz, M, M)
        and scratch_buffer[0].device == orig_device
        and scratch_buffer[0].dtype == compute_dtype
    ):
        A, A2, B = scratch_buffer
    else:
        A = torch.empty((B_sz, M, M), device=orig_device, dtype=compute_dtype)
        A2 = torch.empty((B_sz, M, M), device=orig_device, dtype=compute_dtype)
        B = torch.empty((B_sz, M, M), device=orig_device, dtype=compute_dtype)

    for _ in range(steps):
        # In-place GEMM and polynomial evaluation with zero allocations
        torch.bmm(X, X.transpose(1, 2), out=A)
        torch.bmm(A, A, out=A2)
        B.copy_(A).mul_(b).add_(A2, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    if transposed:
        X = X.transpose(1, 2)

    res = X.to(device=orig_device, dtype=orig_dtype)
    if return_scratch:
        return res, (A, A2, B)
    return res

