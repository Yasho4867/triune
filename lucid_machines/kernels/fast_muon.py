from __future__ import annotations

import torch

NS5_COEFFS = (1.8750, -1.2500, 0.3750)


def compute_muon_update_scale(d_out: int, d_in: int, rms_target: float | None = None) -> float:
    aspect = max(1.0, float(d_out) / float(d_in)) ** 0.5
    if rms_target is not None and rms_target > 0:
        return aspect * float(rms_target)
    return aspect


def fast_zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    scratch_buffer: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    return_scratch: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if G.ndim != 2:
        raise ValueError(f"Matrix orthogonalization requires 2D tensor, got ndim={G.ndim}")

    orig_device = G.device
    orig_dtype = G.dtype

    if orig_device.type == "cuda" and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    X = G.to(device=orig_device, dtype=compute_dtype).clone()

    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True

    m, n = X.shape

    v = torch.ones((n, 1), device=orig_device, dtype=compute_dtype) / (float(n) ** 0.5)
    u = X @ v
    u = u / torch.linalg.vector_norm(u).clamp_min(1e-12)
    v = X.t() @ u
    v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)
    u = X @ v
    sigma_est = torch.linalg.vector_norm(u).clamp_min(1e-12)
    f_norm = torch.linalg.vector_norm(X)
    sigma_safe = torch.maximum(1.15 * sigma_est, f_norm / (float(min(m, n)) ** 0.5))
    # Invariant: sigma_safe must not be below 0.70 * ||X||_F to guarantee spectral norm <= 1/0.70 < sqrt(7/3),
    # strictly preventing Newton-Schulz polynomial divergence on low-rank/sparse gradients.
    sigma_safe = torch.maximum(sigma_safe, f_norm * 0.70).clamp_min(eps)
    X = X.div_(sigma_safe)

    a, b, c = NS5_COEFFS

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
        torch.mm(X, X.t(), out=A)
        torch.addmm(A, A, A, beta=b, alpha=c, out=B)
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
    if G_batch.ndim != 3:
        raise ValueError(f"Batched orthogonalization requires 3D tensor [B, M, N], got ndim={G_batch.ndim}")

    orig_device = G_batch.device
    orig_dtype = G_batch.dtype

    if orig_device.type == "cuda" and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    X = G_batch.to(device=orig_device, dtype=compute_dtype).clone()

    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True

    B_sz, M, N = X.shape

    v = torch.ones((B_sz, N, 1), device=orig_device, dtype=compute_dtype) / (float(N) ** 0.5)
    u = torch.bmm(X, v)
    u = u / torch.linalg.vector_norm(u, dim=1, keepdim=True).clamp_min(1e-12)
    v = torch.bmm(X.transpose(1, 2), u)
    v = v / torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-12)
    u = torch.bmm(X, v)
    sigma_est = torch.linalg.vector_norm(u, dim=1, keepdim=True).clamp_min(1e-12)
    f_norm = torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True)
    sigma_safe = torch.maximum(1.15 * sigma_est, f_norm / (float(min(M, N)) ** 0.5))
    # Invariant: sigma_safe must not be below 0.70 * ||X||_F to guarantee spectral norm <= 1/0.70 < sqrt(7/3),
    # strictly preventing Newton-Schulz polynomial divergence on low-rank/sparse gradients.
    sigma_safe = torch.maximum(sigma_safe, f_norm * 0.70).clamp_min(eps)
    X = X.div_(sigma_safe)

    a, b, c = NS5_COEFFS

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
        torch.bmm(X, X.transpose(1, 2), out=A)
        torch.baddbmm(A, A, A, beta=b, alpha=c, out=B)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    if transposed:
        X = X.transpose(1, 2)

    res = X.to(device=orig_device, dtype=orig_dtype)
    if return_scratch:
        return res, (A, A2, B)
    return res
