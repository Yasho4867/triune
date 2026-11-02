"""CentroidSteer Native Optimization Operator."""

from __future__ import annotations

import torch


def cholesky_qr(Y_f: torch.Tensor) -> torch.Tensor:
    """Robust Cholesky-QR orthogonalization with Gram preconditioning and QR fallback."""
    col_norms = torch.linalg.norm(Y_f, dim=0, keepdim=True).clamp_min(1e-12)
    Y_norm = Y_f / col_norms
    Gram = torch.mm(Y_norm.t(), Y_norm)
    Gram.diagonal().add_(1e-7)
    try:
        L = torch.linalg.cholesky(Gram)
        return torch.linalg.solve_triangular(L, Y_norm.t(), upper=False).t()
    except torch.linalg.LinAlgError:
        Q, _ = torch.linalg.qr(Y_f)
        return Q


def subspace_projection(grad: torch.Tensor, rank: int, niter: int = 2) -> torch.Tensor:
    """Cholesky-QR Subspace Iteration on Tensor Cores."""
    m, n = grad.shape
    q = min(rank + 10, m, n)
    side = 'left' if m <= n else 'right'
    if side == 'left':
        Omega = torch.randn(n, q, device=grad.device, dtype=grad.dtype)
        Y = torch.mm(grad, Omega)
        Q = cholesky_qr(Y.float())
        for _ in range(niter):
            Z = torch.mm(grad.t(), Q.to(grad.dtype))
            Q_z = cholesky_qr(Z.float())
            Y = torch.mm(grad, Q_z.to(grad.dtype))
            Q = cholesky_qr(Y.float())
        return Q[:, :rank].to(device=grad.device, dtype=grad.dtype)
    else:
        Omega = torch.randn(m, q, device=grad.device, dtype=grad.dtype)
        Y = torch.mm(grad.t(), Omega)
        Q = cholesky_qr(Y.float())
        for _ in range(niter):
            Z = torch.mm(grad, Q.to(grad.dtype))
            Q_z = cholesky_qr(Z.float())
            Y = torch.mm(grad.t(), Q_z.to(grad.dtype))
            Q = cholesky_qr(Y.float())
        return Q[:, :rank].to(device=grad.device, dtype=grad.dtype)


def triangular_steer_update(
    P: torch.Tensor,
    Grad_proj: torch.Tensor,
    steer_scale: float = 1.0,
    momentum: float = 0.9,
) -> torch.Tensor:
    """Fused triangular CentroidSteer projection update."""
    delta = (Grad_proj - P) * steer_scale
    P.mul_(momentum).add_(delta)
    return P
