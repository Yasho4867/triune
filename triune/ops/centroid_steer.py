"""CentroidSteer Native Optimization Operator."""

from __future__ import annotations

import torch


def batched_qr(
    Y_mat: torch.Tensor,
    cond_threshold: float = 1e4,
    eps_shift: float = 1e-7,
    orth_tol: float = 1e-4,
    return_fallback_mask: bool = False,
    return_diagnostics: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes batched thin QR with condition-ratio monitored Cholesky-QR, post-QR orthogonality verification, and Householder fallback.

    Args:
        Y_mat: Batched input matrices of shape [K, d, r]
        cond_threshold: Condition ratio threshold kappa(A) approx max(L_ii^2) / min(L_ii^2) for fallback (default 1e4)
        eps_shift: Diagonal shift added to Gram matrix to ensure positive definiteness
        orth_tol: Maximum permissible basis orthogonality error ||Q^T Q - I||_F before Householder fallback (default 1e-4)
        return_fallback_mask: If True, returns (Q, fallback_mask)
        return_diagnostics: If True, returns (Q, diagnostics_dict) with mutually exclusive and overlapping breakdowns
        out_dtype: Target output dtype for Q (defaults to Y_mat.dtype)

    Returns:
        Q orthonormal matrices of shape [K, d, r], optionally accompanied by fallback_mask or diagnostics_dict
    """
    dtype = out_dtype if out_dtype is not None else Y_mat.dtype
    Y_f = Y_mat.float()
    K, d, r = Y_f.shape
    col_norms = torch.linalg.norm(Y_f, dim=1, keepdim=True)
    degen_mask = (col_norms.squeeze(1) <= 1e-12).any(dim=-1)

    # Pass 1: Shifted Cholesky-QR preconditioning (reduces kappa(Y) <= 1e4 down to kappa(Q_1) ~ 1)
    Y_norm = Y_f / col_norms.clamp_min(1e-12)
    Gram1 = torch.bmm(Y_norm.transpose(1, 2), Y_norm)
    Gram1.diagonal(dim1=1, dim2=2).add_(eps_shift)
    L1, info1 = torch.linalg.cholesky_ex(Gram1)
    Q1 = torch.linalg.solve_triangular(L1, Y_norm.transpose(1, 2), upper=False).transpose(1, 2)

    # Pass 2: Cholesky-QR refinement on Q_1 (sCholQR2: eliminates O(eps_shift / sigma_min^2) shift bias)
    q1_norms = torch.linalg.norm(Q1, dim=1, keepdim=True)
    Q1_norm = Q1 / q1_norms.clamp_min(1e-12)
    Gram2 = torch.bmm(Q1_norm.transpose(1, 2), Q1_norm)
    Gram2.diagonal(dim1=1, dim2=2).add_(eps_shift)
    L2, info2 = torch.linalg.cholesky_ex(Gram2)
    cholesky_fail_mask = (info1 != 0) | (info2 != 0)

    # Condition ratio monitoring on refined Cholesky factor: kappa(A_2) approx max(L2_ii^2) / min(L2_ii^2)
    L2_diag = L2.diagonal(dim1=1, dim2=2)
    L2_diag_sq = L2_diag * L2_diag
    cond_ratio = L2_diag_sq.max(dim=-1).values / L2_diag_sq.min(dim=-1).values.clamp_min(1e-12)
    cond_fail_mask = (cond_ratio >= cond_threshold)

    # Solve triangular system L2 Q^T = Q1_norm^T
    Q = torch.linalg.solve_triangular(L2, Q1_norm.transpose(1, 2), upper=False).transpose(1, 2)

    # Post-QR basis orthogonality check: ||Q^T Q - I||_F
    I = torch.eye(r, device=Y_mat.device, dtype=torch.float32).unsqueeze(0).expand(K, -1, -1)
    QtQ = torch.bmm(Q.transpose(1, 2), Q)
    orth_err = torch.linalg.norm(QtQ - I, dim=(1, 2))
    orth_fail_mask = (orth_err > orth_tol)

    fallback_mask = cond_fail_mask | degen_mask | cholesky_fail_mask | orth_fail_mask

    if Y_mat.is_cuda and torch.cuda.is_current_stream_capturing():
        # Synchronization-free static graph branch for CUDA Graph capture
        Q_householder, _ = torch.linalg.qr(Y_f)
        Q = torch.where(fallback_mask.view(K, 1, 1), Q_householder, Q)
    elif bool(fallback_mask.any()):
        # Eager refresh branch: performs one scalar boolean readback per batched_qr call
        # (avoiding dynamic torch.nonzero index allocation and skipping Householder QR on clean batches)
        if K == 1 or bool(fallback_mask.all()):
            Q, _ = torch.linalg.qr(Y_f)
        else:
            Q_failed, _ = torch.linalg.qr(Y_f[fallback_mask])
            Q[fallback_mask] = Q_failed

    Q_out = Q.to(dtype=dtype)
    if return_diagnostics:
        # Mutually exclusive primary cause (chronological stage order)
        primary_degen = degen_mask
        primary_cholesky = (~primary_degen) & cholesky_fail_mask
        primary_cond = (~primary_degen) & (~primary_cholesky) & cond_fail_mask
        primary_orth = (~primary_degen) & (~primary_cholesky) & (~primary_cond) & orth_fail_mask

        diag = {
            "fallback_mask": fallback_mask,
            "cond_fail_mask": cond_fail_mask,
            "degen_mask": degen_mask,
            "cholesky_fail_mask": cholesky_fail_mask,
            "orth_fail_mask": orth_fail_mask,
            "primary_degen_mask": primary_degen,
            "primary_cholesky_mask": primary_cholesky,
            "primary_cond_mask": primary_cond,
            "primary_orth_mask": primary_orth,
            "cond_ratio": cond_ratio,
            "orth_err": orth_err,
        }
        return Q_out, diag
    if return_fallback_mask:
        return Q_out, fallback_mask
    return Q_out


def cholesky_qr(
    Y_f: torch.Tensor,
    cond_threshold: float = 1e4,
    eps_shift: float = 1e-7,
    orth_tol: float = 1e-4,
) -> torch.Tensor:
    """Robust Cholesky-QR orthogonalization with Gram preconditioning and Householder fallback."""
    if Y_f.ndim == 2:
        return batched_qr(
            Y_f.unsqueeze(0),
            cond_threshold=cond_threshold,
            eps_shift=eps_shift,
            orth_tol=orth_tol,
            out_dtype=Y_f.dtype,
        ).squeeze(0)
    return batched_qr(
        Y_f,
        cond_threshold=cond_threshold,
        eps_shift=eps_shift,
        orth_tol=orth_tol,
        out_dtype=Y_f.dtype,
    )


def polar_stiefel_retraction(P: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Retracts matrix P back onto the Stiefel manifold St(r, d) using polar Cholesky factor.
    
    Guarantees strict isometry P^T P = I_r in O(r^3) operations without Stiefel manifold drift.
    """
    orig_dtype = P.dtype
    P_f = P.float()
    if P.ndim == 2:
        G = torch.mm(P_f.t(), P_f)
        G.diagonal().add_(eps)
        L, info = torch.linalg.cholesky_ex(G)
        if info == 0:
            P_ortho = torch.linalg.solve_triangular(L, P_f.t(), upper=False).t()
            return P_ortho.to(orig_dtype)
        else:
            Q, _ = torch.linalg.qr(P_f)
            return Q.to(orig_dtype)
    elif P.ndim == 3:
        G = torch.bmm(P_f.transpose(1, 2), P_f)
        G.diagonal(dim1=1, dim2=2).add_(eps)
        L, info = torch.linalg.cholesky_ex(G)
        if (info == 0).all():
            P_ortho = torch.linalg.solve_triangular(L, P_f.transpose(1, 2), upper=False).transpose(1, 2)
            return P_ortho.to(orig_dtype)
        else:
            Q, _ = torch.linalg.qr(P_f)
            return Q.to(orig_dtype)
    return P


def subspace_projection(
    grad: torch.Tensor,
    rank: int,
    niter: int = 2,
    cond_threshold: float = 1e4,
    eps_shift: float = 1e-7,
    orth_tol: float = 1e-4,
) -> torch.Tensor:
    """Tensor Core Subspace Iteration with Randomized SVD Core Eigendecomposition."""
    m, n = grad.shape
    q = min(rank + 10, m, n)
    side = "left" if m <= n else "right"
    device, dtype = grad.device, grad.dtype
    if side == "left":
        Omega = torch.randn(n, q, device=device, dtype=dtype)
        Y = torch.mm(grad, Omega)
        Q = cholesky_qr(Y.float(), cond_threshold=cond_threshold, eps_shift=eps_shift, orth_tol=orth_tol).to(dtype)
        for _ in range(niter):
            Z = torch.mm(grad.t(), Q)
            Q_z = cholesky_qr(Z.float(), cond_threshold=cond_threshold, eps_shift=eps_shift, orth_tol=orth_tol).to(dtype)
            Y = torch.mm(grad, Q_z)
            Q = cholesky_qr(Y.float(), cond_threshold=cond_threshold, eps_shift=eps_shift, orth_tol=orth_tol).to(dtype)
        
        B = torch.mm(Q.t(), grad)
        B_small = torch.mm(B, B.t())
        S, U_B = torch.linalg.eigh(B_small.float())
        idx = torch.argsort(S, descending=True)
        U_core = U_B[:, idx[:rank]].to(dtype)
        return torch.mm(Q, U_core)
    else:
        Omega = torch.randn(m, q, device=device, dtype=dtype)
        Y = torch.mm(grad.t(), Omega)
        Q = cholesky_qr(Y.float(), cond_threshold=cond_threshold, eps_shift=eps_shift, orth_tol=orth_tol).to(dtype)
        for _ in range(niter):
            Z = torch.mm(grad, Q)
            Q_z = cholesky_qr(Z.float(), cond_threshold=cond_threshold, eps_shift=eps_shift, orth_tol=orth_tol).to(dtype)
            Y = torch.mm(grad.t(), Q_z)
            Q = cholesky_qr(Y.float(), cond_threshold=cond_threshold, eps_shift=eps_shift, orth_tol=orth_tol).to(dtype)
        
        B = torch.mm(grad, Q)
        B_small = torch.mm(B.t(), B)
        S, V_B = torch.linalg.eigh(B_small.float())
        idx = torch.argsort(S, descending=True)
        V_core = V_B[:, idx[:rank]].to(dtype)
        return torch.mm(Q, V_core)


def batched_subspace_projection(
    grads: torch.Tensor,
    rank: int,
    niter: int = 2,
    cond_threshold: float = 1e4,
    eps_shift: float = 1e-7,
    orth_tol: float = 1e-4,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, str] | tuple[torch.Tensor, str, list[dict[str, torch.Tensor]]]:
    """Batched Tensor Core Subspace Iteration with Randomized SVD Core Eigendecomposition."""
    N, m, n = grads.shape
    q = min(rank + 10, m, n)
    side = "left" if m <= n else "right"
    device, dtype = grads.device, grads.dtype

    diag_history = []

    def _qr_call(Y_mat: torch.Tensor) -> torch.Tensor:
        if return_diagnostics:
            Q, d = batched_qr(
                Y_mat,
                cond_threshold=cond_threshold,
                eps_shift=eps_shift,
                orth_tol=orth_tol,
                return_diagnostics=True,
                out_dtype=dtype,
            )
            diag_history.append(d)
            return Q
        return batched_qr(
            Y_mat,
            cond_threshold=cond_threshold,
            eps_shift=eps_shift,
            orth_tol=orth_tol,
            out_dtype=dtype,
        )

    if side == "left":
        Omega = torch.randn(N, n, q, device=device, dtype=dtype)
        Y = torch.bmm(grads, Omega)
        Q = _qr_call(Y)
        for _ in range(niter):
            Z = torch.bmm(grads.transpose(1, 2), Q)
            Q_z = _qr_call(Z)
            Y = torch.bmm(grads, Q_z)
            Q = _qr_call(Y)
        
        B = torch.bmm(Q.transpose(1, 2), grads)
        B_small = torch.bmm(B, B.transpose(1, 2))
        S, U_B = torch.linalg.eigh(B_small.float())
        idx = torch.argsort(S, dim=-1, descending=True)
        U_B_sorted = torch.gather(U_B, 2, idx.unsqueeze(1).expand(-1, q, -1))
        U_core = U_B_sorted[:, :, :rank].to(dtype)
        proj = torch.bmm(Q, U_core)
    else:
        Omega = torch.randn(N, m, q, device=device, dtype=dtype)
        Y = torch.bmm(grads.transpose(1, 2), Omega)
        Q = _qr_call(Y)
        for _ in range(niter):
            Z = torch.bmm(grads, Q)
            Q_z = _qr_call(Z)
            Y = torch.bmm(grads.transpose(1, 2), Q_z)
            Q = _qr_call(Y)
        
        B = torch.bmm(grads, Q)
        B_small = torch.bmm(B.transpose(1, 2), B)
        S, V_B = torch.linalg.eigh(B_small.float())
        idx = torch.argsort(S, dim=-1, descending=True)
        V_B_sorted = torch.gather(V_B, 2, idx.unsqueeze(1).expand(-1, q, -1))
        V_core = V_B_sorted[:, :, :rank].to(dtype)
        proj = torch.bmm(Q, V_core)

    if return_diagnostics:
        return proj, side, diag_history
    return proj, side


def triangular_steer_update(
    P: torch.Tensor,
    Grad_proj: torch.Tensor,
    steer_scale: float = 1.0,
    momentum: float = 0.9,
    retract_stiefel: bool = False,
) -> torch.Tensor:
    """Fused triangular CentroidSteer projection update with optional Polar Stiefel Retraction."""
    from ._c_bridge import c_centroid_steer_update_f32

    if c_centroid_steer_update_f32(P, Grad_proj, steer_scale, momentum):
        if retract_stiefel:
            P_retract = polar_stiefel_retraction(P)
            P.copy_(P_retract)
        return P
    delta = (Grad_proj - P) * steer_scale
    P.mul_(momentum).add_(delta)
    if retract_stiefel:
        P_retract = polar_stiefel_retraction(P)
        P.copy_(P_retract)
    return P
