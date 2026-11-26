from __future__ import annotations

import math

import torch

NS5_COEFFS = (1.8750, -1.2500, 0.3750)
POLAR_EXPRESS_CUSHION = 0.02407327424182761
POLAR_EXPRESS_SAFETY_FACTOR = 1.01

POLAR_EXPRESS_COEFFS_4: tuple[tuple[float, float, float], ...] = (
    (7.9102606081, -22.0788217460, 15.8691137368),
    (3.6137308254, -2.6533241851, 0.5124429161),
    (2.6635018995, -1.9471022980, 0.4358754063),
    (1.9720051091, -1.3603352724, 0.3909666383),
)

POLAR_EXPRESS_COEFFS_BY_STEPS: dict[int, tuple[tuple[float, float, float], ...]] = {
    1: ((7.6830609004, -21.8757083477, 16.0391611837),),
    2: (
        (7.6069909905, -21.2323470012, 15.2607115243),
        (3.2276150128, -2.4241353320, 0.5040505538),
    ),
    3: (
        (7.6069909905, -21.2323470012, 15.2607115243),
        (3.1956584285, -2.3528418705, 0.4795868068),
        (2.2365647230, -1.6172085240, 0.4168206840),
    ),
    4: POLAR_EXPRESS_COEFFS_4,
    5: (
        (7.9102606081, -22.0788217460, 15.8691137368),
        (3.6137308254, -2.6533241851, 0.5124429161),
        (2.6635018995, -1.9471022980, 0.4358754063),
        (1.9524803061, -1.3203280133, 0.3719913414),
        (1.8780048818, -1.2557719974, 0.3777664381),
    ),
    6: (
        (7.9102606081, -22.0788217460, 15.8691137368),
        (3.6137308254, -2.6533241851, 0.5124429161),
        (2.6635018995, -1.9471022980, 0.4358754063),
        (1.9524803061, -1.3203280133, 0.3719913414),
        (1.8594107741, -1.2188399287, 0.3594318038),
        (1.8750262350, -1.2500524705, 0.3750262355),
    ),
}


def _optimal_odd_quintic(l_val: float, u_val: float) -> tuple[float, float, float]:
    """Compute minimax odd quintic p(x) = a*x + b*x^3 + c*x^5 approximating 1 on [l_val, u_val]."""
    pts = [
        l_val,
        l_val + 0.35 * (u_val - l_val),
        l_val + 0.72 * (u_val - l_val),
        u_val,
    ]
    a, b, c = NS5_COEFFS
    for _ in range(32):
        aug = [[x, x**3, x**5, (-1.0) ** idx, 1.0] for idx, x in enumerate(pts)]
        for col in range(4):
            piv = max(range(col, 4), key=lambda row: abs(aug[row][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            denom = aug[col][col]
            if abs(denom) < 1e-18:
                return a, b, c
            for j in range(col, 5):
                aug[col][j] /= denom
            for row in range(4):
                if row != col:
                    factor = aug[row][col]
                    for j in range(col, 5):
                        aug[row][j] -= factor * aug[col][j]
        a, b, c = aug[0][4], aug[1][4], aug[2][4]
        disc = 9.0 * b * b - 20.0 * a * c
        if disc < 0.0 or abs(c) < 1e-18:
            break
        sq = math.sqrt(disc)
        cand = []
        for root_sq in ((-3.0 * b - sq) / (10.0 * c), (-3.0 * b + sq) / (10.0 * c)):
            if root_sq > 0.0:
                cand.append(math.sqrt(max(l_val * l_val, min(u_val * u_val, root_sq))))
        if len(cand) != 2:
            break
        cand.sort()
        new_pts = [l_val, cand[0], cand[1], u_val]
        if max(abs(n - o) for n, o in zip(new_pts, pts)) < 1e-13:
            pts = new_pts
            break
        pts = new_pts
    return a, b, c


def compute_polar_express_coeffs(
    steps: int,
    lower_bound: float | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Return minimax Polar Express polynomial coefficients (a, b, c) per iteration."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if lower_bound is None and steps in POLAR_EXPRESS_COEFFS_BY_STEPS:
        return POLAR_EXPRESS_COEFFS_BY_STEPS[steps]

    l_cur = float(lower_bound if lower_bound is not None else (2e-2 if steps <= 3 else 1e-2))
    l_cur = max(1e-5, min(0.99, l_cur))
    u_cur = 1.0
    coeffs: list[tuple[float, float, float]] = []

    def _eval_poly(a: float, b: float, c: float, x: float) -> float:
        x2 = x * x
        return x * (a + x2 * (b + x2 * c))

    for step_idx in range(steps):
        eff_l = max(l_cur, POLAR_EXPRESS_CUSHION * u_cur)
        if eff_l >= u_cur * 0.999999:
            coeffs.append((1.0, 0.0, 0.0))
            continue
        a, b, c = _optimal_odd_quintic(eff_l, u_cur)
        pl = _eval_poly(a, b, c, l_cur)
        pu = _eval_poly(a, b, c, u_cur)
        denom = pl + pu
        if abs(denom) > 1e-12:
            scale = 2.0 / denom
            a *= scale
            b *= scale
            c *= scale
        l_next = _eval_poly(a, b, c, l_cur)
        u_next = 2.0 - l_next
        if step_idx < steps - 1:
            a /= POLAR_EXPRESS_SAFETY_FACTOR
            b /= POLAR_EXPRESS_SAFETY_FACTOR**3
            c /= POLAR_EXPRESS_SAFETY_FACTOR**5
        coeffs.append((a, b, c))
        l_cur, u_cur = max(1e-6, min(0.999999, l_next)), max(1.000001, u_next) if u_next > 1.0 else max(l_next + 1e-6, u_next)
    return tuple(coeffs)


def _normalize_for_polar_express(X: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale 2D tensor X by the exact Frobenius/Holder spectral upper bound * safety_factor."""
    f_norm = torch.linalg.vector_norm(X)
    abs_x = X.abs()
    n1 = abs_x.sum(dim=0).amax()
    ninf = abs_x.sum(dim=1).amax()
    upper = torch.minimum(f_norm, torch.sqrt(n1 * ninf))
    return X.div_((upper * POLAR_EXPRESS_SAFETY_FACTOR).clamp_min(eps))


def _normalize_for_polar_express_batched(X: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale batched 3D tensor X [B, M, N] by exact spectral upper bound * safety_factor."""
    f_norm = torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True)
    abs_x = X.abs()
    n1 = abs_x.sum(dim=1).amax(dim=-1, keepdim=True).unsqueeze(-1)
    ninf = abs_x.sum(dim=2).amax(dim=-1, keepdim=True).unsqueeze(-1)
    upper = torch.minimum(f_norm, torch.sqrt(n1 * ninf))
    return X.div_((upper * POLAR_EXPRESS_SAFETY_FACTOR).clamp_min(eps))


def _normalize_for_ns5(X: torch.Tensor, eps: float) -> torch.Tensor:
    orig_device = X.device
    compute_dtype = X.dtype
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
    return X.div_(sigma_safe)


def _normalize_for_ns5_batched(X: torch.Tensor, eps: float) -> torch.Tensor:
    orig_device = X.device
    compute_dtype = X.dtype
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
    sigma_safe = torch.maximum(sigma_safe, f_norm * 0.70).clamp_min(eps)
    return X.div_(sigma_safe)


def compute_muon_update_scale(d_out: int, d_in: int, rms_target: float | None = None) -> float:
    aspect = max(1.0, float(d_out) / float(d_in)) ** 0.5
    if rms_target is not None and rms_target > 0:
        return aspect * float(rms_target)
    return aspect


def fast_zeropower_via_polar_express(
    G: torch.Tensor,
    steps: int = 4,
    eps: float = 1e-7,
    scratch_buffer: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    return_scratch: bool = False,
    lower_bound: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute matrix orthogonalization using Polar Express minimax quintic iterations (default 4 steps)."""
    return fast_zeropower_via_newtonschulz5(
        G,
        steps=steps,
        eps=eps,
        scratch_buffer=scratch_buffer,
        return_scratch=return_scratch,
        method="polar_express",
        lower_bound=lower_bound,
    )


def fast_zeropower_via_polar_express_batched(
    G_batch: torch.Tensor,
    steps: int = 4,
    eps: float = 1e-7,
    scratch_buffer: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_scratch: bool = False,
    lower_bound: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Batched 3D [B, M, N] matrix orthogonalization via Polar Express minimax quintic iterations."""
    return fast_zeropower_via_newtonschulz5_batched(
        G_batch,
        steps=steps,
        eps=eps,
        scratch_buffer=scratch_buffer,
        return_scratch=return_scratch,
        method="polar_express",
        lower_bound=lower_bound,
    )


def fast_zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int | None = None,
    eps: float = 1e-7,
    scratch_buffer: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    return_scratch: bool = False,
    *,
    method: str = "polar_express",
    lower_bound: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if G.ndim != 2:
        raise ValueError(f"Matrix orthogonalization requires 2D tensor, got ndim={G.ndim}")

    method_norm = str(method or "polar_express").strip().lower()
    if method_norm not in ("polar_express", "newtonschulz5", "ns5"):
        raise ValueError(f"Unknown orthogonalization method: {method!r}")

    eff_steps = (4 if method_norm == "polar_express" else 5) if steps is None else int(steps)
    if eff_steps < 1:
        raise ValueError(f"steps must be >= 1, got {eff_steps}")

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

    if method_norm == "polar_express":
        X = _normalize_for_polar_express(X, eps)
        coeff_schedule = compute_polar_express_coeffs(eff_steps, lower_bound=lower_bound)
    else:
        X = _normalize_for_ns5(X, eps)
        coeff_schedule = (NS5_COEFFS,) * eff_steps

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

    for a, b, c in coeff_schedule:
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
    steps: int | None = None,
    eps: float = 1e-7,
    scratch_buffer: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_scratch: bool = False,
    *,
    method: str = "polar_express",
    lower_bound: float | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if G_batch.ndim != 3:
        raise ValueError(f"Batched orthogonalization requires 3D tensor [B, M, N], got ndim={G_batch.ndim}")

    method_norm = str(method or "polar_express").strip().lower()
    if method_norm not in ("polar_express", "newtonschulz5", "ns5"):
        raise ValueError(f"Unknown orthogonalization method: {method!r}")

    eff_steps = (4 if method_norm == "polar_express" else 5) if steps is None else int(steps)
    if eff_steps < 1:
        raise ValueError(f"steps must be >= 1, got {eff_steps}")

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

    if method_norm == "polar_express":
        X = _normalize_for_polar_express_batched(X, eps)
        coeff_schedule = compute_polar_express_coeffs(eff_steps, lower_bound=lower_bound)
    else:
        X = _normalize_for_ns5_batched(X, eps)
        coeff_schedule = (NS5_COEFFS,) * eff_steps

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

    for a, b, c in coeff_schedule:
        torch.bmm(X, X.transpose(1, 2), out=A)
        torch.baddbmm(A, A, A, beta=b, alpha=c, out=B)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    if transposed:
        X = X.transpose(1, 2)

    res = X.to(device=orig_device, dtype=orig_dtype)
    if return_scratch:
        return res, (A, A2, B)
    return res
