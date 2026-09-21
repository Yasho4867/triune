"""Muon (MomentUm Orthogonalized by Newton-Schulz) optimizer.

Reference: Keller Jordan (2024), https://kellerjordan.github.io/posts/muon/
Designed for 2D hidden linear weights in neural networks.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to compute the approximate zeroth power (orthogonalization) of G.
    
    Uses a quintic polynomial whose coefficients (a=3.4445, b=-4.7750, c=2.0315) are optimized
    for rapid convergence to the nearest semi-orthogonal matrix.
    """
    if G.ndim != 2:
        raise ValueError(
            f"zeropower_via_newtonschulz5 requires a 2D matrix, got ndim={G.ndim} with shape {tuple(G.shape)}"
        )
    
    orig_device = G.device
    orig_dtype = G.dtype
    compute_device = orig_device
    
    # Run in bfloat16 on CUDA Tensor Cores for speedup if supported
    if compute_device.type == "cuda" and torch.cuda.is_bf16_supported():
        X = G.to(device=compute_device, dtype=torch.bfloat16)
    else:
        X = G.to(device=compute_device, dtype=torch.float32)
        
    norm = X.norm()
    X = X / (norm + eps)
    
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
        
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if transposed:
        X = X.T
        
    return X.to(device=orig_device, dtype=orig_dtype)


class Muon(Optimizer):
    """Muon optimizer for 2D parameter matrices in hidden layers.
    
    Internally maintains standard SGD momentum and post-processes each update
    via Newton-Schulz orthogonalization.
    
    Arguments:
        params: Iterable of parameters to optimize (should be 2D weight matrices).
        lr: Learning rate (units of spectral norm per update, default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        weight_decay: Decoupled weight decay coefficient (default: 0.01).
        nesterov: Whether to use Nesterov-style momentum blending (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon optimizer only supports 2D parameter matrices, got shape {tuple(p.shape)} (ndim={p.ndim})"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon optimizer only supports 2D parameter matrices, got shape {tuple(p.shape)} (ndim={p.ndim})"
                    )
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p.data)

                state["step"] += 1
                momentum = state["momentum"]

                # Momentum accumulation
                momentum.lerp_(grad, 1.0 - beta)

                # Nesterov blend
                if nesterov:
                    update = grad.lerp(momentum, beta)
                else:
                    update = momentum.clone()

                # Newton-Schulz orthogonalization
                ortho_update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Aspect ratio scaling: max(1, d_out / d_in) ** 0.5
                d_out = ortho_update.size(0)
                d_in = ortho_update.size(1) if ortho_update.ndim > 1 else 1
                scale = max(1.0, d_out / d_in) ** 0.5
                ortho_update.mul_(scale)

                # Decoupled weight decay
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                # Update parameter
                p.data.add_(ortho_update.to(dtype=p.dtype, device=p.device), alpha=-lr)

        return loss
