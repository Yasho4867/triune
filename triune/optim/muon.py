"""Muon (MomentUm Orthogonalized by Newton-Schulz) optimizer.

Reference: Keller Jordan (2024), https://kellerjordan.github.io/posts/muon/
Designed for 2D hidden linear weights in neural networks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    scratch_buffer: Any = None,
    return_scratch: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Any]:
    """Newton-Schulz iteration to compute the approximate zeroth power (orthogonalization) of G.

    Delegates to the fused Triune kernel for in-place buffer reuse and Tensor Core GEMM optimization.
    """
    try:
        from triune.kernels.fast_muon import fast_zeropower_via_newtonschulz5

        return fast_zeropower_via_newtonschulz5(
            G, steps=steps, eps=eps, scratch_buffer=scratch_buffer, return_scratch=return_scratch
        )
    except Exception:
        pass

    if G.ndim != 2:
        raise ValueError(
            f"zeropower_via_newtonschulz5 requires a 2D matrix, got ndim={G.ndim} with shape {tuple(G.shape)}"
        )

    orig_device = G.device
    orig_dtype = G.dtype
    compute_device = orig_device

    # Run in bfloat16 on CUDA Tensor Cores for speedup if supported
    if compute_device.type == "cuda" and torch.cuda.is_bf16_supported():
        X = G.to(device=compute_device, dtype=torch.bfloat16).clone()
    else:
        X = G.to(device=compute_device, dtype=torch.float32).clone()

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

    res = X.to(device=orig_device, dtype=orig_dtype)
    if return_scratch:
        return res, None
    return res


class Muon(Optimizer):
    """Muon optimizer for 2D parameter matrices in hidden layers.

    Internally maintains standard SGD momentum and post-processes each update
    via Newton-Schulz orthogonalization with zero-allocation persistent scratch buffers.

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
        ns_steps: int = 3,
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

            buckets = {}
            all_moms = []
            all_grads = []
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

                # Fix H-5: ensure momentum is on the same device as grad
                if momentum.device != grad.device or momentum.dtype != grad.dtype:
                    state["momentum"] = momentum.to(device=grad.device, dtype=grad.dtype)
                    momentum = state["momentum"]

                all_moms.append(momentum)
                all_grads.append(grad)

                key = (p.shape, p.device, p.dtype)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append((p, grad, momentum, state))

            if not all_moms:
                continue

            # Fused multi-tensor momentum accumulation across all 2D matrices simultaneously
            torch._foreach_lerp_(all_moms, all_grads, 1.0 - beta)

            for (shape, device, dtype), items in buckets.items():
                K = len(items)
                d_out, d_in = shape
                scale = max(1.0, d_out / d_in) ** 0.5
                if K >= 2:
                    mom_stack = torch.stack([m for _, _, m, _ in items], dim=0)
                    if nesterov:
                        grad_stack = torch.stack([g for _, g, _, _ in items], dim=0)
                        update_stack = torch.lerp(grad_stack, mom_stack, beta)
                    else:
                        update_stack = mom_stack
                    try:
                        from triune.kernels.fast_muon import fast_zeropower_via_newtonschulz5_batched

                        ortho_stack = fast_zeropower_via_newtonschulz5_batched(update_stack, steps=ns_steps)
                    except Exception:
                        ortho_stack = torch.stack(
                            [zeropower_via_newtonschulz5(update_stack[i], steps=ns_steps) for i in range(K)], dim=0
                        )

                    ortho_stack = ortho_stack.to(dtype=dtype, device=device)
                    params_b = [p.data for p, _, _, _ in items]
                    if wd != 0.0:
                        torch._foreach_mul_(params_b, 1.0 - lr * wd)
                    torch._foreach_add_(params_b, list(ortho_stack.unbind(0)), alpha=-lr * scale)
                else:
                    p, grad, momentum, state = items[0]
                    update = grad.lerp(momentum, beta) if nesterov else momentum
                    ortho_update = zeropower_via_newtonschulz5(
                        update,
                        steps=ns_steps,
                    )

                    if wd != 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.data.add_(ortho_update.to(dtype=p.dtype, device=p.device), alpha=-lr * scale)

        return loss
