import torch
from torch.optim import Optimizer


class LowSpaceWeightProjector:
    def __init__(self, rank: int = 16, update_gap: int = 50, scale: float = 1.0):
        self.rank = rank
        self.update_gap = update_gap
        self.scale = scale
        self.step_count = 0
        self.projection = None
        self.side = "left"

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        m, n = grad.shape
        self.side = "left" if m <= n else "right"
        if self.projection is None or self.step_count % self.update_gap == 0:
            q = min(self.rank + 8, m, n)
            device, dtype = grad.device, grad.dtype
            if self.side == "left":
                omega = torch.randn((n, q), device=device, dtype=dtype)
                y = torch.mm(grad, omega)
                q_mat, _ = torch.linalg.qr(y.float())
                self.projection = q_mat[:, : self.rank].to(dtype)
            else:
                omega = torch.randn((m, q), device=device, dtype=dtype)
                y = torch.mm(grad.t(), omega)
                q_mat, _ = torch.linalg.qr(y.float())
                self.projection = q_mat[:, : self.rank].to(dtype)

        self.step_count += 1
        if self.side == "left":
            return torch.mm(self.projection.t(), grad) * self.scale
        return torch.mm(grad, self.projection) * self.scale

    def project_back(self, low_rank_update: torch.Tensor) -> torch.Tensor:
        if self.side == "left":
            return torch.mm(self.projection, low_rank_update)
        return torch.mm(low_rank_update, self.projection.t())


class UnitaryManifoldOptimizer(Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9, eps: float = 1e-8):
        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue

                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p.data)

                state["step"] += 1
                mom = state["momentum"]
                mom.mul_(beta).add_(grad, alpha=1.0 - beta)

                w = p.data
                m, n = w.shape
                norm_sq = (w * w).sum().clamp_min(eps)

                if m <= n:
                    omega = (torch.mm(mom, w.t()) - torch.mm(w, mom.t())) / norm_sq
                    eye = torch.eye(m, device=w.device, dtype=w.dtype)
                    cayley_inv = torch.linalg.inv(eye - (0.5 * lr) * omega)
                    cayley_fwd = eye + (0.5 * lr) * omega
                    r_mat = torch.mm(cayley_inv, cayley_fwd)
                    p.data.copy_(torch.mm(r_mat, w))
                else:
                    omega = (torch.mm(mom.t(), w) - torch.mm(w.t(), mom)) / norm_sq
                    eye = torch.eye(n, device=w.device, dtype=w.dtype)
                    cayley_inv = torch.linalg.inv(eye - (0.5 * lr) * omega)
                    cayley_fwd = eye + (0.5 * lr) * omega
                    r_mat = torch.mm(cayley_inv, cayley_fwd)
                    p.data.copy_(torch.mm(w, r_mat))

        return loss


QuantumUnitaryOptimizer = UnitaryManifoldOptimizer
