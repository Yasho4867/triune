import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmaxRouter(nn.Module):
    """
    A mathematically rigorous Gumbel-Softmax Straight-Through router.
    Enables end-to-end differentiability of discrete early-exit decisions
    while maintaining physical execution sparsity in the forward pass.
    """
    def __init__(self, hidden_dim: int, target_depth_dist=(0.34, 0.33, 0.33), balance_coef: float | None = None):
        super().__init__()
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Outputs: logits for [Reflex, Limbic, Cortex]
        )
        target_tensor = torch.as_tensor(target_depth_dist, dtype=torch.float32)
        if target_tensor.ndim != 1 or target_tensor.shape[0] != 3:
            raise ValueError(f"target_depth_dist must have 3 elements, got {target_depth_dist}")
        if abs(target_tensor.sum().item() - 1.0) > 1e-4:
            raise ValueError(f"target_depth_dist must sum to 1.0, got {target_tensor.sum().item():.4f}")
        self.register_buffer("target_dist", target_tensor)

    def forward(self, x: torch.Tensor, temperature: float = 1.0, force_depth: int = None):
        """
        Args:
            x (Tensor): Input activations of shape [B, T, D] from prefix layers.
            temperature (float): Relaxation temperature for Gumbel Softmax (must be > 0 and finite).
            force_depth (int, optional): Force routing to exit 0, 1, or 2 if specified.
        Returns:
            logits (Tensor): Raw router logits [B, 3].
            y_route (Tensor): One-hot-like tensor of shape [B, 3] indicating selected path.
            balance_loss (Tensor): Load-balancing regularization loss (unweighted MSE).
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"Gumbel-Softmax temperature must be a finite positive number, got {temperature}")
        B, T, D = x.shape
        pooled = x.mean(dim=1)  # Sequence pooling: [B, D]
        logits = self.router_mlp(pooled)  # Routing logits: [B, 3]

        if force_depth is not None:
            y_route = F.one_hot(torch.full((B,), force_depth, device=x.device, dtype=torch.long), num_classes=3).float()
            balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return logits, y_route, balance_loss

        if self.training:
            # 1. Sample standard Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0,1)
            unif = torch.rand_like(logits)
            gumbels = -torch.log(-torch.log(unif + 1e-20) + 1e-20)
            
            # 2. Compute continuous relaxed softmax
            y_soft = F.softmax((logits + gumbels) / temperature, dim=-1)
            
            # 3. Discretization using Straight-Through (ST) estimator
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            
            # Straight-through gradient identity block
            y_route = y_hard + y_soft - y_soft.detach()
        else:
            # Greedy inference path (deterministic argmax)
            index = logits.argmax(dim=-1)
            y_route = F.one_hot(index, num_classes=3).float()

        # Compute Load Balancing Regularization loss natively in the forward pass (unweighted MSE)
        routing_mean = y_route.mean(dim=0)
        balance_loss = (routing_mean - self.target_dist).pow(2).mean()

        return logits, y_route, balance_loss
