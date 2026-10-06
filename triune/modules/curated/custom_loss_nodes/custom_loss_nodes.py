"""Custom Loss & Router DAG Nodes Plugin for Triune Studio."""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class _DummyModule:
        def __init__(self, *args, **kwargs): pass
    class _NN:
        Module = _DummyModule
    nn = _NN()
    torch = None
    F = None

from triune.plugins.registry import register_node


@register_node(
    name="FocalCrossEntropyLoss",
    category="Loss",
    description="Focal loss focusing on hard tokens by down-weighting well-classified examples with gamma exponent.",
    inputs=["logits", "targets"],
    outputs=["loss"]
)
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


@register_node(
    name="CentroidBalanceLoss",
    category="Loss",
    description="Multi-expert load balancing penalty regularizing divergence from uniform routing distribution.",
    inputs=["router_probs"],
    outputs=["balance_loss"]
)
class CentroidBalanceLoss(nn.Module):
    def __init__(self, coeff: float = 0.01):
        super().__init__()
        self.coeff = coeff

    def forward(self, router_probs: torch.Tensor) -> torch.Tensor:
        expert_load = router_probs.mean(dim=0)
        target = 1.0 / router_probs.size(-1)
        divergence = ((expert_load - target) ** 2).sum()
        return self.coeff * divergence


@register_node(
    name="TemperatureAnnealer",
    category="Optimizer",
    description="Exponentially anneals Gumbel-Softmax exploration temperature from temp_max to temp_min over training steps.",
    inputs=["global_step"],
    outputs=["temperature"]
)
def anneal_temperature(global_step: int = 0, temp_max: float = 1.0, temp_min: float = 0.1, decay_steps: int = 5000) -> float:
    ratio = min(1.0, max(0.0, global_step / max(1, decay_steps)))
    return temp_max - ratio * (temp_max - temp_min)
