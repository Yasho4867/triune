from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.model.norms import RMSNorm


@dataclass
class JEVRoutingResult:
    should_exit: torch.Tensor
    confidence: torch.Tensor
    distance: torch.Tensor


class JEVLatentPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        d_pred: Optional[int] = None,
        eps: float = 1e-6,
        bias: bool = False,
        normalize_output: bool = True,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.d_pred = d_pred if d_pred is not None else max(1, dim // 2)
        self.eps = eps
        self.normalize_output = normalize_output
        self.use_residual = use_residual

        self.norm = RMSNorm(dim, eps=eps)
        self.w1 = nn.Linear(dim, self.d_pred, bias=bias)
        self.act = nn.GELU()
        self.w2 = nn.Linear(self.d_pred, dim, bias=bias)
        self.out_norm = RMSNorm(dim, eps=eps) if normalize_output else None

        self.down_proj = self.w1
        self.up_proj = self.w2
        self.in_norm = self.norm

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.w1.weight, a=math.sqrt(5))
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        orig_dtype = z.dtype
        z_norm = self.norm(z)
        h = self.act(self.w1(z_norm))
        z_pred = self.w2(h)

        if self.use_residual:
            z_pred = z_norm + z_pred

        if self.out_norm is not None:
            z_pred = self.out_norm(z_pred)

        return z_pred.to(orig_dtype)


class JEVObjective(nn.Module):
    def __init__(
        self,
        lambda_stp: float = 1.0,
        lambda_cov: float = 0.04,
        eps_tube: float = 0.1,
        eps_sim: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_stp = float(lambda_stp)
        self.lambda_cov = float(lambda_cov)
        self.eps_tube = float(eps_tube)
        self.eps_sim = float(eps_sim)

        self.last_sim_loss: Optional[float] = None
        self.last_stp_loss: Optional[float] = None
        self.last_cov_loss: Optional[float] = None
        self.last_total_loss: Optional[float] = None

    def compute_sim_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        target_sg = z_target.detach().float()
        pred_f = z_pred.float()
        cos_sim = F.cosine_similarity(pred_f, target_sg, dim=-1, eps=self.eps_sim)
        return (1.0 - cos_sim).mean()

    def compute_stp_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        target_sg = z_target.detach().float()
        pred_f = z_pred.float()
        dist_sq = torch.sum((pred_f - target_sg) ** 2, dim=-1)
        return torch.relu(dist_sq - (self.eps_tube ** 2)).mean()

    def compute_cov_loss(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.float().reshape(-1, z.size(-1))
        n_samples, dim = z_flat.shape
        if n_samples <= 1:
            return torch.zeros((), device=z.device, dtype=torch.float32, requires_grad=True)

        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (n_samples - 1)
        cov_off_diag = cov - torch.diag(torch.diagonal(cov))
        return (cov_off_diag ** 2).sum() / dim

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        l_sim = self.compute_sim_loss(z_pred, z_target)
        l_stp = self.compute_stp_loss(z_pred, z_target)
        l_cov = self.compute_cov_loss(z_pred)

        total_loss = l_sim + (self.lambda_stp * l_stp) + (self.lambda_cov * l_cov)

        self.last_sim_loss = float(l_sim.detach().item())
        self.last_stp_loss = float(l_stp.detach().item())
        self.last_cov_loss = float(l_cov.detach().item())
        self.last_total_loss = float(total_loss.detach().item())

        if return_dict:
            return total_loss, {
                "l_sim": l_sim,
                "l_stp": l_stp,
                "l_cov": l_cov,
                "total_loss": total_loss,
            }
        return total_loss


class JEVExitRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        threshold: float = 0.85,
        distance_metric: str = "cosine",
        predictor: Optional[JEVLatentPredictor] = None,
        eps_tube: float = 0.1,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if distance_metric not in ("cosine", "euclidean", "geodesic", "stp_tube", "learned"):
            raise ValueError(f"Unsupported distance_metric: {distance_metric}")

        self.dim = dim
        self.threshold = float(threshold)
        self.distance_metric = distance_metric
        self.eps_tube = float(eps_tube)
        self.predictor = predictor

        self.gate = nn.Linear(dim, 1)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate.bias)

    def compute_latent_distance(
        self,
        z: torch.Tensor,
        z_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_f = z.float()
        if z_ref is not None:
            ref_f = z_ref.float()
            if self.distance_metric == "cosine":
                cos_sim = F.cosine_similarity(z_f, ref_f, dim=-1, eps=1e-8)
                return (1.0 - cos_sim).clamp(min=0.0, max=2.0)
            elif self.distance_metric == "euclidean":
                return torch.norm(z_f - ref_f, p=2, dim=-1)
            elif self.distance_metric == "geodesic":
                cos_sim = F.cosine_similarity(z_f, ref_f, dim=-1, eps=1e-8).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                return torch.arccos(cos_sim)
            elif self.distance_metric == "stp_tube":
                dist_sq = torch.sum((z_f - ref_f) ** 2, dim=-1)
                return torch.relu(dist_sq - (self.eps_tube ** 2))

        return F.softplus(self.gate(z)).squeeze(-1)

    def compute_latent_confidence(
        self,
        z: torch.Tensor,
        z_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_ref is not None and self.distance_metric == "cosine":
            return F.cosine_similarity(z.float(), z_ref.float(), dim=-1, eps=1e-8).clamp(0.0, 1.0)
        elif z_ref is not None and self.distance_metric in ("euclidean", "stp_tube"):
            return torch.exp(-self.compute_latent_distance(z, z_ref))

        return torch.sigmoid(self.gate(z)).squeeze(-1)

    def evaluate_latent_exit(
        self,
        z: torch.Tensor,
        z_target: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        thresh = float(threshold) if threshold is not None else self.threshold
        z_pred = self.predictor(z) if self.predictor is not None else z
        confidence = self.compute_latent_confidence(z_pred, z_target)
        return confidence >= thresh, confidence

    def forward(
        self,
        z: torch.Tensor,
        z_target: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> JEVRoutingResult:
        thresh = float(threshold) if threshold is not None else self.threshold
        z_pred = self.predictor(z) if self.predictor is not None else z
        distance = self.compute_latent_distance(z_pred, z_target)
        confidence = self.compute_latent_confidence(z_pred, z_target)
        return JEVRoutingResult(
            should_exit=confidence >= thresh,
            confidence=confidence,
            distance=distance,
        )


LatentPredictor = JEVLatentPredictor
JointEmbeddingObjective = JEVObjective
ContinuousLatentExitRouter = JEVExitRouter
TriuneLatentPredictor = JEVLatentPredictor
TriuneLatentObjective = JEVObjective
TriuneContinuousExitRouter = JEVExitRouter

__all__ = [
    "JEVLatentPredictor",
    "JEVObjective",
    "JEVExitRouter",
    "JEVRoutingResult",
    "LatentPredictor",
    "JointEmbeddingObjective",
    "ContinuousLatentExitRouter",
    "TriuneLatentPredictor",
    "TriuneLatentObjective",
    "TriuneContinuousExitRouter",
]
