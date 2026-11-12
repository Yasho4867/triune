import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalAdaptiveEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 131072, hidden_dim: int = 1536, rank: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.fingerprints = nn.Embedding(vocab_size, rank)
        self.basis = nn.Linear(rank, hidden_dim, bias=False)

        nn.init.normal_(self.fingerprints.weight, std=1.0 / math.sqrt(rank))
        nn.init.orthogonal_(self.basis.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.basis(self.fingerprints(input_ids))

    def compute_logits(self, hidden_states: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
        h_spec = hidden_states @ self.basis.weight
        if top_k is None or top_k >= self.vocab_size:
            return F.linear(h_spec, self.fingerprints.weight)

        scores = F.linear(h_spec, self.fingerprints.weight)
        topk_indices = torch.topk(scores, k=top_k, dim=-1).indices
        return scores.gather(-1, topk_indices), topk_indices

    @classmethod
    def from_pretrained_weights(cls, weight: torch.Tensor, rank: int = 64) -> "UniversalAdaptiveEmbedding":
        V, D = weight.shape
        q = min(rank + 8, V, D)
        omega = torch.randn((D, q), device=weight.device, dtype=weight.dtype)
        y = torch.mm(weight, omega)
        Q, _ = torch.linalg.qr(y.float())
        B = torch.mm(Q.t(), weight.float())
        U_b, S_b, Vh_b = torch.linalg.svd(B, full_matrices=False)
        u_core = U_b[:, :rank]
        s_core = torch.diag(S_b[:rank])
        v_core = Vh_b[:rank, :]

        fp_init = torch.mm(Q, u_core) @ torch.sqrt(s_core)
        basis_init = (torch.sqrt(s_core) @ v_core).t()

        mod = cls(vocab_size=V, hidden_dim=D, rank=rank)
        mod.fingerprints.weight.data.copy_(fp_init.to(weight.dtype))
        mod.basis.weight.data.copy_(basis_init.to(weight.dtype))
        return mod
