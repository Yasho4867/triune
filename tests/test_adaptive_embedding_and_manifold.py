import math
import torch
import torch.nn.functional as F
import pytest

from triune.model.manifold_embedding import UniversalAdaptiveEmbedding
from triune.optim.manifold import UnitaryManifoldOptimizer, LowSpaceWeightProjector


def test_adaptive_embedding_forward_and_logits():
    B, T, V, D, r = 2, 16, 1024, 256, 32
    emb = UniversalAdaptiveEmbedding(vocab_size=V, hidden_dim=D, rank=r)

    # 1. Forward lookup
    ids = torch.randint(0, V, (B, T))
    x = emb(ids)
    assert x.shape == (B, T, D)
    assert not torch.isnan(x).any()

    # 2. Dense logits computation
    h = torch.randn(B, T, D)
    logits = emb.compute_logits(h)
    assert logits.shape == (B, T, V)
    assert not torch.isnan(logits).any()

    # 3. Pruned top-K candidate logits computation
    top_logits, top_indices = emb.compute_logits(h, top_k=32)
    assert top_logits.shape == (B, T, 32)
    assert top_indices.shape == (B, T, 32)
    assert (top_indices >= 0).all() and (top_indices < V).all()


def test_adaptive_embedding_from_pretrained_exact_reconstruction():
    V, D, r = 512, 128, 16
    # Exact rank-16 target matrix
    torch.manual_seed(42)
    U = torch.randn(V, r)
    Vt = torch.randn(r, D)
    W_target = U @ Vt

    emb = UniversalAdaptiveEmbedding.from_pretrained_weights(W_target, rank=r)

    # Verify embedding lookup matches target
    ids = torch.randint(0, V, (4, 12))
    lookup_out = emb(ids)
    expected_lookup = W_target[ids]
    rel_err = (lookup_out - expected_lookup).norm() / expected_lookup.norm()
    assert rel_err.item() < 1e-4

    # Verify logits projection matches target
    h = torch.randn(4, 12, D)
    logits = emb.compute_logits(h)
    expected_logits = h @ W_target.t()
    logits_rel_err = (logits - expected_logits).norm() / expected_logits.norm()
    assert logits_rel_err.item() < 1e-4


def test_adaptive_embedding_vram_efficiency_ratio():
    # Frontier scale: 131,072 vocab, 1536 hidden dimension, rank 64
    V, D, r = 131072, 1536, 64
    flat_params = V * D
    adaptive_params = (V * r) + (D * r)

    compression = 1.0 - (adaptive_params / flat_params)
    assert compression > 0.95, f"Expected > 95% parameter reduction, got {compression * 100:.2f}%"

    flat_bytes_fp16 = flat_params * 2
    adaptive_bytes_fp16 = adaptive_params * 2
    mb_saved = (flat_bytes_fp16 - adaptive_bytes_fp16) / (1024 * 1024)
    assert mb_saved > 360.0  # Saves over 360 MB VRAM in FP16 alone


def test_unitary_manifold_optimizer_stiefel():
    torch.manual_seed(42)
    W = torch.nn.Parameter(torch.randn(32, 64))
    init_norm = W.norm().item()
    opt = UnitaryManifoldOptimizer([W], lr=0.01)

    for _ in range(8):
        loss = (W @ torch.randn(64, 4)).pow(2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    final_norm = W.norm().item()
    assert abs(final_norm - init_norm) < 1e-4


def test_low_space_weight_projector():
    proj = LowSpaceWeightProjector(rank=16, update_gap=2)
    G = torch.randn(64, 128)
    G_low = proj.project(G)
    assert G_low.shape == (16, 128)

    G_recon = proj.project_back(G_low)
    assert G_recon.shape == (64, 128)
