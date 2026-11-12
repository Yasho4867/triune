import torch
import torch.nn.functional as F
from triune.kernels.fast_muon import fast_zeropower_via_newtonschulz5, fast_zeropower_via_newtonschulz5_batched
from triune.ops.centroid_steer import polar_stiefel_retraction, subspace_projection, batched_subspace_projection
from triune.kernels.fast_gla import _run_gla_chunks
from triune.model.adapters import generate_adaptive
from triune.optim.manifold import LowSpaceWeightProjector, UnitaryManifoldOptimizer, QuantumUnitaryOptimizer


def test_muon_remez_superlinear():
    G = torch.randn(64, 64)
    O = fast_zeropower_via_newtonschulz5(G, steps=3)
    diff = (O @ O.t() - torch.eye(64)).abs().mean().item()
    assert diff < 0.05, f"Remez-minimax NS5 expected error < 0.05, got {diff}"

    # Batched parity
    G_batch = torch.randn(4, 32, 48)
    O_b = fast_zeropower_via_newtonschulz5_batched(G_batch, steps=3)
    for i in range(4):
        O_s = fast_zeropower_via_newtonschulz5(G_batch[i].clone(), steps=3)
        assert (O_b[i] - O_s).abs().max().item() < 1e-4


def test_centroid_steer_svd_and_polar_retraction():
    # 1. Polar Stiefel Retraction
    P = torch.randn(128, 16)
    P_ortho = polar_stiefel_retraction(P)
    err = (P_ortho.t() @ P_ortho - torch.eye(16)).abs().max().item()
    assert err < 1e-5, f"Polar Stiefel Retraction orthogonality error too high: {err}"

    # 2. Randomized SVD Subspace Projection
    grad = torch.randn(128, 256)
    proj = subspace_projection(grad, rank=16, niter=2)
    assert proj.shape == (128, 16)
    err_proj = (proj.t() @ proj - torch.eye(16)).abs().max().item()
    assert err_proj < 1e-4, f"SVD subspace projection orthogonality error: {err_proj}"

    # Batched subspace projection
    grads = torch.randn(2, 128, 256)
    proj_b, side = batched_subspace_projection(grads, rank=16, niter=2)
    assert proj_b.shape == (2, 128, 16)


def test_gla_unfactored_decay():
    B, T, H, HD = 1, 64, 2, 32
    q = torch.randn(B, T, H, HD)
    k = torch.randn(B, T, H, HD)
    v = torch.randn(B, T, H, HD)
    g = F.logsigmoid(torch.randn(B, T, H, HD))
    state = torch.zeros(B, H, HD, HD)
    out, next_state = _run_gla_chunks(q, k, v, g, state, C=64)
    assert out.shape == (B, T, H, HD)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_quantum_unitary_optimizer_and_low_space():
    W = torch.nn.Parameter(torch.randn(32, 64))
    init_norm = W.norm().item()
    opt = QuantumUnitaryOptimizer([W], lr=0.02)
    for _ in range(5):
        loss = (W @ torch.randn(64, 1)).pow(2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    final_norm = W.norm().item()
    assert abs(final_norm - init_norm) < 1e-4, "QuantumUnitaryOptimizer must preserve norm"

    proj = LowSpaceWeightProjector(rank=8, update_gap=2)
    G = torch.randn(32, 64)
    G_low = proj.project(G)
    assert G_low.shape == (8, 64)
    G_rec = proj.project_back(G_low)
    assert G_rec.shape == (32, 64)
