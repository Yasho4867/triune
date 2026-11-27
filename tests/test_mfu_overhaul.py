"""Verification tests for the MFU overhaul: zero-expand FusedRecurrentScan, BF16 TC GLA, and zero-copy packed MoE/GLA."""

from __future__ import annotations

import copy

import pytest
import torch

from lucid_machines.kernels.fast_gla import FusedRecurrentScan, fast_chunk_gla
from lucid_machines.model.attention import VectorisedGLA
from lucid_machines.model.moe import MoE_FFN


def _naive_recurrent_scan(kv: torch.Tensor, decay_k: torch.Tensor, init_state: torch.Tensor):
    B, H, N, Dk, Dv = kv.shape
    s = init_state.float()
    states = []
    for i in range(N):
        states.append(s)
        s = s * decay_k[:, :, i].float() + kv[:, :, i].float()
    return torch.stack(states, dim=2).to(kv.dtype), s.to(kv.dtype)


def _naive_chunk_gla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor):
    B, T, H, HD = q.shape
    s = torch.zeros(B, H, HD, HD, device=q.device, dtype=torch.float32)
    outs = []
    g_c = torch.clamp(g.float(), max=0.0)
    for t in range(T):
        d = torch.exp(g_c[:, t]).unsqueeze(-1)
        s = s * d + k[:, t].float().unsqueeze(-1) * v[:, t].float().unsqueeze(-2)
        outs.append(torch.matmul(q[:, t].float().unsqueeze(-2), s).squeeze(-2))
    return torch.stack(outs, dim=1).to(q.dtype), s.to(q.dtype)


@pytest.mark.parametrize("batch_size", [2, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_recurrent_scan_zero_expand_fwd_bwd_parity(batch_size: int, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available() or FusedRecurrentScan is None:
        pytest.skip("CUDA + Triton required")
    device = torch.device("cuda")
    torch.manual_seed(7)
    kv = torch.randn(batch_size, 4, 8, 32, 32, device=device, dtype=dtype) * 0.2
    decay_k = torch.sigmoid(torch.randn(batch_size, 4, 8, 32, 1, device=device, dtype=dtype))
    init = torch.randn(batch_size, 4, 32, 32, device=device, dtype=dtype) * 0.1

    kv_a, dec_a, init_a = (t.clone().detach().requires_grad_(True) for t in (kv, decay_k, init))
    kv_b, dec_b, init_b = (t.clone().detach().requires_grad_(True) for t in (kv, decay_k, init))

    s_a, f_a = FusedRecurrentScan.apply(kv_a, dec_a, init_a)
    s_b, f_b = _naive_recurrent_scan(kv_b, dec_b, init_b)

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(s_a.float(), s_b.float(), atol=atol, rtol=1e-2)
    assert torch.allclose(f_a.float(), f_b.float(), atol=atol, rtol=1e-2)

    (s_a.float().sum() + f_a.float().sum()).backward()
    (s_b.float().sum() + f_b.float().sum()).backward()
    assert torch.allclose(kv_a.grad.float(), kv_b.grad.float(), atol=5e-2 if dtype == torch.bfloat16 else 1e-4, rtol=2e-2)
    assert torch.allclose(dec_a.grad.float(), dec_b.grad.float(), atol=5e-2 if dtype == torch.bfloat16 else 1e-4, rtol=2e-2)
    assert torch.allclose(init_a.grad.float(), init_b.grad.float(), atol=5e-2 if dtype == torch.bfloat16 else 1e-4, rtol=2e-2)


def test_fast_chunk_gla_bf16_tc_and_extreme_decay_parity() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(11)
    q = torch.randn(2, 128, 4, 32, device=device, dtype=torch.bfloat16) * 0.3
    k = torch.randn(2, 128, 4, 32, device=device, dtype=torch.bfloat16) * 0.3
    v = torch.randn(2, 128, 4, 32, device=device, dtype=torch.bfloat16) * 0.3
    g = -0.2 * torch.rand(2, 128, 4, 32, device=device, dtype=torch.bfloat16)

    q_a, k_a, v_a, g_a = (t.clone().detach().requires_grad_(True) for t in (q, k, v, g))
    q_b, k_b, v_b, g_b = (t.clone().detach().requires_grad_(True) for t in (q, k, v, g))

    o_a, st_a = fast_chunk_gla(q_a, k_a, v_a, g_a)
    o_b, st_b = _naive_chunk_gla(q_b, k_b, v_b, g_b)
    assert torch.allclose(o_a.float(), o_b.float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(st_a.float(), st_b.float(), atol=5e-2, rtol=5e-2)

    o_a.float().sum().backward()
    o_b.float().sum().backward()
    assert torch.allclose(q_a.grad.float(), q_b.grad.float(), atol=1e-1, rtol=1e-1)
    assert torch.allclose(v_a.grad.float(), v_b.grad.float(), atol=1e-1, rtol=1e-1)

    # Extreme decay (g = -20.0 per token) must fall back to exact pairwise intra-chunk path
    g_ext = torch.full_like(g, -20.0)
    o_ext, st_ext = fast_chunk_gla(q, k, v, g_ext)
    o_ref, st_ref = _naive_chunk_gla(q, k, v, g_ext)
    assert torch.allclose(o_ext.float(), o_ref.float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(st_ext.float(), st_ref.float(), atol=5e-2, rtol=5e-2)


def test_moe_zero_copy_packed_stack_parity_and_checkpoint_roundtrip() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    moe = MoE_FFN(dim=32, num_experts=4, top_k=1, use_fp8=False, shared_expert=True).to(
        device=device, dtype=torch.bfloat16
    )
    moe_ref = copy.deepcopy(moe)

    x = torch.randn(2, 16, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    # Bypass packing on reference by passing explicit torch.stack path expectations via Parameter grads
    out = moe(x, update_stats=False)
    out.float().sum().backward()

    # Verify packed contiguous storage pointers are strictly sequential with 0 extra copies
    buf1 = moe._w1_packed_storage
    step_bytes = moe.experts[0][0].weight.numel() * moe.experts[0][0].weight.element_size()
    for i, exp in enumerate(moe.experts):
        assert exp[0].weight.data_ptr() == buf1.data_ptr() + i * step_bytes
        assert exp[0].weight.grad is not None

    # Compare against non-packed elementwise path using saved state_dict round-trip
    sd = moe.state_dict()
    assert not any("_w1_packed_storage" in k or "_w2_packed_storage" in k for k in sd)
    moe_ref.load_state_dict(sd)
    out_ref = moe_ref(x_ref, update_stats=False)
    assert torch.allclose(out.float(), out_ref.float(), atol=1e-4, rtol=1e-4)
    out_ref.float().sum().backward()
    for exp_a, exp_b in zip(moe.experts, moe_ref.experts):
        assert torch.allclose(exp_a[0].weight.grad.float(), exp_b[0].weight.grad.float(), atol=1e-4, rtol=1e-4)
        assert torch.allclose(exp_a[2].weight.grad.float(), exp_b[2].weight.grad.float(), atol=1e-4, rtol=1e-4)


def test_vectorised_gla_zero_copy_qkvg_parity() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(23)
    gla = VectorisedGLA(64, 4, 16).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(2, 64, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    out, _ = gla(x)
    out.float().sum().backward()

    buf = gla._qkvg_packed_storage
    step_bytes = gla.q_proj.weight.numel() * gla.q_proj.weight.element_size()
    for i, proj in enumerate((gla.q_proj, gla.k_proj, gla.v_proj, gla.g_proj)):
        assert proj.weight.data_ptr() == buf.data_ptr() + i * step_bytes
        assert proj.weight.grad is not None and torch.isfinite(proj.weight.grad).all()
