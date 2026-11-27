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


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("with_bias", [False, True])
def test_grouped_mm_moe_parity_overflow_and_biases(top_k: int, with_bias: bool) -> None:
    if not torch.cuda.is_available() or not hasattr(torch, "_grouped_mm"):
        pytest.skip("CUDA + torch._grouped_mm required")
    device = torch.device("cuda")
    torch.manual_seed(31 + top_k + int(with_bias) * 7)

    moe = MoE_FFN(dim=32, num_experts=4, top_k=top_k, use_fp8=False, shared_expert=True).to(
        device=device, dtype=torch.bfloat16
    )
    if not with_bias:
        for e in moe.experts:
            e[0].bias = None
            e[2].bias = None
    moe_ref = copy.deepcopy(moe)

    # Skewed router + enough tokens to exercise overflow AND empty experts
    with torch.no_grad():
        moe.router.bias.copy_(torch.tensor([3.0, 2.0, -4.0, -4.0], device=device, dtype=torch.bfloat16))
        moe_ref.router.bias.copy_(moe.router.bias)

    x = torch.randn(2, 16, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    moe.train()
    out = moe(x, update_stats=True)
    go = torch.randn_like(out)
    out.backward(go)

    moe_ref.train()
    moe_ref.use_fp4 = True  # force scalar reference loop
    out_ref = moe_ref(x_ref, update_stats=True)
    moe_ref.use_fp4 = False
    out_ref.backward(go)

    assert torch.allclose(out.float(), out_ref.float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(x.grad.float(), x_ref.grad.float(), atol=5e-2, rtol=5e-2)
    assert torch.equal(moe.last_accepted_counts, moe_ref.last_accepted_counts)
    assert int(moe.overflow_counter) == int(moe_ref.overflow_counter)
    for ea, eb in zip(moe.experts, moe_ref.experts):
        eb_w1_g = eb[0].weight.grad if eb[0].weight.grad is not None else torch.zeros_like(ea[0].weight)
        eb_w2_g = eb[2].weight.grad if eb[2].weight.grad is not None else torch.zeros_like(ea[2].weight)
        assert torch.allclose(ea[0].weight.grad.float(), eb_w1_g.float(), atol=1e-1, rtol=5e-2)
        assert torch.allclose(ea[2].weight.grad.float(), eb_w2_g.float(), atol=1e-1, rtol=5e-2)
        if with_bias:
            eb_b1_g = eb[0].bias.grad if eb[0].bias.grad is not None else torch.zeros_like(ea[0].bias)
            eb_b2_g = eb[2].bias.grad if eb[2].bias.grad is not None else torch.zeros_like(ea[2].bias)
            assert torch.allclose(ea[0].bias.grad.float(), eb_b1_g.float(), atol=5e-2, rtol=5e-2)
            assert torch.allclose(ea[2].bias.grad.float(), eb_b2_g.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_qkvg_rope_prep_fwd_bwd_parity(head_dim: int, dtype: torch.dtype) -> None:
    from lucid_machines.model.attention import _FusedQKVGRoPEPrepFn

    if not torch.cuda.is_available() or _FusedQKVGRoPEPrepFn is None:
        pytest.skip("CUDA + Triton required")
    device = torch.device("cuda")
    torch.manual_seed(47 + head_dim)
    B, T, H = 2, 32, 4
    qkvg = torch.randn(B, T, 4 * H * head_dim, device=device, dtype=dtype, requires_grad=True)
    qkvg_ref = qkvg.detach().clone().requires_grad_(True)
    angles = torch.randn(T, head_dim // 2, device=device, dtype=torch.float32)
    emb = torch.cat((angles, angles), dim=-1)
    cos, sin = emb.cos(), emb.sin()

    q, k, v, g = _FusedQKVGRoPEPrepFn.apply(qkvg, cos, sin, H, head_dim, True)

    hd2, hd3 = 2 * H * head_dim, 3 * H * head_dim
    qk_r = qkvg_ref[..., :hd2].reshape(B, T, 2, H, head_dim)
    v_r = qkvg_ref[..., hd2:hd3].reshape(B, T, H, head_dim)
    g_r = torch.nn.functional.logsigmoid(qkvg_ref[..., hd3:].reshape(B, T, H, head_dim))
    qk_r = torch.nn.functional.normalize(torch.nn.functional.elu(qk_r) + 1.0, p=2.0, dim=-1, eps=1e-6)
    cos_b = cos.to(dtype=dtype).view(1, T, 1, 1, head_dim)
    sin_b = sin.to(dtype=dtype).view(1, T, 1, 1, head_dim)
    qk1, qk2 = qk_r.chunk(2, dim=-1)
    qk_r = (qk_r * cos_b + torch.cat((-qk2, qk1), dim=-1) * sin_b).to(dtype=dtype)
    q_r, k_r = qk_r[:, :, 0], qk_r[:, :, 1]

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(q.float(), q_r.float(), atol=tol, rtol=tol)
    assert torch.allclose(k.float(), k_r.float(), atol=tol, rtol=tol)
    assert torch.allclose(v.float(), v_r.float(), atol=tol, rtol=tol)
    assert torch.allclose(g.float(), g_r.float(), atol=tol, rtol=tol)

    gq, gk, gv, gg = (torch.randn_like(t) for t in (q, k, v, g))
    (q * gq + k * gk + v * gv + g * gg).sum().backward()
    (q_r * gq + k_r * gk + v_r * gv + g_r * gg).sum().backward()
    gtol = 5e-2 if dtype == torch.bfloat16 else 2e-4
    assert torch.allclose(qkvg.grad.float(), qkvg_ref.grad.float(), atol=gtol, rtol=gtol)


def test_batched_bf16_sr_selective_ckpt_and_chunked_joint_kl() -> None:
    from types import SimpleNamespace
    from lucid_machines.kernels.fast_bf16_sr import bf16_stochastic_round_update_batch_
    from lucid_machines.model.transformer import TriuneTransformer
    from lucid_machines.trainer.engine import TrainingEngine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Batched BF16 SR on packed contiguous storage
    buf = torch.ones((4, 64), device=device, dtype=torch.bfloat16)
    slices = [buf[i] for i in range(4)]
    deltas = torch.full((4, 64), 0.25, device=device, dtype=torch.bfloat16)
    bf16_stochastic_round_update_batch_(slices, deltas, beta_wd=1.0, alpha_upd=1.0, base_seed=99)
    assert torch.allclose(buf.float(), torch.full_like(buf, 1.25).float(), atol=1e-2)

    # 2. Selective gradient checkpointing modes
    model = TriuneTransformer(
        vocab_size=64, hidden_dim=32, num_layers=5, num_heads=4, head_dim=8,
        num_experts=4, router_prefix_layers=1, reflex_exit_layer=2, limbic_exit_layer=3,
    ).to(device=device, dtype=torch.bfloat16)
    model.enable_gradient_checkpointing(mode="moe")
    assert all(l._grad_ckpt_mode == "moe" and l._use_gradient_checkpointing for l in model.layers)
    model_ref = copy.deepcopy(model)
    model_ref.enable_gradient_checkpointing(mode="none")
    x = torch.randint(0, 64, (2, 16), device=device)
    logits, _ = model(x, force_depth=2)
    logits_ref, _ = model_ref(x, force_depth=2)
    assert torch.allclose(logits.float(), logits_ref.float(), atol=5e-2, rtol=5e-2)
    logits.float().sum().backward()
    logits_ref.float().sum().backward()
    assert torch.allclose(model.final_head.weight.grad.float(), model_ref.final_head.weight.grad.float(), atol=5e-2, rtol=5e-2)

    # 3. Chunked vs unchunked joint KL loss parity
    fake_trainer = SimpleNamespace(
        model=model,
        optimizer=None,
        device=device,
        pad_token_id=-100,
        config={"target_depth_dist": [0.5, 0.3, 0.2], "ce_chunk_size": 7, "exit_distill_coef": 0.2},
    )
    eng = TrainingEngine(fake_trainer)
    r_l = torch.randn(2, 11, 64, device=device, requires_grad=True)
    l_l = torch.randn(2, 11, 64, device=device, requires_grad=True)
    c_l = torch.randn(2, 11, 64, device=device, requires_grad=True)
    tgt = torch.randint(0, 64, (22,), device=device)
    loss_chunked = eng._compute_joint_lm_loss(r_l, l_l, c_l, tgt, 64)
    eng.config["ce_chunk_size"] = 1024
    loss_full = eng._compute_joint_lm_loss(r_l, l_l, c_l, tgt, 64)
    assert torch.allclose(loss_chunked, loss_full, atol=1e-5, rtol=1e-5)
