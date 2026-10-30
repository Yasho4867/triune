"""Unit test for MoE batched BMM vs sequential execution parity.

Verifies:
1. Exact mathematical equivalence in forward outputs between sequential expert loop and batched BMM.
2. Non-zero biases on linear layers (b1 and b2) are preserved and correctly added.
3. Gradient equivalence (backward pass) on inputs, weights, and biases.
4. Seamless fallback when biases are None.
"""

import copy
import pytest
import torch
import torch.nn as nn
from triune.model.moe import MoE_FFN


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU testing")
def test_moe_batched_parity_with_biases():
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    dim = 64
    num_experts = 4
    top_k = 2
    B, T = 2, 8

    # 1. Instantiate model with standard nn.Linear experts (which have bias=True)
    moe_batched = MoE_FFN(dim=dim, num_experts=num_experts, top_k=top_k, shared_expert=False).to(device=device, dtype=torch.float32)
    # Ensure biases are non-zero
    for e in moe_batched.experts:
        nn.init.uniform_(e[0].bias, 0.1, 1.0)
        nn.init.uniform_(e[2].bias, 0.1, 1.0)

    # 2. Deepcopy to create an identical model forced to run sequential path
    moe_seq = copy.deepcopy(moe_batched)

    # Inputs with requires_grad=True
    x_batched = torch.randn(B, T, dim, device=device, dtype=torch.float32, requires_grad=True)
    x_seq = x_batched.clone().detach().requires_grad_(True)

    # Forward batched
    moe_batched.train()
    out_batched = moe_batched(x_batched, update_stats=False)

    # Forward sequential: monkeypatch can_batch_bmm check by setting use_fp4 temporarily or patching
    moe_seq.train()
    # Force sequential path
    orig_modules = moe_seq.experts
    try:
        # Setting use_fp4 = True on moe_seq triggers the sequential fallback path without changing linear dtypes
        moe_seq.use_fp4 = True
        out_seq = moe_seq(x_seq, update_stats=False)
    finally:
        moe_seq.use_fp4 = False

    # Check forward output parity
    forward_diff = (out_batched - out_seq).abs().max().item()
    assert forward_diff < 1e-4, f"Forward outputs differ by {forward_diff:.2e}"

    # Backward parity
    loss_batched = out_batched.sum()
    loss_batched.backward()

    loss_seq = out_seq.sum()
    loss_seq.backward()

    # Check input gradients
    x_grad_diff = (x_batched.grad - x_seq.grad).abs().max().item()
    assert x_grad_diff < 1e-4, f"Input gradients differ by {x_grad_diff:.2e}"

    # Check weight and bias gradients for each expert
    for e_idx in range(num_experts):
        w1_diff = (moe_batched.experts[e_idx][0].weight.grad - moe_seq.experts[e_idx][0].weight.grad).abs().max().item()
        b1_diff = (moe_batched.experts[e_idx][0].bias.grad - moe_seq.experts[e_idx][0].bias.grad).abs().max().item()
        w2_diff = (moe_batched.experts[e_idx][2].weight.grad - moe_seq.experts[e_idx][2].weight.grad).abs().max().item()
        b2_diff = (moe_batched.experts[e_idx][2].bias.grad - moe_seq.experts[e_idx][2].bias.grad).abs().max().item()

        assert w1_diff < 1e-4, f"Expert {e_idx} W1 grad diff {w1_diff:.2e}"
        assert b1_diff < 1e-4, f"Expert {e_idx} b1 grad diff {b1_diff:.2e}"
        assert w2_diff < 1e-4, f"Expert {e_idx} W2 grad diff {w2_diff:.2e}"
        assert b2_diff < 1e-4, f"Expert {e_idx} b2 grad diff {b2_diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU testing")
def test_moe_batched_parity_bias_none():
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    dim = 64
    num_experts = 4
    top_k = 2
    B, T = 2, 8

    # Model with bias=None
    moe_batched = MoE_FFN(dim=dim, num_experts=num_experts, top_k=top_k, shared_expert=False).to(device=device, dtype=torch.float32)
    for e in moe_batched.experts:
        e[0].bias = None
        e[2].bias = None

    moe_seq = copy.deepcopy(moe_batched)

    x_batched = torch.randn(B, T, dim, device=device, dtype=torch.float32, requires_grad=True)
    x_seq = x_batched.clone().detach().requires_grad_(True)

    moe_batched.train()
    out_batched = moe_batched(x_batched, update_stats=False)

    moe_seq.train()
    moe_seq.use_fp4 = True
    out_seq = moe_seq(x_seq, update_stats=False)

    forward_diff = (out_batched - out_seq).abs().max().item()
    assert forward_diff < 1e-4, f"Forward outputs differ by {forward_diff:.2e} when bias=None"

    loss_batched = out_batched.sum()
    loss_batched.backward()

    loss_seq = out_seq.sum()
    loss_seq.backward()

    x_grad_diff = (x_batched.grad - x_seq.grad).abs().max().item()
    assert x_grad_diff < 1e-4, f"Input gradients differ by {x_grad_diff:.2e} when bias=None"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU testing")
def test_moe_mixed_bias_fallback():
    """Verifies that mixed bias configurations across experts safely fall back to sequential loop."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    dim = 64
    num_experts = 4
    top_k = 2
    B, T = 2, 8

    # Create MoE where expert 0 has no bias but expert 1-3 have biases
    moe = MoE_FFN(dim=dim, num_experts=num_experts, top_k=top_k, shared_expert=False).to(device=device, dtype=torch.float32)
    moe.experts[0][0].bias = None

    x = torch.randn(B, T, dim, device=device, dtype=torch.float32, requires_grad=True)
    moe.train()
    out = moe(x, update_stats=False)
    assert out.shape == (B, T, dim)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert moe.experts[1][0].bias.grad is not None
    print("✅ MoE mixed bias fallback handled cleanly!")
