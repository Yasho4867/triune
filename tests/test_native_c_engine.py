"""Tests for the Native C/C++/AVX2/CUDA Engine Bridge (`triune.ops._c_bridge`) and NeuronScope."""
from __future__ import annotations

import torch

import triune
from triune.ops import (
    NativeTriuneEngine,
    c_centroid_steer_update_f32,
    c_compute_neuron_heatmap_f32,
    c_gla_recurrent_step_f32,
    c_route_tiers_f32,
    get_native_c_info,
    is_native_c_available,
    route_tiers,
    triangular_steer_update,
)
from triune.runtime.neuron_scope import NeuronScope


def test_native_c_info_schema():
    info = get_native_c_info()
    assert isinstance(info, dict)
    assert "available" in info
    assert "version" in info
    assert "cuda_runtime_bound" in info


def test_centroid_steer_update_and_router_ops():
    P = torch.ones((16, 32), dtype=torch.float32)
    G = torch.full((16, 32), 2.0, dtype=torch.float32)
    out_P = triangular_steer_update(P.clone(), G, steer_scale=0.5, momentum=0.9)
    assert out_P.shape == (16, 32)
    assert torch.allclose(out_P, torch.full((16, 32), 0.9 * 1.0 + 0.5 * (2.0 - 1.0)))

    logits = torch.tensor([[4.0, 0.1, -1.0], [-1.0, 3.0, 0.2], [-2.0, -1.0, 5.0]], dtype=torch.float32)
    probs, tiers = route_tiers(logits, reflex_threshold=0.70)
    assert probs.shape == (3, 3)
    assert tiers.tolist() == [0, 1, 2]


def test_native_c_direct_entrypoints_when_loaded():
    if not is_native_c_available():
        return
    info = get_native_c_info()
    assert "2.1.0" in str(info["version"])

    # 1. Native C AVX2 CentroidSteer update
    P = torch.ones((8, 64), dtype=torch.float32)
    G = torch.zeros((8, 64), dtype=torch.float32)
    assert c_centroid_steer_update_f32(P, G, steer_scale=0.25, momentum=0.8) is True
    assert torch.allclose(P, torch.full((8, 64), 0.55), atol=1e-5)

    # 2. Native C 3-Tier Router
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=torch.float32)
    res = c_route_tiers_f32(logits, reflex_threshold=0.70)
    assert res is not None
    probs, tiers = res
    assert tiers.tolist() == [0, 1, 2]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-2)

    # 3. Native C AVX2 Neuron Heatmap Binning
    act = torch.randn(16, 64, dtype=torch.float32)
    hm = c_compute_neuron_heatmap_f32(act, num_bins=32)
    assert hm is not None
    bins, rgba = hm
    assert len(bins) == 32
    assert len(rgba) == 32
    assert max(bins) <= 1.0001

    # 4. Native C AVX2 O(1) GLA Recurrent Step
    q = torch.randn(2, 16, dtype=torch.float32)
    k = torch.randn(2, 16, dtype=torch.float32)
    v = torch.randn(2, 16, dtype=torch.float32)
    g = torch.full((2, 16), -0.1, dtype=torch.float32)
    state = torch.zeros(2, 16, 16, dtype=torch.float32)
    gla_res = c_gla_recurrent_step_f32(q, k, v, g, state)
    assert gla_res is not None
    out, next_state = gla_res
    assert out.shape == (2, 16)
    assert next_state.shape == (2, 16, 16)

    # 5. Native C++/CUDA Arena & Graph Lifecycle
    eng = NativeTriuneEngine(vocab_size=1024, hidden_dim=128, num_layers=4, max_seq_len=64)
    assert eng.is_ready
    step_info = eng.step(batch_size=2, seq_len=32)
    assert step_info["ok"] is True
    assert step_info["tokens_processed"] == 64
    eng.close()


def test_neuron_scope_end_to_end():
    model = triune.Model("triune-nano", vocab_size=256)
    scope = NeuronScope(model)
    scope.attach()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        model(x)
    scope.record_routing(1)
    snap = scope.get_snapshot()
    assert len(snap["layers"]) > 0
    assert snap["active_tier"] in (0, 1, 2)
    assert "native_c_engine" in snap
    scope.detach()
