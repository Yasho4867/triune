import unittest
import torch
import torch.nn as nn

from triune.model.transformer import TriuneTransformer
from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig
from triune.optim.centroid import CentroidSteerOptimizer


def test_layer_streaming_engine_lifecycle():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 1. Build a model with 6 layers, 4 experts
    model = TriuneTransformer(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        head_dim=64,
        num_experts=4,
        router_prefix_layers=2,
        reflex_exit_layer=3,
        limbic_exit_layer=4,
        use_fp8=False,
    )

    # 2. Enable layer streaming
    engine = model.enable_layer_streaming(device=device, chunk_size=1, pin_memory=True, instant_optimizer=False)
    assert engine.is_attached
    assert model._layer_streaming_active

    # 3. Check parameter placement:
    # Root components on GPU
    assert model.token_embed.weight.device.type == "cuda"
    assert model.final_head.weight.device.type == "cuda"

    # Layer components on CPU and pinned
    for layer in model.layers:
        for p in layer.parameters():
            assert p.device.type == "cpu"
            assert p.data.is_pinned()

    initial_vram_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"\nInitial VRAM allocated after streaming attach: {initial_vram_mb:.2f} MB")
    assert initial_vram_mb < 200.0, f"Expected initial VRAM < 200MB, got {initial_vram_mb} MB"

    # 4. Forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    logits, route_logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, 1000)
    assert route_logits.shape == (batch_size, 3)

    forward_vram_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"Forward pass completed. VRAM allocated: {forward_vram_mb:.2f} MB")

    # 5. Backward pass with gradient offload verification
    loss = logits.sum() + route_logits.sum()
    loss.backward()

    # Check that layer parameters offloaded their gradients to CPU
    has_cpu_grad = False
    for layer in model.layers:
        for p in layer.parameters():
            if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                has_cpu_grad = True
                assert p._cpu_grad.device.type == "cpu"
                # GPU gradient should be cleared to save VRAM
                assert p.grad is None
    assert has_cpu_grad, "Expected layer parameters to have _cpu_grad populated"

    # 6. Optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Restore cpu grads to param.grad for optimizer step
    for p in model.parameters():
        if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
            p.grad = p._cpu_grad

    optimizer.step()

    # Clean up cpu grads
    for p in model.parameters():
        if hasattr(p, "_cpu_grad"):
            p.grad = None
            p._cpu_grad = None

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak VRAM during full forward/backward/step: {peak_vram_mb:.2f} MB")
    assert peak_vram_mb < 500.0, f"Expected peak VRAM < 500MB, got {peak_vram_mb} MB"
    print("✅ Layer Streaming Engine unit test passed successfully!")


def test_layer_streaming_with_centroid_optimizer():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    model = TriuneTransformer(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        head_dim=64,
        num_experts=4,
        router_prefix_layers=2,
        reflex_exit_layer=3,
        limbic_exit_layer=4,
        use_fp8=False,
    )
    engine = model.enable_layer_streaming(device=device, chunk_size=1, pin_memory=False, instant_optimizer=False)
    optimizer = CentroidSteerOptimizer(
        model, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05, rank=32, update_gap=5
    )

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    logits, route_logits = model(input_ids)
    loss = logits.sum() + route_logits.sum()
    loss.backward()

    for p in model.parameters():
        if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
            p.grad = p._cpu_grad

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    for p in model.parameters():
        if hasattr(p, "_cpu_grad"):
            p.grad = None
            p._cpu_grad = None

    print("✅ CentroidSteerOptimizer with Layer Streaming passed successfully!")


def test_layer_streaming_optimizer_gpu_execution():
    if not torch.cuda.is_available():
        return

    torch.set_default_dtype(torch.bfloat16)
    try:
        model = TriuneTransformer(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=6,
            num_heads=4,
            head_dim=64,
            num_experts=4,
            router_prefix_layers=2,
            reflex_exit_layer=3,
            limbic_exit_layer=4,
            use_fp8=False,
        )
    finally:
        torch.set_default_dtype(torch.float32)

    device = torch.device("cuda")
    optimizer = CentroidSteerOptimizer(
        model, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05, rank=32, update_gap=5
    )
    engine = model.enable_layer_streaming(
        device=device, chunk_size=1, pin_memory=False, fp8_weights=True
    )
    engine.set_optimizer(optimizer)

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    engine.zero_grad()
    logits, route_logits = model(input_ids)
    loss = logits.sum() + route_logits.sum()
    loss.backward()

    # Step optimizer on GPU Tensor Cores layer-by-layer
    engine.step_streaming_optimizer(optimizer=optimizer, grad_clip=1.0)

    # Verify all gradients were cleared and freed
    for p in model.parameters():
        assert not hasattr(p, "_cpu_grad") or p._cpu_grad is None
        assert p.grad is None

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"Peak VRAM during Layer-Streaming GPU Optimizer: {peak_vram_mb:.2f} MB")
    assert peak_vram_mb < 500.0, f"Expected peak VRAM < 500MB, got {peak_vram_mb} MB"
    print("✅ Layer-Streaming GPU Optimizer passed successfully!")


def test_streaming_gradient_numerical_parity():
    """Verify exact numerical gradient equivalence between streaming and non-streaming models."""
    if not torch.cuda.is_available():
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    kwargs = dict(
        vocab_size=100,
        hidden_dim=64,
        num_layers=4,
        num_heads=2,
        head_dim=32,
        num_experts=2,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
        use_fp4=False,
        use_fp8=False,
    )

    model_std = TriuneTransformer(**kwargs).to(device).eval()
    model_str = TriuneTransformer(**kwargs).eval()
    model_str.load_state_dict(model_std.state_dict())

    engine = model_str.enable_layer_streaming(
        device=device, chunk_size=1, pin_memory=False, instant_optimizer=False
    )

    x = torch.randint(0, 100, (2, 8), device=device)

    # Standard backward with all exits enabled (eval mode guarantees zero Gumbel noise drift)
    model_std.zero_grad()
    r1, l1, c1, rt1 = model_std.forward_all_exits(x)
    loss_std = r1.sum() + l1.sum() + c1.sum() + rt1.sum()
    loss_std.backward()

    # Streaming backward with all exits enabled
    model_str.zero_grad()
    r2, l2, c2, rt2 = model_str.forward_all_exits(x)
    loss_str = r2.sum() + l2.sum() + c2.sum() + rt2.sum()
    loss_str.backward()

    assert torch.allclose(r1, r2, atol=1e-4, rtol=1e-3), "Reflex logits mismatch"
    assert torch.allclose(l1, l2, atol=1e-4, rtol=1e-3), "Limbic logits mismatch"
    assert torch.allclose(c1, c2, atol=1e-4, rtol=1e-3), "Cortex logits mismatch"
    assert torch.allclose(rt1, rt2, atol=1e-4, rtol=1e-3), "Route logits mismatch"

    param_count = 0
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model_std.named_parameters(), model_str.named_parameters()):
        param_count += 1
        g1 = p1.grad.cpu() if p1.grad is not None else None
        g2 = p2._cpu_grad if hasattr(p2, "_cpu_grad") and p2._cpu_grad is not None else (p2.grad.cpu() if p2.grad is not None else None)
        assert g1 is not None, f"{n1} has no grad in standard model"
        assert g2 is not None, f"{n2} has no grad in streaming model"
        diff = (g1 - g2).abs().max().item()
        if diff > max_diff:
            max_diff = diff
        assert torch.allclose(g1, g2, atol=1e-4, rtol=1e-3), f"Gradient mismatch in {n1}: diff {diff}"

    print(f"✅ Streaming gradient numerical parity verified across all {param_count} parameters (max diff: {max_diff})")


class TestLayerStreaming(unittest.TestCase):
    def test_lifecycle(self):
        test_layer_streaming_engine_lifecycle()

    def test_centroid_optimizer(self):
        test_layer_streaming_with_centroid_optimizer()

    def test_gpu_execution(self):
        test_layer_streaming_optimizer_gpu_execution()

    def test_gradient_numerical_parity(self):
        test_streaming_gradient_numerical_parity()


if __name__ == "__main__":
    unittest.main()
