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


if __name__ == "__main__":
    test_layer_streaming_engine_lifecycle()
    test_layer_streaming_with_centroid_optimizer()
    test_layer_streaming_optimizer_gpu_execution()
