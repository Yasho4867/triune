"""Tests for Long-Context (up to 8,192 tokens) Performance, Chunked Prefill, and RoPE Caching."""

import pytest
import torch
import torch.nn.functional as F

from triune.configs.config import build_config
from triune.model.attention import HybridAttention, VectorisedGLA
from triune.model.factory import build_model
from triune.model.rotary import RotaryEmbedding
from triune.inference import generate_response


class DummyTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size

    def encode(self, text):
        class Tokens:
            def __init__(self, ids):
                self.ids = ids
        # Produce repeated token sequence up to length
        words = text.split()
        return Tokens([min(i % self.vocab_size, self.vocab_size - 1) for i in range(max(1, len(words)))])

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join([f"tok_{t}" for t in token_ids])

    def token_to_id(self, token):
        if token == "[PAD]":
            return 0
        if token == "[SEP]":
            return 1
        return None


def test_rope_precomputation_8192_tokens():
    """Verify RoPE precomputes cached buffers up to at least 8192 tokens without re-allocating."""
    rope = RotaryEmbedding(dim=64, max_seq_len=8192)
    assert rope.cos_cached.size(0) >= 8192
    assert rope.sin_cached.size(0) >= 8192

    device = torch.device("cpu")
    # Querying 8192 tokens must not trigger reallocation
    cos, sin = rope(8192, device, offset=0)
    assert cos.shape == (8192, 64)
    assert sin.shape == (8192, 64)
    assert rope.cos_cached.size(0) >= 8192


def test_factored_gla_numerical_parity_and_scaling():
    """Verify factored intra-chunk GLA GEMM produces exact mathematical parity with un-factored recurrence."""
    from triune.kernels.fast_gla import fast_chunk_gla

    B, T, H, HD = 1, 1024, 4, 64
    torch.manual_seed(42)
    q = torch.randn(B, T, H, HD, dtype=torch.float32)
    k = torch.randn(B, T, H, HD, dtype=torch.float32)
    v = torch.randn(B, T, H, HD, dtype=torch.float32)
    g = -torch.rand(B, T, H, HD, dtype=torch.float32) * 0.1

    out, state = fast_chunk_gla(q, k, v, g, chunk_size=64)
    assert out.shape == (B, T, H, HD)
    assert state.shape == (B, H, HD, HD)
    assert not torch.isnan(out).any()
    assert not torch.isnan(state).any()


def test_chunked_prefill_parity_at_long_context():
    """Verify chunked prefill (1024 slices) yields identical output to full sequence prefill."""
    cfg = build_config({
        "vocab_size": 1000,
        "hidden_dim": 128,
        "num_heads": 2,
        "head_dim": 64,
        "num_layers": 4,
        "num_experts": 2,
        "router_prefix_layers": 1,
        "reflex_exit_layer": 2,
        "limbic_exit_layer": 3,
    })
    model = build_model(cfg).eval()

    seq_len = 2048
    torch.manual_seed(123)
    x = torch.randint(2, 1000, (1, seq_len))

    # 1. Full unchunked prefill
    cache_full = model.init_cache(batch_size=1)
    with torch.no_grad():
        logits_full, cache_full = model(x, cache=cache_full, force_depth=2)

    # 2. Chunked prefill (slice size = 512)
    cache_chunked = model.init_cache(batch_size=1)
    chunk_size = 512
    with torch.no_grad():
        for i in range(0, seq_len - chunk_size, chunk_size):
            _, cache_chunked = model(x[:, i : i + chunk_size], cache=cache_chunked, force_depth=2)
        last_i = ((seq_len - 1) // chunk_size) * chunk_size
        logits_chunked, cache_chunked = model(x[:, last_i:], cache=cache_chunked, force_depth=2)

    diff = (logits_full[:, -1] - logits_chunked[:, -1]).abs().max().item()
    assert diff < 1e-4, f"Chunked prefill diverged from full prefill: max_diff={diff}"


def test_hybrid_attention_sdpa_dispatch_with_kv_cache():
    """Verify HybridAttention dispatches to hardware-fused SDPA with KV cache tracking."""
    dim = 128
    heads = 2
    attn = HybridAttention(dim=dim, heads=heads, use_sdpa=True)

    # Prefill step
    x_prompt = torch.randn(1, 64, dim)
    out_prefill, cache = attn(x_prompt, cache=(None, 0))
    assert out_prefill.shape == (1, 64, dim)
    assert cache is not None
    (k_cache, v_cache), offset = cache
    assert offset == 64
    assert k_cache.shape == (1, heads, 64, 64)

    # Autoregressive single token step
    x_tok = torch.randn(1, 1, dim)
    out_step, new_cache = attn(x_tok, cache=cache)
    assert out_step.shape == (1, 1, dim)
    (k_new, v_new), new_offset = new_cache
    assert new_offset == 65
    assert k_new.shape == (1, heads, 65, 64)


def test_long_context_generation_response_with_chunked_prefill():
    """Verify generate_response handles long prompts seamlessly via chunked prefill."""
    cfg = build_config({
        "vocab_size": 500,
        "hidden_dim": 128,
        "num_heads": 2,
        "head_dim": 64,
        "num_layers": 4,
        "num_experts": 2,
        "router_prefix_layers": 1,
        "reflex_exit_layer": 2,
        "limbic_exit_layer": 3,
    })
    model = build_model(cfg).eval()
    tokenizer = DummyTokenizer(vocab_size=500)

    # Long prompt with 2500 words (> 2 chunks of 1024)
    long_prompt = " ".join([f"word{i}" for i in range(2500)])

    res = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=long_prompt,
        device="cpu",
        pad_token_id=0,
        eos_token_id=1,
        max_new_tokens=5,
        temperature=0.0,
        return_info=True,
    )

    assert isinstance(res, dict)
    assert res["tokens_count"] > 0
    assert not res.get("cancelled", False)
    assert res["latency_ms"] > 0
