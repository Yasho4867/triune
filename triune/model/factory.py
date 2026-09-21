"""Model construction for callable applications."""

from __future__ import annotations

from .transformer import TriuneTransformer


def build_model(config: dict | None = None) -> TriuneTransformer:
    """Build the established Triune model from a per-run config dictionary."""
    if config is None:
        return TriuneTransformer()
    return TriuneTransformer(
        vocab_size=config.get("vocab_size", 32000),
        hidden_dim=config.get("hidden_dim", 1536),
        num_layers=config.get("num_layers", 24),
        num_heads=config.get("num_heads", 12),
        head_dim=config.get("head_dim", 128),
        num_experts=config.get("num_experts", 8),
        router_prefix_layers=config.get("router_prefix_layers", 3),
        reflex_exit_layer=config.get("reflex_exit_layer", 6),
        limbic_exit_layer=config.get("limbic_exit_layer", 16),
        use_fp4=config.get("use_fp4", False),
        use_fp8=config.get("use_fp8", False),
        streaming_fp8_weights=config.get("streaming_fp8_weights", False),
    )
