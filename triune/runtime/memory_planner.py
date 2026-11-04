"""VRAM Memory Planner & Automated Hardware Budgeting Engine.

Calculates memory footprints across parameters, optimizer states, activations,
and gradient accumulation buffers to fit models cleanly on consumer GPUs (e.g. RTX 5070 8GB/12GB VRAM).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MemoryEstimate:
    total_params: int
    param_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    gradient_memory_gb: float
    total_vram_gb: float
    recommended_batch_size: int
    recommended_grad_accum: int
    recommended_checkpointing: bool
    recommended_precision: str


class MemoryPlanner:
    """Hardware-aware VRAM Memory Planner.

    Accurately models the Triune architecture:
    - Embedding: vocab_size * hidden_dim
    - Per-layer attention: 4 projections (Q,K,V,G) + output = 5 * hidden_dim^2
    - Per-layer MoE FFN: num_experts * 2 * hidden_dim * (hidden_dim * expert_multiplier) + shared expert
    - Per-layer dense FFN (prefix layers): 2 * hidden_dim * (hidden_dim * 4)
    - Exit heads at reflex/limbic layers: hidden_dim * vocab_size each
    - Router: hidden_dim * 3
    - GaLore optimizer: rank-64 subspace projections (~0.15 bytes/param effective)
    """

    @staticmethod
    def estimate_vram(
        config: dict,
        target_vram_gb: float = 8.0,
        available_vram_gb: float | None = None,
    ) -> MemoryEstimate:
        """Estimate VRAM allocation breakdown and recommend optimal training hyperparameters."""
        if available_vram_gb is None and torch.cuda.is_available():
            available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_vram_gb = available_vram_gb or target_vram_gb

        vocab_size = config.get("vocab_size", 32000)
        hidden_dim = config.get("hidden_dim", 1536)
        num_layers = config.get("num_layers", 24)
        num_experts = config.get("num_experts", 8)
        seq_len = config.get("seq_len", 256)
        use_fp4 = config.get("use_fp4", False)
        use_fp8 = config.get("use_fp8", False)
        expert_multiplier = config.get("expert_hidden_multiplier", 6)
        reflex_exit_layer = config.get("reflex_exit_layer", 6)
        router_prefix_layers = config.get("router_prefix_layers", 3)
        galore_rank = config.get("galore_rank", 64)

        # ── Accurate Parameter Count ──────────────────────────
        # Embedding + final norm
        embed_params = vocab_size * hidden_dim + hidden_dim  # embedding + final RMSNorm

        # Per-layer params
        total_layer_params = 0
        for i in range(num_layers):
            # Attention: Q, K, V, gate, out projections + 2 RMSNorms
            attn_params = 5 * (hidden_dim**2) + 2 * hidden_dim

            # FFN: MoE layers (after reflex_exit_layer) vs dense prefix layers
            if i > reflex_exit_layer:
                # MoE: num_experts routed + 1 shared, each with 2 linear layers
                expert_ffn_dim = hidden_dim * expert_multiplier
                routed_params = num_experts * (2 * hidden_dim * expert_ffn_dim)
                shared_params = 2 * hidden_dim * expert_ffn_dim  # shared expert
                gate_params = hidden_dim * num_experts  # gating projection
                ffn_params = routed_params + shared_params + gate_params
            else:
                # Dense prefix: 2 linear layers with 4x expansion
                ffn_params = 2 * hidden_dim * (hidden_dim * 4)

            total_layer_params += attn_params + ffn_params

        # Exit heads (reflex + limbic + final cortex head)
        exit_head_params = 3 * (hidden_dim * vocab_size) + 3 * vocab_size

        # Depth router + RMSNorms
        router_params = hidden_dim * 3 + (num_layers * 2 * hidden_dim) + hidden_dim

        total_params = embed_params + total_layer_params + exit_head_params + router_params

        # ── Bytes Per Parameter ──────────────────────────────
        if use_fp4:
            bytes_per_param = 0.5  # 4-bit (0.5 bytes per parameter)
        elif use_fp8:
            bytes_per_param = 1.0  # 8-bit (1.0 byte per parameter)
        else:
            bytes_per_param = 2.0  # 16-bit BF16 (2.0 bytes per parameter)

        param_memory_gb = (total_params * bytes_per_param) / (1024**3)

        # ── Optimizer Memory (GaLore Subspace) ────────────────
        # GaLore stores rank-r projection matrices + rank-r momentum/variance
        # For a (m, n) weight: projection is max(m,n) x rank, momentum/variance are rank x min(m,n) each
        # Effective: ~(rank / min(m,n)) * 8 bytes per param (FP32 momentum + variance)
        rank_ratio = min(1.0, galore_rank / hidden_dim)
        optimizer_2d_bytes = 8.0 * rank_ratio  # FP32 momentum + variance in low-rank subspace
        projection_bytes = 2.0 * rank_ratio
        optimizer_bytes_per_param = optimizer_2d_bytes + projection_bytes

        norm_params = num_layers * 2 * hidden_dim
        optimizer_memory_gb = ((total_params - norm_params) * optimizer_bytes_per_param + norm_params * 8.0) / (1024**3)

        # ── Activation Memory (with Gradient Checkpointing) ───
        # With selective gradient checkpointing, only layer boundary activations are stored
        checkpointing = True
        activation_per_sample_bytes = num_layers * seq_len * hidden_dim * 2

        # ── Gradient Memory ──────────────────────────────────
        # With GaLore low-rank projection, gradients are compressed into subspace buffers
        gradient_memory_gb = (total_params * bytes_per_param * 0.5) / (1024**3)

        # ── Target Safety Threshold ──────────────────────────
        cuda_context_gb = 0.5  # PyTorch CUDA context + cuBLAS workspace
        base_overhead_gb = param_memory_gb + optimizer_memory_gb + gradient_memory_gb + cuda_context_gb
        target_limit_gb = available_vram_gb * 0.90  # 90% safety margin

        remaining_vram_gb = max(0.1, target_limit_gb - base_overhead_gb)

        # ── Optimal Batch Size & Grad Accumulation ───────────
        activation_per_sample_gb = activation_per_sample_bytes / (1024**3)

        # Find largest batch that fits
        recommended_batch_size = 1
        for bs in [8, 4, 2, 1]:
            if bs * activation_per_sample_gb <= remaining_vram_gb:
                recommended_batch_size = bs
                break

        # Target effective batch of 8-16 tokens
        effective_target = 8
        recommended_grad_accum = max(1, effective_target // recommended_batch_size)

        # Precision recommendation
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                recommended_precision = "fp8"
            else:
                recommended_precision = "bf16"
        else:
            recommended_precision = "bf16"

        activation_memory_gb = recommended_batch_size * activation_per_sample_gb
        total_vram_gb = (
            param_memory_gb + optimizer_memory_gb + activation_memory_gb + gradient_memory_gb + cuda_context_gb
        )

        return MemoryEstimate(
            total_params=total_params,
            param_memory_gb=round(param_memory_gb, 3),
            optimizer_memory_gb=round(optimizer_memory_gb, 3),
            activation_memory_gb=round(activation_memory_gb, 3),
            gradient_memory_gb=round(gradient_memory_gb, 3),
            total_vram_gb=round(total_vram_gb, 3),
            recommended_batch_size=recommended_batch_size,
            recommended_grad_accum=recommended_grad_accum,
            recommended_checkpointing=checkpointing,
            recommended_precision=recommended_precision,
        )
