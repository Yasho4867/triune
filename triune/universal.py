"""Universal Multi-Architecture Engine for Triune Transformer.

Enables Triune to operate as a high-performance, 'one-shoe-fits-all' framework
capable of training, fine-tuning (LoRA/QLoRA), and serving:
1. Any Dense Transformer (Llama-style, Mistral-style, GPT-style)
2. Any Mixture-of-Experts (MoE) Transformer (DeepSeek-style, Mixtral-style, Triune-style)
3. Any Arbitrary User PyTorch Model (via universal instrumentation and memory gating)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from triune.kernels import (
    FastGELU,
    chunked_cross_entropy_from_hidden,
    fast_chunk_gla,
    fast_cross_entropy,
    fast_fused_add_rmsnorm,
    fast_gate_up_lora_linear,
    fast_lora_linear,
    fast_qkv_lora_linear,
    fast_rmsnorm,
    fast_swiglu,
)
from triune.optim.centroid import CentroidSteerOptimizer
from triune.trainer.finetune import LoRAConfig, TriuneFineTuner


def _extract_lora_components(mod: nn.Module):
    """Extract (weight, bias, lora_A, lora_B, scaling) from an nn.Linear or LoRALayer wrapper."""
    if hasattr(mod, "original_layer") and hasattr(mod, "lora_A") and mod.lora_A is not None and mod.lora_B is not None:
        orig = mod.original_layer
        w = getattr(orig, "weight", None)
        b = getattr(orig, "bias", None)
        return w, b, mod.lora_A, mod.lora_B, float(getattr(mod, "scaling", 2.0))
    elif (
        type(mod) is nn.Linear
        and hasattr(mod, "lora_A")
        and hasattr(mod, "lora_B")
        and mod.lora_A is not None
        and mod.lora_B is not None
    ):
        return mod.weight, mod.bias, mod.lora_A, mod.lora_B, float(getattr(mod, "lora_scaling", 2.0))
    return None


def _linear_or_lora(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Dispatches fused addmm LoRA when lora_A/lora_B are attached to an nn.Linear, else standard forward."""
    if (
        type(mod) is nn.Linear
        and hasattr(mod, "lora_A")
        and hasattr(mod, "lora_B")
        and mod.lora_A is not None
        and mod.lora_B is not None
    ):
        scaling = getattr(mod, "lora_scaling", 2.0)
        return fast_lora_linear(x, mod.weight, mod.bias, mod.lora_A, mod.lora_B, scaling)
    return mod(x)


# =========================================================================
# 1. Canonical Dense Transformer Architecture
# =========================================================================
class _FusedBaseQKVFunction(torch.autograd.Function):
    """Fused base QKV linear projection with analytical backward gradient slicing.
    Computes [flat_x @ W_qkv.T] in 1 GEMM and [flat_grad.T @ flat_x] in 1 GEMM,
    eliminating CatBackward autograd overhead and intermediate slice allocations.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        D = orig_shape[-1]
        flat_x = x.contiguous().reshape(-1, D)
        D_q, D_k, D_v = W_q.shape[0], W_k.shape[0], W_v.shape[0]
        W_qkv = torch.cat((W_q, W_k, W_v), dim=0)
        out = torch.mm(flat_x, W_qkv.t())
        ctx.save_for_backward(flat_x, W_q, W_k, W_v)
        ctx.orig_shape = orig_shape
        ctx.dims = (D_q, D_k, D_v)
        return out.reshape(*orig_shape[:-1], D_q + D_k + D_v)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        flat_x, W_q, W_k, W_v = ctx.saved_tensors
        D_q, D_k, D_v = ctx.dims
        flat_grad = grad_output.contiguous().reshape(-1, D_q + D_k + D_v)

        grad_x = None
        if ctx.needs_input_grad[0]:
            W_qkv = torch.cat((W_q, W_k, W_v), dim=0)
            grad_x = torch.mm(flat_grad, W_qkv).reshape(ctx.orig_shape)

        grad_W_q = grad_W_k = grad_W_v = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            grad_W_all = torch.mm(flat_grad.t(), flat_x)
            grad_W_q = grad_W_all[:D_q].contiguous() if ctx.needs_input_grad[1] else None
            grad_W_k = grad_W_all[D_q : D_q + D_k].contiguous() if ctx.needs_input_grad[2] else None
            grad_W_v = grad_W_all[D_q + D_k :].contiguous() if ctx.needs_input_grad[3] else None

        return grad_x, grad_W_q, grad_W_k, grad_W_v


class _FusedBaseGateUpFunction(torch.autograd.Function):
    """Fused base SwiGLU Gate+Up linear projection with analytical backward gradient slicing.
    Computes [flat_x @ W_gu.T] in 1 GEMM and [flat_grad.T @ flat_x] in 1 GEMM.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, W_g: torch.Tensor, W_u: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        D = orig_shape[-1]
        flat_x = x.contiguous().reshape(-1, D)
        D_g, D_u = W_g.shape[0], W_u.shape[0]
        W_gu = torch.cat((W_g, W_u), dim=0)
        out = torch.mm(flat_x, W_gu.t())
        ctx.save_for_backward(flat_x, W_g, W_u)
        ctx.orig_shape = orig_shape
        ctx.dims = (D_g, D_u)
        return out.reshape(*orig_shape[:-1], D_g + D_u)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        flat_x, W_g, W_u = ctx.saved_tensors
        D_g, D_u = ctx.dims
        flat_grad = grad_output.contiguous().reshape(-1, D_g + D_u)

        grad_x = None
        if ctx.needs_input_grad[0]:
            W_gu = torch.cat((W_g, W_u), dim=0)
            grad_x = torch.mm(flat_grad, W_gu).reshape(ctx.orig_shape)

        grad_W_g = grad_W_u = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            grad_W_all = torch.mm(flat_grad.t(), flat_x)
            grad_W_g = grad_W_all[:D_g].contiguous() if ctx.needs_input_grad[1] else None
            grad_W_u = grad_W_all[D_g:].contiguous() if ctx.needs_input_grad[2] else None

        return grad_x, grad_W_g, grad_W_u


class DenseAttention(nn.Module):
    """Standard Multi-Head / Grouped-Query Attention with optional Fast GLA."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        use_gla: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.use_gla = use_gla

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        if use_gla:
            self.gate_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        can_fuse_base = (
            type(self.q_proj) is nn.Linear
            and type(self.k_proj) is nn.Linear
            and type(self.v_proj) is nn.Linear
            and not hasattr(self.q_proj, "lora_A")
            and not hasattr(self.k_proj, "lora_A")
            and not hasattr(self.v_proj, "lora_A")
        )
        lora_q = _extract_lora_components(self.q_proj)
        lora_k = _extract_lora_components(self.k_proj)
        lora_v = _extract_lora_components(self.v_proj)
        can_fuse_lora = (
            lora_q is not None
            and lora_k is not None
            and lora_v is not None
            and lora_q[2].shape[0] == lora_k[2].shape[0] == lora_v[2].shape[0]
            and lora_q[4] == lora_k[4] == lora_v[4]
        )

        if self.use_gla:
            hd_total = q_dim
            if can_fuse_base and type(self.gate_proj) is nn.Linear and not hasattr(self.gate_proj, "lora_A"):
                W_qkvg = torch.cat(
                    (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, self.gate_proj.weight),
                    dim=0,
                )
                b_qkvg = (
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias, self.gate_proj.bias), dim=0)
                    if self.q_proj.bias is not None
                    else None
                )
                qkvg = F.linear(x, W_qkvg, b_qkvg)
                q = qkvg[..., :hd_total].reshape(B, T, self.num_heads, self.head_dim)
                k = qkvg[..., hd_total : hd_total + kv_dim].reshape(B, T, self.num_kv_heads, self.head_dim)
                v = qkvg[..., hd_total + kv_dim : hd_total + 2 * kv_dim].reshape(B, T, self.num_kv_heads, self.head_dim)
                g = -F.softplus(qkvg[..., hd_total + 2 * kv_dim :].reshape(B, T, self.num_heads, self.head_dim))
            else:
                q = _linear_or_lora(self.q_proj, x).reshape(B, T, self.num_heads, self.head_dim)
                k = _linear_or_lora(self.k_proj, x).reshape(B, T, self.num_kv_heads, self.head_dim)
                v = _linear_or_lora(self.v_proj, x).reshape(B, T, self.num_kv_heads, self.head_dim)
                g = -F.softplus(_linear_or_lora(self.gate_proj, x).reshape(B, T, self.num_heads, self.head_dim))
            if self.num_kv_heads < self.num_heads:
                k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
                v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            out, new_state = fast_chunk_gla(q, k, v, g, initial_state=state)
            out = out.reshape(B, T, -1)
            return _linear_or_lora(self.o_proj, out), new_state
        else:
            if can_fuse_lora:
                qkv = fast_qkv_lora_linear(
                    x,
                    lora_q[0],
                    lora_q[1],
                    lora_k[0],
                    lora_k[1],
                    lora_v[0],
                    lora_v[1],
                    lora_q[2],
                    lora_q[3],
                    lora_k[2],
                    lora_k[3],
                    lora_v[2],
                    lora_v[3],
                    scaling=lora_q[4],
                )
                if self.num_kv_heads == self.num_heads:
                    q, k, v = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
                else:
                    q = qkv[..., :q_dim].view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                    k = qkv[..., q_dim : q_dim + kv_dim].view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v = qkv[..., q_dim + kv_dim :].view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            else:
                q = _linear_or_lora(self.q_proj, x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                k = _linear_or_lora(self.k_proj, x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = _linear_or_lora(self.v_proj, x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if self.num_kv_heads < self.num_heads:
                try:
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                except TypeError:
                    k_rep = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    v_rep = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, T, -1)
            return _linear_or_lora(self.o_proj, out), None


class DenseMLP(nn.Module):
    """Dense Feed-Forward Network with SwiGLU or GELU activation."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, act_type: str = "swiglu"):
        super().__init__()
        hidden = hidden_dim or (dim * 4)
        self.act_type = act_type.lower()
        if self.act_type == "swiglu":
            self.gate_proj = nn.Linear(dim, hidden, bias=False)
            self.up_proj = nn.Linear(dim, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, dim, bias=False)
        else:
            self.linear1 = nn.Linear(dim, hidden, bias=False)
            self.act = FastGELU()
            self.linear2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_type == "swiglu":
            lora_g = _extract_lora_components(self.gate_proj)
            lora_u = _extract_lora_components(self.up_proj)
            can_fuse_gu_lora = (
                lora_g is not None
                and lora_u is not None
                and lora_g[2].shape[0] == lora_u[2].shape[0]
                and lora_g[4] == lora_u[4]
            )
            if can_fuse_gu_lora:
                gu = fast_gate_up_lora_linear(
                    x,
                    lora_g[0],
                    lora_g[1],
                    lora_u[0],
                    lora_u[1],
                    lora_g[2],
                    lora_g[3],
                    lora_u[2],
                    lora_u[3],
                    scaling=lora_g[4],
                )
                gate, up = gu.chunk(2, dim=-1)
                return _linear_or_lora(self.down_proj, fast_swiglu(gate, up))
            gate = _linear_or_lora(self.gate_proj, x)
            up = _linear_or_lora(self.up_proj, x)
            return _linear_or_lora(self.down_proj, fast_swiglu(gate, up))
        return _linear_or_lora(self.linear2, self.act(_linear_or_lora(self.linear1, x)))


class DenseTransformerBlock(nn.Module):
    """A standard dense transformer block featuring in-place RMSNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        use_gla: bool = False,
    ):
        super().__init__()
        self.norm1_w = nn.Parameter(torch.ones(dim))
        self.attn = DenseAttention(dim, num_heads, num_kv_heads=num_kv_heads, use_gla=use_gla)
        self.norm2_w = nn.Parameter(torch.ones(dim))
        self.mlp = DenseMLP(dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h1 = fast_rmsnorm(x, self.norm1_w)
        attn_out, new_state = self.attn(h1, state=state)
        h2, x = fast_fused_add_rmsnorm(attn_out, x, self.norm2_w)
        mlp_out = self.mlp(h2)
        x = x + mlp_out
        return x, new_state


class UniversalDenseTransformer(nn.Module):
    """High-performance Universal Dense Transformer Architecture."""

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 6,
        hidden_dim: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        use_gla: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                DenseTransformerBlock(
                    dim,
                    num_heads,
                    hidden_dim=hidden_dim,
                    num_kv_heads=num_kv_heads,
                    use_gla=use_gla,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm_w = nn.Parameter(torch.ones(dim))
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Optional[torch.Tensor]]] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        x = self.embed(input_ids)
        new_states = []
        for i, layer in enumerate(self.layers):
            s = states[i] if states and i < len(states) else None
            x, new_s = layer(x, state=s)
            new_states.append(new_s)
        x = fast_rmsnorm(x, self.final_norm_w)
        if return_hidden:
            return x, new_states
        logits = self.lm_head(x)
        return logits, new_states


# =========================================================================
# 2. Canonical Mixture-of-Experts (MoE) Transformer Architecture
# =========================================================================
class UniversalMoETransformer(nn.Module):
    """High-performance Universal Mixture-of-Experts Transformer Architecture."""

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 6,
        head_dim: int = 64,
        num_experts: int = 4,
        top_k: int = 1,
        shared_expert: bool = True,
        router_prefix_layers: int = 1,
        reflex_exit_layer: int = 2,
        limbic_exit_layer: int = 5,
        scale_residual_projections: bool = False,
    ):
        super().__init__()
        from triune.model.transformer import TriuneTransformer

        self.core = TriuneTransformer(
            vocab_size=vocab_size,
            hidden_dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            router_prefix_layers=router_prefix_layers,
            reflex_exit_layer=reflex_exit_layer,
            limbic_exit_layer=limbic_exit_layer,
            scale_residual_projections=scale_residual_projections,
        )
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_experts = num_experts

    def forward(
        self, input_ids: torch.Tensor, force_depth: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.core(input_ids, force_depth=force_depth)


# =========================================================================
# 3. Universal Multi-Model Execution & Training Engine
# =========================================================================
class TriuneUniversalEngine:
    """The One-Shoe-Fits-All Universal Training, Fine-Tuning & Serving Engine.

    Accepts ANY Dense Transformer, MoE Transformer, or user-provided PyTorch module.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: Literal["pretrain", "finetune", "inference"] = "pretrain",
        lora_config: Optional[LoRAConfig] = None,
        lr: float = 1e-3,
        use_centroid_steer: bool = True,
        rank: int = 16,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (
            torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        )
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.mode = mode
        self.lr = lr
        self.use_centroid_steer = use_centroid_steer
        self.rank = rank

        # Detect whether architecture is MoE or Dense
        self.is_moe = any("expert" in name.lower() for name, _ in self.model.named_modules())

        # Fine-Tuning Setup
        self.finetuner: Optional[TriuneFineTuner] = None
        if self.mode == "finetune":
            self.finetuner = TriuneFineTuner(self.model, lora_config=lora_config or LoRAConfig(rank=rank))

        # Build or use provided Optimizer
        self.optimizer = optimizer if optimizer is not None else self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.mode == "inference":
            return None

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if self.use_centroid_steer and self.mode == "pretrain":
            # Use 3-Tier Partitioned CentroidSteer + Muon + 8-bit AdamW across both MoE and Dense
            try:
                target_mod = self.model.core if hasattr(self.model, "core") else self.model
                return CentroidSteerOptimizer(
                    target_mod,
                    lr=self.lr,
                    rank=self.rank,
                    steer_scale=0.1 if self.is_moe else 0.0,
                    use_muon=True,
                    use_8bit_base=True,
                )
            except Exception:
                pass

        # Standard AdamW fallback
        return torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=0.01)

    def train_step(self, input_ids: torch.Tensor, targets: torch.Tensor, chunk_size: int = 1024) -> Dict[str, float]:
        """Execute a single accelerated training step with zero-allocation cross-entropy."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        if self.is_moe and hasattr(self.model, "core"):
            logits, _ = self.model(input_ids, force_depth=2)
            vocab_size = logits.size(-1)
            loss = fast_cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), chunk_size=chunk_size)
        elif hasattr(self.model, "layers") and isinstance(self.model, UniversalDenseTransformer):
            hidden, _ = self.model(input_ids, return_hidden=True)
            loss = chunked_cross_entropy_from_hidden(hidden, self.model.lm_head, targets, chunk_size=chunk_size)
        else:
            out = self.model(input_ids)
            logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
            vocab_size = logits.size(-1)
            loss = fast_cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), chunk_size=chunk_size)

        loss.backward()

        self.optimizer.step()
        return {"loss": float(loss.item())}

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressively generate tokens with GPU acceleration."""
        self.model.eval()
        prompt_ids = prompt_ids.to(self.device)
        curr_ids = prompt_ids

        for _ in range(max_new_tokens):
            if self.is_moe and hasattr(self.model, "core"):
                logits, _ = self.model(curr_ids, force_depth=2)
            elif hasattr(self.model, "layers") and isinstance(self.model, UniversalDenseTransformer):
                logits, _ = self.model(curr_ids)
            else:
                out = self.model(curr_ids)
                logits = out[0] if isinstance(out, tuple) else out

            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                vals, idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < vals[:, -1:]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        return curr_ids


def upgrade_model(
    model: nn.Module,
    *,
    lora_rank: Optional[int] = None,
    lora_alpha: float = 32.0,
    streaming: bool = False,
    resident_gpu_layers: int = 0,
    as_engine: bool = False,
    return_optimizer: bool = False,
    lr: float = 1e-3,
    rank: int = 16,
) -> Union[nn.Module, Tuple[nn.Module, torch.optim.Optimizer], TriuneUniversalEngine]:
    """Universal 1-line Triune AI drop-in accelerator for any PyTorch or Hugging Face model.

    Automatically patches:
    - RMSNorm modules (LlamaRMSNorm, Qwen2RMSNorm, MistralRMSNorm, GemmaRMSNorm, nn.RMSNorm) -> fast_rmsnorm
    - SwiGLU MLP modules (with gate_proj, up_proj, down_proj) -> fast_swiglu
    - GELU activations -> FastGELU
    - Optional LoRA adapters (when lora_rank is specified) -> fused addmm LoRALayer
    - Optional LayerStreamingEngine (when streaming=True)
    - Optional 3-Tier CentroidSteer + Muon optimizer (when return_optimizer=True)
    - Optional TriuneUniversalEngine wrap (when as_engine=True)
    """
    import types

    for name, mod in list(model.named_modules()):
        cls_name = mod.__class__.__name__
        # 1. Patch any RMSNorm variant
        if "rmsnorm" in cls_name.lower() and hasattr(mod, "weight") and mod.weight is not None:
            eps = float(getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6)) or 1e-6)

            def _fast_rmsnorm_fwd(self_mod, hidden_states: torch.Tensor, _eps: float = eps) -> torch.Tensor:
                return fast_rmsnorm(hidden_states, self_mod.weight, eps=_eps)

            mod.forward = types.MethodType(_fast_rmsnorm_fwd, mod)
            mod._triune_upgraded_rmsnorm = True

        # 2. Patch standard HuggingFace SwiGLU MLPs (gate_proj, up_proj, down_proj)
        elif (
            hasattr(mod, "gate_proj")
            and hasattr(mod, "up_proj")
            and hasattr(mod, "down_proj")
            and not isinstance(mod, DenseMLP)
            and not getattr(mod, "_triune_upgraded_swiglu", False)
        ):

            def _fast_swiglu_mlp_fwd(self_mod, x: torch.Tensor) -> torch.Tensor:
                return _linear_or_lora(
                    self_mod.down_proj,
                    fast_swiglu(_linear_or_lora(self_mod.gate_proj, x), _linear_or_lora(self_mod.up_proj, x)),
                )

            mod.forward = types.MethodType(_fast_swiglu_mlp_fwd, mod)
            mod._triune_upgraded_swiglu = True

    if lora_rank is not None and lora_rank > 0:
        tuner = TriuneFineTuner(model, LoRAConfig(rank=lora_rank, alpha=lora_alpha))
        model._triune_finetuner = tuner

    if streaming:
        from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig

        dev = next((p.device for p in model.parameters()), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if dev.type == "cpu" and torch.cuda.is_available():
            dev = torch.device("cuda:0")
        engine = LayerStreamingEngine(
            model,
            device=dev,
            config=StreamingConfig(enabled=True, async_prefetch=True, resident_gpu_layers=resident_gpu_layers),
        )
        engine.attach()

    if as_engine or return_optimizer:
        is_moe = any("expert" in n.lower() for n, _ in model.named_modules())
        target_mod = model.core if hasattr(model, "core") else model
        opt = CentroidSteerOptimizer(
            target_mod,
            lr=lr,
            rank=rank,
            steer_scale=0.1 if is_moe else 0.0,
            use_muon=True,
            use_8bit_base=True,
        )
        if getattr(model, "_streaming_engine", None) is not None:
            model._streaming_engine.set_optimizer(opt)
        if as_engine:
            return TriuneUniversalEngine(model, optimizer=opt)
        return model, opt

    return model


upgrade = upgrade_model
