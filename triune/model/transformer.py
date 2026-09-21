import math
from typing import Any, List, Optional
import torch
import torch.nn as nn

from .block import *
from .fp4 import te
from .norms import *
from .router import GumbelSoftmaxRouter
from .config import *


class TriuneTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        head_dim: int = GLA_HEAD_DIM,
        num_experts: int = NUM_EXPERTS,
        router_prefix_layers: int = ROUTER_PREFIX_LAYERS,
        reflex_exit_layer: int = REFLEX_EXIT_LAYER,
        limbic_exit_layer: int = LIMBIC_EXIT_LAYER,
        target_depth_dist: tuple[float, float, float] = TARGET_DEPTH_DIST,
        balance_coef: float = DEPTH_BALANCE_COEF,
        use_fp4: bool = False,
        use_fp8: bool = False,
        streaming_fp8_weights: bool = False,
    ):
        super().__init__()
        if not (0 < router_prefix_layers < reflex_exit_layer < limbic_exit_layer < num_layers):
            raise ValueError(
                f"Layer hierarchy violation: expected 0 < router_prefix_layers ({router_prefix_layers}) < "
                f"reflex_exit_layer ({reflex_exit_layer}) < limbic_exit_layer ({limbic_exit_layer}) < "
                f"num_layers ({num_layers})"
            )
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1; got {num_experts}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1; got {vocab_size}")
        if num_heads < 1 or head_dim < 1:
            raise ValueError(f"num_heads ({num_heads}) and head_dim ({head_dim}) must be >= 1")
        if not math.isfinite(balance_coef) or balance_coef < 0:
            raise ValueError(f"balance_coef must be a finite non-negative float; got {balance_coef}")
        if any(not math.isfinite(d) or d < 0 for d in target_depth_dist):
            raise ValueError(f"target_depth_dist elements must be finite non-negative floats; got {target_depth_dist}")
        if abs(sum(target_depth_dist) - 1.0) > 1e-4:
            raise ValueError(f"target_depth_dist must sum to 1.0; got {target_depth_dist}")

        expected_hidden_dim = num_heads * head_dim
        if hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal num_heads * head_dim ({expected_hidden_dim})"
            )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.router_prefix_layers = router_prefix_layers
        self.reflex_exit_layer = reflex_exit_layer
        self.limbic_exit_layer = limbic_exit_layer
        self.use_fp4 = use_fp4
        self.use_fp8 = use_fp8

        self.last_route_logits = None
        self.last_depth_choice = None
        self.last_balance_loss = None

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        exit_layers = [reflex_exit_layer, limbic_exit_layer]
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, i, vocab_size, exit_layers, num_experts=num_experts, use_moe=(i > reflex_exit_layer), use_fp4=use_fp4, use_fp8=use_fp8)
            for i in range(num_layers)
        ])
        self.router = GumbelSoftmaxRouter(hidden_dim, target_depth_dist=target_depth_dist, balance_coef=balance_coef)
        self.final_norm = RMSNorm(hidden_dim)
        self.final_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.token_embed.weight = self.final_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_cache(self, batch_size: int = 1) -> list[Any]:
        """Initializes empty recurrent GLA cache states for all layers."""
        return [(None, 0) for _ in range(self.num_layers)]

    def enable_gradient_checkpointing(self):
        for layer in self.layers:
            layer._use_gradient_checkpointing = True

    def enable_layer_streaming(
        self,
        device: torch.device,
        chunk_size: int = 1,
        pin_memory: bool | None = None,
        async_prefetch: bool = True,
        fp8_weights: bool = False,
        grad_clip: float = 1.0,
        instant_optimizer: bool = True,
    ):
        """Enables AirLLM-style layer streaming, keeping layers in CPU RAM and streaming on-demand."""
        from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig
        cfg = StreamingConfig(
            enabled=True,
            chunk_size=chunk_size,
            pin_memory=pin_memory,
            async_prefetch=async_prefetch,
            fp8_weights=fp8_weights,
            grad_clip=grad_clip,
            instant_optimizer=instant_optimizer,
        )
        engine = LayerStreamingEngine(self, device=device, config=cfg)
        engine.attach()
        self._streaming_engine = engine
        return engine

    def _forward_block(self, layer, x, return_exit, cache=None, update_stats=True):
        return layer(x, return_exit, cache, update_stats=update_stats)

    def _run_layers(self, x, start_layer, end_layer, cache=None, update_stats=True):
        new_cache = list(cache) if cache is not None else None
        for i in range(start_layer, end_layer):
            layer_cache = cache[i] if cache is not None and i < len(cache) else None
            if cache is not None and layer_cache is None:
                layer_cache = (None, 0)
            x, _, new_c = self._forward_block(self.layers[i], x, False, cache=layer_cache, update_stats=update_stats)
            if new_cache is not None:
                new_cache[i] = new_c
        return x, new_cache

    def _route(self, x, force_depth, B, device, temperature=1.0):
        logits, y_route, balance_loss = self.router(x, temperature=temperature, force_depth=force_depth)
        if force_depth is None:
            depth_choice = y_route.argmax(dim=-1)
        else:
            depth_choice = torch.full((B,), force_depth, device=device, dtype=torch.long)
        return logits, depth_choice, balance_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        force_depth: Optional[int] = None,
        cache: Optional[List[Any]] = None,
        temperature: float = 1.0,
        return_balance_loss: bool = False,
    ):
        if force_depth is not None and force_depth not in (0, 1, 2):
            raise ValueError(f"force_depth must be 0, 1, or 2; got {force_depth}")
        B, T = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids)

        x_prefix, new_cache = self._run_layers(x, 0, self.router_prefix_layers, cache=cache)
        route_logits, depth_choice, balance_loss = self._route(x_prefix, force_depth, B, device, temperature=temperature)
        self.last_route_logits = route_logits
        self.last_depth_choice = depth_choice
        self.last_balance_loss = balance_loss

        if force_depth is not None:
            if force_depth == 0:
                x_out, new_cache = self._run_layers(x_prefix, self.router_prefix_layers, self.reflex_exit_layer, cache=new_cache)
                layer_c = new_cache[self.reflex_exit_layer] if new_cache is not None else None
                if new_cache is not None and layer_c is None:
                    layer_c = (None, 0)
                _, logits, new_c = self._forward_block(self.layers[self.reflex_exit_layer], x_out, True, cache=layer_c)
                if new_cache is not None:
                    new_cache[self.reflex_exit_layer] = new_c
            elif force_depth == 1:
                x_out, new_cache = self._run_layers(x_prefix, self.router_prefix_layers, self.limbic_exit_layer, cache=new_cache)
                layer_c = new_cache[self.limbic_exit_layer] if new_cache is not None else None
                if new_cache is not None and layer_c is None:
                    layer_c = (None, 0)
                _, logits, new_c = self._forward_block(self.layers[self.limbic_exit_layer], x_out, True, cache=layer_c)
                if new_cache is not None:
                    new_cache[self.limbic_exit_layer] = new_c
            else:
                x_out, new_cache = self._run_layers(x_prefix, self.router_prefix_layers, self.num_layers, cache=new_cache)
                logits = self.final_head(self.final_norm(x_out))

            if cache is not None:
                return logits, new_cache
            if return_balance_loss:
                return logits, route_logits, balance_loss
            return logits, route_logits

        final_logits = torch.empty(B, T, self.final_head.out_features, device=device, dtype=x.dtype)
        for d in (0, 1, 2):
            idx = (depth_choice == d).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_d = x_prefix.index_select(0, idx)
            d_cache = None
            if cache is not None:
                d_cache = []
                for entry in new_cache:
                    if entry is not None and isinstance(entry, tuple) and entry[0] is not None:
                        s, off = entry
                        d_cache.append((s.index_select(0, idx), off))
                    else:
                        d_cache.append(entry)

            if d == 0:
                x_d, d_cache = self._run_layers(x_d, self.router_prefix_layers, self.reflex_exit_layer, cache=d_cache)
                layer_c = d_cache[self.reflex_exit_layer] if d_cache is not None else None
                if d_cache is not None and layer_c is None:
                    layer_c = (None, 0)
                _, logits_d, new_c = self._forward_block(self.layers[self.reflex_exit_layer], x_d, True, cache=layer_c)
                if d_cache is not None:
                    d_cache[self.reflex_exit_layer] = new_c
            elif d == 1:
                x_d, d_cache = self._run_layers(x_d, self.router_prefix_layers, self.limbic_exit_layer, cache=d_cache)
                layer_c = d_cache[self.limbic_exit_layer] if d_cache is not None else None
                if d_cache is not None and layer_c is None:
                    layer_c = (None, 0)
                _, logits_d, new_c = self._forward_block(self.layers[self.limbic_exit_layer], x_d, True, cache=layer_c)
                if d_cache is not None:
                    d_cache[self.limbic_exit_layer] = new_c
            else:
                x_d, d_cache = self._run_layers(x_d, self.router_prefix_layers, self.num_layers, cache=d_cache)
                logits_d = self.final_head(self.final_norm(x_d))

            final_logits.index_copy_(0, idx, logits_d)
            if cache is not None and d_cache is not None:
                for l_idx, entry in enumerate(d_cache):
                    if entry is not None and isinstance(entry, tuple) and entry[0] is not None:
                        s_d, off = entry
                        if new_cache[l_idx] is not None and isinstance(new_cache[l_idx], tuple) and new_cache[l_idx][0] is not None:
                            new_s, _ = new_cache[l_idx]
                            new_s.index_copy_(0, idx, s_d)
                            new_cache[l_idx] = (new_s, off)
                        else:
                            full_s = torch.zeros(B, *s_d.shape[1:], device=device, dtype=s_d.dtype)
                            full_s.index_copy_(0, idx, s_d)
                            new_cache[l_idx] = (full_s, off)

        if cache is not None:
            return final_logits, new_cache
        if return_balance_loss:
            return final_logits, route_logits, balance_loss
        return final_logits, route_logits

    def forward_all_exits(self, input_ids, update_stats=False):
        """Generate labels for depth router. If update_stats=False, MoE layers skip bias/centroid updates."""
        B, T = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids)
        x_prefix, _ = self._run_layers(x, 0, self.router_prefix_layers)
        route_logits, _, _ = self._route(x_prefix, None, B, device)

        # Run every MoE block with the same update_stats value.  Label generation
        # must not alter routing statistics.
        x6, _ = self._run_layers(
            x_prefix, self.router_prefix_layers, self.reflex_exit_layer, update_stats=update_stats
        )
        reflex_out, reflex_logits, _ = self._forward_block(
            self.layers[self.reflex_exit_layer], x6, True, update_stats=update_stats
        )

        # Limbic: continue from reflex_out (layers 7-15) then layer 16 full block
        x16, _ = self._run_layers(
            reflex_out, self.reflex_exit_layer + 1, self.limbic_exit_layer, update_stats=update_stats
        )
        limbic_out, limbic_logits, _ = self._forward_block(
            self.layers[self.limbic_exit_layer], x16, True, update_stats=update_stats
        )

        # Cortex: continue from limbic_out (layers 17-23)
        x24, _ = self._run_layers(
            limbic_out, self.limbic_exit_layer + 1, self.num_layers, update_stats=update_stats
        )
        cortex_logits = self.final_head(self.final_norm(x24))

        return reflex_logits, limbic_logits, cortex_logits, route_logits
