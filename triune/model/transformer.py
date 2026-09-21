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
        use_fp4: bool = False,
        use_fp8: bool = False,
        streaming_fp8_weights: bool = False,
    ):
        super().__init__()
        if num_layers <= limbic_exit_layer:
            raise ValueError(
                f"num_layers must exceed limbic_exit_layer ({limbic_exit_layer}); got {num_layers}"
            )
        expected_hidden_dim = num_heads * head_dim
        if hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal num_heads * head_dim ({expected_hidden_dim})"
            )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.router_prefix_layers = router_prefix_layers
        self.reflex_exit_layer = reflex_exit_layer
        self.limbic_exit_layer = limbic_exit_layer
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)

        use_fp8_e4m3 = streaming_fp8_weights and hasattr(torch, "float8_e4m3fn")
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            block = TransformerBlock(
                dim=hidden_dim,
                heads=num_heads,
                layer_idx=i,
                vocab_size=vocab_size,
                exit_layers=(self.reflex_exit_layer, self.limbic_exit_layer),
                num_experts=num_experts,
                use_moe=(i > self.reflex_exit_layer),
                use_fp4=use_fp4,
                use_fp8=use_fp8,
            )
            if use_fp8_e4m3:
                for p in block.parameters():
                    if p.dim() >= 2:
                        p.data = p.data.to(torch.float8_e4m3fn)
            self.layers.append(block)

        self.router = GumbelSoftmaxRouter(hidden_dim)
        self.final_norm = RMSNorm(hidden_dim)
        self.final_head = nn.Linear(hidden_dim, vocab_size)
        self._use_gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and isinstance(m, (nn.Linear, getattr(te, 'Linear', nn.Linear))):
                if m.weight.dtype != torch.float8_e4m3fn:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None and m.bias.dtype != torch.float8_e4m3fn:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def gradient_checkpointing_enable(self):
        self._use_gradient_checkpointing = True
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
        for i in range(start_layer, end_layer):
            layer_cache = cache[i] if cache is not None and i < len(cache) else None
            x, _, _ = self._forward_block(self.layers[i], x, False, cache=layer_cache, update_stats=update_stats)
        return x

    def _route(self, x, force_depth, B, device, temperature=1.0):
        logits, y_route, balance_loss = self.router(x, temperature=temperature, force_depth=force_depth)
        if force_depth is None:
            depth_choice = y_route.argmax(dim=-1)
        else:
            depth_choice = torch.full((B,), force_depth, device=device, dtype=torch.long)
        return logits, depth_choice, balance_loss

    def forward(self, input_ids, force_depth=None, cache=None, temperature=1.0, return_balance_loss=False):
        B, T = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids)
        x_prefix = self._run_layers(x, 0, self.router_prefix_layers)
        route_logits, depth_choice, balance_loss = self._route(x_prefix, force_depth, B, device, temperature=temperature)
        self.last_balance_loss = balance_loss

        if force_depth is not None:
            if force_depth not in (0, 1, 2):
                raise ValueError(f"force_depth must be 0, 1, or 2; got {force_depth}")
            if force_depth == 0:
                x_out = self._run_layers(x_prefix, self.router_prefix_layers, self.reflex_exit_layer)
                _, logits, _ = self._forward_block(self.layers[self.reflex_exit_layer], x_out, True)
            elif force_depth == 1:
                x_out = self._run_layers(x_prefix, self.router_prefix_layers, self.limbic_exit_layer)
                _, logits, _ = self._forward_block(self.layers[self.limbic_exit_layer], x_out, True)
            else:
                x_out = self._run_layers(x_prefix, self.router_prefix_layers, self.num_layers)
                logits = self.final_head(self.final_norm(x_out))
            if return_balance_loss:
                return logits, route_logits, balance_loss
            return logits, route_logits

        final_logits = torch.empty(B, T, self.final_head.out_features, device=device, dtype=x.dtype)
        for d in (0, 1, 2):
            idx = (depth_choice == d).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_d = x_prefix.index_select(0, idx)
            if d == 0:
                x_d = self._run_layers(x_d, self.router_prefix_layers, self.reflex_exit_layer)
                _, logits_d, _ = self._forward_block(self.layers[self.reflex_exit_layer], x_d, True)
            elif d == 1:
                x_d = self._run_layers(x_d, self.router_prefix_layers, self.limbic_exit_layer)
                _, logits_d, _ = self._forward_block(self.layers[self.limbic_exit_layer], x_d, True)
            else:
                x_d = self._run_layers(x_d, self.router_prefix_layers, self.num_layers)
                logits_d = self.final_head(self.final_norm(x_d))
            final_logits.index_copy_(0, idx, logits_d)
        if return_balance_loss:
            return final_logits, route_logits, balance_loss
        return final_logits, route_logits

    def forward_all_exits(self, input_ids, update_stats=False):
        """Generate labels for depth router. If update_stats=False, MoE layers skip bias/centroid updates."""
        B, T = input_ids.shape
        device = input_ids.device
        x = self.token_embed(input_ids)
        x_prefix = self._run_layers(x, 0, self.router_prefix_layers)
        route_logits, _, _ = self._route(x_prefix, None, B, device)

        # Run every MoE block with the same update_stats value.  Label generation
        # must not alter routing statistics.
        x6 = self._run_layers(
            x_prefix, self.router_prefix_layers, self.reflex_exit_layer, update_stats=update_stats
        )
        reflex_out, reflex_logits, _ = self._forward_block(
            self.layers[self.reflex_exit_layer], x6, True, update_stats=update_stats
        )

        # Limbic: continue from reflex_out (layers 7-15) then layer 16 full block
        x16 = self._run_layers(
            reflex_out, self.reflex_exit_layer + 1, self.limbic_exit_layer, update_stats=update_stats
        )
        limbic_out, limbic_logits, _ = self._forward_block(
            self.layers[self.limbic_exit_layer], x16, True, update_stats=update_stats
        )

        # Cortex: continue from limbic_out (layers 17-23)
        x24 = self._run_layers(
            limbic_out, self.limbic_exit_layer + 1, self.num_layers, update_stats=update_stats
        )
        cortex_logits = self.final_head(self.final_norm(x24))

        return reflex_logits, limbic_logits, cortex_logits, route_logits
