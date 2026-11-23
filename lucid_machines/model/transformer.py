import contextlib
import math
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .config import (
    DEPTH_BALANCE_COEF,
    EXPERT_HIDDEN_MULTIPLIER,
    GLA_HEAD_DIM,
    HIDDEN_DIM,
    LIMBIC_EXIT_LAYER,
    NUM_EXPERTS,
    NUM_HEADS,
    NUM_LAYERS,
    REFLEX_EXIT_LAYER,
    ROPE_MAX_SEQ_LEN,
    ROUTER_PREFIX_LAYERS,
    SHARED_EXPERT,
    SHARED_EXPERT_SCALE,
    TARGET_DEPTH_DIST,
    TOP_K_EXPERTS,
    VOCAB_SIZE,
)
from .norms import RMSNorm
from .router import GumbelSoftmaxRouter


class TriuneTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        head_dim: int = GLA_HEAD_DIM,
        num_experts: int = NUM_EXPERTS,
        top_k_experts: int = TOP_K_EXPERTS,
        shared_expert: bool = SHARED_EXPERT,
        shared_expert_scale: float = SHARED_EXPERT_SCALE,
        expert_hidden_multiplier: int = EXPERT_HIDDEN_MULTIPLIER,
        rope_max_seq_len: int = ROPE_MAX_SEQ_LEN,
        router_prefix_layers: int = ROUTER_PREFIX_LAYERS,
        reflex_exit_layer: int = REFLEX_EXIT_LAYER,
        limbic_exit_layer: int = LIMBIC_EXIT_LAYER,
        target_depth_dist: tuple[float, float, float] = TARGET_DEPTH_DIST,
        balance_coef: float = DEPTH_BALANCE_COEF,
        use_fp4: bool = False,
        use_fp8: bool = False,
        streaming_fp8_weights: bool = False,
        scale_residual_projections: bool = False,
        use_reflex_sdpa: bool = False,
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
        if top_k_experts < 1 or top_k_experts > num_experts:
            raise ValueError(f"top_k_experts must be in [1, num_experts={num_experts}]; got {top_k_experts}")
        if expert_hidden_multiplier < 1:
            raise ValueError(f"expert_hidden_multiplier must be >= 1; got {expert_hidden_multiplier}")
        if rope_max_seq_len < 1:
            raise ValueError(f"rope_max_seq_len must be >= 1; got {rope_max_seq_len}")
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
            raise ValueError(f"hidden_dim ({hidden_dim}) must equal num_heads * head_dim ({expected_hidden_dim})")
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.top_k_experts = int(top_k_experts)
        self.shared_expert = bool(shared_expert)
        self.shared_expert_scale = float(shared_expert_scale)
        self.expert_hidden_multiplier = int(expert_hidden_multiplier)
        self.rope_max_seq_len = int(rope_max_seq_len)
        self.router_prefix_layers = router_prefix_layers
        self.reflex_exit_layer = reflex_exit_layer
        self.limbic_exit_layer = limbic_exit_layer
        self.use_fp4 = use_fp4
        self.use_fp8 = use_fp8

        self.last_route_logits = None
        self.last_depth_choice = None
        self.last_balance_loss = None
        self.last_calm_exit_depths = None
        self._composite_id_map: dict[int, int] = {}
        self._base_vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        exit_layers = [reflex_exit_layer, limbic_exit_layer]
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    num_heads,
                    i,
                    vocab_size,
                    exit_layers,
                    num_experts=num_experts,
                    top_k_experts=self.top_k_experts,
                    shared_expert=self.shared_expert,
                    shared_expert_scale=self.shared_expert_scale,
                    expert_hidden_multiplier=self.expert_hidden_multiplier,
                    rope_max_seq_len=self.rope_max_seq_len,
                    use_moe=(i > reflex_exit_layer),
                    use_fp4=use_fp4,
                    use_fp8=use_fp8,
                    use_sdpa=use_reflex_sdpa and (i <= reflex_exit_layer),
                )
                for i in range(num_layers)
            ]
        )
        self.router = GumbelSoftmaxRouter(hidden_dim, target_depth_dist=target_depth_dist, balance_coef=balance_coef)
        self.jev_router = None
        self.final_norm = RMSNorm(hidden_dim)
        self.final_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.token_embed.weight = self.final_head.weight
        if scale_residual_projections and self.num_layers > 0:
            residual_std = 0.02 / math.sqrt(2 * self.num_layers)
            for name, p in self.named_parameters():
                if "out_proj" in name or name.endswith(".2.weight"):
                    nn.init.normal_(p, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_jev_routing(self, threshold: float = 0.85, distance_metric: str = "geodesic"):
        """Enable Joint-Embedding Predictive Architecture (JEV) representation-driven exit routing."""
        from lucid_machines.model.jev import JEVExitRouter

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.jev_router = JEVExitRouter(dim=self.hidden_dim, threshold=threshold, distance_metric=distance_metric).to(
            device=device, dtype=dtype
        )
        return self.jev_router

    def embed_compressed(
        self,
        input_ids: torch.Tensor,
        tokenizer: Any,
        mean_pool: bool = True,
        compress_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Embed DynamicTokenizer ids, expanding composite ids into base-vocabulary embeddings.

        Args:
            input_ids: Token tensor containing base and/or composite dynamic IDs.
            tokenizer: DynamicTokenizer instance supporting ``decompress_tokens``.
            mean_pool: Used when ``compress_mode`` is ``None``.
            compress_mode: Optional explicit mode overriding ``mean_pool``:
                - ``"mean_pool"`` (``mean_pool=True``): averages constituent base embeddings per
                  compressed position ($T_{comp} < T_{base}$). Preserves lossless tokenizer text
                  reconstruction, but alters sequence length and is permutation-invariant within spans.
                - ``"order_pool"``: applies orthogonal intra-span phase rotation (identity at offset 0,
                  norm-preserving RoPE-style pairs for offset > 0) prior to segment averaging,
                  preserving constituent token order (e.g., ``"ab" != "ba"``) while leaving singletons exact.
                - ``"expand"`` (``mean_pool=False``): expands composite IDs to base tokens before
                  embedding lookup, yielding bit-for-bit identical logits to uncompressed input IDs
                  (requires equal expanded length across batch rows).
        """
        if compress_mode is not None:
            if compress_mode not in ("mean_pool", "order_pool", "expand"):
                raise ValueError(
                    f"Unknown compress_mode={compress_mode!r}; expected 'mean_pool', 'order_pool', or 'expand'"
                )
            mode = compress_mode
        else:
            mode = "mean_pool" if bool(mean_pool) else "expand"

        uniq, inverse = input_ids.unique(return_inverse=True)
        expansions = [tokenizer.decompress_tokens([t]) for t in uniq.tolist()]
        base_limit = getattr(self, "_base_vocab_size", self.vocab_size)
        bad = [b for e in expansions for b in e if not 0 <= b < base_limit]
        if bad:
            raise ValueError(f"Expanded base id {bad[0]} outside model vocab_size={self.vocab_size}")
        dev = input_ids.device
        if mode in ("mean_pool", "order_pool"):
            lens = torch.tensor([len(e) for e in expansions], device=dev)
            emb = self.token_embed(torch.tensor([b for e in expansions for b in e], device=dev))
            if mode == "order_pool":
                starts = torch.cumsum(lens, dim=0) - lens
                intra_pos = (torch.arange(emb.size(0), device=dev) - torch.repeat_interleave(starts, lens)).float()
                pair_dim = (emb.size(-1) // 2) * 2
                if pair_dim > 0 and bool((intra_pos > 0).any()):
                    inv_freq = 1.0 / (
                        10000.0 ** (torch.arange(0, pair_dim, 2, device=dev, dtype=torch.float32) / max(pair_dim, 2))
                    )
                    angles = intra_pos.unsqueeze(-1) * inv_freq.unsqueeze(0)
                    cos_a = torch.cos(angles).to(emb.dtype)
                    sin_a = torch.sin(angles).to(emb.dtype)
                    emb_pairs = emb[:, :pair_dim]
                    e0 = emb_pairs[:, 0::2]
                    e1 = emb_pairs[:, 1::2]
                    rot = torch.empty_like(emb_pairs)
                    rot[:, 0::2] = e0 * cos_a - e1 * sin_a
                    rot[:, 1::2] = e0 * sin_a + e1 * cos_a
                    if pair_dim < emb.size(-1):
                        emb = torch.cat([rot, emb[:, pair_dim:]], dim=-1)
                    else:
                        emb = rot
            seg = torch.repeat_interleave(torch.arange(uniq.numel(), device=dev), lens)
            pooled = emb.new_zeros(uniq.numel(), emb.size(-1)).index_add_(0, seg, emb)
            return (pooled / lens.unsqueeze(-1).to(emb.dtype))[inverse]
        by_id = dict(zip(uniq.tolist(), expansions))
        rows = [[b for t in row for b in by_id[t]] for row in input_ids.tolist()]
        if len({len(r) for r in rows}) != 1:
            raise ValueError("mean_pool=False requires equal expanded length across batch rows")
        return self.token_embed(torch.tensor(rows, device=dev))

    def extend_vocabulary_for_tokenizer(self, tokenizer: Any) -> dict[int, int]:
        """Extend embeddings, final_head, and exit_heads to natively address DynamicTokenizer composite IDs.

        Maps composite token IDs ``>= _base_vocab_size`` into contiguous vocabulary rows initialized
        to the mean of their constituent base embeddings, enabling zero-pooling native forward
        execution and fine-tuning on compressed sequences.
        """
        base_limit = getattr(self, "_base_vocab_size", self.vocab_size)
        candidates = sorted(
            tid
            for tid in (
                set(getattr(tokenizer, "expansion_cache", {}).keys())
                | set(getattr(tokenizer, "decomp_rules", {}).keys())
            )
            if int(tid) >= base_limit and int(tid) not in self._composite_id_map
        )
        valid_pairs: list[tuple[int, list[int]]] = []
        for tid in candidates:
            base_ids = tokenizer.decompress_tokens([tid])
            if base_ids and all(0 <= int(b) < base_limit for b in base_ids):
                valid_pairs.append((int(tid), [int(b) for b in base_ids]))
        if not valid_pairs:
            return dict(self._composite_id_map)

        old_vocab = self.vocab_size
        new_vocab = old_vocab + len(valid_pairs)
        dev = self.token_embed.weight.device
        dtype = self.token_embed.weight.dtype

        with torch.no_grad():
            old_w = self.final_head.weight.detach()
            new_head = nn.Linear(self.hidden_dim, new_vocab, bias=False, device=dev, dtype=dtype)
            new_head.weight[:old_vocab].copy_(old_w)
            for offset, (tid, base_ids) in enumerate(valid_pairs):
                row_idx = old_vocab + offset
                idx_t = torch.tensor(base_ids, device=dev, dtype=torch.long)
                new_head.weight[row_idx].copy_(old_w.index_select(0, idx_t).mean(dim=0))
                self._composite_id_map[tid] = row_idx

            new_embed = nn.Embedding(new_vocab, self.hidden_dim, device=dev, dtype=dtype)
            new_embed.weight = new_head.weight
            self.final_head = new_head
            self.token_embed = new_embed

            for layer in self.layers:
                if getattr(layer, "exit_head", None) is not None:
                    old_ex = layer.exit_head
                    has_bias = old_ex.bias is not None
                    new_ex = nn.Linear(self.hidden_dim, new_vocab, bias=has_bias, device=dev, dtype=old_ex.weight.dtype)
                    new_ex.weight[:old_vocab].copy_(old_ex.weight.detach())
                    if has_bias:
                        new_ex.bias[:old_vocab].copy_(old_ex.bias.detach())
                    for offset, (_, base_ids) in enumerate(valid_pairs):
                        row_idx = old_vocab + offset
                        idx_t = torch.tensor(base_ids, device=dev, dtype=torch.long)
                        new_ex.weight[row_idx].copy_(old_ex.weight.detach().index_select(0, idx_t).mean(dim=0))
                        if has_bias:
                            new_ex.bias[row_idx].copy_(old_ex.bias.detach().index_select(0, idx_t).mean(dim=0))
                    layer.exit_head = new_ex

        self.vocab_size = new_vocab
        return dict(self._composite_id_map)

    @staticmethod
    def _slice_cache(cache: Optional[List[Any]], idx: torch.Tensor) -> Optional[List[Any]]:
        if cache is None:
            return None
        sliced = []
        for entry in cache:
            if entry is not None and isinstance(entry, tuple) and entry[0] is not None:
                s, off = entry
                sliced.append((s.index_select(0, idx), off))
            else:
                sliced.append(entry)
        return sliced

    @staticmethod
    def _scatter_cache(
        dst_cache: Optional[List[Any]],
        src_cache: Optional[List[Any]],
        idx: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Optional[List[Any]]:
        if dst_cache is None or src_cache is None:
            return dst_cache
        for l_idx, entry in enumerate(src_cache):
            if entry is not None and isinstance(entry, tuple) and entry[0] is not None:
                s_d, off = entry
                if (
                    dst_cache[l_idx] is not None
                    and isinstance(dst_cache[l_idx], tuple)
                    and dst_cache[l_idx][0] is not None
                ):
                    new_s, _ = dst_cache[l_idx]
                    new_s = new_s.index_copy(0, idx, s_d.to(new_s.dtype))
                    dst_cache[l_idx] = (new_s, off)
                else:
                    full_s = torch.zeros(batch_size, *s_d.shape[1:], device=device, dtype=s_d.dtype)
                    full_s = full_s.index_copy(0, idx, s_d)
                    dst_cache[l_idx] = (full_s, off)
        return dst_cache

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
        resident_gpu_layers: int = 0,
        activation_checkpointing: bool = True,
    ):
        """Enables AirLLM-style layer streaming, keeping layers in CPU RAM and streaming on-demand."""
        from lucid_machines.runtime.streaming import LayerStreamingEngine, StreamingConfig

        cfg = StreamingConfig(
            enabled=True,
            chunk_size=chunk_size,
            pin_memory=pin_memory,
            async_prefetch=async_prefetch,
            fp8_weights=fp8_weights,
            grad_clip=grad_clip,
            instant_optimizer=instant_optimizer,
            resident_gpu_layers=resident_gpu_layers,
            activation_checkpointing=activation_checkpointing,
        )
        engine = LayerStreamingEngine(self, device=device, config=cfg)
        engine.attach()
        self._streaming_engine = engine
        return engine

    def _forward_block(
        self,
        layer,
        x,
        return_exit,
        cache=None,
        update_stats=True,
        exit_decider=None,
        exit_last_token_only=False,
        active_mask=None,
    ):
        if active_mask is not None:
            return layer(
                x,
                return_exit,
                cache,
                update_stats=update_stats,
                exit_decider=exit_decider,
                exit_last_token_only=exit_last_token_only,
                active_mask=active_mask,
            )
        return layer(
            x,
            return_exit,
            cache,
            update_stats=update_stats,
            exit_decider=exit_decider,
            exit_last_token_only=exit_last_token_only,
        )

    def _run_layers(self, x, start_layer, end_layer, cache=None, update_stats=True, active_mask=None):
        new_cache = list(cache) if cache is not None else None
        for i in range(start_layer, end_layer):
            layer_cache = cache[i] if cache is not None and i < len(cache) else None
            if cache is not None and layer_cache is None:
                layer_cache = (None, 0)
            x, _, new_c = self._forward_block(
                self.layers[i],
                x,
                False,
                cache=layer_cache,
                update_stats=update_stats,
                active_mask=active_mask,
            )
            if new_cache is not None:
                new_cache[i] = new_c
        return x, new_cache

    def _route(self, x, force_depth, B, device, temperature=1.0):
        logits, y_route, balance_loss = self.router(x, temperature=temperature, force_depth=force_depth)
        if force_depth is None:
            depth_choice = y_route.argmax(dim=-1)
        else:
            depth_choice = torch.full((B,), force_depth, device=device, dtype=torch.long)
        return logits, y_route, depth_choice, balance_loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        force_depth: Optional[int] = None,
        cache: Optional[List[Any]] = None,
        temperature: float = 1.0,
        return_balance_loss: bool = False,
        last_token_only: bool = False,
        exit_policy: str = "gumbel",
        confidence_threshold: float = 0.8,
        inputs_embeds: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
    ):
        if force_depth is not None and force_depth not in (0, 1, 2):
            raise ValueError(f"force_depth must be 0, 1, or 2; got {force_depth}")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            active_tok = tokenizer if tokenizer is not None else getattr(self, "tokenizer", None)
            base_v = getattr(self, "_base_vocab_size", self.vocab_size)
            if input_ids.numel() > 0 and int(input_ids.max().item()) >= base_v:
                comp_map = getattr(self, "_composite_id_map", None) or {}
                if comp_map:
                    high_ids = input_ids[input_ids >= base_v].unique().tolist()
                    if all(int(tid) in comp_map for tid in high_ids):
                        remapped = input_ids.clone()
                        for tid in high_ids:
                            remapped[input_ids == int(tid)] = int(comp_map[int(tid)])
                        x = self.token_embed(remapped)
                    elif active_tok is not None:
                        x = self.embed_compressed(input_ids, active_tok)
                    else:
                        x = self.token_embed(input_ids)
                elif active_tok is not None and int(input_ids.max().item()) >= self.vocab_size:
                    x = self.embed_compressed(input_ids, active_tok)
                else:
                    x = self.token_embed(input_ids)
            else:
                x = self.token_embed(input_ids)
        B, T = x.shape[:2]
        device = x.device

        x_prefix, new_cache = self._run_layers(x, 0, self.router_prefix_layers, cache=cache)
        route_logits, y_route, depth_choice, balance_loss = self._route(
            x_prefix, force_depth, B, device, temperature=temperature
        )
        self.last_route_logits = route_logits.detach()
        self.last_depth_choice = depth_choice.detach()
        self.last_balance_loss = balance_loss

        if exit_policy in ("calm", "confidence") and force_depth is None:
            # Confident Adaptive Language Modeling (CALM - Schuster et al., Google Research NeurIPS)
            # Step 1: Execute prefix through Reflex Exit (L2)
            x_reflex, new_cache = self._run_layers(
                x_prefix, self.router_prefix_layers, self.reflex_exit_layer, cache=new_cache
            )
            blk_ref = self.layers[self.reflex_exit_layer]
            layer_c = new_cache[self.reflex_exit_layer] if new_cache is not None else None
            x_reflex, logits_r, new_c = self._forward_block(
                blk_ref, x_reflex, True, cache=layer_c, exit_last_token_only=last_token_only
            )
            if new_cache is not None:
                new_cache[self.reflex_exit_layer] = new_c

            probs_r = F.softmax(logits_r.float(), dim=-1)
            top2_r = torch.topk(probs_r, 2, dim=-1).values
            margin_r = top2_r[..., 0] - top2_r[..., 1]  # [B, T_out]
            settled_r = margin_r >= confidence_threshold

            logits_out = logits_r.clone()
            calm_exit_depths = torch.full(settled_r.shape, 2, dtype=torch.long, device=device)
            calm_exit_depths = torch.where(settled_r, torch.zeros_like(calm_exit_depths), calm_exit_depths)
            total_exit_tokens = int(settled_r.numel())

            need_limbic = (~settled_r.all(dim=1)).nonzero(as_tuple=True)[0]
            if need_limbic.numel() == 0:
                self.last_calm_exit_depths = calm_exit_depths.detach()
                self.last_depth_choice = calm_exit_depths[:, -1].detach()
                self.last_calm_compaction_stats = {
                    "total_tokens": total_exit_tokens,
                    "reflex_tokens": total_exit_tokens,
                    "limbic_tokens": 0,
                    "cortex_tokens": 0,
                    "limbic_rows": 0,
                    "cortex_rows": 0,
                    "limbic_seq_len": 0,
                    "cortex_seq_len": 0,
                }
                if cache is not None:
                    return logits_out, new_cache
                if return_balance_loss:
                    return logits_out, route_logits, balance_loss
                return logits_out, route_logits

            # Step 2: Compact to unsettled batch items & tokens and continue Reflex -> Limbic Exit (L5)
            x_r_act = x_reflex.index_select(0, need_limbic)
            settled_r_act = settled_r.index_select(0, need_limbic)
            unsettled_r_act = ~settled_r_act
            n_limbic_tokens = int(unsettled_r_act.sum().item())
            n_limbic_rows = int(need_limbic.numel())
            l_cache = self._slice_cache(new_cache, need_limbic) if cache is not None else None

            if not last_token_only and cache is None:
                lim_cols = unsettled_r_act.any(dim=0).nonzero(as_tuple=True)[0]
                T_lim = int(lim_cols[-1].item()) + 1
                if T_lim < x_r_act.size(1):
                    x_r_act = x_r_act[:, :T_lim, :]
                    unsettled_r_act = unsettled_r_act[:, :T_lim]
                    settled_r_act = settled_r_act[:, :T_lim]
            else:
                T_lim = int(x_r_act.size(1))

            active_mask_l = unsettled_r_act if (not last_token_only or x_r_act.size(1) == 1) else None
            x_limbic_act, l_cache = self._run_layers(
                x_r_act,
                self.reflex_exit_layer + 1,
                self.limbic_exit_layer,
                cache=l_cache,
                active_mask=active_mask_l,
            )
            blk_lim = self.layers[self.limbic_exit_layer]
            layer_c = l_cache[self.limbic_exit_layer] if l_cache is not None else None

            if not last_token_only and getattr(blk_lim, "exit_head", None) is not None and active_mask_l is not None:
                x_limbic_act, _, new_c = self._forward_block(
                    blk_lim,
                    x_limbic_act,
                    False,
                    cache=layer_c,
                    active_mask=active_mask_l,
                )
                if l_cache is not None:
                    l_cache[self.limbic_exit_layer] = new_c
                    new_cache = self._scatter_cache(new_cache, l_cache, need_limbic, B, device)

                r_idx_l, t_idx_l = active_mask_l.nonzero(as_tuple=True)
                x_pos_l = x_limbic_act[r_idx_l, t_idx_l]
                logits_l_compact = blk_lim.exit_head(blk_lim.exit_norm(x_pos_l.unsqueeze(0))).squeeze(0)
                probs_l = F.softmax(logits_l_compact.float(), dim=-1)
                top2_l = torch.topk(probs_l, 2, dim=-1).values
                margin_l = top2_l[..., 0] - top2_l[..., 1]
                settled_l_flat = margin_l >= confidence_threshold * 0.85

                abs_rows_l = need_limbic.index_select(0, r_idx_l)
                logits_out = logits_out.index_put((abs_rows_l, t_idx_l), logits_l_compact.to(logits_out.dtype))

                cur_depths_l = calm_exit_depths[abs_rows_l, t_idx_l]
                new_depths_l = torch.where(settled_l_flat, torch.ones_like(cur_depths_l), cur_depths_l)
                calm_exit_depths = calm_exit_depths.index_put((abs_rows_l, t_idx_l), new_depths_l)

                settled_l_act = torch.zeros_like(active_mask_l)
                settled_l_act[r_idx_l, t_idx_l] = settled_l_flat
                still_unsettled_act = active_mask_l & (~settled_l_act)
            else:
                x_limbic_act, logits_l_act, new_c = self._forward_block(
                    blk_lim,
                    x_limbic_act,
                    True,
                    cache=layer_c,
                    exit_last_token_only=last_token_only,
                    active_mask=active_mask_l,
                )
                if l_cache is not None:
                    l_cache[self.limbic_exit_layer] = new_c
                    new_cache = self._scatter_cache(new_cache, l_cache, need_limbic, B, device)

                probs_l = F.softmax(logits_l_act.float(), dim=-1)
                top2_l = torch.topk(probs_l, 2, dim=-1).values
                margin_l = top2_l[..., 0] - top2_l[..., 1]  # [B_act, T_out]
                settled_l_act = (~settled_r_act) & (margin_l >= confidence_threshold * 0.85)

                prev_logits_act = logits_out.index_select(0, need_limbic)
                updated_l_act = torch.where(
                    settled_r_act.unsqueeze(-1),
                    prev_logits_act,
                    logits_l_act.to(logits_out.dtype),
                )
                logits_out = logits_out.index_copy(0, need_limbic, updated_l_act)

                prev_depths_act = calm_exit_depths.index_select(0, need_limbic)
                updated_depths_l = torch.where(settled_l_act, torch.ones_like(prev_depths_act), prev_depths_act)
                calm_exit_depths = calm_exit_depths.index_copy(0, need_limbic, updated_depths_l)

                still_unsettled_act = ~(settled_r_act | settled_l_act)

            rel_cortex = still_unsettled_act.any(dim=1).nonzero(as_tuple=True)[0]
            if rel_cortex.numel() == 0:
                self.last_calm_exit_depths = calm_exit_depths.detach()
                self.last_depth_choice = calm_exit_depths[:, -1].detach()
                self.last_calm_compaction_stats = {
                    "total_tokens": total_exit_tokens,
                    "reflex_tokens": total_exit_tokens,
                    "limbic_tokens": n_limbic_tokens,
                    "cortex_tokens": 0,
                    "limbic_rows": n_limbic_rows,
                    "cortex_rows": 0,
                    "limbic_seq_len": T_lim,
                    "cortex_seq_len": 0,
                }
                if cache is not None:
                    return logits_out, new_cache
                if return_balance_loss:
                    return logits_out, route_logits, balance_loss
                return logits_out, route_logits

            # Step 3: Compact to remaining unsettled items & tokens and continue Limbic -> Cortex depth (L8)
            abs_cortex = need_limbic.index_select(0, rel_cortex)
            x_l_act = x_limbic_act.index_select(0, rel_cortex)
            still_unsettled_c = still_unsettled_act.index_select(0, rel_cortex)
            n_cortex_tokens = int(still_unsettled_c.sum().item())
            n_cortex_rows = int(rel_cortex.numel())
            c_cache = self._slice_cache(l_cache, rel_cortex) if cache is not None else None

            if not last_token_only and cache is None:
                cor_cols = still_unsettled_c.any(dim=0).nonzero(as_tuple=True)[0]
                T_cor = int(cor_cols[-1].item()) + 1
                if T_cor < x_l_act.size(1):
                    x_l_act = x_l_act[:, :T_cor, :]
                    still_unsettled_c = still_unsettled_c[:, :T_cor]
            else:
                T_cor = int(x_l_act.size(1))

            active_mask_c = still_unsettled_c if (not last_token_only or x_l_act.size(1) == 1) else None
            x_cortex_act, c_cache = self._run_layers(
                x_l_act,
                self.limbic_exit_layer + 1,
                self.num_layers,
                cache=c_cache,
                active_mask=active_mask_c,
            )
            if cache is not None and c_cache is not None:
                new_cache = self._scatter_cache(new_cache, c_cache, abs_cortex, B, device)

            if not last_token_only:
                cr_idx, ct_idx = still_unsettled_c.nonzero(as_tuple=True)
                x_pos_c = x_cortex_act[cr_idx, ct_idx]
                logits_c_compact = self.final_head(self.final_norm(x_pos_c.unsqueeze(0))).squeeze(0)
                abs_rows_c = abs_cortex.index_select(0, cr_idx)
                logits_out = logits_out.index_put((abs_rows_c, ct_idx), logits_c_compact.to(logits_out.dtype))
            else:
                x_pos_c = x_cortex_act[:, -1:, :]
                logits_c_act = self.final_head(self.final_norm(x_pos_c))
                prev_logits_c = logits_out.index_select(0, abs_cortex)
                updated_c_act = torch.where(
                    still_unsettled_c.unsqueeze(-1),
                    logits_c_act.to(logits_out.dtype),
                    prev_logits_c,
                )
                logits_out = logits_out.index_copy(0, abs_cortex, updated_c_act)

            self.last_calm_exit_depths = calm_exit_depths.detach()
            self.last_depth_choice = calm_exit_depths[:, -1].detach()
            self.last_calm_compaction_stats = {
                "total_tokens": total_exit_tokens,
                "reflex_tokens": total_exit_tokens,
                "limbic_tokens": n_limbic_tokens,
                "cortex_tokens": n_cortex_tokens,
                "limbic_rows": n_limbic_rows,
                "cortex_rows": n_cortex_rows,
                "limbic_seq_len": T_lim,
                "cortex_seq_len": T_cor,
            }
            if cache is not None:
                return logits_out, new_cache
            if return_balance_loss:
                return logits_out, route_logits, balance_loss
            return logits_out, route_logits

        if exit_policy == "jev" and self.jev_router is not None and force_depth is None:
            # Joint-Embedding Predictive Architecture (JEV) representation routing
            x_reflex, new_cache = self._run_layers(
                x_prefix, self.router_prefix_layers, self.reflex_exit_layer, cache=new_cache
            )
            blk_ref = self.layers[self.reflex_exit_layer]

            jev_res = None

            def should_exit(hidden):
                nonlocal jev_res
                jev_res = self.jev_router(hidden)
                return jev_res.should_exit.float().mean() >= 0.5

            x_reflex, jev_logits, new_c = self._forward_block(
                blk_ref, x_reflex, False, exit_decider=should_exit, exit_last_token_only=last_token_only
            )
            if new_cache is not None:
                new_cache[self.reflex_exit_layer] = new_c

            if jev_res is not None and jev_res.should_exit.float().mean() >= 0.5:
                logits = jev_logits
                if cache is not None:
                    return logits, new_cache
                if return_balance_loss:
                    return logits, route_logits, balance_loss
                return logits, route_logits

        if force_depth is not None:
            if force_depth == 0:
                x_out, new_cache = self._run_layers(
                    x_prefix, self.router_prefix_layers, self.reflex_exit_layer, cache=new_cache
                )
                layer_c = new_cache[self.reflex_exit_layer] if new_cache is not None else None
                if new_cache is not None and layer_c is None:
                    layer_c = (None, 0)
                blk = self.layers[self.reflex_exit_layer]
                x_out, logits, new_c = self._forward_block(
                    blk, x_out, True, cache=layer_c, exit_last_token_only=last_token_only
                )
                if new_cache is not None:
                    new_cache[self.reflex_exit_layer] = new_c
            elif force_depth == 1:
                x_out, new_cache = self._run_layers(
                    x_prefix, self.router_prefix_layers, self.limbic_exit_layer, cache=new_cache
                )
                layer_c = new_cache[self.limbic_exit_layer] if new_cache is not None else None
                if new_cache is not None and layer_c is None:
                    layer_c = (None, 0)
                blk = self.layers[self.limbic_exit_layer]
                x_out, logits, new_c = self._forward_block(
                    blk, x_out, True, cache=layer_c, exit_last_token_only=last_token_only
                )
                if new_cache is not None:
                    new_cache[self.limbic_exit_layer] = new_c
            else:
                x_out, new_cache = self._run_layers(
                    x_prefix, self.router_prefix_layers, self.num_layers, cache=new_cache
                )
                x_final = x_out[:, -1:, :] if last_token_only else x_out
                logits = self.final_head(self.final_norm(x_final))

            if cache is not None:
                return logits, new_cache
            if return_balance_loss:
                return logits, route_logits, balance_loss
            return logits, route_logits

        final_logits = None
        for d in (0, 1, 2):
            idx = (depth_choice == d).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_d = x_prefix.index_select(0, idx)
            d_cache = self._slice_cache(new_cache, idx) if cache is not None else None

            if d == 0:
                x_d, d_cache = self._run_layers(x_d, self.router_prefix_layers, self.reflex_exit_layer, cache=d_cache)
                layer_c = d_cache[self.reflex_exit_layer] if d_cache is not None else None
                if d_cache is not None and layer_c is None:
                    layer_c = (None, 0)
                blk = self.layers[self.reflex_exit_layer]
                x_d, logits_d, new_c = self._forward_block(
                    blk, x_d, True, cache=layer_c, exit_last_token_only=last_token_only
                )
                if d_cache is not None:
                    d_cache[self.reflex_exit_layer] = new_c
            elif d == 1:
                x_d, d_cache = self._run_layers(x_d, self.router_prefix_layers, self.limbic_exit_layer, cache=d_cache)
                layer_c = d_cache[self.limbic_exit_layer] if d_cache is not None else None
                if d_cache is not None and layer_c is None:
                    layer_c = (None, 0)
                blk = self.layers[self.limbic_exit_layer]
                x_d, logits_d, new_c = self._forward_block(
                    blk, x_d, True, cache=layer_c, exit_last_token_only=last_token_only
                )
                if d_cache is not None:
                    d_cache[self.limbic_exit_layer] = new_c
            else:
                x_d, d_cache = self._run_layers(x_d, self.router_prefix_layers, self.num_layers, cache=d_cache)
                x_final = x_d[:, -1:, :] if last_token_only else x_d
                logits_d = self.final_head(self.final_norm(x_final))

            T_out = 1 if last_token_only else T
            if final_logits is None:
                final_logits = torch.empty(B, T_out, self.final_head.out_features, device=device, dtype=logits_d.dtype)
            final_logits = final_logits.index_copy(0, idx, logits_d.to(final_logits.dtype))
            if cache is not None and d_cache is not None:
                new_cache = self._scatter_cache(new_cache, d_cache, idx, B, device)

        if final_logits is None:
            final_logits = torch.empty(B, T, self.final_head.out_features, device=device, dtype=x_prefix.dtype)

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
        route_logits, _, depth_choice, balance_loss = self._route(x_prefix, None, B, device)
        self.last_route_logits = route_logits.detach()
        self.last_depth_choice = depth_choice.detach()
        self.last_balance_loss = balance_loss

        # Run every MoE block with the same update_stats value.  Label generation
        # must not alter routing statistics.
        x6, _ = self._run_layers(x_prefix, self.router_prefix_layers, self.reflex_exit_layer, update_stats=update_stats)
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
        x24, _ = self._run_layers(limbic_out, self.limbic_exit_layer + 1, self.num_layers, update_stats=update_stats)
        cortex_logits = self.final_head(self.final_norm(x24))

        return reflex_logits, limbic_logits, cortex_logits, route_logits

    def distill_exit_heads(
        self,
        calibration_ids: torch.Tensor,
        steps: int = 50,
        lr: float = 3e-3,
        batch_size: int = 7,
        weight_decay: float = 0.01,
    ) -> dict[str, float | int]:
        """Self-distill intermediate exit heads toward frozen Cortex predictions via KL divergence."""
        was_training = self.training
        prev_grad_flags = [(p, p.requires_grad) for p in self.parameters()]
        exit_params: list[nn.Parameter] = []
        for idx in (self.reflex_exit_layer, self.limbic_exit_layer):
            layer = self.layers[idx]
            if getattr(layer, "exit_norm", None) is not None:
                exit_params.extend(layer.exit_norm.parameters())
            if getattr(layer, "exit_head", None) is not None:
                exit_params.extend(layer.exit_head.parameters())

        if not exit_params or steps <= 0 or calibration_ids.numel() == 0:
            return {"steps": 0, "initial_kl_loss": 0.0, "final_kl_loss": 0.0}

        try:
            for p in self.parameters():
                p.requires_grad_(False)
            for p in exit_params:
                p.requires_grad_(True)

            model_device = next(self.parameters()).device
            cal_ids = calibration_ids.to(model_device)
            batches = list(cal_ids.split(max(1, int(batch_size))))
            is_cuda = model_device.type == "cuda"

            self.eval()
            torch.manual_seed(42)
            opt = torch.optim.AdamW(exit_params, lr=lr, weight_decay=weight_decay)

            init_loss: float | None = None
            final_loss = 0.0
            autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if is_cuda else contextlib.nullcontext()
            for _ in range(steps):
                step_loss = 0.0
                for batch in batches:
                    opt.zero_grad()
                    with autocast_ctx:
                        r_logits, l_logits, c_logits, _ = self.forward_all_exits(batch, update_stats=False)
                        v = c_logits.size(-1)
                        p_cortex = F.softmax(c_logits.detach().float().reshape(-1, v), dim=-1)
                        loss = F.kl_div(
                            F.log_softmax(r_logits.float().reshape(-1, v), dim=-1),
                            p_cortex,
                            reduction="batchmean",
                        ) + F.kl_div(
                            F.log_softmax(l_logits.float().reshape(-1, v), dim=-1),
                            p_cortex,
                            reduction="batchmean",
                        )
                    loss.backward()
                    opt.step()
                    step_loss += float(loss.detach().item())
                avg_step_loss = step_loss / max(1, len(batches))
                if init_loss is None:
                    init_loss = avg_step_loss
                final_loss = avg_step_loss

            return {
                "steps": int(steps),
                "initial_kl_loss": round(init_loss or 0.0, 4),
                "final_kl_loss": round(final_loss, 4),
            }
        finally:
            for p, prev_req in prev_grad_flags:
                p.requires_grad_(prev_req)
            self.train(was_training)
