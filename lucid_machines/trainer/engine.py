"""Stateful execution engine for LucidMachines training."""

from __future__ import annotations

import inspect
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from lucid_machines.kernels.fast_cross_entropy import chunked_cross_entropy_from_hidden, fast_cross_entropy
from lucid_machines.model import MoE_FFN
from lucid_machines.runtime.cuda_graph import CUDAGraphEngine
from lucid_machines.runtime.resource_manager import LiveVRAMMonitor
from lucid_machines.trainer.evaluation import forward_causal_lm


class TrainingEngine:
    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.step = 0
        self.best_eval_loss = float("inf")
        self.target_depth_dist = torch.tensor(
            trainer.config["target_depth_dist"], device=trainer.device, dtype=torch.float32
        )
        self.depth_usage_ema = self.target_depth_dist.clone()
        self.vram_monitor = LiveVRAMMonitor(trainer.device)
        self.cuda_graph = CUDAGraphEngine(trainer.device) if trainer.config.get("cuda_graph", False) else None
        self.sync_streaming_engine()

    def sync_streaming_engine(self) -> None:
        streaming_engine = getattr(self.model, "_streaming_engine", None)
        if streaming_engine is not None and self.optimizer is not None:
            streaming_engine.set_optimizer(self.optimizer)
            streaming_engine.config.grad_clip = self.config.get("grad_clip", 1.0)

    @property
    def model(self):
        return self.trainer.model

    @property
    def optimizer(self):
        return self.trainer.optimizer

    @property
    def config(self):
        return self.trainer.config

    def _exploration_rate(self) -> float:
        mode = self.config["exploration"]
        steps = self.config["exploration_steps"]
        if mode == "none" or steps <= 0:
            return 0.0
        if mode == "linear":
            return max(0.0, 1.0 - self.step / steps)
        if mode == "cosine" and self.step < steps:
            return 0.5 * (1 + math.cos(math.pi * self.step / steps))
        return 0.0

    def _compute_lm_loss(self, logits: torch.Tensor, y_flat: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Compute mean language modeling loss normalized strictly by valid non-padding tokens."""
        pad_id = getattr(self.trainer, "pad_token_id", -100)
        if pad_id is None:
            pad_id = -100
        return fast_cross_entropy(logits.reshape(-1, vocab_size), y_flat, ignore_index=int(pad_id))

    @staticmethod
    def _is_valid_chunked_head(head: object, vocab_size: int, device: torch.device) -> bool:
        if type(head) is not nn.Linear:
            return False
        if getattr(head, "lora_A", None) is not None or getattr(head, "lora_B", None) is not None:
            return False
        weight = getattr(head, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.is_meta or weight.device != device:
            return False
        return int(head.out_features) == int(vocab_size)

    def _forward_lm_loss(
        self,
        input_ids: torch.Tensor,
        y_flat: torch.Tensor,
        vocab_size: int,
        *,
        force_depth: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward model and compute CE loss directly from hidden states via chunked_cross_entropy_from_hidden when safe."""
        pad_id = getattr(self.trainer, "pad_token_id", -100)
        pad_idx = -100 if pad_id is None else int(pad_id)
        use_chunked = bool(self.config.get("use_chunked_ce", True))
        is_capturing = bool(input_ids.is_cuda and torch.cuda.is_current_stream_capturing())

        if use_chunked and not is_capturing:
            chunk_size = max(1, int(self.config.get("ce_chunk_size", 512)))
            base = getattr(self.model, "module", self.model)

            # 1. UniversalDenseTransformer or custom model exposing forward(..., return_hidden=True) + lm_head
            supports_ret_hidden = getattr(base, "_lucid_supports_return_hidden", None)
            if supports_ret_hidden is None:
                try:
                    sig = inspect.signature(base.forward)
                    supports_ret_hidden = "return_hidden" in sig.parameters
                except (TypeError, ValueError):
                    supports_ret_hidden = False
                try:
                    base._lucid_supports_return_hidden = supports_ret_hidden
                except Exception:
                    pass

            lm_head = getattr(base, "lm_head", None)
            if supports_ret_hidden and self._is_valid_chunked_head(lm_head, vocab_size, input_ids.device):
                hidden, _ = self.model(input_ids, return_hidden=True)
                if hidden.dtype != lm_head.weight.dtype:
                    hidden = hidden.to(lm_head.weight.dtype)
                loss = chunked_cross_entropy_from_hidden(
                    hidden, lm_head, y_flat, chunk_size=chunk_size, ignore_index=pad_idx
                )
                return loss, None

            # 2. TriuneTransformer / UniversalMoETransformer / TriuneForCausalLM on full-depth Cortex (force_depth == 2)
            core = getattr(base, "core", getattr(base, "transformer", base))
            final_head = getattr(core, "final_head", None)
            if (
                force_depth == 2
                and hasattr(core, "final_norm")
                and hasattr(core, "layers")
                and self._is_valid_chunked_head(final_head, vocab_size, input_ids.device)
            ):
                captured: dict[str, torch.Tensor | None] = {"h": None}
                orig_head_forward = final_head.forward

                def _capture_head(h: torch.Tensor) -> torch.Tensor:
                    captured["h"] = h
                    return h.new_empty((*h.shape[:-1], 0))

                final_head.forward = _capture_head
                try:
                    _, route_logits = forward_causal_lm(self.model, input_ids, force_depth=2)
                finally:
                    final_head.forward = orig_head_forward

                hidden = captured["h"]
                if hidden is not None:
                    if hidden.dtype != final_head.weight.dtype:
                        hidden = hidden.to(final_head.weight.dtype)
                    loss = chunked_cross_entropy_from_hidden(
                        hidden, final_head, y_flat, chunk_size=chunk_size, ignore_index=pad_idx
                    )
                    return loss, route_logits

            # 3. Hugging Face AutoModelForCausalLM (.model + .lm_head)
            backbone = getattr(base, "model", None)
            if (
                force_depth in (None, 2)
                and isinstance(backbone, nn.Module)
                and getattr(base, "config", None) is not None
                and self._is_valid_chunked_head(lm_head, vocab_size, input_ids.device)
            ):
                try:
                    sig_bb = inspect.signature(backbone.forward)
                    accepts_use_cache = "use_cache" in sig_bb.parameters or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_bb.parameters.values()
                    )
                except (TypeError, ValueError):
                    accepts_use_cache = False
                if accepts_use_cache:
                    bb_out = backbone(input_ids=input_ids, use_cache=False)
                else:
                    bb_out = backbone(input_ids)
                hidden = bb_out.last_hidden_state if hasattr(bb_out, "last_hidden_state") else bb_out[0]
                if hidden.dtype != lm_head.weight.dtype:
                    hidden = hidden.to(lm_head.weight.dtype)
                loss = chunked_cross_entropy_from_hidden(
                    hidden, lm_head, y_flat, chunk_size=chunk_size, ignore_index=pad_idx
                )
                return loss, None

        logits, route_logits = forward_causal_lm(self.model, input_ids, force_depth=force_depth)
        return self._compute_lm_loss(logits, y_flat, vocab_size), route_logits

    def _compute_joint_lm_loss(
        self,
        reflex_logits: torch.Tensor,
        limbic_logits: torch.Tensor,
        cortex_logits: torch.Tensor,
        y_flat: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Combine Reflex/Limbic/Cortex CE losses with optional Cortex-to-exit KL self-distillation."""
        loss_ref = self._compute_lm_loss(reflex_logits, y_flat, vocab_size)
        loss_limb = self._compute_lm_loss(limbic_logits, y_flat, vocab_size)
        loss_cort = self._compute_lm_loss(cortex_logits, y_flat, vocab_size)
        if loss_cort.detach() > loss_limb.detach():
            lm_loss = 0.22 * loss_ref + 0.23 * loss_limb + 0.55 * loss_cort
        else:
            lm_loss = 0.25 * loss_ref + 0.30 * loss_limb + 0.45 * loss_cort
        exit_distill_coef = float(self.config.get("exit_distill_coef", 0.2))
        if exit_distill_coef > 0.0:
            p_cort_det = F.softmax(cortex_logits.detach().float().reshape(-1, vocab_size), dim=-1)
            kl_ref = F.kl_div(
                F.log_softmax(reflex_logits.float().reshape(-1, vocab_size), dim=-1),
                p_cort_det,
                reduction="batchmean",
            )
            kl_limb = F.kl_div(
                F.log_softmax(limbic_logits.float().reshape(-1, vocab_size), dim=-1),
                p_cort_det,
                reduction="batchmean",
            )
            lm_loss = lm_loss + exit_distill_coef * (kl_ref + kl_limb)
        return lm_loss

    def _router_labels_from_logits(self, reflex, limbic, cortex, x, y):
        with torch.no_grad():
            y_flat = y.reshape(-1)
            valid_mask = y_flat != self.trainer.pad_token_id
            valid_per_sample = valid_mask.reshape(x.size(0), -1).sum(dim=1).clamp_min(1)
            losses = []
            pad_id = getattr(self.trainer, "pad_token_id", -100) or -100
            for logits in (reflex, limbic, cortex):
                token_loss = F.cross_entropy(
                    logits.reshape(-1, self.trainer.vocab_size), y_flat, ignore_index=pad_id, reduction="none"
                )
                losses.append((token_loss * valid_mask.float()).reshape(x.size(0), -1).sum(dim=1) / valid_per_sample)
            all_losses = torch.stack(losses, dim=1)
            if self.target_depth_dist.device != all_losses.device:
                self.target_depth_dist = self.target_depth_dist.to(all_losses.device)
                self.depth_usage_ema = self.depth_usage_ema.to(all_losses.device)

            # Mean-center exit losses across the batch so an under-trained exit (e.g. Cortex)
            # is never permanently starved out of argmin, and route harder samples deeper.
            if x.size(0) > 1:
                centered = all_losses - all_losses.mean(dim=0, keepdim=True)
                sample_diff = all_losses[:, 0]
                z_diff = (sample_diff - sample_diff.mean()) / (sample_diff.std(unbiased=False) + 1e-4)
                diff_cost = torch.stack([z_diff, z_diff.abs() * 0.6, -z_diff], dim=1) * 0.35
                cost = 0.65 * centered + diff_cost
            else:
                cost = all_losses - all_losses.mean(dim=1, keepdim=True)

            bias_scale = max(1.0, float(self.config.get("bias_strength", 0.5)))
            adjusted = cost - bias_scale * (self.target_depth_dist - self.depth_usage_ema).unsqueeze(0)
            labels = adjusted.argmin(dim=1)
            usage = F.one_hot(labels, num_classes=3).float().mean(dim=0)
            self.depth_usage_ema.mul_(self.config["usage_ema_decay"]).add_(usage * (1 - self.config["usage_ema_decay"]))
        return labels

    def _router_labels(self, x, y):
        with torch.no_grad():
            slice_size = max(1, min(4, 2048 // max(1, x.size(1))))
            if x.size(0) > slice_size:
                chunk_labels = []
                for chunk_x, chunk_y in zip(x.split(slice_size), y.split(slice_size)):
                    with self.trainer.model_autocast():
                        reflex, limbic, cortex, _ = self.model.forward_all_exits(chunk_x, update_stats=False)
                    lbls = self._router_labels_from_logits(reflex, limbic, cortex, chunk_x, chunk_y)
                    chunk_labels.append(lbls)
                    del reflex, limbic, cortex
                return torch.cat(chunk_labels, dim=0)
            else:
                with self.trainer.model_autocast():
                    reflex, limbic, cortex, _ = self.model.forward_all_exits(x, update_stats=False)
                labels = self._router_labels_from_logits(reflex, limbic, cortex, x, y)
                del reflex, limbic, cortex
                return labels

    def _log_training(
        self, *, loss, lm_loss, router_loss, balance_loss, lr, exploration_rate, overflow, labels, route_logits
    ):
        if self.step % self.config["log_every"]:
            return
        telemetry = self.vram_monitor.format_telemetry()
        vram_gib = self.vram_monitor.last_allocated_gb
        depth_names = ("Reflex", "Limbic", "Cortex")
        chosen = route_logits.argmax(dim=-1)[0].item()
        target = labels[0].item()
        usage = " ".join(f"{name[0]}:{value:.2f}" for name, value in zip(depth_names, self.depth_usage_ema))
        print(
            f"Step {self.step:6d}/{self.config['total_steps']} | Loss: {loss:.4f} "
            f"(LM: {lm_loss:.4f}, Router: {router_loss:.4f}, Bal: {balance_loss:.4f}) "
            f"| Router: {depth_names[chosen]} (target: {depth_names[target]}) "
            f"| Usage: {usage} | LR: {lr:.2e} "
            f"| {telemetry} "
            f"| Best Eval: {self.best_eval_loss:.4f} | Overflow: {overflow}",
            flush=True,
        )
        usage_vals = self.depth_usage_ema.tolist()
        self.trainer.logger.log(
            {
                "train/loss": loss,
                "train/lm_loss": lm_loss,
                "train/router_loss": router_loss,
                "train/balance_loss": balance_loss,
                "train/lr": lr,
                "usage/reflex": usage_vals[0],
                "usage/limbic": usage_vals[1],
                "usage/cortex": usage_vals[2],
                "vram": vram_gib,
                "best_eval_loss": self.best_eval_loss,
                "exploration_rate": exploration_rate,
                "overflow": overflow,
            },
            step=self.step,
        )

    def step_once(self, *, advance_step: bool = False) -> dict:
        """Executes a single canonical LucidMachines training step shared across CLI and Studio."""
        self.model.train()
        self.vram_monitor.tick()
        lr = self.trainer.lr_schedule(self.config, self.step)
        explicit_expert_lr = getattr(self.trainer, "centroid_lr", getattr(self.trainer, "expert_lr", None))
        if explicit_expert_lr is not None:
            peak_lr = max(
                1e-12, float(self.config.get("lr", self.config.get("lr_peak", getattr(self.trainer, "current_lr", lr))))
            )
            scheduled_expert_lr = float(explicit_expert_lr) * (lr / peak_lr)
        else:
            scheduled_expert_lr = None
        try:
            self.trainer.set_optimizer_lr(self.optimizer, lr, expert_lr=scheduled_expert_lr)
        except TypeError:
            self.trainer.set_optimizer_lr(self.optimizer, lr)
        for module in self.model.modules():
            if isinstance(module, MoE_FFN):
                module._global_step = self.step
                module._step_managed_by_engine = True

        streaming_engine = getattr(self.model, "_streaming_engine", None)
        self.optimizer.zero_grad()
        if streaming_engine is not None:
            streaming_engine.zero_grad()

        totals = {"loss": 0.0, "lm": 0.0, "router": 0.0, "balance": 0.0, "z_loss": 0.0, "overflow": 0, "tokens": 0}
        loss_accum = torch.zeros(5, device=self.trainer.device, dtype=torch.float32)
        last_labels = last_route_logits = None
        exploration_rate = self._exploration_rate()
        depth_mode = self.config.get("depth_mode", "dynamic")
        if depth_mode in ("joint", "dynamic") and not hasattr(self.model, "forward_all_exits"):
            depth_mode = "cortex"
        grad_accum = int(getattr(self.trainer, "grad_accum", self.config.get("grad_accum_steps", 1)))
        vocab_size = int(getattr(self.trainer, "vocab_size", getattr(self.model, "vocab_size", 32000)))
        z_loss_coef = float(getattr(self.trainer, "z_loss_coef", 1e-3))
        balance_coef = float(self.config.get("balance_coef", 0.3))
        ema_decay = float(self.config.get("usage_ema_decay", 0.99))

        for _ in range(grad_accum):
            x, y = self.trainer.next_batch()
            if x.device != self.trainer.device:
                x = x.to(self.trainer.device, non_blocking=True)
            if y.device != self.trainer.device:
                y = y.to(self.trainer.device, non_blocking=True)
            totals["tokens"] += int(x.numel())
            y_flat = y.reshape(-1)
            route_logits = torch.zeros((x.size(0), 3), device=self.trainer.device)

            if depth_mode == "cortex":
                labels = torch.full((x.size(0),), 2, dtype=torch.long, device=self.trainer.device)
                max_slice = max(1, min(x.size(0), 4096 // max(1, x.size(1))))
                if x.size(0) > max_slice:
                    sub_xs = x.split(max_slice)
                    sub_ys = y.split(max_slice)
                    num_sub = len(sub_xs)
                    lm_sum = torch.zeros((), device=self.trainer.device, dtype=torch.float32)
                    route_chunks = []
                    for sub_x, sub_y in zip(sub_xs, sub_ys):
                        sub_y_flat = sub_y.reshape(-1)
                        with self.trainer.model_autocast():
                            sub_lm_loss, sub_route_logits = self._forward_lm_loss(
                                sub_x, sub_y_flat, vocab_size, force_depth=2
                            )
                        (sub_lm_loss / (grad_accum * num_sub)).backward()
                        lm_sum += sub_lm_loss.detach().float()
                        if sub_route_logits is not None:
                            route_chunks.append(sub_route_logits.detach())
                    lm_loss = lm_sum / num_sub
                    route_logits = (
                        torch.cat(route_chunks, dim=0)
                        if route_chunks
                        else torch.zeros((x.size(0), 3), device=self.trainer.device)
                    )
                    router_loss = torch.tensor(0.0, device=self.trainer.device)
                    z_loss = torch.tensor(0.0, device=self.trainer.device)
                    balance_loss = torch.tensor(0.0, device=self.trainer.device)
                    micro_loss = lm_loss
                else:
                    with self.trainer.model_autocast():
                        lm_loss, parsed_route = self._forward_lm_loss(x, y_flat, vocab_size, force_depth=2)
                        route_logits = (
                            parsed_route
                            if parsed_route is not None
                            else torch.zeros((x.size(0), 3), device=self.trainer.device)
                        )
                    router_loss = torch.tensor(0.0, device=self.trainer.device)
                    z_loss = torch.tensor(0.0, device=self.trainer.device)
                    balance_loss = torch.tensor(0.0, device=self.trainer.device)
                    micro_loss = lm_loss
                    (micro_loss / grad_accum).backward()
                usage = torch.tensor([0.0, 0.0, 1.0], device=self.depth_usage_ema.device)
                self.depth_usage_ema.mul_(ema_decay).add_(usage * (1.0 - ema_decay))
            elif depth_mode == "joint":
                max_slice = max(1, min(x.size(0), 2048 // max(1, x.size(1))))
                if x.size(0) > max_slice:
                    sub_xs = x.split(max_slice)
                    sub_ys = y.split(max_slice)
                    num_sub = len(sub_xs)
                    component_sums = [
                        torch.zeros((), device=self.trainer.device, dtype=torch.float32) for _ in range(4)
                    ]
                    labels_chunks = []
                    route_chunks = []
                    for sub_x, sub_y in zip(sub_xs, sub_ys):
                        sub_y_flat = sub_y.reshape(-1)
                        with self.trainer.model_autocast():
                            reflex_l, limbic_l, cortex_l, route_l = self.model.forward_all_exits(
                                sub_x, update_stats=True
                            )
                        sub_lbls = self._router_labels_from_logits(reflex_l, limbic_l, cortex_l, sub_x, sub_y)
                        sub_lm = self._compute_joint_lm_loss(reflex_l, limbic_l, cortex_l, sub_y_flat, vocab_size)
                        rf = route_l.float()
                        router_loss_fn = getattr(self.trainer, "router_loss_fn", F.cross_entropy)
                        sub_r_loss = router_loss_fn(rf, sub_lbls)
                        sub_z = torch.logsumexp(rf, dim=-1).pow(2).mean()
                        sub_bal = getattr(self.model, "last_balance_loss", None)
                        if sub_bal is None:
                            sub_bal = (torch.softmax(rf, dim=-1).mean(dim=0) - self.target_depth_dist).pow(2).mean()
                        sub_micro = sub_lm + 0.5 * sub_r_loss + z_loss_coef * sub_z + balance_coef * sub_bal
                        (sub_micro / (grad_accum * num_sub)).backward()
                        component_sums[0] += sub_lm.detach().float()
                        component_sums[1] += sub_r_loss.detach().float()
                        component_sums[2] += sub_bal.detach().float()
                        component_sums[3] += sub_z.detach().float()
                        labels_chunks.append(sub_lbls.detach())
                        route_chunks.append(route_l.detach())
                    lm_loss, router_loss, balance_loss, z_loss = (part / num_sub for part in component_sums)
                    micro_loss = lm_loss + 0.5 * router_loss + z_loss_coef * z_loss + balance_coef * balance_loss
                    labels = torch.cat(labels_chunks, dim=0)
                    route_logits = torch.cat(route_chunks, dim=0)
                else:
                    with self.trainer.model_autocast():
                        reflex_logits, limbic_logits, cortex_logits, route_logits = self.model.forward_all_exits(
                            x, update_stats=True
                        )
                    labels = self._router_labels_from_logits(reflex_logits, limbic_logits, cortex_logits, x, y)
                    lm_loss = self._compute_joint_lm_loss(
                        reflex_logits, limbic_logits, cortex_logits, y_flat, vocab_size
                    )
                    route_f = route_logits.float()
                    router_loss_fn = getattr(self.trainer, "router_loss_fn", F.cross_entropy)
                    router_loss = router_loss_fn(route_f, labels)
                    z_loss = torch.logsumexp(route_f, dim=-1).pow(2).mean()
                    balance_loss = getattr(self.model, "last_balance_loss", None)
                    if balance_loss is None:
                        balance_loss = (
                            (torch.softmax(route_f, dim=-1).mean(dim=0) - self.target_depth_dist).pow(2).mean()
                        )
                    micro_loss = lm_loss + 0.5 * router_loss + z_loss_coef * z_loss + balance_coef * balance_loss
                    (micro_loss / grad_accum).backward()
            else:
                labels = self._router_labels(x, y)
                chosen_depth = random.choice((0, 1, 2)) if random.random() < exploration_rate else None
                max_slice = max(1, min(x.size(0), 2048 // max(1, x.size(1))))
                if x.size(0) > max_slice:
                    sub_xs = x.split(max_slice)
                    sub_ys = y.split(max_slice)
                    sub_lbls = labels.split(max_slice)
                    num_sub = len(sub_xs)
                    component_sums = [
                        torch.zeros((), device=self.trainer.device, dtype=torch.float32) for _ in range(4)
                    ]
                    route_chunks = []
                    for sub_x, sub_y, sub_lbl in zip(sub_xs, sub_ys, sub_lbls):
                        sub_y_flat = sub_y.reshape(-1)
                        with self.trainer.model_autocast():
                            sub_lm_loss, sub_route_logits = self._forward_lm_loss(
                                sub_x, sub_y_flat, vocab_size, force_depth=chosen_depth
                            )
                            if sub_route_logits is None:
                                sub_route_logits = torch.zeros((sub_x.size(0), 3), device=self.trainer.device)
                        sub_route_f = sub_route_logits.float()
                        router_loss_fn = getattr(self.trainer, "router_loss_fn", F.cross_entropy)
                        sub_r_loss = router_loss_fn(sub_route_f, sub_lbl)
                        sub_z = torch.logsumexp(sub_route_f, dim=-1).pow(2).mean()
                        sub_bal = getattr(self.model, "last_balance_loss", None)
                        if sub_bal is None:
                            sub_bal = (
                                (torch.softmax(sub_route_f, dim=-1).mean(dim=0) - self.target_depth_dist).pow(2).mean()
                            )
                        sub_micro = sub_lm_loss + 0.5 * sub_r_loss + z_loss_coef * sub_z + balance_coef * sub_bal
                        (sub_micro / (grad_accum * num_sub)).backward()
                        component_sums[0] += sub_lm_loss.detach().float()
                        component_sums[1] += sub_r_loss.detach().float()
                        component_sums[2] += sub_bal.detach().float()
                        component_sums[3] += sub_z.detach().float()
                        route_chunks.append(sub_route_logits.detach())
                    lm_loss, router_loss, balance_loss, z_loss = (part / num_sub for part in component_sums)
                    micro_loss = lm_loss + 0.5 * router_loss + z_loss_coef * z_loss + balance_coef * balance_loss
                    route_logits = torch.cat(route_chunks, dim=0)
                else:
                    with self.trainer.model_autocast():
                        lm_loss, route_logits = self._forward_lm_loss(
                            x, y_flat, vocab_size, force_depth=chosen_depth
                        )
                        if route_logits is None:
                            route_logits = torch.zeros((x.size(0), 3), device=self.trainer.device)
                        last_choice = getattr(self.model, "last_depth_choice", None)
                        saved_balance = getattr(self.model, "last_balance_loss", None)
                        cortex_anchor_loss = None
                        if (
                            streaming_engine is None
                            and chosen_depth is None
                            and last_choice is not None
                            and not (last_choice == 2).any()
                        ):
                            cortex_anchor_loss, _ = self._forward_lm_loss(
                                x[:1], y[:1].reshape(-1), vocab_size, force_depth=2
                            )
                            self.model.last_route_logits = route_logits
                            self.model.last_depth_choice = last_choice
                            self.model.last_balance_loss = saved_balance
                    route_f = route_logits.float()
                    if cortex_anchor_loss is not None:
                        lm_loss = 0.80 * lm_loss + 0.20 * cortex_anchor_loss
                    router_loss_fn = getattr(self.trainer, "router_loss_fn", F.cross_entropy)
                    router_loss = router_loss_fn(route_f, labels)
                    z_loss = torch.logsumexp(route_f, dim=-1).pow(2).mean()
                    balance_loss = getattr(self.model, "last_balance_loss", None)
                    if balance_loss is None:
                        balance_loss = (
                            (torch.softmax(route_f, dim=-1).mean(dim=0) - self.target_depth_dist).pow(2).mean()
                        )
                    micro_loss = lm_loss + 0.5 * router_loss + z_loss_coef * z_loss + balance_coef * balance_loss
                    (micro_loss / grad_accum).backward()

            loss_accum[0] += micro_loss.detach()
            loss_accum[1] += lm_loss.detach()
            loss_accum[2] += router_loss.detach()
            loss_accum[3] += balance_loss.detach()
            loss_accum[4] += z_loss.detach()
            for module in self.model.modules():
                if isinstance(module, MoE_FFN):
                    totals["overflow"] += module.overflow_counter
                    module.overflow_counter = 0
            last_labels, last_route_logits = labels, route_logits.detach()

        if streaming_engine is not None:
            streaming_engine.step_streaming_optimizer(optimizer=self.optimizer, grad_clip=self.config["grad_clip"])
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.optimizer.step()

        for module in self.model.modules():
            if isinstance(module, MoE_FFN) and hasattr(module, "step_bias"):
                module.step_bias()
        self.model.last_route_logits = None
        self.model.last_depth_choice = None
        self.model.last_balance_loss = None
        loss_vals = (loss_accum / grad_accum).tolist()
        totals["loss"] = loss_vals[0]
        totals["lm"] = loss_vals[1]
        totals["router"] = loss_vals[2]
        totals["balance"] = loss_vals[3]
        totals["z_loss"] = loss_vals[4]

        executed_step = self.step
        if advance_step:
            self.step += 1

        return {
            "step": executed_step,
            "loss": totals["loss"],
            "lm_loss": totals["lm"],
            "router_loss": totals["router"],
            "balance_loss": totals["balance"],
            "z_loss": totals["z_loss"],
            "lr": lr,
            "exploration_rate": exploration_rate,
            "overflow": totals["overflow"],
            "labels": last_labels,
            "route_logits": last_route_logits,
            "tokens_in_step": totals["tokens"],
            "depth_usage_ema": self.depth_usage_ema.detach().clone(),
        }

    def train(self):
        self.model.train()
        try:
            while self.step < self.config["total_steps"]:
                step_stats = self.step_once(advance_step=False)

                if self.step and self.step % self.config["eval_every"] == 0:
                    cortex_loss = self.trainer.run_eval(force_depth=2)
                    dynamic_loss = self.trainer.run_eval()
                    if cortex_loss < self.best_eval_loss:
                        self.best_eval_loss = cortex_loss
                        self.trainer.save_best(self.step, cortex_loss)
                    sample = self.trainer.run_sample()
                    print(
                        f"Eval cortex: {cortex_loss:.4f} | dynamic: {dynamic_loss:.4f} | perplexity: {math.exp(min(cortex_loss, 20)):.2f}"
                    )
                    print(f"Sample: {self.trainer.sample_prompt}{sample}\n")
                    self.trainer.logger.log(
                        {
                            "eval/loss": cortex_loss,
                            "eval/dynamic_loss": dynamic_loss,
                            "eval/perplexity": math.exp(min(cortex_loss, 20)),
                        },
                        step=self.step,
                    )

                if self.step and self.step % self.config["save_every"] == 0:
                    self.trainer.save_latest(self.step, step_stats["loss"])
                self._log_training(
                    loss=step_stats["loss"],
                    lm_loss=step_stats["lm_loss"],
                    router_loss=step_stats["router_loss"],
                    balance_loss=step_stats["balance_loss"],
                    lr=step_stats["lr"],
                    exploration_rate=step_stats["exploration_rate"],
                    overflow=step_stats["overflow"],
                    labels=step_stats["labels"],
                    route_logits=step_stats["route_logits"],
                )
                self.step += 1
        except BaseException:
            self.trainer.save_latest(self.step, 0.0)
            raise

        if self.step:
            self.trainer.save_latest(self.step, 0.0)
        return {"step": self.step, "best_eval_loss": self.best_eval_loss}
