"""Stateful execution engine for Triune training."""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from triune.model import MoE_FFN
from triune.runtime.resource_manager import LiveVRAMMonitor


class TrainingEngine:
    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.step = 0
        self.best_eval_loss = float("inf")
        self.target_depth_dist = torch.tensor(trainer.config["target_depth_dist"], device=trainer.device)
        self.depth_usage_ema = self.target_depth_dist.clone()
        self.vram_monitor = LiveVRAMMonitor(trainer.device)
        streaming_engine = getattr(self.model, "_streaming_engine", None)
        if streaming_engine is not None:
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

    def _router_labels(self, x, y):
        with torch.no_grad():
            reflex, limbic, cortex, _ = self.model.forward_all_exits(x, update_stats=False)
            y_flat = y.reshape(-1)
            valid_mask = y_flat != self.trainer.pad_token_id
            valid_per_sample = valid_mask.reshape(x.size(0), -1).sum(dim=1).clamp_min(1)
            losses = []
            for logits in (reflex, limbic, cortex):
                token_loss = self.trainer.loss_fn(logits.reshape(-1, self.trainer.vocab_size), y_flat)
                losses.append((token_loss * valid_mask.float()).reshape(x.size(0), -1).sum(dim=1) / valid_per_sample)
            del reflex, limbic, cortex, logits
            all_losses = torch.stack(losses, dim=1)
            adjusted = all_losses - self.config["bias_strength"] * (self.target_depth_dist - self.depth_usage_ema).unsqueeze(0)
            labels = adjusted.argmin(dim=1)
            usage = F.one_hot(labels, num_classes=3).float().mean(dim=0)
            self.depth_usage_ema.mul_(self.config["usage_ema_decay"]).add_(usage * (1 - self.config["usage_ema_decay"]))
        return labels

    def _log_training(self, *, loss, lm_loss, router_loss, balance_loss, lr, exploration_rate, overflow, labels, route_logits):
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
            flush=True
        )
        self.trainer.logger.log(
            {
                "train/loss": loss,
                "train/lm_loss": lm_loss,
                "train/router_loss": router_loss,
                "train/balance_loss": balance_loss,
                "train/lr": lr,
                "usage/reflex": self.depth_usage_ema[0].item(),
                "usage/limbic": self.depth_usage_ema[1].item(),
                "usage/cortex": self.depth_usage_ema[2].item(),
                "vram": vram_gib,
                "best_eval_loss": self.best_eval_loss,
                "exploration_rate": exploration_rate,
                "overflow": overflow,
            },
            step=self.step,
        )

    def train(self):
        self.model.train()
        try:
            while self.step < self.config["total_steps"]:
                self.vram_monitor.tick()
                lr = self.trainer.lr_schedule(self.config, self.step)
                self.trainer.set_optimizer_lr(self.optimizer, lr)
                for module in self.model.modules():
                    if isinstance(module, MoE_FFN):
                        module._global_step = self.step

                streaming_engine = getattr(self.model, "_streaming_engine", None)
                self.optimizer.zero_grad()
                if streaming_engine is not None:
                    streaming_engine.zero_grad()

                totals = {"loss": 0.0, "lm": 0.0, "router": 0.0, "balance": 0.0, "overflow": 0}
                last_labels = last_route_logits = None
                exploration_rate = self._exploration_rate()

                for _ in range(self.trainer.grad_accum):
                    x, y = self.trainer.next_batch()
                    x, y = x.to(self.trainer.device), y.to(self.trainer.device)
                    labels = self._router_labels(x, y)
                    # Intentional batched exploration: same forced depth across all sequences in the batch
                    # for gradient correlation. Per-sequence exploration would require per-item depth tensors.
                    chosen_depth = random.choice((0, 1, 2)) if random.random() < exploration_rate else None

                    with self.trainer.model_autocast():
                        logits, route_logits = self.model(x, force_depth=chosen_depth)
                    y_flat = y.reshape(-1)
                    logits_f = logits.float()
                    route_f = route_logits.float()
                    lm_loss = self.trainer.loss_fn(logits_f.reshape(-1, self.trainer.vocab_size), y_flat).mean()
                    router_loss = self.trainer.router_loss_fn(route_logits, labels)
                    z_loss = torch.logsumexp(route_f, dim=-1).pow(2).mean()
                    # Use router's native balance loss if recorded during forward pass, else compute from softmax distribution
                    balance_loss = getattr(self.model, "last_balance_loss", None)
                    if balance_loss is None:
                        balance_loss = (torch.softmax(route_f, dim=-1).mean(dim=0) - self.target_depth_dist).pow(2).mean()
                    micro_loss = lm_loss + 0.5 * router_loss + self.trainer.z_loss_coef * z_loss + self.config["balance_coef"] * balance_loss
                    (micro_loss / self.trainer.grad_accum).backward()

                    totals["loss"] += micro_loss.item()
                    totals["lm"] += lm_loss.item()
                    totals["router"] += router_loss.item()
                    totals["balance"] += balance_loss.item()
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
                # overflow is a cumulative count, not an average
                for key in ("loss", "lm", "router", "balance"):
                    totals[key] /= self.trainer.grad_accum

                if self.step and self.step % self.config["eval_every"] == 0:
                    cortex_loss = self.trainer.run_eval(force_depth=2)
                    dynamic_loss = self.trainer.run_eval()
                    if cortex_loss < self.best_eval_loss:
                        self.best_eval_loss = cortex_loss
                        self.trainer.save_best(self.step, cortex_loss)
                    sample = self.trainer.run_sample()
                    print(f"Eval cortex: {cortex_loss:.4f} | dynamic: {dynamic_loss:.4f} | perplexity: {math.exp(min(cortex_loss, 20)):.2f}")
                    print(f"Sample: {self.trainer.sample_prompt}{sample}\n")
                    self.trainer.logger.log(
                        {"eval/loss": cortex_loss, "eval/dynamic_loss": dynamic_loss, "eval/perplexity": math.exp(min(cortex_loss, 20))},
                        step=self.step,
                    )

                if self.step and self.step % self.config["save_every"] == 0:
                    self.trainer.save_latest(self.step, totals["loss"])
                self._log_training(
                    loss=totals["loss"], lm_loss=totals["lm"], router_loss=totals["router"],
                    balance_loss=totals["balance"], lr=lr, exploration_rate=exploration_rate,
                    overflow=totals["overflow"], labels=last_labels, route_logits=last_route_logits,
                )
                self.step += 1
        except BaseException:
            self.trainer.save_latest(self.step, 0.0)
            raise

        if self.step:
            self.trainer.save_latest(self.step, 0.0)
        return {"step": self.step, "best_eval_loss": self.best_eval_loss}
