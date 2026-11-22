import math


def lr_schedule(config: dict, step: int) -> float:
    warmup_steps = config.get("warmup_steps", 150)
    cycle_start = config.get("cycle_start_step", 0)
    cycle_steps = config.get("cycle_steps", config.get("total_steps", 25000))

    rel_step = max(0, step - cycle_start)
    if warmup_steps and rel_step < warmup_steps:
        return config["lr"] * (rel_step + 1) / warmup_steps
    if cycle_steps <= warmup_steps:
        return config["min_lr"]
    span = max(1, cycle_steps - warmup_steps)
    if rel_step > cycle_steps:
        # When training continues past cycle_steps, wrap into a fresh cosine cycle instead of pinning at min_lr
        rel_progress = ((rel_step - warmup_steps) % span) / span
    else:
        rel_progress = min(max(0, rel_step - warmup_steps) / span, 1.0)
    return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * rel_progress))


def set_optimizer_lr(optimizer, lr: float, expert_lr: float | None = None) -> None:
    if hasattr(optimizer, "set_lr"):
        if expert_lr is not None:
            try:
                optimizer.set_lr(lr, expert_lr=expert_lr)
                return
            except TypeError:
                pass
        optimizer.set_lr(lr)
    else:
        for param_group in optimizer.param_groups:
            if param_group.get("is_expert", False) and expert_lr is not None:
                param_group["lr"] = expert_lr
            else:
                param_group["lr"] = lr
