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
    progress = min(max(0, rel_step - warmup_steps) / max(1, cycle_steps - warmup_steps), 1.0)
    return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr: float) -> None:
    if hasattr(optimizer, "set_lr"):
        optimizer.set_lr(lr)
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
