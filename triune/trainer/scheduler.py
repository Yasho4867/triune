import math


def lr_schedule(config: dict, step: int) -> float:
    warmup_steps = config["warmup_steps"]
    total_steps = config["total_steps"]
    if warmup_steps and step < warmup_steps:
        return config["lr"] * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return config["min_lr"]
    progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
    return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr: float) -> None:
    if hasattr(optimizer, "set_lr"):
        optimizer.set_lr(lr)
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
