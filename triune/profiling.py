"""Bounded training-step profiling utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from .configs import build_config
from .data import load_tokenizer
from .model import build_model


def profile_model(
    *,
    checkpoint_path: str | Path = "checkpoints_full/latest.pt",
    tokenizer_path: str | Path = "triune_tokenizer.json",
    output_path: str | Path = "profiler_output.txt",
    batch_size: int = 4,
    seq_len: int = 256,
    warmup_steps: int = 5,
    active_steps: int = 10,
) -> str:
    """Profile a bounded number of Cortex training steps and write the table."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profiling requires CUDA")
    device = torch.device("cuda")
    tokenizer = load_tokenizer(tokenizer_path)
    config = build_config({"vocab_size": tokenizer.get_vocab_size(), "seq_len": seq_len})
    state_dict = None
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state"]
        saved = checkpoint.get("config", {})
        config = build_config({
            "vocab_size": saved.get("vocab_size", tokenizer.get_vocab_size()),
            "hidden_dim": saved.get("hidden_dim"),
            "num_layers": saved.get("num_layers"),
            "use_fp4": saved.get("use_fp4", any(".ffn.experts." in key and ".linear.weight" in key for key in state_dict)),
            "seq_len": seq_len,
        })
        if any(key.startswith("_orig_mod.") for key in state_dict):
            state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    model = build_model(config).to(device).bfloat16().train()
    if state_dict is not None:
        model.load_state_dict(state_dict)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    def training_step():
        model.zero_grad(set_to_none=True)
        inputs = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
        labels = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs, force_depth=2)
            loss = loss_fn(logits.reshape(-1, config["vocab_size"]), labels.reshape(-1))
        loss.backward()

    for _ in range(warmup_steps):
        training_step()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as profiler:
        for _ in range(active_steps):
            training_step()
    table = profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    Path(output_path).write_text(table, encoding="utf-8")
    return table
