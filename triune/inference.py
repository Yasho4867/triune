"""Callable checkpoint loading and text generation utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .configs.config import build_config
from .data.tokenizer import load_tokenizer
from .model.factory import build_model


def load_checkpoint_model(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path = "triune_tokenizer.json",
    *,
    device: str | torch.device | None = None,
):
    """Load a training checkpoint for generation and return model plus tokenizer."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = load_tokenizer(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    eos_token_id = tokenizer.token_to_id("[SEP]")
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define [PAD] and [SEP] special tokens")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    saved_config = checkpoint.get("config", {})
    use_fp4 = saved_config.get(
        "use_fp4", any(".ffn.experts." in key and ".linear.weight" in key for key in state_dict)
    )
    config = build_config({
        "vocab_size": saved_config.get("vocab_size", tokenizer.get_vocab_size()),
        "hidden_dim": saved_config.get("hidden_dim"),
        "num_layers": saved_config.get("num_layers"),
        "use_fp4": use_fp4,
    })
    model = build_model(config)
    model.load_state_dict(state_dict)
    model = model.to(device).bfloat16().eval()
    return model, tokenizer, checkpoint, device, pad_token_id, eos_token_id


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str | torch.device,
    pad_token_id: int,
    eos_token_id: int,
    force_depth: int | None = None,
    temperature: float = 0.7,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 100,
) -> str:
    """Generate a response using the model's established depth-routing options."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if repetition_penalty < 0:
        raise ValueError("repetition_penalty must be non-negative")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    formatted = f"User: {prompt}\nAssistant:"
    ids = torch.tensor(tokenizer.encode(formatted).ids, dtype=torch.long, device=device).unsqueeze(0)
    generated, seen = [], set()
    
    # Process full prompt once, initialize cache
    cache = model.init_cache(batch_size=1)
    logits, cache = model(ids, cache=cache, force_depth=force_depth)
    
    for _ in range(max_new_tokens):
        next_logits = logits[0, -1].float() / temperature
        
        # Standard multiplicative repetition penalty
        for token_id in seen:
            if next_logits[token_id] > 0:
                next_logits[token_id] /= repetition_penalty
            else:
                next_logits[token_id] *= repetition_penalty
                
        # Top-k filtering
        top_k = min(50, next_logits.size(-1))
        indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
        next_logits[indices_to_remove] = float('-inf')
        
        next_logits[pad_token_id] = -torch.inf
        
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id == eos_token_id:
            break
        generated.append(next_id)
        seen.add(next_id)
        
        # Feed ONLY the new token with cache — O(1) per step
        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, cache = model(next_input, cache=cache, force_depth=force_depth)
        
    return tokenizer.decode(generated)
