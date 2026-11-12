"""Callable checkpoint loading and text generation utilities."""

from __future__ import annotations

from pathlib import Path

import torch

from .configs.config import build_config
from .data.tokenizer import load_tokenizer
from .model.factory import build_model


def load_checkpoint_model(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path = "triune_tokenizer.json",
    *,
    device: str | torch.device | None = None,
    adapter_path: str | Path | None = None,
):
    """Load a training checkpoint for generation and return model plus tokenizer."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = load_tokenizer(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    eos_token_id = tokenizer.token_to_id("[SEP]")
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define [PAD] and [SEP] special tokens")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state", checkpoint)
    if any(str(key).startswith("_orig_mod.") for key in state_dict):
        state_dict = {str(key).removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    saved_config = checkpoint.get("config") or {}
    if not isinstance(saved_config, dict):
        saved_config = {}
    use_fp4 = saved_config.get("use_fp4", any(".ffn.experts." in key and ".linear.weight" in key for key in state_dict))

    import re

    embed_weight = state_dict.get("token_embed.weight")
    inferred_vocab = (
        int(embed_weight.shape[0])
        if embed_weight is not None and embed_weight.ndim == 2
        else tokenizer.get_vocab_size()
    )
    inferred_dim = int(embed_weight.shape[1]) if embed_weight is not None and embed_weight.ndim == 2 else None
    layer_ids = [int(m.group(1)) for k in state_dict if (m := re.match(r"layers\.(\d+)\.", str(k)))]
    expert_ids = [int(m.group(1)) for k in state_dict if (m := re.match(r"layers\.\d+\.ffn\.experts\.(\d+)\.", str(k)))]
    exit_ids = [int(m.group(1)) for k in state_dict if (m := re.match(r"layers\.(\d+)\.exit_head\.weight$", str(k)))]
    inferred_layers = (max(layer_ids) + 1) if layer_ids else None
    inferred_experts = (max(expert_ids) + 1) if expert_ids else None

    hidden_dim = saved_config.get("hidden_dim") or inferred_dim
    num_layers = saved_config.get("num_layers") or inferred_layers
    head_dim = saved_config.get("head_dim") or (
        64 if hidden_dim and hidden_dim % 64 == 0 and hidden_dim < 1536 else 128
    )
    num_heads = saved_config.get("num_heads") or (hidden_dim // head_dim if hidden_dim and head_dim else None)
    reflex_exit = saved_config.get("reflex_exit_layer") or (min(exit_ids) if len(exit_ids) >= 2 else None)
    limbic_exit = saved_config.get("limbic_exit_layer") or (max(exit_ids) if len(exit_ids) >= 2 else None)
    prefix_layers = saved_config.get("router_prefix_layers") or (
        1 if reflex_exit is not None and reflex_exit <= 3 else None
    )

    config = build_config(
        {
            "vocab_size": saved_config.get("vocab_size", inferred_vocab),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_experts": saved_config.get("num_experts") or inferred_experts,
            "router_prefix_layers": prefix_layers,
            "reflex_exit_layer": reflex_exit,
            "limbic_exit_layer": limbic_exit,
            "use_fp4": use_fp4,
        }
    )
    model = build_model(config)
    model.load_state_dict(state_dict)
    model = model.to(device).bfloat16().eval()
    if adapter_path is not None and Path(adapter_path).exists():
        from triune.trainer.finetune import LoRAConfig, TriuneFineTuner

        cfg = LoRAConfig()
        tuner = TriuneFineTuner(model, cfg)
        tuner.load_adapter(adapter_path)
    return model, tokenizer, checkpoint, device, pad_token_id, eos_token_id


class IncrementalStreamingDecoder:
    def __init__(self, tokenizer, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_cache: list[int] = []
        self.print_len = 0
        self.cumulative_text = ""

    def step(self, token_id: int) -> tuple[str, str]:
        self.token_cache.append(token_id)
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
        if text.endswith("\ufffd"):
            return self.cumulative_text, ""
        delta = text[self.print_len:]
        self.cumulative_text += delta
        if text.endswith(" ") or text.endswith("\n") or len(self.token_cache) >= 16:
            self.token_cache = []
            self.print_len = 0
        else:
            self.print_len = len(text)
        return self.cumulative_text, delta

    def flush(self) -> tuple[str, str]:
        if self.token_cache:
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=self.skip_special_tokens)
            if self.print_len < len(text):
                delta = text[self.print_len:]
                self.cumulative_text += delta
                self.token_cache = []
                self.print_len = 0
                return self.cumulative_text, delta
        return self.cumulative_text, ""


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
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 100,
    prompt_is_formatted: bool = False,
    cancel_event=None,
    on_text=None,
    return_info: bool = False,
    cache=None,
) -> str | dict[str, object]:
    """Generate a response using the model's established depth-routing options."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    import time

    formatted = prompt if prompt_is_formatted else f"User: {prompt}\nAssistant:"
    token_ids = tokenizer.encode(formatted).ids
    if not token_ids:
        raise ValueError("The configured tokenizer returned no prompt tokens")
    vocab_size = tokenizer.get_vocab_size()
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise ValueError("Tokenizer prompt IDs exceed the active model vocabulary")
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: list[int] = []
    seen = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    token_counts = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    if cache is None and hasattr(model, "init_cache"):
        cache = model.init_cache(batch_size=1)
    stream_decoder = IncrementalStreamingDecoder(tokenizer) if on_text is not None else None
    started = time.perf_counter()
    first_token_latency_ms: float | None = None
    decode_started: float | None = None
    with torch.inference_mode():
        prompt_len = ids.size(1)
        prefill_chunk_size = 1024
        if prompt_len > prefill_chunk_size and cache is not None:
            # Chunked prompt prefill: processes long texts (up to 8,192+ tokens) in constant-memory slices
            # Keeps peak activation VRAM bounded and enables responsive early cancellation
            for start_idx in range(0, prompt_len - prefill_chunk_size, prefill_chunk_size):
                if cancel_event is not None and cancel_event.is_set():
                    return (
                        {
                            "text": "",
                            "cancelled": True,
                            "tokens_count": 0,
                            "tokens_per_sec": 0.0,
                            "latency_ms": 0.0,
                            "route": "AUTO",
                        }
                        if return_info
                        else ""
                    )
                chunk_ids = ids[:, start_idx : start_idx + prefill_chunk_size]
                _, cache = model(chunk_ids, cache=cache, force_depth=force_depth)
            last_start = ((prompt_len - 1) // prefill_chunk_size) * prefill_chunk_size
            rem_ids = ids[:, last_start:]
            logits, cache = model(rem_ids, cache=cache, force_depth=force_depth)
        elif cache is not None:
            logits, cache = model(ids, cache=cache, force_depth=force_depth)
        else:
            logits, _ = model(ids, force_depth=force_depth)
        # A recurrent cache belongs to one exit path; keep the prompt's choice
        # stable through decoding instead of switching into stale layer states.
        decode_depth = force_depth
        if decode_depth is None and getattr(model, "last_depth_choice", None) is not None:
            decode_depth = int(model.last_depth_choice[0].item())

        cancelled = False
        for _ in range(min(max_new_tokens, 8192)):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            next_logits = logits[0, -1].float()
            if temperature > 0:
                next_logits = next_logits / temperature
            if repetition_penalty != 1.0:
                penalized = torch.where(
                    next_logits > 0,
                    next_logits / repetition_penalty,
                    next_logits * repetition_penalty,
                )
                next_logits = torch.where(seen, penalized, next_logits)
                # Frequency penalty prevents degenerate punctuation/token loops over long sequences
                next_logits = next_logits - 0.35 * (token_counts - 1.0).clamp_min(0.0)
            if 0 <= pad_token_id < next_logits.numel():
                next_logits[pad_token_id] = float("-inf")
            if temperature == 0:
                # Top-k masking cannot change an argmax result, so avoid the
                # extra vocabulary-wide topk kernel on deterministic decoding.
                next_input = torch.argmax(next_logits).view(1, 1)
                next_id = int(next_input.item())
            else:
                effective_top_k = min(top_k, next_logits.numel())
                top_vals, top_indices = torch.topk(next_logits, effective_top_k)
                sampled_pos = torch.multinomial(torch.softmax(top_vals, dim=-1), 1)
                next_input = top_indices.gather(0, sampled_pos).view(1, 1)
                next_id = int(next_input.item())
            if next_id == eos_token_id:
                break
            generated.append(next_id)
            if first_token_latency_ms is None:
                # next_id.item()/multinomial.item() synchronizes pending device
                # work, so this includes prompt prefill and first-token sampling.
                decode_started = time.perf_counter()
                first_token_latency_ms = (decode_started - started) * 1000.0
            if stream_decoder is not None:
                cur_tier = 0 if decode_depth is None else int(decode_depth)
                cur_route = "AUTO" if decode_depth is None else ("REFLEX", "LIMBIC", "CORTEX")[decode_depth]
                def _emit_text(txt: str) -> None:
                    if on_text is not None:
                        try:
                            on_text(txt, tier=cur_tier, route=cur_route, token_id=next_id)
                        except TypeError:
                            on_text(txt)

                cum_text, delta = stream_decoder.step(next_id)
                if delta:
                    hit_stop = False
                    for stop_seq in ("\nUser:", "\nSystem:", "User:", "Assistant:"):
                        if stop_seq in cum_text:
                            clean_text = cum_text.split(stop_seq, 1)[0]
                            _emit_text(clean_text)
                            hit_stop = True
                            break
                    if hit_stop:
                        break
                    _emit_text(cum_text)
            seen[next_id] = True
            token_counts[next_id] += 1.0
            if cache is not None:
                logits, cache = model(next_input, cache=cache, force_depth=decode_depth)
            else:
                ids = torch.cat((ids, next_input), dim=1)
                logits, _ = model(ids, force_depth=decode_depth)

        if stream_decoder is not None:
            cum_text, delta = stream_decoder.flush()
            if delta:
                _emit_text(cum_text)

    finished = time.perf_counter()
    elapsed = max(finished - started, 1e-3)
    decode_elapsed = max(finished - decode_started, 1e-3) if decode_started is not None else 0.0
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    for stop_seq in ("\nUser:", "\nSystem:", "User:", "Assistant:"):
        if stop_seq in text:
            text = text.split(stop_seq, 1)[0].strip()
    if not return_info:
        return text
    route_name = "AUTO" if decode_depth is None else ("REFLEX", "LIMBIC", "CORTEX")[decode_depth]
    return {
        "text": text,
        "tokens_count": len(generated),
        "tokens_per_sec": len(generated) / elapsed,
        "latency_ms": round(elapsed * 1000, 1),
        "ttft_ms": round(first_token_latency_ms, 1) if first_token_latency_ms is not None else None,
        "decode_tokens_per_sec": round(max(0, len(generated) - 1) / decode_elapsed, 2) if decode_elapsed else None,
        "route": route_name,
        "cancelled": cancelled,
        "cache": cache,
    }
