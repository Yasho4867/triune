"""Callable checkpoint loading and text generation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .configs.config import build_config
from .data.tokenizer import load_tokenizer
from .model.factory import build_model

_GEMMA_RMS_NORM_CLASSES = frozenset({"GemmaRMSNorm", "Gemma2RMSNorm", "Gemma3RMSNorm"})
_STD_RMS_NORM_CLASSES = frozenset({"LlamaRMSNorm", "Qwen2RMSNorm", "Qwen3RMSNorm", "MistralRMSNorm"})
_SUPPORTED_HF_CG_CLASSES = frozenset(
    {
        "Gemma3ForCausalLM",
        "Gemma2ForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
    }
)
_TRIUNE_STATIC_CACHE_CLS = None


def optimize_hf_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Patch eager HF RMSNorm layers to fused ``torch.nn.functional.rms_norm`` during inference.

    Preserves exact gradient-enabled training paths when ``torch.is_grad_enabled()``
    or ``module.training`` is True, and stores any pre-shifted ``(1 + weight)`` Gemma norm
    scales as non-persistent buffers so ``state_dict()`` remains unmodified.
    """
    if not hasattr(F, "rms_norm"):
        return model
    if getattr(model, "_streaming_engine", None) is not None or getattr(model, "_layer_streaming_active", False):
        return model

    for mod in model.modules():
        if getattr(mod, "_triune_rms_fused", False):
            continue
        cls_name = type(mod).__name__
        is_gemma = cls_name in _GEMMA_RMS_NORM_CLASSES
        if not is_gemma and cls_name not in _STD_RMS_NORM_CLASSES:
            continue
        weight = getattr(mod, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 1
            or weight.numel() == 0
            or weight.device.type == "meta"
        ):
            continue

        eps = float(getattr(mod, "eps", getattr(mod, "variance_epsilon", 1e-6)))
        norm_shape = (int(weight.shape[0]),)
        orig_forward = mod.forward

        if is_gemma:
            w_eff = (1.0 + weight.detach().float()).to(dtype=weight.dtype, device=weight.device)
            mod.register_buffer("_triune_w_eff", w_eff, persistent=False)

            def _fused_gemma_rms(x: torch.Tensor, _m=mod, _orig=orig_forward, _shape=norm_shape, _eps=eps):
                if torch.is_grad_enabled() or _m.training:
                    return _orig(x)
                w = _m._triune_w_eff
                if w.device != x.device or w.dtype != x.dtype:
                    w = (1.0 + _m.weight.detach().float()).to(device=x.device, dtype=x.dtype)
                out = F.rms_norm(x, _shape, w, _eps)
                return out if out.dtype == x.dtype else out.to(dtype=x.dtype)

            mod.forward = _fused_gemma_rms
        else:

            def _fused_std_rms(x: torch.Tensor, _m=mod, _orig=orig_forward, _shape=norm_shape, _eps=eps):
                if torch.is_grad_enabled() or _m.training:
                    return _orig(x)
                w = _m.weight
                if w.dtype != x.dtype:
                    w = w.to(dtype=x.dtype)
                out = F.rms_norm(x, _shape, w, _eps)
                return out if out.dtype == x.dtype else out.to(dtype=x.dtype)

            mod.forward = _fused_std_rms

        mod._triune_rms_fused = True

    return model


def _get_static_cache_cls():
    global _TRIUNE_STATIC_CACHE_CLS
    if _TRIUNE_STATIC_CACHE_CLS is not None:
        return _TRIUNE_STATIC_CACHE_CLS
    from transformers.cache_utils import Cache

    class _TriuneStaticCache(Cache):
        """Pre-allocated static KV cache supporting multi-token prefill and 1-token CUDA-graph decode."""

        def __init__(
            self,
            n_layers: int,
            n_kv_heads: int,
            head_dim: int,
            max_cache_len: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            self.layers = []
            self.keys = [
                torch.zeros(1, n_kv_heads, max_cache_len, head_dim, dtype=dtype, device=device) for _ in range(n_layers)
            ]
            self.values = [
                torch.zeros(1, n_kv_heads, max_cache_len, head_dim, dtype=dtype, device=device) for _ in range(n_layers)
            ]
            self._max_cache_len = int(max_cache_len)
            self._n_layers = int(n_layers)
            self._seen = 0
            self._prefill_mode = True
            self._write_pos = torch.zeros(1, dtype=torch.long, device=device)

        @property
        def max_cache_len(self) -> int:
            return self._max_cache_len

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            k_buf = self.keys[layer_idx]
            if key_states.dtype != k_buf.dtype:
                key_states = key_states.to(dtype=k_buf.dtype)
                value_states = value_states.to(dtype=k_buf.dtype)
            if self._prefill_mode:
                seq_len = key_states.shape[2]
                start = self._seen
                k_buf[:, :, start : start + seq_len].copy_(key_states)
                self.values[layer_idx][:, :, start : start + seq_len].copy_(value_states)
                return k_buf[:, :, : start + seq_len], self.values[layer_idx][:, :, : start + seq_len]
            k_buf.index_copy_(2, self._write_pos, key_states)
            self.values[layer_idx].index_copy_(2, self._write_pos, value_states)
            return k_buf, self.values[layer_idx]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self._seen

        def get_max_cache_shape(self, layer_idx: int = 0) -> int:
            return self._max_cache_len

        def get_mask_sizes(self, q_len_or_pos, layer_idx: int = 0) -> tuple[int, int]:
            q_len = int(q_len_or_pos.shape[0]) if hasattr(q_len_or_pos, "shape") else int(q_len_or_pos)
            return (self._seen + q_len if self._prefill_mode else self._max_cache_len), 0

        def __len__(self) -> int:
            return self._n_layers

    _TRIUNE_STATIC_CACHE_CLS = _TriuneStaticCache
    return _TRIUNE_STATIC_CACHE_CLS


class _HFCUDAGraphDecodeRunner:
    """Captures 1-token HF autoregressive decode into a CUDA Graph over preallocated static KV/mask buffers."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        cfg = getattr(getattr(model, "config", None), "text_config", getattr(model, "config", None))
        n_layers = int(getattr(cfg, "num_hidden_layers", 0))
        n_heads = int(getattr(cfg, "num_attention_heads", 1))
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
        head_dim = int(getattr(cfg, "head_dim", int(getattr(cfg, "hidden_size", n_heads)) // max(1, n_heads)))

        cache_cls = _get_static_cache_cls()
        self.model = model
        self.device = device
        self.dtype = dtype
        self.max_cache_len = int(max_cache_len)
        self.cache = cache_cls(n_layers, n_kv_heads, head_dim, self.max_cache_len, dtype, device)
        self.sliding_window: int | None = getattr(cfg, "sliding_window", None)
        has_layer_types = bool(getattr(cfg, "layer_types", None))
        self.neg_inf = float(torch.finfo(dtype).min)

        self.full_mask = torch.full((1, 1, 1, self.max_cache_len), self.neg_inf, dtype=dtype, device=device)
        self.slide_mask = (
            torch.full((1, 1, 1, self.max_cache_len), self.neg_inf, dtype=dtype, device=device)
            if self.sliding_window is not None
            else None
        )
        if has_layer_types:
            attn_arg: Any = {"full_attention": self.full_mask}
            if self.slide_mask is not None:
                attn_arg["sliding_attention"] = self.slide_mask
            self.attn_arg = attn_arg
        else:
            self.attn_arg = self.slide_mask if self.sliding_window is not None else self.full_mask

        self.static_tok = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.static_pos = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_out: torch.Tensor | None = None

    def _clear_exit_acts(self) -> None:
        exit_mgr = getattr(self.model, "_exit_manager", None)
        acts = getattr(exit_mgr, "intermediate_activations", None) if exit_mgr is not None else None
        if isinstance(acts, dict):
            acts.clear()

    def capture(self) -> None:
        self.full_mask.fill_(self.neg_inf)
        self.full_mask[0, 0, 0, 0] = 0.0
        if self.slide_mask is not None:
            self.slide_mask.fill_(self.neg_inf)
            self.slide_mask[0, 0, 0, 0] = 0.0

        self.cache._prefill_mode = False
        self.cache._seen = 1
        # Dedicated scratch slot at max_cache_len - 1 prevents capture/warmup from mutating active KV slots.
        self.cache._write_pos.fill_(self.max_cache_len - 1)
        self.static_pos.fill_(0)
        self.static_tok.fill_(0)

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.amp.autocast("cuda", enabled=False):
            for _ in range(2):
                _ = self.model(
                    input_ids=self.static_tok,
                    attention_mask=self.attn_arg,
                    position_ids=self.static_pos,
                    past_key_values=self.cache,
                    use_cache=True,
                )
                self._clear_exit_acts()
        torch.cuda.current_stream(self.device).wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.amp.autocast("cuda", enabled=False):
            self._static_out = self.model(
                input_ids=self.static_tok,
                attention_mask=self.attn_arg,
                position_ids=self.static_pos,
                past_key_values=self.cache,
                use_cache=True,
            ).logits
        self._clear_exit_acts()
        self._graph = graph

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.full_mask.fill_(self.neg_inf)
        if self.slide_mask is not None:
            self.slide_mask.fill_(self.neg_inf)

        self.cache._prefill_mode = True
        self.cache._seen = 0
        self._clear_exit_acts()
        with torch.amp.autocast("cuda", enabled=False):
            out = self.model(input_ids=input_ids, past_key_values=self.cache, use_cache=True)
        self._clear_exit_acts()

        prompt_len = int(input_ids.shape[1])
        self.cache._seen = prompt_len
        self.cache._prefill_mode = False
        self.full_mask[0, 0, 0, :prompt_len] = 0.0
        if self.slide_mask is not None and self.sliding_window is not None:
            win_start = max(0, prompt_len - self.sliding_window)
            self.slide_mask[0, 0, 0, win_start:prompt_len] = 0.0
        return out.logits

    def decode_step(self, next_input: torch.Tensor) -> torch.Tensor:
        pos = self.cache._seen
        self.static_tok.copy_(next_input)
        self.cache._write_pos.fill_(pos)
        self.static_pos.fill_(pos)
        self.full_mask[0, 0, 0, pos] = 0.0
        if self.slide_mask is not None and self.sliding_window is not None:
            self.slide_mask[0, 0, 0, pos] = 0.0
            if pos >= self.sliding_window:
                self.slide_mask[0, 0, 0, pos - self.sliding_window] = self.neg_inf
        self._graph.replay()
        self.cache._seen = pos + 1
        return self._static_out

    def release(self) -> None:
        if self._graph is not None:
            try:
                self._graph.reset()
            except Exception:
                pass
        self._graph = None
        self._static_out = None
        self.cache = None
        self.full_mask = None
        self.slide_mask = None
        self.attn_arg = None
        self.model = None


def release_cuda_graph_runner(model: torch.nn.Module | None) -> None:
    """Release any captured static KV cache and CUDA graph attached to ``model``."""
    if model is None:
        return
    runner = getattr(model, "_triune_cg_runner", None)
    if runner is not None:
        try:
            runner.release()
        except Exception:
            pass
        try:
            delattr(model, "_triune_cg_runner")
        except Exception:
            model._triune_cg_runner = None


def _get_or_create_hf_cg_runner(
    model: torch.nn.Module,
    *,
    prompt_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> _HFCUDAGraphDecodeRunner | None:
    if os.environ.get("TRIUNE_DISABLE_CUDAGRAPH"):
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if type(model).__name__ not in _SUPPORTED_HF_CG_CLASSES or model.training:
        return None
    if getattr(model, "_layer_streaming_active", False) or getattr(model, "_streaming_engine", None) is not None:
        return None
    if any(getattr(m, "lora_A", None) is not None for m in model.modules()):
        return None

    cfg = getattr(getattr(model, "config", None), "text_config", getattr(model, "config", None))
    if cfg is None:
        return None
    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    if n_layers <= 0:
        return None
    max_pos = int(getattr(cfg, "max_position_embeddings", 8192) or 8192)
    required_len = prompt_len + max_new_tokens + 2
    if required_len > max_pos:
        return None

    ref_param = next((p for p in model.parameters() if p.is_floating_point()), None)
    if ref_param is not None and ref_param.device.type == "cuda":
        device = ref_param.device
    elif device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    existing = getattr(model, "_triune_cg_runner", None)
    if (
        existing is not None
        and existing.device == device
        and existing.max_cache_len >= required_len
        and existing._graph is not None
    ):
        return existing

    release_cuda_graph_runner(model)

    dtype = ref_param.dtype if ref_param is not None else torch.bfloat16
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    n_heads = int(getattr(cfg, "num_attention_heads", 1))
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", int(getattr(cfg, "hidden_size", n_heads)) // max(1, n_heads)))
    per_tok_bytes = max(1, n_layers * 2 * n_kv_heads * head_dim * dtype_bytes)

    free_vram, _ = torch.cuda.mem_get_info(device)
    cap: int | None = None
    for desired, frac in (
        (min(max_pos, max(required_len, 2048)), 0.25),
        (min(max_pos, max(required_len, 1024)), 0.40),
        (required_len, 0.50),
    ):
        if desired >= required_len and desired * per_tok_bytes <= int(free_vram * frac):
            cap = desired
            break
    if cap is None:
        return None

    runner = _HFCUDAGraphDecodeRunner(model, max_cache_len=cap, device=device, dtype=dtype)
    try:
        runner.capture()
    except Exception:
        runner.release()
        return None
    model._triune_cg_runner = runner
    return runner


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

        cfg = LoRAConfig.from_pretrained(adapter_path)
        tuner = TriuneFineTuner(model, cfg)
        tuner.load_adapter(adapter_path)
    return model, tokenizer, checkpoint, device, pad_token_id, eos_token_id


class _HFInferenceTokenizerAdapter:
    """Wrap Hugging Face AutoTokenizer with Triune's tokenizer.encode().ids & vocab contract."""

    def __init__(self, hf_tokenizer, model_vocab_size: int | None = None):
        self.hf = hf_tokenizer
        self._model_vocab = int(model_vocab_size) if model_vocab_size else None
        self.eos_token_id = hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else 2
        self.pad_token_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else self.eos_token_id
        self.chat_template = getattr(hf_tokenizer, "chat_template", None)

    def get_vocab_size(self) -> int:
        tok_len = len(self.hf)
        return max(tok_len, self._model_vocab or 0)

    def token_to_id(self, token: str):
        if token == "[PAD]":
            return self.pad_token_id
        if token in ("[SEP]", "[EOS]"):
            return self.eos_token_id
        if token == "[BOS]":
            return getattr(self.hf, "bos_token_id", None)
        return self.hf.convert_tokens_to_ids(token)

    def encode(self, text: str):
        ids = self.hf.encode(text, add_special_tokens=False)

        class _EncResult:
            def __init__(self, tok_ids: list[int]):
                self.ids = tok_ids

        return _EncResult([int(i) for i in ids])

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.hf.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @staticmethod
    def _normalize_chat_messages(messages: list[dict], *, system_as_content_blocks: bool = False) -> list[dict]:
        sys_parts: list[str] = []
        turns: list[dict] = []
        for m in messages or []:
            role = str(m.get("role") or "user").strip().lower()
            content = str(m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                sys_parts.append(content)
                continue
            if role not in ("user", "assistant"):
                role = "user"
            # Collapse consecutive same-role messages for strict alternating templates (Gemma / Mistral)
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] = f"{turns[-1]['content']}\n\n{content}"
            else:
                turns.append({"role": role, "content": content})

        if sys_parts:
            sys_blob = "\n\n".join(sys_parts)
            if system_as_content_blocks:
                return [{"role": "system", "content": [{"type": "text", "text": sys_blob}]}, *turns]
            if turns and turns[0]["role"] == "user":
                turns[0] = {"role": "user", "content": f"{sys_blob}\n\n{turns[0]['content']}"}
            else:
                turns.insert(0, {"role": "user", "content": sys_blob})
        return turns

    def apply_chat_template(self, messages, *, tokenize: bool = False, add_generation_prompt: bool = True):
        try:
            return self.hf.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            for as_blocks in (True, False):
                norm = self._normalize_chat_messages(messages, system_as_content_blocks=as_blocks)
                if not norm:
                    continue
                try:
                    return self.hf.apply_chat_template(
                        norm,
                        tokenize=tokenize,
                        add_generation_prompt=add_generation_prompt,
                    )
                except Exception:
                    continue
            raise


def load_model_for_inference(
    source: str | Path,
    tokenizer_path: str | Path = "triune_tokenizer.json",
    *,
    device: str | torch.device | None = None,
    adapter_path: str | Path | None = None,
    quantize: str | None = None,
    disk_stream: bool = False,
    disk_cache_dir: str | Path | None = None,
    quantize_lm_head: bool = False,
    hadamard: bool = True,
    local_files_only: bool = False,
):
    """Unified loader for Triune .pt checkpoints, NVFP4 disk caches, GGUF archives, and HF repos/dirs."""
    raw_str = str(source).strip()
    src_path = Path(raw_str)
    is_direct_gguf = False
    if src_path.is_file() and src_path.suffix.lower() not in (".pt", ".pth", ".ckpt", ".safetensors"):
        from triune.runtime.gguf_loader import is_gguf_file

        is_direct_gguf = src_path.suffix.lower() == ".gguf" or is_gguf_file(src_path)

    if raw_str.lower().startswith("ollama:") or is_direct_gguf:
        from triune.runtime.gguf_loader import load_gguf_hf_model, resolve_ollama_gguf_blob

        gguf_file = src_path if is_direct_gguf else resolve_ollama_gguf_blob(raw_str)
        if gguf_file is None or not gguf_file.is_file():
            raise FileNotFoundError(f"Could not locate local GGUF / Ollama blob for '{raw_str}'.")
        model, tokenizer, metadata, target_dev, pad_id, eos_id = load_gguf_hf_model(
            gguf_file,
            device=device,
            quantize=quantize,
            hadamard=hadamard,
        )
        if adapter_path is not None and Path(adapter_path).exists():
            from triune.trainer.finetune import LoRAConfig, TriuneFineTuner

            tuner = TriuneFineTuner(
                model,
                LoRAConfig.from_pretrained(adapter_path),
                tokenizer=getattr(tokenizer, "hf", tokenizer),
                adapter_device=target_dev,
            )
            tuner.load_adapter(adapter_path)
        return model, tokenizer, metadata, target_dev, pad_id, eos_id

    if src_path.is_file() and src_path.suffix.lower() in (".pt", ".pth", ".ckpt", ".bin"):
        model, tokenizer, ckpt, dev, pad_id, eos_id = load_checkpoint_model(
            src_path, tokenizer_path, device=device, adapter_path=adapter_path
        )
        if quantize and "nvfp4" in quantize.lower():
            from triune.model.fp4 import convert_to_nvfp4

            convert_to_nvfp4(model, hadamard=hadamard, protect_sensitive=True)
        return model, tokenizer, ckpt, dev, pad_id, eos_id

    target_dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_disk = bool(disk_stream or (src_path / "manifest.json").is_file() or (quantize and "disk" in quantize.lower()))

    if use_disk:
        from triune.runtime.disk_streaming import load_disk_streamed_model

        model, engine, resolved_cache = load_disk_streamed_model(
            source,
            cache_dir=disk_cache_dir,
            device=target_dev,
            hadamard=hadamard,
            quantize_lm_head=quantize_lm_head,
            local_files_only=local_files_only,
        )
        tok_source = str(resolved_cache)
        metadata: dict = {
            "source": str(source),
            "disk_cache": str(resolved_cache),
            "streaming_engine": engine,
            "best_eval_loss": "n/a (disk-streamed NVFP4)",
        }
    else:
        from transformers import AutoModelForCausalLM

        dtype = torch.bfloat16 if target_dev.type == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(str(source), dtype=dtype, local_files_only=local_files_only)
        except Exception as exc:
            # Fallback to Triune presets/configs if not an HF CausalLM repo
            try:
                from triune.model.zoo import load_model

                model = load_model(source)
                if not getattr(model, "_layer_streaming_active", False):
                    model = model.to(device=target_dev, dtype=dtype)
                tok = load_tokenizer(tokenizer_path)
                if quantize and "nvfp4" in quantize.lower():
                    from triune.model.fp4 import convert_to_nvfp4

                    convert_to_nvfp4(model, hadamard=hadamard, protect_sensitive=True)
                if adapter_path is not None and Path(adapter_path).exists():
                    from triune.trainer.finetune import LoRAConfig, TriuneFineTuner

                    tuner = TriuneFineTuner(model, LoRAConfig.from_pretrained(adapter_path))
                    tuner.load_adapter(adapter_path)
                model.eval()
                return (
                    model,
                    tok,
                    {"source": str(source), "best_eval_loss": "n/a"},
                    target_dev,
                    tok.token_to_id("[PAD]"),
                    tok.token_to_id("[SEP]"),
                )
            except Exception:
                raise exc
        if not getattr(model, "_layer_streaming_active", False):
            model = model.to(device=target_dev)
        if quantize and "nvfp4" in quantize.lower():
            from triune.model.fp4 import convert_to_nvfp4

            convert_to_nvfp4(model, hadamard=hadamard, protect_sensitive=True)
        tok_source = str(source)
        metadata = {"source": str(source), "best_eval_loss": "n/a (HF pretrained)"}

    from transformers import AutoTokenizer

    raw_tok = AutoTokenizer.from_pretrained(tok_source, local_files_only=local_files_only)
    cfg_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    tokenizer = _HFInferenceTokenizerAdapter(raw_tok, model_vocab_size=cfg_vocab)
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id)

    extra_eos: set[int] = {eos_id}
    gen_cfg = getattr(model, "generation_config", None)
    cfg_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if isinstance(cfg_eos, (list, tuple, set)):
        extra_eos.update(int(x) for x in cfg_eos)
    elif isinstance(cfg_eos, int):
        extra_eos.add(int(cfg_eos))
    model._triune_extra_eos_ids = extra_eos

    if adapter_path is not None and Path(adapter_path).exists():
        from triune.trainer.finetune import LoRAConfig, TriuneFineTuner

        tuner = TriuneFineTuner(
            model,
            LoRAConfig.from_pretrained(adapter_path),
            tokenizer=raw_tok,
            adapter_device=target_dev,
        )
        tuner.load_adapter(adapter_path)
    model.eval()
    optimize_hf_model_for_inference(model)
    return model, tokenizer, metadata, target_dev, pad_id, eos_id


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
        delta = text[self.print_len :]
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
                delta = text[self.print_len :]
                self.cumulative_text += delta
                self.token_cache = []
                self.print_len = 0
                return self.cumulative_text, delta
        return self.cumulative_text, ""


def _extract_exit_hidden(acts: dict, key_name: str, layer_idx: int | None):
    """Fetch captured intermediate hidden states by symbolic tier name or layer index."""
    if not acts:
        return None
    if key_name in acts and acts[key_name] is not None:
        return acts[key_name]
    if layer_idx is not None and layer_idx in acts and acts[layer_idx] is not None:
        return acts[layer_idx]
    return None


def _compute_tier_logits(head: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Run an exit projection head cleanly with dtype/device alignment."""
    ref_param = next(head.parameters(), None)
    if ref_param is not None:
        hidden = hidden.to(device=ref_param.device, dtype=ref_param.dtype)
    return head(hidden)


def _confidence_margin(logits: torch.Tensor) -> float:
    """Compute last-token top-1 vs top-2 probability margin for early-exit gating."""
    last = logits[:, -1, :].float()
    probs = torch.softmax(last, dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
    if top2.size(-1) < 2:
        return float(top2[0, 0].item())
    return float((top2[0, 0] - top2[0, 1]).item())


def _forward_generation_step(model, ids_step: torch.Tensor, cache, force_depth: int | None):
    """Execute one generation step transparently across TriuneTransformer and HF CausalLM models."""
    if hasattr(model, "init_cache") or getattr(model, "_is_triune", False):
        return model(ids_step, cache=cache, force_depth=force_depth)

    exit_mgr = getattr(model, "_exit_manager", None)
    acts = getattr(exit_mgr, "intermediate_activations", None) if exit_mgr is not None else None
    if isinstance(acts, dict):
        acts.clear()

    out = model(input_ids=ids_step, past_key_values=cache, use_cache=True)
    cortex_logits = out.logits if hasattr(out, "logits") else out[0]
    next_cache = getattr(out, "past_key_values", None)

    if exit_mgr is not None and getattr(model, "_is_triune_transmuted", False) and isinstance(acts, dict):
        try:
            r_idx = getattr(model, "reflex_layer_idx", getattr(model, "reflex_exit_layer", None))
            l_idx = getattr(model, "limbic_layer_idx", getattr(model, "limbic_exit_layer", None))
            h_reflex = _extract_exit_hidden(acts, "reflex", r_idx)
            h_limbic = _extract_exit_hidden(acts, "limbic", l_idx)
            reflex_head = getattr(model, "reflex_head", None)
            limbic_head = getattr(model, "limbic_head", None)

            selected_logits = cortex_logits
            chosen_depth = 2

            if force_depth == 0 and h_reflex is not None and reflex_head is not None:
                selected_logits = _compute_tier_logits(reflex_head, h_reflex)
                chosen_depth = 0
            elif force_depth == 1 and h_limbic is not None and limbic_head is not None:
                selected_logits = _compute_tier_logits(limbic_head, h_limbic)
                chosen_depth = 1
            elif force_depth is None and getattr(model, "_exits_calibrated", False):
                settled = False
                if h_reflex is not None and reflex_head is not None:
                    r_logits = _compute_tier_logits(reflex_head, h_reflex)
                    if _confidence_margin(r_logits) >= 0.85:
                        selected_logits = r_logits
                        chosen_depth = 0
                        settled = True
                if not settled and h_limbic is not None and limbic_head is not None:
                    l_logits = _compute_tier_logits(limbic_head, h_limbic)
                    if _confidence_margin(l_logits) >= 0.78:
                        selected_logits = l_logits
                        chosen_depth = 1

            model.last_depth_choice = torch.tensor([chosen_depth], device=cortex_logits.device, dtype=torch.long)
            selected_logits = selected_logits.to(dtype=cortex_logits.dtype)
            return selected_logits, next_cache
        finally:
            acts.clear()

    return cortex_logits, next_cache


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
    """Generate a response using Triune depth-routing or Hugging Face CausalLM execution."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    import time

    if prompt_is_formatted:
        formatted = prompt
    elif hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = f"User: {prompt}\nAssistant:"
    else:
        formatted = f"User: {prompt}\nAssistant:"
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
    is_triune_model = (
        hasattr(model, "init_cache")
        or getattr(model, "_is_triune", False)
        or getattr(model, "_is_triune_transmuted", False)
    )
    if cache is None and hasattr(model, "init_cache"):
        cache = model.init_cache(batch_size=1)
    stop_eos_ids: set[int] = set(getattr(model, "_triune_extra_eos_ids", (eos_token_id,)))
    stop_eos_ids.add(int(eos_token_id))
    stream_decoder = IncrementalStreamingDecoder(tokenizer) if on_text is not None else None
    started = time.perf_counter()
    first_token_latency_ms: float | None = None
    decode_started: float | None = None
    with torch.inference_mode():
        prompt_len = ids.size(1)
        gen_limit = min(max_new_tokens, 8192)
        cg_runner: _HFCUDAGraphDecodeRunner | None = None
        can_use_cg = (
            cache is None
            and not hasattr(model, "init_cache")
            and not getattr(model, "_is_triune", False)
            and force_depth not in (0, 1)
            and not (force_depth is None and getattr(model, "_exits_calibrated", False))
        )
        if can_use_cg:
            cg_runner = _get_or_create_hf_cg_runner(
                model,
                prompt_len=prompt_len,
                max_new_tokens=gen_limit,
                device=torch.device(device),
            )

        prefill_chunk_size = 1024
        if cg_runner is not None:
            logits = cg_runner.prefill(ids)
            if getattr(model, "_is_triune_transmuted", False):
                model.last_depth_choice = torch.tensor([2], device=device, dtype=torch.long)
        elif is_triune_model and prompt_len > prefill_chunk_size and cache is not None:
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
                _, cache = _forward_generation_step(model, chunk_ids, cache, force_depth)
            last_start = ((prompt_len - 1) // prefill_chunk_size) * prefill_chunk_size
            rem_ids = ids[:, last_start:]
            logits, cache = _forward_generation_step(model, rem_ids, cache, force_depth)
        else:
            logits, cache = _forward_generation_step(model, ids, cache, force_depth)

        decode_depth = force_depth
        if is_triune_model and decode_depth is None and getattr(model, "last_depth_choice", None) is not None:
            decode_depth = int(model.last_depth_choice[0].item())

        cancelled = False
        for step_idx in range(gen_limit):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            next_logits = logits[0, -1].float()
            if next_logits.numel() > seen.numel():
                # Guard against padded vocab difference
                next_logits = next_logits[: seen.numel()]
            if temperature > 0:
                next_logits = next_logits / temperature
            if repetition_penalty != 1.0:
                penalized = torch.where(
                    next_logits > 0,
                    next_logits / repetition_penalty,
                    next_logits * repetition_penalty,
                )
                next_logits = torch.where(seen, penalized, next_logits)
                next_logits = next_logits - 0.35 * (token_counts - 1.0).clamp_min(0.0)
            if 0 <= pad_token_id < next_logits.numel() and pad_token_id not in stop_eos_ids:
                next_logits[pad_token_id] = float("-inf")
            if temperature == 0:
                next_input = torch.argmax(next_logits).view(1, 1)
                next_id = int(next_input.item())
            else:
                effective_top_k = min(top_k, next_logits.numel())
                top_vals, top_indices = torch.topk(next_logits, effective_top_k)
                sampled_pos = torch.multinomial(torch.softmax(top_vals, dim=-1), 1)
                next_input = top_indices.gather(0, sampled_pos).view(1, 1)
                next_id = int(next_input.item())
            if next_id in stop_eos_ids:
                break
            generated.append(next_id)
            if first_token_latency_ms is None:
                decode_started = time.perf_counter()
                first_token_latency_ms = (decode_started - started) * 1000.0
            if stream_decoder is not None:
                cur_tier = 0 if decode_depth is None else int(decode_depth)
                cur_route = (
                    ("AUTO" if decode_depth is None else ("REFLEX", "LIMBIC", "CORTEX")[decode_depth])
                    if is_triune_model
                    else "HF"
                )

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
            if cg_runner is not None:
                if step_idx + 1 < gen_limit:
                    logits = cg_runner.decode_step(next_input)
            elif cache is not None:
                logits, cache = _forward_generation_step(model, next_input, cache, decode_depth)
            else:
                ids = torch.cat((ids, next_input), dim=1)
                logits, cache = _forward_generation_step(model, ids, None, decode_depth)

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
    route_name = (
        ("AUTO" if decode_depth is None else ("REFLEX", "LIMBIC", "CORTEX")[decode_depth]) if is_triune_model else "HF"
    )
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
