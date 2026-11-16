"""Model weight transmutation and knowledge distillation utilities for Triune.

Supports two complementary conversion modes when upgrading an existing model
(HuggingFace CausalLM, local SafeTensors directory, or Ollama candidate) into a
3-tier Triune adaptive architecture:

1. **Cortex-Preserving Early-Exit Transmutation** (``transmute_hf_to_triune``):
   Loads the source model's weights cleanly (including BF16, NVFP4, or
   disk-streamed 4-bit execution) and attaches Triune's weight-tied Reflex and
   Limbic early-exit heads plus optional JEV routing routers. The original
   full-depth backbone weights remain intact for Cortex execution while attached
   early-exit heads are marked uncalibrated until tuned/validated.

2. **Structural Weight Surgery + Distillation** (``transfer_weights_to_native_triune``
   and ``distill_native_triune_step``):
   Maps embeddings, norms, Q/K/V/O projections (with GQA KV-head expansion),
   and MLP/SwiGLU blocks (with MoE expert upcycling) directly into a native
   ``TriuneTransformer`` (Hybrid GLA + Sparse MoE), then aligns all three exit
   tiers to the HF teacher via KL logit distillation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import WeightTiedExitProjection, attach_early_exits
from .transformer import TriuneTransformer

logger = logging.getLogger(__name__)

# Curated Ollama tag -> HuggingFace repo ID candidates (used for guidance/lookup).
_OLLAMA_HF_FAMILY_MAP: dict[str, str] = {
    "gemma3:1b": "google/gemma-3-1b-it",
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma3:12b": "google/gemma-3-12b-it",
    "gemma3:27b": "google/gemma-3-27b-it",
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
    "gemma2:27b": "google/gemma-2-27b-it",
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5:72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-coder:1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-coder:3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder:14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "qwen2.5-coder:32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "qwen3:0.6b": "Qwen/Qwen3-0.6B",
    "qwen3:1.7b": "Qwen/Qwen3-1.7B",
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen3:8b": "Qwen/Qwen3-8B",
    "qwen3:14b": "Qwen/Qwen3-14B",
    "qwen3:32b": "Qwen/Qwen3-32B",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1:70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3:8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3:70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi4:14b": "microsoft/phi-4",
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi3:3.8b": "microsoft/Phi-3-mini-4k-instruct",
    "smollm2:135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2:360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2:1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1:14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}


def ollama_to_hf_candidate(name: str) -> str | None:
    """Resolve an Ollama model reference (e.g. ``ollama:gemma3:4b``) to an HF repo ID."""
    if not name:
        return None
    cleaned = str(name).strip()
    if cleaned.lower().startswith("ollama:"):
        cleaned = cleaned[len("ollama:") :].strip()

    # Pass through if already formatted like an org/repo HF identifier or local path.
    if "/" in cleaned and ":" not in cleaned:
        return cleaned

    norm = cleaned.lower()
    if norm in _OLLAMA_HF_FAMILY_MAP:
        return _OLLAMA_HF_FAMILY_MAP[norm]

    # Strip trailing quantization suffix like :q4_k_m if present (`gemma3:4b-instruct-q4_K_M` -> `gemma3:4b`).
    base_name, _, tag = norm.partition(":")
    tag_size = re.match(r"^([0-9]+(?:\.[0-9]+)?[bm])", tag or "")
    if base_name and tag_size:
        short_key = f"{base_name}:{tag_size.group(1)}"
        if short_key in _OLLAMA_HF_FAMILY_MAP:
            return _OLLAMA_HF_FAMILY_MAP[short_key]

    # Default size tags when ':latest' or no tag was specified.
    if not tag or tag == "latest":
        for default_tag in ("4b", "7b", "8b", "3b", "1.7b", "mini"):
            candidate = _OLLAMA_HF_FAMILY_MAP.get(f"{base_name}:{default_tag}")
            if candidate:
                return candidate
    return None


def _resolve_hf_backbone(hf_model: nn.Module):
    """Extract (embed_tokens, lm_head, layers, final_norm, text_config, attach_target) cleanly.

    Supports standard CausalLM models (Llama, Qwen2/3, Mistral, Gemma2/3-text, GPT2,
    GPTNeoX) as well as multimodal wrappers such as ``Gemma3ForConditionalGeneration``.
    """
    cfg = getattr(hf_model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)

    lm_head = getattr(hf_model, "lm_head", None) or getattr(hf_model, "embed_out", None)
    inner = (
        getattr(hf_model, "model", None)
        or getattr(hf_model, "transformer", None)
        or getattr(hf_model, "gpt_neox", None)
    )

    attach_target = hf_model
    # Multimodal wrappers (e.g. Gemma3ForConditionalGeneration) nest text backbone inside language_model.
    if getattr(hf_model, "language_model", None) is not None and not hasattr(inner, "layers"):
        lang_mod = hf_model.language_model
        attach_target = lang_mod
        if hasattr(lang_mod, "model"):
            inner = lang_mod.model
        else:
            inner = lang_mod
        if lm_head is None:
            lm_head = getattr(lang_mod, "lm_head", None)

    if inner is None:
        inner = hf_model

    embed_tokens = (
        getattr(inner, "embed_tokens", None) or getattr(inner, "wte", None) or getattr(inner, "embed_in", None)
    )
    if embed_tokens is None and hasattr(hf_model, "get_input_embeddings"):
        embed_tokens = hf_model.get_input_embeddings()

    if lm_head is None and hasattr(hf_model, "get_output_embeddings"):
        lm_head = hf_model.get_output_embeddings()

    layers = getattr(inner, "layers", None) or getattr(inner, "h", None) or getattr(inner, "blocks", None)
    final_norm = (
        getattr(inner, "norm", None) or getattr(inner, "ln_f", None) or getattr(inner, "final_layer_norm", None)
    )

    return embed_tokens, lm_head, layers, final_norm, text_cfg, attach_target


def _compute_tier_indices(num_layers: int, reflex_ratio: float = 0.25) -> tuple[int, int]:
    """Compute valid (reflex_layer_idx, limbic_layer_idx) for a given total layer count."""
    n = max(3, int(num_layers))
    reflex_ratio_clamped = max(0.15, min(0.45, float(reflex_ratio)))
    reflex = max(1, min(n - 2, int(round(n * reflex_ratio_clamped))))
    limbic = max(reflex + 1, min(n - 1, int(round(n * 0.60))))
    if limbic <= reflex:
        limbic = min(n - 1, reflex + 1)
    if reflex >= limbic and reflex > 0:
        reflex = limbic - 1
    return reflex, limbic


def _align_exit_attachments_to_head(model: nn.Module, attach_target: nn.Module, lm_head: nn.Module | None) -> None:
    """Ensure attached exit heads and JEV routers match the LM head device and non-quantized dtype."""
    if lm_head is None:
        return
    ref_param = next(lm_head.parameters(), None)
    if ref_param is None or ref_param.device.type == "meta":
        return
    target_device = ref_param.device
    target_dtype = ref_param.dtype if ref_param.dtype.is_floating_point else torch.float32

    attr_names = ("reflex_head", "limbic_head", "reflex_jev_router", "limbic_jev_router")
    for container in (attach_target, model):
        for attr in attr_names:
            mod = getattr(container, attr, None)
            if isinstance(mod, nn.Module):
                mod.to(device=target_device, dtype=target_dtype)


def transmute_hf_to_triune(
    source: str | Path,
    *,
    reflex_ratio: float = 0.25,
    quantize: str | None = None,
    disk_stream: bool = False,
    disk_cache_dir: str | Path | None = None,
    rank: int = 64,
    use_jev: bool = True,
    jev_threshold: float = 0.85,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    tokenizer_path: str = "triune_tokenizer.json",
) -> tuple[nn.Module, Any, dict[str, Any], torch.device, int, int]:
    """Load real HF CausalLM weights and attach Triune's 3-tier adaptive exit system.

    Returns ``(model, tokenizer, metadata, target_dev, pad_id, eos_id)`` compatible with
    :func:`triune.inference.generate_response` and Studio's ``PyTorchEngineState``.
    """
    from triune.inference import load_model_for_inference

    model, tokenizer, metadata, target_dev, pad_id, eos_id = load_model_for_inference(
        source,
        tokenizer_path=tokenizer_path,
        device=device,
        quantize=quantize,
        disk_stream=disk_stream,
        disk_cache_dir=disk_cache_dir,
        local_files_only=local_files_only,
    )

    _, lm_head, layers, _, text_cfg, attach_target = _resolve_hf_backbone(model)
    num_layers = (
        len(layers)
        if layers is not None
        else int(getattr(text_cfg, "num_hidden_layers", getattr(text_cfg, "n_layer", 12)) or 12)
    )
    reflex_idx, limbic_idx = _compute_tier_indices(num_layers, reflex_ratio=reflex_ratio)

    # Prefer Triune's canonical attach_early_exits when attach_target exposes .model.layers
    attached = False
    if hasattr(attach_target, "model") and hasattr(attach_target.model, "layers") and hasattr(attach_target, "lm_head"):
        try:
            attach_early_exits(
                attach_target,
                reflex_layer=reflex_idx,
                limbic_layer=limbic_idx,
                rank=rank,
                use_jev=use_jev,
                jev_threshold=jev_threshold,
            )
            attached = True
        except Exception as exc:
            logger.warning("attach_early_exits fell back to manual attachment: %s", exc)

    if not attached and layers is not None and lm_head is not None and len(layers) >= 3:
        hidden_dim = int(getattr(text_cfg, "hidden_size", getattr(lm_head, "in_features", 768)))
        attach_target.reflex_head = WeightTiedExitProjection(lm_head, hidden_dim, rank=rank)
        attach_target.limbic_head = WeightTiedExitProjection(lm_head, hidden_dim, rank=rank)
        attach_target.reflex_layer_idx = reflex_idx
        attach_target.limbic_layer_idx = limbic_idx

        from .adapters import EarlyExitManager

        mgr = EarlyExitManager(attach_target, reflex_idx, limbic_idx)
        mgr.attach_hooks()
        attach_target._exit_manager = mgr

    # Mirror attachment properties onto the outer model if attach_target was a sub-wrapper (e.g. language_model)
    if attach_target is not model:
        for attr in (
            "reflex_head",
            "limbic_head",
            "reflex_layer_idx",
            "limbic_layer_idx",
            "_exit_manager",
            "reflex_jev_router",
            "limbic_jev_router",
            "use_jev",
        ):
            if hasattr(attach_target, attr):
                setattr(model, attr, getattr(attach_target, attr))

    _align_exit_attachments_to_head(model, attach_target, lm_head)

    model._is_triune_transmuted = True
    model._exits_calibrated = False
    model.reflex_exit_layer = int(getattr(model, "reflex_layer_idx", reflex_idx))
    model.limbic_exit_layer = int(getattr(model, "limbic_layer_idx", limbic_idx))
    model.last_depth_choice = None

    enriched_meta = dict(metadata or {})
    enriched_meta.update(
        {
            "weights_transferred": True,
            "transmute_mode": "early_exit",
            "reflex_exit_layer": model.reflex_exit_layer,
            "limbic_exit_layer": model.limbic_exit_layer,
            "num_layers": num_layers,
            "use_jev": bool(use_jev),
        }
    )
    return model, tokenizer, enriched_meta, target_dev, pad_id, eos_id


def _weights_are_tied(emb_w: torch.Tensor | None, head_w: torch.Tensor | None) -> bool:
    """Return True when input embeddings and LM head share identical storage or values."""
    if emb_w is None or head_w is None:
        return True
    if emb_w.data_ptr() == head_w.data_ptr() and emb_w.shape == head_w.shape:
        return True
    if emb_w.shape != head_w.shape:
        return False
    with torch.no_grad():
        return bool(torch.allclose(emb_w.detach().float(), head_w.detach().float(), atol=1e-6, rtol=1e-5))


def _copy_matrix_projected(dst: torch.Tensor, src: torch.Tensor) -> int:
    """Copy or crop ``src`` into ``dst`` without memory blowups and return exact copied elements."""
    with torch.no_grad():
        s = src.detach().to(device=dst.device, dtype=dst.dtype)
        if s.shape == dst.shape:
            dst.copy_(s)
            return int(dst.numel())

        if s.ndim == 1 and dst.ndim == 1:
            n = min(dst.shape[0], s.shape[0])
            dst[:n].copy_(s[:n])
            if dst.shape[0] > n:
                dst[n:].zero_()
            return int(n)

        out_d, in_d = dst.shape[:2]
        src_out, src_in = s.shape[:2]
        rows = min(out_d, src_out)
        cols = min(in_d, src_in)
        dst.zero_()
        dst[:rows, :cols].copy_(s[:rows, :cols])
        return int(rows * cols)


def _copy_linear_projected(dst_linear: nn.Module, src_linear: nn.Module | None) -> int:
    """Copy linear weight (and bias if compatible) into ``dst_linear``. Returns copied element count."""
    if src_linear is None or not hasattr(dst_linear, "weight") or not hasattr(src_linear, "weight"):
        return 0
    src_w = getattr(src_linear, "weight", None)
    dst_w = getattr(dst_linear, "weight", None)
    if src_w is None or dst_w is None or src_w.device.type == "meta" or dst_w.device.type == "meta":
        return 0
    copied = _copy_matrix_projected(dst_w, src_w)
    dst_b = getattr(dst_linear, "bias", None)
    src_b = getattr(src_linear, "bias", None)
    if dst_b is not None and src_b is not None and src_b.device.type != "meta":
        copied += _copy_matrix_projected(dst_b, src_b)
    return copied


def _expand_gqa_weight(weight: torch.Tensor, num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
    """Expand GQA [kv_heads * head_dim, hidden] projection weight into [q_heads * head_dim, hidden]."""
    if num_kv_heads <= 0 or num_q_heads <= 0 or num_kv_heads == num_q_heads:
        return weight
    if num_q_heads % num_kv_heads != 0:
        return weight
    repeats = num_q_heads // num_kv_heads
    head_dim = weight.shape[0] // num_kv_heads
    if head_dim * num_kv_heads != weight.shape[0]:
        return weight
    reshaped = weight.view(num_kv_heads, head_dim, weight.shape[1])
    expanded = reshaped.repeat_interleave(repeats, dim=0)
    return expanded.reshape(num_q_heads * head_dim, weight.shape[1])


def _extract_teacher_attn_and_mlp(layer: nn.Module):
    """Extract (norm1, norm2, q, k, v, o, gate_proj, up_proj, down_proj) across common HF layer layouts."""
    norm1 = getattr(layer, "input_layernorm", None) or getattr(layer, "ln_1", None)
    norm2 = getattr(layer, "post_attention_layernorm", None) or getattr(layer, "ln_2", None)
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    q_proj = getattr(attn, "q_proj", None) if attn is not None else None
    k_proj = getattr(attn, "k_proj", None) if attn is not None else None
    v_proj = getattr(attn, "v_proj", None) if attn is not None else None
    o_proj = getattr(attn, "o_proj", None) or (getattr(attn, "out_proj", None) if attn is not None else None)

    mlp = getattr(layer, "mlp", None)
    gate_proj = getattr(mlp, "gate_proj", None) if mlp is not None else None
    up_proj = getattr(mlp, "up_proj", None) if mlp is not None else None
    down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
    if up_proj is None and mlp is not None:
        up_proj = getattr(mlp, "c_fc", None) or getattr(mlp, "fc_in", None) or getattr(mlp, "dense_h_to_4h", None)
    if down_proj is None and mlp is not None:
        down_proj = getattr(mlp, "c_proj", None) or getattr(mlp, "fc_out", None) or getattr(mlp, "dense_4h_to_h", None)

    return norm1, norm2, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj


def _extract_source_weight(mod: nn.Module | None) -> torch.Tensor | None:
    if mod is None or not hasattr(mod, "weight"):
        return None
    w = getattr(mod, "weight", None)
    if w is None or getattr(w, "device", torch.device("cpu")).type == "meta":
        return None
    # Packed 4-bit weights cannot be copied raw into float linear layers without dequantization
    if not w.dtype.is_floating_point:
        return None
    return w


def transfer_weights_to_native_triune(
    hf_model: nn.Module,
    tokenizer: Any = None,
    *,
    reflex_ratio: float = 0.25,
    num_experts: int = 8,
    top_k: int = 2,
    shared_expert: bool = True,
    target_dim: int | None = None,
    target_layers: int | None = None,
) -> tuple[TriuneTransformer, dict[str, Any]]:
    """Perform structural weight surgery from an HF CausalLM into a native ``TriuneTransformer``.

    Copies embeddings, final norm, per-layer norms, Q/K/V/O attention projections
    (expanding GQA KV heads when needed), and MLP/SwiGLU upcycled dense + MoE
    FFN weights. Returns ``(student_model, transfer_report)``.
    """
    embed_tokens, lm_head, t_layers, t_final_norm, text_cfg, _ = _resolve_hf_backbone(hf_model)
    if t_layers is None or len(t_layers) == 0:
        raise ValueError("Source model exposes no transformer layers for structural weight transfer.")

    t_vocab = int(
        getattr(
            text_cfg,
            "vocab_size",
            embed_tokens.weight.shape[0] if embed_tokens is not None else 32000,
        )
    )
    t_dim = int(getattr(text_cfg, "hidden_size", embed_tokens.weight.shape[1] if embed_tokens is not None else 768))
    t_num_layers = len(t_layers)
    t_num_heads = int(getattr(text_cfg, "num_attention_heads", max(1, t_dim // 64)) or max(1, t_dim // 64))
    t_num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", t_num_heads) or t_num_heads)

    s_dim = int(target_dim or t_dim)
    s_head_dim = 64
    if s_dim % s_head_dim != 0:
        for cand_hd in (128, 64, 32, 16):
            if s_dim % cand_hd == 0:
                s_head_dim = cand_hd
                break
        else:
            s_head_dim = max(16, s_dim // max(1, t_num_heads))
    s_num_heads = max(1, s_dim // s_head_dim)
    s_dim = s_num_heads * s_head_dim

    s_num_layers = max(4, int(target_layers or t_num_layers))
    prefix_layers = 1
    reflex_idx, limbic_idx = _compute_tier_indices(s_num_layers, reflex_ratio=reflex_ratio)
    if reflex_idx <= prefix_layers:
        reflex_idx = prefix_layers + 1
    if limbic_idx <= reflex_idx:
        limbic_idx = min(s_num_layers - 1, reflex_idx + 1)

    student = TriuneTransformer(
        vocab_size=t_vocab,
        hidden_dim=s_dim,
        num_layers=s_num_layers,
        num_heads=s_num_heads,
        head_dim=s_head_dim,
        num_experts=max(1, int(num_experts)),
        router_prefix_layers=prefix_layers,
        reflex_exit_layer=reflex_idx,
        limbic_exit_layer=limbic_idx,
    )

    ref_param = next((p for p in hf_model.parameters() if p.device.type != "meta" and p.dtype.is_floating_point), None)
    if ref_param is not None:
        student = student.to(device=ref_param.device, dtype=ref_param.dtype)

    transferred_tensors = 0
    transferred_params = 0

    def _record_copy(copied_elems: int) -> None:
        nonlocal transferred_tensors, transferred_params
        if copied_elems > 0:
            transferred_tensors += 1
            transferred_params += int(copied_elems)

    # 1. Embeddings (token_embed.weight defaults to shared with final_head.weight in TriuneTransformer)
    emb_w = _extract_source_weight(embed_tokens)
    if emb_w is not None:
        _record_copy(_copy_matrix_projected(student.token_embed.weight, emb_w))

    # 1b. Explicit LM head transfer when teacher lm_head is untied, shaped differently, or biased
    lm_head_w = _extract_source_weight(lm_head)
    head_bias = getattr(lm_head, "bias", None) if lm_head is not None else None
    has_head_bias = (
        head_bias is not None
        and getattr(head_bias, "device", torch.device("cpu")).type != "meta"
        and head_bias.dtype.is_floating_point
    )
    lm_head_transferred = False
    lm_head_untied = False
    if lm_head_w is not None:
        tied_to_embed = emb_w is not None and _weights_are_tied(emb_w, lm_head_w) and not has_head_bias
        if tied_to_embed:
            lm_head_transferred = True
        else:
            dev = student.token_embed.weight.device
            dt = student.token_embed.weight.dtype
            untied_head = nn.Linear(s_dim, t_vocab, bias=has_head_bias, device=dev, dtype=dt)
            copied_head = _copy_matrix_projected(untied_head.weight, lm_head_w)
            _record_copy(copied_head)
            if has_head_bias:
                _record_copy(_copy_matrix_projected(untied_head.bias, head_bias))
            student.final_head = untied_head
            lm_head_transferred = True
            lm_head_untied = True

    # 2. Final RMSNorm
    norm_w = _extract_source_weight(t_final_norm)
    if norm_w is not None and hasattr(student.final_norm, "weight"):
        _record_copy(_copy_matrix_projected(student.final_norm.weight, norm_w))

    # 3. Transformer blocks
    for s_idx, s_layer in enumerate(student.layers):
        t_idx = min(t_num_layers - 1, int(round(s_idx * (t_num_layers - 1) / max(1, s_num_layers - 1))))
        norm1, norm2, q_p, k_p, v_p, o_p, gate_p, up_p, down_p = _extract_teacher_attn_and_mlp(t_layers[t_idx])

        n1_w = _extract_source_weight(norm1)
        if n1_w is not None and hasattr(s_layer.norm1, "weight"):
            _record_copy(_copy_matrix_projected(s_layer.norm1.weight, n1_w))

        n2_w = _extract_source_weight(norm2)
        if n2_w is not None and hasattr(s_layer.norm2, "weight"):
            _record_copy(_copy_matrix_projected(s_layer.norm2.weight, n2_w))

        # Attention projections (HybridAttention.gla)
        gla = getattr(getattr(s_layer, "attn", None), "gla", None)
        if gla is not None:
            q_w = _extract_source_weight(q_p)
            if q_w is not None:
                _record_copy(_copy_matrix_projected(gla.q_proj.weight, q_w))

            k_w = _extract_source_weight(k_p)
            if k_w is not None:
                k_exp = _expand_gqa_weight(k_w, t_num_heads, t_num_kv_heads)
                _record_copy(_copy_matrix_projected(gla.k_proj.weight, k_exp))
                # Seed GLA decay gate smoothly near identity/moderate decay from k_proj structure
                if hasattr(gla, "g_proj"):
                    _record_copy(_copy_matrix_projected(gla.g_proj.weight, k_exp * 0.1))

            v_w = _extract_source_weight(v_p)
            if v_w is not None:
                v_exp = _expand_gqa_weight(v_w, t_num_heads, t_num_kv_heads)
                _record_copy(_copy_matrix_projected(gla.v_proj.weight, v_exp))

            _record_copy(_copy_linear_projected(gla.out_proj, o_p))

        # FFN / MoE upcycling (blend gate_proj + up_proj when teacher is SwiGLU)
        gate_w = _extract_source_weight(gate_p)
        up_w = _extract_source_weight(up_p)
        down_w = _extract_source_weight(down_p)
        fc1_src = None
        if gate_w is not None and up_w is not None and gate_w.shape == up_w.shape:
            fc1_src = 0.5 * (gate_w + up_w)
        elif up_w is not None:
            fc1_src = up_w

        ffn = getattr(s_layer, "ffn", None)
        if isinstance(ffn, nn.Sequential) and len(ffn) >= 3:
            if fc1_src is not None and hasattr(ffn[0], "weight"):
                _record_copy(_copy_matrix_projected(ffn[0].weight, fc1_src))
            if down_w is not None and hasattr(ffn[2], "weight"):
                _record_copy(_copy_matrix_projected(ffn[2].weight, down_w))
        elif ffn is not None and hasattr(ffn, "experts"):
            if getattr(ffn, "shared", None) is not None and len(ffn.shared) >= 3:
                if fc1_src is not None:
                    _record_copy(_copy_matrix_projected(ffn.shared[0].weight, fc1_src))
                if down_w is not None:
                    _record_copy(_copy_matrix_projected(ffn.shared[2].weight, down_w))

            n_exp = len(ffn.experts)
            for e_idx, exp in enumerate(ffn.experts):
                if len(exp) >= 3:
                    # Slight deterministic specialization factor across experts avoids dead collapse
                    spec_scale = 1.0 + (e_idx - (n_exp - 1) / 2.0) * 1e-3
                    if fc1_src is not None:
                        _record_copy(_copy_matrix_projected(exp[0].weight, fc1_src * spec_scale))
                    if down_w is not None:
                        _record_copy(_copy_matrix_projected(exp[2].weight, down_w / spec_scale))

        # Initialize exit_norm from teacher's final norm for stable early-exit logits
        if getattr(s_layer, "exit_norm", None) is not None and norm_w is not None:
            _record_copy(_copy_matrix_projected(s_layer.exit_norm.weight, norm_w))

    if tokenizer is not None:
        student.tokenizer = tokenizer
    student._exits_calibrated = False

    total_student_params = sum(int(p.numel()) for p in student.parameters())
    report = {
        "weights_transferred": transferred_tensors > 0,
        "transmute_mode": "native_distill",
        "transferred_tensors": transferred_tensors,
        "transferred_params": transferred_params,
        "total_student_params": total_student_params,
        "coverage_ratio": round(min(1.0, transferred_params / max(1, total_student_params)), 4),
        "lm_head_transferred": lm_head_transferred,
        "lm_head_untied": lm_head_untied,
        "exits_calibrated": False,
        "teacher_layers": t_num_layers,
        "student_layers": s_num_layers,
        "teacher_dim": t_dim,
        "student_dim": s_dim,
        "reflex_exit_layer": reflex_idx,
        "limbic_exit_layer": limbic_idx,
        "needs_distillation": True,
    }
    return student, report


def distill_native_triune_step(
    student: TriuneTransformer,
    teacher: nn.Module,
    input_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    temperature: float = 2.0,
    ce_weight: float = 0.1,
    reflex_weight: float = 0.25,
    limbic_weight: float = 0.35,
) -> dict[str, float]:
    """Run one teacher-student KL distillation step aligning all three Triune exit tiers."""
    student.train()
    teacher.eval()
    temp = max(0.1, float(temperature))

    with torch.no_grad():
        teacher_out = teacher(input_ids=input_ids, use_cache=False)
        teacher_logits = teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out[0]
        target_probs = F.softmax(teacher_logits.float() / temp, dim=-1)

    optimizer.zero_grad(set_to_none=True)

    logits_cortex, _ = student(input_ids, force_depth=2)
    logits_limbic, _ = student(input_ids, force_depth=1)
    logits_reflex, _ = student(input_ids, force_depth=0)

    def _kl_loss(student_logits: torch.Tensor) -> torch.Tensor:
        # Align vocab slice if student & teacher differ in padding vocab length,
        # and flatten [B, T, V] -> [B*T, V] so batchmean averages per token cleanly,
        # keeping loss weighting invariant to sequence length and aligned with CE.
        v = min(student_logits.size(-1), target_probs.size(-1))
        log_p = F.log_softmax(student_logits[..., :v].float() / temp, dim=-1).reshape(-1, v)
        q_p = target_probs[..., :v].reshape(-1, v)
        if v < target_probs.size(-1):
            q_p = q_p / q_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return F.kl_div(log_p, q_p, reduction="batchmean") * (temp * temp)

    cortex_kl = _kl_loss(logits_cortex)
    limbic_kl = _kl_loss(logits_limbic)
    reflex_kl = _kl_loss(logits_reflex)

    total_loss = cortex_kl + limbic_weight * limbic_kl + reflex_weight * reflex_kl
    if ce_weight > 0.0 and input_ids.size(1) > 1:
        ce = F.cross_entropy(
            logits_cortex[:, :-1, :].reshape(-1, logits_cortex.size(-1)).float(),
            input_ids[:, 1:].reshape(-1),
        )
        total_loss = total_loss + float(ce_weight) * ce

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(total_loss.detach().item()),
        "cortex_kl": float(cortex_kl.detach().item()),
        "limbic_kl": float(limbic_kl.detach().item()),
        "reflex_kl": float(reflex_kl.detach().item()),
    }
