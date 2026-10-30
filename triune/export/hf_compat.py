"""Hugging Face Hub Compatibility & Export Engine for Triune.

Provides seamless export of Triune models to Hugging Face Transformers format,
enabling any external developer or platform to load Triune with:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("path/to/exported_dir", trust_remote_code=True)
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Optional
import torch
import torch.nn as nn

from .exporter import export_safetensors


_CONFIGURATION_TRIUNE_CODE = '''"""Triune Model Configuration for Hugging Face Transformers."""

from transformers.configuration_utils import PretrainedConfig


class TriuneConfig(PretrainedConfig):
    model_type = "triune"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        head_dim: int = 64,
        num_experts: int = 4,
        router_prefix_layers: int = 1,
        reflex_exit_layer: int = 2,
        limbic_exit_layer: int = 4,
        use_fp4: bool = False,
        use_fp8: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.router_prefix_layers = router_prefix_layers
        self.reflex_exit_layer = reflex_exit_layer
        self.limbic_exit_layer = limbic_exit_layer
        self.use_fp4 = use_fp4
        self.use_fp8 = use_fp8
'''

_MODELING_TRIUNE_CODE = '''"""Triune Causal Language Model implementation for Hugging Face Transformers."""

from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_triune import TriuneConfig
from triune.model.transformer import TriuneTransformer


class TriunePreTrainedModel(PreTrainedModel):
    config_class = TriuneConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TriuneTransformerBlock", "VectorisedGLA", "MoE_FFN"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()


class TriuneForCausalLM(TriunePreTrainedModel):
    def __init__(self, config: TriuneConfig):
        super().__init__(config)
        self.transformer = TriuneTransformer(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            num_experts=config.num_experts,
            router_prefix_layers=config.router_prefix_layers,
            reflex_exit_layer=config.reflex_exit_layer,
            limbic_exit_layer=config.limbic_exit_layer,
            use_fp4=config.use_fp4,
            use_fp8=config.use_fp8,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.token_embed

    def set_input_embeddings(self, value):
        self.transformer.token_embed = value

    def get_output_embeddings(self):
        return self.transformer.final_head

    def set_output_embeddings(self, new_embeddings):
        self.transformer.final_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        force_depth: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        res = self.transformer(
            input_ids=input_ids,
            force_depth=force_depth,
            cache=past_key_values,
        )

        if past_key_values is not None:
            logits, new_cache = res
        else:
            logits = res[0] if isinstance(res, (tuple, list)) else res
            new_cache = None

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,)
            if new_cache is not None:
                output = output + (new_cache,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_cache,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs
    ):
        force_depth = model_kwargs.get("force_depth", None)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "force_depth": force_depth,
        }
'''


def extract_triune_hf_config(model: nn.Module) -> dict[str, Any]:
    """Extracts standard Hugging Face config dictionary from a Triune model instance."""
    # Handle wrapped or unwrapped models
    core = getattr(model, "transformer", model)
    return {
        "architectures": ["TriuneForCausalLM"],
        "model_type": "triune",
        "auto_map": {
            "AutoConfig": "configuration_triune.TriuneConfig",
            "AutoModelForCausalLM": "modeling_triune.TriuneForCausalLM",
        },
        "vocab_size": getattr(core, "vocab_size", 2048),
        "hidden_dim": getattr(core, "hidden_dim", getattr(core, "dim", 256)),
        "num_layers": getattr(core, "num_layers", 6),
        "num_heads": getattr(core, "num_heads", 4),
        "head_dim": getattr(core, "head_dim", 64),
        "num_experts": getattr(core, "num_experts", 4),
        "router_prefix_layers": getattr(core, "router_prefix_layers", 1),
        "reflex_exit_layer": getattr(core, "reflex_exit_layer", 2),
        "limbic_exit_layer": getattr(core, "limbic_exit_layer", 4),
        "use_fp4": getattr(core, "use_fp4", False),
        "use_fp8": getattr(core, "use_fp8", False),
        "torch_dtype": str(next(core.parameters()).dtype).split(".")[-1],
        "transformers_version": "4.49.0",
    }


def export_to_huggingface(
    model: nn.Module,
    save_directory: str | Path,
    tokenizer: Optional[Any] = None,
    tokenizer_path: Optional[str | Path] = None,
) -> Path:
    """Exports model weights, configurations, and auto-loading python scripts to Hugging Face format.

    Creates:
    - config.json (model architecture and auto_map)
    - generation_config.json (decoding defaults)
    - configuration_triune.py (PretrainedConfig subclass)
    - modeling_triune.py (PreTrainedModel subclass)
    - model.safetensors (deduplicated weights)
    - tokenizer.json (if provided)
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate and save config.json
    hf_config = extract_triune_hf_config(model)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # 2. Save generation_config.json
    generation_config = {
        "_from_model_config": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "transformers_version": "4.49.0",
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
    }
    with open(save_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2)

    # 3. Write configuration_triune.py and modeling_triune.py
    with open(save_dir / "configuration_triune.py", "w", encoding="utf-8") as f:
        f.write(_CONFIGURATION_TRIUNE_CODE)

    with open(save_dir / "modeling_triune.py", "w", encoding="utf-8") as f:
        f.write(_MODELING_TRIUNE_CODE)

    # 4. Save model weights as safetensors
    # Normalize state dict keys: if model is direct TriuneTransformer, prefix with "transformer."
    # so TriuneForCausalLM loads cleanly
    core_model = getattr(model, "transformer", model)
    raw_state = core_model.state_dict()
    prefixed_state = {}
    seen_ptrs = set()
    for k, v in raw_state.items():
        pref_key = f"transformer.{k}"
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            prefixed_state[pref_key] = v.clone()
        else:
            seen_ptrs.add(ptr)
            prefixed_state[pref_key] = v

    # Use export_safetensors logic with prefixed keys
    safetensors_path = save_dir / "model.safetensors"
    try:
        from safetensors.torch import save_file
        save_file(prefixed_state, str(safetensors_path))
    except ImportError:
        from .exporter import _save_safetensors_fallback
        _save_safetensors_fallback(prefixed_state, safetensors_path)

    # 5. Copy or save tokenizer if available
    if tokenizer_path is not None and Path(tokenizer_path).exists():
        shutil.copy(str(tokenizer_path), str(save_dir / "tokenizer.json"))
    elif tokenizer is not None and hasattr(tokenizer, "save"):
        tokenizer.save(str(save_dir / "tokenizer.json"))

    return save_dir
