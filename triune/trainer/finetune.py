"""Unified LoRA & QLoRA Fine-Tuning Engine Abstraction."""

from __future__ import annotations

import contextlib
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from triune.callbacks import global_emitter
from triune.export import export_model
from triune.kernels.fast_cross_entropy import chunked_cross_entropy_from_hidden, fast_cross_entropy
from triune.kernels.fast_lora import fast_lora_forward, fast_lora_linear
from triune.kernels.fast_nvfp4 import NVFP4_GROUP_SIZE, NVFP4Linear, fwht_16, nvfp4_dequantize, nvfp4_quantize


def _infer_layer_device_dtype(
    layer: nn.Module, fallback_device: torch.device | None = None
) -> tuple[torch.device, torch.dtype]:
    """Infer target LoRA device and dtype without triggering full NVFP4 dequantization or meta failures."""
    if isinstance(layer, NVFP4Linear):
        dt = getattr(layer, "dtype", torch.bfloat16)
        sc = getattr(layer, "scale", None)
        dev = (
            sc.device
            if (sc is not None and sc.numel() > 0 and sc.device.type != "meta")
            else (fallback_device or torch.device("cpu"))
        )
        return dev, dt
    orig_weight = getattr(layer, "weight", None)
    if orig_weight is None and hasattr(layer, "linear"):
        orig_weight = getattr(layer.linear, "weight", None)
    if orig_weight is None:
        return (fallback_device or torch.device("cpu")), torch.float32
    dev = orig_weight.device if orig_weight.device.type != "meta" else (fallback_device or torch.device("cpu"))
    dt = (
        orig_weight.dtype
        if orig_weight.is_floating_point() and "float8" not in str(orig_weight.dtype)
        else torch.float32
    )
    return dev, dt


class LoRALayer(nn.Module):
    """Low-Rank Adapter (LoRA) Layer Wrapper supporting BF16, FP8, and NVFP4 base layers."""

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        adapter_device: torch.device | None = None,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = getattr(original_layer, "in_features", 1536)
        out_features = getattr(original_layer, "out_features", 1536)

        param_dev, param_dtype = _infer_layer_device_dtype(original_layer, fallback_device=adapter_device)
        if adapter_device is not None:
            param_dev = adapter_device

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=param_dev, dtype=param_dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=param_dev, dtype=param_dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Freeze original layer
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            type(self.original_layer) is nn.Linear
            and (isinstance(self.dropout, nn.Identity) or not self.training)
            and x.dtype == self.original_layer.weight.dtype
            and x.dtype == self.lora_A.dtype
        ):
            return fast_lora_linear(
                x,
                self.original_layer.weight,
                self.original_layer.bias,
                self.lora_A,
                self.lora_B,
                self.scaling,
            )
        orig_out = self.original_layer(x)
        drop_x = self.dropout(x)
        if drop_x.device != self.lora_A.device or drop_x.dtype != self.lora_A.dtype:
            drop_x = drop_x.to(device=self.lora_A.device, dtype=self.lora_A.dtype)
        lora_out = fast_lora_forward(drop_x, self.lora_A, self.lora_B, self.scaling)
        return orig_out + lora_out.to(device=orig_out.device, dtype=orig_out.dtype)

    @property
    def weight(self):
        return getattr(self.original_layer, "weight", None)

    @property
    def bias(self):
        return getattr(self.original_layer, "bias", None)

    def merge(self) -> None:
        """Merge adapter weights directly into the original linear (or NVFP4Linear) layer."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        if isinstance(self.original_layer, NVFP4Linear):
            if self.original_layer.packed_weight.numel() == 0:
                return
            w = nvfp4_dequantize(
                self.original_layer.packed_weight,
                self.original_layer.scale,
                dtype=self.original_layer.dtype,
            )
            if self.original_layer.hadamard:
                w = fwht_16(w)
            w_merged = w + delta.to(device=w.device, dtype=w.dtype)
            w_pack = fwht_16(w_merged) if self.original_layer.hadamard else w_merged
            packed, scale = nvfp4_quantize(w_pack)
            self.original_layer.packed_weight.copy_(packed)
            self.original_layer.scale.copy_(scale)
        elif hasattr(self.original_layer, "weight") and self.original_layer.weight is not None:
            self.original_layer.weight.data.add_(
                delta.to(device=self.original_layer.weight.device, dtype=self.original_layer.weight.dtype)
            )


class LoRAConfig:
    """Configuration container for LoRA / QLoRA adapters."""

    def __init__(
        self,
        rank: int = 16,
        r: Optional[int] = None,
        alpha: float = 32.0,
        lora_alpha: Optional[float] = None,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        use_qlora: bool = False,
        quantization: Optional[str] = None,
        quantize_base: Optional[bool] = None,
        hadamard: bool = True,
    ):
        self.rank = r if r is not None else rank
        self.alpha = lora_alpha if lora_alpha is not None else alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "qkv_proj",
            "gate_up_proj",
            "0",
            "2",
        ]
        self.quantization = quantization or ("4-bit NVFP4" if use_qlora else "FP16")
        q_low = str(self.quantization).lower()
        self.use_qlora = bool(use_qlora or ("4-bit" in q_low or "nf4" in q_low or "nvfp4" in q_low))
        self.quantize_base = bool(quantize_base if quantize_base is not None else ("nvfp4" in q_low))
        self.hadamard = bool(hadamard)

    @classmethod
    def from_pretrained(cls, adapter_dir: str | Path) -> "LoRAConfig":
        """Load LoRAConfig from adapter_config.json when present, else return defaults."""
        p = Path(adapter_dir)
        cfg_file = (p.parent if p.is_file() else p) / "adapter_config.json"
        if not cfg_file.is_file():
            return cls()
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        return cls(
            rank=int(data.get("r", data.get("rank", 16))),
            alpha=float(data.get("lora_alpha", data.get("alpha", 32.0))),
            dropout=float(data.get("lora_dropout", data.get("dropout", 0.05))),
            target_modules=data.get("target_modules"),
            quantization=data.get("quantization"),
            quantize_base=data.get("quantize_base"),
        )


_BASE_ADAPTER_MODULES = (nn.Linear, NVFP4Linear)


class TriuneFineTuner:
    """Unified High-Level Fine-Tuning Loop Abstraction for LoRA/QLoRA."""

    def __init__(
        self,
        model: nn.Module,
        lora_config: Optional[LoRAConfig] = None,
        *,
        tokenizer: Optional[Any] = None,
        model_name_or_path: Optional[str] = None,
        adapter_device: Optional[torch.device | str] = None,
    ):
        self.model = model
        self.lora_config = lora_config or LoRAConfig()
        self.tokenizer = getattr(tokenizer, "hf", tokenizer) if tokenizer is not None else None
        self.model_name_or_path = (
            model_name_or_path
            or getattr(model, "name_or_path", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None)
        )
        if adapter_device is not None:
            self._adapter_device: Optional[torch.device] = torch.device(adapter_device)
        else:
            eng = getattr(model, "_streaming_engine", None)
            if eng is not None and hasattr(eng, "device"):
                self._adapter_device = torch.device(eng.device)
            else:
                self._adapter_device = None
        self.applied_adapters: Dict[str, LoRALayer] = {}
        self.attach_lora()

    def attach_lora(self) -> None:
        """Attach LoRA adapter weights to target modules, freezing base parameters."""
        if self.applied_adapters:
            return

        # Freeze base model parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Infer fallback adapter device (useful when base modules are on meta/0-byte placeholders)
        fallback_dev = self._adapter_device
        if fallback_dev is None:
            for p in self.model.parameters():
                if p.device.type != "meta" and p.numel() > 0:
                    fallback_dev = p.device
                    break

        named_mods = dict(self.model.named_modules())
        for name, module in list(self.model.named_modules()):
            if isinstance(module, LoRALayer):
                continue
            if any(target == name or target in name for target in self.lora_config.target_modules) and isinstance(
                module, _BASE_ADAPTER_MODULES
            ):
                parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                if attr_name in ("original_layer", "lm_head", "final_head", "exit_head", "router"):
                    continue
                parent = named_mods.get(parent_name, self.model) if parent_name else self.model
                if isinstance(parent, LoRALayer):
                    continue
                base_mod = module
                if (
                    self.lora_config.quantize_base
                    and type(module) is nn.Linear
                    and module.in_features % NVFP4_GROUP_SIZE == 0
                    and module.weight.device.type != "meta"
                    and module.weight.numel() > 0
                ):
                    base_mod = NVFP4Linear.from_linear(module, hadamard=self.lora_config.hadamard)
                adapter = LoRALayer(
                    base_mod,
                    rank=self.lora_config.rank,
                    alpha=self.lora_config.alpha,
                    dropout=self.lora_config.dropout,
                    adapter_device=fallback_dev,
                )
                setattr(parent, attr_name, adapter)
                self.applied_adapters[name] = adapter

    def detach_lora(self) -> None:
        """Detach LoRA adapter weights and restore original layers."""
        named_mods = dict(self.model.named_modules())
        for name, adapter in list(self.applied_adapters.items()):
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = named_mods.get(parent_name, self.model) if parent_name else self.model
            if hasattr(parent, attr_name) and isinstance(getattr(parent, attr_name), LoRALayer):
                setattr(parent, attr_name, adapter.original_layer)
        self.applied_adapters.clear()
        # Unfreeze base model parameters
        for p in self.model.parameters():
            p.requires_grad = True

    def merge_and_unload(self) -> None:
        """Merge adapter weights directly into the original layers and detach."""
        for adapter in self.applied_adapters.values():
            adapter.merge()
        self.detach_lora()

    def save_adapter(self, save_dir: str | Path) -> Path:
        """Save LoRA adapter weights and metadata in Hugging Face PEFT standard format."""
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        state_dict = {}
        for name, adapter in self.applied_adapters.items():
            a_cpu = adapter.lora_A.detach().cpu().contiguous()
            b_cpu = adapter.lora_B.detach().cpu().contiguous()
            state_dict[f"{name}.lora_A"] = a_cpu
            state_dict[f"{name}.lora_B"] = b_cpu
            # Distinct clones avoid safetensors shared-storage errors when adapters live on CPU.
            state_dict[f"base_model.model.{name}.lora_A.weight"] = a_cpu.clone()
            state_dict[f"base_model.model.{name}.lora_B.weight"] = b_cpu.clone()

        base_name = (
            self.model_name_or_path
            or getattr(self.model, "active_model_id", None)
            or getattr(self.model, "name_or_path", None)
            or "triune-base"
        )
        config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": str(base_name),
            "r": self.lora_config.rank,
            "lora_alpha": self.lora_config.alpha,
            "lora_dropout": self.lora_config.dropout,
            "target_modules": self.lora_config.target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "rank": self.lora_config.rank,
            "alpha": self.lora_config.alpha,
            "dropout": self.lora_config.dropout,
            "quantization": getattr(self.lora_config, "quantization", "4-bit NVFP4"),
            "quantize_base": bool(getattr(self.lora_config, "quantize_base", False)),
            "trainable_params": self.count_trainable_parameters(),
        }
        with open(p / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        try:
            from safetensors.torch import save_file

            save_file(state_dict, str(p / "adapter_model.safetensors"))
        except Exception:
            torch.save(state_dict, str(p / "adapter_model.pt"))
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            try:
                self.tokenizer.save_pretrained(str(p))
            except Exception:
                pass
        return p

    def load_adapter(self, load_dir: str | Path) -> None:
        """Load LoRA weights into attached adapters from native or Hugging Face PEFT format."""
        p = Path(load_dir)
        if p.is_file():
            weights_file = p
        else:
            weights_file = p / "adapter_model.safetensors"
            if not weights_file.is_file():
                weights_file = p / "adapter_model.pt"

        if not weights_file.is_file():
            raise FileNotFoundError(f"Adapter weights not found at {load_dir}")

        if weights_file.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(weights_file))
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        self.attach_lora()
        for name, adapter in self.applied_adapters.items():
            cand_a = [
                f"{name}.lora_A",
                f"{name}.lora_A.weight",
                f"base_model.model.{name}.lora_A.weight",
                f"base_model.model.{name}.lora_A",
            ]
            cand_b = [
                f"{name}.lora_B",
                f"{name}.lora_B.weight",
                f"base_model.model.{name}.lora_B.weight",
                f"base_model.model.{name}.lora_B",
            ]
            for ka in cand_a:
                if ka in state_dict:
                    adapter.lora_A.data.copy_(state_dict[ka].to(adapter.lora_A.device))
                    break
            for kb in cand_b:
                if kb in state_dict:
                    adapter.lora_B.data.copy_(state_dict[kb].to(adapter.lora_B.device))
                    break

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return list of all parameters requiring gradient in the model."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_optimizer(
        self,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        use_8bit: bool = False,
    ) -> torch.optim.Optimizer:
        """Create high-performance optimizer for trainable LoRA adapter parameters."""
        trainable_params = self.get_trainable_parameters()
        if not trainable_params:
            raise RuntimeError("No trainable LoRA parameters found on model.")
        if use_8bit:
            try:
                import bitsandbytes as bnb

                return bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=weight_decay)
            except Exception:
                pass
        p_first = trainable_params[0]
        use_fused = p_first.device.type == "cuda"
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, fused=use_fused)

    @staticmethod
    def _encode_ids(tokenizer: Any, text: str) -> List[int]:
        if hasattr(tokenizer, "encode"):
            try:
                enc = tokenizer.encode(text, add_special_tokens=False)
                if isinstance(enc, list):
                    return [int(x) for x in enc]
                if hasattr(enc, "ids"):
                    return [int(x) for x in enc.ids]
            except TypeError:
                enc = tokenizer.encode(text)
                return [int(x) for x in (enc.ids if hasattr(enc, "ids") else enc)]
        raise TypeError("Unsupported tokenizer object for fine-tuning.")

    def _resolve_tokenizer(self, tokenizer: Optional[Any]) -> Any:
        if tokenizer is not None:
            return getattr(tokenizer, "hf", tokenizer)
        if self.tokenizer is not None:
            return getattr(self.tokenizer, "hf", self.tokenizer)
        repo_hint = self.model_name_or_path
        if repo_hint:
            try:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(str(repo_hint))
            except Exception:
                pass
        for cand in ("triune_tokenizer.json", "tokenizer.json"):
            if Path(cand).is_file():
                try:
                    from triune.data.tokenizer import load_tokenizer

                    return load_tokenizer(cand)
                except Exception:
                    pass
        raise RuntimeError(
            "A tokenizer is required for LoRA fine-tuning. Pass tokenizer=..., use an HF model with "
            "tokenizer files, or provide triune_tokenizer.json."
        )

    @staticmethod
    def _parse_dataset_records(
        dataset_path: str | Path | None,
        *,
        dataset_split: str = "train",
        max_samples: int = 2000,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not dataset_path:
            return records
        dpath = Path(dataset_path)
        if dpath.is_file():
            if dpath.suffix.lower() == ".jsonl":
                with open(dpath, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            item = json.loads(s)
                            records.append(item if isinstance(item, dict) else {"text": str(item)})
                        except Exception:
                            records.append({"text": s})
            elif dpath.suffix.lower() == ".csv":
                with open(dpath, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        raise ValueError(f"CSV dataset has no header row: {dpath}")
                    text_column = next(
                        (col for col in ("text", "content", "story", "prompt") if col in reader.fieldnames),
                        None,
                    )
                    if text_column is None:
                        raise ValueError(
                            f"CSV dataset needs one of these text columns: text, content, story, or prompt ({dpath})."
                        )
                    for row in reader:
                        val = str(row.get(text_column) or "").strip()
                        if val:
                            records.append({"text": val})
            else:
                with open(dpath, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            records.append({"text": s})
            return records

        # Remote / cached Hugging Face dataset identifier fallback (e.g. 'HuggingFaceTB/smoltalk2')
        from datasets import load_dataset

        ds = load_dataset(str(dataset_path), split=dataset_split, streaming=True)
        for idx, row in enumerate(ds):
            if max_samples and idx >= max_samples:
                break
            if isinstance(row, dict):
                records.append(row)
        return records

    def _build_training_examples(
        self,
        records: List[Dict[str, Any]],
        tokenizer: Any,
        seq_len: int,
        *,
        mask_prompt_tokens: bool = True,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Tokenize raw text or SFT dialogue records into (input_ids, labels) tensors of length seq_len+1."""
        inputs_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        raw_stream: List[int] = []
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None and hasattr(tokenizer, "token_to_id"):
            eos_id = tokenizer.token_to_id("[SEP]") or tokenizer.token_to_id("</s>")
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None and hasattr(tokenizer, "token_to_id"):
            pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = int(eos_id) if eos_id is not None else 0

        has_chat_tmpl = bool(hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None))

        for item in records:
            prompt_str: Optional[str] = None
            full_str: Optional[str] = None
            if "messages" in item and isinstance(item["messages"], list) and item["messages"]:
                msgs = [
                    {
                        "role": (
                            "user"
                            if (m.get("role") or "").lower() in ("user", "human")
                            else ("system" if (m.get("role") or "").lower() == "system" else "assistant")
                        ),
                        "content": str(m.get("content", "")),
                    }
                    for m in item["messages"]
                ]
                if has_chat_tmpl:
                    try:
                        full_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                        # Find prompt prefix up to final assistant turn for SFT label masking
                        if mask_prompt_tokens and msgs[-1]["role"] == "assistant" and len(msgs) >= 2:
                            prompt_str = tokenizer.apply_chat_template(
                                msgs[:-1], tokenize=False, add_generation_prompt=True
                            )
                    except Exception:
                        full_str = None
                if full_str is None:
                    turns = [f"{m['role'].capitalize()}: {m['content']}" for m in msgs]
                    full_str = "\n\n".join(turns)
                    if mask_prompt_tokens and msgs[-1]["role"] == "assistant" and len(msgs) >= 2:
                        prompt_str = "\n\n".join(turns[:-1]) + "\n\nAssistant:"
            elif "instruction" in item and "output" in item:
                inp = f"\nContext: {item['input']}" if item.get("input") else ""
                user_msg = f"{item['instruction']}{inp}"
                ans_msg = str(item.get("output", ""))
                if has_chat_tmpl:
                    try:
                        msgs = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": ans_msg}]
                        full_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                        if mask_prompt_tokens:
                            prompt_str = tokenizer.apply_chat_template(
                                msgs[:-1], tokenize=False, add_generation_prompt=True
                            )
                    except Exception:
                        full_str = None
                if full_str is None:
                    prompt_str = f"User: {user_msg}\n\nAssistant:" if mask_prompt_tokens else None
                    full_str = f"User: {user_msg}\n\nAssistant: {ans_msg}"
            elif "prompt" in item and "completion" in item:
                prompt_str = f"User: {item['prompt']}\n\nAssistant:" if mask_prompt_tokens else None
                full_str = f"User: {item['prompt']}\n\nAssistant: {item['completion']}"
            else:
                text = str(item.get("text", item.get("content", "")))
                if text:
                    ids = self._encode_ids(tokenizer, text)
                    raw_stream.extend(ids)
                    if eos_id is not None:
                        raw_stream.append(int(eos_id))
                continue

            if full_str:
                full_ids = self._encode_ids(tokenizer, full_str)
                if eos_id is not None and (not full_ids or full_ids[-1] != int(eos_id)):
                    full_ids.append(int(eos_id))
                if len(full_ids) < 4:
                    continue
                if mask_prompt_tokens and prompt_str:
                    prompt_ids = self._encode_ids(tokenizer, prompt_str)
                    n_mask = min(len(prompt_ids), max(0, len(full_ids) - 1))
                    ex_labels = ([-100] * n_mask) + full_ids[n_mask:]
                    # Pad/truncate per-example SFT record so assistant masks stay aligned cleanly
                    target_len = seq_len + 1
                    if len(full_ids) > target_len:
                        full_ids = full_ids[:target_len]
                        ex_labels = ex_labels[:target_len]
                    elif len(full_ids) < target_len:
                        pad_n = target_len - len(full_ids)
                        full_ids = full_ids + ([int(pad_id)] * pad_n)
                        ex_labels = ex_labels + ([-100] * pad_n)
                    if any(v != -100 for v in ex_labels[1:]):
                        inputs_list.append(torch.tensor(full_ids, dtype=torch.long))
                        labels_list.append(torch.tensor(ex_labels, dtype=torch.long))
                else:
                    raw_stream.extend(full_ids)

        if len(raw_stream) > seq_len + 1:
            n_chunks = len(raw_stream) // (seq_len + 1)
            for i in range(n_chunks):
                chunk = raw_stream[i * (seq_len + 1) : (i + 1) * (seq_len + 1)]
                t = torch.tensor(chunk, dtype=torch.long)
                inputs_list.append(t)
                labels_list.append(t)

        return inputs_list, labels_list

    def fit(
        self,
        dataset_path: str | Path | None = None,
        output_dir: str | Path = "checkpoints/finetuned",
        epochs: int = 3,
        batch_size: int = 4,
        seq_len: int = 64,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        use_8bit: bool = False,
        export_formats: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        on_step: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        dataset_split: str = "train",
        max_samples: int = 2000,
        grad_accum_steps: int = 1,
        stream_layers: bool = False,
        stream_chunk_size: int = 1,
        gradient_checkpointing: bool = False,
        use_amp: bool = True,
        mask_prompt_tokens: bool = True,
        use_chunked_ce: bool = True,
    ) -> Dict[str, Any]:
        """Run LoRA/QLoRA fine-tuning with optional chat prompt masking, chunked CE, and streaming."""
        self.attach_lora()
        trainable_params = self.get_trainable_parameters()
        if not trainable_params:
            raise RuntimeError("No trainable LoRA parameters found on model.")

        resolved_tok = self._resolve_tokenizer(tokenizer)
        self.tokenizer = resolved_tok

        records = self._parse_dataset_records(dataset_path, dataset_split=dataset_split, max_samples=max_samples)
        input_batches, label_batches = self._build_training_examples(
            records, resolved_tok, seq_len, mask_prompt_tokens=mask_prompt_tokens
        )
        if not input_batches:
            raise RuntimeError(
                "The selected fine-tuning dataset produced no complete training sequences. "
                f"Verify the file/dataset and tokenizer, and ensure it contains more than {seq_len} tokens."
            )

        device = self._adapter_device or trainable_params[0].device
        if device.type == "meta":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        existing_engine = getattr(self.model, "_streaming_engine", None)
        created_stream_engine = None
        if stream_layers and existing_engine is None:
            from triune.runtime.streaming import enable_layer_streaming

            created_stream_engine = enable_layer_streaming(
                self.model,
                device=device,
                chunk_size=max(1, int(stream_chunk_size)),
                activation_checkpointing=True,
            )
            existing_engine = created_stream_engine

        if gradient_checkpointing and existing_engine is None and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        optimizer = self.get_optimizer(lr=lr, weight_decay=weight_decay, use_8bit=use_8bit)
        if created_stream_engine is not None:
            created_stream_engine.set_optimizer(optimizer)

        is_hf_chunked_eligible = bool(
            use_chunked_ce
            and hasattr(self.model, "model")
            and hasattr(self.model, "lm_head")
            and type(self.model.lm_head) is nn.Linear
            and getattr(self.model.lm_head, "weight", None) is not None
            and not self.model.lm_head.weight.is_meta
            and self.model.lm_head.weight.device == device
        )

        accum = max(1, int(grad_accum_steps))
        n_examples = len(input_batches)
        total_steps = max_steps or max(1, (epochs * n_examples) // max(1, batch_size * accum))
        total_steps = max(1, total_steps)
        warmup_steps = max(1, min(total_steps // 10, 50)) if total_steps > 4 else 0

        print(
            f"[LoRA Fine-Tuning] Starting: {n_examples} sequences, {total_steps} steps (accum={accum}) -> {output_dir}"
        )
        self.model.train()

        step = 0
        final_loss = 0.0
        batch_idx = 0

        try:
            while step < total_steps:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    print(f"[LoRA Fine-Tuning] Cancellation requested at step {step}/{total_steps}.")
                    break

                # Cosine warmup schedule across trainable groups
                if total_steps > 1:
                    if step < warmup_steps:
                        scale_lr = float(step + 1) / float(max(1, warmup_steps))
                    else:
                        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                        scale_lr = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr * scale_lr

                optimizer.zero_grad(set_to_none=True)
                accum_loss_val = 0.0
                router_loss_val = 0.0

                for _micro in range(accum):
                    b_in = []
                    b_lbl = []
                    for _ in range(batch_size):
                        pos = batch_idx % n_examples
                        b_in.append(input_batches[pos])
                        b_lbl.append(label_batches[pos])
                        batch_idx += 1

                    in_tensor = torch.stack(b_in).to(device)
                    lbl_tensor = torch.stack(b_lbl).to(device)
                    input_ids = in_tensor[:, :-1]
                    targets = lbl_tensor[:, 1:].contiguous()

                    amp_ctx = (
                        torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if (use_amp and device.type == "cuda")
                        else contextlib.nullcontext()
                    )
                    with amp_ctx:
                        if is_hf_chunked_eligible:
                            inner_out = self.model.model(input_ids=input_ids, use_cache=False)
                            hidden = inner_out[0] if isinstance(inner_out, tuple) else inner_out.last_hidden_state
                            lm_loss = chunked_cross_entropy_from_hidden(
                                hidden,
                                self.model.lm_head,
                                targets,
                                chunk_size=256,
                                ignore_index=-100,
                                force_chunked=True,
                            )
                        else:
                            res = self.model(input_ids)
                            if isinstance(res, tuple):
                                logits = res[0]
                                if len(res) > 2 and res[2] is not None:
                                    router_loss_val = float(res[2].item()) if torch.is_tensor(res[2]) else float(res[2])
                            elif hasattr(res, "logits"):
                                logits = res.logits
                            else:
                                logits = res
                            lm_loss = fast_cross_entropy(
                                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
                            )
                        micro_loss = lm_loss / float(accum)

                    micro_loss.backward()
                    accum_loss_val += float(lm_loss.item()) / float(accum)

                if hasattr(existing_engine, "step_streaming_optimizer"):
                    existing_engine.step_streaming_optimizer(optimizer, grad_clip=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                step += 1
                final_loss = accum_loss_val
                if on_step is not None:
                    try:
                        on_step(step, total_steps, final_loss, final_loss)
                    except Exception:
                        pass
                global_emitter.emit_loss_update(
                    step=step,
                    loss=final_loss,
                    lm_loss=final_loss,
                    router_loss=router_loss_val,
                )
        finally:
            if created_stream_engine is not None:
                created_stream_engine.detach()

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save lightweight PEFT LoRA adapter + tokenizer bundle
        self.save_adapter(out_path)

        total_model_params = sum(p.numel() for p in self.model.parameters())
        if export_formats:
            for fmt in export_formats:
                export_model(self.model, out_path / f"model_finetuned.{fmt}", fmt=fmt)
        elif existing_engine is None and not self.lora_config.quantize_base and total_model_params <= 250_000_000:
            try:
                export_model(self.model, out_path / "model_finetuned.safetensors", fmt="safetensors")
            except Exception as e:
                print(f"[LoRA Fine-Tuning] Full model export skipped: {e}")

        return {
            "status": "completed",
            "final_loss": round(final_loss, 4),
            "total_steps": step,
            "trainable_params": self.count_trainable_parameters(),
            "output_dir": str(out_path),
        }


def list_adapters(scan_dirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Discover all saved LoRA/QLoRA adapters on disk across standard directories."""
    import time

    scan_dirs = scan_dirs or ["checkpoints", "checkpoints/adapters", "exports"]
    adapters: List[Dict[str, Any]] = []
    seen_paths = set()
    for sdir in scan_dirs:
        p = Path(sdir)
        if not p.is_dir():
            continue
        for item in p.iterdir():
            clean_path = str(item).replace("\\", "/")
            if clean_path in seen_paths:
                continue
            if item.is_dir():
                cfg_file = item / "adapter_config.json"
                weights_file = item / "adapter_model.safetensors"
                if not weights_file.exists():
                    weights_file = item / "adapter_model.pt"
                if cfg_file.exists() or weights_file.exists():
                    seen_paths.add(clean_path)
                    rank_val = 16
                    alpha_val = 32.0
                    quant_val = "4-bit NF4"
                    trainable_val = 0
                    if cfg_file.exists():
                        try:
                            with open(cfg_file, "r", encoding="utf-8") as f:
                                cdata = json.load(f)
                                rank_val = cdata.get("rank", rank_val)
                                alpha_val = cdata.get("alpha", alpha_val)
                                quant_val = cdata.get("quantization", quant_val)
                                trainable_val = cdata.get("trainable_params", 0)
                        except Exception:
                            pass
                    sz_mb = round(weights_file.stat().st_size / (1024 * 1024), 2) if weights_file.exists() else 0.0
                    adapters.append(
                        {
                            "id": item.name,
                            "name": item.name.replace("_", " ").title(),
                            "path": clean_path,
                            "rank": rank_val,
                            "alpha": alpha_val,
                            "quantization": quant_val,
                            "trainable_params": trainable_val,
                            "size_mb": sz_mb,
                            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(item.stat().st_mtime)),
                        }
                    )
            elif item.is_file() and "adapter" in item.name.lower() and item.suffix in (".safetensors", ".pt"):
                seen_paths.add(clean_path)
                sz_mb = round(item.stat().st_size / (1024 * 1024), 2)
                adapters.append(
                    {
                        "id": item.stem,
                        "name": item.stem.replace("_", " ").title(),
                        "path": clean_path,
                        "rank": 16,
                        "alpha": 32.0,
                        "quantization": "4-bit NF4",
                        "trainable_params": 0,
                        "size_mb": sz_mb,
                        "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(item.stat().st_mtime)),
                    }
                )
    return adapters


def export_adapter(adapter_path: str | Path, output_path: str | Path | None = None, fmt: str = "safetensors") -> Path:
    """Export an adapter as a standalone SafeTensors package."""
    import shutil

    p = Path(adapter_path)
    if not p.exists():
        raise FileNotFoundError(f"Adapter source not found: {adapter_path}")
    src = p / "adapter_model.safetensors" if p.is_dir() else p
    if not src.exists():
        src = p / "adapter_model.pt"
    if not src.exists():
        raise FileNotFoundError(f"No adapter weight file in {adapter_path}")
    if output_path is None:
        name = p.name if p.is_dir() else p.stem
        out = Path("exports") / f"{name}.{fmt}"
    else:
        out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    return out


def merge_adapter_checkpoint(
    adapter_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    tokenizer_path: str | Path = "triune_tokenizer.json",
) -> Path:
    """Load base model checkpoint, attach and merge adapter, and save updated checkpoint."""
    from triune.inference import load_checkpoint_model

    model, tokenizer, checkpoint, device, _, _ = load_checkpoint_model(checkpoint_path, tokenizer_path)
    cfg = LoRAConfig()
    tuner = TriuneFineTuner(model, cfg)
    tuner.load_adapter(adapter_path)
    tuner.merge_and_unload()
    out = Path(output_path or f"checkpoints/merged_{Path(checkpoint_path).name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["model_state"] = model.state_dict()
    torch.save(checkpoint, str(out))
    return out
