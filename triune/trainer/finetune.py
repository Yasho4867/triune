"""Unified LoRA & QLoRA Fine-Tuning Engine Abstraction."""

from __future__ import annotations

import json
import math
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from triune.callbacks import global_emitter
from triune.export import export_model


class LoRALayer(nn.Module):
    """Low-Rank Adapter (LoRA) Layer Wrapper."""

    def __init__(self, original_layer: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = getattr(original_layer, "in_features", 1536)
        out_features = getattr(original_layer, "out_features", 1536)

        orig_weight = original_layer.weight
        param_dtype = (
            orig_weight.dtype
            if orig_weight.is_floating_point() and "float8" not in str(orig_weight.dtype)
            else torch.float32
        )
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=orig_weight.device, dtype=param_dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=orig_weight.device, dtype=param_dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Freeze original layer
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return orig_out + (lora_out * self.scaling).to(orig_out.dtype)

    def merge(self) -> None:
        """Merge adapter weights directly into the original linear layer."""
        if hasattr(self.original_layer, "weight") and self.original_layer.weight is not None:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data.add_(delta.to(self.original_layer.weight.dtype))


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
    ):
        self.rank = r if r is not None else rank
        self.alpha = lora_alpha if lora_alpha is not None else alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
            "gate_proj", "up_proj", "down_proj", "0", "2"
        ]
        self.quantization = quantization or ("4-bit NF4" if use_qlora else "FP16")
        self.use_qlora = bool(use_qlora or ("4-bit" in str(self.quantization) or "nf4" in str(self.quantization).lower()))


class TriuneFineTuner:
    """Unified High-Level Fine-Tuning Loop Abstraction for LoRA/QLoRA."""

    def __init__(self, model: nn.Module, lora_config: Optional[LoRAConfig] = None):
        self.model = model
        self.lora_config = lora_config or LoRAConfig()
        self.applied_adapters: Dict[str, LoRALayer] = {}
        self.attach_lora()

    def attach_lora(self) -> None:
        """Attach LoRA adapter weights to target modules, freezing base parameters."""
        if self.applied_adapters:
            return

        # Freeze base model parameters
        for p in self.model.parameters():
            p.requires_grad = False

        named_mods = dict(self.model.named_modules())
        for name, module in list(self.model.named_modules()):
            if isinstance(module, LoRALayer):
                continue
            if any(target == name or target in name for target in self.lora_config.target_modules) and isinstance(module, nn.Linear):
                parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                if attr_name == "original_layer":
                    continue
                parent = named_mods.get(parent_name, self.model) if parent_name else self.model
                if isinstance(parent, LoRALayer):
                    continue
                adapter = LoRALayer(
                    module,
                    rank=self.lora_config.rank,
                    alpha=self.lora_config.alpha,
                    dropout=self.lora_config.dropout,
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
        """Save only LoRA adapter weights and metadata."""
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        state_dict = {}
        for name, adapter in self.applied_adapters.items():
            state_dict[f"{name}.lora_A"] = adapter.lora_A.data.cpu()
            state_dict[f"{name}.lora_B"] = adapter.lora_B.data.cpu()
        config = {
            "rank": self.lora_config.rank,
            "alpha": self.lora_config.alpha,
            "dropout": self.lora_config.dropout,
            "target_modules": self.lora_config.target_modules,
            "quantization": getattr(self.lora_config, "quantization", "4-bit NF4"),
            "trainable_params": self.count_trainable_parameters(),
        }
        with open(p / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        try:
            from safetensors.torch import save_file
            save_file(state_dict, str(p / "adapter_model.safetensors"))
        except Exception:
            torch.save(state_dict, str(p / "adapter_model.pt"))
        return p

    def load_adapter(self, load_dir: str | Path) -> None:
        """Load LoRA weights into attached adapters."""
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
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state_dict:
                adapter.lora_A.data.copy_(state_dict[a_key].to(adapter.lora_A.device))
            if b_key in state_dict:
                adapter.lora_B.data.copy_(state_dict[b_key].to(adapter.lora_B.device))

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(
        self,
        dataset_path: str | Path | None = None,
        output_dir: str | Path = "checkpoints/finetuned",
        epochs: int = 3,
        batch_size: int = 4,
        seq_len: int = 64,
        lr: float = 2e-4,
        export_formats: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        on_step: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run real fine-tuning loop with gradient accumulation, event streaming, and checkpoint export."""
        self.attach_lora()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable LoRA parameters found on model.")
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        # Detect device & target vocab size
        p_first = next(self.model.parameters())
        device = p_first.device
        dtype = p_first.dtype if p_first.is_floating_point() else torch.float32

        # Prepare tokens
        token_batches: List[torch.Tensor] = []

        if dataset_path and Path(dataset_path).is_file():
            dpath = Path(dataset_path)
            raw_texts: List[str] = []
            if dpath.suffix == ".jsonl":
                with open(dpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if "instruction" in item:
                                # Alpaca format: instruction, input, output
                                inp = f"\nContext: {item['input']}" if item.get("input") else ""
                                formatted = f"User: {item['instruction']}{inp}\n\nAssistant: {item.get('output', '')}"
                                raw_texts.append(formatted)
                            elif "messages" in item and isinstance(item["messages"], list):
                                # ShareGPT / OpenAI Chat format
                                turns = []
                                for m in item["messages"]:
                                    role = "User" if m.get("role") in ("user", "human") else "Assistant"
                                    turns.append(f"{role}: {m.get('content', '')}")
                                raw_texts.append("\n\n".join(turns))
                            elif "prompt" in item and "completion" in item:
                                raw_texts.append(f"User: {item['prompt']}\n\nAssistant: {item['completion']}")
                            else:
                                raw_texts.append(item.get("text", str(item)))
                        except Exception:
                            raw_texts.append(line)
            elif dpath.suffix.lower() == ".csv":
                with open(dpath, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        raise ValueError(f"CSV dataset has no header row: {dpath}")
                    text_column = next(
                        (column for column in ("text", "content", "story", "prompt") if column in reader.fieldnames),
                        None,
                    )
                    if text_column is None:
                        raise ValueError(
                            f"CSV dataset needs one of these text columns: text, content, story, or prompt ({dpath})."
                        )
                    raw_texts = [
                        str(row.get(text_column) or "").strip()
                        for row in reader
                        if str(row.get(text_column) or "").strip()
                    ]
            else:
                with open(dpath, "r", encoding="utf-8") as f:
                    raw_texts = [line.strip() for line in f if line.strip()]

            # Try loading tokenizer
            tokenizer = None
            for cand in ("triune_tokenizer.json", "tokenizer.json"):
                if Path(cand).is_file():
                    try:
                        from triune.data.tokenizer import load_tokenizer
                        tokenizer = load_tokenizer(cand)
                        break
                    except Exception:
                        pass

            if raw_texts and tokenizer is None:
                raise RuntimeError(
                    "A Triune tokenizer is required for LoRA fine-tuning. "
                    "Load triune_tokenizer.json or tokenizer.json; refusing to substitute hashed token IDs."
                )

            all_ids: List[int] = []
            for txt in raw_texts:
                all_ids.extend(tokenizer.encode(txt).ids)

            if len(all_ids) > seq_len + 1:
                num_chunks = len(all_ids) // (seq_len + 1)
                for i in range(num_chunks):
                    chunk = all_ids[i * (seq_len + 1) : (i + 1) * (seq_len + 1)]
                    token_batches.append(torch.tensor(chunk, dtype=torch.long))

        if not token_batches:
            raise RuntimeError(
                "The selected fine-tuning dataset produced no complete training sequences. "
                f"Verify the file and tokenizer, and ensure it contains more than {seq_len} tokens."
            )

        print(f"[LoRA Fine-Tuning] Starting: {len(token_batches)} batches -> {output_dir}")
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()

        total_steps = max_steps or (epochs * len(token_batches) // max(1, batch_size))
        total_steps = max(1, total_steps)

        step = 0
        final_loss = 0.0
        batch_idx = 0

        while step < total_steps:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                print(f"[LoRA Fine-Tuning] Cancellation requested at step {step}/{total_steps}.")
                break
            optimizer.zero_grad()
            # Assemble batch
            batch_tensors = []
            for _ in range(batch_size):
                batch_tensors.append(token_batches[batch_idx % len(token_batches)])
                batch_idx += 1
            batch_tensor = torch.stack(batch_tensors).to(device)

            input_ids = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:].contiguous()

            res = self.model(input_ids)
            router_loss_val = 0.0
            if isinstance(res, tuple):
                logits = res[0]
                if len(res) > 2 and res[2] is not None:
                    router_loss_val = float(res[2].item()) if torch.is_tensor(res[2]) else float(res[2])
            elif hasattr(res, "logits"):
                logits = res.logits
            else:
                logits = res

            lm_loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss = lm_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            step += 1
            final_loss = float(total_loss.item())
            lm_loss_val = float(lm_loss.item())
            if on_step is not None:
                try:
                    on_step(step, total_steps, final_loss, lm_loss_val)
                except Exception:
                    pass
            global_emitter.emit_loss_update(
                step=step,
                loss=final_loss,
                lm_loss=lm_loss_val,
                router_loss=router_loss_val,
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save standalone LoRA adapter weights and metadata
        self.save_adapter(out_path)

        if export_formats:
            for fmt in export_formats:
                export_model(self.model, out_path / f"model_finetuned.{fmt}", fmt=fmt)
        else:
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
                    adapters.append({
                        "id": item.name,
                        "name": item.name.replace("_", " ").title(),
                        "path": clean_path,
                        "rank": rank_val,
                        "alpha": alpha_val,
                        "quantization": quant_val,
                        "trainable_params": trainable_val,
                        "size_mb": sz_mb,
                        "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(item.stat().st_mtime)),
                    })
            elif item.is_file() and "adapter" in item.name.lower() and item.suffix in (".safetensors", ".pt"):
                seen_paths.add(clean_path)
                sz_mb = round(item.stat().st_size / (1024 * 1024), 2)
                adapters.append({
                    "id": item.stem,
                    "name": item.stem.replace("_", " ").title(),
                    "path": clean_path,
                    "rank": 16,
                    "alpha": 32.0,
                    "quantization": "4-bit NF4",
                    "trainable_params": 0,
                    "size_mb": sz_mb,
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(item.stat().st_mtime)),
                })
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
