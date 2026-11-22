"""NVMe-to-VRAM Quantized Layer Streaming Engine for Consumer GPUs.

Enables loading, running, and QLoRA fine-tuning multi-billion-parameter Hugging Face models
(e.g. 7B-27B+) on laptops with tight host RAM (16 GB) and consumer VRAM (8 GB) without ever
materializing full BF16 weights in RAM or GPU memory:

1. ``build_nvfp4_disk_cache`` opens safetensors shards one layer at a time via zero-copy mmap,
   applies optional FWHT-16 Hadamard rotation + 4-bit block-scaled NVFP4 packing, and writes
   compact per-layer shards (~0.625 bytes/param) to disk using < 1.5 GB peak RAM.
2. ``DiskNVFP4StreamingEngine`` instantiates the HF skeleton on ``meta``, keeps only root I/O,
   tiny 1D norms/biases, and trainable LoRA adapters resident on GPU/CPU, and double-buffers
   single layers from NVMe -> pinned CPU -> CUDA prefetch stream -> VRAM -> release, including
   full gradient-checkpointed backward recomputation for QLoRA.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from lucid_machines.kernels.fast_nvfp4 import (
    NVFP4_GROUP_SIZE,
    NVFP4Linear,
    fwht_16,
    nvfp4_quantize,
)

_EMPTY_U8 = torch.empty(0, dtype=torch.uint8)
_EMPTY_BF16 = torch.empty(0, dtype=torch.bfloat16)

_LAYER_KEY_PATTERNS = (
    re.compile(r"^(model\.layers)\.(\d+)\.(.+)$"),
    re.compile(r"^(transformer\.h)\.(\d+)\.(.+)$"),
    re.compile(r"^(model\.decoder\.layers)\.(\d+)\.(.+)$"),
    re.compile(r"^(layers)\.(\d+)\.(.+)$"),
)

_SENSITIVE_SUBSTRINGS = ("lm_head", "embed_tokens", "wte", "embed")


def resolve_hf_model_dir(model_id_or_dir: str | Path, *, local_files_only: bool = False) -> Path:
    """Resolve a local model directory or Hugging Face repo ID to a local directory with shards."""
    p = Path(model_id_or_dir)
    if p.is_dir():
        return p

    from huggingface_hub import snapshot_download

    allow = [
        "*.json",
        "*.safetensors",
        "tokenizer*",
        "*.model",
        "*.txt",
        "*.tiktoken",
    ]
    if not local_files_only:
        try:
            snap = Path(snapshot_download(str(model_id_or_dir), allow_patterns=allow, local_files_only=True))
            if (
                (snap / "model.safetensors").is_file()
                or (snap / "model.safetensors.index.json").is_file()
                or any(snap.glob("*.safetensors"))
            ):
                return snap
        except Exception:
            pass
    snap = snapshot_download(str(model_id_or_dir), allow_patterns=allow, local_files_only=local_files_only)
    return Path(snap)


def _discover_safetensors_map(model_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    """Return mapping from state-dict key -> shard Path without loading weights into RAM."""
    from safetensors import safe_open

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        with open(index_file, encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        resolved = {k: model_dir / rel for k, rel in weight_map.items()}
        shards = sorted(set(resolved.values()))
        return resolved, shards

    single_file = model_dir / "model.safetensors"
    if single_file.is_file():
        with safe_open(str(single_file), framework="pt", device="cpu") as f:
            keys = list(f.keys())
        return {k: single_file for k in keys}, [single_file]

    cand_shards = sorted(model_dir.glob("*.safetensors"))
    if cand_shards:
        resolved: dict[str, Path] = {}
        for sh in cand_shards:
            with safe_open(str(sh), framework="pt", device="cpu") as f:
                for k in f.keys():
                    resolved[k] = sh
        return resolved, cand_shards

    raise FileNotFoundError(
        f"No .safetensors checkpoints found in '{model_dir}'. "
        "Disk NVFP4 streaming requires Safetensors weights for zero-copy per-layer mmap conversion."
    )


def _detect_layer_layout(keys: list[str]) -> tuple[str, int]:
    """Infer transformer layer prefix (e.g. 'model.layers') and layer count from state keys."""
    counts: dict[str, int] = {}
    max_idx: dict[str, int] = {}
    for k in keys:
        for pat in _LAYER_KEY_PATTERNS:
            m = pat.match(k)
            if m:
                prefix, idx = m.group(1), int(m.group(2))
                counts[prefix] = counts.get(prefix, 0) + 1
                max_idx[prefix] = max(max_idx.get(prefix, -1), idx)
                break
    if not counts:
        raise ValueError("Could not detect transformer layer list prefix in checkpoint state dict.")
    best_prefix = max(counts, key=lambda name: counts[name])
    return best_prefix, max_idx[best_prefix] + 1


def _get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    cur: Any = root
    for part in path.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _set_module_by_path(root: nn.Module, path: str, new_mod: nn.Module) -> None:
    if "." not in path:
        setattr(root, path, new_mod)
        return
    parent_path, attr = path.rsplit(".", 1)
    parent = _get_module_by_path(root, parent_path)
    if attr.isdigit():
        parent[int(attr)] = new_mod
    else:
        setattr(parent, attr, new_mod)


def _should_quantize_layer_weight(
    sub_key: str,
    tensor: torch.Tensor,
    *,
    protect_sensitive: bool = True,
) -> bool:
    if not sub_key.endswith(".weight") or tensor.dim() != 2:
        return False
    out_f, in_f = int(tensor.size(0)), int(tensor.size(1))
    if in_f % NVFP4_GROUP_SIZE != 0:
        return False
    leaf = sub_key[: -len(".weight")].rsplit(".", 1)[-1].lower()
    if "norm" in leaf:
        return False
    if protect_sensitive and (leaf == "gate" and out_f <= 64 or "router" in sub_key.lower()):
        return False
    if protect_sensitive and any(s in sub_key.lower() for s in _SENSITIVE_SUBSTRINGS):
        return False
    return True


def _pack_tensor_nvfp4(
    weight: torch.Tensor,
    *,
    hadamard: bool,
    compute_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.to(device=compute_device, dtype=torch.bfloat16)
    if hadamard:
        w = fwht_16(w)
    packed, scale = nvfp4_quantize(w)
    return packed.cpu().contiguous(), scale.cpu().contiguous()


def default_nvfp4_cache_dir(model_id_or_dir: str | Path, *, hadamard: bool = True) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(model_id_or_dir)).strip("_")
    suffix = "nvfp4_fwht16" if hadamard else "nvfp4"
    return Path.home() / ".cache" / "lucid_machines" / "nvfp4_cache" / f"{slug}_{suffix}"


def build_nvfp4_disk_cache(
    model_id_or_dir: str | Path,
    cache_dir: str | Path | None = None,
    *,
    hadamard: bool = True,
    protect_sensitive: bool = True,
    quantize_lm_head: bool = False,
    compute_device: str | torch.device | None = None,
    force: bool = False,
    local_files_only: bool = False,
) -> Path:
    """Convert an HF safetensors model shard-by-shard into a compact NVFP4 disk cache.

    Never materializes the full unquantized checkpoint in RAM or VRAM, enabling conversion of
    14B-27B+ models on 16 GB RAM consumer machines.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    target_dir = (
        Path(cache_dir) if cache_dir is not None else default_nvfp4_cache_dir(model_id_or_dir, hadamard=hadamard)
    )
    manifest_path = target_dir / "manifest.json"
    if manifest_path.is_file() and not force:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                existing = json.load(f)
            if (
                bool(existing.get("hadamard")) == bool(hadamard)
                and bool(existing.get("quantize_lm_head", False)) == bool(quantize_lm_head)
                and existing.get("num_layers", 0) > 0
            ):
                return target_dir
        except Exception:
            pass

    model_dir = resolve_hf_model_dir(model_id_or_dir, local_files_only=local_files_only)
    weight_map, _shards = _discover_safetensors_map(model_dir)
    layer_prefix, num_layers = _detect_layer_layout(list(weight_map.keys()))

    layers_dir = target_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    # Copy config + tokenizer files cleanly so the cache directory is self-contained
    for fname in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    ):
        src_file = model_dir / fname
        if src_file.is_file():
            shutil.copy2(src_file, target_dir / fname)

    tied_embeddings = False
    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            raw_cfg = json.load(f)
        tied_embeddings = bool(raw_cfg.get("tie_word_embeddings", False))

    if compute_device is None:
        pack_dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        pack_dev = torch.device(compute_device)

    layer_keys: list[dict[str, Path]] = [{} for _ in range(num_layers)]
    root_keys: dict[str, Path] = {}
    prefix_dot = f"{layer_prefix}."
    for key, shard_path in weight_map.items():
        if key.startswith(prefix_dot):
            rest = key[len(prefix_dot) :]
            idx_str, sub_key = rest.split(".", 1)
            layer_idx = int(idx_str)
            layer_keys[layer_idx][sub_key] = shard_path
        else:
            root_keys[key] = shard_path

    raw_bytes = 0
    packed_bytes = 0
    layer_files: list[str] = []
    quantized_layer_modules: list[list[str]] = []

    open_handles: dict[Path, Any] = {}

    def _get_handle(shard_p: Path):
        h = open_handles.get(shard_p)
        if h is None:
            h = safe_open(str(shard_p), framework="pt", device="cpu")
            open_handles[shard_p] = h
        return h

    try:
        # 1. Pack root tensors
        root_tensors: dict[str, torch.Tensor] = {}
        root_quantized_modules: list[str] = []
        for rkey, rshard in root_keys.items():
            t = _get_handle(rshard).get_tensor(rkey)
            raw_bytes += t.numel() * t.element_size()
            if (
                quantize_lm_head
                and rkey == "lm_head.weight"
                and not tied_embeddings
                and t.dim() == 2
                and t.size(1) % NVFP4_GROUP_SIZE == 0
            ):
                p_u8, sc_bf16 = _pack_tensor_nvfp4(t, hadamard=hadamard, compute_device=pack_dev)
                root_tensors["lm_head.packed_weight"] = p_u8
                root_tensors["lm_head.scale"] = sc_bf16
                root_quantized_modules.append("lm_head")
                packed_bytes += p_u8.numel() * p_u8.element_size() + sc_bf16.numel() * sc_bf16.element_size()
            else:
                bf_t = t.to(torch.bfloat16).cpu().contiguous() if t.is_floating_point() else t.cpu().contiguous()
                root_tensors[rkey] = bf_t
                packed_bytes += bf_t.numel() * bf_t.element_size()
            del t

        root_rel = "root.safetensors"
        save_file(root_tensors, str(target_dir / root_rel))
        del root_tensors

        # 2. Pack each transformer layer independently (only 1 layer live in memory at any moment)
        for idx in range(num_layers):
            out_dict: dict[str, torch.Tensor] = {}
            q_mods: list[str] = []
            for sub_key, shard_p in layer_keys[idx].items():
                full_key = f"{prefix_dot}{idx}.{sub_key}"
                t = _get_handle(shard_p).get_tensor(full_key)
                raw_bytes += t.numel() * t.element_size()
                if _should_quantize_layer_weight(sub_key, t, protect_sensitive=protect_sensitive):
                    mod_sub = sub_key[: -len(".weight")]
                    p_u8, sc_bf16 = _pack_tensor_nvfp4(t, hadamard=hadamard, compute_device=pack_dev)
                    out_dict[f"{mod_sub}.packed_weight"] = p_u8
                    out_dict[f"{mod_sub}.scale"] = sc_bf16
                    q_mods.append(mod_sub)
                    packed_bytes += p_u8.numel() * p_u8.element_size() + sc_bf16.numel() * sc_bf16.element_size()
                else:
                    bf_t = t.to(torch.bfloat16).cpu().contiguous() if t.is_floating_point() else t.cpu().contiguous()
                    out_dict[sub_key] = bf_t
                    packed_bytes += bf_t.numel() * bf_t.element_size()
                del t

            rel_layer = f"layers/layer_{idx:04d}.safetensors"
            save_file(out_dict, str(target_dir / rel_layer))
            layer_files.append(rel_layer)
            quantized_layer_modules.append(sorted(set(q_mods)))
            del out_dict

        if pack_dev.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        for h in open_handles.values():
            with contextlib.suppress(Exception):
                h.__exit__(None, None, None)

    manifest = {
        "format": "lucid-nvfp4-disk-cache-v1",
        "source_model": str(model_id_or_dir),
        "layer_prefix": layer_prefix,
        "num_layers": num_layers,
        "hadamard": bool(hadamard),
        "protect_sensitive": bool(protect_sensitive),
        "quantize_lm_head": bool(quantize_lm_head),
        "tied_embeddings": tied_embeddings,
        "root_file": root_rel,
        "root_quantized_modules": root_quantized_modules,
        "layer_files": layer_files,
        "quantized_layer_modules": quantized_layer_modules,
        "raw_bytes": raw_bytes,
        "packed_bytes": packed_bytes,
        "compression_ratio": round(raw_bytes / max(1, packed_bytes), 3),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return target_dir


class DiskNVFP4StreamingEngine:
    """Streams frozen NVFP4 layer weights directly from NVMe into VRAM during forward & QLoRA backward."""

    def __init__(
        self,
        model: nn.Module,
        cache_dir: str | Path,
        manifest: dict[str, Any],
        device: torch.device,
        *,
        activation_checkpointing: bool = True,
        async_prefetch: bool = True,
        pin_memory: bool = False,
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest
        self.device = torch.device(device)
        self.activation_checkpointing = activation_checkpointing
        self.async_prefetch = async_prefetch
        self.pin_memory = pin_memory and self.device.type == "cuda"

        self.layer_prefix: str = manifest["layer_prefix"]
        self.layers: list[nn.Module] = list(_get_module_by_path(model, self.layer_prefix))
        self.layer_files: list[Path] = [self.cache_dir / rel for rel in manifest["layer_files"]]
        self.quantized_layer_modules: list[list[str]] = manifest["quantized_layer_modules"]
        self._streamed_tensor_targets: list[list[tuple[nn.Module, str, str, bool]]] = []
        self._hooks: list[Any] = []
        self.is_attached = False

        self.prefetch_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if (async_prefetch and self.device.type == "cuda") else None
        )
        self._io_pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if async_prefetch else None
        self._io_lock = threading.Lock()
        self._prefetched_cpu: dict[int, Future] = {}
        self._staged_gpu: dict[int, dict[str, torch.Tensor]] = {}
        self.telemetry = {
            "disk_layer_reads": 0,
            "h2d_bytes_transferred": 0,
            "prefetched_hits": 0,
        }

    def _build_layer_targets(self) -> None:
        """Precompute module attribute references for ultra-fast layer staging/release."""
        self._streamed_tensor_targets.clear()
        for idx, layer in enumerate(self.layers):
            layer_path = self.layer_files[idx]
            from safetensors import safe_open

            with safe_open(str(layer_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())

            targets: list[tuple[nn.Module, str, str, bool]] = []
            for tensor_key in keys:
                mod_sub, attr = tensor_key.rsplit(".", 1)
                mod = _get_module_by_path(layer, mod_sub)
                is_param = attr in mod._parameters
                targets.append((mod, attr, tensor_key, is_param))
            self._streamed_tensor_targets.append(targets)

    def _release_cpu_host_slot(self, layer_idx: int) -> None:
        self._prefetched_cpu.pop(layer_idx, None)

    def _load_layer_dict_cpu(self, layer_idx: int) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        data = load_file(str(self.layer_files[layer_idx]), device="cpu")
        if self.pin_memory:
            pinned = {}
            for k, v in data.items():
                try:
                    pinned[k] = v.pin_memory()
                except Exception:
                    pinned[k] = v
            return pinned
        return data

    def _schedule_prefetch(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= len(self.layers) or self._io_pool is None:
            return
        with self._io_lock:
            if layer_idx not in self._prefetched_cpu and layer_idx not in self._staged_gpu:
                self._prefetched_cpu[layer_idx] = self._io_pool.submit(self._load_layer_dict_cpu, layer_idx)

    def _stage_layer(self, layer_idx: int) -> None:
        if layer_idx in self._staged_gpu:
            return
        fut: Optional[Future] = None
        with self._io_lock:
            fut = self._prefetched_cpu.pop(layer_idx, None)
        if fut is not None:
            cpu_dict = fut.result()
            self.telemetry["prefetched_hits"] += 1
        else:
            cpu_dict = self._load_layer_dict_cpu(layer_idx)
        self.telemetry["disk_layer_reads"] += 1

        gpu_dict: dict[str, torch.Tensor] = {}
        bytes_tx = 0
        dev = self.device
        for mod, attr, tensor_key, is_param in self._streamed_tensor_targets[layer_idx]:
            src = cpu_dict[tensor_key]
            non_blk = bool(self.pin_memory and getattr(src, "is_pinned", lambda: False)())
            t_gpu = src.to(dev, non_blocking=non_blk) if dev.type == "cuda" else src
            bytes_tx += t_gpu.numel() * t_gpu.element_size()
            gpu_dict[tensor_key] = t_gpu
            if is_param:
                param = mod._parameters[attr]
                if param is None or param.device != dev or param.shape != t_gpu.shape:
                    mod._parameters[attr] = nn.Parameter(t_gpu, requires_grad=False)
                else:
                    param.data = t_gpu
            else:
                setattr(mod, attr, t_gpu)

        if self.pin_memory and dev.type == "cuda":
            torch.cuda.current_stream(dev).synchronize()
        self.telemetry["h2d_bytes_transferred"] += bytes_tx
        self._staged_gpu[layer_idx] = gpu_dict
        del cpu_dict

    def _release_layer(self, layer_idx: int) -> None:
        if layer_idx not in self._staged_gpu:
            return
        for mod, attr, _tensor_key, is_param in self._streamed_tensor_targets[layer_idx]:
            if is_param:
                p = mod._parameters.get(attr)
                if p is not None:
                    placeholder = _EMPTY_BF16 if p.dtype == torch.bfloat16 else torch.empty(0, dtype=p.dtype)
                    mod._parameters[attr] = nn.Parameter(placeholder, requires_grad=False)
            else:
                cur = getattr(mod, attr, None)
                if isinstance(cur, torch.Tensor):
                    placeholder = _EMPTY_U8 if cur.dtype == torch.uint8 else _EMPTY_BF16
                    setattr(mod, attr, placeholder)
        self._staged_gpu.pop(layer_idx, None)

    def attach(self) -> None:
        """Attach NVMe layer-streaming and optional checkpointing hooks."""
        if self.is_attached or not self.layers:
            return
        self._build_layer_targets()

        # Offload frozen layer placeholders off-GPU upfront; trainable LoRA params stay on device
        for idx in range(len(self.layers)):
            for mod, attr, _tensor_key, is_param in self._streamed_tensor_targets[idx]:
                if is_param:
                    p = mod._parameters.get(attr)
                    dt = p.dtype if p is not None else torch.bfloat16
                    mod._parameters[attr] = nn.Parameter(
                        _EMPTY_BF16 if dt == torch.bfloat16 else torch.empty(0, dtype=dt), requires_grad=False
                    )
                else:
                    cur = getattr(mod, attr, None)
                    dt = (
                        cur.dtype
                        if isinstance(cur, torch.Tensor)
                        else (torch.uint8 if attr == "packed_weight" else torch.bfloat16)
                    )
                    setattr(mod, attr, _EMPTY_U8 if dt == torch.uint8 else _EMPTY_BF16)

        for idx, layer in enumerate(self.layers):

            def make_fwd_pre(i: int, l_ref: nn.Module):
                def fwd_pre(module, args, kwargs):
                    in_bwd = getattr(l_ref, "_in_backward", False)
                    self._stage_layer(i)
                    if not in_bwd:
                        self._schedule_prefetch(i + 1)
                    dev = self.device

                    def _move_obj(obj):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(dev) if obj.device != dev else obj
                        if isinstance(obj, tuple):
                            return tuple(_move_obj(v) for v in obj)
                        if isinstance(obj, list):
                            return [_move_obj(v) for v in obj]
                        return obj

                    new_args = tuple(_move_obj(a) for a in args) if isinstance(args, tuple) else _move_obj(args)
                    if (
                        torch.is_grad_enabled()
                        and isinstance(new_args, tuple)
                        and new_args
                        and isinstance(new_args[0], torch.Tensor)
                        and new_args[0].is_floating_point()
                        and not new_args[0].requires_grad
                    ):
                        new_args = (new_args[0].detach().requires_grad_(True),) + new_args[1:]
                    new_kwargs = {k: _move_obj(v) for k, v in kwargs.items()} if kwargs else kwargs
                    return new_args, new_kwargs

                return fwd_pre

            def make_fwd_post(i: int, l_ref: nn.Module):
                def fwd_post(module, args, output):
                    if not getattr(l_ref, "_in_backward", False):
                        self._release_layer(i)
                    return output

                return fwd_post

            def make_bwd_pre(i: int, l_ref: nn.Module):
                def bwd_pre(module, grad_output):
                    l_ref._in_backward = True
                    self._stage_layer(i)
                    if i > 0:
                        self._schedule_prefetch(i - 1)

                return bwd_pre

            def make_bwd_post(i: int, l_ref: nn.Module):
                def bwd_post(module, grad_input, grad_output):
                    l_ref._in_backward = False
                    self._release_layer(i)

                return bwd_post

            self._hooks.append(layer.register_forward_pre_hook(make_fwd_pre(idx, layer), with_kwargs=True))
            try:
                self._hooks.append(layer.register_forward_hook(make_fwd_post(idx, layer), always_call=True))
            except TypeError:
                self._hooks.append(layer.register_forward_hook(make_fwd_post(idx, layer)))
            self._hooks.append(layer.register_full_backward_pre_hook(make_bwd_pre(idx, layer)))
            self._hooks.append(layer.register_full_backward_hook(make_bwd_post(idx, layer)))

            if self.activation_checkpointing:
                from lucid_machines.model.config import (
                    checkpoint_bwd_ctx,
                    checkpoint_fwd_ctx,
                    is_checkpoint_recomputing,
                )

                orig_fwd = layer.forward
                layer._orig_forward_for_disk_streaming = orig_fwd

                def make_ckpt_fwd(l_ref: nn.Module, orig_fn):
                    def ckpt_fwd(*f_args, **f_kwargs):
                        has_cache = bool(
                            f_kwargs.get("use_cache", False)
                            or f_kwargs.get("past_key_value") is not None
                            or f_kwargs.get("past_key_values") is not None
                        )
                        first_is_tensor = len(f_args) > 0 and isinstance(f_args[0], torch.Tensor)
                        if (
                            torch.is_grad_enabled()
                            and not has_cache
                            and first_is_tensor
                            and not getattr(l_ref, "_in_backward", False)
                            and not is_checkpoint_recomputing()
                        ):
                            x0 = f_args[0]
                            if x0.is_floating_point() and not x0.requires_grad:
                                f_args = (x0.detach().requires_grad_(True),) + f_args[1:]

                            def _ckpt_call(x_arg, *rest_args):
                                return orig_fn(x_arg, *rest_args, **f_kwargs)

                            return torch.utils.checkpoint.checkpoint(
                                _ckpt_call,
                                f_args[0],
                                *f_args[1:],
                                use_reentrant=False,
                                preserve_rng_state=True,
                                context_fn=lambda: (checkpoint_fwd_ctx(), checkpoint_bwd_ctx()),
                            )
                        return orig_fn(*f_args, **f_kwargs)

                    return ckpt_fwd

                layer.forward = make_ckpt_fwd(layer, orig_fwd)

        self.model._streaming_engine = self
        self.model._layer_streaming_active = True
        self.is_attached = True

    def detach(self) -> None:
        """Detach disk streaming hooks and shut down background prefetch pool."""
        for h in self._hooks:
            with contextlib.suppress(Exception):
                h.remove()
        self._hooks.clear()
        for layer in self.layers:
            if hasattr(layer, "_orig_forward_for_disk_streaming"):
                layer.forward = layer._orig_forward_for_disk_streaming
                delattr(layer, "_orig_forward_for_disk_streaming")
            layer._in_backward = False
        for i in list(self._staged_gpu.keys()):
            self._release_layer(i)
        with self._io_lock:
            self._prefetched_cpu.clear()
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=False)
            self._io_pool = None
        self.model._streaming_engine = None
        self.model._layer_streaming_active = False
        self.is_attached = False


@contextlib.contextmanager
def _init_empty_weights():
    """Initialize module parameters on ``meta`` without requiring ``accelerate``.

    Buffers remain on CPU so derived initialization buffers (such as rotary
    embedding ``inv_freq``) retain their real computed values.
    """
    old_register_parameter = nn.Module.register_parameter

    def _register_empty_parameter(module: nn.Module, name: str, param: nn.Parameter | None) -> None:
        old_register_parameter(module, name, param)
        registered = module._parameters.get(name)
        if registered is not None and registered.device.type != "meta":
            param_cls = type(registered)
            new_param = param_cls(registered.to("meta"), requires_grad=registered.requires_grad)
            if registered.__dict__:
                new_param.__dict__.update(registered.__dict__)
            module._parameters[name] = new_param

    nn.Module.register_parameter = _register_empty_parameter  # type: ignore[method-assign]
    try:
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter  # type: ignore[method-assign]


def load_disk_streamed_model(
    model_or_cache_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    hadamard: bool = True,
    protect_sensitive: bool = True,
    quantize_lm_head: bool = False,
    activation_checkpointing: bool = True,
    async_prefetch: bool = True,
    pin_memory: bool = False,
    local_files_only: bool = False,
) -> tuple[nn.Module, DiskNVFP4StreamingEngine, Path]:
    """Instantiate an HF CausalLM on meta, load root weights to device, and stream NVFP4 layers from NVMe."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    target_dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cand_cache = Path(model_or_cache_dir)
    if (cand_cache / "manifest.json").is_file():
        resolved_cache = cand_cache
    else:
        resolved_cache = build_nvfp4_disk_cache(
            model_or_cache_dir,
            cache_dir=cache_dir,
            hadamard=hadamard,
            protect_sensitive=protect_sensitive,
            quantize_lm_head=quantize_lm_head,
            local_files_only=local_files_only,
        )

    with open(resolved_cache / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    hf_cfg = AutoConfig.from_pretrained(str(resolved_cache))
    with _init_empty_weights():
        try:
            model = AutoModelForCausalLM.from_config(hf_cfg, dtype=torch.bfloat16)
        except TypeError:
            model = AutoModelForCausalLM.from_config(hf_cfg, torch_dtype=torch.bfloat16)

    # Swap quantized linear modules inside decoder layers into lightweight NVFP4Linear skeletons on meta
    layers: list[nn.Module] = list(_get_module_by_path(model, manifest["layer_prefix"]))
    use_hadamard = bool(manifest.get("hadamard", hadamard))
    for idx, layer in enumerate(layers):
        for mod_sub in manifest["quantized_layer_modules"][idx]:
            old_mod = _get_module_by_path(layer, mod_sub)
            in_f = int(getattr(old_mod, "in_features"))
            out_f = int(getattr(old_mod, "out_features"))
            has_bias = getattr(old_mod, "bias", None) is not None
            nv_mod = NVFP4Linear.empty_skeleton(in_f, out_f, bias=has_bias, dtype=torch.bfloat16, hadamard=use_hadamard)
            _set_module_by_path(layer, mod_sub, nv_mod)

    # Load root weights (embed_tokens, final norm, optional lm_head) directly onto target device
    root_tensors = load_file(str(resolved_cache / manifest["root_file"]), device="cpu")
    for mod_sub in manifest.get("root_quantized_modules", []):
        old_mod = _get_module_by_path(model, mod_sub)
        nv_root = NVFP4Linear.empty_skeleton(
            int(old_mod.in_features),
            int(old_mod.out_features),
            bias=getattr(old_mod, "bias", None) is not None,
            dtype=torch.bfloat16,
            hadamard=use_hadamard,
        )
        nv_root.packed_weight = root_tensors.pop(f"{mod_sub}.packed_weight").to(target_dev)
        nv_root.scale = root_tensors.pop(f"{mod_sub}.scale").to(target_dev)
        _set_module_by_path(model, mod_sub, nv_root)

    for key, t in root_tensors.items():
        mod_path, attr = key.rsplit(".", 1)
        mod = _get_module_by_path(model, mod_path)
        t_dev = t.to(target_dev)
        if attr in mod._parameters:
            mod._parameters[attr] = nn.Parameter(t_dev, requires_grad=False)
        else:
            setattr(mod, attr, t_dev)
    del root_tensors

    # Materialize any tied lm_head and non-persistent buffers (e.g. rotary_emb.inv_freq) on target device
    if manifest.get("tied_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()
    for mod in model.modules():
        if any(mod is l or any(mod is sm for sm in l.modules()) for l in layers):
            continue
        for b_name, b_val in list(mod.named_buffers(recurse=False)):
            if b_val is not None and b_val.device != target_dev:
                if b_val.device.type == "meta":
                    setattr(mod, b_name, torch.zeros(b_val.shape, dtype=b_val.dtype, device=target_dev))
                else:
                    setattr(mod, b_name, b_val.to(target_dev))

    for idx, layer in enumerate(layers):
        for sub_name, sub in layer.named_modules():
            for b_name, b_val in list(sub.named_buffers(recurse=False)):
                if b_val is not None and b_name not in ("packed_weight", "scale") and b_val.device != target_dev:
                    if b_val.device.type != "meta":
                        setattr(sub, b_name, b_val.to(target_dev))

    model.eval()
    engine = DiskNVFP4StreamingEngine(
        model,
        resolved_cache,
        manifest,
        device=target_dev,
        activation_checkpointing=activation_checkpointing,
        async_prefetch=async_prefetch,
        pin_memory=pin_memory,
    )
    engine.attach()
    return model, engine, resolved_cache
