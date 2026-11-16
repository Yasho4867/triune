"""Local PC Model Scanner & Architecture Metadata Inspector.

Discovers models stored across Ollama, Hugging Face Hub caches, vLLM caches,
Hermes/Nous Research homes, LM Studio libraries, GGUF files, NVFP4 disk caches,
and workspace checkpoints so every model on the machine can be inspected, loaded,
or transmuted into a Triune 3-tier architecture.
"""

from __future__ import annotations

import json
import os
import re
import struct
import time
from pathlib import Path
from typing import Any, Optional

_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".locks",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "$recycle.bin",
        "system volume information",
        "windows",
        "program files",
        "program files (x86)",
    }
)

_GGUF_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "IQ3_XXS",
    22: "IQ1_S",
    23: "IQ4_NL",
    24: "IQ3_S",
    25: "IQ2_S",
    26: "IQ4_XS",
    27: "I8",
    28: "I16",
    29: "I32",
    30: "I64",
    31: "F64",
    32: "IQ1_M",
    33: "BF16",
}


def _format_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "—"
    gb = size_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    mb = size_bytes / (1024**2)
    return f"{mb:.1f} MB"


def _dir_weights_bytes(model_dir: Path) -> int:
    total = 0
    try:
        for f in model_dir.iterdir():
            if f.is_file() and f.suffix.lower() in {".safetensors", ".bin", ".gguf", ".pt"}:
                total += f.stat().st_size
    except OSError:
        pass
    return total


def parse_gguf_header(gguf_path: Path, max_kv: int = 96) -> Optional[dict[str, Any]]:
    """Parse scalar architecture metadata from a GGUF binary header without loading weights."""
    try:
        with open(gguf_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            _version = struct.unpack("<I", f.read(4))[0]
            _n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def _read_str() -> str:
                ln = struct.unpack("<Q", f.read(8))[0]
                if ln > 16384:
                    f.seek(ln, 1)
                    return ""
                return f.read(ln).decode("utf-8", errors="replace")

            def _skip_str() -> None:
                ln = struct.unpack("<Q", f.read(8))[0]
                f.seek(ln, 1)

            def _read_val(vtype: int, key: str = "") -> Any:
                if vtype == 0:
                    return struct.unpack("<B", f.read(1))[0]
                if vtype == 1:
                    return struct.unpack("<b", f.read(1))[0]
                if vtype == 2:
                    return struct.unpack("<H", f.read(2))[0]
                if vtype == 3:
                    return struct.unpack("<h", f.read(2))[0]
                if vtype == 4:
                    return struct.unpack("<I", f.read(4))[0]
                if vtype == 5:
                    return struct.unpack("<i", f.read(4))[0]
                if vtype == 6:
                    return struct.unpack("<f", f.read(4))[0]
                if vtype == 7:
                    return bool(struct.unpack("<B", f.read(1))[0])
                if vtype == 8:
                    return _read_str()
                if vtype == 10:
                    return struct.unpack("<Q", f.read(8))[0]
                if vtype == 11:
                    return struct.unpack("<q", f.read(8))[0]
                if vtype == 12:
                    return struct.unpack("<d", f.read(8))[0]
                if vtype == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    clen = struct.unpack("<Q", f.read(8))[0]
                    if key == "tokenizer.ggml.tokens":
                        return clen
                    elem_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                    if et in elem_sizes:
                        f.seek(clen * elem_sizes[et], 1)
                    elif et == 8:
                        for _ in range(clen):
                            _skip_str()
                    return None
                return None

            meta: dict[str, Any] = {}
            for _ in range(min(int(n_kv), max_kv)):
                k = _read_str()
                vt = struct.unpack("<I", f.read(4))[0]
                val = _read_val(vt, key=k)
                if k == "tokenizer.ggml.tokens" and isinstance(val, int):
                    meta["vocab_size"] = val
                    # Tokens array usually follows architecture keys; stop before merges/scores.
                    if "general.architecture" in meta:
                        break
                elif not k.startswith("tokenizer.") and val is not None:
                    meta[k] = val

            arch = str(meta.get("general.architecture") or "").strip()
            layers = meta.get(f"{arch}.block_count") or meta.get("llama.block_count")
            hidden_dim = meta.get(f"{arch}.embedding_length") or meta.get("llama.embedding_length")
            heads = meta.get(f"{arch}.attention.head_count") or meta.get("llama.attention.head_count")
            ffn_dim = meta.get(f"{arch}.feed_forward_length") or meta.get("llama.feed_forward_length")
            vocab_size = meta.get("vocab_size") or meta.get(f"{arch}.vocab_size")
            file_type = meta.get("general.file_type")
            return {
                "architecture": arch or "gguf",
                "name": str(meta.get("general.name") or gguf_path.stem),
                "num_layers": int(layers) if layers is not None else None,
                "hidden_dim": int(hidden_dim) if hidden_dim is not None else None,
                "num_heads": int(heads) if heads is not None else None,
                "ffn_dim": int(ffn_dim) if ffn_dim is not None else None,
                "vocab_size": int(vocab_size) if vocab_size is not None else None,
                "quantization": _GGUF_FILE_TYPES.get(int(file_type), "GGUF") if file_type is not None else None,
            }
    except Exception:
        return None


_NON_CAUSAL_MODEL_PREFIXES = (
    "bert",
    "roberta",
    "albert",
    "electra",
    "deberta",
    "vit",
    "clip",
    "siglip",
    "depth",
    "dino",
    "sam",
    "detr",
    "wav2vec",
    "whisper",
)


def _is_hf_causal_config(cfg: dict[str, Any]) -> bool:
    """Return True if a Hugging Face config.json describes an autoregressive CausalLM."""
    arch_list = [str(a) for a in (cfg.get("architectures") or []) if isinstance(a, str)]
    if arch_list:
        return any(a.endswith(("ForCausalLM", "LMHeadModel", "CausalLM")) for a in arch_list)
    model_type_low = str(cfg.get("model_type") or "").lower()
    return not any(model_type_low.startswith(prefix) for prefix in _NON_CAUSAL_MODEL_PREFIXES)


def _parse_hf_config_dir(model_dir: Path) -> Optional[dict[str, Any]]:
    cfg_p = model_dir / "config.json"
    if not cfg_p.is_file():
        return None
    try:
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    n_l = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    h_d = cfg.get("hidden_size") or cfg.get("n_embd")
    heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    ffn = cfg.get("intermediate_size") or cfg.get("n_inner")
    vocab = cfg.get("vocab_size")
    has_w = bool(
        (model_dir / "model.safetensors").is_file()
        or (model_dir / "model.safetensors.index.json").is_file()
        or any(model_dir.glob("*.safetensors"))
        or any(model_dir.glob("*.bin"))
    )
    return {
        "model_type": str(cfg.get("model_type") or "hf"),
        "is_causal_lm": _is_hf_causal_config(cfg),
        "num_layers": int(n_l) if n_l is not None else None,
        "hidden_dim": int(h_d) if h_d is not None else None,
        "num_heads": int(heads) if heads is not None else None,
        "ffn_dim": int(ffn) if ffn is not None else None,
        "vocab_size": int(vocab) if vocab is not None else None,
        "has_weights": has_w,
        "size_bytes": _dir_weights_bytes(model_dir) if has_w else 0,
    }


def _build_hf_scan_entry(
    entry_id: str,
    name: str,
    path_str: str,
    *,
    source: str,
    kind: str,
    prefix: str,
    cfg_meta: dict[str, Any],
    obj: Optional[str] = None,
    spec_fallback: str = "HF Config",
    status_suffix: Optional[str] = None,
) -> dict[str, Any]:
    n_l, h_d = cfg_meta.get("num_layers"), cfg_meta.get("hidden_dim")
    is_causal = bool(cfg_meta.get("is_causal_lm", True))
    has_weights = bool(cfg_meta.get("has_weights"))
    spec = f"{n_l}L / {h_d}D" if (n_l and h_d) else spec_fallback
    extra_tag = f", {status_suffix}" if status_suffix else ""
    if not is_causal:
        extra_tag = f"{extra_tag}, non-CausalLM"
    return {
        "id": entry_id,
        "name": name,
        "path": path_str,
        "source": source,
        "object": obj or kind,
        "kind": kind,
        "model_type": cfg_meta.get("model_type"),
        "is_causal_lm": is_causal,
        "num_layers": n_l,
        "hidden_dim": h_d,
        "num_heads": cfg_meta.get("num_heads"),
        "ffn_dim": cfg_meta.get("ffn_dim"),
        "vocab_size": cfg_meta.get("vocab_size"),
        "size_formatted": _format_size(cfg_meta.get("size_bytes", 0)),
        "has_weights": has_weights,
        "can_load_direct": bool(has_weights and is_causal),
        "can_transmute": True,
        "description": f"{prefix}: {name} ({spec}{extra_tag})",
    }


def _collect_wsl_subpaths(rel_subpath: str) -> list[Path]:
    out: list[Path] = []
    if os.name != "nt":
        return out
    wsl_root = Path("\\\\wsl.localhost")
    if not wsl_root.exists():
        return out
    try:
        for distro in wsl_root.iterdir():
            home_base = distro / "home"
            if not home_base.is_dir():
                continue
            for user_dir in home_base.iterdir():
                cand = user_dir / rel_subpath
                if cand.is_dir():
                    out.append(cand)
    except OSError:
        pass
    return out


def _infer_model_family_source(name: str, default_source: str) -> str:
    low = name.lower()
    if "hermes" in low and default_source in ("huggingface", "hf_cache"):
        return "hermes"
    return default_source


def _read_hermes_config_refs(hermes_dir: Path) -> list[str]:
    refs: list[str] = []
    for cfg_name in ("config.json", "config.yaml", "config.yml", "settings.json"):
        cp = hermes_dir / cfg_name
        if not cp.is_file():
            continue
        try:
            text = cp.read_text(encoding="utf-8", errors="replace")
            if cfg_name.endswith(".json"):
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ("model", "model_name", "default_model", "model_path", "local_model"):
                        val = data.get(key)
                        if isinstance(val, str) and val.strip():
                            refs.append(val.strip())
            else:
                for m in re.finditer(
                    r"^\s*(?:model|model_name|model_path|default_model)\s*:\s*['\"]?([^\s'\"]+)", text, re.M
                ):
                    refs.append(m.group(1).strip())
        except Exception:
            continue
    return refs


def resolve_ollama_host(base_url: str | None = None) -> str:
    """Resolve local Ollama HTTP base URL from override or OLLAMA_HOST environment variable."""
    host = (base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def query_ollama_running_tags(base_url: str | None = None, timeout: float = 0.6) -> set[str]:
    """Query the local Ollama daemon (/api/tags) and return available model tags in lowercase."""
    import urllib.request

    url = f"{resolve_ollama_host(base_url)}/api/tags"
    req = urllib.request.Request(url, method="GET")
    tags: set[str] = set()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return tags
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        for item in payload.get("models") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip().lower()
            if name:
                tags.add(name)
                if name.endswith(":latest"):
                    tags.add(name[: -len(":latest")])
    except Exception:
        pass
    return tags


def stream_ollama_chat(
    tag: str,
    messages: list[Any],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    base_url: str | None = None,
    cancel_event: Any = None,
    on_text: Any = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Stream a multi-turn chat completion from the local Ollama daemon (/api/chat)."""
    import urllib.request

    clean_tag = str(tag or "").strip()
    if clean_tag.lower().startswith("ollama:"):
        clean_tag = clean_tag[len("ollama:") :].strip()
    if not clean_tag:
        raise RuntimeError("No Ollama model tag specified.")

    formatted_msgs = []
    for m in messages or []:
        role = m.role if hasattr(m, "role") else m.get("role", "user")
        content = m.content if hasattr(m, "content") else m.get("content", "")
        formatted_msgs.append({"role": str(role), "content": str(content)})

    url = f"{resolve_ollama_host(base_url)}/api/chat"
    payload = json.dumps(
        {
            "model": clean_tag,
            "messages": formatted_msgs,
            "stream": True,
            "options": {
                "num_predict": max(1, int(max_new_tokens)),
                "temperature": max(0.0, float(temperature)),
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    chunks: list[str] = []
    eval_count = 0
    eval_ns = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            if cancel_event is not None and cancel_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                pkt = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_obj = pkt.get("message") if isinstance(pkt, dict) else None
            delta = str(msg_obj.get("content") or "") if isinstance(msg_obj, dict) else ""
            if delta:
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                chunks.append(delta)
                eval_count += 1
                if on_text is not None:
                    on_text(delta)
            if pkt.get("done"):
                if pkt.get("eval_count"):
                    eval_count = int(pkt["eval_count"])
                if pkt.get("eval_duration"):
                    eval_ns = int(pkt["eval_duration"])
                break

    elapsed_s = max(1e-4, (eval_ns / 1e9) if eval_ns > 0 else (time.perf_counter() - t0))
    total_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    tps = round(eval_count / elapsed_s, 1) if eval_count > 0 else 0.0
    cancelled = bool(cancel_event is not None and cancel_event.is_set())
    return {
        "text": "".join(chunks),
        "route": "OLLAMA",
        "latency_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tps,
        "decode_tokens_per_sec": tps,
        "tokens_count": int(eval_count),
        "tokens_generated": int(eval_count),
        "weights_loaded": True,
        "checkpoint": clean_tag,
        "cache": None,
        "cancelled": cancelled,
        "status": "cancelled" if cancelled else "ok",
    }


def _is_tag_live_in_ollama(tag: str, live_tags: set[str]) -> bool:
    """Return True when a model tag (with optional 'ollama:' prefix) is live in local Ollama."""
    if not live_tags:
        return False
    clean = str(tag or "").strip().lower()
    if clean.startswith("ollama:"):
        clean = clean[len("ollama:") :].strip()
    if not clean:
        return False
    if clean in live_tags or f"{clean}:latest" in live_tags:
        return True
    if clean.endswith(":latest") and clean[: -len(":latest")] in live_tags:
        return True
    return False


def scan_pc_models(
    extra_paths: Optional[list[str]] = None,
    *,
    include_wsl: bool = True,
    max_depth: int = 4,
    installed_modules: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Scan local checkpoints, Ollama, Hugging Face, vLLM, Hermes, LM Studio, and custom paths."""
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    scanned_roots: list[str] = []
    running_ollama_tags = query_ollama_running_tags()

    def _add(entry: dict[str, Any]) -> None:
        eid = str(entry.get("id") or "")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            results.append(entry)

    # 1. Local workspace checkpoints, exports, NVFP4 disk caches, & HF directories
    for folder in ("checkpoints", "checkpoints_full", "caches", "models", "exports"):
        root = Path(folder)
        if not root.is_dir():
            continue
        scanned_roots.append(str(root.resolve()))
        try:
            items = sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        except OSError:
            continue
        for item in items:
            try:
                rel = str(item).replace("\\", "/")
                if item.is_file():
                    sfx = item.suffix.lower()
                    if sfx in {".pt", ".safetensors", ".bin", ".ckpt"}:
                        if "adapter" in item.stem.lower():
                            continue
                        sz = _format_size(item.stat().st_size)
                        _add(
                            {
                                "id": rel,
                                "name": item.name,
                                "path": rel,
                                "source": "checkpoint",
                                "object": "checkpoint",
                                "kind": "checkpoint",
                                "size_formatted": sz,
                                "has_weights": True,
                                "can_load_direct": True,
                                "can_transmute": True,
                                "description": f"Checkpoint: {item.name} ({sz})",
                            }
                        )
                    elif sfx == ".gguf":
                        sz = _format_size(item.stat().st_size)
                        meta = parse_gguf_header(item) or {}
                        n_l, h_d = meta.get("num_layers"), meta.get("hidden_dim")
                        spec = f"{n_l}L / {h_d}D" if (n_l and h_d) else "GGUF"
                        _add(
                            {
                                "id": rel,
                                "name": item.name,
                                "path": rel,
                                "source": "gguf",
                                "object": "gguf",
                                "kind": "gguf",
                                "num_layers": n_l,
                                "hidden_dim": h_d,
                                "num_heads": meta.get("num_heads"),
                                "ffn_dim": meta.get("ffn_dim"),
                                "vocab_size": meta.get("vocab_size"),
                                "quantization": meta.get("quantization", "GGUF"),
                                "size_formatted": sz,
                                "has_weights": True,
                                "can_load_direct": False,
                                "can_transmute": True,
                                "description": f"GGUF: {item.name} ({spec}, {sz})",
                            }
                        )
                elif item.is_dir():
                    manifest_p = item / "manifest.json"
                    if manifest_p.is_file():
                        m_data = {}
                        try:
                            m_data = json.loads(manifest_p.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                        n_l = m_data.get("num_layers")
                        cfg_meta = _parse_hf_config_dir(item) or {}
                        h_d = cfg_meta.get("hidden_dim")
                        meta_tag = f"{n_l}L" + (f" / {h_d}D" if h_d else "") + " NVFP4 Cache"
                        _add(
                            {
                                "id": rel,
                                "name": item.name,
                                "path": rel,
                                "source": "nvfp4_cache",
                                "object": "nvfp4_cache",
                                "kind": "nvfp4_cache",
                                "num_layers": n_l,
                                "hidden_dim": h_d,
                                "has_weights": True,
                                "can_load_direct": True,
                                "can_transmute": True,
                                "description": f"NVFP4 Cache: {item.name} ({meta_tag})",
                            }
                        )
                    elif (cfg_meta := _parse_hf_config_dir(item)) is not None:
                        _add(
                            _build_hf_scan_entry(
                                rel,
                                item.name,
                                rel,
                                source=_infer_model_family_source(item.name, "huggingface"),
                                kind="hf_local",
                                prefix="Local HF",
                                cfg_meta=cfg_meta,
                            )
                        )
            except OSError:
                continue

    # 2. Ollama local manifests + GGUF binary header inspection
    ollama_roots: list[Path] = []
    if os.environ.get("OLLAMA_MODELS"):
        ollama_roots.append(Path(os.environ["OLLAMA_MODELS"]))
    ollama_roots.append(Path.home() / ".ollama" / "models")
    if include_wsl:
        ollama_roots.extend(_collect_wsl_subpaths(".ollama/models"))

    seen_ollama_roots: set[str] = set()
    for ol_root in ollama_roots:
        r_str = str(ol_root)
        if not ol_root.is_dir() or r_str in seen_ollama_roots:
            continue
        seen_ollama_roots.add(r_str)
        scanned_roots.append(r_str)
        blobs_dir = ol_root / "blobs"
        for m_sub in ("manifests", "manifests-v2"):
            m_dir = ol_root / m_sub
            if not m_dir.is_dir():
                continue
            for mf in sorted(m_dir.rglob("*")):
                if not mf.is_file():
                    continue
                try:
                    manifest = json.loads(mf.read_text(encoding="utf-8"))
                    layers_list = manifest.get("layers")
                    if not isinstance(layers_list, list) or not layers_list:
                        continue
                    rel_parts = mf.relative_to(m_dir).parts
                    # Typical layout: registry.ollama.ai/library/<model>/<tag>
                    if len(rel_parts) >= 3 and rel_parts[1] == "library":
                        display_name = f"{rel_parts[2]}:{rel_parts[-1]}"
                    elif len(rel_parts) >= 2:
                        display_name = f"{'/'.join(rel_parts[1:-1])}:{rel_parts[-1]}"
                    else:
                        display_name = mf.name

                    model_layer = next(
                        (l for l in layers_list if str(l.get("mediaType", "")).endswith(".model")),
                        None,
                    )
                    if model_layer is None:
                        continue
                    blob_name = str(model_layer.get("digest", "")).replace(":", "-")
                    blob_path = blobs_dir / blob_name
                    total_bytes = sum(int(l.get("size", 0) or 0) for l in layers_list)
                    sz_str = _format_size(total_bytes)

                    cfg_meta_ol = {}
                    cfg_digest = (manifest.get("config") or {}).get("digest")
                    if isinstance(cfg_digest, str):
                        cfg_blob = blobs_dir / cfg_digest.replace(":", "-")
                        if cfg_blob.is_file():
                            try:
                                cfg_meta_ol = json.loads(cfg_blob.read_text(encoding="utf-8"))
                            except Exception:
                                pass

                    gguf_meta = parse_gguf_header(blob_path) if blob_path.is_file() else None
                    n_l = (gguf_meta or {}).get("num_layers")
                    h_d = (gguf_meta or {}).get("hidden_dim")
                    quant = (gguf_meta or {}).get("quantization") or cfg_meta_ol.get("file_type") or "GGUF"
                    spec = f"{n_l}L / {h_d}D" if (n_l and h_d) else (cfg_meta_ol.get("model_type") or "GGUF")
                    _add(
                        {
                            "id": f"ollama:{display_name}",
                            "name": display_name,
                            "path": str(blob_path).replace("\\", "/"),
                            "source": _infer_model_family_source(display_name, "ollama"),
                            "object": "ollama",
                            "kind": "ollama",
                            "num_layers": n_l,
                            "hidden_dim": h_d,
                            "num_heads": (gguf_meta or {}).get("num_heads"),
                            "ffn_dim": (gguf_meta or {}).get("ffn_dim"),
                            "vocab_size": (gguf_meta or {}).get("vocab_size"),
                            "quantization": quant,
                            "size_formatted": sz_str,
                            "has_weights": blob_path.is_file(),
                            "can_load_direct": False,
                            "ollama_running": _is_tag_live_in_ollama(display_name, running_ollama_tags),
                            "can_transmute": True,
                            "description": f"Ollama: {display_name} ({spec}, {quant}, {sz_str})",
                        }
                    )
                except Exception:
                    continue

    # 3. Hugging Face Hub Cache snapshots (plus Hermes/Nous cached repos)
    hf_hub_dirs: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        hf_hub_dirs.append(Path(os.environ["HF_HUB_CACHE"]))
    elif os.environ.get("HF_HOME"):
        hf_hub_dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    else:
        hf_hub_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    if include_wsl:
        hf_hub_dirs.extend(_collect_wsl_subpaths(".cache/huggingface/hub"))

    seen_hf_roots: set[str] = set()
    for hub_dir in hf_hub_dirs:
        if not hub_dir.is_dir() or str(hub_dir) in seen_hf_roots:
            continue
        seen_hf_roots.add(str(hub_dir))
        scanned_roots.append(str(hub_dir))
        try:
            model_dirs = sorted(hub_dir.glob("models--*"))
        except OSError:
            continue
        for mdir in model_dirs:
            snaps_dir = mdir / "snapshots"
            if not snaps_dir.is_dir():
                continue
            repo_id = mdir.name.removeprefix("models--").replace("--", "/", 1)
            try:
                snaps = sorted(
                    (s for s in snaps_dir.iterdir() if s.is_dir()), key=lambda x: x.stat().st_mtime, reverse=True
                )
            except OSError:
                continue
            for snap in snaps:
                cfg_meta = _parse_hf_config_dir(snap)
                if cfg_meta is None:
                    continue
                sz_bytes = cfg_meta.get("size_bytes", 0)
                w_tag = _format_size(sz_bytes) if cfg_meta["has_weights"] else "config cached"
                src = _infer_model_family_source(repo_id, "huggingface")
                _add(
                    _build_hf_scan_entry(
                        repo_id,
                        repo_id,
                        str(snap).replace("\\", "/"),
                        source=src,
                        kind="hf_cache",
                        prefix="Hermes" if src == "hermes" else "HF Cache",
                        cfg_meta=cfg_meta,
                        spec_fallback="cached config",
                        status_suffix=w_tag,
                    )
                )
                break

    # 4. vLLM cache / local vLLM serving model folders
    vllm_dirs: list[Path] = []
    if os.environ.get("VLLM_CACHE_ROOT"):
        vllm_dirs.append(Path(os.environ["VLLM_CACHE_ROOT"]))
    vllm_dirs.extend([Path.home() / ".cache" / "vllm", Path.home() / ".config" / "vllm"])
    if include_wsl:
        vllm_dirs.extend(_collect_wsl_subpaths(".cache/vllm"))
    for v_root in vllm_dirs:
        if not v_root.is_dir():
            continue
        scanned_roots.append(str(v_root))
        for cfg_file in v_root.rglob("config.json"):
            mdir = cfg_file.parent
            cfg_meta = _parse_hf_config_dir(mdir)
            if cfg_meta is None:
                continue
            rel = str(mdir).replace("\\", "/")
            _add(
                _build_hf_scan_entry(
                    rel,
                    mdir.name,
                    rel,
                    source="vllm",
                    kind="vllm",
                    prefix="vLLM",
                    cfg_meta=cfg_meta,
                    spec_fallback="vLLM Model",
                )
            )

    # 5. Hermes / Nous Research local directories & configs
    hermes_dirs: list[Path] = []
    if os.environ.get("HERMES_HOME"):
        hermes_dirs.append(Path(os.environ["HERMES_HOME"]))
    hermes_dirs.extend([Path.home() / ".hermes", Path.home() / ".cache" / "hermes"])
    if include_wsl:
        hermes_dirs.extend(_collect_wsl_subpaths(".hermes"))
        hermes_dirs.extend(_collect_wsl_subpaths(".cache/hermes"))
    for h_root in hermes_dirs:
        if not h_root.is_dir():
            continue
        scanned_roots.append(str(h_root))
        for cfg_file in h_root.rglob("config.json"):
            mdir = cfg_file.parent
            cfg_meta = _parse_hf_config_dir(mdir)
            if cfg_meta is None:
                continue
            rel = str(mdir).replace("\\", "/")
            _add(
                _build_hf_scan_entry(
                    rel,
                    mdir.name,
                    rel,
                    source="hermes",
                    kind="hermes",
                    prefix="Hermes",
                    cfg_meta=cfg_meta,
                    spec_fallback="Hermes Model",
                )
            )
        for gguf_f in h_root.rglob("*.gguf"):
            rel = str(gguf_f).replace("\\", "/")
            sz = _format_size(gguf_f.stat().st_size)
            meta = parse_gguf_header(gguf_f) or {}
            n_l, h_d = meta.get("num_layers"), meta.get("hidden_dim")
            spec = f"{n_l}L / {h_d}D" if (n_l and h_d) else "GGUF"
            _add(
                {
                    "id": rel,
                    "name": gguf_f.name,
                    "path": rel,
                    "source": "hermes",
                    "object": "hermes",
                    "kind": "gguf",
                    "num_layers": n_l,
                    "hidden_dim": h_d,
                    "num_heads": meta.get("num_heads"),
                    "ffn_dim": meta.get("ffn_dim"),
                    "vocab_size": meta.get("vocab_size"),
                    "quantization": meta.get("quantization", "GGUF"),
                    "size_formatted": sz,
                    "has_weights": True,
                    "can_load_direct": False,
                    "can_transmute": True,
                    "description": f"Hermes GGUF: {gguf_f.name} ({spec}, {sz})",
                }
            )
        for ref in _read_hermes_config_refs(h_root):
            ref_p = Path(ref)
            if ref_p.is_file() and ref_p.suffix.lower() == ".gguf":
                meta = parse_gguf_header(ref_p) or {}
                sz = _format_size(ref_p.stat().st_size)
                _add(
                    {
                        "id": str(ref_p).replace("\\", "/"),
                        "name": ref_p.name,
                        "path": str(ref_p).replace("\\", "/"),
                        "source": "hermes",
                        "object": "hermes",
                        "kind": "gguf",
                        "num_layers": meta.get("num_layers"),
                        "hidden_dim": meta.get("hidden_dim"),
                        "size_formatted": sz,
                        "has_weights": True,
                        "can_load_direct": False,
                        "can_transmute": True,
                        "description": f"Hermes Config Model: {ref_p.name} ({sz})",
                    }
                )

    # 6. LM Studio local model directories & standalone GGUFs
    lms_dirs: list[Path] = [
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "lm-studio" / "models",
    ]
    if include_wsl:
        lms_dirs.extend(_collect_wsl_subpaths(".lmstudio/models"))
        lms_dirs.extend(_collect_wsl_subpaths(".cache/lm-studio/models"))
    for lm_root in lms_dirs:
        if not lm_root.is_dir():
            continue
        scanned_roots.append(str(lm_root))
        for item in lm_root.rglob("*.gguf"):
            try:
                rel = str(item).replace("\\", "/")
                sz = _format_size(item.stat().st_size)
                meta = parse_gguf_header(item) or {}
                n_l, h_d = meta.get("num_layers"), meta.get("hidden_dim")
                spec = f"{n_l}L / {h_d}D" if (n_l and h_d) else "GGUF"
                _add(
                    {
                        "id": rel,
                        "name": item.name,
                        "path": rel,
                        "source": "lmstudio",
                        "object": "lmstudio",
                        "kind": "gguf",
                        "num_layers": n_l,
                        "hidden_dim": h_d,
                        "num_heads": meta.get("num_heads"),
                        "ffn_dim": meta.get("ffn_dim"),
                        "vocab_size": meta.get("vocab_size"),
                        "quantization": meta.get("quantization", "GGUF"),
                        "size_formatted": sz,
                        "has_weights": True,
                        "can_load_direct": False,
                        "can_transmute": True,
                        "description": f"LM Studio: {item.name} ({spec}, {sz})",
                    }
                )
            except OSError:
                continue
        for cfg_f in lm_root.rglob("config.json"):
            mdir = cfg_f.parent
            cfg_meta = _parse_hf_config_dir(mdir)
            if cfg_meta is None:
                continue
            rel = str(mdir).replace("\\", "/")
            _add(
                _build_hf_scan_entry(
                    rel,
                    mdir.name,
                    rel,
                    source="lmstudio",
                    obj="lmstudio",
                    kind="hf_local",
                    prefix="LM Studio HF",
                    cfg_meta=cfg_meta,
                    spec_fallback="HF Model",
                )
            )

    # 7. Installed / Cloned module repositories with model weights
    if installed_modules:
        for mod in installed_modules:
            weights = (mod.get("artifacts") or {}).get("weights") or []
            inst_path = mod.get("installed_at")
            if mod.get("type") == "model" or weights:
                target = str(weights[0] if weights else (inst_path or mod.get("id", ""))).replace("\\", "/")
                if target:
                    _add(
                        {
                            "id": target,
                            "name": mod.get("name") or Path(target).name,
                            "path": target,
                            "source": "installed_model",
                            "object": "installed_model",
                            "kind": "installed_model",
                            "has_weights": True,
                            "can_load_direct": True,
                            "can_transmute": True,
                            "description": f"Installed Repo: {mod.get('name', target)}",
                        }
                    )

    # 8. User-supplied extra directories (bounded depth-first walk)
    for raw_dir in extra_paths or []:
        s = str(raw_dir or "").strip()
        if not s:
            continue
        root = Path(s).expanduser()
        if not root.is_dir():
            continue
        scanned_roots.append(str(root))
        base_depth = len(root.parts)
        for cur_root, dirnames, filenames in os.walk(root):
            cur_p = Path(cur_root)
            depth = len(cur_p.parts) - base_depth
            dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIR_NAMES and not d.startswith(".git")]
            if depth >= max_depth:
                dirnames.clear()

            rel_dir = str(cur_p).replace("\\", "/")
            if "manifest.json" in filenames:
                try:
                    m_data = json.loads((cur_p / "manifest.json").read_text(encoding="utf-8"))
                    cfg_meta = _parse_hf_config_dir(cur_p) or {}
                    n_l = m_data.get("num_layers") or cfg_meta.get("num_layers")
                    _add(
                        {
                            "id": rel_dir,
                            "name": cur_p.name,
                            "path": rel_dir,
                            "source": "nvfp4_cache",
                            "object": "nvfp4_cache",
                            "kind": "nvfp4_cache",
                            "num_layers": n_l,
                            "hidden_dim": cfg_meta.get("hidden_dim"),
                            "has_weights": True,
                            "can_load_direct": True,
                            "can_transmute": True,
                            "description": f"NVFP4 Cache: {cur_p.name} ({n_l or '?'}L)",
                        }
                    )
                    dirnames.clear()
                    continue
                except Exception:
                    pass

            if "config.json" in filenames and (cfg_meta := _parse_hf_config_dir(cur_p)) is not None:
                _add(
                    _build_hf_scan_entry(
                        rel_dir,
                        cur_p.name,
                        rel_dir,
                        source=_infer_model_family_source(cur_p.name, "hf_local"),
                        kind="hf_local",
                        prefix="Local Model",
                        cfg_meta=cfg_meta,
                    )
                )
                dirnames.clear()
                continue

            for fname in filenames:
                fp = cur_p / fname
                sfx = fp.suffix.lower()
                rel_fp = str(fp).replace("\\", "/")
                if sfx == ".gguf":
                    sz = _format_size(fp.stat().st_size)
                    meta = parse_gguf_header(fp) or {}
                    n_l, h_d = meta.get("num_layers"), meta.get("hidden_dim")
                    spec = f"{n_l}L / {h_d}D" if (n_l and h_d) else "GGUF"
                    _add(
                        {
                            "id": rel_fp,
                            "name": fname,
                            "path": rel_fp,
                            "source": "gguf",
                            "object": "gguf",
                            "kind": "gguf",
                            "num_layers": n_l,
                            "hidden_dim": h_d,
                            "num_heads": meta.get("num_heads"),
                            "ffn_dim": meta.get("ffn_dim"),
                            "vocab_size": meta.get("vocab_size"),
                            "quantization": meta.get("quantization", "GGUF"),
                            "size_formatted": sz,
                            "has_weights": True,
                            "can_load_direct": False,
                            "can_transmute": True,
                            "description": f"GGUF: {fname} ({spec}, {sz})",
                        }
                    )
                elif sfx in {".pt", ".ckpt"} and "adapter" not in fp.stem.lower():
                    sz = _format_size(fp.stat().st_size)
                    _add(
                        {
                            "id": rel_fp,
                            "name": fname,
                            "path": rel_fp,
                            "source": "checkpoint",
                            "object": "checkpoint",
                            "kind": "checkpoint",
                            "size_formatted": sz,
                            "has_weights": True,
                            "can_load_direct": True,
                            "can_transmute": True,
                            "description": f"Checkpoint: {fname} ({sz})",
                        }
                    )

    counts_by_source: dict[str, int] = {}
    for m in results:
        src = str(m.get("source") or "other")
        counts_by_source[src] = counts_by_source.get(src, 0) + 1

    return {
        "models": results,
        "counts_by_source": counts_by_source,
        "scanned_roots": scanned_roots,
        "scan_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def inspect_model_spec_for_upgrade(
    raw_source: str,
    *,
    checkpoint_arch_fn: Optional[Any] = None,
    fallback_spec: Optional[dict[str, Any]] = None,
) -> Optional[tuple[int, int, int, int, int, str]]:
    """Inspect any saved checkpoint, Ollama model, GGUF binary, NVFP4 cache, HF dir, or HF repo ID."""
    raw_str = str(raw_source or "").strip()
    if not raw_str:
        return None

    # 1. Resolve Ollama model identifiers ("ollama:gemma3:4b") to their backing GGUF blob
    if raw_str.lower().startswith("ollama:"):
        target_id = raw_str.lower()
        scanned = scan_pc_models(include_wsl=True)
        for m in scanned["models"]:
            if str(m.get("id", "")).lower() == target_id or str(m.get("name", "")).lower() == target_id.removeprefix(
                "ollama:"
            ):
                if m.get("num_layers") and m.get("hidden_dim"):
                    layers = int(m["num_layers"])
                    h_dim = int(m["hidden_dim"])
                    heads = int(m.get("num_heads") or max(1, h_dim // 64))
                    ffn = int(m.get("ffn_dim") or h_dim * 4)
                    vocab = int(m.get("vocab_size") or 32000)
                    quant = m.get("quantization") or "GGUF"
                    return layers, h_dim, heads, ffn, vocab, f"Ollama ({m['name']} · {quant})"
                raw_str = str(m.get("path") or raw_str)
                break

    p = Path(raw_str)
    # 2. Standalone GGUF file
    if p.is_file() and p.suffix.lower() == ".gguf":
        meta = parse_gguf_header(p)
        if meta and (meta.get("num_layers") or meta.get("hidden_dim")):
            layers = int(meta.get("num_layers") or 24)
            h_dim = int(meta.get("hidden_dim") or 1024)
            heads = int(meta.get("num_heads") or max(1, h_dim // 64))
            ffn = int(meta.get("ffn_dim") or h_dim * 4)
            vocab = int(meta.get("vocab_size") or 32000)
            arch = str(meta.get("architecture") or "GGUF").upper()
            return layers, h_dim, heads, ffn, vocab, f"{arch} GGUF ({p.name})"

    # 3. PyTorch / SafeTensors checkpoint file
    if p.is_file() and p.suffix.lower() in {".pt", ".bin", ".ckpt", ".safetensors"}:
        try:
            if p.suffix.lower() == ".safetensors":
                from safetensors import safe_open

                with safe_open(str(p), framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    layer_nums = [
                        int(m.group(1))
                        for k in keys
                        if (m := re.match(r"(?:model\.layers|transformer\.h|layers)\.(\d+)\.", str(k)))
                    ]
                    layers = (max(layer_nums) + 1) if layer_nums else 8
                    emb_key = next((k for k in keys if "embed" in k and len(f.get_slice(k).get_shape()) == 2), None)
                    if emb_key:
                        emb_shape = f.get_slice(emb_key).get_shape()
                        vocab_size, hidden_dim = int(emb_shape[0]), int(emb_shape[1])
                    else:
                        vocab_size, hidden_dim = 32000, 384
                    ffn_key = next(
                        (k for k in keys if any(t in k for t in ("gate_proj", "up_proj", "fc1", "experts.0.0"))),
                        None,
                    )
                    ffn_dim = int(max(f.get_slice(ffn_key).get_shape())) if ffn_key else hidden_dim * 4
                    heads = max(1, hidden_dim // 64)
                    return layers, hidden_dim, heads, ffn_dim, vocab_size, f"Saved SafeTensors ({p.name})"

            import torch

            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and checkpoint_arch_fn is not None:
                _, arch = checkpoint_arch_fn(ckpt, fallback_spec or {})
                layers = int(arch.get("num_layers") or 8)
                hidden_dim = int(arch.get("hidden_dim") or 384)
                heads = int(arch.get("num_heads") or max(1, hidden_dim // 64))
                vocab_size = int(arch.get("vocab_size") or 32000)
                ffn_dim = int(
                    (ckpt.get("config") or {}).get("ffn_dim", hidden_dim * 4)
                    if isinstance(ckpt.get("config"), dict)
                    else hidden_dim * 4
                )
                return layers, hidden_dim, heads, ffn_dim, vocab_size, f"Saved Checkpoint ({p.name})"
        except Exception:
            return None

    # 4. Local HF / vLLM / Hermes / NVFP4 Cache directory or HF Hub repo ID
    cfg_data: Optional[dict[str, Any]] = None
    label_name = p.name if p.is_dir() else raw_str
    if p.is_dir():
        cfg_data = _parse_hf_config_dir(p)
        if cfg_data is not None:
            layers = int(cfg_data.get("num_layers") or 24)
            hidden_dim = int(cfg_data.get("hidden_dim") or 1024)
            heads = int(cfg_data.get("num_heads") or max(1, hidden_dim // 64))
            ffn_dim = int(cfg_data.get("ffn_dim") or hidden_dim * 4)
            vocab_size = int(cfg_data.get("vocab_size") or 32000)
            mtype = str(cfg_data.get("model_type") or "hf").upper()
            return layers, hidden_dim, heads, ffn_dim, vocab_size, f"{mtype} ({label_name})"
    elif "/" in raw_str and not raw_str.endswith((".pt", ".bin", ".safetensors", ".ckpt")):
        try:
            from transformers import AutoConfig

            try:
                cfg_obj = AutoConfig.from_pretrained(raw_str, local_files_only=True)
            except Exception:
                cfg_obj = AutoConfig.from_pretrained(raw_str)
            cfg_dict = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else vars(cfg_obj)
            if isinstance(cfg_dict, dict) and (
                cfg_dict.get("num_hidden_layers")
                or cfg_dict.get("n_layer")
                or cfg_dict.get("hidden_size")
                or cfg_dict.get("n_embd")
            ):
                layers = int(cfg_dict.get("num_hidden_layers") or cfg_dict.get("n_layer") or 24)
                hidden_dim = int(cfg_dict.get("hidden_size") or cfg_dict.get("n_embd") or 1024)
                heads = int(cfg_dict.get("num_attention_heads") or cfg_dict.get("n_head") or max(1, hidden_dim // 64))
                ffn_dim = int(cfg_dict.get("intermediate_size") or cfg_dict.get("n_inner") or hidden_dim * 4)
                vocab_size = int(cfg_dict.get("vocab_size") or 32000)
                mtype = str(cfg_dict.get("model_type") or "hf").upper()
                return layers, hidden_dim, heads, ffn_dim, vocab_size, f"{mtype} ({label_name})"
        except Exception:
            return None

    return None
