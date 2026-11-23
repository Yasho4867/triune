"""Pure-Python / NumPy / PyTorch GGUF v3 reader and Ollama blob weight loader.

Reads local ``.gguf`` archives and Ollama model manifests/blobs directly on disk
without requiring Hugging Face downloads, gated repo auth tokens, or optional
third-party ``gguf`` / ``llama-cpp-python`` packages. Supports vectorised
dequantization of F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, and Q6_K
blocks directly into Hugging Face CausalLM modules and LucidMachines' transmuter.
"""

from __future__ import annotations

import gc
import json
import math
import mmap
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_GGUF_TOKENIZER_CACHE: dict[tuple[str, int, int], tuple[Any, frozenset[int], int]] = {}
_GGUF_TOKENIZER_CACHE_MAX = 8

# GGML tensor type IDs
_GGML_F32 = 0
_GGML_F16 = 1
_GGML_Q4_0 = 2
_GGML_Q4_1 = 3
_GGML_Q5_0 = 6
_GGML_Q5_1 = 7
_GGML_Q8_0 = 8
_GGML_Q4_K = 12
_GGML_Q5_K = 13
_GGML_Q6_K = 14
_GGML_BF16 = 30

_GGUF_SCALAR_FMTS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<B", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}


def is_gguf_file(path: str | Path) -> bool:
    """Return True when ``path`` is an existing file beginning with the 4-byte GGUF magic header."""
    try:
        p = Path(path)
        if not p.is_file():
            return False
        with open(p, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def resolve_ollama_gguf_blob(ollama_ref: str) -> Path | None:
    """Resolve ``ollama:<name>:<tag>`` to the underlying GGUF model blob file on disk."""
    raw = str(ollama_ref or "").strip()
    if raw.lower().startswith("ollama:"):
        raw = raw[len("ollama:") :].strip()
    if not raw:
        return None

    # Allow direct blob path if already passed
    cand = Path(raw)
    if cand.is_file():
        return cand

    if ":" in raw:
        repo_part, tag_part = raw.rsplit(":", 1)
    else:
        repo_part, tag_part = raw, "latest"

    ollama_models_root = Path(os.environ.get("OLLAMA_MODELS") or (Path.home() / ".ollama" / "models"))
    manifests_base = ollama_models_root / "manifests"
    blobs_dir = ollama_models_root / "blobs"
    if not manifests_base.is_dir() or not blobs_dir.is_dir():
        return None

    candidate_paths: list[Path] = []
    if "/" in repo_part:
        candidate_paths.append(manifests_base / repo_part / tag_part)
        candidate_paths.append(manifests_base / "registry.ollama.ai" / repo_part / tag_part)
    else:
        candidate_paths.append(manifests_base / "registry.ollama.ai" / "library" / repo_part / tag_part)

    for mp in candidate_paths:
        if not mp.is_file():
            continue
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for layer in manifest.get("layers") or []:
            media = str(layer.get("mediaType") or "")
            if "image.model" in media or media.endswith(".model"):
                digest = str(layer.get("digest") or "").replace(":", "-")
                bp = blobs_dir / digest
                if bp.is_file():
                    return bp
    return None


def read_gguf_directory(gguf_path: Path, *, load_tokenizer_arrays: bool = True) -> dict[str, Any]:
    """Parse GGUF v2/v3 header, metadata (optionally tokenizer arrays), and tensor index."""
    gguf_path = Path(gguf_path)
    with open(gguf_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a valid GGUF archive: {gguf_path}")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version} in {gguf_path}")

        def _read_str() -> str:
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln).decode("utf-8", errors="replace")

        def _skip_str() -> None:
            ln = struct.unpack("<Q", f.read(8))[0]
            f.seek(ln, 1)

        def _read_scalar(vtype: int) -> Any:
            fmt, size = _GGUF_SCALAR_FMTS[vtype]
            val = struct.unpack(fmt, f.read(size))[0]
            return bool(val) if vtype == 7 else val

        def _read_value(vtype: int, key: str) -> Any:
            if vtype in _GGUF_SCALAR_FMTS:
                return _read_scalar(vtype)
            if vtype == 8:
                if load_tokenizer_arrays or not key.startswith("tokenizer.") or key == "tokenizer.chat_template":
                    return _read_str()
                _skip_str()
                return None
            if vtype == 9:
                et, clen = struct.unpack("<IQ", f.read(12))
                keep = load_tokenizer_arrays and key in {
                    "tokenizer.ggml.tokens",
                    "tokenizer.ggml.merges",
                    "tokenizer.ggml.scores",
                    "tokenizer.ggml.token_type",
                    "tokenizer.ggml.eos_token_ids",
                }
                if et in _GGUF_SCALAR_FMTS:
                    fmt, sz = _GGUF_SCALAR_FMTS[et]
                    if keep:
                        raw_bytes = f.read(clen * sz)
                        np_dtype = {
                            0: "<u1",
                            1: "<i1",
                            2: "<u2",
                            3: "<i2",
                            4: "<u4",
                            5: "<i4",
                            6: "<f4",
                            7: "<?",
                            10: "<u8",
                            11: "<i8",
                            12: "<f8",
                        }[et]
                        return np.frombuffer(raw_bytes, dtype=np.dtype(np_dtype)).tolist()
                    f.seek(clen * sz, 1)
                    return None
                if et == 8:
                    if keep:
                        return [_read_str() for _ in range(clen)]
                    for _ in range(clen):
                        _skip_str()
                    return None
                raise ValueError(f"Unsupported GGUF array element type {et} for key '{key}'")
            raise ValueError(f"Unsupported GGUF value type {vtype} for key '{key}'")

        meta: dict[str, Any] = {}
        for _ in range(int(n_kv)):
            k = _read_str()
            vt = struct.unpack("<I", f.read(4))[0]
            val = _read_value(vt, k)
            if val is not None:
                meta[k] = val

        tensors: dict[str, dict[str, Any]] = {}
        for _ in range(int(n_tensors)):
            tname = _read_str()
            ndim = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors[tname] = {"dims": tuple(int(d) for d in dims), "ggml_type": int(ttype), "offset": int(offset)}

        pos = f.tell()
        align = int(meta.get("general.alignment", 32) or 32)
        data_start = ((pos + align - 1) // align) * align

    return {"meta": meta, "tensors": tensors, "data_start": data_start, "path": gguf_path}


def _get_scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized unpack of Q4_K / Q5_K 6-bit scales and mins across all blocks [N_blk, 12]."""
    sc = np.empty((scales.shape[0], 8), dtype=np.float32)
    mn = np.empty((scales.shape[0], 8), dtype=np.float32)
    sc[:, 0:4] = scales[:, 0:4] & 63
    mn[:, 0:4] = scales[:, 4:8] & 63
    sc[:, 4:8] = (scales[:, 8:12] & 0x0F) | ((scales[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)
    return sc, mn


def _dequantize_q4_0(raw: np.ndarray, numel: int) -> np.ndarray:
    nb = numel // 32
    blocks = np.frombuffer(raw[: nb * 18], dtype=np.uint8).reshape(nb, 18)
    d = blocks[:, :2].view(np.float16).astype(np.float32)
    qs = blocks[:, 2:18]
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, :16] = (qs & 0x0F).astype(np.float32) - 8.0
    out[:, 16:] = (qs >> 4).astype(np.float32) - 8.0
    return (out * d).reshape(numel)


def _dequantize_q4_1(raw: np.ndarray, numel: int) -> np.ndarray:
    nb = numel // 32
    blocks = np.frombuffer(raw[: nb * 20], dtype=np.uint8).reshape(nb, 20)
    d = blocks[:, :2].view(np.float16).astype(np.float32)
    m = blocks[:, 2:4].view(np.float16).astype(np.float32)
    qs = blocks[:, 4:20]
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, :16] = (qs & 0x0F).astype(np.float32)
    out[:, 16:] = (qs >> 4).astype(np.float32)
    return (out * d + m).reshape(numel)


def _dequantize_q5_0(raw: np.ndarray, numel: int) -> np.ndarray:
    nb = numel // 32
    blocks = np.frombuffer(raw[: nb * 22], dtype=np.uint8).reshape(nb, 22)
    d = blocks[:, :2].view(np.float16).astype(np.float32)
    qh = blocks[:, 2:6].view(np.uint32)
    qs = blocks[:, 6:22]
    shifts_lo = np.arange(16, dtype=np.uint32)[None, :]
    shifts_hi = np.arange(16, 32, dtype=np.uint32)[None, :]
    bit_lo = ((qh >> shifts_lo) & 1) << 4
    bit_hi = ((qh >> shifts_hi) & 1) << 4
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, :16] = ((qs & 0x0F).astype(np.uint32) | bit_lo).astype(np.float32) - 16.0
    out[:, 16:] = ((qs >> 4).astype(np.uint32) | bit_hi).astype(np.float32) - 16.0
    return (out * d).reshape(numel)


def _dequantize_q8_0(raw: np.ndarray, numel: int) -> np.ndarray:
    nb = numel // 32
    blocks = np.frombuffer(raw[: nb * 34], dtype=np.uint8).reshape(nb, 34)
    d = blocks[:, :2].view(np.float16).astype(np.float32)
    qs = blocks[:, 2:34].view(np.int8).astype(np.float32)
    return (qs * d).reshape(numel)


def _dequantize_q4_k(raw: bytes | bytearray | memoryview, numel: int) -> np.ndarray:
    nb = numel // 256
    blocks = np.frombuffer(raw[: nb * 144], dtype=np.uint8).reshape(nb, 144)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32)
    sc, mn = _get_scale_min_k4(blocks[:, 4:16])
    d_sc = (d * sc).reshape(nb, 4, 2, 1)
    m_mn = (dmin * mn).reshape(nb, 4, 2, 1)
    qs = blocks[:, 16:144].reshape(nb, 4, 32)
    out = np.empty((nb, 4, 2, 32), dtype=np.float32)
    np.multiply((qs & 0x0F).astype(np.float32), d_sc[:, :, 0, :], out=out[:, :, 0, :])
    out[:, :, 0, :] -= m_mn[:, :, 0, :]
    np.multiply((qs >> 4).astype(np.float32), d_sc[:, :, 1, :], out=out[:, :, 1, :])
    out[:, :, 1, :] -= m_mn[:, :, 1, :]
    return out.reshape(numel)


def _dequantize_q5_k(raw: bytes | bytearray | memoryview, numel: int) -> np.ndarray:
    nb = numel // 256
    blocks = np.frombuffer(raw[: nb * 176], dtype=np.uint8).reshape(nb, 176)
    d = blocks[:, 0:2].view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].view(np.float16).astype(np.float32)
    sc, mn = _get_scale_min_k4(blocks[:, 4:16])
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176].reshape(nb, 4, 32)
    bits = ((qh[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1) << 4
    q_pairs = np.stack([qs & 0x0F, qs >> 4], axis=2).reshape(nb, 8, 32) | bits
    out = q_pairs.astype(np.float32) * (d * sc)[:, :, None] - (dmin * mn)[:, :, None]
    return out.reshape(numel)


def _dequantize_q6_k(raw: bytes | bytearray | memoryview, numel: int) -> np.ndarray:
    nb = numel // 256
    blocks = np.frombuffer(raw[: nb * 210], dtype=np.uint8).reshape(nb, 210)
    ql = blocks[:, 0:128].reshape(nb, 2, 2, 32)
    qh = blocks[:, 128:192].reshape(nb, 2, 32)
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32).reshape(nb, 2, 4, 2, 1)
    d = blocks[:, 208:210].view(np.float16).astype(np.float32).reshape(nb, 1, 1, 1, 1)
    q1 = ((ql[:, :, 0] & 0x0F) | (((qh >> 0) & 3) << 4)).astype(np.int16) - 32
    q2 = ((ql[:, :, 1] & 0x0F) | (((qh >> 2) & 3) << 4)).astype(np.int16) - 32
    q3 = ((ql[:, :, 0] >> 4) | (((qh >> 4) & 3) << 4)).astype(np.int16) - 32
    q4 = ((ql[:, :, 1] >> 4) | (((qh >> 6) & 3) << 4)).astype(np.int16) - 32
    q_all = np.stack([q1, q2, q3, q4], axis=2).reshape(nb, 2, 4, 2, 16).astype(np.float32)
    return (q_all * (d * scales)).reshape(numel)


def _unpack_scale_min_k4_torch(scales_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nb = scales_u8.size(0)
    s = scales_u8.to(torch.int32)
    sc = torch.empty((nb, 8), dtype=torch.float32, device=scales_u8.device)
    mn = torch.empty((nb, 8), dtype=torch.float32, device=scales_u8.device)
    sc[:, 0:4] = s[:, 0:4] & 63
    mn[:, 0:4] = s[:, 4:8] & 63
    sc[:, 4:8] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, mn


def _dequantize_ggml_torch_chunk(b: torch.Tensor, ggml_type: int, target_dtype: torch.dtype) -> torch.Tensor:
    """Vectorized GPU/PyTorch block dequantizer matching the NumPy CPU paths bit-for-bit."""
    nb = b.size(0)
    dev = b.device
    if ggml_type == _GGML_Q4_0:
        d = b[:, :2].view(torch.float16).float()
        qs = b[:, 2:18]
        q = torch.cat([qs & 0x0F, qs >> 4], dim=1).float() - 8.0
        out = q * d
    elif ggml_type == _GGML_Q4_1:
        d = b[:, :2].view(torch.float16).float()
        m = b[:, 2:4].view(torch.float16).float()
        qs = b[:, 4:20]
        q = torch.cat([qs & 0x0F, qs >> 4], dim=1).float()
        out = q * d + m
    elif ggml_type == _GGML_Q5_0:
        d = b[:, :2].view(torch.float16).float()
        qb = b[:, 2:6].to(torch.int32)
        qh = qb[:, 0:1] | (qb[:, 1:2] << 8) | (qb[:, 2:3] << 16) | (qb[:, 3:4] << 24)
        qs = b[:, 6:22]
        bits = (qh >> torch.arange(32, device=dev, dtype=torch.int32).unsqueeze(0)) & 1
        q_lo = (qs & 0x0F).to(torch.int32) | (bits[:, :16] << 4)
        q_hi = (qs >> 4).to(torch.int32) | (bits[:, 16:] << 4)
        out = (torch.cat([q_lo, q_hi], dim=1).float() - 16.0) * d
    elif ggml_type == _GGML_Q8_0:
        d = b[:, :2].view(torch.float16).float()
        qs = b[:, 2:34].view(torch.int8).float()
        out = qs * d
    elif ggml_type == _GGML_Q4_K:
        dm = b[:, :4].view(torch.float16).float()
        sc, mn = _unpack_scale_min_k4_torch(b[:, 4:16])
        qs = b[:, 16:144].reshape(nb, 4, 32)
        q_pairs = torch.stack([qs & 0x0F, qs >> 4], dim=2).reshape(nb, 8, 32).float()
        out = q_pairs * (dm[:, 0:1] * sc).unsqueeze(-1) - (dm[:, 1:2] * mn).unsqueeze(-1)
    elif ggml_type == _GGML_Q5_K:
        dm = b[:, :4].view(torch.float16).float()
        sc, mn = _unpack_scale_min_k4_torch(b[:, 4:16])
        qh = b[:, 16:48]
        qs = b[:, 48:176].reshape(nb, 4, 32)
        bits = ((qh.unsqueeze(1) >> torch.arange(8, device=dev, dtype=torch.int32).view(1, 8, 1)) & 1) << 4
        q_pairs = torch.stack([qs & 0x0F, qs >> 4], dim=2).reshape(nb, 8, 32).to(torch.int32) | bits
        out = q_pairs.float() * (dm[:, 0:1] * sc).unsqueeze(-1) - (dm[:, 1:2] * mn).unsqueeze(-1)
    elif ggml_type == _GGML_Q6_K:
        ql = b[:, 0:128].reshape(nb, 2, 2, 32)
        qh = b[:, 128:192].reshape(nb, 2, 32)
        scales = b[:, 192:208].view(torch.int8).float().reshape(nb, 2, 4, 2, 1)
        d = b[:, 208:210].view(torch.float16).float().reshape(nb, 1, 1, 1, 1)
        q1 = ((ql[:, :, 0] & 0x0F) | (((qh >> 0) & 3) << 4)).to(torch.int16) - 32
        q2 = ((ql[:, :, 1] & 0x0F) | (((qh >> 2) & 3) << 4)).to(torch.int16) - 32
        q3 = ((ql[:, :, 0] >> 4) | (((qh >> 4) & 3) << 4)).to(torch.int16) - 32
        q4 = ((ql[:, :, 1] >> 4) | (((qh >> 6) & 3) << 4)).to(torch.int16) - 32
        out = torch.stack([q1, q2, q3, q4], dim=2).reshape(nb, 2, 4, 2, 16).float() * (d * scales)
    else:
        raise NotImplementedError(f"GGML quantization type {ggml_type} is not supported on Torch path.")
    return out.reshape(-1).to(dtype=target_dtype)


_BLOCK_META: dict[int, tuple[int, int]] = {
    _GGML_F32: (1, 4),
    _GGML_F16: (1, 2),
    _GGML_BF16: (1, 2),
    _GGML_Q4_0: (32, 18),
    _GGML_Q4_1: (32, 20),
    _GGML_Q5_0: (32, 22),
    _GGML_Q8_0: (32, 34),
    _GGML_Q4_K: (256, 144),
    _GGML_Q5_K: (256, 176),
    _GGML_Q6_K: (256, 210),
}


def dequantize_ggml_tensor(
    raw_bytes: bytes | bytearray | memoryview,
    dims: tuple[int, ...],
    ggml_type: int,
    target_dtype: torch.dtype,
    *,
    compute_device: torch.device | None = None,
    return_on_device: bool = False,
) -> torch.Tensor:
    """Dequantize a GGUF tensor buffer and return a PyTorch tensor shaped ``tuple(reversed(dims))``."""
    numel = 1
    for d in dims:
        numel *= int(d)
    pt_shape = tuple(reversed(dims))
    dst_device = compute_device if (return_on_device and compute_device is not None) else None

    if ggml_type == _GGML_F32:
        buf_view = np.frombuffer(raw_bytes, dtype="<f4", count=numel)
        try:
            arr = np.array(buf_view, copy=True)
        finally:
            del buf_view
        t = torch.from_numpy(arr).reshape(pt_shape).to(dtype=target_dtype)
        return t.to(device=dst_device) if dst_device is not None else t
    if ggml_type == _GGML_F16:
        buf_view = np.frombuffer(raw_bytes, dtype="<f2", count=numel)
        try:
            arr = np.array(buf_view, copy=True)
        finally:
            del buf_view
        t = torch.from_numpy(arr).reshape(pt_shape).to(dtype=target_dtype)
        return t.to(device=dst_device) if dst_device is not None else t
    if ggml_type == _GGML_BF16:
        buf_view = np.frombuffer(raw_bytes, dtype="<u2", count=numel)
        try:
            arr = np.array(buf_view, copy=True)
        finally:
            del buf_view
        t = torch.from_numpy(arr).view(torch.bfloat16).reshape(pt_shape).to(dtype=target_dtype)
        return t.to(device=dst_device) if dst_device is not None else t

    if ggml_type not in _BLOCK_META:
        raise NotImplementedError(f"GGML quantization type {ggml_type} is not yet supported for direct dequantization.")

    blk_elems, blk_bytes = _BLOCK_META[ggml_type]
    if compute_device is not None and compute_device.type == "cuda" and numel >= 65536 and torch.cuda.is_available():
        try:
            nb_total = numel // blk_elems
            # Bound temporary CUDA scratch VRAM (<128 MB) via block chunking even for huge embeddings
            chunk_blocks = 131072 if blk_elems == 256 else 524288
            blocks_np = np.frombuffer(raw_bytes, dtype=np.uint8, count=nb_total * blk_bytes).reshape(
                nb_total, blk_bytes
            )
            try:
                if return_on_device and nb_total <= chunk_blocks:
                    chunk_u8 = torch.from_numpy(np.array(blocks_np, copy=True)).to(compute_device, non_blocking=False)
                    deq_chunk = _dequantize_ggml_torch_chunk(chunk_u8, ggml_type, target_dtype)
                    del chunk_u8
                    return deq_chunk.reshape(pt_shape)

                dst = torch.empty(numel, dtype=target_dtype, device=compute_device if return_on_device else "cpu")
                for b_start in range(0, nb_total, chunk_blocks):
                    b_end = min(b_start + chunk_blocks, nb_total)
                    chunk_u8 = torch.from_numpy(np.array(blocks_np[b_start:b_end], copy=True)).to(
                        compute_device, non_blocking=False
                    )
                    deq_chunk = _dequantize_ggml_torch_chunk(chunk_u8, ggml_type, target_dtype)
                    del chunk_u8
                    dst[b_start * blk_elems : b_end * blk_elems].copy_(
                        deq_chunk if return_on_device else deq_chunk.cpu()
                    )
                    del deq_chunk
                return dst.reshape(pt_shape)
            finally:
                del blocks_np
        except Exception:
            pass

    if ggml_type == _GGML_Q4_0:
        f32 = _dequantize_q4_0(raw_bytes, numel)
    elif ggml_type == _GGML_Q4_1:
        f32 = _dequantize_q4_1(raw_bytes, numel)
    elif ggml_type == _GGML_Q5_0:
        f32 = _dequantize_q5_0(raw_bytes, numel)
    elif ggml_type == _GGML_Q8_0:
        f32 = _dequantize_q8_0(raw_bytes, numel)
    elif ggml_type == _GGML_Q4_K:
        f32 = _dequantize_q4_k(raw_bytes, numel)
    elif ggml_type == _GGML_Q5_K:
        f32 = _dequantize_q5_k(raw_bytes, numel)
    elif ggml_type == _GGML_Q6_K:
        f32 = _dequantize_q6_k(raw_bytes, numel)
    else:
        raise NotImplementedError(f"GGML quantization type {ggml_type} is not yet supported for direct dequantization.")

    out = torch.from_numpy(f32).reshape(pt_shape).to(dtype=target_dtype)
    return out.to(device=dst_device) if dst_device is not None else out


def _read_gguf_tensor(
    f,
    data_start: int,
    info: dict[str, Any],
    target_dtype: torch.dtype,
    *,
    mm: mmap.mmap | None = None,
    compute_device: torch.device | None = None,
    return_on_device: bool = False,
) -> torch.Tensor:
    dims = info["dims"]
    ggml_type = info["ggml_type"]
    numel = 1
    for d in dims:
        numel *= int(d)
    if ggml_type not in _BLOCK_META:
        raise NotImplementedError(f"Unsupported GGML tensor type: {ggml_type}")
    blk_elems, blk_bytes = _BLOCK_META[ggml_type]
    n_bytes = (numel // blk_elems) * blk_bytes
    offset = data_start + info["offset"]
    if mm is not None:
        mv = memoryview(mm)
        mv_slice = None
        try:
            mv_slice = mv[offset : offset + n_bytes]
            return dequantize_ggml_tensor(
                mv_slice,
                dims,
                ggml_type,
                target_dtype,
                compute_device=compute_device,
                return_on_device=return_on_device,
            )
        finally:
            if mv_slice is not None:
                mv_slice.release()
            mv.release()
    f.seek(offset)
    raw = f.read(n_bytes)
    return dequantize_ggml_tensor(
        raw,
        dims,
        ggml_type,
        target_dtype,
        compute_device=compute_device,
        return_on_device=return_on_device,
    )


def build_hf_tokenizer_from_gguf(meta: dict[str, Any]):
    """Reconstruct a fast Hugging Face PreTrainedTokenizerFast directly from GGUF vocabulary & merges."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokens = meta.get("tokenizer.ggml.tokens") or []
    if not tokens:
        raise ValueError("GGUF metadata contains no tokenizer vocabulary.")

    token_types = meta.get("tokenizer.ggml.token_type") or []
    vocab = {str(tok): idx for idx, tok in enumerate(tokens)}
    bos_id = int(meta.get("tokenizer.ggml.bos_token_id", 1))
    eos_id = int(meta.get("tokenizer.ggml.eos_token_id", 2))
    unk_id = int(meta.get("tokenizer.ggml.unknown_token_id", 0))
    pad_id = int(meta.get("tokenizer.ggml.padding_token_id", eos_id))

    bos_tok = str(tokens[bos_id]) if 0 <= bos_id < len(tokens) else "<s>"
    eos_tok = str(tokens[eos_id]) if 0 <= eos_id < len(tokens) else "</s>"
    unk_tok = str(tokens[unk_id]) if 0 <= unk_id < len(tokens) else "<unk>"
    pad_tok = str(tokens[pad_id]) if 0 <= pad_id < len(tokens) else eos_tok

    raw_merges = meta.get("tokenizer.ggml.merges")
    if raw_merges:
        merges: list[tuple[str, str]] = []
        for entry in raw_merges:
            parts = str(entry).split(" ")
            if len(parts) == 2 and parts[0] in vocab and parts[1] in vocab:
                merges.append((parts[0], parts[1]))
        tok_obj = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=unk_tok, ignore_merges=True))
        # Handle Metaspace (SentencePiece U+2581) vs GPT2 ByteLevel
        if any(t.startswith("▁") for t in tokens[: min(1000, len(tokens))]):
            tok_obj.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
            tok_obj.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback()])
        else:
            tok_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok_obj.decoder = decoders.ByteLevel()
    else:
        scores = meta.get("tokenizer.ggml.scores") or [0.0] * len(tokens)
        unigram_vocab = [(str(t), float(scores[i]) if i < len(scores) else -10.0) for i, t in enumerate(tokens)]
        tok_obj = Tokenizer(models.Unigram(unigram_vocab, unk_id=unk_id))
        tok_obj.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
        tok_obj.decoder = decoders.Metaspace(replacement="▁")

    # Register control / structural tokens with exact vocab IDs preserved so templates encode cleanly
    special_tokens = [
        AddedToken(str(tokens[i]), special=True, normalized=False)
        for i, tt in enumerate(token_types)
        if tt == 3 and i < len(tokens)
    ]
    for st_name in ("<start_of_turn>", "<end_of_turn>", "<|eot_id|>", "<|im_start|>", "<|im_end|>"):
        if st_name in vocab and all(s.content != st_name for s in special_tokens):
            special_tokens.append(AddedToken(st_name, special=True, normalized=False))
    if special_tokens:
        tok_obj.add_special_tokens(special_tokens)

    chat_tmpl = meta.get("tokenizer.chat_template")
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token=bos_tok,
        eos_token=eos_tok,
        unk_token=unk_tok,
        pad_token=pad_tok,
    )
    if isinstance(chat_tmpl, str) and chat_tmpl.strip():
        fast_tok.chat_template = chat_tmpl
    return fast_tok


def _unpermute_llama_qk(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Undo llama.cpp interleaved RoPE head pair ordering -> HuggingFace half-split ordering."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // n_heads
    return w.view(n_heads, head_dim // 2, 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim).contiguous()


def _layer_gguf_to_hf_map(
    layer_idx: int,
    *,
    is_gemma_norm: bool,
    need_llama_unperm: bool,
    n_heads: int,
    n_kv_heads: int,
    has_pre_ffn_norm: bool,
) -> dict[str, tuple[str, bool, int | None]]:
    i = layer_idx
    return {
        f"blk.{i}.attn_norm.weight": (f"model.layers.{i}.input_layernorm.weight", is_gemma_norm, None),
        f"blk.{i}.post_attention_norm.weight": (
            f"model.layers.{i}.post_attention_layernorm.weight",
            is_gemma_norm,
            None,
        ),
        f"blk.{i}.ffn_norm.weight": (
            f"model.layers.{i}.pre_feedforward_layernorm.weight"
            if has_pre_ffn_norm
            else f"model.layers.{i}.post_attention_layernorm.weight",
            is_gemma_norm,
            None,
        ),
        f"blk.{i}.post_ffw_norm.weight": (
            f"model.layers.{i}.post_feedforward_layernorm.weight",
            is_gemma_norm,
            None,
        ),
        f"blk.{i}.attn_q_norm.weight": (f"model.layers.{i}.self_attn.q_norm.weight", is_gemma_norm, None),
        f"blk.{i}.attn_k_norm.weight": (f"model.layers.{i}.self_attn.k_norm.weight", is_gemma_norm, None),
        f"blk.{i}.attn_q.weight": (
            f"model.layers.{i}.self_attn.q_proj.weight",
            False,
            n_heads if need_llama_unperm else None,
        ),
        f"blk.{i}.attn_k.weight": (
            f"model.layers.{i}.self_attn.k_proj.weight",
            False,
            n_kv_heads if need_llama_unperm else None,
        ),
        f"blk.{i}.attn_v.weight": (f"model.layers.{i}.self_attn.v_proj.weight", False, None),
        f"blk.{i}.attn_output.weight": (f"model.layers.{i}.self_attn.o_proj.weight", False, None),
        f"blk.{i}.attn_q.bias": (f"model.layers.{i}.self_attn.q_proj.bias", False, None),
        f"blk.{i}.attn_k.bias": (f"model.layers.{i}.self_attn.k_proj.bias", False, None),
        f"blk.{i}.attn_v.bias": (f"model.layers.{i}.self_attn.v_proj.bias", False, None),
        f"blk.{i}.ffn_gate.weight": (f"model.layers.{i}.mlp.gate_proj.weight", False, None),
        f"blk.{i}.ffn_up.weight": (f"model.layers.{i}.mlp.up_proj.weight", False, None),
        f"blk.{i}.ffn_down.weight": (f"model.layers.{i}.mlp.down_proj.weight", False, None),
    }


def _reinit_hf_nonpersistent_buffers(model: nn.Module, hidden_size: int, device: torch.device) -> None:
    """Reinitialize non-persistent RoPE and embed_scale buffers wiped by ``meta`` construction."""
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if "rotary" in type(child).__name__.lower() and hasattr(child, "config"):
                setattr(parent, child_name, type(child)(child.config).to(device))
    for mod in model.modules():
        if hasattr(mod, "rope_init_fn") and hasattr(mod, "inv_freq"):
            inv_freq, att_scaling = mod.rope_init_fn(mod.config, "cpu")
            inv_dev = inv_freq.to(device=mod.inv_freq.device, dtype=mod.inv_freq.dtype)
            with torch.no_grad():
                mod.inv_freq.copy_(inv_dev)
                if hasattr(mod, "original_inv_freq"):
                    mod.original_inv_freq.copy_(inv_dev.to(device=mod.original_inv_freq.device))
                mod.attention_scaling = att_scaling
    embed_tok = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed_tok is not None and hasattr(embed_tok, "embed_scale"):
        scale_val = math.sqrt(hidden_size)
        if embed_tok.embed_scale.device.type == "meta" or embed_tok.embed_scale.device != device:
            embed_tok.register_buffer("embed_scale", torch.tensor(scale_val, device=device), persistent=False)
        else:
            with torch.no_grad():
                embed_tok.embed_scale.fill_(scale_val)


def _swap_layers_to_nvfp4_skeletons(
    module: nn.Module,
    *,
    dtype: torch.dtype,
    hadamard: bool,
    nvfp4_cls: type[nn.Module],
    group_size: int,
    sensitive_names: frozenset[str],
) -> None:
    """Replace eligible ``nn.Linear`` layers under ``module`` with unallocated ``NVFP4Linear`` skeletons."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, nvfp4_cls):
            if name in sensitive_names or (name == "gate" and child.out_features <= 64):
                continue
            if child.in_features % group_size == 0:
                setattr(
                    module,
                    name,
                    nvfp4_cls.empty_skeleton(
                        child.in_features,
                        child.out_features,
                        bias=(child.bias is not None),
                        dtype=dtype,
                        hadamard=hadamard,
                    ),
                )
        else:
            _swap_layers_to_nvfp4_skeletons(
                child,
                dtype=dtype,
                hadamard=hadamard,
                nvfp4_cls=nvfp4_cls,
                group_size=group_size,
                sensitive_names=sensitive_names,
            )


def _resolve_parent_leaf(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str] | None:
    if "." not in dotted_name:
        return model, dotted_name
    parent_path, leaf = dotted_name.rsplit(".", 1)
    try:
        return model.get_submodule(parent_path), leaf
    except AttributeError:
        return None


def _finalize_streaming_meta_tensors(
    model: nn.Module,
    target_dev: torch.device,
    target_dtype: torch.dtype,
    nvfp4_cls: type[nn.Module],
) -> None:
    """Ensure no ``meta`` tensors or unallocated ``NVFP4Linear.bias`` remain after streaming."""
    for mod in model.modules():
        if (
            isinstance(mod, nvfp4_cls)
            and mod.bias is not None
            and (mod.bias.numel() == 0 or mod.bias.device.type == "meta")
        ):
            mod.bias = nn.Parameter(
                torch.zeros(mod.out_features, dtype=target_dtype, device=target_dev),
                requires_grad=False,
            )
        for pname, p in list(mod.named_parameters(recurse=False)):
            if p is not None and p.device.type == "meta":
                setattr(
                    mod,
                    pname,
                    nn.Parameter(torch.zeros(p.shape, dtype=target_dtype, device=target_dev), requires_grad=False),
                )
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf is not None and buf.device.type == "meta":
                persistent = bname not in getattr(mod, "_non_persistent_buffers_set", ())
                mod.register_buffer(
                    bname,
                    torch.zeros(buf.shape, dtype=buf.dtype, device=target_dev),
                    persistent=persistent,
                )


def _compute_extra_eos_ids(hf_tok: Any, meta: dict[str, Any], eos_id: int) -> set[int]:
    extra_eos: set[int] = {eos_id}
    raw_eos_arr = meta.get("tokenizer.ggml.eos_token_ids")
    if isinstance(raw_eos_arr, (list, tuple)):
        extra_eos.update(int(x) for x in raw_eos_arr)
    vocab_map = hf_tok.get_vocab() if hasattr(hf_tok, "get_vocab") else {}
    unk_id = getattr(hf_tok, "unk_token_id", None)
    for stop_sym in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
        if stop_sym in vocab_map:
            sid = int(vocab_map[stop_sym])
            if sid >= 0 and sid != unk_id:
                extra_eos.add(sid)
    return extra_eos


def _build_or_get_cached_gguf_tokenizer(
    cache_key: tuple[str, int, int],
    meta: dict[str, Any],
    eos_id: int,
) -> tuple[Any, set[int]]:
    cached = _GGUF_TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        hf_tok, extra_frozen, _ = cached
        extra = set(extra_frozen)
        extra.add(int(eos_id))
        return hf_tok, extra
    hf_tok = build_hf_tokenizer_from_gguf(meta)
    extra = _compute_extra_eos_ids(hf_tok, meta, eos_id)
    if len(_GGUF_TOKENIZER_CACHE) >= _GGUF_TOKENIZER_CACHE_MAX:
        oldest_key = next(iter(_GGUF_TOKENIZER_CACHE))
        _GGUF_TOKENIZER_CACHE.pop(oldest_key, None)
    _GGUF_TOKENIZER_CACHE[cache_key] = (hf_tok, frozenset(extra), len(hf_tok))
    return hf_tok, set(extra)


def _tie_lm_head_if_requested(model: nn.Module, *, tie: bool) -> None:
    if not tie:
        return
    embed_tok = getattr(getattr(model, "model", None), "embed_tokens", None)
    lm_head = getattr(model, "lm_head", None)
    if embed_tok is not None and lm_head is not None and hasattr(embed_tok, "weight"):
        lm_head.weight = embed_tok.weight


def load_gguf_hf_model(
    gguf_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    quantize: str | None = None,
    hadamard: bool = True,
) -> tuple[nn.Module, Any, dict[str, Any], torch.device, int, int]:
    """Dequantize a local GGUF archive directly into an HF CausalLM on CPU or CUDA."""
    from transformers import (
        Gemma2Config,
        Gemma2ForCausalLM,
        Gemma3ForCausalLM,
        Gemma3TextConfig,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    from lucid_machines.inference import _HFInferenceTokenizerAdapter, optimize_hf_model_for_inference

    p = Path(gguf_path)
    st = p.stat()
    tok_cache_key = (str(p.resolve()), int(st.st_size), int(st.st_mtime_ns))
    cached_tok_entry = _GGUF_TOKENIZER_CACHE.get(tok_cache_key)
    directory = read_gguf_directory(p, load_tokenizer_arrays=(cached_tok_entry is None))
    meta = directory["meta"]
    tensors_idx = directory["tensors"]
    data_start = directory["data_start"]

    arch = str(meta.get("general.architecture") or "llama").strip().lower()
    if arch.startswith("qwen3") or arch == "qwen3":
        arch_cls_name = "qwen3"
    else:
        arch_cls_name = arch

    supported_arches = {"gemma", "gemma2", "gemma3", "llama", "mistral", "qwen2", "qwen3"}
    if arch_cls_name not in supported_arches:
        raise ValueError(
            f"GGUF architecture '{arch}' is not supported for direct PyTorch weight transfer. "
            "Supported architectures: Gemma 2/3, Llama, Mistral, and Qwen 2/3."
        )

    n_layers = int(meta.get(f"{arch}.block_count") or meta.get("llama.block_count"))
    hidden_size = int(meta.get(f"{arch}.embedding_length") or meta.get("llama.embedding_length"))
    n_heads = int(meta.get(f"{arch}.attention.head_count") or meta.get("llama.attention.head_count"))
    n_kv_heads = int(meta.get(f"{arch}.attention.head_count_kv") or n_heads)
    head_dim = int(meta.get(f"{arch}.attention.key_length") or (hidden_size // max(1, n_heads)))
    inter_size = int(meta.get(f"{arch}.feed_forward_length") or (hidden_size * 4))
    max_pos = int(meta.get(f"{arch}.context_length") or 8192)
    rms_eps = float(meta.get(f"{arch}.attention.layer_norm_rms_epsilon") or 1e-6)
    rope_theta = float(meta.get(f"{arch}.rope.freq_base") or meta.get(f"{arch}.rope.global.freq_base") or 10000.0)

    emb_info = tensors_idx.get("token_embd.weight")
    fallback_vocab_len = (
        len(meta.get("tokenizer.ggml.tokens") or []) if cached_tok_entry is None else (cached_tok_entry[2] or 0)
    )
    vocab_size = int(emb_info["dims"][1]) if emb_info else fallback_vocab_len
    bos_id = int(meta.get("tokenizer.ggml.bos_token_id", 1))
    eos_id = int(meta.get("tokenizer.ggml.eos_token_id", 2))
    pad_id = int(meta.get("tokenizer.ggml.padding_token_id", eos_id))
    has_output_head = "output.weight" in tensors_idx

    target_dev = (
        torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    target_dtype = dtype or (
        torch.bfloat16
        if (target_dev.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
        if target_dev.type == "cuda"
        else torch.float32
    )

    if arch == "gemma3":
        cfg = Gemma3TextConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_pos,
            rms_norm_eps=rms_eps,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            tie_word_embeddings=not has_output_head,
            sliding_window=int(meta.get("gemma3.attention.sliding_window") or 512),
            final_logit_softcapping=meta.get("gemma3.final_logit_softcapping"),
            attn_logit_softcapping=meta.get("gemma3.attn_logit_softcapping"),
            query_pre_attn_scalar=int(head_dim),
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": float(meta.get("gemma3.rope.local.freq_base") or 10000.0),
                },
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": float(meta.get("gemma3.rope.global.freq_base") or 1000000.0),
                },
            },
        )
        model_cls = Gemma3ForCausalLM
    elif arch == "gemma2":
        cfg = Gemma2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_pos,
            rms_norm_eps=rms_eps,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            tie_word_embeddings=not has_output_head,
        )
        model_cls = Gemma2ForCausalLM
    elif arch_cls_name in ("qwen2", "qwen3"):
        if arch_cls_name == "qwen3":
            try:
                from transformers import Qwen3Config, Qwen3ForCausalLM

                cfg = Qwen3Config(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    intermediate_size=inter_size,
                    num_hidden_layers=n_layers,
                    num_attention_heads=n_heads,
                    num_key_value_heads=n_kv_heads,
                    head_dim=head_dim,
                    max_position_embeddings=max_pos,
                    rms_norm_eps=rms_eps,
                    rope_theta=rope_theta,
                    tie_word_embeddings=not has_output_head,
                )
                model_cls = Qwen3ForCausalLM
            except ImportError:
                cfg = Qwen2Config(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    intermediate_size=inter_size,
                    num_hidden_layers=n_layers,
                    num_attention_heads=n_heads,
                    num_key_value_heads=n_kv_heads,
                    max_position_embeddings=max_pos,
                    rms_norm_eps=rms_eps,
                    rope_theta=rope_theta,
                    tie_word_embeddings=not has_output_head,
                )
                model_cls = Qwen2ForCausalLM
        else:
            cfg = Qwen2Config(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=inter_size,
                num_hidden_layers=n_layers,
                num_attention_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                max_position_embeddings=max_pos,
                rms_norm_eps=rms_eps,
                rope_theta=rope_theta,
                tie_word_embeddings=not has_output_head,
            )
            model_cls = Qwen2ForCausalLM
    elif arch == "mistral":
        cfg = MistralConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            max_position_embeddings=max_pos,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=not has_output_head,
        )
        model_cls = MistralForCausalLM
    else:
        cfg = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            max_position_embeddings=max_pos,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=not has_output_head,
        )
        model_cls = LlamaForCausalLM

    with torch.device("meta"):
        model = model_cls(cfg)

    meta_param_shapes = {name: tuple(p_meta.shape) for name, p_meta in model.named_parameters()}
    dtype_bytes = torch.empty((), dtype=target_dtype).element_size()
    est_bf16_bytes = 0
    est_embed_bytes = 0
    for pname, p_meta in model.named_parameters():
        if not has_output_head and pname == "lm_head.weight":
            continue
        b = int(p_meta.numel()) * dtype_bytes
        est_bf16_bytes += b
        if "embed_tokens" in pname or pname == "lm_head.weight":
            est_embed_bytes += b

    initial_free_vram = 0
    if target_dev.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        initial_free_vram, _ = torch.cuda.mem_get_info(target_dev)

    use_nvfp4 = bool(quantize and "nvfp4" in str(quantize).lower())
    if not use_nvfp4 and target_dev.type == "cuda" and est_bf16_bytes > int(initial_free_vram * 0.75):
        use_nvfp4 = True

    if target_dev.type == "cuda" and use_nvfp4:
        est_nvfp4_bytes = est_embed_bytes + int((est_bf16_bytes - est_embed_bytes) * (0.625 / max(1, dtype_bytes)))
        if est_nvfp4_bytes > int(initial_free_vram * 0.92):
            target_dev = torch.device("cpu")

    is_gemma_norm = arch in ("gemma", "gemma2", "gemma3")
    need_llama_unperm = arch in ("llama", "mistral")

    if target_dev.type == "cuda" and use_nvfp4:
        from lucid_machines.kernels.fast_nvfp4 import NVFP4_GROUP_SIZE, NVFP4Linear, fwht_16, nvfp4_quantize
        from lucid_machines.model.fp4 import _SENSITIVE_NVFP4_EXACT_NAMES

        inner_layers = getattr(getattr(model, "model", None), "layers", None)
        if inner_layers is not None:
            _swap_layers_to_nvfp4_skeletons(
                inner_layers,
                dtype=target_dtype,
                hadamard=hadamard,
                nvfp4_cls=NVFP4Linear,
                group_size=NVFP4_GROUP_SIZE,
                sensitive_names=_SENSITIVE_NVFP4_EXACT_NAMES,
            )

        def _assign_streamed_param(param_name: str, tensor: torch.Tensor) -> None:
            expected = meta_param_shapes.get(param_name)
            if expected is None:
                return
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"Shape mismatch assigning '{param_name}': expected {expected}, got {tuple(tensor.shape)}"
                )
            resolved = _resolve_parent_leaf(model, param_name)
            if resolved is None:
                return
            parent_mod, leaf = resolved
            setattr(
                parent_mod,
                leaf,
                nn.Parameter(tensor.to(device=target_dev, dtype=target_dtype), requires_grad=False),
            )

        with open(p, "rb") as f:
            mm: mmap.mmap | None = None
            try:
                if st.st_size > 0:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                mm = None
            try:
                if "token_embd.weight" in tensors_idx and "model.embed_tokens.weight" in meta_param_shapes:
                    emb_t = _read_gguf_tensor(
                        f,
                        data_start,
                        tensors_idx["token_embd.weight"],
                        target_dtype,
                        mm=mm,
                        compute_device=target_dev,
                        return_on_device=True,
                    )
                    _assign_streamed_param("model.embed_tokens.weight", emb_t)
                    del emb_t
                if has_output_head and "output.weight" in tensors_idx and "lm_head.weight" in meta_param_shapes:
                    out_t = _read_gguf_tensor(
                        f,
                        data_start,
                        tensors_idx["output.weight"],
                        target_dtype,
                        mm=mm,
                        compute_device=target_dev,
                        return_on_device=True,
                    )
                    _assign_streamed_param("lm_head.weight", out_t)
                    del out_t
                if "output_norm.weight" in tensors_idx and "model.norm.weight" in meta_param_shapes:
                    norm_t = _read_gguf_tensor(
                        f, data_start, tensors_idx["output_norm.weight"], torch.float32, mm=mm, compute_device=None
                    )
                    _assign_streamed_param("model.norm.weight", (norm_t - 1.0) if is_gemma_norm else norm_t)
                    del norm_t

                for i in range(n_layers):
                    gguf_to_hf = _layer_gguf_to_hf_map(
                        i,
                        is_gemma_norm=is_gemma_norm,
                        need_llama_unperm=need_llama_unperm,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        has_pre_ffn_norm=(f"model.layers.{i}.pre_feedforward_layernorm.weight" in meta_param_shapes),
                    )
                    for g_key, (hf_key, sub_one, unperm_h) in gguf_to_hf.items():
                        if g_key not in tensors_idx or hf_key not in meta_param_shapes:
                            continue
                        resolved = _resolve_parent_leaf(model, hf_key)
                        if resolved is None:
                            continue
                        parent_mod, leaf = resolved
                        if isinstance(parent_mod, NVFP4Linear) and leaf == "weight":
                            w_gpu = _read_gguf_tensor(
                                f,
                                data_start,
                                tensors_idx[g_key],
                                target_dtype,
                                mm=mm,
                                compute_device=target_dev,
                                return_on_device=True,
                            )
                            expected = meta_param_shapes[hf_key]
                            if tuple(w_gpu.shape) != expected:
                                raise ValueError(
                                    f"Shape mismatch assigning '{hf_key}': expected {expected}, got {tuple(w_gpu.shape)}"
                                )
                            if unperm_h is not None:
                                w_gpu = _unpermute_llama_qk(w_gpu, unperm_h)
                            w_q = fwht_16(w_gpu) if hadamard else w_gpu
                            packed, scale = nvfp4_quantize(w_q)
                            parent_mod.packed_weight = packed
                            parent_mod.scale = scale
                            del w_gpu, w_q
                        elif isinstance(parent_mod, NVFP4Linear) and leaf == "bias":
                            b_gpu = _read_gguf_tensor(
                                f,
                                data_start,
                                tensors_idx[g_key],
                                target_dtype,
                                mm=mm,
                                compute_device=target_dev,
                                return_on_device=True,
                            )
                            expected = meta_param_shapes[hf_key]
                            if tuple(b_gpu.shape) != expected:
                                raise ValueError(
                                    f"Shape mismatch assigning '{hf_key}': expected {expected}, got {tuple(b_gpu.shape)}"
                                )
                            parent_mod.bias = nn.Parameter(b_gpu, requires_grad=False)
                        else:
                            t = _read_gguf_tensor(
                                f,
                                data_start,
                                tensors_idx[g_key],
                                torch.float32 if sub_one else target_dtype,
                                mm=mm,
                                compute_device=None if sub_one else target_dev,
                                return_on_device=not sub_one,
                            )
                            if sub_one:
                                t = t - 1.0
                            if unperm_h is not None:
                                t = _unpermute_llama_qk(t, unperm_h)
                            _assign_streamed_param(hf_key, t)
                            del t
                    if (i + 1) % 8 == 0:
                        torch.cuda.empty_cache()
            finally:
                if mm is not None:
                    mm.close()

        _reinit_hf_nonpersistent_buffers(model, hidden_size, target_dev)
        _tie_lm_head_if_requested(model, tie=not has_output_head)
        _finalize_streaming_meta_tensors(model, target_dev, target_dtype, NVFP4Linear)
        _tie_lm_head_if_requested(model, tie=not has_output_head)
        torch.cuda.empty_cache()
    else:
        model = model.to_empty(device="cpu").to(dtype=target_dtype)
        param_map = dict(model.named_parameters())

        def _assign(param_name: str, tensor: torch.Tensor) -> None:
            if param_name not in param_map:
                return
            p_dst = param_map[param_name]
            if p_dst.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch assigning '{param_name}': expected {tuple(p_dst.shape)}, got {tuple(tensor.shape)}"
                )
            with torch.no_grad():
                p_dst.copy_(tensor.to(dtype=p_dst.dtype))

        with open(p, "rb") as f:
            mm: mmap.mmap | None = None
            try:
                if st.st_size > 0:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                mm = None
            try:
                if "token_embd.weight" in tensors_idx:
                    _assign(
                        "model.embed_tokens.weight",
                        _read_gguf_tensor(
                            f,
                            data_start,
                            tensors_idx["token_embd.weight"],
                            target_dtype,
                            mm=mm,
                            compute_device=target_dev,
                        ),
                    )
                if has_output_head and "lm_head.weight" in param_map:
                    _assign(
                        "lm_head.weight",
                        _read_gguf_tensor(
                            f,
                            data_start,
                            tensors_idx["output.weight"],
                            target_dtype,
                            mm=mm,
                            compute_device=target_dev,
                        ),
                    )

                if "output_norm.weight" in tensors_idx:
                    norm_t = _read_gguf_tensor(
                        f, data_start, tensors_idx["output_norm.weight"], torch.float32, mm=mm, compute_device=None
                    )
                    _assign("model.norm.weight", (norm_t - 1.0) if is_gemma_norm else norm_t)

                for i in range(n_layers):
                    gguf_to_hf = _layer_gguf_to_hf_map(
                        i,
                        is_gemma_norm=is_gemma_norm,
                        need_llama_unperm=need_llama_unperm,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        has_pre_ffn_norm=(f"model.layers.{i}.pre_feedforward_layernorm.weight" in param_map),
                    )
                    for g_key, (hf_key, sub_one, unperm_h) in gguf_to_hf.items():
                        if g_key in tensors_idx and hf_key in param_map:
                            t = _read_gguf_tensor(
                                f,
                                data_start,
                                tensors_idx[g_key],
                                torch.float32 if sub_one else target_dtype,
                                mm=mm,
                                compute_device=target_dev if not sub_one else None,
                            )
                            if sub_one:
                                t = t - 1.0
                            if unperm_h is not None:
                                t = _unpermute_llama_qk(t, unperm_h)
                            _assign(hf_key, t)
            finally:
                if mm is not None:
                    mm.close()

        _tie_lm_head_if_requested(model, tie=not has_output_head)
        _reinit_hf_nonpersistent_buffers(model, hidden_size, torch.device("cpu"))

        if use_nvfp4:
            from lucid_machines.model.fp4 import convert_to_nvfp4

            convert_to_nvfp4(model, hadamard=hadamard, protect_sensitive=True)

        try:
            model = model.to(target_dev)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            target_dev = torch.device("cpu")
            model = model.to(target_dev)
        _tie_lm_head_if_requested(model, tie=not has_output_head)

    model.eval()
    optimize_hf_model_for_inference(model)
    hf_tok, extra_eos = _build_or_get_cached_gguf_tokenizer(tok_cache_key, meta, eos_id)
    tok_adapter = _HFInferenceTokenizerAdapter(hf_tok, model_vocab_size=vocab_size)
    model._lm_extra_eos_ids = extra_eos
    model._lucid_extra_eos_ids = extra_eos
    model._triune_extra_eos_ids = extra_eos

    metadata = {
        "source": str(gguf_path),
        "gguf_arch": arch,
        "num_layers": n_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "quantized_nvfp4": use_nvfp4,
        "gguf_loaded_direct": True,
    }
    return model, tok_adapter, metadata, target_dev, pad_id, eos_id
