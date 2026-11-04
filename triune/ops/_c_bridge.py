"""
Triune Native C/C++/CUDA Bridge (`ctypes` FFI to `triune_core.dll` / `libtriune_core.so`).
Provides direct zero-overhead access to the compiled AVX2 + CUDA Runtime C engine (`csrc/`).
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class TriuneConfigStruct(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_int),
        ("hidden_dim", ctypes.c_int),
        ("num_layers", ctypes.c_int),
        ("num_heads", ctypes.c_int),
        ("head_dim", ctypes.c_int),
        ("num_experts", ctypes.c_int),
        ("top_k", ctypes.c_int),
        ("max_seq_len", ctypes.c_int),
        ("reflex_threshold", ctypes.c_float),
    ]


class TriuneStepStatsStruct(ctypes.Structure):
    _fields_ = [
        ("loss", ctypes.c_float),
        ("lm_loss", ctypes.c_float),
        ("router_loss", ctypes.c_float),
        ("active_tier", ctypes.c_int),
        ("tokens_processed", ctypes.c_int),
        ("step_time_ms", ctypes.c_float),
        ("vram_used_mb", ctypes.c_float),
        ("cuda_runtime_bound", ctypes.c_int),
    ]


_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[str] = None


def _load_native_library() -> Optional[ctypes.CDLL]:
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB

    ops_dir = Path(__file__).resolve().parent
    candidates = []
    if sys.platform == "win32":
        candidates.append(ops_dir / "triune_core.dll")
        # Also allow studio/windows directory fallback
        candidates.append(ops_dir.parent.parent / "studio" / "windows" / "TriuneStudio.Windows" / "triune_core.dll")
        # Pre-load PyTorch's bundled cudart64 DLL into process address space if present
        torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib_dir.is_dir():
            try:
                os.add_dll_directory(str(torch_lib_dir))
            except Exception:
                pass
    elif sys.platform == "darwin":
        candidates.append(ops_dir / "libtriune_core.dylib")
    else:
        candidates.append(ops_dir / "libtriune_core.so")

    for path in candidates:
        if path.is_file():
            try:
                lib = ctypes.CDLL(str(path))
                # Bind function signatures
                lib.triune_engine_version.argtypes = []
                lib.triune_engine_version.restype = ctypes.c_char_p

                lib.triune_has_cuda_runtime.argtypes = []
                lib.triune_has_cuda_runtime.restype = ctypes.c_int

                lib.triune_engine_create.argtypes = [ctypes.POINTER(TriuneConfigStruct), ctypes.c_int]
                lib.triune_engine_create.restype = ctypes.c_void_p

                lib.triune_engine_destroy.argtypes = [ctypes.c_void_p]
                lib.triune_engine_destroy.restype = None

                lib.triune_engine_step.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(TriuneStepStatsStruct),
                ]
                lib.triune_engine_step.restype = ctypes.c_int

                lib.triune_cuda_graph_capture.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
                lib.triune_cuda_graph_capture.restype = ctypes.c_int

                lib.triune_cuda_graph_replay.argtypes = [ctypes.c_void_p]
                lib.triune_cuda_graph_replay.restype = ctypes.c_int

                lib.triune_centroid_steer_update_f32.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_float,
                ]
                lib.triune_centroid_steer_update_f32.restype = None

                lib.triune_route_tiers_f32.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_float,
                ]
                lib.triune_route_tiers_f32.restype = None

                lib.triune_compute_neuron_heatmap_f32.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.triune_compute_neuron_heatmap_f32.restype = None

                lib.triune_gla_recurrent_step_f32.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.triune_gla_recurrent_step_f32.restype = None

                _LIB = lib
                _LIB_PATH = str(path)
                return _LIB
            except Exception:
                continue
    return None


def is_native_c_available() -> bool:
    return _load_native_library() is not None


def get_native_c_info() -> Dict[str, Any]:
    lib = _load_native_library()
    if lib is None:
        return {
            "available": False,
            "version": None,
            "cuda_runtime_bound": False,
            "library_path": None,
        }
    ver_bytes = lib.triune_engine_version()
    ver = ver_bytes.decode("utf-8", errors="replace") if ver_bytes else "unknown"
    cuda_bound = bool(lib.triune_has_cuda_runtime())
    return {
        "available": True,
        "version": ver,
        "cuda_runtime_bound": cuda_bound,
        "library_path": _LIB_PATH,
    }


def c_centroid_steer_update_f32(
    P: torch.Tensor,
    Grad_proj: torch.Tensor,
    steer_scale: float,
    momentum: float,
) -> bool:
    """
    Executes the AVX2 native C CentroidSteer update in-place if tensors are CPU contiguous float32.
    Returns True if executed via native C, False otherwise.
    """
    lib = _load_native_library()
    if (
        lib is None
        or P.device.type != "cpu"
        or Grad_proj.device.type != "cpu"
        or P.dtype != torch.float32
        or Grad_proj.dtype != torch.float32
        or not P.is_contiguous()
        or not Grad_proj.is_contiguous()
    ):
        return False
    rows = P.shape[0]
    cols = P.numel() // max(1, rows)
    lib.triune_centroid_steer_update_f32(
        ctypes.c_void_p(P.data_ptr()),
        ctypes.c_void_p(Grad_proj.data_ptr()),
        int(rows),
        int(cols),
        float(steer_scale),
        float(momentum),
    )
    return True


def c_route_tiers_f32(
    logits: torch.Tensor,
    reflex_threshold: float = 0.72,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Executes the native C 3-Tier Router (Reflex / Limbic / Cortex) on CPU float32 logits.
    """
    lib = _load_native_library()
    if lib is None or logits.device.type != "cpu" or logits.dtype != torch.float32:
        return None
    logits_c = logits.contiguous()
    batch_size = logits_c.numel() // 3
    probs_out = torch.empty_like(logits_c)
    tiers_out = torch.empty((batch_size,), dtype=torch.int32, device="cpu")
    lib.triune_route_tiers_f32(
        ctypes.c_void_p(logits_c.data_ptr()),
        ctypes.c_void_p(probs_out.data_ptr()),
        ctypes.c_void_p(tiers_out.data_ptr()),
        int(batch_size),
        float(reflex_threshold),
    )
    return probs_out, tiers_out.to(torch.int64).view(logits.shape[:-1])


def c_compute_neuron_heatmap_f32(
    activations: torch.Tensor,
    num_bins: int = 64,
) -> Optional[Tuple[List[float], List[int]]]:
    """
    Computes normalized 64-bin neuron firing intensities and RGBA32 color buffer via native C/AVX2.
    """
    lib = _load_native_library()
    if lib is None:
        return None
    flat = activations.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    if flat.numel() == 0:
        return None
    bins_t = torch.zeros((num_bins,), dtype=torch.float32, device="cpu")
    rgba_t = torch.zeros((num_bins,), dtype=torch.uint32 if hasattr(torch, "uint32") else torch.int32, device="cpu")
    lib.triune_compute_neuron_heatmap_f32(
        ctypes.c_void_p(flat.data_ptr()),
        ctypes.c_void_p(bins_t.data_ptr()),
        ctypes.c_void_p(rgba_t.data_ptr()),
        int(flat.numel()),
        int(num_bins),
    )
    return [round(float(x), 4) for x in bins_t.tolist()], [int(x) & 0xFFFFFFFF for x in rgba_t.tolist()]


def c_gla_recurrent_step_f32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Executes the AVX2 native C O(1) GLA single-token recurrent decode step on CPU float32 tensors.
    """
    lib = _load_native_library()
    if (
        lib is None
        or q.device.type != "cpu"
        or q.dtype != torch.float32
        or not q.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
        or not g.is_contiguous()
    ):
        return None
    head_dim = q.shape[-1]
    batch_heads = q.numel() // head_dim
    state_c = state.contiguous().clone()
    out = torch.empty_like(q)
    lib.triune_gla_recurrent_step_f32(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(g.data_ptr()),
        ctypes.c_void_p(state_c.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        int(batch_heads),
        int(head_dim),
    )
    return out, state_c


class NativeTriuneEngine:
    """
    Python wrapper around the native `TriuneEngine*` C++/CUDA arena & CUDA Graph executor.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 384,
        num_layers: int = 8,
        num_heads: int = 6,
        head_dim: int = 64,
        num_experts: int = 4,
        top_k: int = 2,
        max_seq_len: int = 512,
        reflex_threshold: float = 0.72,
        device_id: int = 0,
    ) -> None:
        self._lib = _load_native_library()
        self._handle: Optional[int] = None
        if self._lib is not None:
            cfg = TriuneConfigStruct(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                num_experts=num_experts,
                top_k=top_k,
                max_seq_len=max_seq_len,
                reflex_threshold=reflex_threshold,
            )
            self._handle = self._lib.triune_engine_create(ctypes.byref(cfg), int(device_id))

    @property
    def is_ready(self) -> bool:
        return bool(self._lib is not None and self._handle)

    def capture_graph(self, batch_size: int = 4, seq_len: int = 256) -> bool:
        if not self.is_ready:
            return False
        return self._lib.triune_cuda_graph_capture(self._handle, int(batch_size), int(seq_len)) == 0

    def replay_graph(self) -> bool:
        if not self.is_ready:
            return False
        return self._lib.triune_cuda_graph_replay(self._handle) == 0

    def step(self, batch_size: int = 4, seq_len: int = 256) -> Dict[str, Any]:
        if not self.is_ready:
            return {"ok": False}
        stats = TriuneStepStatsStruct()
        rc = self._lib.triune_engine_step(
            self._handle,
            None,
            None,
            int(batch_size),
            int(seq_len),
            ctypes.byref(stats),
        )
        return {
            "ok": rc == 0,
            "loss": round(float(stats.loss), 4),
            "lm_loss": round(float(stats.lm_loss), 4),
            "router_loss": round(float(stats.router_loss), 4),
            "active_tier": int(stats.active_tier),
            "tokens_processed": int(stats.tokens_processed),
            "step_time_ms": round(float(stats.step_time_ms), 4),
            "vram_used_mb": round(float(stats.vram_used_mb), 2),
            "cuda_runtime_bound": bool(stats.cuda_runtime_bound),
        }

    def close(self) -> None:
        if self._lib is not None and self._handle:
            self._lib.triune_engine_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
