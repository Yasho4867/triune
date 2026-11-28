"""
LucidMachines Native C/C++/CUDA Bridge (`ctypes` FFI to `lucid_machines_core.dll` / `liblucid_machines_core.so`).
Provides direct zero-overhead access to the compiled AVX2 + CUDA Runtime C engine (`csrc/`).
Includes backward-compatible fallback loading for `triune_core.*` and `triune_*` C ABI symbols.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class LucidMachinesConfigStruct(ctypes.Structure):
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


class LucidMachinesStepStatsStruct(ctypes.Structure):
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


TriuneConfigStruct = LucidMachinesConfigStruct
TriuneStepStatsStruct = LucidMachinesStepStatsStruct

_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[str] = None
_SYMS: Dict[str, Any] = {}


def _resolve_sym(lib: ctypes.CDLL, suffix: str) -> Any:
    fn = getattr(lib, f"lucid_machines_{suffix}", None)
    if fn is not None:
        return fn
    return getattr(lib, f"triune_{suffix}")


def _load_native_library() -> Optional[ctypes.CDLL]:
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB

    ops_dir = Path(__file__).resolve().parent
    candidates = []
    if sys.platform == "win32":
        candidates.append(ops_dir / "lucid_machines_core.dll")
        candidates.append(
            ops_dir.parent.parent / "studio" / "windows" / "PolysomStudio.Windows" / "lucid_machines_core.dll"
        )
        candidates.append(ops_dir / "triune_core.dll")
        candidates.append(ops_dir.parent.parent / "studio" / "windows" / "PolysomStudio.Windows" / "triune_core.dll")
        # Pre-load PyTorch's bundled cudart64 DLL into process address space if present
        torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib_dir.is_dir():
            try:
                os.add_dll_directory(str(torch_lib_dir))
            except Exception:
                pass
    elif sys.platform == "darwin":
        candidates.append(ops_dir / "liblucid_machines_core.dylib")
        candidates.append(ops_dir / "libtriune_core.dylib")
    else:
        candidates.append(ops_dir / "liblucid_machines_core.so")
        candidates.append(ops_dir / "libtriune_core.so")

    for path in candidates:
        if path.is_file():
            try:
                lib = ctypes.CDLL(str(path))
                syms: Dict[str, Any] = {}

                syms["engine_version"] = _resolve_sym(lib, "engine_version")
                syms["engine_version"].argtypes = []
                syms["engine_version"].restype = ctypes.c_char_p

                syms["has_cuda_runtime"] = _resolve_sym(lib, "has_cuda_runtime")
                syms["has_cuda_runtime"].argtypes = []
                syms["has_cuda_runtime"].restype = ctypes.c_int

                syms["engine_create"] = _resolve_sym(lib, "engine_create")
                syms["engine_create"].argtypes = [ctypes.POINTER(LucidMachinesConfigStruct), ctypes.c_int]
                syms["engine_create"].restype = ctypes.c_void_p

                syms["engine_destroy"] = _resolve_sym(lib, "engine_destroy")
                syms["engine_destroy"].argtypes = [ctypes.c_void_p]
                syms["engine_destroy"].restype = None

                syms["engine_step"] = _resolve_sym(lib, "engine_step")
                syms["engine_step"].argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(LucidMachinesStepStatsStruct),
                ]
                syms["engine_step"].restype = ctypes.c_int

                syms["cuda_graph_capture"] = _resolve_sym(lib, "cuda_graph_capture")
                syms["cuda_graph_capture"].argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
                syms["cuda_graph_capture"].restype = ctypes.c_int

                syms["cuda_graph_replay"] = _resolve_sym(lib, "cuda_graph_replay")
                syms["cuda_graph_replay"].argtypes = [ctypes.c_void_p]
                syms["cuda_graph_replay"].restype = ctypes.c_int

                syms["centroid_steer_update_f32"] = _resolve_sym(lib, "centroid_steer_update_f32")
                syms["centroid_steer_update_f32"].argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_float,
                ]
                syms["centroid_steer_update_f32"].restype = None

                syms["route_tiers_f32"] = _resolve_sym(lib, "route_tiers_f32")
                syms["route_tiers_f32"].argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_float,
                ]
                syms["route_tiers_f32"].restype = None

                syms["compute_neuron_heatmap_f32"] = _resolve_sym(lib, "compute_neuron_heatmap_f32")
                syms["compute_neuron_heatmap_f32"].argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                syms["compute_neuron_heatmap_f32"].restype = None

                syms["gla_recurrent_step_f32"] = _resolve_sym(lib, "gla_recurrent_step_f32")
                syms["gla_recurrent_step_f32"].argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                syms["gla_recurrent_step_f32"].restype = None

                _SYMS.clear()
                _SYMS.update(syms)
                _LIB = lib
                _LIB_PATH = str(path)
                return _LIB
            except Exception:
                continue
    return None


def is_native_c_available() -> bool:
    return _load_native_library() is not None


def get_current_cuda_stream_ptr(device_id: int = 0) -> int:
    """Return the raw cudaStream_t integer handle for PyTorch's active stream, or 0 on CPU."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return 0
    return int(torch.cuda.current_stream(device_id).cuda_stream)


def _sync_pytorch_cuda_stream(device_id: int = 0) -> None:
    """Synchronize the active PyTorch CUDA stream before crossing into unmanaged C ABI calls."""
    if (
        torch.cuda.is_available()
        and torch.cuda.is_initialized()
        and not torch.cuda.is_current_stream_capturing()
    ):
        torch.cuda.current_stream(device_id).synchronize()


def get_native_c_info() -> Dict[str, Any]:
    lib = _load_native_library()
    if lib is None:
        return {
            "available": False,
            "version": None,
            "cuda_runtime_bound": False,
            "library_path": None,
        }
    ver_bytes = _SYMS["engine_version"]()
    ver = ver_bytes.decode("utf-8", errors="replace") if ver_bytes else "unknown"
    cuda_bound = bool(_SYMS["has_cuda_runtime"]())
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
    _SYMS["centroid_steer_update_f32"](
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
    _SYMS["route_tiers_f32"](
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
    _SYMS["compute_neuron_heatmap_f32"](
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
    _SYMS["gla_recurrent_step_f32"](
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


class NativeLucidMachinesEngine:
    """
    Python wrapper around the native `LucidMachinesEngine*` C++/CUDA arena & CUDA Graph executor.
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
        self._device_id: int = int(device_id)
        if self._lib is not None:
            cfg = LucidMachinesConfigStruct(
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
            self._handle = _SYMS["engine_create"](ctypes.byref(cfg), int(device_id))

    @property
    def is_ready(self) -> bool:
        return bool(self._lib is not None and self._handle)

    def capture_graph(self, batch_size: int = 4, seq_len: int = 256) -> bool:
        if not self.is_ready:
            return False
        _sync_pytorch_cuda_stream(self._device_id)
        return _SYMS["cuda_graph_capture"](self._handle, int(batch_size), int(seq_len)) == 0

    def replay_graph(self) -> bool:
        if not self.is_ready:
            return False
        _sync_pytorch_cuda_stream(self._device_id)
        return _SYMS["cuda_graph_replay"](self._handle) == 0

    def step(self, batch_size: int = 4, seq_len: int = 256) -> Dict[str, Any]:
        if not self.is_ready:
            return {"ok": False}
        _sync_pytorch_cuda_stream(self._device_id)
        stats = LucidMachinesStepStatsStruct()
        rc = _SYMS["engine_step"](
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
            _SYMS["engine_destroy"](self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


NativeTriuneEngine = NativeLucidMachinesEngine
