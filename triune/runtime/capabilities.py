"""Hardware and execution runtime precision capability detection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)

# Cache probe results per device index to avoid redundant probe overhead
_SCALED_MM_PROBE_CACHE: dict[int, bool] = {}


def _probe_cublas_scaled_mm(device: torch.device) -> bool:
    """Probes whether torch._scaled_mm can actually execute a matmul on the given device.

    This avoids CUBLAS_STATUS_NOT_SUPPORTED crashes when PyTorch exposes _scaled_mm
    as an API symbol but cuBLAS lacks kernel support for the specific architecture
    (e.g. SM120 on CUDA < 12.8).
    """
    if not (hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn")):
        return False
    if device.type != "cuda":
        return False

    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    if dev_idx in _SCALED_MM_PROBE_CACHE:
        return _SCALED_MM_PROBE_CACHE[dev_idx]

    supported = False
    try:
        a = torch.zeros((16, 16), device=device, dtype=torch.float8_e4m3fn)
        b = torch.zeros((16, 16), device=device, dtype=torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device=device)
        sb = torch.tensor(1.0, device=device)
        torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        supported = True
    except Exception:
        supported = False

    _SCALED_MM_PROBE_CACHE[dev_idx] = supported
    return supported


@dataclass
class PrecisionCapabilities:
    """Hardware and library capabilities for lower-precision execution."""

    device: torch.device
    compute_capability: Tuple[int, int]
    fp8_gemm: bool
    fp4_gemm: bool
    transformer_engine: bool
    native_scaled_mm: bool
    bf16_supported: bool

    @classmethod
    def detect(cls, device: Optional[torch.device | str] = None) -> "PrecisionCapabilities":
        """Dynamically probes the environment for precision capabilities without hard-coding."""
        if device is None:
            dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        elif isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = device

        if dev.type != "cuda" or not torch.cuda.is_available():
            return cls(
                device=dev,
                compute_capability=(0, 0),
                fp8_gemm=False,
                fp4_gemm=False,
                transformer_engine=False,
                native_scaled_mm=False,
                bf16_supported=False,
            )

        major, minor = torch.cuda.get_device_capability(dev)
        bf16_supported = (major >= 8) and torch.cuda.is_bf16_supported()

        # Check Transformer Engine
        has_te = False
        try:
            import transformer_engine.pytorch as te  # noqa: F401
            has_te = True
        except (ImportError, Exception):
            has_te = False

        # Check native scaled_mm actual kernel execution
        native_scaled_mm = _probe_cublas_scaled_mm(dev)

        # FP8 Tensor Cores exist on SM89 (Ada), SM90 (Hopper), SM100/SM120 (Blackwell)
        is_sm89_plus = (major > 8) or (major == 8 and minor >= 9)
        fp8_gemm = is_sm89_plus and (native_scaled_mm or has_te)

        # FP4 GEMM requires Blackwell (SM100+ / SM120+) and Transformer Engine with NVFP4 support
        has_te_fp4 = False
        if has_te and major >= 10:
            try:
                from transformer_engine.common.recipe import NVFP4BlockScaling  # noqa: F401
                has_te_fp4 = True
            except (ImportError, Exception):
                has_te_fp4 = False
        fp4_gemm = (major >= 10) and has_te_fp4

        return cls(
            device=dev,
            compute_capability=(major, minor),
            fp8_gemm=fp8_gemm,
            fp4_gemm=fp4_gemm,
            transformer_engine=has_te,
            native_scaled_mm=native_scaled_mm,
            bf16_supported=bf16_supported,
        )

    def resolve_precision(self, requested: str, fallback: bool = True) -> str:
        """Resolves requested precision string against detected capabilities.

        Supported requested strings: 'fp4', 'fp8', 'bf16', 'bfloat16', 'fp32', 'float32'.
        If requested precision is unsupported:
          - If fallback=True: returns highest supported precision and logs warning.
          - If fallback=False: raises RuntimeError with clear context.
        """
        req = requested.lower().strip()
        major, minor = self.compute_capability

        if req == "fp4":
            if self.fp4_gemm:
                return "fp4"
            msg = (
                f"FP4 requested but unsupported on {self.device} (SM{major}.{minor}). "
                f"Requires SM100+ and Transformer Engine with NVFP4BlockScaling (TE={self.transformer_engine})."
            )
            if not fallback:
                raise RuntimeError(msg)
            logger.warning(f"{msg} Falling back to BF16.")
            return "bf16" if self.bf16_supported else "fp32"

        if req == "fp8":
            if self.fp8_gemm:
                return "fp8"
            msg = (
                f"FP8 GEMM requested but unsupported on {self.device} (SM{major}.{minor}). "
                f"Requires SM89+ and functional scaled_mm or Transformer Engine "
                f"(native_scaled_mm={self.native_scaled_mm}, TE={self.transformer_engine})."
            )
            if not fallback:
                raise RuntimeError(msg)
            logger.warning(f"{msg} Falling back to BF16.")
            return "bf16" if self.bf16_supported else "fp32"

        if req in ("bf16", "bfloat16"):
            if self.bf16_supported:
                return "bf16"
            msg = f"BF16 requested but unsupported on {self.device} (SM{major}.{minor})."
            if not fallback:
                raise RuntimeError(msg)
            logger.warning(f"{msg} Falling back to FP32.")
            return "fp32"

        if req in ("fp32", "float32"):
            return "fp32"

        raise ValueError(f"Unknown requested precision: {requested}")
