"""Triune Core Package with PEP 562 Lazy Loading."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

# Ensure UTF-8 standard streams on Windows pipes/consoles so Unicode log symbols never raise UnicodeEncodeError
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(_stream, "reconfigure"):
                _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Eager core imports (fast, zero heavy web/agent/export dependencies)
try:
    from .configs.config import build_config, default_config, validate_config
    from .kernels import (
        fast_cross_entropy,
        fast_rmsnorm,
        fast_rope,
        fast_swiglu,
        fast_gelu,
        FastSwiGLU,
        FastGELU,
        fast_chunk_gla,
        fast_zeropower_via_newtonschulz5,
        fast_lora_forward,
        get_kernel_telemetry,
        is_turbo_mode_enabled,
        set_turbo_mode,
    )
    from .model import (
        TriuneModel,
        TriuneTransformer,
        attach_early_exits,
        build_model,
        load_model,
        register_model,
    )
    from .optim import CentroidSteerOptimizer, Muon, build_optimizer
except ImportError:
    # Allows lightweight tooling and desktop launchers to initialize without direct torch installation
    pass

# Mapping of heavy/secondary components to their host submodule
_LAZY_IMPORTS = {
    # Agents
    "Agent": ".agents",
    "MultiAgentOrchestrator": ".agents",
    "ProviderConfig": ".agents",
    "ProviderManager": ".agents",
    # API / Web Server
    "create_app": ".api",
    "run_server": ".api",
    # Callbacks
    "Callback": ".callbacks",
    "CallbackList": ".callbacks",
    "EventEmitter": ".callbacks",
    "TelemetryCallback": ".callbacks",
    "global_emitter": ".callbacks",
    # Execution
    "DAGParser": ".execution",
    "ExecutionEngine": ".execution",
    "NodeExecutionError": ".execution",
    # Export
    "export_gguf": ".export",
    "export_model": ".export",
    "export_onnx": ".export",
    "export_safetensors": ".export",
    # Plugins
    "NodeRegistry": ".plugins",
    "register_node": ".plugins",
    # Runtime
    "AutoOffloader": ".runtime",
    "DynamicResourceManager": ".runtime",
    "LayerStreamingEngine": ".runtime",
    "LiveVRAMMonitor": ".runtime",
    "MemoryPlan": ".runtime",
    "MemoryPlanner": ".runtime",
    "PythonSandbox": ".runtime",
    "VRAMProfiler": ".runtime",
    "enable_layer_streaming": ".runtime",
    # Trainer
    "Engine": ".trainer",
    "Trainer": ".trainer",
    "LoRAConfig": ".trainer",
    "LoRALayer": ".trainer",
    "TriuneFineTuner": ".trainer",
}

__all__ = [
    # Core (eager)
    "build_config",
    "default_config",
    "validate_config",
    "fast_cross_entropy",
    "fast_rmsnorm",
    "fast_rope",
    "fast_swiglu",
    "fast_gelu",
    "FastSwiGLU",
    "FastGELU",
    "fast_chunk_gla",
    "fast_zeropower_via_newtonschulz5",
    "fast_lora_forward",
    "get_kernel_telemetry",
    "is_turbo_mode_enabled",
    "set_turbo_mode",
    "TriuneModel",
    "TriuneTransformer",
    "attach_early_exits",
    "build_model",
    "load_model",
    "register_model",
    "CentroidSteerOptimizer",
    "Muon",
    "build_optimizer",
    # Lazy
    *_LAZY_IMPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __package__)
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))
