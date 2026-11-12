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

# Pure PEP 562 Lazy Loading: Zero-overhead startup for CLI, tooling, and desktop launchers.
# PyTorch, Triton, and native CUDA extensions are loaded strictly on first access.
_CORE_IMPORTS = {
    # Core public API
    "Model": ".model",
    "Config": ".configs.config",
    "CentroidSteer": ".optim",
    "Muon": ".optim",
    "ops": ".ops",
    # Configs
    "build_config": ".configs.config",
    "default_config": ".configs.config",
    "validate_config": ".configs.config",
    # Kernels
    "FastGELU": ".kernels",
    "FastSwiGLU": ".kernels",
    "fast_chunk_gla": ".kernels",
    "fast_cross_entropy": ".kernels",
    "fast_gelu": ".kernels",
    "fast_lora_forward": ".kernels",
    "fast_rmsnorm": ".kernels",
    "fast_rope": ".kernels",
    "fast_swiglu": ".kernels",
    "fast_zeropower_via_newtonschulz5": ".kernels",
    "get_kernel_telemetry": ".kernels",
    "is_turbo_mode_enabled": ".kernels",
    "set_turbo_mode": ".kernels",
    # Model
    "TriuneModel": ".model",
    "TriuneTransformer": ".model",
    "attach_early_exits": ".model",
    "build_model": ".model",
    "load_model": ".model",
    "register_model": ".model",
    # Optim
    "CentroidSteerOptimizer": ".optim",
    "build_optimizer": ".optim",
}

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
    # Universal Framework
    "UniversalDenseTransformer": ".universal",
    "UniversalMoETransformer": ".universal",
    "TriuneUniversalEngine": ".universal",
    "upgrade_model": ".universal",
    "upgrade": ".universal",
}

__all__ = [
    # Minimalist Public API
    "Model",
    "Config",
    "CentroidSteer",
    "Muon",
    "ops",
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


_ALL_LAZY = {**_CORE_IMPORTS, **_LAZY_IMPORTS}


def __getattr__(name: str) -> Any:
    if name in _ALL_LAZY:
        module_path = _ALL_LAZY[name]
        module = importlib.import_module(module_path, __package__)
        if name == "Model":
            val = getattr(module, "build_model")
        elif name == "Config":
            val = getattr(module, "build_config")
        elif name == "CentroidSteer":
            val = getattr(module, "CentroidSteerOptimizer")
        elif name == "ops":
            val = module
        else:
            val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_ALL_LAZY.keys()))
