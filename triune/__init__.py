"""Triune Core Package with PEP 562 Lazy Loading."""

from __future__ import annotations

import importlib
from typing import Any

# Eager core imports (fast, zero heavy web/agent/export dependencies)
from .configs.config import build_config, default_config, validate_config
from .kernels import fast_cross_entropy, fast_rmsnorm, fast_rope
from .model import (
    TriuneModel,
    TriuneTransformer,
    attach_early_exits,
    build_model,
    load_model,
    register_model,
)
from .optim import CentroidSteerOptimizer, Muon, build_optimizer

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
