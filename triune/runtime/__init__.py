try:
    from .memory_planner import MemoryEstimate, MemoryPlanner, MemoryEstimate as MemoryPlan
    from .resource_manager import (
        AllotmentPlan,
        DynamicResourceManager,
        HardwareProfile,
        LiveVRAMMonitor,
        VRAMBudgetExceededError,
    )
    from .streaming import (
        LayerStreamingEngine,
        StreamingConfig,
        apply_layer_streaming_if_requested,
        enable_layer_streaming,
    )
    from .capabilities import PrecisionCapabilities
    from .staging import FP8StagingBuffer, ParameterStager, StagingBuffer
    from .vram import AutoOffloader, VRAMProfiler
except ImportError:
    MemoryEstimate = None
    MemoryPlan = None
    MemoryPlanner = None
    AllotmentPlan = None
    DynamicResourceManager = None
    HardwareProfile = None
    LiveVRAMMonitor = None
    VRAMBudgetExceededError = None
    LayerStreamingEngine = None
    StreamingConfig = None
    apply_layer_streaming_if_requested = None
    enable_layer_streaming = None
    PrecisionCapabilities = None
    FP8StagingBuffer = None
    ParameterStager = None
    StagingBuffer = None
    AutoOffloader = None
    VRAMProfiler = None

from .sandbox import PythonSandbox, SandboxResult

__all__ = [
    "MemoryEstimate",
    "MemoryPlan",
    "MemoryPlanner",
    "DynamicResourceManager",
    "LiveVRAMMonitor",
    "AllotmentPlan",
    "HardwareProfile",
    "VRAMBudgetExceededError",
    "LayerStreamingEngine",
    "StreamingConfig",
    "enable_layer_streaming",
    "apply_layer_streaming_if_requested",
    "PrecisionCapabilities",
    "ParameterStager",
    "StagingBuffer",
    "FP8StagingBuffer",
    "PythonSandbox",
    "SandboxResult",
    "VRAMProfiler",
    "AutoOffloader",
]

