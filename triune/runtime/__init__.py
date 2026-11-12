try:
    from .capabilities import PrecisionCapabilities
    from .cuda_graph import CUDAGraphEngine
    from .memory_planner import MemoryEstimate, MemoryPlanner
    from .memory_planner import MemoryEstimate as MemoryPlan
    from .neuron_scope import NeuronScope
    from .parallel_engine import GLAPipelineBlock, MoERoutingBlock, ParallelEngine
    from .resource_manager import (
        AllotmentPlan,
        DynamicResourceManager,
        HardwareProfile,
        LiveVRAMMonitor,
        VRAMBudgetExceededError,
    )
    from .staging import FP8StagingBuffer, ParameterStager, StagingBuffer
    from .streaming import (
        LayerStreamingEngine,
        StreamingConfig,
        apply_layer_streaming_if_requested,
        enable_layer_streaming,
    )
    from .vram import AutoOffloader, VRAMProfiler
except ImportError:
    CUDAGraphEngine = None
    NeuronScope = None
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
    "CUDAGraphEngine",
    "NeuronScope",
    "ParallelEngine",
    "GLAPipelineBlock",
    "MoERoutingBlock",
]
