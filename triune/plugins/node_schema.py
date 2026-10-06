"""Node-based Visual Pipeline Schema with Strongly-Typed Ports and Execution Contracts.

Defines serializable Node, Port, and Graph Data Transfer Objects for Triune Studio GUI,
DAG Validation, and CLI pipeline execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PortType(str, Enum):
    """Strongly typed port dataflow classifications."""
    TEXT = "text"
    TENSOR = "tensor"
    DATASET_STREAM = "dataset_stream"
    MODEL_HANDLE = "model_handle"
    OPTIMIZER_HANDLE = "optimizer_handle"
    LOSS = "loss"
    METRICS = "metrics"
    CONFIG = "config"
    ANY = "any"


class NodeExecutionType(str, Enum):
    """Categorization of node runtime behavior."""
    EXECUTION = "execution"         # Runs genuine tensor operations / kernel forward-backward
    SPECIFICATION = "specification" # Configures structural hyperparams, blueprints, precision recipes


def is_port_compatible(source_type: Any, target_type: Any) -> bool:
    """Validate dataflow type compatibility between source port and target port."""
    s = getattr(source_type, "value", str(source_type)).lower()
    t = getattr(target_type, "value", str(target_type)).lower()

    if s == PortType.ANY.value or t == PortType.ANY.value:
        return True
    if s == t:
        return True

    # Valid dataflow coercions
    # 1. Dataset stream can emit raw text samples
    if s == PortType.DATASET_STREAM.value and t == PortType.TEXT.value:
        return True
    # 2. Loss is a scalar autograd Tensor and also a Metric
    if s == PortType.LOSS.value and t in (PortType.TENSOR.value, PortType.METRICS.value):
        return True
    # 3. Model output tensors can feed loss computation
    if s == PortType.TENSOR.value and t == PortType.LOSS.value:
        return True
    # 4. Metrics can accept scalar tensors
    if s == PortType.TENSOR.value and t == PortType.METRICS.value:
        return True

    return False


@dataclass
class Port:
    """A strongly typed input or output terminal on a DAG node."""
    name: str
    type: str = PortType.ANY.value
    description: str = ""
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeSchema:
    """Declarative specification for a Triune DAG visual computation node."""
    id: str
    name: str
    category: str  # "Data", "Model", "Optimizer", "Loss", "Runtime", "Evaluation", "Export", "Custom"
    execution_type: str = NodeExecutionType.EXECUTION.value
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)
    config_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    requires_cuda: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "execution_type": self.execution_type,
            "inputs": [p.to_dict() if hasattr(p, "to_dict") else asdict(p) for p in self.inputs],
            "outputs": [p.to_dict() if hasattr(p, "to_dict") else asdict(p) for p in self.outputs],
            "config_params": self.config_params,
            "description": self.description,
            "requires_cuda": self.requires_cuda,
        }
