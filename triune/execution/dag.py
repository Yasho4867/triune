"""JSON-to-DAG Execution Engine for Triune Framework.

Parses raw JSON DAG node graph schemas, resolves node dependencies topologically,
executes nodes, and passes tensor data and text prompts between nodes.
"""

from __future__ import annotations

import collections
import inspect
import json
import math
import time
from typing import Any, Callable, Dict, List, Optional


_NODE_TITLE_ALIASES = {
    "triunemoettransformer": "TriuneTransformer",
    "huggingfacestreamer": "HuggingFaceStreamer",
    "bpetokenizer": "BPETokenizer",
    "cyclingdataloader": "CyclingDataLoader",
    "routerstabilityzloss": "RouterZLoss",
    "safetensorsexporter": "SafeTensorsExport",
    "centroidsteeroptimizer": "CentroidSteerOptimizer",
    "muonmatrixoptimizer": "MuonOptimizer",
    "autogradtrainingstep": "TrainingStep",
    "localfilereader": "LocalFileReader",
    "layerstreamingengine": "LayerStreamingEngine",
    "vramprofiler": "VRAMProfiler",
    "fp8scaledlinear": "FP8Linear",
}


class NodeExecutionError(Exception):
    """Raised when a DAG node fails execution."""
    pass


class DAGParser:
    """Parses JSON DAG node graph definitions into executable dependency graphs."""

    @staticmethod
    def _node_metadata(node: Dict[str, Any], registry: Any) -> Dict[str, Any] | None:
        """Resolve a displayed node title before its broad UI category (e.g. Model/Data)."""
        node_type = node.get("type")
        # Group sockets are instance-defined, not catalog-defined. Publish their
        # declared schemas to the same parser path used for built-in/plugin nodes
        # so typed edges are checked before any nested graph is executed.
        if node_type == "Subgraph" or node_type == "SubgraphInput":
            def typed_ports(value: Any) -> list[Dict[str, Any]]:
                if not isinstance(value, list):
                    return []
                ports = []
                for item in value:
                    if isinstance(item, str):
                        ports.append({"name": item, "type": "any", "required": True})
                    elif isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]:
                        ports.append({
                            "name": item["name"],
                            "type": item.get("type", "any"),
                            "required": item.get("required", True),
                        })
                return ports

            return {
                "name": node_type,
                "typed_inputs": typed_ports(node.get("inputs", [])) if node_type == "Subgraph" else [],
                "typed_outputs": typed_ports(node.get("outputs", [])),
            }
        for candidate in (node.get("name"), node.get("title")):
            if isinstance(candidate, str) and candidate:
                metadata = registry.get_node(candidate)
                if metadata:
                    return metadata
        title = str(node.get("title") or node.get("name") or "")
        normalized = "".join(ch.lower() for ch in title if ch.isalnum())
        alias = _NODE_TITLE_ALIASES.get(normalized)
        if alias:
            return registry.get_node(alias)
        if normalized:
            for name in registry._nodes:
                candidate = "".join(ch.lower() for ch in name if ch.isalnum())
                if candidate and (normalized == candidate or normalized.startswith(candidate) or candidate in normalized):
                    return registry.get_node(name)
        return registry.get_node(node_type) if isinstance(node_type, str) else None

    @classmethod
    def _validate_graph(cls, graph_json: Dict[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        if not isinstance(graph_json, dict):
            raise NodeExecutionError("A DAG must be an object containing nodes and edges.")
        nodes = graph_json.get("nodes", [])
        edges = graph_json.get("edges", [])
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise NodeExecutionError("DAG nodes and edges must both be arrays.")

        node_map: Dict[str, Dict[str, Any]] = {}
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise NodeExecutionError(f"Node {index + 1} must be an object.")
            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id.strip():
                raise NodeExecutionError(f"Node {index + 1} must have a non-empty string id.")
            if node_id in node_map:
                raise NodeExecutionError(f"Duplicate node id: {node_id}")
            node_map[node_id] = node

        try:
            from triune.plugins.node_schema import is_port_compatible
            from triune.plugins.registry import global_registry
        except ImportError:
            is_port_compatible = None
            global_registry = None

        seen_edges: set[tuple[Any, ...]] = set()
        occupied_inputs: set[tuple[str, str]] = set()
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                raise NodeExecutionError(f"Edge {index + 1} must be an object.")
            src, target = edge.get("source"), edge.get("target")
            if src not in node_map or target not in node_map:
                raise NodeExecutionError(f"Edge {index + 1} references a missing node ({src!r} → {target!r}).")
            src_port, target_port = edge.get("source_port"), edge.get("target_port")
            if (src_port is None) != (target_port is None):
                raise NodeExecutionError(f"Edge {index + 1} must specify both source_port and target_port, or neither.")
            if src_port is not None and (not isinstance(src_port, str) or not src_port or not isinstance(target_port, str) or not target_port):
                raise NodeExecutionError(f"Edge {index + 1} port names must be non-empty strings.")
            signature = (src, src_port, target, target_port)
            if signature in seen_edges:
                raise NodeExecutionError(f"Duplicate edge: {src}:{src_port or '*'} → {target}:{target_port or '*'}")
            seen_edges.add(signature)
            if target_port is not None:
                target_key = (target, target_port)
                if target_key in occupied_inputs:
                    raise NodeExecutionError(f"Input {target}:{target_port} has more than one incoming edge.")
                occupied_inputs.add(target_key)

            if src_port is None or global_registry is None:
                continue  # Untyped legacy edges remain supported.
            src_meta = cls._node_metadata(node_map[src], global_registry)
            tgt_meta = cls._node_metadata(node_map[target], global_registry)
            if not src_meta or not tgt_meta:
                continue  # Custom nodes may publish schemas at runtime.
            src_outputs = {p["name"]: p.get("type", "any") for p in src_meta.get("typed_outputs", [])}
            tgt_inputs = {p["name"]: p.get("type", "any") for p in tgt_meta.get("typed_inputs", [])}
            if src_port not in src_outputs:
                raise NodeExecutionError(f"Node '{node_map[src].get('title', src)}' has no output port '{src_port}'.")
            if target_port not in tgt_inputs:
                raise NodeExecutionError(f"Node '{node_map[target].get('title', target)}' has no input port '{target_port}'.")
            if is_port_compatible and not is_port_compatible(src_outputs[src_port], tgt_inputs[target_port]):
                raise NodeExecutionError(
                    f"Type mismatch: cannot connect '{src_port}' ({src_outputs[src_port]}) on "
                    f"'{node_map[src].get('title', src)}' to '{target_port}' ({tgt_inputs[target_port]}) on "
                    f"'{node_map[target].get('title', target)}'."
                )
        return nodes, edges, node_map

    @classmethod
    def from_json(cls, graph_json: Dict[str, Any]) -> Dict[str, Any]:
        nodes, edges, node_map = cls._validate_graph(graph_json)
        deps: Dict[str, List[str]] = {node_id: [] for node_id in node_map}
        node_ids = set(node_map)
        for edge in edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src in node_ids and tgt in node_ids:
                deps[tgt].append(src)

        res = {}
        for node_id, node in node_map.items():
            res[node_id] = {
                "id": node_id,
                "type": node.get("type"),
                "details": node.get("details"),
                "dependencies": deps[node_id],
            }
        return res

    @classmethod
    def parse(cls, graph_json: Dict[str, Any]) -> Dict[str, Any]:
        nodes, edges, node_map = cls._validate_graph(graph_json)

        adj: Dict[str, List[str]] = collections.defaultdict(list)
        in_degree: Dict[str, int] = {n["id"]: 0 for n in nodes}

        for edge in edges:
            src = edge.get("source")
            target = edge.get("target")
            adj[src].append(target)
            in_degree[target] = in_degree.get(target, 0) + 1

        queue = collections.deque([n_id for n_id, deg in in_degree.items() if deg == 0])
        execution_order = []

        while queue:
            curr = queue.popleft()
            execution_order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(execution_order) != len(nodes):
            raise NodeExecutionError("Cycle detected in DAG node graph!")

        return {
            "execution_order": execution_order,
            "node_map": node_map,
            "adj": dict(adj)
        }


def _sanitize_for_json(val: Any) -> Any:
    """Recursively converts PyTorch Tensors, Modules, and non-serializable objects into JSON-safe representations."""
    try:
        import torch
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return round(float(val.item()), 6)
            return f"<Tensor shape={list(val.shape)} dtype={str(val.dtype)}>"
        if isinstance(val, torch.nn.Module):
            return f"<{val.__class__.__name__} Module>"
        if isinstance(val, torch.optim.Optimizer):
            return f"<{val.__class__.__name__} Optimizer>"
    except ImportError:
        pass

    if isinstance(val, dict):
        return {str(k): _sanitize_for_json(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_sanitize_for_json(x) for x in val]
    elif isinstance(val, (int, float, bool, type(None))):
        if isinstance(val, float):
            import math
            if math.isnan(val) or math.isinf(val):
                return str(val)
        return val
    elif isinstance(val, str):
        return val
    else:
        return str(val)


def _first_not_none(*args: Any) -> Any:
    """Return the first argument that is not None, avoiding ambiguous Tensor boolean evaluation."""
    for a in args:
        if a is not None:
            return a
    return None


def _accepted_kwargs(signature: inspect.Signature, values: Dict[str, Any]) -> Dict[str, Any]:
    """Filter configuration values to a callable's named keyword parameters."""
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(values)
    return {
        name: values[name]
        for name, parameter in signature.parameters.items()
        if name in values and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


class ExecutionEngine:
    """Executes DAG graph pipelines with state propagation and graceful error handling."""

    def __init__(self):
        self.node_registry: Dict[str, Callable] = {}
        self.execution_state: Dict[str, Any] = {}
        self.event_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._register_default_handlers()

    @staticmethod
    def _node_params(node: Dict[str, Any]) -> Dict[str, Any]:
        """Merge displayed detail fields with explicit params/config (explicit wins)."""
        params: Dict[str, Any] = {}
        details = node.get("details", "")
        if isinstance(details, str):
            for line in details.splitlines():
                key, sep, raw_value = line.partition("=")
                if not sep or not key.strip():
                    continue
                raw_value = raw_value.strip()
                try:
                    params[key.strip()] = json.loads(raw_value)
                except (ValueError, TypeError):
                    params[key.strip()] = raw_value
        explicit_params = node.get("params")
        if isinstance(explicit_params, dict):
            params.update(explicit_params)
        config = node.get("config")
        if isinstance(config, dict):
            for key, value in config.items():
                params[key] = value.get("value", value) if isinstance(value, dict) else value
        return params

    def _resolve_handler(self, node: Dict[str, Any]) -> Callable | None:
        """Prefer an actual node name/title over broad UI categories such as `Model`."""
        for candidate in (node.get("name"), node.get("title")):
            if isinstance(candidate, str) and candidate in self.node_registry:
                return self.node_registry[candidate]
        title = str(node.get("title") or node.get("name") or "")
        normalized_title = "".join(ch.lower() for ch in title if ch.isalnum())
        alias = _NODE_TITLE_ALIASES.get(normalized_title)
        if alias and alias in self.node_registry:
            return self.node_registry[alias]
        for name, handler in self.node_registry.items():
            normalized_name = "".join(ch.lower() for ch in name if ch.isalnum())
            if normalized_name and (normalized_title == normalized_name or normalized_name in normalized_title):
                return handler
        node_type = node.get("type")
        if isinstance(node_type, str):
            return self.node_registry.get(node_type)
        return None

    def _resolve_plugin_handler(self, node: Dict[str, Any]) -> Callable | None:
        """Adapt a registered plugin to the same wired input/output contract as built-ins."""
        try:
            from triune.plugins.registry import global_registry
        except ImportError:
            return None
        schema = (
            global_registry.get_node(node.get("name"))
            or global_registry.get_node(node.get("title"))
            or global_registry.get_node(node.get("type"))
        )
        if not schema or not callable(schema.get("target")):
            return None
        target = schema["target"]
        output_names = [port["name"] for port in schema.get("typed_outputs", [])]
        execution_type = schema.get("execution_type", "execution")

        def _invoke_plugin(node_value: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            params = self._node_params(node_value)
            wired_inputs = context.get("current_inputs", {})
            if not isinstance(wired_inputs, dict):
                return {"status": "failed", "error": "Plugin inputs must be a port-name mapping.", "execution_type": execution_type, "real_computation": False}
            values = {**params, **wired_inputs}
            required_inputs = {
                port["name"] for port in schema.get("typed_inputs", [])
                if port.get("required", True)
            }
            missing_inputs = required_inputs.difference(values)
            if missing_inputs:
                return {
                    "status": "failed",
                    "error": f"Plugin '{schema['name']}' is missing required input port(s): {', '.join(sorted(missing_inputs))}.",
                    "execution_type": execution_type,
                    "real_computation": False,
                }
            try:
                is_class = inspect.isclass(target)
                instance = target(**_accepted_kwargs(inspect.signature(target), params)) if is_class else target
                callable_target = instance.forward if is_class and callable(getattr(instance, "forward", None)) else instance
                signature = inspect.signature(callable_target)
                names = [
                    name for name, parameter in signature.parameters.items()
                    if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                ]
                if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
                    call_kwargs = values
                elif len(names) == 1 and names[0] in {"inputs", "input_values", "values"}:
                    call_kwargs = {names[0]: values}
                else:
                    call_kwargs = {name: values[name] for name in names if name in values}
                missing = [
                    name for name, parameter in signature.parameters.items()
                    if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                    and parameter.default is inspect.Parameter.empty and name not in call_kwargs
                ]
                if missing:
                    # Preserve simple legacy plugins whose one-argument callable
                    # expects the full input/config mapping.
                    if len(names) == 1 and not wired_inputs and not is_class:
                        result = callable_target(values)
                    else:
                        raise TypeError(f"missing required plugin input(s): {', '.join(missing)}")
                else:
                    result = callable_target(**call_kwargs)
            except Exception as exc:
                return {"status": "failed", "error": f"Plugin '{schema['name']}' failed: {exc}", "execution_type": execution_type, "real_computation": False}

            if isinstance(result, dict) and result.get("status") in {"failed", "error"}:
                return {**result, "execution_type": result.get("execution_type", execution_type), "real_computation": False}

            declared_values: Dict[str, Any] = {}
            if isinstance(result, dict):
                declared_values = {name: result[name] for name in output_names if name in result}
                if not declared_values and len(output_names) == 1 and "status" not in result:
                    declared_values[output_names[0]] = result
                required_outputs = {
                    port["name"] for port in schema.get("typed_outputs", [])
                    if port.get("required", True)
                }
                missing_outputs = required_outputs.difference(declared_values)
                if missing_outputs:
                    return {
                        "status": "failed",
                        "error": f"Plugin '{schema['name']}' did not return required output port(s): {', '.join(sorted(missing_outputs))}.",
                        "execution_type": execution_type,
                        "real_computation": False,
                    }
            elif len(output_names) == 1:
                declared_values[output_names[0]] = result
            elif result is not None:
                return {"status": "failed", "error": f"Plugin '{schema['name']}' must return a mapping for its {len(output_names)} declared outputs.", "execution_type": execution_type, "real_computation": False}

            node_id = node_value.get("id", schema["name"])
            context.setdefault("port_data", {}).setdefault(node_id, {}).update(declared_values)
            payload = dict(result) if isinstance(result, dict) else {}
            payload.update(declared_values)
            payload.update({
                "status": payload.get("status", "completed" if execution_type == "execution" else "configured"),
                "plugin": schema["name"],
                "execution_type": execution_type,
                "real_computation": execution_type == "execution",
            })
            return payload

        return _invoke_plugin

    def register_event_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Register an observer callback for execution telemetry events."""
        if cb not in self.event_callbacks:
            self.event_callbacks.append(cb)

    def unregister_event_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Unregister an observer callback."""
        if cb in self.event_callbacks:
            self.event_callbacks.remove(cb)

    def _emit_event(self, event_data: Dict[str, Any]) -> None:
        """Dispatch event to registered callbacks."""
        for cb in list(self.event_callbacks):
            try:
                cb(event_data)
            except Exception:
                pass

    def _register_default_handlers(self) -> None:
        """Register default handlers for all 30 built-in DAG node types."""
        # Data & Tokenization
        self.register_handler("Data", self._handle_data_node)
        self.register_handler("HuggingFaceStreamer", self._handle_hf_stream_node)
        self.register_handler("URLDatasetStreamer", self._handle_url_stream_node)
        self.register_handler("LocalFileReader", self._handle_data_node)
        self.register_handler("BPETokenizer", self._handle_tokenizer_node)
        self.register_handler("CyclingDataLoader", self._handle_dataloader_node)

        # Model & Attention Layers
        self.register_handler("Model", self._handle_model_node)
        self.register_handler("TriuneTransformer", self._handle_model_node)
        self.register_handler("VectorisedGLA", self._handle_attention_node)
        self.register_handler("HybridAttention", self._handle_attention_node)
        self.register_handler("MoE_FFN", self._handle_moe_node)
        self.register_handler("DepthRouter", self._handle_router_node)
        self.register_handler("FP8Linear", self._handle_fp8_node)
        self.register_handler("FP4Linear", self._handle_fp4_node)
        self.register_handler("TensorInspector", self._handle_tensor_inspector_node)
        self.register_handler("TokenBatchInspector", self._handle_token_batch_inspector_node)
        self.register_handler("RMSNorm", self._handle_rmsnorm_node)
        self.register_handler("LoRAAdapter", self._handle_lora_node)

        # Optimizers & Schedulers
        self.register_handler("Optimizer", self._handle_optimizer_node)
        self.register_handler("CentroidSteerOptimizer", self._handle_centroid_node)
        self.register_handler("MuonOptimizer", self._handle_muon_node)
        self.register_handler("AdamWOptimizer", self._handle_adamw_node)
        self.register_handler("CosineLRScheduler", self._handle_scheduler_node)

        # Loss Functions & Kernels
        self.register_handler("Loss", self._handle_loss_node)
        self.register_handler("FastCrossEntropy", self._handle_loss_node)
        self.register_handler("JointExitLoss", self._handle_joint_loss_node)
        self.register_handler("RouterZLoss", self._handle_zloss_node)
        self.register_handler("FastRoPE", self._handle_rope_node)
        self.register_handler("FastRMSNorm", self._handle_rmsnorm_node)

        # Runtime & System
        self.register_handler("Runtime", self._handle_runtime_node)
        self.register_handler("LayerStreamingEngine", self._handle_streaming_engine_node)
        self.register_handler("GradientAccumulator", self._handle_gradient_accumulator_node)
        self.register_handler("CheckpointManager", self._handle_checkpoint_manager_node)
        self.register_handler("DynamicResourceManager", self._handle_resource_manager_node)
        self.register_handler("VRAMProfiler", self._handle_vram_node)
        self.register_handler("PythonSandbox", self._handle_sandbox_node)
        self.register_handler("WandbLogger", self._handle_wandb_node)

        # Inference & Agents
        self.register_handler("AutoregressiveGenerator", self._handle_generator_node)
        self.register_handler("BYOKChatRouter", self._handle_byok_router_node)

        # Evaluation & Training
        self.register_handler("Subgraph", self._handle_subgraph_node)
        self.register_handler("SubgraphInput", self._handle_subgraph_input_node)
        self.register_handler("Evaluation", self._handle_evaluator_node)
        self.register_handler("TrainingStep", self._handle_training_step_node)
        self.register_handler("PerplexityEvaluator", self._handle_evaluator_node)
        self.register_handler("LoRAFineTuner", self._handle_lora_finetune_node)

        # Exporters
        self.register_handler("Export", self._handle_export_node)
        self.register_handler("SafeTensorsExport", self._handle_export_node)
        self.register_handler("GGUFExport", self._handle_gguf_node)
        self.register_handler("ONNXExport", self._handle_onnx_node)

    @staticmethod
    def _handle_hf_stream_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        d_name = params.get("dataset_name", "roneneldan/TinyStories")

        try:
            runtime_state = context.get("_runtime_training_state")
            if runtime_state is None:
                from triune.api.routes import pytorch_state as runtime_state
            if runtime_state is None:
                raise RuntimeError("The PyTorch streaming backend is unavailable.")
            if getattr(runtime_state, "is_training", False):
                raise RuntimeError("Pause background training before changing its active dataset stream.")
            res = runtime_state.connect_hf_stream(
                d_name,
                config=params.get("config") or None,
                split=params.get("split", "train"),
                text_column=params.get("text_column", "text"),
            )
            preview = res.get("sample_preview")
            if not isinstance(preview, str) or not preview.strip():
                raise RuntimeError("The connected dataset did not provide a usable text sample.")
        except Exception as exc:
            return {"status": "failed", "error": f"Could not initialize dataset stream '{d_name}': {exc}"}

        context["data_source"] = d_name
        context["is_streaming"] = True
        node_id = node.get("id", "hf_stream")
        context.setdefault("port_data", {}).setdefault(node_id, {}).update({
            "dataset_stream": d_name,
            "text_sample": preview,
        })
        return {
            "status": "connected",
            "source": f"Hugging Face: {d_name}",
            "stream_mode": "backend_stream_initialized",
            "buffered_chunks": res.get("buffered_chunks", 0),
            "preview": preview[:160],
            "execution_type": "execution",
            "real_computation": True,
        }

    @staticmethod
    def _handle_url_stream_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        url = str(params.get("url", "")).strip()
        if not url:
            return {"status": "failed", "error": "URLDatasetStreamer requires a dataset URL."}
        try:
            runtime_state = context.get("_runtime_training_state")
            if runtime_state is None:
                from triune.api.routes import pytorch_state as runtime_state
            if runtime_state is None:
                raise RuntimeError("The PyTorch streaming backend is unavailable.")
            if getattr(runtime_state, "is_training", False):
                raise RuntimeError("Pause background training before changing its active dataset stream.")
            res = runtime_state.connect_url_stream(url, text_column=params.get("text_column"))
            preview = res.get("sample_preview")
            if not isinstance(preview, str) or not preview.strip():
                raise RuntimeError("The remote dataset did not provide a usable text sample.")
        except Exception as exc:
            return {"status": "failed", "error": f"Could not initialize URL dataset stream: {exc}"}
        context["data_source"] = url
        context["is_streaming"] = True
        node_id = node.get("id", "url_stream")
        context.setdefault("port_data", {}).setdefault(node_id, {}).update({
            "dataset_stream": url,
            "text_sample": preview,
        })
        return {
            "status": "connected",
            "source": f"Remote URL: {url}",
            "stream_mode": "backend_stream_initialized",
            "buffered_chunks": res.get("buffered_chunks", 0),
            "preview": preview[:160],
            "execution_type": "execution",
            "real_computation": True,
        }

    @staticmethod
    def _handle_tokenizer_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        # Read text from input port, context, or default sample
        text = _first_not_none(context.get("current_inputs", {}).get("text"), context.get("text"))
        if text is None or text == "":
            return {"status": "failed", "error": "Tokenizer has no text input. Connect a text-producing node or provide input explicitly."}
        if isinstance(text, list):
            text = text[0] if text else ""
        if not isinstance(text, str) or not text.strip():
            return {"status": "failed", "error": "Tokenizer input must be non-empty text."}

        params = ExecutionEngine._node_params(node)
        tokenizer = None
        runtime_state = context.get("_runtime_training_state")
        if runtime_state is not None and bool(params.get("configure_runtime", True)):
            tokenizer = runtime_state.load_tokenizer()
        if tokenizer is None:
            from pathlib import Path
            from triune.data.tokenizer import load_tokenizer
            project_root = Path(__file__).resolve().parents[2]
            tok_file = Path(str(params.get("tokenizer_path", project_root / "triune_tokenizer.json"))).expanduser()
            if not tok_file.is_absolute():
                tok_file = project_root / tok_file
            if not tok_file.is_file():
                return {"status": "failed", "error": f"Configured tokenizer was not found: {tok_file}"}
            tokenizer = load_tokenizer(tok_file)
        try:
            token_ids = tokenizer.encode(text).ids
        except Exception as exc:
            return {"status": "failed", "error": f"Tokenizer could not encode the connected text: {exc}"}
        if not token_ids:
            return {"status": "failed", "error": "Tokenizer returned no token IDs for the connected text."}
        vocab_size = int(tokenizer.get_vocab_size())
        if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
            return {"status": "failed", "error": "Tokenizer produced token IDs outside its declared vocabulary."}

        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        context["tokens"] = token_tensor
        context["input_ids"] = token_tensor
        node_id = node.get("id", "tokenizer")
        context.setdefault("port_data", {}).setdefault(node_id, {})
        context["port_data"][node_id]["tokens"] = token_tensor

        return {
            "status": "ready",
            "execution_type": "execution",
            "type": "BPE Tokenizer",
            "vocab_size": vocab_size,
            "seq_len": int(token_tensor.shape[1]),
            "real_computation": True,
            "preview": str(text)[:80],
        }

    @staticmethod
    def _handle_dataloader_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        tokens = _first_not_none(context.get("current_inputs", {}).get("tokens"), context.get("tokens"))
        if tokens is None or not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.size(1) < 2:
            return {"status": "failed", "error": "DataLoader requires a rank-2 token tensor with at least two tokens per sequence."}

        params = ExecutionEngine._node_params(node)
        seq_len = params.get("seq_len")
        if seq_len is not None:
            seq_len = int(seq_len)
            if seq_len < 1:
                return {"status": "failed", "error": "DataLoader seq_len must be positive."}
            tokens = tokens[:, :seq_len + 1]
        if tokens.size(1) < 2:
            return {"status": "failed", "error": "DataLoader input is too short to create next-token targets."}

        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]
        context["input_ids"] = input_ids
        context["targets"] = targets
        node_id = node.get("id", "dataloader")
        context.setdefault("port_data", {}).setdefault(node_id, {})
        context["port_data"][node_id]["input_ids"] = input_ids
        context["port_data"][node_id]["targets"] = targets

        return {
            "status": "active",
            "execution_type": "execution",
            "batch_size": int(input_ids.size(0)),
            "seq_len": int(input_ids.size(1)),
            "grad_accum_steps": 4,
            "real_computation": True,
        }

    @staticmethod
    def _handle_attention_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        params = ExecutionEngine._node_params(node)
        hidden_states = _first_not_none(context.get("current_inputs", {}).get("hidden_states"), context.get("hidden_states"))
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3 or not hidden_states.is_floating_point():
            return {"status": "failed", "execution_type": "execution", "error": "GLA attention requires floating hidden_states shaped [batch, sequence, hidden_dim].", "real_computation": False}
        dim = int(params.get("hidden_dim", hidden_states.size(-1)))
        heads = int(params.get("num_heads", 4))
        if dim != hidden_states.size(-1) or heads <= 0:
            return {"status": "failed", "execution_type": "execution", "error": "Attention hidden_dim must match the input tensor and num_heads must be positive.", "real_computation": False}
        try:
            from triune.model.attention import HybridAttention
            layer = HybridAttention(dim=dim, heads=heads).to(device=hidden_states.device, dtype=hidden_states.dtype)
            output, cache = layer(hidden_states)
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"GLA attention failed: {exc}", "real_computation": False}
        node_id = node.get("id", "attention")
        context["attention_layer"] = layer
        context.setdefault("port_data", {}).setdefault(node_id, {}).update({"attention_out": output})
        return {"status": "computed", "execution_type": "execution", "layer": node.get("title", "VectorisedGLA"), "heads": heads, "head_dim": dim // heads if dim % heads == 0 else None, "output_shape": list(output.shape), "cache_initialized": cache is not None, "real_computation": True}

    @staticmethod
    def _handle_moe_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        params = ExecutionEngine._node_params(node)
        hidden_states = _first_not_none(context.get("current_inputs", {}).get("hidden_states"), context.get("hidden_states"))
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3 or not hidden_states.is_floating_point():
            return {"status": "failed", "execution_type": "execution", "error": "MoE_FFN requires floating hidden_states shaped [batch, sequence, hidden_dim].", "real_computation": False}
        num_experts = int(params.get("num_experts", 8))
        top_k = int(params.get("top_k", 2))
        if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
            return {"status": "failed", "execution_type": "execution", "error": "MoE requires 1 <= top_k <= num_experts.", "real_computation": False}
        try:
            from triune.model.moe import MoE_FFN
            layer = MoE_FFN(
                dim=hidden_states.size(-1), num_experts=num_experts, top_k=top_k,
                use_fp4=bool(params.get("use_fp4", False)), use_fp8=bool(params.get("use_fp8", False)),
                shared_expert=bool(params.get("shared_expert", True)),
            ).to(device=hidden_states.device, dtype=hidden_states.dtype)
            output = layer(hidden_states)
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"MoE_FFN execution failed: {exc}", "real_computation": False}
        node_id = node.get("id", "moe")
        context["moe_layer"] = layer
        context.setdefault("port_data", {}).setdefault(node_id, {}).update({"moe_out": output})
        return {"status": "computed", "execution_type": "execution", "num_experts": num_experts, "top_k": top_k, "output_shape": list(output.shape), "real_computation": True}

    @staticmethod
    def _handle_router_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        params = ExecutionEngine._node_params(node)
        hidden_states = _first_not_none(context.get("current_inputs", {}).get("hidden_states"), context.get("hidden_states"))
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3 or not hidden_states.is_floating_point():
            return {"status": "failed", "execution_type": "execution", "error": "DepthRouter requires floating hidden_states shaped [batch, sequence, hidden_dim].", "real_computation": False}
        try:
            from triune.model.router import GumbelSoftmaxRouter
            target_dist = params.get("target_depth_dist", (0.34, 0.33, 0.33))
            router = GumbelSoftmaxRouter(hidden_states.size(-1), target_depth_dist=target_dist).to(device=hidden_states.device, dtype=hidden_states.dtype)
            logits, routes, balance_loss = router(
                hidden_states,
                temperature=float(params.get("temperature", 1.0)),
                force_depth=params.get("force_depth"),
            )
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"DepthRouter execution failed: {exc}", "real_computation": False}
        choices = routes.argmax(dim=-1)
        node_id = node.get("id", "router")
        context["router_layer"] = router
        context.setdefault("port_data", {}).setdefault(node_id, {}).update({"exit_choice": choices, "router_logits": logits, "balance_loss": balance_loss})
        return {"status": "routed", "execution_type": "execution", "choices": choices.detach().cpu().tolist(), "target_depth_dist": [float(value) for value in target_dist], "balance_loss": float(balance_loss.detach().item()), "real_computation": True}

    @staticmethod
    def _handle_fp8_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        params = ExecutionEngine._node_params(node)
        x = _first_not_none(context.get("current_inputs", {}).get("x"), context.get("x"))
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or not x.is_floating_point():
            return {"status": "failed", "execution_type": "execution", "error": "FP8Linear requires a floating input tensor with at least one dimension.", "real_computation": False}
        in_features = int(params["in_features"]) if params.get("in_features") is not None else x.shape[-1]
        out_features = int(params["out_features"]) if params.get("out_features") is not None else in_features
        if in_features != x.shape[-1] or out_features <= 0:
            return {"status": "failed", "execution_type": "execution", "error": "FP8Linear feature dimensions do not match the input tensor.", "real_computation": False}
        try:
            from triune.model.fp8 import FP8Linear
            layer = FP8Linear(in_features, out_features, bias=bool(params.get("bias", False)), device=x.device, dtype=x.dtype)
            output = layer(x)
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"FP8Linear execution failed: {exc}", "real_computation": False}
        node_id = node.get("id", "fp8")
        context["fp8_layer"] = layer
        context.setdefault("port_data", {}).setdefault(node_id, {})["linear_out"] = output
        return {"status": "computed", "execution_type": "execution", "precision": "FP8-aware linear", "output_shape": list(output.shape), "real_computation": True}

    @staticmethod
    def _handle_fp4_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        params = ExecutionEngine._node_params(node)
        x = _first_not_none(context.get("current_inputs", {}).get("x"), context.get("x"))
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or not x.is_floating_point():
            return {"status": "failed", "execution_type": "execution", "error": "FP4Linear requires a floating input tensor with at least one dimension.", "real_computation": False}
        in_features = int(params["in_features"]) if params.get("in_features") is not None else x.shape[-1]
        out_features = int(params["out_features"]) if params.get("out_features") is not None else in_features
        if in_features != x.shape[-1] or out_features <= 0:
            return {"status": "failed", "execution_type": "execution", "error": "FP4Linear feature dimensions do not match the input tensor.", "real_computation": False}
        try:
            from triune.model.fp4 import FP4Linear
            from contextlib import nullcontext
            compute_dtype = torch.bfloat16 if x.dtype not in (torch.bfloat16, torch.float16) else x.dtype
            layer = FP4Linear(in_features, out_features, bias=bool(params.get("bias", True))).to(device=x.device, dtype=compute_dtype)
            runtime_state = context.get("_runtime_training_state")
            precision_context = runtime_state.model_autocast() if runtime_state is not None else nullcontext()
            with precision_context:
                output = layer(x.to(dtype=compute_dtype)).to(dtype=x.dtype)
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"FP4Linear execution failed: {exc}", "real_computation": False}
        node_id = node.get("id", "fp4")
        context["fp4_layer"] = layer
        context.setdefault("port_data", {}).setdefault(node_id, {})["linear_out"] = output
        return {"status": "computed", "execution_type": "execution", "precision": "NVFP4 recipe when active; supported BF16 fallback otherwise", "output_shape": list(output.shape), "real_computation": True}

    @staticmethod
    def _handle_tensor_inspector_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        tensor = _first_not_none(context.get("current_inputs", {}).get("x"), context.get("x"))
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0 or not tensor.is_floating_point():
            return {"status": "failed", "execution_type": "execution", "error": "TensorInspector requires a non-empty floating tensor on input port 'x'.", "real_computation": False}
        finite_mask = torch.isfinite(tensor)
        finite_count = int(finite_mask.sum().item())
        if finite_count == 0:
            return {"status": "failed", "execution_type": "execution", "error": "TensorInspector received no finite values.", "real_computation": False}
        finite_values = tensor[finite_mask].float()
        metrics = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
            "finite_count": finite_count,
            "nonfinite_count": int(tensor.numel()) - finite_count,
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
        }
        node_id = node.get("id", "tensor_inspector")
        context.setdefault("port_data", {}).setdefault(node_id, {})["metrics"] = metrics
        return {"status": "inspected", "execution_type": "execution", "metrics": metrics, "real_computation": True}

    @staticmethod
    def _handle_token_batch_inspector_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and summarize integer token IDs without requiring model/GPU execution."""
        import torch

        tokens = _first_not_none(context.get("current_inputs", {}).get("tokens"), context.get("tokens"))
        if not isinstance(tokens, torch.Tensor) or tokens.numel() == 0:
            return {"status": "failed", "execution_type": "execution", "error": "Token Batch Inspector requires a non-empty integer tensor on input port 'tokens'.", "real_computation": False}
        if tokens.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            return {"status": "failed", "execution_type": "execution", "error": f"Token IDs must use an integer tensor dtype, received {tokens.dtype}.", "real_computation": False}
        metrics = {
            "shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "device": str(tokens.device),
            "token_count": int(tokens.numel()),
            "unique_token_count": int(torch.unique(tokens).numel()),
            "min_token_id": int(tokens.min().item()),
            "max_token_id": int(tokens.max().item()),
            "rank": int(tokens.ndim),
            "has_next_token_pairs": bool(tokens.ndim >= 1 and tokens.shape[-1] >= 2),
        }
        node_id = node.get("id", "token_batch_inspector")
        context.setdefault("port_data", {}).setdefault(node_id, {})["metrics"] = metrics
        return {"status": "inspected", "execution_type": "execution", "metrics": metrics, "real_computation": True}

    @staticmethod
    def _handle_rmsnorm_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        params = ExecutionEngine._node_params(node)
        x = _first_not_none(context.get("current_inputs", {}).get("x"), context.get("x"))
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or not x.is_floating_point():
            return {"status": "failed", "execution_type": "execution", "error": "RMSNorm requires a floating input tensor.", "real_computation": False}
        dim = int(params["dim"]) if params.get("dim") is not None else x.shape[-1]
        if dim != x.shape[-1]:
            return {"status": "failed", "execution_type": "execution", "error": f"RMSNorm dim {dim} does not match input width {x.shape[-1]}.", "real_computation": False}
        try:
            eps = float(params.get("eps", 1e-6))
            weight = context.get("current_inputs", {}).get("weight")
            if (node.get("type") == "FastRMSNorm" or node.get("title") == "FastRMSNorm") and weight is None:
                raise ValueError("FastRMSNorm requires its connected weight tensor.")
            if weight is not None:
                if not isinstance(weight, torch.Tensor) or weight.shape != (dim,):
                    raise ValueError(f"RMSNorm weight must be a tensor shaped [{dim}].")
                from triune.kernels import fast_rmsnorm
                output = fast_rmsnorm(x, weight.to(device=x.device, dtype=x.dtype), eps)
                layer = None
            else:
                from triune.model.norms import RMSNorm
                layer = RMSNorm(dim, eps=eps).to(device=x.device, dtype=x.dtype)
                output = layer(x)
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"RMSNorm execution failed: {exc}", "real_computation": False}
        node_id = node.get("id", "rmsnorm")
        if layer is not None:
            context["rmsnorm_layer"] = layer
        context.setdefault("port_data", {}).setdefault(node_id, {})["norm_out"] = output
        return {"status": "computed", "execution_type": "execution", "kernel": "Triune RMSNorm", "output_shape": list(output.shape), "real_computation": True}

    @staticmethod
    def _handle_lora_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from triune.trainer.finetune import LoRALayer
        params = ExecutionEngine._node_params(node)
        base = _first_not_none(context.get("current_inputs", {}).get("base_layer"), context.get("current_inputs", {}).get("model_handle"), context.get("model"))
        if not isinstance(base, torch.nn.Module):
            return {"status": "failed", "execution_type": "execution", "error": "LoRAAdapter requires a connected model or base_layer.", "real_computation": False}
        rank = int(params.get("rank", 16))
        alpha = float(params.get("alpha", 32.0))
        dropout = float(params.get("dropout", 0.05))
        targets = params.get("target_modules", ("q_proj", "v_proj", "out_proj"))
        if isinstance(targets, str):
            targets = (targets,)
        try:
            if isinstance(base, torch.nn.Linear):
                adapted = base if isinstance(base, LoRALayer) else LoRALayer(base, rank=rank, alpha=alpha, dropout=dropout)
                if adapted is not base:
                    context["adapted_layer"] = adapted
            else:
                modules = dict(base.named_modules())
                targets_to_wrap = [
                    (name, module) for name, module in modules.items()
                    if name and any(str(target) in name for target in targets) and isinstance(module, torch.nn.Linear)
                ]
                if not targets_to_wrap and not any(isinstance(module, LoRALayer) for module in base.modules()):
                    raise RuntimeError(f"No supported linear modules matched LoRA targets {list(targets)}.")
                for name, module in targets_to_wrap:
                    parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = modules[parent_name] if parent_name else base
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
                    setattr(parent, attr_name, LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout))
                adapted = base
                context["model"] = base
            node_id = node.get("id", "lora_adapter")
            context.setdefault("port_data", {}).setdefault(node_id, {})["adapted_layer"] = adapted
            return {"status": "attached", "execution_type": "execution", "rank": rank, "alpha": alpha, "adapter_count": sum(1 for module in adapted.modules() if isinstance(module, LoRALayer)), "real_computation": True}
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"LoRA attachment failed: {exc}", "real_computation": False}

    @staticmethod
    def _handle_centroid_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return ExecutionEngine._handle_optimizer_node(node, context)

    @staticmethod
    def _handle_muon_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return ExecutionEngine._handle_optimizer_node(node, context)

    @staticmethod
    def _handle_adamw_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return ExecutionEngine._handle_optimizer_node(node, context)

    @staticmethod
    def _handle_scheduler_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        state = context.get("_runtime_training_state")
        step = int(params.get("step", getattr(state, "step", context.get("step", 0))))
        warmup_steps = int(params.get("warmup_steps", getattr(state, "warmup_steps", 200)))
        peak_lr = float(params.get("peak_lr", getattr(state, "lr_peak", params.get("lr", 5e-4))))
        min_lr_ratio = float(params.get("min_lr_ratio", 0.1))
        total_steps = int(params.get("total_steps", 25000))
        if step < 0 or warmup_steps < 0 or total_steps <= 0 or peak_lr < 0 or not 0 <= min_lr_ratio <= 1:
            return {"status": "failed", "execution_type": "execution", "error": "Scheduler parameters are outside valid ranges.", "real_computation": False}
        min_lr = peak_lr * min_lr_ratio
        if warmup_steps and step < warmup_steps:
            lr = peak_lr * (step + 1) / warmup_steps
        else:
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer = _first_not_none(context.get("current_inputs", {}).get("optimizer_handle"), context.get("current_inputs", {}).get("optimizer"), context.get("optimizer"))
        if optimizer is not None:
            groups = getattr(optimizer, "param_groups", None)
            if not isinstance(groups, list) or not groups:
                return {"status": "failed", "execution_type": "execution", "error": "Connected optimizer does not expose parameter groups.", "real_computation": False}
            for group in groups:
                group["lr"] = lr
        node_id = node.get("id", "scheduler")
        context.setdefault("port_data", {}).setdefault(node_id, {})["learning_rate"] = lr
        context["port_data"][node_id]["scheduled_lr"] = lr
        return {"status": "applied" if optimizer is not None else "computed", "execution_type": "execution", "schedule": "cosine_with_warmup", "step": step, "learning_rate": lr, "warmup_steps": warmup_steps, "total_steps": total_steps, "optimizer_updated": optimizer is not None, "real_computation": True}

    @staticmethod
    def _handle_loss_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        logits = _first_not_none(context.get("current_inputs", {}).get("logits"), context.get("logits"))
        targets = _first_not_none(context.get("current_inputs", {}).get("targets"), context.get("targets"))

        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            return {"status": "failed", "error": "Cross-entropy requires connected logits shaped [batch, sequence, vocabulary]."}
        if not isinstance(targets, torch.Tensor) or targets.ndim != 2:
            return {"status": "failed", "error": "Cross-entropy requires connected token targets shaped [batch, sequence]."}
        if logits.shape[:2] != targets.shape:
            return {
                "status": "failed",
                "error": f"Logit/target shape mismatch: logits {tuple(logits.shape[:2])}, targets {tuple(targets.shape)}.",
            }

        vocab_size = logits.size(-1)
        targets = torch.clamp(targets, 0, vocab_size - 1)
        lm_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        params = ExecutionEngine._node_params(node)
        loss = lm_loss
        balance_loss = context.get("current_inputs", {}).get("balance_loss")
        balance_weight = float(params.get("balance_loss_weight", 0.0))
        if balance_loss is not None:
            if not isinstance(balance_loss, torch.Tensor) or not balance_loss.requires_grad:
                return {"status": "failed", "error": "Connected balance_loss must be a differentiable tensor."}
            if balance_weight < 0:
                return {"status": "failed", "error": "balance_loss_weight must be non-negative."}
            loss = loss + balance_weight * balance_loss
        router_logits = context.get("current_inputs", {}).get("router_logits")
        z_loss_weight = float(params.get("router_z_loss_weight", 0.0))
        if router_logits is not None:
            if not isinstance(router_logits, torch.Tensor) or not router_logits.is_floating_point():
                return {"status": "failed", "error": "Connected router_logits must be a floating tensor."}
            if z_loss_weight < 0:
                return {"status": "failed", "error": "router_z_loss_weight must be non-negative."}
            loss = loss + z_loss_weight * torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        loss_val = round(float(loss.item()), 4)
        context["loss"] = loss
        node_id = node.get("id", "loss")
        context.setdefault("port_data", {}).setdefault(node_id, {})
        context["port_data"][node_id]["loss"] = loss

        outputs = {"loss": loss, "lm_loss": lm_loss}
        context["port_data"][node_id].update(outputs)
        return {
            "status": "computed",
            "execution_type": "execution",
            "loss_fn": "FastCrossEntropy",
            "loss_val": loss_val,
            "lm_loss": round(float(lm_loss.detach().item()), 4),
            "balance_loss": round(float(balance_loss.detach().item()), 6) if balance_loss is not None else 0.0,
            "router_z_loss_weight": z_loss_weight,
            "real_computation": True,
        }

    @staticmethod
    def _handle_zloss_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        router_logits = _first_not_none(context.get("current_inputs", {}).get("router_logits"), context.get("exit_logits"))
        if not isinstance(router_logits, torch.Tensor):
            return {"status": "failed", "error": "RouterZLoss requires connected router logits."}
        params = ExecutionEngine._node_params(node)
        coeff = float(params.get("coeff", 1e-3))
        z_loss_t = coeff * torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        z_val = round(float(z_loss_t.detach().item()), 6)
        node_id = node.get("id", "router_z_loss")
        context.setdefault("port_data", {}).setdefault(node_id, {})["z_loss"] = z_loss_t
        return {
            "status": "computed",
            "execution_type": "execution",
            "loss_fn": "RouterZLoss (logsumexp^2)",
            "coeff": coeff,
            "z_loss_val": z_val,
            "real_computation": True,
        }

    @staticmethod
    def _handle_runtime_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        runtime_state = context.get("_runtime_training_state")
        model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"), getattr(runtime_state, "model", None))
        if runtime_state is None and model is None:
            return {"status": "unavailable", "execution_type": "execution", "error": "Runtime status requires an active Studio runtime or connected model.", "real_computation": False}
        streaming_engine = getattr(runtime_state, "streaming_engine", None) if runtime_state is not None else getattr(model, "_streaming_engine", None)
        return {
            "status": "active",
            "execution_type": "execution",
            "engine": "Triune TrainingEngine" if runtime_state is not None else type(model).__name__,
            "model_loaded": model is not None,
            "step": getattr(runtime_state, "step", None),
            "streaming_enabled": streaming_engine is not None,
            "real_computation": True,
        }

    @staticmethod
    def _handle_streaming_engine_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        params = ExecutionEngine._node_params(node)
        runtime_state = context.get("_runtime_training_state")
        if runtime_state is not None and getattr(runtime_state, "is_training", False):
            return {"status": "failed", "execution_type": "execution", "error": "Pause background training before changing the active model's layer-streaming configuration.", "real_computation": False}
        model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"), getattr(runtime_state, "model", None))
        if model is None and runtime_state is not None:
            try:
                runtime_state.lazy_init_model()
                model = runtime_state.model
            except Exception as exc:
                return {"status": "failed", "execution_type": "execution", "error": f"Could not load the active Studio model: {exc}", "real_computation": False}
        if not isinstance(model, torch.nn.Module):
            return {"status": "failed", "execution_type": "execution", "error": "LayerStreamingEngine requires a connected live model_handle.", "real_computation": False}
        current = getattr(model, "_streaming_engine", None) or getattr(runtime_state, "streaming_engine", None)
        if current is None:
            try:
                from triune.runtime.streaming import enable_layer_streaming
                optimizer = getattr(runtime_state, "optimizer", None) if runtime_state is not None else context.get("optimizer")
                current = enable_layer_streaming(
                    model,
                    device=next(model.parameters()).device,
                    chunk_size=int(params.get("chunk_size", 1)),
                    pin_memory=params.get("pin_memory"),
                    async_prefetch=bool(params.get("prefetch", True)),
                    optimizer=optimizer,
                )
            except Exception as exc:
                return {"status": "failed", "execution_type": "execution", "error": f"Could not enable layer streaming: {exc}", "real_computation": False}
        if runtime_state is not None:
            runtime_state.streaming_engine = current
        node_id = node.get("id", "streaming")
        context.setdefault("port_data", {}).setdefault(node_id, {})["streaming_model"] = model
        return {"status": "enabled", "execution_type": "execution", "engine": type(current).__name__, "device": str(getattr(current, "device", next(model.parameters()).device)), "real_computation": True}

    @staticmethod
    def _handle_vram_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch
            params = ExecutionEngine._node_params(node)
            requested_device = torch.device(str(params.get("device", "cuda:0")))
            if requested_device.type != "cuda" or not torch.cuda.is_available():
                return {"status": "unavailable", "execution_type": "execution", "device": str(requested_device), "error": "CUDA VRAM metrics require an available CUDA device.", "real_computation": False}
            device = requested_device.index if requested_device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
            res = torch.cuda.memory_reserved(device) / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            total = props.total_memory / (1024 ** 3)
            threshold = float(params.get("alert_threshold_gb", 7.2))
            if threshold <= 0:
                raise ValueError("alert_threshold_gb must be positive")
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"VRAM profiling failed: {exc}", "real_computation": False}
        output = {
            "status": "healthy",
            "execution_type": "execution",
            "device": torch.cuda.get_device_name(device),
            "allocated_gb": round(alloc, 3),
            "reserved_gb": round(res, 3),
            "peak_allocated_gb": round(peak, 3),
            "total_gb": round(total, 3),
            "oom_risk": alloc >= threshold or res >= threshold,
            "real_computation": True,
        }
        context.setdefault("port_data", {}).setdefault(node.get("id", "vram"), {})["vram_stats"] = {key: value for key, value in output.items() if key not in {"execution_type", "real_computation"}}
        return output

    @staticmethod
    def _handle_sandbox_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        code = _first_not_none(context.get("current_inputs", {}).get("code"), params.get("code"))
        if not isinstance(code, str) or not code.strip():
            return {"status": "failed", "execution_type": "execution", "error": "PythonSandbox requires connected code text.", "real_computation": False}
        try:
            from triune.runtime.sandbox import PythonSandbox
            sandbox = PythonSandbox(timeout=int(params.get("timeout", 10)), max_memory_mb=int(params.get("max_memory_mb", 256)))
            result = sandbox.execute_code(code, params.get("inputs") if isinstance(params.get("inputs"), dict) else {})
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"Sandbox execution failed: {exc}", "real_computation": False}
        if not result.get("success"):
            return {"status": "failed", "execution_type": "execution", "error": result.get("error", "Sandbox execution failed."), "real_computation": False}
        node_id = node.get("id", "sandbox")
        context.setdefault("port_data", {}).setdefault(node_id, {}).update({"result": json.dumps(result, ensure_ascii=False), "stdout": result.get("stdout", "")})
        return {"status": "completed", "execution_type": "execution", "sandbox": "PythonSandbox", "result": result, "real_computation": True}

    @staticmethod
    def _handle_wandb_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        metrics = _first_not_none(context.get("current_inputs", {}).get("metrics"), context.get("metrics"))
        if not isinstance(metrics, dict) or not metrics:
            return {"status": "failed", "execution_type": "execution", "error": "WandbLogger requires a connected non-empty metrics object.", "real_computation": False}
        try:
            import wandb
            if wandb.run is None:
                return {"status": "failed", "execution_type": "execution", "error": "No active Weights & Biases run. Start/log in to a run before using this node.", "real_computation": False}
            wandb.log(metrics, step=params.get("step"))
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"Weights & Biases logging failed: {exc}", "real_computation": False}
        output = {"status": "logged", "execution_type": "execution", "project": wandb.run.project, "metrics_logged": list(metrics), "real_computation": True}
        context.setdefault("port_data", {}).setdefault(node.get("id", "wandb"), {})["log_status"] = output
        return output

    @staticmethod
    def _handle_joint_loss_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F
        params = ExecutionEngine._node_params(node)
        weights = [float(params.get(key, default)) for key, default in (
            ("lambda_reflex", 0.20), ("lambda_limbic", 0.30), ("lambda_cortex", 0.50)
        )]
        weight_sum = sum(weights)
        if any(weight < 0 for weight in weights) or weight_sum <= 0:
            return {"status": "failed", "error": "Joint exit loss weights must be non-negative with a positive sum."}
        exit_logits = context.get("current_inputs", {}).get("exit_logits")
        targets = context.get("current_inputs", {}).get("targets")
        if not isinstance(exit_logits, (list, tuple)) or len(exit_logits) != 3 or not all(isinstance(item, torch.Tensor) for item in exit_logits):
            return {"status": "failed", "error": "JointExitLoss requires all three connected exit logits from the model node."}
        if not isinstance(targets, torch.Tensor) or targets.ndim != 2:
            return {"status": "failed", "error": "JointExitLoss requires connected next-token targets."}
        per_exit = []
        for index, logits in enumerate(exit_logits):
            if logits.ndim != 3 or logits.shape[:2] != targets.shape:
                return {"status": "failed", "error": f"Exit {index + 1} logits do not match target shape {tuple(targets.shape)}."}
            per_exit.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)))
        joint_loss = sum((weight / weight_sum) * loss for weight, loss in zip(weights, per_exit))
        z_loss = context.get("current_inputs", {}).get("z_loss")
        if z_loss is not None:
            if not isinstance(z_loss, torch.Tensor):
                return {"status": "failed", "error": "The connected router z-loss is not a tensor."}
            joint_loss = joint_loss + z_loss
        loss_val = round(float(joint_loss.detach().item()), 4)

        node_id = node.get("id", "joint_loss")
        context.setdefault("port_data", {}).setdefault(node_id, {})
        context["loss"] = joint_loss
        context["port_data"][node_id]["joint_loss"] = joint_loss
        context["port_data"][node_id]["loss"] = joint_loss
        context["port_data"][node_id]["per_exit_loss"] = [float(loss.detach().item()) for loss in per_exit]

        return {
            "status": "computed",
            "execution_type": "execution",
            "loss": loss_val,
            "weights": {"reflex": weights[0] / weight_sum, "limbic": weights[1] / weight_sum, "cortex": weights[2] / weight_sum},
            "per_exit_loss": [round(float(loss.detach().item()), 4) for loss in per_exit],
            "z_loss_included": z_loss is not None,
            "real_computation": True,
        }

    @staticmethod
    def _handle_gradient_accumulator_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        accum_steps = int(params.get("grad_accum_steps", 4))
        batch_size = context.get("batch_size")
        seq_len = context.get("seq_len")
        eff_tokens = int(batch_size) * int(seq_len) * accum_steps if batch_size is not None and seq_len is not None else None
        if accum_steps < 1:
            return {"status": "failed", "execution_type": "specification", "error": "grad_accum_steps must be positive.", "real_computation": False}
        context["grad_accum_steps"] = accum_steps
        node_id = node.get("id", "gradient_accumulator")
        accum_status = {"grad_accum_steps": accum_steps, "effective_batch_tokens": eff_tokens, "configured_only": True}
        context.setdefault("port_data", {}).setdefault(node_id, {})["accum_status"] = accum_status
        return {
            "status": "configured",
            "execution_type": "specification",
            "grad_accum_steps": accum_steps,
            "effective_batch_tokens": eff_tokens,
            "loss_scaling": f"1/{accum_steps}",
            "configured_only": True,
            "real_computation": False,
        }

    @staticmethod
    def _handle_checkpoint_manager_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        from pathlib import Path
        import torch
        from triune.trainer.checkpoint import _atomic_save_checkpoint, compute_checkpoint_fingerprint
        params = ExecutionEngine._node_params(node)
        model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"))
        runtime_state = context.get("_runtime_training_state")
        if model is None and runtime_state is not None:
            runtime_state.lazy_init_model()
            model = runtime_state.model
        if not isinstance(model, torch.nn.Module):
            return {"status": "failed", "execution_type": "execution", "error": "CheckpointManager requires a live model_handle.", "real_computation": False}
        save_dir = Path(str(params.get("save_dir", "checkpoints"))).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        step = int(getattr(runtime_state, "step", context.get("step", 0)))
        config = dict(getattr(runtime_state, "config", {}) or {})
        spec = getattr(model, "model_spec", None)
        if spec is not None and hasattr(spec, "to_dict"):
            config.setdefault("model_spec", spec.to_dict())
        fingerprint = compute_checkpoint_fingerprint(config)
        payload = {
            "model_state": model.state_dict(),
            "step": step,
            "config": config,
            "fingerprint": fingerprint,
            "schema_version": 2,
            "total_tokens_trained": int(getattr(runtime_state, "total_tokens_trained", 0)),
            "tokens_trained": int(getattr(runtime_state, "total_tokens_trained", 0)),
        }
        optimizer = _first_not_none(
            context.get("current_inputs", {}).get("optimizer_handle"),
            getattr(runtime_state, "optimizer", None) if runtime_state is not None else None,
            context.get("optimizer"),
        )
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        target = save_dir / f"triune_step_{step}.pt"
        try:
            _atomic_save_checkpoint(payload, target)
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"Checkpoint save failed: {exc}", "real_computation": False}
        keep_last_n = max(1, int(params.get("keep_last_n", 5)))
        checkpoints = sorted(save_dir.glob("triune_step_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
        for old_path in checkpoints[keep_last_n:]:
            try:
                old_path.unlink()
            except OSError:
                pass
        node_id = node.get("id", "checkpoint")
        context.setdefault("port_data", {}).setdefault(node_id, {})["checkpoint_path"] = str(target)
        return {
            "status": "saved",
            "execution_type": "execution",
            "save_directory": str(save_dir),
            "checkpoint_path": str(target),
            "fingerprint": fingerprint,
            "atomic_write": True,
            "real_computation": True,
        }

    @staticmethod
    def _handle_resource_manager_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        try:
            import torch
            requested_device = str(params.get("target_device", "cuda:0"))
            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                return {"status": "unavailable", "execution_type": "execution", "device": "cpu", "error": "CUDA was requested but is unavailable.", "real_computation": False}
            device = torch.device(requested_device if requested_device.startswith("cuda") else "cpu")
            if device.type == "cuda":
                index = device.index if device.index is not None else torch.cuda.current_device()
                props = torch.cuda.get_device_properties(index)
                total = props.total_memory / (1024**3)
                free, _ = torch.cuda.mem_get_info(index)
                free /= 1024**3
                name = props.name
                fp8_ready = hasattr(torch, "float8_e4m3fn") and props.major >= 9
                safety_margin = float(params.get("safety_margin_gb", 1.0))
                if safety_margin < 0:
                    raise ValueError("safety_margin_gb must be non-negative")
                headroom = free - safety_margin
            else:
                total = free = None
                name = "CPU"
                fp8_ready = False
                safety_margin = float(params.get("safety_margin_gb", 1.0))
                headroom = None
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"Resource probe failed: {exc}", "real_computation": False}
        output = {
            "status": "probed",
            "execution_type": "execution",
            "device": str(device),
            "device_name": name,
            "vram_total_gb": round(total, 2) if total is not None else None,
            "vram_free_gb": round(free, 2) if free is not None else None,
            "hardware_fp8_ready": fp8_ready,
            "safety_margin_gb": safety_margin,
            "available_after_safety_margin_gb": round(headroom, 2) if headroom is not None else None,
            "safety_margin_met": headroom >= 0 if headroom is not None else None,
            "real_computation": True,
        }
        context.setdefault("port_data", {}).setdefault(node.get("id", "resource_manager"), {})["feasibility_report"] = {key: value for key, value in output.items() if key not in {"execution_type", "real_computation"}}
        return output

    @staticmethod
    def _handle_generator_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        prompt = _first_not_none(context.get("current_inputs", {}).get("prompt"), params.get("prompt"))
        if not isinstance(prompt, str) or not prompt.strip():
            return {"status": "failed", "execution_type": "execution", "error": "AutoregressiveGenerator requires a non-empty connected prompt.", "real_computation": False}
        try:
            import torch
            from triune.inference import generate_response
            model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"))
            runtime_state = context.get("_runtime_training_state")
            if model is None and runtime_state is not None:
                runtime_state.lazy_init_model()
                model = runtime_state.model
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError("Connect a live model_handle or execute this graph in the Studio model session.")
            tokenizer = runtime_state.load_tokenizer() if runtime_state is not None else None
            if tokenizer is None:
                from pathlib import Path
                from triune.data.tokenizer import load_tokenizer
                tokenizer_path = Path(str(params.get("tokenizer_path", Path(__file__).resolve().parents[2] / "triune_tokenizer.json"))).expanduser()
                if not tokenizer_path.is_absolute():
                    tokenizer_path = Path(__file__).resolve().parents[2] / tokenizer_path
                if not tokenizer_path.is_file():
                    raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
                tokenizer = load_tokenizer(tokenizer_path)
            pad_token_id = tokenizer.token_to_id("[PAD]")
            eos_token_id = tokenizer.token_to_id("[SEP]")
            if pad_token_id is None or eos_token_id is None:
                raise ValueError("Tokenizer must define [PAD] and [SEP] special tokens.")
            device = next(model.parameters()).device
            started_at = time.perf_counter()
            max_tokens = int(params.get("max_tokens", 128))
            from contextlib import nullcontext
            model_context = runtime_state.model_autocast() if runtime_state is not None and model is runtime_state.model else nullcontext()
            with model_context:
                output_text = generate_response(
                    model, tokenizer, prompt, device=device, pad_token_id=int(pad_token_id), eos_token_id=int(eos_token_id),
                    max_new_tokens=max_tokens, temperature=float(params.get("temperature", 0.7)),
                    top_k=int(params.get("top_k", 50)),
                    repetition_penalty=float(params.get("repetition_penalty", 1.2)), force_depth=params.get("force_depth"),
                )
            generated_tokens = len(tokenizer.encode(output_text).ids) if output_text else 0
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            node_id = node.get("id", "generator")
            context.setdefault("port_data", {}).setdefault(node_id, {}).update({
                "generated_text": output_text,
                "tokens_per_sec": {"tokens_per_sec": generated_tokens / elapsed, "tokens_generated": generated_tokens},
            })
            return {
                "status": "completed",
                "execution_type": "execution",
                "prompt": prompt,
                "tokens_generated": generated_tokens,
                "tokens_per_sec": round(generated_tokens / elapsed, 3),
                "generated_text": output_text,
                "output_preview": output_text[:160],
                "real_computation": True,
            }
        except Exception as e:
            return {
                "status": "failed",
                "execution_type": "execution",
                "prompt": prompt,
                "error": f"Generation did not run: {e}",
                "real_computation": False,
            }

    @staticmethod
    def _handle_rope_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        params = ExecutionEngine._node_params(node)
        q = _first_not_none(context.get("current_inputs", {}).get("q"), context.get("q"))
        k = _first_not_none(context.get("current_inputs", {}).get("k"), context.get("k"))
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or q.shape != k.shape or q.ndim != 4 or not q.is_floating_point() or not k.is_floating_point():
            return {"status": "failed", "execution_type": "execution", "error": "FastRoPE requires matching q and k tensors shaped [batch, heads, sequence, head_dim].", "real_computation": False}
        dim = int(params.get("dim", q.shape[-1]))
        if dim != q.shape[-1] or dim % 2:
            return {"status": "failed", "execution_type": "execution", "error": "FastRoPE dim must match the even final dimension of q and k.", "real_computation": False}
        try:
            from triune.model.rotary import RotaryEmbedding, apply_rotary
            rotary = RotaryEmbedding(
                dim,
                max_seq_len=int(params.get("max_seq_len", 4096)),
                base=float(params.get("base", params.get("base_theta", 10000.0))),
            ).to(q.device)
            cos, sin = rotary(q.shape[-2], q.device)
            q_rot, k_rot = apply_rotary(q, k, cos, sin)
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"FastRoPE execution failed: {exc}", "real_computation": False}
        node_id = node.get("id", "rope")
        context.setdefault("port_data", {}).setdefault(node_id, {}).update({"q_rot": q_rot, "k_rot": k_rot})
        return {"status": "computed", "execution_type": "execution", "output_shape": list(q_rot.shape), "real_computation": True}

    @staticmethod
    def _handle_byok_router_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = ExecutionEngine._node_params(node)
        provider = params.get("provider", "triune-local")
        model = params.get("model", "triune-base")
        output = {
            "status": "configured",
            "execution_type": "specification",
            "provider": provider,
            "model": model,
            "request_dispatched": False,
            "note": "Provider settings only; no remote request or credential check is performed by this node.",
            "real_computation": False,
        }
        context.setdefault("port_data", {}).setdefault(node.get("id", "byok_router"), {})["provider_config"] = {
            "provider": provider,
            "model": model,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 256),
        }
        return output

    @staticmethod
    def _handle_training_step_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        # Studio DAG training must run the same TrainingEngine step as the CLI,
        # including the configured routing, balancing, precision, accumulation,
        # and optimizer behavior. The explicit tensor/model path below remains
        # available for standalone custom DAGs.
        runtime_state = context.get("_runtime_training_state")
        has_explicit_training_inputs = any(
            key in context.get("current_inputs", {})
            for key in ("model_handle", "optimizer_handle", "tokens", "input_ids")
        )
        if runtime_state is not None and not has_explicit_training_inputs:
            if not context.get("current_inputs", {}).get("dataset_stream"):
                return {"status": "failed", "error": "Connect a dataset source before running the shared TrainingEngine step."}
            if getattr(runtime_state, "is_training", False):
                return {"status": "failed", "error": "A background training run is active. Pause it before running a DAG training step."}
            params = ExecutionEngine._node_params(node)
            try:
                if "grad_accum_steps" in params:
                    accum = int(params["grad_accum_steps"])
                    if accum < 1:
                        raise ValueError("grad_accum_steps must be a positive integer")
                    runtime_state.grad_accum_steps = accum
                if "max_grad_norm" in params:
                    runtime_state.grad_clip = float(params["max_grad_norm"])
                    if runtime_state.grad_clip < 0:
                        raise ValueError("max_grad_norm must be non-negative")
                with runtime_state.worker_lock:
                    if getattr(runtime_state, "is_training", False):
                        raise RuntimeError("A background training run became active; retry after pausing it.")
                    step = runtime_state.step_once()
                if not isinstance(step, dict) or "loss" not in step or "step" not in step:
                    raise RuntimeError("The shared training engine did not return step metrics.")
                node_id = node.get("id", "training_step")
                metrics = {
                    key: step.get(key)
                    for key in (
                        "step", "loss", "lm_loss", "router_loss", "balance_loss", "z_loss",
                        "lr", "throughput", "vram_gb", "tokens_in_step", "tokens_trained", "overflow",
                    )
                }
                context.setdefault("port_data", {}).setdefault(node_id, {}).update({
                    "step_metrics": metrics,
                    "loss": torch.tensor(float(step["loss"]), dtype=torch.float32),
                    "model_handle": runtime_state.model,
                })
                return {
                    "status": "completed",
                    "execution_type": "execution",
                    "engine": "Triune TrainingEngine (shared with CLI and Studio training)",
                    "step_completed": 1,
                    "step": int(step["step"]),
                    "loss": float(step["loss"]),
                    "lm_loss": float(step.get("lm_loss", step["loss"])),
                    "router_loss": float(step.get("router_loss", 0.0)),
                    "balance_loss": float(step.get("balance_loss", 0.0)),
                    "z_loss": float(step.get("z_loss", 0.0)),
                    "learning_rate": float(step.get("lr", 0.0)),
                    "throughput": int(step.get("throughput", 0)),
                    "tokens_in_step": int(step.get("tokens_in_step", 0)),
                    "real_computation": True,
                }
            except Exception as exc:
                return {"status": "failed", "execution_type": "execution", "error": f"Shared training step failed: {exc}", "real_computation": False}

        model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"))
        optimizer = _first_not_none(context.get("current_inputs", {}).get("optimizer_handle"), context.get("optimizer"))
        tokens = _first_not_none(context.get("current_inputs", {}).get("tokens"), context.get("current_inputs", {}).get("input_ids"), context.get("tokens"), context.get("input_ids"))
        if not isinstance(model, torch.nn.Module):
            return {"status": "failed", "error": "TrainingStep requires a connected model_handle."}
        if not isinstance(optimizer, torch.optim.Optimizer) and not callable(getattr(optimizer, "step", None)):
            return {"status": "failed", "error": "TrainingStep requires a connected optimizer_handle."}
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.size(1) < 2:
            return {"status": "failed", "error": "TrainingStep requires rank-2 token batches of at least two tokens."}

        params = ExecutionEngine._node_params(node)
        model.train()
        input_ids, targets = tokens[:, :-1], tokens[:, 1:]
        result = model(input_ids)
        logits = result[0] if isinstance(result, tuple) else result
        if logits.ndim != 3 or logits.shape[:2] != targets.shape:
            return {"status": "failed", "error": f"TrainingStep output/target shape mismatch: {tuple(logits.shape[:2])} vs {tuple(targets.shape)}."}
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        max_grad_norm = float(params.get("max_grad_norm", 1.0))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) if max_grad_norm > 0 else torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()
        loss_value = float(loss.detach().item())
        node_id = node.get("id", "training_step")
        context["loss"] = loss
        context.setdefault("port_data", {}).setdefault(node_id, {}).update({
            "step_metrics": {"loss": loss_value, "grad_norm": float(grad_norm.detach().item()), "step": 1},
            "loss": loss,
            "model_handle": model,
        })
        return {
            "status": "completed",
            "execution_type": "execution",
            "loss": round(loss_value, 4),
            "grad_norm": round(float(grad_norm.detach().item()), 4),
            "tokens": int(targets.numel()),
            "real_computation": True,
        }

    @staticmethod
    def _handle_subgraph_input_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Expose one named parent-graph value as an output in a nested graph."""
        params = ExecutionEngine._node_params(node)
        public_name = params.get("public_name")
        parent_inputs = context.get("_runtime_subgraph_inputs", {})
        if not isinstance(public_name, str) or public_name not in parent_inputs:
            return {"status": "failed", "error": f"Subgraph input '{public_name}' was not supplied."}
        node_id = node.get("id", "subgraph_input")
        value = parent_inputs[public_name]
        context.setdefault("port_data", {}).setdefault(node_id, {})["value"] = value
        return {"status": "completed", "execution_type": "execution", "real_computation": True}

    def _handle_subgraph_node(self, node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an encapsulated graph and publish its declared output ports."""
        definition = node.get("subgraph")
        if not isinstance(definition, dict) or not isinstance(definition.get("nodes"), list) or not isinstance(definition.get("edges"), list):
            return {"status": "failed", "error": "Subgraph node requires a valid nested nodes/edges graph."}
        depth = int(context.get("_runtime_subgraph_depth", 0))
        if depth >= 8:
            return {"status": "failed", "error": "Nested subgraph depth exceeds the safety limit (8)."}

        parent_inputs = context.get("current_inputs", {})
        declared_inputs = definition.get("inputs", [])
        missing_inputs = [
            item.get("name", "?") for item in declared_inputs
            if item.get("required", True) and item.get("name") not in parent_inputs
        ]
        if missing_inputs:
            return {"status": "failed", "error": f"Subgraph is missing required input(s): {', '.join(missing_inputs)}."}

        child = ExecutionEngine()
        for callback in self.event_callbacks:
            child.register_event_callback(callback)
        child_result = child.execute_graph(
            {"nodes": definition["nodes"], "edges": definition["edges"]},
            initial_context={
                "_runtime_training_state": context.get("_runtime_training_state"),
                "_runtime_subgraph_depth": depth + 1,
                "_runtime_subgraph_inputs": parent_inputs,
            },
            _return_runtime_context=True,
        )
        if child_result.get("status") == "failed":
            return {
                "status": "failed",
                "error": f"Subgraph '{node.get('title', node.get('id', 'Subgraph'))}' failed at '{child_result.get('failed_node')}': {child_result.get('error', 'nested node failed')}",
                "execution_type": "execution",
                "real_computation": False,
                "nested_results": child_result.get("results", {}),
            }

        child_context = child_result.get("_runtime_context", {})
        child_ports = child_context.get("port_data", {})
        exposed_outputs = {}
        for item in definition.get("outputs", []):
            public_name = item.get("name")
            source_node = item.get("source_node")
            source_port = item.get("source_port")
            if not public_name or not source_node or not source_port:
                return {"status": "failed", "error": "Each subgraph output must declare name, source_node, and source_port."}
            source_ports = child_ports.get(source_node, {})
            if source_port not in source_ports:
                if item.get("required", True):
                    return {"status": "failed", "error": f"Subgraph output '{public_name}' has no value from {source_node}:{source_port}."}
                continue
            exposed_outputs[public_name] = source_ports[source_port]

        node_id = node.get("id", "subgraph")
        context.setdefault("port_data", {}).setdefault(node_id, {}).update(exposed_outputs)
        return {
            "status": "completed",
            "execution_type": "execution",
            "real_computation": True,
            "group_name": node.get("title", "Subgraph"),
            "nested_nodes_executed": len(child_result.get("results", {})),
            "nested_status": child_result.get("status", "success"),
            "nested_specification_nodes": child_result.get("specification_nodes", []),
            "outputs": list(exposed_outputs),
        }

    @staticmethod
    def _handle_evaluator_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import math
        try:
            import torch
            import torch.nn.functional as F
            model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"))
            runtime_state = context.get("_runtime_training_state")
            if model is None and runtime_state is not None:
                runtime_state.lazy_init_model()
                model = runtime_state.model
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError("PerplexityEvaluator requires a connected model_handle or an active Studio model.")
            tokens = _first_not_none(context.get("current_inputs", {}).get("tokens"), context.get("current_inputs", {}).get("input_ids"), context.get("tokens"), context.get("input_ids"))
            targets = context.get("current_inputs", {}).get("targets")
            if tokens is not None and (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.numel() == 0):
                raise ValueError("Evaluation input_ids must be a non-empty rank-2 token tensor.")
            if targets is None and (not isinstance(tokens, torch.Tensor) or tokens.size(1) < 2):
                raise RuntimeError("PerplexityEvaluator requires connected targets or a token batch with at least two tokens.")
            if targets is not None and (not isinstance(targets, torch.Tensor) or targets.ndim != 2 or targets.numel() == 0):
                raise ValueError("Evaluation targets must be a non-empty rank-2 token tensor.")
            vocab_size = int(getattr(model, "vocab_size", 0))
            if vocab_size <= 0:
                raise ValueError("Evaluator could not determine the model vocabulary size.")
            device = next(model.parameters()).device
            if tokens is not None:
                tokens = tokens.to(device)
                if torch.any(tokens < 0) or torch.any(tokens >= vocab_size):
                    raise ValueError(f"Evaluation token IDs must be within the model vocabulary [0, {vocab_size - 1}].")
            if targets is None:
                targets = tokens[:, 1:]
            else:
                targets = targets.to(device)
            if torch.any(targets < 0) or torch.any(targets >= vocab_size):
                raise ValueError(f"Evaluation target IDs must be within the model vocabulary [0, {vocab_size - 1}].")

            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    eval_targets = targets.contiguous()
                    exits = context.get("current_inputs", {}).get("exit_logits")
                    per_exit_metrics = None
                    if isinstance(exits, (tuple, list)):
                        if len(exits) != 3:
                            raise ValueError("Multi-exit evaluation requires exactly Reflex, Limbic, and Cortex logits.")
                        per_exit_metrics = []
                        for name, exit_output in zip(("reflex", "limbic", "cortex"), exits):
                            if not isinstance(exit_output, torch.Tensor) or exit_output.ndim != 3 or exit_output.shape[:2] != eval_targets.shape:
                                raise ValueError(f"{name} exit logits must match token targets shaped {tuple(eval_targets.shape)}.")
                            if exit_output.size(-1) != vocab_size:
                                raise ValueError(f"{name} exit vocabulary size {exit_output.size(-1)} does not match {vocab_size}.")
                            exit_loss = F.cross_entropy(exit_output.reshape(-1, vocab_size), eval_targets.reshape(-1))
                            exit_loss_value = float(exit_loss.item())
                            per_exit_metrics.append({"exit": name, "val_loss": round(exit_loss_value, 4), "perplexity": round(math.exp(min(exit_loss_value, 20.0)), 2)})
                        val_loss_t = torch.tensor(per_exit_metrics[-1]["val_loss"], device=device)
                    else:
                        if tokens is None:
                            raise ValueError("A model_handle and tokens are required when no exit_logits are connected.")
                        res = model(tokens)
                        logits = res[0] if isinstance(res, tuple) else res
                        if logits.ndim != 3 or logits.shape[:2] not in (tokens.shape, eval_targets.shape):
                            raise ValueError(f"Model logits must match the token input or target shape, got {tuple(logits.shape)}.")
                        eval_logits = logits[:, :-1, :].contiguous() if logits.shape[:2] == tokens.shape and logits.shape[:2] != eval_targets.shape else logits
                        vocab_s = eval_logits.size(-1)
                        if vocab_s != vocab_size or eval_logits.shape[:2] != eval_targets.shape:
                            raise ValueError(f"Model output vocabulary size {vocab_s} does not match model token vocabulary size {vocab_size}.")
                        val_loss_t = F.cross_entropy(eval_logits.reshape(-1, vocab_s), eval_targets.reshape(-1))
                    val_loss = round(float(val_loss_t.item()), 4)
                    perplexity = round(math.exp(min(val_loss, 20.0)), 2)
            finally:
                model.train(was_training)

            node_id = node.get("id", "evaluator")
            context.setdefault("port_data", {}).setdefault(node_id, {}).update({
                "val_loss": val_loss,
                "perplexity": {"perplexity": perplexity, "tokens_evaluated": int(eval_targets.numel())},
                "per_exit_metrics": per_exit_metrics,
            })

            return {
                "status": "completed",
                "execution_type": "execution",
                "evaluator": "PerplexityEvaluator",
                "val_loss": val_loss,
                "perplexity": perplexity,
                "per_exit_metrics": per_exit_metrics,
                "tokens_evaluated": int(eval_targets.numel()),
                "real_computation": True,
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "execution_type": "execution",
                "evaluator": "PerplexityEvaluator",
                "val_loss": None,
                "perplexity": None,
                "note": f"Evaluator error: {e}",
                "real_computation": False,
            }

    @staticmethod
    def _handle_lora_finetune_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch
            import torch.nn.functional as F
            params = ExecutionEngine._node_params(node)
            model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"))
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError("LoRAFineTuner requires a connected live model_handle.")
            tokens = _first_not_none(context.get("current_inputs", {}).get("tokens"), context.get("current_inputs", {}).get("input_ids"), context.get("tokens"), context.get("input_ids"))
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.size(1) < 2:
                raise RuntimeError("LoRAFineTuner requires real token input shaped [batch, sequence] with sequence length >= 2.")
            vocab_size = int(getattr(model, "vocab_size", 0))
            if vocab_size <= 0 or torch.any(tokens < 0) or torch.any(tokens >= vocab_size):
                raise ValueError(f"Fine-tuning token IDs must be within the model vocabulary [0, {vocab_size - 1}].")
            from triune.trainer.finetune import LoRALayer
            rank = int(params.get("rank", 16))
            alpha = float(params.get("alpha", 32.0))
            targets = params.get("target_modules", ("q_proj", "v_proj", "out_proj"))
            if isinstance(targets, str):
                targets = (targets,)
            existing_adapters = sum(1 for module in model.modules() if isinstance(module, LoRALayer))
            if existing_adapters == 0:
                modules = dict(model.named_modules())
                target_layers = [
                    (name, module) for name, module in modules.items()
                    if name and any(str(target) in name for target in targets) and isinstance(module, torch.nn.Linear)
                ]
                if not target_layers:
                    raise RuntimeError(f"No supported linear target modules found for LoRA targets {list(targets)}.")
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                for name, module in target_layers:
                    parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = modules[parent_name] if parent_name else model
                    setattr(parent, attr_name, LoRALayer(module, rank=rank, alpha=alpha))
            adapter_params = [parameter for module in model.modules() if isinstance(module, LoRALayer) for parameter in (module.lora_A, module.lora_B)]
            if not adapter_params:
                raise RuntimeError(f"No supported linear target modules found for LoRA targets {list(targets)}.")
            # A graph may connect a separately executed LoRAAdapter node first.
            # Freeze every base parameter in either path, then explicitly leave
            # only adapter matrices trainable so this remains memory-efficient.
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in adapter_params:
                parameter.requires_grad_(True)
            learning_rate = float(params.get("lr", 2e-4))
            if learning_rate <= 0:
                raise ValueError("LoRA learning rate must be positive.")
            optimizer = torch.optim.AdamW(adapter_params, lr=learning_rate)
            model.train()
            tokens = tokens.to(next(model.parameters()).device)
            optimizer.zero_grad(set_to_none=True)
            res = model(tokens)
            logits = res[0] if isinstance(res, tuple) else res
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params, float(params.get("max_grad_norm", 1.0)))
            optimizer.step()

            final_loss = round(float(loss.item()), 4)
            node_id = node.get("id", "lora_finetune")
            context["model"] = model
            context.setdefault("port_data", {}).setdefault(node_id, {}).update({"finetuned_adapter": model, "final_loss": final_loss})
            return {
                "status": "completed",
                "execution_type": "execution",
                "finetuner": "LoRAFineTuner",
                "optimizer_steps": 1,
                "adapter_count": sum(1 for module in model.modules() if isinstance(module, LoRALayer)),
                "final_loss": final_loss,
                "adapter_saved": False,
                "real_computation": True,
            }
        except Exception as e:
            return {
                "status": "failed",
                "execution_type": "execution",
                "finetuner": "LoRAFineTuner",
                "adapter_saved": False,
                "error": str(e),
                "real_computation": False,
            }

    @staticmethod
    def _handle_gguf_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = node.get("params", {})
        quant = params.get("quantization", "Q4_K_M")
        return {
            "status": "configured",
            "execution_type": "specification",
            "format": "GGUF (llama.cpp / Ollama)",
            "quantization": quant,
            "compatible_runtimes": ["Ollama", "llama.cpp", "LM Studio"],
            "real_computation": False,
        }

    @staticmethod
    def _handle_onnx_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = node.get("params", {})
        opset = int(params.get("opset_version", 17))
        return {
            "status": "configured",
            "execution_type": "specification",
            "format": "ONNX",
            "opset": opset,
            "dynamic_batch": True,
            "acceleration": "TensorRT / DirectML",
            "real_computation": False,
        }

    @staticmethod
    def _handle_data_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        title = node.get("title", "DataLoader")
        params = ExecutionEngine._node_params(node)
        from pathlib import Path
        import json

        project_root = Path(__file__).resolve().parents[2]
        requested_path = Path(str(params.get("file_path", "data/fineweb_sample.jsonl"))).expanduser()
        chosen_file = requested_path if requested_path.is_absolute() else (project_root / requested_path)
        if not chosen_file.is_file():
            return {"status": "failed", "error": f"Dataset file does not exist: {chosen_file}"}

        sample_texts = []
        lines_count = 0
        try:
            with chosen_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    lines_count += 1
                    if len(sample_texts) < 5:
                        try:
                            data = json.loads(line)
                            text = (data.get("text", "") or data.get("input", "")) if isinstance(data, dict) else str(data)
                            if str(text).strip():
                                sample_texts.append(str(text)[:1000].strip())
                        except json.JSONDecodeError:
                            sample_texts.append(line[:1000])
        except (OSError, UnicodeError) as exc:
            return {"status": "failed", "error": f"Could not read dataset file {chosen_file}: {exc}"}
        if not sample_texts:
            return {"status": "failed", "error": f"Dataset file contains no non-empty text records: {chosen_file}"}

        context["data_samples"] = sample_texts
        context["dataset_file"] = str(chosen_file)
        sample_out = sample_texts[0]
        context["text"] = sample_out

        node_id = node.get("id", "data")
        context.setdefault("port_data", {}).setdefault(node_id, {})
        context["port_data"][node_id]["dataset"] = str(chosen_file)
        context["port_data"][node_id]["text_samples"] = sample_texts
        context["port_data"][node_id]["text"] = sample_out
        context["port_data"][node_id]["text_sample"] = sample_out
        context["port_data"][node_id]["dataset_stream"] = str(chosen_file)

        runtime_state = context.get("_runtime_training_state")
        # LocalFileReader is also useful for inspection/preview-only graphs.
        # Honor the explicit opt-out so those graphs do not replace the live
        # trainer's dataset as a surprising side effect.
        if runtime_state is not None and bool(params.get("configure_runtime", True)):
            try:
                if runtime_state.load_tokenizer() is None:
                    raise RuntimeError("No configured tokenizer is available.")
                batch_size = int(params.get("batch_size", runtime_state.batch_size))
                seq_len = int(params.get("seq_len", runtime_state.seq_len))
                if batch_size < 1 or seq_len < 2:
                    raise ValueError("LocalFileReader batch_size must be positive and seq_len must be at least 2.")
                with runtime_state.worker_lock:
                    if getattr(runtime_state, "is_training", False):
                        raise RuntimeError("Pause background training before replacing its active dataset.")
                    runtime_state.batch_size = batch_size
                    runtime_state.seq_len = seq_len
                    runtime_state.is_streaming_hf = False
                    runtime_state.hf_stream_buffer.clear()
                    runtime_state.dataset_path = str(chosen_file)
                    runtime_state.load_training_data(str(chosen_file))
            except Exception as exc:
                return {"status": "failed", "error": f"Could not prepare local training data: {exc}"}

        return {
            "status": "active",
            "execution_type": "execution",
            "source": title,
            "dataset_file": str(chosen_file),
            "samples_count": lines_count,
            "preview": sample_out,
            "real_computation": True,
        }

    @staticmethod
    def _handle_model_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        title = node.get("title") or node.get("name") or "TriuneTransformer"
        import torch
        from triune.model.base import get_model_spec, build_model_from_spec

        params = ExecutionEngine._node_params(node)
        preset_id = str(params.get("model_id") or params.get("preset") or "triune-small")
        runtime_state = context.get("_runtime_training_state")
        connected_model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"))
        if isinstance(connected_model, torch.nn.Module):
            model = connected_model
            spec = getattr(model, "model_spec", None)
        elif runtime_state is not None and not bool(params.get("use_node_config", False)):
            runtime_state.lazy_init_model()
            model = runtime_state.model
            spec = getattr(model, "model_spec", None)
        else:
            overrides: Dict[str, Any] = {"name": str(title)}
            for key in (
                "vocab_size", "hidden_dim", "num_layers", "num_heads", "head_dim",
                "router_prefix_layers", "reflex_exit_layer", "limbic_exit_layer",
                "depth_mode", "precision", "use_fp4", "use_fp8", "streaming",
            ):
                if key in params and params[key] is not None:
                    overrides[key] = params[key]
            if "num_experts" in params or "experts_routed" in params:
                overrides["num_experts"] = int(params.get("num_experts", params.get("experts_routed", 4)))
            spec = get_model_spec(preset_id, **overrides)
            model = build_model_from_spec(spec)
        precision_str = getattr(spec, "precision", getattr(runtime_state, "precision", "unknown"))
        context["model"] = model
        if spec is not None:
            context["model_spec"] = spec

        if bool(params.get("initialize_only", False)):
            node_id = node.get("id", "model")
            context["model_handle"] = model
            context.setdefault("port_data", {}).setdefault(node_id, {})["model_handle"] = model
            total_params = sum(parameter.numel() for parameter in model.parameters())
            trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            return {
                "status": "initialized",
                "execution_type": "execution",
                "architecture": title,
                "precision": getattr(spec, "precision", getattr(runtime_state, "precision", "unknown")),
                "total_params": f"{total_params / 1e6:.1f}M",
                "trainable_params": f"{trainable_params / 1e6:.1f}M",
                "layers_ready": True,
                "forward_skipped": True,
                "real_computation": True,
            }

        # Real forward pass
        tokens = _first_not_none(context.get("current_inputs", {}).get("input_ids"), context.get("input_ids"), context.get("tokens"))
        if tokens is None or not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.numel() == 0:
            return {"status": "failed", "error": "Model node requires connected input_ids shaped [batch, sequence]."}
        if torch.any(tokens < 0) or torch.any(tokens >= model.vocab_size):
            return {"status": "failed", "error": f"Input token IDs must be in [0, {model.vocab_size - 1}]."}
        tokens = tokens.to(next(model.parameters()).device)

        was_training = model.training
        if bool(params.get("evaluation_mode", False)):
            model.eval()
        balance_loss = None
        try:
            from contextlib import nullcontext
            model_context = runtime_state.model_autocast() if runtime_state is not None and model is runtime_state.model else nullcontext()
            with model_context:
                if params.get("return_all_exits"):
                    if not hasattr(model, "forward_all_exits"):
                        return {"status": "failed", "error": "This model does not expose forward_all_exits for JointExitLoss."}
                    reflex_logits, limbic_logits, cortex_logits, route_logits = model.forward_all_exits(tokens, update_stats=True)
                    logits = cortex_logits
                    exit_logits = (reflex_logits, limbic_logits, cortex_logits)
                else:
                    res = model(tokens, return_balance_loss=bool(params.get("return_balance_loss", False)))
                    if isinstance(res, tuple):
                        logits = res[0]
                        route_logits = res[1] if len(res) > 1 else None
                        balance_loss = res[2] if len(res) > 2 else None
                    else:
                        logits = res
                        route_logits = None
                    exit_logits = None
        except Exception as exc:
            return {"status": "failed", "execution_type": "execution", "error": f"Model forward pass failed: {exc}", "real_computation": False}
        finally:
            if bool(params.get("evaluation_mode", False)):
                model.train(was_training)

        context["logits"] = logits
        context["model_handle"] = model
        context["input_ids"] = tokens
        node_id = node.get("id", "model")
        context.setdefault("port_data", {}).setdefault(node_id, {})
        context["port_data"][node_id]["logits"] = logits
        context["port_data"][node_id]["model_handle"] = model
        if exit_logits is not None:
            context["port_data"][node_id]["exit_logits"] = exit_logits
        if route_logits is not None:
            context["port_data"][node_id]["router_logits"] = route_logits
            context["port_data"][node_id]["selected_depth"] = route_logits.argmax(dim=-1)
        if balance_loss is not None:
            context["port_data"][node_id]["balance_loss"] = balance_loss

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "status": "initialized",
            "execution_type": "execution",
            "architecture": title,
            "precision": precision_str,
            "total_params": f"{total_params / 1e6:.1f}M",
            "trainable_params": f"{trainable_params / 1e6:.1f}M",
            "logits_shape": list(logits.shape),
            "balance_loss": round(float(balance_loss.detach().item()), 6) if balance_loss is not None else None,
            "layers_ready": True,
            "real_computation": True,
        }

    @staticmethod
    def _handle_optimizer_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from triune.optim import CentroidSteerOptimizer, Muon

        title = str(node.get("title") or node.get("type") or "Optimizer")
        params = ExecutionEngine._node_params(node)
        model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"))
        loss = _first_not_none(context.get("current_inputs", {}).get("loss"), context.get("loss"))
        if not isinstance(model, torch.nn.Module):
            return {"status": "failed", "error": "Optimizer requires a connected model_handle."}
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return {"status": "failed", "error": "Optimizer requires a differentiable loss input; no synthetic batch is generated."}

        kind = "".join(ch.lower() for ch in title if ch.isalnum())
        lr = float(params.get("lr", params.get("learning_rate", 1e-4)))
        weight_decay = float(params.get("weight_decay", 0.01))
        if lr <= 0 or weight_decay < 0:
            return {"status": "failed", "error": "Optimizer learning rate must be positive and weight decay non-negative."}

        if "centroid" in kind or "galore" in kind:
            optimizer = CentroidSteerOptimizer(
                model,
                lr=lr,
                betas=tuple(params.get("betas", (0.9, 0.95))),
                weight_decay=weight_decay,
                rank=int(params.get("galore_rank", params.get("rank", 128))),
                update_gap=int(params.get("update_gap", 200)),
                steer_scale=float(params.get("steer_scale", 0.2)),
            )
        elif "muon" in kind:
            matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
            other_params = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]
            if not matrix_params:
                return {"status": "failed", "error": "Muon requires at least one trainable matrix parameter."}
            optimizer = Muon(
                matrix_params,
                lr=lr,
                momentum=float(params.get("momentum", 0.95)),
                weight_decay=weight_decay,
                ns_steps=int(params.get("ns_steps", 5)),
            )
            if other_params:
                # Keep embeddings, normalization vectors, and other non-matrix parameters trainable.
                optimizer._triune_aux_optimizer = torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay)
        elif "adamw" in kind:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=tuple(params.get("betas", (0.9, 0.999))),
                eps=float(params.get("eps", 1e-8)),
                weight_decay=weight_decay,
            )
        else:
            return {"status": "failed", "error": f"Unsupported optimizer node '{title}'. Choose CentroidSteer, Muon, or AdamW."}

        model.train()
        optimizer.zero_grad(set_to_none=True)
        aux_optimizer = getattr(optimizer, "_triune_aux_optimizer", None)
        if aux_optimizer is not None:
            aux_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        max_grad_norm = float(params.get("max_grad_norm", 1.0))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) if max_grad_norm > 0 else torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()
        if aux_optimizer is not None:
            aux_optimizer.step()
        loss_val = round(float(loss.detach().item()), 4)
        grad_norm_val = round(float(grad_norm.detach().item() if hasattr(grad_norm, "item") else grad_norm), 4)

        context["optimizer"] = optimizer
        context["optimizer_handle"] = optimizer
        node_id = node.get("id", "optimizer")
        context.setdefault("port_data", {}).setdefault(node_id, {})
        context["port_data"][node_id]["optimizer_handle"] = optimizer
        context["port_data"][node_id]["metrics"] = {"loss": loss_val, "grad_norm": grad_norm_val, "step": 1}
        return {
            "status": "completed",
            "execution_type": "execution",
            "optimizer": title,
            "step_completed": 1,
            "loss": loss_val,
            "grad_norm": grad_norm_val,
            "real_computation": True,
        }

    @staticmethod
    def _handle_export_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        title = node.get("title", "SafeTensors")
        from pathlib import Path
        import re

        project_root = Path(__file__).resolve().parents[2]
        params = ExecutionEngine._node_params(node)
        export_dir = project_root / "exports"
        filename = str(params.get("filename", "dag_pipeline_model.safetensors"))
        if not filename.lower().endswith(".safetensors") or Path(filename).name != filename or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.safetensors", filename, re.IGNORECASE):
            return {"status": "failed", "error": "Export filename must be a simple .safetensors filename without a directory path."}
        export_file = export_dir / filename
        if export_file.exists() and not params.get("overwrite", False):
            return {"status": "failed", "error": f"Export already exists: {export_file.name}. Set overwrite=True to replace it."}
        model = _first_not_none(context.get("current_inputs", {}).get("model_handle"), context.get("model"))
        try:
            import torch
            if not isinstance(model, torch.nn.Module):
                return {"status": "failed", "error": "SafeTensors export requires a connected model_handle."}
            export_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"status": "failed", "error": f"Could not create export directory: {exc}"}

        try:
            from triune.export.exporter import export_lora_safetensors, export_safetensors
            adapter_only = bool(params.get("adapter_only", False))
            if adapter_only:
                export_lora_safetensors(model, export_file)
            else:
                export_safetensors(model, export_file)
            size_mb = export_file.stat().st_size / (1024 * 1024)
            node_id = node.get("id", "export")
            context.setdefault("port_data", {}).setdefault(node_id, {})["safetensors_file"] = str(export_file)

            return {
                "status": "exported",
                "target": title,
                "format": "safetensors",
                "quantized": False,
                "adapter_only": adapter_only,
                "file_path": str(export_file),
                "file_size_mb": round(size_mb, 2),
                "execution_type": "execution",
                "real_computation": True,
            }
        except Exception as e:
            return {
                "status": "failed",
                "target": title,
                "format": "safetensors",
                "error": f"Model export failed: {e}",
                "execution_type": "execution",
                "real_computation": False,
            }

    def register_handler(self, node_type: str, handler: Callable):
        self.node_registry[node_type] = handler

    def execute_node(self, node_or_name: Any, inputs: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a single DAG node by name or node dictionary."""
        if context is None:
            context = {"inputs": inputs or {}, "outputs": {}, "port_data": collections.defaultdict(dict)}
        else:
            context.setdefault("inputs", {})
            context.setdefault("outputs", {})
            context.setdefault("port_data", collections.defaultdict(dict))
        current_inputs = context.setdefault("current_inputs", {})
        if inputs:
            current_inputs.update(inputs)
        if isinstance(node_or_name, str):
            node = {"id": "single_node", "type": node_or_name, "title": node_or_name, "params": inputs or {}}
        else:
            node = dict(node_or_name)
            if inputs:
                node.setdefault("params", {}).update(inputs)

        node_type = node.get("type", "default")
        handler = self._resolve_handler(node) or self._resolve_plugin_handler(node)

        if handler:
            return handler(node, context)

        return {"status": "failed", "error": f"No execution handler is registered for node '{node.get('title', node_type)}' ({node_type})."}

    def run(self, graph_json: Dict[str, Any]) -> Dict[str, Any]:
        res = self.execute_graph(graph_json)
        return res["results"]

    def execute_graph(
        self,
        graph_json: Dict[str, Any],
        *,
        initial_context: Optional[Dict[str, Any]] = None,
        _return_runtime_context: bool = False,
    ) -> Dict[str, Any]:
        parsed = DAGParser.parse(graph_json)
        execution_order = parsed["execution_order"]
        node_map = parsed["node_map"]
        edges = graph_json.get("edges", [])

        # Index incoming edges by target node
        incoming_edges: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for edge in edges:
            tgt = edge.get("target")
            if tgt:
                incoming_edges[tgt].append(edge)

        results = {}
        context = {
            "inputs": {},
            "outputs": {},
            "port_data": collections.defaultdict(dict),
            "current_inputs": {}
        }
        if initial_context:
            context.update(initial_context)
        context.setdefault("inputs", {})
        context.setdefault("outputs", {})
        context.setdefault("port_data", collections.defaultdict(dict))
        context.setdefault("current_inputs", {})

        print(f"[ExecutionEngine] Executing DAG graph ({len(execution_order)} nodes)...")

        for node_id in execution_order:
            node = node_map[node_id]
            node_type = node.get("type", "default")
            node_name = node.get("title", node.get("name", node_id))

            # Resolve incoming port data for this node
            current_inputs = {}
            for edge in incoming_edges.get(node_id, []):
                src_id = edge.get("source")
                src_port = edge.get("source_port")
                tgt_port = edge.get("target_port")

                src_ports = context["port_data"].get(src_id, {})
                if src_port and src_port in src_ports:
                    val = src_ports[src_port]
                    if tgt_port:
                        current_inputs[tgt_port] = val
                    else:
                        current_inputs[src_port] = val
                elif not src_port:
                    # Untyped legacy edge: copy all port data from source
                    current_inputs.update(src_ports)
                    if src_id in context["outputs"] and isinstance(context["outputs"][src_id], dict):
                        for k, v in context["outputs"][src_id].items():
                            if k not in current_inputs:
                                current_inputs[k] = v

            context["current_inputs"] = current_inputs

            # Resolve the actual node identity before broad UI categories (`Model`, `Data`, etc.).
            handler = self._resolve_handler(node) or self._resolve_plugin_handler(node)

            if not handler:
                t_lower = (node_type + " " + str(node.get("title", ""))).lower()
                if any(k in t_lower for k in ("hugging", "hf", "stream")):
                    handler = self._handle_hf_stream_node
                elif any(k in t_lower for k in ("token", "bpe")):
                    handler = self._handle_tokenizer_node
                elif any(k in t_lower for k in ("centroid", "galore")):
                    handler = self._handle_centroid_node
                elif "muon" in t_lower:
                    handler = self._handle_muon_node
                elif any(k in t_lower for k in ("optim", "adam")):
                    handler = self._handle_optimizer_node
                elif any(k in t_lower for k in ("schedul", "lr")):
                    handler = self._handle_scheduler_node
                elif any(k in t_lower for k in ("joint", "exitloss")):
                    handler = self._handle_joint_loss_node
                elif any(k in t_lower for k in ("accumul", "microbatch")):
                    handler = self._handle_gradient_accumulator_node
                elif any(k in t_lower for k in ("checkpoint", "ckpt")):
                    handler = self._handle_checkpoint_manager_node
                elif any(k in t_lower for k in ("resource", "feasib")):
                    handler = self._handle_resource_manager_node
                elif any(k in t_lower for k in ("generat", "autoregress", "decode")):
                    handler = self._handle_generator_node
                elif any(k in t_lower for k in ("rope", "rotary")):
                    handler = self._handle_rope_node
                elif any(k in t_lower for k in ("byok", "frontier", "llm")):
                    handler = self._handle_byok_router_node
                elif any(k in t_lower for k in ("loss", "entropy")):
                    handler = self._handle_loss_node
                elif any(k in t_lower for k in ("router", "exit")):
                    handler = self._handle_router_node
                elif any(k in t_lower for k in ("gla", "attention")):
                    handler = self._handle_attention_node
                elif any(k in t_lower for k in ("moe", "expert")):
                    handler = self._handle_moe_node
                elif any(k in t_lower for k in ("lora", "peft", "adapt")):
                    handler = self._handle_lora_node
                elif "vram" in t_lower or "profil" in t_lower:
                    handler = self._handle_vram_node
                elif "eval" in t_lower or "perplex" in t_lower:
                    handler = self._handle_evaluator_node
                elif "gguf" in t_lower:
                    handler = self._handle_gguf_node
                elif "onnx" in t_lower:
                    handler = self._handle_onnx_node
                elif "export" in t_lower or "safetensor" in t_lower:
                    handler = self._handle_export_node
                elif "model" in t_lower or "transformer" in t_lower:
                    handler = self._handle_model_node
                elif "data" in t_lower or "dataset" in t_lower:
                    handler = self._handle_data_node

            start_time = time.time()
            self._emit_event({
                "event": "node_started",
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
                "timestamp": start_time,
            })

            try:
                if handler:
                    output = handler(node, context)
                else:
                    output = {
                        "status": "failed",
                        "error": f"No execution handler is registered for node '{node_name}' ({node_type}).",
                    }

                elapsed = time.time() - start_time
                if not isinstance(output, dict):
                    output = {"status": "failed", "error": f"Node returned unsupported output type: {type(output).__name__}."}
                node_failed = output.get("status") in {"failed", "error"} or (
                    output.get("execution_type") == "execution" and output.get("real_computation") is False
                )
                if node_failed:
                    output.setdefault("status", "failed")
                    output.setdefault("error", output.get("note", "Node did not complete its declared computation."))
                context["outputs"][node_id] = output
                sanitized_output = _sanitize_for_json(output)
                output_kind = output.get("execution_type") if isinstance(output, dict) else None
                result_status = "failed" if node_failed else ("specification" if output_kind == "specification" else "completed")
                results[node_id] = {
                    "status": result_status,
                    "elapsed_sec": round(elapsed, 4),
                    "output": sanitized_output
                }
                if node_failed:
                    results[node_id]["error"] = output.get("error", "Node reported failure")

                self._emit_event({
                    "event": "node_failed" if node_failed else "node_finished",
                    "node_id": node_id,
                    "node_name": node_name,
                    "node_type": node_type,
                    "elapsed_sec": round(elapsed, 4),
                    "output": sanitized_output,
                    "timestamp": time.time(),
                })
                if node_failed:
                    print(f"  [FAIL] Node [{node_name}] ({node_type}) reported failure: {output.get('error')}")
                    break
                else:
                    print(f"  [OK] Node [{node_name}] ({node_type}) completed in {elapsed:.4f}s")
            except Exception as err:
                elapsed = time.time() - start_time
                print(f"  [FAIL] Node [{node_name}] failed: {err}")
                results[node_id] = {"status": "failed", "error": str(err), "elapsed_sec": round(elapsed, 4)}
                self._emit_event({
                    "event": "node_failed",
                    "node_id": node_id,
                    "node_name": node_name,
                    "node_type": node_type,
                    "elapsed_sec": round(elapsed, 4),
                    "error": str(err),
                    "timestamp": time.time(),
                })
                break

        safe_context = _sanitize_for_json({key: value for key, value in context.items() if not key.startswith("_runtime_")})
        failed_entry = next(((nid, res) for nid, res in results.items() if res.get("status") == "failed"), None)
        specification_nodes = [nid for nid, res in results.items() if res.get("status") == "specification"]
        if failed_entry:
            failed_id, failed_res = failed_entry
            response = {
                "status": "failed",
                "failed_node": failed_id,
                "error": failed_res.get("error", "Node execution failed"),
                "results": results,
                "context": safe_context,
                "specification_nodes": specification_nodes,
            }
            if _return_runtime_context:
                response["_runtime_context"] = context
            return response
        response = {
            "status": "partial" if specification_nodes else "success",
            "results": results,
            "context": safe_context,
            "specification_nodes": specification_nodes,
        }
        if _return_runtime_context:
            response["_runtime_context"] = context
        return response

    def execute_single_node(self, node: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a single DAG node in isolation with immediate response telemetry."""
        ctx = context or {"outputs": {}}
        node_id = node.get("id", "single_node")
        node_name = node.get("title") or node.get("name") or node.get("type", "Node")
        node_type = node.get("type") or node.get("name", "Custom")

        handler = self._resolve_handler(node) or self._resolve_plugin_handler(node)

        if not handler:
            t_lower = (str(node_type) + " " + str(node_name)).lower()
            if any(k in t_lower for k in ("hugging", "hf", "stream")):
                handler = self._handle_hf_stream_node
            elif any(k in t_lower for k in ("token", "bpe")):
                handler = self._handle_tokenizer_node
            elif any(k in t_lower for k in ("centroid", "galore")):
                handler = self._handle_centroid_node
            elif "muon" in t_lower:
                handler = self._handle_muon_node
            elif any(k in t_lower for k in ("joint", "exitloss")):
                handler = self._handle_joint_loss_node
            elif any(k in t_lower for k in ("accumul", "microbatch")):
                handler = self._handle_gradient_accumulator_node
            elif any(k in t_lower for k in ("checkpoint", "ckpt")):
                handler = self._handle_checkpoint_manager_node
            elif any(k in t_lower for k in ("resource", "feasib")):
                handler = self._handle_resource_manager_node
            elif any(k in t_lower for k in ("generat", "autoregress")):
                handler = self._handle_generator_node
            elif any(k in t_lower for k in ("rope", "rotary")):
                handler = self._handle_rope_node
            elif any(k in t_lower for k in ("byok", "frontier")):
                handler = self._handle_byok_router_node
            elif any(k in t_lower for k in ("loss", "entropy")):
                handler = self._handle_loss_node
            elif any(k in t_lower for k in ("optim", "adam")):
                handler = self._handle_optimizer_node
            elif any(k in t_lower for k in ("router", "exit")):
                handler = self._handle_router_node
            elif any(k in t_lower for k in ("gla", "attention")):
                handler = self._handle_attention_node
            elif any(k in t_lower for k in ("moe", "expert")):
                handler = self._handle_moe_node
            elif any(k in t_lower for k in ("lora", "peft", "adapt")):
                handler = self._handle_lora_node
            elif "vram" in t_lower or "profil" in t_lower:
                handler = self._handle_vram_node
            elif "eval" in t_lower or "perplex" in t_lower:
                handler = self._handle_evaluator_node
            elif "export" in t_lower or "safetensor" in t_lower:
                handler = self._handle_export_node
            elif "model" in t_lower or "transformer" in t_lower:
                handler = self._handle_model_node
            elif "data" in t_lower or "dataset" in t_lower:
                handler = self._handle_data_node

        start_time = time.time()
        self._emit_event({
            "event": "node_started",
            "node_id": node_id,
            "node_name": node_name,
            "node_type": node_type,
            "timestamp": start_time,
        })
        try:
            if handler:
                output = handler(node, ctx)
            else:
                output = {
                    "status": "failed",
                    "error": f"No execution handler is registered for node '{node_name}' ({node_type}).",
                }
            elapsed = round(time.time() - start_time, 4)
            if not isinstance(output, dict):
                output = {"status": "failed", "error": f"Node returned unsupported output type: {type(output).__name__}."}
            node_failed = output.get("status") in {"failed", "error"} or (
                output.get("execution_type") == "execution" and output.get("real_computation") is False
            )
            sanitized_output = _sanitize_for_json(output)
            res = {
                "status": "failed" if node_failed else ("specification" if output.get("execution_type") == "specification" else "completed"),
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
                "elapsed_sec": elapsed,
                "output": sanitized_output
            }
            if node_failed:
                res["error"] = output.get("error", "Node reported failure")
            self._emit_event({
                "event": "node_failed" if node_failed else "node_finished",
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
                "elapsed_sec": elapsed,
                "output": sanitized_output,
                "timestamp": time.time(),
            })
            return res
        except Exception as err:
            elapsed = round(time.time() - start_time, 4)
            self._emit_event({
                "event": "node_failed",
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
                "elapsed_sec": elapsed,
                "error": str(err),
                "timestamp": time.time(),
            })
            return {
                "status": "failed",
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
                "elapsed_sec": elapsed,
                "error": str(err)
            }
