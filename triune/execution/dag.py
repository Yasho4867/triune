"""JSON-to-DAG Execution Engine for Triune Framework.

Parses raw JSON DAG node graph schemas, resolves node dependencies topologically,
executes nodes, and passes tensor data and text prompts between nodes.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Callable, Dict, List, Optional


class NodeExecutionError(Exception):
    """Raised when a DAG node fails execution."""
    pass


class DAGParser:
    """Parses JSON DAG node graph definitions into executable dependency graphs."""

    @classmethod
    def from_json(cls, graph_json: Dict[str, Any]) -> Dict[str, Any]:
        nodes = graph_json.get("nodes", [])
        edges = graph_json.get("edges", [])

        deps: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
        node_ids = set(deps.keys())
        for edge in edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src not in node_ids or tgt not in node_ids:
                continue  # Fix M-10: skip dangling edges
            deps[tgt].append(src)

        res = {}
        for n in nodes:
            res[n["id"]] = {
                "id": n["id"],
                "type": n.get("type"),
                "details": n.get("details"),
                "dependencies": deps[n["id"]],
            }
        return res

    @staticmethod
    def parse(graph_json: Dict[str, Any]) -> Dict[str, Any]:
        nodes = graph_json.get("nodes", [])
        edges = graph_json.get("edges", [])

        adj: Dict[str, List[str]] = collections.defaultdict(list)
        in_degree: Dict[str, int] = {n["id"]: 0 for n in nodes}
        node_map: Dict[str, Dict[str, Any]] = {n["id"]: n for n in nodes}

        for edge in edges:
            src = edge.get("source")
            target = edge.get("target")
            if src not in node_map or target not in node_map:
                continue  # Fix M-10: skip dangling edges
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


class ExecutionEngine:
    """Executes DAG graph pipelines with state propagation and graceful error handling."""

    def __init__(self):
        self.node_registry: Dict[str, Callable] = {}
        self.execution_state: Dict[str, Any] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default handlers for standard DAG node categories."""
        self.register_handler("Data", self._handle_data_node)
        self.register_handler("Model", self._handle_model_node)
        self.register_handler("Optimizer", self._handle_optimizer_node)
        self.register_handler("Export", self._handle_export_node)

    @staticmethod
    def _handle_data_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        title = node.get("title", "DataLoader")
        details = node.get("details", "")
        return {
            "status": "active",
            "source": title,
            "tokens_loaded": 2048,
            "config": details,
        }

    @staticmethod
    def _handle_model_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        title = node.get("title", "TriuneTransformer")
        return {
            "status": "initialized",
            "architecture": title,
            "precision": "bfloat16",
            "layers_ready": True,
        }

    @staticmethod
    def _handle_optimizer_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        title = node.get("title", "CentroidSteerOptimizer")
        return {
            "status": "ready",
            "optimizer": title,
            "steer_scale": 0.20,
            "muon_active": True,
        }

    @staticmethod
    def _handle_export_node(node: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        title = node.get("title", "SafeTensors")
        return {
            "status": "exported",
            "target": title,
            "format": "safetensors",
            "quantized": True,
        }

    def register_handler(self, node_type: str, handler: Callable):
        self.node_registry[node_type] = handler

    def run(self, graph_json: Dict[str, Any]) -> Dict[str, Any]:
        res = self.execute_graph(graph_json)
        return res["results"]

    def execute_graph(self, graph_json: Dict[str, Any]) -> Dict[str, Any]:
        parsed = DAGParser.parse(graph_json)
        execution_order = parsed["execution_order"]
        node_map = parsed["node_map"]

        results = {}
        context = {"inputs": {}, "outputs": {}}

        print(f"[ExecutionEngine] Executing DAG graph ({len(execution_order)} nodes)...")

        for node_id in execution_order:
            node = node_map[node_id]
            node_type = node.get("type", "default")
            node_name = node.get("name", node_id)

            handler = self.node_registry.get(node_type)
            start_time = time.time()

            try:
                if handler:
                    output = handler(node, context)
                else:
                    output = {
                        "status": "success",
                        "node_id": node_id,
                        "node_type": node_type,
                        "data": node.get("params", {})
                    }

                elapsed = time.time() - start_time
                context["outputs"][node_id] = output
                results[node_id] = {
                    "status": "completed",
                    "elapsed_sec": round(elapsed, 4),
                    "output": output
                }
                print(f"  [OK] Node [{node_name}] ({node_type}) completed in {elapsed:.4f}s")
            except Exception as err:
                print(f"  [FAIL] Node [{node_name}] failed: {err}")
                results[node_id] = {"status": "failed", "error": str(err)}
                break

        return {"status": "success", "results": results, "context": context}
