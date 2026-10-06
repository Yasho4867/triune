"""Unit tests for Strongly Typed Port DAG Schema and Real Execution Contracts."""

import math
import unittest
import torch

from triune.plugins.node_schema import (
    NodeExecutionType,
    Port,
    PortType,
    is_port_compatible,
)
from triune.plugins.registry import global_registry
from triune.execution.dag import DAGParser, ExecutionEngine, NodeExecutionError, _sanitize_for_json


class TestTypedDAGSchema(unittest.TestCase):
    """Test suite for typed port contracts, compatibility, and DAG execution."""

    def test_port_compatibility_rules(self):
        """Verify strict and polymorphic port compatibility rules."""
        # Exact match
        self.assertTrue(is_port_compatible(PortType.TENSOR, PortType.TENSOR))
        self.assertTrue(is_port_compatible(PortType.LOSS, PortType.LOSS))
        self.assertTrue(is_port_compatible(PortType.TEXT, PortType.TEXT))

        # Loss is a Tensor (subtyping compatibility)
        self.assertTrue(is_port_compatible(PortType.LOSS, PortType.TENSOR))

        # ANY compatibility
        self.assertTrue(is_port_compatible(PortType.ANY, PortType.TENSOR))
        self.assertTrue(is_port_compatible(PortType.TENSOR, PortType.ANY))

        # Incompatible ports
        self.assertFalse(is_port_compatible(PortType.TEXT, PortType.TENSOR))
        self.assertFalse(is_port_compatible(PortType.DATASET_STREAM, PortType.OPTIMIZER_HANDLE))
        self.assertFalse(is_port_compatible(PortType.MODEL_HANDLE, PortType.LOSS))

    def test_catalog_node_registration_and_types(self):
        """Verify global registry exposes 30 built-in nodes with typed contracts."""
        catalog = global_registry.get_catalog()
        self.assertGreaterEqual(len(catalog), 30)

        # Check execution contracts
        exec_nodes = [n for n in catalog if n.get("execution_type") == NodeExecutionType.EXECUTION.value]
        spec_nodes = [n for n in catalog if n.get("execution_type") == NodeExecutionType.SPECIFICATION.value]

        self.assertGreater(len(exec_nodes), 5)
        self.assertGreater(len(spec_nodes), 10)

        # Verify key nodes
        model_node = global_registry.get_node("TriuneTransformer")
        self.assertIsNotNone(model_node)
        self.assertEqual(model_node["execution_type"], NodeExecutionType.EXECUTION.value)

        fp8_node = global_registry.get_node("FP8Linear")
        self.assertIsNotNone(fp8_node)
        self.assertEqual(fp8_node["execution_type"], NodeExecutionType.SPECIFICATION.value)

    def test_dag_parser_type_mismatch_rejection(self):
        """Verify DAGParser rejects incompatible connections between typed ports."""
        graph = {
            "nodes": [
                {"id": "node_text", "type": "Data", "title": "Data"},
                {"id": "node_optim", "type": "CentroidSteerOptimizer", "title": "CentroidSteerOptimizer"}
            ],
            "edges": [
                {
                    "source": "node_text",
                    "target": "node_optim",
                    "source_port": "text_sample",  # type: text
                    "target_port": "loss"           # type: loss
                }
            ]
        }

        with self.assertRaises(NodeExecutionError) as ctx:
            DAGParser.parse(graph)
        self.assertIn("Type mismatch", str(ctx.exception))

    def test_dag_parser_cycle_detection(self):
        """Verify DAGParser detects cycles in execution order."""
        graph = {
            "nodes": [
                {"id": "n1", "type": "Data"},
                {"id": "n2", "type": "Model"}
            ],
            "edges": [
                {"source": "n1", "target": "n2"},
                {"source": "n2", "target": "n1"}
            ]
        }
        with self.assertRaises(NodeExecutionError) as ctx:
            DAGParser.parse(graph)
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_real_execution_dataflow(self):
        """Execute a genuine PyTorch pipeline: Tokenizer -> Dataloader -> Model -> Loss -> Optimizer."""
        engine = ExecutionEngine()
        events = []
        engine.register_event_callback(lambda ev: events.append(ev))

        graph = {
            "nodes": [
                {"id": "tok", "type": "BPETokenizer", "title": "BPETokenizer"},
                {"id": "dl", "type": "CyclingDataLoader", "title": "CyclingDataLoader"},
                {"id": "mod", "type": "TriuneTransformer", "title": "TriuneTransformer", "params": {"hidden_dim": 64, "num_layers": 2, "vocab_size": 256}},
                {"id": "loss", "type": "FastCrossEntropy", "title": "FastCrossEntropy"},
                {"id": "opt", "type": "CentroidSteerOptimizer", "title": "CentroidSteerOptimizer"},
                {"id": "eval", "type": "PerplexityEvaluator", "title": "PerplexityEvaluator"}
            ],
            "edges": [
                {"source": "tok", "target": "dl", "source_port": "tokens_tensor", "target_port": "dataset"},
                {"source": "dl", "target": "mod", "source_port": "batch_inputs", "target_port": "input_ids"},
                {"source": "mod", "target": "loss", "source_port": "logits", "target_port": "logits"},
                {"source": "dl", "target": "loss", "source_port": "batch_targets", "target_port": "targets"},
                {"source": "loss", "target": "opt", "source_port": "loss", "target_port": "loss"},
                {"source": "mod", "target": "opt", "source_port": "model_handle", "target_port": "model_handle"},
                {"source": "mod", "target": "eval", "source_port": "model_handle", "target_port": "model_handle"}
            ]
        }

        res = engine.execute_graph(graph)
        self.assertEqual(res["status"], "success")

        results = res["results"]
        # Verify real loss was computed
        loss_res = results["loss"]["output"]
        self.assertTrue(loss_res["real_computation"])
        self.assertIsInstance(loss_res["loss_val"], (float, int))
        self.assertGreater(loss_res["loss_val"], 0.0)

        # Verify real perplexity was computed
        eval_res = results["eval"]["output"]
        self.assertTrue(eval_res["real_computation"])
        self.assertAlmostEqual(eval_res["perplexity"], round(math.exp(eval_res["val_loss"]), 2), places=1)

        # Verify optimizer executed step
        opt_res = results["opt"]["output"]
        self.assertTrue(opt_res["real_computation"])
        self.assertEqual(opt_res["step_completed"], 1)

        # Verify context is fully sanitized for JSON
        import json
        json_str = json.dumps(res["context"])
        self.assertIsInstance(json_str, str)

        # Verify event callbacks fired
        event_types = [e["event"] for e in events]
        self.assertIn("node_started", event_types)
        self.assertIn("node_finished", event_types)
        self.assertEqual(event_types.count("node_started"), 6)
        self.assertEqual(event_types.count("node_finished"), 6)

    def test_json_sanitizer(self):
        """Verify _sanitize_for_json converts torch tensors and models into clean JSON objects."""
        raw = {
            "tensor_scalar": torch.tensor(3.14159),
            "tensor_multi": torch.randn(2, 3),
            "module": torch.nn.Linear(10, 5),
            "nested": [torch.tensor(1.0), {"t": torch.zeros(1, 1)}],
            "primitive": "hello",
            "num": 42
        }
        sanitized = _sanitize_for_json(raw)
        import json
        serialized = json.dumps(sanitized)
        self.assertIsInstance(serialized, str)
        self.assertEqual(sanitized["num"], 42)
        self.assertEqual(sanitized["primitive"], "hello")
        self.assertAlmostEqual(sanitized["tensor_scalar"], 3.14159, places=4)
        self.assertIn("<Tensor shape=[2, 3]", sanitized["tensor_multi"])
        self.assertIn("<Linear Module>", sanitized["module"])


if __name__ == "__main__":
    unittest.main()
