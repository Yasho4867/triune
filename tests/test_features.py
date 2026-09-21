"""Unit tests for Triune Framework core 5 features:
1. JSON-to-DAG Execution Engine
2. Telemetry & Event Emitter Hooks
3. Dynamic Plugin Node Registry (@register_node)
4. Smart VRAM Profiler & Dynamic Layer Offloader
5. Unified LoRA & QLoRA Fine-Tuning Abstraction
"""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import triune
from triune.execution import DAGParser, ExecutionEngine
from triune.callbacks import EventEmitter, global_emitter
from triune.plugins import NodeRegistry, register_node, global_registry
from triune.runtime import VRAMProfiler, AutoOffloader
from triune.trainer import LoRAConfig, LoRALayer, TriuneFineTuner


class NewFrameworkFeaturesTest(unittest.TestCase):
    def test_json_to_dag_execution_engine(self):
        nodes = [
            {"id": "node_1", "type": "Data", "details": "batch=4"},
            {"id": "node_2", "type": "Model", "details": "arch=MoE"},
            {"id": "node_3", "type": "Export", "details": "quant=Q4_K_M"}
        ]
        edges = [
            {"id": "e1", "source": "node_1", "target": "node_2"},
            {"id": "e2", "source": "node_2", "target": "node_3"}
        ]
        dag_dict = DAGParser.from_json({"nodes": nodes, "edges": edges})
        self.assertIn("node_1", dag_dict)
        self.assertEqual(dag_dict["node_2"]["dependencies"], ["node_1"])

        engine = ExecutionEngine()
        results = engine.run({"nodes": nodes, "edges": edges})
        self.assertEqual(len(results), 3)
        self.assertEqual(results["node_3"]["status"], "completed")

    def test_event_emitter_telemetry(self):
        emitter = EventEmitter()
        received = []

        def on_step(data):
            received.append(data)

        emitter.on("train_step", on_step)
        emitter.emit("train_step", {"step": 42, "loss": 1.25})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["step"], 42)

    def test_node_registry_decorator(self):
        @register_node("CustomFilterNode", inputs=["tensor"], outputs=["filtered_tensor"])
        def custom_node(data):
            return data

        nodes = global_registry.list_nodes()
        node_names = [n["name"] for n in nodes]
        self.assertIn("CustomFilterNode", node_names)
        schema = global_registry.get_schema("CustomFilterNode")
        self.assertEqual(schema["inputs"], ["tensor"])

    def test_vram_profiler(self):
        stats = VRAMProfiler.get_vram_stats()
        self.assertIn("total_gb", stats)
        self.assertIn("allocated_gb", stats)
        self.assertIn("oom_risk", stats)

        offloader = AutoOffloader(vram_threshold_gb=100.0)
        offloaded = offloader.check_and_offload()
        self.assertTrue(isinstance(offloaded, bool))

    def test_lora_fine_tuner(self):
        base_model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        config = LoRAConfig(r=8, lora_alpha=16, target_modules=["0", "2"])
        tuner = TriuneFineTuner(base_model, config)
        trainable_count = tuner.count_trainable_parameters()
        self.assertGreater(trainable_count, 0)


if __name__ == "__main__":
    unittest.main()
