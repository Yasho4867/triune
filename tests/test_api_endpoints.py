"""In-process API regression tests for the Studio node graph endpoints."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from triune.api.server import create_app


class TestStudioApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_app())

    def test_plugins_endpoint_returns_serializable_nodes(self) -> None:
        response = self.client.get("/v1/plugins")
        self.assertEqual(response.status_code, 200)
        nodes = response.json()
        self.assertIsInstance(nodes, list)
        self.assertTrue(any(node["name"] == "TriuneTransformer" for node in nodes))

    def test_execute_single_specification_nodes(self) -> None:
        cases = (
            {"id": "joint-loss", "type": "JointExitLoss", "params": {"lambda_reflex": 0.2}},
            {"id": "rope", "type": "FastRoPE", "params": {"dim": 64, "max_seq_len": 2048}},
            {"id": "accumulator", "type": "GradientAccumulator", "params": {"grad_accum_steps": 4}},
        )
        for node in cases:
            with self.subTest(node=node["type"]):
                response = self.client.post("/v1/dag/execute_node", json={"node": node})
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertIn(payload["status"], {"success", "completed"})
                self.assertIn("output", payload)


    def test_dag_compile_to_training(self) -> None:
        payload = {
            "nodes": [
                {
                    "id": "node-data",
                    "type": "HuggingFaceStreamer",
                    "config": {
                        "dataset_name": {"value": "roneneldan/TinyStories"},
                        "split": {"value": "train"},
                        "config": {"value": "default"},
                        "text_column": {"value": "text"},
                    },
                },
                {
                    "id": "node-loader",
                    "type": "DataLoader",
                    "params": {
                        "batch_size": 2,
                        "seq_len": 32,
                    },
                },
                {
                    "id": "node-router",
                    "type": "HierarchicalDepthRouter",
                    "details": "depth_mode=joint\nbalance_loss_weight=0.45",
                },
                {
                    "id": "node-opt",
                    "type": "CentroidSteerOptimizer",
                    "params": {
                        "lr": 0.0003,
                        "muon_lr": 0.015,
                        "steer_scale": 0.25,
                    },
                },
                {
                    "id": "node-accum",
                    "type": "GradientAccumulator",
                    "params": {
                        "grad_accum_steps": 2,
                    },
                },
            ],
            "edges": [],
        }
        response = self.client.post("/v1/dag/compile_to_training", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        applied = data["applied_config"]
        self.assertEqual(applied.get("batch_size"), 2)
        self.assertEqual(applied.get("seq_len"), 32)
        self.assertEqual(applied.get("depth_mode"), "joint")
        self.assertEqual(applied.get("balance_loss_weight"), 0.45)
        self.assertEqual(applied.get("lr"), 0.0003)
        self.assertEqual(applied.get("muon_lr"), 0.015)
        self.assertEqual(applied.get("steer_scale"), 0.25)
        self.assertEqual(applied.get("grad_accum_steps"), 2)
        self.assertIn("dataset", applied)

        engine_state = data["current_engine_state"]
        self.assertEqual(engine_state["batch_size"], 2)
        self.assertEqual(engine_state["seq_len"], 32)
        self.assertEqual(engine_state["depth_mode"], "joint")
        self.assertEqual(engine_state["lr"], 0.0003)
        self.assertEqual(engine_state["grad_accum_steps"], 2)

        # Alias route check
        alias_response = self.client.post("/v1/model/compile_dag", json=payload)
        self.assertEqual(alias_response.status_code, 200)
        self.assertEqual(alias_response.json()["status"], "success")

    def test_live_optimizer_update(self) -> None:
        payload = {
            "lr": 0.00015,
            "muon_lr": 0.012,
            "centroid_lr": 0.0002,
            "steer_scale": 0.35,
            "momentum": 0.92,
        }
        response = self.client.post("/v1/training/optimizer", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertAlmostEqual(data["lr"], 0.00015)
        self.assertAlmostEqual(data["muon_lr"], 0.012)
        self.assertAlmostEqual(data["centroid_lr"], 0.0002)
        self.assertAlmostEqual(data["steer_scale"], 0.35)
        self.assertAlmostEqual(data["momentum"], 0.92)

    def test_router_temperature_update(self) -> None:
        payload = {
            "temperature": 0.85,
            "balance_loss_weight": 0.4,
        }
        response = self.client.post("/v1/model/router/temperature", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertAlmostEqual(data["temperature"], 0.85)
        self.assertAlmostEqual(data["balance_loss_weight"], 0.4)

    def test_single_training_step(self) -> None:
        from triune.api.routes import pytorch_state
        pytorch_state.batch_size = 1
        pytorch_state.seq_len = 16
        pytorch_state.grad_accum_steps = 1
        pytorch_state.is_streaming_hf = False
        pytorch_state.dataset_path = "data/fineweb_sample.jsonl"
        pytorch_state.load_training_data("data/fineweb_sample.jsonl")

        response = self.client.post("/v1/training/step", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("step", data)
        self.assertIn("loss", data)
        self.assertGreaterEqual(data["step"], 1)
        self.assertGreaterEqual(data["loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
