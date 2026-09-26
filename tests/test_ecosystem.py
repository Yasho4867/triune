"""Verification test suite for Triune Ecosystem Phase 1 modules."""

import sys
import unittest
import torch

import triune
from triune import (
    Agent,
    MemoryPlanner,
    MultiAgentOrchestrator,
    ProviderManager,
    PythonSandbox,
    TelemetryCallback,
    TriuneModel,
    export_safetensors,
    load_model,
    register_model,
    register_node,
)
from triune.api.server import HAS_FASTAPI, create_app
from triune.plugins import list_registered_nodes


class TestTriuneEcosystem(unittest.TestCase):

    def test_memory_planner(self):
        config = triune.build_config({})
        estimate = MemoryPlanner.estimate_vram(config, target_vram_gb=8.0)
        self.assertGreater(estimate.total_params, 0)
        self.assertIn(estimate.recommended_batch_size, [1, 2, 4])
        self.assertIsNotNone(estimate.recommended_precision)

    def test_sandbox(self):
        sandbox = PythonSandbox()
        result = sandbox.execute_code("res = a + b", locals_dict={"a": 15, "b": 25})
        self.assertEqual(result["res"], 40)

    def test_sandbox_security(self):
        """Verify sandbox rejects dangerous operations."""
        sandbox = PythonSandbox(timeout=5)
        
        # Test: import os should fail or be restricted
        result = sandbox.execute_code('import os; os.system("echo pwned")')
        self.assertFalse(result.get('success', False), "Sandbox should reject importing os")
        
        # Test: file access should fail or be isolated
        result = sandbox.execute_code('open("/etc/passwd").read()')
        self.assertFalse(result.get('success', True), "Sandbox should not allow arbitrary file reads")
        
        # Test: timeout enforcement
        result = sandbox.execute_code('import time; time.sleep(100)')
        self.assertFalse(result.get('success', True), "Sandbox should enforce timeout")

    def test_model_zoo(self):
        model = load_model("triune-small")
        self.assertIsNotNone(model)

        @register_model("dummy_test_model")
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 10)

            def forward(self, x, force_depth=None):
                return self.fc(x), None

        dummy = load_model("dummy_test_model")
        self.assertIsInstance(dummy, DummyModel)

    def test_plugins_registry(self):
        @register_node("Test Node", category="custom", description="A test node")
        def dummy_node(x):
            return x * 2

        nodes = list_registered_nodes()
        self.assertTrue(any(n["name"] == "Test Node" for n in nodes))

    def test_agent_engine(self):
        agent = Agent(name="Researcher", role="ML Expert", system_prompt="Analyze code")
        output = agent.run_task("Evaluate batch size")
        self.assertIn("Researcher", output)

    def test_provider_manager(self):
        pm = ProviderManager()
        pm.register_provider("openai", api_key="sk-test-key")
        prov = pm.get_provider("openai")
        self.assertIsNotNone(prov)
        self.assertEqual(prov.api_key, "sk-test-key")

    def test_api_server(self):
        if HAS_FASTAPI:
            app = create_app()
            self.assertEqual(app.title, "Triune Framework Server")

    def test_export_safetensors(self):
        model = torch.nn.Linear(10, 10)
        path = export_safetensors(model, "scratch/model_test.safetensors")
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
