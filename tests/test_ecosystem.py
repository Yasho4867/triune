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

    def test_module_manager_search(self):
        """Verify ModuleManager returns curated components and handles multi-source queries."""
        from triune.modules.manager import ModuleManager
        mgr = ModuleManager()
        results = mgr.search_marketplace(query="", module_type="all", source="curated")
        self.assertIn("curated", results)
        self.assertGreater(len(results["curated"]), 0)
        # Verify custom-loss-nodes is present in curated registry
        loss_mod = next((m for m in results["curated"] if m["id"] == "custom-loss-nodes"), None)
        self.assertIsNotNone(loss_mod, "custom-loss-nodes should be in curated registry")
        self.assertEqual(loss_mod["type"], "plugin")

    def test_module_manager_inspect_and_register_nodes(self):
        """Verify dynamic discovery and registration of DAG nodes from an installed module."""
        import tempfile
        import shutil
        from pathlib import Path
        from triune.modules.manager import ModuleManager
        from triune.plugins.registry import global_registry
        from triune.execution.dag import ExecutionEngine

        temp_dir = Path(tempfile.mkdtemp(prefix="triune_mod_test_"))
        try:
            # Create a sample plugin script with a custom registered DAG node
            script_path = temp_dir / "my_custom_node.py"
            script_content = '''
from triune.plugins.registry import register_node

@register_node("DynamicEchoNode", category="Plugin", description="Dynamic test node")
def dynamic_echo_handler(inputs: dict) -> dict:
    val = inputs.get("val", 0)
    return {"result": val * 10}
'''
            script_path.write_text(script_content, encoding="utf-8")

            # Also create a dummy weights file
            weights_path = temp_dir / "adapter_model.safetensors"
            weights_path.write_bytes(b"dummy_safetensors_bytes")

            mgr = ModuleManager()
            inspection = mgr.inspect_directory(temp_dir)
            self.assertGreater(inspection["file_count"], 0)
            self.assertIn("adapter_model.safetensors", inspection["artifacts"]["weights"])
            self.assertIn("my_custom_node.py", inspection["artifacts"]["nodes"])

            # Test dynamic registration
            discovered = mgr.discover_and_register_nodes(temp_dir)
            self.assertIn("DynamicEchoNode", discovered)
            self.assertIn("DynamicEchoNode", global_registry._nodes)

            # Test execution of dynamically registered node via ExecutionEngine
            engine = ExecutionEngine()
            out = engine.execute_node("DynamicEchoNode", {"val": 7})
            self.assertEqual(out.get("result"), 70)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
