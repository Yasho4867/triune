"""Fast DAG contract tests that do not require the optional torch runtime."""

import unittest
import threading
from types import SimpleNamespace

from triune.execution.dag import DAGParser, ExecutionEngine, NodeExecutionError
from triune.plugins.node_schema import NodeExecutionType
from triune.plugins.registry import global_registry


class TestDAGExecutionContracts(unittest.TestCase):
    def test_execute_node_passes_explicit_inputs_to_handler(self):
        engine = ExecutionEngine()
        engine.register_handler(
            "InputContract",
            lambda node, context: {"status": "completed", "value": context["current_inputs"]["value"]},
        )

        result = engine.execute_node("InputContract", {"value": "connected"})

        self.assertEqual(result["value"], "connected")

    def test_rejects_duplicate_node_ids(self):
        with self.assertRaisesRegex(NodeExecutionError, "Duplicate node id"):
            DAGParser.parse({"nodes": [{"id": "same"}, {"id": "same"}], "edges": []})

    def test_rejects_missing_edge_endpoint(self):
        graph = {
            "nodes": [{"id": "source"}],
            "edges": [{"source": "source", "target": "missing"}],
        }
        with self.assertRaisesRegex(NodeExecutionError, "missing node"):
            DAGParser.parse(graph)

    def test_rejects_duplicate_incoming_port(self):
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {"source": "a", "target": "c", "source_port": "out", "target_port": "in"},
                {"source": "b", "target": "c", "source_port": "out", "target_port": "in"},
            ],
        }
        with self.assertRaisesRegex(NodeExecutionError, "more than one incoming edge"):
            DAGParser.parse(graph)

    def test_rejects_cycles(self):
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
        }
        with self.assertRaisesRegex(NodeExecutionError, "Cycle detected"):
            DAGParser.parse(graph)

    def test_studio_hierarchical_graph_ports_match_registered_schema(self):
        graph = {
            "nodes": [
                {"id": "model", "type": "Model", "title": "Triune MoE Transformer"},
                {"id": "zloss", "type": "Loss", "title": "Router Stability Z-Loss"},
            ],
            "edges": [
                {
                    "source": "model",
                    "source_port": "router_logits",
                    "target": "zloss",
                    "target_port": "router_logits",
                }
            ],
        }
        self.assertEqual(DAGParser.parse(graph)["execution_order"], ["model", "zloss"])

    def test_cli_shared_training_graph_has_valid_typed_ports(self):
        graph = {
            "nodes": [
                {"id": "stream", "type": "Data", "title": "Hugging Face Streamer"},
                {"id": "step", "type": "Evaluation", "title": "Autograd Training Step"},
            ],
            "edges": [
                {"source": "stream", "source_port": "dataset_stream", "target": "step", "target_port": "dataset_stream"},
            ],
        }
        self.assertEqual(DAGParser.parse(graph)["execution_order"], ["stream", "step"])

    def test_unknown_node_is_not_reported_as_success(self):
        result = ExecutionEngine().execute_graph(
            {"nodes": [{"id": "unknown", "type": "unregistered", "title": "Unknown"}], "edges": []}
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["results"]["unknown"]["status"], "failed")

    def test_single_unknown_node_is_not_reported_as_success(self):
        result = ExecutionEngine().execute_single_node({"id": "unknown", "type": "unregistered", "title": "Unknown"})
        self.assertEqual(result["status"], "failed")

    def test_specification_output_is_distinguished_from_execution(self):
        engine = ExecutionEngine()
        engine.register_handler(
            "TestSpecification",
            lambda node, context: {
                "status": "configured",
                "execution_type": "specification",
                "real_computation": False,
            },
        )
        result = engine.execute_graph(
            {"nodes": [{"id": "spec", "type": "TestSpecification", "title": "TestSpecification"}], "edges": []}
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"]["spec"]["status"], "specification")

    def test_runtime_context_reaches_handlers_but_is_not_serialized(self):
        runtime = object()
        observed = []
        engine = ExecutionEngine()
        engine.register_handler("RuntimeAware", lambda node, context: observed.append(context.get("_runtime_training_state")) or {
            "status": "completed", "execution_type": "execution", "real_computation": True,
        })

        result = engine.execute_graph(
            {"nodes": [{"id": "runtime", "type": "RuntimeAware", "title": "RuntimeAware"}], "edges": []},
            initial_context={"_runtime_training_state": runtime},
        )

        self.assertEqual(observed, [runtime])
        self.assertNotIn("_runtime_training_state", result["context"])
        self.assertEqual(result["status"], "success")

    def test_registered_plugin_consumes_wired_ports_and_emits_declared_ports(self):
        from triune.plugins.registry import global_registry

        name = "DAGContractMultiplyPlugin"
        previous = global_registry._nodes.get(name)

        @global_registry.register(name=name, category="Test", inputs=["value"], outputs=["result"])
        def multiply(value, factor=2):
            return {"result": value * factor}

        engine = ExecutionEngine()
        engine.register_handler("DAGContractSource", lambda node, context: (
            context.setdefault("port_data", {}).setdefault(node["id"], {}).update({"value": 7})
            or {"status": "completed", "execution_type": "execution", "real_computation": True}
        ))
        try:
            result = engine.execute_graph({
                "nodes": [
                    {"id": "source", "type": "DAGContractSource", "title": "DAGContractSource"},
                    {"id": "plugin", "type": name, "title": name, "params": {"factor": 6}},
                ],
                "edges": [{"source": "source", "source_port": "value", "target": "plugin", "target_port": "value"}],
            })
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["context"]["port_data"]["plugin"]["result"], 42)
            self.assertEqual(result["results"]["plugin"]["output"]["result"], 42)
        finally:
            if previous is None:
                global_registry._nodes.pop(name, None)
            else:
                global_registry._nodes[name] = previous

    def test_registered_plugin_missing_required_output_fails_instead_of_claiming_success(self):
        name = "DAGContractMissingOutputPlugin"
        previous = global_registry._nodes.get(name)

        @global_registry.register(name=name, inputs=[{"name": "value", "required": True}], outputs=["result"])
        def missing_output(value):
            return {"status": "completed"}

        try:
            result = ExecutionEngine().execute_graph({
                "nodes": [{"id": "plugin", "type": name, "title": name, "params": {"value": 1}}],
                "edges": [],
            })
            self.assertEqual(result["status"], "failed")
            self.assertIn("required output", result["error"])
        finally:
            if previous is None:
                global_registry._nodes.pop(name, None)
            else:
                global_registry._nodes[name] = previous

    def test_registered_plugin_missing_required_input_fails(self):
        name = "DAGContractRequiredInputPlugin"
        previous = global_registry._nodes.get(name)

        @global_registry.register(name=name, inputs=[{"name": "value", "required": True}], outputs=["result"])
        def add_one(value):
            return {"result": value + 1}

        try:
            result = ExecutionEngine().execute_graph({
                "nodes": [{"id": "plugin", "type": name, "title": name}],
                "edges": [],
            })
            self.assertEqual(result["status"], "failed")
            self.assertIn("missing required input", result["error"])
        finally:
            if previous is None:
                global_registry._nodes.pop(name, None)
            else:
                global_registry._nodes[name] = previous

    def test_subgraph_exposes_inputs_and_outputs_as_reusable_ports(self):
        engine = ExecutionEngine()
        observed = []
        engine.register_handler("SubgraphContractSource", lambda node, context: (
            context.setdefault("port_data", {}).setdefault(node["id"], {}).update({"value": 23})
            or {"status": "completed", "execution_type": "execution", "real_computation": True}
        ))
        engine.register_handler("SubgraphContractSink", lambda node, context: (
            observed.append(context.get("current_inputs", {}).get("answer"))
            or {"status": "completed", "execution_type": "execution", "real_computation": True}
        ))
        subgraph = {
            "nodes": [{
                "id": "public_value",
                "type": "SubgraphInput",
                "title": "Input: value",
                "params": {"public_name": "value"},
                "outputs": [{"name": "value", "type": "tensor"}],
            }],
            "edges": [],
            "inputs": [{"name": "value", "type": "tensor"}],
            "outputs": [{
                "name": "result", "type": "tensor",
                "source_node": "public_value", "source_port": "value",
            }],
        }
        result = engine.execute_graph({
            "nodes": [
                {"id": "source", "type": "SubgraphContractSource", "title": "Source"},
                {
                    "id": "group", "type": "Subgraph", "title": "Reusable Group",
                    "inputs": [{"name": "value", "type": "tensor"}],
                    "outputs": [{"name": "result", "type": "tensor"}],
                    "subgraph": subgraph,
                },
                {"id": "sink", "type": "SubgraphContractSink", "title": "Sink"},
            ],
            "edges": [
                {"source": "source", "source_port": "value", "target": "group", "target_port": "value"},
                {"source": "group", "source_port": "result", "target": "sink", "target_port": "answer"},
            ],
        })

        self.assertEqual(result["status"], "success")
        self.assertEqual(observed, [23])
        self.assertEqual(result["context"]["port_data"]["group"]["result"], 23)

    def test_subgraph_instance_ports_are_type_checked_by_the_parser(self):
        graph = {
            "nodes": [
                {"id": "tokenizer", "type": "BPETokenizer", "title": "BPETokenizer"},
                {
                    "id": "group", "type": "Subgraph", "title": "Typed Group",
                    "inputs": [{"name": "model", "type": "model_handle"}],
                    "outputs": [],
                    "subgraph": {"nodes": [], "edges": [], "inputs": [], "outputs": []},
                },
            ],
            "edges": [{
                "source": "tokenizer", "source_port": "tokens",
                "target": "group", "target_port": "model",
            }],
        }
        with self.assertRaisesRegex(NodeExecutionError, "Type mismatch"):
            DAGParser.parse(graph)

    def test_fp8_studio_title_resolves_to_registered_operator_ports(self):
        graph = {
            "nodes": [
                {"id": "model", "title": "Triune MoE Transformer", "type": "Model"},
                {"id": "fp8", "title": "FP8 Scaled Linear", "type": "Model"},
                {"id": "inspect", "title": "Tensor Inspector", "type": "Analysis"},
            ],
            "edges": [
                {"source": "model", "source_port": "router_logits", "target": "fp8", "target_port": "x"},
                {"source": "fp8", "source_port": "linear_out", "target": "inspect", "target_port": "x"},
            ],
        }

        parsed = DAGParser.parse(graph)

        self.assertEqual(parsed["execution_order"], ["model", "fp8", "inspect"])

    def test_scheduler_updates_connected_optimizer_and_emits_its_output_port(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
        context = {"current_inputs": {"optimizer": optimizer}, "port_data": {}}
        node = {
            "id": "schedule",
            "type": "CosineLRScheduler",
            "params": {"step": 0, "warmup_steps": 2, "total_steps": 10, "peak_lr": 1.0},
        }

        output = ExecutionEngine._handle_scheduler_node(node, context)

        self.assertEqual(output["status"], "applied")
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        self.assertEqual(context["port_data"]["schedule"]["scheduled_lr"], 0.5)

    def test_local_reader_applies_batch_and_sequence_settings_to_shared_runtime(self):
        loaded = []
        runtime = SimpleNamespace(
            is_training=False,
            batch_size=1,
            seq_len=8,
            worker_lock=threading.Lock(),
            hf_stream_buffer=[],
            load_tokenizer=lambda: object(),
            load_training_data=lambda path: loaded.append(path),
        )
        result = ExecutionEngine._handle_data_node(
            {
                "id": "local",
                "title": "Local File Reader",
                "params": {"file_path": "data/fineweb_sample.jsonl", "batch_size": 3, "seq_len": 32},
            },
            {"_runtime_training_state": runtime, "port_data": {}},
        )

        self.assertEqual(result["status"], "active")
        self.assertEqual((runtime.batch_size, runtime.seq_len), (3, 32))
        self.assertEqual(len(loaded), 1)

    def test_local_reader_does_not_replace_dataset_during_background_training(self):
        loaded = []
        runtime = SimpleNamespace(
            is_training=True,
            batch_size=1,
            seq_len=8,
            worker_lock=threading.Lock(),
            hf_stream_buffer=[],
            load_tokenizer=lambda: object(),
            load_training_data=lambda path: loaded.append(path),
        )
        result = ExecutionEngine._handle_data_node(
            {"id": "local", "title": "Local File Reader", "params": {"file_path": "data/fineweb_sample.jsonl"}},
            {"_runtime_training_state": runtime, "port_data": {}},
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(loaded, [])

    def test_local_reader_preview_mode_does_not_reconfigure_shared_runtime(self):
        loaded = []
        runtime = SimpleNamespace(
            is_training=False,
            batch_size=1,
            seq_len=8,
            worker_lock=threading.Lock(),
            hf_stream_buffer=[],
            load_tokenizer=lambda: object(),
            load_training_data=lambda path: loaded.append(path),
        )
        result = ExecutionEngine._handle_data_node(
            {
                "id": "preview",
                "title": "Local File Reader",
                "params": {
                    "file_path": "data/fineweb_sample.jsonl",
                    "configure_runtime": False,
                    "batch_size": 32,
                    "seq_len": 1024,
                },
            },
            {"_runtime_training_state": runtime, "port_data": {}},
        )

        self.assertEqual(result["status"], "active")
        self.assertEqual((runtime.batch_size, runtime.seq_len), (1, 8))
        self.assertEqual(loaded, [])

    def test_tensor_operator_nodes_are_catalogued_as_executable(self):
        executable_nodes = {
            "VectorisedGLA", "HybridAttention", "MoE_FFN", "DepthRouter",
            "FP8Linear", "FP4Linear", "RMSNorm", "FastRoPE", "FastRMSNorm",
            "LoRAAdapter", "TokenBatchInspector", "CosineLRScheduler", "CheckpointManager", "DynamicResourceManager",
        }
        for name in executable_nodes:
            with self.subTest(node=name):
                self.assertEqual(global_registry.get_node(name)["execution_type"], NodeExecutionType.EXECUTION.value)


if __name__ == "__main__":
    unittest.main()
