"""In-process API regression tests for the Studio node graph endpoints."""

from __future__ import annotations

import unittest
import inspect
import tempfile
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from triune.api.server import create_app


class TestStudioApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The API intentionally requires bearer auth for non-loopback peers.
        # Model the local Studio client rather than Starlette's default
        # "testclient" hostname, which is not an IP address and is treated as
        # remote by that production security middleware.
        client_kwargs = {}
        if "client" in inspect.signature(TestClient).parameters:
            client_kwargs["client"] = ("127.0.0.1", 43210)
        cls.client = TestClient(create_app(), **client_kwargs)

    def test_plugins_endpoint_returns_serializable_nodes(self) -> None:
        response = self.client.get("/v1/plugins")
        self.assertEqual(response.status_code, 200)
        nodes = response.json()
        self.assertIsInstance(nodes, list)
        self.assertTrue(any(node["name"] == "TriuneTransformer" for node in nodes))

    def test_checkpoint_delete_rejects_unrelated_path_containing_checkpoint_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outside_checkpoint_dir = Path(temp_dir) / "checkpoints"
            outside_checkpoint_dir.mkdir()
            protected_file = outside_checkpoint_dir / "keep.pt"
            protected_file.write_bytes(b"not a checkpoint")

            response = self.client.post(
                "/v1/checkpoints/delete",
                json={"checkpoint_path": str(protected_file)},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "error")
            self.assertTrue(protected_file.exists())

    def test_checkpoint_export_rejects_unmanaged_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outside_checkpoint_dir = Path(temp_dir) / "checkpoints"
            outside_checkpoint_dir.mkdir()
            protected_file = outside_checkpoint_dir / "keep.pt"
            protected_file.write_bytes(b"not a checkpoint")

            response = self.client.post(
                "/v1/checkpoints/export",
                json={"checkpoint_path": str(protected_file)},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "error")
            self.assertTrue(protected_file.exists())

    def test_tokenizer_endpoint_reports_real_tokenizer(self) -> None:
        response = self.client.post(
            "/v1/tokenizer/tokenize",
            json={"text": "Triune framework"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        if payload["status"] == "unavailable":
            self.assertEqual(payload["tokens"], [])
            self.assertIn("tokenizer", payload["message"].lower())
        else:
            self.assertEqual(payload["tokenizer"], "BPE")
            self.assertTrue(all(isinstance(token["id"], int) for token in payload["tokens"]))

    def test_legacy_checkpoint_architecture_preserves_router_prefix(self) -> None:
        from triune.api.routes import _checkpoint_architecture

        class TensorShape:
            def __init__(self, *shape: int) -> None:
                self.shape = shape

        checkpoint = {
            "model_state": {
                "token_embed.weight": TensorShape(32000, 256),
                "layers.0.attn.gla.q_proj.weight": TensorShape(256, 256),
                "layers.2.exit_head.weight": TensorShape(32000, 256),
                "layers.4.exit_head.weight": TensorShape(32000, 256),
                "layers.5.norm1.weight": TensorShape(256),
                "layers.3.ffn.experts.0.0.weight": TensorShape(1024, 256),
                "layers.3.ffn.experts.3.0.weight": TensorShape(1024, 256),
            }
        }
        fallback = {
            "preset_id": "triune-base",
            "hidden_dim": 384,
            "num_layers": 8,
            "num_heads": 6,
            "head_dim": 64,
            "num_experts": 4,
            "router_prefix_layers": 1,
        }

        model_id, architecture = _checkpoint_architecture(checkpoint, fallback)
        self.assertEqual(model_id, "triune-small")
        self.assertEqual(architecture["hidden_dim"], 256)
        self.assertEqual(architecture["num_layers"], 6)
        self.assertEqual(architecture["num_heads"], 4)
        self.assertEqual(architecture["num_experts"], 4)
        self.assertEqual(architecture["router_prefix_layers"], 1)
        self.assertEqual(architecture["reflex_exit_layer"], 2)
        self.assertEqual(architecture["limbic_exit_layer"], 4)

    def test_local_chat_cancel_signals_active_generation(self) -> None:
        import threading
        from triune.api import routes

        previous = routes.pytorch_state.active_generation_cancel
        cancel_event = threading.Event()
        routes.pytorch_state.active_generation_cancel = cancel_event
        try:
            response = self.client.post("/v1/chat/cancel")
        finally:
            routes.pytorch_state.active_generation_cancel = previous

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelling")
        self.assertTrue(cancel_event.is_set())

    def test_byok_save_with_blank_fields_preserves_existing_credentials(self) -> None:
        from triune.api import routes

        class ConfigStub:
            saved = None

            def get_config(self):
                return {"byok_keys": {"openai": "existing-secret"}}

            def save_config(self, config, **kwargs):
                self.saved = config
                self.save_options = kwargs

        stub = ConfigStub()
        with patch.object(routes, "module_manager", stub):
            response = self.client.post(
                "/v1/byok/save",
                json={"keys": {"openai": "", "anthropic": "new-secret", "gemini": "", "huggingface": ""}},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "saved")
        self.assertEqual(stub.saved["byok_keys"]["openai"], "existing-secret")
        self.assertEqual(stub.saved["byok_keys"]["anthropic"], "new-secret")
        self.assertEqual(stub.save_options["force_secure_secrets"], {"byok:anthropic"})

    def test_byok_save_reports_secure_store_failure_instead_of_success(self) -> None:
        from triune.api import routes

        class ConfigStub:
            def get_config(self):
                return {"byok_keys": {}}

            def save_config(self, config, **kwargs):
                raise RuntimeError("Windows Credential Manager bridge unavailable")

        with patch.object(routes, "module_manager", ConfigStub()):
            response = self.client.post(
                "/v1/byok/save",
                json={"keys": {"openai": "sk-test-value"}},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Credential Manager", response.json()["detail"])

    def test_byok_test_performs_authenticated_provider_request(self) -> None:
        from triune.api import routes
        from unittest.mock import MagicMock

        provider_response = MagicMock()
        provider_response.status = 200
        provider_response.__enter__.return_value = provider_response
        with patch.object(routes.urllib.request, "urlopen", return_value=provider_response) as urlopen:
            response = self.client.post("/v1/byok/test", json={"provider": "openai", "key": "sk-test-value"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "valid")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-test-value")

    def test_byok_chat_missing_key_is_an_api_error_not_an_assistant_reply(self) -> None:
        from triune.api import routes

        class ConfigStub:
            def get_config(self):
                return {"byok_keys": {}}

        with patch.object(routes, "module_manager", ConfigStub()), patch.dict(
            "os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": ""}
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
            )
        self.assertEqual(response.status_code, 424)
        self.assertIn("not configured", response.json()["detail"])

    def test_byok_chat_uses_selected_current_provider_model(self) -> None:
        from triune.api import routes

        class ConfigStub:
            def get_config(self):
                return {"byok_keys": {"gemini": "configured-secret"}}

        with patch.object(routes, "module_manager", ConfigStub()), patch.object(
            routes, "_call_gemini_api", return_value="provider response"
        ) as provider_call:
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hello"}]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "provider response")
        provider_call.assert_called_once()
        self.assertEqual(provider_call.call_args.args[1], "configured-secret")
        self.assertEqual(provider_call.call_args.kwargs["model"], "gemini-3.8-flash")

    def test_byok_chat_distinguishes_authentication_from_quota_failures(self) -> None:
        import urllib.error
        from triune.api import routes

        class ConfigStub:
            def get_config(self):
                return {"byok_keys": {"gemini": "configured-secret"}}

        cases = (
            (401, "authentication failure, not a rate/quota response"),
            (403, "denied access"),
            (429, "rate or quota limit reached"),
        )
        for status_code, expected_message in cases:
            with self.subTest(status_code=status_code), patch.object(
                routes, "module_manager", ConfigStub()
            ), patch.object(
                routes,
                "_call_gemini_api",
                side_effect=urllib.error.HTTPError(
                    "https://generativelanguage.googleapis.com/", status_code, "rejected", {}, None
                ),
            ):
                response = self.client.post(
                    "/v1/chat/completions",
                    json={"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hello"}]},
                )
                self.assertEqual(response.status_code, 502)
                self.assertIn(expected_message, response.json()["detail"])

    def test_byok_chat_identifies_provider_socket_timeouts(self) -> None:
        import urllib.error
        from triune.api import routes

        class ConfigStub:
            def get_config(self):
                return {"byok_keys": {"gemini": "configured-secret"}}

        with patch.object(routes, "module_manager", ConfigStub()), patch.object(
            routes,
            "_call_gemini_api",
            side_effect=urllib.error.URLError(TimeoutError("timed out")),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"model": "gemini-3.8-flash", "messages": [{"role": "user", "content": "hello"}]},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("90-second request timeout", response.json()["detail"])
        self.assertIn("not local model inference", response.json()["detail"])

    def test_execute_single_specification_nodes(self) -> None:
        cases = (
            ({"id": "joint-loss", "type": "JointExitLoss", "params": {"lambda_reflex": 0.2}}, "failed", "requires all three"),
            ({"id": "rope", "type": "FastRoPE", "params": {"dim": 64, "max_seq_len": 2048}}, "failed", "requires matching q and k"),
            ({"id": "accumulator", "type": "GradientAccumulator", "params": {"grad_accum_steps": 4}}, "specification", None),
        )
        for node, expected_status, expected_error in cases:
            with self.subTest(node=node["type"]):
                response = self.client.post("/v1/dag/execute_node", json={"node": node})
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["status"], expected_status)
                if expected_error:
                    self.assertIn(expected_error, payload.get("error", ""))
                self.assertIn("node_id", payload)


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
        from triune.api import routes
        pytorch_state = routes.pytorch_state
        initial_step = pytorch_state.step
        pytorch_state.batch_size = 1
        pytorch_state.seq_len = 16
        pytorch_state.grad_accum_steps = 1
        pytorch_state.is_streaming_hf = False
        pytorch_state.dataset_path = "data/fineweb_sample.jsonl"
        pytorch_state.load_training_data("data/fineweb_sample.jsonl")

        response = self.client.post("/v1/training/step", json={})
        if not routes.HAS_TORCH:
            self.assertEqual(response.status_code, 503)
            self.assertEqual(pytorch_state.step, initial_step)
            self.assertNotIn("loss", response.json())
            return
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("step", data)
        self.assertIn("loss", data)
        self.assertGreaterEqual(data["step"], 1)
        self.assertGreaterEqual(data["loss"], 0.0)

    def test_models_endpoint_lists_all_canonical_presets(self) -> None:
        from triune.model.base import CANONICAL_MODEL_SPECS

        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        ids = {item["id"] for item in data}
        for preset_id in CANONICAL_MODEL_SPECS:
            self.assertIn(preset_id, ids)

    def test_model_preset_and_hardware_tune_sync(self) -> None:
        # 1. Switch preset to triune-nano with fp8
        preset_res = self.client.post(
            "/v1/models/preset",
            json={
                "preset_id": "triune-nano",
                "num_layers": 4,
                "hidden_dim": 192,
                "num_heads": 3,
                "head_dim": 64,
                "num_experts": 4,
                "batch_size": 2,
                "seq_len": 64,
                "depth_mode": "dynamic",
                "precision": "fp8",
            },
        )
        self.assertEqual(preset_res.status_code, 200)
        preset_data = preset_res.json()
        self.assertEqual(preset_data["status"], "success")
        self.assertEqual(preset_data["architecture"]["preset_id"], "triune-nano")
        self.assertEqual(preset_data["architecture"]["num_layers"], 4)
        self.assertTrue(preset_data["architecture"]["use_fp8"])

        # 2. Tune hardware with lr, warmup_steps, precision=bf16, gpu_layers=2
        tune_res = self.client.post(
            "/v1/hardware/tune",
            json={
                "preset": "custom",
                "batch_size": 4,
                "seq_len": 64,
                "grad_accum_steps": 2,
                "depth_mode": "joint",
                "precision": "bf16",
                "gpu_layers": 2,
                "lr": 0.0008,
                "warmup_steps": 150,
            },
        )
        self.assertEqual(tune_res.status_code, 200)
        tune_data = tune_res.json()
        self.assertEqual(tune_data["status"], "success")
        cc = tune_data["profile"]["current_config"]
        self.assertEqual(cc["batch_size"], 4)
        self.assertEqual(cc["depth_mode"], "joint")
        self.assertEqual(cc["precision"], "bf16")
        self.assertAlmostEqual(cc["lr"], 0.0008)
        self.assertEqual(cc["warmup_steps"], 150)
        self.assertEqual(tune_data["profile"]["gpu_layers"], 2)

        # 3. Verify /v1/training/status reflects synchronized state
        status_res = self.client.get("/v1/training/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertEqual(status_data["active_model"], "triune-nano")
        self.assertEqual(status_data["precision"], "bf16")
        self.assertEqual(status_data["gpu_layers"], 2)
        self.assertEqual(status_data["warmup_steps"], 150)

    def test_finetune_adapter_attach_and_detach_sync(self) -> None:
        attach_res = self.client.post(
            "/v1/finetune/adapter",
            json={
                "action": "attach",
                "name": "LoRA-r32",
                "rank": 32,
                "alpha": 64,
                "quantization": "4-bit NF4",
            },
        )
        self.assertEqual(attach_res.status_code, 200)
        attach_data = attach_res.json()
        self.assertEqual(attach_data["status"], "success")
        self.assertEqual(attach_data["active_adapter"]["name"], "LoRA-r32")
        self.assertEqual(attach_data["active_adapter"]["rank"], 32)

        status_res = self.client.get("/v1/training/status")
        self.assertEqual(status_res.json()["active_adapter"]["rank"], 32)

        detach_res = self.client.post("/v1/finetune/adapter", json={"action": "detach"})
        self.assertEqual(detach_res.status_code, 200)
        self.assertIsNone(detach_res.json()["active_adapter"])

    def test_dag_compile_precision_adapter_and_token_ids_exclusion(self) -> None:
        payload = {
            "nodes": [
                {
                    "id": "node_2",
                    "title": "Token IDs",
                    "type": "Data",
                    "details": "vocab_size=32000\ndtype=int64\npin_memory=True",
                },
                {
                    "id": "node_fp8",
                    "title": "FP8 Linear (E4M3)",
                    "type": "FP8Linear",
                    "details": "dtype=fp8_e4m3fn",
                },
                {
                    "id": "node_lora",
                    "title": "LoRA Adapter",
                    "type": "LoRAAdapter",
                    "details": "rank=16\nalpha=32",
                },
                {
                    "id": "node_streamer",
                    "title": "Layer Streamer",
                    "type": "LayerStreamer",
                    "details": "gpu_layers=2",
                },
            ],
            "edges": [],
        }
        res = self.client.post("/v1/dag/compile_to_training", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        applied = data["applied_config"]
        # Token IDs must NOT be treated as a dataset streamer
        self.assertNotIn("dataset", applied)
        self.assertEqual(applied.get("precision"), "fp8")
        self.assertEqual(applied.get("gpu_layers"), 2)
        self.assertEqual(applied["active_adapter"]["rank"], 16)

    def test_byok_wandb_save_and_test(self) -> None:
        import os
        from triune.api import routes

        class ConfigStub:
            def __init__(self):
                self.saved = {}
            def get_config(self):
                return {"byok_keys": {}}
            def save_config(self, config, **kwargs):
                self.saved = config

        stub = ConfigStub()
        with patch.object(routes, "module_manager", stub):
            res = self.client.post(
                "/v1/byok/save",
                json={"keys": {"wandb": "test-wandb-api-key-40-hex-characters-long"}}
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "saved")
            self.assertEqual(os.environ.get("WANDB_API_KEY"), "test-wandb-api-key-40-hex-characters-long")

        # Test wandb verification
        with patch("wandb.login", return_value=True, create=True):
            test_res = self.client.post(
                "/v1/byok/test",
                json={"provider": "wandb", "key": "test-wandb-api-key-40-hex-characters-long"}
            )
            self.assertEqual(test_res.status_code, 200)
            self.assertEqual(test_res.json()["status"], "valid")

    def test_word_boundary_chunking_pads_and_preserves_words(self) -> None:
        from triune.api import routes
        engine = routes.pytorch_state
        engine.load_tokenizer()
        if engine.tokenizer:
            # Token ID 296 ('ĠH') is word start; 22713 ('aines') is word continuation
            self.assertTrue(engine._is_word_boundary(296))
            self.assertFalse(engine._is_word_boundary(22713))


if __name__ == "__main__":
    unittest.main()
