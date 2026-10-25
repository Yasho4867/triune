"""Comprehensive Unit Tests for Triune LoRA/QLoRA Fine-Tuning Engine and API Routes."""

import json
import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from triune.api.server import create_app
from triune.model import TriuneTransformer
from triune.trainer.finetune import LoRAConfig, LoRALayer, TriuneFineTuner


class TestFineTuneCore(unittest.TestCase):
    def setUp(self):
        self.model = TriuneTransformer(
            vocab_size=500,
            hidden_dim=64,
            num_layers=4,
            num_heads=2,
            head_dim=32,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=3,
        )

    def test_lora_config(self):
        cfg = LoRAConfig(rank=8, alpha=16.0, quantization="4-bit NF4")
        self.assertEqual(cfg.rank, 8)
        self.assertEqual(cfg.alpha, 16.0)
        self.assertTrue(cfg.use_qlora)

    def test_lora_layer_forward_and_merge(self):
        linear = nn.Linear(32, 64)
        orig_weight = linear.weight.clone()
        adapter = LoRALayer(linear, rank=4, alpha=8.0)

        # Base layer must be frozen
        for p in adapter.original_layer.parameters():
            self.assertFalse(p.requires_grad)

        # LoRA parameters must be trainable
        self.assertTrue(adapter.lora_A.requires_grad)
        self.assertTrue(adapter.lora_B.requires_grad)

        # Forward pass
        x = torch.randn(2, 5, 32)
        out = adapter(x)
        self.assertEqual(out.shape, (2, 5, 64))

        # Test merge
        adapter.lora_B.data.fill_(0.5)
        adapter.lora_A.data.fill_(0.2)
        adapter.merge()
        self.assertFalse(torch.equal(linear.weight, orig_weight))

    def test_triune_finetuner_lifecycle(self):
        cfg = LoRAConfig(rank=4, alpha=8.0)
        tuner = TriuneFineTuner(self.model, cfg)

        # Base model parameters must be frozen
        self.assertFalse(self.model.token_embed.weight.requires_grad)
        self.assertFalse(self.model.final_head.weight.requires_grad)

        trainable = tuner.count_trainable_parameters()
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(trainable, 0)
        self.assertLess(trainable, total_params * 0.1)  # LoRA is less than 10% of model

        # Idempotent attach: attach_lora() must not double wrap
        adapter_count_before = len(tuner.applied_adapters)
        tuner.attach_lora()
        self.assertEqual(len(tuner.applied_adapters), adapter_count_before)

        # Test save and load adapter
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_lora_ckpt"
            tuner.save_adapter(out_path)
            self.assertTrue((out_path / "adapter_config.json").is_file())
            has_weights = (out_path / "adapter_model.safetensors").is_file() or (out_path / "adapter_model.pt").is_file()
            self.assertTrue(has_weights)

            # Detach
            tuner.detach_lora()
            self.assertEqual(len(tuner.applied_adapters), 0)
            self.assertFalse(any(isinstance(m, LoRALayer) for m in self.model.modules()))

            # Reload
            tuner.load_adapter(out_path)
            self.assertGreater(len(tuner.applied_adapters), 0)
            self.assertTrue(any(isinstance(m, LoRALayer) for m in self.model.modules()))

            # Merge and unload
            tuner.merge_and_unload()
            self.assertEqual(len(tuner.applied_adapters), 0)
            self.assertFalse(any(isinstance(m, LoRALayer) for m in self.model.modules()))


class TestFineTuneApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(), client=("127.0.0.1", 43210))

    def test_finetune_status_endpoint(self):
        res = self.client.get("/v1/finetune/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("step", data)
        self.assertIn("trainable_params", data)

    def test_finetune_adapters_list_endpoint(self):
        res = self.client.get("/v1/finetune/adapters")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("adapters", data)
        self.assertIsInstance(data["adapters"], list)

    def test_finetune_adapter_attach_detach(self):
        attach_res = self.client.post("/v1/finetune/adapter", json={
            "action": "attach",
            "name": "Test-Adapter-Attach",
            "rank": 8,
            "alpha": 16.0,
            "quantization": "4-bit NF4"
        })
        self.assertEqual(attach_res.status_code, 200)
        attach_data = attach_res.json()
        self.assertEqual(attach_data["status"], "success")
        self.assertEqual(attach_data["active_adapter"]["rank"], 8)

        # Merge endpoint test with active adapter
        merge_res = self.client.post("/v1/finetune/merge")
        self.assertEqual(merge_res.status_code, 200)
        self.assertEqual(merge_res.json()["status"], "success")

        # Detach endpoint test
        detach_res = self.client.post("/v1/finetune/adapter", json={"action": "detach"})
        self.assertEqual(detach_res.status_code, 200)
        detach_data = detach_res.json()
        self.assertEqual(detach_data["status"], "success")
        self.assertIsNone(detach_data["active_adapter"])

    def test_finetune_cancel_endpoint(self):
        res = self.client.post("/v1/finetune/cancel")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)


if __name__ == "__main__":
    unittest.main()
