"""Test suite verifying Hugging Face streaming dataset integration, BYOK authentication,
and zero-dummy-phrase training steps with RealPyTorchEngineState and FastAPI routes.
"""

import os
import sys
import unittest
import asyncio

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from triune.api.routes import pytorch_state, router
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestLiveApiStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FastAPI()
        cls.app.include_router(router)
        cls.client = TestClient(cls.app)

    def test_01_byok_save_and_test(self):
        """Verify saving and testing Hugging Face token via BYOK endpoints."""
        # 1. Test invalid token format
        res = self.client.post("/v1/byok/test", json={"provider": "huggingface", "key": "short"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "invalid")
        self.assertIn("hf_", data["message"])

        # 2. Test valid-format token (offline / format check)
        dummy_hf_token = "hf_testtoken1234567890abcdefghijklmnopqrstuvwxyz"
        save_res = self.client.post("/v1/byok/save", json={"keys": {"huggingface": dummy_hf_token}})
        self.assertEqual(save_res.status_code, 200)
        self.assertEqual(pytorch_state.hf_token, dummy_hf_token)
        self.assertEqual(os.environ.get("HF_TOKEN"), dummy_hf_token)

    def test_02_stream_connect_hf_dataset(self):
        """Verify connecting to a real Hugging Face dataset (TinyStories) streams real tokens."""
        # We test with roneneldan/TinyStories which is fast and public
        res = self.client.post("/v1/datasets/stream/connect", json={
            "dataset_name": "roneneldan/TinyStories",
            "split": "train",
            "text_column": "text"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[HF Stream Connect Result]:", data)
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["is_streaming"])
        self.assertEqual(data["text_column"], "text")
        self.assertGreaterEqual(data["buffered_chunks"], 1)
        self.assertTrue(len(data["sample_preview"]) > 0)
        # Ensure it's not any of the hardcoded dummy sentences!
        dummy_markers = ["Triune transformer achieves state-of-the-art", "Synthetic text generation", "Vectorised GLA attention"]
        for marker in dummy_markers:
            self.assertNotIn(marker, data["sample_preview"])

    def test_03_stream_status_endpoint(self):
        """Verify GET /v1/datasets/stream/status returns active streaming telemetry."""
        res = self.client.get("/v1/datasets/stream/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        print("\n[Stream Status]:", data)
        self.assertTrue(data["is_streaming"])
        self.assertEqual(data["dataset_name"], "roneneldan/TinyStories")
        self.assertGreaterEqual(data["buffered_chunks"], 1)
        self.assertTrue(data["hf_token_configured"])

    def test_04_training_step_uses_streamed_tokens(self):
        """Verify POST /v1/training/step consumes from the streaming buffer and updates batch preview."""
        initial_tokens = pytorch_state.tokens_trained
        step_res = self.client.post("/v1/training/step")
        self.assertEqual(step_res.status_code, 200)
        step_data = step_res.json()
        print("\n[Training Step Result]:", step_data)
        self.assertIn("loss", step_data)
        self.assertIn("batch_preview", step_data)
        self.assertEqual(step_data["dataset_name"], "roneneldan/TinyStories")
        self.assertGreater(step_data["tokens_trained"], initial_tokens)
        self.assertTrue(len(step_data["batch_preview"]) > 0)


if __name__ == "__main__":
    unittest.main()
