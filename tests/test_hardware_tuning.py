"""Regression coverage for applying hardware tuning to the live model."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from triune.api import routes
from triune.api.routes import _apply_model_gpu_residency
from triune.runtime.streaming import StreamingConfig


class _FakeStreamingEngine:
    def __init__(self, model, device, config, optimizer=None):
        self.model = model
        self.device = device
        self.config = config
        self.optimizer = optimizer
        self.all_layers = list(model.layers)
        count = max(0, min(len(self.all_layers), config.resident_gpu_layers))
        self.resident_layers = self.all_layers[:count]
        self.layers = self.all_layers[count:]
        self.detached = False

    def cleanup_resident_layers(self):
        pass

    def detach(self):
        self.detached = True
        if getattr(self.model, "_streaming_engine", None) is self:
            self.model._streaming_engine = None

    def attach(self, optimizer=None):
        self.model._streaming_engine = self


class TestHardwareTuning(unittest.TestCase):
    def test_gpu_layer_change_rebuilds_live_streaming_wrapper(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
        old = _FakeStreamingEngine(
            model,
            torch.device("cuda"),
            StreamingConfig(resident_gpu_layers=1),
        )
        model._streaming_engine = old
        state = SimpleNamespace(
            model=model,
            device=torch.device("cuda"),
            optimizer=object(),
            precision="bf16",
            is_training=False,
        )

        with patch("triune.runtime.streaming.LayerStreamingEngine", _FakeStreamingEngine), \
             patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 8 * 1024**3)):
            applied = _apply_model_gpu_residency(state, 4)

        self.assertTrue(old.detached)
        self.assertEqual(applied, 4)
        self.assertEqual(len(model._streaming_engine.resident_layers), 4)
        self.assertEqual(model._streaming_engine.optimizer, state.optimizer)

    def test_cpu_runtime_never_claims_gpu_resident_layers(self):
        state = SimpleNamespace(model=nn.Linear(2, 2), device=torch.device("cpu"))
        self.assertEqual(_apply_model_gpu_residency(state, 1), 0)

    def test_cpu_profile_does_not_claim_gpu_offload_or_invent_speed(self):
        with patch.object(routes.pytorch_state, "cuda_available", False), \
             patch.object(routes.pytorch_state, "model", None), \
             patch.object(routes.pytorch_state, "history", []):
            profile = routes._compute_hardware_profile()

        self.assertEqual(profile["recommended_preset"], "eco")
        self.assertTrue(all(preset["gpu_layers"] == 0 for preset in profile["presets"].values()))
        self.assertTrue(all(preset["estimated_tok_s"] is None for preset in profile["presets"].values()))


if __name__ == "__main__":
    unittest.main()
