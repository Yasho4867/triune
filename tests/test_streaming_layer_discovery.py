import unittest

import torch
import torch.nn as nn

from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig


class TestStreamingLayerDiscovery(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_standard_model_discovery(self):
        """Standard model with model.layers ModuleList is discovered automatically."""

        class DummyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 16)
                self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])
                self.head = nn.Linear(16, 10)

        model = DummyTransformer()
        engine = LayerStreamingEngine(model, device=self.device, config=StreamingConfig(enabled=True))
        layers = engine._resolve_layers()
        self.assertEqual(len(layers), 4)
        self.assertIs(layers[0], model.layers[0])

    def test_explicit_layers_path_resolution(self):
        """Explicit layers_path in config resolves target nested ModuleList."""

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.custom_blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(5)])

        model = CustomModel()
        cfg = StreamingConfig(enabled=True, layers_path="backbone.custom_blocks")
        engine = LayerStreamingEngine(model, device=self.device, config=cfg)
        layers = engine._resolve_layers()
        self.assertEqual(len(layers), 5)
        self.assertIs(layers[0], model.backbone.custom_blocks[0])

    def test_ambiguous_module_list_raises_value_error(self):
        """Model with multiple competing ModuleLists of equal length raises ValueError."""

        class AmbiguousModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_stack = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])
                self.decoder_stack = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])

        model = AmbiguousModel()
        with self.assertRaises(ValueError) as ctx:
            LayerStreamingEngine(model, device=self.device, config=StreamingConfig(enabled=True))
        self.assertIn("Ambiguous layer discovery", str(ctx.exception))
        self.assertIn("layers_path", str(ctx.exception))

    def test_invalid_layers_path_raises_value_error(self):
        """Invalid layers_path raises ValueError."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

        model = SimpleModel()
        cfg = StreamingConfig(enabled=True, layers_path="non_existent_path")
        with self.assertRaises(ValueError) as ctx:
            LayerStreamingEngine(model, device=self.device, config=cfg)
        self.assertIn("Specified layers_path", str(ctx.exception))

    def test_chunk_size_greater_than_one_runs_windowed_chunks(self):
        """chunk_size > 1 partitions layers into fixed-size staging windows and runs cleanly."""

        class DummyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = DummyTransformer()
        cfg = StreamingConfig(enabled=True, chunk_size=2, activation_checkpointing=False)
        engine = LayerStreamingEngine(model, device=self.device, config=cfg)
        try:
            self.assertEqual(engine._chunk_for_layer_idx(0), (0, 0, 1))
            self.assertEqual(engine._chunk_for_layer_idx(1), (0, 0, 1))
            self.assertEqual(engine._chunk_for_layer_idx(2), (1, 2, 3))
            engine.attach()
            x = torch.randn(2, 16, device=self.device, requires_grad=True)
            loss = model(x).sum()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
        finally:
            engine.detach()


if __name__ == "__main__":
    unittest.main()
