"""Test Hugging Face Hub export and loading compatibility."""

import json
from pathlib import Path
import tempfile
import unittest
import torch

from triune.model.transformer import TriuneTransformer
from triune.export.hf_compat import export_to_huggingface, extract_triune_hf_config


class TestHuggingFaceExport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TriuneTransformer(
            vocab_size=128,
            hidden_dim=64,
            num_layers=6,
            num_heads=2,
            head_dim=32,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
        )

    def test_extract_triune_hf_config(self):
        cfg = extract_triune_hf_config(self.model)
        self.assertEqual(cfg["model_type"], "triune")
        self.assertEqual(cfg["vocab_size"], 128)
        self.assertEqual(cfg["hidden_dim"], 64)
        self.assertEqual(cfg["num_layers"], 6)
        self.assertEqual(cfg["num_experts"], 2)
        self.assertIn("auto_map", cfg)
        self.assertEqual(cfg["auto_map"]["AutoConfig"], "configuration_triune.TriuneConfig")
        self.assertEqual(cfg["auto_map"]["AutoModelForCausalLM"], "modeling_triune.TriuneForCausalLM")

    def test_export_to_huggingface_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = export_to_huggingface(self.model, tmp_dir)
            out_path = Path(out_dir)

            self.assertTrue((out_path / "config.json").exists())
            self.assertTrue((out_path / "generation_config.json").exists())
            self.assertTrue((out_path / "configuration_triune.py").exists())
            self.assertTrue((out_path / "modeling_triune.py").exists())
            self.assertTrue((out_path / "model.safetensors").exists())

            # Verify config.json content
            with open(out_path / "config.json", "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
            self.assertEqual(saved_cfg["architectures"], ["TriuneForCausalLM"])
            self.assertEqual(saved_cfg["vocab_size"], 128)

            # Verify generation_config.json
            with open(out_path / "generation_config.json", "r", encoding="utf-8") as f:
                saved_gen = json.load(f)
            self.assertEqual(saved_gen["bos_token_id"], 1)
            self.assertEqual(saved_gen["eos_token_id"], 2)


if __name__ == "__main__":
    unittest.main()
