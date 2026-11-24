"""Architecture compatibility regression tests for Lucid Machines.

Verifies:
1. Lucid Machines exports the canonical model, trainer, engine, and objective names.
2. `triune-*` architecture presets (`triune-nano`, `triune-small`, `triune-base`, etc.) and
   `_LEGACY_PRESET_ALIASES` (`triune-micro` -> `triune-nano`, `triune-1.5b` -> `triune-1b`,
   `triune-moe` -> `triune-7b`) resolve cleanly alongside `models/configs/junkie.json`.
3. `LUCID_MACHINES_*` env vars override `TRIUNE_*` fallback, and `TRIUNE_*` fallback still works.
4. Hugging Face `model_type == "triune"` (`TriuneConfig`, `TriuneForCausalLM`) exports and loads cleanly.
5. Preserved `triune_tokenizer.json`, `models/configs/triune-*.json`, and `models/configs/junkie.json` exist.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class TestTriuneBackwardCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_canonical_lucid_machines_exports(self) -> None:
        lm = importlib.import_module("lucid_machines")

        self.assertTrue(hasattr(lm, "TriuneTransformer"))
        self.assertTrue(hasattr(lm, "TriuneModel"))
        self.assertTrue(hasattr(lm, "TriuneConfig"))
        self.assertTrue(hasattr(lm, "TriuneFineTuner"))
        self.assertTrue(hasattr(lm, "LucidMachinesUniversalEngine"))

    def test_root_triune_compat_folder_removed_and_lucid_machines_exports_triune_architecture(self) -> None:
        self.assertFalse((self.repo_root / "triune").exists())
        sys.modules.pop("triune", None)
        importlib.invalidate_caches()
        self.assertIsNone(importlib.util.find_spec("triune"))

        lm = importlib.import_module("lucid_machines")
        for symbol in (
            "TriuneTransformer",
            "TriuneModel",
            "TriuneConfig",
            "TriuneForCausalLM",
            "TriuneFineTuner",
            "CentroidSteerOptimizer",
        ):
            self.assertTrue(hasattr(lm, symbol), f"lucid_machines missing export {symbol}")

        cfg_dir = self.repo_root / "models" / "configs"
        self.assertGreaterEqual(len(list(cfg_dir.glob("triune-*.json"))), 4)
        self.assertTrue((cfg_dir / "junkie.json").is_file())
        self.assertTrue((self.repo_root / "triune_tokenizer.json").is_file())

    def test_keyring_service_prefers_polysom_studio_with_legacy_fallback(self) -> None:
        from lucid_machines.modules import manager as mgr_mod

        self.assertEqual(mgr_mod._KEYRING_SERVICE, "Polysom Studio")
        self.assertIn("Triune Studio", mgr_mod._LEGACY_KEYRING_SERVICES)

    def test_canonical_triune_presets_and_legacy_aliases_resolve(self) -> None:
        from lucid_machines.model.base import _LEGACY_PRESET_ALIASES, CANONICAL_MODEL_SPECS, get_model_spec

        for canonical_id in ("triune-nano", "triune-small", "triune-base", "triune-7b"):
            self.assertIn(canonical_id, CANONICAL_MODEL_SPECS)

        expected_aliases = {
            "triune-micro": "triune-nano",
            "triune-drafter": "triune-nano",
            "triune-1.5b": "triune-1b",
            "triune-2.5b": "triune-3b",
            "triune-moe": "triune-7b",
        }
        for alias_id, target_id in expected_aliases.items():
            self.assertEqual(
                _LEGACY_PRESET_ALIASES.get(alias_id),
                target_id,
                f"Expected _LEGACY_PRESET_ALIASES[{alias_id!r}] == {target_id!r}",
            )
            alias_spec = get_model_spec(alias_id)
            canonical_spec = get_model_spec(target_id)
            self.assertEqual(alias_spec.num_layers, canonical_spec.num_layers)
            self.assertEqual(alias_spec.hidden_dim, canonical_spec.hidden_dim)
            self.assertEqual(alias_spec.num_heads, canonical_spec.num_heads)
            self.assertEqual(alias_spec.num_experts, canonical_spec.num_experts)

    def test_lucid_machines_env_vars_override_triune_fallback(self) -> None:
        from lucid_machines.api import server as server_mod

        # 1. TRIUNE_* fallback works when LUCID_MACHINES_* is unset
        legacy_token = "legacy_" + ("a" * 34)
        new_token = "lucid__" + ("b" * 34)
        with patch.dict(os.environ, {"LUCID_MACHINES_API_TOKEN": "", "TRIUNE_API_TOKEN": legacy_token}):
            self.assertEqual(server_mod._get_env("API_TOKEN"), legacy_token)

            client = TestClient(server_mod.create_app(), client=("127.0.0.1", 43210))
            unauth = client.get("/v1/system/config", headers={"X-Forwarded-For": "203.0.113.10"})
            self.assertEqual(unauth.status_code, 401)
            ok = client.get(
                "/v1/system/config",
                headers={"X-Forwarded-For": "203.0.113.10", "Authorization": f"Bearer {legacy_token}"},
            )
            self.assertEqual(ok.status_code, 200)

        # 2. LUCID_MACHINES_* takes precedence over TRIUNE_* when both are set
        with patch.dict(os.environ, {"LUCID_MACHINES_API_TOKEN": new_token, "TRIUNE_API_TOKEN": legacy_token}):
            self.assertEqual(server_mod._get_env("API_TOKEN"), new_token)

            client = TestClient(server_mod.create_app(), client=("127.0.0.1", 43210))
            stale = client.get(
                "/v1/system/config",
                headers={"X-Forwarded-For": "203.0.113.10", "Authorization": f"Bearer {legacy_token}"},
            )
            self.assertEqual(stale.status_code, 401)
            ok_new = client.get(
                "/v1/system/config",
                headers={"X-Forwarded-For": "203.0.113.10", "Authorization": f"Bearer {new_token}"},
            )
            self.assertEqual(ok_new.status_code, 200)

    def test_hf_model_type_triune_exports_and_loads(self) -> None:
        from lucid_machines.export import hf_compat
        from lucid_machines.model.transformer import TriuneTransformer

        model = TriuneTransformer(
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

        cfg = hf_compat.extract_triune_hf_config(model)
        self.assertEqual(cfg["model_type"], "triune")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = hf_compat.export_to_huggingface(model, tmpdir)
            cfg_path = Path(out_dir) / "config.json"
            saved_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_cfg["model_type"], "triune")
            self.assertEqual(saved_cfg["architectures"], ["TriuneForCausalLM"])

            from transformers import AutoConfig, AutoModelForCausalLM

            loaded_cfg = AutoConfig.from_pretrained(str(out_dir), trust_remote_code=True)
            self.assertEqual(loaded_cfg.model_type, "triune")
            loaded_model = AutoModelForCausalLM.from_pretrained(str(out_dir), trust_remote_code=True)
            self.assertIsNotNone(loaded_model)

    def test_preserved_tokenizer_triune_templates_and_junkie_model_config_exist(self) -> None:
        self.assertTrue((self.repo_root / "triune_tokenizer.json").is_file())
        cfg_dir = self.repo_root / "models" / "configs"
        if cfg_dir.is_dir():
            triune_cfgs = list(cfg_dir.glob("triune-*.json"))
            self.assertGreaterEqual(len(triune_cfgs), 4)
            self.assertTrue((cfg_dir / "junkie.json").is_file())


if __name__ == "__main__":
    unittest.main()
