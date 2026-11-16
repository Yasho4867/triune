"""Regression & integration tests for real HF weight transfer, 3-tier transmutation, and local Ollama runtime."""

from __future__ import annotations

import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from triune.api.routes import pytorch_state
from triune.api.server import create_app
from triune.inference import generate_response
from triune.model.transformer import TriuneTransformer
from triune.model.transmuter import (
    distill_native_triune_step,
    ollama_to_hf_candidate,
    transfer_weights_to_native_triune,
    transmute_hf_to_triune,
)
from triune.runtime.model_scanner import _is_tag_live_in_ollama, query_ollama_running_tags


def _build_tiny_hf_llama_dir(target_dir: str) -> Path:
    """Create a tiny offline LlamaForCausalLM + PreTrainedTokenizerFast checkpoint on disk."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for idx in range(3, 64):
        vocab[f"tok{idx}"] = idx
    vocab["hello"] = 64
    vocab["triune"] = 65
    vocab["world"] = 66

    tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
    )
    fast_tok.save_pretrained(target_dir)

    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(target_dir)
    return Path(target_dir)


class TestWeightTransmutationAndOllamaRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._session_file = Path("checkpoints/session_state.json")
        cls._session_backup = cls._session_file.read_text(encoding="utf-8") if cls._session_file.is_file() else None
        client_kwargs = {}
        if "client" in inspect.signature(TestClient).parameters:
            client_kwargs["client"] = ("127.0.0.1", 43210)
        cls.client = TestClient(create_app(), **client_kwargs)

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_session_backup", None) is not None:
            cls._session_file.write_text(cls._session_backup, encoding="utf-8")
        elif cls._session_file.is_file():
            try:
                cls._session_file.unlink()
            except OSError:
                pass

    @staticmethod
    def _snapshot_engine_state() -> dict[str, object]:
        return {
            "model": pytorch_state.model,
            "tokenizer": pytorch_state.tokenizer,
            "optimizer": pytorch_state.optimizer,
            "active_model_id": pytorch_state.active_model_id,
            "active_checkpoint": pytorch_state.active_checkpoint,
            "weights_loaded": pytorch_state.weights_loaded,
            "explicit_architecture_active": pytorch_state.explicit_architecture_active,
            "active_runtime": getattr(pytorch_state, "active_runtime", "pytorch"),
            "ollama_model_tag": getattr(pytorch_state, "ollama_model_tag", None),
            "distill_teacher": getattr(pytorch_state, "distill_teacher", None),
            "architecture_spec": dict(pytorch_state.architecture_spec) if pytorch_state.architecture_spec else None,
            "step": pytorch_state.step,
            "total_tokens_trained": pytorch_state.total_tokens_trained,
            "history": list(pytorch_state.history),
            "_pad_id": getattr(pytorch_state, "_pad_id", None),
            "_eos_id": getattr(pytorch_state, "_eos_id", None),
        }

    @staticmethod
    def _restore_engine_state(snap: dict[str, object]) -> None:
        for key, val in snap.items():
            setattr(pytorch_state, key, val)

    def test_ollama_to_hf_candidate_mapping(self) -> None:
        self.assertEqual(ollama_to_hf_candidate("ollama:gemma3:4b"), "google/gemma-3-4b-it")
        self.assertEqual(ollama_to_hf_candidate("gemma3:4b-instruct-q4_K_M"), "google/gemma-3-4b-it")
        self.assertEqual(ollama_to_hf_candidate("ollama:qwen2.5:7b"), "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(ollama_to_hf_candidate("Qwen/Qwen2.5-1.5B-Instruct"), "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertIsNone(ollama_to_hf_candidate("ollama:unknown-private-model:latest"))

    def test_early_exit_transmutation_and_three_tier_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _build_tiny_hf_llama_dir(tmpdir)
            model, tok, meta, dev, pad_id, eos_id = transmute_hf_to_triune(
                tmpdir,
                reflex_ratio=0.25,
                rank=16,
                use_jev=True,
                device="cpu",
                local_files_only=True,
            )
            try:
                self.assertTrue(meta["weights_transferred"])
                self.assertEqual(meta["transmute_mode"], "early_exit")
                self.assertTrue(getattr(model, "_is_triune_transmuted", False))
                self.assertTrue(hasattr(model, "reflex_head"))
                self.assertTrue(hasattr(model, "limbic_head"))
                self.assertTrue(hasattr(model, "_exit_manager"))

                expected_routes = {0: "REFLEX", 1: "LIMBIC", 2: "CORTEX"}
                for depth, label in expected_routes.items():
                    info = generate_response(
                        model,
                        tok,
                        "hello triune world",
                        device=dev,
                        max_new_tokens=4,
                        temperature=0.0,
                        force_depth=depth,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                        return_info=True,
                    )
                    self.assertIsInstance(info, dict)
                    self.assertIsInstance(info.get("text"), str)
                    self.assertEqual(info.get("route"), label)
                    n_toks = info.get("tokens_count", info.get("tokens_generated", 0))
                    self.assertGreaterEqual(int(n_toks), 1)

                auto_info = generate_response(
                    model,
                    tok,
                    "hello triune world",
                    device=dev,
                    max_new_tokens=4,
                    temperature=0.0,
                    force_depth=None,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    return_info=True,
                )
                auto_route = auto_info.get("route")
                self.assertIn(auto_route, {"REFLEX", "LIMBIC", "CORTEX"})
            finally:
                exit_mgr = getattr(model, "_exit_manager", None)
                if exit_mgr is not None and hasattr(exit_mgr, "remove_hooks"):
                    exit_mgr.remove_hooks()

    def test_native_weight_surgery_and_kl_distillation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _build_tiny_hf_llama_dir(tmpdir)
            from triune.inference import load_model_for_inference

            teacher, tok, _, dev, _, _ = load_model_for_inference(tmpdir, device="cpu", local_files_only=True)
            student, report = transfer_weights_to_native_triune(
                teacher,
                tok,
                reflex_ratio=0.25,
                num_experts=4,
                top_k=2,
            )
            self.assertIsInstance(student, TriuneTransformer)
            self.assertTrue(report["weights_transferred"])
            self.assertEqual(report["transmute_mode"], "native_distill")
            self.assertGreater(report["transferred_tensors"], 0)
            self.assertGreater(report["transferred_params"], 0)
            self.assertTrue(
                torch.allclose(
                    student.token_embed.weight.detach(),
                    teacher.model.embed_tokens.weight.detach(),
                    atol=1e-5,
                )
            )

            opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
            batch = torch.tensor([[1, 64, 65, 66, 3, 4, 5]], dtype=torch.long, device=dev)
            m1 = distill_native_triune_step(student, teacher, batch, opt, temperature=2.0)
            m2 = distill_native_triune_step(student, teacher, batch, opt, temperature=2.0)
            self.assertTrue(torch.isfinite(torch.tensor(m1["loss"])))
            self.assertGreaterEqual(m1["cortex_kl"], 0.0)
            self.assertGreaterEqual(m1["limbic_kl"], 0.0)
            self.assertGreaterEqual(m1["reflex_kl"], 0.0)
            self.assertTrue(torch.isfinite(torch.tensor(m2["loss"])))

    def test_upgrade_endpoint_real_weight_transfer_and_distill(self) -> None:
        snap = self._snapshot_engine_state()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                _build_tiny_hf_llama_dir(tmpdir)

                # 1. Zero-loss 3-tier early-exit transmutation via /v1/models/upgrade_to_triune
                resp = self.client.post(
                    "/v1/models/upgrade_to_triune",
                    json={
                        "model_name": tmpdir,
                        "apply_as_active": True,
                        "transfer_weights": True,
                        "transfer_mode": "early_exit",
                        "reflex_ratio": 0.25,
                    },
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["status"], "success")
                self.assertTrue(data["activated"])
                self.assertTrue(data["weights_transferred"])
                self.assertEqual(data["activation_kind"], "transferred_early_exit")
                self.assertTrue(pytorch_state.weights_loaded)
                self.assertFalse(pytorch_state.explicit_architecture_active)

                chat_resp = self.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": data["active_model_id"],
                        "messages": [{"role": "user", "content": "hello triune"}],
                        "max_tokens": 4,
                        "temperature": 0.0,
                        "stream": False,
                    },
                )
                self.assertEqual(chat_resp.status_code, 200)
                chat_payload = chat_resp.json()
                self.assertIn("choices", chat_payload)
                self.assertTrue(chat_payload["choices"][0]["message"]["content"])

                # 2. Structural surgery + /v1/models/distill endpoint
                resp_native = self.client.post(
                    "/v1/models/upgrade_to_triune",
                    json={
                        "model_name": tmpdir,
                        "apply_as_active": True,
                        "transfer_weights": True,
                        "transfer_mode": "native_distill",
                        "reflex_ratio": 0.25,
                        "limbic_experts": 4,
                    },
                )
                self.assertEqual(resp_native.status_code, 200)
                native_data = resp_native.json()
                self.assertTrue(native_data["weights_transferred"])
                self.assertEqual(native_data["activation_kind"], "transferred_native_distill")
                self.assertIsInstance(pytorch_state.model, TriuneTransformer)

                distill_resp = self.client.post(
                    "/v1/models/distill",
                    json={"steps": 2, "lr": 5e-4, "prompts": ["hello triune world"]},
                )
                self.assertEqual(distill_resp.status_code, 200)
                d_data = distill_resp.json()
                self.assertEqual(d_data["status"], "success")
                self.assertEqual(d_data["steps_completed"], 2)
                self.assertIsNotNone(d_data["final_loss"])
        finally:
            self._restore_engine_state(snap)

    def test_ollama_scanner_load_and_streaming_chat(self) -> None:
        self.assertTrue(_is_tag_live_in_ollama("ollama:gemma3:4b", {"gemma3:4b", "llama3:latest"}))
        self.assertTrue(_is_tag_live_in_ollama("llama3", {"llama3:latest"}))
        self.assertFalse(_is_tag_live_in_ollama("qwen-custom:latest", {"gemma3:4b"}))

        snap = self._snapshot_engine_state()
        try:
            with patch(
                "triune.runtime.model_scanner.query_ollama_running_tags",
                return_value={"gemma3:4b"},
            ):
                load_ok = self.client.post("/v1/models/load", json={"checkpoint_path": "ollama:gemma3:4b"})
                self.assertEqual(load_ok.status_code, 200)
                payload = load_ok.json()
                self.assertEqual(payload["status"], "success")
                self.assertTrue(payload["weights_loaded"])
                self.assertEqual(payload["runtime"], "ollama")
                self.assertEqual(pytorch_state.active_runtime, "ollama")
                self.assertEqual(pytorch_state.ollama_model_tag, "gemma3:4b")

                ndjson_body = (
                    json.dumps({"message": {"role": "assistant", "content": "Hello from "}, "done": False}).encode()
                    + b"\n"
                    + json.dumps(
                        {
                            "message": {"role": "assistant", "content": "local Ollama!"},
                            "done": True,
                            "eval_count": 6,
                            "eval_duration": 100_000_000,
                        }
                    ).encode()
                    + b"\n"
                )

                class _FakeResp(io.BytesIO):
                    status = 200

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        self.close()

                with patch("urllib.request.urlopen", return_value=_FakeResp(ndjson_body)):
                    chat_res = self.client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "ollama:gemma3:4b",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 32,
                            "stream": False,
                        },
                    )
                    self.assertEqual(chat_res.status_code, 200)
                    cjson = chat_res.json()
                    self.assertEqual(cjson["choices"][0]["message"]["content"], "Hello from local Ollama!")
                    self.assertEqual(cjson["telemetry"]["route"], "OLLAMA")

                # Unpulled Ollama tag still directs users to Transmute cleanly
                load_missing = self.client.post("/v1/models/load", json={"checkpoint_path": "ollama:not-pulled:latest"})
                self.assertEqual(load_missing.status_code, 200)
                self.assertEqual(load_missing.json()["status"], "error")
                self.assertIn("Transmute", load_missing.json()["message"])
        finally:
            self._restore_engine_state(snap)

        # Verify query_ollama_running_tags handles connection refusal cleanly
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            self.assertEqual(query_ollama_running_tags(), set())


if __name__ == "__main__":
    unittest.main()
