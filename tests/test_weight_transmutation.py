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

    def test_gguf_block_dequantization_and_synthetic_archive_transmutation(self) -> None:
        import struct

        import numpy as np

        from triune.runtime.gguf_loader import dequantize_ggml_tensor, load_gguf_hf_model

        # 1. Verify Q4_0 and Q8_0 dequantization precision on synthetic blocks
        d_f16 = np.float16(0.5).tobytes()
        qs_q8 = np.arange(-16, 16, dtype=np.int8).tobytes()
        t_q8 = dequantize_ggml_tensor(d_f16 + qs_q8, (32,), 8, torch.float32)
        expected_q8 = torch.arange(-16, 16, dtype=torch.float32) * 0.5
        self.assertTrue(torch.allclose(t_q8, expected_q8, atol=1e-4))

        qs_q4 = bytes([0x90] * 16)  # low nibble=0 (-8*0.5=-4.0), high nibble=9 ((9-8)*0.5=+0.5)
        t_q4 = dequantize_ggml_tensor(d_f16 + qs_q4, (32,), 2, torch.float32)
        self.assertAlmostEqual(float(t_q4[:16].mean()), -4.0, places=4)
        self.assertAlmostEqual(float(t_q4[16:].mean()), 0.5, places=4)

        # 2. Write a minimal valid synthetic GGUF v3 Llama archive (2L, H=32, 2 heads, vocab=64) and transmute via API
        def _pack_str(s: str) -> bytes:
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        vocab = ["<unk>", "<s>", "</s>"] + [f"▁tok{i}" for i in range(3, 61)] + ["▁hello", "▁triune", "▁world"]
        kvs: list[tuple[str, int, bytes]] = [
            ("general.architecture", 8, _pack_str("llama")),
            ("general.alignment", 4, struct.pack("<I", 32)),
            ("llama.block_count", 4, struct.pack("<I", 4)),
            ("llama.embedding_length", 4, struct.pack("<I", 32)),
            ("llama.attention.head_count", 4, struct.pack("<I", 2)),
            ("llama.attention.head_count_kv", 4, struct.pack("<I", 2)),
            ("llama.feed_forward_length", 4, struct.pack("<I", 64)),
            ("llama.context_length", 4, struct.pack("<I", 128)),
            ("tokenizer.ggml.model", 8, _pack_str("llama")),
            ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 1)),
            ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
            ("tokenizer.ggml.unknown_token_id", 4, struct.pack("<I", 0)),
            ("tokenizer.ggml.padding_token_id", 4, struct.pack("<I", 0)),
            (
                "tokenizer.ggml.tokens",
                9,
                struct.pack("<IQ", 8, len(vocab)) + b"".join(_pack_str(t) for t in vocab),
            ),
            (
                "tokenizer.ggml.scores",
                9,
                struct.pack("<IQ", 6, len(vocab)) + np.full(len(vocab), -1.0, dtype="<f4").tobytes(),
            ),
        ]

        rng = np.random.default_rng(7)
        tensor_specs: list[tuple[str, tuple[int, ...], bytes]] = [
            ("token_embd.weight", (32, 64), (rng.standard_normal((64, 32)).astype("<f4") * 0.05).tobytes()),
            ("output_norm.weight", (32,), np.ones(32, dtype="<f4").tobytes()),
        ]
        for li in range(4):
            tensor_specs.extend(
                [
                    (f"blk.{li}.attn_norm.weight", (32,), np.ones(32, dtype="<f4").tobytes()),
                    (f"blk.{li}.ffn_norm.weight", (32,), np.ones(32, dtype="<f4").tobytes()),
                    (
                        f"blk.{li}.attn_q.weight",
                        (32, 32),
                        (rng.standard_normal((32, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.attn_k.weight",
                        (32, 32),
                        (rng.standard_normal((32, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.attn_v.weight",
                        (32, 32),
                        (rng.standard_normal((32, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.attn_output.weight",
                        (32, 32),
                        (rng.standard_normal((32, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.ffn_gate.weight",
                        (32, 64),
                        (rng.standard_normal((64, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.ffn_up.weight",
                        (32, 64),
                        (rng.standard_normal((64, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.ffn_down.weight",
                        (64, 32),
                        (rng.standard_normal((32, 64)).astype("<f4") * 0.05).tobytes(),
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "tiny_test.gguf"
            header = bytearray()
            header.extend(b"GGUF")
            header.extend(struct.pack("<IQQ", 3, len(tensor_specs), len(kvs)))
            for k, vt, raw_val in kvs:
                header.extend(_pack_str(k))
                header.extend(struct.pack("<I", vt))
                header.extend(raw_val)

            offset = 0
            tensor_data = bytearray()
            for tname, dims, buf in tensor_specs:
                header.extend(_pack_str(tname))
                header.extend(struct.pack("<I", len(dims)))
                header.extend(struct.pack(f"<{len(dims)}Q", *dims))
                header.extend(struct.pack("<IQ", 0, offset))
                pad_len = (32 - (len(buf) % 32)) % 32
                tensor_data.extend(buf)
                tensor_data.extend(b"\x00" * pad_len)
                offset += len(buf) + pad_len

            head_pad = (32 - (len(header) % 32)) % 32
            gguf_bytes = bytes(header) + (b"\x00" * head_pad) + bytes(tensor_data)
            gguf_path.write_bytes(gguf_bytes)
            # Also save as extensionless Ollama-style sha256 blob (regression test for local blob transmutation)
            blob_path = Path(tmpdir) / "sha256-deadbeef12345678"
            blob_path.write_bytes(gguf_bytes)

            m_gguf, _, meta_g, dev_g, _, _ = load_gguf_hf_model(gguf_path, device="cpu")
            self.assertTrue(meta_g["gguf_loaded_direct"])
            self.assertEqual(meta_g["gguf_arch"], "llama")
            self.assertEqual(meta_g["num_layers"], 4)

            snap = self._snapshot_engine_state()
            try:
                for cand_path in (gguf_path, blob_path):
                    up = self.client.post(
                        "/v1/models/upgrade_to_triune",
                        json={"model_name": str(cand_path), "apply_as_active": True, "transfer_weights": True},
                    )
                    self.assertEqual(up.status_code, 200)
                    uj = up.json()
                    self.assertEqual(uj["status"], "success")
                    self.assertTrue(uj["activated"])
                    self.assertTrue(uj["weights_transferred"])
            finally:
                self._restore_engine_state(snap)

    def test_native_untied_lm_head_exact_crop_coverage_and_seq_invariant_kl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from transformers import AutoModelForCausalLM

            tiny_dir = _build_tiny_hf_llama_dir(tmpdir)
            teacher = AutoModelForCausalLM.from_pretrained(str(tiny_dir))
            # Untie lm_head explicitly with distinct weights from embed_tokens
            with torch.no_grad():
                untied_head = torch.nn.Linear(teacher.config.hidden_size, teacher.config.vocab_size, bias=False)
                untied_head.weight.copy_(torch.randn_like(teacher.model.embed_tokens.weight) * 0.25 + 0.5)
                teacher.lm_head = untied_head

            student, rep = transfer_weights_to_native_triune(teacher, num_experts=4)
            self.assertTrue(rep["lm_head_transferred"])
            self.assertTrue(rep["lm_head_untied"])
            self.assertFalse(getattr(student, "_exits_calibrated", True))
            self.assertNotEqual(student.final_head.weight.data_ptr(), student.token_embed.weight.data_ptr())
            self.assertTrue(torch.allclose(student.final_head.weight, teacher.lm_head.weight, atol=1e-5))

            # Expanding hidden_dim from 64 -> 128 only initializes 64 columns cleanly; coverage must reflect real copy size
            wide_student, wide_rep = transfer_weights_to_native_triune(teacher, num_experts=4, target_dim=128)
            self.assertLess(wide_rep["coverage_ratio"], 0.7)
            self.assertGreater(wide_student.hidden_dim, teacher.config.hidden_size)

            # Sequence-length invariance check for per-token KL distillation normalization
            student.eval()
            torch.manual_seed(7)
            seq_short = torch.randint(0, 64, (2, 8))
            seq_long = seq_short.repeat(1, 4)  # T=32 repeating identical chunks

            opt_zero = torch.optim.SGD(student.parameters(), lr=0.0)
            m_short = distill_native_triune_step(student, teacher, seq_short, opt_zero, ce_weight=0.0)
            m_long = distill_native_triune_step(student, teacher, seq_long, opt_zero, ce_weight=0.0)
            self.assertAlmostEqual(m_short["cortex_kl"], m_long["cortex_kl"], delta=0.15)
            self.assertLess(abs(m_short["reflex_kl"] - m_long["reflex_kl"]), 0.25)

    def test_early_exit_cortex_logits_match_teacher_equivalently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from transformers import AutoModelForCausalLM

            tiny_dir = _build_tiny_hf_llama_dir(tmpdir)
            teacher_ref = AutoModelForCausalLM.from_pretrained(str(tiny_dir)).eval()
            triune_wrapped, tok, report, dev, pad_id, eos_id = transmute_hf_to_triune(
                str(tiny_dir),
                reflex_ratio=0.25,
                rank=16,
                use_jev=True,
                device="cpu",
                local_files_only=True,
            )
            try:
                triune_wrapped.eval()
                self.assertFalse(getattr(triune_wrapped, "_exits_calibrated", True))
                self.assertEqual(report["transmute_mode"], "early_exit")

                x = torch.randint(0, 64, (1, 10))
                with torch.no_grad():
                    ref_logits = teacher_ref(x).logits
                    cortex_logits = triune_wrapped(x).logits
                    self.assertTrue(torch.allclose(cortex_logits, ref_logits, atol=1e-5))

                # Uncalibrated AUTO exit must fall back safely to full CORTEX depth, matching teacher generation
                auto_info = generate_response(
                    triune_wrapped,
                    tok,
                    "hello triune world",
                    device=dev,
                    max_new_tokens=6,
                    temperature=0.0,
                    force_depth=None,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    return_info=True,
                )
                cortex_info = generate_response(
                    triune_wrapped,
                    tok,
                    "hello triune world",
                    device=dev,
                    max_new_tokens=6,
                    temperature=0.0,
                    force_depth=2,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    return_info=True,
                )
                self.assertEqual(auto_info["route"], "CORTEX")
                self.assertEqual(auto_info["text"], cortex_info["text"])

                # Reflex and Limbic exit heads run independently on intermediate layers
                reflex_info = generate_response(
                    triune_wrapped,
                    tok,
                    "hello triune world",
                    device=dev,
                    max_new_tokens=4,
                    temperature=0.0,
                    force_depth=0,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    return_info=True,
                )
                self.assertEqual(reflex_info["route"], "REFLEX")
                self.assertGreaterEqual(int(reflex_info["tokens_count"]), 1)
            finally:
                exit_mgr = getattr(triune_wrapped, "_exit_manager", None)
                if exit_mgr is not None and hasattr(exit_mgr, "remove_hooks"):
                    exit_mgr.remove_hooks()

    def test_gemma3_gguf_rotary_buffers_system_chat_template_and_upgrade_guards(self) -> None:
        import struct

        import numpy as np

        from triune.runtime.gguf_loader import load_gguf_hf_model

        def _pack_str(s: str) -> bytes:
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        # Gemma3 GGUF with multimodal system template & sliding/full RoPE bases
        gemma3_tmpl = (
            "{{ bos_token }}"
            "{%- if messages[0]['role'] == 'system' -%}"
            "{%- set first_user_prefix = messages[0]['content'][0]['text'] + '\\n\\n' -%}"
            "{%- set loop_messages = messages[1:] -%}"
            "{%- else -%}"
            "{%- set first_user_prefix = '' -%}"
            "{%- set loop_messages = messages -%}"
            "{%- endif -%}"
            "{%- for message in loop_messages -%}"
            "{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\\n' + (first_user_prefix if loop.first else '') + message['content'] + '<end_of_turn>\\n' }}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\\n' }}{%- endif -%}"
        )
        vocab = (
            ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
            + [f"▁w{i}" for i in range(6, 61)]
            + ["▁hello", "▁triune", "▁world"]
        )
        token_types = [3] * 6 + [1] * (len(vocab) - 6)
        kvs: list[tuple[str, int, bytes]] = [
            ("general.architecture", 8, _pack_str("gemma3")),
            ("general.alignment", 4, struct.pack("<I", 32)),
            ("gemma3.block_count", 4, struct.pack("<I", 6)),
            ("gemma3.embedding_length", 4, struct.pack("<I", 32)),
            ("gemma3.attention.head_count", 4, struct.pack("<I", 2)),
            ("gemma3.attention.head_count_kv", 4, struct.pack("<I", 1)),
            ("gemma3.attention.key_length", 4, struct.pack("<I", 16)),
            ("gemma3.attention.value_length", 4, struct.pack("<I", 16)),
            ("gemma3.attention.sliding_window", 4, struct.pack("<I", 512)),
            ("gemma3.feed_forward_length", 4, struct.pack("<I", 64)),
            ("gemma3.context_length", 4, struct.pack("<I", 128)),
            ("gemma3.rope.local.freq_base", 6, struct.pack("<f", 10000.0)),
            ("gemma3.rope.global.freq_base", 6, struct.pack("<f", 1000000.0)),
            ("tokenizer.ggml.model", 8, _pack_str("llama")),
            ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 2)),
            ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 1)),
            ("tokenizer.ggml.unknown_token_id", 4, struct.pack("<I", 3)),
            ("tokenizer.ggml.padding_token_id", 4, struct.pack("<I", 0)),
            ("tokenizer.chat_template", 8, _pack_str(gemma3_tmpl)),
            ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, len(vocab)) + b"".join(_pack_str(t) for t in vocab)),
            (
                "tokenizer.ggml.token_type",
                9,
                struct.pack("<IQ", 4, len(vocab)) + struct.pack(f"<{len(vocab)}I", *token_types),
            ),
            (
                "tokenizer.ggml.scores",
                9,
                struct.pack("<IQ", 6, len(vocab)) + np.full(len(vocab), -1.0, dtype="<f4").tobytes(),
            ),
        ]
        rng = np.random.default_rng(11)
        tensor_specs: list[tuple[str, tuple[int, ...], bytes]] = [
            ("token_embd.weight", (32, 64), (rng.standard_normal((64, 32)).astype("<f4") * 0.05).tobytes()),
            ("output_norm.weight", (32,), np.ones(32, dtype="<f4").tobytes()),
        ]
        for li in range(6):
            tensor_specs.extend(
                [
                    (f"blk.{li}.attn_norm.weight", (32,), np.ones(32, dtype="<f4").tobytes()),
                    (f"blk.{li}.post_attention_norm.weight", (32,), np.ones(32, dtype="<f4").tobytes()),
                    (f"blk.{li}.ffn_norm.weight", (32,), np.ones(32, dtype="<f4").tobytes()),
                    (f"blk.{li}.post_ffw_norm.weight", (32,), np.ones(32, dtype="<f4").tobytes()),
                    (f"blk.{li}.attn_q_norm.weight", (16,), np.ones(16, dtype="<f4").tobytes()),
                    (f"blk.{li}.attn_k_norm.weight", (16,), np.ones(16, dtype="<f4").tobytes()),
                    (
                        f"blk.{li}.attn_q.weight",
                        (32, 32),
                        (rng.standard_normal((32, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.attn_k.weight",
                        (32, 16),
                        (rng.standard_normal((16, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.attn_v.weight",
                        (32, 16),
                        (rng.standard_normal((16, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.attn_output.weight",
                        (32, 32),
                        (rng.standard_normal((32, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.ffn_gate.weight",
                        (32, 64),
                        (rng.standard_normal((64, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.ffn_up.weight",
                        (32, 64),
                        (rng.standard_normal((64, 32)).astype("<f4") * 0.05).tobytes(),
                    ),
                    (
                        f"blk.{li}.ffn_down.weight",
                        (64, 32),
                        (rng.standard_normal((32, 64)).astype("<f4") * 0.05).tobytes(),
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            g3_blob = Path(tmpdir) / "sha256-gemma3test"
            header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensor_specs), len(kvs)))
            for k, vt, raw_val in kvs:
                header.extend(_pack_str(k) + struct.pack("<I", vt) + raw_val)
            offset = 0
            tensor_data = bytearray()
            for tname, dims, buf in tensor_specs:
                header.extend(
                    _pack_str(tname)
                    + struct.pack("<I", len(dims))
                    + struct.pack(f"<{len(dims)}Q", *dims)
                    + struct.pack("<IQ", 0, offset)
                )
                pad_len = (32 - (len(buf) % 32)) % 32
                tensor_data.extend(buf + b"\x00" * pad_len)
                offset += len(buf) + pad_len
            head_pad = (32 - (len(header) % 32)) % 32
            g3_blob.write_bytes(bytes(header) + (b"\x00" * head_pad) + bytes(tensor_data))

            m_g3, tok_g3, meta_g3, dev_g3, _, _ = load_gguf_hf_model(g3_blob, device="cpu")
            self.assertEqual(meta_g3["gguf_arch"], "gemma3")
            # 1. Non-parameter rotary buffers must be properly initialized (never to_empty() stack garbage)
            for bname, buf in m_g3.named_buffers():
                if "inv_freq" in bname:
                    self.assertTrue(torch.isfinite(buf).all(), f"{bname} has non-finite values")
                    self.assertGreater(float(buf.min()), 0.0, f"{bname} uninitialized min")
                    self.assertLessEqual(float(buf.max()), 1.0, f"{bname} uninitialized max")
            # 2. <unk> (id=3) must never pollute _triune_extra_eos_ids
            self.assertNotIn(3, m_g3._triune_extra_eos_ids)
            self.assertIn(5, m_g3._triune_extra_eos_ids)

            # 3. Studio DEFAULT_SYSTEM_PROMPT must format cleanly via _HFInferenceTokenizerAdapter
            rendered = tok_g3.apply_chat_template(
                [
                    {"role": "system", "content": "You are Triune Transformer, an engine with dynamic routing."},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "triune"},
                    {"role": "user", "content": "world"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            self.assertIn("<start_of_turn>user", rendered)
            self.assertIn("You are Triune Transformer", rendered)

            # 4. Upgrade endpoint -> Chat Playground inference (streaming + non-streaming), blueprint guard & .pt checkpoint
            snap = self._snapshot_engine_state()
            try:
                up_g3 = self.client.post(
                    "/v1/models/upgrade_to_triune",
                    json={"model_name": str(g3_blob), "apply_as_active": True, "transfer_weights": True},
                )
                self.assertEqual(up_g3.status_code, 200)
                uj = up_g3.json()
                self.assertEqual(uj["status"], "success")
                self.assertTrue(uj["weights_transferred"])
                self.assertTrue(bool(uj.get("message")))

                chat_res = self.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": uj["active_model_id"],
                        "messages": [
                            {"role": "system", "content": "You are Triune Transformer."},
                            {"role": "user", "content": "hello triune"},
                        ],
                        "max_tokens": 6,
                        "temperature": 0.0,
                        "stream": False,
                    },
                )
                self.assertEqual(chat_res.status_code, 200)
                self.assertGreaterEqual(chat_res.json()["telemetry"]["tokens_count"], 1)

                # Blueprint preset guard when transfer_weights=True
                pre_res = self.client.post(
                    "/v1/models/upgrade_to_triune",
                    json={"model_name": "llama-3-8b", "apply_as_active": True, "transfer_weights": True},
                )
                self.assertEqual(pre_res.json()["status"], "error")
                self.assertIn("blueprint", pre_res.json()["message"])

                # Triune native .pt checkpoint upgrade path
                pt_path = Path(tmpdir) / "native_triune.pt"
                tiny_m = TriuneTransformer(
                    vocab_size=64,
                    hidden_dim=64,
                    num_layers=4,
                    num_heads=2,
                    head_dim=32,
                    num_experts=2,
                    router_prefix_layers=1,
                    reflex_exit_layer=2,
                    limbic_exit_layer=3,
                )
                torch.save(
                    {
                        "model_state": tiny_m.state_dict(),
                        "config": {
                            "vocab_size": 64,
                            "num_layers": 4,
                            "hidden_dim": 64,
                            "num_heads": 2,
                            "head_dim": 32,
                            "num_experts": 2,
                            "router_prefix_layers": 1,
                            "reflex_exit_layer": 2,
                            "limbic_exit_layer": 3,
                        },
                    },
                    pt_path,
                )
                pt_res = self.client.post(
                    "/v1/models/upgrade_to_triune",
                    json={"model_name": str(pt_path), "apply_as_active": True, "transfer_weights": True},
                )
                self.assertEqual(pt_res.status_code, 200)
                self.assertEqual(pt_res.json()["activation_kind"], "triune_checkpoint_loaded")
                self.assertTrue(pt_res.json()["weights_transferred"])
            finally:
                self._restore_engine_state(snap)

            # Unsupported GGUF architecture (phi2) fails fast & explicitly
            phi_blob = Path(tmpdir) / "unsupported_phi2.gguf"
            phi_kvs = [("general.architecture", 8, _pack_str("phi2")), ("phi2.block_count", 4, struct.pack("<I", 2))]
            phi_hdr = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(phi_kvs)))
            for k, vt, raw_val in phi_kvs:
                phi_hdr.extend(_pack_str(k) + struct.pack("<I", vt) + raw_val)
            phi_blob.write_bytes(bytes(phi_hdr))
            with self.assertRaises(ValueError):
                load_gguf_hf_model(phi_blob, device="cpu")


if __name__ == "__main__":
    unittest.main()
