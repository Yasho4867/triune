"""End-to-end tests for Phase 1 & Phase 2: NVMe NVFP4 disk streaming, chunked layer streaming,
QLoRA SFT with assistant masking, unified HF/Triune inference, MemoryPlanner HF sizing, and Studio API.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch


def _create_synthetic_qwen2_dir(model_dir: Path, *, tie_word_embeddings: bool = True) -> Path:
    """Create a tiny self-contained sharded Qwen2ForCausalLM directory + ByteLevel Qwen2 tokenizer."""
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {sym: idx for idx, sym in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    tok_obj = Tokenizer(BPE(vocab=vocab, merges=[]))
    tok_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok_obj.decoder = decoders.ByteLevel()

    chat_tpl = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        chat_template=chat_tpl,
    )
    hf_tok.add_special_tokens({"additional_special_tokens": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]})
    hf_tok.save_pretrained(str(model_dir))

    vocab_size = max(288, len(hf_tok))
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=int(hf_tok.eos_token_id or 0),
        eos_token_id=int(hf_tok.eos_token_id or 0),
        pad_token_id=int(hf_tok.pad_token_id or 0),
        tie_word_embeddings=tie_word_embeddings,
    )
    ref = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    # Save as 2 shards (< 50 KB each) so multi-shard safe_open streaming is exercised directly.
    ref.save_pretrained(str(model_dir), safe_serialization=True, max_shard_size="30KB")
    return model_dir


class TestDiskStreamingAndUnifiedHF(unittest.TestCase):
    def test_build_and_load_disk_streamed_qwen2(self):
        from transformers import AutoModelForCausalLM

        from triune.kernels.fast_nvfp4 import NVFP4Linear
        from triune.runtime.disk_streaming import build_nvfp4_disk_cache, load_disk_streamed_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hf_dir = _create_synthetic_qwen2_dir(tmp_path / "tiny_qwen2")
            cache_dir = tmp_path / "nvfp4_cache"

            resolved_cache = build_nvfp4_disk_cache(
                hf_dir,
                cache_dir=cache_dir,
                hadamard=True,
                protect_sensitive=True,
                quantize_lm_head=False,
                compute_device=device,
                local_files_only=True,
            )
            manifest = json.loads((resolved_cache / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_layers"], 2)
            self.assertGreater(manifest["compression_ratio"], 1.5)
            self.assertTrue((resolved_cache / "layers" / "layer_0000.safetensors").is_file())
            self.assertTrue((resolved_cache / "layers" / "layer_0001.safetensors").is_file())
            self.assertTrue((resolved_cache / "root.safetensors").is_file())

            ref = AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype=torch.bfloat16).to(device).eval()
            model, engine, _ = load_disk_streamed_model(
                resolved_cache,
                device=device,
                hadamard=True,
                activation_checkpointing=True,
                async_prefetch=True,
                local_files_only=True,
            )
            try:
                # Off-GPU decoder blocks hold zero-byte placeholders until staged for forward
                for layer in model.model.layers:
                    for mod in layer.modules():
                        if isinstance(mod, NVFP4Linear):
                            self.assertEqual(mod.packed_weight.numel(), 0)
                            self.assertEqual(mod.scale.numel(), 0)

                ids = torch.tensor([[1, 3, 6, 7, 2, 1, 4]], dtype=torch.long, device=device)
                with torch.no_grad():
                    ref_logits = ref(input_ids=ids).logits.float().view(-1)
                    stream_logits = model(input_ids=ids).logits.float().view(-1)
                cos = torch.nn.functional.cosine_similarity(ref_logits, stream_logits, dim=0).item()
                self.assertGreater(cos, 0.94)

                # Multi-token autoregressive generation with KV cache (use_cache=True)
                with torch.no_grad():
                    out_ids = model.generate(input_ids=ids, max_new_tokens=3, do_sample=False, use_cache=True)
                self.assertEqual(out_ids.shape, (1, ids.shape[1] + 3))

                # Placeholders returned to 0 elements after forward exit
                for layer in model.model.layers:
                    for mod in layer.modules():
                        if isinstance(mod, NVFP4Linear):
                            self.assertEqual(mod.packed_weight.numel(), 0)
            finally:
                engine.detach()

    def test_init_empty_weights_without_accelerate_preserves_buffers_and_ties(self):
        from unittest.mock import patch

        from triune.runtime.disk_streaming import _init_empty_weights

        class TinyRotaryAndLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = torch.nn.Linear(16, 16, bias=False)
                self.proj_out = torch.nn.Linear(16, 16, bias=False)
                self.proj_out.weight = self.proj_in.weight
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8.0))
                self.register_buffer("inv_freq", inv_freq, persistent=False)

        orig_reg = torch.nn.Module.register_parameter
        with patch.dict("sys.modules", {"accelerate": None}):
            with _init_empty_weights():
                m = TinyRotaryAndLinear()
            self.assertEqual(m.proj_in.weight.device.type, "meta")
            self.assertIs(m.proj_out.weight, m.proj_in.weight)
            self.assertEqual(m.inv_freq.device.type, "cpu")
            self.assertTrue(torch.isfinite(m.inv_freq).all())
            self.assertGreater(m.inv_freq.abs().sum().item(), 0.0)
        self.assertIs(torch.nn.Module.register_parameter, orig_reg)

    def test_gguf_vectorized_cpu_and_torch_dequant_parity(self):
        import numpy as np

        from triune.runtime.gguf_loader import (
            _BLOCK_META,
            _GGML_Q4_0,
            _GGML_Q4_1,
            _GGML_Q4_K,
            _GGML_Q5_0,
            _GGML_Q5_K,
            _GGML_Q6_K,
            _GGML_Q8_0,
            _dequantize_ggml_torch_chunk,
            dequantize_ggml_tensor,
        )

        rng = np.random.default_rng(7)
        devs = [torch.device("cpu")]
        if torch.cuda.is_available():
            devs.append(torch.device("cuda"))

        for ggml_type in (_GGML_Q4_0, _GGML_Q4_1, _GGML_Q5_0, _GGML_Q8_0, _GGML_Q4_K, _GGML_Q5_K, _GGML_Q6_K):
            blk_elems, blk_bytes = _BLOCK_META[ggml_type]
            nb = 256
            raw = rng.integers(0, 256, size=(nb, blk_bytes), dtype=np.uint8)
            if ggml_type == _GGML_Q6_K:
                raw[:, 208:210] = 0x38
            elif ggml_type in (_GGML_Q4_1, _GGML_Q4_K, _GGML_Q5_K):
                raw[:, :4] = 0x38
            else:
                raw[:, :2] = 0x38
            if ggml_type == _GGML_Q5_0:
                raw[0, 5] |= 0x80
            raw_bytes = raw.tobytes()
            dims = (64, (nb * blk_elems) // 64)

            ref = dequantize_ggml_tensor(raw_bytes, dims, ggml_type, torch.float32, compute_device=None)
            for dev in devs:
                got = (
                    _dequantize_ggml_torch_chunk(torch.from_numpy(raw.copy()).to(dev), ggml_type, torch.float32)
                    .cpu()
                    .reshape(tuple(reversed(dims)))
                )
                self.assertEqual(float((ref - got).abs().max().item()), 0.0)

    def test_disk_streamed_qlora_sft_masking_and_step(self):
        from triune.inference import generate_response, load_model_for_inference
        from triune.runtime.disk_streaming import build_nvfp4_disk_cache, load_disk_streamed_model
        from triune.trainer.finetune import LoRAConfig, TriuneFineTuner

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hf_dir = _create_synthetic_qwen2_dir(tmp_path / "tiny_qwen2_sft")
            cache_dir = build_nvfp4_disk_cache(
                hf_dir,
                cache_dir=tmp_path / "nvfp4_cache_sft",
                hadamard=True,
                quantize_lm_head=False,
                compute_device=device,
                local_files_only=True,
            )

            ds_path = tmp_path / "sft_chat.jsonl"
            samples = [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "ok yes answer"},
                    ]
                }
                for _ in range(8)
            ]
            ds_path.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")

            model, engine, _ = load_disk_streamed_model(
                cache_dir,
                device=device,
                hadamard=True,
                activation_checkpointing=True,
                async_prefetch=True,
                local_files_only=True,
            )
            adapter_out = tmp_path / "adapter_out"
            try:
                cfg = LoRAConfig(rank=4, alpha=8.0, quantization="NVFP4", quantize_base=True, hadamard=True)
                tuner = TriuneFineTuner(
                    model,
                    cfg,
                    model_name_or_path=str(hf_dir),
                    adapter_device=device,
                )
                self.assertGreater(tuner.count_trainable_parameters(), 0)
                res = tuner.fit(
                    dataset_path=ds_path,
                    output_dir=adapter_out,
                    epochs=1,
                    batch_size=2,
                    seq_len=48,
                    lr=1e-3,
                    max_steps=2,
                    grad_accum_steps=2,
                    mask_prompt_tokens=True,
                    use_chunked_ce=True,
                )
                self.assertTrue(torch.isfinite(torch.tensor(float(res["final_loss"]))))
                self.assertEqual(res["total_steps"], 2)
                self.assertTrue((adapter_out / "adapter_config.json").is_file())
                self.assertTrue((adapter_out / "adapter_model.safetensors").is_file())
                # Verify adapter-only artifact stays compact (no full base model dump)
                self.assertLess((adapter_out / "adapter_model.safetensors").stat().st_size, 2 * 1024 * 1024)
            finally:
                engine.detach()

            # Reload via unified loader with trained LoRA adapter attached
            loaded_model, tok, _meta, dev, pad_id, eos_id = load_model_for_inference(
                cache_dir,
                device=device,
                adapter_path=adapter_out,
                disk_stream=True,
                local_files_only=True,
            )
            try:
                out = generate_response(
                    loaded_model,
                    tok,
                    "hi",
                    device=dev,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    max_new_tokens=3,
                    temperature=0.0,
                    return_info=True,
                )
                self.assertEqual(out["route"], "HF")
                self.assertGreaterEqual(out["tokens_count"], 1)
            finally:
                if getattr(loaded_model, "_streaming_engine", None) is not None:
                    loaded_model._streaming_engine.detach()

    def test_chunked_layer_streaming_with_fp4_readonly_invariance(self):
        from triune.kernels.fast_nvfp4 import NVFP4Linear
        from triune.model.fp4 import convert_to_nvfp4
        from triune.model.transformer import TriuneTransformer
        from triune.runtime.streaming import enable_layer_streaming

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TriuneTransformer(
            vocab_size=64,
            hidden_dim=64,
            num_layers=6,
            num_heads=2,
            head_dim=32,
            num_experts=2,
            router_prefix_layers=1,
            reflex_exit_layer=2,
            limbic_exit_layer=4,
        )
        if device.type == "cuda":
            model = model.to(torch.bfloat16)
        convert_to_nvfp4(model, hadamard=True, protect_sensitive=True)
        engine = enable_layer_streaming(
            model,
            device=device,
            chunk_size=2,
            fp4_weights=True,
            resident_gpu_layers=0,
            activation_checkpointing=True,
        )
        try:
            nv_mods = [m for m in model.modules() if isinstance(m, NVFP4Linear)]
            self.assertGreater(len(nv_mods), 0)
            # Check quantized weight staging buffers are marked read_only so D2H copy-back is skipped
            nv_ro = [
                engine.stager.buffers[id(m.packed_weight)].read_only
                for m in nv_mods
                if id(m.packed_weight) in engine.stager.buffers
            ]
            self.assertGreater(len(nv_ro), 0)
            self.assertTrue(all(nv_ro))

            # Trainable params still update across chunked windowed streaming
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=1e-3)
            engine.set_optimizer(opt)
            x = torch.randint(0, 64, (2, 12), device=device)
            logits, _ = model(x, force_depth=2)
            loss = logits.float().sum()
            loss.backward()
            engine.step_streaming_optimizer()
            self.assertTrue(torch.isfinite(loss))
        finally:
            engine.detach()

    def test_unified_inference_and_memory_planner_offline(self):
        from triune.inference import generate_response, load_model_for_inference
        from triune.runtime.memory_planner import MemoryPlanner

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hf_dir = _create_synthetic_qwen2_dir(tmp_path / "tiny_qwen2_infer")

            est = MemoryPlanner.estimate_hf_model(hf_dir, local_files_only=True)
            self.assertGreater(est["total_params"], 50_000)
            self.assertIn(
                est["recommended_mode"],
                {"resident_bf16", "resident_nvfp4", "ram_layer_streaming", "disk_nvfp4_streaming"},
            )
            self.assertLess(est["nvfp4_disk_cache_gb"], est["bf16_weights_gb"] + 0.01)

            # Verify synthetic 27B config triggers disk_nvfp4_streaming on 8GB VRAM + 15GB RAM consumer profile
            q27_dir = tmp_path / "fake_qwen_27b"
            q27_dir.mkdir(parents=True, exist_ok=True)
            (q27_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2",
                        "architectures": ["Qwen2ForCausalLM"],
                        "hidden_size": 5120,
                        "intermediate_size": 17408,
                        "num_hidden_layers": 64,
                        "num_attention_heads": 40,
                        "num_key_value_heads": 8,
                        "vocab_size": 151936,
                    }
                ),
                encoding="utf-8",
            )
            est_27b = MemoryPlanner.estimate_hf_model(
                q27_dir, available_vram_gb=8.0, available_ram_gb=15.0, local_files_only=True
            )
            self.assertEqual(est_27b["recommended_mode"], "disk_nvfp4_streaming")
            self.assertGreater(est_27b["bf16_weights_gb"], 40.0)
            self.assertLess(est_27b["nvfp4_disk_cache_gb"], est_27b["bf16_weights_gb"])

            model, tok, _meta, dev, pad_id, eos_id = load_model_for_inference(
                hf_dir, device=device, quantize="nvfp4", local_files_only=True
            )
            info = generate_response(
                model,
                tok,
                "hello world",
                device=dev,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                max_new_tokens=3,
                temperature=0.0,
                return_info=True,
            )
            self.assertEqual(info["route"], "HF")
            self.assertGreaterEqual(info["tokens_count"], 1)

    def test_studio_api_hf_load_and_chat(self):
        from fastapi.testclient import TestClient

        from triune.api.routes import pytorch_state
        from triune.api.server import create_app

        session_file = Path("checkpoints/session_state.json")
        session_backup = session_file.read_text(encoding="utf-8") if session_file.is_file() else None
        prev_testing = os.environ.get("TRIUNE_TESTING")
        os.environ["TRIUNE_TESTING"] = "1"

        saved_model_streaming_engine = getattr(pytorch_state.model, "_streaming_engine", None)
        saved_state = {
            "model": pytorch_state.model,
            "tokenizer": pytorch_state.tokenizer,
            "optimizer": pytorch_state.optimizer,
            "training_engine": pytorch_state.training_engine,
            "streaming_engine": pytorch_state.streaming_engine,
            "active_adapter": pytorch_state.active_adapter,
            "active_finetuner": getattr(pytorch_state, "active_finetuner", None),
            "active_model_id": pytorch_state.active_model_id,
            "active_checkpoint": pytorch_state.active_checkpoint,
            "weights_loaded": pytorch_state.weights_loaded,
            "architecture_spec": dict(pytorch_state.architecture_spec),
            "precision": pytorch_state.precision,
            "step": pytorch_state.step,
            "total_tokens_trained": pytorch_state.total_tokens_trained,
            "tokens_trained": pytorch_state.tokens_trained,
            "pad_token_id": pytorch_state.pad_token_id,
            "history": list(pytorch_state.history),
        }

        client_kwargs = {}
        if "client" in inspect.signature(TestClient).parameters:
            client_kwargs["client"] = ("127.0.0.1", 43210)
        client = TestClient(create_app(), **client_kwargs)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                bad_file = tmp_path / "payload.exe"
                bad_file.write_bytes(b"bad")
                bad_resp = client.post("/v1/models/load", json={"checkpoint_path": str(bad_file)})
                self.assertEqual(bad_resp.status_code, 200)
                self.assertEqual(bad_resp.json()["status"], "error")
                self.assertIn("Security restriction", bad_resp.json()["message"])

                hf_dir = _create_synthetic_qwen2_dir(tmp_path / "tiny_studio_qwen2")
                load_resp = client.post(
                    "/v1/models/load",
                    json={"checkpoint_path": str(hf_dir), "nvfp4": True, "disk_stream": False},
                )
                self.assertEqual(load_resp.status_code, 200)
                body = load_resp.json()
                self.assertEqual(body["status"], "success")
                self.assertTrue(body["weights_loaded"])
                self.assertEqual(body["architecture"]["num_layers"], 2)

                chat_resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "hello world"}],
                        "max_tokens": 3,
                        "temperature": 0.0,
                    },
                )
                self.assertEqual(chat_resp.status_code, 200)
                chat_json = chat_resp.json()
                self.assertIn("choices", chat_json)

                spec_resp = client.get("/v1/playground/neurons")
                self.assertEqual(spec_resp.status_code, 200)
                self.assertEqual(spec_resp.json()["num_layers"], 2)
        finally:
            if prev_testing is None:
                os.environ.pop("TRIUNE_TESTING", None)
            else:
                os.environ["TRIUNE_TESTING"] = prev_testing
            for eng_attr in ("_streaming_engine", "_disk_streaming_engine"):
                eng = getattr(pytorch_state.model, eng_attr, None)
                if eng is not None and hasattr(eng, "detach"):
                    try:
                        eng.detach()
                    except Exception:
                        pass
            pytorch_state.model = saved_state["model"]
            pytorch_state.tokenizer = saved_state["tokenizer"]
            pytorch_state.optimizer = saved_state["optimizer"]
            pytorch_state.training_engine = saved_state["training_engine"]
            pytorch_state.streaming_engine = saved_state["streaming_engine"]
            pytorch_state.active_adapter = saved_state["active_adapter"]
            pytorch_state.active_finetuner = saved_state["active_finetuner"]
            pytorch_state.active_model_id = saved_state["active_model_id"]
            pytorch_state.active_checkpoint = saved_state["active_checkpoint"]
            pytorch_state.weights_loaded = saved_state["weights_loaded"]
            pytorch_state.architecture_spec = saved_state["architecture_spec"]
            pytorch_state.precision = saved_state["precision"]
            pytorch_state.step = saved_state["step"]
            pytorch_state.total_tokens_trained = saved_state["total_tokens_trained"]
            pytorch_state.tokens_trained = saved_state["tokens_trained"]
            pytorch_state.pad_token_id = saved_state["pad_token_id"]
            pytorch_state.history = saved_state["history"]
            if saved_model_streaming_engine is not None and not saved_model_streaming_engine.is_attached:
                saved_model_streaming_engine.attach(optimizer=saved_state["optimizer"])
            if session_backup is not None:
                session_file.write_text(session_backup, encoding="utf-8")
            elif session_file.is_file():
                session_file.unlink()

    def test_studio_saved_models_discovery_and_upgrade(self):
        from fastapi.testclient import TestClient

        from triune.api.server import create_app
        from triune.model.transformer import TriuneTransformer
        from triune.runtime.disk_streaming import build_nvfp4_disk_cache

        client_kwargs = {}
        if "client" in inspect.signature(TestClient).parameters:
            client_kwargs["client"] = ("127.0.0.1", 43210)
        client = TestClient(create_app(), **client_kwargs)

        res_list = client.get("/v1/models")
        self.assertEqual(res_list.status_code, 200)
        list_data = res_list.json()
        self.assertIn("saved_models", list_data)
        saved_ids = {m["id"] for m in list_data["saved_models"]}
        self.assertFalse(any("lora_adapter" in mid.lower() for mid in saved_ids))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hf_dir = _create_synthetic_qwen2_dir(tmp_path / "saved_hf_qwen2")
            cache_dir = build_nvfp4_disk_cache(
                hf_dir,
                cache_dir=tmp_path / "saved_nvfp4_cache",
                hadamard=True,
                quantize_lm_head=False,
                local_files_only=True,
            )
            pt_path = tmp_path / "saved_triune.pt"
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
            torch.save({"model_state": tiny_m.state_dict(), "config": {"num_layers": 4, "hidden_dim": 64}}, pt_path)

            for target, exp_layers, exp_dim in [
                (str(hf_dir), 2, 64),
                (str(cache_dir), 2, 64),
                (str(pt_path), 4, 64),
            ]:
                up_res = client.post(
                    "/v1/models/upgrade_to_triune",
                    json={"model_name": target, "reflex_ratio": 0.25, "limbic_experts": 4, "apply_as_active": False},
                )
                self.assertEqual(up_res.status_code, 200)
                up_json = up_res.json()
                self.assertEqual(up_json["status"], "success")
                self.assertEqual(len(up_json["layer_mapping"]), exp_layers)
                self.assertEqual(up_json["tier_summary"]["hidden_dim"], exp_dim)

        app_js = Path("studio/src/app.js").read_text(encoding="utf-8")
        self.assertIn("Saved Models, Checkpoints & HF Cache", app_js)
        self.assertIn("fetchSavedModels", app_js)
        self.assertIn("handleLoadSavedModel", app_js)
        self.assertIn("useState('llama-3-8b')", app_js)

    def test_pc_model_scanner_ollama_vllm_hermes_gguf_and_loading_menu(self):
        import struct

        from fastapi.testclient import TestClient

        from triune.api.server import create_app

        def _write_gguf(
            out_path: Path,
            *,
            arch: str,
            layers: int,
            hidden_dim: int,
            heads: int,
            ffn_dim: int,
            vocab_size: int,
            file_type: int = 15,
        ) -> Path:
            def _s(txt: str) -> bytes:
                b = txt.encode("utf-8")
                return struct.pack("<Q", len(b)) + b

            kv_entries = [
                ("general.architecture", 8, _s(arch)),
                ("general.name", 8, _s(f"test-{arch}")),
                ("general.file_type", 4, struct.pack("<I", file_type)),
                (f"{arch}.block_count", 4, struct.pack("<I", layers)),
                (f"{arch}.embedding_length", 4, struct.pack("<I", hidden_dim)),
                (f"{arch}.attention.head_count", 4, struct.pack("<I", heads)),
                (f"{arch}.feed_forward_length", 4, struct.pack("<I", ffn_dim)),
                ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, vocab_size)),
            ]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"GGUF")
                f.write(struct.pack("<IQQ", 3, 0, len(kv_entries)))
                for k, vtype, payload in kv_entries:
                    f.write(_s(k))
                    f.write(struct.pack("<I", vtype))
                    f.write(payload)
            return out_path

        client_kwargs = {}
        if "client" in inspect.signature(TestClient).parameters:
            client_kwargs["client"] = ("127.0.0.1", 43210)
        client = TestClient(create_app(), **client_kwargs)

        env_keys = ("OLLAMA_MODELS", "HF_HUB_CACHE", "VLLM_CACHE_ROOT", "HERMES_HOME")
        saved_env = {k: os.environ.get(k) for k in env_keys}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tp = Path(tmp)

                # 1. Synthetic Ollama manifest + config + GGUF blob
                ol_root = tp / "ollama_models"
                blobs = ol_root / "blobs"
                blobs.mkdir(parents=True, exist_ok=True)
                cfg_digest = "sha256:1111"
                mod_digest = "sha256:2222"
                (blobs / "sha256-1111").write_text(
                    json.dumps({"model_format": "gguf", "model_type": "7B", "file_type": "Q4_K_M"}),
                    encoding="utf-8",
                )
                _write_gguf(
                    blobs / "sha256-2222",
                    arch="qwen2",
                    layers=28,
                    hidden_dim=3584,
                    heads=28,
                    ffn_dim=18944,
                    vocab_size=151936,
                    file_type=15,
                )
                mf = ol_root / "manifests" / "registry.ollama.ai" / "library" / "qwen-custom" / "latest"
                mf.parent.mkdir(parents=True, exist_ok=True)
                mf.write_text(
                    json.dumps(
                        {
                            "config": {"digest": cfg_digest},
                            "layers": [
                                {"mediaType": "application/vnd.ollama.image.model", "digest": mod_digest, "size": 4096}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                # 2. Synthetic HF Hub cache (Hermes repo detected cleanly)
                hf_hub = tp / "hf_hub"
                snap = hf_hub / "models--NousResearch--Hermes-Test" / "snapshots" / "rev1"
                snap.mkdir(parents=True, exist_ok=True)
                (snap / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": "llama",
                            "num_hidden_layers": 16,
                            "hidden_size": 2048,
                            "num_attention_heads": 16,
                            "intermediate_size": 5632,
                            "vocab_size": 32000,
                        }
                    ),
                    encoding="utf-8",
                )

                # 3. Synthetic vLLM cache dir
                vllm_dir = tp / "vllm_cache" / "qwen3_serving"
                vllm_dir.mkdir(parents=True, exist_ok=True)
                (vllm_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": "qwen2",
                            "num_hidden_layers": 32,
                            "hidden_size": 4096,
                            "num_attention_heads": 32,
                            "intermediate_size": 12288,
                            "vocab_size": 151936,
                        }
                    ),
                    encoding="utf-8",
                )

                # 4. Synthetic Hermes local GGUF + Extra folder 27B GGUF
                hermes_dir = tp / "hermes_home"
                _write_gguf(
                    hermes_dir / "models" / "hermes_local.gguf",
                    arch="llama",
                    layers=12,
                    hidden_dim=1024,
                    heads=8,
                    ffn_dim=4096,
                    vocab_size=32000,
                )
                extra_dir = tp / "extra_models"
                extra_gguf = _write_gguf(
                    extra_dir / "qwen27b_custom.gguf",
                    arch="qwen2",
                    layers=64,
                    hidden_dim=5120,
                    heads=40,
                    ffn_dim=17408,
                    vocab_size=151936,
                    file_type=15,
                )

                os.environ["OLLAMA_MODELS"] = str(ol_root)
                os.environ["HF_HUB_CACHE"] = str(hf_hub)
                os.environ["VLLM_CACHE_ROOT"] = str(tp / "vllm_cache")
                os.environ["HERMES_HOME"] = str(hermes_dir)

                scan_res = client.post(
                    "/v1/models/scan",
                    json={"extra_paths": [str(extra_dir)], "include_wsl": False, "max_depth": 4},
                )
                self.assertEqual(scan_res.status_code, 200)
                sdata = scan_res.json()
                self.assertEqual(sdata["status"], "success")
                found_by_id = {m["id"]: m for m in sdata["models"]}

                self.assertIn("ollama:qwen-custom:latest", found_by_id)
                self.assertEqual(found_by_id["ollama:qwen-custom:latest"]["num_layers"], 28)
                self.assertEqual(found_by_id["ollama:qwen-custom:latest"]["quantization"], "Q4_K_M")

                self.assertIn("NousResearch/Hermes-Test", found_by_id)
                self.assertEqual(found_by_id["NousResearch/Hermes-Test"]["source"], "hermes")

                vllm_id = str(vllm_dir).replace("\\", "/")
                self.assertIn(vllm_id, found_by_id)
                self.assertEqual(found_by_id[vllm_id]["source"], "vllm")

                extra_id = str(extra_gguf).replace("\\", "/")
                self.assertIn(extra_id, found_by_id)
                self.assertEqual(found_by_id[extra_id]["num_layers"], 64)
                self.assertEqual(found_by_id[extra_id]["hidden_dim"], 5120)

                # Every scanned model source must transmute cleanly using its true dimensions
                for target, exp_l, exp_d in [
                    ("ollama:qwen-custom:latest", 28, 3584),
                    (vllm_id, 32, 4096),
                    (extra_id, 64, 5120),
                ]:
                    tr = client.post(
                        "/v1/models/upgrade_to_triune",
                        json={"model_name": target, "apply_as_active": False},
                    )
                    self.assertEqual(tr.status_code, 200)
                    tj = tr.json()
                    self.assertEqual(len(tj["layer_mapping"]), exp_l)
                    self.assertEqual(tj["tier_summary"]["hidden_dim"], exp_d)

                # Helpful GGUF/Ollama message when attempting direct PyTorch load
                ol_load = client.post("/v1/models/load", json={"checkpoint_path": "ollama:qwen-custom:latest"})
                self.assertEqual(ol_load.status_code, 200)
                self.assertEqual(ol_load.json()["status"], "error")
                self.assertIn("Transmute", ol_load.json()["message"])
        finally:
            for k, old in saved_env.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

        app_js = Path("studio/src/app.js").read_text(encoding="utf-8")
        self.assertIn("__scan_pc__", app_js)
        self.assertIn("showModelScannerModal", app_js)
        self.assertIn("PC Model Scanner & Universal Transmuter", app_js)
        self.assertIn("handleScanPcModels", app_js)

    def test_custom_and_transmuted_inference_routing_fixes(self):
        """Verify transmuted models aren't overwritten, local google/openai models bypass BYOK, and HF file/encoder guards work."""
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from triune.api import routes as routes_mod
        from triune.api.server import create_app

        app = create_app()
        client = TestClient(app)

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            # 1. Encoder-only HF config (e.g. BERT) should be flagged non-loadable in scanner & blocked cleanly
            bert_dir = tp / "bert-custom"
            bert_dir.mkdir(parents=True)
            (bert_dir / "config.json").write_text(
                json.dumps({"model_type": "bert", "architectures": ["BertForMaskedLM"], "hidden_size": 768}),
                encoding="utf-8",
            )
            (bert_dir / "model.safetensors").write_bytes(b"stub")

            scan = client.post(
                "/v1/models/scan",
                json={"extra_paths": [str(tp)], "include_wsl": False, "max_depth": 3},
            ).json()
            found = {m["id"]: m for m in scan["models"]}
            bert_id = str(bert_dir).replace("\\", "/")
            self.assertIn(bert_id, found)
            self.assertFalse(found[bert_id]["is_causal_lm"])
            self.assertFalse(found[bert_id]["can_load_direct"])

            bert_load = client.post("/v1/models/load", json={"checkpoint_path": str(bert_dir / "model.safetensors")})
            self.assertEqual(bert_load.status_code, 200)
            self.assertEqual(bert_load.json()["status"], "error")
            self.assertIn("CausalLM", bert_load.json()["message"])

        # 2. Transmuted active model must not call load_model_checkpoint during generate_text
        session_file = Path("studio_session.json")
        session_backup = session_file.read_text(encoding="utf-8") if session_file.is_file() else None
        st = routes_mod.pytorch_state
        saved_stream = getattr(st.model, "_streaming_engine", None)
        saved_state = {
            "model": st.model,
            "tokenizer": st.tokenizer,
            "optimizer": st.optimizer,
            "training_engine": st.training_engine,
            "streaming_engine": st.streaming_engine,
            "active_adapter": st.active_adapter,
            "active_finetuner": getattr(st, "active_finetuner", None),
            "active_model_id": st.active_model_id,
            "active_checkpoint": st.active_checkpoint,
            "weights_loaded": st.weights_loaded,
            "explicit_architecture_active": getattr(st, "explicit_architecture_active", False),
            "architecture_spec": dict(st.architecture_spec),
            "precision": st.precision,
            "step": st.step,
            "total_tokens_trained": st.total_tokens_trained,
            "tokens_trained": st.tokens_trained,
            "pad_token_id": st.pad_token_id,
            "history": list(st.history),
            "batch_size": st.batch_size,
            "seq_len": st.seq_len,
            "gpu_layers": st.gpu_layers,
            "gpu_offload_pct": st.gpu_offload_pct,
        }
        try:
            up = client.post(
                "/v1/models/upgrade_to_triune",
                json={"model_name": "qwen-2.5-7b", "apply_as_active": True},
            )
            self.assertEqual(up.status_code, 200)
            self.assertEqual(st.active_model_id, "triune-qwen-2.5-7b")
            self.assertTrue(st.explicit_architecture_active)

            with (
                patch.object(st, "load_model_checkpoint") as mock_load_ckpt,
                patch(
                    "triune.inference.generate_response",
                    return_value={
                        "text": "ok",
                        "route": "CORTEX",
                        "tokens_count": 1,
                        "tokens_per_sec": 10,
                        "latency_ms": 5,
                        "cancelled": False,
                    },
                ),
            ):
                st.generate_text("hello", max_new_tokens=4)
                mock_load_ckpt.assert_not_called()

            # 3. Loaded local google/... model must run locally rather than routing to Gemini BYOK
            st.active_model_id = "google/gemma-2-2b"
            st.weights_loaded = True
            st.explicit_architecture_active = False
            with patch.object(
                st,
                "generate_text",
                return_value={
                    "text": "local gemma reply",
                    "route": "CORTEX",
                    "tokens_count": 3,
                    "tokens_per_sec": 30,
                    "latency_ms": 12,
                    "vram_gb": 2.1,
                    "checkpoint": "gemma-2-2b",
                    "weights_loaded": True,
                    "cancelled": False,
                },
            ) as mock_gen:
                chat_res = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "google/gemma-2-2b",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                )
            self.assertEqual(chat_res.status_code, 200)
            self.assertEqual(chat_res.json()["choices"][0]["message"]["content"], "local gemma reply")
            mock_gen.assert_called_once()
        finally:
            for eng_attr in ("_streaming_engine", "_disk_streaming_engine"):
                eng = getattr(st.model, eng_attr, None)
                if eng is not None and eng is not saved_stream and hasattr(eng, "detach"):
                    try:
                        eng.detach()
                    except Exception:
                        pass
            for k, v in saved_state.items():
                setattr(st, k, v)
            if saved_stream is not None and not saved_stream.is_attached:
                saved_stream.attach(optimizer=saved_state["optimizer"])
            if session_backup is not None:
                session_file.write_text(session_backup, encoding="utf-8")
            elif session_file.is_file():
                session_file.unlink()


if __name__ == "__main__":
    unittest.main()
