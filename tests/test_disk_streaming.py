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


if __name__ == "__main__":
    unittest.main()
