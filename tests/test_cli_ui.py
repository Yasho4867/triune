"""Tests for the Rich Lucid Machines CLI, new scan/transmute/agent subcommands, and chat slash commands."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from lucid_machines.cli import main as cli_main
from scripts.chat import main as chat_main

try:
    from tests.test_weight_transmutation import _build_tiny_hf_llama_dir
except ModuleNotFoundError:
    from test_weight_transmutation import _build_tiny_hf_llama_dir


class TestLucidMachinesBeautifiedCLI(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> str:
        buf = io.StringIO()
        with patch.object(sys, "argv", ["lucid-machines", *argv]), redirect_stdout(buf):
            cli_main()
        return buf.getvalue()

    def test_help_models_gpucheck_doctor_and_memory_plan(self) -> None:
        help_out = self._run_cli(["--help"])
        self.assertIn("LUCID MACHINES", help_out.upper())
        self.assertIn("lucid-machines agent", help_out)
        self.assertIn("lucid-machines transmute", help_out)
        self.assertIn("lucid-machines scan", help_out)

        models_out = self._run_cli(["models"])
        self.assertIn("Model Architecture Zoo", models_out)
        self.assertIn("triune-small", models_out)

        gpu_out = self._run_cli(["gpucheck"])
        self.assertIn("GPU", gpu_out.upper())

        doc_out = self._run_cli(["doctor"])
        self.assertIn("SYSTEM DOCTOR", doc_out.upper())
        self.assertIn("CentroidSteer", doc_out)

        mem_out = self._run_cli(["plan-memory", "--vram-gb", "8"])
        self.assertIn("VRAM Memory Plan Estimate", mem_out)

    def test_scan_command_renders_live_ollama_and_hf_status(self) -> None:
        fake_scan = {
            "models": [
                {
                    "id": "ollama:gemma3:4b",
                    "name": "gemma3:4b",
                    "source": "ollama",
                    "kind": "ollama",
                    "architecture": "gemma3",
                    "quantization": "Q4_K_M",
                    "size_str": "3.3 GB",
                    "ollama_running": True,
                    "can_load_direct": False,
                    "can_transmute": True,
                },
                {
                    "id": "models/qwen",
                    "name": "qwen-local",
                    "source": "huggingface",
                    "kind": "hf_local",
                    "architecture": "Qwen2ForCausalLM",
                    "quantization": "BF16",
                    "size_str": "1.1 GB",
                    "ollama_running": False,
                    "can_load_direct": True,
                    "can_transmute": True,
                },
            ]
        }
        with patch("lucid_machines.runtime.model_scanner.scan_pc_models", return_value=fake_scan):
            out = self._run_cli(["scan"])
        self.assertIn("PC MODEL SCANNER", out)
        self.assertIn("ollama:gemma3:4b", out)
        self.assertIn("LIVE (Ollama)", out)
        self.assertIn("Direct PyTorch", out)

    def test_cli_transmute_command_early_exit_and_native_distill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _build_tiny_hf_llama_dir(tmpdir)
            out_ee = self._run_cli(
                [
                    "transmute",
                    "--model",
                    tmpdir,
                    "--mode",
                    "early-exit",
                    "--device",
                    "cpu",
                    "--prompt",
                    "hello lucid",
                    "--max-tokens",
                    "3",
                ]
            )
            self.assertIn("WEIGHT TRANSMUTER", out_ee)
            self.assertIn("Yes (real weights intact)", out_ee)

            out_nat = self._run_cli(
                [
                    "transmute",
                    "--model",
                    tmpdir,
                    "--mode",
                    "native-distill",
                    "--experts",
                    "4",
                    "--device",
                    "cpu",
                ]
            )
            self.assertIn("Transferred Tensors", out_nat)

    def test_chat_slash_commands_and_ollama_streaming(self) -> None:
        ndjson_body = (
            json.dumps({"message": {"role": "assistant", "content": "Hello "}, "done": False}).encode()
            + b"\n"
            + json.dumps(
                {
                    "message": {"role": "assistant", "content": "world!"},
                    "done": True,
                    "eval_count": 3,
                    "eval_duration": 50_000_000,
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

        inputs = iter(["/help", "/route cortex", "/temp 0.4", "/tokens 128", "hello there", "/clear", "/exit"])
        out_buf = io.StringIO()
        with (
            patch("lucid_machines.runtime.model_scanner.query_ollama_running_tags", return_value={"gemma3:4b"}),
            patch("urllib.request.urlopen", return_value=_FakeResp(ndjson_body)),
            patch("builtins.input", side_effect=lambda *_: next(inputs)),
            redirect_stdout(out_buf),
        ):
            chat_main(["--model", "ollama:gemma3:4b"])

        out = out_buf.getvalue()
        self.assertIn("CHAT", out.upper())
        self.assertIn("Interactive Chat Commands", out)
        self.assertIn("Route set to CORTEX", out)
        self.assertIn("Temperature updated to 0.40", out)
        self.assertIn("Hello world!", out)
        self.assertIn("Conversation history cleared", out)


if __name__ == "__main__":
    unittest.main()
