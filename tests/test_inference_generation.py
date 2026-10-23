"""Regression tests for the shared CLI and Studio autoregressive decoder."""

from __future__ import annotations

import unittest

import torch

from triune.inference import generate_response


class _Tokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, text: str):
        self.prompts.append(text)
        return type("Encoding", (), {"ids": [1, 2]})()

    @staticmethod
    def get_vocab_size() -> int:
        return 8

    @staticmethod
    def decode(ids, skip_special_tokens=True) -> str:
        return " ".join(str(token) for token in ids)


class _Model:
    def __init__(self) -> None:
        self.last_depth_choice = None
        self.force_depths: list[int | None] = []
        self.calls = 0

    @staticmethod
    def init_cache(batch_size: int):
        return {"step": 0}

    def __call__(self, ids, *, cache=None, force_depth=None):
        self.force_depths.append(force_depth)
        self.last_depth_choice = torch.tensor([1])
        logits = torch.full((1, ids.shape[1], 8), -20.0)
        # Emit one normal token, then EOS.
        logits[0, -1, 3 if self.calls == 0 else 7] = 20.0
        self.calls += 1
        if cache is not None:
            return logits, cache
        return logits, None


class TestSharedGeneration(unittest.TestCase):
    def test_locks_adaptive_route_after_prefill(self) -> None:
        model = _Model()
        tokenizer = _Tokenizer()
        result = generate_response(
            model,
            tokenizer,
            "hello",
            device="cpu",
            pad_token_id=0,
            eos_token_id=7,
            temperature=0,
            max_new_tokens=4,
            return_info=True,
        )
        self.assertEqual(result["text"], "3")
        self.assertEqual(result["route"], "LIMBIC")
        self.assertEqual(model.force_depths, [None, 1])

    def test_does_not_reformat_api_conversation_prompt(self) -> None:
        model = _Model()
        tokenizer = _Tokenizer()
        generate_response(
            model,
            tokenizer,
            "System: concise\nUser: hello\nAssistant:",
            device="cpu",
            pad_token_id=0,
            eos_token_id=7,
            temperature=0,
            max_new_tokens=1,
            prompt_is_formatted=True,
        )
        self.assertEqual(tokenizer.prompts, ["System: concise\nUser: hello\nAssistant:"])

