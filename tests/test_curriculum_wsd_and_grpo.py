from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lucid_machines.data.dataset import (
    WeightedMultiStreamDataset,
    _extract_sample_text,
    parse_dataset_mix,
)
from lucid_machines.model.transformer import TriuneTransformer
from lucid_machines.trainer.grpo import (
    GRPOHarnessTrainer,
    VerifiableTask,
    default_verifiable_task_bank,
    verify_rollout,
)
from lucid_machines.trainer.scheduler import lr_schedule
from scripts.train_grpo_harness import main as grpo_main


class _WordTokenizer:
    def __init__(self, vocab_size: int = 256) -> None:
        self._vocab = vocab_size

    def get_vocab_size(self) -> int:
        return self._vocab

    def token_to_id(self, tok: str) -> int | None:
        return {"[PAD]": 0, "[SEP]": 1}.get(tok)

    def id_to_token(self, tid: int) -> str:
        return {0: "[PAD]", 1: "[SEP]"}.get(tid, f"t{tid}")

    def encode(self, text: str):
        class _Enc:
            def __init__(self, ids):
                self.ids = ids

        ids = [2 + (b % (self._vocab - 4)) for b in text.encode("utf-8", errors="ignore")]
        return _Enc(ids)

    def decode(self, ids: list[int]) -> str:
        return bytes(32 + (i % 90) for i in ids).decode("ascii", errors="ignore")


def test_wsd_lr_schedule_phases() -> None:
    cfg = {
        "lr": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 10,
        "total_steps": 100,
        "lr_schedule": "wsd",
        "wsd_decay_frac": 0.20,  # last 18 steps of 90 post-warmup decay (start=82)
    }
    # Linear warmup
    assert lr_schedule(cfg, 0) == pytest.approx(1e-4)
    assert lr_schedule(cfg, 9) == pytest.approx(1e-3)
    # Constant stable plateau across steps 10..81
    assert lr_schedule(cfg, 10) == pytest.approx(1e-3)
    assert lr_schedule(cfg, 50) == pytest.approx(1e-3)
    assert lr_schedule(cfg, 81) == pytest.approx(1e-3)
    # Cooldown decay across steps 82..100
    mid = lr_schedule(cfg, 91)
    assert 1e-4 < mid < 1e-3
    assert lr_schedule(cfg, 100) == pytest.approx(1e-4)


def test_parse_dataset_mix_and_extract_chat_text() -> None:
    preset = parse_dataset_mix("junkie_stage1")
    assert len(preset) == 4
    assert sum(w for _, _, w in preset) == pytest.approx(1.0)

    custom = parse_dataset_mix("repo/a:cfg1:0.7,repo/b::0.3")
    assert custom == [("repo/a", "cfg1", 0.7), ("repo/b", None, 0.3)]

    chat_txt = _extract_sample_text({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]})
    assert "<user>" in chat_txt and "<assistant>" in chat_txt and "ok" in chat_txt


def test_weighted_multi_stream_resilient_fallback(monkeypatch) -> None:
    def _fake_stream(self):
        if self.dataset_name == "bad/repo":
            raise RuntimeError("offline stream")
        text = f"domain={self.dataset_name} " * 40
        for _ in range(12):
            yield {"text": text}

    monkeypatch.setattr("lucid_machines.data.dataset.TokenStreamDataset._stream", _fake_stream)
    tok = _WordTokenizer()
    ds = WeightedMultiStreamDataset(
        tok,
        streams=[("good/edu", "v1", 0.7), ("bad/repo", None, 0.2), ("good/math", None, 0.1)],
        seq_len=16,
        sep_token_id=1,
        shuffle_buffer=8,
        seed=7,
    )
    items = [batch for _, batch in zip(range(6), ds)]
    assert len(items) == 6
    assert all(x.shape == (16,) and y.shape == (16,) for x, y in items)


def test_verify_rollout_math_python_sandbox_and_tool_harness() -> None:
    bank = {t.task_id: t for t in default_verifiable_task_bank()}

    r_math_ok, meta = verify_rollout(bank["math_arith_1"], "Therefore \\boxed{350}")
    assert r_math_ok == pytest.approx(1.0) and meta["correct"] is True
    r_math_fmt, _ = verify_rollout(bank["math_arith_1"], "Therefore \\boxed{999}")
    assert r_math_fmt == pytest.approx(0.2)

    py_ok = "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"
    r_py_ok, py_meta = verify_rollout(bank["py_fib_1"], py_ok)
    assert r_py_ok == pytest.approx(1.0) and py_meta["correct"] is True

    py_bad = "```python\ndef fib(n):\n    return 0\n```"
    r_py_bad, _ = verify_rollout(bank["py_fib_1"], py_bad)
    assert r_py_bad == pytest.approx(0.2)

    tool_ok = '<tool_call>{"name": "search_code", "arguments": {"pattern": "TriuneTransformer", "directory": "lucid_machines"}}</tool_call>'
    r_tool_ok, t_meta = verify_rollout(bank["tool_search_1"], tool_ok)
    assert r_tool_ok == pytest.approx(1.0) and t_meta["correct"] is True


def test_grpo_harness_trainer_step_and_dry_run(tmp_path: Path) -> None:
    torch.manual_seed(0)
    tok = _WordTokenizer(vocab_size=128)
    model = TriuneTransformer(
        vocab_size=128,
        hidden_dim=64,
        num_layers=4,
        num_heads=2,
        head_dim=32,
        num_experts=2,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=3,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    task = VerifiableTask(
        task_id="t_any",
        category="math",
        prompt="<user>\n1+1?\n</user>\n<assistant>\n",
        expected="2",
    )
    trainer = GRPOHarnessTrainer(
        model=model,
        optimizer=opt,
        tokenizer=tok,
        tasks=[task],
        device=torch.device("cpu"),
        output_dir=tmp_path,
        group_size=2,
        max_new_tokens=6,
        rft_every=0,
        replay_threshold=0.0,  # force recording verified trace to test JSONL path
    )
    metrics = trainer.train_step(task)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert trainer.trace_path.is_file()
    lines = [json.loads(line) for line in trainer.trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    dry = grpo_main(["--model_config", "junkie-mini", "--dry_run"])
    assert dry["dry_run"] is True and dry["tasks"] >= 6
