"""Tests for LocalMemmapTokenDataset, prepare_local_dataset, and Colab Cell 2B/Cell 3 configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lucid_machines.data.dataset import LocalMemmapTokenDataset, build_dataloader
from scripts.prepare_local_dataset import prepare_dataset
from scripts.train import build_parser


class _FakeEncoded:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _WordTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"[PAD]": 0, "[SEP]": 1}

    def get_vocab_size(self) -> int:
        return 32000

    def token_to_id(self, tok: str) -> int | None:
        return self._vocab.get(tok)

    def encode(self, text: str) -> _FakeEncoded:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 10
            ids.append(self._vocab[word])
        return _FakeEncoded(ids)

    def encode_batch(self, texts: list[str]) -> list[_FakeEncoded]:
        return [self.encode(t) for t in texts]


def test_local_memmap_split_shift_and_epoch_wrap(tmp_path: Path) -> None:
    bin_path = tmp_path / "tokens.bin"
    arr = np.arange(500, dtype=np.uint16) + 10
    bin_path.write_bytes(arr.tobytes())

    eval_tokens = 64
    seq_len = 16
    holdout_ds = LocalMemmapTokenDataset(bin_path, seq_len=seq_len, is_holdout=True, eval_tokens=eval_tokens)
    holdout_batches = list(holdout_ds)
    assert len(holdout_batches) == (eval_tokens - 1) // seq_len
    first_x, first_y = holdout_batches[0]
    assert first_x.shape == (seq_len,)
    assert torch.equal(first_y[:-1], first_x[1:])
    assert int(first_x[0].item()) == 10

    train_ds = LocalMemmapTokenDataset(bin_path, seq_len=seq_len, is_holdout=False, eval_tokens=eval_tokens, seed=7)
    assert train_ds.slice_start == eval_tokens
    it = iter(train_ds)
    seen_starts = set()
    # Consume more windows than one epoch to confirm wrap-around and zero holdout overlap
    for _ in range(train_ds.num_windows + 5):
        x, y = next(it)
        assert int(x.min().item()) >= 10 + eval_tokens
        assert torch.equal(y[:-1], x[1:])
        seen_starts.add(int(x[0].item()))
    assert len(seen_starts) == train_ds.num_windows


def test_build_dataloader_dispatches_to_local_memmap(tmp_path: Path) -> None:
    bin_path = tmp_path / "stage1.bin"
    arr = (np.arange(1024, dtype=np.uint16) % 5000) + 1
    bin_path.write_bytes(arr.tobytes())

    cfg = {
        "eval_batches": 2,
        "batch_size": 2,
        "seq_len": 32,
        "num_workers": 0,
        "local_tokens_bin": str(bin_path),
        "dataset_mix": "junkie_stage1",
        "dataset_name": "unused",
        "dataset_config": None,
        "shuffle_buffer": 64,
    }
    eval_loader = build_dataloader(None, cfg, sep_token_id=1, is_holdout=True)
    train_loader = build_dataloader(None, cfg, sep_token_id=1, is_holdout=False)
    ex, ey = next(iter(eval_loader))
    tx, ty = next(iter(train_loader))
    assert ex.shape == (2, 32)
    assert tx.shape == (2, 32)
    assert int(ex[0, 0].item()) == 1
    assert int(tx.min().item()) >= 1 + 2 * 2 * 32

    # Also auto-detects when dataset_name directly points to a .bin file
    cfg_alt = dict(cfg)
    cfg_alt["local_tokens_bin"] = None
    cfg_alt["dataset_name"] = str(bin_path)
    alt_loader = build_dataloader(None, cfg_alt, sep_token_id=1, is_holdout=False)
    ax, _ = next(iter(alt_loader))
    assert ax.shape == (2, 32)


def test_prepare_dataset_mock_streams_cache_and_fallback(tmp_path: Path) -> None:
    out_dir = tmp_path / "data"
    cache_dir = tmp_path / "drive_cache"
    tok = _WordTokenizer()

    def _factory(repo: str, cfg: str | None):
        if cfg == "python-edu":
            raise RuntimeError("simulated gated stream")
        docs = [{"text": f"{repo} sample document {i} with tokens"} for i in range(40)]
        return iter(docs)

    res = prepare_dataset(
        "junkie_stage1",
        target_tokens=600,
        output_dir=out_dir,
        cache_dir=cache_dir,
        batch_docs=8,
        stream_factory=_factory,
        tokenizer=tok,
    )
    bin_file = Path(res["bin_path"])
    assert bin_file.is_file()
    assert res["tokens"] == 600
    assert res["cached"] is False
    assert bin_file.stat().st_size == 1200  # uint16
    meta = json.loads((out_dir / "junkie_stage1.meta.json").read_text(encoding="utf-8"))
    assert meta["tokens"] == 600
    assert meta["dtype"] == "uint16"
    assert (cache_dir / "junkie_stage1.bin").is_file()

    # Re-running skips work when >= target_tokens
    res_skip = prepare_dataset(
        "junkie_stage1",
        target_tokens=600,
        output_dir=out_dir,
        stream_factory=lambda *_: (_ for _ in ()).throw(AssertionError("should not reopen stream")),
        tokenizer=tok,
    )
    assert res_skip["cached"] is True

    # Restores from cache_dir if local bin is deleted
    bin_file.unlink()
    res_restored = prepare_dataset(
        "junkie_stage1",
        target_tokens=600,
        output_dir=out_dir,
        cache_dir=cache_dir,
        stream_factory=lambda *_: (_ for _ in ()).throw(AssertionError("should restore from cache")),
        tokenizer=tok,
    )
    assert res_restored["cached"] is True
    assert bin_file.is_file()


def test_train_cli_local_tokens_bin_flag_and_colab_cell2b_cell3(tmp_path: Path) -> None:
    parsed = build_parser().parse_args(["--local_tokens_bin", "/content/data/junkie_stage1.bin"])
    assert parsed.local_tokens_bin == "/content/data/junkie_stage1.bin"

    nb_path = Path(__file__).resolve().parents[1] / "cloud" / "lucid_machines_colab_runner.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 11
    cell2b_src = "".join(nb["cells"][2]["source"])
    cell3_src = "".join(nb["cells"][3]["source"])
    assert cell2b_src.lstrip().startswith("# Cell 2B:")
    assert cell3_src.lstrip().startswith("# Cell 3:")

    snippet = []
    for line in cell3_src.splitlines():
        if line.startswith("if vram_gb >= 70.0:") or snippet:
            if line.startswith("fp8_flag ="):
                break
            snippet.append(line)

    # Check L4 tier (22.5 GiB): 600,000 steps, batch=4, rank=32, fp8 preserved from supports_fp8
    ns_l4 = {
        "vram_gb": 22.5,
        "ckpt_3b": tmp_path / "none",
        "torch": torch,
        "os": os,
        "supports_fp8": True,
        "depth_mode_val": "cortex",
    }
    exec("\n".join(snippet), ns_l4)
    assert ns_l4["steps_val"] == 600000
    assert ns_l4["batch_val"] == 4
    assert ns_l4["rank_val"] == 32

    # Check A100-80GB tier
    ns_a80 = {
        "vram_gb": 79.6,
        "ckpt_3b": tmp_path / "none",
        "torch": torch,
        "os": os,
        "supports_fp8": False,
        "depth_mode_val": "cortex",
    }
    exec("\n".join(snippet), ns_a80)
    assert ns_a80["steps_val"] == 300000
    assert ns_a80["rank_val"] == 64


def test_sharded_memmap_live_auto_discovery_and_bootstrap_resume(tmp_path: Path) -> None:
    shard_dir = tmp_path / "data" / "junkie_stage1"
    shard_dir.mkdir(parents=True)
    seq_len = 16
    eval_tokens = 64
    arr0 = np.arange(320, dtype=np.uint16) + 10
    (shard_dir / "shard_0000.bin").write_bytes(arr0.tobytes())

    ds = LocalMemmapTokenDataset(shard_dir, seq_len=seq_len, is_holdout=False, eval_tokens=eval_tokens, seed=3)
    it = iter(ds)
    # Consume 5 windows from shard_0000, then drop in shard_0001 with distinct token values (>= 10000)
    for _ in range(5):
        x, _ = next(it)
        assert int(x.min().item()) >= 10 + eval_tokens
        assert int(x.max().item()) < 500
    arr1 = np.arange(320, dtype=np.uint16) + 10000
    (shard_dir / "shard_0001.bin").write_bytes(arr1.tobytes())

    saw_shard1 = False
    for _ in range(ds.num_windows + 25):
        x, _ = next(it)
        if int(x.min().item()) >= 10000:
            saw_shard1 = True
            break
    assert saw_shard1, "LocalMemmapTokenDataset must auto-discover newly completed shard_0001.bin live"

    # Also test prepare_dataset sharded bootstrap (1 shard) -> resume remaining shards
    tok = _WordTokenizer()

    def _factory(repo: str, cfg: str | None):
        return iter([{"text": f"{repo} token stream document {i}"} for i in range(60)])

    out_dir = tmp_path / "sharded_prep"
    res1 = prepare_dataset(
        "junkie_stage1",
        target_tokens=700,
        output_dir=out_dir,
        batch_docs=4,
        sharded=True,
        shard_tokens=300,
        max_new_shards=1,
        stream_factory=_factory,
        tokenizer=tok,
    )
    assert res1["completed_shards"] == 1
    assert res1["total_shards"] == 3
    assert (out_dir / "junkie_stage1" / "shard_0000.bin").is_file()
    assert not (out_dir / "junkie_stage1" / "shard_0001.bin").exists()

    res2 = prepare_dataset(
        "junkie_stage1",
        target_tokens=700,
        output_dir=out_dir,
        batch_docs=4,
        sharded=True,
        shard_tokens=300,
        max_new_shards=None,
        stream_factory=_factory,
        tokenizer=tok,
    )
    assert res2["completed_shards"] == 3
    assert res2["tokens"] == 700
    assert (out_dir / "junkie_stage1" / "shard_0002.bin").stat().st_size == 200  # 100 uint16 tokens

