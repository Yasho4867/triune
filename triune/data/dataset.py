"""Streaming token datasets with no module-level runtime state."""

from __future__ import annotations

import random
from collections.abc import Iterator

import torch
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader, IterableDataset


class TokenStreamDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        *,
        seq_len: int,
        sep_token_id: int,
        max_tokens: int | None = None,
        shuffle_buffer: int = 4096,
        offset: int = 0,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str | None = "sample-10BT",
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sep_token_id = sep_token_id
        self.max_tokens = max_tokens
        self.shuffle_buffer = shuffle_buffer
        self.offset = offset
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

    def _stream(self):
        import os
        if os.path.exists(self.dataset_name):
            ext = os.path.splitext(self.dataset_name)[1].lower()
            if ext in (".jsonl", ".json"):
                return load_dataset("json", data_files=self.dataset_name, split="train", streaming=True)
            elif ext == ".parquet":
                return load_dataset("parquet", data_files=self.dataset_name, split="train", streaming=True)
            elif ext in (".txt", ".md", ".py", ".c", ".cpp"):
                return load_dataset("text", data_files=self.dataset_name, split="train", streaming=True)
        return load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split="train",
            streaming=True,
            download_config=DownloadConfig(max_retries=10, resume_download=True),
        )

    @staticmethod
    def _flush(buffer: list[tuple[torch.Tensor, torch.Tensor]]) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        random.shuffle(buffer)
        yield from buffer
        buffer.clear()

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        import time
        max_retries = 3
        worker_info = torch.utils.data.get_worker_info()

        for attempt in range(max_retries):
            token_buffer: list[int] = []
            batch_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
            token_count = 0
            sample_idx = 0
            tokens_to_skip = self.offset // worker_info.num_workers if worker_info is not None else self.offset

            try:
                stream = self._stream()
                for sample in stream:
                    if isinstance(sample, str):
                        text = sample
                    else:
                        text = sample.get("text") or sample.get("content") or sample.get("story") or sample.get("prompt") or ""
                    if not text.strip():
                        continue

                    # Simple sharding: let workers skip samples round-robin
                    if worker_info is not None:
                        if sample_idx % worker_info.num_workers != worker_info.id:
                            sample_idx += 1
                            continue
                    sample_idx += 1

                    ids = self.tokenizer.encode(text).ids
                    if tokens_to_skip:
                        if tokens_to_skip >= len(ids):
                            tokens_to_skip -= len(ids)
                            continue
                        ids = ids[tokens_to_skip:]
                        tokens_to_skip = 0

                    token_buffer.extend(ids)
                    token_buffer.append(self.sep_token_id)

                    while len(token_buffer) >= self.seq_len + 1:
                        chunk = token_buffer[: self.seq_len + 1]
                        del token_buffer[: self.seq_len + 1]
                        batch_buffer.append(
                            (torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long))
                        )
                        token_count += self.seq_len

                        if len(batch_buffer) >= self.shuffle_buffer:
                            yield from self._flush(batch_buffer)
                        if self.max_tokens is not None and token_count >= self.max_tokens:
                            yield from self._flush(batch_buffer)
                            return

                yield from self._flush(batch_buffer)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


def build_dataloader(tokenizer, config: dict, sep_token_id: int, *, is_holdout: bool) -> DataLoader:
    eval_tokens = config["eval_batches"] * config["batch_size"] * config["seq_len"]
    dataset = TokenStreamDataset(
        tokenizer,
        seq_len=config["seq_len"],
        sep_token_id=sep_token_id,
        max_tokens=eval_tokens if is_holdout else None,
        offset=0 if is_holdout else eval_tokens,
        shuffle_buffer=config["shuffle_buffer"],
        dataset_name=config["dataset_name"],
        dataset_config=config["dataset_config"],
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        drop_last=True,
    )
