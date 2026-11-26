"""Streaming token datasets with no module-level runtime state."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

import torch
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader, IterableDataset

CURRICULUM_MIX_PRESETS: dict[str, list[tuple[str, str | None, float]]] = {
    "junkie_stage1": [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.45),
        ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 0.25),
        ("HuggingFaceTB/finemath", "finemath-4plus", 0.15),
        ("HuggingFaceTB/smollm-corpus", "python-edu", 0.15),
    ],
    "junkie_stage2": [
        ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 0.35),
        ("HuggingFaceTB/finemath", "finemath-4plus", 0.35),
        ("HuggingFaceTB/smollm-corpus", "python-edu", 0.20),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.10),
    ],
    "junkie_tool_sft": [
        ("HuggingFaceTB/smoltalk", "all", 0.50),
        ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 0.25),
        ("HuggingFaceTB/finemath", "finemath-4plus", 0.25),
    ],
}


def parse_dataset_mix(mix_spec: str) -> list[tuple[str, str | None, float]]:
    """Parse a preset name or 'repo:config:weight,repo2:config2:weight2' specification."""
    key = str(mix_spec).strip()
    if key in CURRICULUM_MIX_PRESETS:
        return list(CURRICULUM_MIX_PRESETS[key])
    entries: list[tuple[str, str | None, float]] = []
    for raw_part in key.split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = [f.strip() for f in part.split(":")]
        if len(fields) == 2:
            repo, weight_s = fields
            cfg = None
        elif len(fields) == 3:
            repo, cfg, weight_s = fields
            cfg = cfg or None
        else:
            raise ValueError(
                f"Invalid dataset mix entry {part!r}; expected preset name or 'repo:config:weight'"
            )
        weight = float(weight_s)
        if weight <= 0:
            raise ValueError(f"Dataset mix weight must be positive; got {weight} in {part!r}")
        entries.append((repo, cfg, weight))
    if not entries:
        raise ValueError(f"Dataset mix specification {mix_spec!r} produced no streams")
    return entries


def _extract_sample_text(sample: object) -> str:
    """Extract normalized text from plain web text, code, QA, or multi-turn chat samples."""
    if isinstance(sample, str):
        return sample
    if not isinstance(sample, dict):
        return ""
    for k in ("text", "content", "story"):
        val = sample.get(k)
        if isinstance(val, str) and val.strip():
            return val
    for chat_key in ("messages", "conversations"):
        turns = sample.get(chat_key)
        if isinstance(turns, list) and turns:
            lines: list[str] = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role") or turn.get("from") or "user").strip().lower()
                body = str(turn.get("content") or turn.get("value") or "").strip()
                if body:
                    lines.append(f"<{role}>\n{body}\n</{role}>")
            if lines:
                return "\n".join(lines)
    if sample.get("instruction"):
        inst = str(sample.get("instruction", "")).strip()
        inp = str(sample.get("input", "")).strip()
        out = str(sample.get("output") or sample.get("response") or "").strip()
        parts = [f"<user>\n{inst}" + (f"\n{inp}" if inp else "") + "\n</user>"]
        if out:
            parts.append(f"<assistant>\n{out}\n</assistant>")
        return "\n".join(parts)
    if sample.get("question") and sample.get("answer"):
        return f"<user>\n{str(sample['question']).strip()}\n</user>\n<assistant>\n{str(sample['answer']).strip()}\n</assistant>"
    if sample.get("prompt"):
        p = str(sample.get("prompt", "")).strip()
        c = str(sample.get("completion") or sample.get("response") or "").strip()
        return f"{p}\n{c}".strip() if c else p
    return ""


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
        split: str = "train",
        token: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sep_token_id = sep_token_id
        self.max_tokens = max_tokens
        self.shuffle_buffer = shuffle_buffer
        self.offset = offset
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def _is_word_boundary(self, token_id: int) -> bool:
        """Determine if a token represents the start of a whole word, whitespace, or punctuation."""
        if not hasattr(self, "tokenizer") or not self.tokenizer:
            return True
        try:
            t = self.tokenizer.id_to_token(token_id)
            if not t:
                return True
            if (t.startswith("[") and t.endswith("]")) or (t.startswith("<|") and t.endswith("|>")):
                return True
            if t.startswith("Ġ") or t.startswith("Ċ") or t.startswith("\n") or t.startswith(" "):
                return True
            if not t[0].isalnum():
                return True
            return False
        except Exception:
            return True

    def _stream(self):
        import os

        token_val = self.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
        if os.path.exists(self.dataset_name):
            ext = os.path.splitext(self.dataset_name)[1].lower()
            if ext in (".jsonl", ".json"):
                return load_dataset("json", data_files=self.dataset_name, split=self.split, streaming=True)
            elif ext == ".parquet":
                return load_dataset("parquet", data_files=self.dataset_name, split=self.split, streaming=True)
            elif ext in (".txt", ".md", ".py", ".c", ".cpp"):
                return load_dataset("text", data_files=self.dataset_name, split=self.split, streaming=True)
        return load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=True,
            token=token_val,
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
                    text = _extract_sample_text(sample)
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

                    chunk_len = self.seq_len + 1
                    while len(token_buffer) >= chunk_len:
                        split_idx = chunk_len
                        # Word-boundary alignment: ensure next chunk never starts with a severed subword
                        if split_idx < len(token_buffer) and not self._is_word_boundary(token_buffer[split_idx]):
                            bt = split_idx
                            while bt > split_idx - 16 and bt > 0 and not self._is_word_boundary(token_buffer[bt]):
                                bt -= 1
                            if bt > 0:
                                split_idx = bt

                        chunk = token_buffer[:split_idx]
                        del token_buffer[:split_idx]
                        if len(chunk) < chunk_len:
                            chunk.extend([0] * (chunk_len - len(chunk)))

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
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise


class WeightedMultiStreamDataset(IterableDataset):
    """Proportional multi-domain curriculum sampler across weighted TokenStreamDataset sources."""

    def __init__(
        self,
        tokenizer,
        *,
        streams: list[tuple[str, str | None, float]],
        seq_len: int,
        sep_token_id: int,
        shuffle_buffer: int = 4096,
        offset: int = 0,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.stream_specs = list(streams)
        self.seq_len = seq_len
        self.sep_token_id = sep_token_id
        self.shuffle_buffer = shuffle_buffer
        self.offset = offset
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + worker_id * 1009)

        active: list[tuple[str, float, Iterator[tuple[torch.Tensor, torch.Tensor]], TokenStreamDataset]] = []
        sub_buf = max(256, self.shuffle_buffer // max(1, len(self.stream_specs)))
        for repo, cfg, weight in self.stream_specs:
            ds = TokenStreamDataset(
                self.tokenizer,
                seq_len=self.seq_len,
                sep_token_id=self.sep_token_id,
                shuffle_buffer=sub_buf,
                offset=self.offset if weight >= 0.30 else 0,
                dataset_name=repo,
                dataset_config=cfg,
            )
            try:
                it = iter(ds)
                first_item = next(it)
                # Prepend buffered first item so initialization eagerly validates sub-stream health
                def _chained(first, rest):
                    yield first
                    yield from rest

                active.append((f"{repo}:{cfg}", float(weight), _chained(first_item, it), ds))
            except Exception as exc:
                print(f"⚠️ [CurriculumMix] Skipping unavailable stream {repo} ({cfg}): {exc}", flush=True)

        if not active:
            raise RuntimeError("WeightedMultiStreamDataset: all configured curriculum streams failed to initialize")

        while active:
            weights = [entry[1] for entry in active]
            idx = rng.choices(range(len(active)), weights=weights, k=1)[0]
            label, _, it, _ = active[idx]
            try:
                yield next(it)
            except StopIteration:
                repo, cfg, w = next((r, c, wt) for r, c, wt in self.stream_specs if f"{r}:{c}" == label)
                try:
                    fresh_ds = active[idx][3]
                    active[idx] = (label, w, iter(fresh_ds), fresh_ds)
                except Exception:
                    active.pop(idx)
            except Exception as exc:
                print(f"⚠️ [CurriculumMix] Dropping failing stream {label}: {exc}", flush=True)
                active.pop(idx)


def build_dataloader(tokenizer, config: dict, sep_token_id: int, *, is_holdout: bool) -> DataLoader:
    eval_tokens = config["eval_batches"] * config["batch_size"] * config["seq_len"]
    mix_spec = None if is_holdout else config.get("dataset_mix")
    if mix_spec:
        streams = parse_dataset_mix(str(mix_spec))
        dataset: IterableDataset = WeightedMultiStreamDataset(
            tokenizer,
            streams=streams,
            seq_len=config["seq_len"],
            sep_token_id=sep_token_id,
            shuffle_buffer=config["shuffle_buffer"],
            offset=eval_tokens,
        )
    else:
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
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
