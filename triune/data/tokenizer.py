"""Tokenizer loading and optional tokenizer training helpers."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def build_tokenizer(
    output_path: str | Path,
    *,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    target_chars: int = 5_000_000_000,
    dataset_name: str = "wikitext",
    dataset_config: str | None = "wikitext-103-raw-v1",
) -> Tokenizer:
    """Train and save a BPE tokenizer directly from a dataset stream."""
    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS)
    seen = 0

    def texts():
        nonlocal seen
        for sample in dataset:
            text = sample.get("text", "")
            if not text.strip():
                continue
            seen += len(text)
            yield text
            if seen >= target_chars:
                return

    tokenizer.train_from_iterator(texts(), trainer=trainer)
    tokenizer.save(str(output_path))
    return tokenizer
