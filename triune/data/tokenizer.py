"""Tokenizer loading and optional tokenizer training helpers."""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def bytes_to_unicode() -> Dict[int, str]:
    """Returns mapping between utf-8 byte values (0..255) and unicode characters for byte-level BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))


class PurePythonBPETokenizer:
    """Pure-Python high-throughput ByteLevel BPE tokenizer matching OpenAI / HuggingFace format.
    Ensures zero decoding artifacts and works across any operating system without C-extension dependencies.
    """

    def __init__(self, json_path: Union[str, Path]):
        with open(str(json_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model", {})
        self.vocab: Dict[str, int] = model.get("vocab", {})
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        raw_merges = model.get("merges", [])
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for rank, item in enumerate(raw_merges):
            if isinstance(item, list) and len(item) == 2:
                self.bpe_ranks[tuple(item)] = rank
            elif isinstance(item, str):
                parts = tuple(item.split(" ", 1))
                if len(parts) == 2:
                    self.bpe_ranks[parts] = rank

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")

    def _get_pairs(self, word: Tuple[str, ...]):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @functools.lru_cache(maxsize=32768)
    def bpe(self, token: str) -> str:
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        return " ".join(word)

    def encode(self, text: str):
        bpe_tokens: List[int] = []
        for match in re.finditer(self.pat, text):
            token = match.group(0)
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            for bpe_token in self.bpe(token).split(" "):
                tid = self.vocab.get(bpe_token, self.vocab.get("[UNK]", 1))
                bpe_tokens.append(tid)

        class EncResult:
            def __init__(self, ids: List[int]):
                self.ids = ids

        return EncResult(bpe_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        byte_arr = bytearray()
        for tid in ids:
            tok = self.inv_vocab.get(tid)
            if tok is None:
                continue
            if skip_special_tokens and (
                tok in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|endoftext|>"]
                or (tok.startswith("[") and tok.endswith("]"))
            ):
                continue
            for ch in tok:
                if ch in self.byte_decoder:
                    byte_arr.append(self.byte_decoder[ch])
                else:
                    byte_arr.extend(ch.encode("utf-8"))
        return byte_arr.decode("utf-8", errors="replace")

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.vocab.get(token)

    def id_to_token(self, tid: int) -> Optional[str]:
        return self.inv_vocab.get(tid)


def load_tokenizer(path: Union[str, Path]) -> Any:
    """Load BPE tokenizer from json file, preferring HuggingFace Tokenizers C-lib or PurePython fallback."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    # Try fast C-extension Tokenizer if installed
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        try:
            tok = Tokenizer.from_file(str(p))
            if tok.decoder is None:
                tok.decoder = ByteLevelDecoder()
            return tok
        except Exception:
            with open(str(p), "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("decoder") and data["decoder"].get("type") == "ByteLevel":
                data["decoder"] = {
                    "type": "ByteLevel",
                    "add_prefix_space": True,
                    "trim_offsets": True,
                    "use_regex": True,
                }
            tok = Tokenizer.from_str(json.dumps(data))
            if tok.decoder is None:
                tok.decoder = ByteLevelDecoder()
            return tok
    except (ImportError, Exception):
        # High-performance Pure Python Byte-Level fallback
        return PurePythonBPETokenizer(p)


def build_tokenizer(
    output_path: Union[str, Path],
    *,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    target_chars: int = 5_000_000_000,
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = "wikitext-103-raw-v1",
) -> Any:
    """Train and save a BPE tokenizer directly from a dataset stream."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import ByteLevel as ByteLevelProcessor
        from tokenizers.trainers import BpeTrainer
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(f"Tokenizer training requires 'tokenizers' and 'datasets' packages: {e}")

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
