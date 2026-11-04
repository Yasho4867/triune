"""Build a Triune tokenizer from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triune.data import build_tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a Triune BPE tokenizer")
    parser.add_argument("--output", default="triune_tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--target_chars", type=int, default=5_000_000_000)
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    args = parser.parse_args(argv)
    tokenizer = build_tokenizer(
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        target_chars=args.target_chars,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
    )
    print(f"Tokenizer saved to {args.output}; vocabulary size: {tokenizer.get_vocab_size():,}")


if __name__ == "__main__":
    main()
