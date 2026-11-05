"""Fetch a deterministic FineWeb-Edu (sample-10BT) shard for the paper experiments.

Streams the first ``--skip + --num-docs`` documents of the public split, drops the first
``--skip`` (so the shard is disjoint from ``data/fineweb_edu_real.jsonl``, which was drawn
from the head of the stream), and writes ``{"text": ...}`` JSON lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-docs", type=int, default=8000)
    p.add_argument("--skip", type=int, default=1000)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "fineweb_edu_large.jsonl")
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i < args.skip:
                continue
            text = row.get("text", "")
            if text.strip():
                f.write(json.dumps({"text": text, "id": row.get("id")}, ensure_ascii=False) + "\n")
                n += 1
            if n >= args.num_docs:
                break
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {n} docs -> {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
