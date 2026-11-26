"""Profile Lucid Machines training steps from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lucid_machines.cli import resolve_default_checkpoint
from lucid_machines.cli_ui import render_notice
from lucid_machines.profiling import profile_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile Lucid Machines training steps")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path (auto-resolves active/latest checkpoint when omitted)",
    )
    parser.add_argument("--tokenizer_path", default="triune_tokenizer.json")
    parser.add_argument("--output", default="profiler_output.txt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--active_steps", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        ckpt_path = resolve_default_checkpoint(args.checkpoint, prefer="latest")
        print(
            profile_model(
                checkpoint_path=str(ckpt_path),
                tokenizer_path=args.tokenizer_path,
                output_path=args.output,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup_steps=args.warmup_steps,
                active_steps=args.active_steps,
            )
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        render_notice(f"Profiling failed: {exc}", level="error")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
