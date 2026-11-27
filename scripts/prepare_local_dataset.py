"""Download and pre-tokenize multi-domain curriculum datasets into local uint16 memmap shards."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from datasets import DownloadConfig, load_dataset
from tokenizers import Tokenizer

from lucid_machines.data.dataset import _extract_sample_text, parse_dataset_mix

DEFAULT_PRESET_TOKENS: dict[str, int] = {
    "junkie_stage1": 500_000_000,
    "junkie_stage2": 100_000_000,
    "junkie_tool_sft": 50_000_000,
}


def _resolve_sep_token_id(tokenizer) -> int:
    for cand in ("[SEP]", "<|endoftext|>", "</s>", "<sep>"):
        try:
            tid = tokenizer.token_to_id(cand)
        except Exception:
            tid = None
        if tid is not None:
            return int(tid)
    return 0


def _encode_texts(tokenizer, texts: list[str]) -> list[list[int]]:
    if hasattr(tokenizer, "encode_batch"):
        return [list(enc.ids) for enc in tokenizer.encode_batch(texts)]
    return [list(tokenizer.encode(t).ids) for t in texts]


def _default_stream_factory(repo: str, cfg: str | None) -> Iterator[dict]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    ds = load_dataset(
        repo,
        name=cfg,
        split="train",
        streaming=True,
        token=token,
        download_config=DownloadConfig(max_retries=10, resume_download=True),
    )
    return iter(ds)


def prepare_dataset(
    preset: str,
    target_tokens: int,
    output_dir: str | Path = "/content/data",
    *,
    tokenizer_path: str | Path = "triune_tokenizer.json",
    cache_dir: str | Path | None = None,
    batch_docs: int = 512,
    force: bool = False,
    seed: int = 42,
    stream_factory: Callable[[str, str | None], Iterator[dict]] | None = None,
    tokenizer=None,
) -> dict:
    """Stream, proportionally sample, pre-tokenize, and write a uint16 memmap `.bin` shard."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = preset if preset.isidentifier() else "custom_mix"
    bin_path = out_dir / f"{safe_name}.bin"
    meta_path = out_dir / f"{safe_name}.meta.json"

    if not force and bin_path.is_file():
        existing_tokens = bin_path.stat().st_size // 2
        if existing_tokens >= target_tokens:
            print(f"✅ Already prepared {bin_path} ({existing_tokens:,} tokens >= {target_tokens:,})", flush=True)
            return {"preset": preset, "bin_path": str(bin_path), "tokens": existing_tokens, "cached": True}

    if cache_dir is not None:
        cache_bin = Path(cache_dir) / f"{safe_name}.bin"
        cache_meta = Path(cache_dir) / f"{safe_name}.meta.json"
        if not force and cache_bin.is_file() and (cache_bin.stat().st_size // 2) >= target_tokens:
            shutil.copy2(cache_bin, bin_path)
            if cache_meta.is_file():
                shutil.copy2(cache_meta, meta_path)
            restored_tokens = bin_path.stat().st_size // 2
            print(f"✅ Restored cached shard {cache_bin} -> {bin_path} ({restored_tokens:,} tokens)", flush=True)
            return {"preset": preset, "bin_path": str(bin_path), "tokens": restored_tokens, "cached": True}

    tok = tokenizer if tokenizer is not None else Tokenizer.from_file(str(tokenizer_path))
    vocab_size = int(tok.get_vocab_size())
    if vocab_size > 65535:
        raise ValueError(f"Tokenizer vocab_size={vocab_size} exceeds uint16 max (65535)")
    sep_token_id = _resolve_sep_token_id(tok)

    stream_specs = parse_dataset_mix(preset)
    factory = stream_factory or _default_stream_factory
    active: list[tuple[str, str | None, float, Iterator[dict]]] = []
    for repo, cfg, weight in stream_specs:
        try:
            it = factory(repo, cfg)
            first_doc = next(it)

            def _chain(first, rest):
                yield first
                yield from rest

            active.append((repo, cfg, float(weight), _chain(first_doc, it)))
        except Exception as exc:
            print(f"⚠️ Skipping unavailable stream {repo} ({cfg}): {exc}", flush=True)

    if not active:
        raise RuntimeError(f"All streams for preset {preset!r} failed to initialize")

    rng = random.Random(seed)
    tmp_path = out_dir / f"{safe_name}.bin.tmp"
    flush_threshold = 1_000_000
    log_interval = 10_000_000
    next_log = log_interval
    total_written = 0
    chunk_buf: list[int] = []
    t_start = time.perf_counter()

    try:
        with open(tmp_path, "wb") as fout:
            while total_written < target_tokens and active:
                weights = [entry[2] for entry in active]
                idx = rng.choices(range(len(active)), weights=weights, k=1)[0]
                repo, cfg, w, it = active[idx]
                docs_batch: list[str] = []
                exhausted = False
                failed = False
                for _ in range(batch_docs):
                    try:
                        sample = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    except Exception as exc:
                        print(f"⚠️ Stream {repo} ({cfg}) error ({exc}); dropping stream", flush=True)
                        failed = True
                        break
                    txt = _extract_sample_text(sample)
                    if txt.strip():
                        docs_batch.append(txt)

                if docs_batch:
                    encoded = _encode_texts(tok, docs_batch)
                    for ids in encoded:
                        if not ids:
                            continue
                        chunk_buf.extend(ids)
                        chunk_buf.append(sep_token_id)

                if exhausted:
                    try:
                        active[idx] = (repo, cfg, w, factory(repo, cfg))
                    except Exception:
                        active.pop(idx)
                elif failed:
                    active.pop(idx)

                if len(chunk_buf) >= flush_threshold or (total_written + len(chunk_buf) >= target_tokens) or not active:
                    remaining = max(0, target_tokens - total_written)
                    to_write = chunk_buf[:remaining]
                    if to_write:
                        arr = np.asarray(to_write, dtype=np.uint16)
                        fout.write(arr.tobytes())
                        total_written += len(to_write)
                    chunk_buf.clear()
                    if total_written >= next_log or total_written >= target_tokens:
                        elapsed = max(1e-6, time.perf_counter() - t_start)
                        rate = total_written / elapsed
                        eta_s = max(0.0, (target_tokens - total_written) / max(1.0, rate))
                        print(
                            f"[{safe_name}] {total_written / 1e6:.1f}M / {target_tokens / 1e6:.1f}M tokens "
                            f"({rate / 1e3:.1f}k tok/s | ETA {eta_s / 60:.1f}m)",
                            flush=True,
                        )
                        next_log += log_interval

        if total_written == 0:
            raise RuntimeError(f"Zero tokens written for preset {preset!r}")

        tmp_path.replace(bin_path)
        meta = {
            "preset": preset,
            "tokens": total_written,
            "dtype": "uint16",
            "vocab_size": vocab_size,
            "sep_token_id": sep_token_id,
            "streams": [[r, c, float(wt)] for r, c, wt in stream_specs],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if cache_dir is not None:
            cache_out = Path(cache_dir)
            cache_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bin_path, cache_out / bin_path.name)
            shutil.copy2(meta_path, cache_out / meta_path.name)
        print(f"✅ Wrote {bin_path} ({total_written:,} tokens, {bin_path.stat().st_size / (1024**2):.1f} MiB)", flush=True)
        return {"preset": preset, "bin_path": str(bin_path), "tokens": total_written, "cached": False}
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-download and tokenize Junkie curriculum shards into uint16 .bin")
    parser.add_argument("--preset", type=str, default="junkie_stage1", help="Preset name, 'all', or repo:cfg:weight list")
    parser.add_argument("--target_tokens", type=int, default=None, help="Tokens to collect for single preset")
    parser.add_argument("--stage1_tokens", type=int, default=DEFAULT_PRESET_TOKENS["junkie_stage1"])
    parser.add_argument("--stage2_tokens", type=int, default=DEFAULT_PRESET_TOKENS["junkie_stage2"])
    parser.add_argument("--sft_tokens", type=int, default=DEFAULT_PRESET_TOKENS["junkie_tool_sft"])
    parser.add_argument("--output_dir", type=Path, default=Path("/content/data"))
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--tokenizer_path", type=Path, default=Path("triune_tokenizer.json"))
    parser.add_argument("--batch_docs", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_parser().parse_args(argv)
    if args.preset == "all":
        targets = [
            ("junkie_stage1", int(args.stage1_tokens)),
            ("junkie_stage2", int(args.stage2_tokens)),
            ("junkie_tool_sft", int(args.sft_tokens)),
        ]
    else:
        default_n = DEFAULT_PRESET_TOKENS.get(args.preset, 100_000_000)
        targets = [(args.preset, int(args.target_tokens or default_n))]

    results = []
    for preset_name, n_tok in targets:
        res = prepare_dataset(
            preset_name,
            n_tok,
            output_dir=args.output_dir,
            tokenizer_path=args.tokenizer_path,
            cache_dir=args.cache_dir,
            batch_docs=args.batch_docs,
            force=args.force,
        )
        results.append(res)
    return results


if __name__ == "__main__":
    main()
