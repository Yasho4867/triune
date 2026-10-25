"""Measure steady-state Triune CLI or Studio inference on one checkpoint.

Run one mode at a time so a second copy of the checkpoint does not compete for
GPU memory. Use identical checkpoint, prompt, depth, temperature, and token
budget for the CLI and Studio runs.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def _depth(value: str) -> int | None:
    return {"auto": None, "reflex": 0, "limbic": 1, "cortex": 2}[value]


def benchmark_cli(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from triune.inference import generate_response
    from triune.inference import load_checkpoint_model

    load_started = time.perf_counter()
    model, tokenizer, checkpoint, device, pad_id, eos_id = load_checkpoint_model(
        args.checkpoint,
        args.tokenizer,
        device=args.device,
    )
    load_seconds = time.perf_counter() - load_started
    result = generate_response(
        model,
        tokenizer,
        args.prompt,
        device=device,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        force_depth=_depth(args.depth),
        temperature=args.temperature,
        top_k=50,
        repetition_penalty=1.2,
        max_new_tokens=args.max_tokens,
        return_info=True,
    )
    return {
        "mode": "cli",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "model_load_seconds": round(load_seconds, 3),
        **result,
        "cuda_allocated_gb": round(torch.cuda.memory_allocated(device) / 1024**3, 3)
        if device.type == "cuda" else 0.0,
    }


def _read_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def benchmark_studio(args: argparse.Namespace) -> dict[str, object]:
    base = args.studio_url.rstrip("/")
    status = _read_json(f"{base}/v1/training/status")
    active = status.get("active_checkpoint")
    expected = Path(args.checkpoint).name
    if not status.get("weights_loaded") or not active or Path(str(active)).name != expected:
        raise RuntimeError(
            f"Studio must have the same checkpoint loaded first. Expected {expected!r}; "
            f"Studio reports {active!r} (weights_loaded={status.get('weights_loaded')})."
        )

    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "route": args.depth,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_ms = None
    text_parts: list[str] = []
    telemetry: dict[str, object] = {}
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                detail = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Studio did not return an inference stream: {detail[:1000]}")
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]" or not data:
                    continue
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"].get("message", "Studio inference failed"))
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    if first_token_ms is None and delta["content"]:
                        first_token_ms = round((time.perf_counter() - started) * 1000, 1)
                    if chunk.get("triune_snapshot"):
                        text_parts[:] = [str(delta["content"])]
                    else:
                        text_parts.append(str(delta["content"]))
                if chunk.get("telemetry"):
                    telemetry = chunk["telemetry"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Studio returned HTTP {exc.code}: {detail[:1000]}") from exc

    return {
        "mode": "studio",
        "checkpoint": str(active),
        "device": telemetry.get("device"),
        "client_ttft_ms": first_token_ms,
        "client_total_ms": round((time.perf_counter() - started) * 1000, 1),
        "backend_ttft_ms": telemetry.get("ttft_ms"),
        "decode_tokens_per_sec": telemetry.get("decode_tokens_per_sec"),
        "tokens_per_sec": telemetry.get("tokens_per_sec"),
        "tokens_count": telemetry.get("tokens_count"),
        "vram_gb": telemetry.get("vram_gb"),
        "route": telemetry.get("route"),
        "sample": "".join(text_parts)[:240],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cli", "studio"), required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file to benchmark")
    parser.add_argument("--tokenizer", default="triune_tokenizer.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--depth", choices=("auto", "reflex", "limbic", "cortex"), default="auto")
    parser.add_argument("--device", default=None, help="CLI device; defaults to CUDA when available")
    parser.add_argument("--studio-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.max_tokens > 8192:
        parser.error("--max-tokens must be between 1 and 8192")
    if args.mode == "studio" and args.device is not None:
        parser.error("--device applies only to --mode cli; Studio reports its actual runtime device")
    result = benchmark_cli(args) if args.mode == "cli" else benchmark_studio(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
