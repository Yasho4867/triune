"""Interactive terminal chat entry point backed by Lucid Machines inference, transmutation, and local Ollama."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lucid_machines.cli_ui import (
    ensure_utf8_stdio,
    is_tty,
    render_banner,
    render_kv_panel,
    render_notice,
    render_table,
    route_badge_ansi,
    status_spinner,
)
from lucid_machines.inference import (
    generate_response,
    load_checkpoint_model,
    load_model_for_inference,
    resolve_default_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser shared by scripts/chat.py and `lucid-machines chat`."""
    parser = argparse.ArgumentParser(
        description="Chat with a Lucid Machines checkpoint, HF/transmuted model, or local Ollama"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HF model repo ID, local HF directory, NVFP4 cache, or ollama:<tag>",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path (auto-resolves active/latest checkpoint when omitted)",
    )
    parser.add_argument("--tokenizer_path", default="triune_tokenizer.json")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory or safetensors path to attach")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--speculative", action="store_true", help="Enable Reflex-to-Cortex self-speculative decoding")
    parser.add_argument("--k-draft", type=int, default=3, help="Number of draft tokens per speculative cycle")
    parser.add_argument("--nvfp4", action="store_true", help="Quantize linear blocks to Hadamard-rotated NVFP4")
    parser.add_argument("--disk-stream", action="store_true", help="Stream NVFP4 layers directly from NVMe disk")
    parser.add_argument("--disk-cache", default=None, help="Optional directory for NVFP4 disk cache")
    parser.add_argument("--transmute", action="store_true", help="Attach Triune 3-tier early-exit heads to HF weights")
    parser.add_argument("--reflex-ratio", type=float, default=0.25, help="Reflex exit layer fraction for --transmute")
    parser.add_argument(
        "--depth",
        default="auto",
        choices=("auto", "reflex", "limbic", "cortex"),
        help="Initial routing depth override (default: auto)",
    )
    parser.add_argument("--device", default=None, help="Target device (cuda / cpu)")
    return parser


def _render_chat_help() -> None:
    render_table(
        "Interactive Chat Commands",
        columns=[
            ("Command", "left", "bold cyan"),
            ("Action", "left", "white"),
        ],
        rows=[
            ["/help", "Show interactive slash commands"],
            ["/route <tier>", "Switch depth tier: auto | reflex | limbic | cortex"],
            ["/temp <float>", "Set sampling temperature (e.g. /temp 0.3)"],
            ["/tokens <int>", "Set max generated tokens per turn (e.g. /tokens 512)"],
            ["/speculative", "Toggle Reflex-to-Cortex speculative decoding (native Triune)"],
            ["/scan", "List live Ollama, Hugging Face & local models on this PC"],
            ["/clear", "Reset conversation history"],
            ["/exit", "Exit chat session (or type 'quit')"],
        ],
    )


def main(argv: list[str] | None = None) -> None:
    """Run an interactive terminal chat loop with live streaming and telemetry."""
    ensure_utf8_stdio()
    args = build_parser().parse_args(argv)
    if not args.model:
        try:
            args.checkpoint = str(resolve_default_checkpoint(args.checkpoint))
        except FileNotFoundError as exc:
            render_notice(str(exc), level="error")
            raise SystemExit(1) from exc
    source = args.model or args.checkpoint
    depth_map: dict[str, int | None] = {"auto": None, "reflex": 0, "limbic": 1, "cortex": 2}
    active_depth: int | None = depth_map.get(args.depth)
    active_temp = float(args.temperature)
    active_max_tokens = int(args.max_new_tokens)
    use_speculative = bool(args.speculative)

    is_ollama = bool(args.model and str(args.model).strip().lower().startswith("ollama:"))
    model: Any = None
    tokenizer: Any = None
    metadata: dict[str, Any] = {}
    device: Any = "ollama"
    pad_id = 0
    eos_id = 0

    if is_ollama:
        from lucid_machines.runtime.model_scanner import _is_tag_live_in_ollama, query_ollama_running_tags

        ollama_tag = str(args.model).strip()
        if not _is_tag_live_in_ollama(ollama_tag, query_ollama_running_tags()):
            render_notice(
                f"'{ollama_tag}' is not currently running in local Ollama. "
                f"Run `ollama run {ollama_tag.removeprefix('ollama:')}` or run `lucid-machines scan` to inspect live models.",
                level="error",
            )
            raise SystemExit(1)
        mode_label = "Ollama Live Runtime"
    else:
        quantize = "nvfp4" if args.nvfp4 else None
        try:
            with status_spinner(f"Loading model '{source}'..."):
                if getattr(args, "transmute", False) and (args.model or Path(source).is_dir()):
                    from lucid_machines.model import transmuter as _transmuter

                    _transmute_fn = getattr(_transmuter, "transmute_hf_to_triune", None) or getattr(
                        _transmuter, "transmute_hf_to_lucid", None
                    )
                    model, tokenizer, metadata, device, pad_id, eos_id = _transmute_fn(
                        source,
                        reflex_ratio=args.reflex_ratio,
                        quantize=quantize,
                        disk_stream=args.disk_stream,
                        disk_cache_dir=args.disk_cache,
                        device=args.device,
                        tokenizer_path=args.tokenizer_path,
                    )
                    mode_label = "Triune 3-Tier Transmuted HF"
                elif args.model is not None or args.nvfp4 or args.disk_stream or Path(source).is_dir():
                    model, tokenizer, metadata, device, pad_id, eos_id = load_model_for_inference(
                        source,
                        args.tokenizer_path,
                        device=args.device,
                        adapter_path=args.adapter,
                        quantize=quantize,
                        disk_stream=args.disk_stream,
                        disk_cache_dir=args.disk_cache,
                    )
                    mode_label = "Triune Native" if hasattr(model, "init_cache") else "HF / NVFP4 PyTorch"
                else:
                    model, tokenizer, metadata, device, pad_id, eos_id = load_checkpoint_model(
                        args.checkpoint, args.tokenizer_path, device=args.device, adapter_path=args.adapter
                    )
                    mode_label = "Triune Checkpoint"
        except (FileNotFoundError, OSError, RuntimeError, ValueError, EOFError, KeyError) as exc:
            render_notice(f"Failed to load model '{source}': {exc}", level="error")
            raise SystemExit(1) from exc

    is_triune_model = bool(
        not is_ollama
        and (
            hasattr(model, "init_cache")
            or getattr(model, "_is_triune", False)
            or getattr(model, "_is_triune_transmuted", False)
        )
    )
    has_chat_tmpl = bool(
        not is_ollama and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)
    )

    render_banner("LUCID MACHINES CHAT", subtitle=str(source), badge=mode_label)
    session_rows: list[tuple[str, Any]] = [
        ("Device / Runtime", str(device)),
        ("Route", args.depth.upper() if is_triune_model else ("OLLAMA" if is_ollama else "DENSE HF")),
        ("Sampling", f"temp={active_temp}  ·  max_tokens={active_max_tokens}  ·  rep={args.repetition_penalty}"),
    ]
    if args.adapter:
        session_rows.append(("LoRA Adapter", args.adapter))
    if metadata.get("reflex_exit_layer") is not None:
        session_rows.append(
            (
                "Exit Layers",
                f"Reflex L{metadata['reflex_exit_layer']}  ·  Limbic L{metadata.get('limbic_exit_layer')}",
            )
        )
    if metadata.get("best_eval_loss") is not None:
        session_rows.append(("Best Eval Loss", metadata["best_eval_loss"]))
    render_kv_panel(
        "Session Ready",
        session_rows,
        border_style="cyan",
        footer="Type `/help` for slash commands, `/clear` to reset context, or `exit` to quit.",
    )

    history: list[dict[str, str]] = []
    tty = is_tty()
    user_prefix = "\n\033[1;33myou ❯ \033[0m" if tty else "\nYou: "

    try:
        while True:
            try:
                raw_prompt = input(user_prefix)
            except (EOFError, KeyboardInterrupt):
                print()
                render_notice("Session ended.", level="info")
                return

            prompt = raw_prompt.strip()
            if not prompt:
                continue
            low = prompt.lower()
            if low in {"exit", "quit", "/exit", "/quit"}:
                render_notice("Goodbye.", level="info")
                return

            if low in {"/help", "help", "?"}:
                _render_chat_help()
                continue

            if low in {"/clear", "clear"}:
                history.clear()
                render_notice("Conversation history cleared.", level="success")
                continue

            if low == "/scan":
                from lucid_machines.runtime.model_scanner import scan_pc_models

                scan_data = scan_pc_models()
                rows = [
                    [
                        m.get("source", m.get("kind", "")),
                        m.get("name") or m.get("id"),
                        "LIVE" if m.get("ollama_running") else ("Direct" if m.get("can_load_direct") else "Transmute"),
                    ]
                    for m in scan_data.get("models", [])[:20]
                ]
                render_table(
                    f"Discovered Models ({len(scan_data.get('models', []))} found)",
                    [("Source", "left", "cyan"), ("Model", "left", "white"), ("Status", "left", "green")],
                    rows,
                )
                continue

            if low.startswith("/temp "):
                try:
                    active_temp = max(0.0, min(2.0, float(prompt.split(None, 1)[1])))
                    render_notice(f"Temperature updated to {active_temp:.2f}", level="success")
                except ValueError:
                    render_notice("Usage: /temp <float> (e.g. /temp 0.7)", level="warn")
                continue

            if low.startswith("/tokens "):
                try:
                    active_max_tokens = max(1, int(prompt.split(None, 1)[1]))
                    render_notice(f"Max tokens updated to {active_max_tokens}", level="success")
                except ValueError:
                    render_notice("Usage: /tokens <int> (e.g. /tokens 256)", level="warn")
                continue

            if low.startswith("/route "):
                tier = low.split(None, 1)[1].strip()
                if tier in depth_map:
                    active_depth = depth_map[tier]
                    render_notice(f"Route set to {tier.upper()}", level="success")
                else:
                    render_notice("Usage: /route <auto|reflex|limbic|cortex>", level="warn")
                continue

            if low == "/speculative":
                use_speculative = not use_speculative
                render_notice(
                    f"Speculative decoding {'enabled' if use_speculative else 'disabled'}",
                    level="success",
                )
                continue

            turn_depth = active_depth
            if is_triune_model:
                for command, depth in (("reflex ", 0), ("limbic ", 1), ("cortex ", 2)):
                    if low.startswith(command):
                        turn_depth, prompt = depth, prompt[len(command) :]
                        break
                else:
                    if low.startswith("auto "):
                        turn_depth, prompt = None, prompt[5:]

            if is_ollama:
                from lucid_machines.runtime.model_scanner import stream_ollama_chat

                history.append({"role": "user", "content": prompt})
                sys.stdout.write(f"{route_badge_ansi('OLLAMA')} ")
                sys.stdout.flush()
                try:
                    ol_res = stream_ollama_chat(
                        str(args.model),
                        history,
                        max_new_tokens=active_max_tokens,
                        temperature=active_temp,
                        on_text=lambda d: (sys.stdout.write(d), sys.stdout.flush()),
                    )
                    print()
                    history.append({"role": "assistant", "content": ol_res["text"]})
                    ttft_s = f"{ol_res['ttft_ms']:.0f}ms" if ol_res.get("ttft_ms") else "N/A"
                    dim_open, dim_close = ("\033[2m", "\033[0m") if tty else ("", "")
                    print(
                        f"{dim_open}  ↳ {ol_res['tokens_per_sec']:.1f} tok/s · {ol_res['tokens_count']} tok · TTFT {ttft_s}{dim_close}"
                    )
                except Exception as exc:
                    print()
                    history.pop()
                    render_notice(f"Ollama generation failed: {exc}", level="error")
                continue

            if use_speculative and hasattr(model, "init_cache"):
                from lucid_machines.model.speculative import SelfSpeculativeGenerator

                spec_gen = SelfSpeculativeGenerator(
                    model,
                    tokenizer,
                    k_draft=args.k_draft,
                    temperature=active_temp,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
                spec_res = spec_gen.generate(prompt, max_new_tokens=active_max_tokens)
                print(f"{route_badge_ansi('SPEC')} {spec_res['text']}")
                dim_open, dim_close = ("\033[2m", "\033[0m") if tty else ("", "")
                print(
                    f"{dim_open}  ↳ K={args.k_draft} · acc={spec_res['acceptance_rate']:.1%} · {spec_res['tokens_per_second']} tok/s{dim_close}"
                )
                continue

            effective_prompt = prompt
            prompt_is_formatted = False
            if has_chat_tmpl and not hasattr(model, "init_cache"):
                history.append({"role": "user", "content": prompt})
                effective_prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                prompt_is_formatted = True

            initial_label = {0: "REFLEX", 1: "LIMBIC", 2: "CORTEX"}.get(
                turn_depth, "TRIUNE" if is_triune_model else "HF"
            )
            sys.stdout.write(f"{route_badge_ansi(initial_label)} ")
            sys.stdout.flush()
            printed_len = [0]

            def _stream_delta(cum_text: Any, **_: Any) -> None:
                chunk = str(cum_text)[printed_len[0] :]
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    printed_len[0] = len(str(cum_text))

            res = generate_response(
                model,
                tokenizer,
                effective_prompt,
                device=device,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                force_depth=turn_depth,
                temperature=active_temp,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=active_max_tokens,
                prompt_is_formatted=prompt_is_formatted,
                on_text=_stream_delta,
                return_info=True,
            )
            print()
            if has_chat_tmpl and not hasattr(model, "init_cache"):
                history.append({"role": "assistant", "content": str(res["text"])})
            dim_open, dim_close = ("\033[2m", "\033[0m") if tty else ("", "")
            ttft_str = f"{res['ttft_ms']:.0f}ms" if res.get("ttft_ms") else "N/A"
            print(
                f"{dim_open}  ↳ {res.get('route', initial_label)} · {res.get('tokens_per_sec', 0):.1f} tok/s · {res.get('tokens_count', 0)} tok · TTFT {ttft_str}{dim_close}"
            )
    finally:
        if getattr(model, "_streaming_engine", None) is not None:
            model._streaming_engine.detach()
        if getattr(model, "_exit_manager", None) is not None and hasattr(model._exit_manager, "remove_hooks"):
            model._exit_manager.remove_hooks()


if __name__ == "__main__":
    main()
