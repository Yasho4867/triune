"""Terminal visual system and Rich component helpers for the Lucid Machines CLI."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
from typing import Any, Iterable

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:  # pragma: no cover - fallback for bare environments without rich
    HAS_RICH = False


def ensure_utf8_stdio() -> None:
    """Reconfigure Windows stdio streams to UTF-8 so Unicode tables render cleanly."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                if hasattr(stream, "reconfigure"):
                    stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _terminal_width() -> int:
    if not is_tty():
        try:
            cols = shutil.get_terminal_size(fallback=(140, 30)).columns
            return max(140, int(cols))
        except Exception:
            return 140
    try:
        cols = shutil.get_terminal_size(fallback=(104, 30)).columns
        return max(72, min(140, int(cols)))
    except Exception:
        return 104


def _make_render_console(width: int | None = None, capture: bool = True) -> tuple[Any, io.StringIO]:
    buf = io.StringIO()
    tty = is_tty()
    # Allow explicit override cleanly while respecting non-TTY redirected output
    color_sys = "truecolor" if tty else None
    console = Console(
        file=buf if capture else sys.stdout,
        force_terminal=tty,
        color_system=color_sys,
        no_color=not tty,
        width=width or _terminal_width(),
        legacy_windows=False,
        soft_wrap=True,
    )
    return console, buf


def emit_renderable(renderable: Any) -> None:
    """Render a Rich object (or string) and print via built-in print() for clean testability."""
    if not HAS_RICH:
        print(str(renderable))
        return
    console, buf = _make_render_console()
    console.print(renderable)
    rendered = buf.getvalue()
    if rendered.endswith("\n"):
        rendered = rendered[:-1]
    print(rendered)


def route_badge_markup(route: str) -> str:
    """Return a styled Rich markup badge for a Triune routing tier or Ollama."""
    r = str(route or "AUTO").upper()
    if "REFLEX" in r:
        return "[bold cyan][REFLEX][/]"
    if "LIMBIC" in r:
        return "[bold yellow][LIMBIC][/]"
    if "CORTEX" in r:
        return "[bold magenta][CORTEX][/]"
    if "OLLAMA" in r:
        return "[bold green][OLLAMA][/]"
    if "SPEC" in r:
        return "[bold bright_cyan][SPECULATIVE][/]"
    return f"[bold green][{r}][/]"


def route_badge_ansi(route: str) -> str:
    """Return a colored ANSI route badge for interactive prompt headers."""
    r = str(route or "AUTO").upper()
    tty = is_tty()
    if not tty:
        return f"[{r}]"
    if "REFLEX" in r:
        return "\033[1;36m[REFLEX]\033[0m"
    if "LIMBIC" in r:
        return "\033[1;33m[LIMBIC]\033[0m"
    if "CORTEX" in r:
        return "\033[1;35m[CORTEX]\033[0m"
    if "OLLAMA" in r:
        return "\033[1;32m[OLLAMA]\033[0m"
    return f"\033[1;32m[{r}]\033[0m"


def render_banner(title: str = "LUCID MACHINES", subtitle: str | None = None, badge: str | None = None) -> None:
    """Display a sleek compact Lucid Machines header panel."""
    if not HAS_RICH:
        badge_part = f"  [{badge}]" if badge else ""
        sub_part = f" — {subtitle}" if subtitle else ""
        print(f"\n▲ {title}{badge_part}{sub_part}")
        return

    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=3)
    grid.add_column(justify="right", ratio=1)

    left = Text()
    left.append("▲ ", style="bold #f59e0b")
    left.append(title, style="bold white")
    if subtitle:
        left.append(f"  ·  {subtitle}", style="dim")

    right = Text(badge or "", style="bold cyan")
    grid.add_row(left, right)

    emit_renderable(
        Panel(
            grid,
            box=box.ROUNDED,
            border_style="#f59e0b",
            padding=(0, 2),
        )
    )


def render_kv_panel(
    title: str,
    rows: Iterable[tuple[str, Any]],
    *,
    border_style: str = "bright_black",
    footer: str | None = None,
) -> None:
    """Display an aligned key-value configuration or hardware card."""
    rows_list = list(rows)
    if not HAS_RICH:
        print(f"\n[{title}]")
        w = max((len(str(k)) for k, _ in rows_list), default=12)
        for k, v in rows_list:
            print(f"  {str(k):<{w}} : {v}")
        if footer:
            print(f"  {footer}")
        print()
        return

    grid = Table.grid(padding=(0, 2), expand=False)
    grid.add_column(style="bold dim", no_wrap=True)
    grid.add_column(style="white")
    for key, val in rows_list:
        grid.add_row(str(key), str(val))

    if footer:
        grid.add_row("", "")
        grid.add_row(Text("Note", style="bold yellow"), Text(footer, style="dim"))

    emit_renderable(
        Panel(
            grid,
            title=f"[bold white]{title}[/]",
            title_align="left",
            box=box.ROUNDED,
            border_style=border_style,
            padding=(0, 2),
        )
    )


def render_table(
    title: str,
    columns: list[tuple[str, str, str]],
    rows: list[list[Any]],
    *,
    caption: str | None = None,
    border_style: str = "bright_black",
) -> None:
    """Render a rounded multi-column table. Columns are (header, justify, style)."""
    if not HAS_RICH:
        print(f"\n{title}")
        print("-" * 76)
        for row in rows:
            print("  " + " | ".join(str(x) for x in row))
        if caption:
            print(caption)
        print()
        return

    table = Table(
        title=f"[bold white]{title}[/]",
        title_justify="left",
        box=box.ROUNDED,
        border_style=border_style,
        header_style="bold #f59e0b",
        caption=f"[dim]{caption}[/]" if caption else None,
        caption_justify="left",
        show_lines=False,
        pad_edge=True,
    )
    for col_name, col_justify, col_style in columns:
        table.add_column(col_name, justify=col_justify, style=col_style)

    for r in rows:
        table.add_row(*(str(c) for c in r))

    emit_renderable(table)


def render_notice(message: str, level: str = "info") -> None:
    """Print a styled status line (info / success / warn / error)."""
    icons = {"info": "•", "success": "✓", "warn": "!", "error": "✗"}
    styles = {"info": "cyan", "success": "bold green", "warn": "bold yellow", "error": "bold red"}
    icon = icons.get(level, "•")
    if not HAS_RICH:
        print(f"{icon} {message}")
        return
    style = styles.get(level, "white")
    emit_renderable(Text.from_markup(f"[{style}]{icon}[/] {message}"))


def render_telemetry_bar(
    route: str,
    *,
    tok_sec: float = 0.0,
    ttft_ms: float | None = None,
    tokens: int = 0,
    latency_ms: float = 0.0,
    vram_gb: float = 0.0,
    load_s: float | None = None,
    extra: str | None = None,
) -> None:
    """Render a crisp single-row telemetry pill footer."""
    ttft_str = f"{ttft_ms:.1f}ms" if ttft_ms else "N/A"
    parts = [
        route_badge_markup(route),
        f"[bold cyan]{tok_sec:.1f} tok/s[/]",
        f"[dim]TTFT[/] [yellow]{ttft_str}[/]",
        f"[dim]Tokens[/] [white]{tokens}[/] [dim]({latency_ms:.0f}ms)[/]",
        f"[dim]VRAM[/] [white]{vram_gb:.2f} GB[/]",
    ]
    if load_s is not None:
        parts.append(f"[dim]Load[/] [white]{load_s:.2f}s[/]")
    if extra:
        parts.append(f"[dim]{extra}[/]")

    if not HAS_RICH:
        print(
            f"Telemetry: [{route}] | {tok_sec:.1f} tok/s | TTFT: {ttft_str} | {tokens} tok in {latency_ms:.0f}ms | VRAM: {vram_gb:.2f} GB"
        )
        return

    emit_renderable(
        Panel(
            Text.from_markup("  [dim]│[/]  ".join(parts)),
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(0, 2),
        )
    )


def render_rule(title: str = "") -> None:
    if not HAS_RICH:
        print(f"\n--- {title} ---" if title else "\n" + "-" * 60)
        return
    emit_renderable(Rule(f"[dim]{title}[/]" if title else "", style="bright_black"))


@contextlib.contextmanager
def status_spinner(message: str):
    """Display an animated terminal spinner on TTYs, or a clean status note otherwise."""
    if HAS_RICH and is_tty():
        console = Console(force_terminal=True, color_system="truecolor", legacy_windows=False)
        with console.status(f"[bold cyan]{message}[/]", spinner="dots"):
            yield
    else:
        print(f"• {message}", flush=True)
        yield


def render_help_overview() -> None:
    """Render the grouped flagship `lucid-machines --help` command dashboard."""
    render_banner(
        "LUCID MACHINES CLI",
        subtitle="Triune 3-Tier Adaptive MoE + Weight Transmutation & Local Runtime Engine",
        badge="v1.0.0",
    )

    groups: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Inference & Local Models",
            [
                (
                    "agent",
                    "Self-learning cognitive agent harness (adaptive personas, skills, memory, integrated tools)",
                ),
                ("chat", "Interactive terminal chat (supports checkpoints, HF, NVFP4, --transmute, or ollama:tag)"),
                ("infer", "Single-turn streaming generation with Reflex / Limbic / Cortex telemetry"),
                ("scan", "Scan PC for Ollama (live status), Hugging Face, vLLM, Hermes, LM Studio & checkpoints"),
                ("transmute", "Convert HF/SafeTensors into 3-Tier Triune (early-exit heads or native surgical KL)"),
            ],
        ),
        (
            "Polysom Studio & API Server",
            [
                ("studio", "Launch Polysom Studio native desktop / browser workspace"),
                ("serve", "Start the embedded FastAPI + OpenAI-compatible server"),
            ],
        ),
        (
            "Training & Weights",
            [
                ("finetune", "LoRA / 4-bit NVFP4 QLoRA & layer-streamed fine-tuning"),
                ("train", "Run full pre-training or distributed training scripts"),
                ("adapter", "List, merge, or export LoRA/QLoRA adapters"),
                ("checkpoint", "Inspect and export local .pt training checkpoints"),
                ("export", "Export checkpoints directly to SafeTensors, GGUF, ONNX, or HF"),
                ("tokenizer", "Build or train a domain BPE tokenizer"),
            ],
        ),
        (
            "Zoo & Diagnostics",
            [
                ("models", "List canonical Triune Model Zoo architecture presets"),
                ("doctor", "Full diagnostic dashboard (CUDA, C/AVX2 engine, Ollama, session state)"),
                ("gpucheck", "CUDA device, compute capability, VRAM, and PyTorch check"),
                ("plan-memory", "Estimate VRAM/RAM & NVFP4 disk-stream budget for Triune or HF models"),
                ("cache-nvfp4", "Pack HF SafeTensors shards into an NVMe NVFP4 stream cache (~0.625 B/param)"),
                ("benchmark", "Benchmark kernel, optimizer, streaming, context, or inference latency"),
                ("profile", "Profile step compute & memory breakdowns"),
                ("install", "Run environment setup or native installer wizard"),
                ("build-native", "Package desktop installers for Windows, Linux, or macOS"),
            ],
        ),
    ]

    if not HAS_RICH:
        for grp_title, cmds in groups:
            print(f"\n[{grp_title}]")
            for cmd, desc in cmds:
                print(f"  {cmd:<14} {desc}")
        print("\nExamples:")
        print("  lucid-machines scan")
        print("  lucid-machines chat --model ollama:gemma3:4b")
        print('  lucid-machines transmute --model Qwen/Qwen2.5-1.5B-Instruct --prompt "Hello!"')
        print("  lucid-machines studio\n")
        return

    for grp_title, cmds in groups:
        render_table(
            grp_title,
            columns=[
                ("Command", "left", "bold cyan"),
                ("Description", "left", "white"),
            ],
            rows=[[f"lucid-machines {cmd}", desc] for cmd, desc in cmds],
            border_style="bright_black",
        )

    examples = [
        ("Scan local runtimes", "lucid-machines scan"),
        ("Chat with live Ollama", "lucid-machines chat --model ollama:gemma3:4b"),
        ("Zero-loss 3-tier transmute", "lucid-machines transmute --model <hf_dir_or_repo> --mode early-exit"),
        ("Run diagnostics", "lucid-machines doctor"),
        ("Launch desktop IDE", "lucid-machines studio"),
    ]
    render_kv_panel(
        "Quickstart Examples",
        examples,
        border_style="cyan",
        footer="Run `lucid-machines <command> --help` for detailed flags on any subcommand.",
    )
