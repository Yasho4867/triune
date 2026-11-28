#!/usr/bin/env python3
"""Build, audit, and sync publication-ready PDF research papers using Tectonic.

Compiles:
1. papers/paper1_centroidsteer/main.tex -> papers/paper1_centroidsteer/centroidsteer.pdf
2. papers/paper2_triune_systems/main.tex -> papers/paper2_triune_systems/triune_architecture.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = REPO_ROOT / "papers"
WEBSITE_PAPERS_DIR = REPO_ROOT / "website" / "papers"
WEBSITE_PUBLIC_PAPERS_DIR = REPO_ROOT / "website" / "public" / "papers"

INTERMEDIATE_SUFFIXES = (".aux", ".out", ".bbl", ".blg", ".log", ".xdv")

PAPER_SPECS = {
    "paper1": {
        "dir": PAPERS_DIR / "paper1_centroidsteer",
        "src": "main.tex",
        "pdf": "centroidsteer.pdf",
        "website_tex": "paper1_centroidsteer.tex",
    },
    "paper2": {
        "dir": PAPERS_DIR / "paper2_triune_systems",
        "src": "main.tex",
        "pdf": "triune_architecture.pdf",
        "website_tex": "paper2_triune_systems.tex",
    },
}


def find_tectonic() -> Path:
    env_path = Path(os.environ.get("LOCALAPPDATA", "")) / "tectonic" / "tectonic.exe"
    if env_path.exists():
        return env_path
    which_t = shutil.which("tectonic")
    if which_t:
        return Path(which_t)
    raise FileNotFoundError(f"Tectonic executable not found at {env_path} or on PATH")


@dataclass
class AuditReport:
    paper_key: str
    pages: int = 0
    preflight_errors: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    overfull_hboxes: list[tuple[float, str]] = field(default_factory=list)
    hyperref_warnings: list[str] = field(default_factory=list)
    utf8_warnings: list[str] = field(default_factory=list)
    object_defined_warnings: list[str] = field(default_factory=list)
    undefined_warnings: list[str] = field(default_factory=list)


def strip_tex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        # Strip unescaped % comments
        stripped = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(stripped)
    return "\n".join(lines)


def run_preflight_checks(paper_dir: Path, tex_file: Path) -> list[str]:
    errors: list[str] = []
    raw_text = tex_file.read_text(encoding="utf-8")
    text = strip_tex_comments(raw_text)

    if re.search(r"\*\*[^*\n]+\*\*", text):
        errors.append(f"{tex_file.name}: raw Markdown bold '**...**' detected")
    if re.search(r"(?<!`)`[A-Za-z_\\]", text):
        errors.append(f"{tex_file.name}: raw Markdown inline code backtick '`...' detected")
    if re.search(r"\btriune\b[^\n.]{0,80}\bshim|\bshim[^\n.]{0,80}\btriune\b", text, re.IGNORECASE):
        errors.append(f"{tex_file.name}: stale 'triune' shim claim detected")
    if re.search(r"research@lucidmachines\.co|\[author-email\]|\[corresponding-email|placeholder", text, re.IGNORECASE):
        errors.append(f"{tex_file.name}: placeholder email or placeholder text detected")

    for fig_rel in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        fig_path = paper_dir / fig_rel
        if not fig_path.exists():
            errors.append(f"{tex_file.name}: missing figure target '{fig_rel}'")

    bib_match = re.search(r"\\bibliography\{([^}]+)\}", text)
    if bib_match:
        for bib_name in [b.strip() for b in bib_match.group(1).split(",")]:
            bib_file = paper_dir / (bib_name if bib_name.endswith(".bib") else f"{bib_name}.bib")
            if not bib_file.exists():
                errors.append(f"{tex_file.name}: missing bibliography file '{bib_file.name}'")

    return errors


def parse_log(log_text: str, report: AuditReport) -> None:
    for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\).*?(lines? [0-9-]+)", log_text):
        report.overfull_hboxes.append((float(m.group(1)), m.group(2)))

    for line in log_text.splitlines():
        if re.search(r"Undefined control sequence|LaTeX Error:|Fatal error|^! ", line):
            report.fatal_errors.append(line.strip())
        if re.search(r"Package hyperref Warning:.*Token not allowed", line):
            report.hyperref_warnings.append(line.strip())
        if "Invalid UTF-8" in line:
            report.utf8_warnings.append(line.strip())
        if re.search(r"Object .* already defined", line):
            report.object_defined_warnings.append(line.strip())
        if re.search(r"There were undefined references|Citation .* undefined|Reference .* undefined", line):
            report.undefined_warnings.append(line.strip())

    page_match = re.search(r"Output written on .*?\((\d+)\s+pages?", log_text)
    if page_match:
        report.pages = int(page_match.group(1))


def clean_intermediates(paper_dir: Path, stem: str) -> None:
    for suffix in INTERMEDIATE_SUFFIXES:
        candidate = paper_dir / f"{stem}{suffix}"
        if candidate.exists():
            candidate.unlink()


def compile_paper(
    tectonic_path: Path,
    paper_key: str,
    spec: dict,
    strict: bool,
    max_overfull_pt: float,
    keep_intermediates: bool,
) -> tuple[bool, AuditReport]:
    paper_dir: Path = spec["dir"]
    src_name: str = spec["src"]
    out_name: str = spec["pdf"]
    tex_file = paper_dir / src_name
    pdf_out = paper_dir / out_name
    stem = tex_file.stem

    report = AuditReport(paper_key=paper_key)
    print(f"--> Compiling {paper_dir.name}/{src_name}...", flush=True)

    report.preflight_errors = run_preflight_checks(paper_dir, tex_file)
    if report.preflight_errors:
        for err in report.preflight_errors:
            print(f"    [PREFLIGHT FAIL] {err}", flush=True)
        if strict:
            return False, report

    # Clean stale intermediates before build
    clean_intermediates(paper_dir, stem)

    cmd = [str(tectonic_path), "--keep-logs"]
    if keep_intermediates:
        cmd.append("--keep-intermediates")
    cmd.append(str(tex_file))

    res = subprocess.run(cmd, cwd=str(paper_dir), capture_output=True, text=True, encoding="utf-8", errors="replace")

    log_file = paper_dir / f"{stem}.log"
    combined_diag = (res.stdout or "") + "\n" + (res.stderr or "")
    if log_file.exists():
        combined_diag = log_file.read_text(encoding="utf-8", errors="replace") + "\n" + combined_diag

    parse_log(combined_diag, report)

    if not keep_intermediates:
        clean_intermediates(paper_dir, stem)

    if res.returncode != 0 or report.fatal_errors:
        print(f"[FAIL] Tectonic compilation failed for {tex_file} (exit={res.returncode}):", flush=True)
        for err in report.fatal_errors[:10]:
            print(f"    {err}", flush=True)
        if res.stderr:
            print(res.stderr.strip(), flush=True)
        return False, report

    default_pdf = paper_dir / f"{stem}.pdf"
    if default_pdf.exists() and default_pdf != pdf_out:
        if pdf_out.exists():
            pdf_out.unlink()
        default_pdf.rename(pdf_out)

    if not pdf_out.exists():
        print(f"[FAIL] Expected output {pdf_out} not found.", flush=True)
        return False, report

    bad_overfull = [(pt, loc) for pt, loc in report.overfull_hboxes if pt > max_overfull_pt]
    size_kb = pdf_out.stat().st_size / 1024
    print(
        f"[OK] Generated {pdf_out} ({size_kb:.1f} KB, {report.pages} pages) | "
        f"Overfull>{max_overfull_pt:.1f}pt: {len(bad_overfull)} (total={len(report.overfull_hboxes)}) | "
        f"hyperref: {len(report.hyperref_warnings)} | UTF-8: {len(report.utf8_warnings)} | "
        f"obj-dup: {len(report.object_defined_warnings)} | undef: {len(report.undefined_warnings)}",
        flush=True,
    )

    if strict:
        strict_failures: list[str] = []
        if report.preflight_errors:
            strict_failures.append(f"preflight={len(report.preflight_errors)}")
        if bad_overfull:
            details = ", ".join(f"{pt:.2f}pt@{loc}" for pt, loc in bad_overfull)
            strict_failures.append(f"overfull>{max_overfull_pt:.1f}pt=[{details}]")
        if report.hyperref_warnings:
            strict_failures.append(f"hyperref={len(report.hyperref_warnings)}")
        if report.utf8_warnings:
            strict_failures.append(f"utf8={len(report.utf8_warnings)}")
        if report.object_defined_warnings:
            strict_failures.append(f"obj_defined={len(report.object_defined_warnings)}")
        if report.undefined_warnings:
            strict_failures.append(f"undefined={len(report.undefined_warnings)}")
        if strict_failures:
            print(f"[STRICT FAIL] {paper_key}: " + "; ".join(strict_failures), flush=True)
            return False, report

    return True, report


def sync_website(selected_keys: list[str]) -> None:
    WEBSITE_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    WEBSITE_PUBLIC_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    for key in selected_keys:
        spec = PAPER_SPECS[key]
        paper_dir: Path = spec["dir"]
        src_pdf = paper_dir / spec["pdf"]
        src_tex = paper_dir / spec["src"]
        dst_pdf = WEBSITE_PAPERS_DIR / spec["pdf"]
        dst_pub_pdf = WEBSITE_PUBLIC_PAPERS_DIR / spec["pdf"]
        dst_tex = WEBSITE_PAPERS_DIR / spec["website_tex"]
        shutil.copy2(src_pdf, dst_pdf)
        shutil.copy2(src_pdf, dst_pub_pdf)
        shutil.copy2(src_tex, dst_tex)
        print(f"[SYNC] {src_pdf.name} & {dst_tex.name} -> {WEBSITE_PAPERS_DIR} (+ {WEBSITE_PUBLIC_PAPERS_DIR})", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit Lucid Machines research papers.")
    parser.add_argument(
        "--paper",
        choices=["all", "paper1", "paper2"],
        default="all",
        help="Which manuscript(s) to compile (default: all).",
    )
    parser.add_argument(
        "--with-figures",
        action="store_true",
        help="Regenerate vector figures via generate_figures.py before compiling.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on preflight leaks, Overfull \\hbox > --max-overfull-pt, hyperref/UTF-8/duplicate-object/undefined warnings.",
    )
    parser.add_argument(
        "--max-overfull-pt",
        type=float,
        default=2.0,
        help="Maximum allowed Overfull \\hbox in pt under --strict (default: 2.0).",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Retain .aux/.out/.bbl/.blg/.log/.xdv files after compilation (default: False).",
    )
    parser.add_argument(
        "--sync-website",
        action="store_true",
        help="Copy built PDFs and .tex sources to website/papers/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tectonic_path = find_tectonic()

    print("=" * 70, flush=True)
    print("LUCID MACHINES RESEARCH PAPERS BUILD PIPELINE", flush=True)
    print(f"Compiler: {tectonic_path}", flush=True)
    print("=" * 70, flush=True)

    if args.with_figures:
        fig_script = PAPERS_DIR / "scripts" / "generate_figures.py"
        print(f"--> Regenerating figures via {fig_script}...", flush=True)
        fig_res = subprocess.run([sys.executable, str(fig_script)], cwd=str(REPO_ROOT), check=False)
        if fig_res.returncode != 0:
            print("Figure generation failed.", file=sys.stderr)
            sys.exit(fig_res.returncode)

    keys = ["paper1", "paper2"] if args.paper == "all" else [args.paper]
    all_ok = True
    for key in keys:
        ok, _ = compile_paper(
            tectonic_path=tectonic_path,
            paper_key=key,
            spec=PAPER_SPECS[key],
            strict=args.strict,
            max_overfull_pt=args.max_overfull_pt,
            keep_intermediates=args.keep_intermediates,
        )
        all_ok = all_ok and ok

    if not all_ok:
        print("Build failed.", file=sys.stderr)
        sys.exit(1)

    if args.sync_website:
        sync_website(keys)

    print("=" * 70)
    print("[SUCCESS] All requested research paper PDFs built cleanly!")
    for key in keys:
        spec = PAPER_SPECS[key]
        print(f"  - {spec['dir'] / spec['pdf']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
