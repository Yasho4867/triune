#!/usr/bin/env python3
"""Build publication-ready PDF research papers using Tectonic.

Compiles:
1. papers/paper1_centroidsteer/main.tex -> papers/paper1_centroidsteer/centroidsteer.pdf
2. papers/paper2_triune_systems/main.tex -> papers/paper2_triune_systems/triune_architecture.pdf
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = REPO_ROOT / "papers"
TECTONIC_PATH = Path(os.environ.get("LOCALAPPDATA", "")) / "tectonic" / "tectonic.exe"

if not TECTONIC_PATH.exists():
    which_t = shutil.which("tectonic")
    if which_t:
        TECTONIC_PATH = Path(which_t)
    else:
        print(f"Error: tectonic executable not found at {TECTONIC_PATH}", file=sys.stderr)
        sys.exit(1)


def compile_paper(paper_dir: Path, src_name: str, out_name: str) -> bool:
    print(f"--> Compiling {paper_dir.name}/{src_name}...", flush=True)
    tex_file = paper_dir / src_name
    pdf_out = paper_dir / out_name

    cmd = [str(TECTONIC_PATH), "--keep-intermediates", str(tex_file)]
    res = subprocess.run(cmd, cwd=str(paper_dir), capture_output=True, text=True)

    if res.returncode != 0:
        print(f"[FAIL] Tectonic compilation failed for {tex_file}:", flush=True)
        print(res.stdout, flush=True)
        print(res.stderr, flush=True)
        return False

    default_pdf = paper_dir / (tex_file.stem + ".pdf")
    if default_pdf.exists() and default_pdf != pdf_out:
        if pdf_out.exists():
            pdf_out.unlink()
        default_pdf.rename(pdf_out)

    if pdf_out.exists():
        size_kb = pdf_out.stat().st_size / 1024
        print(f"[OK] Generated {pdf_out} ({size_kb:.1f} KB)", flush=True)
        return True
    else:
        print(f"[FAIL] Expected output {pdf_out} not found.", flush=True)
        return False


def main() -> None:
    print("=" * 70, flush=True)
    print("TRIUNE RESEARCH PAPERS BUILD PIPELINE", flush=True)
    print(f"Compiler: {TECTONIC_PATH}", flush=True)
    print("=" * 70, flush=True)

    p1_dir = PAPERS_DIR / "paper1_centroidsteer"
    p2_dir = PAPERS_DIR / "paper2_triune_systems"

    ok1 = compile_paper(p1_dir, "main.tex", "centroidsteer.pdf")
    ok2 = compile_paper(p2_dir, "main.tex", "triune_architecture.pdf")

    if ok1 and ok2:
        print("=" * 70)
        print("[SUCCESS] All research paper PDFs built successfully!")
        print(f"1. {p1_dir / 'centroidsteer.pdf'}")
        print(f"2. {p2_dir / 'triune_architecture.pdf'}")
        print("=" * 70)
    else:
        print("Build failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
