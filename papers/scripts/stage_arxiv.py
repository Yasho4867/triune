#!/usr/bin/env python3
"""Stage clean, self-contained local arXiv source tarballs under papers/dist/arxiv/.

This script is strictly LOCAL-ONLY (no network calls, no arXiv submission, no git push).
For each requested paper:
1. Runs pre-flight checks from build_papers.py.
2. Compiles main.tex + references.bib + figures/*.pdf in an isolated temp directory
   to verify clean compilation and generate main.bbl.
3. Builds a clean stage_root/ with ONLY:
   - main.tex (with \\bibliographystyle{...} + \\bibliography{...} replaced by the
     generated main.bbl \\begin{thebibliography}...\\end{thebibliography} block so the
     bundle compiles identically under both arXiv pdflatex AutoTeX and Tectonic)
   - figures/<name>.pdf for each \\includegraphics target referenced in main.tex
   (zero .png duplicates, zero compiled root PDFs, zero .aux/.out/.blg/.log files).
4. Packs deterministic .tar.gz archives (POSIX '/' paths, root-level entries, sorted
   order) into papers/dist/arxiv/<paper>-arxiv.tar.gz (gitignored via dist/).
5. Extracts each tarball into a SECOND fresh temp directory, verifies archive contents,
   runs Tectonic self-compile verification asserting exit 0, zero undefined refs/cites,
   zero Overfull \\hbox > 2.0pt, and matching page count, and prints SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_papers import (  # noqa: E402
    PAPER_SPECS,
    PAPERS_DIR,
    REPO_ROOT,
    find_tectonic,
    parse_log,
    run_preflight_checks,
    strip_tex_comments,
    AuditReport,
)

DIST_ARXIV_DIR = PAPERS_DIR / "dist" / "arxiv"
ARCHIVE_NAMES = {
    "paper1": "paper1_centroidsteer-arxiv.tar.gz",
    "paper2": "paper2_triune_systems-arxiv.tar.gz",
}
FORBIDDEN_SUFFIXES = {".aux", ".out", ".blg", ".log", ".toc", ".fls", ".fdb_latexmk", ".xdv", ".png"}


def inline_bbl_into_tex(tex_text: str, bbl_text: str) -> str:
    # Remove \bibliographystyle{...} and replace \bibliography{...} with inlined .bbl
    without_style = re.sub(r"\\bibliographystyle\{[^}]+\}\s*\n?", "", tex_text)
    bbl_clean = bbl_text.strip()
    replaced, count = re.subn(
        r"\\bibliography\{[^}]+\}",
        lambda _m: bbl_clean,
        without_style,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Expected exactly one \\bibliography{...} command to inline main.bbl.")
    return replaced


def collect_referenced_figures(tex_text: str) -> list[str]:
    clean = strip_tex_comments(tex_text)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean)
    # Preserve deterministic sorted unique order
    return sorted(set(figs))


def stage_single_paper(tectonic_path: Path, paper_key: str, max_overfull_pt: float) -> tuple[Path, str, list[str], int]:
    spec = PAPER_SPECS[paper_key]
    paper_dir: Path = spec["dir"]
    tex_src = paper_dir / spec["src"]

    preflight_errors = run_preflight_checks(paper_dir, tex_src)
    if preflight_errors:
        raise RuntimeError(f"{paper_key} failed preflight checks: {preflight_errors}")

    raw_tex = tex_src.read_text(encoding="utf-8")
    referenced_figs = collect_referenced_figures(raw_tex)

    with tempfile.TemporaryDirectory(prefix=f"lm_arxiv_build_{paper_key}_") as build_tmp:
        build_dir = Path(build_tmp)
        shutil.copy2(tex_src, build_dir / "main.tex")
        bib_src = paper_dir / "references.bib"
        if bib_src.exists():
            shutil.copy2(bib_src, build_dir / "references.bib")
        for fig_rel in referenced_figs:
            src_fig = paper_dir / fig_rel
            dst_fig = build_dir / fig_rel
            dst_fig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_fig, dst_fig)

        res = subprocess.run(
            [str(tectonic_path), "--keep-intermediates", "--keep-logs", "main.tex"],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0:
            raise RuntimeError(f"Initial Tectonic build failed for {paper_key}:\n{res.stdout}\n{res.stderr}")

        initial_report = AuditReport(paper_key=paper_key)
        parse_log((build_dir / "main.log").read_text(encoding="utf-8", errors="replace"), initial_report)
        if initial_report.fatal_errors or initial_report.undefined_warnings:
            raise RuntimeError(
                f"Initial build errors/undefined refs for {paper_key}: "
                f"fatal={initial_report.fatal_errors}, undef={initial_report.undefined_warnings}"
            )
        expected_pages = initial_report.pages

        bbl_path = build_dir / "main.bbl"
        if bbl_path.exists():
            staged_tex = inline_bbl_into_tex(raw_tex, bbl_path.read_text(encoding="utf-8"))
        else:
            staged_tex = raw_tex

    with tempfile.TemporaryDirectory(prefix=f"lm_arxiv_stage_{paper_key}_") as stage_tmp:
        stage_root = Path(stage_tmp) / "stage_root"
        stage_root.mkdir(parents=True, exist_ok=True)
        (stage_root / "main.tex").write_text(staged_tex, encoding="utf-8", newline="\n")

        for fig_rel in referenced_figs:
            src_fig = paper_dir / fig_rel
            dst_fig = stage_root / fig_rel
            dst_fig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_fig, dst_fig)

        DIST_ARXIV_DIR.mkdir(parents=True, exist_ok=True)
        tar_path = DIST_ARXIV_DIR / ARCHIVE_NAMES[paper_key]
        if tar_path.exists():
            tar_path.unlink()

        files_to_pack = sorted(p for p in stage_root.rglob("*") if p.is_file())
        with tarfile.open(tar_path, "w:gz", format=tarfile.PAX_FORMAT) as tf:
            for file_path in files_to_pack:
                arcname = file_path.relative_to(stage_root).as_posix()
                info = tf.gettarinfo(str(file_path), arcname=arcname)
                # Normalize deterministic metadata
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 1700000000
                with file_path.open("rb") as f:
                    tf.addfile(info, f)

    # Second isolated verification: extract tarball and self-compile
    with tempfile.TemporaryDirectory(prefix=f"lm_arxiv_verify_{paper_key}_") as verify_tmp:
        verify_dir = Path(verify_tmp)
        with tarfile.open(tar_path, "r:gz") as tf:
            members = tf.getnames()
            for member in members:
                if member.startswith("/") or ".." in member.split("/"):
                    raise RuntimeError(f"Unsafe archive entry: {member}")
                suffix = Path(member).suffix.lower()
                if suffix in FORBIDDEN_SUFFIXES:
                    raise RuntimeError(f"Forbidden file extension in tarball: {member}")
                if "/" not in member and suffix == ".pdf":
                    raise RuntimeError(f"Compiled root PDF must not be inside arXiv tarball: {member}")
            tf.extractall(verify_dir, filter="data")

        res2 = subprocess.run(
            [str(tectonic_path), "--keep-logs", "main.tex"],
            cwd=str(verify_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res2.returncode != 0:
            raise RuntimeError(f"Tarball self-compile failed for {paper_key}:\n{res2.stdout}\n{res2.stderr}")

        verify_report = AuditReport(paper_key=paper_key)
        verify_log = (verify_dir / "main.log").read_text(encoding="utf-8", errors="replace")
        parse_log(verify_log, verify_report)

        bad_overfull = [(pt, loc) for pt, loc in verify_report.overfull_hboxes if pt > max_overfull_pt]
        if verify_report.fatal_errors or verify_report.undefined_warnings or bad_overfull or verify_report.hyperref_warnings:
            raise RuntimeError(
                f"Tarball verification warnings for {paper_key}: "
                f"fatal={verify_report.fatal_errors}, undef={verify_report.undefined_warnings}, "
                f"overfull={bad_overfull}, hyperref={verify_report.hyperref_warnings}"
            )
        if expected_pages and verify_report.pages != expected_pages:
            raise RuntimeError(
                f"Page count mismatch for {paper_key}: initial={expected_pages}, tarball={verify_report.pages}"
            )
        out_pdf = verify_dir / "main.pdf"
        if not out_pdf.exists() or out_pdf.stat().st_size < 10_000:
            raise RuntimeError(f"Self-compiled PDF missing or unexpectedly small for {paper_key}")

    sha256 = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    return tar_path, sha256, members, verify_report.pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage and self-verify local arXiv source tarballs.")
    parser.add_argument(
        "--paper",
        choices=["all", "paper1", "paper2"],
        default="all",
        help="Which paper(s) to stage (default: all).",
    )
    parser.add_argument(
        "--max-overfull-pt",
        type=float,
        default=2.0,
        help="Maximum allowed Overfull \\hbox in pt during tarball self-compile check (default: 2.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tectonic_path = find_tectonic()
    keys = ["paper1", "paper2"] if args.paper == "all" else [args.paper]

    print("=" * 70, flush=True)
    print("LUCID MACHINES LOCAL ARXIV SOURCE STAGING (NO EXTERNAL UPLOAD)", flush=True)
    print(f"Compiler: {tectonic_path}", flush=True)
    print("=" * 70, flush=True)

    for key in keys:
        tar_path, sha256, members, pages = stage_single_paper(tectonic_path, key, args.max_overfull_pt)
        size_kb = tar_path.stat().st_size / 1024
        print(f"[STAGED & VERIFIED] {tar_path.relative_to(REPO_ROOT).as_posix()} ({size_kb:.1f} KB, {pages} pages)", flush=True)
        print(f"  SHA-256 : {sha256}", flush=True)
        print(f"  Entries : {', '.join(members)}", flush=True)

    print("=" * 70, flush=True)
    print("[SUCCESS] Local arXiv tarballs staged and verified cleanly.", flush=True)


if __name__ == "__main__":
    main()
