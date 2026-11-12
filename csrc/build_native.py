#!/usr/bin/env python3
"""Cross-Platform Native C ABI Shared Library Builder for Triune AI.

Discovers host C/C++ toolchains (MSVC, GCC, Clang, Apple Clang), builds
`triune_core.dll` (Windows), `libtriune_core.so` (Linux), or `libtriune_core.dylib` (macOS),
and places artifacts in `triune/ops/` and `studio/windows/TriuneStudio.Windows/`.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

CSRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = CSRC_DIR.parent
OPS_DIR = REPO_ROOT / "triune" / "ops"
STUDIO_WINDOWS_DIR = REPO_ROOT / "studio" / "windows" / "TriuneStudio.Windows"


def find_vswhere() -> Path | None:
    """Locate vswhere.exe to find installed Visual Studio instances."""
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        return vswhere
    return None


def find_vcvars64() -> Path | None:
    """Find vcvars64.bat across Visual Studio 2022, 2019, 2017 or Build Tools."""
    vswhere = find_vswhere()
    if vswhere:
        try:
            cmd = [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            vs_path = res.stdout.strip()
            if vs_path:
                vcvars = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if vcvars.is_file():
                    return vcvars
        except Exception:
            pass

    # Common search paths
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p
    return None


def build_windows_msvc(vcvars64: Path | None = None) -> bool:
    """Build triune_core.dll using MSVC cl.exe."""
    engine_cpp = CSRC_DIR / "engine" / "engine.cpp"
    def_file = CSRC_DIR / "engine" / "kernel32.def"
    lib_file = CSRC_DIR / "engine" / "kernel32.lib"
    dll_out = OPS_DIR / "triune_core.dll"
    obj_out = CSRC_DIR / "engine" / "engine.obj"

    OPS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if cl is already in PATH
    cl_in_path = shutil.which("cl.exe") is not None
    if not cl_in_path:
        vcvars64 = vcvars64 or find_vcvars64()
        if not vcvars64:
            print("[build_native] MSVC compiler (cl.exe) or vcvars64.bat not found.")
            return False

    batch_commands = []
    if not cl_in_path and vcvars64:
        batch_commands.append(f'call "{vcvars64}"')

    batch_commands.extend([
        f'lib.exe /NOLOGO /DEF:"{def_file}" /OUT:"{lib_file}" /MACHINE:X64',
        f'cl.exe /nologo /O2 /Oi /Ot /arch:AVX2 /GS- /LD "{engine_cpp}" /Fe:"{dll_out}" /Fo:"{obj_out}" /link /NODEFAULTLIB /ENTRY:DllMain "{lib_file}" libvcruntime.lib',
    ])

    cmd_script = "\n".join(batch_commands)
    temp_bat = CSRC_DIR / "_temp_build.bat"
    temp_bat.write_text(cmd_script, encoding="utf-8")

    try:
        res = subprocess.run(["cmd.exe", "/c", str(temp_bat)], cwd=str(REPO_ROOT), capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            return False
    finally:
        if temp_bat.is_file():
            temp_bat.unlink(missing_ok=True)
        for junk in [lib_file, obj_out, CSRC_DIR / "engine" / "kernel32.exp", OPS_DIR / "triune_core.exp", OPS_DIR / "triune_core.lib"]:
            if junk.is_file():
                junk.unlink(missing_ok=True)

    # Copy to studio windows directory if present
    if STUDIO_WINDOWS_DIR.is_dir() and dll_out.is_file():
        try:
            shutil.copy2(dll_out, STUDIO_WINDOWS_DIR / "triune_core.dll")
        except OSError:
            pass

    print(f"[build_native] Successfully built Windows C ABI engine: {dll_out}")
    return True


def build_posix(compiler: str, target_so: Path) -> bool:
    """Build libtriune_core.so / libtriune_core.dylib with GCC or Clang."""
    engine_cpp = CSRC_DIR / "engine" / "engine.cpp"
    include_dir = CSRC_DIR / "engine"

    OPS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        "-mavx2",
        "-mfma",
        f"-I{include_dir}",
        str(engine_cpp),
        "-o",
        str(target_so),
    ]
    if platform.system() != "Darwin":
        cmd.extend(["-ldl", "-lpthread"])

    try:
        print(f"[build_native] Compiling with {compiler}: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        if res.stdout:
            print(res.stdout)
        print(f"[build_native] Successfully built POSIX C ABI engine: {target_so}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[build_native] Compilation failed:\n{e.stderr}", file=sys.stderr)
        return False


def build() -> bool:
    """Build native shared library for the current platform."""
    sys_name = platform.system()
    if sys_name == "Windows":
        return build_windows_msvc()
    elif sys_name == "Darwin":
        compiler = shutil.which("clang++") or shutil.which("clang") or shutil.which("g++") or "g++"
        target = OPS_DIR / "libtriune_core.dylib"
        return build_posix(compiler, target)
    else:  # Linux / POSIX
        compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("gcc") or "g++"
        target = OPS_DIR / "libtriune_core.so"
        return build_posix(compiler, target)


def main():
    parser = argparse.ArgumentParser(description="Triune AI Cross-Platform Native Shared Library Builder")
    parser.add_argument("--clean", action="store_true", help="Clean previously compiled shared libraries")
    args = parser.parse_args()

    if args.clean:
        for f in [OPS_DIR / "triune_core.dll", OPS_DIR / "libtriune_core.so", OPS_DIR / "libtriune_core.dylib", STUDIO_WINDOWS_DIR / "triune_core.dll"]:
            if f.is_file():
                f.unlink()
                print(f"[build_native] Removed {f}")
        return

    success = build()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
