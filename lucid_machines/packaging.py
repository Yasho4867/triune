"""Cross-Platform Native Application & Installer Builder for Polysom Studio (Lucid Machines).

Builds native desktop executables and application bundles across:
- Windows: Native .NET WPF WebView2 structured app/ bundle (PolysomStudio.exe & PolysomStudioSetup.exe)
- Linux: Portable AppDir / ELF launcher bundle (PolysomStudio.AppDir + .desktop + AppImage support)
- macOS: Portable macOS Application Bundle (Polysom Studio.app + Info.plist + bundled runtime)
Support: contact@lucidmachines.co | https://lucidmachines.co
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

POLYSOM_SVG_ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="256" height="256">
  <rect width="48" height="48" rx="8" fill="#f4ecda"/>
  <path d="M3 25C13 10 35 10 45 25 35 40 13 40 3 25Z" fill="none" stroke="#7b3f3c" stroke-width="1.5"/>
  <path d="m8 17-3-4m11-1-2-5m10 3V4m8 8 2-5m6 10 3-4M14 37l-2 4m12-3v6m10-7 2 4" stroke="#7b3f3c" stroke-width="1.2"/>
  <ellipse cx="24" cy="25" rx="8.5" ry="10.5" fill="#7b3f3c"/>
  <ellipse cx="24" cy="25" rx="3" ry="5.5" fill="#f4ecda"/>
  <circle cx="27" cy="21" r="1.8" fill="#f4ecda"/>
</svg>
"""


def _make_executable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _bundle_workspace_runtime(bundle_root: Path, workspace: Path) -> None:
    """Copy portable Python/Studio runtime payload into bundle_root without build artifacts or venvs."""
    bundle_root.mkdir(parents=True, exist_ok=True)
    ignore_patterns = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "studio_env", ".venv", ".git")

    for dir_rel in ("lucid_machines", "studio/src", "studio/installer", "models", "docs"):
        src_dir = workspace / dir_rel
        dst_dir = bundle_root / dir_rel
        if src_dir.is_dir():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir, ignore=ignore_patterns)

    for file_rel in (
        "pyproject.toml",
        "README.md",
        "requirements.txt",
        "scripts/launch_studio.py",
        "scripts/benchmark_inference.py",
    ):
        src_file = workspace / file_rel
        if src_file.is_file():
            dst_file = bundle_root / file_rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)


def build_windows_native(workspace: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    """Compile and publish structured studio/publish/app/ bundle on Windows."""
    app_proj = workspace / "studio" / "windows" / "PolysomStudio.Windows" / "PolysomStudio.Windows.csproj"
    setup_proj = workspace / "studio" / "windows" / "PolysomStudio.Installer" / "PolysomStudio.Installer.csproj"
    publish_dir = workspace / "studio" / "publish" / "app"
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    publish_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("dotnet"):
        raise RuntimeError("The .NET SDK ('dotnet') is required to build Windows native executables.")

    print("[Native Build - Windows] Publishing Polysom Studio (structured app/)...")
    subprocess.run(
        [
            "dotnet",
            "publish",
            str(app_proj),
            "-c",
            "Release",
            "-r",
            "win-x64",
            "--self-contained",
            "true",
            "-p:PublishSingleFile=false",
            "-o",
            str(publish_dir),
        ],
        check=True,
    )

    print("[Native Build - Windows] Publishing Polysom Studio Setup (structured app/)...")
    subprocess.run(
        [
            "dotnet",
            "publish",
            str(setup_proj),
            "-c",
            "Release",
            "-r",
            "win-x64",
            "--self-contained",
            "true",
            "-p:PublishSingleFile=false",
            "-o",
            str(publish_dir),
        ],
        check=True,
    )

    app_exe = publish_dir / "PolysomStudio.exe"
    setup_exe = publish_dir / "PolysomStudioSetup.exe"

    return {
        "platform": "windows",
        "app_dir": str(publish_dir),
        "app_executable": str(app_exe),
        "installer_executable": str(setup_exe),
    }


def build_linux_native(workspace: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    """Generate a portable native Linux AppDir & ELF/POSIX desktop bundle."""
    dist_linux = workspace / "dist" / "linux"
    appdir = dist_linux / "PolysomStudio.AppDir"
    usr_bin = appdir / "usr" / "bin"
    usr_share_app = appdir / "usr" / "share" / "polysom-studio"
    usr_apps = appdir / "usr" / "share" / "applications"
    usr_icons = appdir / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"

    for d in (usr_bin, usr_share_app, usr_apps, usr_icons):
        d.mkdir(parents=True, exist_ok=True)

    _bundle_workspace_runtime(usr_share_app, workspace)

    # Write vector icon
    icon_path = appdir / "polysom-studio.svg"
    icon_path.write_text(POLYSOM_SVG_ICON, encoding="utf-8", newline="\n")
    (usr_icons / "polysom-studio.svg").write_text(POLYSOM_SVG_ICON, encoding="utf-8", newline="\n")

    # Write portable AppRun and usr/bin launchers (zero hardcoded build paths)
    apprun_script = """#!/usr/bin/env bash
set -e
SELF_PATH="$(readlink -f "$0" 2>/dev/null || printf "%s" "$0")"
APPDIR="${APPDIR:-$(cd "$(dirname "$SELF_PATH")" && pwd)}"
exec "$APPDIR/usr/bin/polysom-studio" "$@"
"""
    bin_studio_script = """#!/usr/bin/env bash
set -e
SELF_PATH="$(readlink -f "$0" 2>/dev/null || printf "%s" "$0")"
SELF_DIR="$(cd "$(dirname "$SELF_PATH")" && pwd)"
APPDIR="${APPDIR:-$(cd "$SELF_DIR/../.." && pwd)}"
BUNDLE_ROOT="$APPDIR/usr/share/polysom-studio"
if [ ! -d "$BUNDLE_ROOT/lucid_machines" ] && [ -n "${POLYSOM_WORKSPACE:-}" ] && [ -d "$POLYSOM_WORKSPACE/lucid_machines" ]; then
    BUNDLE_ROOT="$POLYSOM_WORKSPACE"
fi
export PYTHONUTF8=1
export PYTHONPATH="$BUNDLE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$BUNDLE_ROOT"
if [ -x "$BUNDLE_ROOT/studio/studio_env/bin/python" ]; then
    exec "$BUNDLE_ROOT/studio/studio_env/bin/python" "$BUNDLE_ROOT/scripts/launch_studio.py" "$@"
elif [ -x "$BUNDLE_ROOT/.venv/bin/python" ]; then
    exec "$BUNDLE_ROOT/.venv/bin/python" "$BUNDLE_ROOT/scripts/launch_studio.py" "$@"
else
    exec python3 "$BUNDLE_ROOT/scripts/launch_studio.py" "$@"
fi
"""
    setup_script = """#!/usr/bin/env bash
set -e
SELF_PATH="$(readlink -f "$0" 2>/dev/null || printf "%s" "$0")"
SELF_DIR="$(cd "$(dirname "$SELF_PATH")" && pwd)"
APPDIR="${APPDIR:-$(cd "$SELF_DIR/../.." && pwd)}"
BUNDLE_ROOT="$APPDIR/usr/share/polysom-studio"
if [ ! -d "$BUNDLE_ROOT/lucid_machines" ] && [ -n "${POLYSOM_WORKSPACE:-}" ] && [ -d "$POLYSOM_WORKSPACE/lucid_machines" ]; then
    BUNDLE_ROOT="$POLYSOM_WORKSPACE"
fi
export PYTHONUTF8=1
export PYTHONPATH="$BUNDLE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$BUNDLE_ROOT"
if [ -x "$BUNDLE_ROOT/studio/studio_env/bin/python" ]; then
    exec "$BUNDLE_ROOT/studio/studio_env/bin/python" "$BUNDLE_ROOT/studio/installer/installer.py" "$@"
elif [ -x "$BUNDLE_ROOT/.venv/bin/python" ]; then
    exec "$BUNDLE_ROOT/.venv/bin/python" "$BUNDLE_ROOT/studio/installer/installer.py" "$@"
else
    exec python3 "$BUNDLE_ROOT/studio/installer/installer.py" "$@"
fi
"""
    apprun_path = appdir / "AppRun"
    apprun_path.write_text(apprun_script, encoding="utf-8", newline="\n")
    _make_executable(apprun_path)

    bin_studio = usr_bin / "polysom-studio"
    bin_studio.write_text(bin_studio_script, encoding="utf-8", newline="\n")
    _make_executable(bin_studio)

    bin_setup = usr_bin / "polysom-studio-setup"
    bin_setup.write_text(setup_script, encoding="utf-8", newline="\n")
    _make_executable(bin_setup)

    # If running on Linux with gcc/cc, compile a portable native ELF binary
    if sys.platform.startswith("linux") and (shutil.which("gcc") or shutil.which("cc")):
        compiler = shutil.which("gcc") or shutil.which("cc")
        c_src = dist_linux / "launcher.c"
        c_code = """#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    char exe_path[PATH_MAX];
    char target[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len <= 0) {
        return 1;
    }
    exe_path[len] = '\\0';
    char *dir = dirname(exe_path);
    if (snprintf(target, sizeof(target), "%s/polysom-studio", dir) >= (int)sizeof(target)) {
        return 1;
    }
    setenv("PYTHONUTF8", "1", 1);
    execv(target, argv);
    return 1;
}
"""
        try:
            c_src.write_text(c_code, encoding="utf-8", newline="\n")
            subprocess.run([compiler, "-O2", str(c_src), "-o", str(usr_bin / "PolysomStudio")], check=True)
            c_src.unlink(missing_ok=True)
        except Exception:
            pass

    desktop_content = """[Desktop Entry]
Type=Application
Name=Polysom Studio
GenericName=Lucid Machines AI Engine & Research IDE
Comment=Polysom Studio — Triune Mixture-of-Experts AI Engine & Visual Research IDE by Lucid Machines
Exec=polysom-studio
Icon=polysom-studio
Terminal=false
Categories=Development;Science;ArtificialIntelligence;
StartupWMClass=PolysomStudio
"""
    desktop_path = appdir / "polysom-studio.desktop"
    desktop_path.write_text(desktop_content, encoding="utf-8", newline="\n")
    _make_executable(desktop_path)
    (usr_apps / "polysom-studio.desktop").write_text(desktop_content, encoding="utf-8", newline="\n")

    appimage_out = None
    if sys.platform.startswith("linux") and shutil.which("appimagetool"):
        candidate_img = dist_linux / "PolysomStudio-x86_64.AppImage"
        try:
            subprocess.run(["appimagetool", str(appdir), str(candidate_img)], check=True)
            appimage_out = str(candidate_img)
        except Exception:
            pass

    print(f"[Native Build - Linux] Created portable Linux AppDir bundle at: {appdir}")
    return {
        "platform": "linux",
        "appdir": str(appdir),
        "bundle_root": str(usr_share_app),
        "launcher": str(bin_studio),
        "installer": str(bin_setup),
        "desktop_entry": str(desktop_path),
        "appimage": appimage_out,
    }


def build_macos_native(workspace: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    """Generate a portable native macOS Application Bundle (Polysom Studio.app)."""
    dist_macos = workspace / "dist" / "macos"
    app_bundle = dist_macos / "Polysom Studio.app"
    contents = app_bundle / "Contents"
    macos_dir = contents / "MacOS"
    resources_dir = contents / "Resources"
    bundle_app_dir = resources_dir / "app"

    for d in (macos_dir, resources_dir, bundle_app_dir):
        d.mkdir(parents=True, exist_ok=True)

    _bundle_workspace_runtime(bundle_app_dir, workspace)

    plist_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>Polysom Studio</string>
    <key>CFBundleExecutable</key>
    <string>PolysomStudio</string>
    <key>CFBundleIconFile</key>
    <string>polysom-studio.svg</string>
    <key>CFBundleIdentifier</key>
    <string>dev.lucidmachines.polysom</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Polysom Studio</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""
    (contents / "Info.plist").write_text(plist_content, encoding="utf-8", newline="\n")
    (resources_dir / "polysom-studio.svg").write_text(POLYSOM_SVG_ICON, encoding="utf-8", newline="\n")

    resolve_bundle_preamble = """#!/usr/bin/env bash
set -e
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    case "$SOURCE" in
        /*) ;;
        *) SOURCE="$DIR/$SOURCE" ;;
    esac
done
MACOS_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
CONTENTS_DIR="$(cd -P "$MACOS_DIR/.." && pwd)"
BUNDLE_ROOT="$CONTENTS_DIR/Resources/app"
if [ ! -d "$BUNDLE_ROOT/lucid_machines" ] && [ -n "${POLYSOM_WORKSPACE:-}" ] && [ -d "$POLYSOM_WORKSPACE/lucid_machines" ]; then
    BUNDLE_ROOT="$POLYSOM_WORKSPACE"
fi
export PYTHONUTF8=1
export PYTHONPATH="$BUNDLE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$BUNDLE_ROOT"
"""
    macos_launcher = resolve_bundle_preamble + """if [ -x "$BUNDLE_ROOT/studio/studio_env/bin/python" ]; then
    exec "$BUNDLE_ROOT/studio/studio_env/bin/python" "$BUNDLE_ROOT/scripts/launch_studio.py" "$@"
elif [ -x "$BUNDLE_ROOT/.venv/bin/python" ]; then
    exec "$BUNDLE_ROOT/.venv/bin/python" "$BUNDLE_ROOT/scripts/launch_studio.py" "$@"
else
    exec python3 "$BUNDLE_ROOT/scripts/launch_studio.py" "$@"
fi
"""
    macos_setup = resolve_bundle_preamble + """if [ -x "$BUNDLE_ROOT/studio/studio_env/bin/python" ]; then
    exec "$BUNDLE_ROOT/studio/studio_env/bin/python" "$BUNDLE_ROOT/studio/installer/installer.py" "$@"
elif [ -x "$BUNDLE_ROOT/.venv/bin/python" ]; then
    exec "$BUNDLE_ROOT/.venv/bin/python" "$BUNDLE_ROOT/studio/installer/installer.py" "$@"
else
    exec python3 "$BUNDLE_ROOT/studio/installer/installer.py" "$@"
fi
"""
    exec_path = macos_dir / "PolysomStudio"
    exec_path.write_text(macos_launcher, encoding="utf-8", newline="\n")
    _make_executable(exec_path)

    setup_path = macos_dir / "PolysomStudioSetup"
    setup_path.write_text(macos_setup, encoding="utf-8", newline="\n")
    _make_executable(setup_path)

    print(f"[Native Build - macOS] Created portable macOS Application Bundle at: {app_bundle}")
    return {
        "platform": "macos",
        "app_bundle": str(app_bundle),
        "bundle_root": str(bundle_app_dir),
        "executable": str(exec_path),
        "installer": str(setup_path),
    }


def build_native_targets(target: str = "all", workspace: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    """Build native desktop applications and installers for the requested target(s)."""
    results: Dict[str, Any] = {}
    target_norm = target.lower().strip()

    if target_norm in ("windows", "win", "all"):
        if sys.platform == "win32" or shutil.which("dotnet"):
            results["windows"] = build_windows_native(workspace)
    if target_norm in ("linux", "all"):
        results["linux"] = build_linux_native(workspace)
    if target_norm in ("macos", "mac", "darwin", "all"):
        results["macos"] = build_macos_native(workspace)

    return results
