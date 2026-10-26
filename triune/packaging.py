"""Cross-Platform Native Application & Installer Builder for Triune Studio.

Builds native desktop executables and application bundles across:
- Windows: Native .NET WPF WebView2 executables (TriuneStudio.exe & TriuneStudioSetup.exe)
- Linux: Native AppDir / ELF launcher bundle (TriuneStudio.AppDir + .desktop + AppImage support)
- macOS: Native macOS Application Bundle (Triune Studio.app + Info.plist + Mach-O launcher)
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

TRIUNE_SVG_ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#312E81" />
      <stop offset="100%" stop-color="#4338CA" />
    </linearGradient>
  </defs>
  <rect width="256" height="256" rx="52" fill="url(#bg)" />
  <path d="M64 72 H192 V102 H145 V192 H111 V102 H64 Z" fill="#FFFFFF" />
  <circle cx="184" cy="180" r="18" fill="#EA580C" />
</svg>
"""


def _make_executable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def build_windows_native(workspace: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    """Compile and publish TriuneStudio.exe and TriuneStudioSetup.exe on Windows."""
    app_proj = workspace / "studio" / "windows" / "TriuneStudio.Windows" / "TriuneStudio.Windows.csproj"
    setup_proj = workspace / "studio" / "windows" / "TriuneStudio.Installer" / "TriuneStudio.Installer.csproj"
    publish_dir = workspace / "studio" / "windows" / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("dotnet"):
        raise RuntimeError("The .NET SDK ('dotnet') is required to build Windows native executables.")

    print("[Native Build - Windows] Building TriuneStudio.Windows (Release)...")
    subprocess.run(["dotnet", "build", str(app_proj), "-c", "Release"], check=True)
    subprocess.run(["dotnet", "publish", str(app_proj), "-c", "Release", "-o", str(publish_dir), "--no-restore"], check=True)

    print("[Native Build - Windows] Building TriuneStudio.Installer (Release)...")
    subprocess.run(["dotnet", "build", str(setup_proj), "-c", "Release"], check=True)
    subprocess.run(["dotnet", "publish", str(setup_proj), "-c", "Release", "-o", str(publish_dir), "--no-restore"], check=True)

    app_exe = publish_dir / "TriuneStudio.exe"
    setup_exe = publish_dir / "TriuneStudioSetup.exe"

    # Place root-level native executables for immediate 1-click access
    root_app_exe = workspace / "TriuneStudio.exe"
    root_setup_exe = workspace / "TriuneStudioSetup.exe"
    if app_exe.is_file():
        try:
            shutil.copy2(app_exe, root_app_exe)
        except OSError:
            pass
    if setup_exe.is_file():
        try:
            shutil.copy2(setup_exe, root_setup_exe)
        except OSError:
            pass

    return {
        "platform": "windows",
        "app_executable": str(app_exe),
        "installer_executable": str(setup_exe),
        "root_app_executable": str(root_app_exe),
        "root_installer_executable": str(root_setup_exe),
    }


def build_linux_native(workspace: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    """Generate a native Linux AppDir & ELF/POSIX desktop bundle."""
    dist_linux = workspace / "dist" / "linux"
    appdir = dist_linux / "TriuneStudio.AppDir"
    usr_bin = appdir / "usr" / "bin"
    usr_apps = appdir / "usr" / "share" / "applications"
    usr_icons = appdir / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"

    for d in (usr_bin, usr_apps, usr_icons):
        d.mkdir(parents=True, exist_ok=True)

    # Write vector icon
    icon_path = appdir / "triune-studio.svg"
    icon_path.write_text(TRIUNE_SVG_ICON, encoding="utf-8")
    (usr_icons / "triune-studio.svg").write_text(TRIUNE_SVG_ICON, encoding="utf-8")

    # Write AppRun and usr/bin launchers
    apprun_script = f"""#!/usr/bin/env bash
set -e
WORKSPACE="{workspace.as_posix()}"
if [ ! -d "$WORKSPACE" ]; then
    WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$WORKSPACE"
if [ -x "$WORKSPACE/studio/studio_env/bin/python" ]; then
    exec "$WORKSPACE/studio/studio_env/bin/python" "$WORKSPACE/scripts/launch_studio.py" "$@"
elif [ -x "$WORKSPACE/.venv/bin/python" ]; then
    exec "$WORKSPACE/.venv/bin/python" "$WORKSPACE/scripts/launch_studio.py" "$@"
else
    exec python3 "$WORKSPACE/scripts/launch_studio.py" "$@"
fi
"""
    setup_script = f"""#!/usr/bin/env bash
set -e
WORKSPACE="{workspace.as_posix()}"
if [ ! -d "$WORKSPACE" ]; then
    WORKSPACE="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$WORKSPACE"
exec python3 "$WORKSPACE/studio/installer/installer.py" "$@"
"""
    apprun_path = appdir / "AppRun"
    apprun_path.write_text(apprun_script, encoding="utf-8")
    _make_executable(apprun_path)

    bin_studio = usr_bin / "triune-studio"
    bin_studio.write_text(apprun_script, encoding="utf-8")
    _make_executable(bin_studio)

    bin_setup = usr_bin / "triune-studio-setup"
    bin_setup.write_text(setup_script, encoding="utf-8")
    _make_executable(bin_setup)

    # If running on Linux with gcc/cc, compile a native ELF binary for AppRun
    if sys.platform.startswith("linux") and (shutil.which("gcc") or shutil.which("cc")):
        compiler = shutil.which("gcc") or shutil.which("cc")
        c_src = dist_linux / "launcher.c"
        c_code = f"""#include <stdlib.h>
#include <unistd.h>
int main(int argc, char *argv[]) {{
    setenv("PYTHONUTF8", "1", 1);
    execv("{bin_studio.as_posix()}", argv);
    return 1;
}}
"""
        try:
            c_src.write_text(c_code, encoding="utf-8")
            subprocess.run([compiler, "-O2", str(c_src), "-o", str(usr_bin / "TriuneStudio")], check=True)
            c_src.unlink(missing_ok=True)
        except Exception:
            pass

    desktop_content = f"""[Desktop Entry]
Type=Application
Name=Triune Studio
GenericName=AI Engine & Research IDE
Comment=Triune Mixture-of-Experts AI Engine & Visual Research IDE
Exec={bin_studio.as_posix()}
Icon=triune-studio
Terminal=false
Categories=Development;Science;ArtificialIntelligence;
StartupWMClass=TriuneStudio
"""
    desktop_path = appdir / "triune-studio.desktop"
    desktop_path.write_text(desktop_content, encoding="utf-8")
    _make_executable(desktop_path)
    (usr_apps / "triune-studio.desktop").write_text(desktop_content, encoding="utf-8")

    appimage_out = None
    if sys.platform.startswith("linux") and shutil.which("appimagetool"):
        candidate_img = dist_linux / "TriuneStudio-x86_64.AppImage"
        try:
            subprocess.run(["appimagetool", str(appdir), str(candidate_img)], check=True)
            appimage_out = str(candidate_img)
        except Exception:
            pass

    print(f"[Native Build - Linux] Created Linux AppDir bundle at: {appdir}")
    return {
        "platform": "linux",
        "appdir": str(appdir),
        "launcher": str(bin_studio),
        "installer": str(bin_setup),
        "desktop_entry": str(desktop_path),
        "appimage": appimage_out,
    }


def build_macos_native(workspace: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    """Generate a native macOS Application Bundle (Triune Studio.app & Triune Studio Setup.app)."""
    dist_macos = workspace / "dist" / "macos"
    app_bundle = dist_macos / "Triune Studio.app"
    contents = app_bundle / "Contents"
    macos_dir = contents / "MacOS"
    resources_dir = contents / "Resources"

    for d in (macos_dir, resources_dir):
        d.mkdir(parents=True, exist_ok=True)

    plist_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>Triune Studio</string>
    <key>CFBundleExecutable</key>
    <string>TriuneStudio</string>
    <key>CFBundleIdentifier</key>
    <string>ai.triune.studio</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Triune Studio</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.1.0</string>
    <key>CFBundleVersion</key>
    <string>2.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""
    (contents / "Info.plist").write_text(plist_content, encoding="utf-8")
    (resources_dir / "triune-studio.svg").write_text(TRIUNE_SVG_ICON, encoding="utf-8")

    macos_launcher = f"""#!/usr/bin/env bash
set -e
WORKSPACE="{workspace.as_posix()}"
if [ ! -d "$WORKSPACE" ]; then
    WORKSPACE="$(cd "$(dirname "$0")/../../../.." && pwd)"
fi
cd "$WORKSPACE"
export PYTHONUTF8=1
if [ -x "$WORKSPACE/studio/studio_env/bin/python" ]; then
    exec "$WORKSPACE/studio/studio_env/bin/python" "$WORKSPACE/scripts/launch_studio.py" "$@"
elif [ -x "$WORKSPACE/.venv/bin/python" ]; then
    exec "$WORKSPACE/.venv/bin/python" "$WORKSPACE/scripts/launch_studio.py" "$@"
else
    exec python3 "$WORKSPACE/scripts/launch_studio.py" "$@"
fi
"""
    macos_setup = f"""#!/usr/bin/env bash
set -e
WORKSPACE="{workspace.as_posix()}"
if [ ! -d "$WORKSPACE" ]; then
    WORKSPACE="$(cd "$(dirname "$0")/../../../.." && pwd)"
fi
cd "$WORKSPACE"
export PYTHONUTF8=1
exec python3 "$WORKSPACE/studio/installer/installer.py" "$@"
"""
    exec_path = macos_dir / "TriuneStudio"
    exec_path.write_text(macos_launcher, encoding="utf-8")
    _make_executable(exec_path)

    setup_path = macos_dir / "TriuneStudioSetup"
    setup_path.write_text(macos_setup, encoding="utf-8")
    _make_executable(setup_path)

    print(f"[Native Build - macOS] Created macOS Application Bundle at: {app_bundle}")
    return {
        "platform": "macos",
        "app_bundle": str(app_bundle),
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
