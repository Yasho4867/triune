"""Cross-Platform Native Application & Installer Builder for Polysom Studio (Lucid Machines).

Builds native desktop executables and application bundles across:
- Windows: Native .NET WPF WebView2 executables (PolysomStudio.exe & PolysomStudioSetup.exe)
- Linux: Native AppDir / ELF launcher bundle (PolysomStudio.AppDir + .desktop + AppImage support)
- macOS: Native macOS Application Bundle (Polysom Studio.app + Info.plist + Mach-O launcher)
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

POLYSOM_SVG_ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <rect width="256" height="256" rx="48" fill="#0E0D17" stroke="#2D294E" stroke-width="6" />
  <path d="M 60 178 L 128 54 L 196 178" fill="none" stroke="#8B7CFF" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" />
  <path d="M 82 141 L 174 141" fill="none" stroke="#38BDF8" stroke-width="7" stroke-linecap="round" />
  <path d="M 48 128 L 84 128 L 98 102 L 114 156 L 128 88 L 143 156 L 159 110 L 173 128 L 208 128" fill="none" stroke="#10B981" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" />
  <circle cx="128" cy="54" r="10" fill="#0E0D17" stroke="#8B7CFF" stroke-width="4" />
</svg>
"""


def _make_executable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def build_windows_native(workspace: Path = WORKSPACE_ROOT) -> Dict[str, Any]:
    """Compile and publish PolysomStudio.exe and PolysomStudioSetup.exe on Windows."""
    app_proj = workspace / "studio" / "windows" / "TriuneStudio.Windows" / "TriuneStudio.Windows.csproj"
    setup_proj = workspace / "studio" / "windows" / "TriuneStudio.Installer" / "TriuneStudio.Installer.csproj"
    publish_dir = workspace / "studio" / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("dotnet"):
        raise RuntimeError("The .NET SDK ('dotnet') is required to build Windows native executables.")

    print("[Native Build - Windows] Building Polysom Studio (Release)...")
    subprocess.run(["dotnet", "build", str(app_proj), "-c", "Release"], check=True)
    subprocess.run(
        ["dotnet", "publish", str(app_proj), "-c", "Release", "-o", str(publish_dir), "--no-restore"], check=True
    )

    print("[Native Build - Windows] Building Polysom Studio Setup (Release)...")
    subprocess.run(["dotnet", "build", str(setup_proj), "-c", "Release"], check=True)
    subprocess.run(
        ["dotnet", "publish", str(setup_proj), "-c", "Release", "-o", str(publish_dir), "--no-restore"], check=True
    )

    app_exe = publish_dir / "PolysomStudio.exe"
    setup_exe = publish_dir / "PolysomStudioSetup.exe"

    # Place root-level native executables for immediate 1-click access
    root_app_exe = workspace / "PolysomStudio.exe"
    root_setup_exe = workspace / "PolysomStudioSetup.exe"
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
    appdir = dist_linux / "PolysomStudio.AppDir"
    usr_bin = appdir / "usr" / "bin"
    usr_apps = appdir / "usr" / "share" / "applications"
    usr_icons = appdir / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"

    for d in (usr_bin, usr_apps, usr_icons):
        d.mkdir(parents=True, exist_ok=True)

    # Write vector icon
    icon_path = appdir / "polysom-studio.svg"
    icon_path.write_text(POLYSOM_SVG_ICON, encoding="utf-8")
    (usr_icons / "polysom-studio.svg").write_text(POLYSOM_SVG_ICON, encoding="utf-8")

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

    bin_studio = usr_bin / "polysom-studio"
    bin_studio.write_text(apprun_script, encoding="utf-8")
    _make_executable(bin_studio)

    bin_setup = usr_bin / "polysom-studio-setup"
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
            subprocess.run([compiler, "-O2", str(c_src), "-o", str(usr_bin / "PolysomStudio")], check=True)
            c_src.unlink(missing_ok=True)
        except Exception:
            pass

    desktop_content = f"""[Desktop Entry]
Type=Application
Name=Polysom Studio
GenericName=Lucid Machines AI Engine & Research IDE
Comment=Polysom Studio — Triune Mixture-of-Experts AI Engine & Visual Research IDE by Lucid Machines
Exec={bin_studio.as_posix()}
Icon=polysom-studio
Terminal=false
Categories=Development;Science;ArtificialIntelligence;
StartupWMClass=PolysomStudio
"""
    desktop_path = appdir / "polysom-studio.desktop"
    desktop_path.write_text(desktop_content, encoding="utf-8")
    _make_executable(desktop_path)
    (usr_apps / "polysom-studio.desktop").write_text(desktop_content, encoding="utf-8")

    appimage_out = None
    if sys.platform.startswith("linux") and shutil.which("appimagetool"):
        candidate_img = dist_linux / "PolysomStudio-x86_64.AppImage"
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
    """Generate a native macOS Application Bundle (Polysom Studio.app)."""
    dist_macos = workspace / "dist" / "macos"
    app_bundle = dist_macos / "Polysom Studio.app"
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
    <string>Polysom Studio</string>
    <key>CFBundleExecutable</key>
    <string>PolysomStudio</string>
    <key>CFBundleIdentifier</key>
    <string>ai.lucidmachines.polysom</string>
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
    (contents / "Info.plist").write_text(plist_content, encoding="utf-8")
    (resources_dir / "polysom-studio.svg").write_text(POLYSOM_SVG_ICON, encoding="utf-8")

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
    exec_path = macos_dir / "PolysomStudio"
    exec_path.write_text(macos_launcher, encoding="utf-8")
    _make_executable(exec_path)

    setup_path = macos_dir / "PolysomStudioSetup"
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
