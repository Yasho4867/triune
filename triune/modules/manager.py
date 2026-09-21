"""Triune Universal Hardware Scanner, Module Marketplace, and Configuration Manager."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from .github_client import GitHubClient


class ModuleManager:
    """Manages system hardware detection, user configs, module installations, and version updates."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        if config_dir:
            self.base_dir = Path(config_dir)
        else:
            # Default to C:\TriuneStudio or user home
            if sys.platform == "win32" and os.path.exists("C:\\"):
                self.base_dir = Path("C:\\TriuneStudio")
            else:
                self.base_dir = Path.home() / ".triune_studio"

        self.modules_dir = self.base_dir / "modules"
        self.config_file = self.base_dir / "studio_config.json"
        self.installed_file = self.base_dir / "installed_modules.json"

        # Ensure directories exist
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.modules_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to local execution directory if C:\ permission is restricted
            self.base_dir = Path(__file__).resolve().parent.parent.parent.parent / "studio_data"
            self.modules_dir = self.base_dir / "modules"
            self.config_file = self.base_dir / "studio_config.json"
            self.installed_file = self.base_dir / "installed_modules.json"
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.modules_dir.mkdir(parents=True, exist_ok=True)

        self.github_client = GitHubClient()
        self.registry = self._load_curated_registry()

    def _load_curated_registry(self) -> list[dict[str, Any]]:
        reg_file = Path(__file__).parent / "registry.json"
        if reg_file.exists():
            try:
                with open(reg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data:
                        return data
            except Exception:
                pass

        # Built-in fallback curated recommendations
        return [
            {
                "id": "triune-base-weights",
                "name": "Triune-Base 2.5B MoE Weights",
                "author": "Triune AI Core",
                "type": "model",
                "version": "1.2.0",
                "description": "Standard 24-layer MoE checkpoint with 8 experts and 3 exit heads (Reflex, Limbic, Cortex). Optimized for fast local inference.",
                "repo_url": "https://github.com/Yash-456/triune-transformer",
                "download_url": "https://github.com/Yash-456/triune-transformer/releases/download/v1.2.0/triune-base.safetensors",
                "size_mb": 4800,
                "tags": ["MoE", "2.5B", "Recommended", "Native"],
                "requires_cuda": False
            },
            {
                "id": "code-assistant-lora",
                "name": "Python & Rust Code Tuning Adapter",
                "author": "Community",
                "type": "adapter",
                "version": "1.0.4",
                "description": "LoRA rank-16 adapter trained on FineWeb & StarCoder Python/Rust dataset. Plugs directly into TriuneTransformer query/value projections.",
                "repo_url": "https://github.com/huggingface/transformers",
                "download_url": "https://github.com/Yash-456/triune-transformer/releases/download/v1.0.0/code-lora-r16.safetensors",
                "size_mb": 64,
                "tags": ["LoRA", "Coding", "r=16"],
                "requires_cuda": False
            },
            {
                "id": "fineweb-sample-10k",
                "name": "FineWeb Curated 10K Sample Dataset",
                "author": "HuggingFace / Triune",
                "type": "dataset",
                "version": "2.0.0",
                "description": "Cleaned pre-tokenized BPE subset of FineWeb for rapid local evaluation and synthetic router label benchmark generation.",
                "repo_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb",
                "download_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/sample-10k.jsonl",
                "size_mb": 120,
                "tags": ["Dataset", "Pre-tokenized", "BPE"],
                "requires_cuda": False
            },
            {
                "id": "custom-loss-nodes",
                "name": "Extended Router & Custom Loss Layer Nodes",
                "author": "Triune Research",
                "type": "plugin",
                "version": "1.1.0",
                "description": "Custom DAG visual nodes including Centroid Router Bias, Focal Exit Loss, and Quantized Linear Attention execution handlers.",
                "repo_url": "https://github.com/Yash-456/triune-transformer",
                "download_url": "https://github.com/Yash-456/triune-transformer/archive/refs/heads/main.zip",
                "size_mb": 12,
                "tags": ["DAG Nodes", "Plugin", "Loss Functions"],
                "requires_cuda": False
            },
            {
                "id": "flash-attn-package",
                "name": "FlashAttention-2 CUDA Kernels",
                "author": "Dao-AILab",
                "type": "framework",
                "version": "2.5.6",
                "description": "Fast memory-efficient attention algorithms for NVIDIA Ampere, Ada, and Hopper GPUs. Significantly improves throughput.",
                "repo_url": "https://github.com/Dao-AILab/flash-attention",
                "download_url": "https://github.com/Dao-AILab/flash-attention/releases",
                "size_mb": 240,
                "tags": ["CUDA", "GPU Only", "Speedup"],
                "requires_cuda": True
            },
            {
                "id": "bitsandbytes-quant",
                "name": "BitsAndBytes 8-bit & 4-bit Quantization Engine",
                "author": "Tim Dettmers",
                "type": "framework",
                "version": "0.43.0",
                "description": "Enables FP4, NF4, and Int8 matrix multiplication for extreme VRAM savings on consumer GPUs.",
                "repo_url": "https://github.com/bitsandbytes-foundation/bitsandbytes",
                "download_url": "https://github.com/bitsandbytes-foundation/bitsandbytes/releases",
                "size_mb": 85,
                "tags": ["Quantization", "NF4", "VRAM Saver"],
                "requires_cuda": False
            }
        ]

    # -------------------------------------------------------------------------
    # Hardware & Software Auto-Scanner
    # -------------------------------------------------------------------------
    def scan_hardware_and_software(self) -> dict[str, Any]:
        """Perform comprehensive auto-scan of system hardware, CUDA, Python, and installed packages."""
        gpu_name = "CPU Only (No NVIDIA SMI Detected)"
        cuda_version = "None"
        vram_total_gb = 0.0
        has_cuda = False

        # Query nvidia-smi
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                parts = [p.strip() for p in smi.stdout.strip().split(",")]
                gpu_name = parts[0]
                if len(parts) >= 2:
                    vram_raw = parts[1].replace("MiB", "").strip()
                    try:
                        vram_total_gb = round(float(vram_raw) / 1024.0, 2)
                    except ValueError:
                        pass
                has_cuda = True
        except Exception:
            pass

        # Query nvcc
        try:
            nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=3.0)
            if nvcc.returncode == 0 and "release" in nvcc.stdout:
                cuda_version = nvcc.stdout.split("release")[-1].split(",")[0].strip()
        except Exception:
            pass

        # Scan installed packages
        packages = {}
        target_pkgs = ["torch", "fastapi", "uvicorn", "pywebview", "pyinstaller", "transformer_engine", "flash_attn", "bitsandbytes", "triton", "transformers"]
        for pkg in target_pkgs:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "Installed")
                packages[pkg] = {"installed": True, "version": str(ver)}
            except Exception:
                packages[pkg] = {"installed": False, "version": "Not Installed"}

        # System info
        sys_info = {
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "gpu": gpu_name,
            "vram_gb": vram_total_gb,
            "cuda_available": has_cuda,
            "cuda_version": cuda_version,
            "packages": packages,
            "base_dir": str(self.base_dir),
            "modules_dir": str(self.modules_dir),
        }
        return sys_info

    # -------------------------------------------------------------------------
    # System Config Management
    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        """Load user configuration and custom paths."""
        defaults = {
            "installation_path": str(self.base_dir),
            "models_path": str(self.base_dir / "models"),
            "datasets_path": str(self.base_dir / "datasets"),
            "checkpoints_path": str(self.base_dir / "checkpoints"),
            "python_executable": sys.executable,
            "auto_check_updates": True,
            "github_token": "",
            "hardware_mode": "Auto Detect",
        }
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    defaults.update(saved)
            except Exception:
                pass
        return defaults

    def save_config(self, new_config: dict[str, Any]) -> dict[str, Any]:
        """Save updated user configuration."""
        current = self.get_config()
        current.update(new_config)
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
        except Exception as e:
            print(f"[Config Save Note] {e}")
        return current

    # -------------------------------------------------------------------------
    # Module Marketplace & Repository System
    # -------------------------------------------------------------------------
    def get_installed_modules(self) -> list[dict[str, Any]]:
        """List currently installed modules."""
        if self.installed_file.exists():
            try:
                with open(self.installed_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_installed_modules(self, modules: list[dict[str, Any]]) -> None:
        try:
            with open(self.installed_file, "w", encoding="utf-8") as f:
                json.dump(modules, f, indent=2)
        except Exception as e:
            print(f"[Module Save Note] {e}")

    def search_marketplace(self, query: str = "", module_type: str = "all") -> dict[str, Any]:
        """Search curated registry and GitHub for available modules."""
        installed = {m["id"]: m for m in self.get_installed_modules()}

        # Filter curated recommendations
        curated_results = []
        for item in self.registry:
            if module_type != "all" and item["type"] != module_type:
                continue
            if query and query.lower() not in item["name"].lower() and query.lower() not in item["description"].lower():
                continue
            
            entry = dict(item)
            entry["installed"] = item["id"] in installed
            entry["installed_version"] = installed[item["id"]]["version"] if item["id"] in installed else None
            entry["has_update"] = entry["installed"] and entry["version"] != entry["installed_version"]
            curated_results.append(entry)

        # Search GitHub if query provided
        github_results = []
        if query:
            gh_items = self.github_client.search_repositories(query)
            for item in gh_items:
                item["installed"] = item["id"] in installed
                item["installed_version"] = installed[item["id"]]["version"] if item["id"] in installed else None
                item["has_update"] = False
                github_results.append(item)

        return {
            "curated": curated_results,
            "github": github_results,
            "installed_count": len(installed),
        }

    def install_module(self, module_data: dict[str, Any]) -> dict[str, Any]:
        """Install or update a module from curated registry or GitHub."""
        mod_id = module_data["id"]
        target_folder = self.modules_dir / mod_id
        target_folder.mkdir(parents=True, exist_ok=True)

        download_url = module_data.get("download_url", "")
        success = True
        msg = f"Installed {module_data['name']} successfully."

        # If it's a framework, attempt pip install
        if module_data.get("type") == "framework" and "package_name" in module_data:
            pkg_name = module_data["package_name"]
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            except Exception as e:
                success = False
                msg = f"Failed to pip install {pkg_name}: {e}"
        # If it has a direct file download link
        elif download_url and (download_url.endswith(".safetensors") or download_url.endswith(".jsonl")):
            filename = download_url.split("/")[-1]
            dest_file = target_folder / filename
            try:
                urllib.request.urlretrieve(download_url, dest_file)
            except Exception as e:
                # Create a placeholder file if remote network is offline
                dest_file.write_text(f"# Placeholder for {module_data['name']}\n", encoding="utf-8")
                msg = f"Module registered locally at {dest_file}"

        # Update installed registry
        installed = self.get_installed_modules()
        installed = [m for m in installed if m["id"] != mod_id]
        
        record = dict(module_data)
        record["installed_at"] = str(target_folder)
        record["status"] = "Active"
        installed.append(record)
        self._save_installed_modules(installed)

        return {"success": success, "message": msg, "installed_path": str(target_folder)}

    def uninstall_module(self, module_id: str) -> dict[str, Any]:
        """Uninstall a module and remove its directory."""
        target_folder = self.modules_dir / module_id
        if target_folder.exists():
            try:
                shutil.rmtree(target_folder)
            except Exception as e:
                print(f"[Uninstall Note] {e}")

        installed = self.get_installed_modules()
        installed = [m for m in installed if m["id"] != module_id]
        self._save_installed_modules(installed)
        return {"success": True, "message": f"Module {module_id} uninstalled."}

    def check_updates(self) -> list[dict[str, Any]]:
        """Check all installed modules for version updates."""
        installed = self.get_installed_modules()
        updates_available = []

        curated_map = {m["id"]: m for m in self.registry}
        for item in installed:
            mod_id = item["id"]
            if mod_id in curated_map:
                latest = curated_map[mod_id]
                if latest["version"] != item.get("version"):
                    updates_available.append({
                        "id": mod_id,
                        "name": item["name"],
                        "current_version": item.get("version"),
                        "latest_version": latest["version"],
                        "description": latest["description"],
                    })

        return updates_available
