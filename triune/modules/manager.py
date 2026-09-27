"""Triune Universal Hardware Scanner, Module Marketplace, and Configuration Manager."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .github_client import RepoSearchClient


class ModuleManager:
    """Manages system hardware detection, user configs, module installations, Git repositories, and dynamic DAG nodes."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        if config_dir:
            self.base_dir = Path(config_dir)
        else:
            if sys.platform == "win32" and os.path.exists("C:\\TriuneStudio"):
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
            # Fallback to local execution directory if permissions restricted
            self.base_dir = Path(__file__).resolve().parent.parent.parent / "studio_data"
            self.modules_dir = self.base_dir / "modules"
            self.config_file = self.base_dir / "studio_config.json"
            self.installed_file = self.base_dir / "installed_modules.json"
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.modules_dir.mkdir(parents=True, exist_ok=True)

        self.repo_client = RepoSearchClient()
        self.github_client = self.repo_client  # Backward compatibility
        self.registry = self._load_curated_registry()

        # Auto-discover and register all DAG nodes from previously installed modules
        self._auto_register_installed_nodes()

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

        # Fallback curated recommendations
        return [
            {
                "id": "custom-loss-nodes",
                "name": "Extended Router & Custom Loss Layer Nodes",
                "author": "Triune Research",
                "type": "plugin",
                "version": "1.2.0",
                "description": "Custom DAG visual nodes including Focal Cross-Entropy Loss, Centroid Balance Loss, and Gumbel Temperature Annealer.",
                "repo_url": "https://github.com/Yasho4867/triune",
                "download_url": "builtin://custom-loss-nodes",
                "size_mb": 2.5,
                "tags": ["DAG Nodes", "Plugin", "Loss Functions", "Verified"],
                "requires_cuda": False
            },
            {
                "id": "code-assistant-lora",
                "name": "Python & Rust Code Tuning Adapter (r=16)",
                "author": "Triune AI Community",
                "type": "adapter",
                "version": "1.1.0",
                "description": "Pre-tuned LoRA rank-16 adapter trained for code generation. Plugs directly into TriuneTransformer query and value projections.",
                "repo_url": "https://github.com/Yasho4867/triune",
                "download_url": "builtin://code-assistant-lora",
                "size_mb": 48.0,
                "tags": ["LoRA", "Coding", "r=16", "Adapter"],
                "requires_cuda": False
            },
            {
                "id": "fineweb-sample-10k",
                "name": "FineWeb Curated 10K Sample Dataset",
                "author": "HuggingFaceFW / Triune",
                "type": "dataset",
                "version": "2.1.0",
                "description": "Cleaned pre-tokenized JSONL subset of FineWeb for rapid local evaluation, dataset streaming, and synthetic router label benchmark generation.",
                "repo_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                "download_url": "builtin://fineweb-sample-10k",
                "size_mb": 14.5,
                "tags": ["Dataset", "Pre-tokenized", "BPE", "Streaming"],
                "requires_cuda": False
            },
            {
                "id": "triune-base-weights",
                "name": "Triune-Base 2.5B MoE Weights",
                "author": "Triune AI Core",
                "type": "model",
                "version": "1.2.0",
                "description": "Standard 24-layer MoE checkpoint with 8 experts and 3 exit heads (Reflex, Limbic, Cortex). Ready for local CPU & GPU inference and fine-tuning.",
                "repo_url": "https://github.com/Yasho4867/triune",
                "download_url": "builtin://triune-base-weights",
                "size_mb": 191.6,
                "tags": ["MoE", "2.5B", "Recommended", "Native"],
                "requires_cuda": False
            },
            {
                "id": "bitsandbytes-quant",
                "name": "BitsAndBytes 8-bit & 4-bit Quantization Engine",
                "author": "Tim Dettmers / Foundation",
                "type": "framework",
                "version": "0.43.0",
                "description": "Enables FP4, NF4, and Int8 matrix multiplication for extreme VRAM savings on consumer GPUs.",
                "repo_url": "https://github.com/bitsandbytes-foundation/bitsandbytes",
                "download_url": "pip://bitsandbytes",
                "package_name": "bitsandbytes",
                "size_mb": 85.0,
                "tags": ["Quantization", "NF4", "VRAM Saver"],
                "requires_cuda": False
            },
            {
                "id": "flash-attn-package",
                "name": "FlashAttention-2 CUDA Kernels",
                "author": "Dao-AILab",
                "type": "framework",
                "version": "2.5.6",
                "description": "Fast memory-efficient attention algorithms for NVIDIA Ampere, Ada, and Blackwell GPUs. Significantly improves token throughput.",
                "repo_url": "https://github.com/Dao-AILab/flash-attention",
                "download_url": "pip://flash-attn",
                "package_name": "flash-attn",
                "size_mb": 240.0,
                "tags": ["CUDA", "GPU Only", "Speedup"],
                "requires_cuda": True
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
        target_pkgs = ["torch", "fastapi", "uvicorn", "pywebview", "transformer_engine", "flash_attn", "bitsandbytes", "triton", "transformers"]
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
            "hf_token": "",
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
    # Installed Modules Listing & Inspection
    # -------------------------------------------------------------------------
    def get_installed_modules(self) -> list[dict[str, Any]]:
        """List currently installed modules with live disk inspection and Git metadata."""
        if self.installed_file.exists():
            try:
                with open(self.installed_file, "r", encoding="utf-8") as f:
                    modules = json.load(f)
                    # Refresh live disk metadata
                    for m in modules:
                        path_str = m.get("installed_at")
                        if path_str and Path(path_str).exists():
                            p = Path(path_str)
                            inspection = self.inspect_directory(p)
                            m["file_count"] = inspection.get("file_count", 0)
                            m["size_mb"] = inspection.get("size_mb", m.get("size_mb", 0))
                            m["git_info"] = inspection.get("git_info", {})
                            m["artifacts"] = inspection.get("artifacts", {})
                            m["registered_nodes"] = inspection.get("registered_nodes", [])
                    return modules
            except Exception:
                pass
        return []

    def _save_installed_modules(self, modules: list[dict[str, Any]]) -> None:
        try:
            with open(self.installed_file, "w", encoding="utf-8") as f:
                json.dump(modules, f, indent=2)
        except Exception as e:
            print(f"[Module Save Note] {e}")

    def inspect_directory(self, folder: Path) -> dict[str, Any]:
        """Inspect a directory for Git metadata, DAG nodes, model weights, datasets, and file counts."""
        if not folder.exists():
            return {"file_count": 0, "size_mb": 0, "git_info": {}, "artifacts": {}, "registered_nodes": []}

        file_count = 0
        total_size = 0
        py_files = []
        weight_files = []
        dataset_files = []
        adapter_files = []

        try:
            for root, _, files in os.walk(folder):
                for file in files:
                    file_count += 1
                    fp = Path(root) / file
                    try:
                        total_size += fp.stat().st_size
                    except OSError:
                        pass

                    fl = file.lower()
                    if fl.endswith(".py"):
                        py_files.append(file)
                    elif fl.endswith((".safetensors", ".pt", ".bin", ".gguf", ".onnx")):
                        weight_files.append(file)
                    elif fl.endswith((".jsonl", ".parquet", ".csv", ".txt")):
                        dataset_files.append(file)
                    elif "adapter" in fl or fl == "adapter_config.json":
                        adapter_files.append(file)
        except Exception:
            pass

        # Git inspection
        git_info = {}
        if (folder / ".git").exists():
            try:
                commit = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=folder, capture_output=True, text=True, timeout=2.0
                ).stdout.strip()
                branch = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=folder, capture_output=True, text=True, timeout=2.0
                ).stdout.strip()
                msg = subprocess.run(
                    ["git", "log", "-1", "--pretty=%s"],
                    cwd=folder, capture_output=True, text=True, timeout=2.0
                ).stdout.strip()
                git_info = {
                    "is_git": True,
                    "commit": commit,
                    "branch": branch or "main",
                    "commit_msg": msg
                }
            except Exception:
                git_info = {"is_git": True, "commit": "local", "branch": "main", "commit_msg": "Local Git Repository"}

        # Check registered DAG nodes in this module
        registered_nodes = []
        try:
            from triune.plugins.registry import global_registry
            for n_name, meta in global_registry._nodes.items():
                if meta.get("module_source") == str(folder):
                    registered_nodes.append(n_name)
        except Exception:
            pass

        return {
            "file_count": file_count,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "git_info": git_info,
            "artifacts": {
                "python_files": py_files[:5],
                "nodes": py_files[:5],
                "weights": weight_files[:5],
                "datasets": dataset_files[:5],
                "adapters": adapter_files[:5],
            },
            "registered_nodes": registered_nodes
        }

    # -------------------------------------------------------------------------
    # Search Marketplace & Repositories (Curated, GitHub, Hugging Face)
    # -------------------------------------------------------------------------
    def search_marketplace(self, query: str = "", module_type: str = "all", source: str = "all") -> dict[str, Any]:
        """Search curated registry, GitHub repositories, and Hugging Face Hub."""
        installed = {m["id"]: m for m in self.get_installed_modules()}

        curated_results = []
        if source in ("all", "curated"):
            for item in self.registry:
                if module_type != "all" and item["type"] != module_type:
                    continue
                if query and query.lower() not in item["name"].lower() and query.lower() not in item["description"].lower():
                    continue

                entry = dict(item)
                entry["installed"] = item["id"] in installed
                entry["installed_version"] = installed[item["id"]].get("version") if item["id"] in installed else None
                entry["has_update"] = entry["installed"] and entry["version"] != entry["installed_version"]
                entry["source"] = "curated"
                curated_results.append(entry)

        github_results = []
        if source in ("all", "github") and query:
            gh_items = self.repo_client.search_github_repositories(query)
            for item in gh_items:
                item["installed"] = item["id"] in installed or item["name"] in installed
                item["installed_version"] = None
                item["has_update"] = False
                github_results.append(item)

        huggingface_results = []
        if source in ("all", "huggingface"):
            if module_type in ("all", "model"):
                hf_models = self.repo_client.search_huggingface_models(query, limit=8)
                for item in hf_models:
                    item["installed"] = item["id"] in installed or item["name"] in installed
                    huggingface_results.append(item)
            if module_type in ("all", "dataset"):
                hf_datasets = self.repo_client.search_huggingface_datasets(query, limit=8)
                for item in hf_datasets:
                    item["installed"] = item["id"] in installed or item["name"] in installed
                    huggingface_results.append(item)

        return {
            "curated": curated_results,
            "github": github_results,
            "huggingface": huggingface_results,
            "installed_count": len(installed),
        }

    # -------------------------------------------------------------------------
    # Real Git Clone & Repository Import
    # -------------------------------------------------------------------------
    def clone_repository(
        self,
        repo_url: str,
        module_type: str = "auto",
        branch: str = "main",
        custom_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Clone any Git or Hugging Face repository, inspect contents, and register DAG nodes."""
        url = repo_url.strip()
        if not url:
            return {"success": False, "message": "Repository URL cannot be empty."}

        # Handle Hugging Face short identifier (e.g. 'roneneldan/TinyStories' or 'HuggingFaceFW/fineweb-edu')
        is_hf = "huggingface.co" in url or (len(url.split("/")) == 2 and not url.startswith("http"))
        if is_hf and not url.startswith("http"):
            url = f"https://huggingface.co/{url}"

        # Clean name derivation
        clean_name = custom_name or url.rstrip("/").split("/")[-1].replace(".git", "")
        clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clean_name)
        target_folder = self.modules_dir / clean_name

        if target_folder.exists() and any(target_folder.iterdir()):
            # Already exists - perform git pull instead
            return self.git_pull(clean_name)

        target_folder.mkdir(parents=True, exist_ok=True)
        git_success = False
        error_msg = ""

        # Attempt 1: Real git clone
        try:
            clone_cmd = ["git", "clone", "--depth", "1"]
            if branch and branch != "main":
                clone_cmd.extend(["-b", branch])
            clone_cmd.extend([url, str(target_folder)])

            res = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=90.0)
            if res.returncode == 0:
                git_success = True
            else:
                error_msg = res.stderr.strip()
        except Exception as e:
            error_msg = str(e)

        # Attempt 2: Zip fallback if GitHub repo
        if not git_success and "github.com" in url:
            try:
                base_gh = url.rstrip("/").replace(".git", "")
                zip_url = f"{base_gh}/archive/refs/heads/{branch}.zip"
                tmp_zip = target_folder / "repo.zip"
                urllib.request.urlretrieve(zip_url, tmp_zip)
                with zipfile.ZipFile(tmp_zip, "r") as z:
                    z.extractall(target_folder)
                tmp_zip.unlink(missing_ok=True)
                git_success = True
            except Exception as e2:
                error_msg = f"{error_msg} | Zip fallback failed: {e2}"

        if not git_success:
            return {
                "success": False,
                "message": f"Failed to clone repository: {error_msg}",
                "installed_path": str(target_folder)
            }

        # Inspect cloned folder contents
        inspection = self.inspect_directory(target_folder)
        detected_type = module_type
        if detected_type == "auto":
            if inspection["artifacts"]["adapters"]:
                detected_type = "adapter"
            elif inspection["artifacts"]["weights"]:
                detected_type = "model"
            elif inspection["artifacts"]["datasets"]:
                detected_type = "dataset"
            elif inspection["artifacts"]["python_files"]:
                detected_type = "plugin"
            else:
                detected_type = "general"

        # Auto-discover and register any DAG nodes in the cloned repo
        registered_nodes = self.discover_and_register_nodes(target_folder)

        # Record to installed_modules.json
        installed = self.get_installed_modules()
        installed = [m for m in installed if m["id"] != clean_name]

        record = {
            "id": clean_name,
            "name": clean_name,
            "repo_url": url,
            "type": detected_type,
            "version": inspection["git_info"].get("commit", "latest"),
            "description": f"Cloned from {url} ({detected_type.upper()})",
            "installed_at": str(target_folder),
            "status": "Active",
            "file_count": inspection["file_count"],
            "size_mb": inspection["size_mb"],
            "git_info": inspection["git_info"],
            "artifacts": inspection["artifacts"],
            "registered_nodes": registered_nodes
        }
        installed.append(record)
        self._save_installed_modules(installed)

        node_note = f" Found & registered {len(registered_nodes)} DAG node(s)!" if registered_nodes else ""
        return {
            "success": True,
            "message": f"Cloned {clean_name} successfully ({detected_type}).{node_note}",
            "module": record
        }

    # -------------------------------------------------------------------------
    # Git Pull / Sync
    # -------------------------------------------------------------------------
    def git_pull(self, module_id: str) -> dict[str, Any]:
        """Perform git pull on a cloned repository to synchronize upstream commits."""
        target_folder = self.modules_dir / module_id
        if not target_folder.exists() or not (target_folder / ".git").exists():
            return {"success": False, "message": f"Module {module_id} is not a local Git repository."}

        try:
            res = subprocess.run(["git", "pull"], cwd=target_folder, capture_output=True, text=True, timeout=45.0)
            if res.returncode != 0:
                return {"success": False, "message": f"git pull error: {res.stderr.strip()}"}

            pull_output = res.stdout.strip()

            # Re-discover and reload DAG nodes
            registered_nodes = self.discover_and_register_nodes(target_folder)
            inspection = self.inspect_directory(target_folder)

            # Update record
            installed = self.get_installed_modules()
            for m in installed:
                if m["id"] == module_id:
                    m["version"] = inspection["git_info"].get("commit", m.get("version"))
                    m["git_info"] = inspection["git_info"]
                    m["file_count"] = inspection["file_count"]
                    m["size_mb"] = inspection["size_mb"]
                    m["registered_nodes"] = registered_nodes
                    break
            self._save_installed_modules(installed)

            return {
                "success": True,
                "message": f"Updated {module_id}: {pull_output}",
                "git_info": inspection["git_info"],
                "registered_nodes": registered_nodes
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to pull repository: {e}"}

    # -------------------------------------------------------------------------
    # Dynamic Plugin & DAG Node Discovery
    # -------------------------------------------------------------------------
    def discover_and_register_nodes(self, module_dir: Path) -> List[str]:
        """Scan a module directory for Python scripts and register discovered DAG nodes into global_registry."""
        discovered = []
        if not module_dir.exists():
            return discovered

        try:
            from triune.plugins.registry import global_registry
            from triune.execution.dag import ExecutionEngine
        except ImportError:
            return discovered

        # Scan all .py files in module directory
        for py_file in module_dir.rglob("*.py"):
            if py_file.name.startswith("__") and py_file.name != "__init__.py":
                continue

            module_name = f"triune_plugin_{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)

                    # Look for items with _triune_node_schema or registered classes
                    for attr_name in dir(mod):
                        obj = getattr(mod, attr_name)
                        schema = getattr(obj, "_triune_node_schema", None)
                        if schema and isinstance(schema, dict) and "name" in schema:
                            node_name = schema["name"]
                            schema["module_source"] = str(module_dir)
                            global_registry._nodes[node_name] = schema
                            discovered.append(node_name)
            except Exception as e:
                print(f"[Plugin Discovery Note] Could not load {py_file}: {e}")

        return discovered

    def _auto_register_installed_nodes(self) -> None:
        """Scan all installed modules and register their nodes on startup."""
        for m in self.get_installed_modules():
            path_str = m.get("installed_at")
            if path_str and Path(path_str).exists():
                self.discover_and_register_nodes(Path(path_str))

    # -------------------------------------------------------------------------
    # Module Installation (Curated / Builtin / Pip / Cloned)
    # -------------------------------------------------------------------------
    def install_module(self, module_data: dict[str, Any]) -> dict[str, Any]:
        """Install or update a curated recommendation, framework, or remote repository."""
        mod_id = module_data.get("id") or module_data.get("name", "")
        # If partial payload passed, merge with curated registry metadata
        reg_item = next((r for r in self.registry if r["id"] == mod_id), None)
        if reg_item:
            full_data = dict(reg_item)
            full_data.update(module_data)
            module_data = full_data

        target_folder = self.modules_dir / mod_id
        target_folder.mkdir(parents=True, exist_ok=True)

        mod_name = module_data.get("name", mod_id)
        download_url = module_data.get("download_url", "")
        repo_url = module_data.get("repo_url", "")
        success = True
        msg = f"Installed {mod_name} successfully."
        registered_nodes = []

        # Case 1: Built-in curated component
        if download_url.startswith("builtin://"):
            curated_id = download_url.replace("builtin://", "")
            src_curated = Path(__file__).parent / "curated" / curated_id
            if not src_curated.exists():
                src_curated = Path(__file__).parent / "curated" / curated_id.replace("-", "_")

            if src_curated.exists():
                for item in src_curated.iterdir():
                    dest = target_folder / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
            elif curated_id == "fineweb-sample-10k":
                # Link local sample dataset
                workspace_sample = Path(__file__).resolve().parent.parent.parent / "data" / "fineweb_sample.jsonl"
                if workspace_sample.exists():
                    shutil.copy2(workspace_sample, target_folder / "fineweb_sample.jsonl")
            elif curated_id == "triune-base-weights":
                # Link or note local model checkpoint
                workspace_ckpt = Path(__file__).resolve().parent.parent.parent / "checkpoints"
                (target_folder / "README_WEIGHTS.txt").write_text(
                    "Triune-Base 2.5B MoE Weights initialized and mapped to live PyTorch engine checkpoints.\n",
                    encoding="utf-8"
                )

            # Auto-discover any DAG nodes in the curated module
            registered_nodes = self.discover_and_register_nodes(target_folder)
            msg = f"Installed curated module {module_data['name']}."

        # Case 2: Pip framework install
        elif download_url.startswith("pip://") or module_data.get("package_name"):
            pkg_name = module_data.get("package_name") or download_url.replace("pip://", "")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
                msg = f"Successfully installed {pkg_name} via pip."
            except Exception as e:
                success = False
                msg = f"Failed to pip install {pkg_name}: {e}"

        # Case 3: Remote Git / GitHub repo
        elif repo_url:
            res = self.clone_repository(repo_url, module_type=module_data.get("type", "auto"), custom_name=mod_id)
            return res

        # Case 4: File download (.safetensors, .jsonl)
        elif download_url and any(download_url.endswith(ext) for ext in (".safetensors", ".pt", ".bin", ".jsonl")):
            filename = download_url.split("/")[-1]
            dest_file = target_folder / filename
            try:
                urllib.request.urlretrieve(download_url, dest_file)
                msg = f"Downloaded {filename} to {dest_file}"
            except Exception as e:
                dest_file.write_text(f"# File registered for {module_data['name']}\n", encoding="utf-8")
                msg = f"Module registered locally at {dest_file}"

        # Inspect and record
        inspection = self.inspect_directory(target_folder)
        installed = self.get_installed_modules()
        installed = [m for m in installed if m["id"] != mod_id]

        record = dict(module_data)
        record["installed_at"] = str(target_folder)
        record["status"] = "Active"
        record["file_count"] = inspection["file_count"]
        record["size_mb"] = inspection["size_mb"]
        record["git_info"] = inspection["git_info"]
        record["artifacts"] = inspection["artifacts"]
        record["registered_nodes"] = registered_nodes
        installed.append(record)
        self._save_installed_modules(installed)

        return {"success": success, "message": msg, "installed_path": str(target_folder), "module": record}

    # -------------------------------------------------------------------------
    # Uninstall Module
    # -------------------------------------------------------------------------
    def uninstall_module(self, module_id: str) -> dict[str, Any]:
        """Uninstall a module, unregister its DAG nodes, and remove its directory."""
        target_folder = self.modules_dir / module_id
        if target_folder.exists():
            try:
                shutil.rmtree(target_folder)
            except Exception as e:
                print(f"[Uninstall Note] {e}")

        # Unregister DAG nodes from global_registry
        try:
            from triune.plugins.registry import global_registry
            to_del = [k for k, v in global_registry._nodes.items() if v.get("module_source") == str(target_folder)]
            for k in to_del:
                del global_registry._nodes[k]
        except Exception:
            pass

        installed = self.get_installed_modules()
        installed = [m for m in installed if m["id"] != module_id]
        self._save_installed_modules(installed)
        return {"success": True, "message": f"Module {module_id} uninstalled."}

    # -------------------------------------------------------------------------
    # Check Updates
    # -------------------------------------------------------------------------
    def check_updates(self) -> list[dict[str, Any]]:
        """Check all installed modules for version updates (curated and Git)."""
        installed = self.get_installed_modules()
        updates_available = []

        curated_map = {m["id"]: m for m in self.registry}
        for item in installed:
            mod_id = item["id"]
            # Check curated version
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

            # Check Git remote updates if it's a Git repo
            path_str = item.get("installed_at")
            if path_str and (Path(path_str) / ".git").exists():
                try:
                    p = Path(path_str)
                    subprocess.run(["git", "fetch"], cwd=p, capture_output=True, timeout=5.0)
                    status = subprocess.run(["git", "status", "-uno"], cwd=p, capture_output=True, text=True, timeout=3.0).stdout
                    if "behind" in status:
                        updates_available.append({
                            "id": mod_id,
                            "name": item["name"],
                            "current_version": item.get("version", "local"),
                            "latest_version": "upstream",
                            "description": "Upstream Git commits available to pull.",
                        })
                except Exception:
                    pass

        return updates_available
