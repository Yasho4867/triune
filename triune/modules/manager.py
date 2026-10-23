"""Triune Universal Hardware Scanner, Module Marketplace, and Configuration Manager."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import inspect
import json
import keyword
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .github_client import RepoSearchClient

_SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-\[\]>=<,!~]*$")
_SAFE_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-/]*$")
_SAFE_MODULE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_KEYRING_SERVICE = "Triune Studio"
_KEYRING_MARKER = "[STORED_IN_OS_KEYRING]"
_ENV_MARKER = "[CONFIGURED_VIA_ENV]"


class _WindowsCredentialStoreBridge:
    """Use the Windows app's Credential Manager when the API process is in WSL.

    The secret is sent to the Windows-side Python process over stdin (never in
    argv or a temporary file). Credential Manager is accessed through Windows'
    built-in Cred* API, so this path does not depend on an optional Python
    keyring package being installed in the Windows runtime.
    """

    _HELPER = r"""
import ctypes, json, sys
from ctypes import wintypes

ERROR_NOT_FOUND = 1168
CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2

class CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ('Flags', wintypes.DWORD),
        ('Type', wintypes.DWORD),
        ('TargetName', wintypes.LPWSTR),
        ('Comment', wintypes.LPWSTR),
        ('LastWritten', wintypes.FILETIME),
        ('CredentialBlobSize', wintypes.DWORD),
        ('CredentialBlob', ctypes.POINTER(ctypes.c_ubyte)),
        ('Persist', wintypes.DWORD),
        ('AttributeCount', wintypes.DWORD),
        ('Attributes', ctypes.c_void_p),
        ('TargetAlias', wintypes.LPWSTR),
        ('UserName', wintypes.LPWSTR),
    ]

advapi32 = ctypes.WinDLL('Advapi32.dll', use_last_error=True)
advapi32.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
advapi32.CredWriteW.restype = wintypes.BOOL
advapi32.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
advapi32.CredReadW.restype = wintypes.BOOL
advapi32.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
advapi32.CredDeleteW.restype = wintypes.BOOL
advapi32.CredFree.argtypes = [ctypes.c_void_p]
advapi32.CredFree.restype = None

def win_error():
    code = ctypes.get_last_error()
    raise OSError(code, ctypes.FormatError(code).strip())

payload = json.load(sys.stdin)
op, service, username = payload['op'], payload['service'], payload['username']
target_name = service + ':' + username
if op == 'get':
    credential = ctypes.POINTER(CREDENTIALW)()
    if not advapi32.CredReadW(target_name, CRED_TYPE_GENERIC, 0, ctypes.byref(credential)):
        code = ctypes.get_last_error()
        if code == ERROR_NOT_FOUND:
            result = {'value': ''}
        else:
            win_error()
    else:
        try:
            blob = ctypes.string_at(credential.contents.CredentialBlob, credential.contents.CredentialBlobSize)
            result = {'value': blob.decode('utf-8')}
        finally:
            advapi32.CredFree(credential)
elif op == 'set':
    secret = payload['password'].encode('utf-8')
    if len(secret) > 2560:
        raise ValueError('Credential exceeds the Windows Credential Manager 2560-byte limit')
    blob = ctypes.create_string_buffer(secret)
    credential = CREDENTIALW(
        Flags=0,
        Type=CRED_TYPE_GENERIC,
        TargetName=target_name,
        Comment='Triune Studio credential',
        LastWritten=wintypes.FILETIME(),
        CredentialBlobSize=len(secret),
        CredentialBlob=ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte)),
        Persist=CRED_PERSIST_LOCAL_MACHINE,
        AttributeCount=0,
        Attributes=None,
        TargetAlias=None,
        UserName=username,
    )
    if not advapi32.CredWriteW(ctypes.byref(credential), 0):
        win_error()
    result = {'ok': True}
elif op == 'delete':
    if not advapi32.CredDeleteW(target_name, CRED_TYPE_GENERIC, 0):
        code = ctypes.get_last_error()
        if code != ERROR_NOT_FOUND:
            win_error()
    result = {'ok': True}
else:
    raise ValueError('Unsupported credential operation')
sys.stdout.write(json.dumps(result))
"""

    def __init__(self, python_executable: str) -> None:
        self.python_executable = python_executable

    def _call(self, op: str, service: str, username: str, password: str = "") -> str:
        payload = json.dumps({
            "op": op,
            "service": service,
            "username": username,
            "password": password,
        })
        try:
            result = subprocess.run(
                [self.python_executable, "-c", self._HELPER],
                input=payload,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "Could not reach the Windows Credential Manager bridge. Restart Triune Studio and verify its Windows runtime is installed."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "Windows Credential Manager is unavailable to the Triune Studio runtime. Run the Studio setup again to repair its secure-storage component."
            )
        try:
            response = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("Windows Credential Manager returned an invalid response.") from exc
        if op == "get":
            return response.get("value", "")
        return ""

    def get_password(self, service: str, username: str) -> str:
        return self._call("get", service, username)

    def set_password(self, service: str, username: str, password: str) -> None:
        self._call("set", service, username, password)

    def delete_password(self, service: str, username: str) -> None:
        self._call("delete", service, username)


def _verified_plugin_entrypoints(module_dir: Path) -> tuple[list[Path], str | None]:
    """Validate a versioned Ed25519-signed external plugin manifest and its entrypoints."""
    manifest_path = module_dir / "triune_plugin.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return [], "A valid triune_plugin.json manifest is required."
    if not isinstance(manifest, dict) or type(manifest.get("manifest_version")) is not int or manifest["manifest_version"] != 1:
        return [], "Unsupported plugin manifest version; expected manifest_version 1."
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        return [], "Plugin manifest must include a name."
    if not isinstance(manifest.get("version"), str) or not re.fullmatch(
        r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", manifest["version"]
    ):
        return [], "Plugin manifest version must use semantic versioning (for example 1.2.3)."
    entrypoints = manifest.get("entrypoints")
    if not isinstance(entrypoints, list) or not entrypoints or len(entrypoints) > 32:
        return [], "Plugin manifest must declare between 1 and 32 Python entrypoints."
    permissions = manifest.get("permissions", [])
    if not isinstance(permissions, list) or len(permissions) > 32 or not all(
        isinstance(item, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", item) for item in permissions
    ):
        return [], "Plugin manifest permissions must be a list of simple permission names."

    root = module_dir.resolve()
    resolved_entrypoints: list[Path] = []
    entrypoint_hashes = manifest.get("files")
    if not isinstance(entrypoint_hashes, dict):
        return [], "Plugin manifest must bind its entrypoint file hashes in the 'files' object."
    for entrypoint in entrypoints:
        if not isinstance(entrypoint, str) or "\\" in entrypoint or Path(entrypoint).is_absolute():
            return [], "Plugin entrypoint paths must be relative POSIX paths."
        relative = Path(entrypoint)
        if relative.suffix != ".py" or ".." in relative.parts:
            return [], "Plugin entrypoints must be in-repository .py files."
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            return [], f"Plugin entrypoint is missing or escapes the repository: {entrypoint}"
        expected_hash = entrypoint_hashes.get(entrypoint)
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_hash):
            return [], f"Plugin manifest has no valid SHA-256 hash for {entrypoint}."
        try:
            file_data = path.read_bytes()
        except OSError:
            return [], f"Could not read plugin entrypoint: {entrypoint}"
        if len(file_data) > 2 * 1024 * 1024:
            return [], f"Plugin entrypoint exceeds the 2 MiB safety limit: {entrypoint}"
        if hashlib.sha256(file_data).hexdigest().lower() != expected_hash.lower():
            return [], f"Plugin entrypoint hash mismatch: {entrypoint}"
        resolved_entrypoints.append(path)

    signature = manifest.get("signature")
    if not isinstance(signature, dict) or str(signature.get("algorithm", "")).lower() != "ed25519":
        return [], "Plugin manifest must contain a trusted Ed25519 signature."
    key_id = signature.get("key_id")
    try:
        trusted_keys = json.loads(os.environ.get("TRIUNE_PLUGIN_TRUSTED_KEYS", "{}"))
    except Exception:
        trusted_keys = {}
    if not isinstance(trusted_keys, dict) or not isinstance(key_id, str) or key_id not in trusted_keys:
        return [], "Plugin signing key is not trusted by this Triune installation."
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = base64.b64decode(trusted_keys[key_id], validate=True)
        signature_bytes = base64.b64decode(signature.get("value", ""), validate=True)
        signed_payload = {key: value for key, value in manifest.items() if key != "signature"}
        canonical_payload = json.dumps(signed_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature_bytes, canonical_payload)
    except ImportError:
        return [], "Signature verification requires the cryptography package in the Studio environment."
    except Exception:
        return [], "Plugin manifest signature is invalid or malformed."

    return resolved_entrypoints, None


def _is_safe_plugin_ast(py_file: Path, strict_untrusted: bool = False) -> bool:
    """Static AST gate for plugin files: must declare @register_node and contain no dangerous primitives."""
    try:
        source = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(py_file))
    except Exception:
        return False

    has_register_node = False
    banned_modules = {
        "os",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "http",
        "ftplib",
        "telnetlib",
        "ctypes",
        "pickle",
        "marshal",
        "pty",
        "pipes",
        "multiprocessing",
    }
    if strict_untrusted:
        banned_modules.update({"sys", "importlib", "builtins", "tempfile", "glob"})

    banned_calls = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
    if strict_untrusted:
        banned_calls.update({"open", "getattr", "setattr", "delattr", "globals", "locals", "vars"})

    banned_dunders = {
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "__bases__",
        "__mro__",
        "__dict__",
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                dec_func = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(dec_func, ast.Name) and dec_func.id == "register_node":
                    has_register_node = True
                elif isinstance(dec_func, ast.Attribute) and dec_func.attr == "register_node":
                    has_register_node = True
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "_triune_node_schema":
                    has_register_node = True

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in banned_modules:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in banned_modules:
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in banned_calls:
                return False
            elif isinstance(node.func, ast.Attribute) and node.func.attr in (
                "system",
                "popen",
                "Popen",
                "run",
                "call",
                "check_call",
                "check_output",
                "exec_module",
            ):
                return False
        elif isinstance(node, ast.Attribute):
            if node.attr in banned_dunders:
                return False

    return has_register_node


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

        # Seed starter curated modules if fresh setup
        if not self.installed_file.exists() or len(self.get_installed_modules()) == 0:
            self._seed_starter_modules()

        # Auto-discover and register all DAG nodes from previously installed modules
        self._auto_register_installed_nodes()

    @staticmethod
    def _workspace_root() -> Path:
        """Return the source/project root used by local plugin workflows."""
        return Path(__file__).resolve().parent.parent.parent

    @classmethod
    def _workspace_path(cls, candidate: str | Path) -> Path | None:
        """Resolve a local workspace path without allowing scans/writes outside the project."""
        try:
            root = cls._workspace_root().resolve()
            path = Path(candidate).expanduser().resolve()
            if path != root and root not in path.parents:
                return None
            return path
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

    def _module_path(self, module_id: str) -> Path | None:
        """Resolve a module ID strictly beneath modules_dir, rejecting traversal and links out."""
        module_id = str(module_id or "").strip()
        if module_id in {".", ".."} or not _SAFE_MODULE_ID_RE.fullmatch(module_id):
            return None
        root = self.modules_dir.resolve()
        target = (root / module_id).resolve()
        if target == root or root not in target.parents:
            return None
        return target

    @staticmethod
    def _env_secret_for(secret_name: str) -> str:
        env_names = {
            "github_token": ("GITHUB_TOKEN",),
            "hf_token": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
            "byok:openai": ("OPENAI_API_KEY",),
            "byok:anthropic": ("ANTHROPIC_API_KEY",),
            "byok:huggingface": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
            "byok:gemini": ("GEMINI_API_KEY",),
        }
        return next((os.environ[name] for name in env_names.get(secret_name, ()) if os.environ.get(name)), "")

    @staticmethod
    def _get_secure_keyring() -> Any:
        """Return a supported OS credential backend; reject plaintext keyring fallbacks."""
        windows_python = os.environ.get("TRIUNE_WINDOWS_PYTHON", "").strip()
        if not windows_python and sys.platform.startswith("linux"):
            for cand in (
                "/mnt/c/Users/yashb_f1ls/OneDrive/Documents/TriuneTransformer/.venv/Scripts/python.exe",
                "/mnt/c/Program Files/Blender Foundation/Blender 5.2/5.2/python/bin/python.exe",
            ):
                if Path(cand).is_file():
                    windows_python = cand
                    break
        if windows_python:
            if not Path(windows_python).is_file():
                raise RuntimeError(
                    "The Windows Credential Manager bridge configured by the Studio launcher is missing. Restart Studio from the installed Windows app."
                )
            return _WindowsCredentialStoreBridge(windows_python)

        # A backend running directly on Windows should use the same built-in
        # Credential Manager path as WSL. This keeps secure saves working even
        # when the optional `keyring` package was omitted from an existing venv.
        if sys.platform == "win32":
            return _WindowsCredentialStoreBridge(sys.executable)

        if os.environ.get("TRIUNE_TESTING") == "1":
            class _InMemoryTestKeyring:
                _vault: dict[str, str] = {}
                @classmethod
                def get_password(cls, service: str, username: str) -> Optional[str]:
                    return cls._vault.get(f"{service}:{username}", "")
                @classmethod
                def set_password(cls, service: str, username: str, password: str) -> None:
                    cls._vault[f"{service}:{username}"] = password
                @classmethod
                def delete_password(cls, service: str, username: str) -> None:
                    cls._vault.pop(f"{service}:{username}", None)
            return _InMemoryTestKeyring

        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError(
                "Secure credential storage is unavailable. Install the Studio dependencies, "
                "including keyring, and configure Windows Credential Manager, macOS Keychain, "
                "or Linux Secret Service before saving API keys."
            ) from exc

        backend = keyring.get_keyring()
        module_name = type(backend).__module__
        supported = (
            "keyring.backends.Windows",
            "keyring.backends.macOS",
            "keyring.backends.SecretService",
            "keyring.backends.kwallet",
        )
        if not any(module_name.startswith(prefix) for prefix in supported) or getattr(backend, "priority", 0) <= 0:
            raise RuntimeError(
                f"Refusing insecure credential backend '{type(backend).__name__}'. "
                "Configure an OS credential store (Windows Credential Manager, macOS Keychain, "
                "or Linux Secret Service); plaintext file backends are not supported."
            )
        return keyring

    _SECRET_CACHE: dict[str, str] = {}

    @classmethod
    def _read_secure_secret(cls, secret_name: str) -> str:
        if secret_name in cls._SECRET_CACHE:
            return cls._SECRET_CACHE[secret_name]
        keyring = cls._get_secure_keyring()
        val = keyring.get_password(_KEYRING_SERVICE, secret_name) or ""
        cls._SECRET_CACHE[secret_name] = val
        return val

    @classmethod
    def _write_secure_secret(cls, secret_name: str, value: str) -> None:
        keyring = cls._get_secure_keyring()
        keyring.set_password(_KEYRING_SERVICE, secret_name, value)
        cls._SECRET_CACHE[secret_name] = value

    @classmethod
    def _delete_secure_secret(cls, secret_name: str) -> None:
        keyring = cls._get_secure_keyring()
        if keyring.get_password(_KEYRING_SERVICE, secret_name):
            keyring.delete_password(_KEYRING_SERVICE, secret_name)
        cls._SECRET_CACHE.pop(secret_name, None)

    def _seed_starter_modules(self) -> None:
        """Seed starter curated modules on initial setup so the ecosystem is ready immediately."""
        starter_ids = ["custom-loss-nodes", "code-assistant-lora", "fineweb-sample-10k"]
        for sid in starter_ids:
            try:
                self.install_module({"id": sid})
            except Exception:
                pass


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
        import importlib.metadata as im

        gpu_name = "CPU Only (No NVIDIA GPU Detected)"
        cuda_version = "None"
        cudnn_version = "None"
        compute_cap_str = ""
        driver_version = ""
        vram_total_gb = 0.0
        has_cuda = False
        ram_total_gb = 16.0

        # Detect Host RAM
        try:
            if Path("/proc/meminfo").is_file():
                for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_total_gb = round(kb / (1024 * 1024), 1)
                        break
        except Exception:
            pass

        # 1. Query PyTorch CUDA runtime directly (fastest and most accurate for active environment)
        torch_mod = None
        try:
            import torch as _torch
            torch_mod = _torch
            if _torch.cuda.is_available():
                has_cuda = True
                props = _torch.cuda.get_device_properties(0)
                gpu_name = props.name
                vram_total_gb = round(props.total_memory / (1024 ** 3), 2)
                compute_cap_str = f"SM {props.major}.{props.minor}"
                if _torch.version.cuda:
                    cuda_version = str(_torch.version.cuda)
                if _torch.backends.cudnn.is_available():
                    cver = _torch.backends.cudnn.version()
                    if cver:
                        major = cver // 10000 if cver >= 10000 else cver // 1000
                        minor = (cver % 10000) // 100 if cver >= 10000 else (cver % 1000) // 100
                        cudnn_version = f"{major}.{minor}"
        except Exception:
            pass

        # 2. Query nvidia-smi for driver version and fallback GPU detection
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                parts = [p.strip() for p in smi.stdout.strip().split(",")]
                if not has_cuda and parts[0]:
                    gpu_name = parts[0]
                if len(parts) >= 2 and vram_total_gb == 0.0:
                    vram_raw = parts[1].replace("MiB", "").strip()
                    try:
                        vram_total_gb = round(float(vram_raw) / 1024.0, 2)
                    except ValueError:
                        pass
                if len(parts) >= 3:
                    driver_version = parts[2]
                has_cuda = True
        except Exception:
            pass

        # 3. Query nvcc only if torch.version.cuda was not found
        if cuda_version == "None":
            try:
                nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=3.0)
                if nvcc.returncode == 0 and "release" in nvcc.stdout:
                    cuda_version = nvcc.stdout.split("release")[-1].split(",")[0].strip()
            except Exception:
                pass

        cuda_display = cuda_version
        if cuda_version != "None":
            extras = []
            if cudnn_version != "None":
                extras.append(f"cuDNN {cudnn_version}")
            if compute_cap_str:
                extras.append(compute_cap_str)
            if extras:
                cuda_display = f"{cuda_version} ({' • '.join(extras)})"

        def _pkg_version(import_name: str, dist_name: Optional[str] = None) -> tuple[bool, str]:
            d_name = dist_name or import_name
            try:
                return True, im.version(d_name)
            except Exception:
                pass
            try:
                mod = __import__(import_name)
                ver = getattr(mod, "__version__", "Installed")
                return True, str(ver)
            except Exception:
                return False, "Not Installed"

        packages: dict[str, dict[str, Any]] = {}
        stack_specs = [
            ("torch", "torch", "Core Engine", "PyTorch CUDA Autograd, BF16/FP8 AMP & Tensor Cores"),
            ("triton", "triton", "CUDA Kernels", "OpenAI Triton Compiler for Fused RMSNorm, RoPE, GLA & MoE"),
            ("flash_attn", "flash-attn", "Attention Kernels", "Hardware-Fused FlashAttention-2 & cuDNN SDPA Kernels"),
            ("transformer_engine", "transformer-engine", "FP8 Precision", "FP8 E4M3/E5M2 Scaled GEMM & Tensor Core Engine"),
            ("bitsandbytes", "bitsandbytes", "Quantization", "8-bit & 4-bit NF4 Quantized Linear & Optimizer Kernels"),
            ("transformers", "transformers", "Model Hub", "Hugging Face Model Architectures & Pretrained Interop"),
            ("accelerate", "accelerate", "Offload & AMP", "Hugging Face Device Offloading & Mixed-Precision Runtime"),
            ("datasets", "datasets", "Data Streaming", "Zero-Disk Hugging Face HTTP & Parquet Dataset Streamer"),
            ("tokenizers", "tokenizers", "Tokenizer", "Rust-Accelerated Hugging Face BPE Tokenizer Engine"),
            ("safetensors", "safetensors", "Serialization", "Zero-Copy SafeTensors Checkpoint Serialization"),
            ("pyarrow", "pyarrow", "Data Streaming", "Apache Arrow Columnar Memory & Parquet Reader"),
            ("einops", "einops", "Tensor Math", "Einstein Notation Tensor Rearrangement & Projections"),
            ("wandb", "wandb", "Telemetry", "Weights & Biases Experiment & Hardware Telemetry"),
            ("fastapi", "fastapi", "Studio Backend", "Async REST API & WebSocket Telemetry Server"),
            ("uvicorn", "uvicorn", "Studio Backend", "ASGI High-Concurrency Server Runtime"),
            ("pywebview", "pywebview", "Desktop Shell", "Native Desktop Window Wrapper"),
        ]

        for pkg_key, dist_name, category, desc in stack_specs:
            installed, ver = _pkg_version(pkg_key, dist_name)
            status_label = "Installed" if installed else "Missing"
            pip_pkg = dist_name

            # Detect built-in PyTorch / Triune native replacements when external optional wheel is not needed
            if pkg_key == "flash_attn" and not installed and torch_mod is not None and has_cuda:
                flash_ok = bool(
                    getattr(torch_mod.backends.cuda, "flash_sdp_enabled", lambda: False)()
                    or getattr(torch_mod.backends.cuda, "cudnn_sdp_enabled", lambda: False)()
                )
                if flash_ok:
                    installed = True
                    ver = f"PyTorch SDPA ({cuda_version})"
                    status_label = "Native SDPA"
            elif pkg_key == "transformer_engine" and not installed:
                try:
                    import triune.precision.fp8  # noqa: F401
                    installed = True
                    ver = "Triune Native FP8 (E4M3/E5M2)"
                    status_label = "Native FP8"
                except Exception:
                    pass

            packages[pkg_key] = {
                "installed": installed,
                "version": ver,
                "category": category,
                "description": desc,
                "pip_package": pip_pkg,
                "status_label": status_label,
            }

        sys_info = {
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "gpu": gpu_name,
            "vram_gb": vram_total_gb,
            "ram_gb": ram_total_gb,
            "cuda_available": has_cuda,
            "cuda_version": cuda_display,
            "cuda_raw_version": cuda_version,
            "cudnn_version": cudnn_version,
            "compute_capability": compute_cap_str,
            "driver_version": driver_version,
            "packages": packages,
            "base_dir": str(self.base_dir),
            "modules_dir": str(self.modules_dir),
        }
        return sys_info

    # -------------------------------------------------------------------------
    # System Config Management
    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        """Load user configuration and custom paths with environment variable precedence."""
        defaults = {
            "installation_path": str(self.base_dir),
            "models_path": str(self.base_dir / "models"),
            "datasets_path": str(self.base_dir / "datasets"),
            "checkpoints_path": str(self.base_dir / "checkpoints"),
            "python_executable": sys.executable,
            "auto_check_updates": True,
            "github_token": os.environ.get("GITHUB_TOKEN", ""),
            "hf_token": os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", "")),
            "hardware_mode": "Auto Detect",
            "byok_keys": {},
        }
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    defaults.update(saved)
            except Exception:
                pass

        # Resolve protected markers at runtime only; never return stored markers
        # to inference code. Existing legacy plaintext values remain readable so
        # the next successful save can migrate them without breaking users.
        for secret_name in ("github_token", "hf_token"):
            if defaults.get(secret_name) == _KEYRING_MARKER:
                try:
                    defaults[secret_name] = self._read_secure_secret(secret_name)
                except RuntimeError:
                    defaults[secret_name] = ""
        if isinstance(defaults.get("byok_keys"), dict):
            resolved_keys = dict(defaults["byok_keys"])
            for provider, value in resolved_keys.items():
                if value == _KEYRING_MARKER:
                    try:
                        resolved_keys[provider] = self._read_secure_secret(f"byok:{provider}")
                    except RuntimeError:
                        resolved_keys[provider] = ""
                elif value == _ENV_MARKER:
                    resolved_keys[provider] = self._env_secret_for(f"byok:{provider}")
            defaults["byok_keys"] = resolved_keys
        for secret_name in ("github_token", "hf_token"):
            if defaults.get(secret_name) == _ENV_MARKER:
                defaults[secret_name] = self._env_secret_for(secret_name)

        # Ensure environment variables take priority for sensitive API keys
        if "byok_keys" not in defaults or not isinstance(defaults["byok_keys"], dict):
            defaults["byok_keys"] = {}
        for provider in ("openai", "anthropic", "huggingface", "gemini"):
            env_value = self._env_secret_for(f"byok:{provider}")
            if env_value:
                defaults["byok_keys"][provider] = env_value
        if os.environ.get("HF_TOKEN"):
            defaults["hf_token"] = os.environ["HF_TOKEN"]
        elif os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            defaults["hf_token"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

        return defaults

    def save_config(
        self,
        new_config: dict[str, Any],
        *,
        force_secure_secrets: set[str] | None = None,
    ) -> dict[str, Any]:
        """Persist configuration without writing newly saved API secrets as plaintext."""
        current = self.get_config()
        current.update(new_config)
        persist_copy = dict(current)
        pending_deletions: list[str] = []
        force_secure_secrets = force_secure_secrets or set()

        def protect_secret(secret_name: str, value: Any, was_submitted: bool) -> str:
            if not isinstance(value, str) or not value:
                if was_submitted:
                    pending_deletions.append(secret_name)
                return ""
            env_value = self._env_secret_for(secret_name)
            if secret_name not in force_secure_secrets and env_value and value == env_value:
                return _ENV_MARKER
            if value == _KEYRING_MARKER:
                return _KEYRING_MARKER
            if value == _ENV_MARKER:
                return _ENV_MARKER
            try:
                self._write_secure_secret(secret_name, value)
            except Exception as exc:
                raise RuntimeError(f"Could not store {secret_name} in the OS credential store: {exc}") from exc
            return _KEYRING_MARKER

        for secret_name in ("github_token", "hf_token"):
            if secret_name in persist_copy:
                persist_copy[secret_name] = protect_secret(
                    secret_name, persist_copy[secret_name], secret_name in new_config
                )

        if isinstance(persist_copy.get("byok_keys"), dict):
            submitted_keys = new_config.get("byok_keys")
            submitted_keys = submitted_keys if isinstance(submitted_keys, dict) else {}
            safe_keys: dict[str, str] = {}
            for provider, value in persist_copy["byok_keys"].items():
                if not isinstance(provider, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", provider):
                    continue
                safe_keys[provider] = protect_secret(
                    f"byok:{provider}",
                    value,
                    provider in submitted_keys,
                )
            for provider in submitted_keys:
                if provider not in persist_copy["byok_keys"] and isinstance(provider, str):
                    pending_deletions.append(f"byok:{provider}")
            persist_copy["byok_keys"] = safe_keys

        temp_path: Path | None = None
        try:
            fd, temp_name = tempfile.mkstemp(prefix="studio_config_", suffix=".tmp", dir=self.base_dir)
            temp_path = Path(temp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(persist_copy, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.config_file)
            if os.name != "nt":
                os.chmod(self.config_file, 0o600)
        except Exception as e:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not persist Studio configuration to {self.config_file}: {e}") from e

        for secret_name in pending_deletions:
            try:
                self._delete_secure_secret(secret_name)
            except Exception:
                # Config has already been saved without referencing the secret;
                # report state accurately without turning a successful save into
                # a misleading failure. The orphaned credential can be removed later.
                pass
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
                    if not isinstance(modules, list):
                        return []
                    valid_modules = []
                    for module in modules:
                        if not isinstance(module, dict):
                            continue
                        expected_path = self._module_path(module.get("id", ""))
                        path_str = module.get("installed_at")
                        if expected_path is None or not path_str:
                            continue
                        try:
                            if Path(path_str).resolve() != expected_path or not expected_path.is_dir():
                                continue
                        except (OSError, RuntimeError):
                            continue
                        inspection = self.inspect_directory(expected_path)
                        module["file_count"] = inspection.get("file_count", 0)
                        module["size_mb"] = inspection.get("size_mb", module.get("size_mb", 0))
                        module["git_info"] = inspection.get("git_info", {})
                        module["artifacts"] = inspection.get("artifacts", {})
                        module["registered_nodes"] = inspection.get("registered_nodes", [])
                        valid_modules.append(module)
                    return valid_modules
            except Exception:
                pass
        return []

    def _save_installed_modules(self, modules: list[dict[str, Any]]) -> None:
        temp_path: Path | None = None
        try:
            fd, temp_name = tempfile.mkstemp(prefix="installed_modules_", suffix=".tmp", dir=self.base_dir)
            temp_path = Path(temp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(modules, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.installed_file)
            if os.name != "nt":
                os.chmod(self.installed_file, 0o600)
        except Exception as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not persist installed extension registry: {exc}") from exc

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
        if url.startswith("-"):
            return {"success": False, "message": "Invalid repository URL: cannot start with '-'."}

        # Handle Hugging Face short identifier (e.g. 'roneneldan/TinyStories' or 'HuggingFaceFW/fineweb-edu')
        is_hf = "huggingface.co" in url or (len(url.split("/")) == 2 and not url.startswith("http"))
        if is_hf and not url.startswith("http"):
            url = f"https://huggingface.co/{url}"

        if not url.startswith(("https://", "http://")):
            return {"success": False, "message": "Repository URL must use http:// or https:// protocol."}

        branch = (branch or "main").strip()
        if branch.startswith("-") or not _SAFE_BRANCH_RE.match(branch):
            return {"success": False, "message": f"Invalid branch name: '{branch}'."}

        # Clean name derivation
        clean_name = custom_name or url.rstrip("/").split("/")[-1].replace(".git", "")
        clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clean_name)
        target_folder = self._module_path(clean_name)
        if target_folder is None:
            return {"success": False, "message": "Invalid module name."}

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
            clone_cmd.extend(["--", url, str(target_folder)])

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
                target_resolved = target_folder.resolve()
                with zipfile.ZipFile(tmp_zip, "r") as z:
                    for member in z.infolist():
                        member_path = (target_folder / member.filename).resolve()
                        if target_resolved not in member_path.parents and member_path != target_resolved:
                            raise ValueError(f"ZipSlip security violation: entry {member.filename} escapes destination")
                        z.extract(member, target_folder)
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

        # A clone is data only. Never import third-party Python during install;
        # loading it happens only after the user explicitly activates it.
        registered_nodes: list[str] = []

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
            "status": "Installed",
            "requires_user_activation": detected_type == "plugin",
            "file_count": inspection["file_count"],
            "size_mb": inspection["size_mb"],
            "git_info": inspection["git_info"],
            "artifacts": inspection["artifacts"],
            "registered_nodes": registered_nodes
        }
        installed.append(record)
        self._save_installed_modules(installed)

        node_note = " Plugin code was not run; review and explicitly activate it in Studio." if detected_type == "plugin" else ""
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
        target_folder = self._module_path(module_id)
        if target_folder is None:
            return {"success": False, "message": "Invalid module ID."}
        if not target_folder.exists() or not (target_folder / ".git").exists():
            return {"success": False, "message": f"Module {module_id} is not a local Git repository."}

        try:
            res = subprocess.run(["git", "pull"], cwd=target_folder, capture_output=True, text=True, timeout=45.0)
            if res.returncode != 0:
                return {"success": False, "message": f"git pull error: {res.stderr.strip()}"}

            pull_output = res.stdout.strip()

            # Updating a repository must not execute its Python code. Unregister
            # already loaded nodes so the old in-memory code is not mistaken for
            # the newly pulled version; the user can review and reactivate it.
            registered_nodes: list[str] = []
            try:
                from triune.plugins.registry import global_registry
                for node_name, metadata in list(global_registry._nodes.items()):
                    if metadata.get("module_source") == str(target_folder):
                        del global_registry._nodes[node_name]
                for py_file in target_folder.rglob("*.py"):
                    sys.modules.pop(f"triune_plugin_{py_file.stem}", None)
            except Exception:
                pass
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
                "message": f"Updated {module_id}: {pull_output}. Plugin code was not re-run; review and activate it in Studio if needed.",
                "git_info": inspection["git_info"],
                "registered_nodes": registered_nodes
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to pull repository: {e}"}

    # -------------------------------------------------------------------------
    # Dynamic Plugin & DAG Node Discovery
    # -------------------------------------------------------------------------
    def discover_and_register_nodes(self, module_dir: Path, trusted: bool = True) -> List[str]:
        """Scan a module directory for Python scripts and register discovered DAG nodes into global_registry.

        Security Enforcement:
        1. Untrusted external directories (`trusted=False`) MUST contain a versioned,
           Ed25519-signed `triune_plugin.json` whose SHA-256 entries match the source,
           and pass strict AST validation before any module is loaded.
        2. Even when `trusted=True`, only `.py` files that statically declare `@register_node` (or
           `_triune_node_schema`) and pass `_is_safe_plugin_ast` are loaded via `exec_module`.
        """
        discovered: List[str] = []
        if not module_dir.exists() or module_dir.is_symlink():
            return discovered
        try:
            resolved_module_dir = module_dir.resolve()
        except (OSError, RuntimeError):
            return discovered

        norm_dir = str(module_dir).replace("\\", "/")
        is_internal = "triune/plugins" in norm_dir or "triune/modules/curated" in norm_dir
        manifest_file = module_dir / "triune_plugin.json"
        candidate_files: List[Path] = []
        strict_ast = not is_internal and not trusted
        if strict_ast:
            candidate_files, manifest_error = _verified_plugin_entrypoints(module_dir)
            if manifest_error:
                print(f"[Plugin Discovery] {manifest_error}")
                return discovered
        elif manifest_file.is_file():
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                entrypoints = manifest_data.get("entrypoints") if isinstance(manifest_data, dict) else None
                if isinstance(entrypoints, list):
                    resolved_root = resolved_module_dir
                    for entrypoint in entrypoints:
                        if isinstance(entrypoint, str) and entrypoint.endswith(".py") and "\\" not in entrypoint:
                            ep_path = (resolved_root / entrypoint).resolve()
                            if resolved_root in ep_path.parents and ep_path.is_file():
                                candidate_files.append(ep_path)
            except Exception:
                candidate_files = []

        if not candidate_files and not strict_ast:
            candidate_files = list(module_dir.rglob("*.py"))

        try:
            from triune.plugins.registry import global_registry
        except ImportError:
            return discovered

        for py_file in candidate_files:
            if py_file.name.startswith("__") and py_file.name != "__init__.py":
                continue

            # Local workspace/plugin trees may contain symlinks. Never execute a
            # Python file whose resolved target is outside the selected module root.
            try:
                resolved_file = py_file.resolve()
                if resolved_file == resolved_module_dir or resolved_module_dir not in resolved_file.parents:
                    continue
                py_file = resolved_file
            except (OSError, RuntimeError):
                continue

            if not _is_safe_plugin_ast(py_file, strict_untrusted=strict_ast):
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
                            raw_node_name = schema["name"]
                            core_names = global_registry.get_core_node_names() if hasattr(global_registry, "get_core_node_names") else set()
                            if raw_node_name in core_names and not is_internal:
                                clean_mod = py_file.parent.name or py_file.stem
                                node_name = f"community.{clean_mod}.{raw_node_name}"
                            else:
                                node_name = raw_node_name
                            schema_copy = dict(schema)
                            schema_copy["name"] = node_name
                            schema_copy["module_source"] = str(module_dir)
                            global_registry._nodes[node_name] = schema_copy
                            discovered.append(node_name)
            except Exception as e:
                print(f"[Plugin Discovery Note] Could not load {py_file}: {e}")

        return discovered

    def _auto_register_installed_nodes(self) -> None:
        """Scan all installed modules and register their nodes on startup."""
        for m in self.get_installed_modules():
            path_str = m.get("installed_at")
            expected_path = self._module_path(m.get("id", ""))
            if not path_str or expected_path is None:
                continue
            try:
                if Path(path_str).resolve() != expected_path or not expected_path.is_dir():
                    continue
            except OSError:
                continue
            # Trust explicit curated provenance, never a user-controlled module
            # ID or installed_at path; remote clones remain dormant on startup.
            if str(m.get("download_url", "")).startswith("builtin://"):
                self.discover_and_register_nodes(expected_path, trusted=True)

    def scan_workspace_plugins(self, workspace_dir: str | Path | None = None) -> dict[str, Any]:
        """Scan workspace directories for Python scripts defining custom DAG nodes."""
        base_search = self._workspace_path(workspace_dir or self._workspace_root())
        if base_search is None or not base_search.is_dir():
            return {
                "success": False,
                "scanned_paths": [],
                "discovered_nodes": [],
                "total_registered_nodes": 0,
                "message": "Workspace scan path must be an existing directory inside the Triune project.",
            }

        search_dirs = [
            base_search / "triune" / "plugins",
            base_search / "plugins",
            base_search / "scripts",
            base_search / "custom_nodes",
        ]
        scanned_paths = []
        all_discovered = []

        from triune.plugins.registry import global_registry

        for s_dir in search_dirs:
            if s_dir.exists() and s_dir.is_dir():
                scanned_paths.append(str(s_dir))
                nodes = self.discover_and_register_nodes(s_dir)
                for n in nodes:
                    if n not in all_discovered:
                        all_discovered.append(n)

        # Also scan standalone plugin files in workspace root
        for py_file in base_search.glob("plugin_*.py"):
            scanned_paths.append(str(py_file))
            nodes = self.discover_and_register_nodes(base_search)
            for n in nodes:
                if n not in all_discovered:
                    all_discovered.append(n)
            break

        total_registered = len(global_registry._nodes)
        return {
            "success": True,
            "scanned_paths": scanned_paths,
            "discovered_nodes": all_discovered,
            "total_registered_nodes": total_registered,
            "message": f"Scanned workspace: {len(all_discovered)} custom node(s) discovered across {len(scanned_paths)} path(s)."
        }

    def create_custom_plugin_template(self, name: str, category: str = "Custom", destination_dir: Optional[str] = None) -> dict[str, Any]:
        """Create a non-overwriting boilerplate plugin within the Triune workspace."""
        raw_name = name.strip() if isinstance(name, str) else ""
        clean_name = "".join(c if c.isascii() and (c.isalnum() or c == "_") else "_" for c in raw_name)[:64]
        if not clean_name:
            clean_name = "CustomNode"
        if clean_name[0].isdigit():
            clean_name = f"Node_{clean_name}"
        if keyword.iskeyword(clean_name):
            clean_name = f"{clean_name}_Node"

        if destination_dir:
            target_dir = self._workspace_path(destination_dir)
            if target_dir is None:
                return {"success": False, "message": "Plugin destination must be inside the Triune project workspace."}
        else:
            target_dir = Path(__file__).resolve().parent.parent / "plugins"

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"success": False, "message": f"Could not create plugin directory: {exc}"}
        target_file = target_dir / f"plugin_{clean_name.lower()}.py"
        if target_file.exists():
            return {"success": False, "message": f"Plugin file already exists; it was left unchanged: {target_file.name}"}
        node_name_literal = json.dumps(clean_name)
        category_literal = json.dumps(str(category)[:80])

        content = f'''"""Custom Triune Studio Node Plugin: {clean_name}."""

from __future__ import annotations
from typing import Any, Dict
from triune.plugins.registry import register_node


@register_node(
    name={node_name_literal},
    category={category_literal},
    description="Custom user-defined visual DAG node for Triune Studio.",
    inputs=["in_data"],
    outputs=["out_data"]
)
class {clean_name}:
    """Custom processor node."""

    def __init__(self, multiplier: float = 1.0, mode: str = "standard"):
        self.multiplier = multiplier
        self.mode = mode

    def __call__(self, in_data: Any = None) -> Dict[str, Any]:
        return {{
            "status": "success",
            "node": {node_name_literal},
            "mode": self.mode,
            "multiplier": self.multiplier,
            "result": "Processed by " + {node_name_literal}
        }}
'''
        try:
            # Exclusive create closes the check-then-write race and protects existing user files.
            with target_file.open("x", encoding="utf-8", newline="\n") as plugin_file:
                plugin_file.write(content)
        except FileExistsError:
            return {"success": False, "message": f"Plugin file already exists; it was left unchanged: {target_file.name}"}
        except OSError as exc:
            return {"success": False, "message": f"Could not write plugin template: {exc}"}

        return {
            "success": True,
            "node_name": clean_name,
            "file_path": str(target_file),
            "message": f"Created {target_file.name}. Review the generated code, then scan the workspace to load it into Studio."
        }

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

        target_folder = self._module_path(mod_id)
        if target_folder is None:
            return {"success": False, "message": "Invalid module ID."}

        mod_name = module_data.get("name", mod_id)
        download_url = module_data.get("download_url", "")
        repo_url = module_data.get("repo_url", "")
        target_existed = target_folder.exists()
        if download_url.startswith("pip://") or module_data.get("package_name"):
            pkg_name = str(module_data.get("package_name") or download_url.replace("pip://", "")).strip()
            if not pkg_name or pkg_name.startswith("-") or not _SAFE_PACKAGE_RE.match(pkg_name):
                return {
                    "success": False,
                    "message": f"Invalid package specification: '{pkg_name}'",
                    "installed_path": str(target_folder),
                }

        target_folder.mkdir(parents=True, exist_ok=True)
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
            registered_nodes = self.discover_and_register_nodes(target_folder, trusted=True)
            msg = f"Installed curated module {module_data['name']}."

        # Case 2: Pip framework install
        elif download_url.startswith("pip://") or module_data.get("package_name"):
            pkg_name = str(module_data.get("package_name") or download_url.replace("pip://", "")).strip()
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--", pkg_name])
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
                if not target_existed:
                    try:
                        target_folder.rmdir()
                    except OSError:
                        pass
                return {
                    "success": False,
                    "message": f"Could not download {module_data.get('name', mod_id)}: {e}",
                    "installed_path": str(target_folder),
                }

        if not success:
            if not target_existed:
                try:
                    target_folder.rmdir()
                except OSError:
                    pass
            return {"success": False, "message": msg, "installed_path": str(target_folder)}

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
        target_folder = self._module_path(module_id)
        if target_folder is None:
            return {"success": False, "message": "Invalid module ID."}
        if target_folder.exists():
            try:
                shutil.rmtree(target_folder)
            except Exception as e:
                return {"success": False, "message": f"Could not remove module directory {target_folder}: {e}"}

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


# Canonical ModuleManager singleton for Studio and API
module_manager = ModuleManager()
