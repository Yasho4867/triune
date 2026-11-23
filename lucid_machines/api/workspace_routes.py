"""Project Workspace IDE & Autonomous Agentic Loop API Routes for Polysom Studio."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from fastapi import APIRouter, Body, HTTPException, Query
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None
    Body = None
    HTTPException = None
    Query = None
    JSONResponse = None
    StreamingResponse = None
    BaseModel = object
    Field = None

from lucid_machines.agents.engine import AgentHarness
from lucid_machines.agents.loops import (
    AgentLoopManager,
    resolve_executable_argv,
    run_agent_loop,
    validate_terminal_command,
)
from lucid_machines.agents.tools import _has_traversal_syntax, _is_protected_workspace_path

_DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB
_MAX_SEARCH_FILE_BYTES = 1 * 1024 * 1024  # 1 MB
_MAX_TERMINAL_OUTPUT_BYTES = 64 * 1024  # 64 KB

_TREE_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        "studio_env",
        "checkpoints",
        "data",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
    }
)

_SEARCH_SKIP_DIRS: frozenset[str] = _TREE_SKIP_DIRS | frozenset({"dist", "build", ".mypy_cache"})

_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pt",
        ".pth",
        ".safetensors",
        ".gguf",
        ".bin",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".whl",
        ".pyc",
        ".pyo",
        ".o",
        ".obj",
        ".lib",
        ".a",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".mp3",
        ".wav",
        ".mp4",
    }
)

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".jsx": "javascriptreact",
    ".json": "json",
    ".jsonl": "json",
    ".css": "css",
    ".scss": "scss",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".bat": "bat",
    ".cmd": "bat",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".cs": "csharp",
    ".rs": "rust",
    ".go": "go",
    ".sql": "sql",
    ".xml": "xml",
    ".svg": "xml",
    ".txt": "plaintext",
    ".env": "dotenv",
    ".gitignore": "plaintext",
}


def _detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[ext]
    if path.name.lower() in {"dockerfile", "makefile"}:
        return path.name.lower()
    return "plaintext"


def _is_binary_bytes(raw: bytes, ext: str = "") -> bool:
    if ext.lower() in _BINARY_EXTENSIONS:
        return True
    sample = raw[:8192]
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError:
        # Check if just cut mid-UTF8 sequence at 8192 boundary
        for trim in range(1, min(4, len(sample))):
            try:
                sample[:-trim].decode("utf-8")
                return False
            except UnicodeDecodeError:
                continue
        return True


def _is_forbidden_workspace_root(resolved: Path) -> bool:
    """Block OS root directories and sensitive system directories from being opened as workspaces."""
    if resolved.parent == resolved:
        return True
    if os.name == "nt":
        win_dir = Path(os.environ.get("WINDIR", r"C:\Windows")).resolve(strict=False)
        sys_roots = [win_dir]
        for env_k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
            val = os.environ.get(env_k)
            if val:
                sys_roots.append(Path(val).resolve(strict=False))
        for sr in sys_roots:
            if resolved == sr or resolved.is_relative_to(sr):
                return True
    else:
        posix_forbidden = (
            "/etc",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/var",
            "/proc",
            "/sys",
            "/dev",
            "/boot",
            "/System",
        )
        for p in posix_forbidden:
            fp = Path(p)
            if resolved == fp or resolved.is_relative_to(fp):
                return True

    extra = os.environ.get("LUCID_MACHINES_ALLOWED_WORKSPACE_ROOTS", "").strip()
    if extra:
        allowed = [Path(x.strip()).resolve(strict=False) for x in extra.split(os.pathsep) if x.strip()]
        if allowed and not any(resolved == a or resolved.is_relative_to(a) for a in allowed):
            return True
    return False


def _is_headless_environment() -> bool:
    """Return True when running in tests, CI, or without an interactive desktop display."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if os.environ.get("LUCID_MACHINES_TESTING") == "1" or os.environ.get("TRIUNE_TESTING") == "1":
        return True
    if os.environ.get("LUCID_MACHINES_HEADLESS", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return True
    if os.name == "nt":
        sess = os.environ.get("SESSIONNAME", "").strip().lower()
        if sess == "services":
            return True
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_native_folder_dialog(initial_dir: str) -> Optional[str]:
    """Open local OS folder dialog only when non-headless; returns None on cancel or headless."""
    if _is_headless_environment():
        return None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked = filedialog.askdirectory(initialdir=initial_dir, title="Open Workspace Folder — Polysom Studio")
        root.destroy()
        return str(picked) if picked else ""
    except Exception:
        return None


def _run_git_info(root: Path, target_rel: Optional[str] = None) -> dict[str, Any]:
    """Collect git repository status safely with shell=False."""
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            shell=False,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if inside.returncode != 0 or inside.stdout.strip().lower() != "true":
            return {"is_git": False, "branch": "", "dirty_count": 0, "entries": [], "diff": ""}

        br_proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            shell=False,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        branch = br_proc.stdout.strip() if br_proc.returncode == 0 else ""
        if not branch or branch == "HEAD":
            sym_proc = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                shell=False,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if sym_proc.returncode == 0 and sym_proc.stdout.strip():
                branch = sym_proc.stdout.strip()

        status_cmd = ["git", "status", "--porcelain"]
        if target_rel:
            status_cmd.extend(["--", target_rel])
        st_proc = subprocess.run(
            status_cmd,
            shell=False,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        entries: list[dict[str, str]] = []
        if st_proc.returncode == 0:
            for line in st_proc.stdout.splitlines():
                if len(line) >= 4:
                    entries.append({"status": line[:2].strip() or line[:2], "path": line[3:].strip()})

        diff_cmd = ["git", "diff"]
        if target_rel:
            diff_cmd.extend(["--", target_rel])
        diff_proc = subprocess.run(
            diff_cmd,
            shell=False,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=8,
        )
        diff_txt = (diff_proc.stdout or "")[:_MAX_TERMINAL_OUTPUT_BYTES]
        return {
            "is_git": True,
            "branch": branch,
            "dirty_count": len(entries),
            "entries": entries,
            "diff": diff_txt,
        }
    except Exception:
        return {"is_git": False, "branch": "", "dirty_count": 0, "entries": [], "diff": ""}


class WorkspaceManager:
    """Thread-safe active workspace root & AgentHarness/AgentLoopManager binding."""

    def __init__(self, initial_root: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        root = (initial_root or _DEFAULT_WORKSPACE_ROOT).resolve(strict=False)
        self._root: Path = root
        self._recent: list[str] = [str(root)]
        self.loop_manager = AgentLoopManager(workspace=root)
        self._harness: Optional[AgentHarness] = None

    @property
    def root(self) -> Path:
        with self._lock:
            return self._root

    @property
    def recent(self) -> list[str]:
        with self._lock:
            return list(self._recent)

    def get_harness(self, *, model: Optional[str] = None, completion_fn: Any = None) -> AgentHarness:
        with self._lock:
            if self._harness is None:
                self._harness = AgentHarness(
                    workspace=self._root,
                    model=model,
                    allow_shell=False,
                    allow_writes=True,
                    completion_fn=completion_fn,
                )
            else:
                if self._harness.workspace != self._root:
                    self._harness.rebind_workspace(self._root)
                if model:
                    self._harness.model = model
                if completion_fn is not None:
                    self._harness.completion_fn = completion_fn
            return self._harness

    def open_workspace(self, raw_path: str) -> dict[str, Any]:
        clean = str(raw_path or "").strip()
        if not clean:
            raise ValueError("Workspace path cannot be empty.")
        if "\x00" in clean or clean.startswith(("\\\\", "//")):
            raise PermissionError("UNC or null-byte workspace path rejected.")

        cand = Path(clean).expanduser()
        if not cand.exists():
            raise FileNotFoundError(f"Directory does not exist: {clean}")
        resolved = cand.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {clean}")
        if _is_forbidden_workspace_root(resolved):
            raise PermissionError(f"Opening system or restricted root '{resolved}' is forbidden.")

        with self._lock:
            self._root = resolved
            s_root = str(resolved)
            self._recent = [s_root, *[p for p in self._recent if p != s_root]][:15]
            self.loop_manager.rebind_workspace(resolved)
            if self._harness is not None:
                self._harness.rebind_workspace(resolved)
            recent_copy = list(self._recent)

        git_info = _run_git_info(resolved)
        return {
            "ok": True,
            "root": str(resolved),
            "branch": git_info["branch"],
            "dirty_count": git_info["dirty_count"],
            "is_git": git_info["is_git"],
            "recent": recent_copy,
        }

    def state(self) -> dict[str, Any]:
        root = self.root
        git_info = _run_git_info(root)
        return {
            "root": str(root),
            "recent": self.recent,
            "branch": git_info["branch"],
            "dirty_count": git_info["dirty_count"],
            "is_git": git_info["is_git"],
        }

    def resolve_confined(self, raw_path: str, *, allow_root: bool = False) -> Path:
        raw = str(raw_path or "").strip()
        root_resolved = self.root.resolve(strict=True)
        if not raw or raw == ".":
            if allow_root:
                return root_resolved
            raise PermissionError("Target path cannot be empty or workspace root.")
        if _has_traversal_syntax(raw):
            raise PermissionError(f"Path traversal syntax rejected: '{raw_path}'")

        p_raw = Path(raw)
        if p_raw.is_absolute():
            cand = p_raw.resolve(strict=False)
            parent_cand = p_raw.parent.resolve(strict=False)
        else:
            joined = root_resolved / raw
            cand = joined.resolve(strict=False)
            parent_cand = joined.parent.resolve(strict=False)

        if not (cand == root_resolved or cand.is_relative_to(root_resolved)):
            raise PermissionError(f"Path '{raw_path}' escapes workspace root '{root_resolved}'.")
        if not (parent_cand == root_resolved or parent_cand.is_relative_to(root_resolved)):
            raise PermissionError(f"Parent of '{raw_path}' escapes workspace root '{root_resolved}'.")
        if cand == root_resolved and not allow_root:
            raise PermissionError("Operation cannot target workspace root itself.")
        return cand


workspace_manager = WorkspaceManager()


if HAS_FASTAPI:
    workspace_router = APIRouter()

    class WorkspaceOpenRequest(BaseModel):
        path: str

    class WorkspaceFileWriteRequest(BaseModel):
        path: str
        content: str

    class WorkspaceFileCreateRequest(BaseModel):
        path: str
        content: str = ""
        is_dir: bool = False

    class WorkspaceDeleteRequest(BaseModel):
        path: str

    class WorkspaceSearchRequest(BaseModel):
        query: str
        regex: bool = False
        case_sensitive: bool = False
        max_results: int = Field(default=200, ge=1, le=500)

    class WorkspaceTerminalRunRequest(BaseModel):
        command: Union[str, List[str]]
        timeout: int = Field(default=15, ge=1, le=30)

    class AgentLoopStreamRequest(BaseModel):
        goal: str
        mode: str = "react"
        autonomy: str = "supervised"
        active_file: Optional[str] = None
        max_steps: int = Field(default=10, ge=1, le=50)
        model: Optional[str] = None
        persona: Optional[str] = "auto"
        reasoning_depth: Optional[str] = "auto"
        compact_char_budget: Optional[int] = Field(default=None, ge=400, le=200000)

    class AgentApprovalRequest(BaseModel):
        approved: bool
        reason: Optional[str] = None

    class SupervisorStartRequest(BaseModel):
        poll_interval_s: float = Field(default=2.0, ge=0.1, le=60.0)
        loss_spike_ratio: float = Field(default=2.0, ge=1.1, le=100.0)
        grad_norm_threshold: float = Field(default=50.0, ge=1.0, le=10000.0)
        auto_checkpoint: bool = True

    def _error_response(status_code: int, detail: str) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"status": "error", "detail": detail})

    @workspace_router.get("/v1/workspace/state")
    async def get_workspace_state() -> Dict[str, Any]:
        return workspace_manager.state()

    @workspace_router.post("/v1/workspace/open")
    async def open_workspace_endpoint(req: WorkspaceOpenRequest) -> Any:
        try:
            return workspace_manager.open_workspace(req.path)
        except PermissionError as exc:
            return _error_response(403, str(exc))
        except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
            return _error_response(400, str(exc))

    @workspace_router.post("/v1/workspace/pick")
    async def pick_workspace_endpoint() -> Dict[str, Any]:
        current_root = str(workspace_manager.root)
        if _is_headless_environment():
            return {"ok": False, "headless": True, "root": current_root}
        picked = await asyncio.to_thread(_open_native_folder_dialog, current_root)
        if picked is None:
            return {"ok": False, "headless": True, "root": current_root}
        if picked == "":
            return {"ok": False, "headless": False, "cancelled": True, "root": current_root}
        try:
            opened = workspace_manager.open_workspace(picked)
            return {"headless": False, **opened}
        except Exception as exc:
            return {"ok": False, "headless": False, "detail": str(exc), "root": current_root}

    def _build_tree_entries(root_resolved: Path, directory: Path, depth: int) -> list[dict[str, Any]]:
        if depth <= 0 or not directory.is_dir():
            return []
        entries: list[dict[str, Any]] = []
        try:
            children_iter = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return entries

        for item in children_iter:
            if item.name in _TREE_SKIP_DIRS:
                continue
            try:
                resolved_item = item.resolve(strict=False)
                if not resolved_item.is_relative_to(root_resolved):
                    continue
            except (OSError, RuntimeError, ValueError):
                continue

            rel = str(resolved_item.relative_to(root_resolved)).replace("\\", "/")
            if item.is_dir():
                sub_children = _build_tree_entries(root_resolved, resolved_item, depth - 1) if depth > 1 else []
                entries.append(
                    {
                        "name": item.name,
                        "path": str(resolved_item),
                        "rel_path": rel,
                        "type": "dir",
                        "size": None,
                        "children": sub_children,
                    }
                )
            elif item.is_file():
                try:
                    sz = item.stat().st_size
                except OSError:
                    sz = 0
                entries.append(
                    {
                        "name": item.name,
                        "path": str(resolved_item),
                        "rel_path": rel,
                        "type": "file",
                        "size": sz,
                        "children": [],
                    }
                )
        return entries

    @workspace_router.get("/v1/workspace/tree")
    async def get_workspace_tree(path: str = "", depth: int = Query(default=2, ge=1, le=6)) -> Any:
        root_resolved = workspace_manager.root.resolve(strict=True)
        try:
            target = workspace_manager.resolve_confined(path, allow_root=True)
        except PermissionError as exc:
            return _error_response(403, str(exc))
        if not target.exists():
            return _error_response(404, f"Directory not found: {path}")
        if not target.is_dir():
            return _error_response(400, f"Path is not a directory: {path}")

        entries = _build_tree_entries(root_resolved, target, depth)
        return {"root": str(root_resolved), "path": str(target), "entries": entries}

    @workspace_router.get("/v1/workspace/file")
    async def read_workspace_file(path: str = Query(...)) -> Any:
        root_resolved = workspace_manager.root.resolve(strict=True)
        try:
            target = workspace_manager.resolve_confined(path, allow_root=False)
        except PermissionError as exc:
            return _error_response(403, str(exc))

        if not target.exists():
            return _error_response(404, f"File not found: {path}")
        if not target.is_file():
            return _error_response(400, f"Path is not a regular file: {path}")

        size = target.stat().st_size
        if size > _MAX_FILE_BYTES:
            return _error_response(413, f"File exceeds 2MB editor limit ({size} bytes).")

        raw = target.read_bytes()
        if _is_binary_bytes(raw, target.suffix):
            return _error_response(400, f"Binary files cannot be opened in the text editor: {target.name}")

        content = raw.decode("utf-8")
        rel_path = str(target.relative_to(root_resolved)).replace("\\", "/")
        return {
            "path": str(target),
            "rel_path": rel_path,
            "content": content,
            "size": size,
            "language": _detect_language(target),
        }

    def _atomic_write_text(target: Path, content: str) -> int:
        data = content.encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".polysom_", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return len(data)

    @workspace_router.post("/v1/workspace/file")
    async def write_workspace_file(req: WorkspaceFileWriteRequest) -> Any:
        root_resolved = workspace_manager.root.resolve(strict=True)
        try:
            target = workspace_manager.resolve_confined(req.path, allow_root=False)
        except PermissionError as exc:
            return _error_response(403, str(exc))

        if _is_protected_workspace_path(root_resolved, target):
            return _error_response(
                403, f"Security restriction: protected workspace path cannot be modified ({req.path})."
            )

        encoded_len = len(req.content.encode("utf-8"))
        if encoded_len > _MAX_FILE_BYTES:
            return _error_response(413, "Content exceeds 2MB workspace write limit.")

        old_content = ""
        if target.exists():
            if target.is_dir():
                return _error_response(400, f"Cannot overwrite directory as a file: {req.path}")
            old_raw = target.read_bytes()
            if _is_binary_bytes(old_raw, target.suffix):
                return _error_response(400, "Cannot overwrite binary file via text editor endpoint.")
            old_content = old_raw.decode("utf-8")

        rel_path = str(target.relative_to(root_resolved)).replace("\\", "/")
        diff_lines = list(
            difflib.unified_diff(
                old_content.splitlines(),
                req.content.splitlines(),
                fromfile=f"a/{rel_path}" if old_content else "/dev/null",
                tofile=f"b/{rel_path}",
                lineterm="",
            )
        )
        bytes_written = _atomic_write_text(target, req.content)
        return {
            "ok": True,
            "path": str(target),
            "rel_path": rel_path,
            "diff": "\n".join(diff_lines),
            "bytes_written": bytes_written,
        }

    @workspace_router.post("/v1/workspace/file/create")
    async def create_workspace_entry(req: WorkspaceFileCreateRequest) -> Any:
        root_resolved = workspace_manager.root.resolve(strict=True)
        try:
            target = workspace_manager.resolve_confined(req.path, allow_root=False)
        except PermissionError as exc:
            return _error_response(403, str(exc))

        if _is_protected_workspace_path(root_resolved, target):
            return _error_response(403, f"Security restriction: cannot create inside protected directory ({req.path}).")
        if target.exists():
            return _error_response(409, f"Path already exists: {req.path}")

        rel_path = str(target.relative_to(root_resolved)).replace("\\", "/")
        if req.is_dir:
            target.mkdir(parents=True, exist_ok=False)
        else:
            _atomic_write_text(target, req.content)
        return {"ok": True, "path": str(target), "rel_path": rel_path}

    @workspace_router.delete("/v1/workspace/file")
    async def delete_workspace_entry(
        path: Optional[str] = Query(default=None),
        body: Optional[WorkspaceDeleteRequest] = Body(default=None),
    ) -> Any:
        raw_path = path if path is not None else (body.path if body is not None else "")
        root_resolved = workspace_manager.root.resolve(strict=True)
        try:
            target = workspace_manager.resolve_confined(raw_path, allow_root=False)
        except PermissionError as exc:
            return _error_response(403, str(exc))

        if _is_protected_workspace_path(root_resolved, target):
            return _error_response(
                403, f"Security restriction: protected workspace path cannot be deleted ({raw_path})."
            )
        if not target.exists() and not target.is_symlink():
            return _error_response(404, f"Path not found: {raw_path}")

        rel_path = str(target.relative_to(root_resolved)).replace("\\", "/")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
        return {"ok": True, "deleted": rel_path}

    @workspace_router.post("/v1/workspace/search")
    async def search_workspace_endpoint(req: WorkspaceSearchRequest) -> Any:
        query = str(req.query or "")
        if not query:
            return {"query": query, "matches": [], "files": []}

        root_resolved = workspace_manager.root.resolve(strict=True)
        flags = 0 if req.case_sensitive else re.IGNORECASE
        try:
            rx = re.compile(query if req.regex else re.escape(query), flags)
        except re.error as exc:
            return _error_response(400, f"Invalid search regex: {exc}")

        max_res = max(1, min(int(req.max_results), 500))
        matches: list[dict[str, Any]] = []
        matched_files: list[str] = []
        seen_files: set[str] = set()

        for cur_dir, dirnames, filenames in os.walk(root_resolved):
            dirnames[:] = sorted(d for d in dirnames if d not in _SEARCH_SKIP_DIRS)
            for fname in sorted(filenames):
                fpath = Path(cur_dir) / fname
                try:
                    resolved_f = fpath.resolve(strict=False)
                    if not resolved_f.is_relative_to(root_resolved):
                        continue
                    rel = str(resolved_f.relative_to(root_resolved)).replace("\\", "/")
                except (OSError, RuntimeError, ValueError):
                    continue

                if rx.search(rel) or rx.search(fname):
                    if rel not in seen_files:
                        seen_files.add(rel)
                        matched_files.append(rel)

                if resolved_f.suffix.lower() in _BINARY_EXTENSIONS:
                    continue
                try:
                    if resolved_f.stat().st_size > _MAX_SEARCH_FILE_BYTES:
                        continue
                    raw = resolved_f.read_bytes()
                    if _is_binary_bytes(raw, resolved_f.suffix):
                        continue
                    text = raw.decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

                for idx, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        matches.append({"rel_path": rel, "line": idx, "text": line[:300]})
                        if rel not in seen_files:
                            seen_files.add(rel)
                            matched_files.append(rel)
                        if len(matches) >= max_res:
                            return {"query": query, "matches": matches, "files": matched_files}

        return {"query": query, "matches": matches, "files": matched_files}

    @workspace_router.get("/v1/workspace/git")
    async def get_workspace_git(path: str = "") -> Any:
        root_resolved = workspace_manager.root.resolve(strict=True)
        rel_target: Optional[str] = None
        if path and path.strip() not in {"", "."}:
            try:
                target = workspace_manager.resolve_confined(path, allow_root=True)
            except PermissionError as exc:
                return _error_response(403, str(exc))
            if target != root_resolved:
                rel_target = str(target.relative_to(root_resolved)).replace("\\", "/")
        return _run_git_info(root_resolved, target_rel=rel_target)

    @workspace_router.post("/v1/workspace/terminal/run")
    async def run_workspace_terminal(req: WorkspaceTerminalRunRequest) -> Any:
        root_resolved = workspace_manager.root.resolve(strict=True)
        try:
            argv = validate_terminal_command(req.command, root_resolved)
        except PermissionError as exc:
            return _error_response(403, str(exc))

        exec_argv = resolve_executable_argv(argv)
        timeout_s = max(1, min(30, int(req.timeout)))
        t0 = time.perf_counter()

        def _exec() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                exec_argv,
                shell=False,
                cwd=str(root_resolved),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )

        try:
            proc = await asyncio.to_thread(_exec)
            duration_ms = int(round((time.perf_counter() - t0) * 1000.0))
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": (proc.stdout or "")[:_MAX_TERMINAL_OUTPUT_BYTES],
                "stderr": (proc.stderr or "")[:_MAX_TERMINAL_OUTPUT_BYTES],
                "duration_ms": duration_ms,
                "command": argv,
            }
        except subprocess.TimeoutExpired:
            duration_ms = int(round((time.perf_counter() - t0) * 1000.0))
            return {
                "ok": False,
                "exit_code": 124,
                "stdout": "",
                "stderr": f"Command timed out after {timeout_s}s.",
                "duration_ms": duration_ms,
                "command": argv,
            }
        except FileNotFoundError:
            duration_ms = int(round((time.perf_counter() - t0) * 1000.0))
            return {
                "ok": False,
                "exit_code": 127,
                "stdout": "",
                "stderr": f"Executable not found on system PATH: {argv[0]}",
                "duration_ms": duration_ms,
                "command": argv,
            }

    def _resolve_loop_backend(req_model: Optional[str]) -> tuple[Optional[str], Any]:
        if req_model:
            return req_model, None
        try:
            from .routes import pytorch_state

            if getattr(pytorch_state, "weights_loaded", False):
                if getattr(pytorch_state, "active_runtime", "pytorch") == "ollama" and pytorch_state.ollama_model_tag:
                    return f"ollama:{pytorch_state.ollama_model_tag}", None

                def _local_pt_completion(messages, **_kwargs):
                    prompt_txt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages[-6:])
                    with pytorch_state.worker_lock:
                        res = pytorch_state.generate_text(
                            prompt_txt,
                            max_new_tokens=256,
                            temperature=float(_kwargs.get("temperature", 0.35)),
                            prompt_is_formatted=False,
                        )
                    return {"content": res["text"], "tool_calls": []}

                return pytorch_state.active_model_id, _local_pt_completion
        except Exception:
            pass
        return None, None

    @workspace_router.post("/v1/agents/loop/stream")
    async def stream_agent_loop(req: AgentLoopStreamRequest) -> StreamingResponse:
        run_id = f"loop-{uuid.uuid4().hex[:10]}"
        cancel_ev = workspace_manager.loop_manager.register_run(run_id, req.mode, req.goal)
        chosen_model, local_completion = _resolve_loop_backend(req.model)
        harness = AgentHarness(
            workspace=workspace_manager.root,
            model=chosen_model,
            allow_shell=False,
            allow_writes=True,
            completion_fn=local_completion,
        )
        if req.persona and req.persona.lower() != "auto":
            harness.state_controller.shift(
                persona=req.persona,
                reasoning_depth=None
                if (not req.reasoning_depth or req.reasoning_depth.lower() == "auto")
                else req.reasoning_depth,
                reason="loop stream request",
            )
        elif req.reasoning_depth and req.reasoning_depth.lower() != "auto":
            harness.state_controller.shift(reasoning_depth=req.reasoning_depth, reason="loop stream request")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()

        def _emit(ev: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ev)

        def _worker() -> None:
            try:
                loop_kwargs: dict[str, Any] = {}
                if req.compact_char_budget is not None:
                    loop_kwargs["compact_char_budget"] = req.compact_char_budget
                run_agent_loop(
                    harness=harness,
                    goal=req.goal,
                    mode=req.mode,
                    autonomy=req.autonomy,
                    active_file=req.active_file,
                    max_steps=req.max_steps,
                    run_id=run_id,
                    emit=_emit,
                    cancel_event=cancel_ev,
                    approval_manager=workspace_manager.loop_manager.approvals,
                    supervisor=workspace_manager.loop_manager.supervisor,
                    **loop_kwargs,
                )
            finally:
                workspace_manager.loop_manager.mark_finished(run_id)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_worker, daemon=True, name=f"AgentLoop-{run_id}").start()

        async def _sse_gen():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, default=str)}\n\n"

        return StreamingResponse(
            _sse_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @workspace_router.post("/v1/agents/approve/{action_id}")
    async def approve_agent_action(action_id: str, req: AgentApprovalRequest) -> Any:
        try:
            return workspace_manager.loop_manager.approvals.resolve(action_id, approved=req.approved, reason=req.reason)
        except KeyError as exc:
            return _error_response(404, str(exc))

    @workspace_router.post("/v1/agents/loop/cancel/{run_id}")
    async def cancel_agent_loop(run_id: str) -> Any:
        ok = workspace_manager.loop_manager.cancel_run(run_id)
        if not ok:
            return _error_response(404, f"Unknown agent loop run_id: {run_id}")
        return {"ok": True, "run_id": run_id, "cancelled": True}

    @workspace_router.post("/v1/agents/supervisor/start")
    async def start_training_supervisor(
        req: Optional[SupervisorStartRequest] = Body(default=None),
    ) -> Dict[str, Any]:
        cfg = req or SupervisorStartRequest()
        return await workspace_manager.loop_manager.supervisor.start(
            poll_interval_s=cfg.poll_interval_s,
            loss_spike_ratio=cfg.loss_spike_ratio,
            grad_norm_threshold=cfg.grad_norm_threshold,
            auto_checkpoint=cfg.auto_checkpoint,
        )

    @workspace_router.post("/v1/agents/supervisor/stop")
    async def stop_training_supervisor() -> Dict[str, Any]:
        return await workspace_manager.loop_manager.supervisor.stop()

    @workspace_router.get("/v1/agents/supervisor/status")
    async def get_training_supervisor_status() -> Dict[str, Any]:
        return workspace_manager.loop_manager.supervisor.status()
