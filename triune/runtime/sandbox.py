"""Restricted Python runner for trusted custom code.

Snippets run in a subprocess with a timeout and, on POSIX, an address-space
limit. AST and import restrictions are guardrails, not a security boundary: the
child retains the current user's operating-system permissions. Do not execute
code from untrusted users without stronger OS/container isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict
import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap


class SandboxError(Exception):
    """Raised when restricted code validation or execution fails."""


@dataclass
class SandboxResult:
    """Result of code execution inside PythonSandbox."""
    success: bool
    output: Any
    error: str | None = None
    exec_time_sec: float = 0.0


_ALLOWED_SANDBOX_MODULES = frozenset({
    "math",
    "random",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "json",
    "re",
})

_BANNED_SANDBOX_CALLS = frozenset({
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "type",
    "breakpoint",
    "input",
    "memoryview",
})


def _validate_sandbox_ast(code_str: str) -> None:
    """Reject obvious unsafe syntax and imports; this is not a security boundary."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error in sandbox code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in _ALLOWED_SANDBOX_MODULES:
                    raise SandboxError(f"Disallowed import in sandbox: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            root_mod = (node.module or "").split(".")[0]
            if root_mod not in _ALLOWED_SANDBOX_MODULES:
                raise SandboxError(f"Disallowed import in sandbox: 'from {node.module}'")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SandboxError(f"Disallowed dunder attribute access in sandbox: '{node.attr}'")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                raise SandboxError(f"Disallowed dunder identifier in sandbox: '{node.id}'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BANNED_SANDBOX_CALLS:
                raise SandboxError(f"Disallowed builtin call in sandbox: '{node.func.id}()'")


class PythonSandbox:
    """Subprocess runner for trusted custom nodes, agents, and model code."""

    def __init__(self, timeout: int = 10, max_memory_mb: int = 256):
        if timeout <= 0:
            raise ValueError("timeout must be a positive number of seconds")
        if max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be a positive number")
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute_code(self, code_str: str, locals_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Execute restricted code and return JSON-compatible values from its local scope."""
        try:
            _validate_sandbox_ast(code_str)
        except SandboxError as err:
            return {"success": False, "error": str(err)}

        context = locals_dict or {}
        try:
            serialized_context = json.dumps(context)
        except (TypeError, ValueError) as err:
            return {"success": False, "error": f"Sandbox inputs must be JSON serializable: {err}"}
        allowed_list = list(_ALLOWED_SANDBOX_MODULES)
        memory_limit_setup = []
        if os.name == "posix":
            memory_limit_bytes = self.max_memory_mb * 1024 * 1024
            memory_limit_setup = [
                "import resource",
                f"_memory_limit = {memory_limit_bytes}",
                "resource.setrlimit(resource.RLIMIT_AS, (_memory_limit, _memory_limit))",
            ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            wrapper = [
                "import json",
                "import builtins as _builtins",
                *memory_limit_setup,
                f"_ALLOWED_MODS = set({json.dumps(allowed_list)})",
                "def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):",
                "    if level != 0 or name.split('.')[0] not in _ALLOWED_MODS:",
                "        raise ImportError(f'Import of {name!r} is not permitted in PythonSandbox')",
                "    return _builtins.__import__(name, globals, locals, fromlist, level)",
                'safe_builtins = {"abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate, "filter": filter, "float": float, "int": int, "len": len, "list": list, "map": map, "max": max, "min": min, "pow": pow, "range": range, "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip, "print": print, "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError, "IndexError": IndexError, "NameError": NameError, "AttributeError": AttributeError, "ZeroDivisionError": ZeroDivisionError, "True": True, "False": False, "None": None, "__import__": _safe_import}',
                'safe_globals = {"__builtins__": safe_builtins}',
                f"_ctx = json.loads({json.dumps(serialized_context)})",
                "safe_locals = dict(_ctx)",
                f"_code = {json.dumps(code_str)}",
                "try:",
                "    exec(_code, safe_globals, safe_locals)",
                "    _exported = {}",
                "    for _k, _v in safe_locals.items():",
                "        if not _k.startswith('_'):",
                "            try:",
                "                json.dumps(_v)",
                "                _exported[_k] = _v",
                "            except TypeError:",
                "                pass",
                '    _exported["success"] = True',
                "    print(json.dumps(_exported))",
                "except Exception as e:",
                '    print(json.dumps({"success": False, "error": str(e)}))',
            ]
            f.write("\n".join(wrapper))
            temp_path = f.name

        minimal_env = {
            k: v
            for k, v in os.environ.items()
            if k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-u", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
                env=minimal_env,
            )
            stdout = result.stdout.strip()
            if stdout:
                try:
                    return json.loads(stdout.split("\n")[-1])
                except json.JSONDecodeError:
                    if result.returncode != 0:
                        return {"success": False, "error": result.stderr or stdout}
                    return {"success": True, "result": stdout, "stdout": stdout, "stderr": result.stderr}
            return {"success": False, "error": result.stderr or "No output", "stdout": "", "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Execution timed out after {self.timeout}s"}
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def execute_function(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a source-backed, JSON-compatible function in the restricted subprocess."""
        try:
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError) as error:
            return {"success": False, "error": f"Function source is unavailable for isolated execution: {error}"}

        fn_name = getattr(fn, "__name__", "")
        if not fn_name.isidentifier():
            return {"success": False, "error": "Only named Python functions can run in the restricted subprocess"}

        call_code = f"{src}\nresult = {fn_name}(*_sandbox_args, **_sandbox_kwargs)"
        result = self.execute_code(
            call_code,
            {"_sandbox_args": list(args), "_sandbox_kwargs": kwargs},
        )
        if not result.get("success"):
            return result
        if "result" not in result:
            return {"success": False, "error": "Function result must be JSON serializable"}
        return {"success": True, "result": result["result"]}
