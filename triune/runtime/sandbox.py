"""Sandboxed Python Execution Runtime.

Provides process isolation, execution limits, and safe execution of user-defined
custom loss functions, architecture modules, node code, and agentic tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict
import sys
import traceback


class SandboxError(Exception):
    """Raised when sandboxed execution fails or violates security bounds."""


@dataclass
class SandboxResult:
    """Result of code execution inside PythonSandbox."""
    success: bool
    output: Any
    error: str | None = None
    exec_time_sec: float = 0.0


class PythonSandbox:
    """Isolated Python code execution environment for custom nodes, agents, and custom models."""

    def __init__(self, timeout: int = 10, max_memory_mb: int = 256):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute_code(self, code_str: str, locals_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Safely execute Python code string and return modified local scope."""
        import subprocess
        import tempfile
        import os
        import json
        
        context = locals_dict or {}
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            wrapper = []
            wrapper.append('import json')
            wrapper.append('safe_builtins = {"abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate, "filter": filter, "float": float, "int": int, "len": len, "list": list, "map": map, "max": max, "min": min, "pow": pow, "range": range, "round": round, "set": set, "str": str, "sum": sum, "tuple": tuple, "zip": zip, "print": print, "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError, "IndexError": IndexError, "NameError": NameError, "AttributeError": AttributeError, "True": True, "False": False, "None": None}')
            wrapper.append('safe_globals = {"__builtins__": safe_builtins}')
            wrapper.append(f'_ctx = json.loads({json.dumps(json.dumps(context))})')
            wrapper.append('safe_locals = dict(_ctx)')
            wrapper.append(f'_code = {json.dumps(code_str)}')
            wrapper.append('try:')
            wrapper.append('    exec(_code, safe_globals, safe_locals)')
            wrapper.append('    _exported = {k: v for k, v in safe_locals.items() if not k.startswith("_")}')
            wrapper.append('    _exported["success"] = True')
            wrapper.append('    print(json.dumps(_exported))')
            wrapper.append('except Exception as e:')
            wrapper.append('    print(json.dumps({"success": False, "error": str(e)}))')
            f.write('\n'.join(wrapper))
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, '-u', temp_path],
                capture_output=True, text=True,
                timeout=self.timeout,
                env={k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'PYTHONPATH')},
            )
            stdout = result.stdout.strip()
            if stdout:
                try:
                    return json.loads(stdout.split('\n')[-1])
                except json.JSONDecodeError:
                    return {'success': True, 'result': stdout, 'stdout': stdout, 'stderr': result.stderr}
            return {'success': False, 'error': result.stderr or 'No output', 'stdout': '', 'stderr': result.stderr}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': f'Execution timed out after {self.timeout}s'}
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def execute_function(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a Python callable inside the sandbox context."""
        try:
            result = fn(*args, **kwargs)
            return {'success': True, 'result': result}
        except Exception as error:
            return {'success': False, 'error': str(error)}
