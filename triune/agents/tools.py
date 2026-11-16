"""Agent tool abstraction, reliability-tracked registry, and integrated Triune toolbelt."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from triune.runtime.sandbox import PythonSandbox

_ALLOWED_LOCAL_HTTP_PORTS: frozenset[int] = frozenset({11434, *range(8000, 8021)})


def validate_http_url(url: str, *, allow_local_services: bool = False) -> str:
    """Validate HTTP(S) URLs against SSRF targets (private, metadata, link-local, arbitrary loopback)."""
    clean = str(url or "").strip()
    parsed = urllib.parse.urlparse(clean)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are permitted.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded user credentials are not permitted.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL is missing a valid hostname.")

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve hostname '{hostname}': {exc}") from exc

    if not addr_infos:
        raise ValueError(f"Hostname '{hostname}' resolved to no addresses.")

    for info in addr_infos:
        raw_ip = str(info[4][0]).split("%")[0]
        ip = ipaddress.ip_address(raw_ip)
        if ip.is_loopback:
            if not (allow_local_services and port in _ALLOWED_LOCAL_HTTP_PORTS):
                raise PermissionError(f"SSRF guard blocked loopback access to {hostname}:{port}.")
            continue
        if (
            ip.is_link_local
            or ip.is_private
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
            or not ip.is_global
        ):
            raise PermissionError(f"SSRF guard blocked non-public IP {ip} for host '{hostname}'.")

    return clean


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforce SSRF URL validation across every HTTP redirect hop."""

    def __init__(self, *, allow_local_services: bool = False, max_redirects: int = 3) -> None:
        super().__init__()
        self.allow_local_services = allow_local_services
        self.max_redirects = max_redirects
        self._hop_count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._hop_count += 1
        if self._hop_count > self.max_redirects:
            raise PermissionError("Exceeded safe HTTP redirect limit.")
        target = urllib.parse.urljoin(req.full_url, newurl)
        validate_http_url(target, allow_local_services=self.allow_local_services)
        return super().redirect_request(req, fp, code, msg, headers, target)


@dataclass
class Tool:
    """Single callable capability exposed to the agent harness."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    category: str = "general"

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class ToolRegistry:
    """Registry with JSON Schema export, telemetry tracking, and learned hints."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters_schema: Dict[str, Any] | None = None,
        *,
        category: str = "general",
    ) -> Tool:
        tool = Tool(
            name=name,
            description=description,
            func=func,
            parameters_schema=parameters_schema or {"type": "object", "properties": {}},
            category=category,
        )
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema,
                "category": t.category,
            }
            for t in self._tools.values()
        ]

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Export tools as OpenAI/Ollama function-calling schema descriptors."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema or {"type": "object", "properties": {}},
                },
            }
            for t in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        arguments: Dict[str, Any] | None = None,
        *,
        memory: Any = None,
        context: str | None = None,
    ) -> Dict[str, Any]:
        """Run a registered tool with timing and automatic tool-learning feedback."""
        tool = self._tools.get(name)
        t0 = time.perf_counter()
        if tool is None:
            err = f"Unknown tool '{name}'. Available: {', '.join(self._tools)}"
            if memory is not None:
                memory.record_tool_outcome(name, success=False, latency_ms=0.0, error=err, context=context)
            return {"ok": False, "result": None, "error": err, "latency_ms": 0.0}

        kwargs = dict(arguments or {})
        try:
            res = tool.run(**kwargs)
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            # Treat explicit {"status": "error"} returns as failures for learning accuracy
            is_err_dict = isinstance(res, dict) and (res.get("status") == "error" or res.get("success") is False)
            if is_err_dict:
                err_msg = str(res.get("error") or res.get("message") or "Tool reported error")
                if memory is not None:
                    memory.record_tool_outcome(name, success=False, latency_ms=dt_ms, error=err_msg, context=context)
                return {"ok": False, "result": res, "error": err_msg, "latency_ms": dt_ms}

            if memory is not None:
                memory.record_tool_outcome(name, success=True, latency_ms=dt_ms, context=context)
            return {"ok": True, "result": res, "error": None, "latency_ms": dt_ms}
        except Exception as exc:
            dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            err_msg = f"{type(exc).__name__}: {exc}"
            if memory is not None:
                memory.record_tool_outcome(name, success=False, latency_ms=dt_ms, error=err_msg, context=context)
            return {"ok": False, "result": None, "error": err_msg, "latency_ms": dt_ms}

    def format_prompt_catalog(
        self,
        memory: Any = None,
        preferred_tools: list[str] | None = None,
        *,
        context: str | None = None,
    ) -> str:
        """Format available tools along with empirical reliability hints and pitfalls."""
        hints = memory.rank_tools_hint(self.names(), context=context) if memory is not None else {}
        preferred = set(preferred_tools or [])
        lines: list[str] = []
        ordered = sorted(
            self._tools.values(),
            key=lambda t: (0 if t.name in preferred else 1, t.category, t.name),
        )
        for t in ordered:
            schema_props = (t.parameters_schema or {}).get("properties") or {}
            args_sig = ", ".join(schema_props.keys())
            star = " [preferred]" if t.name in preferred else ""
            hint_part = f" ({hints[t.name]})" if t.name in hints else ""
            lines.append(f"- {t.name}({args_sig}){star}: {t.description}{hint_part}")
        return "\n".join(lines)


def _resolve_workspace_path(workspace: Path, raw_path: str) -> Path:
    ws_resolved = workspace.resolve()
    cand = (ws_resolved / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
    try:
        cand.relative_to(ws_resolved)
    except ValueError as exc:
        raise PermissionError(f"Path '{raw_path}' escapes workspace root '{ws_resolved}'") from exc
    return cand


def build_integrated_toolbelt(
    *,
    workspace: str | Path = ".",
    memory: Any = None,
    skills: Any = None,
    state_controller: Any = None,
    subagent_runner: Callable[[str, str | None], Any] | None = None,
    allow_shell: bool = False,
    allow_writes: bool = True,
    allow_local_http: bool = False,
) -> ToolRegistry:
    """Create a rich integrated ToolRegistry spanning filesystem, runtime, memory, skills & personas."""
    ws = Path(workspace).resolve()
    sandbox = PythonSandbox(timeout=10, max_memory_mb=256)
    reg = ToolRegistry()

    # ---- 1. Filesystem & Workspace Tools ----
    def tool_list_dir(path: str = ".", max_entries: int = 80) -> list[str]:
        target = _resolve_workspace_path(ws, path)
        if not target.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
        items = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        out = []
        for entry in items[: max(1, int(max_entries))]:
            suffix = "/" if entry.is_dir() else ""
            out.append(entry.name + suffix)
        return out

    reg.register(
        "list_dir",
        "List files and subdirectories inside a workspace directory.",
        tool_list_dir,
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path (default '.')"},
                "max_entries": {"type": "integer", "description": "Max items to return (default 80)"},
            },
        },
        category="workspace",
    )

    def tool_read_file(path: str, start_line: int = 1, end_line: int = 150) -> dict[str, Any]:
        target = _resolve_workspace_path(ws, path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        s_idx = max(1, int(start_line))
        e_idx = min(len(lines), max(s_idx, int(end_line)))
        numbered = [f"{i}: {lines[i - 1]}" for i in range(s_idx, e_idx + 1)]
        return {
            "path": str(target.relative_to(ws)).replace("\\", "/"),
            "total_lines": len(lines),
            "showing": f"{s_idx}-{e_idx}",
            "content": "\n".join(numbered),
        }

    reg.register(
        "read_file",
        "Read line-numbered text slices of a file in the workspace.",
        tool_read_file,
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
        category="workspace",
    )

    def tool_search_code(pattern: str, directory: str = ".", max_matches: int = 30) -> list[str]:
        target = _resolve_workspace_path(ws, directory)
        rx = re.compile(pattern)
        skip_dirs = {".git", "__pycache__", "studio_env", "node_modules", ".pytest_cache", ".ruff_cache"}
        hits: list[str] = []
        for root_dir, dirnames, filenames in os.walk(target):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                if Path(fname).suffix.lower() not in {
                    ".py",
                    ".js",
                    ".css",
                    ".md",
                    ".json",
                    ".toml",
                    ".yaml",
                    ".yml",
                    ".txt",
                }:
                    continue
                fpath = Path(root_dir) / fname
                try:
                    for idx, line in enumerate(fpath.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if rx.search(line):
                            rel = str(fpath.relative_to(ws)).replace("\\", "/")
                            hits.append(f"{rel}:{idx}: {line.strip()[:140]}")
                            if len(hits) >= max_matches:
                                return hits
                except OSError:
                    continue
        return hits

    reg.register(
        "search_code",
        "Search source/config files in the workspace using a Python regular expression.",
        tool_search_code,
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "directory": {"type": "string"},
                "max_matches": {"type": "integer"},
            },
            "required": ["pattern"],
        },
        category="workspace",
    )

    if allow_writes:

        def tool_write_file(path: str, content: str) -> dict[str, Any]:
            target = _resolve_workspace_path(ws, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"status": "ok", "path": str(target.relative_to(ws)).replace("\\", "/"), "bytes": len(content)}

        reg.register(
            "write_file",
            "Create or overwrite a text file inside the workspace.",
            tool_write_file,
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            category="workspace",
        )

        def tool_edit_file(path: str, target_text: str, replacement_text: str) -> dict[str, Any]:
            target = _resolve_workspace_path(ws, path)
            if not target.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            raw = target.read_text(encoding="utf-8")
            if target_text not in raw:
                raise ValueError("Target snippet does not match exact file text; read_file first.")
            count = raw.count(target_text)
            target.write_text(raw.replace(target_text, replacement_text), encoding="utf-8")
            return {"status": "ok", "path": str(target.relative_to(ws)).replace("\\", "/"), "replacements": count}

        reg.register(
            "edit_file",
            "Replace an exact target snippet inside a workspace file.",
            tool_edit_file,
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target_text": {"type": "string"},
                    "replacement_text": {"type": "string"},
                },
                "required": ["path", "target_text", "replacement_text"],
            },
            category="workspace",
        )

    def tool_git_status() -> dict[str, Any]:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"returncode": proc.returncode, "status": proc.stdout.strip()}

    reg.register(
        "git_status", "Show short git status of modified/untracked files.", tool_git_status, category="workspace"
    )

    # ---- 2. Compute, Sandbox, HTTP & Shell ----
    def tool_run_python_sandbox(code: str, locals_json: str = "{}") -> dict[str, Any]:
        ctx = json.loads(locals_json) if locals_json else {}
        return sandbox.execute_code(code, locals_dict=ctx)

    reg.register(
        "run_python_sandbox",
        "Execute restricted pure Python calculations safely via PythonSandbox.",
        tool_run_python_sandbox,
        {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "locals_json": {"type": "string", "description": "Optional JSON object of input variables"},
            },
            "required": ["code"],
        },
        category="compute",
    )

    def tool_http_fetch(url: str, max_chars: int = 4000) -> dict[str, Any]:
        clean = validate_http_url(url, allow_local_services=allow_local_http)
        req = urllib.request.Request(clean, headers={"User-Agent": "TriuneAgentHarness/2.1"})
        opener = urllib.request.build_opener(_SafeRedirectHandler(allow_local_services=allow_local_http))
        with opener.open(req, timeout=12) as resp:
            final_url = getattr(resp, "url", clean) or clean
            if final_url != clean:
                validate_http_url(final_url, allow_local_services=allow_local_http)
            raw = resp.read(max(1024, int(max_chars) * 2)).decode("utf-8", errors="replace")
            return {"status": getattr(resp, "status", 200), "body": raw[: max(200, int(max_chars))]}

    reg.register(
        "http_fetch",
        "Fetch text/JSON from SSRF-validated public HTTP(S) URLs cleanly.",
        tool_http_fetch,
        {
            "type": "object",
            "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
            "required": ["url"],
        },
        category="integration",
    )

    if allow_shell:

        def tool_run_shell(command: str, timeout: int = 30) -> dict[str, Any]:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=max(1, min(120, int(timeout))),
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-2000:],
            }

        reg.register(
            "run_shell",
            "Run a command in the workspace shell and capture stdout/stderr.",
            tool_run_shell,
            {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
            },
            category="compute",
        )

    # ---- 3. Triune Native & Runtime Tools ----
    def tool_inspect_hardware() -> dict[str, Any]:
        from triune.ops import get_native_c_info
        from triune.runtime.model_scanner import query_ollama_running_tags
        from triune.utils.device import device_diagnostic

        return {
            "device": device_diagnostic(),
            "native_c": get_native_c_info(),
            "ollama_live_tags": sorted(t for t in query_ollama_running_tags() if ":" in t),
        }

    reg.register(
        "inspect_hardware",
        "Inspect CUDA GPUs, VRAM, PyTorch, Native C/AVX2 kernels, and live local Ollama models.",
        tool_inspect_hardware,
        category="triune",
    )

    def tool_scan_local_models() -> dict[str, Any]:
        from triune.runtime.model_scanner import scan_pc_models

        res = scan_pc_models()
        return {
            "count": len(res.get("models", [])),
            "counts_by_source": res.get("counts_by_source", {}),
            "models": [
                {
                    "id": m.get("id"),
                    "source": m.get("source"),
                    "kind": m.get("kind"),
                    "ollama_running": bool(m.get("ollama_running", False)),
                    "can_load_direct": bool(m.get("can_load_direct", True)),
                    "size_str": m.get("size_str"),
                }
                for m in (res.get("models") or [])[:30]
            ],
        }

    reg.register(
        "scan_local_models",
        "Scan local PC runtimes (Ollama, HF, vLLM, Hermes, LM Studio, NVFP4, Checkpoints).",
        tool_scan_local_models,
        category="triune",
    )

    def tool_plan_vram_memory(model: str = "", vram_gb: float = 8.0) -> dict[str, Any]:
        from dataclasses import asdict

        from triune.configs import build_config
        from triune.runtime import MemoryPlanner

        if model:
            return MemoryPlanner.estimate_hf_model(model, available_vram_gb=vram_gb)
        return asdict(MemoryPlanner.estimate_vram(build_config({}), target_vram_gb=vram_gb))

    reg.register(
        "plan_vram_memory",
        "Estimate GPU VRAM, host RAM, and NVFP4 disk-stream budgets for a Triune or HF model.",
        tool_plan_vram_memory,
        {
            "type": "object",
            "properties": {"model": {"type": "string"}, "vram_gb": {"type": "number"}},
        },
        category="triune",
    )

    def tool_resolve_ollama_hf(ollama_tag: str) -> dict[str, Any]:
        from triune.model.transmuter import ollama_to_hf_candidate

        cand = ollama_to_hf_candidate(ollama_tag)
        return {"ollama_tag": ollama_tag, "hf_candidate": cand}

    reg.register(
        "resolve_ollama_hf",
        "Resolve an Ollama tag (e.g. ollama:gemma3:4b) to its upstream Hugging Face repo candidate.",
        tool_resolve_ollama_hf,
        {
            "type": "object",
            "properties": {"ollama_tag": {"type": "string"}},
            "required": ["ollama_tag"],
        },
        category="triune",
    )

    # ---- 4. Meta-Cognitive Self-Learning, Persona & Skill Tools ----
    if state_controller is not None:

        def tool_shift_persona_state(
            persona: str = "",
            reasoning_depth: str = "",
            temperature: float | None = None,
            focus_note: str = "",
            custom_title: str = "",
            custom_prompt: str = "",
        ) -> dict[str, Any]:
            if custom_prompt and persona:
                state_controller.register_persona(
                    persona,
                    custom_title or persona.title(),
                    custom_prompt,
                    default_depth=reasoning_depth or "limbic",
                    temperature=temperature if temperature is not None else 0.35,
                )
            st = state_controller.shift(
                persona=persona or None,
                reasoning_depth=reasoning_depth or None,
                temperature=temperature,
                focus_note=focus_note or None,
                reason="agent self-shift",
            )
            return {
                "persona": st.persona,
                "reasoning_depth": st.reasoning_depth,
                "temperature": st.temperature,
                "focus_note": st.focus_note,
            }

        reg.register(
            "shift_persona_state",
            "Switch active personality/cognitive state (architect, engineer, researcher, debugger, optimizer, tutor, sentinel, or create a new custom persona) and reasoning tier (reflex, limbic, cortex).",
            tool_shift_persona_state,
            {
                "type": "object",
                "properties": {
                    "persona": {"type": "string"},
                    "reasoning_depth": {"type": "string", "enum": ["reflex", "limbic", "cortex"]},
                    "temperature": {"type": "number"},
                    "focus_note": {"type": "string"},
                    "custom_title": {"type": "string"},
                    "custom_prompt": {"type": "string"},
                },
            },
            category="metacognition",
        )

    if memory is not None:

        def tool_remember_user_preference(
            key: str, value: str, category: str = "preference", priority: int = 7
        ) -> dict[str, Any]:
            pref = memory.remember_preference(key, value, category=category, priority=priority)
            return {"saved": pref.key, "value": pref.value, "category": pref.category}

        reg.register(
            "remember_user_preference",
            "Persist an inferred or stated user want, preference, constraint, or environment fact.",
            tool_remember_user_preference,
            {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "category": {"type": "string", "enum": ["preference", "constraint", "goal", "environment"]},
                    "priority": {"type": "integer"},
                },
                "required": ["key", "value"],
            },
            category="memory",
        )

        def tool_record_lesson(lesson: str, source_tool: str = "", confidence: float = 0.8) -> dict[str, Any]:
            entry = memory.record_lesson(lesson, source_tool=source_tool or None, confidence=confidence)
            return {"lesson_id": entry.lesson_id, "confidence": entry.confidence, "lesson": entry.lesson}

        reg.register(
            "record_lesson",
            "Record a self-learned operational insight, pitfall fix, or tool optimization for future tasks.",
            tool_record_lesson,
            {
                "type": "object",
                "properties": {
                    "lesson": {"type": "string"},
                    "source_tool": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["lesson"],
            },
            category="memory",
        )

    if skills is not None:

        def tool_create_skill(
            name: str,
            description: str,
            instructions: str,
            triggers: list[str] | None = None,
            steps: list[str] | None = None,
            recommended_tools: list[str] | None = None,
            helper_code: str = "",
        ) -> dict[str, Any]:
            skill = skills.create_skill(
                name,
                description,
                instructions,
                triggers=triggers,
                steps=steps,
                recommended_tools=recommended_tools,
                helper_code=helper_code or None,
            )
            return {"slug": skill.slug, "path": skill.source_path, "triggers": skill.triggers}

        reg.register(
            "create_skill",
            "Synthesize and persist a new reusable SKILL.md (plus optional Python sandbox helper script).",
            tool_create_skill,
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "instructions": {"type": "string"},
                    "triggers": {"type": "array", "items": {"type": "string"}},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "recommended_tools": {"type": "array", "items": {"type": "string"}},
                    "helper_code": {"type": "string"},
                },
                "required": ["name", "description", "instructions"],
            },
            category="skills",
        )

        def tool_evolve_skill(name_or_slug: str, amendment: str, add_tools: list[str] | None = None) -> dict[str, Any]:
            sk = skills.evolve_skill(name_or_slug, amendment, add_tools=add_tools)
            return {"slug": sk.slug, "updated_path": sk.source_path}

        reg.register(
            "evolve_skill",
            "Update an existing skill with newly learned refinements, edge cases, or better tools.",
            tool_evolve_skill,
            {
                "type": "object",
                "properties": {
                    "name_or_slug": {"type": "string"},
                    "amendment": {"type": "string"},
                    "add_tools": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name_or_slug", "amendment"],
            },
            category="skills",
        )

        def tool_run_skill_helper(name_or_slug: str, locals_json: str = "{}") -> dict[str, Any]:
            from .skills import _slugify

            sk = skills.skills.get(_slugify(name_or_slug))
            if sk is None or not sk.helper_code:
                raise FileNotFoundError(f"Skill '{name_or_slug}' has no runnable helper.py.")
            ctx = json.loads(locals_json) if locals_json else {}
            res = sandbox.execute_code(sk.helper_code, locals_dict=ctx)
            ok_exec = bool(res.get("success", False))
            skills.record_outcome(sk.slug, ok_exec, verified=ok_exec)
            return res

        reg.register(
            "run_skill_helper",
            "Execute a skill's helper.py inside PythonSandbox safely.",
            tool_run_skill_helper,
            {
                "type": "object",
                "properties": {"name_or_slug": {"type": "string"}, "locals_json": {"type": "string"}},
                "required": ["name_or_slug"],
            },
            category="skills",
        )

    if subagent_runner is not None:

        def tool_delegate_subagent(goal: str, persona: str = "researcher") -> dict[str, Any]:
            return subagent_runner(goal, persona)

        reg.register(
            "delegate_subagent",
            "Spawn a focused specialist subagent with a tailored persona to solve and return a subtask.",
            tool_delegate_subagent,
            {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "persona": {"type": "string"},
                },
                "required": ["goal"],
            },
            category="metacognition",
        )

    return reg
