"""Upgraded Agentic Loop Engine: streaming ReAct/PEV/Multi-Agent loops, HITL approvals, stuck detection, context compaction, and Training Supervisor."""

from __future__ import annotations

import asyncio
import difflib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .tools import (
    _SHELL_METACHAR_TOKENS,
    _has_traversal_syntax,
    _resolve_workspace_path,
    build_integrated_toolbelt,
)

_APPROVAL_GATED_TOOLS: frozenset[str] = frozenset(
    {"write_file", "edit_file", "delete_file", "run_shell", "terminal_run"}
)

ALLOWED_TERMINAL_COMMANDS: frozenset[str] = frozenset(
    {
        "pytest",
        "python",
        "python3",
        "ruff",
        "git",
        "ls",
        "dir",
        "cat",
        "head",
        "tail",
        "node",
        "npm",
        "dotnet",
        "rg",
        "find",
    }
)


def validate_terminal_command(command: str | list[str], workspace: Path) -> list[str]:
    """Validate and parse a dev terminal command under strict shell=False, allowlist, and workspace confinement."""
    ws_resolved = Path(workspace).resolve(strict=False)
    if isinstance(command, list):
        raw_tokens = [str(x) for x in command]
        raw_joined = " ".join(raw_tokens)
    else:
        raw_joined = str(command or "").strip()
        if not raw_joined:
            raise PermissionError("Command cannot be empty.")
        for bad in _SHELL_METACHAR_TOKENS:
            if bad in raw_joined:
                raise PermissionError(f"Rejected disallowed shell metacharacter '{bad}'.")
        raw_tokens = [
            tok[1:-1] if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'") else tok
            for tok in shlex.split(raw_joined, posix=(os.name != "nt"))
        ]

    if not raw_tokens:
        raise PermissionError("Command cannot be empty.")

    for tok in raw_tokens:
        for bad in _SHELL_METACHAR_TOKENS:
            if bad in tok:
                raise PermissionError(f"Rejected disallowed shell metacharacter '{bad}' in argument.")

    exe_name = Path(raw_tokens[0]).name.lower()
    if exe_name.endswith(".exe"):
        exe_name = exe_name[:-4]
    if exe_name not in ALLOWED_TERMINAL_COMMANDS:
        raise PermissionError(f"Command '{raw_tokens[0]}' is not in the permitted developer command allowlist.")

    for arg in raw_tokens[1:]:
        if _has_traversal_syntax(arg):
            raise PermissionError(f"Path traversal syntax rejected in command argument: '{arg}'")
        if arg.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", arg) or arg.startswith("~"):
            cand = Path(arg).expanduser().resolve(strict=False)
            if not cand.is_relative_to(ws_resolved):
                raise PermissionError(f"Command argument '{arg}' escapes workspace root.")
        elif not arg.startswith("-") and (
            "/" in arg or "\\" in arg or (ws_resolved / arg).exists() or (ws_resolved / arg).is_symlink()
        ):
            cand = (ws_resolved / arg).resolve(strict=False)
            if not cand.is_relative_to(ws_resolved):
                raise PermissionError(f"Command argument '{arg}' escapes workspace root via symlink or path.")

    return raw_tokens


def resolve_executable_argv(argv: list[str]) -> list[str]:
    """Resolve python/python3/pytest cleanly to the active interpreter when missing on PATH."""
    resolved = list(argv)
    head = Path(resolved[0]).name.lower().removesuffix(".exe")
    if head in {"python", "python3"}:
        resolved[0] = sys.executable
    elif head == "pytest" and shutil.which(resolved[0]) is None:
        resolved = [sys.executable, "-m", "pytest", *resolved[1:]]
    elif head == "ruff" and shutil.which(resolved[0]) is None:
        resolved = [sys.executable, "-m", "ruff", *resolved[1:]]
    return resolved


def compute_action_preview(
    workspace: Path, tool_name: str, args: dict[str, Any]
) -> tuple[str, Optional[str], Optional[str]]:
    """Compute human-readable summary, unified diff, and command preview for HITL approval gates."""
    if tool_name in {"run_shell", "terminal_run"}:
        raw_cmd = args.get("command", "")
        cmd_str = " ".join(str(x) for x in raw_cmd) if isinstance(raw_cmd, list) else str(raw_cmd)
        return f"Execute command: {cmd_str}", None, cmd_str

    raw_path = str(args.get("path") or "")
    rel_label = raw_path.replace("\\", "/")
    old_text = ""
    try:
        target = _resolve_workspace_path(workspace, raw_path)
        rel_label = str(target.relative_to(workspace.resolve(strict=False))).replace("\\", "/")
        if target.is_file():
            old_text = target.read_text(encoding="utf-8", errors="replace")
    except Exception:
        target = None

    if tool_name == "write_file":
        new_text = str(args.get("content") or "")
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{rel_label}" if old_text else "/dev/null",
                tofile=f"b/{rel_label}",
                lineterm="",
            )
        )
        return f"Write file {rel_label} ({len(new_text)} chars)", "\n".join(diff_lines), None

    if tool_name == "edit_file":
        target_text = str(args.get("target_text") or "")
        replacement_text = str(args.get("replacement_text") or "")
        new_text = old_text.replace(target_text, replacement_text) if target_text else old_text
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{rel_label}",
                tofile=f"b/{rel_label}",
                lineterm="",
            )
        )
        return f"Edit file {rel_label}", "\n".join(diff_lines), None

    if tool_name == "delete_file":
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(),
                [],
                fromfile=f"a/{rel_label}",
                tofile="/dev/null",
                lineterm="",
            )
        )
        return f"Delete file {rel_label}", "\n".join(diff_lines), None

    return f"Run {tool_name}", None, None


class StuckLoopDetector:
    """Detect repeated identical tool calls or consecutive identical failing calls and trigger a pivot."""

    def __init__(self, *, repeat_threshold: int = 3, fail_threshold: int = 2, window: int = 8) -> None:
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.fail_threshold = max(2, int(fail_threshold))
        self.window = max(self.repeat_threshold, int(window))
        self.recent_calls: list[tuple[str, str]] = []
        self.consecutive_failures: list[tuple[str, str]] = []
        self.triggered_patterns: dict[str, int] = {}

    def record(
        self, tool_name: str, args: dict[str, Any], *, ok: bool, error: str | None = None
    ) -> dict[str, Any] | None:
        args_sig = json.dumps(args or {}, sort_keys=True, default=str)
        sig = (tool_name, args_sig)
        self.recent_calls.append(sig)
        if len(self.recent_calls) > self.window:
            self.recent_calls.pop(0)

        identical_count = sum(1 for item in self.recent_calls if item == sig)
        if identical_count >= self.repeat_threshold:
            pat_key = f"repeat:{tool_name}:{args_sig}"
            count = self.triggered_patterns.get(pat_key, 0) + 1
            self.triggered_patterns[pat_key] = count
            return {
                "triggered": True,
                "reason": "stuck_loop_identical_call",
                "repeat_count": count,
                "pattern": f"{tool_name}({args_sig[:120]})",
                "message": (
                    f"Stuck-loop detector caught {identical_count} identical calls to `{tool_name}` with identical arguments. "
                    "Pivot strategy: try a different tool, inspect different arguments, or conclude with verified findings."
                ),
            }

        if not ok:
            err_norm = str(error or "error").strip()[:120]
            self.consecutive_failures.append((tool_name, err_norm))
            if len(self.consecutive_failures) >= self.fail_threshold:
                tail = self.consecutive_failures[-self.fail_threshold :]
                if all(t == tool_name for t, _ in tail):
                    pat_key = f"fail:{tool_name}:{err_norm}"
                    count = self.triggered_patterns.get(pat_key, 0) + 1
                    self.triggered_patterns[pat_key] = count
                    return {
                        "triggered": True,
                        "reason": "stuck_loop_repeated_failure",
                        "repeat_count": count,
                        "pattern": f"{tool_name}: {err_norm}",
                        "message": (
                            f"Stuck-loop detector caught {self.fail_threshold} consecutive failures in `{tool_name}` ({err_norm}). "
                            "Stop retrying the failing call blindly; inspect root cause or pivot to an alternative approach."
                        ),
                    }
        else:
            self.consecutive_failures.clear()

        return None


class ContextCompactor:
    """Compress older middle tool observations when conversation history exceeds character budget."""

    def __init__(self, *, char_budget: int = 16000, keep_recent: int = 4) -> None:
        self.char_budget = max(800, int(char_budget))
        self.keep_recent = max(2, int(keep_recent))

    def maybe_compact(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        if len(messages) <= 2 + self.keep_recent:
            return None
        chars_before = sum(len(str(m.get("content") or "")) for m in messages[1:])
        if chars_before <= self.char_budget:
            return None

        middle = messages[2 : -self.keep_recent]
        if not middle:
            return None

        tool_notes: list[str] = []
        for msg in middle:
            txt = str(msg.get("content") or "").strip().replace("\n", " ")
            if txt:
                tool_notes.append(txt[:160])

        summary_lines = tool_notes[:6]
        if len(tool_notes) > 6:
            summary_lines.append(f"... ({len(tool_notes) - 6} earlier entries compressed)")
        summary_body = "\n".join(f"- {ln}" for ln in summary_lines)
        compacted_msg = {
            "role": "user",
            "content": f"[Context Compaction Summary · {len(middle)} earlier messages compressed]\n{summary_body}",
        }
        messages[2 : -self.keep_recent] = [compacted_msg]
        chars_after = sum(len(str(m.get("content") or "")) for m in messages[1:])
        return {
            "dropped_messages": len(middle),
            "chars_before": chars_before,
            "chars_after": chars_after,
            "summary": summary_body[:320],
        }


@dataclass
class PendingApproval:
    action_id: str
    run_id: str
    tool: str
    summary: str
    diff: Optional[str]
    command: Optional[str]
    arguments: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    approved: Optional[bool] = None
    reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class ApprovalGateManager:
    """Thread-safe registry for Human-in-the-Loop approval gates during supervised loops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, PendingApproval] = {}

    def create_pending(
        self,
        *,
        run_id: str,
        tool: str,
        summary: str,
        diff: Optional[str],
        command: Optional[str],
        arguments: dict[str, Any],
    ) -> PendingApproval:
        action_id = f"act-{uuid.uuid4().hex[:12]}"
        item = PendingApproval(
            action_id=action_id,
            run_id=run_id,
            tool=tool,
            summary=summary,
            diff=diff,
            command=command,
            arguments=dict(arguments),
        )
        with self._lock:
            self._pending[action_id] = item
        return item

    def wait_for_decision(
        self, pending: PendingApproval, cancel_event: threading.Event, *, timeout: float = 300.0
    ) -> tuple[bool, Optional[str]]:
        deadline = time.monotonic() + max(1.0, float(timeout))
        while not pending.event.is_set():
            if cancel_event.is_set():
                return False, "Cancelled by user"
            if time.monotonic() >= deadline:
                return False, "Approval timed out"
            pending.event.wait(0.05)
        return bool(pending.approved), pending.reason

    def resolve(self, action_id: str, *, approved: bool, reason: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            item = self._pending.get(action_id)
            if item is None:
                raise KeyError(f"Unknown approval action_id '{action_id}'")
            item.approved = bool(approved)
            item.reason = reason
            item.event.set()
        return {
            "ok": True,
            "action_id": action_id,
            "run_id": item.run_id,
            "approved": bool(approved),
            "reason": reason,
        }

    def cancel_for_run(self, run_id: str) -> None:
        with self._lock:
            for item in self._pending.values():
                if item.run_id == run_id and not item.event.is_set():
                    item.approved = False
                    item.reason = "Run cancelled"
                    item.event.set()


def _parse_plan_steps(content: str) -> list[str]:
    steps: list[str] = []
    text = str(content or "").strip()
    if not text:
        return steps
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("plan"), list):
            return [str(s).strip() for s in obj["plan"] if str(s).strip()][:8]
    except Exception:
        pass
    for line in text.splitlines():
        clean = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if clean:
            steps.append(clean)
    return steps[:8]


def _run_validated_terminal(workspace: Path, command: str | list[str], timeout: int = 30) -> dict[str, Any]:
    argv = validate_terminal_command(command, workspace)
    exec_argv = resolve_executable_argv(argv)
    t0 = time.perf_counter()
    proc = subprocess.run(
        exec_argv,
        shell=False,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=max(1, min(30, int(timeout))),
    )
    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[:65536],
        "stderr": (proc.stderr or "")[:65536],
        "duration_ms": dt_ms,
        "command": argv,
    }


def _ensure_loop_toolbelt(harness: Any, workspace: Path) -> Any:
    """Build a workspace-confined toolbelt supporting gated writes & validated terminal execution."""
    reg = build_integrated_toolbelt(
        workspace=workspace,
        memory=getattr(harness, "memory", None),
        skills=getattr(harness, "skills", None),
        state_controller=getattr(harness, "state_controller", None),
        subagent_runner=None,
        allow_shell=False,
        allow_writes=True,
        allow_local_http=bool(getattr(harness, "allow_local_http", False)),
    )

    def tool_terminal_run(command: str | list[str], timeout: int = 30) -> dict[str, Any]:
        return _run_validated_terminal(workspace, command, timeout=timeout)

    reg.register(
        "run_shell",
        "Execute a developer command in the workspace under strict shell=False allowlist confinement.",
        tool_terminal_run,
        {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"],
        },
        category="compute",
    )
    reg.register(
        "terminal_run",
        "Execute a developer command in the workspace under strict shell=False allowlist confinement.",
        tool_terminal_run,
        {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"],
        },
        category="compute",
    )
    existing_tools = getattr(getattr(harness, "tools", None), "_tools", None)
    if isinstance(existing_tools, dict):
        for name, tool_obj in existing_tools.items():
            if name not in reg._tools:
                reg.register(
                    name,
                    tool_obj.description,
                    tool_obj.func,
                    tool_obj.parameters_schema,
                    category=tool_obj.category,
                )
    return reg


def _execute_react_phase(
    *,
    harness: Any,
    tools: Any,
    workspace: Path,
    messages: list[dict[str, Any]],
    run_id: str,
    mode: str,
    autonomy: str,
    agent_name: str,
    max_steps: int,
    emit: Callable[[dict[str, Any]], None],
    cancel_event: threading.Event,
    approval_manager: ApprovalGateManager,
    stuck_detector: StuckLoopDetector,
    compactor: ContextCompactor,
    allow_mutations: bool = True,
    initial_response: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], bool]:
    from .engine import _parse_model_turn

    trace: list[dict[str, Any]] = []
    final_answer = ""
    cancelled = False

    for step_num in range(1, max(1, int(max_steps)) + 1):
        if cancel_event.is_set():
            cancelled = True
            break

        compact_info = compactor.maybe_compact(messages)
        if compact_info is not None:
            emit({"type": "compaction", "run_id": run_id, "step": step_num, "agent": agent_name, **compact_info})

        if step_num == 1 and initial_response is not None:
            resp = initial_response
        else:
            resp = harness._invoke_model(messages)

        if cancel_event.is_set():
            cancelled = True
            break

        thought, calls, answer = _parse_model_turn(resp)
        if thought:
            emit({"type": "thought", "run_id": run_id, "step": step_num, "agent": agent_name, "thought": thought})

        if answer is not None and not calls:
            final_answer = answer
            break

        if not calls:
            final_answer = str(resp.get("content") or "").strip()
            break

        messages.append(
            {
                "role": "assistant",
                "content": str(resp.get("content") or json.dumps({"thought": thought, "tool_calls": calls})),
            }
        )

        break_for_stuck = False
        for call in calls:
            if cancel_event.is_set():
                cancelled = True
                break

            t_name = str(call["name"])
            t_args = dict(call.get("arguments") or {})

            emit(
                {
                    "type": "tool_call",
                    "run_id": run_id,
                    "step": step_num,
                    "agent": agent_name,
                    "tool": t_name,
                    "arguments": t_args,
                }
            )

            if not allow_mutations and t_name in _APPROVAL_GATED_TOOLS:
                outcome = {
                    "ok": False,
                    "result": None,
                    "error": f"Stage '{agent_name}' is read-only; mutation tool '{t_name}' is restricted.",
                    "latency_ms": 0.0,
                }
                approved_flag: Optional[bool] = False
            else:
                approved_flag = None
                if autonomy == "supervised" and t_name in _APPROVAL_GATED_TOOLS:
                    summary, diff_preview, cmd_preview = compute_action_preview(workspace, t_name, t_args)
                    pending = approval_manager.create_pending(
                        run_id=run_id,
                        tool=t_name,
                        summary=summary,
                        diff=diff_preview,
                        command=cmd_preview,
                        arguments=t_args,
                    )
                    emit(
                        {
                            "type": "approval_required",
                            "run_id": run_id,
                            "step": step_num,
                            "agent": agent_name,
                            "action_id": pending.action_id,
                            "tool": t_name,
                            "summary": summary,
                            "diff": diff_preview,
                            "command": cmd_preview,
                            "arguments": t_args,
                        }
                    )
                    is_approved, rej_reason = approval_manager.wait_for_decision(pending, cancel_event)
                    if cancel_event.is_set():
                        cancelled = True
                        break
                    approved_flag = is_approved
                    if not is_approved:
                        outcome = {
                            "ok": False,
                            "result": None,
                            "error": f"Action rejected by user: {rej_reason or 'denied'}",
                            "latency_ms": 0.0,
                        }
                    else:
                        outcome = tools.execute(t_name, t_args, memory=harness.memory, context=agent_name)
                else:
                    outcome = tools.execute(t_name, t_args, memory=harness.memory, context=agent_name)

            trace.append(
                {
                    "step": step_num,
                    "agent": agent_name,
                    "tool": t_name,
                    "arguments": t_args,
                    "ok": outcome["ok"],
                    "error": outcome["error"],
                    "latency_ms": outcome["latency_ms"],
                }
            )
            emit(
                {
                    "type": "tool_result",
                    "run_id": run_id,
                    "step": step_num,
                    "agent": agent_name,
                    "tool": t_name,
                    "ok": outcome["ok"],
                    "error": outcome["error"],
                    "latency_ms": outcome["latency_ms"],
                    "approved": approved_flag,
                    "result": str(outcome["result"])[:1200] if outcome["ok"] else None,
                }
            )

            obs_payload = {
                "tool": t_name,
                "ok": outcome["ok"],
                "result": outcome["result"] if outcome["ok"] else None,
                "error": outcome["error"],
            }
            messages.append(
                {"role": "user", "content": f"Tool Observation:\n{json.dumps(obs_payload, default=str)[:4500]}"}
            )

            stuck = stuck_detector.record(t_name, t_args, ok=outcome["ok"], error=outcome["error"])
            if stuck is not None:
                emit(
                    {
                        "type": "reflection",
                        "run_id": run_id,
                        "step": step_num,
                        "agent": agent_name,
                        "reason": stuck["reason"],
                        "pattern": stuck["pattern"],
                        "message": stuck["message"],
                    }
                )
                messages.append({"role": "user", "content": f"System Reflection:\n{stuck['message']}"})
                if stuck["repeat_count"] >= 2:
                    final_answer = (
                        final_answer
                        or f"Stopped repeating stuck pattern ({stuck['pattern']}). Partial progress recorded across {len(trace)} tool calls."
                    )
                    break_for_stuck = True
                    break

            compact_after = compactor.maybe_compact(messages)
            if compact_after is not None:
                emit({"type": "compaction", "run_id": run_id, "step": step_num, "agent": agent_name, **compact_after})

        if cancelled or break_for_stuck:
            break

    if not final_answer and not cancelled:
        final_answer = f"Completed {len(trace)} tool step(s) in {mode} mode."

    return final_answer, trace, cancelled


def run_agent_loop(
    *,
    harness: Any,
    goal: str,
    mode: str = "react",
    autonomy: str = "supervised",
    active_file: Optional[str] = None,
    max_steps: int = 10,
    run_id: Optional[str] = None,
    emit: Callable[[dict[str, Any]], None],
    cancel_event: Optional[threading.Event] = None,
    approval_manager: Optional[ApprovalGateManager] = None,
    supervisor: Optional["TrainingSupervisor"] = None,
    compact_char_budget: int = 16000,
    compact_keep_recent: int = 4,
    stuck_repeat_threshold: int = 3,
    stuck_fail_threshold: int = 2,
) -> dict[str, Any]:
    """Execute a streaming agentic loop across ReAct, PEV, Multi-Agent, or Training Supervisor modes."""
    from .engine import _parse_model_turn

    rid = run_id or f"loop-{uuid.uuid4().hex[:10]}"
    c_event = cancel_event or threading.Event()
    gate = approval_manager or ApprovalGateManager()
    norm_mode = str(mode or "react").strip().lower()
    if norm_mode not in {"react", "pev", "multi_agent", "training_supervisor"}:
        norm_mode = "react"
    norm_autonomy = str(autonomy or "supervised").strip().lower()
    if norm_autonomy not in {"supervised", "autonomous_sandbox"}:
        norm_autonomy = "supervised"

    workspace = Path(harness.workspace).resolve(strict=False)
    tools = _ensure_loop_toolbelt(harness, workspace)
    stuck_detector = StuckLoopDetector(
        repeat_threshold=stuck_repeat_threshold,
        fail_threshold=stuck_fail_threshold,
    )
    compactor = ContextCompactor(char_budget=compact_char_budget, keep_recent=compact_keep_recent)

    emit(
        {
            "type": "start",
            "run_id": rid,
            "mode": norm_mode,
            "autonomy": norm_autonomy,
            "workspace": str(workspace),
            "active_file": active_file,
            "goal": goal,
        }
    )

    try:
        if c_event.is_set():
            emit({"type": "cancelled", "run_id": rid, "reason": "Cancelled before start"})
            return {"run_id": rid, "cancelled": True, "answer": "", "steps": 0}

        if norm_mode == "training_supervisor":
            sup = supervisor or TrainingSupervisor()
            emit(
                {
                    "type": "plan",
                    "run_id": rid,
                    "phase": "plan",
                    "steps": [
                        "Inspect live training loss, grad-norm, and NaN/Inf health",
                        "Inspect MoE expert utilization, CentroidSteer energy, and Triune 3-tier balance",
                        "Emit diagnostics, safety checkpoint status, and actionable recommendations",
                    ],
                }
            )
            emit(
                {
                    "type": "thought",
                    "run_id": rid,
                    "step": 1,
                    "agent": "supervisor",
                    "thought": f"Inspecting live training metrics and mind telemetry for goal: {goal}",
                }
            )
            emit(
                {
                    "type": "tool_call",
                    "run_id": rid,
                    "step": 1,
                    "agent": "supervisor",
                    "tool": "inspect_training_telemetry",
                    "arguments": {},
                }
            )
            snap = sup.inspect_once()
            emit(
                {
                    "type": "tool_result",
                    "run_id": rid,
                    "step": 1,
                    "agent": "supervisor",
                    "tool": "inspect_training_telemetry",
                    "ok": True,
                    "error": None,
                    "latency_ms": 1.0,
                    "result": json.dumps(snap, default=str)[:1200],
                }
            )
            anomalies = snap.get("anomalies") or []
            recs = snap.get("recommendations") or []
            summary_msg = (
                f"Training Supervisor checked step {snap.get('step', 0)}: {len(anomalies)} anomaly(ies) detected."
            )
            if recs:
                summary_msg += " Recommendations: " + "; ".join(recs[:3])
            emit(
                {
                    "type": "reflection",
                    "run_id": rid,
                    "step": 1,
                    "agent": "supervisor",
                    "reason": "supervisor_inspection",
                    "message": summary_msg,
                }
            )
            answer = summary_msg
            if harness.completion_fn is not None or harness.model:
                try:
                    sys_p = harness._build_system_prompt(goal, [])
                    llm_resp = harness._invoke_model(
                        [
                            {"role": "system", "content": sys_p},
                            {
                                "role": "user",
                                "content": f"Training Supervisor telemetry snapshot:\n{json.dumps(snap, default=str)[:3000]}\n\nGoal: {goal}",
                            },
                        ]
                    )
                    _, _, ans = _parse_model_turn(llm_resp)
                    if ans or llm_resp.get("content"):
                        answer = str(ans or llm_resp.get("content")).strip()
                except Exception:
                    pass
            emit(
                {
                    "type": "done",
                    "run_id": rid,
                    "mode": norm_mode,
                    "answer": answer,
                    "verified": True,
                    "steps": 1,
                }
            )
            return {"run_id": rid, "cancelled": False, "answer": answer, "steps": 1, "snapshot": snap}

        if harness.completion_fn is None and not harness.model:
            err_msg = (
                "No LLM backend configured for AgentHarness. "
                "Start local Ollama, configure a BYOK key, or specify a model."
            )
            emit({"type": "error", "run_id": rid, "error": err_msg})
            return {"run_id": rid, "cancelled": False, "error": err_msg, "steps": 0}

        harness.state_controller.adapt_for_goal(goal, auto_persona=True, auto_depth=True)
        matched_skills = harness.skills.match_skills(goal, limit=3)
        sys_prompt = harness._build_system_prompt(goal, matched_skills)
        if active_file:
            sys_prompt += f"\n\nActive Workspace File: {active_file}"

        if norm_mode == "multi_agent":
            emit(
                {
                    "type": "plan",
                    "run_id": rid,
                    "phase": "multi_agent_pipeline",
                    "steps": [
                        "Explorer: inspect workspace files and locate relevant symbols",
                        "Implementer: apply targeted edits or run scoped commands",
                        "Verifier: validate outputs and summarize verified result",
                    ],
                }
            )
            stages = [
                ("explorer", "architect", "Explore the workspace and gather concrete findings for the goal.", False),
                ("implementer", "engineer", "Implement the required changes using the Explorer findings.", True),
                (
                    "verifier",
                    "sentinel",
                    "Verify the implemented changes and produce the final verified report.",
                    False,
                ),
            ]
            all_trace: list[dict[str, Any]] = []
            prior_notes = ""
            final_answer = ""
            per_stage_steps = max(2, min(int(max_steps), 6))
            for stage_name, persona_name, stage_guidance, can_mutate in stages:
                if c_event.is_set():
                    emit({"type": "cancelled", "run_id": rid, "reason": "Cancelled by user"})
                    return {"run_id": rid, "cancelled": True, "answer": final_answer, "steps": len(all_trace)}
                harness.state_controller.shift(persona=persona_name, reason=f"multi_agent:{stage_name}")
                stage_sys = harness._build_system_prompt(goal, matched_skills)
                if active_file:
                    stage_sys += f"\n\nActive Workspace File: {active_file}"
                stage_user = f"[{stage_name.upper()}] {stage_guidance}\nGoal: {goal}"
                if prior_notes:
                    stage_user += f"\n\nFindings from prior stage:\n{prior_notes[:2000]}"
                stage_msgs = [
                    {"role": "system", "content": stage_sys},
                    {"role": "user", "content": stage_user},
                ]
                stage_ans, stage_trace, was_cancelled = _execute_react_phase(
                    harness=harness,
                    tools=tools,
                    workspace=workspace,
                    messages=stage_msgs,
                    run_id=rid,
                    mode=norm_mode,
                    autonomy=norm_autonomy,
                    agent_name=stage_name,
                    max_steps=per_stage_steps,
                    emit=emit,
                    cancel_event=c_event,
                    approval_manager=gate,
                    stuck_detector=stuck_detector,
                    compactor=compactor,
                    allow_mutations=can_mutate,
                )
                all_trace.extend(stage_trace)
                prior_notes = stage_ans
                final_answer = stage_ans
                if was_cancelled:
                    emit({"type": "cancelled", "run_id": rid, "reason": "Cancelled by user"})
                    return {"run_id": rid, "cancelled": True, "answer": final_answer, "steps": len(all_trace)}

            emit(
                {
                    "type": "reflection",
                    "run_id": rid,
                    "step": len(all_trace),
                    "agent": "verifier",
                    "reason": "pipeline_verified",
                    "message": f"Multi-agent pipeline completed across Explorer -> Implementer -> Verifier ({len(all_trace)} tool call(s)).",
                }
            )
            emit(
                {
                    "type": "done",
                    "run_id": rid,
                    "mode": norm_mode,
                    "answer": final_answer,
                    "verified": True,
                    "steps": len(all_trace),
                }
            )
            return {
                "run_id": rid,
                "cancelled": False,
                "answer": final_answer,
                "steps": len(all_trace),
                "trace": all_trace,
            }

        if norm_mode == "pev":
            plan_msgs = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Plan->Execute->Verify Mode. First produce a concise numbered plan for: {goal}"
                        + (f" (active file: {active_file})" if active_file else "")
                    ),
                },
            ]
            plan_resp = harness._invoke_model(plan_msgs)
            if c_event.is_set():
                emit({"type": "cancelled", "run_id": rid, "reason": "Cancelled by user"})
                return {"run_id": rid, "cancelled": True, "answer": "", "steps": 0}

            p_thought, p_calls, p_ans = _parse_model_turn(plan_resp)
            initial_exec_resp: dict[str, Any] | None = None
            if p_calls:
                plan_steps = [
                    p_thought or f"Execute {p_calls[0]['name']}",
                    "Verify tool outputs and summarize outcome",
                ]
                initial_exec_resp = plan_resp
            else:
                plan_steps = _parse_plan_steps(p_ans or str(plan_resp.get("content") or "")) or [
                    f"Analyze {goal}",
                    "Execute workspace actions",
                    "Verify result",
                ]

            emit({"type": "plan", "run_id": rid, "phase": "plan", "steps": plan_steps})
            exec_msgs = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Goal: {goal}\n"
                        + (f"Active file: {active_file}\n" if active_file else "")
                        + "Approved Plan:\n"
                        + "\n".join(f"{idx}. {st}" for idx, st in enumerate(plan_steps, 1))
                    ),
                },
            ]
            final_answer, trace, was_cancelled = _execute_react_phase(
                harness=harness,
                tools=tools,
                workspace=workspace,
                messages=exec_msgs,
                run_id=rid,
                mode=norm_mode,
                autonomy=norm_autonomy,
                agent_name="pev_executor",
                max_steps=max_steps,
                emit=emit,
                cancel_event=c_event,
                approval_manager=gate,
                stuck_detector=stuck_detector,
                compactor=compactor,
                allow_mutations=True,
                initial_response=initial_exec_resp,
            )
            if was_cancelled:
                emit({"type": "cancelled", "run_id": rid, "reason": "Cancelled by user"})
                return {"run_id": rid, "cancelled": True, "answer": final_answer, "steps": len(trace)}

            ok_calls = sum(1 for t in trace if t["ok"])
            fail_calls = len(trace) - ok_calls
            emit(
                {
                    "type": "reflection",
                    "run_id": rid,
                    "step": len(trace),
                    "agent": "pev_verifier",
                    "reason": "pev_verify",
                    "message": f"PEV Verify: {ok_calls} succeeded, {fail_calls} failed across {len(trace)} tool call(s).",
                }
            )
            verified = bool(final_answer.strip()) and (not trace or trace[-1]["ok"])
            emit(
                {
                    "type": "done",
                    "run_id": rid,
                    "mode": norm_mode,
                    "answer": final_answer,
                    "verified": verified,
                    "steps": len(trace),
                }
            )
            return {"run_id": rid, "cancelled": False, "answer": final_answer, "steps": len(trace), "trace": trace}

        # Default: ReAct mode
        react_msgs = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": goal + (f"\nActive file: {active_file}" if active_file else ""),
            },
        ]
        final_answer, trace, was_cancelled = _execute_react_phase(
            harness=harness,
            tools=tools,
            workspace=workspace,
            messages=react_msgs,
            run_id=rid,
            mode=norm_mode,
            autonomy=norm_autonomy,
            agent_name="react",
            max_steps=max_steps,
            emit=emit,
            cancel_event=c_event,
            approval_manager=gate,
            stuck_detector=stuck_detector,
            compactor=compactor,
            allow_mutations=True,
        )
        if was_cancelled:
            emit({"type": "cancelled", "run_id": rid, "reason": "Cancelled by user"})
            return {"run_id": rid, "cancelled": True, "answer": final_answer, "steps": len(trace)}

        verified = bool(final_answer.strip()) and (not trace or trace[-1]["ok"])
        emit(
            {
                "type": "done",
                "run_id": rid,
                "mode": norm_mode,
                "answer": final_answer,
                "verified": verified,
                "steps": len(trace),
            }
        )
        return {"run_id": rid, "cancelled": False, "answer": final_answer, "steps": len(trace), "trace": trace}
    except Exception as exc:
        emit({"type": "error", "run_id": rid, "error": f"{type(exc).__name__}: {exc}"})
        return {"run_id": rid, "cancelled": False, "error": str(exc), "steps": 0}
    finally:
        gate.cancel_for_run(rid)


class TrainingSupervisor:
    """Autonomous Training Supervisor inspecting live training, MoE, CentroidSteer, and tier balance metrics."""

    def __init__(
        self,
        *,
        metrics_provider: Optional[Callable[[], dict[str, Any]]] = None,
        checkpoint_callback: Optional[Callable[[str, dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        self._metrics_provider = metrics_provider
        self._checkpoint_callback = checkpoint_callback
        self._lock = threading.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._poll_interval_s = 2.0
        self._loss_spike_ratio = 2.0
        self._grad_norm_threshold = 50.0
        self._auto_checkpoint = True
        self._loss_ema: Optional[float] = None
        self._loss_samples = 0
        self._inspections = 0
        self._anomalies: list[dict[str, Any]] = []
        self._diagnostics: list[str] = []
        self._recommendations: list[str] = []
        self._checkpoints_triggered: list[dict[str, Any]] = []
        self._last_snapshot: Optional[dict[str, Any]] = None
        self._last_checked_at: Optional[float] = None

    def _collect_metrics(self) -> dict[str, Any]:
        if self._metrics_provider is not None:
            return dict(self._metrics_provider())
        try:
            from lucid_machines.api.routes import pytorch_state
            from lucid_machines.api.telemetry_helpers import collect_mind_telemetry

            mind = collect_mind_telemetry(pytorch_state)
            latest_h = dict(pytorch_state.history[-1]) if getattr(pytorch_state, "history", None) else {}
            return {
                "step": int(getattr(pytorch_state, "step", 0) or 0),
                "is_training": bool(getattr(pytorch_state, "is_training", False)),
                "loss": latest_h.get("loss", getattr(pytorch_state, "last_loss", None)),
                "lm_loss": latest_h.get("lm_loss"),
                "router_loss": latest_h.get("router_loss"),
                "grad_norm": latest_h.get("grad_norm"),
                "depth_mode": getattr(pytorch_state, "depth_mode", "dynamic"),
                "mind": mind,
                "model_loaded": getattr(pytorch_state, "model", None) is not None,
            }
        except Exception as exc:
            return {"step": 0, "is_training": False, "error": str(exc), "mind": {}}

    def _default_safety_checkpoint(self, reason: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        if self._checkpoint_callback is not None:
            return dict(self._checkpoint_callback(reason, snapshot))
        try:
            import torch

            from lucid_machines.api.routes import pytorch_state

            if getattr(pytorch_state, "model", None) is None:
                return {"ok": False, "skipped": True, "reason": "No active PyTorch model loaded"}
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            step_n = int(getattr(pytorch_state, "step", 0) or 0)
            fname = f"supervisor_safety_step_{step_n}.pt"
            ckpt_path = ckpt_dir / fname
            loss_val = float(snapshot.get("loss") or 0.0) if snapshot.get("loss") is not None else 0.0
            payload = pytorch_state._build_checkpoint_payload(loss_val, to_cpu=True)
            tmp_path = ckpt_path.with_suffix(".tmp")
            torch.save(payload, tmp_path)
            if tmp_path.exists():
                tmp_path.replace(ckpt_path)
            return {"ok": True, "path": str(ckpt_path), "step": step_n, "reason": reason}
        except Exception as exc:
            return {"ok": False, "reason": f"{reason} (checkpoint failed: {exc})"}

    def inspect_once(self) -> dict[str, Any]:
        """Run one full health & anomaly inspection pass over live training and mind metrics."""
        raw = self._collect_metrics()
        step = int(raw.get("step") or 0)
        mind = raw.get("mind") or {}
        loss_val = raw.get("loss")
        lm_loss = raw.get("lm_loss")
        router_loss = raw.get("router_loss")
        grad_norm = raw.get("grad_norm")

        anomalies: list[dict[str, Any]] = []
        diagnostics: list[str] = []
        recommendations: list[str] = []
        severe = False

        # 1. NaN / Inf checks
        for metric_name, val in (
            ("loss", loss_val),
            ("lm_loss", lm_loss),
            ("router_loss", router_loss),
            ("grad_norm", grad_norm),
        ):
            if isinstance(val, (int, float)) and (math.isnan(float(val)) or math.isinf(float(val))):
                severe = True
                anomalies.append(
                    {
                        "kind": "nan_or_inf",
                        "severity": "critical",
                        "metric": metric_name,
                        "value": str(val),
                        "step": step,
                    }
                )
                diagnostics.append(f"Detected non-finite {metric_name}={val} at step {step}.")
                recommendations.append(
                    "Halt/rollback to last clean checkpoint, lower peak LR by 0.5x, and verify BF16/FP8 scaling."
                )

        # 2. Loss spike detection against rolling EMA
        if isinstance(loss_val, (int, float)) and math.isfinite(float(loss_val)) and float(loss_val) >= 0:
            cur_loss = float(loss_val)
            with self._lock:
                if self._loss_ema is not None and self._loss_samples >= 3 and self._loss_ema > 1e-6:
                    if cur_loss > self._loss_spike_ratio * self._loss_ema:
                        severe = True
                        anomalies.append(
                            {
                                "kind": "loss_spike",
                                "severity": "high",
                                "loss": round(cur_loss, 4),
                                "ema": round(self._loss_ema, 4),
                                "ratio": round(cur_loss / self._loss_ema, 2),
                                "step": step,
                            }
                        )
                        diagnostics.append(
                            f"Loss spike at step {step}: {cur_loss:.4f} vs EMA {self._loss_ema:.4f} ({cur_loss / self._loss_ema:.2f}x)."
                        )
                        recommendations.append(
                            "Save safety checkpoint, inspect outlier batch tokens, and reduce learning rate or enable gradient clipping."
                        )
                self._loss_samples += 1
                if self._loss_ema is None:
                    self._loss_ema = cur_loss
                elif cur_loss <= self._loss_spike_ratio * max(self._loss_ema, 1e-6):
                    self._loss_ema = 0.8 * self._loss_ema + 0.2 * cur_loss

        # 3. Grad-norm explosion
        if isinstance(grad_norm, (int, float)) and math.isfinite(float(grad_norm)):
            gn = float(grad_norm)
            if gn > self._grad_norm_threshold:
                severe = True
                anomalies.append(
                    {
                        "kind": "grad_norm_explosion",
                        "severity": "high",
                        "grad_norm": round(gn, 3),
                        "threshold": self._grad_norm_threshold,
                        "step": step,
                    }
                )
                diagnostics.append(
                    f"Gradient norm explosion at step {step}: {gn:.2f} > {self._grad_norm_threshold:.2f}."
                )
                recommendations.append("Clip global gradient norm <= 1.0 and lower Muon/CentroidSteer step size.")

        # 4. MoE expert collapse
        moe = mind.get("moe_routing") or {}
        util_pct = moe.get("utilization_pct") or []
        if isinstance(util_pct, list) and len(util_pct) >= 2 and sum(util_pct) > 0:
            min_u = min(float(x) for x in util_pct)
            max_u = max(float(x) for x in util_pct)
            if min_u < 2.0 or max_u > 85.0:
                severe = severe or (max_u > 90.0 or min_u == 0.0)
                anomalies.append(
                    {
                        "kind": "moe_expert_collapse",
                        "severity": "high" if (max_u > 90.0 or min_u == 0.0) else "medium",
                        "utilization_pct": util_pct,
                        "step": step,
                    }
                )
                diagnostics.append(f"MoE expert utilization collapse/skew detected: {util_pct}.")
                recommendations.append(
                    "Increase router load-balance loss weight (`balance_coef`) and verify router temperature."
                )

        # 5. CentroidSteer alignment energy & drift
        cs = mind.get("centroid_steer") or {}
        if cs.get("active"):
            energy = cs.get("alignment_energy")
            drift = cs.get("centroid_drift")
            if isinstance(energy, (int, float)):
                if (
                    math.isnan(float(energy))
                    or math.isinf(float(energy))
                    or float(energy) <= 1e-7
                    or float(energy) > 1e4
                ):
                    anomalies.append(
                        {
                            "kind": "centroid_steer_anomaly",
                            "severity": "medium",
                            "alignment_energy": energy,
                            "centroid_drift": drift,
                            "step": step,
                        }
                    )
                    diagnostics.append(f"CentroidSteer alignment energy out of healthy band: {energy}.")
                    recommendations.append(
                        "Adjust `steer_scale` or `steer_update_gap` and inspect routed activation subspace norm."
                    )

        # 6. Reflex / Limbic / Cortex routing balance
        tiers = mind.get("triune_tiers") or {}
        exit_usage = tiers.get("exit_usage")
        depth_mode = str(tiers.get("depth_mode") or raw.get("depth_mode") or "dynamic").lower()
        if isinstance(exit_usage, dict) and depth_mode in {"dynamic", "joint"}:
            vals = [float(exit_usage.get(k, 0.0) or 0.0) for k in ("reflex", "limbic", "cortex")]
            if sum(vals) > 0 and (min(vals) < 1.0 or max(vals) > 95.0):
                anomalies.append(
                    {
                        "kind": "tier_routing_imbalance",
                        "severity": "medium",
                        "exit_usage": exit_usage,
                        "step": step,
                    }
                )
                diagnostics.append(f"Triune 3-tier exit balance skewed in {depth_mode} mode: {exit_usage}.")
                recommendations.append("Rebalance `target_depth_dist` or tune exit router temperature.")

        if not diagnostics:
            diagnostics.append(f"Step {step}: training and mind telemetry within nominal thresholds.")

        snapshot = {
            "step": step,
            "loss": loss_val,
            "grad_norm": grad_norm,
            "anomalies": anomalies,
            "diagnostics": diagnostics,
            "recommendations": list(dict.fromkeys(recommendations)),
            "moe_utilization_pct": util_pct,
            "centroid_alignment_energy": cs.get("alignment_energy"),
            "exit_usage": exit_usage,
            "checked_at": time.time(),
        }

        if severe and self._auto_checkpoint:
            reason = "; ".join(a["kind"] for a in anomalies)
            ckpt_res = self._default_safety_checkpoint(reason, snapshot)
            with self._lock:
                self._checkpoints_triggered.append(
                    {
                        "step": step,
                        "reason": reason,
                        "timestamp": time.time(),
                        **ckpt_res,
                    }
                )
                self._checkpoints_triggered = self._checkpoints_triggered[-20:]
            snapshot["safety_checkpoint"] = ckpt_res

        with self._lock:
            self._inspections += 1
            self._last_checked_at = snapshot["checked_at"]
            self._last_snapshot = snapshot
            if anomalies:
                self._anomalies.extend(anomalies)
                self._anomalies = self._anomalies[-50:]
            self._diagnostics = diagnostics[-10:]
            self._recommendations = list(dict.fromkeys(recommendations))[-10:]

        return snapshot

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await asyncio.to_thread(self.inspect_once)
            except Exception as exc:
                with self._lock:
                    self._diagnostics = [f"Supervisor inspection notice: {exc}"]
            await asyncio.sleep(max(0.1, float(self._poll_interval_s)))

    async def start(
        self,
        *,
        poll_interval_s: float = 2.0,
        loss_spike_ratio: float = 2.0,
        grad_norm_threshold: float = 50.0,
        auto_checkpoint: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            self._poll_interval_s = max(0.1, min(60.0, float(poll_interval_s)))
            self._loss_spike_ratio = max(1.1, float(loss_spike_ratio))
            self._grad_norm_threshold = max(1.0, float(grad_norm_threshold))
            self._auto_checkpoint = bool(auto_checkpoint)
            already = self._running and self._task is not None and not self._task.done()
            if not already:
                self._running = True
                self._task = asyncio.create_task(self._run_loop())
        return self.status()

    async def stop(self) -> dict[str, Any]:
        with self._lock:
            self._running = False
            task = self._task
            self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": bool(self._running),
                "poll_interval_s": self._poll_interval_s,
                "loss_spike_ratio": self._loss_spike_ratio,
                "grad_norm_threshold": self._grad_norm_threshold,
                "auto_checkpoint": self._auto_checkpoint,
                "inspections": self._inspections,
                "last_checked_at": self._last_checked_at,
                "anomalies": list(self._anomalies),
                "diagnostics": list(self._diagnostics),
                "recommendations": list(self._recommendations),
                "checkpoints_triggered": list(self._checkpoints_triggered),
                "last_snapshot": self._last_snapshot,
            }


class AgentLoopManager:
    """Coordinates active streaming runs, workspace rebinding, cancellation, and HITL approvals."""

    def __init__(self, workspace: str | Path = ".") -> None:
        self._lock = threading.Lock()
        self.workspace = Path(workspace).resolve(strict=False)
        self.approvals = ApprovalGateManager()
        self.supervisor = TrainingSupervisor()
        self._runs: dict[str, dict[str, Any]] = {}

    def rebind_workspace(self, workspace: str | Path) -> Path:
        resolved = Path(workspace).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Workspace path is not a directory: {workspace}")
        with self._lock:
            self.workspace = resolved
        return resolved

    def register_run(self, run_id: str, mode: str, goal: str) -> threading.Event:
        cancel_ev = threading.Event()
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "mode": mode,
                "goal": goal,
                "cancel_event": cancel_ev,
                "cancelled": False,
                "started_at": time.time(),
                "finished": False,
            }
        return cancel_ev

    def mark_finished(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["finished"] = True

    def cancel_run(self, run_id: str) -> bool:
        with self._lock:
            entry = self._runs.get(run_id)
            if entry is None:
                return False
            entry["cancelled"] = True
            entry["cancel_event"].set()
        self.approvals.cancel_for_run(run_id)
        return True

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._runs[run_id]) if run_id in self._runs else None
