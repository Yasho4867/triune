"""Dynamic personas and cognitive state transitions for the Triune Agent Harness."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PersonaProfile:
    """Behavioral personality + reasoning bias tailored to a work domain."""

    name: str
    title: str
    system_prompt: str
    default_depth: str = "limbic"  # "reflex" | "limbic" | "cortex"
    temperature: float = 0.4
    preferred_tools: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    custom: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveState:
    """Mutable working state dynamically adjusted before and during a task."""

    persona: str = "engineer"
    reasoning_depth: str = "limbic"  # "reflex" (fast), "limbic" (balanced), "cortex" (deep)
    autonomy: str = "autonomous"  # "guided" | "collaborative" | "autonomous"
    temperature: float = 0.4
    verbosity: str = "concise"  # "minimal" | "concise" | "detailed"
    focus_note: str = ""
    consecutive_errors: int = 0
    history: list[str] = field(default_factory=list)


BUILTIN_PERSONAS: dict[str, PersonaProfile] = {
    "architect": PersonaProfile(
        name="architect",
        title="Systems Architect",
        system_prompt=(
            "Think in modular boundaries, failure modes, interfaces, and long-term maintainability. "
            "Map dependencies before modifying code, and articulate clear trade-offs."
        ),
        default_depth="cortex",
        temperature=0.3,
        preferred_tools=["list_dir", "search_code", "read_file", "plan_vram_memory", "create_skill"],
        traits=["structured", "systems-thinking", "foresight"],
    ),
    "engineer": PersonaProfile(
        name="engineer",
        title="Principal Software & ML Engineer",
        system_prompt=(
            "Be decisive, inspect exact code before editing, reuse repository abstractions cleanly, "
            "avoid duplicate logic, and verify changes with concrete tool outputs."
        ),
        default_depth="limbic",
        temperature=0.25,
        preferred_tools=["search_code", "read_file", "edit_file", "write_file", "run_shell"],
        traits=["pragmatic", "verified", "concise"],
    ),
    "researcher": PersonaProfile(
        name="researcher",
        title="ML & Algorithm Researcher",
        system_prompt=(
            "Reason from first principles and empirical evidence. Question unverified assumptions, "
            "compare baselines carefully, and run controlled checks before drawing conclusions."
        ),
        default_depth="cortex",
        temperature=0.35,
        preferred_tools=["read_file", "run_python_sandbox", "http_fetch", "scan_local_models", "record_lesson"],
        traits=["empirical", "rigorous", "curious"],
    ),
    "debugger": PersonaProfile(
        name="debugger",
        title="Root-Cause Triage Specialist",
        system_prompt=(
            "Never patch symptoms blindly. Isolate the smallest reproducible failure, trace exact inputs/outputs, "
            "verify environment/runtime assumptions, and fix the true root cause."
        ),
        default_depth="cortex",
        temperature=0.15,
        preferred_tools=["read_file", "search_code", "run_shell", "run_python_sandbox", "inspect_hardware"],
        traits=["skeptical", "methodical", "exact"],
    ),
    "optimizer": PersonaProfile(
        name="optimizer",
        title="Performance & Hardware Engineer",
        system_prompt=(
            "Focus on VRAM/RAM budgets, kernel fusion, quantization (NVFP4/FP8), latency, and throughput. "
            "Profile bottlenecks cleanly rather than guessing hot paths."
        ),
        default_depth="cortex",
        temperature=0.2,
        preferred_tools=["inspect_hardware", "plan_vram_memory", "scan_local_models", "transmute_model"],
        traits=["quantitative", "hardware-aware", "low-latency"],
    ),
    "tutor": PersonaProfile(
        name="tutor",
        title="Technical Mentor & Explainer",
        system_prompt=(
            "Explain concepts clearly in plain language grounded in concrete files or numbers. "
            "Adapt depth to what the user actually asked without filler."
        ),
        default_depth="reflex",
        temperature=0.5,
        preferred_tools=["read_file", "search_code"],
        traits=["clear", "empathetic", "grounded"],
    ),
    "sentinel": PersonaProfile(
        name="sentinel",
        title="Security & Safety Auditor",
        system_prompt=(
            "Scrutinize path safety, authentication checks, secret handling, resource limits, and command bounds. "
            "Prefer fail-closed designs and explicit boundary enforcement."
        ),
        default_depth="cortex",
        temperature=0.15,
        preferred_tools=["search_code", "read_file", "run_shell"],
        traits=["vigilant", "defensive", "thorough"],
    ),
}


_PERSONA_KEYWORD_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "debugger",
        re.compile(r"\b(bug|fix|error|crash|fail|trace|broken|wtf|why.*not|debug|hang|lag)\b", re.I),
        "cortex",
    ),
    (
        "optimizer",
        re.compile(r"\b(fast|speed|latency|vram|memory|nvfp4|fp8|quant|stream|perf|fps|benchmark)\b", re.I),
        "cortex",
    ),
    ("architect", re.compile(r"\b(architect|design|plan|harness|refactor|system|spec|structure)\b", re.I), "cortex"),
    (
        "researcher",
        re.compile(r"\b(paper|math|prove|theorem|experiment|distill|study|evaluate|compare|why)\b", re.I),
        "cortex",
    ),
    ("sentinel", re.compile(r"\b(auth|token|secret|security|sandbox|permission|cve|vuln|audit)\b", re.I), "cortex"),
    ("tutor", re.compile(r"\b(explain|what is|how does|teach|walk me through|summarize)\b", re.I), "reflex"),
    ("engineer", re.compile(r"\b(implement|build|create|add|write|update|patch|wire|make)\b", re.I), "limbic"),
]


class StateController:
    """Manages personas, custom personality registration, and adaptive state transitions."""

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self.personas: dict[str, PersonaProfile] = {k: v for k, v in BUILTIN_PERSONAS.items()}
        self.state = CognitiveState()
        self._load_custom_personas()

    def _custom_path(self) -> Path | None:
        if self.store_dir is None:
            return None
        return self.store_dir / "personas.json"

    def _load_custom_personas(self) -> None:
        path = self._custom_path()
        if path is None or not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, item in raw.items():
                    if isinstance(item, dict) and item.get("name"):
                        self.personas[str(key).lower()] = PersonaProfile(**item)
        except Exception:
            pass

    def _save_custom_personas(self) -> None:
        path = self._custom_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        custom_only = {k: v.to_dict() for k, v in self.personas.items() if v.custom}
        path.write_text(json.dumps(custom_only, indent=2), encoding="utf-8")

    def register_persona(
        self,
        name: str,
        title: str,
        system_prompt: str,
        *,
        default_depth: str = "limbic",
        temperature: float = 0.35,
        preferred_tools: list[str] | None = None,
        traits: list[str] | None = None,
    ) -> PersonaProfile:
        """Create or update a tailored cognitive personality for specialized work."""
        slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-") or "custom"
        depth = default_depth.lower() if default_depth.lower() in {"reflex", "limbic", "cortex"} else "limbic"
        profile = PersonaProfile(
            name=slug,
            title=title.strip() or slug.title(),
            system_prompt=system_prompt.strip(),
            default_depth=depth,
            temperature=max(0.0, min(1.5, float(temperature))),
            preferred_tools=list(preferred_tools or []),
            traits=list(traits or []),
            custom=True,
        )
        self.personas[slug] = profile
        self._save_custom_personas()
        return profile

    def shift(
        self,
        persona: str | None = None,
        *,
        reasoning_depth: str | None = None,
        temperature: float | None = None,
        verbosity: str | None = None,
        focus_note: str | None = None,
        reason: str = "",
    ) -> CognitiveState:
        """Explicitly alter active personality and cognitive reasoning parameters."""
        if persona:
            key = persona.strip().lower()
            if key not in self.personas:
                raise ValueError(f"Unknown persona '{persona}'. Available: {', '.join(sorted(self.personas))}")
            prof = self.personas[key]
            self.state.persona = key
            self.state.reasoning_depth = reasoning_depth or prof.default_depth
            self.state.temperature = temperature if temperature is not None else prof.temperature
        elif reasoning_depth in {"reflex", "limbic", "cortex"}:
            self.state.reasoning_depth = reasoning_depth
        if reasoning_depth in {"reflex", "limbic", "cortex"}:
            self.state.reasoning_depth = reasoning_depth
        if temperature is not None:
            self.state.temperature = max(0.0, min(1.5, float(temperature)))
        if verbosity in {"minimal", "concise", "detailed"}:
            self.state.verbosity = verbosity
        if focus_note is not None:
            self.state.focus_note = focus_note
        event = f"{self.state.persona}/{self.state.reasoning_depth}"
        if reason:
            event = f"{event} ({reason})"
        self.state.history.append(event)
        return self.state

    def adapt_for_goal(self, goal: str, *, auto_persona: bool = True, auto_depth: bool = True) -> CognitiveState:
        """Select initial personality and Triune reasoning depth based on the user's goal."""
        if not auto_persona and not auto_depth:
            return self.state
        chosen_persona = self.state.persona
        chosen_depth = self.state.reasoning_depth
        for name, pattern, suggested_depth in _PERSONA_KEYWORD_RULES:
            if pattern.search(goal or ""):
                chosen_persona = name
                chosen_depth = suggested_depth
                break
        return self.shift(
            persona=chosen_persona if auto_persona else None,
            reasoning_depth=chosen_depth if auto_depth else None,
            reason="initial goal intake",
        )

    def adapt_after_tool(self, tool_name: str, success: bool, error: str | None = None) -> CognitiveState:
        """Automatically escalate state/personality on repeated tool failures or recovery."""
        if success:
            self.state.consecutive_errors = 0
            return self.state

        self.state.consecutive_errors += 1
        # Escalate to debugger + cortex deep reasoning if tools keep failing
        if self.state.consecutive_errors >= 2 and self.state.persona != "debugger":
            return self.shift(
                "debugger",
                reasoning_depth="cortex",
                temperature=0.15,
                focus_note=f"Investigating repeated error in '{tool_name}': {(error or '')[:120]}",
                reason="auto-escalate on repeated errors",
            )
        if self.state.reasoning_depth == "reflex":
            return self.shift(
                reasoning_depth="cortex",
                focus_note=f"Re-checking '{tool_name}' failure in deeper tier",
                reason="depth escalation after failure",
            )
        return self.state

    @property
    def active_profile(self) -> PersonaProfile:
        return self.personas.get(self.state.persona) or BUILTIN_PERSONAS["engineer"]
