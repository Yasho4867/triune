"""Persistent memory, user-need modeling, tool proficiency learning, and self-reflection."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class UserPreference:
    """Single tracked user want, need, constraint, or workflow habit."""

    key: str
    value: str
    category: str = "preference"  # "preference" | "constraint" | "goal" | "environment"
    priority: int = 5  # 1..10
    updated_at: float = field(default_factory=time.time)


@dataclass
class ToolStats:
    """Empirical reliability and latency profile learned per tool."""

    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_error: str | None = None
    learned_tips: list[str] = field(default_factory=list)
    recovery_transitions: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return (self.successes / self.calls) if self.calls > 0 else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_ms / self.calls) if self.calls > 0 else 0.0


@dataclass
class LearnedLesson:
    """Episodic self-reflection distilled from experience and mistakes."""

    lesson_id: str
    trigger_keywords: list[str]
    lesson: str
    source_tool: str | None = None
    confidence: float = 0.7
    times_applied: int = 0
    created_at: float = field(default_factory=time.time)


class AgentMemoryStore:
    """Persistent self-learning store covering user needs, tool mastery, and lessons."""

    def __init__(self, store_dir: str | Path = "checkpoints/agent_harness") -> None:
        self.store_dir = Path(store_dir)
        self.preferences: dict[str, UserPreference] = {}
        self.tool_stats: dict[str, ToolStats] = {}
        self.lessons: list[LearnedLesson] = []
        self._last_failed_tool: str | None = None
        self.load()

    def _memory_path(self) -> Path:
        return self.store_dir / "memory.json"

    def load(self) -> None:
        path = self._memory_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for key, item in (data.get("preferences") or {}).items():
            if isinstance(item, dict):
                self.preferences[key] = UserPreference(**item)
        for name, item in (data.get("tool_stats") or {}).items():
            if isinstance(item, dict):
                self.tool_stats[name] = ToolStats(**item)
        for item in data.get("lessons") or []:
            if isinstance(item, dict):
                self.lessons.append(LearnedLesson(**item))

    def save(self) -> None:
        path = self._memory_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preferences": {k: asdict(v) for k, v in self.preferences.items()},
            "tool_stats": {k: asdict(v) for k, v in self.tool_stats.items()},
            "lessons": [asdict(x) for x in self.lessons[-200:]],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- User Profile & Needs ----

    def remember_preference(
        self,
        key: str,
        value: str,
        *,
        category: str = "preference",
        priority: int = 6,
    ) -> UserPreference:
        """Store or update what the user wants, dislikes, or needs remembered across sessions."""
        norm_key = re.sub(r"[^a-z0-9._-]+", "_", key.strip().lower()).strip("_") or "pref"
        pref = UserPreference(
            key=norm_key,
            value=str(value).strip(),
            category=category,
            priority=max(1, min(10, int(priority))),
            updated_at=time.time(),
        )
        self.preferences[norm_key] = pref
        self.save()
        return pref

    def infer_user_needs_from_prompt(self, prompt: str) -> list[UserPreference]:
        """Extract implicit wants/constraints when user states explicit preferences in chat."""
        captured: list[UserPreference] = []
        text = (prompt or "").strip()
        if not text:
            return captured

        patterns: list[tuple[str, str, str, int]] = [
            (
                "concise_style",
                r"\b(be concise|keep it short|no fluff|dont overexplain|don't overexplain)\b",
                "preference",
                7,
            ),
            ("real_execution", r"\b(no stubs|no fake|real weights|actual inference|locally)\b", "constraint", 9),
            ("vram_budget", r"\b(8gb vram|8 gb|low vram|laptop gpu)\b", "environment", 8),
            ("beautify_cli", r"\b(pretty cli|beautify.*cli|clean cli|rich terminal)\b", "preference", 6),
        ]
        for key, pat, cat, prio in patterns:
            m = re.search(pat, text, re.I)
            if m:
                captured.append(
                    self.remember_preference(key, f"User expressed: '{m.group(0)}'", category=cat, priority=prio)
                )

        # Capture explicit "I want/need/prefer ..." statements cleanly
        explicit = re.findall(r"(?:i prefer|always|please make sure|remember that)\s+([^.!?\n]{8,110})", text, re.I)
        for idx, stmt in enumerate(explicit[:2]):
            slug = f"user_rule_{abs(hash(stmt.lower())) % 10000}"
            captured.append(self.remember_preference(slug, stmt.strip(), category="preference", priority=8))
        return captured

    # ---- Tool Proficiency & Experience Learning ----

    def record_tool_outcome(
        self,
        tool_name: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        error: str | None = None,
        tip: str | None = None,
    ) -> ToolStats:
        """Update empirical reliability, recovery paths, and learned pitfalls for a tool."""
        st = self.tool_stats.get(tool_name)
        if st is None:
            st = ToolStats(name=tool_name)
            self.tool_stats[tool_name] = st

        st.calls += 1
        st.total_latency_ms += max(0.0, float(latency_ms))
        if success:
            st.successes += 1
            if self._last_failed_tool and self._last_failed_tool != tool_name:
                prev_fail = self.tool_stats.get(self._last_failed_tool)
                if prev_fail is not None:
                    prev_fail.recovery_transitions[tool_name] = prev_fail.recovery_transitions.get(tool_name, 0) + 1
            self._last_failed_tool = None
        else:
            st.failures += 1
            st.last_error = (error or "unknown error")[:200]
            self._last_failed_tool = tool_name
            inferred_tip = tip or self._diagnose_tool_tip(tool_name, st.last_error)
            if inferred_tip and inferred_tip not in st.learned_tips:
                st.learned_tips = (st.learned_tips + [inferred_tip])[-5:]
        if tip and success and tip not in st.learned_tips:
            st.learned_tips = (st.learned_tips + [tip])[-5:]

        self.save()
        return st

    @staticmethod
    def _diagnose_tool_tip(tool_name: str, error_msg: str) -> str:
        low = error_msg.lower()
        if "not found" in low or "no such file" in low:
            return f"Before calling {tool_name}, verify exact path via list_dir or search_code."
        if "exact" in low or "not match" in low:
            return "Read target lines first with read_file so replacement text matches whitespace verbatim."
        if "timeout" in low:
            return f"Narrow scope or split work before retrying {tool_name}."
        if "disallowed" in low or "permission" in low:
            return f"Check sandbox/policy bounds; use a permitted tool instead of raw {tool_name}."
        return f"Validate arguments carefully when calling {tool_name} (previous error: {error_msg[:80]})."

    def rank_tools_hint(self, tool_names: list[str]) -> dict[str, str]:
        """Produce compact empirical performance + pitfall hints for the prompt catalog."""
        hints: dict[str, str] = {}
        for name in tool_names:
            st = self.tool_stats.get(name)
            if st is None or st.calls == 0:
                continue
            rate = int(round(st.success_rate * 100))
            parts = [f"reliability={rate}%", f"avg={st.avg_latency_ms:.0f}ms", f"calls={st.calls}"]
            if st.recovery_transitions:
                best_rec = max(st.recovery_transitions.items(), key=lambda kv: kv[1])[0]
                parts.append(f"on_fail->{best_rec}")
            if st.learned_tips:
                parts.append(f"tip='{st.learned_tips[-1]}'")
            hints[name] = " | ".join(parts)
        return hints

    # ---- Episodic Lessons ----

    def record_lesson(
        self,
        lesson: str,
        *,
        keywords: list[str] | None = None,
        source_tool: str | None = None,
        confidence: float = 0.75,
    ) -> LearnedLesson:
        """Persist a reusable operational lesson extracted from task execution."""
        clean = str(lesson or "").strip()
        kw = [w.lower() for w in (keywords or re.findall(r"[a-zA-Z0-9_]{4,}", clean)[:6])]
        # Reinforce existing near-identical lesson instead of duplicating
        for existing in self.lessons:
            if existing.lesson.lower() == clean.lower():
                existing.confidence = min(1.0, round(existing.confidence + 0.1, 2))
                existing.times_applied += 1
                self.save()
                return existing

        entry = LearnedLesson(
            lesson_id=f"lesson_{len(self.lessons) + 1}_{int(time.time()) % 10000}",
            trigger_keywords=kw,
            lesson=clean,
            source_tool=source_tool,
            confidence=max(0.1, min(1.0, float(confidence))),
        )
        self.lessons.append(entry)
        self.save()
        return entry

    def retrieve_relevant_lessons(self, query: str, limit: int = 5) -> list[LearnedLesson]:
        """Retrieve lessons most relevant to a query via keyword overlap + confidence."""
        tokens = set(re.findall(r"[a-zA-Z0-9_]{3,}", (query or "").lower()))
        scored: list[tuple[float, LearnedLesson]] = []
        for entry in self.lessons:
            overlap = len(tokens & set(entry.trigger_keywords))
            text_hits = sum(1 for t in tokens if t in entry.lesson.lower())
            score = overlap * 2.0 + text_hits * 0.5 + entry.confidence * 0.5
            if tokens and overlap == 0 and text_hits == 0 and len(self.lessons) > limit:
                continue
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def format_context_block(self, query: str) -> str:
        """Build a concise user-needs + self-learned context block for system injection."""
        sections: list[str] = []
        if self.preferences:
            top_prefs = sorted(self.preferences.values(), key=lambda p: (p.priority, p.updated_at), reverse=True)[:10]
            lines = [f"- [{p.category}] {p.key}: {p.value}" for p in top_prefs]
            sections.append("User Wants, Needs & Environment:\n" + "\n".join(lines))

        lessons = self.retrieve_relevant_lessons(query, limit=5)
        if lessons:
            lines = [f"- (conf {l.confidence:.2f}) {l.lesson}" for l in lessons]
            sections.append("Self-Learned Operational Lessons:\n" + "\n".join(lines))

        return "\n\n".join(sections)
