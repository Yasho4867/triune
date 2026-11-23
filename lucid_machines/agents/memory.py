"""Persistent memory, user-need modeling, tool proficiency learning, and self-reflection."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ._storage import atomic_write_json, compute_tfidf_cosine_scores, load_validated_json_dict, tokenize_terms


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
    """Empirical reliability, calibrated confidence, and latency profile learned per tool."""

    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_error: str | None = None
    learned_tips: list[str] = field(default_factory=list)
    recovery_transitions: dict[str, int] = field(default_factory=dict)
    ewma_success_rate: float = 1.0
    context_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return (self.successes / self.calls) if self.calls > 0 else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_ms / self.calls) if self.calls > 0 else 0.0

    def calibrated_reliability(self, context: str | None = None) -> float:
        """Beta-smoothed + EWMA-calibrated reliability estimate, optionally conditioned on task context."""
        if self.calls <= 0:
            return 0.75
        bayesian_prior = (self.successes + 1.0) / (self.calls + 2.0)
        ewma_blend = min(0.7, self.calls / 10.0)
        est = (1.0 - ewma_blend) * bayesian_prior + ewma_blend * max(0.0, min(1.0, self.ewma_success_rate))
        if context:
            cs = self.context_stats.get(context.lower())
            if isinstance(cs, dict) and cs.get("calls", 0) >= 2:
                ctx_rate = (cs.get("successes", 0) + 1.0) / (cs["calls"] + 2.0)
                est = 0.7 * est + 0.3 * ctx_rate
        return max(0.0, min(1.0, round(est, 4)))


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
        data = load_validated_json_dict(self._memory_path())
        if not data:
            return

        raw_prefs = data.get("preferences")
        if isinstance(raw_prefs, dict):
            for key, item in raw_prefs.items():
                if not isinstance(key, str) or not isinstance(item, dict) or "value" not in item:
                    continue
                self.preferences[key] = UserPreference(
                    key=str(item.get("key") or key),
                    value=str(item.get("value") or ""),
                    category=str(item.get("category") or "preference"),
                    priority=max(1, min(10, int(item.get("priority", 5) or 5))),
                    updated_at=float(item.get("updated_at", time.time()) or time.time()),
                )

        raw_tools = data.get("tool_stats")
        if isinstance(raw_tools, dict):
            for name, item in raw_tools.items():
                if not isinstance(name, str) or not isinstance(item, dict):
                    continue
                calls = max(0, int(item.get("calls", 0) or 0))
                succ = max(0, int(item.get("successes", 0) or 0))
                default_ewma = (succ / calls) if calls > 0 else 1.0
                self.tool_stats[name] = ToolStats(
                    name=str(item.get("name") or name),
                    calls=calls,
                    successes=succ,
                    failures=max(0, int(item.get("failures", 0) or 0)),
                    total_latency_ms=max(0.0, float(item.get("total_latency_ms", 0.0) or 0.0)),
                    last_error=(str(item["last_error"]) if item.get("last_error") is not None else None),
                    learned_tips=[str(t) for t in (item.get("learned_tips") or []) if isinstance(t, str)],
                    recovery_transitions={
                        str(k): int(v)
                        for k, v in (item.get("recovery_transitions") or {}).items()
                        if isinstance(v, (int, float))
                    },
                    ewma_success_rate=max(0.0, min(1.0, float(item.get("ewma_success_rate", default_ewma)))),
                    context_stats={
                        str(k): {
                            "calls": int(v.get("calls", 0)),
                            "successes": int(v.get("successes", 0)),
                        }
                        for k, v in (item.get("context_stats") or {}).items()
                        if isinstance(v, dict)
                    },
                )

        raw_lessons = data.get("lessons")
        if isinstance(raw_lessons, list):
            for item in raw_lessons:
                if not isinstance(item, dict) or not item.get("lesson"):
                    continue
                self.lessons.append(
                    LearnedLesson(
                        lesson_id=str(item.get("lesson_id") or f"lesson_{len(self.lessons) + 1}"),
                        trigger_keywords=[str(w) for w in (item.get("trigger_keywords") or []) if isinstance(w, str)],
                        lesson=str(item["lesson"]),
                        source_tool=(str(item["source_tool"]) if item.get("source_tool") is not None else None),
                        confidence=max(0.1, min(1.0, float(item.get("confidence", 0.7) or 0.7))),
                        times_applied=max(0, int(item.get("times_applied", 0) or 0)),
                        created_at=float(item.get("created_at", time.time()) or time.time()),
                    )
                )

    def save(self) -> None:
        payload = {
            "schema_version": 2,
            "preferences": {k: asdict(v) for k, v in self.preferences.items()},
            "tool_stats": {k: asdict(v) for k, v in self.tool_stats.items()},
            "lessons": [asdict(x) for x in self.lessons[-200:]],
        }
        atomic_write_json(self._memory_path(), payload)

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

        # Capture explicit "I want/need/prefer ..." statements cleanly with deterministic key hashing
        explicit = re.findall(r"(?:i prefer|always|please make sure|remember that)\s+([^.!?\n]{8,110})", text, re.I)
        for stmt in explicit[:2]:
            norm_stmt = re.sub(r"\s+", " ", stmt.strip().lower())
            digest = hashlib.sha256(norm_stmt.encode("utf-8")).hexdigest()[:8]
            slug = f"user_rule_{digest}"
            captured.append(self.remember_preference(slug, stmt.strip(), category="preference", priority=8))
        return captured

    def delete_preference(self, key: str) -> bool:
        """Delete a stored user preference by exact or normalized key."""
        norm_key = re.sub(r"[^a-z0-9._-]+", "_", key.strip().lower()).strip("_")
        removed = self.preferences.pop(key, None) or self.preferences.pop(norm_key, None)
        if removed is not None:
            self.save()
            return True
        return False

    def clear_preferences(self) -> int:
        """Remove all stored user preferences and return count deleted."""
        cnt = len(self.preferences)
        self.preferences.clear()
        self.save()
        return cnt

    def clear_lessons(self) -> int:
        """Remove all stored learned lessons and return count deleted."""
        cnt = len(self.lessons)
        self.lessons.clear()
        self.save()
        return cnt

    def clear_all(self) -> dict[str, int]:
        """Reset all stored preferences, lessons, and tool stats."""
        counts = {
            "preferences": len(self.preferences),
            "lessons": len(self.lessons),
            "tool_stats": len(self.tool_stats),
        }
        self.preferences.clear()
        self.lessons.clear()
        self.tool_stats.clear()
        self._last_failed_tool = None
        self.save()
        return counts

    # ---- Tool Proficiency & Experience Learning ----

    def record_tool_outcome(
        self,
        tool_name: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        error: str | None = None,
        tip: str | None = None,
        context: str | None = None,
    ) -> ToolStats:
        """Update empirical reliability, EWMA calibration, recovery paths, and learned pitfalls."""
        st = self.tool_stats.get(tool_name)
        if st is None:
            st = ToolStats(name=tool_name)
            self.tool_stats[tool_name] = st

        st.calls += 1
        st.total_latency_ms += max(0.0, float(latency_ms))
        obs = 1.0 if success else 0.0
        st.ewma_success_rate = obs if st.calls == 1 else round(0.75 * st.ewma_success_rate + 0.25 * obs, 4)

        if context:
            ctx_key = str(context).strip().lower()
            if ctx_key:
                cs = st.context_stats.setdefault(ctx_key, {"calls": 0, "successes": 0})
                cs["calls"] += 1
                if success:
                    cs["successes"] += 1

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

    def rank_tools_hint(self, tool_names: list[str], *, context: str | None = None) -> dict[str, str]:
        """Produce compact empirical performance + calibrated reliability hints for the prompt catalog."""
        hints: dict[str, str] = {}
        for name in tool_names:
            st = self.tool_stats.get(name)
            if st is None or st.calls == 0:
                continue
            rate = int(round(st.calibrated_reliability(context=context) * 100))
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
        kw = [w.lower() for w in (keywords or tokenize_terms(clean, include_char_ngrams=False)[:6])]
        # Reinforce existing near-identical lesson instead of duplicating
        for existing in self.lessons:
            if existing.lesson.lower() == clean.lower():
                existing.confidence = min(1.0, round(existing.confidence + 0.1, 2))
                existing.times_applied += 1
                self.save()
                return existing

        digest = hashlib.sha256(clean.lower().encode("utf-8")).hexdigest()[:6]
        entry = LearnedLesson(
            lesson_id=f"lesson_{len(self.lessons) + 1}_{digest}",
            trigger_keywords=kw,
            lesson=clean,
            source_tool=source_tool,
            confidence=max(0.1, min(1.0, float(confidence))),
        )
        self.lessons.append(entry)
        self.save()
        return entry

    def retrieve_relevant_lessons(self, query: str, limit: int = 5) -> list[LearnedLesson]:
        """Retrieve lessons ranked by TF-IDF subword similarity, confidence, and recency."""
        if not self.lessons:
            return []
        docs = [
            f"{' '.join(entry.trigger_keywords)} {entry.source_tool or ''} {entry.lesson}" for entry in self.lessons
        ]
        sims = compute_tfidf_cosine_scores(query, docs)
        now = time.time()
        has_query = bool(tokenize_terms(query, include_char_ngrams=False))
        scored: list[tuple[float, LearnedLesson]] = []
        for entry, sim in zip(self.lessons, sims):
            if has_query and sim <= 0.01 and len(self.lessons) > limit:
                continue
            age_days = max(0.0, (now - entry.created_at) / 86400.0)
            recency = math.exp(-age_days / 30.0)
            score = sim * 3.0 + entry.confidence * 0.45 + recency * 0.2 + min(0.25, entry.times_applied * 0.05)
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
