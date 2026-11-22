"""Self-authored, evolvable skill library for the LucidMachines Agent Harness."""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._storage import (
    atomic_write_json,
    atomic_write_text,
    compute_tfidf_cosine_scores,
    load_validated_json_dict,
    tokenize_terms,
)


@dataclass
class AgentSkill:
    """Reusable markdown + optional script workflow synthesized or loaded by the agent."""

    slug: str
    name: str
    description: str
    triggers: list[str]
    instructions: str
    steps: list[str] = field(default_factory=list)
    recommended_tools: list[str] = field(default_factory=list)
    helper_code: str | None = None
    uses: int = 0
    successes: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source_path: str | None = None

    @property
    def success_rate(self) -> float:
        return (self.successes / self.uses) if self.uses > 0 else 1.0


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", name.strip().lower()).strip("-") or "custom-skill"


def _format_skill_md(skill: AgentSkill) -> str:
    steps_md = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(skill.steps)) or "1. Apply workflow cleanly."
    tools_str = ", ".join(skill.recommended_tools) if skill.recommended_tools else "any"
    triggers_str = ", ".join(skill.triggers)
    return (
        f"---\n"
        f"name: {skill.name}\n"
        f"slug: {skill.slug}\n"
        f"description: {skill.description}\n"
        f"triggers: [{triggers_str}]\n"
        f"recommended_tools: [{tools_str}]\n"
        f"---\n\n"
        f"# {skill.name}\n\n"
        f"{skill.description}\n\n"
        f"## Instructions\n{skill.instructions.strip()}\n\n"
        f"## Procedure\n{steps_md}\n"
    )


def _parse_skill_md(md_path: Path) -> AgentSkill | None:
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    meta: dict[str, str] = {}
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if m:
        fm, body = m.group(1), m.group(2)
        for line in fm.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip().lower()] = v.strip()

    slug = _slugify(meta.get("slug") or meta.get("name") or md_path.parent.name)
    name = meta.get("name") or slug.replace("-", " ").title()
    desc = meta.get("description") or ""
    triggers_raw = meta.get("triggers", "").strip("[]")
    triggers = [t.strip().lower() for t in triggers_raw.split(",") if t.strip()]
    tools_raw = meta.get("recommended_tools", "").strip("[]")
    tools = [t.strip() for t in tools_raw.split(",") if t.strip() and t.strip() != "any"]
    helper_p = md_path.parent / "helper.py"
    helper_code = helper_p.read_text(encoding="utf-8", errors="replace") if helper_p.is_file() else None
    return AgentSkill(
        slug=slug,
        name=name,
        description=desc,
        triggers=triggers or re.findall(r"[a-zA-Z0-9_]{4,}", f"{name} {desc}".lower())[:6],
        instructions=body.strip(),
        recommended_tools=tools,
        helper_code=helper_code,
        source_path=str(md_path),
    )


class SkillLibrary:
    """Discovers, synthesizes, matches, and evolves reusable agent skills."""

    def __init__(
        self,
        store_dir: str | Path = "checkpoints/agent_harness/skills",
        extra_skill_dirs: list[str | Path] | None = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.extra_skill_dirs = [Path(p) for p in (extra_skill_dirs or [])]
        self.skills: dict[str, AgentSkill] = {}
        self._load_all()

    def _stats_path(self) -> Path:
        return self.store_dir / "_skill_stats.json"

    def _load_stats(self) -> dict[str, dict[str, Any]]:
        raw = load_validated_json_dict(self._stats_path())
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}

    def _save_stats(self) -> None:
        payload = {
            slug: {"uses": s.uses, "successes": s.successes, "updated_at": s.updated_at}
            for slug, s in self.skills.items()
        }
        atomic_write_json(self._stats_path(), payload)

    def _load_all(self) -> None:
        stats = self._load_stats()
        dirs = [self.store_dir, *self.extra_skill_dirs]
        for base in dirs:
            if not base.is_dir():
                continue
            for md in base.glob("*/SKILL.md"):
                skill = _parse_skill_md(md)
                if skill is not None:
                    st = stats.get(skill.slug)
                    if isinstance(st, dict):
                        skill.uses = max(0, int(st.get("uses", 0) or 0))
                        skill.successes = max(0, int(st.get("successes", 0) or 0))
                    self.skills[skill.slug] = skill

    def create_skill(
        self,
        name: str,
        description: str,
        instructions: str,
        *,
        triggers: list[str] | None = None,
        steps: list[str] | None = None,
        recommended_tools: list[str] | None = None,
        helper_code: str | None = None,
    ) -> AgentSkill:
        """Synthesize and atomically persist a SKILL.md (and sync/prune optional helper.py)."""
        slug = _slugify(name)
        trig = [t.strip().lower() for t in (triggers or []) if t.strip()]
        if not trig:
            trig = list(dict.fromkeys(tokenize_terms(f"{name} {description}", include_char_ngrams=False)))[:8]

        existing = self.skills.get(slug)
        skill = AgentSkill(
            slug=slug,
            name=name.strip(),
            description=description.strip(),
            triggers=trig,
            instructions=instructions.strip(),
            steps=list(steps or []),
            recommended_tools=list(recommended_tools or []),
            helper_code=helper_code,
            uses=existing.uses if existing else 0,
            successes=existing.successes if existing else 0,
            created_at=existing.created_at if existing else time.time(),
            updated_at=time.time(),
        )
        skill_dir = self.store_dir / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        md_path = skill_dir / "SKILL.md"
        atomic_write_text(md_path, _format_skill_md(skill))
        helper_path = skill_dir / "helper.py"
        if helper_code is not None:
            atomic_write_text(helper_path, helper_code)
        else:
            helper_path.unlink(missing_ok=True)
        skill.source_path = str(md_path)
        self.skills[slug] = skill
        self._save_stats()
        return skill

    def delete_skill(self, slug_or_name: str) -> bool:
        """Delete a skill from the library and remove its directory if under store_dir."""
        slug = _slugify(slug_or_name)
        if slug not in self.skills:
            return False
        self.skills.pop(slug, None)
        skill_dir = self.store_dir / slug
        if skill_dir.is_dir():
            shutil.rmtree(skill_dir, ignore_errors=True)
        self._save_stats()
        return True

    def evolve_skill(self, slug_or_name: str, amendment: str, *, add_tools: list[str] | None = None) -> AgentSkill:
        """Refine an existing skill cleanly with new edge cases or better steps."""
        slug = _slugify(slug_or_name)
        skill = self.skills.get(slug)
        if skill is None:
            raise KeyError(f"Skill '{slug_or_name}' not found.")
        updated_instructions = f"{skill.instructions.rstrip()}\n\n## Refined Practice\n- {amendment.strip()}"
        merged_tools = list(dict.fromkeys([*skill.recommended_tools, *(add_tools or [])]))
        return self.create_skill(
            skill.name,
            skill.description,
            updated_instructions,
            triggers=skill.triggers,
            steps=skill.steps,
            recommended_tools=merged_tools,
            helper_code=skill.helper_code,
        )

    def match_skills(self, query: str, limit: int = 3) -> list[AgentSkill]:
        """Rank relevant skills via TF-IDF subword cosine similarity + explicit triggers + verified reliability."""
        if not self.skills:
            return []
        q_low = (query or "").lower()
        q_words = set(tokenize_terms(q_low, include_char_ngrams=False))
        skill_list = list(self.skills.values())
        docs = [f"{s.name} {s.description} {' '.join(s.triggers)} {' '.join(s.recommended_tools)}" for s in skill_list]
        sims = compute_tfidf_cosine_scores(query, docs)
        scored: list[tuple[float, AgentSkill]] = []
        for s, sim in zip(skill_list, sims):
            trig_hits = sum(1.5 for t in s.triggers if t in q_low or t in q_words)
            reliability_bonus = 0.35 * ((s.successes + 1.0) / (s.uses + 2.0)) if s.uses > 0 else 0.0
            score = sim * 3.0 + trig_hits + reliability_bonus
            if trig_hits > 0 or sim >= 0.14:
                scored.append((score, s))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def record_outcome(self, slug_or_name: str, success: bool, *, verified: bool = False) -> None:
        """Record skill usage; credit ``successes`` only when verified by execution checks."""
        slug = _slugify(slug_or_name)
        skill = self.skills.get(slug)
        if skill is None:
            return
        skill.uses += 1
        if success and verified:
            skill.successes += 1
        skill.updated_at = time.time()
        self._save_stats()

    def auto_crystallize_from_trace(
        self,
        goal: str,
        trace_steps: list[dict[str, Any]],
        *,
        verified: bool = False,
    ) -> AgentSkill | None:
        """Turn verified, non-trivial multi-tool workflows into a reusable skill."""
        if not verified or not trace_steps:
            return None
        # Do not crystallize if final recorded tool failed without recovery
        if not trace_steps[-1].get("ok"):
            return None
        successful_tools = [str(s.get("tool")) for s in trace_steps if s.get("success") and s.get("tool")]
        unique_tools = list(dict.fromkeys(successful_tools))
        # Only crystallize genuinely non-trivial workflows with >=3 tool calls and >=2 distinct tools
        if len(successful_tools) < 3 or len(unique_tools) < 2:
            return None
        words = [w for w in tokenize_terms(goal, include_char_ngrams=False) if len(w) >= 3][:5]
        if len(words) < 2:
            return None
        name = " ".join(words).title()
        full_name = f"Auto: {name}"
        slug = _slugify(full_name)
        if slug in self.skills:
            return None
        steps = [f"Run `{t}` with validated workspace inputs." for t in unique_tools]
        return self.create_skill(
            name=full_name,
            description=f"Verified workflow ({' -> '.join(unique_tools)}) for goals matching: {goal[:100]}",
            instructions=f"Synthesized from verified execution trace solving: `{goal}`.",
            triggers=words,
            steps=steps,
            recommended_tools=unique_tools,
        )

    def summarize_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "slug": s.slug,
                "name": s.name,
                "description": s.description,
                "triggers": s.triggers,
                "uses": s.uses,
                "success_rate": round(s.success_rate, 3),
                "has_helper": bool(s.helper_code),
            }
            for s in sorted(self.skills.values(), key=lambda x: (x.uses, x.updated_at), reverse=True)
        ]
