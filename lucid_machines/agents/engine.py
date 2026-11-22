"""LucidMachines Self-Learning Cognitive Agent Harness & Multi-Agent Orchestrator."""

from __future__ import annotations

import concurrent.futures
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from lucid_machines.runtime.sandbox import PythonSandbox

from .memory import AgentMemoryStore
from .personas import StateController
from .providers import ProviderManager
from .skills import SkillLibrary
from .tools import ToolRegistry, build_integrated_toolbelt


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract JSON action objects from raw model output (plain JSON or fenced ```json)."""
    objs: list[dict[str, Any]] = []
    if not text:
        return objs
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)\n```", text, re.S | re.I)
    candidates = [*fenced, text.strip()]
    for cand in candidates:
        s = cand.strip()
        if not s:
            continue
        # Scan balanced braces if embedded inside prose
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end > start:
            snippet = s[start : end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    objs.append(parsed)
                    break
            except json.JSONDecodeError:
                pass
    return objs


def _parse_model_turn(resp: Dict[str, Any]) -> tuple[str, list[dict[str, Any]], str | None]:
    """Parse (thought, tool_calls, final_answer) across native tool_calls and JSON ReAct replies."""
    raw_calls = resp.get("tool_calls") or []
    content = str(resp.get("content") or "").strip()

    if raw_calls:
        norm_calls = []
        for tc in raw_calls:
            if isinstance(tc, dict) and tc.get("name"):
                norm_calls.append({"name": str(tc["name"]), "arguments": dict(tc.get("arguments") or {})})
        if norm_calls:
            return content, norm_calls, None

    for obj in _extract_json_objects(content):
        thought = str(obj.get("thought") or obj.get("reasoning") or "").strip()
        action = str(obj.get("action") or obj.get("tool") or "").strip()
        if action.lower() in {"final", "answer", "done", "finish"}:
            ans = str(obj.get("answer") or obj.get("response") or obj.get("output") or content).strip()
            return thought, [], ans
        if action:
            args = obj.get("arguments") or obj.get("args") or obj.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return thought, [{"name": action, "arguments": dict(args)}], None
        if "answer" in obj and isinstance(obj["answer"], str):
            return thought, [], obj["answer"].strip()

    return "", [], content


@dataclass
class Agent:
    """Composable agent node compatible with standalone tasks and MultiAgentOrchestrator."""

    name: str
    role: str
    system_prompt: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    sandbox: PythonSandbox = field(default_factory=PythonSandbox)

    def run_task(self, task_description: str, *, harness: "AgentHarness | None" = None, **kwargs: Any) -> str:
        """Execute task standalone or via full AgentHarness when attached."""
        if harness is not None:
            res = harness.run_goal(
                f"[{self.name} · {self.role}] {task_description}\nGuidance: {self.system_prompt}",
                **kwargs,
            )
            return str(res.get("answer") or "")
        return f"Agent [{self.name} - {self.role}] processing task: {task_description}"


class MultiAgentOrchestrator:
    """Orchestrates sequential pipelines or parallel specialist fan-out across agents."""

    def __init__(self, harness: "AgentHarness | None" = None) -> None:
        self.agents: List[Agent] = []
        self.harness = harness

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def execute_pipeline(self, goal: str, *, model: str | None = None, max_steps: int | None = None) -> List[str]:
        results = []
        prior_context = ""
        for agent in self.agents:
            prompt = f"Collaborative Goal: {goal}"
            if prior_context:
                prompt += f"\nPrior Specialist Findings:\n{prior_context}"
            run_kw: Dict[str, Any] = {}
            if self.harness:
                run_kw["model"] = model
                if max_steps is not None:
                    run_kw["max_steps"] = max_steps
            res = agent.run_task(prompt, harness=self.harness, **run_kw)
            results.append(res)
            prior_context = res[:1200]
        return results

    def execute_parallel(
        self,
        goal: str,
        *,
        model: str | None = None,
        max_workers: int = 4,
        max_steps: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Dispatch all registered specialist agents concurrently with isolated state per worker."""
        if not self.agents:
            return []

        def _run_one(agent: Agent) -> Dict[str, Any]:
            worker_harness: AgentHarness | None = None
            if self.harness is not None:
                worker_harness = AgentHarness(
                    workspace=self.harness.workspace,
                    store_dir=self.harness.store_dir,
                    model=model or self.harness.model,
                    allow_shell=bool(self.harness.tools.get("run_shell") is not None),
                    allow_writes=bool(self.harness.tools.get("write_file") is not None),
                    completion_fn=self.harness.completion_fn,
                    providers=self.harness.providers,
                    depth=self.harness.depth + 1,
                )
            run_kw: Dict[str, Any] = {}
            if worker_harness:
                run_kw["model"] = model
                if max_steps is not None:
                    run_kw["max_steps"] = max_steps
            output = agent.run_task(
                f"Parallel Specialist Goal: {goal}",
                harness=worker_harness,
                **run_kw,
            )
            return {"agent": agent.name, "role": agent.role, "ok": True, "output": output}

        n_workers = max(1, min(int(max_workers), len(self.agents)))
        outputs: List[Dict[str, Any] | None] = [None] * len(self.agents)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_map = {pool.submit(_run_one, ag): idx for idx, ag in enumerate(self.agents)}
            for fut in concurrent.futures.as_completed(future_map):
                idx = future_map[fut]
                ag = self.agents[idx]
                try:
                    outputs[idx] = fut.result()
                except Exception as exc:
                    outputs[idx] = {
                        "agent": ag.name,
                        "role": ag.role,
                        "ok": False,
                        "output": f"Worker error: {exc}",
                    }
        return [o for o in outputs if o is not None]

    def orchestrate_team(
        self,
        goal: str,
        personas: List[str],
        *,
        mode: str = "parallel",
        model: str | None = None,
        max_workers: int = 4,
        max_steps: int = 6,
    ) -> Dict[str, Any]:
        """Build specialist agents from persona profiles and run them in parallel or pipeline mode."""
        known = self.harness.state_controller.personas if self.harness is not None else {}
        selected = [p.strip().lower() for p in personas if str(p).strip()] or ["architect", "engineer", "sentinel"]
        self.agents = []
        for pname in selected:
            prof = known.get(pname)
            role = prof.title if prof is not None else pname.title()
            sys_prompt = prof.system_prompt if prof is not None else f"Specialist perspective: {role}"
            self.add_agent(Agent(name=pname, role=role, system_prompt=sys_prompt))

        clean_mode = str(mode or "parallel").strip().lower()
        if clean_mode == "pipeline":
            raw_outputs = self.execute_pipeline(goal, model=model, max_steps=max_steps)
            results = [
                {"agent": ag.name, "role": ag.role, "ok": True, "output": out}
                for ag, out in zip(self.agents, raw_outputs)
            ]
        else:
            clean_mode = "parallel"
            results = self.execute_parallel(goal, model=model, max_workers=max_workers, max_steps=max_steps)
        return {"mode": clean_mode, "goal": goal, "results": results}


class AgentHarness:
    """Autonomous cognitive harness with adaptive personas, memory, skill creation, and tool learning."""

    def __init__(
        self,
        *,
        workspace: str | Path = ".",
        store_dir: str | Path = "checkpoints/agent_harness",
        model: str | None = None,
        allow_shell: bool = False,
        allow_writes: bool = True,
        allow_local_http: bool = False,
        completion_fn: Callable[..., Dict[str, Any]] | None = None,
        providers: ProviderManager | None = None,
        depth: int = 0,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.store_dir = Path(store_dir)
        self.providers = providers or ProviderManager()
        self.model = model or (None if completion_fn is not None else self.providers.pick_default_model())
        self.completion_fn = completion_fn
        self.depth = depth
        self.allow_local_http = allow_local_http

        self.memory = AgentMemoryStore(self.store_dir)
        self.state_controller = StateController(self.store_dir)
        repo_skills_dir = self.workspace / "checkpoints" / "agent_harness" / "skills"
        extra_dirs = [repo_skills_dir] if repo_skills_dir.resolve() != (self.store_dir / "skills").resolve() else []
        self.skills = SkillLibrary(self.store_dir / "skills", extra_skill_dirs=extra_dirs)

        subagent_runner = None
        if self.depth == 0:
            subagent_runner = self._run_specialist_subagent

        self.tools = build_integrated_toolbelt(
            workspace=self.workspace,
            memory=self.memory,
            skills=self.skills,
            state_controller=self.state_controller,
            subagent_runner=subagent_runner,
            allow_shell=allow_shell,
            allow_writes=allow_writes,
            allow_local_http=allow_local_http,
        )

    def rebind_workspace(self, workspace: str | Path) -> Path:
        """Rebind the harness and all workspace-scoped tools to a canonical project directory."""
        resolved = Path(workspace).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Workspace path is not a directory: {workspace}")
        self.workspace = resolved
        if hasattr(self.tools, "rebind_workspace"):
            self.tools.rebind_workspace(resolved)
        repo_skills_dir = self.workspace / "checkpoints" / "agent_harness" / "skills"
        extra_dirs = [repo_skills_dir] if repo_skills_dir.resolve() != (self.store_dir / "skills").resolve() else []
        self.skills = SkillLibrary(self.store_dir / "skills", extra_skill_dirs=extra_dirs)
        return self.workspace

    def _run_specialist_subagent(self, goal: str, persona: str | None = None) -> Dict[str, Any]:
        """Run a bounded specialist subagent sharing persistent memory & skills cleanly."""
        sub = AgentHarness(
            workspace=self.workspace,
            store_dir=self.store_dir,
            model=self.model,
            allow_shell=bool(self.tools.get("run_shell") is not None),
            allow_writes=bool(self.tools.get("write_file") is not None),
            allow_local_http=self.allow_local_http,
            completion_fn=self.completion_fn,
            providers=self.providers,
            depth=self.depth + 1,
        )
        if persona:
            sub.state_controller.shift(persona=persona, reason="specialist subagent role")
        res = sub.run_goal(goal, max_steps=5, auto_persona=not bool(persona), auto_crystallize=False)
        return {
            "persona": res["persona"],
            "reasoning_depth": res["reasoning_depth"],
            "steps": res["steps"],
            "answer": res["answer"],
        }

    def _build_system_prompt(self, goal: str, matched_skills: list[Any]) -> str:
        st = self.state_controller.state
        prof = self.state_controller.active_profile

        depth_directive = {
            "reflex": "Reasoning Tier: REFLEX — respond swiftly with minimal extra tool hops when straightforward.",
            "limbic": "Reasoning Tier: LIMBIC — balance thorough inspection with concise iterative execution.",
            "cortex": "Reasoning Tier: CORTEX — reason deliberately step-by-step, double-check assumptions, and verify tool outputs before answering.",
        }.get(st.reasoning_depth, "")

        blocks: list[str] = [
            f"You are Lucid Agent operating as [{prof.title}] (persona={st.persona}).",
            prof.system_prompt,
            depth_directive,
        ]
        if st.focus_note:
            blocks.append(f"Current Cognitive Focus: {st.focus_note}")

        mem_ctx = self.memory.format_context_block(goal)
        if mem_ctx:
            blocks.append(mem_ctx)

        if matched_skills:
            sk_lines = []
            for s in matched_skills:
                sk_lines.append(f"### Skill: {s.name} (`{s.slug}`)\n{s.instructions[:700]}")
            blocks.append("Activated Skills:\n" + "\n\n".join(sk_lines))

        catalog = self.tools.format_prompt_catalog(
            self.memory, preferred_tools=prof.preferred_tools, context=st.persona
        )
        blocks.append(
            "Available Tools (use JSON action format when calling without native tool_calls):\n"
            f"{catalog}\n\n"
            "Response Protocol:\n"
            '- To use a tool, reply with JSON: {"thought": "why this step", "action": "<tool_name>", "arguments": {...}}\n'
            "- To alter your persona or create/remember skills or user needs mid-task, call `shift_persona_state`, `create_skill`, `remember_user_preference`, or `record_lesson`.\n"
            '- When ready to reply to the user, reply with JSON: {"thought": "summary check", "action": "final", "answer": "<your verified response>"} or plain text without an action block.'
        )
        return "\n\n".join(blocks)

    def _invoke_model(self, messages: List[Dict[str, Any]], use_native_tools: bool = True) -> Dict[str, Any]:
        st = self.state_controller.state
        tool_schemas = self.tools.to_openai_tools() if use_native_tools else None
        if self.completion_fn is not None:
            return self.completion_fn(
                messages,
                tools=tool_schemas,
                state=st,
                persona=st.persona,
                temperature=st.temperature,
                reasoning_depth=st.reasoning_depth,
            )
        if not self.model:
            raise RuntimeError(
                "No LLM backend configured for AgentHarness. Pull/start a local Ollama model, configure a BYOK key, or pass --model."
            )
        return self.providers.complete_chat(
            self.model,
            messages,
            tools=tool_schemas,
            temperature=st.temperature,
            force_depth=st.reasoning_depth,
        )

    def run_goal(
        self,
        goal: str,
        *,
        model: str | None = None,
        max_steps: int = 10,
        on_event: Callable[[str, Dict[str, Any]], None] | None = None,
        auto_persona: bool = True,
        auto_depth: bool = True,
        auto_crystallize: bool = True,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        """Execute a full Think -> Act -> Observe -> Adapt -> Self-Learn loop for a user goal."""
        if model:
            self.model = model

        def _emit(ev_type: str, data: Dict[str, Any]) -> None:
            if on_event is not None:
                on_event(ev_type, data)

        # 1. Track explicit & implicit user wants/constraints automatically
        inferred_prefs = self.memory.infer_user_needs_from_prompt(goal)
        for p in inferred_prefs:
            _emit("memory_updated", {"key": p.key, "value": p.value, "category": p.category})

        # 2. Adapt cognitive state + personality for task intake
        st = self.state_controller.adapt_for_goal(goal, auto_persona=auto_persona, auto_depth=auto_depth)
        matched_skills = self.skills.match_skills(goal, limit=3)
        _emit(
            "state_selected",
            {
                "persona": st.persona,
                "title": self.state_controller.active_profile.title,
                "reasoning_depth": st.reasoning_depth,
                "matched_skills": [s.slug for s in matched_skills],
            },
        )

        # Rebuild system prompt each step if state/skills change mid-task
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(goal, matched_skills)}
        ]
        if conversation_history:
            messages.extend(conversation_history[-8:])
        messages.append({"role": "user", "content": goal})

        trace: list[dict[str, Any]] = []
        created_skills: list[str] = []
        learned_lessons: list[str] = []
        final_answer = ""
        hit_step_budget = False

        for step_num in range(1, max(1, int(max_steps)) + 1):
            messages[0]["content"] = self._build_system_prompt(goal, matched_skills)
            resp = self._invoke_model(messages)
            thought, calls, answer = _parse_model_turn(resp)

            if thought:
                _emit("thought", {"step": step_num, "persona": self.state_controller.state.persona, "thought": thought})

            if answer is not None and not calls:
                final_answer = answer
                break

            if not calls:
                final_answer = str(resp.get("content") or "")
                break

            # Append assistant call turn
            messages.append(
                {
                    "role": "assistant",
                    "content": str(resp.get("content") or json.dumps({"thought": thought, "tool_calls": calls})),
                }
            )

            for call in calls:
                t_name = str(call["name"])
                t_args = dict(call.get("arguments") or {})
                _emit("tool_start", {"step": step_num, "tool": t_name, "arguments": t_args})

                prev_persona = self.state_controller.state.persona
                prev_depth = self.state_controller.state.reasoning_depth
                outcome = self.tools.execute(t_name, t_args, memory=self.memory, context=prev_persona)
                new_state = self.state_controller.adapt_after_tool(t_name, outcome["ok"], outcome["error"])

                if t_name == "create_skill" and outcome["ok"] and isinstance(outcome["result"], dict):
                    slug = str(outcome["result"].get("slug") or "")
                    if slug:
                        created_skills.append(slug)
                elif t_name == "record_lesson" and outcome["ok"] and isinstance(outcome["result"], dict):
                    les = str(outcome["result"].get("lesson") or "")
                    if les:
                        learned_lessons.append(les)

                trace.append(
                    {
                        "step": step_num,
                        "tool": t_name,
                        "arguments": t_args,
                        "ok": outcome["ok"],
                        "success": outcome["ok"],
                        "latency_ms": outcome["latency_ms"],
                        "error": outcome["error"],
                        "persona": new_state.persona,
                    }
                )
                _emit(
                    "tool_result",
                    {
                        "step": step_num,
                        "tool": t_name,
                        "ok": outcome["ok"],
                        "latency_ms": outcome["latency_ms"],
                        "error": outcome["error"],
                        "result_preview": str(outcome["result"])[:220] if outcome["ok"] else None,
                    },
                )

                if new_state.persona != prev_persona or new_state.reasoning_depth != prev_depth:
                    _emit(
                        "state_shifted",
                        {
                            "from": f"{prev_persona}/{prev_depth}",
                            "to": f"{new_state.persona}/{new_state.reasoning_depth}",
                            "reason": new_state.focus_note,
                        },
                    )

                obs_payload: Dict[str, Any] = {
                    "tool": t_name,
                    "ok": outcome["ok"],
                    "result": outcome["result"] if outcome["ok"] else None,
                    "error": outcome["error"],
                }
                # Surface learned recovery advice immediately when a tool fails
                if not outcome["ok"]:
                    st_tool = self.memory.tool_stats.get(t_name)
                    if st_tool and st_tool.learned_tips:
                        obs_payload["learned_hint"] = st_tool.learned_tips[-1]
                    if st_tool and st_tool.recovery_transitions:
                        obs_payload["suggested_recovery_tool"] = max(
                            st_tool.recovery_transitions.items(), key=lambda kv: kv[1]
                        )[0]

                obs_text = json.dumps(obs_payload, default=str)[:4500]
                messages.append({"role": "user", "content": f"Tool Observation:\n{obs_text}"})
        else:
            hit_step_budget = True
            if not final_answer:
                final_answer = (
                    "Reached maximum reasoning step budget. Partial progress was recorded in the execution trace."
                )

        verified_outcome = (
            (not hit_step_budget)
            and bool(final_answer.strip())
            and bool(trace)
            and any(step["ok"] for step in trace)
            and bool(trace[-1]["ok"])
        )

        # 3. Post-task reflection & verified self-learning
        for sk in matched_skills:
            self.skills.record_outcome(sk.slug, bool(final_answer), verified=verified_outcome)

        # Automatically crystallize failure->recovery transitions into reusable lessons
        for idx in range(len(trace) - 1):
            cur, nxt = trace[idx], trace[idx + 1]
            if not cur["ok"] and nxt["ok"] and cur["tool"] != nxt["tool"]:
                auto_les = f"When `{cur['tool']}` fails ({(cur['error'] or '')[:60]}), recover with `{nxt['tool']}`."
                self.memory.record_lesson(auto_les, source_tool=cur["tool"], confidence=0.8)
                learned_lessons.append(auto_les)

        if auto_crystallize:
            synthesized = self.skills.auto_crystallize_from_trace(goal, trace, verified=verified_outcome)
            if synthesized is not None:
                created_skills.append(synthesized.slug)
                _emit("skill_synthesized", {"slug": synthesized.slug, "name": synthesized.name})

        return {
            "answer": final_answer,
            "verified": verified_outcome,
            "persona": self.state_controller.state.persona,
            "reasoning_depth": self.state_controller.state.reasoning_depth,
            "state_history": list(self.state_controller.state.history),
            "steps": len(trace),
            "trace": trace,
            "matched_skills": [s.slug for s in matched_skills],
            "created_skills": list(dict.fromkeys(created_skills)),
            "learned_lessons": list(dict.fromkeys(learned_lessons)),
        }

    def run_loop_stream(
        self,
        goal: str,
        *,
        mode: str = "react",
        autonomy: str = "supervised",
        active_file: str | None = None,
        max_steps: int = 10,
        model: str | None = None,
        emit: Callable[[Dict[str, Any]], None],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run an upgraded streaming agent loop (ReAct, PEV, Multi-Agent, or Training Supervisor)."""
        from .loops import run_agent_loop

        if model:
            self.model = model
        return run_agent_loop(
            harness=self,
            goal=goal,
            mode=mode,
            autonomy=autonomy,
            active_file=active_file,
            max_steps=max_steps,
            emit=emit,
            **kwargs,
        )
