"""Tests for the Self-Learning Triune Agent Harness (personas, skills, memory, tool mastery, subagents)."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from triune.agents import (
    Agent,
    AgentHarness,
    AgentMemoryStore,
    MultiAgentOrchestrator,
    SkillLibrary,
    StateController,
)
from triune.api.server import create_app


class TestTriuneAgentHarness(unittest.TestCase):
    def test_dynamic_persona_and_state_adaptation_with_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl = StateController(tmpdir)
            st = ctrl.adapt_for_goal("Fix this broken CUDA memory bug and crash in inference")
            self.assertEqual(st.persona, "debugger")
            self.assertEqual(st.reasoning_depth, "cortex")

            # Custom persona creation + disk persistence
            prof = ctrl.register_persona(
                "kernel_wizard",
                "CUDA & Triton Kernel Wizard",
                "Optimize tensor cores, tiling, and coalesced SRAM access.",
                default_depth="cortex",
                temperature=0.2,
                preferred_tools=["inspect_hardware", "read_file"],
            )
            self.assertTrue(prof.custom)
            ctrl.shift("kernel_wizard", focus_note="Fuse RMSNorm + RoPE")
            self.assertEqual(ctrl.state.persona, "kernel_wizard")

            # Reload in fresh StateController instance
            reloaded = StateController(tmpdir)
            self.assertIn("kernel_wizard", reloaded.personas)
            self.assertEqual(reloaded.personas["kernel_wizard"].title, "CUDA & Triton Kernel Wizard")

            # Auto-escalation on repeated tool failures
            reloaded.shift("tutor", reasoning_depth="reflex")
            reloaded.adapt_after_tool("read_file", success=False, error="FileNotFoundError")
            self.assertEqual(reloaded.state.reasoning_depth, "cortex")
            reloaded.adapt_after_tool("read_file", success=False, error="FileNotFoundError")
            self.assertEqual(reloaded.state.persona, "debugger")

    def test_user_needs_memory_and_tool_experience_learning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = AgentMemoryStore(tmpdir)
            mem.infer_user_needs_from_prompt(
                "Please make sure solutions run locally on my 8GB VRAM laptop GPU and avoid stubs."
            )
            self.assertTrue(any(p.key == "vram_budget" for p in mem.preferences.values()))
            self.assertTrue(any(p.key == "real_execution" for p in mem.preferences.values()))

            # Record tool failure followed by recovery tool -> builds transition affinity + pitfall tip
            mem.record_tool_outcome("edit_file", success=False, latency_ms=12.0, error="Target snippet does not match")
            mem.record_tool_outcome("read_file", success=True, latency_ms=8.0)

            hints = mem.rank_tools_hint(["edit_file", "read_file"])
            self.assertIn("on_fail->read_file", hints["edit_file"])
            self.assertIn("Read target lines first", hints["edit_file"])

            # Verify cross-session persistence
            mem2 = AgentMemoryStore(tmpdir)
            ctx = mem2.format_context_block("Edit code cleanly without blowing 8GB VRAM")
            self.assertIn("vram_budget", ctx)

    def test_skill_creation_matching_helper_sandbox_and_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = SkillLibrary(Path(tmpdir) / "skills")
            skill = lib.create_skill(
                name="NVFP4 VRAM Audit",
                description="Audit model VRAM requirements for 8GB GPUs using Python sandbox helpers.",
                instructions="Compute 4-bit weight size + KV headroom cleanly.",
                triggers=["vram", "nvfp4", "8gb", "audit"],
                steps=["Run helper script", "Verify fit inside 8GB"],
                recommended_tools=["plan_vram_memory", "run_skill_helper"],
                helper_code="gb = round(params_b * 0.625, 2)\nfits_8gb = gb <= 6.5\n",
            )
            self.assertTrue((Path(tmpdir) / "skills" / skill.slug / "SKILL.md").is_file())
            self.assertTrue((Path(tmpdir) / "skills" / skill.slug / "helper.py").is_file())

            matched = lib.match_skills("Can we audit nvfp4 vram fit on 8gb gpu?")
            self.assertTrue(any(s.slug == skill.slug for s in matched))

            # Evolve existing skill & run helper in sandbox
            lib.evolve_skill(
                skill.slug, "Always reserve 1.2 GB for KV cache and CUDA graph buffers.", add_tools=["inspect_hardware"]
            )
            reloaded = SkillLibrary(Path(tmpdir) / "skills")
            self.assertIn("reserve 1.2 GB", reloaded.skills[skill.slug].instructions)
            self.assertIn("inspect_hardware", reloaded.skills[skill.slug].recommended_tools)

    def test_autonomous_harness_loop_self_learns_shifts_state_and_synthesizes_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir) / "ws"
            ws.mkdir()
            (ws / "notes.txt").write_text("target_value=42\n", encoding="utf-8")
            store = Path(tmpdir) / "store"

            # Scripted multi-step LLM sequence demonstrating:
            # 1) failing edit -> 2) reading file (recovery!) -> 3) editing cleanly ->
            # 4) shifting persona -> 5) authoring a skill -> 6) running sandbox helper -> 7) final answer
            scripted_turns = iter(
                [
                    {
                        "content": json.dumps(
                            {
                                "thought": "Try replacing value in notes.txt",
                                "action": "edit_file",
                                "arguments": {"path": "notes.txt", "target_text": "missing", "replacement_text": "x"},
                            }
                        )
                    },
                    {
                        "content": json.dumps(
                            {
                                "thought": "edit_file failed, inspect exact contents first",
                                "action": "read_file",
                                "arguments": {"path": "notes.txt", "start_line": 1, "end_line": 10},
                            }
                        )
                    },
                    {
                        "content": json.dumps(
                            {
                                "thought": "Shift to optimizer and remember user's preference for exact edits",
                                "action": "shift_persona_state",
                                "arguments": {"persona": "optimizer", "reasoning_depth": "cortex"},
                            }
                        )
                    },
                    {
                        "content": json.dumps(
                            {
                                "thought": "Remember exact_edit preference",
                                "action": "remember_user_preference",
                                "arguments": {"key": "edit_style", "value": "Always inspect files before editing"},
                            }
                        )
                    },
                    {
                        "content": json.dumps(
                            {
                                "thought": "Create a reusable safe-config-updater skill with sandbox helper",
                                "action": "create_skill",
                                "arguments": {
                                    "name": "Safe Config Updater",
                                    "description": "Read exact text before editing workspace config values.",
                                    "instructions": "Always read_file before edit_file.",
                                    "triggers": ["config", "notes", "update"],
                                    "helper_code": "updated = value * 2\n",
                                },
                            }
                        )
                    },
                    {
                        "content": json.dumps(
                            {
                                "thought": "Verify skill helper runs in PythonSandbox",
                                "action": "run_skill_helper",
                                "arguments": {"name_or_slug": "safe-config-updater", "locals_json": '{"value": 21}'},
                            }
                        )
                    },
                    {
                        "content": json.dumps(
                            {
                                "thought": "Update notes.txt accurately",
                                "action": "edit_file",
                                "arguments": {
                                    "path": "notes.txt",
                                    "target_text": "target_value=42",
                                    "replacement_text": "target_value=84",
                                },
                            }
                        )
                    },
                    {
                        "content": json.dumps(
                            {
                                "thought": "Verified updated file and created reusable skill.",
                                "action": "final",
                                "answer": "Updated target_value to 84, recorded preference, and created safe-config-updater.",
                            }
                        )
                    },
                ]
            )

            events: list[str] = []
            harness = AgentHarness(
                workspace=ws,
                store_dir=store,
                allow_shell=False,
                allow_writes=True,
                completion_fn=lambda *_, **__: next(scripted_turns),
            )
            res = harness.run_goal(
                "Please update notes.txt cleanly",
                max_steps=10,
                on_event=lambda ev, _data: events.append(ev),
            )

            self.assertIn("target_value=84", (ws / "notes.txt").read_text(encoding="utf-8"))
            self.assertEqual(res["persona"], "optimizer")
            self.assertEqual(res["reasoning_depth"], "cortex")
            self.assertIn("safe-config-updater", res["created_skills"])
            self.assertGreaterEqual(len(res["learned_lessons"]), 1)
            self.assertIn("edit_style", harness.memory.preferences)
            self.assertIn("state_selected", events)
            self.assertIn("state_shifted", events)
            # Ensure MultiAgentOrchestrator also integrates cleanly
            orch = MultiAgentOrchestrator(harness=harness)
            orch.add_agent(Agent(name="Reviewer", role="Auditor", system_prompt="Double-check output"))
            harness.completion_fn = lambda *_, **__: {"content": "Verified output looks clean.", "tool_calls": []}
            pipe_out = orch.execute_pipeline("Review updated notes")
            self.assertIn("Verified output", pipe_out[0])

    def test_agent_api_endpoints(self) -> None:
        client_kwargs = {}
        if "client" in inspect.signature(TestClient).parameters:
            client_kwargs["client"] = ("127.0.0.1", 43210)
        client = TestClient(create_app(), **client_kwargs)

        st_res = client.get("/v1/agents/state")
        self.assertEqual(st_res.status_code, 200)
        payload = st_res.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("architect", payload["personas"])
        self.assertIn("debugger", payload["personas"])

        with patch(
            "triune.agents.providers.ProviderManager.complete_chat",
            return_value={"content": '{"thought": "done", "action": "final", "answer": "All systems ready."}'},
        ):
            run_res = client.post(
                "/v1/agents/run",
                json={
                    "goal": "Check system state",
                    "model": "ollama:gemma3:4b",
                    "persona": "researcher",
                    "reasoning_depth": "cortex",
                },
            )
            self.assertEqual(run_res.status_code, 200)
            rjson = run_res.json()
            self.assertEqual(rjson["status"], "success")
            self.assertEqual(rjson["persona"], "researcher")
            self.assertEqual(rjson["answer"], "All systems ready.")


if __name__ == "__main__":
    unittest.main()
