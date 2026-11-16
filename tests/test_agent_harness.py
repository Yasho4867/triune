"""Tests for the Self-Learning Triune Agent Harness (personas, skills, memory, tool mastery, subagents)."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from triune.agents import (
    Agent,
    AgentHarness,
    AgentMemoryStore,
    MultiAgentOrchestrator,
    SkillLibrary,
    StateController,
)
from triune.agents.tools import _SafeRedirectHandler, validate_http_url
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

            # Exercise explicit memory CRUD endpoints
            up_mem = client.post(
                "/v1/agents/memory",
                json={"key": "audit_test_key", "value": "prefer explicit checks", "category": "constraint"},
            )
            self.assertEqual(up_mem.status_code, 200)
            del_mem = client.delete("/v1/agents/memory/audit_test_key")
            self.assertEqual(del_mem.status_code, 200)
            self.assertTrue(del_mem.json()["deleted"])

    def test_ssrf_guard_blocks_private_metadata_and_unsafe_redirects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = AgentHarness(workspace=tmpdir, store_dir=Path(tmpdir) / "store", allow_local_http=False)
            for bad_url in (
                "http://169.254.169.254/latest/meta-data/",
                "http://10.0.0.1/internal",
                "http://192.168.1.50/router",
                "http://127.0.0.1:22/",
                "http://127.0.0.1:11434/api/tags",
                "file:///etc/passwd",
                "http://user@8.8.8.8/",
            ):
                res = harness.tools.execute("http_fetch", {"url": bad_url}, memory=harness.memory)
                self.assertFalse(res["ok"], f"Expected SSRF guard to block {bad_url}")

            # Local allowlist explicitly permits Ollama/Studio ports while still blocking SSH/metadata
            self.assertEqual(
                validate_http_url("http://127.0.0.1:11434/api/tags", allow_local_services=True),
                "http://127.0.0.1:11434/api/tags",
            )
            with self.assertRaises(PermissionError):
                validate_http_url("http://127.0.0.1:22/", allow_local_services=True)
            with self.assertRaises(PermissionError):
                validate_http_url("http://169.254.169.254/", allow_local_services=True)

            # Redirect hops pointing to metadata IPs must fail validation
            redir = _SafeRedirectHandler(allow_local_services=True)
            req = MagicMock()
            req.full_url = "https://example.com/start"
            with self.assertRaises(PermissionError):
                redir.redirect_request(req, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data/")

    def test_memory_determinism_schema_resilience_helper_cleanup_and_parallel_orch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "store"
            mem_a = AgentMemoryStore(store_dir)
            prefs_a = mem_a.infer_user_needs_from_prompt("I prefer clean deterministic hashing across restarts.")
            key_a = [p.key for p in prefs_a if p.key.startswith("user_rule_")][0]

            mem_b = AgentMemoryStore(Path(tmpdir) / "store_b")
            prefs_b = mem_b.infer_user_needs_from_prompt("I prefer clean deterministic hashing across restarts.")
            key_b = [p.key for p in prefs_b if p.key.startswith("user_rule_")][0]
            self.assertEqual(key_a, key_b)

            # Explicit delete & clear controls + corrupt JSON resilience
            self.assertTrue(mem_a.delete_preference(key_a))
            self.assertNotIn(key_a, mem_a.preferences)
            (store_dir / "memory.json").write_text("{corrupt-json", encoding="utf-8")
            mem_recovered = AgentMemoryStore(store_dir)
            self.assertEqual(len(mem_recovered.preferences), 0)

            # Stale helper.py removal when updated with helper_code=None
            lib = SkillLibrary(store_dir / "skills")
            sk = lib.create_skill(
                "Quant Audit",
                "Optimize quantized tensor memory budgets.",
                "Run helper calculation.",
                helper_code="x = 42\n",
            )
            helper_file = store_dir / "skills" / sk.slug / "helper.py"
            self.assertTrue(helper_file.is_file())
            lib.create_skill(
                "Quant Audit", "Optimize quantized tensor memory budgets.", "Manual only.", helper_code=None
            )
            self.assertFalse(helper_file.is_file())

            # Morphological TF-IDF skill retrieval + verified-only reinforcement
            matched = lib.match_skills("optimizing quantization tensors")
            self.assertTrue(any(s.slug == sk.slug for s in matched))

            # Unverified/step-budget exhaustion must NOT credit skill successes or crystallize Auto skill
            harness = AgentHarness(
                workspace=tmpdir,
                store_dir=store_dir,
                completion_fn=lambda *_, **__: {
                    "content": '{"thought": "looping", "action": "list_dir", "arguments": {"path": "."}}'
                },
            )
            budget_res = harness.run_goal("Quant Audit check weights optimize", max_steps=2)
            self.assertFalse(budget_res["verified"])
            self.assertEqual(harness.skills.skills[sk.slug].uses, 1)
            self.assertEqual(harness.skills.skills[sk.slug].successes, 0)
            self.assertEqual(budget_res["created_skills"], [])

            # Parallel multi-agent dispatch
            orch = MultiAgentOrchestrator(harness=harness)
            orch.add_agent(Agent(name="Alpha", role="Researcher", system_prompt="Check A"))
            orch.add_agent(Agent(name="Beta", role="Sentinel", system_prompt="Check B"))
            harness.completion_fn = lambda msgs, **_: {"content": f"Done: {msgs[-1]['content'][:40]}", "tool_calls": []}
            par_res = orch.execute_parallel("Audit architecture concurrently", max_workers=2)
            self.assertEqual(len(par_res), 2)
            self.assertEqual([r["agent"] for r in par_res], ["Alpha", "Beta"])
            self.assertTrue(all(r["ok"] for r in par_res))

            seq_res = orch.orchestrate_team("Sequence check", ["architect", "sentinel"], mode="pipeline", max_steps=1)
            self.assertEqual(seq_res["mode"], "pipeline")
            self.assertEqual([r["agent"] for r in seq_res["results"]], ["architect", "sentinel"])

    def test_orchestrate_api_cli_and_estimate_hf_endpoints(self):
        import io
        import json
        import sys
        from contextlib import redirect_stdout

        from triune.api.server import create_app
        from triune.cli import main as cli_main

        client = TestClient(create_app())

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. /v1/resource/estimate_hf (offline synthetic config)
            cfg_dir = Path(tmpdir) / "tiny_hf_cfg"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen2ForCausalLM"],
                        "model_type": "qwen2",
                        "hidden_size": 128,
                        "intermediate_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "vocab_size": 512,
                    }
                ),
                encoding="utf-8",
            )
            est_res = client.post(
                "/v1/resource/estimate_hf",
                json={"model": str(cfg_dir), "available_vram_gb": 8.0, "local_files_only": True},
            )
            self.assertEqual(est_res.status_code, 200)
            est_data = est_res.json()
            self.assertEqual(est_data["status"], "success")
            self.assertEqual(est_data["layers"], 2)
            self.assertEqual(est_data["recommended_mode"], "resident_bf16")

            # 2. /v1/agents/orchestrate + CLI --team --parallel
            with patch(
                "triune.agents.providers.ProviderManager.complete_chat",
                return_value={"content": '{"answer": "verified specialist output"}', "tool_calls": []},
            ):
                orch_res = client.post(
                    "/v1/agents/orchestrate",
                    json={
                        "goal": "Check system integration",
                        "personas": ["architect", "sentinel"],
                        "mode": "parallel",
                        "max_workers": 2,
                        "max_steps": 2,
                    },
                )
                self.assertEqual(orch_res.status_code, 200)
                orch_data = orch_res.json()
                self.assertEqual(orch_data["status"], "success")
                self.assertEqual(orch_data["mode"], "parallel")
                self.assertEqual([r["agent"] for r in orch_data["results"]], ["architect", "sentinel"])
                self.assertTrue(all(r["ok"] for r in orch_data["results"]))

                buf = io.StringIO()
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "triune",
                            "agent",
                            "--store-dir",
                            tmpdir,
                            "--team",
                            "architect,engineer",
                            "--parallel",
                            "--max-steps",
                            "2",
                            "-g",
                            "Inspect runtime pipeline",
                        ],
                    ),
                    redirect_stdout(buf),
                ):
                    cli_main()
                cli_out = buf.getvalue()
                self.assertIn("architect", cli_out)
                self.assertIn("engineer", cli_out)


if __name__ == "__main__":
    unittest.main()
