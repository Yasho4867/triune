"""Comprehensive tests for Project Workspace IDE Backend & Upgraded Agentic Harness / Loops."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from lucid_machines.agents import AgentHarness
from lucid_machines.agents.loops import (
    ApprovalGateManager,
    StuckLoopDetector,
    TrainingSupervisor,
    run_agent_loop,
)
from lucid_machines.api.server import create_app
from lucid_machines.api.workspace_routes import workspace_manager


def _parse_sse_events(raw_text: str) -> list[dict]:
    events: list[dict] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload:
                events.append(json.loads(payload))
    return events


class TestWorkspaceAndAgentLoops(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls._session_file = cls.repo_root / "checkpoints" / "session_state.json"
        cls._session_backup = cls._session_file.read_text(encoding="utf-8") if cls._session_file.is_file() else None
        cls.app = create_app()
        cls.client = TestClient(cls.app, client=("127.0.0.1", 43210))

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_session_backup", None) is not None:
            cls._session_file.write_text(cls._session_backup, encoding="utf-8")

    def setUp(self) -> None:
        self._orig_root = workspace_manager.root
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name).resolve()
        (self.ws / "src").mkdir()
        (self.ws / "src" / "main.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        (self.ws / "README.md").write_text("# Sample Project\nCentroidSteer\n", encoding="utf-8")
        # Ignored / protected directories
        for d in (".git", "node_modules", "__pycache__", "studio_env", "checkpoints", "data", "exports", "papers"):
            (self.ws / d).mkdir()
            (self.ws / d / "secret.txt").write_text("protected_content\n", encoding="utf-8")
        res = self.client.post("/v1/workspace/open", json={"path": str(self.ws)})
        self.assertEqual(res.status_code, 200, res.text)

    def tearDown(self) -> None:
        workspace_manager.open_workspace(str(self._orig_root))
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # 1. Security Hardening: AST shell=False, Traversal, Symlink, Protected
    # ------------------------------------------------------------------
    def test_01_no_shell_true_anywhere_in_workspace_or_loops(self) -> None:
        for rel in (
            "lucid_machines/api/workspace_routes.py",
            "lucid_machines/agents/loops.py",
            "lucid_machines/agents/tools.py",
        ):
            fpath = self.repo_root / rel
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg == "shell":
                            self.assertFalse(
                                isinstance(kw.value, ast.Constant) and kw.value.value is True,
                                f"shell=True forbidden in {rel}:{node.lineno}",
                            )

    def test_02_open_state_pick_and_forbidden_roots(self) -> None:
        st = self.client.get("/v1/workspace/state")
        self.assertEqual(st.status_code, 200)
        st_json = st.json()
        self.assertEqual(Path(st_json["root"]).resolve(), self.ws)
        self.assertIn(str(self.ws), st_json["recent"])

        # Missing folder -> 400
        missing = self.client.post("/v1/workspace/open", json={"path": str(self.ws / "does_not_exist")})
        self.assertEqual(missing.status_code, 400)

        # Forbidden OS root -> 403
        os_root = "C:\\" if os.name == "nt" else "/"
        forb = self.client.post("/v1/workspace/open", json={"path": os_root})
        self.assertEqual(forb.status_code, 403)

        # Pick endpoint must never hang/block in test/headless mode
        pick = self.client.post("/v1/workspace/pick")
        self.assertEqual(pick.status_code, 200)
        p_json = pick.json()
        self.assertFalse(p_json["ok"])
        self.assertTrue(p_json["headless"])
        self.assertEqual(Path(p_json["root"]).resolve(), self.ws)

        # Middleware auth / Origin still enforced on new endpoints
        csrf = self.client.post(
            "/v1/workspace/open",
            json={"path": str(self.ws)},
            headers={"Origin": "https://evil.example.com"},
        )
        self.assertEqual(csrf.status_code, 403)
        with patch.dict(os.environ, {"LUCID_MACHINES_API_TOKEN": "t" * 40}):
            unauth = self.client.get("/v1/workspace/state", headers={"X-Forwarded-For": "203.0.113.9"})
            self.assertEqual(unauth.status_code, 401)

    def test_03_path_traversal_symlink_escape_protected_dirs_and_binary_caps(self) -> None:
        outside_dir = tempfile.TemporaryDirectory()
        self.addCleanup(outside_dir.cleanup)
        outside_file = Path(outside_dir.name) / "outside_secret.txt"
        outside_file.write_text("TOP_SECRET", encoding="utf-8")

        # Traversal 403 across endpoints
        for trav in ("../outside_secret.txt", "../../etc/passwd", r"\\evil-server\share\x.txt"):
            self.assertEqual(self.client.get("/v1/workspace/file", params={"path": trav}).status_code, 403)
            self.assertEqual(
                self.client.post("/v1/workspace/file", json={"path": trav, "content": "x"}).status_code, 403
            )
            self.assertEqual(
                self.client.post("/v1/workspace/file/create", json={"path": trav, "content": "x"}).status_code, 403
            )
            self.assertEqual(self.client.delete("/v1/workspace/file", params={"path": trav}).status_code, 403)

        # Symlink escape 403 (or verify resolve_confined logic if OS disables unprivileged symlinks)
        link_path = self.ws / "escape_link.txt"
        try:
            os.symlink(outside_file, link_path)
            symlink_created = True
        except OSError:
            symlink_created = False

        if symlink_created:
            self.assertEqual(self.client.get("/v1/workspace/file", params={"path": "escape_link.txt"}).status_code, 403)
            self.assertEqual(
                self.client.post("/v1/workspace/file", json={"path": "escape_link.txt", "content": "x"}).status_code,
                403,
            )
            self.assertEqual(
                self.client.delete("/v1/workspace/file", params={"path": "escape_link.txt"}).status_code, 403
            )

        # Protected directories cannot be modified/created/deleted
        for prot in ("checkpoints", "data", "exports", "papers", ".git"):
            self.assertEqual(
                self.client.post(
                    "/v1/workspace/file", json={"path": f"{prot}/secret.txt", "content": "overwrite"}
                ).status_code,
                403,
            )
            self.assertEqual(
                self.client.post(
                    "/v1/workspace/file/create", json={"path": f"{prot}/new.txt", "content": "x"}
                ).status_code,
                403,
            )
            self.assertEqual(
                self.client.delete("/v1/workspace/file", params={"path": f"{prot}/secret.txt"}).status_code, 403
            )

        # Cannot delete workspace root itself
        self.assertEqual(self.client.delete("/v1/workspace/file", params={"path": "."}).status_code, 403)

        # Binary rejection (400) and >2MB rejection (413)
        (self.ws / "model.bin").write_bytes(b"\x00\x01\x02\xff\xfe")
        self.assertEqual(self.client.get("/v1/workspace/file", params={"path": "model.bin"}).status_code, 400)

        (self.ws / "huge.txt").write_text("A" * (2 * 1024 * 1024 + 128), encoding="utf-8")
        self.assertEqual(self.client.get("/v1/workspace/file", params={"path": "huge.txt"}).status_code, 413)

    # ------------------------------------------------------------------
    # 2. Workspace Tree, CRUD, Diff, Search, Git & Confined Terminal
    # ------------------------------------------------------------------
    def test_04_tree_file_crud_diff_search_git_and_terminal(self) -> None:
        # Tree skips ignored dirs
        tree_res = self.client.get("/v1/workspace/tree", params={"depth": 2})
        self.assertEqual(tree_res.status_code, 200)
        names = {e["name"] for e in tree_res.json()["entries"]}
        self.assertIn("src", names)
        self.assertIn("README.md", names)
        for skipped in (".git", "node_modules", "__pycache__", "studio_env", "checkpoints", "data"):
            self.assertNotIn(skipped, names)

        # Read file
        read_res = self.client.get("/v1/workspace/file", params={"path": "src/main.py"})
        self.assertEqual(read_res.status_code, 200)
        r_data = read_res.json()
        self.assertEqual(r_data["language"], "python")
        self.assertEqual(r_data["rel_path"], "src/main.py")
        self.assertIn("return 'world'", r_data["content"])

        # Save + unified diff
        save_res = self.client.post(
            "/v1/workspace/file",
            json={"path": "src/main.py", "content": "def hello():\n    return 'lucid'\n"},
        )
        self.assertEqual(save_res.status_code, 200)
        s_data = save_res.json()
        self.assertTrue(s_data["ok"])
        self.assertIn("-    return 'world'", s_data["diff"])
        self.assertIn("+    return 'lucid'", s_data["diff"])

        # Create file & dir, then delete via query & JSON body
        c_dir = self.client.post("/v1/workspace/file/create", json={"path": "src/pkg", "is_dir": True})
        self.assertEqual(c_dir.status_code, 200)
        c_file = self.client.post(
            "/v1/workspace/file/create",
            json={"path": "src/pkg/mod.py", "content": "VALUE = 123\n", "is_dir": False},
        )
        self.assertEqual(c_file.status_code, 200)
        self.assertTrue((self.ws / "src" / "pkg" / "mod.py").is_file())

        # Search text & regex
        srch = self.client.post("/v1/workspace/search", json={"query": "VALUE\\s*=\\s*123", "regex": True})
        self.assertEqual(srch.status_code, 200)
        sr_data = srch.json()
        self.assertTrue(any(m["rel_path"] == "src/pkg/mod.py" and m["line"] == 1 for m in sr_data["matches"]))
        self.assertIn("src/pkg/mod.py", sr_data["files"])

        # Delete via JSON body
        del_file = self.client.request("DELETE", "/v1/workspace/file", json={"path": "src/pkg/mod.py"})
        self.assertEqual(del_file.status_code, 200)
        self.assertFalse((self.ws / "src" / "pkg" / "mod.py").exists())
        del_dir = self.client.delete("/v1/workspace/file", params={"path": "src/pkg"})
        self.assertEqual(del_dir.status_code, 200)
        self.assertFalse((self.ws / "src" / "pkg").exists())

        # Git endpoint on real git repo
        if shutil.which("git"):
            # Use a fresh subdir inside ws? No, ws has an empty dummy .git dir; remove dummy .git and git init
            shutil.rmtree(self.ws / ".git")
            subprocess.run(["git", "init"], shell=False, cwd=str(self.ws), check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@lucid.ai"], shell=False, cwd=str(self.ws), check=True)
            subprocess.run(["git", "config", "user.name", "Lucid Test"], shell=False, cwd=str(self.ws), check=True)
            subprocess.run(["git", "add", "README.md"], shell=False, cwd=str(self.ws), check=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], shell=False, cwd=str(self.ws), check=True, capture_output=True
            )
            (self.ws / "README.md").write_text("# Sample Project\nModified\n", encoding="utf-8")

            git_res = self.client.get("/v1/workspace/git")
            self.assertEqual(git_res.status_code, 200)
            g_data = git_res.json()
            self.assertTrue(g_data["is_git"])
            self.assertTrue(bool(g_data["branch"]))
            self.assertGreaterEqual(g_data["dirty_count"], 1)
            self.assertIn("+Modified", g_data["diff"])

        # Terminal run allowlist & metacharacter rejection
        ok_term = self.client.post("/v1/workspace/terminal/run", json={"command": 'python -c "print(6*7)"'})
        self.assertEqual(ok_term.status_code, 200)
        t_data = ok_term.json()
        self.assertTrue(t_data["ok"])
        self.assertEqual(t_data["exit_code"], 0)
        self.assertIn("42", t_data["stdout"])

        for bad_cmd in (
            "rm -rf /",
            "curl https://example.com",
            "python --version && whoami",
            "python --version | cat",
            "python --version; ls",
            "cat ../../etc/passwd",
            "ls /etc",
        ):
            bad_res = self.client.post("/v1/workspace/terminal/run", json={"command": bad_cmd})
            self.assertEqual(bad_res.status_code, 403, f"Expected 403 for '{bad_cmd}', got {bad_res.text}")

    # ------------------------------------------------------------------
    # 3. Harness Rebinding, SSE Loops, Stuck Detector, Compactor, HITL, Cancel
    # ------------------------------------------------------------------
    def test_05_harness_rebind_workspace_and_loop_modes(self) -> None:
        with tempfile.TemporaryDirectory() as other_dir:
            other = Path(other_dir).resolve()
            (other / "hello.txt").write_text("from_other_workspace\n", encoding="utf-8")

            harness = AgentHarness(
                workspace=self.ws, store_dir=self.ws / ".store", completion_fn=lambda *a, **k: {"content": "ok"}
            )
            self.assertIn("README.md", harness.tools.execute("list_dir", {"path": "."})["result"])
            harness.rebind_workspace(other)
            self.assertEqual(harness.workspace, other)
            listed = harness.tools.execute("list_dir", {"path": "."})["result"]
            self.assertIn("hello.txt", listed)
            self.assertNotIn("README.md", listed)

        # Test ReAct, PEV, Multi-Agent SSE endpoints via patched ProviderManager.complete_chat
        turns_react = iter(
            [
                {
                    "content": json.dumps(
                        {"thought": "Inspect README", "action": "read_file", "arguments": {"path": "README.md"}}
                    )
                },
                {"content": json.dumps({"thought": "Done", "action": "final", "answer": "Read README cleanly."})},
            ]
        )
        with patch(
            "lucid_machines.agents.providers.ProviderManager.complete_chat",
            side_effect=lambda *a, **k: next(turns_react),
        ):
            resp = self.client.post(
                "/v1/agents/loop/stream",
                json={
                    "goal": "Read README",
                    "mode": "react",
                    "autonomy": "autonomous_sandbox",
                    "model": "ollama:gemma3:4b",
                    "max_steps": 4,
                },
            )
            self.assertEqual(resp.status_code, 200)
            evs = _parse_sse_events(resp.text)
            ev_types = [e["type"] for e in evs]
            self.assertIn("start", ev_types)
            self.assertIn("thought", ev_types)
            self.assertIn("tool_call", ev_types)
            self.assertIn("tool_result", ev_types)
            self.assertIn("done", ev_types)
            self.assertEqual(evs[-1]["answer"], "Read README cleanly.")

        # PEV mode
        turns_pev = iter(
            [
                {"content": json.dumps({"plan": ["List workspace files", "Summarize findings"]})},
                {"content": json.dumps({"thought": "List files", "action": "list_dir", "arguments": {"path": "."}})},
                {"content": json.dumps({"thought": "Verify", "action": "final", "answer": "PEV complete."})},
            ]
        )
        with patch(
            "lucid_machines.agents.providers.ProviderManager.complete_chat", side_effect=lambda *a, **k: next(turns_pev)
        ):
            resp = self.client.post(
                "/v1/agents/loop/stream",
                json={
                    "goal": "Inspect workspace",
                    "mode": "pev",
                    "autonomy": "autonomous_sandbox",
                    "model": "ollama:gemma3:4b",
                    "max_steps": 4,
                },
            )
            evs = _parse_sse_events(resp.text)
            ev_types = [e["type"] for e in evs]
            self.assertIn("plan", ev_types)
            self.assertIn("reflection", ev_types)
            self.assertEqual(evs[-1]["type"], "done")
            self.assertEqual(evs[-1]["answer"], "PEV complete.")

        # Multi-agent pipeline mode (Explorer -> Implementer -> Verifier)
        turns_ma = iter(
            [
                {
                    "content": json.dumps(
                        {"thought": "Explore files", "action": "list_dir", "arguments": {"path": "src"}}
                    )
                },
                {"content": json.dumps({"thought": "Explorer done", "action": "final", "answer": "Found src/main.py"})},
                {
                    "content": json.dumps(
                        {
                            "thought": "Write artifact",
                            "action": "write_file",
                            "arguments": {"path": "src/out.txt", "content": "built"},
                        }
                    )
                },
                {
                    "content": json.dumps(
                        {"thought": "Implementer done", "action": "final", "answer": "Created src/out.txt"}
                    )
                },
                {
                    "content": json.dumps(
                        {"thought": "Verify artifact", "action": "read_file", "arguments": {"path": "src/out.txt"}}
                    )
                },
                {
                    "content": json.dumps(
                        {"thought": "Verified", "action": "final", "answer": "Pipeline verified src/out.txt"}
                    )
                },
            ]
        )
        with patch(
            "lucid_machines.agents.providers.ProviderManager.complete_chat", side_effect=lambda *a, **k: next(turns_ma)
        ):
            resp = self.client.post(
                "/v1/agents/loop/stream",
                json={
                    "goal": "Create and verify src/out.txt",
                    "mode": "multi_agent",
                    "autonomy": "autonomous_sandbox",
                    "model": "ollama:gemma3:4b",
                    "max_steps": 4,
                },
            )
            evs = _parse_sse_events(resp.text)
            agents_seen = {e.get("agent") for e in evs if e.get("agent")}
            self.assertTrue({"explorer", "implementer", "verifier"}.issubset(agents_seen))
            self.assertEqual((self.ws / "src" / "out.txt").read_text(encoding="utf-8"), "built")
            self.assertEqual(evs[-1]["type"], "done")

    def test_06_stuck_loop_detector_and_context_compactor(self) -> None:
        # Stuck-loop detector catches repeated identical tool calls
        events: list[dict] = []
        harness = AgentHarness(
            workspace=self.ws,
            store_dir=self.ws / ".store",
            completion_fn=lambda *a, **k: {
                "content": json.dumps({"thought": "repeat", "action": "list_dir", "arguments": {"path": "."}})
            },
        )
        run_agent_loop(
            harness=harness,
            goal="Loop test",
            mode="react",
            autonomy="autonomous_sandbox",
            max_steps=8,
            emit=events.append,
            stuck_repeat_threshold=2,
        )
        reflections = [e for e in events if e["type"] == "reflection"]
        self.assertGreaterEqual(len(reflections), 1)
        self.assertIn("stuck_loop", reflections[0]["reason"])
        self.assertEqual(events[-1]["type"], "done")

        # Unit check on StuckLoopDetector & ContextCompactor
        det = StuckLoopDetector(repeat_threshold=3, fail_threshold=2)
        self.assertIsNone(det.record("read_file", {"path": "a.py"}, ok=False, error="FileNotFound"))
        trig = det.record("read_file", {"path": "a.py"}, ok=False, error="FileNotFound")
        self.assertIsNotNone(trig)
        self.assertEqual(trig["reason"], "stuck_loop_repeated_failure")

        # Context compactor inside loop
        (self.ws / "big_text.txt").write_text("X" * 1500 + "\n", encoding="utf-8")
        comp_events: list[dict] = []
        comp_turns = iter(
            [
                {
                    "content": json.dumps(
                        {"thought": "r1", "action": "read_file", "arguments": {"path": "big_text.txt"}}
                    )
                },
                {"content": json.dumps({"thought": "r2", "action": "read_file", "arguments": {"path": "README.md"}})},
                {"content": json.dumps({"thought": "r3", "action": "list_dir", "arguments": {"path": "."}})},
                {"content": json.dumps({"thought": "r4", "action": "list_dir", "arguments": {"path": "src"}})},
                {"content": json.dumps({"thought": "done", "action": "final", "answer": "Compacted cleanly"})},
            ]
        )
        comp_harness = AgentHarness(
            workspace=self.ws,
            store_dir=self.ws / ".store",
            completion_fn=lambda *a, **k: next(comp_turns),
        )
        run_agent_loop(
            harness=comp_harness,
            goal="Read files and compact context",
            mode="react",
            autonomy="autonomous_sandbox",
            max_steps=6,
            emit=comp_events.append,
            compact_char_budget=800,
            compact_keep_recent=2,
        )
        compactions = [e for e in comp_events if e["type"] == "compaction"]
        self.assertGreaterEqual(len(compactions), 1)
        self.assertGreater(compactions[0]["dropped_messages"], 0)
        self.assertLess(compactions[0]["chars_after"], compactions[0]["chars_before"])

    def test_07_hitl_approval_approve_reject_and_loop_cancel(self) -> None:
        # 1. Supervised Approval: APPROVE
        turns_approve = iter(
            [
                {
                    "content": json.dumps(
                        {
                            "thought": "Update README",
                            "action": "write_file",
                            "arguments": {"path": "README.md", "content": "# Approved Edit\n"},
                        }
                    )
                },
                {"content": json.dumps({"thought": "done", "action": "final", "answer": "Saved approved edit."})},
            ]
        )
        gate = ApprovalGateManager()
        events_app: list[dict] = []
        harness_app = AgentHarness(
            workspace=self.ws,
            store_dir=self.ws / ".store",
            completion_fn=lambda *a, **k: next(turns_approve),
        )

        def _approver_worker() -> None:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                req_ev = next((e for e in list(events_app) if e["type"] == "approval_required"), None)
                if req_ev is not None:
                    self.assertIn("+ # Approved Edit".replace("+ #", "+#"), req_ev["diff"])
                    gate.resolve(req_ev["action_id"], approved=True)
                    return
                time.sleep(0.02)

        t = threading.Thread(target=_approver_worker, daemon=True)
        t.start()
        run_agent_loop(
            harness=harness_app,
            goal="Update README.md",
            mode="react",
            autonomy="supervised",
            max_steps=4,
            emit=events_app.append,
            approval_manager=gate,
        )
        t.join(timeout=5)
        self.assertEqual((self.ws / "README.md").read_text(encoding="utf-8"), "# Approved Edit\n")
        tr_app = next(e for e in events_app if e["type"] == "tool_result")
        self.assertTrue(tr_app["ok"])
        self.assertTrue(tr_app["approved"])

        # 2. Supervised Approval: REJECT via HTTP /v1/agents/approve/{action_id}
        turns_reject = iter(
            [
                {
                    "content": json.dumps(
                        {
                            "thought": "Attempt overwrite",
                            "action": "write_file",
                            "arguments": {"path": "README.md", "content": "# REJECTED\n"},
                        }
                    )
                },
                {
                    "content": json.dumps(
                        {"thought": "ack rejection", "action": "final", "answer": "Respecting rejection."}
                    )
                },
            ]
        )
        seen_events: list[dict] = []

        def _reject_via_http() -> None:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                # Poll workspace_manager.loop_manager.approvals for pending action
                with workspace_manager.loop_manager.approvals._lock:
                    pending_ids = [
                        aid
                        for aid, item in workspace_manager.loop_manager.approvals._pending.items()
                        if not item.event.is_set()
                    ]
                if pending_ids:
                    r = self.client.post(
                        f"/v1/agents/approve/{pending_ids[0]}",
                        json={"approved": False, "reason": "Keep existing README"},
                    )
                    self.assertEqual(r.status_code, 200)
                    return
                time.sleep(0.02)

        with patch(
            "lucid_machines.agents.providers.ProviderManager.complete_chat",
            side_effect=lambda *a, **k: next(turns_reject),
        ):
            t_rej = threading.Thread(target=_reject_via_http, daemon=True)
            t_rej.start()
            resp = self.client.post(
                "/v1/agents/loop/stream",
                json={
                    "goal": "Try editing README",
                    "mode": "react",
                    "autonomy": "supervised",
                    "model": "ollama:gemma3:4b",
                    "max_steps": 4,
                },
            )
            t_rej.join(timeout=5)
            seen_events = _parse_sse_events(resp.text)

        self.assertEqual((self.ws / "README.md").read_text(encoding="utf-8"), "# Approved Edit\n")
        tr_rej = next(e for e in seen_events if e["type"] == "tool_result")
        self.assertFalse(tr_rej["ok"])
        self.assertFalse(tr_rej["approved"])

        # 3. Cancel active loop via HTTP /v1/agents/loop/cancel/{run_id}
        cancel_turn = {
            "content": json.dumps(
                {
                    "thought": "Propose edit then wait",
                    "action": "write_file",
                    "arguments": {"path": "README.md", "content": "# SHOULD_NOT_WRITE\n"},
                }
            )
        }

        def _cancel_via_http() -> None:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with workspace_manager.loop_manager.approvals._lock:
                    pending = [
                        item
                        for item in workspace_manager.loop_manager.approvals._pending.values()
                        if not item.event.is_set()
                    ]
                if pending:
                    rid = pending[0].run_id
                    c_res = self.client.post(f"/v1/agents/loop/cancel/{rid}")
                    self.assertEqual(c_res.status_code, 200)
                    self.assertTrue(c_res.json()["cancelled"])
                    return
                time.sleep(0.02)

        with patch("lucid_machines.agents.providers.ProviderManager.complete_chat", return_value=cancel_turn):
            t_can = threading.Thread(target=_cancel_via_http, daemon=True)
            t_can.start()
            resp_c = self.client.post(
                "/v1/agents/loop/stream",
                json={
                    "goal": "Edit README then cancel",
                    "mode": "react",
                    "autonomy": "supervised",
                    "model": "ollama:gemma3:4b",
                    "max_steps": 4,
                },
            )
            t_can.join(timeout=5)
            c_evs = _parse_sse_events(resp_c.text)

        self.assertTrue(any(e["type"] == "cancelled" for e in c_evs))
        self.assertEqual((self.ws / "README.md").read_text(encoding="utf-8"), "# Approved Edit\n")

    # ------------------------------------------------------------------
    # 4. Autonomous Training Supervisor Endpoints & Anomaly Detection
    # ------------------------------------------------------------------
    def test_08_training_supervisor_endpoints_and_anomaly_detection(self) -> None:
        # API lifecycle: start -> status -> stop
        st_start = self.client.post("/v1/agents/supervisor/start", json={"poll_interval_s": 0.2})
        self.assertEqual(st_start.status_code, 200)
        self.assertTrue(st_start.json()["running"])

        time.sleep(0.25)
        st_get = self.client.get("/v1/agents/supervisor/status")
        self.assertEqual(st_get.status_code, 200)
        self.assertGreaterEqual(st_get.json()["inspections"], 1)

        st_stop = self.client.post("/v1/agents/supervisor/stop")
        self.assertEqual(st_stop.status_code, 200)
        self.assertFalse(st_stop.json()["running"])

        # Deterministic anomaly detection + safety checkpoint callback
        ckpts_called: list[str] = []
        samples = iter(
            [
                {"step": 10, "loss": 2.1, "grad_norm": 1.2, "depth_mode": "dynamic", "mind": {}},
                {"step": 11, "loss": 2.0, "grad_norm": 1.1, "depth_mode": "dynamic", "mind": {}},
                {"step": 12, "loss": 1.9, "grad_norm": 1.0, "depth_mode": "dynamic", "mind": {}},
                {
                    "step": 13,
                    "loss": 8.5,  # spike > 2x EMA
                    "grad_norm": 120.0,  # grad explosion > 50
                    "depth_mode": "dynamic",
                    "mind": {
                        "moe_routing": {"utilization_pct": [94.0, 4.0, 2.0, 0.0]},
                        "centroid_steer": {"active": True, "alignment_energy": 0.0, "centroid_drift": 0.0},
                        "triune_tiers": {
                            "depth_mode": "dynamic",
                            "exit_usage": {"reflex": 97.0, "limbic": 2.5, "cortex": 0.5},
                        },
                    },
                },
                {"step": 14, "loss": float("nan"), "grad_norm": 1.0, "depth_mode": "dynamic", "mind": {}},
            ]
        )

        sup = TrainingSupervisor(
            metrics_provider=lambda: next(samples),
            checkpoint_callback=lambda reason, snap: ckpts_called.append(reason) or {"ok": True, "path": "safety.pt"},
        )
        for _ in range(3):
            s = sup.inspect_once()
            self.assertEqual(s["anomalies"], [])

        spike_snap = sup.inspect_once()
        kinds = {a["kind"] for a in spike_snap["anomalies"]}
        self.assertIn("loss_spike", kinds)
        self.assertIn("grad_norm_explosion", kinds)
        self.assertIn("moe_expert_collapse", kinds)
        self.assertIn("centroid_steer_anomaly", kinds)
        self.assertIn("tier_routing_imbalance", kinds)
        self.assertGreaterEqual(len(ckpts_called), 1)
        self.assertTrue(len(spike_snap["recommendations"]) >= 3)

        nan_snap = sup.inspect_once()
        self.assertTrue(any(a["kind"] == "nan_or_inf" for a in nan_snap["anomalies"]))

        # Streaming training_supervisor loop mode
        sup_stream = self.client.post(
            "/v1/agents/loop/stream",
            json={
                "goal": "Audit current training health",
                "mode": "training_supervisor",
                "autonomy": "supervised",
                "max_steps": 2,
            },
        )
        self.assertEqual(sup_stream.status_code, 200)
        sup_evs = _parse_sse_events(sup_stream.text)
        self.assertEqual(sup_evs[0]["type"], "start")
        self.assertEqual(sup_evs[-1]["type"], "done")
        self.assertTrue(any(e["type"] == "reflection" for e in sup_evs))


if __name__ == "__main__":
    unittest.main()
