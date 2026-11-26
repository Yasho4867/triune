"""Verifiable GRPO + Tool/Harness & Self-Improvement RL Trainer for Lucid Machines."""

from __future__ import annotations

import copy
import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lucid_machines.runtime.sandbox import PythonSandbox
from lucid_machines.trainer.evaluation import forward_causal_lm


@dataclass
class VerifiableTask:
    """Single verifiable task covering math, Python sandbox execution, or AgentHarness tools."""

    task_id: str
    category: str  # "math" | "python" | "tool_harness"
    prompt: str
    expected: str | None = None
    test_code: str | None = None
    expected_tool: str | None = None
    required_args: list[str] = field(default_factory=list)
    arg_substrings: dict[str, str] = field(default_factory=dict)


def default_verifiable_task_bank() -> list[VerifiableTask]:
    """Return built-in deterministic tasks for math, Python sandbox coding, and AgentHarness tools."""
    return [
        VerifiableTask(
            task_id="math_arith_1",
            category="math",
            prompt="<user>\nCompute (17 * 24) - 58 and put your final answer inside \\boxed{...}.\n</user>\n<assistant>\n",
            expected="350",
        ),
        VerifiableTask(
            task_id="math_alg_2",
            category="math",
            prompt="<user>\nIf 3x + 7 = 43, find x and put the final integer inside \\boxed{...}.\n</user>\n<assistant>\n",
            expected="12",
        ),
        VerifiableTask(
            task_id="math_pct_3",
            category="math",
            prompt="<user>\nWhat is 15% of 240? Put your final number inside \\boxed{...}.\n</user>\n<assistant>\n",
            expected="36",
        ),
        VerifiableTask(
            task_id="py_fib_1",
            category="python",
            prompt=(
                "<user>\nWrite a Python function `fib(n)` returning the n-th Fibonacci number "
                "(fib(0)=0, fib(1)=1) inside ```python ... ```.\n</user>\n<assistant>\n"
            ),
            test_code=(
                "if fib(0) != 0 or fib(1) != 1 or fib(10) != 55:\n"
                "    raise ValueError('fib failed')\n"
            ),
        ),
        VerifiableTask(
            task_id="py_is_prime_2",
            category="python",
            prompt=(
                "<user>\nWrite a Python function `is_prime(n)` returning True iff integer n >= 2 is prime, "
                "inside ```python ... ```.\n</user>\n<assistant>\n"
            ),
            test_code=(
                "if is_prime(1) or not is_prime(2) or not is_prime(29) or is_prime(51):\n"
                "    raise ValueError('is_prime failed')\n"
            ),
        ),
        VerifiableTask(
            task_id="tool_search_1",
            category="tool_harness",
            prompt=(
                "<user>\nUse the AgentHarness `search_code` tool to find `TriuneTransformer` under "
                "`lucid_machines`. Respond with `<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>`.\n"
                "</user>\n<assistant>\n"
            ),
            expected_tool="search_code",
            required_args=["pattern"],
            arg_substrings={"pattern": "TriuneTransformer"},
        ),
        VerifiableTask(
            task_id="tool_read_2",
            category="tool_harness",
            prompt=(
                "<user>\nUse the AgentHarness `read_file` tool on `scripts/train.py` lines 1 to 60. "
                "Respond with `<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>`.\n"
                "</user>\n<assistant>\n"
            ),
            expected_tool="read_file",
            required_args=["path"],
            arg_substrings={"path": "train.py"},
        ),
        VerifiableTask(
            task_id="tool_sandbox_3",
            category="tool_harness",
            prompt=(
                "<user>\nUse the AgentHarness `run_python_sandbox` tool to compute 2**16. "
                "Respond with `<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>`.\n"
                "</user>\n<assistant>\n"
            ),
            expected_tool="run_python_sandbox",
            required_args=["code"],
            arg_substrings={"code": "2**16"},
        ),
    ]


def load_verifiable_tasks(tasks_jsonl: str | Path | None = None) -> list[VerifiableTask]:
    """Load custom JSONL verifiable tasks or fall back to built-in multi-domain bank."""
    if tasks_jsonl is None:
        return default_verifiable_task_bank()
    path = Path(tasks_jsonl)
    tasks: list[VerifiableTask] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        tasks.append(
            VerifiableTask(
                task_id=str(obj.get("task_id", f"task_{idx}")),
                category=str(obj.get("category", "math")),
                prompt=str(obj["prompt"]),
                expected=obj.get("expected"),
                test_code=obj.get("test_code"),
                expected_tool=obj.get("expected_tool"),
                required_args=list(obj.get("required_args", [])),
                arg_substrings=dict(obj.get("arg_substrings", {})),
            )
        )
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks


def _extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _extract_math_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    final_m = re.search(r"FINAL:\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if final_m:
        return final_m.group(1).strip()
    return None


def _nums_equal(pred: str, expected: str) -> bool:
    p_clean = pred.replace(",", "").strip()
    e_clean = expected.replace(",", "").strip()
    if p_clean == e_clean:
        return True
    try:
        return math.isclose(float(p_clean), float(e_clean), rel_tol=1e-6, abs_tol=1e-9)
    except ValueError:
        return False


def verify_rollout(task: VerifiableTask, completion: str, *, sandbox_timeout: int = 5) -> tuple[float, dict[str, Any]]:
    """Deterministically score a model rollout in [0.0, 1.0] (0.2 format + 0.8 correctness)."""
    cat = task.category.lower()
    if cat == "math":
        ans = _extract_math_answer(completion)
        fmt_ok = ans is not None
        corr_ok = bool(ans is not None and task.expected is not None and _nums_equal(ans, task.expected))
        reward = (0.2 if fmt_ok else 0.0) + (0.8 if corr_ok else 0.0)
        return reward, {"format_ok": fmt_ok, "correct": corr_ok, "extracted": ans}

    if cat == "python":
        has_fence = "```" in completion
        code = _extract_code_block(completion)
        if not code or not task.test_code:
            return (0.2 if has_fence else 0.0), {"format_ok": has_fence, "correct": False, "error": "empty_code"}
        sb = PythonSandbox(timeout=sandbox_timeout, max_memory_mb=256)
        combined = f"{code}\n\n{task.test_code}"
        res = sb.execute_code(combined)
        corr_ok = bool(res.get("success", False))
        reward = (0.2 if has_fence else 0.0) + (0.8 if corr_ok else 0.0)
        return reward, {"format_ok": has_fence, "correct": corr_ok, "error": res.get("error")}

    if cat == "tool_harness":
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", completion, flags=re.DOTALL)
        payload = None
        if m:
            try:
                payload = json.loads(m.group(1))
            except json.JSONDecodeError:
                payload = None
        fmt_ok = isinstance(payload, dict) and isinstance(payload.get("name"), str) and isinstance(
            payload.get("arguments"), dict
        )
        corr_ok = False
        if fmt_ok and payload is not None:
            args = payload["arguments"]
            name_ok = task.expected_tool is None or payload["name"] == task.expected_tool
            keys_ok = all(k in args for k in task.required_args)
            vals_ok = all(sub in str(args.get(k, "")) for k, sub in task.arg_substrings.items())
            corr_ok = name_ok and keys_ok and vals_ok
        reward = (0.2 if fmt_ok else 0.0) + (0.8 if corr_ok else 0.0)
        return reward, {"format_ok": fmt_ok, "correct": corr_ok, "parsed_tool": payload}

    raise ValueError(f"Unknown verifiable task category: {task.category!r}")


class GRPOHarnessTrainer:
    """Group Relative Policy Optimization + online verified self-improvement trace replay."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        tokenizer: Any,
        tasks: list[VerifiableTask],
        device: torch.device,
        output_dir: str | Path,
        group_size: int = 4,
        max_new_tokens: int = 64,
        kl_coef: float = 0.04,
        clip_eps: float = 0.2,
        temperature: float = 0.8,
        rft_every: int = 4,
        replay_threshold: float = 0.95,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.tasks = list(tasks)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "self_improve_traces.jsonl"

        self.group_size = max(2, int(group_size))
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.kl_coef = float(kl_coef)
        self.clip_eps = float(clip_eps)
        self.temperature = max(0.1, float(temperature))
        self.rft_every = max(0, int(rft_every))
        self.replay_threshold = float(replay_threshold)
        self.step_idx = 0
        self.verified_traces: list[tuple[str, str]] = []

        # Use LoRA disable/enable for zero-VRAM reference policy when adapters exist; else frozen CPU/device snapshot
        has_trainable_and_frozen = any(not p.requires_grad for p in self.model.parameters()) and any(
            p.requires_grad for p in self.model.parameters()
        )
        self._lora_ref_mode = has_trainable_and_frozen
        self._ref_model: nn.Module | None = None
        if not self._lora_ref_mode:
            ref = copy.deepcopy(self.model).eval()
            for p in ref.parameters():
                p.requires_grad = False
            self._ref_model = ref.to(self.device)

    @contextmanager
    def _reference_policy(self):
        if not self._lora_ref_mode:
            assert self._ref_model is not None
            yield self._ref_model
            return

        # Temporarily zero LoRA scaling while leaving base weights intact to evaluate pi_ref in-place
        saved: list[tuple[nn.Module, float]] = []
        for mod in self.model.modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "scaling"):
                saved.append((mod, float(mod.scaling)))
                mod.scaling = 0.0
        was_training = self.model.training
        self.model.eval()
        try:
            yield self.model
        finally:
            for mod, orig_sc in saved:
                mod.scaling = orig_sc
            if was_training:
                self.model.train()

    def _encode(self, text: str) -> list[int]:
        enc = self.tokenizer.encode(text)
        return list(enc.ids if hasattr(enc, "ids") else enc)

    def _decode(self, ids: list[int]) -> str:
        return str(self.tokenizer.decode(ids))

    @torch.no_grad()
    def _sample_rollouts(self, prompt_ids: list[int]) -> torch.Tensor:
        self.model.eval()
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        seqs = prompt_t.unsqueeze(0).expand(self.group_size, -1).clone()
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None and hasattr(self.tokenizer, "token_to_id"):
            eos_id = self.tokenizer.token_to_id("[SEP]")

        for _ in range(self.max_new_tokens):
            logits, _ = forward_causal_lm(self.model, seqs, force_depth=2)
            next_logits = logits[:, -1, :].float() / self.temperature
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            seqs = torch.cat([seqs, next_tok], dim=1)
            if eos_id is not None and bool((next_tok == int(eos_id)).all().item()):
                break
        return seqs

    @staticmethod
    def _completion_logprobs(model: nn.Module, seqs: torch.Tensor, prompt_len: int) -> torch.Tensor:
        logits, _ = forward_causal_lm(model, seqs[:, :-1], force_depth=2)
        comp_logits = logits[:, prompt_len - 1 :, :].float()
        targets = seqs[:, prompt_len:]
        log_probs = F.log_softmax(comp_logits, dim=-1)
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def _rft_replay_step(self) -> float | None:
        if not self.verified_traces:
            return None
        prompt_t, comp_t = self.verified_traces[-1]
        p_ids = self._encode(prompt_t)
        c_ids = self._encode(comp_t)
        if not p_ids or not c_ids:
            return None
        full = torch.tensor(p_ids + c_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = forward_causal_lm(self.model, full[:, :-1], force_depth=2)
        targets = full[:, 1:]
        comp_start = max(0, len(p_ids) - 1)
        loss = F.cross_entropy(logits[:, comp_start:, :].float().reshape(-1, logits.size(-1)), targets[:, comp_start:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach().item())

    def train_step(self, task: VerifiableTask) -> dict[str, float]:
        prompt_ids = self._encode(task.prompt)
        if not prompt_ids:
            raise ValueError(f"Empty tokenized prompt for task {task.task_id}")
        seqs = self._sample_rollouts(prompt_ids)
        p_len = len(prompt_ids)

        rewards: list[float] = []
        comp_texts: list[str] = []
        for g_idx in range(self.group_size):
            comp_ids = seqs[g_idx, p_len:].tolist()
            comp_str = self._decode(comp_ids)
            r, _ = verify_rollout(task, comp_str)
            rewards.append(r)
            comp_texts.append(comp_str)
            if r >= self.replay_threshold:
                self.verified_traces.append((task.prompt, comp_str))
                with self.trace_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps({"task_id": task.task_id, "category": task.category, "prompt": task.prompt, "completion": comp_str, "reward": r})
                        + "\n"
                    )

        reward_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        mean_r = reward_t.mean()
        std_r = reward_t.std(unbiased=False)
        adv = (reward_t - mean_r) / (std_r + 1e-6)

        with torch.no_grad():
            old_lp = self._completion_logprobs(self.model, seqs, p_len)
            with self._reference_policy() as ref_model:
                ref_lp = self._completion_logprobs(ref_model, seqs, p_len)

        self.model.train()
        self.optimizer.zero_grad()
        new_lp = self._completion_logprobs(self.model, seqs, p_len)
        ratio = torch.exp((new_lp - old_lp).clamp(-10.0, 10.0))
        adv_exp = adv.unsqueeze(1)
        surr1 = ratio * adv_exp
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_exp
        pg_loss = -torch.minimum(surr1, surr2).mean()

        # Unbiased non-negative Schulman KL approximation: exp(ref - new) - (ref - new) - 1
        kl_diff = (ref_lp - new_lp).clamp(-10.0, 10.0)
        kl_loss = (torch.exp(kl_diff) - kl_diff - 1.0).mean()
        loss = pg_loss + self.kl_coef * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        rft_loss = None
        self.step_idx += 1
        if self.rft_every > 0 and self.step_idx % self.rft_every == 0:
            rft_loss = self._rft_replay_step()

        return {
            "step": float(self.step_idx),
            "loss": float(loss.detach().item()),
            "pg_loss": float(pg_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "mean_reward": float(mean_r.item()),
            "max_reward": float(reward_t.max().item()),
            "verified_traces": float(len(self.verified_traces)),
            "rft_loss": float(rft_loss) if rft_loss is not None else -1.0,
        }

    def train(self, steps: int) -> dict[str, Any]:
        history: list[dict[str, float]] = []
        for s in range(int(steps)):
            task = self.tasks[s % len(self.tasks)]
            metrics = self.train_step(task)
            history.append(metrics)
            if s == 0 or (s + 1) % 5 == 0 or s + 1 == int(steps):
                print(
                    f"[GRPO Step {s + 1}/{steps}] task={task.task_id} reward={metrics['mean_reward']:.3f} "
                    f"loss={metrics['loss']:.4f} kl={metrics['kl_loss']:.4f} traces={int(metrics['verified_traces'])}",
                    flush=True,
                )
        summary = {
            "steps": int(steps),
            "verified_traces": len(self.verified_traces),
            "trace_path": str(self.trace_path),
            "final_mean_reward": history[-1]["mean_reward"] if history else 0.0,
            "history": history,
        }
        (self.output_dir / "grpo_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
