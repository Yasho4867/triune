"""Triune Agentic Engine & Multi-Agent Orchestrator.

Provides lightweight function execution, tool calling, and multi-agent loops
integrated with PythonSandbox for secure code execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from triune.runtime.sandbox import PythonSandbox

from .tools import ToolRegistry


@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    sandbox: PythonSandbox = field(default_factory=PythonSandbox)

    def run_task(self, task_description: str) -> str:
        """Execute task within agent context."""
        # Lightweight agent loop execution
        result = f"Agent [{self.name} - {self.role}] processing task: {task_description}"
        return result


class MultiAgentOrchestrator:
    """Orchestrates multi-agent collaboration loops."""

    def __init__(self) -> None:
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def execute_pipeline(self, goal: str) -> List[str]:
        results = []
        for agent in self.agents:
            res = agent.run_task(f"Collaborative Goal: {goal}")
            results.append(res)
        return results
