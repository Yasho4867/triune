"""Agent Tool Abstraction & Tool Execution System."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters_schema: Dict[str, Any]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class ToolRegistry:
    """Registry for tools exposed to Triune Agentic Execution Engine."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self, name: str, description: str, func: Callable, parameters_schema: Dict[str, Any] | None = None
    ) -> Tool:
        tool = Tool(name=name, description=description, func=func, parameters_schema=parameters_schema or {})
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema,
            }
            for t in self._tools.values()
        ]
