"""Tool registry and schema validation for the Jarvis agent layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    """Represents a callable tool exposed to the agent runtime."""

    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable[..., Any]
    required: List[str] = field(default_factory=list)
    safety_level: str = "low"


class ToolRegistry:
    """Stores available tools and validates parameters before execution."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register or overwrite a tool by name."""
        self._tools[tool.name] = tool

    def register_many(self, tools: List[Tool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_tool_summaries(self) -> List[Dict[str, Any]]:
        """Return serializable summaries for prompts/logging."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "required": tool.required,
                "safety_level": tool.safety_level,
            }
            for tool in self._tools.values()
        ]

    def validate(self, tool_name: str, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate required and typed parameters for a tool."""
        tool = self.get(tool_name)
        if not tool:
            return False, f"Unknown tool: {tool_name}"

        for required_name in tool.required:
            if required_name not in params or params[required_name] in (None, ""):
                return False, f"Missing required parameter: {required_name}"

        for key, value in params.items():
            expected = tool.parameters.get(key)
            if not expected:
                continue
            if not self._matches_type(expected, value):
                return False, f"Invalid type for '{key}': expected {expected}"

        return True, "ok"

    @staticmethod
    def _matches_type(expected: str, value: Any) -> bool:
        """Check basic runtime type compatibility."""
        if expected == "str":
            return isinstance(value, str)
        if expected == "int":
            return isinstance(value, int)
        if expected == "float":
            return isinstance(value, (float, int))
        if expected == "bool":
            return isinstance(value, bool)
        if expected == "list":
            return isinstance(value, list)
        if expected == "dict":
            return isinstance(value, dict)
        return True
