"""Plan execution engine for Jarvis tool calls."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .tool_registry import ToolRegistry
from .safety_policy import SafetyPolicy
from .audit_logger import AuditLogger


class PlanExecutor:
    """Executes direct or multi-step tool calls with validation and policy checks."""

    def __init__(self, registry: ToolRegistry, safety_policy: SafetyPolicy, audit_logger: AuditLogger):
        self.registry = registry
        self.safety_policy = safety_policy
        self.audit_logger = audit_logger

    def execute_direct(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one tool call safely."""
        valid, reason = self.registry.validate(tool_name, params)
        if not valid:
            self.audit_logger.log("tool_rejected", {"tool": tool_name, "params": params, "reason": reason})
            return {"success": False, "message": reason}

        policy = self.safety_policy.validate(tool_name, params)
        if not policy.allowed:
            self.audit_logger.log("policy_blocked", {"tool": tool_name, "params": params, "reason": policy.reason})
            return {"success": False, "message": policy.reason}

        if policy.requires_confirmation:
            self.audit_logger.log("policy_confirmation_required", {"tool": tool_name, "params": params})
            return {"success": False, "message": f"Confirmation required for {tool_name}"}

        tool = self.registry.get(tool_name)
        if not tool:
            return {"success": False, "message": f"Unknown tool: {tool_name}"}

        try:
            result = tool.handler(**params)
            self.audit_logger.log("tool_executed", {"tool": tool_name, "params": params, "result": result})
            if isinstance(result, dict):
                return result
            return {"success": True, "message": str(result)}
        except Exception as exc:
            message = f"Tool execution failed: {exc}"
            self.audit_logger.log("tool_error", {"tool": tool_name, "params": params, "error": str(exc)})
            return {"success": False, "message": message}

    def execute_plan(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute all plan steps in order and aggregate results."""
        if not steps:
            return {"success": False, "message": "Empty plan"}

        outputs = []
        context = {}
        for idx, step in enumerate(steps, start=1):
            tool_name = step.get("tool", "")
            raw_params = step.get("parameters", {}) or {}
            params = self._resolve_parameters(raw_params, context)
            result = self.execute_direct(tool_name, params)
            outputs.append({"step": idx, "tool": tool_name, "result": result})
            context[f"step{idx}"] = result
            if not result.get("success", False):
                return {
                    "success": False,
                    "message": f"Plan failed at step {idx}: {result.get('message', 'unknown error')}",
                    "outputs": outputs,
                }

        summary = "; ".join(item["result"].get("message", "done") for item in outputs)
        return {"success": True, "message": summary, "outputs": outputs}

    def _resolve_parameters(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameter placeholders like $step1.first_path from prior outputs."""
        resolved: Dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = self._resolve_placeholder(value, context)
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _resolve_placeholder(value: str, context: Dict[str, Any]) -> Any:
        if not value.startswith("$step"):
            return value

        match = re.match(r"^\$step(\d+)\.([a-zA-Z0-9_]+)$", value.strip())
        if not match:
            return value

        step_name = f"step{match.group(1)}"
        field_name = match.group(2)
        step_result = context.get(step_name, {})
        if isinstance(step_result, dict):
            return step_result.get(field_name, value)
        return value
