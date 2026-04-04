"""Jarvis agent runtime: tool registry, router, executor, and speech feedback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .action_tools import create_action_tools
from .audit_logger import AuditLogger
from .llm_router import LLMRouter
from .plan_executor import PlanExecutor
from .safety_policy import SafetyPolicy
from .tool_registry import ToolRegistry


class AgentRuntime:
    """Coordinates routing and execution for low-confidence or multi-step requests."""

    def __init__(self, actions, tts=None, config_path: str | None = None):
        self.actions = actions
        self.tts = tts
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config.json")
        self.config = self._load_config()

        agent_cfg = self.config.get("agent", {})
        self.enabled = bool(agent_cfg.get("enabled", True))
        self.speak_responses = bool(agent_cfg.get("speak_responses", True))
        self.low_confidence_threshold = float(agent_cfg.get("low_confidence_threshold", 0.55))

        self.intent_tool_map = {
            "open_application": "open_app",
            "web_browsing": "open_url",
            "search": "search_web",
            "screenshot": "take_screenshot",
            "weather": "get_weather",
            "wikipedia": "wikipedia_summary",
            "news_reporting": "fetch_news",
            "todo_generation": "create_todo",
            "todo_management": "list_todos",
            "volume_control": "volume_up",
        }

        self.registry = ToolRegistry()
        self.registry.register_many(create_action_tools(actions))

        self.safety_policy = SafetyPolicy(config_path=self.config_path)
        self.audit_logger = AuditLogger()
        self.router = LLMRouter(config=self.config)
        self.executor = PlanExecutor(self.registry, self.safety_policy, self.audit_logger)

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def is_candidate(self, text: str, parsed_intent: str, parsed_confidence: float, mode: str) -> bool:
        """Return whether this command should be handled by the agent layer."""
        if not self.enabled or mode != "command":
            return False

        lowered = text.lower().strip()
        if lowered.startswith("agent "):
            return True
        if " and " in lowered:
            return True
        if parsed_intent == "unknown":
            return True
        if parsed_confidence < self.low_confidence_threshold:
            return True

        return False

    def process(self, text: str, parsed_result, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Process a command through router + executor."""
        if not self.enabled:
            return {"handled": False}

        parsed_intent = parsed_result.intent.value
        parsed_confidence = float(parsed_result.confidence)
        parsed_parameters = parsed_result.parameters or {}

        if not self.is_candidate(text, parsed_intent, parsed_confidence, mode="command"):
            return {"handled": False}

        decision = self.router.decide(
            user_text=text,
            parsed_intent=parsed_intent,
            parsed_confidence=parsed_confidence,
            parsed_parameters=parsed_parameters,
            tools=self.registry.list_tool_summaries(),
            context=context or {},
            intent_tool_map=self.intent_tool_map,
        )

        if decision.route == "direct" and decision.tool:
            result = self.executor.execute_direct(decision.tool, decision.parameters)
        elif decision.route == "plan":
            result = self.executor.execute_plan(decision.steps)
        elif decision.route == "clarify":
            result = {"success": False, "message": decision.reply or "Please clarify your request."}
        else:
            return {"handled": False}

        message = result.get("message", "Done")
        if self.tts and self.speak_responses and message:
            self.tts.say(message)

        return {
            "handled": True,
            "success": bool(result.get("success", False)),
            "response": message,
            "decision": {
                "route": decision.route,
                "reason": decision.reason,
            },
            "result": result,
        }

    def health_check(self) -> Dict[str, Any]:
        """Return runtime diagnostics without invoking voice loop."""
        return {
            "enabled": self.enabled,
            "tool_count": len(self.registry.list_tools()),
            "ollama_enabled": self.router.enable_ollama,
            "ollama_model": self.router.ollama_model,
        }
