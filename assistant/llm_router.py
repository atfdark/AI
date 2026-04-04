"""Offline-first routing decisions for the Jarvis agent runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List
from urllib import request as urllib_request

from .query_composer import compose_router_prompt


@dataclass
class RouterDecision:
    """Decision output used by the plan executor."""

    route: str
    reason: str
    reply: str = ""
    tool: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)


class LLMRouter:
    """Offline-first router using heuristics with optional Ollama reasoning."""

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        agent_cfg = config.get("agent", {})
        self.enable_ollama = bool(agent_cfg.get("enable_ollama", True))
        self.ollama_url = agent_cfg.get("ollama_url", "http://localhost:11434/api/generate")
        self.ollama_model = agent_cfg.get("ollama_model", "phi3:mini")
        self.low_confidence_threshold = float(agent_cfg.get("low_confidence_threshold", 0.55))

    def decide(
        self,
        user_text: str,
        parsed_intent: str,
        parsed_confidence: float,
        parsed_parameters: Dict[str, Any],
        tools: List[Dict[str, Any]],
        context: Dict[str, Any] | None = None,
        intent_tool_map: Dict[str, str] | None = None,
    ) -> RouterDecision:
        """Return routing decision from heuristics or optional Ollama output."""
        intent_tool_map = intent_tool_map or {}
        text_lower = user_text.lower().strip()

        # Fast-path heuristic for known confident intents.
        if parsed_intent != "unknown" and parsed_confidence >= self.low_confidence_threshold:
            mapped_tool = intent_tool_map.get(parsed_intent, "")
            if mapped_tool:
                return RouterDecision(
                    route="direct",
                    reason="high-confidence parsed intent",
                    tool=mapped_tool,
                    parameters=parsed_parameters,
                )

        # Fast heuristic for multi-step conjunctions.
        if " and " in text_lower:
            steps = self._build_simple_steps(text_lower)
            if steps:
                return RouterDecision(route="plan", reason="conjunction detected", steps=steps)

        # Ollama reasoning for uncertain or ambiguous requests.
        if self.enable_ollama:
            ollama_result = self._call_ollama(
                user_text,
                parsed_intent,
                parsed_confidence,
                parsed_parameters,
                tools,
                context or {},
            )
            if ollama_result:
                return ollama_result

        # Conservative fallback.
        if parsed_intent == "unknown":
            return RouterDecision(route="clarify", reason="unknown intent", reply="I need more details to help with that request.")

        return RouterDecision(route="fallback", reason="no agent decision")

    def _call_ollama(
        self,
        user_text: str,
        parsed_intent: str,
        parsed_confidence: float,
        parsed_parameters: Dict[str, Any],
        tools: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> RouterDecision | None:
        """Query local Ollama and parse a JSON decision."""
        prompt = compose_router_prompt(
            user_text=user_text,
            parsed_intent=parsed_intent,
            parsed_confidence=parsed_confidence,
            parsed_parameters=parsed_parameters,
            tools=tools,
            context=context,
        )

        body = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            payload = json.dumps(body).encode("utf-8")
            req = urllib_request.Request(
                self.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=15) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            text = data.get("response", "").strip()
            parsed = self._parse_json_payload(text)
            if not parsed:
                return None

            route = parsed.get("route", "fallback")
            return RouterDecision(
                route=route,
                reason=parsed.get("reason", "ollama decision"),
                reply=parsed.get("reply", ""),
                tool=parsed.get("tool", ""),
                parameters=parsed.get("parameters", {}) or {},
                steps=parsed.get("steps", []) or [],
            )
        except Exception:
            return None

    @staticmethod
    def _parse_json_payload(text: str) -> Dict[str, Any] | None:
        """Best-effort extraction of first JSON object from model text."""
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None

    @staticmethod
    def _build_simple_steps(text_lower: str) -> List[Dict[str, Any]]:
        """Build a tiny heuristic plan for common compound commands."""
        steps: List[Dict[str, Any]] = []
        parts = [part.strip() for part in text_lower.split(" and ") if part.strip()]

        for part in parts:
            if part.startswith("open "):
                steps.append({"tool": "open_app", "parameters": {"app_name": part.replace("open ", "", 1).strip().title()}})
            elif part.startswith("search ") or part.startswith("search for "):
                query = part.replace("search for", "", 1).replace("search", "", 1).strip()
                if query:
                    steps.append({"tool": "search_web", "parameters": {"query": query}})
            elif "screenshot" in part:
                steps.append({"tool": "take_screenshot", "parameters": {}})

        return steps
