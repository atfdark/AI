"""Offline-first routing decisions for the Jarvis agent runtime."""

from __future__ import annotations

import json
import re
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
        self.ollama_model = agent_cfg.get("ollama_model", "tinyllama:latest")
        self.low_confidence_threshold = float(agent_cfg.get("low_confidence_threshold", 0.55))
        self.request_timeout = float(agent_cfg.get("ollama_timeout_seconds", 20))

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
        tool_names = {tool.get("name", "") for tool in tools}

        # Goal-directed task chaining for workspace setup requests.
        workspace_steps = self._build_workspace_plan(text_lower, tool_names)
        if workspace_steps:
            return RouterDecision(route="plan", reason="workspace preparation task", steps=workspace_steps)

        # Semantic file discovery for indirect folder or dataset requests.
        file_plan = self._build_file_retrieval_plan(user_text, text_lower, tool_names)
        if file_plan:
            return RouterDecision(route="plan", reason="semantic file discovery", steps=file_plan)

        # Persistent preference memory commands.
        memory_direct = self._build_memory_direct(text_lower, tool_names)
        if memory_direct:
            return memory_direct

        # Knowledge-base management commands.
        knowledge_direct = self._build_knowledge_direct(user_text, text_lower, tool_names)
        if knowledge_direct:
            return knowledge_direct

        # Offline question-answering route.
        if self._looks_like_question(text_lower) and "answer_offline" in tool_names:
            return RouterDecision(
                route="direct",
                reason="offline knowledge question",
                tool="answer_offline",
                parameters={"question": user_text},
            )

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
            steps = self._build_simple_steps(text_lower, tool_names)
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
            "format": "json",
        }

        try:
            payload = json.dumps(body).encode("utf-8")
            req = urllib_request.Request(
                self.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=self.request_timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            text = data.get("response", "").strip()
            parsed = self._parse_json_payload(text)
            if not parsed:
                return None

            allowed = {tool.get("name", "") for tool in tools}
            parsed = self._normalize_decision(parsed, allowed)

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

    def answer_question(self, question: str, memory_context: str = "", knowledge_context: str = "") -> str:
        """Generate an offline answer with optional memory context."""
        if not self.enable_ollama:
            return "I can only answer from stored memory right now because Ollama is disabled."

        prompt_payload = {
            "task": "Answer the question concisely using local knowledge and optional memory context.",
            "rules": [
                "Be factual and concise.",
                "If uncertain, state uncertainty clearly.",
                "Use the provided memory context only when relevant.",
                "When knowledge_context includes citation tags like [K1], preserve and reference those tags in the answer.",
            ],
            "question": question,
            "memory_context": memory_context,
            "knowledge_context": knowledge_context,
        }
        prompt = json.dumps(prompt_payload, ensure_ascii=False)
        answer = self._call_ollama_text(prompt)
        return answer or "I could not produce a reliable offline answer."

    def _call_ollama_text(self, prompt: str) -> str:
        body = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }

        try:
            payload = json.dumps(body).encode("utf-8")
            req = urllib_request.Request(
                self.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=self.request_timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            return data.get("response", "").strip()
        except Exception:
            return ""

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
    def _normalize_decision(raw_decision: Dict[str, Any], allowed_tools: set[str]) -> Dict[str, Any]:
        route = raw_decision.get("route", "fallback")
        normalized = {
            "route": route,
            "reason": raw_decision.get("reason", "ollama decision"),
            "reply": raw_decision.get("reply", ""),
            "tool": "",
            "parameters": raw_decision.get("parameters", {}) or {},
            "steps": [],
        }

        if route == "direct":
            tool = raw_decision.get("tool", "")
            if tool in allowed_tools:
                normalized["tool"] = tool
            else:
                normalized["route"] = "clarify"
                normalized["reply"] = "I need clarification before executing that request."

        if route == "plan":
            steps = raw_decision.get("steps", []) or []
            clean_steps = []
            for step in steps:
                tool = step.get("tool", "")
                if tool not in allowed_tools:
                    continue
                clean_steps.append(
                    {
                        "tool": tool,
                        "parameters": step.get("parameters", {}) or {},
                    }
                )
            if clean_steps:
                normalized["steps"] = clean_steps
            else:
                normalized["route"] = "clarify"
                normalized["reply"] = "I need more details to build a valid plan."

        return normalized

    @staticmethod
    def _build_simple_steps(text_lower: str, tool_names: set[str]) -> List[Dict[str, Any]]:
        """Build a tiny heuristic plan for common compound commands."""
        steps: List[Dict[str, Any]] = []
        parts = [part.strip() for part in text_lower.split(" and ") if part.strip()]

        for part in parts:
            if part.startswith("open ") and "open_app" in tool_names:
                steps.append({"tool": "open_app", "parameters": {"app_name": part.replace("open ", "", 1).strip().title()}})
            elif (part.startswith("search ") or part.startswith("search for ")) and "search_web" in tool_names:
                query = part.replace("search for", "", 1).replace("search", "", 1).strip()
                if query:
                    steps.append({"tool": "search_web", "parameters": {"query": query}})
            elif "screenshot" in part and "take_screenshot" in tool_names:
                steps.append({"tool": "take_screenshot", "parameters": {}})

        return steps

    @staticmethod
    def _looks_like_question(text_lower: str) -> bool:
        question_starts = (
            "what is",
            "what are",
            "who is",
            "how to",
            "how does",
            "why",
            "explain",
            "define",
            "tell me about",
        )
        if any(text_lower.startswith(prefix) for prefix in question_starts):
            return True
        return text_lower.endswith("?")

    @staticmethod
    def _build_workspace_plan(text_lower: str, tool_names: set[str]) -> List[Dict[str, Any]]:
        if "prepare my coding workspace" not in text_lower:
            return []

        steps: List[Dict[str, Any]] = []
        if "open_app" in tool_names:
            steps.append({"tool": "open_app", "parameters": {"app_name": "VS Code"}})
        search_step_index = -1
        if "search_files" in tool_names:
            steps.append({"tool": "search_files", "parameters": {"query": "dataset project notebook", "top_k": 3}})
            search_step_index = len(steps)
        if "open_path" in tool_names:
            if search_step_index > 0:
                steps.append({"tool": "open_path", "parameters": {"path": f"$step{search_step_index}.first_path"}})
        if "open_url" in tool_names:
            steps.append({"tool": "open_url", "parameters": {"url": "https://duckduckgo.com/?q=machine+learning+workspace+references"}})
        return steps

    @staticmethod
    def _build_file_retrieval_plan(user_text: str, text_lower: str, tool_names: set[str]) -> List[Dict[str, Any]]:
        if "search_files" not in tool_names:
            return []

        trigger_words = ["dataset", "folder", "project", "files"]
        if not any(word in text_lower for word in trigger_words):
            return []
        if not any(word in text_lower for word in ["find", "where", "open", "locate"]):
            return []

        cleaned = re.sub(r"\b(open|find|where|is|my|the|folder|files|for|please|stored|locate)\b", " ", text_lower)
        cleaned_query = re.sub(r"\s+", " ", cleaned).strip() or user_text

        steps = [{"tool": "search_files", "parameters": {"query": cleaned_query, "top_k": 5}}]
        if "open" in text_lower and "open_path" in tool_names:
            steps.append({"tool": "open_path", "parameters": {"path": "$step1.first_path"}})
        return steps

    @staticmethod
    def _build_memory_direct(text_lower: str, tool_names: set[str]) -> RouterDecision | None:
        if text_lower.startswith("remember ") and "remember_fact" in tool_names:
            text = text_lower.replace("remember", "", 1).strip()
            return RouterDecision(
                route="direct",
                reason="explicit remember command",
                tool="remember_fact",
                parameters={"text": text, "category": "preference"},
            )

        if ("what do you remember" in text_lower or text_lower.startswith("recall ")) and "recall_memory" in tool_names:
            query = text_lower.replace("recall", "", 1).strip() or "recent preferences"
            return RouterDecision(
                route="direct",
                reason="explicit memory recall",
                tool="recall_memory",
                parameters={"query": query, "top_k": 5},
            )

        return None

    @staticmethod
    def _build_knowledge_direct(user_text: str, text_lower: str, tool_names: set[str]) -> RouterDecision | None:
        if "knowledge_catalog" in tool_names:
            if any(
                token in text_lower
                for token in (
                    "knowledge catalog",
                    "dataset catalog",
                    "list datasets",
                    "which datasets should",
                    "starter bundle",
                )
            ):
                bundle = "starter"
                if "full" in text_lower:
                    bundle = "full"
                elif "medical" in text_lower:
                    bundle = "medical_plus"
                elif "research" in text_lower:
                    bundle = "research_plus"
                elif "core" in text_lower:
                    bundle = "core_plus"
                return RouterDecision(
                    route="direct",
                    reason="knowledge dataset catalog request",
                    tool="knowledge_catalog",
                    parameters={"bundle": bundle},
                )

        if "knowledge_plan" in tool_names and any(
            token in text_lower for token in ("knowledge plan", "dataset plan", "build corpus plan", "create corpus plan")
        ):
            bundle = "starter"
            if "full" in text_lower:
                bundle = "full"
            elif "medical" in text_lower:
                bundle = "medical_plus"
            elif "research" in text_lower:
                bundle = "research_plus"
            elif "core" in text_lower:
                bundle = "core_plus"
            return RouterDecision(
                route="direct",
                reason="dataset plan request",
                tool="knowledge_plan",
                parameters={"bundle": bundle},
            )

        if "register_knowledge_source" in tool_names:
            register_match = re.search(
                r"(?:register|link|connect)\s+([a-z0-9_\-\s]+?)\s+(?:at|to)\s+(.+)$",
                user_text.strip(),
                re.IGNORECASE,
            )
            if register_match:
                dataset_id = register_match.group(1).strip().lower().replace("-", "_").replace(" ", "_")
                path = register_match.group(2).strip().strip("\"'")
                if dataset_id and path:
                    return RouterDecision(
                        route="direct",
                        reason="dataset registration request",
                        tool="register_knowledge_source",
                        parameters={"dataset_id": dataset_id, "path": path},
                    )

        if "ingest_registered_knowledge" in tool_names and any(
            token in text_lower
            for token in (
                "ingest registered",
                "index registered",
                "sync registered datasets",
                "ingest all registered",
            )
        ):
            return RouterDecision(
                route="direct",
                reason="registered corpus ingestion request",
                tool="ingest_registered_knowledge",
                parameters={"max_files_per_dataset": 2000},
            )

        if "list_registered_knowledge" in tool_names and any(
            token in text_lower
            for token in ("registered knowledge", "registered datasets", "show registered sources")
        ):
            return RouterDecision(
                route="direct",
                reason="registered corpus listing request",
                tool="list_registered_knowledge",
                parameters={},
            )

        if "knowledge_stats" in tool_names:
            stats_triggers = (
                "knowledge stats",
                "how much do you know",
                "how many knowledge sources",
                "what is in your knowledge base",
            )
            if any(trigger in text_lower for trigger in stats_triggers):
                return RouterDecision(
                    route="direct",
                    reason="knowledge stats request",
                    tool="knowledge_stats",
                    parameters={},
                )

        if "ingest_knowledge" not in tool_names:
            return None

        match = re.search(r"(?:ingest|index|learn from|add knowledge from)\s+(.+)$", user_text.strip(), re.IGNORECASE)
        if not match:
            return None

        raw_path = match.group(1).strip().strip("\"'")
        if not raw_path:
            return None

        return RouterDecision(
            route="direct",
            reason="knowledge ingestion request",
            tool="ingest_knowledge",
            parameters={"path": raw_path, "max_files": 300},
        )
