"""Prompt composition utilities for the agent router."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def compose_router_prompt(
    user_text: str,
    parsed_intent: str,
    parsed_confidence: float,
    parsed_parameters: Dict[str, Any],
    tools: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
) -> str:
    """Build a compact JSON-driven prompt for local model routing decisions."""
    context = context or {}

    instruction = {
        "task": "Choose how to handle the command using available tools and available memory context.",
        "allowed_routes": ["direct", "plan", "clarify", "fallback"],
        "output_schema": {
            "route": "direct|plan|clarify|fallback",
            "reason": "string",
            "reply": "optional string for user",
            "tool": "required for direct",
            "parameters": "optional object for direct",
            "steps": [
                {
                    "tool": "tool name",
                    "parameters": {"key": "value"},
                }
            ],
        },
        "rules": [
            "Prefer direct route when one tool is enough.",
            "Use plan route for multi-step requests.",
            "For question-answering that does not require desktop actions, use answer_offline.",
            "For expanding offline knowledge from local files/folders, use ingest_knowledge.",
            "For checking offline knowledge coverage, use knowledge_stats.",
            "For dataset recommendations and bundles, use knowledge_catalog.",
            "For creating a dataset acquisition plan, use knowledge_plan.",
            "For connecting downloaded corpora to local paths, use register_knowledge_source.",
            "For bulk indexing all registered corpora, use ingest_registered_knowledge.",
            "For finding local projects/folders/datasets, use search_files first and then open_path if needed.",
            "For long-term user preferences, use remember_fact or recall_memory.",
            "For plan steps, you may reference previous step output fields using $stepN.field (example: $step1.first_path).",
            "Use clarify route when request is ambiguous.",
            "Never invent tools that are not in the tools list.",
            "Return valid JSON only.",
        ],
    }

    payload = {
        "instruction": instruction,
        "user_text": user_text,
        "parsed": {
            "intent": parsed_intent,
            "confidence": parsed_confidence,
            "parameters": parsed_parameters,
        },
        "context": context,
        "tools": tools,
    }

    return json.dumps(payload, ensure_ascii=False)
