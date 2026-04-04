"""Safety policy checks for agent tool execution."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PolicyDecision:
    """Result of policy validation."""

    allowed: bool
    requires_confirmation: bool = False
    reason: str = "ok"


class SafetyPolicy:
    """Simple policy engine with blacklist and per-tool confirmation rules."""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config.json")
        self.config = self._load_config()
        policy = self.config.get("safety_policy", {})

        self.confirmation_tools = set(policy.get("require_confirmation", ["delete_file", "shutdown_system", "restart_system"]))
        self.blocked_tools = set(policy.get("blocked_tools", []))
        self.blocked_path_keywords = [item.lower() for item in policy.get("blocked_path_keywords", ["windows", "system32", "program files"])]

        limits = policy.get("rate_limit", {})
        self.calls_per_minute = int(limits.get("calls_per_minute", 60))
        self._call_timestamps: list[float] = []

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def validate(self, tool_name: str, params: Dict[str, Any]) -> PolicyDecision:
        """Validate whether a tool call is allowed."""
        if tool_name in self.blocked_tools:
            return PolicyDecision(allowed=False, reason=f"Tool '{tool_name}' is blocked by policy")

        if not self._within_rate_limit():
            return PolicyDecision(allowed=False, reason="Rate limit exceeded")

        lower_values = " ".join(str(v).lower() for v in params.values())
        for blocked_piece in self.blocked_path_keywords:
            if blocked_piece in lower_values and tool_name not in {"search_web", "get_weather", "fetch_news"}:
                return PolicyDecision(allowed=False, reason=f"Blocked path keyword detected: {blocked_piece}")

        if tool_name in self.confirmation_tools:
            return PolicyDecision(allowed=True, requires_confirmation=True, reason="Confirmation required")

        return PolicyDecision(allowed=True)

    def _within_rate_limit(self) -> bool:
        """Return whether current call count is under the per-minute threshold."""
        now = time.time()
        one_minute_ago = now - 60
        self._call_timestamps = [ts for ts in self._call_timestamps if ts >= one_minute_ago]
        if len(self._call_timestamps) >= self.calls_per_minute:
            return False
        self._call_timestamps.append(now)
        return True
