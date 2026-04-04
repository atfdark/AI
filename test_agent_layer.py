"""Basic tests for the Jarvis agent layer foundation."""

from __future__ import annotations

import unittest

from assistant.action_tools import create_action_tools
from assistant.safety_policy import SafetyPolicy
from assistant.tool_registry import ToolRegistry


class DummyActions:
    def launch_app(self, app_name):
        return app_name.lower() == "chrome"

    def open_url(self, url):
        return url.startswith("http")

    def perform_search(self, query):
        return f"summary for {query}" if query else None

    def take_screenshot(self):
        return "artifacts/screenshots/screenshot.png"

    def volume_up(self, steps=2):
        return None

    def volume_down(self, steps=2):
        return None

    def get_weather(self, location):
        return f"Weather at {location}"

    def get_wikipedia_summary(self, topic):
        return f"Wiki {topic}"

    def fetch_news(self):
        return "News summary"

    def create_todo_list(self, list_name):
        return True

    def add_todo_task(self, list_name, task):
        return True

    def get_todo_lists(self):
        return {"work": {"tasks": []}}


class AgentLayerTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register_many(create_action_tools(DummyActions()))

    def test_tool_registration(self):
        self.assertGreaterEqual(len(self.registry.list_tools()), 10)

    def test_validation_missing_required(self):
        ok, reason = self.registry.validate("open_app", {})
        self.assertFalse(ok)
        self.assertIn("Missing required", reason)

    def test_validation_type(self):
        ok, _ = self.registry.validate("volume_up", {"steps": "loud"})
        self.assertFalse(ok)

    def test_safety_policy_allows_safe_tool(self):
        policy = SafetyPolicy(config_path="config.json")
        result = policy.validate("search_web", {"query": "python"})
        self.assertTrue(result.allowed)


if __name__ == "__main__":
    unittest.main()
