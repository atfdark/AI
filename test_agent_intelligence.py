"""Tests for semantic memory, semantic file search planning, and chained plan execution."""

from __future__ import annotations

import os
import tempfile
import unittest

from assistant.llm_router import LLMRouter
from assistant.plan_executor import PlanExecutor
from assistant.safety_policy import SafetyPolicy
from assistant.semantic_memory import SemanticMemoryStore
from assistant.tool_registry import Tool, ToolRegistry
from assistant.audit_logger import AuditLogger


class AgentIntelligenceTests(unittest.TestCase):
    def test_semantic_memory_alias_and_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "memory.db")
            memory = SemanticMemoryStore(db_path=db_path, use_embeddings=False)

            memory.remember("My diabetes datasets are in D:/datasets/diabetes", category="project")
            memory.remember("I use VS Code as my coding editor", category="preference")

            hits = memory.search("diabetes dataset folder", top_k=3)
            self.assertTrue(hits)
            self.assertIn("diabetes", hits[0]["text"].lower())

            memory.learn_alias_from_command(
                user_text="open coding editor",
                tool_name="open_app",
                params={"app_name": "VS Code"},
                success=True,
            )
            self.assertEqual(memory.resolve_alias("coding editor"), "VS Code")
            memory.close()

    def test_router_prefers_file_search_plan_for_dataset_request(self):
        router = LLMRouter(config={"agent": {"enable_ollama": False}})
        tools = [
            {"name": "search_files", "description": "", "parameters": {}, "required": []},
            {"name": "open_path", "description": "", "parameters": {}, "required": []},
            {"name": "answer_offline", "description": "", "parameters": {}, "required": []},
        ]

        decision = router.decide(
            user_text="open the folder where my diabetes datasets are stored",
            parsed_intent="unknown",
            parsed_confidence=0.1,
            parsed_parameters={},
            tools=tools,
            context={},
            intent_tool_map={},
        )

        self.assertEqual(decision.route, "plan")
        self.assertTrue(decision.steps)
        self.assertEqual(decision.steps[0].get("tool"), "search_files")

    def test_plan_executor_resolves_step_placeholders(self):
        registry = ToolRegistry()

        def search_files(query: str, top_k: int = 5):
            return {
                "success": True,
                "first_path": "D:/datasets/diabetes",
                "message": "found dataset folder",
            }

        def open_path(path: str):
            return {
                "success": bool(path == "D:/datasets/diabetes"),
                "message": f"opened {path}",
            }

        registry.register(
            Tool(
                name="search_files",
                description="",
                parameters={"query": "str", "top_k": "int"},
                required=["query"],
                handler=search_files,
            )
        )
        registry.register(
            Tool(
                name="open_path",
                description="",
                parameters={"path": "str"},
                required=["path"],
                handler=open_path,
            )
        )

        executor = PlanExecutor(
            registry=registry,
            safety_policy=SafetyPolicy(config_path="config.json"),
            audit_logger=AuditLogger(log_path=os.path.join(tempfile.gettempdir(), "audit_test.jsonl")),
        )

        result = executor.execute_plan(
            [
                {"tool": "search_files", "parameters": {"query": "diabetes dataset", "top_k": 3}},
                {"tool": "open_path", "parameters": {"path": "$step1.first_path"}},
            ]
        )

        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
