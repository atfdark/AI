"""Tests for offline knowledge brain ingestion and retrieval behavior."""

from __future__ import annotations

import os
import tempfile
import unittest

from assistant.knowledge_brain import KnowledgeBrain
from assistant.llm_router import LLMRouter


class KnowledgeBrainTests(unittest.TestCase):
    def test_ingest_text_and_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "knowledge.db")
            brain = KnowledgeBrain(
                db_path=db_path,
                use_embeddings=False,
                chunk_size_chars=500,
                chunk_overlap_chars=50,
            )

            ingest = brain.ingest_text(
                source_title="physics_notes",
                source_path="memory://physics_notes",
                text=(
                    "Physics explains matter, energy, and force. "
                    "Classical mechanics studies motion under forces. "
                    "Thermodynamics studies heat and entropy."
                ),
            )
            self.assertTrue(ingest["success"])
            self.assertGreater(ingest["chunks"], 0)

            hits = brain.search("what is thermodynamics", top_k=3)
            self.assertTrue(hits)
            top_text = hits[0]["text"].lower()
            self.assertIn("thermodynamics", top_text)
            self.assertIn("trust_score", hits[0])

            context = brain.build_context("force and motion", top_k=3)
            self.assertTrue(context)
            self.assertIn("[K1]", context)

            cited = brain.build_cited_context("force and motion", top_k=3)
            self.assertTrue(cited["context"])
            self.assertTrue(cited["citations"])
            self.assertEqual(cited["citations"][0]["id"], "K1")

            stats = brain.stats()
            self.assertIn("avg_trust", stats)

            brain.close()

    def test_router_handles_knowledge_commands(self):
        router = LLMRouter(config={"agent": {"enable_ollama": False}})
        tools = [
            {"name": "ingest_knowledge", "description": "", "parameters": {}, "required": []},
            {"name": "knowledge_stats", "description": "", "parameters": {}, "required": []},
            {"name": "answer_offline", "description": "", "parameters": {}, "required": []},
        ]

        stats_decision = router.decide(
            user_text="show me knowledge stats",
            parsed_intent="unknown",
            parsed_confidence=0.2,
            parsed_parameters={},
            tools=tools,
            context={},
            intent_tool_map={},
        )
        self.assertEqual(stats_decision.route, "direct")
        self.assertEqual(stats_decision.tool, "knowledge_stats")

        ingest_decision = router.decide(
            user_text="ingest knowledge D:/data/books",
            parsed_intent="unknown",
            parsed_confidence=0.1,
            parsed_parameters={},
            tools=tools,
            context={},
            intent_tool_map={},
        )
        self.assertEqual(ingest_decision.route, "direct")
        self.assertEqual(ingest_decision.tool, "ingest_knowledge")


if __name__ == "__main__":
    unittest.main()
