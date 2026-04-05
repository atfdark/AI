"""Tests for dataset catalog and knowledge corpus bootstrap pipeline."""

from __future__ import annotations

import os
import tempfile
import unittest

from assistant.knowledge_bootstrap import KnowledgeBootstrapManager
from assistant.knowledge_brain import KnowledgeBrain
from assistant.knowledge_catalog import build_bundle_plan, get_dataset


class KnowledgeBootstrapTests(unittest.TestCase):
    def test_catalog_and_plan(self):
        plan = build_bundle_plan(bundle="starter")
        self.assertGreaterEqual(plan["dataset_count"], 6)
        self.assertIn("estimated_size_gb", plan)

        wikipedia = get_dataset("wikipedia")
        self.assertIsNotNone(wikipedia)
        self.assertGreater(wikipedia.recommended_trust, 0.0)

    def test_register_and_ingest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = temp_dir
            corpus_dir = os.path.join(temp_dir, "wikipedia_sample")
            os.makedirs(corpus_dir, exist_ok=True)
            sample_file = os.path.join(corpus_dir, "entry.txt")
            with open(sample_file, "w", encoding="utf-8") as handle:
                handle.write("Gravity is a fundamental interaction between mass and energy.")

            manager = KnowledgeBootstrapManager(
                workspace_root=workspace_root,
                registry_path=os.path.join(temp_dir, "dataset_registry.json"),
            )
            manager.register_local_source("wikipedia", corpus_dir)

            db_path = os.path.join(temp_dir, "knowledge.db")
            brain = KnowledgeBrain(db_path=db_path, use_embeddings=False)
            try:
                result = manager.ingest_registered(
                    knowledge_brain=brain,
                    dataset_ids=["wikipedia"],
                    max_files_per_dataset=20,
                )
                self.assertTrue(result["success"])
                self.assertGreater(result["files_ingested"], 0)

                stats = brain.stats()
                self.assertGreater(stats["sources"], 0)
                self.assertGreater(stats["chunks"], 0)
            finally:
                brain.close()


if __name__ == "__main__":
    unittest.main()
