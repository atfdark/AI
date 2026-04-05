#!/usr/bin/env python3
"""CLI for building and syncing the offline knowledge corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from assistant.knowledge_bootstrap import KnowledgeBootstrapManager
from assistant.knowledge_brain import KnowledgeBrain


def _load_config(config_path: str) -> dict:
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _manager_from_config(config_path: str) -> KnowledgeBootstrapManager:
    config = _load_config(config_path)
    corpus_cfg = config.get("knowledge_corpus", {})
    workspace_root = str(Path(__file__).resolve().parent)
    registry_path = corpus_cfg.get("registry_path", "knowledge_sources/dataset_registry.json")
    if not os.path.isabs(registry_path):
        registry_path = os.path.join(workspace_root, registry_path)
    return KnowledgeBootstrapManager(workspace_root=workspace_root, registry_path=registry_path)


def _brain_from_config(config_path: str) -> KnowledgeBrain:
    config = _load_config(config_path)
    root = str(Path(__file__).resolve().parent)
    kb_cfg = config.get("knowledge_brain", {})
    db_path = kb_cfg.get("db_path", "data_versions/knowledge_brain.db")
    if not os.path.isabs(db_path):
        db_path = os.path.join(root, db_path)
    return KnowledgeBrain(
        db_path=db_path,
        embedding_model=kb_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        use_embeddings=bool(kb_cfg.get("enable_embeddings", True)),
        chunk_size_chars=int(kb_cfg.get("chunk_size_chars", 1200)),
        chunk_overlap_chars=int(kb_cfg.get("chunk_overlap_chars", 180)),
        ocr_enabled=bool(kb_cfg.get("ocr_enabled", True)),
        trust_overrides=kb_cfg.get("trust_overrides", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline knowledge corpus bootstrap manager")
    parser.add_argument("--config", default="config.json", help="Path to config file")

    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="Show curated dataset catalog")
    catalog.add_argument("--bundle", default="starter")

    plan = sub.add_parser("plan", help="Generate dataset plan JSON")
    plan.add_argument("--bundle", default="starter")
    plan.add_argument("--output", default="")

    register = sub.add_parser("register", help="Register local dataset path")
    register.add_argument("--dataset", required=True)
    register.add_argument("--path", required=True)
    register.add_argument("--notes", default="")

    sub.add_parser("list-registered", help="List registered dataset paths")

    ingest = sub.add_parser("ingest-registered", help="Ingest all registered datasets")
    ingest.add_argument("--max-files-per-dataset", type=int, default=2000)

    args = parser.parse_args()
    manager = _manager_from_config(args.config)

    if args.command == "catalog":
        data = manager.catalog(bundle=args.bundle)
        print(json.dumps({"bundle": args.bundle, "datasets": data}, indent=2, ensure_ascii=False))
        return

    if args.command == "plan":
        plan_payload = manager.create_plan(bundle=args.bundle, output_path=args.output or None)
        print(json.dumps(plan_payload, indent=2, ensure_ascii=False))
        return

    if args.command == "register":
        try:
            item = manager.register_local_source(dataset_id=args.dataset, local_path=args.path, notes=args.notes)
            print(json.dumps({"success": True, "source": item}, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"success": False, "error": str(exc)}, indent=2, ensure_ascii=False))
        return

    if args.command == "list-registered":
        print(json.dumps({"sources": manager.list_registered(), "stats": manager.stats()}, indent=2, ensure_ascii=False))
        return

    if args.command == "ingest-registered":
        brain = _brain_from_config(args.config)
        try:
            result = manager.ingest_registered(
                knowledge_brain=brain,
                max_files_per_dataset=args.max_files_per_dataset,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        finally:
            brain.close()
        return


if __name__ == "__main__":
    main()
