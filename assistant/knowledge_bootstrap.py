"""Knowledge corpus bootstrap utilities for large offline dataset workflows."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .knowledge_catalog import build_bundle_plan, get_dataset, list_datasets, resolve_dataset_ids


class KnowledgeBootstrapManager:
    """Tracks local dataset copies and helps ingest them into the knowledge brain."""

    def __init__(self, workspace_root: str, registry_path: str | None = None):
        self.workspace_root = os.path.abspath(workspace_root)
        default_registry = os.path.join(self.workspace_root, "knowledge_sources", "dataset_registry.json")
        self.registry_path = os.path.abspath(registry_path or default_registry)
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

    def _new_registry(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "workspace_root": self.workspace_root,
            "updated_at": time.time(),
            "datasets": {},
        }

    def load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(self.registry_path):
            return self._new_registry()
        try:
            with open(self.registry_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "datasets" in data:
                return data
        except Exception:
            pass
        return self._new_registry()

    def save_registry(self, data: Dict[str, Any]) -> None:
        payload = dict(data)
        payload["updated_at"] = time.time()
        with open(self.registry_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def catalog(self, bundle: str | None = None) -> List[Dict[str, Any]]:
        return [spec.to_dict() for spec in list_datasets(bundle=bundle)]

    def create_plan(
        self,
        bundle: str = "starter",
        include_ids: Iterable[str] | None = None,
        output_path: str | None = None,
    ) -> Dict[str, Any]:
        plan = build_bundle_plan(bundle=bundle, include_ids=include_ids)
        plan["created_at"] = time.time()
        plan["steps"] = [
            "Download selected datasets to a local corpus directory.",
            "Register each local dataset path in dataset_registry.json.",
            "Run ingestion to index chunks into the offline knowledge brain.",
            "Ask knowledge_stats to verify source/chunk growth.",
        ]

        target = output_path or os.path.join(self.workspace_root, "knowledge_sources", f"dataset_plan_{bundle}.json")
        resolved = os.path.abspath(target)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=False)

        plan["plan_path"] = resolved
        return plan

    def register_local_source(self, dataset_id: str, local_path: str, notes: str = "") -> Dict[str, Any]:
        normalized_id = (dataset_id or "").strip().lower().replace(" ", "_")
        spec = get_dataset(normalized_id)
        if not spec:
            raise ValueError(f"Unknown dataset id: {dataset_id}")

        resolved_path = os.path.abspath(os.path.expanduser(os.path.expandvars(local_path or "")))
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(resolved_path)

        registry = self.load_registry()
        datasets = registry.setdefault("datasets", {})
        datasets[normalized_id] = {
            "dataset_id": normalized_id,
            "name": spec.name,
            "layer": spec.layer,
            "path": resolved_path,
            "is_dir": os.path.isdir(resolved_path),
            "registered_at": time.time(),
            "recommended_trust": spec.recommended_trust,
            "requires_manual_access": spec.requires_manual_access,
            "notes": notes,
        }
        self.save_registry(registry)
        return datasets[normalized_id]

    def unregister_local_source(self, dataset_id: str) -> bool:
        normalized_id = (dataset_id or "").strip().lower().replace(" ", "_")
        registry = self.load_registry()
        datasets = registry.setdefault("datasets", {})
        if normalized_id not in datasets:
            return False
        datasets.pop(normalized_id)
        self.save_registry(registry)
        return True

    def list_registered(self) -> List[Dict[str, Any]]:
        registry = self.load_registry()
        datasets = registry.get("datasets", {})
        return list(datasets.values())

    def ingest_registered(
        self,
        knowledge_brain,
        dataset_ids: Iterable[str] | None = None,
        max_files_per_dataset: int = 2000,
    ) -> Dict[str, Any]:
        registry = self.load_registry()
        registered = registry.get("datasets", {})

        if dataset_ids:
            wanted = {item.strip().lower().replace(" ", "_") for item in dataset_ids}
        else:
            wanted = set(registered.keys())

        ingested_sources = 0
        files_scanned = 0
        files_ingested = 0
        chunks = 0
        details: List[Dict[str, Any]] = []

        for dataset_id in resolve_dataset_ids(include_ids=wanted):
            info = registered.get(dataset_id)
            if not info:
                continue

            path = info.get("path", "")
            if not path or not os.path.exists(path):
                details.append(
                    {
                        "dataset_id": dataset_id,
                        "success": False,
                        "reason": "path_missing",
                        "path": path,
                    }
                )
                continue

            if os.path.isdir(path):
                result = knowledge_brain.ingest_directory(root_path=path, max_files=max_files_per_dataset)
                files_scanned += int(result.get("files_scanned", 0))
                files_ingested += int(result.get("files_ingested", 0))
                chunks += int(result.get("chunks", 0))
                success = bool(result.get("files_ingested", 0) > 0)
            else:
                result = knowledge_brain.ingest_file(path)
                files_scanned += 1
                file_success = bool(result.get("success", False))
                files_ingested += 1 if file_success else 0
                chunks += int(result.get("chunks", 0))
                success = file_success

            if success:
                ingested_sources += 1

            details.append(
                {
                    "dataset_id": dataset_id,
                    "path": path,
                    "success": success,
                    "result": result,
                }
            )

        return {
            "success": ingested_sources > 0,
            "ingested_sources": ingested_sources,
            "requested_sources": len(wanted),
            "files_scanned": files_scanned,
            "files_ingested": files_ingested,
            "chunks": chunks,
            "details": details,
        }

    def recommended_trust_overrides(self) -> Dict[str, float]:
        overrides: Dict[str, float] = {}
        for item in self.list_registered():
            path = item.get("path", "")
            trust = float(item.get("recommended_trust", 0.6))
            if path:
                normalized = Path(path).name.lower()
                overrides[normalized] = max(0.0, min(1.0, trust))
        return overrides

    def stats(self) -> Dict[str, Any]:
        registered = self.list_registered()
        manual_access = sum(1 for item in registered if item.get("requires_manual_access"))
        accessible = sum(1 for item in registered if os.path.exists(item.get("path", "")))
        return {
            "registry_path": self.registry_path,
            "registered_sources": len(registered),
            "accessible_sources": accessible,
            "manual_access_sources": manual_access,
        }
