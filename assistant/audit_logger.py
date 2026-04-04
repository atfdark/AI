"""Audit trail writer for agent tool decisions and executions."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


class AuditLogger:
    """Writes structured audit entries to logs/audit_trail.jsonl."""

    def __init__(self, log_path: str | None = None):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.log_path = log_path or os.path.join(root_dir, "logs", "audit_trail.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Append one audit event to the JSONL file."""
        record = {
            "timestamp": time.time(),
            "event_type": event_type,
            **payload,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Never break runtime flow for audit logging failures.
            return
