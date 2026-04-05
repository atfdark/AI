"""Persistent semantic memory for long-term assistant preferences and context."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class SemanticMemoryStore:
    """SQLite-backed memory store with optional embedding similarity."""

    def __init__(
        self,
        db_path: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
    ):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.db_path = db_path or os.path.join(root_dir, "data_versions", "semantic_memory.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._embedder = None
        self._embeddings_enabled = bool(use_embeddings and SentenceTransformer)
        if self._embeddings_enabled:
            try:
                self._embedder = SentenceTransformer(embedding_model)
            except Exception:
                self._embedder = None
                self._embeddings_enabled = False

        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    source TEXT NOT NULL DEFAULT 'user',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS aliases (
                    alias TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.6,
                    use_count INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC)")
            self._conn.commit()

    def _encode(self, text: str) -> Optional[List[float]]:
        if not text or not self._embedder:
            return None
        try:
            vector = self._embedder.encode([text], normalize_embeddings=True)[0]
            return [float(v) for v in vector]
        except Exception:
            return None

    @staticmethod
    def _cosine(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        return float(sum(a * b for a, b in zip(v1, v2)))

    @staticmethod
    def _lexical_score(query: str, candidate: str) -> float:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return 0.0
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))
        if not candidate_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(candidate_tokens))
        return overlap / max(1, len(query_tokens))

    def remember(
        self,
        text: str,
        category: str = "general",
        source: str = "user",
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        text = (text or "").strip()
        if not text:
            return -1

        now = time.time()
        metadata = metadata or {}
        embedding = self._encode(text)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO memories (text, category, source, metadata_json, embedding_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    text,
                    category,
                    source,
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(embedding) if embedding else None,
                    now,
                    now,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def search(self, query: str, top_k: int = 5, category: str | None = None) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        with self._lock:
            cur = self._conn.cursor()
            if category:
                cur.execute(
                    """
                    SELECT id, text, category, source, metadata_json, embedding_json, created_at
                    FROM memories
                    WHERE category = ?
                    ORDER BY created_at DESC
                    LIMIT 500
                    """,
                    (category,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, text, category, source, metadata_json, embedding_json, created_at
                    FROM memories
                    ORDER BY created_at DESC
                    LIMIT 500
                    """
                )
            rows = cur.fetchall()

        query_vec = self._encode(query)
        scored: List[Dict[str, Any]] = []
        for row in rows:
            text = row["text"]
            lexical = self._lexical_score(query, text)
            semantic = 0.0
            try:
                if query_vec and row["embedding_json"]:
                    candidate = json.loads(row["embedding_json"])
                    semantic = self._cosine(query_vec, candidate)
            except Exception:
                semantic = 0.0

            score = semantic if semantic > 0 else lexical
            if score <= 0:
                continue

            scored.append(
                {
                    "id": int(row["id"]),
                    "text": text,
                    "category": row["category"],
                    "source": row["source"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "score": float(score),
                    "created_at": float(row["created_at"]),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, int(top_k))]

    def recall_recent(self, limit: int = 5, category: str | None = None) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        with self._lock:
            cur = self._conn.cursor()
            if category:
                cur.execute(
                    """
                    SELECT id, text, category, source, metadata_json, created_at
                    FROM memories
                    WHERE category = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (category, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, text, category, source, metadata_json, created_at
                    FROM memories
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = cur.fetchall()

        return [
            {
                "id": int(row["id"]),
                "text": row["text"],
                "category": row["category"],
                "source": row["source"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def remember_alias(self, alias: str, target: str, confidence: float = 0.6) -> bool:
        alias = (alias or "").strip().lower()
        target = (target or "").strip()
        if not alias or not target:
            return False

        now = time.time()
        confidence = max(0.0, min(1.0, float(confidence)))

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO aliases (alias, target, confidence, use_count, updated_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    target = excluded.target,
                    confidence = (aliases.confidence * aliases.use_count + excluded.confidence) / (aliases.use_count + 1),
                    use_count = aliases.use_count + 1,
                    updated_at = excluded.updated_at
                """,
                (alias, target, confidence, now),
            )
            self._conn.commit()
            return True

    def resolve_alias(self, alias: str, min_confidence: float = 0.6) -> Optional[str]:
        alias = (alias or "").strip().lower()
        if not alias:
            return None

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT target, confidence FROM aliases WHERE alias = ?", (alias,))
            row = cur.fetchone()

        if not row:
            return None
        if float(row["confidence"]) < float(min_confidence):
            return None
        return str(row["target"])

    def learn_alias_from_command(self, user_text: str, tool_name: str, params: Dict[str, Any], success: bool) -> None:
        if not success or tool_name != "open_app":
            return

        target = str(params.get("app_name", "")).strip()
        if not target:
            return

        match = re.search(r"^(?:please\s+)?(?:open|launch|start|run)\s+(.+)$", user_text.strip(), re.IGNORECASE)
        if not match:
            return

        alias = match.group(1).strip().lower()
        alias = re.sub(r"\b(app|application|program|please|now)\b", "", alias).strip()
        if not alias or alias == target.lower():
            return

        self.remember_alias(alias=alias, target=target, confidence=0.7)

    def track_interaction(
        self,
        user_text: str,
        decision_route: str,
        tool_name: str,
        success: bool,
        response: str,
    ) -> None:
        note = {
            "user": user_text,
            "route": decision_route,
            "tool": tool_name,
            "success": bool(success),
            "response": response,
        }
        summary = f"User said: {user_text}. Route: {decision_route}. Tool: {tool_name}. Success: {bool(success)}."
        self.remember(summary, category="interaction", source="agent", metadata=note)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                return
