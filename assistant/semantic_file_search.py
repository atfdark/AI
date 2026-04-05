"""Semantic file indexing and search for assistant file discovery tasks."""

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


class SemanticFileSearchEngine:
    """Indexes workspace files and supports lexical + semantic retrieval."""

    TEXT_EXTENSIONS = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".ts",
        ".js",
        ".html",
        ".css",
        ".xml",
        ".ini",
        ".cfg",
        ".toml",
    }
    SKIP_DIRS = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "artifacts/audio",
    }

    def __init__(
        self,
        workspace_root: str,
        db_path: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
        max_chars_per_file: int = 3000,
    ):
        self.workspace_root = os.path.abspath(workspace_root)
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = os.path.join(self.workspace_root, "data_versions", "file_index.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.max_chars_per_file = int(max_chars_per_file)
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
                CREATE TABLE IF NOT EXISTS indexed_files (
                    path TEXT PRIMARY KEY,
                    rel_path TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    preview_text TEXT NOT NULL,
                    embedding_json TEXT,
                    indexed_at REAL NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_indexed_files_mtime ON indexed_files(mtime DESC)")
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
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @classmethod
    def _lexical_score(cls, query: str, rel_path: str, preview_text: str) -> float:
        q = cls._tokenize(query)
        if not q:
            return 0.0

        p = cls._tokenize(rel_path)
        t = cls._tokenize(preview_text)
        score_path = len(q.intersection(p)) / max(1, len(q))
        score_text = len(q.intersection(t)) / max(1, len(q))
        return (0.7 * score_path) + (0.3 * score_text)

    def _is_candidate_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in self.TEXT_EXTENSIONS

    def _iter_files(self):
        for root, dirs, files in os.walk(self.workspace_root):
            normalized_root = root.replace("\\", "/")
            dirs[:] = [
                d
                for d in dirs
                if d not in self.SKIP_DIRS and not f"{normalized_root}/{d}".endswith(tuple(self.SKIP_DIRS))
            ]

            for file_name in files:
                full_path = os.path.join(root, file_name)
                if self._is_candidate_file(full_path):
                    yield full_path

    def refresh_index(self, max_files: int = 3000, force: bool = False) -> Dict[str, int]:
        indexed = 0
        skipped = 0
        touched = 0
        now = time.time()

        for full_path in self._iter_files():
            touched += 1
            if touched > max_files:
                break

            try:
                stat = os.stat(full_path)
                mtime = float(stat.st_mtime)
                size_bytes = int(stat.st_size)
                rel_path = os.path.relpath(full_path, self.workspace_root)
            except Exception:
                skipped += 1
                continue

            with self._lock:
                cur = self._conn.cursor()
                cur.execute("SELECT mtime FROM indexed_files WHERE path = ?", (full_path,))
                row = cur.fetchone()

            if row and not force and float(row["mtime"]) >= mtime:
                skipped += 1
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                    content = handle.read(self.max_chars_per_file)
            except Exception:
                skipped += 1
                continue

            combined = f"{rel_path}\n{content}"
            embedding = self._encode(combined)

            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO indexed_files (path, rel_path, mtime, size_bytes, preview_text, embedding_json, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        rel_path = excluded.rel_path,
                        mtime = excluded.mtime,
                        size_bytes = excluded.size_bytes,
                        preview_text = excluded.preview_text,
                        embedding_json = excluded.embedding_json,
                        indexed_at = excluded.indexed_at
                    """,
                    (
                        full_path,
                        rel_path,
                        mtime,
                        size_bytes,
                        content,
                        json.dumps(embedding) if embedding else None,
                        now,
                    ),
                )
                self._conn.commit()
            indexed += 1

        return {"indexed": indexed, "skipped": skipped, "touched": touched}

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT path, rel_path, size_bytes, mtime, preview_text, embedding_json
                FROM indexed_files
                ORDER BY indexed_at DESC
                LIMIT 4000
                """
            )
            rows = cur.fetchall()

        if not rows:
            self.refresh_index(max_files=3000, force=False)
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT path, rel_path, size_bytes, mtime, preview_text, embedding_json
                    FROM indexed_files
                    ORDER BY indexed_at DESC
                    LIMIT 4000
                    """
                )
                rows = cur.fetchall()

        query_vec = self._encode(query)
        scored: List[Dict[str, Any]] = []
        for row in rows:
            rel_path = str(row["rel_path"])
            preview = str(row["preview_text"])
            lexical = self._lexical_score(query, rel_path, preview)
            semantic = 0.0

            if query_vec and row["embedding_json"]:
                try:
                    candidate_vec = json.loads(row["embedding_json"])
                    semantic = self._cosine(query_vec, candidate_vec)
                except Exception:
                    semantic = 0.0

            score = max(lexical, semantic)
            if score <= 0:
                continue

            snippet = preview[:220].replace("\n", " ").strip()
            scored.append(
                {
                    "path": str(row["path"]),
                    "rel_path": rel_path,
                    "score": float(score),
                    "size_bytes": int(row["size_bytes"]),
                    "mtime": float(row["mtime"]),
                    "snippet": snippet,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, int(top_k))]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) AS c FROM indexed_files")
            count_row = cur.fetchone()
            total = int(count_row["c"]) if count_row else 0
        return {
            "workspace_root": self.workspace_root,
            "indexed_files": total,
            "embeddings_enabled": self._embeddings_enabled,
            "db_path": self.db_path,
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                return
