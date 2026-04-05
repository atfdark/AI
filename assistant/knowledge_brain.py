"""Persistent offline knowledge base with ingestion and retrieval for broad Q&A."""

from __future__ import annotations

import json
import io
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image
except Exception:
    Image = None


class KnowledgeBrain:
    """Stores curated knowledge chunks and retrieves relevant context for QA."""

    DEFAULT_EXTENSIONS = {".txt", ".md", ".rst", ".json", ".pdf"}

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
        chunk_size_chars: int = 1200,
        chunk_overlap_chars: int = 180,
        ocr_enabled: bool = True,
        trust_overrides: Dict[str, float] | None = None,
    ):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.chunk_size_chars = max(300, int(chunk_size_chars))
        self.chunk_overlap_chars = max(0, min(int(chunk_overlap_chars), self.chunk_size_chars // 2))
        self.ocr_enabled = bool(ocr_enabled)
        self.trust_overrides = trust_overrides or {}

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
                CREATE TABLE IF NOT EXISTS kb_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT UNIQUE,
                    source_title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    trust_score REAL NOT NULL DEFAULT 0.5,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    added_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    keywords_json TEXT NOT NULL DEFAULT '[]',
                    citation_json TEXT NOT NULL DEFAULT '{}',
                    embedding_json TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES kb_sources(id)
                )
                """
            )
            self._ensure_column(cur, "kb_sources", "trust_score", "REAL NOT NULL DEFAULT 0.5")
            self._ensure_column(cur, "kb_sources", "metadata_json", "TEXT NOT NULL DEFAULT '{}'" )
            self._ensure_column(cur, "kb_chunks", "citation_json", "TEXT NOT NULL DEFAULT '{}'" )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_kb_chunks_source_id ON kb_chunks(source_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_kb_chunks_created_at ON kb_chunks(created_at DESC)")
            self._conn.commit()

    @staticmethod
    def _ensure_column(cursor, table: str, column_name: str, column_ddl: str) -> None:
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column_name not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_ddl}")

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
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    @classmethod
    def _extract_keywords(cls, text: str, max_keywords: int = 12) -> List[str]:
        stop_words = {
            "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "is", "are",
            "was", "were", "be", "how", "what", "why", "who", "when", "where", "this", "that",
        }
        tokens = [t for t in cls._tokenize(text) if len(t) > 2 and t not in stop_words]
        freq: Dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        ranked = sorted(freq.items(), key=lambda pair: pair[1], reverse=True)
        return [word for word, _count in ranked[:max_keywords]]

    def _chunk_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        normalized = re.sub(r"\n{3,}", "\n\n", text)
        chunks: List[str] = []
        start = 0
        length = len(normalized)
        while start < length:
            end = min(length, start + self.chunk_size_chars)
            if end < length:
                pivot = normalized.rfind("\n\n", start, end)
                if pivot > start + 200:
                    end = pivot
            piece = normalized[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= length:
                break
            start = max(start + 1, end - self.chunk_overlap_chars)
        return chunks

    def _upsert_source(
        self,
        source_path: str,
        source_title: str,
        source_type: str,
        trust_score: float,
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        now = time.time()
        metadata = metadata or {}
        trust_score = max(0.0, min(1.0, float(trust_score)))
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO kb_sources (source_path, source_title, source_type, trust_score, metadata_json, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    source_title = excluded.source_title,
                    source_type = excluded.source_type,
                    trust_score = excluded.trust_score,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (source_path, source_title, source_type, trust_score, json.dumps(metadata, ensure_ascii=False), now, now),
            )
            cur.execute("SELECT id FROM kb_sources WHERE source_path = ?", (source_path,))
            row = cur.fetchone()
            self._conn.commit()
        return int(row["id"])

    def _infer_trust_score(self, source_path: str, source_type: str) -> float:
        lowered_path = (source_path or "").lower()
        source_type = (source_type or "").lower()

        if lowered_path.startswith("memory://"):
            base = 0.65
        elif source_type == "pdf":
            base = 0.72
        elif lowered_path.endswith((".md", ".rst")):
            base = 0.68
        elif lowered_path.endswith((".txt", ".json")):
            base = 0.62
        else:
            base = 0.58

        if "knowledge_sources" in lowered_path:
            base += 0.1

        for pattern, score in self.trust_overrides.items():
            if pattern.lower() in lowered_path:
                base = float(score)
                break

        return max(0.0, min(1.0, base))

    def _replace_source_chunks(
        self,
        source_id: int,
        chunks: List[str],
        citations: List[Dict[str, Any]],
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM kb_chunks WHERE source_id = ?", (source_id,))
            now = time.time()
            for idx, chunk in enumerate(chunks):
                keywords = self._extract_keywords(chunk)
                embedding = self._encode(chunk)
                citation = citations[idx] if idx < len(citations) else {}
                cur.execute(
                    """
                    INSERT INTO kb_chunks (source_id, chunk_index, text, keywords_json, citation_json, embedding_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        idx,
                        chunk,
                        json.dumps(keywords, ensure_ascii=False),
                        json.dumps(citation, ensure_ascii=False),
                        json.dumps(embedding) if embedding else None,
                        now,
                    ),
                )
            self._conn.commit()

    def ingest_text(self, source_title: str, text: str, source_path: str = "memory://inline") -> Dict[str, Any]:
        chunks = self._chunk_text(text)
        if not chunks:
            return {"success": False, "source": source_title, "chunks": 0}

        trust = self._infer_trust_score(source_path, "text")
        source_id = self._upsert_source(
            source_path=source_path,
            source_title=source_title,
            source_type="text",
            trust_score=trust,
            metadata={"ocr_used": False},
        )

        citations = [{"source": source_title} for _ in chunks]
        self._replace_source_chunks(source_id, chunks, citations)

        return {"success": True, "source": source_title, "chunks": len(chunks)}

    def ingest_pdf(self, path: str) -> Dict[str, Any]:
        full_path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
        if not os.path.exists(full_path):
            return {"success": False, "source": full_path, "chunks": 0, "reason": "not found"}
        if fitz is None:
            return {"success": False, "source": full_path, "chunks": 0, "reason": "PyMuPDF unavailable"}

        title = os.path.basename(full_path)
        trust = self._infer_trust_score(full_path, "pdf")

        page_chunks: List[str] = []
        citations: List[Dict[str, Any]] = []
        ocr_used = False
        page_count = 0

        try:
            with fitz.open(full_path) as doc:
                page_count = len(doc)
                for idx, page in enumerate(doc):
                    page_number = idx + 1
                    page_text = (page.get_text("text") or "").strip()

                    if len(page_text) < 30 and self.ocr_enabled and pytesseract and Image is not None:
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            image = Image.open(io.BytesIO(pix.tobytes("png")))
                            ocr_text = pytesseract.image_to_string(image)
                            if ocr_text and len(ocr_text.strip()) > len(page_text):
                                page_text = ocr_text.strip()
                                ocr_used = True
                        except Exception:
                            pass

                    if not page_text:
                        continue

                    for chunk in self._chunk_text(page_text):
                        page_chunks.append(chunk)
                        citations.append({"source": title, "page": page_number})
        except Exception as exc:
            return {"success": False, "source": full_path, "chunks": 0, "reason": str(exc)}

        if not page_chunks:
            return {"success": False, "source": full_path, "chunks": 0, "reason": "no extractable text"}

        source_id = self._upsert_source(
            source_path=full_path,
            source_title=title,
            source_type="pdf",
            trust_score=trust,
            metadata={"ocr_used": ocr_used, "pages": page_count},
        )
        self._replace_source_chunks(source_id, page_chunks, citations)
        return {
            "success": True,
            "source": title,
            "chunks": len(page_chunks),
            "pages": page_count,
            "ocr_used": ocr_used,
        }

    def ingest_file(self, path: str) -> Dict[str, Any]:
        full_path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
        if not os.path.exists(full_path):
            return {"success": False, "source": full_path, "chunks": 0, "reason": "not found"}

        ext = os.path.splitext(full_path)[1].lower()
        if ext == ".pdf":
            return self.ingest_pdf(full_path)

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except Exception as exc:
            return {"success": False, "source": full_path, "chunks": 0, "reason": str(exc)}

        title = os.path.basename(full_path)
        return self.ingest_text(source_title=title, text=text, source_path=full_path)

    def ingest_directory(self, root_path: str, max_files: int = 1000, extensions: Iterable[str] | None = None) -> Dict[str, Any]:
        root = os.path.abspath(os.path.expandvars(os.path.expanduser(root_path)))
        if not os.path.isdir(root):
            return {"success": False, "root": root, "files_scanned": 0, "files_ingested": 0, "chunks": 0}

        allowed_exts = set(e.lower() for e in (extensions or self.DEFAULT_EXTENSIONS))
        scanned = 0
        ingested = 0
        total_chunks = 0

        for path in Path(root).rglob("*"):
            if scanned >= max_files:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_exts:
                continue

            scanned += 1
            result = self.ingest_file(str(path))
            if result.get("success"):
                ingested += 1
                total_chunks += int(result.get("chunks", 0))

        return {
            "success": True,
            "root": root,
            "files_scanned": scanned,
            "files_ingested": ingested,
            "chunks": total_chunks,
        }

    @staticmethod
    def _lexical_score(query: str, text: str, keywords: List[str]) -> float:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not q:
            return 0.0
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        keyword_tokens = set(k.lower() for k in keywords)
        base = len(q.intersection(tokens)) / max(1, len(q))
        boost = len(q.intersection(keyword_tokens)) / max(1, len(q))
        return min(1.0, base + (0.35 * boost))

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                  SELECT c.id, c.text, c.keywords_json, c.citation_json, c.embedding_json, c.chunk_index,
                      s.source_title, s.source_path, s.source_type, s.trust_score
                FROM kb_chunks c
                JOIN kb_sources s ON c.source_id = s.id
                ORDER BY c.created_at DESC
                LIMIT 5000
                """
            )
            rows = cur.fetchall()

        query_vec = self._encode(query)
        scored: List[Dict[str, Any]] = []
        for row in rows:
            text = str(row["text"])
            keywords = json.loads(row["keywords_json"] or "[]")
            lexical = self._lexical_score(query, text, keywords)

            semantic = 0.0
            if query_vec and row["embedding_json"]:
                try:
                    candidate = json.loads(row["embedding_json"])
                    semantic = self._cosine(query_vec, candidate)
                except Exception:
                    semantic = 0.0

            base_score = max(lexical, semantic)
            trust_score = float(row["trust_score"] or 0.5)
            score = base_score * (0.65 + (0.35 * trust_score))
            if score <= 0:
                continue

            scored.append(
                {
                    "chunk_id": int(row["id"]),
                    "source_title": str(row["source_title"]),
                    "source_path": str(row["source_path"]),
                    "source_type": str(row["source_type"]),
                    "chunk_index": int(row["chunk_index"]),
                    "text": text,
                    "keywords": keywords,
                    "citation": json.loads(row["citation_json"] or "{}"),
                    "trust_score": trust_score,
                    "score": float(score),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, int(top_k))]

    def build_context(self, query: str, top_k: int = 6, max_chars: int = 3600) -> str:
        return self.build_cited_context(query=query, top_k=top_k, max_chars=max_chars).get("context", "")

    def build_cited_context(self, query: str, top_k: int = 6, max_chars: int = 3600) -> Dict[str, Any]:
        matches = self.search(query=query, top_k=top_k)
        if not matches:
            return {"context": "", "citations": []}

        sections: List[str] = []
        citations: List[Dict[str, Any]] = []
        total_chars = 0
        for idx, item in enumerate(matches, start=1):
            cite_id = f"K{idx}"
            page_info = item.get("citation", {}).get("page")
            page_suffix = f", page {page_info}" if page_info else ""
            piece = (
                f"[{cite_id}] Source: {item['source_title']}{page_suffix}\n"
                f"Trust: {item['trust_score']:.2f}\n"
                f"Relevance: {item['score']:.3f}\n"
                f"Content: {item['text']}"
            )
            if total_chars + len(piece) > max_chars:
                break
            sections.append(piece)
            total_chars += len(piece)
            citations.append(
                {
                    "id": cite_id,
                    "source_title": item["source_title"],
                    "source_path": item["source_path"],
                    "source_type": item["source_type"],
                    "page": page_info,
                    "trust_score": item["trust_score"],
                    "score": item["score"],
                }
            )

        return {"context": "\n\n---\n\n".join(sections), "citations": citations}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) AS c FROM kb_sources")
            source_row = cur.fetchone()
            cur.execute("SELECT COUNT(*) AS c FROM kb_chunks")
            chunk_row = cur.fetchone()
            cur.execute("SELECT AVG(trust_score) AS avg_trust FROM kb_sources")
            trust_row = cur.fetchone()
        return {
            "db_path": self.db_path,
            "sources": int(source_row["c"]) if source_row else 0,
            "chunks": int(chunk_row["c"]) if chunk_row else 0,
            "avg_trust": round(float(trust_row["avg_trust"] or 0.0), 3) if trust_row else 0.0,
            "embeddings_enabled": self._embeddings_enabled,
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                return
