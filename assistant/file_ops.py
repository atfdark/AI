"""
file_ops.py
-----------
Voice-driven file read / search / summarise actions for the assistant.

Supported voice workflows
-------------------------
"read my resume"                → reads first ~300 words of a file aloud
"read the file report.pdf"     → same, explicit filename
"summarise budget.xlsx"        → extracts text, speaks a short summary
"search my files for invoice"  → finds matching filenames, speaks results
"open my downloads folder"     → opens Explorer at a known folder
"what files are in documents"  → lists files in a folder aloud

Supported file types
--------------------
.pdf  — PyMuPDF (fitz) → fallback pdfminer → fallback raw text
.docx — python-docx
.xlsx / .csv — openpyxl / csv module
.txt / .md / .py / .log / … — plain open()
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Optional library guards
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Known folder shortcuts (Windows)
# ---------------------------------------------------------------------------

KNOWN_FOLDERS: dict[str, Path] = {
    "desktop":   Path.home() / "Desktop",
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "pictures":  Path.home() / "Pictures",
    "music":     Path.home() / "Music",
    "videos":    Path.home() / "Videos",
    "onedrive":  Path.home() / "OneDrive",
}

# Common filename aliases — maps spoken names to candidate filenames
FILE_ALIASES: dict[str, list[str]] = {
    "resume":     ["resume.pdf", "resume.docx", "cv.pdf", "cv.docx", "Resume.pdf"],
    "cv":         ["cv.pdf", "cv.docx", "resume.pdf", "Resume.docx"],
    "budget":     ["budget.xlsx", "budget.csv", "Budget.xlsx"],
    "report":     ["report.pdf", "report.docx", "Report.pdf", "report.txt"],
    "notes":      ["notes.txt", "notes.md", "Notes.txt"],
    "readme":     ["README.md", "readme.md", "README.txt"],
}

# Folders to search in (in order)
DEFAULT_SEARCH_PATHS: list[Path] = [
    Path.home() / "Desktop",
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home(),
]

# How many words to speak for a "read" command
READ_WORD_LIMIT = 300

# How many words to speak for a "summarise" command
SUMMARY_WORD_LIMIT = 80


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FileOps:
    """Handles file read / search / summarise actions called by the parser."""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        cfg = self.config.get("file_ops", {})

        extra_paths = [Path(p) for p in cfg.get("extra_search_paths", [])]
        self.search_paths: list[Path] = DEFAULT_SEARCH_PATHS + extra_paths

        self.read_word_limit: int   = cfg.get("read_word_limit",    READ_WORD_LIMIT)
        self.summary_word_limit: int = cfg.get("summary_word_limit", SUMMARY_WORD_LIMIT)

    # ------------------------------------------------------------------
    # Public API (called from parser / actions)
    # ------------------------------------------------------------------

    def read_file(self, name_or_path: str) -> tuple[bool, str]:
        """
        Locate and read a file, returning (success, spoken_text).
        spoken_text is truncated to read_word_limit words.
        """
        path = self._resolve_path(name_or_path)
        if path is None:
            return False, f"I couldn't find a file called {name_or_path}"

        text = self._extract_text(path)
        if not text:
            return False, f"I found {path.name} but couldn't read its contents"

        words = text.split()
        truncated = " ".join(words[:self.read_word_limit])
        suffix = f"… and {len(words) - self.read_word_limit} more words." \
                 if len(words) > self.read_word_limit else ""

        spoken = f"Reading {path.name}. {truncated}{suffix}"
        return True, spoken

    def summarise_file(self, name_or_path: str) -> tuple[bool, str]:
        """
        Locate and summarise a file, returning (success, spoken_text).
        Speaks only the first summary_word_limit words + file stats.
        """
        path = self._resolve_path(name_or_path)
        if path is None:
            return False, f"I couldn't find a file called {name_or_path}"

        text = self._extract_text(path)
        if not text:
            return False, f"I found {path.name} but couldn't extract text from it"

        words = text.split()
        total_words = len(words)
        preview = " ".join(words[:self.summary_word_limit])

        size_kb = path.stat().st_size / 1024
        spoken = (
            f"{path.name} — {total_words} words, {size_kb:.0f} kilobytes. "
            f"It begins: {preview}…"
        )
        return True, spoken

    def search_files(self, query: str, folder: str = "") -> tuple[bool, str]:
        """
        Search for files whose names contain query.
        If folder is given, search there; otherwise search all search_paths.
        Returns (success, spoken_text).
        """
        search_roots: list[Path] = []
        if folder:
            resolved_folder = self._resolve_folder(folder)
            if resolved_folder:
                search_roots = [resolved_folder]
            else:
                return False, f"I don't know a folder called {folder}"
        else:
            search_roots = self.search_paths

        matches: list[Path] = []
        query_lower = query.lower()

        for root in search_roots:
            if not root.exists():
                continue
            try:
                for entry in root.iterdir():
                    if query_lower in entry.name.lower():
                        matches.append(entry)
                        if len(matches) >= 10:  # Cap results
                            break
            except PermissionError:
                continue
            if len(matches) >= 10:
                break

        if not matches:
            return False, f"I didn't find any files matching '{query}'"

        names = [m.name for m in matches[:5]]
        extra = f" and {len(matches) - 5} more" if len(matches) > 5 else ""
        spoken = f"I found {len(matches)} file{'s' if len(matches) != 1 else ''}: " \
                 f"{', '.join(names)}{extra}."
        return True, spoken

    def list_folder(self, folder: str) -> tuple[bool, str]:
        """List files in a known folder and return a spoken summary."""
        path = self._resolve_folder(folder)
        if path is None:
            return False, f"I don't know a folder called {folder}"

        if not path.exists():
            return False, f"The {folder} folder doesn't exist on this computer"

        try:
            entries = list(path.iterdir())
        except PermissionError:
            return False, f"I don't have permission to read the {folder} folder"

        files   = [e for e in entries if e.is_file()]
        folders = [e for e in entries if e.is_dir()]

        if not entries:
            return True, f"The {folder} folder is empty"

        spoken = (
            f"Your {folder} folder has {len(files)} file{'s' if len(files) != 1 else ''} "
            f"and {len(folders)} subfolder{'s' if len(folders) != 1 else ''}. "
        )
        if files:
            sample = [f.name for f in files[:4]]
            spoken += f"Files include: {', '.join(sample)}"
            if len(files) > 4:
                spoken += f" and {len(files) - 4} more"
            spoken += "."
        return True, spoken

    def open_folder(self, folder: str) -> tuple[bool, str]:
        """Open a folder in Windows Explorer."""
        path = self._resolve_folder(folder)
        if path is None:
            # Try as a literal path
            literal = Path(folder)
            if literal.exists() and literal.is_dir():
                path = literal
            else:
                return False, f"I don't know a folder called {folder}"

        try:
            subprocess.Popen(["explorer", str(path)])
            return True, f"Opening your {folder} folder"
        except Exception as e:
            return False, f"I couldn't open the {folder} folder: {e}"

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self, name_or_path: str) -> Optional[Path]:
        """Turn a spoken file name into an absolute Path, or None."""
        # 1. Literal absolute path
        p = Path(name_or_path)
        if p.is_absolute() and p.exists():
            return p

        # 2. Relative path from cwd
        if p.exists():
            return p.resolve()

        # 3. Spoken alias ("my resume", "the budget", …)
        clean = re.sub(r'^(?:my|the|a)\s+', '', name_or_path.strip().lower())
        for alias, candidates in FILE_ALIASES.items():
            if alias in clean:
                for folder in self.search_paths:
                    for candidate in candidates:
                        candidate_path = folder / candidate
                        if candidate_path.exists():
                            return candidate_path

        # 4. Fuzzy filename search across search_paths
        name_lower = name_or_path.lower()
        for folder in self.search_paths:
            if not folder.exists():
                continue
            try:
                for entry in folder.iterdir():
                    if name_lower in entry.name.lower() and entry.is_file():
                        return entry
            except PermissionError:
                continue

        return None

    def _resolve_folder(self, folder: str) -> Optional[Path]:
        """Map a spoken folder name to a Path."""
        key = folder.lower().strip()
        # Remove filler words
        key = re.sub(r'^(?:my|the)\s+', '', key)
        key = key.rstrip(" folder")

        return KNOWN_FOLDERS.get(key)

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, path: Path) -> str:
        """Extract plain text from a file based on its extension."""
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._read_pdf(path)
        elif ext == ".docx":
            return self._read_docx(path)
        elif ext in (".xlsx", ".xls"):
            return self._read_xlsx(path)
        elif ext == ".csv":
            return self._read_csv(path)
        elif ext in (".txt", ".md", ".py", ".js", ".json", ".log",
                     ".yaml", ".yml", ".toml", ".ini", ".cfg", ".rst"):
            return self._read_text(path)
        else:
            # Try plain text as last resort
            return self._read_text(path)

    def _read_pdf(self, path: Path) -> str:
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(str(path))
                parts = []
                for page in doc:
                    parts.append(page.get_text())
                    if len(" ".join(parts).split()) > self.read_word_limit * 2:
                        break  # Enough text
                return " ".join(parts).strip()
            except Exception as e:
                print(f"[FileOps] PyMuPDF failed: {e}")

        # Fallback: pdfminer
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            return pdfminer_extract(str(path)) or ""
        except ImportError:
            pass
        except Exception as e:
            print(f"[FileOps] pdfminer failed: {e}")

        return ""

    def _read_docx(self, path: Path) -> str:
        if not DOCX_AVAILABLE:
            return ""
        try:
            doc = DocxDocument(str(path))
            return " ".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            print(f"[FileOps] python-docx failed: {e}")
            return ""

    def _read_xlsx(self, path: Path) -> str:
        if not OPENPYXL_AVAILABLE:
            return ""
        try:
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            parts = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(str(c) for c in row if c is not None)
                    if row_text.strip():
                        parts.append(row_text)
                    if len(" ".join(parts).split()) > self.read_word_limit * 2:
                        break
                if len(" ".join(parts).split()) > self.read_word_limit * 2:
                    break
            return "\n".join(parts)
        except Exception as e:
            print(f"[FileOps] openpyxl failed: {e}")
            return ""

    def _read_csv(self, path: Path) -> str:
        try:
            rows = []
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    rows.append(" | ".join(row))
                    if i > 50:  # Read at most 50 rows
                        break
            return "\n".join(rows)
        except Exception as e:
            print(f"[FileOps] CSV read failed: {e}")
            return ""

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[FileOps] Text read failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_diagnostics(self) -> dict:
        return {
            "pymupdf_available":  PYMUPDF_AVAILABLE,
            "docx_available":     DOCX_AVAILABLE,
            "openpyxl_available": OPENPYXL_AVAILABLE,
            "search_paths":       [str(p) for p in self.search_paths],
        }
