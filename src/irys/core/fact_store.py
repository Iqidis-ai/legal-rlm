"""Fact Store v2 — SQLite + FTS5 backed persistent storage for extracted facts.

facts.db schema:
  facts          — StoredFact rows with compound scoring fields
  fact_fts       — FTS5 BM25 over fact + source (Porter stemmer)
  source_synopses— one ~300-token synopsis per source document
  synopsis_fts   — FTS5 BM25 over synopsis text (Porter stemmer)
  fact_stubs     — archived facts (importance < 35); BM25-searchable summaries

Storage: {repository_path}/.irys/facts.db
"""

import hashlib
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id              INTEGER PRIMARY KEY,
    fact            TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    page            INTEGER,
    quote           TEXT,
    category        TEXT,
    extracted       TEXT    NOT NULL,
    query_context   TEXT,
    scope_type      TEXT    NOT NULL DEFAULT 'snippet',
    importance      REAL    NOT NULL DEFAULT 50.0,
    recency_updated TEXT    NOT NULL,
    tier            TEXT    NOT NULL DEFAULT 'draft',
    content_hash    TEXT    NOT NULL,
    UNIQUE(content_hash)
);

CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(
    fact, source,
    content      = 'facts',
    content_rowid = 'id',
    tokenize     = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO fact_fts(rowid, fact, source) VALUES (new.id, new.fact, new.source);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO fact_fts(fact_fts, rowid, fact, source)
    VALUES ('delete', old.id, old.fact, old.source);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO fact_fts(fact_fts, rowid, fact, source)
    VALUES ('delete', old.id, old.fact, old.source);
    INSERT INTO fact_fts(rowid, fact, source) VALUES (new.id, new.fact, new.source);
END;

CREATE TABLE IF NOT EXISTS source_synopses (
    id          INTEGER PRIMARY KEY,
    source      TEXT    NOT NULL UNIQUE,
    synopsis    TEXT    NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT    NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS synopsis_fts USING fts5(
    synopsis, source,
    content      = 'source_synopses',
    content_rowid = 'id',
    tokenize     = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS synopses_ai AFTER INSERT ON source_synopses BEGIN
    INSERT INTO synopsis_fts(rowid, synopsis, source) VALUES (new.id, new.synopsis, new.source);
END;
CREATE TRIGGER IF NOT EXISTS synopses_ad AFTER DELETE ON source_synopses BEGIN
    INSERT INTO synopsis_fts(synopsis_fts, rowid, synopsis, source)
    VALUES ('delete', old.id, old.synopsis, old.source);
END;
CREATE TRIGGER IF NOT EXISTS synopses_au AFTER UPDATE OF synopsis ON source_synopses BEGIN
    INSERT INTO synopsis_fts(synopsis_fts, rowid, synopsis, source)
    VALUES ('delete', old.id, old.synopsis, old.source);
    INSERT INTO synopsis_fts(rowid, synopsis, source) VALUES (new.id, new.synopsis, new.source);
END;

CREATE TABLE IF NOT EXISTS fact_stubs (
    content_hash  TEXT PRIMARY KEY,
    stub_summary  TEXT NOT NULL,
    original_fact TEXT NOT NULL,
    archived_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_source      ON facts(source);
CREATE INDEX IF NOT EXISTS idx_facts_tier        ON facts(tier);
CREATE INDEX IF NOT EXISTS idx_facts_importance  ON facts(importance DESC);
CREATE INDEX IF NOT EXISTS idx_facts_recency     ON facts(recency_updated);
"""


@dataclass
class StoredFact:
    """A single fact with source citation and scoring metadata."""
    fact:            str
    source:          str
    page:            Optional[int]   = None
    quote:           Optional[str]   = None
    category:        Optional[str]   = None
    extracted:       str             = ""
    query_context:   Optional[str]   = None
    # v2 fields
    scope_type:      str             = "snippet"
    importance:      float           = 50.0
    recency_updated: str             = ""
    tier:            str             = "draft"
    content_hash:    str             = ""

    def __post_init__(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not self.extracted:
            self.extracted = now
        if not self.recency_updated:
            self.recency_updated = now
        if self.page is not None:
            try:
                self.page = int(self.page)
            except (ValueError, TypeError):
                self.page = None
        if not self.content_hash:
            self.content_hash = StoredFact.compute_hash(self.fact, self.source)

    @staticmethod
    def compute_hash(fact: str, source: str) -> str:
        return hashlib.sha256(f"{fact}\x00{source}".encode()).hexdigest()

    def matches_query(self, query_lower: str) -> bool:
        """Simple keyword matching for relevance filtering."""
        fact_lower = self.fact.lower()
        query_words = [w for w in query_lower.split() if len(w) > 3]
        return any(word in fact_lower for word in query_words)

    def to_json_line(self) -> str:
        """Legacy JSONL serialization shim."""
        return json.dumps({
            "fact": self.fact,
            "source": self.source,
            "page": self.page,
            "quote": self.quote,
            "category": self.category,
            "extracted": self.extracted,
            "query_context": self.query_context,
        }, ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> "StoredFact":
        """Legacy JSONL deserialization shim."""
        data = json.loads(line.strip())
        # Only pass known v1 fields; v2 fields will use defaults
        known = {k: v for k, v in data.items() if k in {
            "fact", "source", "page", "quote", "category", "extracted", "query_context"
        }}
        return cls(**known)


@dataclass
class FactStoreStats:
    total_facts:     int
    core_facts:      int
    validated_facts: int
    draft_facts:     int
    archived_stubs:  int
    avg_importance:  float


def _coerce_page(value) -> Optional[int]:
    """Coerce a page value from LLM output to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_page(page) -> int:
    """Return an int page number for sorting, handling str/None gracefully."""
    if page is None:
        return 0
    try:
        return int(page)
    except (ValueError, TypeError):
        return 0


class FactStore:
    """SQLite + FTS5 backed fact store.

    Storage: {repository_path}/.irys/facts.db

    Legacy compatibility:
      - _loaded and _facts attributes are maintained so existing callers and
        tests that set them directly (store._loaded = True; store._facts = [...])
        continue to work. format_for_llm() will use _facts when populated,
        otherwise falls back to the SQLite store.
    """

    STORE_DIR = ".irys"
    DB_FILE   = "facts.db"

    def __init__(self, repository_path: Path, s3_config: Optional[dict] = None):
        self.repository_path = Path(repository_path)
        self.s3_config = s3_config
        self._s3_client = None
        self.store_dir = self.repository_path / self.STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.store_dir / self.DB_FILE
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        # Legacy compat attributes — used by tests that set _facts directly
        self._loaded: bool = False
        self._facts: list = []

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        # executescript issues an implicit COMMIT and may reset connection-level
        # PRAGMAs, so set them explicitly afterward.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Core persistence
    # ------------------------------------------------------------------

    def load(self) -> int:
        """Backward-compatible shim — v2 initialises in __init__. Returns row count."""
        self._loaded = True
        return len(self)

    def save(self) -> int:
        """Flush any in-memory _facts (legacy shim) to SQLite, then return row count."""
        if self._facts:
            for fact in self._facts:
                self._upsert_fact(fact)
            self._conn.commit()
        return len(self)

    def _upsert_fact(self, fact: StoredFact) -> bool:
        """Insert a fact; silently ignore duplicates (same content_hash). Returns True if inserted."""
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO facts
                   (fact, source, page, quote, category, extracted, query_context,
                    scope_type, importance, recency_updated, tier, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.fact, fact.source, fact.page, fact.quote,
                    fact.category, fact.extracted, fact.query_context,
                    fact.scope_type, fact.importance, fact.recency_updated,
                    fact.tier, fact.content_hash,
                ),
            )
            return self._conn.execute("SELECT changes()").fetchone()[0] == 1
        except sqlite3.Error as exc:
            logger.warning("Failed to upsert fact: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(self, fact: StoredFact) -> bool:
        """Add a fact if not duplicate. Returns True if added, False if duplicate."""
        inserted = self._upsert_fact(fact)
        self._conn.commit()
        return inserted

    def add_facts_from_extraction(
        self,
        extraction: dict,
        source_filename: str,
        query_context: Optional[str] = None,
    ) -> int:
        """Add facts from an extract_facts() result dict.

        Args:
            extraction: Result from decisions.extract_facts()
            source_filename: The document these facts came from
            query_context: The query that led to this extraction

        Returns:
            Number of new facts added
        """
        added = 0

        for fact_text in extraction.get("facts", []):
            if not fact_text or not isinstance(fact_text, str):
                continue
            fact = StoredFact(fact=fact_text, source=source_filename, query_context=query_context)
            if self._upsert_fact(fact):
                added += 1

        for quote in extraction.get("quotes", []):
            if not isinstance(quote, dict):
                continue
            quote_text = quote.get("text", "")
            if not quote_text:
                continue
            relevance = quote.get("relevance", "")
            fact_text = f"{relevance}: \"{quote_text}\"" if relevance else quote_text
            fact = StoredFact(
                fact=fact_text,
                source=source_filename,
                page=_coerce_page(quote.get("page")),
                quote=quote_text,
                query_context=query_context,
            )
            if self._upsert_fact(fact):
                added += 1

        self._conn.commit()
        return added

    def get_source_for_hash(self, content_hash: str) -> Optional[str]:
        """Return source filename for a fact identified by content_hash."""
        row = self._conn.execute(
            "SELECT source FROM facts WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row["source"] if row else None

    def get_all(self) -> list[StoredFact]:
        """Return all facts from SQLite as StoredFact objects."""
        rows = self._conn.execute("SELECT * FROM facts").fetchall()
        return [self._row_to_stored_fact(r) for r in rows]

    def get_relevant(self, query: str, max_facts: int = 50) -> list[StoredFact]:
        """Get all cached facts for context (LLM decides relevance).

        Falls back to _facts if set directly by legacy callers.
        """
        facts = self._facts if self._facts else self.get_all()
        if not facts:
            return []
        facts_sorted = sorted(facts, key=lambda f: (f.source, _safe_page(f.page)))
        return facts_sorted[:max_facts]

    def _row_to_stored_fact(self, row) -> StoredFact:
        return StoredFact(
            fact=row["fact"],
            source=row["source"],
            page=row["page"],
            quote=row["quote"],
            category=row["category"],
            extracted=row["extracted"] or "",
            query_context=row["query_context"],
            scope_type=row["scope_type"],
            importance=row["importance"],
            recency_updated=row["recency_updated"] or "",
            tier=row["tier"],
            content_hash=row["content_hash"],
        )

    def format_for_llm(
        self,
        facts: Optional[list] = None,
        max_chars: int = 15_000,
    ) -> str:
        """Format facts as a string for LLM context.

        Budget is distributed proportionally across sources so that no single
        source monopolises the context window.

        Args:
            facts: Facts to format (if None, uses _facts if populated, else DB).
            max_chars: Total character budget across all sources.

        Returns:
            Formatted fact sheet string.
        """
        if facts is None:
            # Legacy test helpers set _facts directly; respect that first.
            facts = self._facts if self._facts else self.get_all()

        if not facts:
            return ""

        by_source: dict = defaultdict(list)
        for f in facts:
            by_source[f.source].append(f)

        n_sources = len(by_source)
        budget_per_source = max(1_000, max_chars // n_sources)

        lines = ["=== CACHED FACTS FROM PREVIOUS INVESTIGATIONS ===", ""]
        total_chars = len(lines[0])

        for source, source_facts in by_source.items():
            if total_chars >= max_chars:
                break

            lines.append(f"\n[{source}]")
            source_chars = 0

            for idx, fact in enumerate(source_facts):
                page_ref = f" (p.{fact.page})" if fact.page else ""
                fact_line = f"  - {fact.fact}{page_ref}"

                if source_chars + len(fact_line) > budget_per_source:
                    remaining = len(source_facts) - idx
                    lines.append(f"  ... ({remaining} more facts from this source)")
                    break

                if total_chars + len(fact_line) > max_chars:
                    lines.append(f"  ... ({len(source_facts) - idx} more facts truncated)")
                    break

                lines.append(fact_line)
                source_chars += len(fact_line)
                total_chars += len(fact_line)

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get statistics about the fact store."""
        all_facts = self.get_all()
        sources = {f.source for f in all_facts}
        return {
            "total_facts": len(all_facts),
            "unique_sources": len(sources),
            "sources": list(sources),
            "store_path": str(self.store_dir / self.DB_FILE),
            "exists": (self.store_dir / self.DB_FILE).exists(),
        }

    def clear(self):
        """Clear all facts (both in-memory _facts and SQLite)."""
        self._facts = []
        self._conn.execute("DELETE FROM facts")
        self._conn.commit()

    def __bool__(self) -> bool:
        # Always return True so `if fact_store:` checks existence, not emptiness
        return True

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
