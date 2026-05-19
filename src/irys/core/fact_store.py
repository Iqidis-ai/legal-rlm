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
import math
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

    # ── Tier thresholds ──────────────────────────────────────────────────────
    _TIER_PROMOTE = {"draft": 65.0, "validated": 85.0, "core": float("inf")}
    _TIER_DEMOTE  = {"core": 60.0, "validated": 35.0, "draft": 0.0}

    def add_facts_from_extraction(
        self,
        facts: list,
        source: str,
        scope,
        query_context: str = "",
        page: Optional[int] = None,
        category: Optional[str] = None,
    ) -> list:
        """Insert facts and return their content_hash values.

        scope.is_targeted → scope_type "targeted"; else "prefix".
        Duplicate (same content_hash) increments importance by +5.
        """
        scope_type = "targeted" if getattr(scope, "is_targeted", False) else "prefix"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result_hashes: list = []

        for fact_text in facts:
            if not fact_text or not isinstance(fact_text, str):
                continue
            content_hash = StoredFact.compute_hash(fact_text, source)
            try:
                self._conn.execute(
                    """INSERT INTO facts
                       (fact, source, page, category, extracted, query_context,
                        scope_type, importance, recency_updated, tier, content_hash)
                       VALUES (?,?,?,?,?,?,?,50.0,?,'draft',?)""",
                    (fact_text, source, page, category, now, query_context or None,
                     scope_type, now, content_hash),
                )
            except sqlite3.IntegrityError:
                self._conn.execute(
                    """UPDATE facts SET
                           importance      = MIN(importance + 5, 100.0),
                           recency_updated = ?
                       WHERE content_hash  = ?""",
                    (now, content_hash),
                )
                self._check_tier(content_hash)
            result_hashes.append(content_hash)

        self._conn.commit()

        if facts and not self.get_synopsis(source):
            self._build_synopsis(source, facts)

        return result_hashes

    def get_synopsis(self, source: str) -> Optional[str]:
        """Return synopsis text for a source, or None if not built yet."""
        row = self._conn.execute(
            "SELECT synopsis FROM source_synopses WHERE source = ?", (source,)
        ).fetchone()
        return row["synopsis"] if row else None

    def _build_synopsis(self, source: str, initial_facts: list) -> None:
        """Build and store a deterministic synopsis for a source document."""
        sample = initial_facts[:3]
        sample_lines = "\n".join(f"  - {f[:80]}" for f in sample)
        synopsis = f"Source: {source}\nSample facts:\n{sample_lines}"
        token_count = len(synopsis.split())
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._conn.execute(
            """INSERT OR IGNORE INTO source_synopses (source, synopsis, token_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            (source, synopsis, token_count, now),
        )
        self._conn.commit()

    def _check_tier(self, content_hash: str) -> None:
        """Evaluate tier transitions after importance change. Does NOT commit."""
        row = self._conn.execute(
            "SELECT importance, tier FROM facts WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if not row:
            return
        importance, tier = row["importance"], row["tier"]
        new_tier = tier

        if tier == "draft" and importance >= 65.0:
            new_tier = "validated"
        elif tier == "validated" and importance >= 85.0:
            new_tier = "core"
        elif tier == "core" and importance < 60.0:
            new_tier = "validated"
        elif tier == "validated" and importance < 35.0:
            new_tier = "draft"

        if new_tier != tier:
            self._conn.execute(
                "UPDATE facts SET tier = ? WHERE content_hash = ?",
                (new_tier, content_hash),
            )
            logger.debug("Fact %s: %s → %s (importance=%.1f)",
                         content_hash[:8], tier, new_tier, importance)

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

    def get_relevant(self, query: str, top_k: int = 40) -> list[StoredFact]:
        """Return top_k facts ranked by multi-granularity RRF + compound score.

        Three retrieval lanes fused via RRF (k=60):
          q1: BM25 over individual fact rows
          q2: BM25 over source synopses (expands to all facts from high-signal sources)
          q3: Importance sweep (query-independent top facts)
        """
        # Legacy compat: if _facts set directly, fall back to simple path
        if self._facts:
            facts_sorted = sorted(self._facts, key=lambda f: (f.source, _safe_page(f.page)))
            return facts_sorted[:top_k]

        try:
            q1_hits = self._bm25_facts(query, top_k * 3)
            q2_sources = self._bm25_synopses(query, top_m=10)
            q3_hits = self._importance_sweep(top_k)
        except sqlite3.OperationalError as e:
            logger.warning("BM25 query failed (%s), falling back to importance sweep", e)
            q1_hits, q2_sources = [], []
            q3_hits = self._importance_sweep(top_k)

        # q2 expansion: collect all fact IDs from top-M synopsis sources
        q2_fact_ids: set[int] = set()
        if q2_sources:
            placeholders = ",".join("?" * len(q2_sources))
            rows = self._conn.execute(
                f"SELECT id FROM facts WHERE source IN ({placeholders})", q2_sources
            ).fetchall()
            q2_fact_ids = {r[0] for r in rows}

        q1_ranks = {fid: rank for rank, (fid, _) in enumerate(q1_hits)}
        q2_ranks = {fid: i for i, fid in enumerate(q2_fact_ids)}
        q3_ranks = {fid: rank for rank, (fid, _) in enumerate(q3_hits)}

        all_ids = (
            {fid for fid, _ in q1_hits}
            | q2_fact_ids
            | {fid for fid, _ in q3_hits}
        )
        if not all_ids:
            return []

        K = 60
        rrf_scores: dict[int, float] = {}
        for fid in all_ids:
            rrf = 0.0
            if fid in q1_ranks:
                rrf += 0.6 / (K + q1_ranks[fid])
            if fid in q2_ranks:
                rrf += 0.3 / (K + q2_ranks[fid])
            if fid in q3_ranks:
                rrf += 0.1 / (K + q3_ranks[fid])
            rrf_scores[fid] = rrf

        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[: top_k * 2]
        placeholders = ",".join("?" * len(top_ids))
        rows = self._conn.execute(
            f"SELECT * FROM facts WHERE id IN ({placeholders})", top_ids
        ).fetchall()

        q1_score_map = {fid: abs(score) for fid, score in q1_hits}
        max_bm25 = max(q1_score_map.values(), default=1.0) or 1.0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        scored: list[tuple[float, StoredFact]] = []
        for row in rows:
            sf = self._row_to_stored_fact(row)
            bm25_norm = q1_score_map.get(row["id"], 0.0) / max_bm25
            compound = self._compound_score(sf, bm25_norm, today)
            scored.append((compound, sf))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [sf for _, sf in scored[:top_k]]

    def _bm25_facts(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return [(fact_id, bm25_score)] ordered best-first."""
        rows = self._conn.execute(
            """SELECT f.id, bm25(fact_fts) AS score
               FROM fact_fts
               JOIN facts f ON f.id = fact_fts.rowid
               WHERE fact_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (query, top_k),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def _bm25_synopses(self, query: str, top_m: int = 10) -> list[str]:
        """Return source paths whose synopses best match the query."""
        rows = self._conn.execute(
            """SELECT s.source
               FROM synopsis_fts
               JOIN source_synopses s ON s.id = synopsis_fts.rowid
               WHERE synopsis_fts MATCH ?
               ORDER BY bm25(synopsis_fts)
               LIMIT ?""",
            (query, top_m),
        ).fetchall()
        return [r[0] for r in rows]

    def _importance_sweep(self, top_n: int) -> list[tuple[int, float]]:
        """Return top-N facts by raw importance (query-independent lane)."""
        rows = self._conn.execute(
            "SELECT id, importance FROM facts ORDER BY importance DESC LIMIT ?",
            (top_n,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    @staticmethod
    def _compound_score(sf: StoredFact, bm25_norm: float, today: str) -> float:
        scope_bonus = {"targeted": 1.0, "prefix": 0.7, "snippet": 0.4}.get(sf.scope_type, 0.4)
        importance_signal = (sf.importance / 100.0) * scope_bonus
        try:
            days_idle = (
                datetime.fromisoformat(today) - datetime.fromisoformat(sf.recency_updated)
            ).days
        except (ValueError, TypeError):
            days_idle = 0
        recency = math.exp(-max(0, days_idle) / 14.0)
        tier_boost = {"core": 1.15, "validated": 1.08, "draft": 1.0}.get(sf.tier, 1.0)
        return (0.60 * bm25_norm + 0.25 * importance_signal + 0.15 * recency) * tier_boost

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

    def on_search_hit(self, content_hash: str) -> None:
        """Increment importance by +3 when a source is re-encountered in search."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._conn.execute(
            """UPDATE facts SET
                   importance      = MIN(importance + 3, 100.0),
                   recency_updated = ?
               WHERE content_hash  = ?""",
            (now, content_hash),
        )
        self._check_tier(content_hash)
        self._conn.commit()

    def on_re_extraction(self, content_hash: str) -> None:
        """Increment importance by +5 on re-extraction."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._conn.execute(
            """UPDATE facts SET
                   importance      = MIN(importance + 5, 100.0),
                   recency_updated = ?
               WHERE content_hash  = ?""",
            (now, content_hash),
        )
        self._check_tier(content_hash)
        self._conn.commit()

    def tick_decay(self) -> int:
        """Apply idle decay: importance × 0.995^days_idle. Returns rows updated."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self._conn.execute(
            "SELECT content_hash, importance, recency_updated FROM facts"
        ).fetchall()
        updated = 0
        for row in rows:
            try:
                delta = (
                    datetime.fromisoformat(today)
                    - datetime.fromisoformat(row["recency_updated"])
                ).days
            except (ValueError, TypeError):
                continue
            if delta <= 0:
                continue
            new_imp = row["importance"] * (0.995 ** delta)
            self._conn.execute(
                "UPDATE facts SET importance = ? WHERE content_hash = ?",
                (new_imp, row["content_hash"]),
            )
            self._check_tier(row["content_hash"])
            updated += 1
        self._conn.commit()
        return updated

    def archive_cold_facts(self) -> int:
        """Archive draft facts with importance < 35 to fact_stubs. Returns count."""
        cold = self._conn.execute(
            "SELECT content_hash, fact, source FROM facts WHERE tier='draft' AND importance < 35"
        ).fetchall()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for row in cold:
            stub_summary = f"Archived fact from {row['source']}: {row['fact'][:150]}"
            self._conn.execute(
                """INSERT OR REPLACE INTO fact_stubs
                   (content_hash, stub_summary, original_fact, archived_at)
                   VALUES (?, ?, ?, ?)""",
                (row["content_hash"], stub_summary, row["fact"], now),
            )
            self._conn.execute(
                "DELETE FROM facts WHERE content_hash = ?", (row["content_hash"],)
            )
        self._conn.commit()
        return len(cold)

    def pack_evidence(self, query: str, token_budget: int = 32_000) -> str:
        """Retrieve relevant facts and pack into a synthesis-ready bundle."""
        from irys.core.evidence_packer import EvidencePacker
        facts = self.get_relevant(query, top_k=80)
        return EvidencePacker.pack(facts, query=query, token_budget=token_budget)

    def stats(self) -> FactStoreStats:
        """Return current fact counts and averages."""
        row = self._conn.execute("""
            SELECT
                COUNT(*)                                       AS total,
                SUM(CASE WHEN tier='core'      THEN 1 ELSE 0 END) AS core,
                SUM(CASE WHEN tier='validated' THEN 1 ELSE 0 END) AS validated,
                SUM(CASE WHEN tier='draft'     THEN 1 ELSE 0 END) AS draft,
                AVG(importance)                                AS avg_imp
            FROM facts
        """).fetchone()
        stubs = self._conn.execute("SELECT COUNT(*) FROM fact_stubs").fetchone()[0]
        return FactStoreStats(
            total_facts=row[0] or 0,
            core_facts=row[1] or 0,
            validated_facts=row[2] or 0,
            draft_facts=row[3] or 0,
            archived_stubs=stubs,
            avg_importance=round(row[4] or 0.0, 2),
        )

    def migrate_from_jsonl(self, jsonl_path: Path) -> int:
        """Migrate legacy facts.jsonl to facts.db.

        All migrated facts get scope_type='snippet', importance=50.0, tier='draft'.
        Renames jsonl_path -> jsonl_path.bak after migration.
        Returns number of facts migrated.
        """
        if not jsonl_path.exists():
            logger.warning("migrate_from_jsonl: %s does not exist", jsonl_path)
            return 0

        migrated = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("migrate_from_jsonl: skipping malformed line %d: %s", line_num, e)
                    continue
                fact_text = data.get("fact", "")
                source = data.get("source", "")
                if not fact_text or not source:
                    continue
                content_hash = StoredFact.compute_hash(fact_text, source)
                extracted = data.get("extracted") or now
                try:
                    self._conn.execute(
                        """INSERT OR IGNORE INTO facts
                           (fact, source, page, category, extracted, query_context,
                            scope_type, importance, recency_updated, tier, content_hash)
                           VALUES (?,?,?,?,?,?,'snippet',50.0,?,'draft',?)""",
                        (fact_text, source,
                         data.get("page"), data.get("category"),
                         extracted, data.get("query_context"),
                         extracted, content_hash),
                    )
                    migrated += 1
                except sqlite3.Error as e:
                    logger.warning("migrate_from_jsonl: insert failed line %d: %s", line_num, e)

        self._conn.commit()
        bak = jsonl_path.with_suffix(".jsonl.bak")
        jsonl_path.rename(bak)
        logger.info("Migrated %d facts from %s (original -> %s)", migrated, jsonl_path, bak)
        return migrated

    def __bool__(self) -> bool:
        # Always return True so `if fact_store:` checks existence, not emptiness
        return True

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
