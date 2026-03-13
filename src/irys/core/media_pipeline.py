"""Media pipeline — Phase 1 additive infrastructure.

ChunkRecord: the atomic unit of indexed evidence (text or media).
MetadataStore: SQLite CRUD for chunk metadata.
process_* functions: Phase 2 stubs raising NotImplementedError.

Frozen files (engine.py, decisions.py, reader.py, search.py) have zero
imports from this module.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ChunkRecord
# =============================================================================

@dataclass
class ChunkRecord:
    """Atomic unit of indexed evidence.

    A single indexed chunk — a text passage, transcript segment, image region,
    or video scene. Phase 1 only populates text chunks (asset_type="text").

    Fields:
        chunk_id:     UUID string — primary key across all stores.
        asset_path:   Absolute path to the source file.
        asset_type:   "text" | "audio" | "image" | "video".
        chunk_index:  Position of this chunk within the asset (0-based).
        text_content: The text (or transcript excerpt) for this chunk.
        start_char:   Character offset into source text (text chunks).
        end_char:     Character offset end (text chunks).
        start_time_s: Start timestamp in seconds (audio/video chunks).
        end_time_s:   End timestamp in seconds (audio/video chunks).
        page_number:  Page number (PDF/DOCX chunks).
        metadata:     Arbitrary extra metadata as JSON-serializable dict.
        indexed_at:   ISO UTC timestamp when this chunk was indexed.
    """
    chunk_id: str
    asset_path: str
    asset_type: str
    chunk_index: int
    text_content: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    indexed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def from_text(
        cls,
        asset_path: str | Path,
        chunk_index: int,
        text_content: str,
        start_char: int,
        end_char: int,
        page_number: Optional[int] = None,
        **metadata,
    ) -> "ChunkRecord":
        """Convenience constructor for text chunks (Phase 1 primary path)."""
        return cls(
            chunk_id=str(uuid.uuid4()),
            asset_path=str(asset_path),
            asset_type="text",
            chunk_index=chunk_index,
            text_content=text_content,
            start_char=start_char,
            end_char=end_char,
            page_number=page_number,
            metadata=metadata,
        )


# =============================================================================
# MetadataStore
# =============================================================================

_SELECT_COLS = (
    "chunk_id, asset_path, asset_type, chunk_index, page_number, "
    "start_char, end_char, text_content, metadata, indexed_at"
)


def _row_to_record(row: tuple) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=row[0], asset_path=row[1], asset_type=row[2],
        chunk_index=row[3], page_number=row[4],
        start_char=row[5], end_char=row[6], text_content=row[7],
        metadata=json.loads(row[8]), indexed_at=row[9],
    )


class MetadataStore:
    """SQLite CRUD for ChunkRecord metadata.

    Stores per-chunk metadata indexed by chunk_id.
    Full embedding vectors live separately in LocalVectorStore.
    """

    _TABLE_SQL = (
        "CREATE TABLE IF NOT EXISTS chunk_meta ("
        "  chunk_id    TEXT PRIMARY KEY,"
        "  asset_path  TEXT NOT NULL,"
        "  asset_type  TEXT NOT NULL,"
        "  chunk_index INTEGER NOT NULL,"
        "  page_number INTEGER,"
        "  start_char  INTEGER,"
        "  end_char    INTEGER,"
        "  text_content TEXT NOT NULL,"
        "  metadata    TEXT NOT NULL,"
        "  indexed_at  TEXT NOT NULL"
        ")"
    )

    def __init__(self, db_path: Path):
        self._db = sqlite3.connect(str(db_path))
        self._db.execute(self._TABLE_SQL)
        self._db.commit()

    def upsert(self, record: ChunkRecord) -> None:
        """Insert or replace a ChunkRecord."""
        self._db.execute(
            "INSERT OR REPLACE INTO chunk_meta "
            f"({_SELECT_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                record.chunk_id, record.asset_path, record.asset_type,
                record.chunk_index, record.page_number,
                record.start_char, record.end_char, record.text_content,
                json.dumps(record.metadata), record.indexed_at,
            ),
        )
        self._db.commit()

    def get(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Fetch a ChunkRecord by chunk_id, or None if not found."""
        row = self._db.execute(
            f"SELECT {_SELECT_COLS} FROM chunk_meta WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return _row_to_record(row) if row else None

    def get_many(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """Fetch multiple ChunkRecords by chunk_id."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._db.execute(
            f"SELECT {_SELECT_COLS} FROM chunk_meta WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def list_by_asset(self, asset_path: str) -> list[ChunkRecord]:
        """Return all chunks for an asset, ordered by chunk_index."""
        rows = self._db.execute(
            f"SELECT {_SELECT_COLS} FROM chunk_meta "
            "WHERE asset_path = ? ORDER BY chunk_index",
            (asset_path,),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def delete_by_asset(self, asset_path: str) -> int:
        """Delete all chunks for an asset. Returns count deleted."""
        cur = self._db.execute(
            "DELETE FROM chunk_meta WHERE asset_path = ?", (asset_path,)
        )
        self._db.commit()
        return cur.rowcount


# =============================================================================
# Phase 2 stubs — raise NotImplementedError until Phase 2
# =============================================================================

def process_audio(path: Path) -> list[ChunkRecord]:
    """Transcribe and chunk an audio file. Phase 2 stub."""
    raise NotImplementedError(f"process_audio is a Phase 2 feature. Path: {path}")


def process_image(path: Path) -> list[ChunkRecord]:
    """OCR/caption an image file. Phase 2 stub."""
    raise NotImplementedError(f"process_image is a Phase 2 feature. Path: {path}")


def process_video(path: Path) -> list[ChunkRecord]:
    """Transcribe and chunk a video file. Phase 2 stub."""
    raise NotImplementedError(f"process_video is a Phase 2 feature. Path: {path}")


# =============================================================================
# IndexCacheRecord + IndexCache (Phase 1 additive)
# =============================================================================

@dataclass
class IndexCacheRecord:
    """Per-asset indexing state — used to decide whether to re-index.

    Fields:
        asset_id:               Unique identifier for the asset (path as string).
        checksum:               SHA-256 of the file content at last index time.
        embedding_model:        Embedding model used for the last index run.
        dimensionality:         Index dimensionality used for the last run.
        chunk_strategy_version: Version of the chunking strategy used.
        indexed_at:             ISO UTC timestamp of last successful index run.
    """
    asset_id: str
    checksum: str
    embedding_model: str
    dimensionality: int
    chunk_strategy_version: int
    indexed_at: str


class IndexCache:
    """SQLite-backed per-asset index state for re-index decisions.

    Stored in the same .irys/index/{matter_id}/ directory as the vector store.
    A re-index is needed when: file is new, checksum changed, model changed,
    dimensionality changed, or chunk_strategy_version changed.
    """

    _TABLE_SQL = (
        "CREATE TABLE IF NOT EXISTS index_cache ("
        "  asset_id               TEXT PRIMARY KEY,"
        "  checksum               TEXT NOT NULL,"
        "  embedding_model        TEXT NOT NULL,"
        "  dimensionality         INTEGER NOT NULL,"
        "  chunk_strategy_version INTEGER NOT NULL,"
        "  indexed_at             TEXT NOT NULL"
        ")"
    )

    def __init__(self, db_path: Path):
        self._db = sqlite3.connect(str(db_path))
        self._db.execute(self._TABLE_SQL)
        self._db.commit()

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, '_db'):
            self._db.close()

    def _compute_checksum(self, asset_path: Path) -> str:
        """Compute SHA-256 checksum of asset file."""
        import hashlib
        sha256 = hashlib.sha256()
        with open(asset_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def needs_reindex(self, asset: "Asset", config: "EmbeddingConfig") -> bool:
        """Return True if the asset needs (re-)indexing.

        True when: no record exists, checksum changed, model changed,
        dimensionality changed, or chunk_strategy_version changed.

        Args:
            asset: The Asset to check.
            config: The EmbeddingConfig with current settings.

        Returns:
            True if reindexing is needed, False otherwise.
        """
        from .models import EmbeddingConfig
        from .repository import Asset

        asset_id = str(asset.path)
        current_checksum = self._compute_checksum(asset.path)

        row = self._db.execute(
            "SELECT checksum, embedding_model, dimensionality, chunk_strategy_version "
            "FROM index_cache WHERE asset_id = ?",
            (asset_id,),
        ).fetchone()

        if row is None:
            return True

        stored_checksum, stored_model, stored_dim, stored_version = row
        return (
            stored_checksum != current_checksum
            or stored_model != config.model_id
            or stored_dim != config.index_dimensionality
            or stored_version != config.chunk_strategy_version
        )

    def mark_indexed(self, asset: "Asset", config: "EmbeddingConfig") -> None:
        """Mark an asset as indexed with the current configuration.

        Creates or updates the cache record for the asset.

        Args:
            asset: The Asset that was indexed.
            config: The EmbeddingConfig used for indexing.
        """
        from .models import EmbeddingConfig
        from .repository import Asset

        asset_id = str(asset.path)
        checksum = self._compute_checksum(asset.path)
        indexed_at = datetime.utcnow().isoformat()

        self._db.execute(
            "INSERT OR REPLACE INTO index_cache "
            "(asset_id, checksum, embedding_model, dimensionality, chunk_strategy_version, indexed_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                asset_id,
                checksum,
                config.model_id,
                config.index_dimensionality,
                config.chunk_strategy_version,
                indexed_at,
            ),
        )
        self._db.commit()

    def get(self, asset_id: str) -> Optional[IndexCacheRecord]:
        """Fetch an IndexCacheRecord by asset_id, or None if not found.

        Args:
            asset_id: The asset identifier (typically the path as string).

        Returns:
            IndexCacheRecord if found, None otherwise.
        """
        row = self._db.execute(
            "SELECT asset_id, checksum, embedding_model, dimensionality, "
            "chunk_strategy_version, indexed_at "
            "FROM index_cache WHERE asset_id = ?",
            (asset_id,),
        ).fetchone()
        if row is None:
            return None
        return IndexCacheRecord(
            asset_id=row[0],
            checksum=row[1],
            embedding_model=row[2],
            dimensionality=row[3],
            chunk_strategy_version=row[4],
            indexed_at=row[5],
        )

    def evict(self, asset_id: str) -> None:
        """Remove the cache record for an asset (e.g. on file deletion).

        Args:
            asset_id: The asset identifier to remove from cache.
        """
        self._db.execute("DELETE FROM index_cache WHERE asset_id = ?", (asset_id,))
        self._db.commit()
