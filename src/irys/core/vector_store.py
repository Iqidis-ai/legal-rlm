"""VectorStore protocol + LocalVectorStore — Phase 1 additive infrastructure.

Two storage layers:
  - FAISS flat index (256-dim) for fast candidate retrieval  (stage 1)
  - SQLite (3072-dim blobs)      for reranking and lookup    (stage 2)

Persisted to .irys/index/{matter_id}/ on disk.
Frozen files (engine.py, decisions.py, reader.py, search.py) have zero
imports from this module.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

import faiss
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol
# =============================================================================

@runtime_checkable
class VectorStore(Protocol):
    """Minimal protocol for a two-stage vector store."""

    def add(self, chunk_id: str, fast_vec: np.ndarray, full_vec: np.ndarray) -> None:
        """Add a chunk's fast and full vectors, keyed by chunk_id."""
        ...

    def search_fast(self, query_vec: np.ndarray, k: int) -> list[str]:
        """Stage 1: return up to k chunk_ids by fast-vector ANN."""
        ...

    def get_full_vector(self, chunk_id: str) -> np.ndarray | None:
        """Stage 2: return stored full vector for reranking."""
        ...

    def persist(self) -> None:
        """Flush all state to disk."""
        ...


# =============================================================================
# Implementation
# =============================================================================

class LocalVectorStore:
    """Two-layer vector store: FAISS (fast) + SQLite (full + metadata).

    Disk layout inside index_dir:
        fast.index      — FAISS IndexFlatL2 (256-dim)
        fast_ids.npy    — chunk_id list aligned with FAISS ordinals
        chunks.db       — SQLite: chunk_id → full_vec blob (3072-dim)

    Cosine similarity is correct for L2 on normalized vectors.
    All vectors must be L2-normalized before being passed to add().
    """

    _FAST_INDEX_FILE = "fast.index"
    _FAST_IDS_FILE = "fast_ids.npy"
    _SQLITE_FILE = "chunks.db"

    def __init__(self, index_dir: Path, fast_dim: int = 256, full_dim: int = 3072):
        self._dir = index_dir
        self._fast_dim = fast_dim
        self._full_dim = full_dim
        index_dir.mkdir(parents=True, exist_ok=True)

        self._faiss: faiss.IndexFlatL2 = faiss.IndexFlatL2(fast_dim)
        self._fast_ids: list[str] = []

        self._db = sqlite3.connect(str(index_dir / self._SQLITE_FILE), check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, chunk_id: str, fast_vec: np.ndarray, full_vec: np.ndarray) -> None:
        """Add normalized fast + full vectors for a chunk."""
        self._faiss.add(fast_vec.reshape(1, -1).astype(np.float32))
        self._fast_ids.append(chunk_id)
        self._db.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, full_vec) VALUES (?, ?)",
            (chunk_id, full_vec.astype(np.float32).tobytes()),
        )
        self._db.commit()

    def search_fast(self, query_vec: np.ndarray, k: int) -> list[str]:
        """Stage 1: ANN search in 256-dim FAISS index, return chunk_ids."""
        if self._faiss.ntotal == 0:
            return []
        k = min(k, self._faiss.ntotal)
        _, indices = self._faiss.search(query_vec.reshape(1, -1).astype(np.float32), k)
        return [self._fast_ids[i] for i in indices[0] if i >= 0]

    def get_full_vector(self, chunk_id: str) -> np.ndarray | None:
        """Stage 2: fetch 3072-dim vector for reranking from SQLite."""
        row = self._db.execute(
            "SELECT full_vec FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def persist(self) -> None:
        """Write FAISS index + id list to disk."""
        faiss.write_index(self._faiss, str(self._dir / self._FAST_INDEX_FILE))
        np.save(str(self._dir / self._FAST_IDS_FILE), np.array(self._fast_ids, dtype=object))
        self._db.commit()
        logger.info("VectorStore persisted to %s (%d vectors)", self._dir, self._faiss.ntotal)

    @classmethod
    def load(cls, index_dir: Path, fast_dim: int = 256, full_dim: int = 3072) -> "LocalVectorStore":
        """Load a persisted LocalVectorStore from disk."""
        store = cls.__new__(cls)
        store._dir = index_dir
        store._fast_dim = fast_dim
        store._full_dim = full_dim

        idx_path = index_dir / cls._FAST_INDEX_FILE
        ids_path = index_dir / cls._FAST_IDS_FILE
        if idx_path.exists() and ids_path.exists():
            store._faiss = faiss.read_index(str(idx_path))
            store._fast_ids = list(np.load(str(ids_path), allow_pickle=True))
        else:
            store._faiss = faiss.IndexFlatL2(fast_dim)
            store._fast_ids = []

        store._db = sqlite3.connect(str(index_dir / cls._SQLITE_FILE), check_same_thread=False)
        store._init_db()
        logger.info("VectorStore loaded from %s (%d vectors)", index_dir, store._faiss.ntotal)
        return store

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(chunk_id TEXT PRIMARY KEY, full_vec BLOB NOT NULL)"
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._faiss.ntotal

    def __repr__(self) -> str:
        return f"LocalVectorStore({self._dir}, {len(self)} vectors)"

