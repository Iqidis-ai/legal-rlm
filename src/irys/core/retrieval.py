"""Evidence retrieval — Phase 1 additive infrastructure.

EvidenceCard: a retrieved chunk with its similarity score and source metadata.
EvidenceRetriever: two-stage search (256-dim FAISS fast pass → 3072-dim rerank).

Frozen files (engine.py, decisions.py, reader.py, search.py) have zero
imports from this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .embeddings import EmbeddingClient
from .media_pipeline import ChunkRecord, MetadataStore
from .models import EmbeddingConfig
from .vector_store import LocalVectorStore

logger = logging.getLogger(__name__)


# =============================================================================
# EvidenceCard
# =============================================================================

@dataclass
class EvidenceCard:
    """A single retrieved evidence item with provenance and similarity score.

    Produced by EvidenceRetriever and consumed by the engine seam (Task 1.9).
    One EvidenceCard corresponds to one ChunkRecord, plus retrieval context.

    Fields:
        chunk_id:       UUID string matching the ChunkRecord.
        asset_path:     Absolute path to the source file.
        asset_type:     "text" | "audio" | "image" | "video".
        text_content:   The text (or transcript excerpt) for this chunk.
        page_number:    Page number if applicable.
        start_char:     Character offset into source text.
        end_char:       Character offset end.
        similarity:     Final reranked cosine similarity score [0, 1].
        query:          The query string that produced this card.
        metadata:       Extra metadata from ChunkRecord.
    """
    chunk_id: str
    asset_path: str
    asset_type: str
    text_content: str
    similarity: float
    query: str
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk: ChunkRecord,
        similarity: float,
        query: str,
    ) -> "EvidenceCard":
        """Build an EvidenceCard from a ChunkRecord and a similarity score."""
        return cls(
            chunk_id=chunk.chunk_id,
            asset_path=chunk.asset_path,
            asset_type=chunk.asset_type,
            text_content=chunk.text_content,
            similarity=similarity,
            query=query,
            page_number=chunk.page_number,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            metadata=chunk.metadata,
        )


# =============================================================================
# EvidenceRetriever
# =============================================================================

class EvidenceRetriever:
    """Two-stage vector retrieval over an indexed matter.

    Stage 1 (fast pass): 256-dim FAISS ANN — returns top_k_candidates chunk_ids.
    Stage 2 (rerank):    3072-dim cosine similarity — returns top_k_results cards.

    Only EvidenceCards above similarity_threshold are returned.
    """

    def __init__(
        self,
        vector_store: LocalVectorStore,
        metadata_store: MetadataStore,
        embedding_client: EmbeddingClient,
        config: EmbeddingConfig,
    ):
        self._vs = vector_store
        self._ms = metadata_store
        self._ec = embedding_client
        self._cfg = config

    def search(self, query: str) -> list[EvidenceCard]:
        """Run two-stage retrieval for a query.

        Returns:
            List of EvidenceCards sorted by descending similarity,
            filtered to >= config.similarity_threshold.
        """
        if len(self._vs) == 0:
            logger.debug("EvidenceRetriever: vector store is empty, returning []")
            return []

        # Stage 1: embed query at 256-dim, ANN search
        fast_vec = self._ec.embed_query(
            query, output_dimensionality=self._cfg.fast_dimensionality
        )
        candidate_ids = self._vs.search_fast(fast_vec, k=self._cfg.top_k_candidates)

        if not candidate_ids:
            return []

        # Stage 2: embed query at index_dimensionality, cosine rerank.
        # Stored vectors in SQLite are at index_dimensionality (768), so the
        # query must match that dimension for np.dot to be well-defined.
        full_query_vec = self._ec.embed_query(
            query, output_dimensionality=self._cfg.index_dimensionality
        )
        chunks = self._ms.get_many(candidate_ids)
        chunk_map = {c.chunk_id: c for c in chunks}

        scored: list[tuple[float, ChunkRecord]] = []
        for cid in candidate_ids:
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue
            full_vec = self._vs.get_full_vector(cid)
            if full_vec is None:
                continue
            sim = float(np.dot(full_query_vec, full_vec))
            if sim >= self._cfg.similarity_threshold:
                scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._cfg.top_k_results]

        return [EvidenceCard.from_chunk(chunk, sim, query) for sim, chunk in top]

    def has_media(self) -> bool:
        """Return True if the index contains any chunks (Phase 1: any chunks at all).

        Phase 1: Returns True if vector store has any indexed chunks (including text).
                 Enables testing of multimodal integration seam with text-only data.
        Phase 2: Will query MetadataStore for asset_type != 'text' specifically.
        """
        return len(self._vs) > 0

