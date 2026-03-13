"""Multimodal embedding client — Phase 1 additive infrastructure.

Wraps gemini-embedding-2-preview via Vertex AI.
Two-stage retrieval: fast_dimensionality (256) for candidate pass,
full_dimensionality (3072) for local reranking.

Frozen files (engine.py, decisions.py, reader.py, search.py) have zero
imports from this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

from .models import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when an embedding API call fails."""


class EmbeddingClient:
    """Wraps gemini-embedding-2-preview via Vertex AI.

    Responsibilities:
    - embed_text: call the Vertex AI embedding API, return a normalized vector
    - embed_query: same but with RETRIEVAL_QUERY task type
    - embed_media: Phase 1 stub — raises NotImplementedError until Phase 2
    - _normalize: L2-normalizes all vectors (required for sub-3072-dim MRL)
    """

    def __init__(self, config: EmbeddingConfig, project: str, region: str | None = None):
        self._config = config
        self._project = project
        self._region = region or config.region
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=self._region,
        )
        logger.info(
            "EmbeddingClient initialized — model=%s project=%s region=%s",
            config.model_id,
            project,
            self._region,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_text(
        self,
        text: str,
        *,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
    ) -> np.ndarray:
        """Embed a text string and return a normalized float32 vector.

        Args:
            text: Input text to embed.
            task_type: Override task type. Defaults to config.index_task_type.
            output_dimensionality: Override output dims. Defaults to
                config.index_dimensionality.

        Returns:
            Normalized float32 array of shape (output_dimensionality,).

        Raises:
            EmbeddingError: on API failure.
        """
        task = task_type or self._config.index_task_type
        dims = output_dimensionality or self._config.index_dimensionality

        try:
            response = self._client.models.embed_content(
                model=self._config.model_id,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task,
                    output_dimensionality=dims,
                ),
            )
        except Exception as exc:
            raise EmbeddingError(f"embed_text API call failed: {exc}") from exc

        raw = response.embeddings[0].values
        vector = np.array(raw, dtype=np.float32)
        return self._normalize(vector)

    def embed_query(self, query: str, output_dimensionality: int | None = None) -> np.ndarray:
        """Embed a retrieval query using RETRIEVAL_QUERY task type.

        Convenience wrapper around embed_text with the query task type.

        Args:
            query: Query string to embed.
            output_dimensionality: Override output dims. Defaults to
                config.index_dimensionality.

        Returns:
            Normalized float32 array.
        """
        return self.embed_text(
            query,
            task_type=self._config.query_task_type,
            output_dimensionality=output_dimensionality,
        )

    def embed_media(self, path: Path) -> np.ndarray:  # noqa: ARG002
        """Embed a media file (audio, video, image).

        Phase 1 stub — raises NotImplementedError until Phase 2.
        This is intentional and part of the Phase 1 acceptance criteria.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"embed_media is a Phase 2 feature. Cannot embed media file: {path}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalize a vector and return it.

        Gemini sub-3072-dim truncated MRL vectors are not normalized by
        default. This must be applied before any cosine similarity comparison.
        """
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm

