"""
FAISS-based vector store for entity embeddings with PostgreSQL backend.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..storage.postgres_database import PostgreSQLDatabase

try:
    import faiss
except ImportError:
    faiss = None

from ..config import EMBEDDING_DIMENSION


class VectorStore:
    """FAISS-based vector store for similarity search with PostgreSQL backend."""

    def __init__(self, db: 'PostgreSQLDatabase', matter_id: str, dimension: int = EMBEDDING_DIMENSION):
        """
        Initialize vector store with PostgreSQL backend.

        Args:
            db: PostgreSQL database instance
            matter_id: UUID of the matter
            dimension: Embedding dimension
        """
        self.db = db
        self.matter_id = matter_id
        self.dimension = dimension

        # ID to index mapping
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}

        # Initialize FAISS index
        if faiss is not None:
            self.index = faiss.IndexFlatL2(self.dimension)
            self._load_from_db()
        else:
            self.index = None
            self._fallback_vectors: Dict[str, np.ndarray] = {}
            self._load_fallback_from_db()

    def _load_from_db(self):
        """Load embeddings from PostgreSQL into FAISS index."""
        if self.index is None:
            return

        embeddings = self.db.get_all_embeddings()
        for entity_id, vector_bytes in embeddings:
            # Convert bytes to numpy array
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            if vector.shape[0] == self.dimension:
                idx = self.index.ntotal
                self.index.add(vector.reshape(1, -1))
                self.id_to_idx[entity_id] = idx
                self.idx_to_id[idx] = entity_id

    def _load_fallback_from_db(self):
        """Load embeddings into fallback dictionary (when FAISS not available)."""
        embeddings = self.db.get_all_embeddings()
        for entity_id, vector_bytes in embeddings:
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            if vector.shape[0] == self.dimension:
                self._fallback_vectors[entity_id] = vector

    def add(self, entity_id: str, embedding: np.ndarray):
        """Add an embedding for an entity."""
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {embedding.shape[0]}")

        # Normalize for cosine similarity
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Store in PostgreSQL
        vector_bytes = embedding.tobytes()
        self.db.store_embedding(entity_id, vector_bytes)

        # Add to FAISS index
        if faiss is not None and self.index is not None:
            idx = self.index.ntotal
            self.index.add(embedding.reshape(1, -1))
            self.id_to_idx[entity_id] = idx
            self.idx_to_id[idx] = entity_id
        else:
            # Fallback: store in dictionary
            self._fallback_vectors[entity_id] = embedding

    def add_batch(self, items: list):
        """Add multiple embeddings at once. items = [(entity_id, embedding), ...]

        Uses batch DB storage for dramatically fewer round-trips.
        """
        if not items:
            return

        db_batch = []  # (entity_id, vector_bytes)
        faiss_batch = []  # (entity_id, normalized_embedding)

        for entity_id, embedding in items:
            if embedding.shape[0] != self.dimension:
                continue
            embedding = embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            db_batch.append((entity_id, embedding.tobytes()))
            faiss_batch.append((entity_id, embedding))

        # Batch store in PostgreSQL
        self.db.store_embeddings_batch(db_batch)

        # Add to FAISS index
        if faiss is not None and self.index is not None:
            for entity_id, embedding in faiss_batch:
                idx = self.index.ntotal
                self.index.add(embedding.reshape(1, -1))
                self.id_to_idx[entity_id] = idx
                self.idx_to_id[idx] = entity_id
        else:
            for entity_id, embedding in faiss_batch:
                self._fallback_vectors[entity_id] = embedding

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar entities.

        Returns list of (entity_id, distance) tuples, sorted by similarity.
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {query_embedding.shape[0]}")

        # Normalize query
        query_embedding = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        if faiss is not None and self.index is not None:
            if self.index.ntotal == 0:
                return []

            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1), k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx in self.idx_to_id:
                    # Convert L2 distance to similarity score (lower is better for L2)
                    similarity = 1 / (1 + dist)
                    results.append((self.idx_to_id[idx], similarity))
            return results
        else:
            # Fallback: brute force search
            if not self._fallback_vectors:
                return []

            results = []
            for entity_id, vec in self._fallback_vectors.items():
                # Cosine similarity (vectors are normalized)
                similarity = float(np.dot(query_embedding, vec))
                results.append((entity_id, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

    def remove(self, entity_id: str):
        """Remove an entity's embedding (PostgreSQL handles this via CASCADE)."""
        # FAISS doesn't support efficient deletion
        # The database CASCADE will handle deletion automatically
        if entity_id in self.id_to_idx:
            del self.id_to_idx[entity_id]
        if hasattr(self, '_fallback_vectors') and entity_id in self._fallback_vectors:
            del self._fallback_vectors[entity_id]

    def save(self):
        """Save is handled automatically by PostgreSQL - this is a no-op for compatibility."""
        pass

    def load(self):
        """Reload from PostgreSQL."""
        self.id_to_idx = {}
        self.idx_to_id = {}

        if faiss is not None and self.index is not None:
            # Reset and reload FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self._load_from_db()
        else:
            # Reset and reload fallback
            self._fallback_vectors = {}
            self._load_fallback_from_db()

    def get_count(self) -> int:
        """Get number of stored embeddings."""
        if faiss is not None and self.index is not None:
            return self.index.ntotal
        return len(self._fallback_vectors)

    def has_entity(self, entity_id: str) -> bool:
        """Check if entity has an embedding."""
        return entity_id in self.id_to_idx or (hasattr(self, '_fallback_vectors') and entity_id in self._fallback_vectors)


class EmbeddingGenerator:
    """Generate embeddings using Gemini API (new google-genai SDK)."""

    def __init__(self, api_key: str):
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model_name = "gemini-embedding-001"  # Correct model name for new SDK
        print(
            f"✓ Initialized EmbeddingGenerator with model: {self.model_name}")

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        try:
            # Use new SDK with task type for document embedding
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text[:2048],
                config=self.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMENSION
                )
            )
            # Access embedding from response
            embedding = result.embeddings[0].values
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    def generate_batch(self, texts: List[str], _batch_size: int = 100) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.

        Gemini embed_content API allows at most 100 items per request,
        so we chunk the input automatically.
        """
        all_embeddings: List[np.ndarray] = []
        truncated_texts = [text[:2048] for text in texts]
        for i in range(0, len(truncated_texts), _batch_size):
            batch = truncated_texts[i:i + _batch_size]
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=self.types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=EMBEDDING_DIMENSION
                    )
                )
                all_embeddings.extend(
                    np.array(emb.values, dtype=np.float32)
                    for emb in result.embeddings
                )
            except Exception as e:
                print(
                    f"Error generating batch embeddings (batch {i // _batch_size}): {e}")
                # Fallback to individual generation for this chunk
                all_embeddings.extend(self.generate(text) for text in batch)
        return all_embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query (different task type for retrieval)."""
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=query,
                config=self.types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=EMBEDDING_DIMENSION
                )
            )
            embedding = result.embeddings[0].values
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
