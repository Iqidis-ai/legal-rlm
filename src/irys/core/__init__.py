"""Core components for document processing and model access."""

from .models import ModelTier, GeminiClient
from .repository import MatterRepository
from .reader import DocumentReader
from .search import DocumentSearch
from .fact_store import FactStore, StoredFact

__all__ = [
    "ModelTier",
    "GeminiClient",
    "MatterRepository",
    "DocumentReader",
    "DocumentSearch",
    "FactStore",
    "StoredFact",
]
