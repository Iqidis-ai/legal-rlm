"""Irys RLM - Recursive Language Model system for legal document analysis."""

__version__ = "0.1.0"

from .api import Irys, IrysConfig, InvestigationResult
from .core.repository import MatterRepository
from .core.search import DocumentSearch, SearchHit, SearchResults
from .core.reader import DocumentReader, DocumentContent
from .rlm.state import InvestigationState
from .rlm.engine import RLMEngine, RLMConfig
from .output import get_formatter

__all__ = [
    "__version__",
    "Irys",
    "IrysConfig",
    "InvestigationResult",
    "MatterRepository",
    "DocumentSearch",
    "SearchHit",
    "SearchResults",
    "DocumentReader",
    "DocumentContent",
    "InvestigationState",
    "RLMEngine",
    "RLMConfig",
    "get_formatter",
]
