"""Irys RLM - Recursive Language Model system for legal document analysis.

Irys is a sophisticated legal document analysis system that uses recursive
language model techniques to investigate queries against document repositories.

Basic Usage:
    from irys import Irys, IrysConfig

    # Initialize
    irys = Irys(api_key="your-gemini-api-key")

    # Run investigation
    result = await irys.investigate(
        query="What are the contract obligations?",
        repository="./documents",
    )

    print(result.output)

Quick Functions:
    from irys import quick_search, quick_summarize

    # Search documents
    results = await quick_search("damages", "./documents")

    # Summarize files
    summary = await quick_summarize(["doc1.pdf", "doc2.pdf"])

Synchronous Usage:
    from irys import investigate_sync, search_sync

    result = investigate_sync("What happened?", "./documents")
"""

__version__ = "0.1.0"

# Main API
from .api import (
    Irys,
    IrysConfig,
    InvestigationResult,
    batch_investigate,
    quick_search,
    quick_summarize,
    analyze_repository,
    get_irys,
    investigate_sync,
    search_sync,
    health_check,
    get_version,
)

# Core components (for advanced usage)
from .core.repository import MatterRepository
from .core.search import DocumentSearch, SearchHit, SearchResults
from .core.reader import DocumentReader, DocumentContent, OcrCallMetadata

# RLM components (for advanced usage)
from .rlm.state import InvestigationState
from .rlm.engine import RLMEngine, RLMConfig
from .rlm.templates import get_template, suggest_template, get_template_names

# Tracing / observability
from .core.tracing import TracingProvider, NoOpProvider, LangfuseProvider, create_tracing_provider  # noqa: F401

# Output formatters
from .output import get_formatter

__all__ = [
    # Version
    "__version__",
    # Main API
    "Irys",
    "IrysConfig",
    "InvestigationResult",
    "batch_investigate",
    "quick_search",
    "quick_summarize",
    "analyze_repository",
    "get_irys",
    "investigate_sync",
    "search_sync",
    "health_check",
    "get_version",
    # Core
    "MatterRepository",
    "DocumentSearch",
    "SearchHit",
    "SearchResults",
    "DocumentReader",
    "DocumentContent",
    "OcrCallMetadata",
    # RLM
    "InvestigationState",
    "RLMEngine",
    "RLMConfig",
    "get_template",
    "suggest_template",
    "get_template_names",
    # Tracing
    "TracingProvider",
    "NoOpProvider",
    "LangfuseProvider",
    "create_tracing_provider",
    # Output
    "get_formatter",
]
