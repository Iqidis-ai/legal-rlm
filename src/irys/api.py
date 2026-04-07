"""High-level API for the Irys RLM system.

Units 41-50: Integration, API, and final polish.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from pathlib import Path
import asyncio
import logging

from .core.models import GeminiClient, ModelTier
from .core.repository import MatterRepository
from .core.cache import ResponseCache, LRUCache
from .core.utils import (
    setup_logging,
    TelemetryCollector,
    ProgressTracker,
    SystemConfig,
    validate_query,
    validate_file_path,
)
from .rlm.engine import RLMEngine, RLMConfig
from .rlm.state import InvestigationState
from .rlm.templates import get_template, suggest_template, get_template_names
from .output import get_formatter

logger = logging.getLogger("irys")


# =============================================================================
# Unit 41: High-Level API
# =============================================================================

@dataclass
class IrysConfig:
    """Configuration for the Irys system."""
    api_key: Optional[str] = None
    max_depth: int = 5
    max_leads_per_level: int = 5
    checkpoint_dir: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    output_format: str = "markdown"
    log_level: str = "INFO"
    enable_matter_model: bool = True  # Persist intelligence to durable SQLite store (default on)


class Irys:
    """
    High-level API for Irys legal document analysis.

    Example usage:
        irys = Irys(api_key="your-api-key")
        result = await irys.investigate(
            query="What are the key contract obligations?",
            repository="./documents",
        )
        print(result.output)
    """

    def __init__(self, config: Optional[IrysConfig] = None, **kwargs):
        """
        Initialize Irys.

        Args:
            config: IrysConfig object or individual parameters as kwargs
        """
        if config:
            self.config = config
        else:
            self.config = IrysConfig(**kwargs)

        # Setup logging
        setup_logging(level=self.config.log_level)

        # Initialize components
        self._client: Optional[GeminiClient] = None
        self._engine: Optional[RLMEngine] = None
        self._cache: Optional[ResponseCache] = None
        self._telemetry = TelemetryCollector()
        self._matter_models: dict[str, Any] = {}  # repo_path → MatterModel

        # Callbacks
        self._on_progress: Optional[Callable] = None
        self._on_step: Optional[Callable] = None

    def _ensure_initialized(self):
        """Ensure components are initialized."""
        if self._client is None:
            self._client = GeminiClient(api_key=self.config.api_key)

        if self._cache is None and self.config.cache_enabled:
            self._cache = ResponseCache(
                ttl_seconds=self.config.cache_ttl_seconds)

        if self._engine is None:
            engine_config = RLMConfig(
                max_depth=self.config.max_depth,
                max_leads_per_level=self.config.max_leads_per_level,
                checkpoint_dir=self.config.checkpoint_dir,
                enable_matter_model=self.config.enable_matter_model,
            )
            self._engine = RLMEngine(
                gemini_client=self._client,
                config=engine_config,
                on_step=self._on_step,
                on_progress=self._on_progress,
            )

    def on_progress(self, callback: Callable[[dict], None]):
        """Register progress callback."""
        self._on_progress = callback
        if self._engine:
            self._engine.on_progress = callback

    def on_step(self, callback: Callable):
        """Register step callback."""
        self._on_step = callback
        if self._engine:
            self._engine.on_step = callback

    async def investigate(
        self,
        query: str,
        repository: str | Path,
        template: Optional[str] = None,
    ) -> "InvestigationResult":
        """
        Run an investigation.

        Args:
            query: The legal question to investigate
            repository: Path to document repository
            template: Optional investigation template name

        Returns:
            InvestigationResult with findings and output
        """
        # Validate inputs
        valid, issues = validate_query(query)
        if not valid:
            raise ValueError(f"Invalid query: {', '.join(issues)}")

        valid, issues = validate_file_path(str(repository))
        if not valid:
            raise ValueError(f"Invalid repository: {', '.join(issues)}")

        self._ensure_initialized()

        # Wire matter model for this repository (SO-1: durable per-repo store)
        if self.config.enable_matter_model:
            repo_key = str(Path(repository).resolve())
            if repo_key not in self._matter_models:
                from .matter import MatterModel
                self._matter_models[repo_key] = MatterModel.open(repo_key)
            self._engine._matter_model = self._matter_models[repo_key]

        # Apply template if specified
        if template:
            tmpl = get_template(template)
            if tmpl:
                # Could enhance query with template guidance
                logger.info(f"Using template: {tmpl.name}")

        # Run investigation
        self._telemetry.start_operation("investigation")
        try:
            state = await self._engine.investigate(query, repository)
        finally:
            self._telemetry.end_operation(
                "investigation",
                "investigate_complete",
                {"query_length": len(query)},
            )

        # Format output
        formatter = get_formatter(self.config.output_format)
        output = formatter.format(state)

        return InvestigationResult(
            state=state,
            output=output,
            format=self.config.output_format,
        )

    async def resume_investigation(
        self,
        checkpoint_path: "str | Path",
        original_run_id: "str | None" = None,
    ) -> "InvestigationResult":
        """Resume a stopped investigation from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (from run_session.next_action)
            original_run_id: The interrupted run_session.id; if it has a pending redirect,
                the redirect is propagated to the new resumed run (SO-3 stop→redirect→resume).

        Returns:
            InvestigationResult with findings and output from the resumed run
        """
        self._ensure_initialized()

        self._telemetry.start_operation("resume_investigation")
        try:
            state = await self._engine.resume_investigation(
                checkpoint_path, original_run_id=original_run_id
            )
        finally:
            self._telemetry.end_operation(
                "resume_investigation",
                "resume_complete",
                {},
            )

        formatter = get_formatter(self.config.output_format)
        output = formatter.format(state)
        return InvestigationResult(state=state, output=output, format=self.config.output_format)

    async def summarize(
        self,
        files: list[str | Path],
        repository: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """
        Summarize documents.

        Args:
            files: List of file paths to summarize
            repository: Optional repository for context

        Returns:
            Dict with individual and collection summaries
        """
        self._ensure_initialized()

        file_paths = [Path(f) for f in files]
        repo = MatterRepository(
            repository or file_paths[0].parent) if repository or file_paths else None

        return await self._engine.summarize_documents(file_paths, repo)

    async def search(
        self,
        query: str,
        repository: str | Path,
        regex: bool = False,
    ) -> list[dict]:
        """
        Search documents.

        Args:
            query: Search term
            repository: Path to repository
            regex: Whether to use regex matching

        Returns:
            List of search hits
        """
        repo = MatterRepository(repository)
        files = list(repo.list_files())
        results = repo.search(query, regex=regex)

        return [
            {
                "file": hit.filename,
                "page": hit.page_num,
                "text": hit.match_text,
                "context": hit.context,
            }
            for hit in results.top(20)
        ]

    def suggest_template(self, query: str) -> Optional[str]:
        """Suggest an investigation template for a query."""
        return suggest_template(query)

    def list_templates(self) -> list[str]:
        """List available investigation templates."""
        return get_template_names()

    def get_telemetry(self) -> dict[str, Any]:
        """Get telemetry summary."""
        return self._telemetry.get_summary()


# =============================================================================
# Unit 42: Investigation Result
# =============================================================================

@dataclass
class InvestigationResult:
    """Result of an investigation."""
    state: InvestigationState
    output: str
    format: str

    @property
    def query(self) -> str:
        return self.state.query

    @property
    def status(self) -> str:
        return self.state.status

    @property
    def success(self) -> bool:
        return self.state.status == "completed"

    @property
    def citations(self) -> list:
        return self.state.citations

    @property
    def entities(self) -> dict:
        return self.state.entities

    @property
    def confidence(self) -> dict:
        return self.state.get_confidence_score()

    @property
    def quality(self):
        return self.state.assess_answer_quality()

    def to_format(self, format_type: str) -> str:
        """Convert to different output format."""
        formatter = get_formatter(format_type)
        return formatter.format(self.state)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "status": self.status,
            "confidence": self.confidence,
            "output": self.output,
            "metrics": self.state.get_progress(),
        }


# =============================================================================
# Unit 43: Batch Investigation
# =============================================================================

async def batch_investigate(
    queries: list[str],
    repository: str | Path,
    config: Optional[IrysConfig] = None,
    parallel: bool = True,
) -> list[InvestigationResult]:
    """
    Run multiple investigations.

    Args:
        queries: List of queries
        repository: Document repository path
        config: Configuration
        parallel: Whether to run in parallel

    Returns:
        List of InvestigationResult objects
    """
    irys = Irys(config=config)

    if parallel:
        tasks = [irys.investigate(q, repository) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, InvestigationResult)]
    else:
        results = []
        for query in queries:
            try:
                result = await irys.investigate(query, repository)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Investigation failed for query '{query[:50]}...': {e}")
        return results


# =============================================================================
# Unit 44: Quick Functions
# =============================================================================

async def quick_search(
    query: str,
    repository: str | Path,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Quick search function."""
    irys = Irys(api_key=api_key)
    return await irys.search(query, repository)


async def quick_summarize(
    files: list[str | Path],
    api_key: Optional[str] = None,
) -> dict:
    """Quick summarize function."""
    irys = Irys(api_key=api_key)
    return await irys.summarize(files)


# =============================================================================
# Unit 45: Repository Analysis
# =============================================================================

async def analyze_repository(
    repository: str | Path,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    Analyze a document repository.

    Returns overview, statistics, and document types.
    """
    repo = MatterRepository(repository)
    stats = repo.get_stats()
    structure = repo.get_structure()

    # Categorize documents
    from .core.clustering import cluster_by_document_type
    files = repo.list_files()
    documents = [f.path for f in files]
    categories = cluster_by_document_type([str(d) for d in documents])

    return {
        "path": str(repository),
        "stats": {
            "total_files": stats.total_files,
            "total_size_mb": round(stats.total_size_bytes / (1024 * 1024), 2),
            "file_types": stats.files_by_type,
        },
        "structure": structure,
        "categories": {k: len(v) for k, v in categories.items()},
        "sample_files": [str(d) for d in documents[:10]],
    }


# =============================================================================
# Unit 46-50: Convenience and Polish
# =============================================================================

# Unit 46: Default instance
_default_irys: Optional[Irys] = None


def get_irys(api_key: Optional[str] = None) -> Irys:
    """Get or create default Irys instance."""
    global _default_irys
    if _default_irys is None:
        _default_irys = Irys(api_key=api_key)
    return _default_irys


# Unit 47: Sync wrappers for async functions
def investigate_sync(
    query: str,
    repository: str | Path,
    api_key: Optional[str] = None,
) -> InvestigationResult:
    """Synchronous wrapper for investigate."""
    irys = Irys(api_key=api_key)
    return asyncio.run(irys.investigate(query, repository))


def search_sync(
    query: str,
    repository: str | Path,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Synchronous wrapper for search."""
    irys = Irys(api_key=api_key)
    return asyncio.run(irys.search(query, repository))


# Unit 48: Version info
__version__ = "0.1.0"


def get_version() -> str:
    """Get Irys version."""
    return __version__


# Unit 49: Health check
async def health_check(api_key: Optional[str] = None) -> dict[str, Any]:
    """
    Check system health.

    Returns status of all components.
    """
    status = {
        "version": __version__,
        "components": {},
    }

    # Check Gemini client
    try:
        client = GeminiClient(api_key=api_key)
        # Could make a test call here
        status["components"]["gemini"] = "ok"
    except Exception as e:
        status["components"]["gemini"] = f"error: {e}"

    return status


# Unit 50: Export all public APIs
__all__ = [
    # Main classes
    "Irys",
    "IrysConfig",
    "InvestigationResult",
    # Functions
    "batch_investigate",
    "quick_search",
    "quick_summarize",
    "analyze_repository",
    "get_irys",
    "investigate_sync",
    "search_sync",
    "health_check",
    "get_version",
    # Version
    "__version__",
]
