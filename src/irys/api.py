"""High-level API for the Irys RLM system."""

from dataclasses import dataclass
from typing import Optional, Any, Callable
from pathlib import Path
import logging

from .core.models import GeminiClient
from .core.repository import MatterRepository
from .core.utils import (
    setup_logging,
    TelemetryCollector,
    validate_query,
    validate_file_path,
)
from .rlm.engine import RLMEngine, RLMConfig
from .rlm.state import InvestigationState, normalize_research_mode
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
        self._telemetry = TelemetryCollector()
        self._matter_models: dict[str, Any] = {}  # repo_path → MatterModel

        # Callbacks
        self._on_progress: Optional[Callable] = None
        self._on_step: Optional[Callable] = None

    def _ensure_initialized(self):
        """Ensure components are initialized."""
        if self._client is None:
            self._client = GeminiClient(api_key=self.config.api_key)

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

    def _attach_usage_summary(
        self,
        state: InvestigationState,
        usage_before: dict,
    ) -> None:
        """Attach Gemini token/cost deltas to the state and run_session."""
        if self._client is None:
            return
        usage = self._client.get_usage_delta(usage_before)
        state.llm_usage = usage
        run_id = getattr(state, "_run_id", None)
        if run_id and self._engine and self._engine._matter_model is not None:
            try:
                self._engine._matter_model.record_run_usage_summary(run_id, usage)
            except Exception:
                logger.debug("Could not persist run usage summary for run %s", run_id)

    async def investigate(
        self,
        query: str,
        repository: str | Path,
        research_mode: "str | None" = None,
    ) -> "InvestigationResult":
        """
        Run an investigation.

        Args:
            query: The legal question to investigate
            repository: Path to document repository

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
        if research_mode is not None:
            research_mode = normalize_research_mode(research_mode, strict=True)

        # Wire matter model for this repository (SO-1: durable per-repo store)
        if self.config.enable_matter_model:
            repo_key = str(Path(repository).resolve())
            if repo_key not in self._matter_models:
                from .matter import MatterModel
                self._matter_models[repo_key] = MatterModel.open(repo_key)
            self._engine._matter_model = self._matter_models[repo_key]

        # Run investigation
        self._telemetry.start_operation("investigation")
        usage_before = self._client.snapshot_usage()
        try:
            state = await self._engine.investigate(
                query,
                repository,
                research_mode=research_mode,
            )
        finally:
            self._telemetry.end_operation(
                "investigation",
                "investigate_complete",
                {"query_length": len(query)},
            )
        self._attach_usage_summary(state, usage_before)

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
        follow_up_query: "str | None" = None,
        research_mode: "str | None" = None,
    ) -> "InvestigationResult":
        """Resume a stopped investigation from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file (from run_session.next_action)
            original_run_id: The interrupted run_session.id; if it has a pending redirect,
                the redirect is propagated to the new resumed run (SO-3 stop→redirect→resume).
            follow_up_query: Optional new user query to continue from the saved state
                with a refined objective.

        Returns:
            InvestigationResult with findings and output from the resumed run
        """
        self._ensure_initialized()
        if research_mode is not None:
            research_mode = normalize_research_mode(research_mode, strict=True)

        self._telemetry.start_operation("resume_investigation")
        usage_before = self._client.snapshot_usage()
        try:
            state = await self._engine.resume_investigation(
                checkpoint_path,
                original_run_id=original_run_id,
                follow_up_query=follow_up_query,
                research_mode=research_mode,
            )
        finally:
            self._telemetry.end_operation(
                "resume_investigation",
                "resume_complete",
                {},
            )
        self._attach_usage_summary(state, usage_before)

        formatter = get_formatter(self.config.output_format)
        output = formatter.format(state)
        return InvestigationResult(state=state, output=output, format=self.config.output_format)

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


# =============================================================================
# Investigation Result
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


__version__ = "0.1.0"

__all__ = [
    "Irys",
    "IrysConfig",
    "InvestigationResult",
    "__version__",
]
