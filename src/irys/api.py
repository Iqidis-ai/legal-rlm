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
from .rlm.governance import (
    CascadeGovernor,
    CascadeDecision,
    QueryFamilyHandler,
    QueryFamilyResult,
    ReadFamilyHandler,
    ReadFamilyResult,
    TraceFamilyHandler,
    TraceFamilyResult,
    decision_cache_key,
)
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
        conversation_history: Optional[list[dict[str, str]]] = None,
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

        # MVI-1 Answerability-Governed Cost Cascade — front door.
        # Decide whether to run the full recursive loop at all, answer
        # from existing matter state (`read`), or bounce back a
        # clarification (`clarify`). See src/irys/rlm/governance.py
        # for the cascade design and codex_master_plan.txt for the
        # founder-approved architecture.
        matter_model = self._engine._matter_model
        governor = CascadeGovernor(client=self._client, matter_model=matter_model)
        decision = await governor.decide(
            query=query,
            conversation_history=conversation_history,
        )

        if decision.family == "query":
            # MVI-2: NANO sub-intent resolution then zero-LLM data
            # fetch. Escalates to read (one synth) when the query
            # doesn't map to any enumeration intent.
            query_result = await QueryFamilyHandler(
                matter_model, client=self._client,
            ).run(query=query, contract=decision.contract)
            if not query_result.escalation_needed:
                self._persist_route_decision(
                    matter_model=matter_model,
                    query=query,
                    decision=decision,
                    research_mode=research_mode,
                    terminal_family="query",
                )
                state = self._make_query_state(
                    query=query,
                    repository=repository,
                    research_mode=research_mode,
                    conversation_history=conversation_history,
                    query_result=query_result,
                    decision=decision,
                )
                return InvestigationResult(
                    state=state,
                    output=query_result.rendered_answer,
                    format=self.config.output_format,
                )
            decision.escalation_reason = query_result.escalation_reason
            # Fall through to read — which may itself escalate to
            # investigate if matter coverage is thin.
            decision.family = "read"
            decision.contract = CascadeGovernor._contract_for("read")

        if decision.family == "trace":
            # MVI-2: provenance + reasoning-ledger lookup, zero LLM.
            trace_result = TraceFamilyHandler(matter_model).run(
                query=query, contract=decision.contract,
            )
            self._persist_route_decision(
                matter_model=matter_model,
                query=query,
                decision=decision,
                research_mode=research_mode,
                terminal_family="trace",
            )
            state = self._make_trace_state(
                query=query,
                repository=repository,
                research_mode=research_mode,
                conversation_history=conversation_history,
                trace_result=trace_result,
                decision=decision,
            )
            return InvestigationResult(
                state=state,
                output=trace_result.rendered_answer,
                format=self.config.output_format,
            )

        if decision.family == "read":
            read_result = await self._run_read_family(
                query=query,
                decision=decision,
                conversation_history=conversation_history,
            )
            if not read_result.escalation_needed:
                # Read handler answered. Persist the route for audit.
                self._persist_route_decision(
                    matter_model=matter_model,
                    query=query,
                    decision=decision,
                    research_mode=research_mode,
                    terminal_family="read",
                )
                state = self._make_read_state(
                    query=query,
                    repository=repository,
                    research_mode=research_mode,
                    conversation_history=conversation_history,
                    read_result=read_result,
                    decision=decision,
                )
                formatter = get_formatter(self.config.output_format)
                return InvestigationResult(
                    state=state,
                    output=read_result.answer or formatter.format(state),
                    format=self.config.output_format,
                )
            # Coverage insufficient — fall through to investigate,
            # recording the escalation reason so the classifier can be
            # tuned against real escalation data.
            decision.escalation_reason = read_result.escalation_reason
            logger.info(
                "Read family escalated to investigate: %s",
                read_result.escalation_reason,
            )

        if decision.family == "clarify":
            # MVI-1: clarify returns the question back to the user as
            # the answer text. No investigation, no read, no further
            # LLM spend. Later MVIs can generate a structured
            # clarification prompt; for now the rationale is the
            # question.
            self._persist_route_decision(
                matter_model=matter_model,
                query=query,
                decision=decision,
                research_mode=research_mode,
                terminal_family="clarify",
            )
            state = self._make_clarify_state(
                query=query,
                repository=repository,
                research_mode=research_mode,
                conversation_history=conversation_history,
                decision=decision,
            )
            clarification_text = (
                f"Need clarification before we can answer: {decision.rationale}"
            )
            return InvestigationResult(
                state=state,
                output=clarification_text,
                format=self.config.output_format,
            )

        # Full investigate path — unchanged from the pre-cascade flow.
        self._telemetry.start_operation("investigation")
        usage_before = self._client.snapshot_usage()
        try:
            state = await self._engine.investigate(
                query,
                repository,
                research_mode=research_mode,
                conversation_history=conversation_history,
            )
        finally:
            self._telemetry.end_operation(
                "investigation",
                "investigate_complete",
                {"query_length": len(query)},
            )
        self._attach_usage_summary(state, usage_before)

        # Persist the decision AFTER the run so we have the run_id.
        self._persist_route_decision(
            matter_model=matter_model,
            query=query,
            decision=decision,
            research_mode=research_mode,
            terminal_family="investigate",
            run_id=getattr(state, "_run_id", None),
        )

        # Format output
        formatter = get_formatter(self.config.output_format)
        output = formatter.format(state)

        return InvestigationResult(
            state=state,
            output=output,
            format=self.config.output_format,
        )

    async def _run_read_family(
        self,
        query: str,
        decision: CascadeDecision,
        conversation_history: Optional[list[dict[str, str]]],
    ) -> ReadFamilyResult:
        """One synth call over existing matter state. MVI-1."""
        handler = ReadFamilyHandler(
            client=self._client,
            matter_model=self._engine._matter_model,
        )
        self._telemetry.start_operation("read_family")
        try:
            return await handler.run(
                query=query,
                contract=decision.contract,
                conversation_history=conversation_history,
            )
        finally:
            self._telemetry.end_operation(
                "read_family",
                "read_complete",
                {"query_length": len(query)},
            )

    def _make_read_state(
        self,
        query: str,
        repository: "str | Path",
        research_mode: Optional[str],
        conversation_history: Optional[list[dict[str, str]]],
        read_result: ReadFamilyResult,
        decision: CascadeDecision,
    ) -> InvestigationState:
        """Build a minimal InvestigationState for a read-family answer
        so downstream formatters / telemetry can treat it uniformly."""
        state = InvestigationState.create(
            query,
            str(Path(repository).resolve()),
            research_mode=research_mode,
            conversation_history=conversation_history,
        )
        state.findings["final_output"] = read_result.answer
        state.findings["route"] = decision.to_audit_dict()
        state.findings["read_confidence"] = read_result.confidence_label
        # Attach citations as document-anchored entries so existing
        # citation consumers have something to render.
        for doc in read_result.citations:
            try:
                state.add_citation(
                    document=doc,
                    page=None,
                    text="",
                    context="Cited by read handler from existing matter state",
                    relevance="supporting",
                )
            except Exception:
                pass
        return state

    def _make_query_state(
        self,
        query: str,
        repository: "str | Path",
        research_mode: Optional[str],
        conversation_history: Optional[list[dict[str, str]]],
        query_result: QueryFamilyResult,
        decision: CascadeDecision,
    ) -> InvestigationState:
        state = InvestigationState.create(
            query,
            str(Path(repository).resolve()),
            research_mode=research_mode,
            conversation_history=conversation_history,
        )
        state.findings["final_output"] = query_result.rendered_answer
        state.findings["route"] = decision.to_audit_dict()
        state.findings["query_intent"] = query_result.intent
        state.findings["query_row_count"] = len(query_result.rows)
        return state

    def _make_trace_state(
        self,
        query: str,
        repository: "str | Path",
        research_mode: Optional[str],
        conversation_history: Optional[list[dict[str, str]]],
        trace_result: TraceFamilyResult,
        decision: CascadeDecision,
    ) -> InvestigationState:
        state = InvestigationState.create(
            query,
            str(Path(repository).resolve()),
            research_mode=research_mode,
            conversation_history=conversation_history,
        )
        state.findings["final_output"] = trace_result.rendered_answer
        state.findings["route"] = decision.to_audit_dict()
        state.findings["trace_target_kind"] = trace_result.target_kind
        state.findings["trace_target_id"] = trace_result.target_id
        return state

    def _make_clarify_state(
        self,
        query: str,
        repository: "str | Path",
        research_mode: Optional[str],
        conversation_history: Optional[list[dict[str, str]]],
        decision: CascadeDecision,
    ) -> InvestigationState:
        state = InvestigationState.create(
            query,
            str(Path(repository).resolve()),
            research_mode=research_mode,
            conversation_history=conversation_history,
        )
        state.findings["final_output"] = (
            f"Need clarification before we can answer: {decision.rationale}"
        )
        state.findings["route"] = decision.to_audit_dict()
        return state

    def _persist_route_decision(
        self,
        matter_model: Any,
        query: str,
        decision: CascadeDecision,
        research_mode: Optional[str],
        terminal_family: str,
        run_id: Optional[str] = None,
    ) -> None:
        """Write the route decision to run_session + ledger_event so
        the front door is auditable. Per Codex master plan acceptance
        criteria: if it's not queryable, it's not tunable."""
        if matter_model is None:
            return
        try:
            if run_id is None:
                # For non-investigate terminal families we still open a
                # run so there's a record of the decision.
                run_id = matter_model.start_run(
                    query=query,
                    objective=f"cascade:{terminal_family}",
                    operation_type=terminal_family,
                    trigger="user",
                    research_mode=research_mode or "deep",
                )
                try:
                    matter_model.complete_run(run_id)
                except Exception:
                    pass
            # Write a structured ledger event capturing the decision.
            try:
                import json as _json
                from .matter.enums import LedgerEventType
                matter_model.ledger.append_event(
                    run_id=run_id,
                    event_type=LedgerEventType.ROUTE_DECISION,
                    summary=(
                        f"Route: {terminal_family} "
                        f"(classifier family={decision.family}, "
                        f"conf={decision.confidence:.2f})"
                    ),
                    why=decision.rationale,
                    snapshot_json=_json.dumps(decision.to_audit_dict()),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Route ledger write failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Route persistence failed: %s", exc)

    async def resume_investigation(
        self,
        checkpoint_path: "str | Path",
        original_run_id: "str | None" = None,
        follow_up_query: "str | None" = None,
        research_mode: "str | None" = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
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
                conversation_history=conversation_history,
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
