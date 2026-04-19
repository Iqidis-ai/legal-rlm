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
    CompareFamilyHandler,
    CompareFamilyResult,
    DeliverableFamilyHandler,
    DeliverableFamilyResult,
    QueryFamilyHandler,
    QueryFamilyResult,
    ReadFamilyHandler,
    ReadFamilyResult,
    ScenarioFamilyHandler,
    ScenarioFamilyResult,
    SteerFamilyHandler,
    SteerFamilyResult,
    TraceFamilyHandler,
    TraceFamilyResult,
    decision_cache_key,
)
from .rlm.state import InvestigationState, StepType, normalize_research_mode
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

        # Adversarial #10 Fix C: never mutate decision.family in
        # place — it's the classifier's original call and must be
        # preserved for audit. `active_family` is what dispatch
        # switches on; when a handler escalates, we reassign
        # active_family (not decision.family). The ledger writer
        # records both.
        classifier_family = decision.family
        active_family = classifier_family
        active_contract = decision.contract

        if active_family == "query":
            # MVI-2: NANO sub-intent resolution then zero-LLM data
            # fetch. Escalates to read (one synth) when the query
            # doesn't map to any enumeration intent.
            query_result = await QueryFamilyHandler(
                matter_model, client=self._client,
            ).run(query=query, contract=active_contract)
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
            active_family = "read"
            active_contract = CascadeGovernor._contract_for("read")

        if active_family == "deliverable":
            # MVI-7: named work-product renderer. MVI-7 ships only the
            # privilege-log renderer; other sub-intents escalate to
            # read for a narrative response.
            deliverable_result = await DeliverableFamilyHandler(
                matter_model, client=self._client,
            ).run(query=query, contract=active_contract)
            if not deliverable_result.escalation_needed:
                self._persist_route_decision(
                    matter_model=matter_model,
                    query=query,
                    decision=decision,
                    research_mode=research_mode,
                    terminal_family="deliverable",
                )
                state = self._make_simple_state(
                    query=query,
                    repository=repository,
                    research_mode=research_mode,
                    conversation_history=conversation_history,
                    output=deliverable_result.rendered_answer,
                    decision=decision,
                    extra={
                        "deliverable_intent": deliverable_result.intent,
                        "deliverable_row_count": deliverable_result.row_count,
                    },
                    terminal_family="deliverable",
                )
                return InvestigationResult(
                    state=state,
                    output=deliverable_result.rendered_answer,
                    format=self.config.output_format,
                )
            decision.escalation_reason = deliverable_result.escalation_reason
            active_family = "read"
            active_contract = CascadeGovernor._contract_for("read")

        if active_family == "compare":
            # MVI-6: diff current state vs last completed run.
            compare_result = CompareFamilyHandler(matter_model).run(
                query=query, contract=active_contract,
            )
            self._persist_route_decision(
                matter_model=matter_model,
                query=query,
                decision=decision,
                research_mode=research_mode,
                terminal_family="compare",
            )
            state = self._make_simple_state(
                query=query,
                repository=repository,
                research_mode=research_mode,
                conversation_history=conversation_history,
                output=compare_result.rendered_answer,
                decision=decision,
                extra={
                    "compare_baseline_run_id": compare_result.baseline_run_id,
                    "compare_assertion_delta": (
                        compare_result.current_assertion_count
                        - compare_result.baseline_assertion_count
                    ),
                },
                terminal_family="compare",
            )
            return InvestigationResult(
                state=state,
                output=compare_result.rendered_answer,
                format=self.config.output_format,
            )

        if active_family == "scenario":
            # MVI-6: NANO-parsed assumption, then read-family answer
            # with the assumption injected as a temporary override.
            # No state mutation.
            scenario_result = await ScenarioFamilyHandler(
                client=self._client, matter_model=matter_model,
            ).run(
                query=query,
                contract=active_contract,
                conversation_history=conversation_history,
            )
            if not scenario_result.escalation_needed:
                self._persist_route_decision(
                    matter_model=matter_model,
                    query=query,
                    decision=decision,
                    research_mode=research_mode,
                    terminal_family="scenario",
                )
                state = self._make_simple_state(
                    query=query,
                    repository=repository,
                    research_mode=research_mode,
                    conversation_history=conversation_history,
                    output=scenario_result.answer,
                    decision=decision,
                    extra={
                        "scenario_assumption": scenario_result.assumption,
                        "scenario_confidence": scenario_result.confidence_label,
                    },
                    terminal_family="scenario",
                )
                return InvestigationResult(
                    state=state,
                    output=scenario_result.answer,
                    format=self.config.output_format,
                )
            decision.escalation_reason = scenario_result.escalation_reason
            active_family = "read"
            active_contract = CascadeGovernor._contract_for("read")

        if active_family == "steer":
            # MVI-4: NANO-parsed correction / mutation. Returns a
            # preview the user confirms via existing UI — no
            # auto-apply. Escalates to read when the intent can't be
            # parsed clearly.
            steer_result = await SteerFamilyHandler(
                matter_model, client=self._client,
            ).run(query=query, contract=active_contract)
            if not steer_result.escalation_needed:
                self._persist_route_decision(
                    matter_model=matter_model,
                    query=query,
                    decision=decision,
                    research_mode=research_mode,
                    terminal_family="steer",
                )
                state = self._make_steer_state(
                    query=query,
                    repository=repository,
                    research_mode=research_mode,
                    conversation_history=conversation_history,
                    steer_result=steer_result,
                    decision=decision,
                )
                return InvestigationResult(
                    state=state,
                    output=steer_result.rendered_answer,
                    format=self.config.output_format,
                )
            decision.escalation_reason = steer_result.escalation_reason
            active_family = "read"
            active_contract = CascadeGovernor._contract_for("read")

        if active_family == "trace":
            # MVI-2: provenance + reasoning-ledger lookup, zero LLM.
            trace_result = TraceFamilyHandler(matter_model).run(
                query=query, contract=active_contract,
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

        if active_family == "read":
            read_result = await self._run_read_family(
                query=query,
                decision=decision,
                conversation_history=conversation_history,
                contract=active_contract,
            )
            # Adversarial #10 finding #6: distinguish "state
            # insufficient" (escalate to investigate) from infra
            # failure (surface an error — do NOT silently run the
            # full AR loop during an outage).
            if read_result.failure_kind == "infra":
                self._persist_route_decision(
                    matter_model=matter_model,
                    query=query,
                    decision=decision,
                    research_mode=research_mode,
                    terminal_family="read_infra_failure",
                )
                state = self._make_simple_state(
                    query=query,
                    repository=repository,
                    research_mode=research_mode,
                    conversation_history=conversation_history,
                    output=(
                        "⚠️ Read handler couldn't reach the LLM service. "
                        f"Reason: {read_result.escalation_reason}. "
                        "Retry when the provider is available, or rerun "
                        "with an explicit investigate request if you "
                        "want the full pipeline to run."
                    ),
                    decision=decision,
                    extra={"read_infra_failure": True},
                    terminal_family="read_infra_failure",
                )
                return InvestigationResult(
                    state=state,
                    output=state.findings["final_output"],
                    format=self.config.output_format,
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
            # State insufficient — fall through to investigate,
            # recording the escalation reason so the classifier can be
            # tuned against real escalation data.
            decision.escalation_reason = read_result.escalation_reason
            logger.info(
                "Read family escalated to investigate (state insufficient): %s",
                read_result.escalation_reason,
            )

        if active_family == "clarify":
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

        # Full investigate path — cascade-aware. MVI-3: we thread the
        # ExecutionContract through so the engine's new termination
        # controller reads family-scoped stop rules instead of the
        # legacy confidence/count heuristics.
        self._telemetry.start_operation("investigation")
        usage_before = self._client.snapshot_usage()
        try:
            state = await self._engine.investigate(
                query,
                repository,
                research_mode=research_mode,
                conversation_history=conversation_history,
                execution_contract=active_contract,
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
        # adv#12 Finding #2: also attach the route audit to
        # state.findings so _extract_cascade_surface (and therefore
        # JobResult.route / SyncInvestigateResponse.route) exposes it
        # to clients. The full investigate path was the only terminal
        # family that wrote to the ledger without setting findings,
        # which left the API response with route=None for investigate.
        try:
            state.findings["route"] = decision.to_audit_dict(
                terminal_family="investigate",
            )
        except Exception as _exc:
            logger.warning("investigate: route findings attach failed: %s", _exc)

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
        contract: Any = None,
    ) -> ReadFamilyResult:
        """One synth call over existing matter state. MVI-1.
        `contract` is the active contract at call time — may differ
        from `decision.contract` when an earlier handler escalated."""
        handler = ReadFamilyHandler(
            client=self._client,
            matter_model=self._engine._matter_model,
        )
        self._telemetry.start_operation("read_family")
        try:
            return await handler.run(
                query=query,
                contract=contract or decision.contract,
                conversation_history=conversation_history,
            )
        finally:
            self._telemetry.end_operation(
                "read_family",
                "read_complete",
                {"query_length": len(query)},
            )

    # Attorney-facing labels for the cascade families. Mirrors the UI
    # chip; kept here so the reasoning-trace prose matches the chip
    # and never leaks an internal token like "read".
    _ROUTE_PROSE = {
        "investigate": "Deep investigation",
        "read": "Quick summary",
        "query": "Direct lookup",
        "trace": "Reasoning trace",
        "steer": "Correction preview",
        "compare": "Change comparison",
        "scenario": "What-if analysis",
        "deliverable": "Document draft",
        "clarify": "Clarification request",
    }

    def _seed_cheap_path_trace(
        self,
        state: InvestigationState,
        decision: CascadeDecision,
        terminal_family: str,
        extra: Optional[list[tuple[StepType, str]]] = None,
    ) -> None:
        """Populate state.thinking_steps for a cheap-path family (read,
        query, trace, steer, compare, scenario, deliverable, clarify)
        so the UI's Reasoning Trace tab isn't blank on these flows.

        Without this, a user running "what do we know" sees an empty
        trace — the full AR loop is the only path that streams
        thinking_steps. This puts three short attorney-facing lines
        on state.thinking_steps regardless of path:
          1. what the classifier decided (and why)
          2. the source of the answer (existing matter state vs.
             fresh investigation)
          3. any family-specific detail the caller passes via `extra`

        Reasoning-trace prose never uses internal family tokens
        ('read' / 'investigate'); it uses the same attorney labels
        the UI chip displays.
        """
        try:
            term_label = self._ROUTE_PROSE.get(
                terminal_family, terminal_family.title(),
            )
            cls = str(decision.family or "").strip()
            rationale = (decision.rationale or "").strip()
            if cls and cls != terminal_family:
                cls_label = self._ROUTE_PROSE.get(cls, cls.title())
                route_line = (
                    f"Routed to {term_label} "
                    f"(escalated from {cls_label})"
                )
            else:
                route_line = f"Routed to {term_label}"
            if rationale:
                route_line += f". Reason: {rationale}"
            state.add_step(StepType.THINKING, route_line)
            state.add_step(
                StepType.THINKING,
                "Answered from existing matter state (no new documents read).",
            )
            if extra:
                for step_type, content in extra:
                    state.add_step(step_type, content)
        except Exception as _exc:
            logger.debug("_seed_cheap_path_trace failed: %s", _exc)

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
        # adv#12 Finding #2: emit classifier_family + terminal_family
        # on every route audit dict so clients can differentiate the
        # NANO choice from the actual terminal behavior after any
        # escalation. Read handler is always the terminal_family here.
        state.findings["route"] = decision.to_audit_dict(terminal_family="read")
        state.findings["read_confidence"] = read_result.confidence_label
        # Populate thinking_steps so the UI reasoning-trace tab shows
        # transparency on the cheap path. See _seed_cheap_path_trace.
        _cit_count = len(read_result.citations or [])
        self._seed_cheap_path_trace(
            state, decision, "read",
            extra=[(
                StepType.SYNTHESIS,
                f"Answer confidence: {read_result.confidence_label} "
                f"({_cit_count} citation{'s' if _cit_count != 1 else ''}).",
            )],
        )
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
        state.status = "completed"
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
        state.findings["route"] = decision.to_audit_dict(terminal_family="query")
        state.findings["query_intent"] = query_result.intent
        state.findings["query_row_count"] = len(query_result.rows)
        _row_count = len(query_result.rows)
        self._seed_cheap_path_trace(
            state, decision, "query",
            extra=[(
                StepType.FINDING,
                f"Query intent: {query_result.intent}. "
                f"Returned {_row_count} matching record"
                f"{'s' if _row_count != 1 else ''}.",
            )],
        )
        state.status = "completed"
        return state

    def _make_simple_state(
        self,
        query: str,
        repository: "str | Path",
        research_mode: Optional[str],
        conversation_history: Optional[list[dict[str, str]]],
        output: str,
        decision: CascadeDecision,
        extra: Optional[dict] = None,
        terminal_family: Optional[str] = None,
    ) -> InvestigationState:
        """Shared builder for zero/single-LLM family results
        (compare, scenario) that don't need a rich state object.

        `terminal_family` overrides the route dict's terminal field
        when the caller is a family that escalated from the
        classifier's initial pick (e.g. deliverable that may fall
        back to read).
        """
        state = InvestigationState.create(
            query,
            str(Path(repository).resolve()),
            research_mode=research_mode,
            conversation_history=conversation_history,
        )
        state.findings["final_output"] = output
        state.findings["route"] = decision.to_audit_dict(terminal_family=terminal_family)
        if extra:
            state.findings.update(extra)
        # Seed a minimal reasoning trace so the UI tab isn't blank on
        # compare / scenario / deliverable / read_infra_failure paths.
        self._seed_cheap_path_trace(
            state, decision,
            terminal_family or str(decision.family or ""),
        )
        state.status = "completed"
        return state

    def _make_steer_state(
        self,
        query: str,
        repository: "str | Path",
        research_mode: Optional[str],
        conversation_history: Optional[list[dict[str, str]]],
        steer_result: SteerFamilyResult,
        decision: CascadeDecision,
    ) -> InvestigationState:
        state = InvestigationState.create(
            query,
            str(Path(repository).resolve()),
            research_mode=research_mode,
            conversation_history=conversation_history,
        )
        state.findings["final_output"] = steer_result.rendered_answer
        state.findings["route"] = decision.to_audit_dict(terminal_family="steer")
        _cand_count = len(steer_result.candidates or [])
        self._seed_cheap_path_trace(
            state, decision, "steer",
            extra=[(
                StepType.THINKING,
                f"Preview only — showing {_cand_count} candidate"
                f"{'s' if _cand_count != 1 else ''}; nothing has "
                "been applied.",
            )],
        )
        state.findings["steer_action"] = steer_result.action
        state.findings["steer_target_hint"] = steer_result.target_hint
        state.findings["steer_candidates"] = steer_result.candidates
        state.status = "completed"
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
        state.findings["route"] = decision.to_audit_dict(terminal_family="trace")
        self._seed_cheap_path_trace(
            state, decision, "trace",
            extra=[(
                StepType.THINKING,
                f"Traced {trace_result.target_kind} — no LLM synthesis, "
                "reading directly from the reasoning ledger.",
            )],
        )
        state.findings["trace_target_kind"] = trace_result.target_kind
        state.findings["trace_target_id"] = trace_result.target_id
        state.status = "completed"
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
        state.findings["route"] = decision.to_audit_dict(terminal_family="clarify")
        self._seed_cheap_path_trace(state, decision, "clarify")
        state.status = "completed"
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
            # Adversarial #10 Fix C: write BOTH the classifier's
            # original family AND the terminal family that actually
            # handled the request. `decision.family` is pristine
            # (never mutated since the governor returned it), so the
            # audit snapshot carries the classifier's real call.
            try:
                import json as _json
                from .matter.enums import LedgerEventType
                audit_payload = decision.to_audit_dict()
                audit_payload["terminal_family"] = terminal_family
                # Codex fallout R3: on stale-cache fallback the
                # decision.family is a REUSED prior route, not a
                # fresh classifier emission. Label the audit
                # accordingly so ledger queries can distinguish
                # "classifier routed X" from "classifier failed,
                # reused cached X". classifier_version carries the
                # sentinel "_stale_cache_fallback" from the governor.
                is_stale_fallback = (
                    decision.classifier_version == "_stale_cache_fallback"
                )
                audit_payload["classifier_family"] = (
                    "_stale_cache_fallback" if is_stale_fallback
                    else decision.family
                )
                if is_stale_fallback:
                    audit_payload["reused_route"] = decision.family
                classifier_label = (
                    "_stale_cache_fallback" if is_stale_fallback
                    else decision.family
                )
                matter_model.ledger.append_event(
                    run_id=run_id,
                    event_type=LedgerEventType.ROUTE_DECISION,
                    summary=(
                        f"Route: classifier={classifier_label} "
                        f"terminal={terminal_family} "
                        f"conf={decision.confidence:.2f}"
                    ),
                    why=decision.rationale,
                    snapshot_json=_json.dumps(audit_payload),
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
