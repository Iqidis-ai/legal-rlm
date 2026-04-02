"""MatterRuntimeAdapter — bridges engine.py to the matter model.

Keeps engine.py on one canonical path while the matter model is
optional (enable_matter_model=False preserves all existing behavior).

When enabled:
- engine reads QueryMatterContext at run start
- extracted facts are converted to AssertionCandidates and upserted
- ledger events are appended in real time
- conflicts trigger BeliefRevision after each write batch
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .matter import MatterModel
from .enums import (
    LedgerEventType, SpeechAct, SourceRole, ModelLayer,
    AssertionKind, OriginKind, RevisionCause, AssertionLinkType,
)
from .models import AssertionCandidate, QueryMatterContext


class MatterRuntimeAdapter:
    """
    Thin adapter used by RLMEngine when enable_matter_model=True.

    The engine creates one adapter per investigation run and calls it
    at key points in the pipeline. When not enabled, all methods are
    no-ops and the engine behaves exactly as before.
    """

    def __init__(
        self,
        matter_model: MatterModel,
        run_id: str,
    ):
        self.model = matter_model
        self.run_id = run_id
        self._pending_assertion_ids: list[str] = []

    # ------------------------------------------------------------------
    # Called from engine._orient()
    # ------------------------------------------------------------------

    def get_context(self) -> QueryMatterContext:
        """Return current matter context for injection into orientation prompt."""
        return self.model.build_query_context()

    def log_objective(self, objective: str) -> None:
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.OBJECTIVE_SET,
            summary=f"Objective: {objective[:200]}",
        )

    # ------------------------------------------------------------------
    # Called from engine._analyze_search_results() / _deep_read_document()
    # ------------------------------------------------------------------

    def record_fact(
        self,
        proposition_text: str,
        document_id: str,
        source_role: SourceRole = SourceRole.UNKNOWN,
        speech_act: SpeechAct = SpeechAct.EXTRACTED,
        model_layer: ModelLayer = ModelLayer.RECORD,
        assertion_kind: AssertionKind = AssertionKind.FACTUAL,
        span_id: Optional[str] = None,
    ) -> str:
        """
        Convert an extracted fact string into a typed assertion.
        Returns assertion_id.
        """
        candidate = AssertionCandidate(
            proposition_text=proposition_text,
            model_layer=model_layer,
            assertion_kind=assertion_kind,
            document_id=document_id,
            span_id=span_id,
            source_role=source_role,
            speech_act=speech_act,
            origin_kind=OriginKind.EXTRACTED,
        )
        assertion_id, is_new = self.model.record_assertion(candidate)
        self._pending_assertion_ids.append(assertion_id)

        if is_new:
            self.model.ledger.append_event(
                run_id=self.run_id,
                event_type=LedgerEventType.ASSERTION_ADDED,
                summary=f"New assertion: {proposition_text[:120]}",
                changed_object_type="assertion",
                changed_object_id=assertion_id,
            )
        return assertion_id

    def flush_revisions(self) -> int:
        """
        Trigger belief revision for all assertions added since last flush.
        Returns count of revised assertions.
        """
        if not self._pending_assertion_ids:
            return 0
        results = self.model.apply_revision(
            seed_assertion_ids=self._pending_assertion_ids,
            cause=RevisionCause.NEW_EVIDENCE,
            run_id=self.run_id,
        )
        for result in results:
            if result.old_belief_state != result.new_belief_state:
                self.model.ledger.append_event(
                    run_id=self.run_id,
                    event_type=LedgerEventType.ASSERTION_REVISED,
                    summary=(
                        f"Belief revised: {result.old_belief_state.value} → "
                        f"{result.new_belief_state.value}"
                    ),
                    changed_object_type="assertion",
                    changed_object_id=result.assertion_id,
                )
        self._pending_assertion_ids.clear()
        return len(results)

    # ------------------------------------------------------------------
    # Called from engine._emit_step() / general progress
    # ------------------------------------------------------------------

    def log_step(self, summary: str, why: Optional[str] = None) -> None:
        """Append a generic step to the reasoning ledger."""
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.BRANCH_SELECTED,
            summary=summary[:500],
            why=why,
        )

    def log_conflict(self, summary: str) -> None:
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.CONFLICT_DETECTED,
            summary=summary[:500],
        )

    def log_gap(self, summary: str, gap_id: Optional[str] = None) -> None:
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.GAP_IDENTIFIED,
            summary=summary[:500],
            changed_object_type="gap" if gap_id else None,
            changed_object_id=gap_id,
        )

    # ------------------------------------------------------------------
    # User steering (called from API layer)
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Signal the engine to stop after the current iteration."""
        self.model.ledger.request_stop(self.run_id)
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.USER_INTERRUPTED,
            summary="User requested stop",
        )

    def is_stop_requested(self) -> bool:
        return self.model.ledger.is_stop_requested(self.run_id)


class NullMatterAdapter:
    """
    No-op adapter used when enable_matter_model=False.

    All methods return safe defaults so engine.py can call the adapter
    unconditionally regardless of whether the matter model is enabled.
    """

    def get_context(self) -> Optional[QueryMatterContext]:
        return None

    def log_objective(self, objective: str) -> None:
        pass

    def record_fact(self, proposition_text: str, document_id: str, **kwargs) -> str:
        return ""

    def flush_revisions(self) -> int:
        return 0

    def log_step(self, summary: str, why: Optional[str] = None) -> None:
        pass

    def log_conflict(self, summary: str) -> None:
        pass

    def log_gap(self, summary: str, gap_id: Optional[str] = None) -> None:
        pass

    def request_stop(self) -> None:
        pass

    def is_stop_requested(self) -> bool:
        return False
