"""MatterRuntimeAdapter — bridges engine.py to the matter model.

Keeps engine.py on one canonical path while the matter model is
optional (enable_matter_model=False preserves all existing behavior).

When enabled:
- engine reads QueryMatterContext at run start
- extracted facts are converted to AssertionCandidates and upserted
- ledger events are appended in real time
- conflicts trigger BeliefRevision after each write batch
"""

import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .matter import MatterModel
from .enums import (
    LedgerEventType, SpeechAct, SourceRole, ModelLayer,
    AssertionKind, OriginKind, RevisionCause, AssertionLinkType, GapType,
)
from .models import AssertionCandidate, QueryMatterContext


# ---------------------------------------------------------------------------
# Source-role inference from document filename/path
# ---------------------------------------------------------------------------
# Maps filename keyword patterns → SourceRole.
# Listed in priority order — first match wins.

_SOURCE_ROLE_PATTERNS: list[tuple[re.Pattern, SourceRole]] = [
    # Draft: check FIRST — a draft contract is not yet operative
    (re.compile(
        r"(draft|redline|redlined|markup|track.?change)",
        re.IGNORECASE,
    ), SourceRole.DRAFT),
    # Advocacy: party-authored pleadings and argument documents (check BEFORE procedural
    # so that "complaint" and "answer" get advocacy weight, not procedural weight).
    # Use \b word boundaries for terms that are substrings of procedural words
    # (e.g. "position" is a substring of "deposition").
    (re.compile(
        r"(complaint|answer|brief|demand|letter|memo\b|memorandum|\bposition\b|argument|appeal)",
        re.IGNORECASE,
    ), SourceRole.ADVOCACY),
    # Procedural: court filings, discovery documents, neutral court records
    (re.compile(
        r"(motion|petition|pleading|filing|discovery|"
        r"subpoena|deposition|interrogator|exhibit|affidavit)",
        re.IGNORECASE,
    ), SourceRole.PROCEDURAL),
    # Authoritative: statutes, regulations, court orders, judicial decisions.
    # Check BEFORE operative so that "order_approving_settlement_agreement" is
    # classified AUTHORITATIVE (it is a court order) rather than OPERATIVE (settlement).
    # Bare "order" is NOT included because "purchase_order", "change_order", and
    # "work_order" would falsely match. Court orders are identified by compound patterns
    # (court_order, order_approv*, final_order, etc.) or by unambiguous terms
    # (statute, ruling, injunction, etc.). A standalone "order.pdf" resolves as UNKNOWN.
    (re.compile(
        r"(statute|regulation|rule|code|opinion|decision|judgment|judgement|"
        r"mandate|injunction|ruling|decree|"
        r"order[._-]approv\w*|order[._-]grant\w*|order[._-]enter\w*|"
        r"order[._-]den[yi]\w*|order[._-]enjoin\w*|order[._-]dismiss\w*|"
        r"court[._-]order|consent[._-]order|final[._-]order|interim[._-]order|"
        r"preliminary[._-]order|protective[._-]order|restraining[._-]order|"
        r"show[._-]cause)",
        re.IGNORECASE,
    ), SourceRole.AUTHORITATIVE),
    # Operative: contracts, agreements, amendments, leases. Checked after authoritative
    # so that court orders that approve settlements are not misclassified as operative.
    (re.compile(
        r"(contract|agreement|msa|sow|nda|lease|license|amendment|addendum|"
        r"settlement|deed|covenant|warrant|indenture|resolution)",
        re.IGNORECASE,
    ), SourceRole.OPERATIVE),
    # Informal: emails, messages, notes, chats, texts
    (re.compile(
        r"(email|mail|message|note|chat|text|sms|slack|teams|whatsapp|"
        r"thread|correspondence)",
        re.IGNORECASE,
    ), SourceRole.INFORMAL),
    # Post-hoc: explanatory memos, expert reports, declarations written after events
    (re.compile(
        r"(report|expert|declaration|analysis|assessment|audit|review|evaluation)",
        re.IGNORECASE,
    ), SourceRole.POST_HOC_EXPLANATORY),
]


def infer_source_role(document_id: str) -> SourceRole:
    """
    Infer SourceRole from document filename/path keywords.
    Returns SourceRole.UNKNOWN if no pattern matches.
    """
    stem = Path(document_id).name  # filename only, not full path
    for pattern, role in _SOURCE_ROLE_PATTERNS:
        if pattern.search(stem):
            return role
    return SourceRole.UNKNOWN


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
        issue_id: Optional[str] = None,
        issue_link_type: str = "supports",
    ) -> str:
        """
        Convert an extracted fact string into a typed assertion.
        Auto-infers source_role from document_id if not explicitly provided.
        Also auto-infers speech_act from source_role when speech_act is EXTRACTED
        (advocacy → alleged, operative → operative, authoritative → operative).
        If issue_id is provided, links the assertion to that issue with issue_link_type.
        Returns assertion_id.
        """
        if source_role == SourceRole.UNKNOWN:
            source_role = infer_source_role(document_id)

        # Auto-elevate speech_act when the caller left it as the generic EXTRACTED default
        if speech_act == SpeechAct.EXTRACTED:
            if source_role == SourceRole.ADVOCACY:
                speech_act = SpeechAct.ALLEGED
            elif source_role in (SourceRole.OPERATIVE, SourceRole.AUTHORITATIVE):
                speech_act = SpeechAct.OPERATIVE

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

        # Link to issue if a focus issue was specified for this lead (SO-4)
        if issue_id is not None:
            self.model.issues.link_assertion(assertion_id, issue_id, issue_link_type)

        return assertion_id

    def record_assertion_link(
        self,
        src_assertion_id: str,
        dst_assertion_id: str,
        link_type: str,
    ) -> None:
        """Create a directed edge between two assertions in the dependency graph (SO-2).

        Called after deep-read relationship extraction to populate the assertion graph
        from LLM-identified logical relationships within a document.
        """
        try:
            lt = AssertionLinkType(link_type)
        except ValueError:
            return  # unknown link type — skip silently
        try:
            self.model.assertions.link(src_assertion_id, dst_assertion_id, lt)
        except Exception:
            pass  # link building must not block fact recording

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

    def record_gap(
        self,
        description: str,
        gap_type: GapType = GapType.MISSING_DOCUMENT,
        expected_artifact: Optional[str] = None,
        materiality: float = 0.5,
        affected_type: Optional[str] = None,
        affected_id: Optional[str] = None,
    ) -> str:
        """
        Persist a structured gap in the gap store and write a ledger event.
        Returns gap_id.
        """
        gap_id = self.model.gaps.record(
            gap_type=gap_type,
            description=description,
            expected_artifact=expected_artifact,
            materiality=materiality,
            affected_type=affected_type,
            affected_id=affected_id,
        )
        self.log_gap(description, gap_id=gap_id)
        return gap_id

    # ------------------------------------------------------------------
    # User steering (called from API layer)
    # ------------------------------------------------------------------

    def request_redirect(self, issue_id: str) -> None:
        """Signal the engine to redirect focus to the given issue on the next iteration."""
        self.model.ledger.request_redirect(self.run_id, issue_id)
        self.model.ledger.append_event(
            run_id=self.run_id,
            event_type=LedgerEventType.BRANCH_SELECTED,
            summary=f"User requested redirect to issue: {issue_id[:60]}",
            branch_issue_id=issue_id,
        )

    def is_redirect_requested(self) -> bool:
        return self.model.ledger.is_redirect_requested(self.run_id)

    def get_redirect_issue_id(self) -> Optional[str]:
        return self.model.ledger.get_redirect_issue_id(self.run_id)

    def clear_redirect(self) -> None:
        self.model.ledger.clear_redirect(self.run_id)

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

    def record_actor(
        self,
        canonical_name: str,
        actor_type: str = "person",
    ) -> str:
        """
        Persist an actor (person or organization) to the durable actor store.
        Idempotent: same normalized name returns the existing actor_id.
        Returns actor_id.
        """
        actor_id, _ = self.model.actors.upsert_actor(
            canonical_name=canonical_name,
            actor_type=actor_type,
        )
        return actor_id

    def record_quant(
        self,
        quant_kind: str,
        raw_text: str,
        amount_value: Optional[float] = None,
        currency: Optional[str] = None,
        date_value: Optional[str] = None,
        rate_value: Optional[float] = None,
        unit: Optional[str] = None,
        subject_type: Optional[str] = None,
        assertion_id: Optional[str] = None,
    ) -> str:
        """
        Persist a structured numeric fact to the quant store (SO-6).
        Returns quant_fact_id.
        """
        return self.model.quant.record(
            quant_kind=quant_kind,
            raw_text=raw_text,
            amount_value=amount_value,
            currency=currency,
            date_value=date_value,
            rate_value=rate_value,
            unit=unit,
            subject_type=subject_type,
            assertion_id=assertion_id,
        )


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

    def record_gap(self, description: str, **kwargs) -> str:
        return ""

    def request_redirect(self, issue_id: str) -> None:
        pass

    def is_redirect_requested(self) -> bool:
        return False

    def get_redirect_issue_id(self) -> Optional[str]:
        return None

    def clear_redirect(self) -> None:
        pass

    def request_stop(self) -> None:
        pass

    def is_stop_requested(self) -> bool:
        return False

    def record_actor(self, canonical_name: str, actor_type: str = "person") -> str:
        return ""

    def record_quant(self, quant_kind: str, raw_text: str, **kwargs) -> str:
        return ""

    def record_assertion_link(
        self,
        src_assertion_id: str,
        dst_assertion_id: str,
        link_type: str,
    ) -> None:
        pass
