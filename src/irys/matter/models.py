"""Command and result dataclasses for the matter model layer."""

from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime, timezone

from .enums import (
    SpeechAct, SourceRole, BeliefState, ModelLayer,
    AssertionKind, AssertionLinkType, RevisionCause,
    GapType, OriginKind, LedgerEventType,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    import uuid
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Assertion inputs / outputs
# ---------------------------------------------------------------------------

@dataclass
class AssertionCandidate:
    """
    An assertion extracted from a document occurrence, ready to be upserted
    into the assertion store.

    The proposition_key is a normalized hash of the canonical proposition text.
    The same proposition from two documents yields ONE assertion row and TWO
    assertion_occurrence rows — this is the deduplication invariant.
    """
    proposition_text: str
    model_layer: ModelLayer = ModelLayer.RECORD
    assertion_kind: AssertionKind = AssertionKind.FACTUAL

    # Source of this specific occurrence
    document_id: str = ""
    span_id: Optional[str] = None
    speaker_actor_id: Optional[str] = None
    source_role: SourceRole = SourceRole.UNKNOWN
    source_side: Optional[str] = None
    speech_act: SpeechAct = SpeechAct.EXTRACTED
    origin_kind: OriginKind = OriginKind.EXTRACTED

    # Optional structured fields
    subject_ref_type: Optional[str] = None
    subject_ref_id: Optional[str] = None
    predicate_key: Optional[str] = None
    object_json: Optional[str] = None
    temporal_scope_start: Optional[str] = None
    temporal_scope_end: Optional[str] = None

    def proposition_key(self) -> str:
        """Deterministic key for deduplication."""
        import hashlib
        normalized = " ".join(self.proposition_text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]


@dataclass
class AssertionRecord:
    """A canonical assertion as stored in the DB."""
    id: str
    matter_id: str
    proposition_key: str
    proposition_text: str
    model_layer: str
    assertion_kind: str
    belief_state: str
    confidence: float
    created_at: str
    updated_at: str
    subject_ref_type: Optional[str] = None
    subject_ref_id: Optional[str] = None
    predicate_key: Optional[str] = None
    object_json: Optional[str] = None
    temporal_scope_start: Optional[str] = None
    temporal_scope_end: Optional[str] = None


@dataclass
class RevisionResult:
    """Result of applying belief revision to one assertion."""
    assertion_id: str
    old_belief_state: BeliefState
    new_belief_state: BeliefState
    old_confidence: float
    new_confidence: float
    cause: RevisionCause
    propagated_to: list[str] = field(default_factory=list)
    propagation_truncated: bool = False  # True when BFS hit MAX_WORK before full convergence
    # Internal retry frontier — NOT serialized to API responses.
    # correct_assertion() uses this for inline retry; after the retry loop, any
    # remaining nodes are surfaced via propagation_truncated=True to the caller.
    truncation_pending: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context read from the matter model at query time
# ---------------------------------------------------------------------------

@dataclass
class QueryMatterContext:
    """
    A snapshot of the matter model loaded at the start of a query run.

    The engine uses this to avoid rediscovering stable structure and to
    focus retrieval on coverage gaps.
    """
    matter_id: str
    matter_name: str
    open_issues: list[dict] = field(default_factory=list)
    open_gaps: list[dict] = field(default_factory=list)
    active_assumptions: list[dict] = field(default_factory=list)
    existing_assertion_count: int = 0
    existing_actor_count: int = 0
    known_document_ids: list[str] = field(default_factory=list)
    known_actors: list[str] = field(default_factory=list)
    answered_clarifications: list[dict] = field(default_factory=list)
    document_annotations: list[dict] = field(default_factory=list)
    weakest_issue_id: Optional[str] = None
    # SO-2: top predicate_key values from the typed assertion graph.
    # Used by orientation to generate SPO-aware search leads (e.g. "Party A owes_money_to Party B").
    key_predicates: list[str] = field(default_factory=list)
    # Document intelligence: how many documents have structured cards
    document_card_count: int = 0


# ---------------------------------------------------------------------------
# Run session
# ---------------------------------------------------------------------------

@dataclass
class RunSessionRecord:
    """A run session as stored in the DB."""
    id: str
    matter_id: str
    query: str
    status: str
    started_at: str
    objective: Optional[str] = None
    active_branch_issue_id: Optional[str] = None
    stop_requested: bool = False
    redirect_requested: bool = False
    next_action: Optional[str] = None
    completed_at: Optional[str] = None
    assertions_at_start: Optional[int] = None
    reuse_rate: Optional[float] = None
    resumed_from: Optional[str] = None
