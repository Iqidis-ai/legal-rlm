"""Command and result dataclasses for the matter model layer."""

import hashlib
import json as _json_mod
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime, timezone

from .enums import (
    SpeechAct, SourceRole, BeliefState, ModelLayer,
    AssertionKind, AssertionLinkType, RevisionCause,
    GapType, OriginKind, LedgerEventType,
)


def _normalize_text(value: Optional[str]) -> str:
    return " ".join((value or "").lower().split())


def _hash_text(value: str, *, length: int = 32) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def _stable_json_text(raw: str) -> str:
    try:
        parsed = _json_mod.loads(raw)
    except Exception:
        return _normalize_text(raw)
    return _json_mod.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------------------
# Claim identity v2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimIdentity:
    """Resolved identity for a canonical assertion claim."""
    claim_key: str
    legacy_proposition_key: str
    identity_version: str
    polarity: str
    canonical_subject_key: Optional[str]
    canonical_object_key: Optional[str]
    temporal_identity_key: str
    speaker_scope_key: Optional[str]
    resolution_strategy: str
    canonicalization_confidence: float


# ---------------------------------------------------------------------------
# Assertion inputs / outputs
# ---------------------------------------------------------------------------

@dataclass
class AssertionCandidate:
    """
    An assertion extracted from a document occurrence, ready to be upserted
    into the assertion store.

    proposition_key() is legacy-text compatibility only.
    New writes resolve identity through resolve_claim_identity().
    """
    proposition_text: str
    model_layer: ModelLayer = ModelLayer.RECORD
    assertion_kind: AssertionKind = AssertionKind.FACTUAL

    # Source of this specific occurrence
    document_id: str = ""
    document_inventory_id: Optional[str] = None
    raw_text: Optional[str] = None
    span_id: Optional[str] = None
    speaker_actor_id: Optional[str] = None
    source_role: SourceRole = SourceRole.UNKNOWN
    source_side: Optional[str] = None
    speech_act: SpeechAct = SpeechAct.EXTRACTED
    origin_kind: OriginKind = OriginKind.EXTRACTED

    # Structured fields
    subject_ref_type: Optional[str] = None
    subject_ref_id: Optional[str] = None
    predicate_key: Optional[str] = None
    object_json: Optional[str] = None
    temporal_scope_start: Optional[str] = None
    temporal_scope_end: Optional[str] = None

    # Claim identity v2
    identity_version: str = "claim_v2"
    polarity: str = "affirmed"
    speaker_scope_key: Optional[str] = None
    extraction_confidence: float = 0.0

    def proposition_key(self) -> str:
        """Legacy text-only compatibility key."""
        return _hash_text(_normalize_text(self.proposition_text), length=32)

    def text_signature(self) -> str:
        return _hash_text(_normalize_text(self.raw_text or self.proposition_text), length=32)

    def canonical_subject_key(self) -> Optional[str]:
        subject_type = _normalize_text(self.subject_ref_type)
        subject_id = _normalize_text(self.subject_ref_id)
        if not subject_type or not subject_id:
            return None
        if subject_type == "free_text":
            return f"np:{_hash_text(subject_id, length=32)}"
        return f"{subject_type}:{subject_id}"

    def canonical_object_key(self) -> Optional[str]:
        if not self.object_json:
            return None
        try:
            parsed = _json_mod.loads(self.object_json)
        except Exception:
            return f"json:{_hash_text(_normalize_text(self.object_json), length=32)}"
        if isinstance(parsed, dict) and parsed.get("ref_type") and parsed.get("ref_id"):
            ref_type = _normalize_text(str(parsed["ref_type"]))
            ref_id = _normalize_text(str(parsed["ref_id"]))
            if ref_type == "free_text":
                return f"np:{_hash_text(ref_id, length=32)}"
            return f"{ref_type}:{ref_id}"
        return f"json:{_hash_text(_stable_json_text(self.object_json), length=32)}"

    def resolved_speaker_scope_key(self) -> Optional[str]:
        if self.speaker_scope_key:
            return _normalize_text(self.speaker_scope_key)
        if self.speaker_actor_id:
            return f"actor:{_normalize_text(self.speaker_actor_id)}"
        if self.source_side:
            return f"side:{_normalize_text(self.source_side)}"
        return None

    def temporal_identity_key(self) -> str:
        start = _normalize_text(self.temporal_scope_start)
        end = _normalize_text(self.temporal_scope_end)
        if not start and not end:
            return "atemporal"
        return f"{start}|{end}"

    def resolve_claim_identity(self) -> ClaimIdentity:
        """Resolve structured claim identity using 3-tier strategy."""
        identity_version = self.identity_version or "claim_v2"
        polarity = _normalize_text(self.polarity) or "affirmed"
        subject_key = self.canonical_subject_key()
        object_key = self.canonical_object_key()
        temporal_key = self.temporal_identity_key()
        speaker_key = self.resolved_speaker_scope_key()
        predicate = _normalize_text(self.predicate_key)
        text_key = f"text:{self.text_signature()}"
        base_conf = max(0.0, min(self.extraction_confidence, 1.0))

        # Tier A: full structured identity (subject + predicate + object)
        if subject_key and predicate and object_key:
            resolution_strategy = "tier_a_structured"
            speaker_component = ""
            canonicalization_confidence = max(base_conf, 0.95)
            inputs = (
                identity_version, self.model_layer.value,
                self.assertion_kind.value, polarity,
                subject_key, predicate, object_key,
                temporal_key, speaker_component,
            )
        # Tier B: partial structured (predicate + one of subject/object)
        elif predicate and (subject_key or object_key):
            resolution_strategy = "tier_b_partial_structured"
            speaker_component = speaker_key or "speaker:unknown"
            canonicalization_confidence = max(base_conf, 0.70)
            inputs = (
                identity_version, self.model_layer.value,
                self.assertion_kind.value, polarity,
                subject_key or text_key, predicate, object_key or text_key,
                temporal_key, speaker_component,
            )
        # Tier C: text signature fallback
        else:
            resolution_strategy = "tier_c_text_signature"
            speaker_component = speaker_key or "speaker:unknown"
            canonicalization_confidence = max(base_conf, 0.40)
            inputs = (
                identity_version, self.model_layer.value,
                self.assertion_kind.value, polarity,
                text_key, predicate or "predicate:unresolved", text_key,
                temporal_key, speaker_component,
            )

        claim_key = hashlib.sha256("|".join(inputs).encode()).hexdigest()[:40]
        return ClaimIdentity(
            claim_key=claim_key,
            legacy_proposition_key=self.proposition_key(),
            identity_version=identity_version,
            polarity=polarity,
            canonical_subject_key=subject_key,
            canonical_object_key=object_key,
            temporal_identity_key=temporal_key,
            speaker_scope_key=speaker_component or None,
            resolution_strategy=resolution_strategy,
            canonicalization_confidence=canonicalization_confidence,
        )


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
    claim_key: Optional[str] = None
    identity_version: str = "legacy_text_v1"
    polarity: str = "affirmed"
    canonical_subject_key: Optional[str] = None
    subject_ref_type: Optional[str] = None
    subject_ref_id: Optional[str] = None
    predicate_key: Optional[str] = None
    canonical_object_key: Optional[str] = None
    object_json: Optional[str] = None
    temporal_scope_start: Optional[str] = None
    temporal_scope_end: Optional[str] = None
    temporal_identity_key: str = "atemporal"
    speaker_scope_key: Optional[str] = None
    canonicalization_confidence: float = 0.0


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
    research_mode: Optional[str] = None
    llm_input_tokens: Optional[int] = None
    llm_cache_read_tokens: Optional[int] = None
    llm_output_tokens: Optional[int] = None
    llm_request_count: Optional[int] = None
    llm_estimated_cost_usd: Optional[float] = None
