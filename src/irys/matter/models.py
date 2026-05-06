"""Command and result dataclasses for the matter model layer."""

import hashlib
import json as _json_mod
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime, timezone

from .enums import (
    SpeechAct, SourceRole, BeliefState, ModelLayer,
    AssertionKind, AssertionLinkType, RevisionCause,
    GapType, OriginKind, LedgerEventType, AbsenceStatus,
    VerificationTargetKind,
)


@dataclass(frozen=True)
class ProvenanceContext:
    """P0.1 Provenance Lite: AI-derived object write context (SO-2).

    Extraction call sites build one of these and pass it through to the
    matter writer (AssertionStore.upsert_occurrence, QuantStore.record,
    AuthorityStore.upsert, DocumentCardStore.upsert,
    EvidenceStore.upsert_edge). The writer forwards it to
    ProvenanceStore.record, which appends a provenance_event row.

    Fields:
    - event_kind: which extraction produced the row (e.g.
      'assertion_extraction', 'edge_write', 'quant_record').
    - writer_name: the fully-qualified writer method, e.g.
      'AssertionStore.upsert_occurrence'.
    - run_id / model_id / model_tier / prompt_version /
      extractor_version / llm_call_id: AI call identity.
    - source_document_ref: relative-path of the source document.
    - source_document_inventory_id: FK into document_inventory.
    - source_span_id / source_span_status: span identity if available.
      P0.1 AC #4 requires absence to be explicit — use 'missing' when
      span is unavailable, 'present' when it is, 'not_applicable' for
      targets that have no natural span (e.g. document_card itself),
      and 'unknown' only for pre-P0.1 imports.
    - note: optional free-form context.
    """

    event_kind: str
    writer_name: str
    run_id: Optional[str] = None
    model_id: Optional[str] = None
    model_tier: Optional[str] = None
    prompt_version: Optional[str] = None
    extractor_version: Optional[str] = None
    llm_call_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    response_hash: Optional[str] = None
    source_document_ref: Optional[str] = None
    source_document_inventory_id: Optional[str] = None
    source_span_id: Optional[str] = None
    source_span_status: str = "unknown"
    note: Optional[str] = None


@dataclass(frozen=True)
class AbsenceStatusRecord:
    """Structured negative/absence result for a searched target.

    Use this for false-premise, out-of-matter, and searched-not-found cases so
    synthesis does not collapse every absence into "missing documents."
    """

    target: str
    status: AbsenceStatus
    searched_documents: int = 0
    searched_terms: tuple[str, ...] = ()
    confidence: float = 0.0
    rationale: str = ""
    source_scope: str = "matter"

    @property
    def is_negative_answer(self) -> bool:
        return self.status in {
            AbsenceStatus.SEARCHED_NOT_FOUND,
            AbsenceStatus.OUT_OF_MATTER,
            AbsenceStatus.FALSE_PREMISE_LIKELY,
        }

    @property
    def requires_more_source(self) -> bool:
        return self.status in {
            AbsenceStatus.NOT_SEARCHED,
            AbsenceStatus.SOURCE_MISSING,
        }

    def to_prompt_line(self) -> str:
        parts = [
            f"target={self.target}",
            f"status={self.status.value}",
            f"searched_documents={self.searched_documents}",
        ]
        if self.searched_terms:
            parts.append("searched_terms=" + ", ".join(self.searched_terms[:8]))
        if self.confidence:
            parts.append(f"confidence={self.confidence:.2f}")
        if self.rationale:
            parts.append(f"rationale={self.rationale}")
        return "; ".join(parts)


@dataclass(frozen=True)
class EvidenceSpanRef:
    """Location of a task-specific evidence object in source material."""

    document_id: str
    span_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_ref: Optional[str] = None
    clause_ref: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    excerpt: str = ""

    @property
    def is_located(self) -> bool:
        return bool(
            self.document_id
            and (
                self.span_id
                or self.section_ref
                or self.clause_ref
                or self.page_start is not None
                or self.excerpt
            )
        )

    def label(self) -> str:
        parts = [self.document_id]
        if self.section_ref:
            parts.append(self.section_ref)
        elif self.clause_ref:
            parts.append(self.clause_ref)
        elif self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"pp. {self.page_start}-{self.page_end}")
            else:
                parts.append(f"p. {self.page_start}")
        elif self.span_id:
            parts.append(f"span {self.span_id}")
        return " | ".join(parts)

    def to_prompt_line(self) -> str:
        parts = [f"source={self.label()}"]
        if self.span_id:
            parts.append(f"span_id={self.span_id}")
        if self.excerpt:
            parts.append(f"excerpt={self.excerpt[:240]}")
        return "; ".join(parts)


@dataclass(frozen=True)
class DefinedTermRecord:
    """Typed extraction result for a defined term."""

    document_id: str
    term: str
    definition_text: str
    first_defined_in: str = ""
    source_span: Optional[EvidenceSpanRef] = None
    aliases: tuple[str, ...] = ()
    confidence: float = 0.0

    target_kind: VerificationTargetKind = VerificationTargetKind.DEFINED_TERM

    @property
    def has_required_fields(self) -> bool:
        return bool(self.document_id and self.term and self.definition_text and self.first_defined_in)

    def identity_key(self) -> str:
        return _hash_text(
            "|".join(
                (
                    _normalize_text(self.document_id),
                    _normalize_text(self.term),
                    _normalize_text(self.first_defined_in),
                )
            ),
            length=32,
        )

    def to_prompt_line(self) -> str:
        parts = [
            f"defined_term={self.term}",
            f"document={self.document_id}",
            f"first_defined_in={self.first_defined_in or 'unknown'}",
            f"definition={self.definition_text[:240]}",
        ]
        if self.source_span:
            parts.append(self.source_span.to_prompt_line())
        if self.confidence:
            parts.append(f"confidence={self.confidence:.2f}")
        return "; ".join(parts)


@dataclass(frozen=True)
class SignatureBlockRecord:
    """Typed extraction result for an executed signature block."""

    document_id: str
    entity_name: str
    signatory_name: str
    title: str = ""
    capacity: str = ""
    execution_date: str = ""
    source_span: Optional[EvidenceSpanRef] = None
    confidence: float = 0.0

    target_kind: VerificationTargetKind = VerificationTargetKind.SIGNATURE_BLOCK

    @property
    def has_name_title_entity(self) -> bool:
        return bool(self.entity_name and self.signatory_name and (self.title or self.capacity))

    def identity_key(self) -> str:
        return _hash_text(
            "|".join(
                (
                    _normalize_text(self.document_id),
                    _normalize_text(self.entity_name),
                    _normalize_text(self.signatory_name),
                    _normalize_text(self.title or self.capacity),
                )
            ),
            length=32,
        )

    def to_prompt_row(self) -> dict[str, str]:
        return {
            "document": self.document_id,
            "entity": self.entity_name,
            "name": self.signatory_name,
            "title": self.title,
            "capacity": self.capacity,
            "execution_date": self.execution_date,
            "source": self.source_span.label() if self.source_span else "",
        }


@dataclass(frozen=True)
class CrossReferenceRecord:
    """Typed extraction result for an in-document reference to another object."""

    source_document_id: str
    target_label: str
    reference_text: str
    source_span: Optional[EvidenceSpanRef] = None
    normalized_target: str = ""
    reference_kind: str = "document"
    confidence: float = 0.0

    target_kind: VerificationTargetKind = VerificationTargetKind.CROSS_REFERENCE

    @property
    def has_reference_span(self) -> bool:
        return bool(
            self.source_document_id
            and self.target_label
            and self.reference_text
            and self.source_span
            and self.source_span.is_located
        )

    def identity_key(self) -> str:
        return _hash_text(
            "|".join(
                (
                    _normalize_text(self.source_document_id),
                    _normalize_text(self.normalized_target or self.target_label),
                    _normalize_text(self.reference_text[:160]),
                )
            ),
            length=32,
        )

    def to_prompt_line(self) -> str:
        parts = [
            f"reference_target={self.target_label}",
            f"source_document={self.source_document_id}",
            f"reference_kind={self.reference_kind}",
            f"text={self.reference_text[:240]}",
        ]
        if self.normalized_target:
            parts.append(f"normalized_target={self.normalized_target}")
        if self.source_span:
            parts.append(self.source_span.to_prompt_line())
        if self.confidence:
            parts.append(f"confidence={self.confidence:.2f}")
        return "; ".join(parts)


@dataclass(frozen=True)
class QuantFactRecord:
    """Scoped quantitative fact used for reconciliation-safe analysis."""

    subject_key: str
    metric_key: str
    value: float
    unit: str = ""
    currency: str = ""
    period_start: str = ""
    period_end: str = ""
    role: str = "observed"
    document_id: str = ""
    source_span: Optional[EvidenceSpanRef] = None
    confidence: float = 0.0

    target_kind: VerificationTargetKind = VerificationTargetKind.QUANT_FACT

    @property
    def reconciliation_scope(self) -> str:
        return "|".join(
            (
                _normalize_text(self.subject_key),
                _normalize_text(self.metric_key),
                _normalize_text(self.unit or self.currency),
                _normalize_text(self.period_start),
                _normalize_text(self.period_end),
            )
        )

    def comparable_to(self, other: "QuantFactRecord") -> bool:
        return self.reconciliation_scope == other.reconciliation_scope

    def identity_key(self) -> str:
        return _hash_text(
            "|".join(
                (
                    self.reconciliation_scope,
                    str(self.value),
                    _normalize_text(self.document_id),
                    _normalize_text(self.role),
                )
            ),
            length=32,
        )

    def to_prompt_line(self) -> str:
        parts = [
            f"subject={self.subject_key}",
            f"metric={self.metric_key}",
            f"value={self.value:g}",
        ]
        if self.currency:
            parts.append(f"currency={self.currency}")
        elif self.unit:
            parts.append(f"unit={self.unit}")
        if self.period_start or self.period_end:
            parts.append(f"period={self.period_start or '?'}..{self.period_end or '?'}")
        parts.append(f"role={self.role}")
        if self.source_span:
            parts.append(self.source_span.to_prompt_line())
        return "; ".join(parts)


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
    # Domain composition: active facets and composed vocabulary from recorded detections
    domain_facets: list[dict] = field(default_factory=list)
    composed_trust_weights: dict[str, float] = field(default_factory=dict)
    primary_domain_profile_id: Optional[str] = None


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
    llm_tool_use_prompt_tokens: Optional[int] = None
    llm_thinking_tokens: Optional[int] = None
    llm_output_tokens: Optional[int] = None
    llm_total_processed_tokens: Optional[int] = None
    llm_request_count: Optional[int] = None
    llm_estimated_cost_usd: Optional[float] = None
