"""Enumerations for the matter model substrate."""

from enum import Enum


class SpeechAct(str, Enum):
    """The illocutionary act performed by a speaker in making an assertion."""
    ALLEGED = "alleged"
    ARGUED = "argued"
    DENIED = "denied"
    ADMITTED = "admitted"
    ORDERED = "ordered"
    PERFORMED = "performed"
    PAID = "paid"
    REQUESTED = "requested"
    THREATENED = "threatened"
    PROMISED = "promised"
    ESTIMATED = "estimated"
    CALCULATED = "calculated"
    OBSERVED = "observed"
    TESTIFIED = "testified"
    STIPULATED = "stipulated"
    AMENDED = "amended"
    WAIVED = "waived"
    TERMINATED = "terminated"
    INFERRED = "inferred"
    OPERATIVE = "operative"  # operative contract/agreement language (creates rights/obligations)
    EXTRACTED = "extracted"  # system-extracted, no attributed speech act yet


class SourceRole(str, Enum):
    """The functional role of a document source in the legal matter."""
    ADVOCACY = "advocacy"                   # pleadings, briefs, demand letters
    OPERATIVE = "operative"                 # contracts, signed agreements, orders
    PROCEDURAL = "procedural"               # court filings, process documents
    AUTHORITATIVE = "authoritative"         # court orders, statutes, regulations
    INFORMAL = "informal"                   # emails, notes, messages
    DRAFT = "draft"                         # negotiation drafts, redlines
    POST_HOC_EXPLANATORY = "post_hoc"       # memos explaining past events
    UNKNOWN = "unknown"


# Legacy legal-domain trust weights (SO-5 fallback).
# New code should use composed weights from domain facets.
SOURCE_TRUST_WEIGHTS: dict[str, float] = {
    "operative": 1.0,
    "authoritative": 1.0,
    "procedural": 0.7,
    "informal": 0.5,
    "unknown": 0.5,
    "draft": 0.4,
    "advocacy": 0.3,
    "post_hoc": 0.3,
}


class BeliefState(str, Enum):
    """The system's current belief about the status of an assertion."""
    ALLEGED = "alleged"
    ARGUED = "argued"
    ADMITTED = "admitted"
    OPERATIVE = "operative"
    PERFORMED = "performed"
    NOT_PERFORMED = "not_performed"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"
    INFERRED = "inferred"
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


class ModelLayer(str, Enum):
    """Which reasoning layer an assertion belongs to."""
    RECORD = "record"               # What the documents literally say
    REALITY = "reality"             # What likely happened in the world
    PROOF = "proof"                 # What can be supported/proved
    LEGAL = "legal"                 # What the law says
    DECISION_CONTEXT = "decision_context"  # What decision-makers care about


class AssertionKind(str, Enum):
    """The logical kind of an assertion."""
    FACTUAL = "factual"
    NORMATIVE = "normative"         # obligation, right, duty
    QUANTITATIVE = "quantitative"   # amount, date, rate
    TEMPORAL = "temporal"           # event, sequence, deadline
    RELATIONAL = "relational"       # actor-to-actor, doc-to-doc


class AssertionLinkType(str, Enum):
    """The type of directed link between two assertions."""
    SUPPORTS = "supports"
    ATTACKS = "attacks"
    DEPENDS_ON = "depends_on"
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    CORROBORATES = "corroborates"


class RevisionCause(str, Enum):
    """Why a belief revision was triggered."""
    NEW_EVIDENCE = "new_evidence"
    USER_CORRECTION = "user_correction"
    CONFLICT_DETECTION = "conflict_detection"
    SUPERSESSION = "supersession"
    ADMISSION = "admission"
    TRUST_OVERRIDE = "trust_override"


class GapType(str, Enum):
    """The type of known missingness."""
    MISSING_DOCUMENT = "missing_document"
    MISSING_METADATA = "missing_metadata"
    MISSING_ISSUE_PREDICATE = "missing_issue_predicate"
    MISSING_AUTHORITY = "missing_authority"
    MISSING_USER_CONTEXT = "missing_user_context"
    MISSING_QUANTITATIVE_INPUT = "missing_quantitative_input"
    UNRESOLVED_CONTRADICTION = "unresolved_contradiction"
    EXPECTED_ABSENT_ATTACHMENT = "expected_absent_attachment"
    EXPECTED_ABSENT_NOTICE = "expected_absent_notice"


class AbsenceStatus(str, Enum):
    """Typed result for searches whose answer may be negative.

    This is intentionally distinct from GapType. A gap says the model lacks
    support for an issue; an absence status says what kind of negative or
    missingness result a search produced.
    """

    NOT_SEARCHED = "not_searched"
    SEARCHED_NOT_FOUND = "searched_not_found"
    FOUND_UNVERIFIED = "found_unverified"
    FOUND_VERIFIED = "found_verified"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    OUT_OF_MATTER = "out_of_matter"
    FALSE_PREMISE_LIKELY = "false_premise_likely"
    SOURCE_MISSING = "source_missing"


class IssueType(str, Enum):
    """The type of an issue node."""
    CLAIM = "claim"
    DEFENSE = "defense"
    CONTRACT_QUESTION = "contract_question"
    CONDITION_PRECEDENT = "condition_precedent"
    WAIVER = "waiver"
    DAMAGES = "damages"
    DILIGENCE_RED_FLAG = "diligence_red_flag"
    COMPLIANCE_FAILURE = "compliance_failure"
    PROCEDURAL_BARRIER = "procedural_barrier"
    EVIDENTIARY_BOTTLENECK = "evidentiary_bottleneck"


class OriginKind(str, Enum):
    """How an assertion occurrence came to be in the system."""
    EXTRACTED = "extracted"         # LLM extracted from document
    INFERRED = "inferred"           # System inferred from other assertions
    USER_SUPPLIED = "user_supplied"  # User explicitly provided
    IMPORTED = "imported"           # Imported from external source


class RunStatus(str, Enum):
    """Status of a reasoning run session."""
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class LedgerEventType(str, Enum):
    """Type of event in the reasoning ledger."""
    RUN_STARTED = "run_started"
    OBJECTIVE_SET = "objective_set"
    BRANCH_SELECTED = "branch_selected"
    ASSERTION_ADDED = "assertion_added"
    ASSERTION_REVISED = "assertion_revised"
    CONFLICT_DETECTED = "conflict_detected"
    GAP_IDENTIFIED = "gap_identified"
    ISSUE_UPDATED = "issue_updated"
    USER_INTERRUPTED = "user_interrupted"
    USER_REDIRECTED = "user_redirected"
    USER_CORRECTION = "user_correction"
    SYNTHESIS_STARTED = "synthesis_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    SYSTEM_WARNING = "system_warning"
    PROGRESS_NOTE = "progress_note"
    ROUTE_DECISION = "route_decision"  # MVI-1 cascade front-door gate


class VerificationStatus(str, Enum):
    """Human-review state of a matter intelligence object (MVP.2, SO-2).

    Truth is not belief: this enum tracks whether a human has reviewed an
    AI-derived object. Proof-state and assertion belief_state are separate.
    """

    CANDIDATE = "candidate"
    VERIFIED = "verified"
    REJECTED = "rejected"
    STALE = "stale"


class VerificationTargetKind(str, Enum):
    """The kinds of intelligence objects that carry a verification state."""

    ASSERTION = "assertion"
    ASSERTION_OCCURRENCE = "assertion_occurrence"
    ISSUE_PREDICATE = "issue_predicate"
    EVIDENCE_EDGE = "evidence_edge"
    QUANT_FACT = "quant_fact"
    AUTHORITY = "authority"
    DOCUMENT_CARD = "document_card"
    PRIVILEGE_CLASSIFICATION = "privilege_classification"
    GAP = "gap"
    DISPUTE = "dispute"
    DISPUTE_POSITION = "dispute_position"
    TIMELINE_EVENT = "timeline_event"
    DEADLINE = "deadline"
    AUTHORITY_TREATMENT = "authority_treatment"
    ACTOR_RELATIONSHIP = "actor_relationship"
    DEFINED_TERM = "defined_term"
    SIGNATURE_BLOCK = "signature_block"
    CROSS_REFERENCE = "cross_reference"
    SECTION_REF = "section_ref"
    SCHEDULE_ENTRY = "schedule_entry"
    REDACTION_MARKER = "redaction_marker"
    ABSENCE_STATUS = "absence_status"
    CAUSATION_EDGE = "causation_edge"
    THEORY = "theory"
    ARTIFACT = "artifact"
    ARTIFACT_MANIFEST_ITEM = "artifact_manifest_item"


class ReviewScope(str, Enum):
    """What aspect a human review has validated."""

    EXTRACTION_CORRECT = "extraction_correct"
    RECORD_TRUTH = "record_truth"
    INFERENCE = "inference"
    LEGAL_CONCLUSION = "legal_conclusion"
    TRUTH_OVERRIDE = "truth_override"
    INTERNAL_PRIVILEGED = "internal_privileged"
    CLEAN_OUTPUT = "clean_output"
    PRIVILEGE_CLASSIFICATION = "privilege_classification"
    DISPUTE_RESOLUTION = "dispute_resolution"
    ARTIFACT_POLICY = "artifact_policy"


class ReviewedByKind(str, Enum):
    """Who performed the review.

    Only 'user' and 'attorney' may promote status to 'verified'. 'system'
    and 'import' are automation markers and must never reach verified.
    """

    USER = "user"
    ATTORNEY = "attorney"
    SYSTEM = "system"
    IMPORT = "import"


class EvidenceRelationType(str, Enum):
    """Relation type persisted on evidence_edge rows (MVP.3, SO-4).

    Intentionally matches the assertion_issue_link vocabulary so MVP.3
    can swap the proof substrate without also rewriting the relation
    vocabulary. ASPIC+-style rebut/undercut come in a later phase.
    """

    SUPPORTS = "supports"
    ESTABLISHES = "establishes"
    ATTACKS = "attacks"
    NEGATES = "negates"


class EvidenceOriginKind(str, Enum):
    """How an evidence_edge row came into existence (MVP.3).

    Separate from the assertion-oriented OriginKind because the
    evidence-edge lifecycle has distinct values — legacy_backfill and
    attorney_annotated do not apply to assertion occurrences.
    """

    AI_EXTRACTED = "ai_extracted"
    ATTORNEY_ANNOTATED = "attorney_annotated"
    SYSTEM_INFERRED = "system_inferred"
    IMPORTED = "imported"
    LEGACY_BACKFILL = "legacy_backfill"
