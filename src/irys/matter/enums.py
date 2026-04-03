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


class IssueType(str, Enum):
    """The type of a legal issue node."""
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
