"""Matter model substrate for Irys RLM.

The matter module provides the persistent intelligence layer:
- Typed assertion graph with truth maintenance
- Reasoning ledger with user-steerable events
- Gap tracking for structured missingness
- Belief revision propagating through dependency graph

Primary entry point: MatterModel.open(repository_path)
"""

from .matter import MatterModel
from .db import SQLiteMatterDB
from .enums import (
    SpeechAct, SourceRole, BeliefState, ModelLayer,
    AssertionKind, AssertionLinkType, RevisionCause,
    GapType, IssueType, OriginKind, RunStatus, LedgerEventType,
)
from .models import (
    AssertionCandidate, AssertionRecord, RevisionResult,
    QueryMatterContext, RunSessionRecord,
)
from .graph import AssertionStore, GapStore, ActorStore, IssueStore, ClarificationStore
from .reasoning import ReasoningLedgerStore
from .belief_revision import BeliefRevisionEngine
from .runtime import MatterRuntimeAdapter, NullMatterAdapter, infer_source_role

__all__ = [
    "MatterModel",
    "SQLiteMatterDB",
    # Enums
    "SpeechAct", "SourceRole", "BeliefState", "ModelLayer",
    "AssertionKind", "AssertionLinkType", "RevisionCause",
    "GapType", "IssueType", "OriginKind", "RunStatus", "LedgerEventType",
    # Models
    "AssertionCandidate", "AssertionRecord", "RevisionResult",
    "QueryMatterContext", "RunSessionRecord",
    # Stores
    "AssertionStore", "GapStore", "ActorStore", "IssueStore", "ClarificationStore",
    "ReasoningLedgerStore",
    "BeliefRevisionEngine",
    # Runtime
    "MatterRuntimeAdapter", "NullMatterAdapter", "infer_source_role",
]
