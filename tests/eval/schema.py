"""Typed fixture schema for the eval harness.

One JSON manifest per synthetic micro-matter. Loaders deserialize into these
dataclasses so fixture data stays reviewable and test failures can blame a
specific field rather than a JSON parse error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class FixtureDocument:
    alias: str
    document_id: str
    filename: str
    source_role: str = "unknown"
    source_side: str = "neutral"
    doc_type: Optional[str] = None
    privilege_status: Optional[str] = None  # e.g. "privileged", "clean", "unknown"
    text: Optional[str] = None


@dataclass(frozen=True)
class FixtureAssertion:
    alias: str
    document_alias: str
    proposition_text: str
    speech_act: str = "extracted"
    source_role: str = "unknown"
    model_layer: str = "record"
    assertion_kind: str = "factual"
    origin_kind: str = "extracted"
    verified: bool = False  # PR.1/MVP.2 — default candidate


@dataclass(frozen=True)
class FixtureIssue:
    alias: str
    title: str
    issue_type: str = "claim"
    materiality: float = 0.6
    salience: float = 0.5
    burden_side: Optional[str] = None
    predicates: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixtureIssueLink:
    assertion_alias: str
    issue_alias: str
    relation_type: str = "supports"


@dataclass(frozen=True)
class FixtureGap:
    alias: str
    gap_type: str
    description: str
    materiality_score: float = 0.5
    affected_type: Optional[str] = None  # e.g. "issue"
    affected_alias: Optional[str] = None


@dataclass(frozen=True)
class FixtureMaintenanceStep:
    step: str  # e.g. "proof_state.compute_all", "engine._detect_proof_gaps"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvariantSpec:
    name: str
    group: str  # e.g. "proof_gap", "evidence_edge", "verification", "privilege"
    requires: tuple[str, ...] = ()  # capability tags; empty means always active
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    intent: str
    sacred_outcomes: tuple[str, ...]
    documents: tuple[FixtureDocument, ...] = ()
    assertions: tuple[FixtureAssertion, ...] = ()
    issues: tuple[FixtureIssue, ...] = ()
    issue_links: tuple[FixtureIssueLink, ...] = ()
    gaps: tuple[FixtureGap, ...] = ()
    maintenance: tuple[FixtureMaintenanceStep, ...] = ()
    scripted_responses: dict[str, str] = field(default_factory=dict)
    invariants: tuple[InvariantSpec, ...] = ()
