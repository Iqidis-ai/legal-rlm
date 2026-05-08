"""Tests for CpSectionExtractorAgent — deterministic CP coverage operator."""

from __future__ import annotations

import asyncio

import pytest

from irys.matter import MatterModel
from irys.rlm.agents import (
    AgentInvocation, AgentRequirement, AgentTaskView,
    CpSectionExtractorAgent, OperatorBudget,
    SubAgentDispatcher, SubAgentRegistry,
)
from irys.rlm.agents.cp_section_extractor import _better_revision


# ---------------------------------------------------------------------------
# _better_revision priority logic
# ---------------------------------------------------------------------------


def test_better_revision_missing_beats_met():
    """Missing finding wins over later met — surface problems, not happy paths."""
    a = {"cp_id": "1", "status": "missing", "severity": "high"}
    b = {"cp_id": "1", "status": "met", "severity": "high"}
    assert _better_revision(a, b) is a


def test_better_revision_critical_severity_wins():
    a = {"cp_id": "1", "status": "missing", "severity": "low"}
    b = {"cp_id": "1", "status": "missing", "severity": "critical"}
    assert _better_revision(a, b) is b


def test_better_revision_partial_beats_met():
    a = {"cp_id": "1", "status": "partial", "severity": "high"}
    b = {"cp_id": "1", "status": "met", "severity": "high"}
    assert _better_revision(a, b) is a


# ---------------------------------------------------------------------------
# Agent end-to-end
# ---------------------------------------------------------------------------


def _seed_cp(matter: MatterModel, gen: str, **kw):
    payload = {"schema_ref": "legal.cp_gap.v1", **kw}
    cp_id = kw.get("cp_id", "CP")
    doc = (kw.get("required_by_document") or "Doc").lower().replace(" ", "_")
    matter.typed_evidence.upsert(
        "cp_gap",
        f"cp_gap_revision:{doc}:{cp_id}:{gen}",
        payload=payload,
        document_id=kw.get("source_document") or "agreement.pdf",
        confidence=0.9,
    )


def _make_invocation(matter_id: str):
    return AgentInvocation(
        matter_id=matter_id, run_id="r",
        agent_id="cp", phase="pre_synthesis",
        persona_id=None, requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(),
        execution_family="investigate", workflow_kind="default",
        budget=OperatorBudget(),
        input_refs=(), input_hash="h",
    )


def test_cp_extractor_no_evidence_returns_success_zero_artifacts():
    m = MatterModel.open_in_memory()
    agent = CpSectionExtractorAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert result.status == "success"
    assert result.artifacts == ()


def test_cp_extractor_groups_by_required_document():
    m = MatterModel.open_in_memory()
    _seed_cp(m, "g1", cp_id="4.01(a)", requirement_text="secretary cert",
             status="missing", severity="critical",
             required_by_document="Credit Agreement.pdf")
    _seed_cp(m, "g1", cp_id="4.01(b)", requirement_text="opinion letter",
             status="met", severity="medium",
             required_by_document="Credit Agreement.pdf")
    _seed_cp(m, "g1", cp_id="2.01", requirement_text="title policy",
             status="partial", severity="high",
             required_by_document="Mortgage.pdf")
    agent = CpSectionExtractorAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert len(result.artifacts) == 2  # one per required document

    by_doc = {a.payload["required_document"]: a for a in result.artifacts}
    ca = by_doc["Credit Agreement.pdf"].payload
    assert ca["n_total"] == 2
    assert ca["n_missing"] == 1
    assert ca["n_met"] == 1
    assert ca["n_critical_missing"] == 1
    # critical missing → verification_state candidate
    assert by_doc["Credit Agreement.pdf"].verification_state == "candidate"

    mort = by_doc["Mortgage.pdf"].payload
    assert mort["n_total"] == 1
    assert mort["n_partial"] == 1
    assert mort["n_critical_missing"] == 0


def test_cp_extractor_collapses_multiple_revisions_to_current_best():
    """Same CP with 'missing' and later 'met' revisions → missing wins."""
    m = MatterModel.open_in_memory()
    _seed_cp(m, "g1", cp_id="4.01(a)", requirement_text="cert",
             status="missing", severity="critical",
             required_by_document="CA.pdf")
    _seed_cp(m, "g2", cp_id="4.01(a)", requirement_text="cert",
             status="met", severity="critical",
             required_by_document="CA.pdf")
    agent = CpSectionExtractorAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    art = out.invocations[0][1].artifacts[0]
    p = art.payload
    assert p["n_total"] == 1  # collapsed to one requirement
    assert p["n_missing"] == 1
    assert p["n_met"] == 0
    items = p["items"]
    assert len(items) == 1
    assert items[0]["status"] == "missing"
    assert items[0]["n_revisions"] == 2


def test_cp_extractor_orders_missing_first():
    m = MatterModel.open_in_memory()
    _seed_cp(m, "g1", cp_id="A", requirement_text="met item", status="met",
             severity="high", required_by_document="X.pdf")
    _seed_cp(m, "g2", cp_id="B", requirement_text="missing item",
             status="missing", severity="critical",
             required_by_document="X.pdf")
    _seed_cp(m, "g3", cp_id="C", requirement_text="partial", status="partial",
             severity="high", required_by_document="X.pdf")
    agent = CpSectionExtractorAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    items = out.invocations[0][1].artifacts[0].payload["items"]
    statuses = [it["status"] for it in items]
    assert statuses == ["missing", "partial", "met"]


def test_cp_extractor_warns_on_multiple_revisions():
    m = MatterModel.open_in_memory()
    _seed_cp(m, "g1", cp_id="X", requirement_text="r", status="missing",
             required_by_document="D.pdf")
    _seed_cp(m, "g2", cp_id="X", requirement_text="r", status="met",
             required_by_document="D.pdf")
    agent = CpSectionExtractorAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert any("cp_revisions" in w for w in result.warnings)


def test_cp_extractor_persists_artifacts():
    m = MatterModel.open_in_memory()
    _seed_cp(m, "g1", cp_id="X", requirement_text="r", status="missing",
             required_by_document="D.pdf")
    agent = CpSectionExtractorAgent()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    n = m.db.execute(
        "SELECT COUNT(*) FROM agent_artifact WHERE artifact_kind='cp.coverage_report'"
    ).fetchone()[0]
    assert n == 1


def test_cp_extractor_capability_tags():
    a = CpSectionExtractorAgent()
    assert "extract.section" in a.capability_tags
    assert "compare.coverage" in a.capability_tags
    assert "verify.extraction" in a.capability_tags


def test_cp_extractor_in_default_registry():
    """The default registry pre-registers all built-in operators."""
    from irys.rlm.agents import default_registry
    reg = default_registry()
    ids = {a.agent_id for a in reg.list()}
    assert "banking.cp_section_extractor.v1" in ids
    assert "antitrust.hhi_market_share.v1" in ids
    assert "finance.numerical_reconciliation.v1" in ids
    assert "document.file_reader.v1" in ids
