"""Unit tests for ObligationCoverageMatrix (Item 1)."""

from __future__ import annotations

import asyncio
import json

import pytest

from irys.matter import MatterModel
from irys.rlm.agents import (
    ObligationCoverageMatrix,
    AgentInvocation,
    AgentRequirement,
    AgentTaskView,
    OperatorBudget,
)
from irys.rlm.agents.obligation_coverage import (
    SCHEMA_REF_OBLIGATION_ROW,
    SCHEMA_REF_TASK_CRITERIA,
    SCHEMA_REF_TASK_DELIVERABLE_SPEC,
    ARTIFACT_KIND_MATRIX,
    ARTIFACT_KIND_VALIDATOR_FAILURE,
    RECORD_KIND_OBLIGATION_ROW,
    RECORD_KIND_TASK_CRITERIA,
    RECORD_KIND_TASK_DELIVERABLE_SPEC,
    SEVERITY_RANK,
    make_obligation_id,
)


def _runtime_for(matter: MatterModel):
    class _R:
        def __init__(self, mm):
            self.matter_model = mm
    return _R(matter)


def _invocation(family: str = "deliverable", *, work_profile: dict | None = None,
                requirement: AgentRequirement = AgentRequirement.REQUIRED):
    return AgentInvocation(
        matter_id="m1",
        run_id="run-1",
        agent_id="obligation_coverage_matrix",
        phase="pre_synthesis",
        persona_id=None,
        requirement=requirement,
        task=AgentTaskView(),
        execution_family=family,
        workflow_kind="default",
        budget=OperatorBudget(),
        input_refs=(),
        input_hash="task-fingerprint-1",
        work_profile=work_profile or {},
    )


def _seed_criteria(matter: MatterModel, criteria: list[dict], *, task_id: str = "t1"):
    payload = {
        "schema_ref": SCHEMA_REF_TASK_CRITERIA,
        "task_id": task_id,
        "criteria": criteria,
    }
    matter.typed_evidence.upsert(
        RECORD_KIND_TASK_CRITERIA,
        f"task_criteria:{task_id}",
        payload=payload, document_id=None, confidence=1.0,
    )


def _seed_deliverable(matter: MatterModel, deliverables: list[dict], *, task_id: str = "t1"):
    payload = {
        "schema_ref": SCHEMA_REF_TASK_DELIVERABLE_SPEC,
        "task_id": task_id,
        "deliverables": deliverables,
    }
    matter.typed_evidence.upsert(
        RECORD_KIND_TASK_DELIVERABLE_SPEC,
        f"task_deliverable:{task_id}",
        payload=payload, document_id=None, confidence=1.0,
    )


# ---------------------------------------------------------------------------
# obligation_id determinism (D1)
# ---------------------------------------------------------------------------


def test_make_obligation_id_is_deterministic_and_prefixed():
    a = make_obligation_id(
        task_fingerprint="t1", source_family="lab_config",
        source_criterion_key_or_text="C-001",
        deliverable_key="memo.docx", required_slot_key="section:summary",
        expected_artifact_kind="draft_document.v1",
    )
    b = make_obligation_id(
        task_fingerprint="t1", source_family="lab_config",
        source_criterion_key_or_text="C-001",
        deliverable_key="memo.docx", required_slot_key="section:summary",
        expected_artifact_kind="draft_document.v1",
    )
    assert a == b, "obligation_id must be deterministic"
    assert a.startswith("obl_v1:")
    assert len(a) == len("obl_v1:") + 24


def test_make_obligation_id_distinguishes_inputs():
    base = dict(
        task_fingerprint="t1", source_family="lab_config",
        source_criterion_key_or_text="C-001",
        deliverable_key="memo.docx", required_slot_key="section:summary",
        expected_artifact_kind="draft_document.v1",
    )
    a = make_obligation_id(**base)
    diff = dict(base, deliverable_key="other.docx")
    b = make_obligation_id(**diff)
    assert a != b


# ---------------------------------------------------------------------------
# match() relevance gating
# ---------------------------------------------------------------------------


def test_match_strong_signal_when_criteria_present():
    agent = ObligationCoverageMatrix()
    inv = _invocation("deliverable", work_profile={
        "task_criteria_count": 5, "task_deliverable_spec_count": 1,
    })
    m = agent.match(inv)
    assert m is not None
    assert m.score >= 0.95
    assert m.requirement == AgentRequirement.BLOCKING_VALIDATOR


def test_match_required_when_investigate_with_criteria():
    agent = ObligationCoverageMatrix()
    inv = _invocation("investigate", work_profile={"task_criteria_count": 3})
    m = agent.match(inv)
    assert m is not None
    assert m.requirement == AgentRequirement.REQUIRED


def test_match_skips_for_read_query_trace_clarify():
    agent = ObligationCoverageMatrix()
    for fam in ("read", "query", "trace", "clarify"):
        assert agent.match(_invocation(fam)) is None


def test_match_falls_through_when_no_signal():
    agent = ObligationCoverageMatrix()
    inv = _invocation("investigate", work_profile={
        "task_criteria_count": 0, "task_deliverable_spec_count": 0,
        "obligation_row_count": 0,
    })
    assert agent.match(inv) is None


def test_match_deliverable_family_no_criteria_still_runs():
    agent = ObligationCoverageMatrix()
    inv = _invocation("deliverable", work_profile={
        "task_criteria_count": 0, "task_deliverable_spec_count": 0,
        "obligation_row_count": 0,
    })
    m = agent.match(inv)
    assert m is not None
    assert m.requirement == AgentRequirement.REQUIRED


# ---------------------------------------------------------------------------
# invoke() — happy paths
# ---------------------------------------------------------------------------


def test_invoke_with_no_criteria_returns_ran_empty_warning():
    agent = ObligationCoverageMatrix()
    matter = MatterModel.open_in_memory()
    inv = _invocation()
    result = asyncio.run(agent.invoke(inv, _runtime_for(matter)))
    assert result.status == "success"
    assert "no_criteria_no_deliverables" in result.warnings
    assert not result.artifacts


def test_invoke_produces_matrix_and_typed_rows_from_criteria():
    agent = ObligationCoverageMatrix()
    matter = MatterModel.open_in_memory()
    _seed_criteria(matter, [
        {"criterion_id": "C-001", "title": "Include market analysis",
         "description": "must analyze HHI for top 4 markets",
         "severity": "critical"},
        {"criterion_id": "C-002", "title": "Cite all sources",
         "description": "every claim has a source", "severity": "required"},
    ])
    _seed_deliverable(matter, [
        {"deliverable_key": "memo.docx", "filename": "antitrust-memo.docx",
         "format": "docx",
         "required_sections": ["Executive Summary", "Market Analysis"],
         "required_tables": [],
         "required_artifact_kinds": ["draft_document.v1"]},
    ])

    inv = _invocation()
    result = asyncio.run(agent.invoke(inv, _runtime_for(matter)))
    assert result.status == "success"
    assert len(result.artifacts) == 1
    matrix = result.artifacts[0]
    assert matrix.artifact_kind == ARTIFACT_KIND_MATRIX
    p = matrix.payload
    assert p["matrix_status"] == "success"
    # 2 criteria × (2 sections + 1 artifact_kind) = 6 rows
    assert p["n_total"] == 6

    # All rows are persisted as typed_evidence_record(obligation_row)
    obs = matter.db.execute(
        "SELECT COUNT(*) AS n FROM typed_evidence_record "
        "WHERE record_kind=?", (RECORD_KIND_OBLIGATION_ROW,),
    ).fetchone()["n"]
    assert obs == 6


def test_invoke_marks_artifact_kind_as_met_when_present():
    agent = ObligationCoverageMatrix()
    matter = MatterModel.open_in_memory()
    _seed_criteria(matter, [
        {"criterion_id": "C-001", "title": "Section map present",
         "severity": "required"},
    ])
    _seed_deliverable(matter, [
        {"deliverable_key": "memo.docx",
         "required_artifact_kinds": ["document.section_map"]},
    ])
    # Pre-seed an artifact of that kind by inserting into agent_artifact directly.
    # Using a fake invocation row to satisfy FK.
    matter.db.execute(
        "INSERT INTO sub_agent_invocation (id, matter_id, run_id, agent_id, "
        "agent_version, phase, status, requirement, invocation_at, input_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("inv1", matter.matter_id, "run-prior",
         "doc.file_reader", 1, "pre_synthesis", "success", "optional",
         "2026-05-08T00:00:00.000000+00:00", "h1"),
    )
    matter.db.execute(
        "INSERT INTO agent_artifact (id, matter_id, invocation_id, "
        "artifact_kind, artifact_key, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("a1", matter.matter_id, "inv1",
         "document.section_map", "k1", "{}",
         "2026-05-08T00:00:00.000000+00:00"),
    )
    inv = _invocation()
    result = asyncio.run(agent.invoke(inv, _runtime_for(matter)))
    p = result.artifacts[0].payload
    rows = p["rows"]
    art_kind_rows = [r for r in rows if r["required_slot"]["kind"] == "artifact_kind"]
    assert art_kind_rows
    assert all(r["status"] == "met" for r in art_kind_rows)


# ---------------------------------------------------------------------------
# verify_output blocking-validator semantics (D8)
# ---------------------------------------------------------------------------


def test_verify_output_halts_when_blocking_critical_missing():
    agent = ObligationCoverageMatrix()
    matter = MatterModel.open_in_memory()
    _seed_criteria(matter, [
        {"criterion_id": "C-001", "title": "Critical finding",
         "severity": "critical"},
    ])
    _seed_deliverable(matter, [
        {"deliverable_key": "memo.docx",
         "required_artifact_kinds": ["draft_document.v1"]},  # absent
    ])
    inv = _invocation(requirement=AgentRequirement.BLOCKING_VALIDATOR)
    result = asyncio.run(agent.invoke(inv, _runtime_for(matter)))
    verified = agent.verify_output(inv, result)
    assert verified.status == "invalid"
    assert verified.error_class == "ObligationCoverageBlocking"
    assert "blocking_validator_halt" in verified.warnings


def test_verify_output_passes_when_all_required_met():
    agent = ObligationCoverageMatrix()
    matter = MatterModel.open_in_memory()
    _seed_criteria(matter, [
        {"criterion_id": "C-001", "title": "Provide section map",
         "severity": "required"},
    ])
    _seed_deliverable(matter, [
        {"deliverable_key": "memo.docx",
         "required_artifact_kinds": ["document.section_map"]},
    ])
    matter.db.execute(
        "INSERT INTO sub_agent_invocation (id, matter_id, run_id, agent_id, "
        "agent_version, phase, status, requirement, invocation_at, input_hash) "
        "VALUES ('inv1', ?, 'run-prior', 'doc.file_reader', 1, "
        "'pre_synthesis', 'success', 'optional', '2026-05-08T00:00:00', 'h1')",
        (matter.matter_id,),
    )
    matter.db.execute(
        "INSERT INTO agent_artifact (id, matter_id, invocation_id, "
        "artifact_kind, artifact_key, payload_json, created_at) "
        "VALUES ('a1', ?, 'inv1', 'document.section_map', 'k1', '{}', "
        "'2026-05-08T00:00:00')",
        (matter.matter_id,),
    )
    inv = _invocation(requirement=AgentRequirement.BLOCKING_VALIDATOR)
    result = asyncio.run(agent.invoke(inv, _runtime_for(matter)))
    verified = agent.verify_output(inv, result)
    assert verified.status == "success"


# ---------------------------------------------------------------------------
# D6 — drift detection across runs
# ---------------------------------------------------------------------------


def test_rerun_marks_old_rows_active_false_when_dropped():
    agent = ObligationCoverageMatrix()
    matter = MatterModel.open_in_memory()
    # Run 1 with 2 criteria
    _seed_criteria(matter, [
        {"criterion_id": "C-001", "title": "First", "severity": "required"},
        {"criterion_id": "C-002", "title": "Second", "severity": "required"},
    ])
    _seed_deliverable(matter, [
        {"deliverable_key": "memo.docx",
         "required_artifact_kinds": ["draft_document.v1"]},
    ])
    inv1 = _invocation()
    asyncio.run(agent.invoke(inv1, _runtime_for(matter)))

    # Run 2: drop C-002. (Update typed_evidence to only have C-001.)
    matter.db.execute(
        "DELETE FROM typed_evidence_record WHERE record_kind=?",
        (RECORD_KIND_TASK_CRITERIA,),
    )
    _seed_criteria(matter, [
        {"criterion_id": "C-001", "title": "First", "severity": "required"},
    ])
    inv2 = _invocation()
    asyncio.run(agent.invoke(inv2, _runtime_for(matter)))

    # The dropped C-002 row should be marked active=False
    rows = matter.db.execute(
        "SELECT payload_json FROM typed_evidence_record WHERE record_kind=?",
        (RECORD_KIND_OBLIGATION_ROW,),
    ).fetchall()
    payloads = [json.loads(r["payload_json"]) for r in rows]
    inactive = [p for p in payloads if not p.get("active", True)]
    assert any(p.get("stale_reason") == "not_in_current_matrix" for p in inactive)


def test_drift_summary_records_added_and_removed():
    agent = ObligationCoverageMatrix()
    matter = MatterModel.open_in_memory()
    _seed_criteria(matter, [
        {"criterion_id": "C-001", "title": "Original", "severity": "required"},
    ])
    _seed_deliverable(matter, [
        {"deliverable_key": "memo.docx",
         "required_artifact_kinds": ["draft_document.v1"]},
    ])
    asyncio.run(agent.invoke(_invocation(), _runtime_for(matter)))

    # Run 2: replace C-001 with C-002
    matter.db.execute(
        "DELETE FROM typed_evidence_record WHERE record_kind=?",
        (RECORD_KIND_TASK_CRITERIA,),
    )
    _seed_criteria(matter, [
        {"criterion_id": "C-002", "title": "Replacement", "severity": "required"},
    ])
    result2 = asyncio.run(agent.invoke(_invocation(), _runtime_for(matter)))
    drift = result2.artifacts[0].payload["drift"]
    assert len(drift["added"]) >= 1
    assert len(drift["removed"]) >= 1
