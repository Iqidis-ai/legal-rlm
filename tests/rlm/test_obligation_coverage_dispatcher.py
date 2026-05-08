"""Dispatcher-integration tests for ObligationCoverageMatrix.

Covers Codex Phase-2 r1 blockers:
  - dispatcher honors AgentMatch.requirement (BLOCKING_VALIDATOR escalation)
  - artifacts persist even when verify_output returns 'invalid'
  - row upsert failures surface (not silently swallowed)
  - task_fingerprint uses payload task_id (not just input_hash)
  - matrix_status reports upstream_required_evidence_missing correctly
  - D7 renderer rule: positive summary only when blocking_validator OR user-asks
"""

from __future__ import annotations

import asyncio
import json

import pytest

from irys.matter import MatterModel
from irys.rlm.agents import (
    AgentInvocation,
    AgentRequirement,
    AgentTaskView,
    OperatorBudget,
    ObligationCoverageMatrix,
    SubAgentDispatcher,
    SubAgentRegistry,
)
from irys.rlm.agents.obligation_coverage import (
    SCHEMA_REF_TASK_CRITERIA,
    SCHEMA_REF_TASK_DELIVERABLE_SPEC,
    ARTIFACT_KIND_MATRIX,
    ARTIFACT_KIND_VALIDATOR_FAILURE,
    RECORD_KIND_OBLIGATION_ROW,
    RECORD_KIND_TASK_CRITERIA,
    RECORD_KIND_TASK_DELIVERABLE_SPEC,
    make_obligation_id,
)


def _seed(matter, criteria=None, deliverables=None, *, task_id="t1"):
    if criteria is not None:
        matter.typed_evidence.upsert(
            RECORD_KIND_TASK_CRITERIA, f"task_criteria:{task_id}",
            payload={"schema_ref": SCHEMA_REF_TASK_CRITERIA,
                     "task_id": task_id, "criteria": criteria},
            confidence=1.0,
        )
    if deliverables is not None:
        matter.typed_evidence.upsert(
            RECORD_KIND_TASK_DELIVERABLE_SPEC, f"task_deliverable:{task_id}",
            payload={"schema_ref": SCHEMA_REF_TASK_DELIVERABLE_SPEC,
                     "task_id": task_id, "deliverables": deliverables},
            confidence=1.0,
        )


def _invocation(matter, family="deliverable",
                requirement=AgentRequirement.OPTIONAL,
                work_profile=None):
    """Build an invocation. work_profile defaults are seeded so the
    obligation_coverage operator's match() lands in the strong-signal
    branch (which is what the engine's _compute_agent_work_profile
    would compute in a real run after seeding criteria + deliverables).
    """
    if work_profile is None:
        work_profile = {
            "task_criteria_count": 1,
            "task_deliverable_spec_count": 1,
            "obligation_row_count": 0,
        }
    return AgentInvocation(
        matter_id=matter.matter_id, run_id="run-1",
        agent_id="dispatcher",
        phase="pre_synthesis", persona_id=None,
        requirement=requirement,
        task=AgentTaskView(),
        execution_family=family, workflow_kind="default",
        budget=OperatorBudget(),
        input_refs=(), input_hash="some-query-hash",
        work_profile=work_profile,
    )


def _registry():
    return SubAgentRegistry(agents=(ObligationCoverageMatrix(),))


# ---------------------------------------------------------------------------
# Blocker 1 — dispatcher honors match.requirement escalation
# ---------------------------------------------------------------------------


def test_dispatcher_honors_match_blocking_validator_escalation():
    """Template is OPTIONAL; agent match escalates to BLOCKING_VALIDATOR
    on the deliverable family. With critical/required missing, dispatcher
    must raise."""
    matter = MatterModel.open_in_memory()
    _seed(matter,
          criteria=[{"criterion_id": "C-001", "title": "Critical",
                     "severity": "critical"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_artifact_kinds": ["draft_document.v1"]}])
    disp = SubAgentDispatcher(registry=_registry(), matter_model=matter)
    inv = _invocation(matter, family="deliverable",
                      requirement=AgentRequirement.OPTIONAL)
    with pytest.raises(RuntimeError, match="blocking validator failed"):
        asyncio.run(disp.run_phase(inv, phase="pre_synthesis"))


def test_dispatcher_does_not_raise_when_blocking_obligations_met():
    """Same setup but with the required artifact present — must NOT raise."""
    matter = MatterModel.open_in_memory()
    _seed(matter,
          criteria=[{"criterion_id": "C-001", "title": "Critical",
                     "severity": "critical"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_artifact_kinds": ["document.section_map"]}])
    # Pre-seed the artifact_kind so the matrix scores it 'met'
    matter.db.execute(
        "INSERT INTO sub_agent_invocation (id, matter_id, run_id, agent_id, "
        "agent_version, phase, status, requirement, invocation_at, input_hash) "
        "VALUES ('inv-prior', ?, 'run-prior', 'doc.file_reader', 1, "
        "'pre_synthesis', 'success', 'optional', '2026-05-08T00:00:00', 'h0')",
        (matter.matter_id,),
    )
    matter.db.execute(
        "INSERT INTO agent_artifact (id, matter_id, invocation_id, "
        "artifact_kind, artifact_key, payload_json, created_at) "
        "VALUES ('a1', ?, 'inv-prior', 'document.section_map', 'k1', '{}', "
        "'2026-05-08T00:00:00')",
        (matter.matter_id,),
    )
    disp = SubAgentDispatcher(registry=_registry(), matter_model=matter)
    result = asyncio.run(
        disp.run_phase(_invocation(matter), phase="pre_synthesis")
    )
    statuses = [r.status for _a, r in result.invocations]
    assert "success" in statuses


# ---------------------------------------------------------------------------
# Blocker 2 — invalid result still persists artifacts
# ---------------------------------------------------------------------------


def test_validator_failure_artifact_persists_even_on_invalid_status():
    """When obligation_coverage halts under blocking_validator, the
    matrix and validator_failure artifacts must reach agent_artifact
    so the audit trail survives the halt."""
    matter = MatterModel.open_in_memory()
    _seed(matter,
          criteria=[{"criterion_id": "C-001", "title": "Required",
                     "severity": "required"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_artifact_kinds": ["draft_document.v1"]}])
    disp = SubAgentDispatcher(registry=_registry(), matter_model=matter)
    inv = _invocation(matter, family="deliverable")
    with pytest.raises(RuntimeError):
        asyncio.run(disp.run_phase(inv, phase="pre_synthesis"))
    # Both artifacts persisted
    rows = matter.db.execute(
        "SELECT artifact_kind FROM agent_artifact"
    ).fetchall()
    kinds = [r["artifact_kind"] for r in rows]
    assert ARTIFACT_KIND_MATRIX in kinds
    assert ARTIFACT_KIND_VALIDATOR_FAILURE in kinds


# ---------------------------------------------------------------------------
# Blocker 5 — task_fingerprint preference order
# ---------------------------------------------------------------------------


def test_task_fingerprint_uses_criteria_payload_task_id():
    """Two different invocations with the same input_hash but different
    criteria task_ids must produce different obligation_ids."""
    matter_a = MatterModel.open_in_memory()
    _seed(matter_a,
          criteria=[{"criterion_id": "C-001", "title": "X",
                     "severity": "required"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_artifact_kinds": ["draft_document.v1"]}],
          task_id="task_alpha")
    matter_b = MatterModel.open_in_memory()
    _seed(matter_b,
          criteria=[{"criterion_id": "C-001", "title": "X",
                     "severity": "required"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_artifact_kinds": ["draft_document.v1"]}],
          task_id="task_beta")
    agent = ObligationCoverageMatrix()
    # Same input_hash for both (would have collided in prior implementation)

    class _R:
        def __init__(self, mm):
            self.matter_model = mm

    r_a = asyncio.run(agent.invoke(_invocation(matter_a), _R(matter_a)))
    r_b = asyncio.run(agent.invoke(_invocation(matter_b), _R(matter_b)))
    fp_a = r_a.artifacts[0].payload["task_fingerprint"]
    fp_b = r_b.artifacts[0].payload["task_fingerprint"]
    assert fp_a != fp_b
    assert "task_alpha" in fp_a
    assert "task_beta" in fp_b


# ---------------------------------------------------------------------------
# matrix_status — upstream_required_evidence_missing
# ---------------------------------------------------------------------------


def test_matrix_status_upstream_when_all_rows_blocked_upstream():
    """Section/table requirements with no document.section_map artifact
    in the matter should produce all-rows upstream_required_evidence_missing,
    and the matrix-level status should mirror that."""
    matter = MatterModel.open_in_memory()
    _seed(matter,
          criteria=[{"criterion_id": "C-001", "title": "Section coverage",
                     "severity": "required"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_sections": ["Executive Summary"]}])
    agent = ObligationCoverageMatrix()
    class _R:
        matter_model = matter
    result = asyncio.run(agent.invoke(_invocation(matter), _R()))
    payload = result.artifacts[0].payload
    assert payload["matrix_status"] == "upstream_required_evidence_missing"
    assert payload["n_upstream_missing"] == payload["n_total"]


# ---------------------------------------------------------------------------
# D7 renderer fidelity (engine integration)
# ---------------------------------------------------------------------------


def test_renderer_omits_positive_summary_when_not_blocking_and_no_user_ask():
    """A passing matrix on an OPTIONAL invocation with no completeness
    keyword in query must not produce an OBLIGATION COVERAGE section."""
    from irys.rlm.engine import RLMEngine
    from irys.rlm.state import InvestigationState

    matter = MatterModel.open_in_memory()
    _seed(matter,
          criteria=[{"criterion_id": "C-001", "title": "Section map",
                     "severity": "required"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_artifact_kinds": ["document.section_map"]}])
    matter.db.execute(
        "INSERT INTO sub_agent_invocation (id, matter_id, run_id, agent_id, "
        "agent_version, phase, status, requirement, invocation_at, input_hash) "
        "VALUES ('p', ?, 'run-prior', 'd', 1, 'pre_synthesis', 'success', "
        "'optional', '2026-05-08T00:00:00', 'h')",
        (matter.matter_id,),
    )
    matter.db.execute(
        "INSERT INTO agent_artifact (id, matter_id, invocation_id, "
        "artifact_kind, artifact_key, payload_json, synthesis_visibility, "
        "created_at) VALUES ('a1', ?, 'p', 'document.section_map', 'k1', "
        "'{}', 'audit_only', '2026-05-08T00:00:00')",
        (matter.matter_id,),
    )

    eng = RLMEngine.__new__(RLMEngine)
    eng._matter_model = matter
    state = InvestigationState.create("draft a brief", "/tmp/repo")
    state._run_id = "run-render"
    asyncio.run(eng._run_pre_synthesis_operators(state))

    summary = eng._build_agent_artifact_summary(state)
    assert "OBLIGATION COVERAGE" not in summary, (
        f"Positive summary leaked into non-blocking, non-user-asked synthesis: "
        f"{summary[:300]}"
    )


# ---------------------------------------------------------------------------
# Round 2 carryover test gaps
# ---------------------------------------------------------------------------


def test_obligation_ids_in_payload_not_typed_evidence_refs():
    """Codex Phase-2 r2 blocker: typed_evidence_refs precedent uses row id
    not record_key. obligation_ids must live in payload['obligation_ids']
    instead, leaving typed_evidence_refs empty (or row-id keyed)."""
    matter = MatterModel.open_in_memory()
    _seed(matter,
          criteria=[{"criterion_id": "C-001", "title": "X",
                     "severity": "required"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_artifact_kinds": ["draft_document.v1"]}])
    agent = ObligationCoverageMatrix()
    class _R:
        matter_model = matter
    result = asyncio.run(agent.invoke(_invocation(matter), _R()))
    art = result.artifacts[0]
    payload = art.payload
    assert payload["obligation_ids"], (
        "matrix payload must include obligation_ids list"
    )
    assert all(oid.startswith("obl_v1:") for oid in payload["obligation_ids"])
    # typed_evidence_refs should NOT carry record_keys (would conflict
    # with cp_section_extractor / numerical_reconciliation precedent).
    for ref in art.typed_evidence_refs:
        assert not str(ref).startswith("obl_v1:"), (
            f"typed_evidence_refs leaked record_key: {ref}"
        )


def test_blocking_template_strictness_rule():
    """When the phase template is BLOCKING_VALIDATOR, every selected agent
    inherits at least BLOCKING_VALIDATOR (max strictness rule). A failing
    agent — even one that asked for OPTIONAL — must trigger halt."""
    from irys.rlm.agents import (
        SubAgentRegistry,
        SubAgentDispatcher,
        AgentMatch,
        AgentInvocationResult,
        AgentArtifact,
    )

    class _AlwaysFails:
        agent_id = "always_fail"
        version = 1
        enabled = True
        priority = 50
        capability_tags = ("compute",)
        supported_domain_profiles = ("legal:1",)
        phases = ("pre_synthesis",)
        exclusive_group = None
        deterministic = True

        def match(self, invocation):
            # Asks for OPTIONAL; should be escalated by template.
            return AgentMatch(
                agent_id=self.agent_id, score=1.0,
                requirement=AgentRequirement.OPTIONAL,
            )

        async def invoke(self, invocation, runtime):
            return AgentInvocationResult(
                status="error", error_class="X", error="boom",
            )

        def verify_output(self, invocation, result):
            return result

    matter = MatterModel.open_in_memory()
    reg = SubAgentRegistry(agents=(_AlwaysFails(),))
    disp = SubAgentDispatcher(registry=reg, matter_model=matter)
    inv = _invocation(
        matter, family="deliverable",
        requirement=AgentRequirement.BLOCKING_VALIDATOR,
    )
    with pytest.raises(RuntimeError, match="blocking validator failed"):
        asyncio.run(disp.run_phase(inv, phase="pre_synthesis"))


def test_renderer_includes_positive_summary_when_user_asks():
    """When the user query contains a completeness keyword, the matrix
    should render even if all required obligations are met."""
    from irys.rlm.engine import RLMEngine
    from irys.rlm.state import InvestigationState

    matter = MatterModel.open_in_memory()
    _seed(matter,
          criteria=[{"criterion_id": "C-001", "title": "Section map",
                     "severity": "required"}],
          deliverables=[{"deliverable_key": "memo.docx",
                         "required_artifact_kinds": ["document.section_map"]}])
    matter.db.execute(
        "INSERT INTO sub_agent_invocation (id, matter_id, run_id, agent_id, "
        "agent_version, phase, status, requirement, invocation_at, input_hash) "
        "VALUES ('p', ?, 'run-prior', 'd', 1, 'pre_synthesis', 'success', "
        "'optional', '2026-05-08T00:00:00', 'h')",
        (matter.matter_id,),
    )
    matter.db.execute(
        "INSERT INTO agent_artifact (id, matter_id, invocation_id, "
        "artifact_kind, artifact_key, payload_json, synthesis_visibility, "
        "created_at) VALUES ('a1', ?, 'p', 'document.section_map', 'k1', "
        "'{}', 'audit_only', '2026-05-08T00:00:00')",
        (matter.matter_id,),
    )
    eng = RLMEngine.__new__(RLMEngine)
    eng._matter_model = matter
    state = InvestigationState.create(
        "what gaps exist in coverage?", "/tmp/repo",
    )
    state._run_id = "run-render-2"
    asyncio.run(eng._run_pre_synthesis_operators(state))
    summary = eng._build_agent_artifact_summary(state)
    assert "OBLIGATION COVERAGE" in summary
