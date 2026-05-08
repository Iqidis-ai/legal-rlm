"""Tests for CrossDocLinker — cross-document linking operator."""

from __future__ import annotations

import asyncio
import json

import pytest

from irys.matter import MatterModel
from irys.rlm.agents import (
    AgentInvocation, AgentRequirement, AgentTaskView,
    CrossDocLinker, OperatorBudget,
    SubAgentDispatcher, SubAgentRegistry,
)


def _seed_section_map(matter, *, doc_path, sections, invocation_id="inv_test"):
    """Insert a fake section_map agent_artifact for the linker to read."""
    matter.db.execute(
        """INSERT OR IGNORE INTO sub_agent_invocation
           (id, matter_id, run_id, agent_id, agent_version, persona_id,
            phase, requirement, invocation_at, input_hash, latency_ms,
            cost_estimate, llm_calls, success_bool, status,
            capability_tags_json, token_estimate)
           VALUES (?, ?, '', 'document.file_reader.v1', 1, NULL,
                   'pre_synthesis', 'optional', '2026-05-08T00:00:00', '', 0,
                   0.0, 0, 1, 'success', '[]', 0)""",
        (invocation_id, matter.matter_id),
    )
    payload = {
        "schema_ref": "agent.document.section_map.v1",
        "document_id": doc_path,
        "document_path": doc_path,
        "sections": sections,
        "n_sections": len(sections),
    }
    import uuid
    artifact_id = uuid.uuid4().hex
    matter.db.execute(
        """INSERT INTO agent_artifact
           (id, matter_id, invocation_id, artifact_kind, artifact_key,
            label, payload_json, synthesis_visibility, confidence,
            source_refs_json, typed_evidence_refs_json,
            verification_state, created_at)
           VALUES (?, ?, ?, 'document.section_map', ?, ?, ?,
                   'audit_only', 0.85, '[]', '[]', 'verified',
                   '2026-05-08T00:00:00')""",
        (artifact_id, matter.matter_id, invocation_id,
         f"section_map:{doc_path}",
         f"Section map [{doc_path}]: {len(sections)} sections",
         json.dumps(payload)),
    )


def _seed_schedule_index(matter, *, doc_path, schedules, invocation_id="inv_test"):
    payload = {
        "schema_ref": "agent.document.schedule_index.v1",
        "document_id": doc_path,
        "document_path": doc_path,
        "schedules": schedules,
        "n_schedules": len(schedules),
    }
    import uuid
    artifact_id = uuid.uuid4().hex
    matter.db.execute(
        """INSERT INTO agent_artifact
           (id, matter_id, invocation_id, artifact_kind, artifact_key,
            label, payload_json, synthesis_visibility, confidence,
            source_refs_json, typed_evidence_refs_json,
            verification_state, created_at)
           VALUES (?, ?, ?, 'document.schedule_index', ?, ?, ?,
                   'audit_only', 0.85, '[]', '[]', 'verified',
                   '2026-05-08T00:00:00')""",
        (artifact_id, matter.matter_id, invocation_id,
         f"schedule_index:{doc_path}",
         f"Schedules [{doc_path}]: {len(schedules)} entries",
         json.dumps(payload)),
    )


def _make_invocation(matter_id):
    return AgentInvocation(
        matter_id=matter_id, run_id="r",
        agent_id="linker", phase="pre_synthesis",
        persona_id=None, requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(),
        execution_family="investigate", workflow_kind="default",
        budget=OperatorBudget(),
        input_refs=(), input_hash="h",
    )


# ---------------------------------------------------------------------------
# Empty path
# ---------------------------------------------------------------------------


def test_linker_no_artifacts_returns_success_zero_artifacts():
    m = MatterModel.open_in_memory()
    agent = CrossDocLinker()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    assert result.status == "success"
    assert result.artifacts == ()


# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------


def test_linker_resolves_section_to_defining_document():
    m = MatterModel.open_in_memory()
    # First, set up sub_agent_invocation row dependency
    inv_id = "inv1"
    m.db.execute(
        """INSERT INTO sub_agent_invocation
           (id, matter_id, run_id, agent_id, agent_version, persona_id,
            phase, requirement, invocation_at, input_hash, latency_ms,
            cost_estimate, llm_calls, success_bool, status,
            capability_tags_json, token_estimate)
           VALUES (?, ?, '', 'reader', 1, NULL, 'pre_synthesis', 'optional',
                   '2026-05-08T00:00:00', '', 0, 0.0, 0, 1, 'success',
                   '[]', 0)""",
        (inv_id, m.matter_id),
    )
    _seed_section_map(
        m, doc_path="agreement.pdf",
        sections=[
            {"label": "Section 4.01", "title": "CP", "line_start": 100, "depth": 2},
            {"label": "Section 4.01(a)", "title": "Officer Cert", "line_start": 105, "depth": 3},
            {"label": "Section 5.02", "title": "Negative Covenants", "line_start": 200, "depth": 2},
        ],
        invocation_id=inv_id,
    )
    agent = CrossDocLinker()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    by_kind = {a.artifact_kind: a for a in result.artifacts}
    assert "link.cross_reference" in by_kind
    p = by_kind["link.cross_reference"].payload
    assert p["n_links"] == 3
    assert p["n_ambiguous"] == 0
    labels = {l["section_label"] for l in p["links"]}
    assert "section 4.01" in labels


def test_linker_flags_ambiguous_sections_across_documents():
    m = MatterModel.open_in_memory()
    inv_id = "inv1"
    m.db.execute(
        """INSERT INTO sub_agent_invocation
           (id, matter_id, run_id, agent_id, agent_version, persona_id,
            phase, requirement, invocation_at, input_hash, latency_ms,
            cost_estimate, llm_calls, success_bool, status,
            capability_tags_json, token_estimate)
           VALUES (?, ?, '', 'reader', 1, NULL, 'pre_synthesis', 'optional',
                   '2026-05-08T00:00:00', '', 0, 0.0, 0, 1, 'success',
                   '[]', 0)""",
        (inv_id, m.matter_id),
    )
    # Same Section 4.01 in two docs → ambiguous
    _seed_section_map(m, doc_path="credit_agreement.pdf",
                      sections=[{"label": "Section 4.01", "title": "CP",
                                 "line_start": 100, "depth": 2}],
                      invocation_id=inv_id)
    _seed_section_map(m, doc_path="amendment.pdf",
                      sections=[{"label": "Section 4.01", "title": "CP",
                                 "line_start": 50, "depth": 2}],
                      invocation_id=inv_id)
    agent = CrossDocLinker()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    by_kind = {a.artifact_kind: a for a in result.artifacts}
    p = by_kind["link.cross_reference"].payload
    assert p["n_ambiguous"] == 1
    assert p["ambiguous"][0]["section_label"] == "section 4.01"
    assert by_kind["link.cross_reference"].verification_state == "candidate"


# ---------------------------------------------------------------------------
# Schedule traversal
# ---------------------------------------------------------------------------


def test_linker_traverses_schedules_to_agreements():
    m = MatterModel.open_in_memory()
    inv_id = "inv1"
    m.db.execute(
        """INSERT INTO sub_agent_invocation
           (id, matter_id, run_id, agent_id, agent_version, persona_id,
            phase, requirement, invocation_at, input_hash, latency_ms,
            cost_estimate, llm_calls, success_bool, status,
            capability_tags_json, token_estimate)
           VALUES (?, ?, '', 'reader', 1, NULL, 'pre_synthesis', 'optional',
                   '2026-05-08T00:00:00', '', 0, 0.0, 0, 1, 'success',
                   '[]', 0)""",
        (inv_id, m.matter_id),
    )
    _seed_schedule_index(
        m, doc_path="credit_agreement.pdf",
        schedules=[
            {"kind": "Schedule", "ref": "3.01(a)", "label": "Schedule 3.01(a)",
             "first_offset": 1000, "line_start": 50, "count": 3},
            {"kind": "Exhibit", "ref": "B", "label": "Exhibit B",
             "first_offset": 2000, "line_start": 100, "count": 1},
        ],
        invocation_id=inv_id,
    )
    agent = CrossDocLinker()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    by_kind = {a.artifact_kind: a for a in result.artifacts}
    assert "link.schedule_to_agreement" in by_kind
    p = by_kind["link.schedule_to_agreement"].payload
    assert p["n_links"] == 2
    labels = {t["schedule_label"] for t in p["traversals"]}
    assert "schedule 3.01(a)" in labels


# ---------------------------------------------------------------------------
# Entity unification
# ---------------------------------------------------------------------------


def test_linker_unifies_corporate_suffix_aliases():
    m = MatterModel.open_in_memory()
    # Seed actor table with two name variants
    m.db.execute(
        """INSERT INTO actor (id, matter_id, canonical_name, normalized_name,
                              actor_type, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'organization', '2026-05-08T00:00:00', '2026-05-08T00:00:00')""",
        ("a1", m.matter_id, "Meridian Industrial Gases, Inc.",
         "meridian industrial gases inc"),
    )
    m.db.execute(
        """INSERT INTO actor (id, matter_id, canonical_name, normalized_name,
                              actor_type, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'organization', '2026-05-08T00:00:00', '2026-05-08T00:00:00')""",
        ("a2", m.matter_id, "Meridian Industrial Gases LLC",
         "meridian industrial gases llc"),
    )
    m.db.execute(
        """INSERT INTO actor (id, matter_id, canonical_name, normalized_name,
                              actor_type, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'organization', '2026-05-08T00:00:00', '2026-05-08T00:00:00')""",
        ("a3", m.matter_id, "Other Corp", "other corp"),
    )
    agent = CrossDocLinker()
    reg = SubAgentRegistry(agents=(agent,))
    disp = SubAgentDispatcher(registry=reg, matter_model=m)
    out = asyncio.run(disp.run_phase(_make_invocation(m.matter_id), phase="pre_synthesis"))
    _, result = out.invocations[0]
    by_kind = {a.artifact_kind: a for a in result.artifacts}
    assert "link.entity_unification" in by_kind
    p = by_kind["link.entity_unification"].payload
    # Two Meridian variants unified into one group
    assert p["n_links"] == 1
    grp = p["unifications"][0]
    assert grp["n_aliases"] == 2
    # "Other Corp" alone — single-member group not emitted


def test_linker_capability_tags():
    a = CrossDocLinker()
    assert "link.cross_reference" in a.capability_tags
    assert "link.schedule_traversal" in a.capability_tags
    assert "link.entity_unification" in a.capability_tags


def test_linker_in_default_registry():
    from irys.rlm.agents import default_registry
    reg = default_registry()
    ids = {a.agent_id for a in reg.list()}
    assert "link.cross_document_linker.v1" in ids
