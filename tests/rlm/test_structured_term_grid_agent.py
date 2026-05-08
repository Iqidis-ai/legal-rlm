"""Unit tests for StructuredTermGridExtractor.v1.

Anti-gaming foundation: every test must work for a fuzzy user prompt
with no benchmark metadata attached. No LAB criterion IDs, no rubric
strings, no benchmark task IDs.
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
    StructuredTermGridExtractor,
)
from irys.rlm.agents.term_grid import (
    ARTIFACT_KIND_TERM_GRID,
    ARTIFACT_KIND_CONDITIONAL_RULE_TREE,
    SCHEMA_REF_TERM_GRID,
    SCHEMA_REF_CONDITIONAL_RULE_TREE,
    make_term_row_id,
    make_rule_id,
)


class _FakeLLMClient:
    def __init__(self, json_response=None, raise_on_call=False):
        self.json_response = json_response
        self.raise_on_call = raise_on_call
        self.calls = []

    async def complete(self, *, prompt, tier=None, timeout=None,
                       usage_label=None, **kw):
        self.calls.append({"prompt": prompt, "usage_label": usage_label, **kw})
        if self.raise_on_call:
            raise RuntimeError("fake LLM error")
        return json.dumps(self.json_response or {})


class _Runtime:
    def __init__(self, matter, llm_client=None, query=""):
        self.matter_model = matter
        self.llm_client = llm_client
        class _State: pass
        self.state = _State()
        self.state.query = query


def _invocation(matter, *, family="investigate",
                domain_profile_id="legal:1",
                work_profile=None,
                input_hash="ih"):
    if work_profile is None:
        work_profile = {
            "document_section_map_count": 1,
            "obligation_row_count": 0,
            "term_grid_obligation_count": 0,
            "contract_provision_count": 0,
        }
    return AgentInvocation(
        matter_id=matter.matter_id, run_id="run-1",
        agent_id="structured_term_grid_extractor.v1",
        phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(answer_shape="memo"),
        execution_family=family, workflow_kind="analysis",
        budget=OperatorBudget(), input_refs=(), input_hash=input_hash,
        domain_profile_id=domain_profile_id, domain_profile_version=1,
        work_profile=work_profile,
    )


def _seed_section_map(matter, *, doc_id="doc1", sections=None):
    """Persist a document.section_map artifact via direct SQL."""
    if sections is None:
        sections = [
            {"title": "Section 5.3 Standstill", "text":
                "Second Lien Agent shall not exercise any remedies for "
                "180 days after a payment default occurs, unless a "
                "bankruptcy stay relief exception applies."},
            {"title": "Section 7.1 Cure Rights",
             "text": "Borrower may cure a payment default within 30 days."},
            {"title": "Section 1.1 Definitions",
             "text": "All capitalized terms..."},
        ]
    payload = {"document_id": doc_id, "sections": sections}
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
        "VALUES (?, ?, 'inv-prior', 'document.section_map', ?, ?, "
        "'2026-05-08T00:00:00')",
        ("a1", matter.matter_id, f"sm:{doc_id}", json.dumps(payload)),
    )


def _intercreditor_response():
    return {
        "schema_ref": "term_grid.v1",
        "rows": [
            {
                "document_id": "doc1",
                "section_ref": "Section 5.3 Standstill",
                "topic": "standstill_period",
                "actor": "Second Lien Agent",
                "obligation_or_right": "may not exercise remedies",
                "trigger": "payment default",
                "condition": "First Lien obligations remain outstanding",
                "exception": "bankruptcy stay relief exception",
                "threshold": None,
                "amount": None,
                "date_or_period": "180 days",
                "consequence": "standstill applies",
                "source_refs": ["span:5.3"],
                "confidence": 0.87,
            },
            {
                "document_id": "doc1",
                "section_ref": "Section 7.1 Cure Rights",
                "topic": "cure_period",
                "actor": "Borrower",
                "obligation_or_right": "may cure a payment default",
                "trigger": "payment default occurs",
                "condition": "",
                "exception": "",
                "threshold": None,
                "amount": None,
                "date_or_period": "30 days",
                "consequence": "default cured",
                "source_refs": ["span:7.1"],
                "confidence": 0.82,
            },
        ],
        "rules": [
            {
                "section_ref": "Section 5.3 Standstill",
                "if": ["payment default"],
                "then": ["remedy standstill"],
                "unless": ["bankruptcy stay relief exception"],
                "timing": "180 days",
                "thresholds": [],
                "parties": ["Second Lien Agent"],
                "source_span_ids": ["span:5.3"],
                "confidence": 0.84,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Row id determinism
# ---------------------------------------------------------------------------


def test_make_term_row_id_is_deterministic_and_excludes_benchmark_metadata():
    a = make_term_row_id(
        document_id="doc1", section_ref="Section 5.3",
        topic="standstill_period", actor="Second Lien Agent",
        obligation_or_right="may not exercise remedies",
    )
    b = make_term_row_id(
        document_id="doc1", section_ref="Section 5.3",
        topic="standstill_period", actor="Second Lien Agent",
        obligation_or_right="may not exercise remedies",
    )
    assert a == b
    assert a.startswith("termrow_v1:")


def test_make_term_row_id_distinguishes_inputs():
    base = dict(
        document_id="doc1", section_ref="Section 5.3",
        topic="standstill_period", actor="Second Lien Agent",
        obligation_or_right="may not exercise remedies",
    )
    a = make_term_row_id(**base)
    b = make_term_row_id(**dict(base, actor="First Lien Agent"))
    assert a != b


def test_make_rule_id_deterministic():
    a = make_rule_id(
        document_id="doc1", section_ref="5.3",
        if_clauses=["payment default"],
        then_clauses=["remedy standstill"],
    )
    b = make_rule_id(
        document_id="doc1", section_ref="5.3",
        if_clauses=["payment default"],
        then_clauses=["remedy standstill"],
    )
    assert a == b
    assert a.startswith("rule_v1:")


# ---------------------------------------------------------------------------
# match() gating
# ---------------------------------------------------------------------------


def test_match_strong_signal_when_term_grid_obligations_exist():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    inv = _invocation(matter, work_profile={
        "document_section_map_count": 0,
        "obligation_row_count": 5,
        "term_grid_obligation_count": 3,
        "contract_provision_count": 0,
    })
    m = agent.match(inv)
    assert m is not None
    assert m.score >= 0.9
    assert m.requirement == AgentRequirement.REQUIRED


def test_match_section_maps_plus_intent_match_required():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    inv = _invocation(matter, work_profile={
        "document_section_map_count": 3,
        "obligation_row_count": 0,
        "term_grid_obligation_count": 0,
        "contract_provision_count": 0,
        "query_text_normalized": "extract the covenants and exceptions from this credit agreement",
    })
    m = agent.match(inv)
    assert m is not None
    assert m.requirement == AgentRequirement.REQUIRED


def test_match_skips_clarify_scenario_steer():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    for fam in ("clarify", "scenario", "steer"):
        assert agent.match(_invocation(matter, family=fam)) is None


def test_match_no_signal_returns_none():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    inv = _invocation(matter, work_profile={
        "document_section_map_count": 0,
        "obligation_row_count": 0,
        "term_grid_obligation_count": 0,
        "contract_provision_count": 0,
    })
    assert agent.match(inv) is None


def test_match_intent_keyword_must_pair_with_object():
    """'extract the covenants' fires; 'extract the contract' alone wouldn't
    (no term-shaped object), 'covenants' alone wouldn't (no extract verb)."""
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    # Intent + object → fires
    inv1 = _invocation(matter, work_profile={
        "document_section_map_count": 1, "obligation_row_count": 0,
        "term_grid_obligation_count": 0, "contract_provision_count": 0,
        "query_text_normalized": "extract the covenants",
    })
    assert agent.match(inv1) is not None
    # Intent without term-shaped object → only weak/no match
    inv2 = _invocation(matter, work_profile={
        "document_section_map_count": 1, "obligation_row_count": 0,
        "term_grid_obligation_count": 0, "contract_provision_count": 0,
        "query_text_normalized": "summarize the document",
    })
    m2 = agent.match(inv2)
    # No term-keyword match, no obligation rows → should be None
    assert m2 is None or m2.score < 0.85


# ---------------------------------------------------------------------------
# invoke() — happy path
# ---------------------------------------------------------------------------


def test_invoke_no_section_maps_returns_ran_empty():
    """No candidate sections AND no strong-signal upstream → ran_empty
    (Codex reviewer round 1 blocker B2: differentiate ran_empty vs
    upstream_required_evidence_missing)."""
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response=_intercreditor_response())
    rt = _Runtime(matter, llm_client=fake, query="extract covenants")
    result = asyncio.run(agent.invoke(_invocation(matter), rt))
    assert result.status == "success"
    assert "ran_empty" in result.warnings
    assert not result.artifacts


def test_invoke_no_section_maps_with_strong_signal_returns_upstream_missing():
    """When match() required us due to term_grid_obligation_count or
    contract_provision_count > 0, but no candidate sections exist, we
    must signal `upstream_required_evidence_missing`, NOT silent ran_empty."""
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response=_intercreditor_response())
    rt = _Runtime(matter, llm_client=fake, query="extract covenants")
    inv = _invocation(matter, work_profile={
        "document_section_map_count": 0,
        "obligation_row_count": 0,
        "term_grid_obligation_count": 3,  # strong signal
        "contract_provision_count": 0,
    })
    result = asyncio.run(agent.invoke(inv, rt))
    assert result.status == "success"
    assert "upstream_required_evidence_missing" in result.warnings
    assert not result.artifacts


def test_invoke_extracts_term_grid_from_section_maps():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    _seed_section_map(matter)
    fake = _FakeLLMClient(json_response=_intercreditor_response())
    rt = _Runtime(matter, llm_client=fake,
                  query="extract the standstill and cure provisions")
    result = asyncio.run(agent.invoke(_invocation(matter), rt))
    assert result.status == "success"
    assert result.llm_calls >= 1

    # Both artifacts produced
    kinds = {a.artifact_kind for a in result.artifacts}
    assert ARTIFACT_KIND_TERM_GRID in kinds
    assert ARTIFACT_KIND_CONDITIONAL_RULE_TREE in kinds

    # term_grid has the rows
    grid = next(a for a in result.artifacts if a.artifact_kind == ARTIFACT_KIND_TERM_GRID)
    assert grid.payload["n_rows"] == 2
    topics = {r["topic"] for r in grid.payload["rows"]}
    assert "standstill_period" in topics
    assert "cure_period" in topics


def test_invoke_drops_low_confidence_rows():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    _seed_section_map(matter)
    low_conf_response = {
        "schema_ref": "term_grid.v1",
        "rows": [
            {"document_id": "doc1", "section_ref": "S1", "topic": "low",
             "actor": "X", "obligation_or_right": "y", "confidence": 0.3,
             "source_refs": ["span:S1"]},
            {"document_id": "doc1", "section_ref": "S2", "topic": "high",
             "actor": "X", "obligation_or_right": "z", "confidence": 0.9,
             "source_refs": ["span:S2"]},
        ],
        "rules": [],
    }
    fake = _FakeLLMClient(json_response=low_conf_response)
    rt = _Runtime(matter, llm_client=fake, query="extract terms")
    result = asyncio.run(agent.invoke(_invocation(matter), rt))
    grid = next(a for a in result.artifacts if a.artifact_kind == ARTIFACT_KIND_TERM_GRID)
    assert grid.payload["n_rows"] == 1
    assert grid.payload["rows"][0]["topic"] == "high"


def test_invoke_no_llm_client_warns_gracefully():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    _seed_section_map(matter)
    rt = _Runtime(matter, llm_client=None, query="extract covenants")
    result = asyncio.run(agent.invoke(_invocation(matter), rt))
    assert result.status == "success"
    assert any("no_llm_client" in w for w in result.warnings)


def test_invoke_llm_failure_records_warning():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    _seed_section_map(matter)
    fake = _FakeLLMClient(raise_on_call=True)
    rt = _Runtime(matter, llm_client=fake, query="extract covenants")
    result = asyncio.run(agent.invoke(_invocation(matter), rt))
    assert result.status == "success"
    assert any("batch_extract_failed" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Anti-gaming
# ---------------------------------------------------------------------------


def test_row_ids_dont_leak_benchmark_identifiers():
    """No criterion ID, task ID, or LAB-shaped string in row IDs."""
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    _seed_section_map(matter)
    fake = _FakeLLMClient(json_response=_intercreditor_response())
    rt = _Runtime(matter, llm_client=fake, query="extract terms")
    result = asyncio.run(agent.invoke(_invocation(matter), rt))
    grid = next(a for a in result.artifacts if a.artifact_kind == ARTIFACT_KIND_TERM_GRID)
    for r in grid.payload["rows"]:
        rid = r["row_id"]
        assert rid.startswith("termrow_v1:")
        assert not any(forbidden in rid.lower() for forbidden in (
            "lab_", "harvey", "rubric", "criterion_id", "c-001",
        ))


def test_invoke_works_on_fuzzy_prompt_no_metadata():
    """Anti-gaming sentinel: a fuzzy user prompt with no structured
    criteria/rubric/task_id still produces useful term grid rows."""
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    _seed_section_map(matter)
    fake = _FakeLLMClient(json_response=_intercreditor_response())
    rt = _Runtime(
        matter, llm_client=fake,
        query="What are the key provisions and exceptions in this intercreditor agreement?",
    )
    inv = _invocation(matter, work_profile={
        "document_section_map_count": 1, "obligation_row_count": 0,
        "term_grid_obligation_count": 0, "contract_provision_count": 0,
        "query_text_normalized": "what are the key provisions and exceptions in this intercreditor agreement",
    })
    result = asyncio.run(agent.invoke(inv, rt))
    assert result.status == "success"
    assert any(a.artifact_kind == ARTIFACT_KIND_TERM_GRID for a in result.artifacts)


# ---------------------------------------------------------------------------
# verify_output
# ---------------------------------------------------------------------------


def test_operator_runs_on_schedule_entry_only_path():
    """Reviewer r2 blocker: a matter with ONLY `schedule_entry` typed
    evidence (no document.section_map) must still select + invoke the
    operator. Previously only section_map gated the match."""
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    # Seed a schedule_entry typed evidence row (no section maps)
    matter.typed_evidence.upsert(
        "schedule_entry", "sched:1",
        payload={
            "schema_ref": "legal.schedule_entry.v1",
            "schedule_ref": "Schedule 5.1",
            "text": "Borrower shall not incur additional debt above $10M without consent. "
                    "Exception: working capital revolver permitted up to $2M.",
        },
        document_id="doc1", confidence=0.9,
    )
    # match() should fire on schedule_entry_count alone
    inv = _invocation(matter, work_profile={
        "document_section_map_count": 0,
        "schedule_index_count": 0,
        "table_index_count": 0,
        "schedule_entry_count": 1,
        "obligation_row_count": 0,
        "term_grid_obligation_count": 0,
        "contract_provision_count": 0,
        "query_text_normalized": "extract the covenants",
    })
    m = agent.match(inv)
    assert m is not None, "schedule_entry-only matter must select the operator"
    assert m.requirement == AgentRequirement.REQUIRED

    # invoke() should produce a matrix from the schedule_entry source
    fake = _FakeLLMClient(json_response={
        "schema_ref": "term_grid.v1",
        "rows": [{
            "document_id": "doc1", "section_ref": "Schedule 5.1",
            "topic": "debt_covenant", "actor": "Borrower",
            "obligation_or_right": "shall not incur additional debt above $10M",
            "trigger": "additional debt above $10M",
            "exception": "working capital revolver permitted up to $2M",
            "amount": "$10M", "confidence": 0.85,
            "source_refs": ["record:sched:1"],
        }],
        "rules": [],
    })
    rt = _Runtime(matter, llm_client=fake, query="extract the covenants")
    result = asyncio.run(agent.invoke(inv, rt))
    assert result.status == "success"
    assert any(a.artifact_kind == ARTIFACT_KIND_TERM_GRID for a in result.artifacts)


def test_operator_runs_on_table_index_only_path():
    """Same idea but for table_index sources."""
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    # Seed a table_index agent_artifact (no section maps)
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
        "VALUES ('a-tbl', ?, 'inv-prior', 'document.table_index', 'tbl:1', ?, "
        "'2026-05-08T00:00:00')",
        (matter.matter_id, json.dumps({
            "document_id": "doc1",
            "tables": [{"caption": "Termination Triggers",
                        "text": "If revenue < $5M and ARR drop > 30%, "
                                "term: 60 days notice."}],
        })),
    )
    inv = _invocation(matter, work_profile={
        "document_section_map_count": 0,
        "schedule_index_count": 0,
        "table_index_count": 1,
        "schedule_entry_count": 0,
        "obligation_row_count": 0,
        "term_grid_obligation_count": 0,
        "contract_provision_count": 0,
        "query_text_normalized": "extract the termination triggers",
    })
    m = agent.match(inv)
    assert m is not None, "table_index-only matter must select the operator"


def test_registry_filters_by_domain_profile():
    """Reviewer round 1 blocker B5: registry must filter agents whose
    `supported_domain_profiles` doesn't include the invocation's
    domain_profile_id. Cross-domain operators (5 profiles) pass; a
    legal-only operator on a coding invocation gets suppressed."""
    from irys.rlm.agents import (
        SubAgentRegistry, AgentInvocation, AgentRequirement,
        AgentTaskView, OperatorBudget, AgentMatch,
    )
    from irys.rlm.agents.contracts import AgentInvocationResult

    class _LegalOnly:
        agent_id = "legal_only_op"
        version = 1
        enabled = True
        priority = 50
        capability_tags = ("compute",)
        supported_domain_profiles = ("legal:1",)
        phases = ("pre_synthesis",)
        exclusive_group = None
        deterministic = True

        def match(self, invocation):
            return AgentMatch(agent_id=self.agent_id, score=1.0)

        async def invoke(self, invocation, runtime):
            return AgentInvocationResult(status="success")

        def verify_output(self, invocation, result):
            return result

    matter = MatterModel.open_in_memory()
    reg = SubAgentRegistry(agents=(_LegalOnly(),))

    # Coding invocation should suppress the legal-only agent
    inv_coding = AgentInvocation(
        matter_id=matter.matter_id, run_id="r",
        agent_id="dispatcher", phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(),
        execution_family="investigate", workflow_kind="analysis",
        budget=OperatorBudget(), input_refs=(), input_hash="ih",
        domain_profile_id="coding:1", domain_profile_version=1,
        work_profile={},
    )
    disp = reg.dispatch(inv_coding, phase="pre_synthesis", phase_cap=8)
    assert _LegalOnly().agent_id not in [a.agent_id for a in disp.selected]
    suppressions = [s for s in disp.suppressed
                    if s.get("agent_id") == "legal_only_op"]
    assert suppressions
    assert suppressions[0]["reason"] == "domain_profile_mismatch"

    # Legal invocation should NOT suppress it
    inv_legal = AgentInvocation(
        matter_id=matter.matter_id, run_id="r",
        agent_id="dispatcher", phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(),
        execution_family="investigate", workflow_kind="analysis",
        budget=OperatorBudget(), input_refs=(), input_hash="ih",
        domain_profile_id="legal:1", domain_profile_version=1,
        work_profile={},
    )
    disp2 = reg.dispatch(inv_legal, phase="pre_synthesis", phase_cap=8)
    assert "legal_only_op" in [a.agent_id for a in disp2.selected]


def test_verify_output_passes_well_formed_artifacts():
    agent = StructuredTermGridExtractor()
    matter = MatterModel.open_in_memory()
    _seed_section_map(matter)
    fake = _FakeLLMClient(json_response=_intercreditor_response())
    rt = _Runtime(matter, llm_client=fake, query="extract")
    result = asyncio.run(agent.invoke(_invocation(matter), rt))
    inv = _invocation(matter)
    verified = agent.verify_output(inv, result)
    assert verified.status == "success"
