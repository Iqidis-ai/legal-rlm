"""Tier 5 (prompt-inferred criteria) regression tests.

Per Codex T5-D9 anti-gaming validation: every test here must demonstrate
that the operator works WITHOUT pre-supplied benchmark metadata. If any
test relies on pre-injected typed_evidence_record(task_criteria) or
ExecutionContract.output_contract.criteria, it does NOT belong here.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from irys.matter import MatterModel
from irys.rlm.agents import (
    AgentInvocation,
    AgentRequirement,
    AgentTaskView,
    OperatorBudget,
    ObligationCoverageMatrix,
)
from irys.rlm.agents.obligation_coverage import (
    SCHEMA_REF_TASK_CRITERIA,
    RECORD_KIND_TASK_CRITERIA,
    RECORD_KIND_OBLIGATION_ROW,
    ARTIFACT_KIND_MATRIX,
    TIER5_INFERENCE_VERSION,
    _tier5_inference_cache_key,
)


# ---------------------------------------------------------------------------
# Fake JSON-mode LLM client
# ---------------------------------------------------------------------------


class _FakeLLMClient:
    """Records prompts; returns a canned JSON string per fixture."""

    def __init__(self, json_response: dict | None = None,
                 string_response: str | None = None,
                 raise_on_call: bool = False):
        self.json_response = json_response
        self.string_response = string_response
        self.raise_on_call = raise_on_call
        self.calls: list[dict] = []

    async def complete(self, *, prompt, tier=None, timeout=None,
                       usage_label=None, **kw):
        self.calls.append({
            "prompt": prompt, "tier": tier,
            "timeout": timeout, "usage_label": usage_label,
        })
        if self.raise_on_call:
            raise RuntimeError("fake LLM error")
        if self.string_response is not None:
            return self.string_response
        return json.dumps(self.json_response or {})


class _Runtime:
    def __init__(self, matter, llm_client=None, query=""):
        self.matter_model = matter
        self.llm_client = llm_client
        # Mimic the engine-shaped state attribute the operator looks at.
        class _State:
            pass
        self.state = _State()
        self.state.query = query

    def domain_profile_id(self):
        return "legal:1"


def _invocation(matter, *, family="deliverable",
                requirement=AgentRequirement.OPTIONAL,
                workflow_kind="drafting",
                answer_shape="memo",
                work_profile=None,
                input_hash="iqh"):
    if work_profile is None:
        work_profile = {
            "task_criteria_count": 0,
            "task_deliverable_spec_count": 0,
            "obligation_row_count": 0,
        }
    return AgentInvocation(
        matter_id=matter.matter_id, run_id="run-1",
        agent_id="obligation_coverage_matrix",
        phase="pre_synthesis", persona_id=None,
        requirement=requirement,
        task=AgentTaskView(answer_shape=answer_shape),
        execution_family=family,
        workflow_kind=workflow_kind,
        budget=OperatorBudget(),
        input_refs=(), input_hash=input_hash,
        domain_profile_id="legal:1",
        domain_profile_version=1,
        work_profile=work_profile,
    )


def _hospital_antitrust_response():
    return {
        "schema_ref": "task.criteria_inference.v1",
        "inference_version": TIER5_INFERENCE_VERSION,
        "confidence": 0.78,
        "should_materialize_matrix": True,
        "vagueness_reason": "",
        "criteria": [
            {"criterion_id": "market_def",
             "title": "Define relevant markets",
             "description": "Address plausible product/geographic markets.",
             "severity": "required",
             "expected_deliverable_shape": "section"},
            {"criterion_id": "concentration",
             "title": "Assess concentration and structural risk",
             "description": "HHI/concentration evidence + structural presumption.",
             "severity": "required",
             "expected_deliverable_shape": "section"},
            {"criterion_id": "evidence_grounding",
             "title": "Ground conclusions in evidence",
             "description": "Cite available matter evidence and flag missing facts.",
             "severity": "critical",
             "expected_deliverable_shape": "citation_set"},
        ],
        "deliverables": [
            {"deliverable_key": "antitrust_memo",
             "format": "markdown",
             "required_sections": ["Issue", "Market definition",
                                    "Concentration", "Evidence gaps",
                                    "Conclusion"],
             "required_tables": [],
             "required_artifact_kinds": []},
        ],
    }


# ---------------------------------------------------------------------------
# Anti-gaming: Tier 5 produces useful obligations from a fuzzy prompt
# ---------------------------------------------------------------------------


def test_tier5_antitrust_memo_no_metadata():
    """No task_criteria, no task_deliverable_spec, no contract criteria,
    just a fuzzy prompt → Tier 5 fires once and produces real rows."""
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response=_hospital_antitrust_response())
    rt = _Runtime(
        matter, llm_client=fake,
        query="Draft me an antitrust memo on this hospital acquisition.",
    )
    agent = ObligationCoverageMatrix()
    result = asyncio.run(agent.invoke(_invocation(matter), rt))
    assert result.status == "success"
    assert result.llm_calls == 1, "Tier 5 must fire one bounded call"
    assert result.artifacts, "matrix should be materialized"

    matrix = next(
        (a for a in result.artifacts if a.artifact_kind == ARTIFACT_KIND_MATRIX),
        None,
    )
    assert matrix is not None
    p = matrix.payload
    assert p["criteria_source"] == "llm_inferred"
    assert p["n_total"] >= 3, f"expected >=3 rows, got {p['n_total']}"

    # Rows should reflect the inferred criteria
    titles = [
        (r.get("source_criterion") or {}).get("title", "")
        for r in p.get("rows", [])
    ]
    assert any("market" in t.lower() for t in titles)
    assert any("concentration" in t.lower() or "structural" in t.lower() for t in titles)


def test_tier5_coding_design_doc_no_metadata():
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response={
        "schema_ref": "task.criteria_inference.v1",
        "inference_version": TIER5_INFERENCE_VERSION,
        "confidence": 0.74,
        "should_materialize_matrix": True,
        "criteria": [
            {"title": "State target behavior", "severity": "required"},
            {"title": "Cover migration and failure modes", "severity": "required"},
            {"title": "Specify test and observability plan", "severity": "required"},
        ],
        "deliverables": [],
    })
    inv = AgentInvocation(
        matter_id=matter.matter_id, run_id="run-1",
        agent_id="obligation_coverage_matrix",
        phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(answer_shape="memo"),
        execution_family="deliverable", workflow_kind="drafting",
        budget=OperatorBudget(), input_refs=(), input_hash="auth_cache_doc",
        domain_profile_id="coding:1", domain_profile_version=1,
        work_profile={"task_criteria_count": 0,
                      "task_deliverable_spec_count": 0,
                      "obligation_row_count": 0},
    )
    rt = _Runtime(
        matter, llm_client=fake,
        query="Write a design doc for replacing the auth cache.",
    )
    result = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert result.status == "success"
    assert result.llm_calls == 1
    matrix = result.artifacts[0]
    assert matrix.payload["n_total"] >= 3


def test_tier5_biomedical_safety_summary_no_metadata():
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response={
        "schema_ref": "task.criteria_inference.v1",
        "inference_version": TIER5_INFERENCE_VERSION,
        "confidence": 0.72,
        "should_materialize_matrix": True,
        "criteria": [
            {"title": "Identify safety populations and exposure",
             "severity": "required"},
            {"title": "Report adverse events", "severity": "critical"},
            {"title": "Flag missing safety evidence", "severity": "required"},
        ],
        "deliverables": [],
    })
    inv = AgentInvocation(
        matter_id=matter.matter_id, run_id="run-1",
        agent_id="obligation_coverage_matrix",
        phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(answer_shape="memo"),
        execution_family="deliverable", workflow_kind="drafting",
        budget=OperatorBudget(), input_refs=(), input_hash="drug_x_safety",
        domain_profile_id="biomedical:1", domain_profile_version=1,
        work_profile={"task_criteria_count": 0,
                      "task_deliverable_spec_count": 0,
                      "obligation_row_count": 0},
    )
    rt = _Runtime(
        matter, llm_client=fake,
        query="Summarize Drug X safety from these trial docs.",
    )
    result = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert result.status == "success"
    assert result.llm_calls == 1


def test_tier5_query_does_not_overfire():
    """A pure factual lookup must NOT produce a matrix even though Tier 5
    is theoretically available. Family + answer-shape gate must catch it."""
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response=_hospital_antitrust_response())
    inv = AgentInvocation(
        matter_id=matter.matter_id, run_id="run-1",
        agent_id="obligation_coverage_matrix",
        phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(answer_shape="narrative_answer"),
        execution_family="query", workflow_kind="lookup",
        budget=OperatorBudget(), input_refs=(),
        input_hash="effective_date_q",
        work_profile={"task_criteria_count": 0,
                      "task_deliverable_spec_count": 0,
                      "obligation_row_count": 0},
    )
    rt = _Runtime(
        matter, llm_client=fake,
        query="What is the effective date of the agreement?",
    )
    # match() should return None for query family with no signals
    m = ObligationCoverageMatrix().match(inv)
    assert m is None, f"query family must not produce matrix; got {m}"


def test_presupplied_criteria_fast_path_not_required():
    """The same prompt with explicit criteria uses Tier 1; without uses
    Tier 5. Both produce non-empty matrices."""
    fake = _FakeLLMClient(json_response=_hospital_antitrust_response())

    # WITH explicit criteria
    matter_a = MatterModel.open_in_memory()
    matter_a.typed_evidence.upsert(
        RECORD_KIND_TASK_CRITERIA, "task_criteria:explicit_t1",
        payload={"schema_ref": SCHEMA_REF_TASK_CRITERIA,
                 "task_id": "explicit_t1",
                 "criteria": [{"criterion_id": "C-001", "title": "Be coherent",
                               "severity": "required"}]},
        confidence=1.0,
    )
    rt_a = _Runtime(matter_a, llm_client=fake, query="Draft an antitrust memo.")
    inv_a = _invocation(matter_a, work_profile={"task_criteria_count": 1,
                                                 "task_deliverable_spec_count": 0,
                                                 "obligation_row_count": 0})
    result_a = asyncio.run(ObligationCoverageMatrix().invoke(inv_a, rt_a))
    assert result_a.artifacts
    assert result_a.artifacts[0].payload["criteria_source"] == "typed_evidence_record"
    assert result_a.llm_calls == 0, "Tier 1 path must NOT call LLM"

    # WITHOUT explicit criteria
    matter_b = MatterModel.open_in_memory()
    rt_b = _Runtime(matter_b, llm_client=fake, query="Draft an antitrust memo.")
    inv_b = _invocation(matter_b)
    result_b = asyncio.run(ObligationCoverageMatrix().invoke(inv_b, rt_b))
    assert result_b.artifacts
    assert result_b.artifacts[0].payload["criteria_source"] == "llm_inferred"
    assert result_b.llm_calls == 1


def test_tier5_no_benchmark_keys_in_persisted_payloads():
    """Anti-gaming sentinel: nothing in the persisted criteria/deliverable
    payloads from Tier 5 should look like a benchmark task ID, criterion
    ID, or LAB scoring key."""
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response=_hospital_antitrust_response())
    rt = _Runtime(
        matter, llm_client=fake,
        query="Draft me an antitrust memo on this hospital acquisition.",
    )
    inv = _invocation(matter)
    asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))

    rows = matter.db.execute(
        "SELECT payload_json FROM typed_evidence_record WHERE record_kind=?",
        (RECORD_KIND_TASK_CRITERIA,),
    ).fetchall()
    assert rows, "Tier 5 must persist inferred criteria"
    blob = " ".join(r["payload_json"] for r in rows).lower()

    # Reject anything that looks like a LAB rubric / benchmark identifier.
    forbidden_patterns = [
        r"\bharvey-lab\b",
        r"\blab[-_]task\b",
        r"\brubric\b",
        r"^c-\d{3}$",  # LAB criteria are formatted as C-001 etc — reject if exact pattern
        r"\bharvey_id\b",
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, blob), (
            f"Tier 5 leaked benchmark identifier matching {pat!r} into "
            f"persisted payload — anti-gaming gate violated"
        )


# ---------------------------------------------------------------------------
# Cache + budget tests
# ---------------------------------------------------------------------------


def test_tier5_cache_hit_avoids_second_llm_call():
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response=_hospital_antitrust_response())
    rt = _Runtime(
        matter, llm_client=fake,
        query="Draft me an antitrust memo on this hospital acquisition.",
    )
    inv = _invocation(matter)

    # First call — should hit LLM
    r1 = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert r1.llm_calls == 1

    # Second call same prompt — cache hit, no LLM
    r2 = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert r2.llm_calls == 0, (
        f"Second invocation with same cache key should not call LLM; got "
        f"{r2.llm_calls} calls (fake client.calls = {len(fake.calls)})"
    )
    # Source label distinguishes cache hit
    assert r2.artifacts[0].payload["criteria_source"] in (
        "llm_inferred_cached", "typed_evidence_record",
    )


def test_tier5_low_confidence_does_not_materialize_matrix():
    """When the LLM returns confidence < 0.40 for non-deliverable family,
    the operator must NOT materialize a matrix."""
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response={
        "schema_ref": "task.criteria_inference.v1",
        "inference_version": TIER5_INFERENCE_VERSION,
        "confidence": 0.20,  # very low
        "should_materialize_matrix": True,
        "criteria": [
            {"title": "Be coherent", "severity": "recommended"},
        ],
        "deliverables": [],
    })
    inv = AgentInvocation(
        matter_id=matter.matter_id, run_id="run-1",
        agent_id="obligation_coverage_matrix",
        phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(answer_shape="report"),
        execution_family="investigate", workflow_kind="drafting",
        budget=OperatorBudget(), input_refs=(), input_hash="vague",
        work_profile={"task_criteria_count": 0,
                      "task_deliverable_spec_count": 0,
                      "obligation_row_count": 0},
    )
    rt = _Runtime(matter, llm_client=fake, query="hmm")
    result = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert result.status == "success"
    assert not result.artifacts, "low confidence + non-deliverable must skip materialization"
    assert any("low_confidence" in w for w in result.warnings)


def test_tier5_unparseable_response_warns_no_crash():
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(string_response="not json at all, just words")
    rt = _Runtime(matter, llm_client=fake, query="Draft me a memo.")
    inv = _invocation(matter)
    result = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert result.status == "success"
    assert result.llm_calls == 1
    assert any("unparseable" in w for w in result.warnings)


def test_tier5_llm_call_failure_returns_clean_warning():
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(raise_on_call=True)
    rt = _Runtime(matter, llm_client=fake, query="Draft me a memo.")
    inv = _invocation(matter)
    result = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert result.status == "success"
    assert any("tier5_llm_call_failed" in w for w in result.warnings)


def test_tier5_no_llm_client_warns_gracefully():
    matter = MatterModel.open_in_memory()
    rt = _Runtime(matter, llm_client=None, query="Draft me a memo.")
    inv = _invocation(matter)
    result = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert result.status == "success"
    assert any("tier5_no_llm_client" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Cache key stability
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Codex Phase-2 r1 carryover regression tests
# ---------------------------------------------------------------------------


def test_inferred_criteria_do_not_leak_across_prompts_in_same_matter():
    """Codex Phase-2 r1 BLOCKER: previously, prior inferred criteria
    persisted as Tier 1 typed evidence and bled into the next prompt.
    Now Tier 1 should EXCLUDE inferred rows so a different prompt fires
    Tier 5 fresh."""
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response=_hospital_antitrust_response())
    rt = _Runtime(
        matter, llm_client=fake,
        query="Draft me an antitrust memo on this hospital acquisition.",
    )
    inv = _invocation(matter)
    r1 = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert r1.llm_calls == 1

    # Second prompt — different content, different cache key
    fake2 = _FakeLLMClient(json_response={
        "schema_ref": "task.criteria_inference.v1",
        "inference_version": TIER5_INFERENCE_VERSION,
        "confidence": 0.75,
        "should_materialize_matrix": True,
        "criteria": [
            {"title": "Identify tax issues", "severity": "required"},
            {"title": "Reconcile schedule M-1", "severity": "required"},
        ],
        "deliverables": [],
    })
    rt2 = _Runtime(
        matter, llm_client=fake2,
        query="Draft me a tax memo on the target's pre-acquisition tax issues.",
    )
    inv2 = _invocation(matter, input_hash="tax_memo_q")
    r2 = asyncio.run(ObligationCoverageMatrix().invoke(inv2, rt2))

    # The second prompt should NOT reuse antitrust criteria
    assert r2.llm_calls == 1, (
        "Different prompt must trigger fresh Tier 5 inference; saw "
        f"llm_calls={r2.llm_calls}"
    )
    assert r2.artifacts
    titles = [
        (r.get("source_criterion") or {}).get("title", "")
        for r in r2.artifacts[0].payload.get("rows", [])
    ]
    assert any("tax" in t.lower() for t in titles), (
        f"Expected tax-related criteria; got titles {titles}"
    )
    assert not any("market" in t.lower() for t in titles), (
        f"Antitrust criteria leaked into tax prompt; titles {titles}"
    )


def test_match_routes_readquery_completeness_via_workprofile_flag():
    """Codex Phase-2 r1 BLOCKER: match() previously hardcoded
    user_asked_completeness=False, so read/query "what gaps?" prompts
    could not select the operator. Now it reads
    work_profile['query_asks_completeness']."""
    matter = MatterModel.open_in_memory()
    inv = AgentInvocation(
        matter_id=matter.matter_id, run_id="r",
        agent_id="obligation_coverage_matrix",
        phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(answer_shape="narrative_answer"),
        execution_family="query", workflow_kind="lookup",
        budget=OperatorBudget(), input_refs=(), input_hash="q",
        work_profile={"task_criteria_count": 0,
                      "task_deliverable_spec_count": 0,
                      "obligation_row_count": 0,
                      "query_asks_completeness": 1},
    )
    m = ObligationCoverageMatrix().match(inv)
    assert m is not None, "completeness query must select obligation operator"
    assert m.requirement == AgentRequirement.OPTIONAL


def test_match_deliverable_no_criteria_escalates_to_blocking_for_drafting():
    """For deliverable family + drafting workflow with no criteria yet,
    match() must escalate to BLOCKING_VALIDATOR so Tier 5 inferred
    missing required rows actually halt synthesis."""
    matter = MatterModel.open_in_memory()
    inv = AgentInvocation(
        matter_id=matter.matter_id, run_id="r",
        agent_id="obligation_coverage_matrix",
        phase="pre_synthesis", persona_id=None,
        requirement=AgentRequirement.OPTIONAL,
        task=AgentTaskView(answer_shape="memo"),
        execution_family="deliverable", workflow_kind="drafting",
        budget=OperatorBudget(), input_refs=(), input_hash="q",
        work_profile={"task_criteria_count": 0,
                      "task_deliverable_spec_count": 0,
                      "obligation_row_count": 0},
    )
    m = ObligationCoverageMatrix().match(inv)
    assert m is not None
    assert m.requirement == AgentRequirement.BLOCKING_VALIDATOR


def test_engine_invocation_propagates_taskview_and_family():
    """Codex Phase-2 r1 BLOCKER: engine previously hardcoded
    AgentTaskView() and execution_family='investigate'. Now it must
    build them from state.execution_contract."""
    from irys.rlm.engine import RLMEngine
    from irys.rlm.state import InvestigationState
    from irys.rlm.governance import ExecutionContract

    matter = MatterModel.open_in_memory()
    eng = RLMEngine.__new__(RLMEngine)
    eng._matter_model = matter
    eng.client = _FakeLLMClient(json_response=_hospital_antitrust_response())

    state = InvestigationState.create(
        "Draft me an antitrust memo on this hospital acquisition.",
        "/tmp/repo",
    )
    state._run_id = "engine-tier5-test"
    # Plumb a deliverable-family ExecutionContract — exactly the case
    # the engine was previously hardcoding away.
    state.execution_contract = ExecutionContract(
        family="deliverable",
        workflow_kind="drafting",
        output_contract={
            "task_spec": {
                "task_type": "antitrust-memo",
                "operation": "draft",
                "answer_shape": "memo",
                "required_evidence": [],
                "fresh_extraction_required": False,
                "cached_state_allowed": True,
                "external_tool_required": False,
            },
        },
    )

    # Run the engine hook directly. Per Codex lifecycle decision E
    # (B-now, D-later): Tier 5 infers criteria but pre-synthesis pending
    # rows (status=unknown/upstream_required_evidence_missing) MUST NOT
    # halt the run — they're placeholders awaiting output, not real
    # failures. Halt comes from a future post-synthesis verification.
    asyncio.run(eng._run_pre_synthesis_operators(state))

    # Matrix produced
    rows = matter.db.execute(
        "SELECT artifact_kind, payload_json FROM agent_artifact "
        "WHERE artifact_kind=?",
        ("obligation.coverage_matrix.v1",),
    ).fetchall()
    assert rows, (
        "engine path did not produce an obligation matrix even though "
        "ExecutionContract.family='deliverable' should reach Tier 5"
    )
    p = json.loads(rows[0]["payload_json"])
    assert p["criteria_source"] == "llm_inferred"
    assert p["invocation_requirement"] == "blocking_validator"

    # Lifecycle decision E observability fields are populated
    assert p.get("matrix_phase") == "pre_synthesis_inference"
    assert p.get("render_policy") == "concrete_gaps_only_until_post_synthesis"
    assert "n_pending_output" in p
    assert "n_verifiable_gaps" in p
    # The whole point of decision E: pending rows exist but no concrete gaps yet
    assert p["n_pending_output"] > 0, (
        "Tier 5 inferred rows should be pending pre-synthesis"
    )

    # No halt was recorded
    halt = state.findings.get("blocking_validator_halt")
    assert halt is None, (
        f"pre-synthesis pending rows must NOT trigger halt; got {halt}"
    )


# ---------------------------------------------------------------------------
# Lifecycle decision E (B-now, D-later) regression tests
# ---------------------------------------------------------------------------


def test_pending_rows_not_rendered_into_synthesis_context():
    """Codex lifecycle decision E: when all matrix rows are pending
    pre-synthesis (unknown / upstream_required_evidence_missing) and
    the user did NOT ask about completeness, the renderer must NOT
    inject the matrix into synthesis context — it would tell the LLM
    'these obligations are missing' before the LLM has written them."""
    from irys.rlm.engine import RLMEngine
    from irys.rlm.state import InvestigationState
    from irys.rlm.governance import ExecutionContract

    matter = MatterModel.open_in_memory()
    eng = RLMEngine.__new__(RLMEngine)
    eng._matter_model = matter
    eng.client = _FakeLLMClient(json_response=_hospital_antitrust_response())

    state = InvestigationState.create(
        "Draft me an antitrust memo on this hospital acquisition.",
        "/tmp/repo",
    )
    state._run_id = "render-no-pending-test"
    state.execution_contract = ExecutionContract(
        family="deliverable",
        workflow_kind="drafting",
        output_contract={
            "task_spec": {
                "task_type": "antitrust-memo",
                "operation": "draft",
                "answer_shape": "memo",
            },
        },
    )
    asyncio.run(eng._run_pre_synthesis_operators(state))

    # Matrix exists
    rows = matter.db.execute(
        "SELECT payload_json FROM agent_artifact WHERE artifact_kind=?",
        ("obligation.coverage_matrix.v1",),
    ).fetchall()
    assert rows
    p = json.loads(rows[0]["payload_json"])
    assert p["n_pending_output"] > 0

    # Synthesis renderer must omit it (no concrete gaps, no completeness ask)
    summary = eng._build_agent_artifact_summary(state)
    assert "OBLIGATION COVERAGE" not in summary, (
        f"Pending Tier 5 rows leaked into synthesis context; got summary "
        f"start: {summary[:200]!r}"
    )


def test_concrete_missing_artifact_renders_into_synthesis():
    """When a row has CONCRETE status (artifact_kind is required + absent),
    the renderer SHOULD inject it into synthesis context as a real gap."""
    from irys.rlm.engine import RLMEngine
    from irys.rlm.state import InvestigationState
    from irys.rlm.governance import ExecutionContract

    matter = MatterModel.open_in_memory()
    # Pre-supply a Tier 1 explicit criterion that requires a specific
    # artifact_kind that doesn't exist in the matter — this creates a
    # CONCRETE pre-synthesis-verifiable gap.
    matter.typed_evidence.upsert(
        RECORD_KIND_TASK_CRITERIA, "task_criteria:explicit_concrete",
        payload={
            "schema_ref": SCHEMA_REF_TASK_CRITERIA,
            "task_id": "explicit_concrete",
            "criteria": [{
                "criterion_id": "C-001",
                "title": "Document section map must exist",
                "severity": "required",
            }],
        },
        confidence=1.0,
    )
    matter.typed_evidence.upsert(
        "task_deliverable_spec", "task_deliverable:explicit_concrete",
        payload={
            "schema_ref": "task.deliverable_spec.v1",
            "task_id": "explicit_concrete",
            "deliverables": [{
                "deliverable_key": "memo.docx",
                "required_artifact_kinds": ["document.section_map"],
            }],
        },
        confidence=1.0,
    )

    eng = RLMEngine.__new__(RLMEngine)
    eng._matter_model = matter
    eng.client = _FakeLLMClient(json_response={})

    state = InvestigationState.create("draft", "/tmp/repo")
    state._run_id = "concrete-gap-test"
    state.execution_contract = ExecutionContract(
        family="deliverable",
        workflow_kind="drafting",
        output_contract={"task_spec": {"task_type": "memo",
                                        "answer_shape": "memo"}},
    )

    # No `document.section_map` artifact has been seeded → row scores
    # 'missing' (concrete), triggering blocking_validator halt. The
    # artifact persists before the raise per round-3 dispatcher fix.
    with pytest.raises(RuntimeError, match="blocking validator"):
        asyncio.run(eng._run_pre_synthesis_operators(state))

    # Even though the run halted, the matrix artifact is still there,
    # and the renderer (when called) shows the concrete gap row.
    summary = eng._build_agent_artifact_summary(state)
    assert "OBLIGATION COVERAGE" in summary, (
        "concrete-gap rows must render into synthesis"
    )


def test_bare_keywords_complete_or_coverage_do_not_trigger_render():
    """Codex holistic review fix: bare keywords like 'complete' or
    'coverage' appear in benign LAB prose ('draft a complete memo with
    comprehensive coverage'). They MUST NOT trigger pending-matrix
    rendering — only explicit gap-analysis phrases do."""
    from irys.rlm.engine import RLMEngine
    from irys.rlm.state import InvestigationState
    from irys.rlm.governance import ExecutionContract

    matter = MatterModel.open_in_memory()
    eng = RLMEngine.__new__(RLMEngine)
    eng._matter_model = matter
    eng.client = _FakeLLMClient(json_response=_hospital_antitrust_response())

    # Common LAB-style prose with bare keywords that are NOT explicit
    # completeness asks
    state = InvestigationState.create(
        "Draft a complete antitrust memo with comprehensive coverage of "
        "the proposed acquisition.",
        "/tmp/repo",
    )
    state._run_id = "bare-keyword-test"
    state.execution_contract = ExecutionContract(
        family="deliverable",
        workflow_kind="drafting",
        output_contract={"task_spec": {"task_type": "antitrust",
                                        "answer_shape": "memo"}},
    )
    asyncio.run(eng._run_pre_synthesis_operators(state))
    summary = eng._build_agent_artifact_summary(state)
    assert "OBLIGATION COVERAGE" not in summary, (
        f"bare keywords 'complete' / 'coverage' must NOT trigger "
        f"pending-matrix rendering; got {summary[:200]!r}"
    )


def test_completeness_query_renders_pending_rows_explicitly():
    """When the user explicitly asks about completeness/coverage, the
    renderer should show pending rows so they can be addressed."""
    from irys.rlm.engine import RLMEngine
    from irys.rlm.state import InvestigationState
    from irys.rlm.governance import ExecutionContract

    matter = MatterModel.open_in_memory()
    eng = RLMEngine.__new__(RLMEngine)
    eng._matter_model = matter
    eng.client = _FakeLLMClient(json_response=_hospital_antitrust_response())

    state = InvestigationState.create(
        "What gaps in coverage are missing from this antitrust analysis?",
        "/tmp/repo",
    )
    state._run_id = "completeness-render-test"
    state.execution_contract = ExecutionContract(
        family="investigate",
        workflow_kind="analysis",
        output_contract={
            "task_spec": {
                "task_type": "antitrust",
                "answer_shape": "memo",
            },
        },
    )
    asyncio.run(eng._run_pre_synthesis_operators(state))
    summary = eng._build_agent_artifact_summary(state)
    assert "OBLIGATION COVERAGE" in summary, (
        "completeness query must render even if all rows are pending"
    )


def test_run_objective_success_criteria_does_not_short_circuit_tier5():
    """Codex Phase-2 r3 (post-smoke v6): RunObjective.success_criteria
    must NOT be sourced as obligation criteria. Even when present,
    Tier 5 must fire because RunObjective is engine-internal scaffolding
    (e.g. 'satisfy task X', 'ground the answer in evidence'), not user
    intent. The smoke v6 failure mode was: Tier 4 read RunObjective →
    short-circuited Tier 5 → noisy matrices → synthesis quality dropped."""
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(json_response=_hospital_antitrust_response())

    # Create a runtime where state.run_objective.success_criteria has
    # exactly the kind of engine boilerplate that smoke v6 picked up.
    class _RunObjective:
        success_criteria = [
            "satisfy task 'quantitative_reconciliation' as v1",
            "ground the answer in required evidence objects",
            "do not rely on cached summaries alone when fresh evidence required",
        ]
    rt = _Runtime(
        matter, llm_client=fake,
        query="Draft me an antitrust memo on this hospital acquisition.",
    )
    rt.state.run_objective = _RunObjective()

    inv = _invocation(matter)
    result = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert result.status == "success"
    assert result.llm_calls == 1, (
        f"Tier 5 must fire even when RunObjective.success_criteria exists; "
        f"got llm_calls={result.llm_calls}"
    )
    assert result.artifacts
    p = result.artifacts[0].payload
    assert p["criteria_source"] == "llm_inferred", (
        f"criteria_source must be llm_inferred, not run_objective; "
        f"got {p['criteria_source']}"
    )
    # Confirm the matrix has actual antitrust content, not engine boilerplate
    titles = [
        (r.get("source_criterion") or {}).get("title", "")
        for r in p.get("rows", [])
    ]
    assert not any("satisfy task" in t.lower() for t in titles), (
        f"RunObjective boilerplate leaked into matrix: {titles}"
    )
    assert any("market" in t.lower() for t in titles), (
        f"Expected antitrust criteria; got titles {titles}"
    )


def test_failed_llm_call_counts_attempted_call():
    """Codex Phase-2 r1 non-blocker: an exception on llm_client.complete
    should still be counted as an attempted call against budget."""
    matter = MatterModel.open_in_memory()
    fake = _FakeLLMClient(raise_on_call=True)
    rt = _Runtime(matter, llm_client=fake, query="Draft me a memo.")
    inv = _invocation(matter)
    result = asyncio.run(ObligationCoverageMatrix().invoke(inv, rt))
    assert result.status == "success"
    assert result.llm_calls == 1, (
        f"failed LLM attempt should still consume one budget call; got "
        f"llm_calls={result.llm_calls}"
    )


def test_cache_key_is_deterministic_and_prompt_based():
    spec = {"task_type": "deliverable", "operation": "draft",
            "answer_shape": "memo", "required_evidence": []}
    a = _tier5_inference_cache_key(
        user_query="Draft me an antitrust memo.",
        task_spec=spec, execution_family="deliverable",
        workflow_kind="drafting", domain_profile_id="legal:1",
        domain_profile_version=1,
    )
    b = _tier5_inference_cache_key(
        user_query="Draft me an antitrust memo.",
        task_spec=spec, execution_family="deliverable",
        workflow_kind="drafting", domain_profile_id="legal:1",
        domain_profile_version=1,
    )
    assert a == b
    # Whitespace/case normalization
    c = _tier5_inference_cache_key(
        user_query="DRAFT ME   an  antitrust   MEMO.",
        task_spec=spec, execution_family="deliverable",
        workflow_kind="drafting", domain_profile_id="legal:1",
        domain_profile_version=1,
    )
    assert a == c
    # Different prompt → different key
    d = _tier5_inference_cache_key(
        user_query="Draft me a tax memo.",
        task_spec=spec, execution_family="deliverable",
        workflow_kind="drafting", domain_profile_id="legal:1",
        domain_profile_version=1,
    )
    assert a != d
