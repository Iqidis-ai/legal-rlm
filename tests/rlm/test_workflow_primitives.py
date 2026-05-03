"""Workflow primitive tests.

These are intentionally small: the first slice only proves that
objective/obligation/working-set/plan/validation state can survive
checkpoint serialization. The planner and validators can build on this
without inventing a second state channel.
"""

import asyncio

from irys.core.models import ModelTier
from irys.rlm.state import (
    InvestigationState,
    Obligation,
    OutputEnvelope,
    PlanAction,
    RunObjective,
    ValidationResult,
    WorkflowKind,
    WorkingSet,
)
from irys.rlm.engine import RLMConfig, RLMEngine
from irys.rlm.governance import CascadeGovernor


class _StubClient:
    pass


class _RepairClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    async def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


def test_workflow_primitives_survive_checkpoint_roundtrip():
    state = InvestigationState.create(
        "draft a motion outline",
        ".",
    )
    objective = RunObjective.create(
        user_goal="Draft a motion outline",
        output_shape="motion_outline",
        workflow_kind=WorkflowKind.DRAFTING.value,
        success_criteria=[
            "address every required element",
            "flag unsupported allegations",
        ],
        constraints=["do not quote attorney guidance"],
        source_query=state.query,
    )
    citation_obligation = Obligation.create(
        description="Every factual paragraph needs source support.",
        obligation_type="citation",
        validator="citation_floor",
    )
    state.run_objective = objective
    state.workflow_obligations = [citation_obligation]
    state.working_set = WorkingSet(
        verified_assertion_ids=["a1"],
        issue_ids=["i1"],
        gap_ids=["g1"],
        document_ids=["doc.pdf"],
        dependency_manifest_hash="dep123",
    )
    state.plan_actions = [
        PlanAction.create(
            action_type="assemble_outline",
            description="Build sections from issues and verified assertions.",
            target_obligation_ids=[citation_obligation.id],
            input_refs=["issue:i1", "assertion:a1"],
            expected_output="section_plan",
        )
    ]
    state.validation_results = [
        ValidationResult(
            validator="draft_validator",
            passed=False,
            score=0.7,
            blocking_issues=["missing authority section"],
            obligation_status={citation_obligation.id: True},
        )
    ]

    restored = InvestigationState.from_dict(state.to_dict())

    assert restored.run_objective is not None
    assert restored.run_objective.workflow_kind == WorkflowKind.DRAFTING.value
    assert restored.run_objective.success_criteria == objective.success_criteria
    assert restored.workflow_obligations[0].validator == "citation_floor"
    assert restored.working_set is not None
    assert restored.working_set.dependency_manifest_hash == "dep123"
    assert restored.plan_actions[0].target_obligation_ids == [citation_obligation.id]
    assert restored.validation_results[0].blocking_issues == [
        "missing authority section",
    ]


def test_output_envelope_survives_checkpoint_roundtrip():
    state = InvestigationState.create("analyze exposure", ".")
    validation = ValidationResult(
        validator="citation_floor",
        passed=False,
        blocking_issues=["citation support 0 < floor 1"],
    )
    state.output_envelope = OutputEnvelope.create(
        output_text="Draft answer",
        workflow_kind=WorkflowKind.ANALYSIS.value,
        output_shape="investigation_memo",
        emitter="test",
        objective_id="obj1",
        dependency_manifest_hash="dep123",
        validation_results=[validation],
    )

    restored = InvestigationState.from_dict(state.to_dict())

    assert restored.output_envelope is not None
    assert restored.output_envelope.output_text == "Draft answer"
    assert restored.output_envelope.dependency_manifest_hash == "dep123"
    assert restored.output_envelope.validation_results[0].validator == "citation_floor"
    assert restored.output_envelope.blocking_issues == [
        "citation support 0 < floor 1",
    ]


def test_engine_seeds_workflow_state_from_execution_contract():
    state = InvestigationState.create("draft a privilege log", ".")
    state.execution_contract = CascadeGovernor._contract_for("deliverable")
    engine = RLMEngine(gemini_client=_StubClient(), config=RLMConfig())

    engine._initialize_workflow_state(state)

    assert state.run_objective is not None
    assert state.run_objective.workflow_kind == WorkflowKind.DRAFTING.value
    assert state.run_objective.output_shape == "legal_work_product"
    assert "follow the selected work-product template" in (
        state.run_objective.success_criteria
    )
    validators = {item.validator for item in state.workflow_obligations}
    assert "draft_template" in validators
    assert "human_review_required" in validators
    assert state.working_set is not None


def test_engine_emit_output_wraps_final_output_and_validates():
    state = InvestigationState.create("analyze exposure", ".")
    state.execution_contract = CascadeGovernor._contract_for("investigate")
    engine = RLMEngine(gemini_client=_StubClient(), config=RLMConfig())
    engine._initialize_workflow_state(state)

    envelope = engine._emit_output(state, "No citations here.", emitter="test")

    assert state.findings["final_output"] == "No citations here."
    assert state.findings["output_envelope"]["id"] == envelope.id
    assert state.output_envelope is envelope
    assert envelope.workflow_kind == WorkflowKind.ANALYSIS.value
    assert any(
        result.validator == "citation_floor" and not result.passed
        for result in envelope.validation_results
    )
    assert any("citation support" in issue for issue in envelope.blocking_issues)


def test_workflow_quality_section_surfaces_contract_for_synthesis():
    state = InvestigationState.create("draft a privilege log", ".")
    state.execution_contract = CascadeGovernor._contract_for("deliverable")
    engine = RLMEngine(gemini_client=_StubClient(), config=RLMConfig())
    engine._initialize_workflow_state(state)
    assert state.working_set is not None
    state.working_set.document_ids = ["doc-1"]
    state.working_set.dependency_manifest_hash = "dep123"

    section = engine._build_workflow_quality_section(state)

    assert "Workflow Quality Contract" in section
    assert "Workflow kind: drafting" in section
    assert "Output shape: legal_work_product" in section
    assert "[draft_template]" in section
    assert "[human_review_required]" in section
    assert "dependency manifest hash: dep123" in section
    assert "Do not claim a draft is ready" in section


def test_repair_output_runs_once_for_fixable_workflow_failure():
    repaired_text = "## Analysis\n\nAssumptions:\n- Payment was made temporarily."
    client = _RepairClient(repaired_text)
    state = InvestigationState.create("what if payment was made?", ".")
    state.execution_contract = CascadeGovernor._contract_for("scenario")
    engine = RLMEngine(gemini_client=client, config=RLMConfig())
    engine._initialize_workflow_state(state)

    repaired = asyncio.run(
        engine._repair_output_if_needed(
            state,
            "Payment was made.",
            emitter="synthesis",
        )
    )

    assert repaired == repaired_text
    assert len(client.calls) == 1
    prompt, kwargs = client.calls[0]
    assert "temporary assumptions were not explicitly labeled" in prompt
    assert "Do not invent citations" in prompt
    assert kwargs["tier"] == ModelTier.PRO
    assert kwargs["usage_label"] == "synthesis_workflow_repair"


def test_repair_output_skips_nonfixable_citation_floor_failure():
    client = _RepairClient("unused")
    state = InvestigationState.create("analyze exposure", ".")
    state.execution_contract = CascadeGovernor._contract_for("investigate")
    engine = RLMEngine(gemini_client=client, config=RLMConfig())
    engine._initialize_workflow_state(state)

    output = asyncio.run(
        engine._repair_output_if_needed(
            state,
            "No citations here.",
            emitter="synthesis",
        )
    )

    assert output == "No citations here."
    assert client.calls == []


def test_fmt_output_envelope_summary_clean():
    from irys.ui.app import _fmt_output_envelope_summary

    envelope = OutputEnvelope.create(
        output_text="Analysis complete.",
        workflow_kind="analysis",
        output_shape="answer",
        emitter="synthesis",
    ).to_dict()
    md = _fmt_output_envelope_summary(envelope)
    assert "analysis" in md
    assert "All output quality checks passed" in md


def test_fmt_output_envelope_summary_with_warnings():
    from irys.ui.app import _fmt_output_envelope_summary

    vr = ValidationResult(
        validator="citation_check",
        passed=False,
        score=0.3,
        blocking_issues=["No citations found"],
        warnings=["Output may lack supporting references"],
    )
    envelope = OutputEnvelope.create(
        output_text="Draft answer.",
        workflow_kind="analysis",
        output_shape="answer",
        emitter="synthesis",
        validation_results=[vr],
        review_required=True,
    ).to_dict()
    md = _fmt_output_envelope_summary(envelope)
    assert "Review required" in md
    assert "No citations found" in md
    assert "citation_check" in md
    assert "failed" in md


def test_fmt_output_envelope_summary_empty():
    from irys.ui.app import _fmt_output_envelope_summary
    assert _fmt_output_envelope_summary(None) == ""
    assert _fmt_output_envelope_summary({}) == ""


def test_fmt_belief_revision_panel_empty():
    from irys.ui.app import _fmt_belief_revision_panel
    result = _fmt_belief_revision_panel([])
    assert "viz-empty" in result
    assert "No belief revisions recorded" in result


def test_fmt_belief_revision_panel_empty_finance():
    from irys.ui.app import _fmt_belief_revision_panel
    result = _fmt_belief_revision_panel([], domain="finance")
    assert "No revisions recorded" in result
    assert "Financial facts" in result


def test_fmt_belief_revision_panel_with_data():
    from irys.ui.app import _fmt_belief_revision_panel
    revisions = [
        {
            "proposition_text": "Defendant breached clause 4.2",
            "old_belief_state": "believed",
            "new_belief_state": "rejected",
            "cause": "contradicted_by_new_evidence",
            "old_confidence": 0.85,
            "new_confidence": 0.20,
        },
        {
            "proposition_text": "Payment was received on March 1",
            "old_belief_state": "unknown",
            "new_belief_state": "believed",
            "cause": "user_correction",
            "old_confidence": 0.0,
            "new_confidence": 0.95,
        },
    ]
    result = _fmt_belief_revision_panel(revisions)
    assert "2 revisions" in result
    assert "Defendant breached" in result
    assert "believed" in result
    assert "rejected" in result


def test_fmt_belief_revision_panel_missing_fields():
    from irys.ui.app import _fmt_belief_revision_panel
    revisions = [{"proposition_text": None, "old_belief_state": None}]
    result = _fmt_belief_revision_panel(revisions)
    assert "1 revision" in result


def test_fmt_contradiction_panel_empty():
    from irys.ui.app import _fmt_contradiction_panel
    result = _fmt_contradiction_panel([])
    assert "viz-empty" in result
    assert "No active contradictions" in result


def test_fmt_contradiction_panel_empty_finance():
    from irys.ui.app import _fmt_contradiction_panel
    result = _fmt_contradiction_panel([], domain="finance")
    assert "No conflicting financial claims" in result


def test_fmt_contradiction_panel_with_data():
    from irys.ui.app import _fmt_contradiction_panel
    conflicts = [
        {
            "attacker_prop": "Payment was never received",
            "attacked_prop": "Payment was received on March 1",
            "link_type": "contradicts",
            "attacker_belief": "operative",
            "attacked_belief": "disputed",
        },
    ]
    result = _fmt_contradiction_panel(conflicts)
    assert "1 active conflict" in result
    assert "Payment was never received" in result
    assert "Payment was received on March 1" in result
    assert "contradicts" in result
    assert "Disputed" in result


def test_fmt_contradiction_panel_skips_non_dict():
    from irys.ui.app import _fmt_contradiction_panel
    result = _fmt_contradiction_panel(["bad", None, 42])
    assert "viz-empty" in result


def test_fmt_contradiction_panel_missing_fields():
    from irys.ui.app import _fmt_contradiction_panel
    result = _fmt_contradiction_panel([{"attacker_prop": None}])
    assert "1 active conflict" in result


# ---------------------------------------------------------------------------
# Document Version Chains panel formatter tests
# ---------------------------------------------------------------------------


def test_fmt_document_versions_panel_empty():
    from irys.ui.app import _fmt_document_versions_panel
    result = _fmt_document_versions_panel([])
    assert "viz-empty" in result
    assert "No document version chains" in result


def test_fmt_document_versions_panel_empty_finance():
    from irys.ui.app import _fmt_document_versions_panel
    result = _fmt_document_versions_panel([], domain="finance")
    assert "No filing version chains" in result


def test_fmt_document_versions_panel_with_data():
    from irys.ui.app import _fmt_document_versions_panel
    families = [
        {
            "family_id": "fam-1",
            "members": [
                {"id": "d1", "relative_path": "contract_v1.pdf", "is_operative": False},
                {"id": "d2", "relative_path": "contract_v2.pdf", "is_operative": True},
            ],
        },
    ]
    result = _fmt_document_versions_panel(families)
    assert "1 version chain" in result
    assert "2 documents" in result
    assert "contract_v1.pdf" in result
    assert "contract_v2.pdf" in result
    assert "Operative (Current)" in result
    assert "Superseded" in result


def test_fmt_document_versions_panel_skips_non_dict():
    from irys.ui.app import _fmt_document_versions_panel
    result = _fmt_document_versions_panel(["bad", None, 42])
    assert "viz-empty" in result


def test_fmt_document_versions_panel_missing_fields():
    from irys.ui.app import _fmt_document_versions_panel
    families = [{"family_id": "f1", "members": [{"id": "d1"}]}]
    result = _fmt_document_versions_panel(families)
    assert "1 version chain" in result


def test_fmt_document_versions_panel_multiple_families():
    from irys.ui.app import _fmt_document_versions_panel
    families = [
        {
            "family_id": "fam-1",
            "members": [
                {"id": "d1", "relative_path": "lease_v1.pdf", "is_operative": False},
                {"id": "d2", "relative_path": "lease_v2.pdf", "is_operative": True},
            ],
        },
        {
            "family_id": "fam-2",
            "members": [
                {"id": "d3", "relative_path": "memo_draft.docx", "is_operative": False},
                {"id": "d4", "relative_path": "memo_final.docx", "is_operative": True},
            ],
        },
    ]
    result = _fmt_document_versions_panel(families)
    assert "2 version chains" in result
    assert "4 documents" in result


# ---------------------------------------------------------------------------
# Quantitative Threshold Violations panel formatter tests
# ---------------------------------------------------------------------------


def test_fmt_quant_thresholds_panel_empty():
    from irys.ui.app import _fmt_quant_thresholds_panel
    result = _fmt_quant_thresholds_panel([])
    assert "viz-empty" in result
    assert "No quantitative threshold violations" in result


def test_fmt_quant_thresholds_panel_empty_finance():
    from irys.ui.app import _fmt_quant_thresholds_panel
    result = _fmt_quant_thresholds_panel([], domain="finance")
    assert "No financial risk thresholds" in result


def test_fmt_quant_thresholds_panel_with_data():
    from irys.ui.app import _fmt_quant_thresholds_panel
    violations = [
        {
            "threshold": "positive_exposure",
            "level": "HIGH",
            "description": "Claimed financial exposure: USD 50,000.00",
            "amount": 50000.0,
        },
        {
            "threshold": "disputed_fraction",
            "level": "MED",
            "description": "Disputed amounts represent 15% of total invoiced",
            "amount": 7500.0,
        },
    ]
    result = _fmt_quant_thresholds_panel(violations)
    assert "2 violations" in result
    assert "1 HIGH" in result
    assert "Positive Exposure" in result
    assert "Disputed Fraction" in result
    assert "pill-red" in result
    assert "pill-orange" in result


def test_fmt_quant_thresholds_panel_skips_non_dict():
    from irys.ui.app import _fmt_quant_thresholds_panel
    result = _fmt_quant_thresholds_panel(["bad", None])
    assert "viz-empty" in result


def test_fmt_quant_thresholds_panel_missing_fields():
    from irys.ui.app import _fmt_quant_thresholds_panel
    result = _fmt_quant_thresholds_panel([{"threshold": None}])
    assert "1 violation" in result


def test_fmt_system_health_panel_empty():
    from irys.ui.app import _fmt_system_health_panel
    result = _fmt_system_health_panel({})
    assert "viz-empty" in result


def test_fmt_system_health_panel_empty_finance():
    from irys.ui.app import _fmt_system_health_panel
    result = _fmt_system_health_panel({}, domain="finance")
    assert "viz-empty" in result


def test_fmt_system_health_panel_none():
    from irys.ui.app import _fmt_system_health_panel
    result = _fmt_system_health_panel(None)
    assert "viz-empty" in result


def test_fmt_system_health_panel_good():
    from irys.ui.app import _fmt_system_health_panel
    health = {
        "assertion_count": 50,
        "disputed_count": 5,
        "disputed_fraction": 0.1,
        "revision_count": 12,
        "open_gap_count": 3,
        "contradiction_count": 2,
        "version_chain_count": 1,
        "oscillating_count": 0,
        "health_score": "good",
    }
    result = _fmt_system_health_panel(health)
    assert "Truth Maintenance Health" in result
    assert "pill-green" in result
    assert "50" in result
    assert "10.0%" in result
    assert "All systems healthy" in result


def test_fmt_system_health_panel_attention_needed():
    from irys.ui.app import _fmt_system_health_panel
    health = {
        "assertion_count": 100,
        "disputed_count": 40,
        "disputed_fraction": 0.4,
        "revision_count": 30,
        "open_gap_count": 8,
        "contradiction_count": 5,
        "version_chain_count": 2,
        "oscillating_count": 3,
        "health_score": "attention_needed",
    }
    result = _fmt_system_health_panel(health)
    assert "Attention Needed" in result
    assert "pill-orange" in result
    assert "pill-red" in result
    assert "40.0%" in result


def test_fmt_system_health_panel_finance_domain():
    from irys.ui.app import _fmt_system_health_panel
    health = {
        "assertion_count": 10,
        "disputed_count": 1,
        "disputed_fraction": 0.1,
        "revision_count": 3,
        "open_gap_count": 0,
        "contradiction_count": 0,
        "version_chain_count": 0,
        "oscillating_count": 0,
        "health_score": "good",
    }
    result = _fmt_system_health_panel(health, domain="finance")
    assert "Analysis Health" in result
    assert "Total Claims" in result
    assert "Position Revisions" in result


def test_fmt_system_health_panel_missing_fields():
    from irys.ui.app import _fmt_system_health_panel
    result = _fmt_system_health_panel({"health_score": "good"})
    assert "Truth Maintenance Health" in result
    assert "0" in result
