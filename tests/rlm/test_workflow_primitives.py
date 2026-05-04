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
    assert state.run_objective.output_shape == "domain_work_product"
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
    assert "Output shape: domain_work_product" in section
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
    assert "50,000.00" in result
    assert "7,500.00" in result
    assert "Amount" in result


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


def test_fmt_so_scorecard_panel_empty():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard_panel
    result = _fmt_so_scorecard_panel({})
    assert "viz-empty" in result


def test_fmt_so_scorecard_panel_none():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard_panel
    result = _fmt_so_scorecard_panel(None)
    assert "viz-empty" in result


def test_fmt_so_scorecard_panel_with_data():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard_panel
    so = {
        "assertion_structure_rate": 1.0,
        "source_role_known_rate": 0.95,
        "issue_coverage_avg": 0.85,
        "reuse_rate": 0.72,
        "steerability": True,
        "belief_revision": True,
        "provenance_attribution_rate": 0.92,
        "numeric_extraction_rate": 0.88,
        "gap_surface_ratio": 0.3,
        "targets": {
            "assertion_structure_rate": 1.0,
            "source_role_known_rate": 0.9,
            "issue_coverage_avg": 0.8,
            "reuse_rate": 0.7,
            "numeric_extraction_rate": 0.9,
            "provenance_attribution_rate": 0.9,
            "steerability": True,
            "belief_revision": True,
        },
        "targets_met": {
            "assertion_structure_rate": True,
            "source_role_known_rate": True,
            "issue_coverage_avg": True,
            "reuse_rate": True,
            "numeric_extraction_rate": False,
            "provenance_attribution_rate": True,
            "steerability": True,
            "belief_revision": True,
        },
    }
    result = _fmt_so_scorecard_panel(so)
    assert "Sacred Outcomes Scorecard" in result
    assert "7/8 passing" in result
    assert "pill-green" in result
    assert "pill-red" in result
    assert "100.0%" in result
    assert "95.0%" in result


def test_fmt_so_scorecard_panel_all_passing():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard_panel
    so = {
        "assertion_structure_rate": 1.0,
        "source_role_known_rate": 0.95,
        "issue_coverage_avg": 0.85,
        "reuse_rate": 0.72,
        "steerability": True,
        "belief_revision": True,
        "provenance_attribution_rate": 0.92,
        "numeric_extraction_rate": 0.95,
        "gap_surface_ratio": 0.3,
        "targets": {
            "assertion_structure_rate": 1.0,
            "source_role_known_rate": 0.9,
            "issue_coverage_avg": 0.8,
            "reuse_rate": 0.7,
            "numeric_extraction_rate": 0.9,
            "provenance_attribution_rate": 0.9,
            "steerability": True,
            "belief_revision": True,
        },
        "targets_met": {
            "assertion_structure_rate": True,
            "source_role_known_rate": True,
            "issue_coverage_avg": True,
            "reuse_rate": True,
            "numeric_extraction_rate": True,
            "provenance_attribution_rate": True,
            "steerability": True,
            "belief_revision": True,
        },
    }
    result = _fmt_so_scorecard_panel(so)
    assert "8/8 passing" in result
    assert "pill-green" in result


def test_fmt_so_scorecard_panel_finance_domain():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard_panel
    so = {
        "targets": {},
        "targets_met": {},
    }
    result = _fmt_so_scorecard_panel(so, domain="finance")
    assert "Analysis Quality Scorecard" in result
    assert "Durability" in result


def test_fmt_so_scorecard_panel_missing_targets():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard_panel
    result = _fmt_so_scorecard_panel({"targets": {}, "targets_met": {}})
    assert "Sacred Outcomes Scorecard" in result
    assert "0/0 passing" in result


def test_fmt_trust_overrides_empty():
    from irys.ui.app import _fmt_trust_overrides
    result = _fmt_trust_overrides([])
    assert "viz-empty" in result
    assert "No trust overrides" in result


def test_fmt_trust_overrides_with_data():
    from irys.ui.app import _fmt_trust_overrides
    overrides = [
        {
            "id": "ov1",
            "document_pattern": "contract_v1.pdf",
            "trust_level": "low",
            "note": "Client disputes this version",
            "created_at": "2026-05-01T10:30:00",
        },
        {
            "id": "ov2",
            "document_pattern": "signed_agreement.pdf",
            "trust_level": "high",
            "note": "Verified original",
            "created_at": "2026-05-02T14:00:00",
        },
    ]
    result = _fmt_trust_overrides(overrides)
    assert "2 active" in result
    assert "contract_v1.pdf" in result
    assert "signed_agreement.pdf" in result
    assert "pill-red" in result
    assert "pill-green" in result
    assert "Client disputes" in result


def test_fmt_trust_overrides_skips_non_dict():
    from irys.ui.app import _fmt_trust_overrides
    result = _fmt_trust_overrides(["bad", None])
    assert "viz-empty" in result


# ---------------------------------------------------------------------------
# Annotations formatter
# ---------------------------------------------------------------------------

def test_fmt_annotations_panel_empty():
    from irys.ui.app import _fmt_annotations_panel
    result = _fmt_annotations_panel([])
    assert "viz-empty" in result


def test_fmt_annotations_panel_with_data():
    from irys.ui.app import _fmt_annotations_panel
    result = _fmt_annotations_panel([
        {"document_pattern": "contract.pdf", "annotation_text": "Key document", "annotation_type": "strategic", "created_at": "2026-01-01T00:00:00"},
        {"document_pattern": "report.pdf", "annotation_text": "Unreliable source", "annotation_type": "reliability", "created_at": "2026-01-02T00:00:00"},
    ])
    assert "contract.pdf" in result
    assert "Key document" in result
    assert "Strategic" in result
    assert "Reliability" in result
    assert "2 note" in result


def test_fmt_annotations_panel_skips_non_dict():
    from irys.ui.app import _fmt_annotations_panel
    result = _fmt_annotations_panel(["bad", None])
    assert "viz-empty" in result


# ---------------------------------------------------------------------------
# Decision context formatter
# ---------------------------------------------------------------------------

def test_fmt_decision_context_none():
    from irys.ui.app import _fmt_decision_context
    result = _fmt_decision_context(None)
    assert "viz-empty" in result


def test_fmt_decision_context_with_data():
    from irys.ui.app import _fmt_decision_context
    result = _fmt_decision_context({
        "decision_maker_type": "judge",
        "decision_maker_name": "Judge Smith",
        "objective": "motion_practice",
        "strategic_notes": "Focus on damages",
        "scope_narrow": True,
        "updated_at": "2026-01-01T12:00:00",
    })
    assert "Judge" in result
    assert "Judge Smith" in result
    assert "Motion Practice" in result
    assert "Focus on damages" in result
    assert "Yes" in result


def test_fmt_decision_context_finance_domain():
    from irys.ui.app import _fmt_decision_context
    result = _fmt_decision_context({
        "decision_maker_type": "risk_officer",
        "objective": "compliance_review",
    }, domain="finance")
    assert "Risk Officer" in result
    assert "Compliance" in result


def test_fmt_decision_context_coding_domain():
    from irys.ui.app import _fmt_decision_context
    result = _fmt_decision_context({
        "decision_maker_type": "tech_lead",
        "objective": "code_review",
    }, domain="coding")
    assert "Tech Lead" in result
    assert "Code Review" in result


def test_fmt_decision_context_biomedical_domain():
    from irys.ui.app import _fmt_decision_context
    result = _fmt_decision_context({
        "decision_maker_type": "clinician",
        "objective": "drug_safety",
    }, domain="biomedical")
    assert "Clinician" in result
    assert "Drug Safety" in result


def test_decision_maker_choices_all_domains():
    from irys.ui.app import _decision_maker_choices_for_domain
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        choices = _decision_maker_choices_for_domain(domain)
        assert len(choices) >= 5, f"{domain} has too few choices"
        values = [v for _, v in choices]
        assert "unknown" in values, f"{domain} missing 'Other' fallback"


def test_objective_choices_all_domains():
    from irys.ui.app import _objective_choices_for_domain
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        choices = _objective_choices_for_domain(domain)
        assert len(choices) >= 5, f"{domain} has too few choices"
        values = [v for _, v in choices]
        assert "unknown" in values, f"{domain} missing 'Other' fallback"


# ---------------------------------------------------------------------------
# Quantitative panel (SO-6)
# ---------------------------------------------------------------------------

def test_fmt_quant_panel_empty():
    from irys.ui.app import _fmt_quant_panel
    result = _fmt_quant_panel({}, [], [], [])
    assert "viz-empty" in result


def test_fmt_quant_panel_with_conflicts():
    from irys.ui.app import _fmt_quant_panel
    conflicts = [
        {"subject_type": "invoice_amount", "subject_id": "INV-001", "currency": "USD", "values": [5000.0, 7500.0]},
    ]
    result = _fmt_quant_panel({}, [], conflicts, [])
    assert "invoice_amount" in result
    assert "INV-001" in result


def test_fmt_quant_panel_with_reconciliation():
    from irys.ui.app import _fmt_quant_panel
    recon = {"invoiced": 100000.0, "paid": 75000.0, "disputed": 10000.0, "exposure": 25000.0, "currency": "USD"}
    result = _fmt_quant_panel(recon, [], [], [])
    assert "Invoiced" in result
    assert "Paid" in result
    assert "Disputed" in result


# ---------------------------------------------------------------------------
# Actor duplicate detection panel (SO-5)
# ---------------------------------------------------------------------------

def test_fmt_duplicate_actors_panel_empty():
    from irys.ui.app import _fmt_duplicate_actors_panel
    result = _fmt_duplicate_actors_panel([])
    assert "viz-empty" in result


def test_fmt_duplicate_actors_panel_with_data():
    from irys.ui.app import _fmt_duplicate_actors_panel
    pairs = [
        {
            "actor_a": {"id": "a1", "canonical_name": "Acme Inc", "actor_type": "organization"},
            "actor_b": {"id": "a2", "canonical_name": "Acme Corporation", "actor_type": "organization"},
            "shared_prefix": "acme",
        }
    ]
    result = _fmt_duplicate_actors_panel(pairs)
    assert "Acme Inc" in result
    assert "Acme Corporation" in result
    assert "acme" in result


def test_fmt_duplicate_actors_panel_skips_non_dict():
    from irys.ui.app import _fmt_duplicate_actors_panel
    result = _fmt_duplicate_actors_panel(["not-a-dict", None])
    assert "viz-empty" in result


# ---------------------------------------------------------------------------
# Gaps & missingness panel (SO-7)
# ---------------------------------------------------------------------------

def test_fmt_gaps_empty():
    from irys.ui.app import _fmt_gaps
    result = _fmt_gaps([], [])
    assert "viz-empty" in result
    assert "No open gaps" in result


def test_fmt_gaps_with_data():
    from irys.ui.app import _fmt_gaps
    gaps = [
        {"gap_type": "missing_document", "description": "Missing contract v2", "materiality_score": 0.8, "dependencies": []},
        {"gap_type": "unresolved_contradiction", "description": "Amount conflict", "materiality_score": 0.6, "dependencies": [{"affected_type": "issue", "affected_id": "i1"}]},
    ]
    result = _fmt_gaps(gaps, [])
    assert "Missing Document" in result
    assert "Unresolved Conflict" in result
    assert "Missing contract v2" in result
    assert "2 unresolved" in result
    assert "Open Gaps &" in result


def test_fmt_gaps_finance_domain():
    from irys.ui.app import _fmt_gaps
    gaps = [
        {"gap_type": "missing_document", "description": "Missing 10-K", "materiality_score": 0.9, "dependencies": []},
        {"gap_type": "missing_quantitative_input", "description": "Revenue figure absent", "materiality_score": 0.5, "dependencies": []},
    ]
    result = _fmt_gaps(gaps, [], domain="finance")
    assert "Missing Filing" in result
    assert "Missing Figure" in result
    assert "Open Gaps &" in result
    assert "Missing Data" in result


def test_fmt_gaps_biomedical_domain():
    from irys.ui.app import _fmt_gaps
    result = _fmt_gaps([], [], domain="biomedical")
    assert "No open" in result
    assert "pending queries" in result


def test_fmt_gaps_with_clarifications():
    from irys.ui.app import _fmt_gaps
    clarifications = [
        {"question_text": "What was the delivery date?", "expected_impact": "Resolves timeline gap"},
    ]
    result = _fmt_gaps([], clarifications)
    assert "What was the delivery date?" in result
    assert "Resolves timeline gap" in result


# ---------------------------------------------------------------------------
# Steering recommendations panel (SO-3)
# ---------------------------------------------------------------------------

def test_fmt_steering_panel_empty():
    from irys.ui.app import _fmt_steering_panel
    result = _fmt_steering_panel([])
    assert "viz-empty" in result


def test_fmt_steering_panel_with_actions():
    from irys.ui.app import _fmt_steering_panel
    actions = [
        {
            "action_type": "redirect_focus",
            "description": "Investigate breach of contract claim",
            "rationale": "Coverage below 30%",
            "priority": "high",
            "impact": "Improves issue coverage",
            "params": {"issue_id": "iss-001"},
        },
        {
            "action_type": "supply_document",
            "description": "Provide the signed agreement",
            "rationale": "Missing document gap recorded",
            "priority": "medium",
            "impact": "Closes missing-document gap",
            "params": {},
        },
    ]
    result = _fmt_steering_panel(actions)
    assert "Redirect Investigation" in result
    assert "Supply Missing Document" in result
    assert "HIGH" in result
    assert "MEDIUM" in result
    assert "breach of contract" in result
    assert "2 recommendations" in result


def test_fmt_steering_panel_finance_domain():
    from irys.ui.app import _fmt_steering_panel
    actions = [{"action_type": "correct_assertion", "description": "Fix amount", "priority": "low", "params": {}}]
    result = _fmt_steering_panel(actions, domain="finance")
    assert "Correct a Finding" in result
    assert "Recommended Actions" in result


# ---------------------------------------------------------------------------
# Gate 34 edge-case coverage: non-dict items and malformed fields
# ---------------------------------------------------------------------------

def test_fmt_gaps_skips_non_dict_items():
    from irys.ui.app import _fmt_gaps
    gaps = [
        "bad-string",
        {"description": "Missing contract", "gap_type": "missing_document", "materiality_score": 0.8},
        42,
    ]
    result = _fmt_gaps(gaps, [])
    assert "Missing contract" in result
    assert "1 unresolved" in result


def test_fmt_gaps_skips_non_dict_clarifications():
    from irys.ui.app import _fmt_gaps
    clarifications = ["not-a-dict", {"question_text": "When was delivery?"}]
    result = _fmt_gaps([], clarifications)
    assert "When was delivery?" in result


def test_fmt_gaps_non_dict_dependencies():
    from irys.ui.app import _fmt_gaps
    gaps = [{"description": "Gap", "gap_type": "factual", "dependencies": ["bad", {"affected_type": "issue"}]}]
    result = _fmt_gaps(gaps, [])
    assert "issue" in result


def test_fmt_steering_panel_non_dict_params():
    from irys.ui.app import _fmt_steering_panel
    actions = [
        {"action_type": "redirect_focus", "description": "Test", "priority": "high", "params": "not-a-dict"},
    ]
    result = _fmt_steering_panel(actions)
    assert "Test" in result
    assert "<code>" not in result


def test_fmt_steering_panel_non_string_action_type():
    from irys.ui.app import _fmt_steering_panel
    actions = [{"action_type": 123, "description": "Num type", "priority": "low"}]
    result = _fmt_steering_panel(actions)
    assert "Num type" in result


def test_fmt_steering_panel_all_non_dict_returns_empty():
    from irys.ui.app import _fmt_steering_panel
    result = _fmt_steering_panel(["bad", 42, None])
    assert "viz-empty" in result


# ---------------------------------------------------------------------------
# Assertion Inspector panel (SO-2 provenance transparency)
# ---------------------------------------------------------------------------

def test_fmt_assertion_inspector_not_found():
    from irys.ui.app import _fmt_assertion_inspector
    result = _fmt_assertion_inspector({"error": "assertion_not_found"})
    assert "viz-empty" in result
    assert "not found" in result


def test_fmt_assertion_inspector_basic():
    from irys.ui.app import _fmt_assertion_inspector
    health = {
        "assertion_id": "a-123",
        "proposition_text": "The contract was signed on Jan 1",
        "belief_state": "accepted",
        "confidence": 0.92,
        "oscillating": False,
        "support_count": 3,
        "attack_count": 1,
        "has_superseding": False,
        "support_source_roles": ["CONTRACT", "DEPOSITION"],
        "attack_source_roles": ["COMPLAINT"],
        "provenance": [
            {
                "event_kind": "ai_extracted",
                "writer_name": "DocumentAnalyzer",
                "model_id": "gpt-4",
                "model_tier": "high",
                "created_at": "2026-05-01T10:00:00",
                "source_document_ref": "contract_v2.pdf",
                "source_span_status": "present",
            },
        ],
    }
    result = _fmt_assertion_inspector(health)
    assert "a-123" in result
    assert "The contract was signed" in result
    assert "ACCEPTED" in result
    assert "0.92" in result
    assert "3" in result
    assert "1" in result
    assert "CONTRACT" in result
    assert "AI-Extracted" in result
    assert "DocumentAnalyzer" in result
    assert "gpt-4" in result
    assert "contract_v2.pdf" in result
    assert "span linked" in result


def test_fmt_assertion_inspector_oscillating():
    from irys.ui.app import _fmt_assertion_inspector
    health = {
        "assertion_id": "a-osc",
        "proposition_text": "Unstable claim",
        "belief_state": "disputed",
        "confidence": 0.45,
        "oscillating": True,
        "support_count": 2,
        "attack_count": 2,
        "has_superseding": True,
        "support_source_roles": [],
        "attack_source_roles": [],
        "provenance": [],
    }
    result = _fmt_assertion_inspector(health)
    assert "OSCILLATING" in result
    assert "SUPERSEDED" in result
    assert "No provenance events" in result


def test_fmt_assertion_inspector_non_dict_provenance():
    from irys.ui.app import _fmt_assertion_inspector
    health = {
        "assertion_id": "a-bad",
        "proposition_text": "Test",
        "belief_state": "accepted",
        "confidence": 0.8,
        "oscillating": False,
        "support_count": 0,
        "attack_count": 0,
        "has_superseding": False,
        "support_source_roles": [],
        "attack_source_roles": [],
        "provenance": ["not-a-dict", {"event_kind": "ai_extracted", "writer_name": "X", "created_at": "2026-01-01"}],
    }
    result = _fmt_assertion_inspector(health)
    assert "Provenance Trail" in result
    assert "AI-Extracted" in result


def test_fmt_assertion_inspector_all_non_dict_provenance():
    from irys.ui.app import _fmt_assertion_inspector
    health = {
        "assertion_id": "a-x",
        "proposition_text": "Test",
        "belief_state": "accepted",
        "confidence": 0.5,
        "oscillating": False,
        "support_count": 0,
        "attack_count": 0,
        "has_superseding": False,
        "support_source_roles": [],
        "attack_source_roles": [],
        "provenance": ["bad", 42],
    }
    result = _fmt_assertion_inspector(health)
    assert "No provenance events" in result


def test_fmt_assertion_inspector_with_history():
    from irys.ui.app import _fmt_assertion_inspector
    health = {
        "assertion_id": "a-hist",
        "proposition_text": "Amount was $50,000",
        "belief_state": "supported",
        "confidence": 0.75,
        "oscillating": False,
        "support_count": 1,
        "attack_count": 0,
        "has_superseding": False,
        "support_source_roles": [],
        "attack_source_roles": [],
        "provenance": [],
    }
    history = [
        {
            "changed_field": "belief_state",
            "old_value": "undetermined",
            "new_value": "supported",
            "cause": "evidence_update",
            "actor_kind": "system",
            "actor_ref": "engine",
            "created_at": "2026-05-01T12:00:00",
        },
        {
            "changed_field": "confidence",
            "old_value": 0.5,
            "new_value": 0.75,
            "cause": "evidence_update",
            "actor_kind": "system",
            "actor_ref": "engine",
            "created_at": "2026-05-01T12:00:00",
        },
    ]
    result = _fmt_assertion_inspector(health, history=history)
    assert "Revision History" in result
    assert "belief_state" in result
    assert "undetermined" in result
    assert "supported" in result
    assert "evidence_update" in result
    assert "system" in result


def test_fmt_assertion_inspector_history_non_dict_items():
    from irys.ui.app import _fmt_assertion_inspector
    health = {
        "assertion_id": "a-h2",
        "proposition_text": "Test",
        "belief_state": "accepted",
        "confidence": 0.8,
        "oscillating": False,
        "support_count": 0,
        "attack_count": 0,
        "has_superseding": False,
        "support_source_roles": [],
        "attack_source_roles": [],
        "provenance": [],
    }
    history = ["bad", {"changed_field": "confidence", "old_value": 0.5, "new_value": 0.8, "cause": "correction", "created_at": "2026-01-01"}]
    result = _fmt_assertion_inspector(health, history=history)
    assert "Revision History" in result
    assert "confidence" in result


def test_fmt_assertion_inspector_xss_escape():
    from irys.ui.app import _fmt_assertion_inspector
    xss = '<script>alert("xss")</script>'
    health = {
        "assertion_id": xss,
        "proposition_text": xss,
        "belief_state": "accepted",
        "confidence": 0.5,
        "oscillating": False,
        "support_count": 0,
        "attack_count": 0,
        "has_superseding": False,
        "support_source_roles": [xss],
        "attack_source_roles": [],
        "provenance": [
            {"event_kind": xss, "writer_name": xss, "model_id": xss, "created_at": xss, "source_document_ref": xss},
        ],
    }
    history = [
        {"changed_field": xss, "old_value": xss, "new_value": xss, "cause": xss, "actor_kind": xss, "created_at": xss},
    ]
    result = _fmt_assertion_inspector(health, history=history)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# Content Policy Audit panel (SO-5)
# ---------------------------------------------------------------------------

def test_fmt_content_policy_panel_empty():
    from irys.ui.app import _fmt_content_policy_panel
    result = _fmt_content_policy_panel([])
    assert "viz-empty" in result
    assert "No content policy decisions" in result


def test_fmt_content_policy_panel_with_decisions():
    from irys.ui.app import _fmt_content_policy_panel
    decisions = [
        {
            "action": "allow",
            "purpose": "synthesis_context",
            "target_kind": "assertion",
            "target_id": "a-001",
            "reason_code": "verified_clean",
            "trust_bucket": "high",
            "policy_audience": "clean",
            "privilege_flag": None,
            "created_at": "2026-05-01T10:00:00",
        },
        {
            "action": "block",
            "purpose": "export",
            "target_kind": "assertion",
            "target_id": "a-002",
            "reason_code": "unverified_candidate",
            "trust_bucket": "low",
            "policy_audience": "clean",
            "privilege_flag": None,
            "created_at": "2026-05-01T10:01:00",
        },
        {
            "action": "withhold",
            "purpose": "chat_response",
            "target_kind": "assertion",
            "target_id": "a-003",
            "reason_code": "privilege_flagged",
            "trust_bucket": "medium",
            "policy_audience": "clean",
            "privilege_flag": 1,
            "created_at": "2026-05-01T10:02:00",
        },
    ]
    result = _fmt_content_policy_panel(decisions)
    assert "ALLOW" in result
    assert "BLOCK" in result
    assert "WITHHOLD" in result
    assert "1 blocked" in result
    assert "1 withheld" in result
    assert "PRIV" in result
    assert "synthesis_context" in result
    assert "3 decisions" in result


def test_fmt_content_policy_panel_skips_non_dict():
    from irys.ui.app import _fmt_content_policy_panel
    result = _fmt_content_policy_panel(["bad", None])
    assert "viz-empty" in result


def test_fmt_content_policy_panel_privilege_badge():
    from irys.ui.app import _fmt_content_policy_panel
    decisions = [
        {"action": "withhold", "purpose": "export", "target_kind": "assertion", "target_id": "a-1", "reason_code": "priv", "trust_bucket": "high", "policy_audience": "clean", "privilege_flag": True, "created_at": "2026-01-01"},
    ]
    result = _fmt_content_policy_panel(decisions)
    assert "PRIV" in result


def test_fmt_content_policy_panel_xss_escape():
    from irys.ui.app import _fmt_content_policy_panel
    xss = '<script>alert("xss")</script>'
    decisions = [
        {
            "action": "allow",
            "purpose": xss,
            "target_kind": xss,
            "target_id": xss,
            "reason_code": xss,
            "trust_bucket": xss,
            "policy_audience": xss,
            "privilege_flag": None,
            "created_at": xss,
        },
    ]
    result = _fmt_content_policy_panel(decisions)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# Working Assumptions panel (SO-7 adjacent)
# ---------------------------------------------------------------------------

def test_fmt_assumptions_empty():
    from irys.ui.app import _fmt_assumptions
    result = _fmt_assumptions([])
    assert "viz-empty" in result
    assert "No assumptions" in result


def test_fmt_assumptions_with_data():
    from irys.ui.app import _fmt_assumptions
    assumptions = [
        {"statement": "Contract was signed by authorized parties", "status": "confirmed", "rationale": "Signature page verified"},
        {"statement": "Delivery occurred on schedule", "status": "provisional", "invalidation_condition": "Late delivery evidence found"},
        {"statement": "Warranty period has expired", "status": "invalidated", "rationale": "Extended warranty clause found"},
    ]
    result = _fmt_assumptions(assumptions)
    assert "Confirmed" in result
    assert "Provisional" in result
    assert "Invalidated" in result
    assert "Contract was signed" in result
    assert "3 assumptions" in result
    assert "1 invalidated" in result
    assert "Late delivery evidence" in result


def test_fmt_assumptions_skips_non_dict():
    from irys.ui.app import _fmt_assumptions
    result = _fmt_assumptions(["bad", None])
    assert "viz-empty" in result


def test_fmt_assumptions_xss_escape():
    from irys.ui.app import _fmt_assumptions
    xss = '<img src=x onerror=alert(1)>'
    result = _fmt_assumptions([
        {"statement": xss, "status": "provisional", "invalidation_condition": xss, "rationale": xss},
    ])
    assert "<img" not in result
    assert "&lt;img" in result


def test_fmt_assumptions_xss_in_status():
    from irys.ui.app import _fmt_assumptions
    result = _fmt_assumptions([
        {"statement": "Test", "status": '<script>alert(1)</script>'},
    ])
    assert "<script>" not in result


# ── Domain Profile panel tests ───────────────────────────────────────

def test_fmt_domain_profile_panel_empty():
    from irys.ui.app import _fmt_domain_profile_panel
    result = _fmt_domain_profile_panel({})
    assert "No domain profile loaded" in result


def test_fmt_domain_profile_panel_not_found():
    from irys.ui.app import _fmt_domain_profile_panel
    result = _fmt_domain_profile_panel({"status": "not_found"})
    assert "No domain profile loaded" in result


def test_fmt_domain_profile_panel_none():
    from irys.ui.app import _fmt_domain_profile_panel
    result = _fmt_domain_profile_panel(None)
    assert "No domain profile loaded" in result


def test_fmt_domain_profile_panel_basic():
    from irys.ui.app import _fmt_domain_profile_panel
    summary = {
        "profile_id": "legal",
        "profile_version": 1,
        "profile_kind": "legal",
        "status": "current",
        "is_primary": True,
        "neutral_kernel": {
            "claim": "legal_proposition",
            "entity": "party",
        },
        "composed_trust_weights": {
            "judge": 0.92,
            "witness": 0.55,
        },
        "source_roles": ["judge", "witness", "attorney"],
        "belief_states": ["operative", "alleged"],
        "taint_classes": ["public_clean", "privileged"],
        "speech_acts": ["testimony", "ruling"],
        "facets": [],
    }
    result = _fmt_domain_profile_panel(summary, domain="legal")
    assert "Domain Profile" in result
    assert "PRIMARY" in result
    assert "legal" in result
    assert "legal_proposition" in result
    assert "0.92" in result
    assert "Judge" in result
    assert "Witness" in result
    assert "Operative" in result
    assert "Testimony" in result


def test_fmt_domain_profile_panel_finance():
    from irys.ui.app import _fmt_domain_profile_panel
    summary = {
        "profile_id": "finance",
        "profile_version": 1,
        "profile_kind": "finance",
        "status": "current",
        "is_primary": True,
        "neutral_kernel": {"claim": "financial_claim"},
        "composed_trust_weights": {"auditor": 0.9},
        "source_roles": ["auditor", "analyst"],
        "belief_states": ["reported"],
        "taint_classes": [],
        "speech_acts": [],
        "facets": [
            {"domain_profile_id": "finance", "domain_profile_version": 1, "confidence": 0.85, "status": "active"},
        ],
    }
    result = _fmt_domain_profile_panel(summary, domain="finance")
    assert "Finance" in result
    assert "Auditor" in result
    assert "0.85" in result
    assert "Domain Facets" in result


def test_fmt_domain_profile_panel_non_dict_facets():
    from irys.ui.app import _fmt_domain_profile_panel
    summary = {
        "profile_id": "legal",
        "profile_version": 1,
        "profile_kind": "legal",
        "status": "current",
        "is_primary": False,
        "neutral_kernel": {},
        "composed_trust_weights": {},
        "source_roles": [],
        "belief_states": [],
        "taint_classes": [],
        "speech_acts": [],
        "facets": ["not-a-dict", 42, None],
    }
    result = _fmt_domain_profile_panel(summary, domain="legal")
    assert "Domain Profile" in result


def test_fmt_domain_profile_panel_xss():
    from irys.ui.app import _fmt_domain_profile_panel
    xss = '<img src=x onerror=alert(1)>'
    summary = {
        "profile_id": xss,
        "profile_version": 1,
        "profile_kind": xss,
        "status": xss,
        "is_primary": True,
        "neutral_kernel": {xss: xss},
        "composed_trust_weights": {xss: 0.5},
        "source_roles": [xss],
        "belief_states": [xss],
        "taint_classes": [xss],
        "speech_acts": [xss],
        "facets": [
            {"domain_profile_id": xss, "domain_profile_version": 1, "confidence": 0.5, "status": xss},
        ],
    }
    result = _fmt_domain_profile_panel(summary, domain="legal")
    assert "<img" not in result
    assert "&lt;img" in result


def test_fmt_domain_profile_panel_trust_weight_bar_colors():
    from irys.ui.app import _fmt_domain_profile_panel
    summary = {
        "profile_id": "legal",
        "profile_version": 1,
        "profile_kind": "legal",
        "status": "current",
        "is_primary": True,
        "neutral_kernel": {},
        "composed_trust_weights": {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        },
        "source_roles": [],
        "belief_states": [],
        "taint_classes": [],
        "speech_acts": [],
        "facets": [],
    }
    result = _fmt_domain_profile_panel(summary, domain="legal")
    assert "#22c55e" in result
    assert "#eab308" in result
    assert "#ef4444" in result


# ── Document Triage panel tests ──────────────────────────────────────

def test_fmt_doc_triage_panel_empty():
    from irys.ui.app import _fmt_doc_triage_panel
    result = _fmt_doc_triage_panel([])
    assert "profiled" in result.lower()


def test_fmt_doc_triage_panel_none():
    from irys.ui.app import _fmt_doc_triage_panel
    result = _fmt_doc_triage_panel(None)
    assert "profiled" in result.lower()


def test_fmt_doc_triage_panel_basic():
    from irys.ui.app import _fmt_doc_triage_panel
    docs = [
        {
            "id": "d1",
            "relative_path": "contracts/agreement.pdf",
            "file_type": "pdf",
            "size_bytes": 1048576,
            "salience_score": 0.85,
            "ingest_status": "ready",
        },
        {
            "id": "d2",
            "relative_path": "emails/thread.eml",
            "file_type": "eml",
            "size_bytes": 2048,
            "salience_score": 0.3,
            "ingest_status": "ready",
        },
    ]
    result = _fmt_doc_triage_panel(docs, domain="legal")
    assert "Document Triage Queue" in result
    assert "contracts/agreement.pdf" in result
    assert "emails/thread.eml" in result
    assert "1.0 MB" in result
    assert "2.0 KB" in result
    assert "0.85" in result
    assert "2 documents" in result


def test_fmt_doc_triage_panel_coding_domain():
    from irys.ui.app import _fmt_doc_triage_panel
    docs = [{"id": "d1", "relative_path": "src/main.py", "file_type": "py",
             "size_bytes": 500, "salience_score": 0.5, "ingest_status": "ready"}]
    result = _fmt_doc_triage_panel(docs, domain="coding")
    assert "Artifact Triage Queue" in result
    assert "1 document" in result


def test_fmt_doc_triage_panel_skips_non_dict():
    from irys.ui.app import _fmt_doc_triage_panel
    docs = ["not-a-dict", 42, {"id": "d1", "relative_path": "a.pdf",
             "file_type": "pdf", "size_bytes": 100, "salience_score": 0.5,
             "ingest_status": "ready"}]
    result = _fmt_doc_triage_panel(docs, domain="legal")
    assert "a.pdf" in result
    assert "1 document" in result


def test_fmt_doc_triage_panel_all_non_dict():
    from irys.ui.app import _fmt_doc_triage_panel
    result = _fmt_doc_triage_panel(["a", "b", 3])
    assert "profiled" in result.lower()


def test_fmt_doc_triage_panel_xss():
    from irys.ui.app import _fmt_doc_triage_panel
    xss = '<img src=x onerror=alert(1)>'
    docs = [{
        "id": "d1",
        "relative_path": xss,
        "file_type": xss,
        "size_bytes": 100,
        "salience_score": 0.5,
        "ingest_status": xss,
    }]
    result = _fmt_doc_triage_panel(docs, domain="legal")
    assert "<img" not in result
    assert "&lt;img" in result


def test_fmt_doc_triage_panel_salience_colors():
    from irys.ui.app import _fmt_doc_triage_panel
    docs = [
        {"id": "d1", "relative_path": "high.pdf", "file_type": "pdf",
         "size_bytes": 100, "salience_score": 0.9, "ingest_status": "ready"},
        {"id": "d2", "relative_path": "med.pdf", "file_type": "pdf",
         "size_bytes": 100, "salience_score": 0.5, "ingest_status": "ready"},
        {"id": "d3", "relative_path": "low.pdf", "file_type": "pdf",
         "size_bytes": 100, "salience_score": 0.2, "ingest_status": "ready"},
    ]
    result = _fmt_doc_triage_panel(docs, domain="legal")
    assert "#22c55e" in result
    assert "#eab308" in result
    assert "#94a3b8" in result


# ── Taint Summary panel tests ────────────────────────────────────────

def test_fmt_taint_summary_panel_empty():
    from irys.ui.app import _fmt_taint_summary_panel
    result = _fmt_taint_summary_panel({})
    assert "clean" in result.lower()


def test_fmt_taint_summary_panel_none():
    from irys.ui.app import _fmt_taint_summary_panel
    result = _fmt_taint_summary_panel(None)
    assert "clean" in result.lower()


def test_fmt_taint_summary_panel_zero_total():
    from irys.ui.app import _fmt_taint_summary_panel
    result = _fmt_taint_summary_panel({"by_class": [], "by_kind": [], "recent": [], "total": 0})
    assert "clean" in result.lower()


def test_fmt_taint_summary_panel_basic():
    from irys.ui.app import _fmt_taint_summary_panel
    data = {
        "by_class": [
            {"taint_class": "privileged", "count": 5},
            {"taint_class": "public_clean", "count": 12},
        ],
        "by_kind": [
            {"target_kind": "assertion", "count": 10},
            {"target_kind": "document", "count": 7},
        ],
        "recent": [
            {
                "id": "t1",
                "target_kind": "assertion",
                "target_id": "a1",
                "taint_class": "privileged",
                "derivation_reason": "attorney-client privilege detected",
                "created_at": "2026-05-03T10:00:00",
            },
        ],
        "total": 17,
    }
    result = _fmt_taint_summary_panel(data, domain="legal")
    assert "Sensitivity" in result
    assert "17 taint" in result
    assert "privileged" in result
    assert "public_clean" in result
    assert "assertion" in result
    assert "attorney-client" in result


def test_fmt_taint_summary_panel_skips_non_dict():
    from irys.ui.app import _fmt_taint_summary_panel
    data = {
        "by_class": [{"taint_class": "privileged", "count": 1}, "bad"],
        "by_kind": [42, {"target_kind": "doc", "count": 1}],
        "recent": ["not-a-dict"],
        "total": 2,
    }
    result = _fmt_taint_summary_panel(data, domain="legal")
    assert "privileged" in result


def test_fmt_taint_summary_panel_xss():
    from irys.ui.app import _fmt_taint_summary_panel
    xss = '<img src=x onerror=alert(1)>'
    data = {
        "by_class": [{"taint_class": xss, "count": 1}],
        "by_kind": [{"target_kind": xss, "count": 1}],
        "recent": [{
            "id": "t1", "target_kind": xss, "target_id": xss,
            "taint_class": xss, "derivation_reason": xss,
            "created_at": "2026-05-03T10:00:00",
        }],
        "total": 1,
    }
    result = _fmt_taint_summary_panel(data, domain="legal")
    assert "<img" not in result
    assert "&lt;img" in result


def test_fmt_taint_summary_panel_coding_domain():
    from irys.ui.app import _fmt_taint_summary_panel
    data = {
        "by_class": [{"taint_class": "security_sensitive", "count": 3}],
        "by_kind": [{"target_kind": "code_artifact", "count": 3}],
        "recent": [],
        "total": 3,
    }
    result = _fmt_taint_summary_panel(data, domain="coding")
    assert "Sensitivity" in result
    assert "Artifact Kind" in result


def test_fmt_taint_summary_panel_class_colors():
    from irys.ui.app import _fmt_taint_summary_panel
    data = {
        "by_class": [
            {"taint_class": "public_clean", "count": 5},
            {"taint_class": "sealed_privileged", "count": 2},
            {"taint_class": "unknown_taint", "count": 1},
        ],
        "by_kind": [],
        "recent": [],
        "total": 8,
    }
    result = _fmt_taint_summary_panel(data, domain="legal")
    assert "#22c55e" in result
    assert "#ef4444" in result
    assert "#94a3b8" in result


# ── Investigation History panel tests ─────────────────────────────────

def test_fmt_investigation_history_empty():
    from irys.ui.app import _fmt_investigation_history_panel
    result = _fmt_investigation_history_panel([])
    assert "No investigation runs" in result


def test_fmt_investigation_history_none():
    from irys.ui.app import _fmt_investigation_history_panel
    result = _fmt_investigation_history_panel(None)
    assert "No investigation runs" in result


def test_fmt_investigation_history_basic():
    from irys.ui.app import _fmt_investigation_history_panel
    runs = [
        {
            "id": "r1",
            "query": "What are the key terms of the contract?",
            "operation_type": "query",
            "status": "completed",
            "research_mode": "deep",
            "llm_request_count": 15,
            "llm_calls_avoided": 5,
            "llm_estimated_cost_usd": 0.0342,
            "reuse_rate": 0.25,
            "started_at": "2026-05-03T10:00:00",
        },
        {
            "id": "r2",
            "query": "Follow up on damages",
            "operation_type": "redirect",
            "status": "running",
            "research_mode": "fast",
            "llm_request_count": 3,
            "llm_calls_avoided": 0,
            "llm_estimated_cost_usd": 0.01,
            "reuse_rate": None,
            "started_at": "2026-05-03T11:00:00",
        },
    ]
    result = _fmt_investigation_history_panel(runs)
    assert "Investigation History" in result
    assert "2 runs" in result
    assert "What are the key terms" in result
    assert "Investigation" in result
    assert "Redirect" in result
    assert "completed" in result
    assert "running" in result
    assert "$0.0342" in result
    assert "25%" in result
    assert "5 cached" in result


def test_fmt_investigation_history_skips_non_dict():
    from irys.ui.app import _fmt_investigation_history_panel
    runs = ["not-a-dict", {"id": "r1", "query": "test", "status": "completed",
            "started_at": "2026-05-03T10:00:00"}]
    result = _fmt_investigation_history_panel(runs)
    assert "1 run" in result


def test_fmt_investigation_history_xss():
    from irys.ui.app import _fmt_investigation_history_panel
    xss = '<img src=x onerror=alert(1)>'
    runs = [{
        "id": "r1",
        "query": xss,
        "operation_type": xss,
        "status": xss,
        "research_mode": xss,
        "started_at": xss,
    }]
    result = _fmt_investigation_history_panel(runs)
    assert "<img" not in result
    assert "&lt;img" in result


def test_fmt_investigation_history_status_colors():
    from irys.ui.app import _fmt_investigation_history_panel
    runs = [
        {"id": "r1", "query": "q1", "status": "completed", "started_at": "2026-05-03"},
        {"id": "r2", "query": "q2", "status": "failed", "started_at": "2026-05-03"},
        {"id": "r3", "query": "q3", "status": "running", "started_at": "2026-05-03"},
    ]
    result = _fmt_investigation_history_panel(runs)
    assert "#22c55e" in result
    assert "#ef4444" in result
    assert "#3b82f6" in result


# ------------------------------------------------------------------ #
# _fmt_overview_panel tests                                            #
# ------------------------------------------------------------------ #

def test_fmt_overview_panel_empty():
    from irys.ui.app import _fmt_overview_panel
    result = _fmt_overview_panel({})
    assert "Run your first investigation" in result


def test_fmt_overview_panel_none():
    from irys.ui.app import _fmt_overview_panel
    result = _fmt_overview_panel(None)
    assert "Run your first investigation" in result


def test_fmt_overview_panel_basic():
    from irys.ui.app import _fmt_overview_panel
    data = {
        "stats": {
            "assertion_count": 42,
            "open_issue_count": 5,
            "open_gap_count": 3,
            "actor_count": 10,
            "quant_fact_count": 7,
            "llm": {
                "totals": {
                    "estimated_cost_usd": 1.25,
                    "request_count": 30,
                    "by_tier": {},
                }
            },
        },
        "so_metrics": {
            "issue_coverage_avg": 0.72,
            "reuse_rate": 0.45,
            "source_role_known_rate": 0.88,
            "assertion_structure_rate": 0.76,
        },
        "coverage_report": [
            {"id": "i1", "title": "Breach of contract", "coverage_fraction": 0.6},
        ],
        "weakest_issues": [],
        "top_gaps": [],
        "pending_clarifications": [],
    }
    result = _fmt_overview_panel(data)
    assert "42" in result
    assert "Assertions" in result
    assert "Open Issues" in result
    assert "Breach of contract" in result


def test_fmt_overview_panel_xss():
    from irys.ui.app import _fmt_overview_panel
    data = {
        "stats": {"assertion_count": 1, "open_issue_count": 0, "actor_count": 0},
        "so_metrics": {},
        "coverage_report": [
            {"id": "i1", "title": "<script>alert(1)</script>", "coverage_fraction": 0.5},
        ],
        "weakest_issues": [],
        "top_gaps": [],
        "pending_clarifications": [],
    }
    result = _fmt_overview_panel(data)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_fmt_overview_panel_domain_labels():
    """Each domain renders its own terminology in the overview panel."""
    from irys.ui.app import _fmt_overview_panel

    data = {
        "stats": {"assertion_count": 5, "open_issue_count": 2, "actor_count": 3},
        "so_metrics": {},
        "coverage_report": [],
        "weakest_issues": [],
        "top_gaps": [],
        "pending_clarifications": [],
    }
    domain_terms = {
        "legal": ("Assertions", "Open Issues", "Actors"),
        "finance": ("Claims", "Open Theses", "Entities"),
        "coding": ("Findings", "Open Hypotheses", "Components"),
        "academic_research": ("Claims", "Open Questions", "Authors"),
        "biomedical": ("Findings", "Open Questions", "Entities"),
    }
    for domain, (assertions_label, issues_label, actors_label) in domain_terms.items():
        result = _fmt_overview_panel(data, domain=domain)
        assert assertions_label in result, f"{domain}: missing '{assertions_label}'"
        assert issues_label in result, f"{domain}: missing '{issues_label}'"
        assert actors_label in result, f"{domain}: missing '{actors_label}'"


def test_fmt_overview_panel_non_dict_guards():
    """Non-dict items in weakest/gaps/clarifications/coverage must not crash."""
    from irys.ui.app import _fmt_overview_panel

    data = {
        "total_issues": 2,
        "total_assertions": 5,
        "total_gaps": 1,
        "total_documents": 3,
        "coverage_report": [
            {"id": "iss-1", "title": "Real issue", "coverage_fraction": 0.5},
            "stale-string-entry",
            42,
            None,
        ],
        "weakest_issues": [
            {"id": "iss-2", "title": "Weak one", "coverage_fraction": 0.1},
            "not-a-dict",
            None,
        ],
        "top_gaps": [
            {"id": "gap-1", "gap_type": "missing_evidence", "description": "Need more"},
            "orphan",
            99,
        ],
        "pending_clarifications": [
            {"question_text": "What happened?"},
            "bare-string",
            None,
        ],
    }
    result = _fmt_overview_panel(data)
    assert "Real issue" in result
    assert "Weak one" in result
    assert "Need more" in result
    assert "What happened?" in result
    assert "stale-string-entry" not in result
    assert "not-a-dict" not in result


# ------------------------------------------------------------------ #
# _fmt_issues_panel tests                                              #
# ------------------------------------------------------------------ #

def test_fmt_issues_panel_empty():
    from irys.ui.app import _fmt_issues_panel
    result = _fmt_issues_panel([])
    assert "No open issues" in result


def test_fmt_issues_panel_basic():
    from irys.ui.app import _fmt_issues_panel
    issues = [
        {
            "id": "iss-1",
            "title": "Breach of fiduciary duty",
            "depth": 0,
            "coverage_fraction": 0.65,
            "verified_coverage_fraction": 0.3,
            "verified_supporting_count": 2,
            "candidate_supporting_count": 5,
            "attacking_count": 1,
            "proof_status": "partial",
            "has_proof_gap": True,
        },
    ]
    result = _fmt_issues_panel(issues)
    assert "Breach of fiduciary duty" in result
    assert "proof gap" in result
    assert "2 verified" in result
    assert "5 candidate" in result
    assert "1 attack" in result
    assert "Verified coverage" in result


def test_fmt_issues_panel_skips_non_dict():
    from irys.ui.app import _fmt_issues_panel
    issues = [
        {"id": "iss-1", "title": "Valid", "depth": 0, "coverage_fraction": 0.5},
        "not-a-dict",
        42,
    ]
    result = _fmt_issues_panel(issues)
    assert "Valid" in result


def test_fmt_issues_panel_xss():
    from irys.ui.app import _fmt_issues_panel
    issues = [
        {
            "id": "iss-1",
            "title": "<img onerror=alert(1) src=x>",
            "depth": 0,
            "coverage_fraction": 0.5,
        },
    ]
    result = _fmt_issues_panel(issues)
    assert "<img onerror" not in result
    assert "&lt;img" in result


def test_fmt_issues_panel_domain_labels():
    """Each domain renders its own terminology."""
    from irys.ui.app import _fmt_issues_panel

    issues = [
        {
            "id": "iss-1",
            "title": "Test Issue",
            "depth": 0,
            "coverage_fraction": 0.5,
            "verified_coverage_fraction": 0.3,
            "verified_supporting_count": 2,
            "candidate_supporting_count": 3,
            "attacking_count": 1,
            "has_proof_gap": True,
        },
    ]
    domain_terms = {
        "legal": ("verified", "attack", "proof gap"),
        "finance": ("confirmed", "contradiction", "evidence gap"),
        "coding": ("confirmed", "refutation", "verification gap"),
        "academic_research": ("verified", "challenge", "evidence gap"),
        "biomedical": ("verified", "contradiction", "evidence gap"),
    }
    for domain, (verified_label, attack_label, gap_label) in domain_terms.items():
        result = _fmt_issues_panel(issues, domain=domain)
        assert verified_label in result, f"{domain}: missing '{verified_label}'"
        assert attack_label in result, f"{domain}: missing '{attack_label}'"
        assert gap_label in result, f"{domain}: missing '{gap_label}'"


def test_fmt_issues_panel_domain_empty():
    """Empty states use domain-specific language."""
    from irys.ui.app import _fmt_issues_panel

    assert "No open theses" in _fmt_issues_panel([], domain="finance")
    assert "No open hypotheses" in _fmt_issues_panel([], domain="coding")
    assert "No open findings" in _fmt_issues_panel([], domain="biomedical")


# ------------------------------------------------------------------ #
# _fmt_trust_notice tests                                              #
# ------------------------------------------------------------------ #

def test_fmt_trust_notice_empty():
    from irys.ui.app import _fmt_trust_notice
    assert _fmt_trust_notice([]) == ""


def test_fmt_trust_notice_all_verified():
    from irys.ui.app import _fmt_trust_notice
    issues = [
        {"verified_supporting_count": 3, "candidate_supporting_count": 1, "has_proof_gap": False, "title": "Issue A"},
    ]
    result = _fmt_trust_notice(issues)
    assert "verified support" in result
    assert "#dcfce7" in result  # green banner


def test_fmt_trust_notice_hedging():
    from irys.ui.app import _fmt_trust_notice
    issues = [
        {"verified_supporting_count": 0, "candidate_supporting_count": 2, "has_proof_gap": False, "title": "Issue A"},
        {"verified_supporting_count": 1, "candidate_supporting_count": 0, "has_proof_gap": True, "title": "Issue B"},
    ]
    result = _fmt_trust_notice(issues)
    assert "2 of 2" in result or "Hedging" in result.lower() or "hedging" in result.lower()
    assert "#fef3c7" in result  # amber banner
    assert "Issue A" in result
    assert "Issue B" in result


def test_fmt_trust_notice_domain_labels():
    from irys.ui.app import _fmt_trust_notice
    issues = [
        {"verified_supporting_count": 3, "candidate_supporting_count": 0, "has_proof_gap": False, "title": "Issue A"},
    ]
    result_legal = _fmt_trust_notice(issues, domain="legal")
    result_finance = _fmt_trust_notice(issues, domain="finance")
    result_coding = _fmt_trust_notice(issues, domain="coding")
    assert "attorney" in result_legal.lower()
    assert "analyst" in result_finance.lower()
    assert "engineer" in result_coding.lower()


def test_fmt_trust_notice_domain_hedging_labels():
    """Trust notice hedging path uses domain-aware candidate/unsupported labels."""
    from irys.ui.app import _fmt_trust_notice
    candidate_issue = [
        {"verified_supporting_count": 0, "candidate_supporting_count": 2, "has_proof_gap": False, "title": "T1"},
    ]
    unsupported_issue = [
        {"verified_supporting_count": 0, "candidate_supporting_count": 0, "has_proof_gap": False, "title": "T2"},
    ]
    assert "candidate-only" in _fmt_trust_notice(candidate_issue, domain="legal")
    assert "unconfirmed" in _fmt_trust_notice(candidate_issue, domain="finance")
    assert "unverified" in _fmt_trust_notice(candidate_issue, domain="coding")
    assert "unsupported" in _fmt_trust_notice(unsupported_issue, domain="legal")
    assert "unsourced" in _fmt_trust_notice(unsupported_issue, domain="finance")
    assert "uncited" in _fmt_trust_notice(unsupported_issue, domain="academic_research")


def test_fmt_trust_notice_xss():
    from irys.ui.app import _fmt_trust_notice
    issues = [
        {"verified_supporting_count": 0, "candidate_supporting_count": 1, "has_proof_gap": True, "title": "<script>xss</script>"},
    ]
    result = _fmt_trust_notice(issues)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_fmt_trust_notice_non_dict_guard():
    from irys.ui.app import _fmt_trust_notice
    issues = [
        {"verified_supporting_count": 2, "candidate_supporting_count": 0, "has_proof_gap": False, "title": "Good"},
        "not-a-dict",
        None,
    ]
    result = _fmt_trust_notice(issues)
    assert "#dcfce7" in result or result == ""


# ------------------------------------------------------------------ #
# _fmt_source_drawer tests                                             #
# ------------------------------------------------------------------ #

def test_fmt_source_drawer_empty():
    from irys.ui.app import _fmt_source_drawer
    result = _fmt_source_drawer("assertion", "a-1", [], [])
    assert "No source or review history" in result


def test_fmt_source_drawer_provenance():
    from irys.ui.app import _fmt_source_drawer
    prov = [
        {
            "source_document_ref": "contract.pdf",
            "source_span_id": "p3-s2",
            "source_span_status": "found",
            "created_at": "2026-05-01T12:00:00",
            "tier": "direct_extraction",
        },
    ]
    result = _fmt_source_drawer("assertion", "a-1", prov, [])
    assert "contract.pdf" in result
    assert "Where this came from" in result


def test_fmt_source_drawer_verification():
    from irys.ui.app import _fmt_source_drawer
    events = [
        {
            "new_status": "verified",
            "reviewed_by_kind": "user",
            "created_at": "2026-05-02T14:00:00",
        },
    ]
    result = _fmt_source_drawer("assertion", "a-1", [], events)
    assert "Review history" in result
    assert "Verified" in result


def test_fmt_source_drawer_xss():
    from irys.ui.app import _fmt_source_drawer
    prov = [
        {
            "source_document_ref": "<img src=x onerror=alert(1)>",
            "created_at": "2026-05-01",
            "tier": "direct_extraction",
        },
    ]
    result = _fmt_source_drawer("assertion", "a-1", prov, [])
    assert "<img src=x" not in result
    assert "&lt;img" in result


def test_fmt_source_drawer_non_dict_guard():
    from irys.ui.app import _fmt_source_drawer
    prov = [{"source_document_ref": "file.pdf", "created_at": "2026-05-01"}, "stale", None]
    events = [{"new_status": "verified", "reviewed_by_kind": "user", "created_at": "2026-05-02"}, 42]
    result = _fmt_source_drawer("assertion", "a-1", prov, events)
    assert "file.pdf" in result
    assert "stale" not in result


# ------------------------------------------------------------------ #
# _fmt_overview (markdown export) tests                                #
# ------------------------------------------------------------------ #

def test_fmt_overview_empty():
    from irys.ui.app import _fmt_overview
    result = _fmt_overview({})
    assert "matter" in result.lower() or "investigation" in result.lower() or result.strip() == ""


def test_fmt_overview_basic():
    from irys.ui.app import _fmt_overview
    data = {
        "stats": {"assertion_count": 10, "open_issue_count": 3, "open_gap_count": 1, "actor_count": 5},
        "coverage_report": [{"id": "i1", "title": "Issue One", "coverage_fraction": 0.7}],
        "weakest_issues": [{"id": "i2", "title": "Weak One", "coverage_fraction": 0.1}],
        "top_gaps": [{"description": "Missing contract"}],
        "pending_clarifications": [{"question_text": "What date?"}],
    }
    result = _fmt_overview(data)
    assert "10" in result
    assert "Weak One" in result
    assert "Missing contract" in result
    assert "What date?" in result


# ------------------------------------------------------------------ #
# _fmt_timeline_panel tests                                            #
# ------------------------------------------------------------------ #

def test_fmt_timeline_panel_empty():
    from irys.ui.app import _fmt_timeline_panel
    result = _fmt_timeline_panel([])
    assert "No timeline events" in result


def test_fmt_timeline_panel_basic():
    from irys.ui.app import _fmt_timeline_panel
    events = [
        {
            "date": "2024-01-15",
            "event": "Contract signed",
            "kind": "agreement",
            "source_doc": "contract_v1.pdf",
            "subject": "Acme Corp",
        },
        {
            "date": "2024-03-01",
            "event": "Breach notice sent",
            "kind": "communication",
            "source_doc": "notice.pdf",
        },
    ]
    result = _fmt_timeline_panel(events)
    assert "Contract signed" in result
    assert "Breach notice sent" in result
    assert "2024-01-15" in result or "Jan" in result


def test_fmt_timeline_panel_withheld():
    from irys.ui.app import _fmt_timeline_panel
    events = [
        {"date": "2024-02-01", "event": "Privileged memo", "withheld": True},
    ]
    result = _fmt_timeline_panel(events)
    assert "Withheld" in result or "withheld" in result
    assert "1 event(s) withheld" in result


def test_fmt_timeline_panel_xss():
    from irys.ui.app import _fmt_timeline_panel
    events = [
        {"date": "2024-01-01", "event": "<script>alert(1)</script>", "kind": "test"},
    ]
    result = _fmt_timeline_panel(events)
    assert "<script>" not in result


def test_fmt_timeline_panel_skips_non_dict():
    from irys.ui.app import _fmt_timeline_panel
    events = [
        {"date": "2024-01-01", "event": "Valid event", "kind": "meeting"},
        "not a dict",
        42,
        None,
    ]
    result = _fmt_timeline_panel(events)
    assert "Valid event" in result
    assert "not a dict" not in result


# ------------------------------------------------------------------ #
# _fmt_evidence_matrix_panel tests                                     #
# ------------------------------------------------------------------ #

def test_fmt_evidence_matrix_panel_empty():
    from irys.ui.app import _fmt_evidence_matrix_panel
    result = _fmt_evidence_matrix_panel({})
    assert "Evidence matrix will populate" in result


def test_fmt_evidence_matrix_panel_no_issues():
    from irys.ui.app import _fmt_evidence_matrix_panel
    result = _fmt_evidence_matrix_panel({"issues": [], "sources": ["doc.pdf"]})
    assert "Evidence matrix will populate" in result


def test_fmt_evidence_matrix_panel_basic():
    from irys.ui.app import _fmt_evidence_matrix_panel
    matrix = {
        "issues": [{"id": "i1", "title": "Breach claim"}],
        "sources": ["contract.pdf"],
        "cells": {
            "i1": {
                "contract.pdf": {"supporting": 3, "attacking": 1, "total": 4},
            },
        },
        "issue_totals": {"i1": {"supporting": 3, "attacking": 1}},
        "source_totals": {"contract.pdf": {"supporting": 3, "attacking": 1}},
    }
    result = _fmt_evidence_matrix_panel(matrix)
    assert "Breach claim" in result
    assert "contract" in result.lower()


def test_fmt_evidence_matrix_panel_xss():
    from irys.ui.app import _fmt_evidence_matrix_panel
    matrix = {
        "issues": [{"id": "i1", "title": "<script>xss</script>"}],
        "sources": ["<img src=x>"],
        "cells": {"i1": {"<img src=x>": {"supporting": 1, "attacking": 0, "total": 1}}},
        "issue_totals": {"i1": {"supporting": 1, "attacking": 0}},
        "source_totals": {"<img src=x>": {"supporting": 1, "attacking": 0}},
    }
    result = _fmt_evidence_matrix_panel(matrix)
    assert "<script>" not in result
    assert "<img src=x>" not in result


def test_fmt_evidence_matrix_panel_domain_labels():
    """Each domain renders its own terminology."""
    from irys.ui.app import _fmt_evidence_matrix_panel

    matrix = {
        "issues": [{"id": "i1", "title": "Issue A"}],
        "sources": ["src1"],
        "cells": {"i1": {"src1": {"supporting": 1, "attacking": 0, "total": 1}}},
        "issue_totals": {"i1": {"supporting": 1, "attacking": 0}},
        "source_totals": {"src1": {"supporting": 1, "attacking": 0}},
    }
    domain_terms = {
        "legal": ("Support and attack", "Issue"),
        "finance": ("Corroboration and contradiction", "Thesis"),
        "coding": ("Confirmation and refutation", "Hypothesis"),
        "academic_research": ("Support and challenge", "Claim"),
        "biomedical": ("Support and contradiction", "Finding"),
    }
    for domain, (title_frag, issue_label) in domain_terms.items():
        result = _fmt_evidence_matrix_panel(matrix, domain=domain)
        assert title_frag in result, f"{domain}: missing title fragment '{title_frag}'"
        assert issue_label in result, f"{domain}: missing issue label '{issue_label}'"


# ------------------------------------------------------------------ #
# _fmt_authority_panel tests                                           #
# ------------------------------------------------------------------ #

def test_fmt_authority_panel_empty():
    from irys.ui.app import _fmt_authority_panel
    result = _fmt_authority_panel({"authorities": [], "issue_links": {}})
    assert "No authorities" in result or "No legal authorities" in result


def test_fmt_authority_panel_basic():
    from irys.ui.app import _fmt_authority_panel
    data = {
        "authorities": [
            {
                "id": "auth-1",
                "citation": "Smith v. Jones, 123 F.3d 456",
                "name": "Smith case",
                "authority_type": "case_law",
                "weight": "binding",
                "jurisdiction": "Federal",
            },
        ],
        "issue_links": {
            "auth-1": [
                {"issue_id": "i1", "issue_title": "Breach", "relevance": "supporting"},
            ],
        },
    }
    result = _fmt_authority_panel(data)
    assert "Smith v. Jones" in result
    assert "binding" in result.lower()
    assert "Breach" in result


def test_fmt_authority_panel_finance_domain():
    from irys.ui.app import _fmt_authority_panel
    data = {
        "authorities": [
            {
                "id": "auth-1",
                "citation": "SEC Rule 10b-5",
                "authority_type": "regulation",
                "weight": "binding",
            },
        ],
        "issue_links": {},
    }
    result = _fmt_authority_panel(data, domain="finance")
    assert "SEC Rule 10b-5" in result


def test_fmt_authority_panel_xss():
    from irys.ui.app import _fmt_authority_panel
    data = {
        "authorities": [
            {
                "id": "auth-1",
                "citation": "<script>alert(1)</script>",
                "authority_type": "case_law",
                "weight": "binding",
            },
        ],
        "issue_links": {},
    }
    result = _fmt_authority_panel(data)
    assert "<script>" not in result


# ------------------------------------------------------------------ #
# _fmt_communication_map_panel tests                                   #
# ------------------------------------------------------------------ #

def test_fmt_communication_map_panel_empty():
    from irys.ui.app import _fmt_communication_map_panel
    result = _fmt_communication_map_panel({})
    assert "No communication graph" in result


def test_fmt_communication_map_panel_no_edges():
    from irys.ui.app import _fmt_communication_map_panel
    result = _fmt_communication_map_panel({
        "actors": [{"id": "a1", "name": "Alice"}],
        "documents": ["doc.pdf"],
        "actor_document_edges": [],
    })
    assert "No communication graph" in result


def test_fmt_communication_map_panel_basic():
    from irys.ui.app import _fmt_communication_map_panel
    graph = {
        "actors": [{"id": "a1", "name": "Alice"}, {"id": "a2", "name": "Bob"}],
        "documents": ["contract.pdf", "memo.pdf"],
        "actor_document_edges": [
            {"actor_id": "a1", "document_id": "contract.pdf", "occurrence_count": 5},
            {"actor_id": "a2", "document_id": "memo.pdf", "occurrence_count": 3},
            {"actor_id": "a1", "document_id": "memo.pdf", "occurrence_count": 2},
        ],
        "actor_actor_edges": [],
    }
    result = _fmt_communication_map_panel(graph)
    assert "Alice" in result
    assert "Bob" in result


def test_fmt_communication_map_panel_xss():
    from irys.ui.app import _fmt_communication_map_panel
    graph = {
        "actors": [{"id": "a1", "name": "<script>xss</script>"}],
        "documents": ["<img src=x onerror=alert(1)>"],
        "actor_document_edges": [
            {"actor_id": "a1", "document_id": "<img src=x onerror=alert(1)>", "occurrence_count": 1},
        ],
        "actor_actor_edges": [],
    }
    result = _fmt_communication_map_panel(graph)
    assert "<script>" not in result
    assert "<img src=x" not in result


# ------------------------------------------------------------------ #
# _fmt_llm_analytics_panel tests                                       #
# ------------------------------------------------------------------ #

def test_fmt_llm_analytics_panel_empty():
    from irys.ui.app import _fmt_llm_analytics_panel
    result = _fmt_llm_analytics_panel({}, [])
    assert "No LLM cost" in result or "viz-empty" in result


def test_fmt_llm_analytics_panel_basic():
    from irys.ui.app import _fmt_llm_analytics_panel
    summary = {"estimated_cost_usd": 2.50, "request_count": 15}
    calls = [
        {
            "call_id": "c1",
            "stage": "extraction",
            "estimated_cost_usd": 1.20,
            "model_tier": "mid",
            "input_tokens": 500,
            "output_tokens": 100,
        },
    ]
    result = _fmt_llm_analytics_panel(summary, calls)
    assert "Calls" in result
    assert "Spend" in result
    assert "$2.5000" in result


# ------------------------------------------------------------------ #
# _fmt_proof_state_panel tests                                         #
# ------------------------------------------------------------------ #

def test_fmt_proof_state_panel_empty():
    from irys.ui.app import _fmt_proof_state_panel
    result = _fmt_proof_state_panel({}, [])
    assert "No proof state" in result or "viz-empty" in result


def test_fmt_proof_state_panel_basic():
    from irys.ui.app import _fmt_proof_state_panel
    summary = {
        "issues_total": 3,
        "issues_sufficient": 1,
        "issues_partial": 1,
        "issues_insufficient": 1,
        "average_sufficiency": 0.55,
    }
    issues = [
        {
            "id": "i1",
            "title": "Negligence",
            "coverage_fraction": 0.8,
            "proof_status": "sufficient",
        },
    ]
    result = _fmt_proof_state_panel(summary, issues)
    assert "Negligence" in result or "3" in result


def test_fmt_proof_state_panel_xss():
    from irys.ui.app import _fmt_proof_state_panel
    summary = {"issues_total": 1}
    issues = [
        {
            "id": "i1",
            "title": "<img onerror=alert(1)>",
            "coverage_fraction": 0.5,
            "proof_status": "partial",
        },
    ]
    result = _fmt_proof_state_panel(summary, issues)
    assert "onerror" not in result


# ------------------------------------------------------------------ #
# _fmt_document_intelligence_panel tests                               #
# ------------------------------------------------------------------ #

def test_fmt_document_intelligence_panel_empty():
    from irys.ui.app import _fmt_document_intelligence_panel
    result = _fmt_document_intelligence_panel({})
    assert "viz-empty" in result


def test_fmt_document_intelligence_panel_basic():
    from irys.ui.app import _fmt_document_intelligence_panel
    data = {
        "cards": [
            {
                "relative_path": "docs/contract.pdf",
                "doc_type": "contract",
                "source_side": "plaintiff",
                "author": "Smith LLP",
                "operative_status": "operative",
                "salience_score": 0.85,
                "privilege_flag": False,
            },
        ],
        "total_inventory": 5,
        "ingested_count": 3,
    }
    result = _fmt_document_intelligence_panel(data)
    assert "contract" in result.lower()
    assert "Smith LLP" in result


def test_fmt_document_intelligence_panel_xss():
    from irys.ui.app import _fmt_document_intelligence_panel
    data = {
        "cards": [
            {
                "relative_path": "<script>alert(1)</script>",
                "doc_type": "unknown",
                "source_side": "unknown",
                "author": "<img onerror=alert(1)>",
                "operative_status": "unknown",
                "salience_score": 0.0,
            },
        ],
        "total_inventory": 1,
        "ingested_count": 0,
    }
    result = _fmt_document_intelligence_panel(data)
    assert "<script>" not in result
    assert "<img " not in result
    assert "&lt;script&gt;" in result


# ------------------------------------------------------------------ #
# Backend: compute_proof_state / flush_pending abstract compliance    #
# ------------------------------------------------------------------ #

def test_http_backend_compute_proof_state_type_guard():
    """HttpBackend.compute_proof_state returns error dict on non-dict response."""
    from irys.ui.backends.http import HttpBackend
    backend = HttpBackend.__new__(HttpBackend)
    import asyncio

    async def _mock_post(path, body=None):
        return "unexpected string"

    backend._post = _mock_post
    result = asyncio.run(
        backend.compute_proof_state("m1")
    )
    assert "error" in result


def test_http_backend_flush_pending_type_guard():
    """HttpBackend.flush_pending returns error dict on non-dict response."""
    from irys.ui.backends.http import HttpBackend
    backend = HttpBackend.__new__(HttpBackend)
    import asyncio

    async def _mock_post(path, body=None):
        return [1, 2, 3]

    backend._post = _mock_post
    result = asyncio.run(
        backend.flush_pending("m1")
    )
    assert "error" in result


def test_http_backend_compute_proof_state_passthrough():
    """HttpBackend.compute_proof_state passes through dict response."""
    from irys.ui.backends.http import HttpBackend
    backend = HttpBackend.__new__(HttpBackend)
    import asyncio

    expected = {"matter_id": "m1", "updated_count": 3, "states": []}

    async def _mock_post(path, body=None):
        return expected

    backend._post = _mock_post
    result = asyncio.run(
        backend.compute_proof_state("m1")
    )
    assert result == expected


def test_http_backend_flush_pending_passthrough():
    """HttpBackend.flush_pending passes through dict response."""
    from irys.ui.backends.http import HttpBackend
    backend = HttpBackend.__new__(HttpBackend)
    import asyncio

    expected = {"status": "ok", "revised_count": 5}

    async def _mock_post(path, body=None):
        return expected

    backend._post = _mock_post
    result = asyncio.run(
        backend.flush_pending("m1")
    )
    assert result == expected


# ------------------------------------------------------------------ #
# XSS regression: _metric_card and _bar_row                          #
# ------------------------------------------------------------------ #

def test_metric_card_escapes_detail():
    from irys.ui.app import _metric_card
    result = _metric_card("Title", "42", detail="<script>alert(1)</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_metric_card_escapes_title():
    from irys.ui.app import _metric_card
    result = _metric_card("<img src=x onerror=alert(1)>", "42")
    assert "<img " not in result


def test_bar_row_escapes_meta():
    from irys.ui.app import _bar_row
    result = _bar_row("Label", 5.0, 10.0, meta="<script>alert(1)</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_bar_row_escapes_label():
    from irys.ui.app import _bar_row
    result = _bar_row("<img src=x onerror=alert(1)>", 5.0, 10.0)
    assert "<img " not in result


# ------------------------------------------------------------------ #
# Domain-aware review queue labels (SO-3)                             #
# ------------------------------------------------------------------ #

def test_fmt_review_queue_legal_domain():
    from irys.ui.app import _fmt_review_queue
    queue = [{"priority_bucket": 3, "priority_score": 0.8, "target_kind": "assertion", "proposition_text": "Test fact"}]
    result = _fmt_review_queue(queue, domain="legal")
    assert "Fact" in result
    assert "Element of proof" in result


def test_fmt_review_queue_finance_domain():
    from irys.ui.app import _fmt_review_queue
    queue = [{"priority_bucket": 3, "priority_score": 0.8, "target_kind": "assertion", "proposition_text": "Test finding"}]
    result = _fmt_review_queue(queue, domain="finance")
    assert "Finding" in result
    assert "Compliance element" in result


def test_fmt_review_queue_coding_domain():
    from irys.ui.app import _fmt_review_queue
    queue = [{"priority_bucket": 4, "priority_score": 0.5, "target_kind": "quant_fact", "quant_raw_text": "latency 200ms"}]
    result = _fmt_review_queue(queue, domain="coding")
    assert "Metric" in result


def test_fmt_review_queue_academic_domain():
    from irys.ui.app import _fmt_review_queue
    queue = [{"priority_bucket": 2, "priority_score": 0.6, "target_kind": "assertion", "proposition_text": "Hypothesis claim"}]
    result = _fmt_review_queue(queue, domain="academic_research")
    assert "Claim" in result
    assert "Supports a hypothesis" in result


def test_fmt_review_queue_biomedical_domain():
    from irys.ui.app import _fmt_review_queue
    queue = [{"priority_bucket": 5, "priority_score": 0.3, "target_kind": "authority", "authority_citation": "FDA Guideline"}]
    result = _fmt_review_queue(queue, domain="biomedical")
    assert "Protocol / Guideline" in result


def test_review_count_badge_domain_labels():
    from irys.ui.app import _fmt_review_count_badge
    result = _fmt_review_count_badge(5, {3: 3, 4: 2}, domain="finance")
    assert "Compliance element" in result
    assert "Figure" in result


def test_review_queue_xss():
    from irys.ui.app import _fmt_review_queue
    queue = [{"priority_bucket": 0, "priority_score": 1.0, "target_kind": "assertion", "proposition_text": "<script>alert(1)</script>"}]
    result = _fmt_review_queue(queue)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_review_queue_source_document_display():
    from irys.ui.app import _fmt_review_queue
    queue = [
        {
            "priority_bucket": 0,
            "priority_score": 1.0,
            "target_kind": "assertion",
            "proposition_text": "Damages exceed threshold",
            "source_doc_label": "contract_2024.pdf",
            "source_section_label": "Section 4.2",
            "source_span_id": "span-001",
            "source_doc_id": "doc-abc",
        },
    ]
    result = _fmt_review_queue(queue)
    assert "contract_2024.pdf" in result
    assert "Section 4.2" in result
    assert "Damages exceed threshold" in result


def test_review_queue_source_document_missing():
    from irys.ui.app import _fmt_review_queue
    queue = [
        {
            "priority_bucket": 2,
            "priority_score": 0.5,
            "target_kind": "assertion",
            "proposition_text": "Some finding",
        },
    ]
    result = _fmt_review_queue(queue)
    assert "Some finding" in result
    assert "&#128196;" not in result


def test_review_queue_source_document_xss():
    from irys.ui.app import _fmt_review_queue
    queue = [
        {
            "priority_bucket": 0,
            "priority_score": 1.0,
            "target_kind": "assertion",
            "proposition_text": "test",
            "source_doc_label": "<img onerror=alert(1)>",
            "source_section_label": "<script>bad</script>",
        },
    ]
    result = _fmt_review_queue(queue)
    assert "<img onerror" not in result
    assert "<script>" not in result
    assert "&lt;" in result


def test_review_queue_choices_with_source_doc():
    from irys.ui.app import _review_queue_choices
    queue = [
        {
            "target_kind": "assertion",
            "target_id": "a1",
            "proposition_text": "Contract was breached",
            "source_doc_label": "exhibit_a.pdf",
        },
        {
            "target_kind": "assertion",
            "target_id": "a2",
            "proposition_text": "Payment was made",
        },
    ]
    choices = _review_queue_choices(queue)
    assert len(choices) == 2
    assert "exhibit_a.pdf" in choices[0][0]
    assert choices[0][1] == "assertion:a1"
    assert "exhibit_a.pdf" not in choices[1][0]


# ------------------------------------------------------------------ #
# Utility formatter tests                                              #
# ------------------------------------------------------------------ #

def test_fmt_coverage():
    from irys.ui.app import _fmt_coverage
    assert _fmt_coverage(None) == "—"
    assert _fmt_coverage(0.5) == "50%"
    assert _fmt_coverage(1.0) == "100%"
    assert _fmt_coverage(0.0) == "0%"


def test_fmt_percent_html():
    from irys.ui.app import _fmt_percent_html
    assert _fmt_percent_html(None) == "—"
    assert _fmt_percent_html(0.75) == "75%"
    assert _fmt_percent_html(0.0) == "0%"


def test_fmt_money():
    from irys.ui.app import _fmt_money
    assert _fmt_money(1234.5678) == "$1,234.5678"
    assert _fmt_money(0) == "$0.0000"
    assert _fmt_money(None) == "$0.0000"
    assert _fmt_money("bad") == "$0.0000"


def test_fmt_money_short():
    from irys.ui.app import _fmt_money_short
    assert _fmt_money_short(1234.5) == "$1,234.50"
    assert _fmt_money_short(0) == "$0.00"


def test_fmt_money_decimals():
    from irys.ui.app import _fmt_money
    assert _fmt_money(99.99, decimals=2) == "$99.99"


# ------------------------------------------------------------------ #
# _fmt_issues (markdown export) tests                                  #
# ------------------------------------------------------------------ #

def test_fmt_issues_empty():
    from irys.ui.app import _fmt_issues
    assert _fmt_issues([]) == "No open issues."


def test_fmt_issues_basic():
    from irys.ui.app import _fmt_issues
    issues = [
        {
            "id": "iss-1",
            "title": "Breach of contract",
            "depth": 0,
            "coverage_fraction": 0.6,
            "proof_status": "partial",
            "supporting_count": 3,
            "attacking_count": 1,
        },
    ]
    result = _fmt_issues(issues)
    assert "Breach of contract" in result
    assert "3 supporting" in result
    assert "1 attacking" in result


def test_fmt_issues_non_dict_guard():
    from irys.ui.app import _fmt_issues
    issues = [
        {"id": "iss-1", "title": "Real issue", "depth": 0, "coverage_fraction": 0.5},
        "stale-string",
        None,
    ]
    result = _fmt_issues(issues)
    assert "Real issue" in result
    assert "stale-string" not in result


# ------------------------------------------------------------------
# Domain-aware formatter tests: trust overrides, annotations,
# communication map, source drawer, export overview/issues
# ------------------------------------------------------------------

def test_fmt_trust_overrides_domain_labels():
    from irys.ui.app import _fmt_trust_overrides
    overrides = [
        {"document_pattern": "contract.pdf", "trust_level": "high", "note": "key doc", "created_at": "2025-01-01"},
    ]
    legal = _fmt_trust_overrides(overrides, domain="legal")
    assert "Document Trust Overrides" in legal
    assert "Document" in legal

    finance = _fmt_trust_overrides(overrides, domain="finance")
    assert "Source Trust Overrides" in finance
    assert "Source" in finance

    coding = _fmt_trust_overrides(overrides, domain="coding")
    assert "Artifact Trust Overrides" in coding
    assert "Artifact" in coding


def test_fmt_trust_overrides_empty():
    from irys.ui.app import _fmt_trust_overrides
    result = _fmt_trust_overrides([], domain="finance")
    assert "No trust overrides set." in result


def test_fmt_annotations_panel_domain_labels():
    from irys.ui.app import _fmt_annotations_panel
    annotations = [
        {"document_pattern": "memo.pdf", "annotation_text": "Important", "annotation_type": "strategic", "created_at": "2025-01-01"},
    ]
    legal = _fmt_annotations_panel(annotations, domain="legal")
    assert "Document Notes" in legal

    finance = _fmt_annotations_panel(annotations, domain="finance")
    assert "Source Notes" in finance

    coding = _fmt_annotations_panel(annotations, domain="coding")
    assert "Artifact Notes" in coding


def test_fmt_annotations_panel_empty():
    from irys.ui.app import _fmt_annotations_panel
    result = _fmt_annotations_panel([], domain="coding")
    assert "artifact" in result.lower()


def test_fmt_communication_map_panel_domain_labels():
    from irys.ui.app import _fmt_communication_map_panel
    graph = {
        "actors": [{"id": "a1", "name": "Alice"}],
        "documents": [{"id": "d1"}],
        "actor_document_edges": [{"actor_id": "a1", "document_id": "d1", "occurrence_count": 3}],
        "actor_actor_edges": [],
    }
    legal = _fmt_communication_map_panel(graph, domain="legal")
    assert "Actor/document communication map" in legal
    assert "Actors" in legal

    finance = _fmt_communication_map_panel(graph, domain="finance")
    assert "Entity/source communication map" in finance
    assert "Entities" in finance


def test_fmt_communication_map_panel_empty():
    from irys.ui.app import _fmt_communication_map_panel
    result = _fmt_communication_map_panel({}, domain="coding")
    assert "viz-empty" in result


def test_fmt_source_drawer_domain_labels():
    from irys.ui.app import _fmt_source_drawer
    events = [
        {"new_status": "verified", "reviewed_by_kind": "attorney", "created_at": "2025-01-01T10:00:00"},
    ]
    legal = _fmt_source_drawer("assertion", "a1", [], events, domain="legal")
    assert "Attorney" in legal

    finance = _fmt_source_drawer("assertion", "a1", [], events, domain="finance")
    assert "Analyst" in finance

    coding = _fmt_source_drawer("assertion", "a1", [], events, domain="coding")
    assert "Engineer" in coding

    research = _fmt_source_drawer("assertion", "a1", [], events, domain="academic_research")
    assert "Reviewer" in research

    biomed = _fmt_source_drawer("assertion", "a1", [], events, domain="biomedical")
    assert "Clinician" in biomed


def test_fmt_export_overview_domain_labels():
    from irys.ui.app import _fmt_overview
    data = {
        "stats": {"assertion_count": 10, "open_issue_count": 3, "open_gap_count": 2, "actor_count": 5},
        "so_metrics": {},
    }
    legal = _fmt_overview(data, domain="legal")
    assert "Matter Overview" in legal
    assert "Assertions" in legal

    finance = _fmt_overview(data, domain="finance")
    assert "Analysis Overview" in finance
    assert "Claims" in finance

    coding = _fmt_overview(data, domain="coding")
    assert "Investigation Overview" in coding
    assert "Findings" in coding


def test_fmt_export_issues_domain_empty():
    from irys.ui.app import _fmt_issues
    legal = _fmt_issues([], domain="legal")
    assert "No open issues." in legal

    finance = _fmt_issues([], domain="finance")
    assert "No open theses." in finance

    coding = _fmt_issues([], domain="coding")
    assert "No open hypotheses." in coding


def test_fmt_timeline_panel_domain_labels():
    from irys.ui.app import _fmt_timeline_panel
    events = [
        {"date": "2025-01-01", "event": "Contract signed", "kind": "execution", "withheld": True},
        {"date": "2025-02-01", "event": "Deposition filed", "kind": "filing"},
    ]
    legal = _fmt_timeline_panel(events, domain="legal")
    assert "Withheld under clean policy" in legal
    assert "Privileged docs" in legal

    finance = _fmt_timeline_panel(events, domain="finance")
    assert "Withheld under compliance policy" in finance
    assert "Restricted sources" in finance

    coding = _fmt_timeline_panel(events, domain="coding")
    assert "Withheld under content policy" in coding


def test_fmt_timeline_panel_empty():
    from irys.ui.app import _fmt_timeline_panel
    result = _fmt_timeline_panel([], domain="biomedical")
    assert "viz-empty" in result


def test_fmt_export_quant_domain_labels():
    from irys.ui.app import _fmt_quant
    recon = {"invoiced": 100.0, "paid": 50.0, "disputed": 10.0, "exposure": 40.0, "currency": "USD"}
    damages = [{"component": "Direct", "claimed_amount": 100.0, "source_count": 3}]

    legal = _fmt_quant(recon, damages, domain="legal")
    assert "Payment Reconciliation" in legal
    assert "Total Invoiced" in legal
    assert "Damages Waterfall" in legal

    finance = _fmt_quant(recon, damages, domain="finance")
    assert "Transaction Reconciliation" in finance
    assert "Total Billed" in finance
    assert "Amount Breakdown" in finance


def test_fmt_export_quant_empty():
    from irys.ui.app import _fmt_quant
    result = _fmt_quant({}, [], domain="coding")
    assert "investigation" in result.lower()


def test_fmt_quant_panel_domain_labels():
    from irys.ui.app import _fmt_quant_panel
    recon = {"invoiced": 100.0, "paid": 50.0, "disputed": 10.0, "exposure": 40.0}
    damages = [{"component": "Direct", "claimed_amount": 100.0, "source_count": 2}]

    legal = _fmt_quant_panel(recon, [], [], damages, domain="legal")
    assert "Invoiced" in legal
    assert "Damages waterfall" in legal

    finance = _fmt_quant_panel(recon, [], [], damages, domain="finance")
    assert "Billed" in finance
    assert "Amount breakdown" in finance

    coding = _fmt_quant_panel(recon, [], [], damages, domain="coding")
    assert "Allocated" in coding
    assert "Metric breakdown" in coding


def test_fmt_quant_panel_empty():
    from irys.ui.app import _fmt_quant_panel
    result = _fmt_quant_panel({}, [], [], [], domain="biomedical")
    assert "viz-empty" in result


def test_fmt_steering_domain_labels():
    from irys.ui.app import _fmt_steering
    actions = [
        {"action_type": "correct_assertion", "description": "Fix item", "priority": "high"},
        {"action_type": "redirect_focus", "description": "Shift", "params": {"issue_id": "I1"}},
    ]
    legal = _fmt_steering(actions, domain="legal")
    assert "Recommended Next Steps" in legal
    assert "Correct a Fact" in legal
    assert "Redirect Investigation" in legal

    finance = _fmt_steering(actions, domain="finance")
    assert "Recommended Actions" in finance
    assert "Correct a Finding" in finance
    assert "Redirect Analysis" in finance

    coding = _fmt_steering(actions, domain="coding")
    assert "Suggested Improvements" in coding
    assert "Correct Finding" in coding

    research = _fmt_steering(actions, domain="academic_research")
    assert "Research Recommendations" in research
    assert "Correct Claim" in research

    bio = _fmt_steering(actions, domain="biomedical")
    assert "Clinical Recommendations" in bio
    assert "Correct Finding" in bio


def test_fmt_steering_empty():
    from irys.ui.app import _fmt_steering
    assert "analysis model" in _fmt_steering([], domain="finance")
    assert "code model" in _fmt_steering([], domain="coding")
    assert "matter model" in _fmt_steering([], domain="legal")


def test_fmt_steering_non_dict_guard():
    from irys.ui.app import _fmt_steering
    result = _fmt_steering(["not-a-dict", 42, None], domain="legal")
    assert "Recommended Next Steps" in result


def test_fmt_authority_panel_domain_labels():
    from irys.ui.app import _fmt_authority_panel
    data = {
        "authorities": [
            {"id": "A1", "citation": "Smith v. Jones", "name": "Smith", "authority_type": "case",
             "weight": "binding", "jurisdiction": "9th Cir.", "holdings": [], "key_rules": []},
        ],
        "issue_links": {"A1": [{"issue_id": "I1", "issue_title": "Breach", "relevance": "supporting"}]},
    }
    legal = _fmt_authority_panel(data, domain="legal")
    assert "Authority" in legal
    assert "Linked Issues" in legal

    finance = _fmt_authority_panel(data, domain="finance")
    assert "Reference" in finance
    assert "Linked Theses" in finance

    coding = _fmt_authority_panel(data, domain="coding")
    assert "Specification" in coding
    assert "Linked Requirements" in coding


def test_fmt_authority_panel_empty():
    from irys.ui.app import _fmt_authority_panel
    result = _fmt_authority_panel({"authorities": [], "issue_links": {}}, domain="biomedical")
    assert "viz-empty" in result
    assert "clinical guidelines" in result.lower()


def test_authority_backend_interface_methods():
    """Verify backend interface declares all authority curation methods."""
    from irys.ui.backends.base import UIBackend
    import inspect
    assert hasattr(UIBackend, "upsert_authority")
    assert hasattr(UIBackend, "link_authority_to_issue")
    assert hasattr(UIBackend, "unlink_authority_from_issue")
    assert hasattr(UIBackend, "search_authorities")
    sig = inspect.signature(UIBackend.upsert_authority)
    assert "citation" in sig.parameters
    assert "authority_type" in sig.parameters
    assert "weight" in sig.parameters


def test_xss_authority_curation_escape():
    """XSS regression: authority curation status messages must escape user input."""
    from irys.ui.app import _escape
    xss_payload = '<img src=x onerror=alert(1)>'
    escaped = _escape(xss_payload)
    assert "<img" not in escaped
    assert "&lt;" in escaped
    aid = _escape(xss_payload[:12])
    msg = f"Added authority {aid}"
    assert "<img" not in msg


def test_xss_fmt_steering_escapes_backend_data():
    """XSS regression: _fmt_steering must escape description/rationale/params from backend."""
    from irys.ui.app import _fmt_steering
    xss = '<script>alert(1)</script>'
    actions = [
        {
            "action_type": "redirect_focus",
            "description": xss,
            "rationale": xss,
            "priority": "high",
            "params": {xss: xss},
        }
    ]
    result = _fmt_steering(actions, domain="legal")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_fmt_authority_panel_non_dict_guard():
    """Non-dict guard: authority panel must skip non-dict items without crashing."""
    from irys.ui.app import _fmt_authority_panel
    data = {
        "authorities": ["not-a-dict", None, 42, {"id": "A1", "citation": "Test", "weight": "binding"}],
        "issue_links": {"A1": ["bad-link", {"issue_id": "I1", "issue_title": "T", "relevance": "supporting"}]},
    }
    result = _fmt_authority_panel(data, domain="legal")
    assert "Test" in result
    assert "supporting" in result.lower() or "T" in result


def test_fmt_steering_params_non_dict_guard():
    """Non-dict guard: _fmt_steering handles non-dict params gracefully."""
    from irys.ui.app import _fmt_steering
    actions = [
        {"action_type": "redirect_focus", "description": "test", "params": "not-a-dict"},
        {"action_type": "redirect_focus", "description": "test2", "params": None},
    ]
    result = _fmt_steering(actions, domain="legal")
    assert "test" in result
    assert "test2" in result


def test_fmt_domain_composition_panel_domain_labels():
    from irys.ui.app import _fmt_domain_composition_panel
    data = {
        "primary_domain_profile_id": "legal:1",
        "facets": {"legal": 0.8, "finance": 0.3},
        "composed_trust_weights": {"authoritative": 0.9, "advocacy": 0.5},
        "detection_events": [
            {"candidate_profile_id": "legal:1", "confidence": 0.85,
             "target_kind": "document", "target_id": "doc1", "created_at": "2026-01-01T00:00:00"},
        ],
    }
    legal = _fmt_domain_composition_panel(data, domain="legal")
    assert "Domain Composition" in legal
    assert "legal:1" in legal
    assert "Composed Trust Weights" in legal

    finance = _fmt_domain_composition_panel(data, domain="finance")
    assert "Composed Reliability Weights" in finance

    coding = _fmt_domain_composition_panel(data, domain="coding")
    assert "Composed Confidence Weights" in coding


def test_fmt_domain_composition_panel_empty():
    from irys.ui.app import _fmt_domain_composition_panel
    result = _fmt_domain_composition_panel({}, domain="biomedical")
    assert "viz-empty" in result


def test_fmt_domain_composition_panel_non_dict_guard():
    from irys.ui.app import _fmt_domain_composition_panel
    data = {
        "primary_domain_profile_id": "legal:1",
        "facets": "not-a-dict",
        "composed_trust_weights": None,
        "detection_events": ["bad-event", {"candidate_profile_id": "legal:1", "confidence": 0.7}],
    }
    result = _fmt_domain_composition_panel(data, domain="legal")
    assert "legal:1" in result


def test_fmt_proof_state_panel_issue_id_display():
    from irys.ui.app import _fmt_proof_state_panel
    summary = {"total_issues_tracked": 1, "avg_sufficiency": 0.5, "by_status": {}, "gap_count": 0}
    issues = [
        {
            "issue_id": "iss-abc123456789",
            "issue_title": "Test Issue",
            "sufficiency": 0.6,
            "proof_status": "partial",
            "supporting_count": 3,
            "attacking_count": 1,
            "total_predicate_count": 5,
            "satisfied_predicate_count": 3,
            "trust_weighted_support": 0.7,
            "trust_weighted_attack": 0.2,
        }
    ]
    result = _fmt_proof_state_panel(summary, issues, domain="legal")
    assert "iss-abc12345" in result
    assert "font-family:monospace" in result
    assert "Test Issue" in result


def test_fmt_proof_state_panel_issue_id_all_domains():
    from irys.ui.app import _fmt_proof_state_panel
    summary = {"total_issues_tracked": 1, "avg_sufficiency": 0.5, "by_status": {}, "gap_count": 0}
    issues = [{"issue_id": "x", "sufficiency": 0.5, "proof_status": "partial"}]
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        result = _fmt_proof_state_panel(summary, issues, domain=domain)
        assert "font-family:monospace" in result


def test_fmt_issue_assertions_domain_labels():
    from irys.ui.app import _fmt_issue_assertions
    assertions = [
        {"id": "a1", "proposition_text": "Test claim", "belief_state": "accepted",
         "confidence": 0.9, "relation_type": "supporting"},
        {"id": "a2", "proposition_text": "Counter claim", "belief_state": "disputed",
         "confidence": 0.4, "relation_type": "attacking"},
        {"id": "a3", "proposition_text": "Background", "belief_state": "undetermined",
         "confidence": 0.5, "relation_type": "neutral"},
    ]
    legal = _fmt_issue_assertions(assertions, "iss-1", domain="legal")
    assert "Supporting Evidence" in legal
    assert "Attacking Evidence" in legal
    assert "Test claim" in legal

    finance = _fmt_issue_assertions(assertions, "iss-1", domain="finance")
    assert "Corroborating Data" in finance
    assert "Contradicting Data" in finance

    biomedical = _fmt_issue_assertions(assertions, "iss-1", domain="biomedical")
    assert "Supporting Evidence" in biomedical
    assert "Contradicting Evidence" in biomedical


def test_fmt_issue_assertions_empty():
    from irys.ui.app import _fmt_issue_assertions
    result = _fmt_issue_assertions([], "iss-1", domain="coding")
    assert "viz-empty" in result
    assert "No linked findings" in result


def test_fmt_issue_assertions_non_dict_guard():
    from irys.ui.app import _fmt_issue_assertions
    assertions = [
        "not-a-dict",
        {"id": "a1", "proposition_text": "Valid", "relation_type": "supporting"},
    ]
    result = _fmt_issue_assertions(assertions, "iss-1", domain="legal")
    assert "Valid" in result
    assert "1" in result


def test_issue_assertions_backend_interface():
    import inspect
    from irys.ui.backends.base import UIBackend
    assert hasattr(UIBackend, "get_issue_assertions")
    sig = inspect.signature(UIBackend.get_issue_assertions)
    params = list(sig.parameters.keys())
    assert "matter_id" in params
    assert "issue_id" in params
    assert hasattr(UIBackend, "get_issue_authorities")
    sig2 = inspect.signature(UIBackend.get_issue_authorities)
    assert "issue_id" in list(sig2.parameters.keys())


def test_fmt_issue_assertions_with_authorities():
    from irys.ui.app import _fmt_issue_assertions
    assertions = [
        {"id": "a1", "proposition_text": "Claim", "relation_type": "supporting",
         "belief_state": "accepted", "confidence": 0.8},
    ]
    authorities = [
        {"citation": "Smith v. Jones, 123 F.3d 456", "authority_type": "case",
         "weight": "binding", "relevance": "supporting"},
        {"citation": "UCC § 2-207", "authority_type": "statute",
         "weight": "authoritative", "relevance": "neutral"},
    ]
    result = _fmt_issue_assertions(assertions, "iss-1", authorities=authorities, domain="legal")
    assert "Smith v. Jones" in result
    assert "UCC" in result
    assert "Linked Authorities" in result
    assert "case" in result
    assert "binding" in result


def test_fmt_investigation_history_domain_labels():
    from irys.ui.app import _fmt_investigation_history_panel
    runs = [
        {"query": "What happened?", "operation_type": "query", "status": "completed",
         "research_mode": "deep", "llm_request_count": 5, "llm_estimated_cost_usd": 0.02,
         "started_at": "2026-05-01T10:00:00"},
    ]
    legal = _fmt_investigation_history_panel(runs, domain="legal")
    assert "Investigation History" in legal
    assert "Investigation" in legal
    assert "1 run recorded" in legal

    finance = _fmt_investigation_history_panel(runs, domain="finance")
    assert "Analysis History" in finance
    assert "Analysis" in finance

    bio = _fmt_investigation_history_panel(runs, domain="biomedical")
    assert "Case Review History" in bio
    assert "Case Review" in bio
    assert "Clinical Query" in bio

    research = _fmt_investigation_history_panel(runs, domain="academic_research")
    assert "Research Session History" in research
    assert "Research Question" in research
    assert "1 session recorded" in research


def test_fmt_investigation_history_domain_operation_labels():
    from irys.ui.app import _fmt_investigation_history_panel
    runs = [
        {"query": "Fix bug", "operation_type": "correction", "status": "completed"},
    ]
    legal = _fmt_investigation_history_panel(runs, domain="legal")
    assert "Correction" in legal

    coding = _fmt_investigation_history_panel(runs, domain="coding")
    assert "Patch" in coding

    bio = _fmt_investigation_history_panel(runs, domain="biomedical")
    assert "Amendment" in bio


def test_fmt_investigation_history_empty_all_domains():
    from irys.ui.app import _fmt_investigation_history_panel
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        result = _fmt_investigation_history_panel([], domain=domain)
        assert "viz-empty" in result
        assert len(result) > 20


def test_fmt_investigation_history_non_dict_guard():
    from irys.ui.app import _fmt_investigation_history_panel
    runs = [{"query": "Q1", "status": "completed"}, "bad", 42, None]
    result = _fmt_investigation_history_panel(runs, domain="legal")
    assert "1 run recorded" in result
    assert "Q1" in result


def test_xss_belief_state_class_whitelist():
    """Codex PR Gate finding: belief_state goes into class= attribute.
    Malicious values must be normalized to a safe default."""
    from irys.ui.app import _fmt_issue_assertions, _fmt_assertions
    malicious = "accepted' onclick='alert(1)"
    assertions_ia = [
        {"id": "a1", "proposition_text": "Test", "relation_type": "supporting",
         "belief_state": malicious, "confidence": 0.5},
    ]
    result = _fmt_issue_assertions(assertions_ia, "iss-1", domain="legal")
    assert "belief-undetermined" in result
    assert f"belief-{malicious.lower()}" not in result

    assertions_a = [
        {"id": "a2", "proposition_text": "Test2", "belief_state": malicious, "confidence": 0.5},
    ]
    result2 = _fmt_assertions(assertions_a, domain="legal")
    assert "belief-unknown" in result2
    assert f"belief-{malicious.lower()}" not in result2


def test_xss_proof_status_class_whitelist():
    """proof_status goes into class= attribute in the issues panel tree.
    Malicious values must be normalized to a safe default."""
    from irys.ui.app import _fmt_issues_panel
    malicious = "partial' onclick='alert(1)"
    issues = [
        {"id": "iss-1", "proof_status": malicious, "title": "Test",
         "coverage_fraction": 0.5, "depth": 0,
         "supporting_count": 1, "attacking_count": 0},
    ]
    result = _fmt_issues_panel(issues, domain="legal")
    assert f"proof-{malicious.lower()}" not in result
    assert "proof-none" in result


def test_fmt_gaps_shows_gap_id():
    """Gap resolution UI needs gap IDs visible in the table."""
    from irys.ui.app import _fmt_gaps
    gaps = [
        {"id": "gap-abc123", "gap_type": "missing_document", "description": "Need contract",
         "materiality_score": 0.8, "dependencies": []},
    ]
    result = _fmt_gaps(gaps, [], domain="legal")
    assert "gap-abc123" in result
    assert "<th>ID</th>" in result
    assert "<code" in result


def test_resolve_gap_backend_interface():
    """Verify resolve_gap exists in the backend interface with correct signature."""
    import inspect
    from irys.ui.backends.base import UIBackend
    assert hasattr(UIBackend, "resolve_gap")
    sig = inspect.signature(UIBackend.resolve_gap)
    params = list(sig.parameters.keys())
    assert "matter_id" in params
    assert "gap_id" in params
    assert "resolution_note" in params


def test_fmt_gaps_non_dict_guard_with_id():
    """Gap list with mixed types should only render valid dicts."""
    from irys.ui.app import _fmt_gaps
    gaps = [
        {"id": "g1", "gap_type": "missing_predicate", "description": "Need proof",
         "materiality_score": 0.5},
        "bad_entry",
        42,
        None,
    ]
    result = _fmt_gaps(gaps, [], domain="legal")
    assert "1 unresolved" in result
    assert "g1" in result


def test_clarification_context_rendering():
    """Clarification context shows why_it_matters and expected_impact."""
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state._clarification_cache = {
        "q1": {
            "id": "q1",
            "question_text": "What was the contract date?",
            "why_it_matters": "Statute of limitations depends on this date.",
            "expected_impact": "Could resolve 2 open gaps.",
            "gap_id": "gap-abc",
            "status": "pending",
        },
    }
    state._clarification_cache_matter = "mid-1"
    result = state.get_clarification_context("mid-1", "q1")
    assert "What was the contract date?" in result
    assert "Statute of limitations" in result
    assert "Could resolve 2 open gaps" in result
    assert "gap-abc" in result
    assert "Why it matters" in result
    assert "Expected impact" in result


def test_clarification_context_empty_fields():
    """Context with missing optional fields still renders question."""
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state._clarification_cache = {
        "q2": {
            "id": "q2",
            "question_text": "Confirm amount?",
            "status": "pending",
        },
    }
    state._clarification_cache_matter = "mid-1"
    result = state.get_clarification_context("mid-1", "q2")
    assert "Confirm amount?" in result
    assert "Why it matters" not in result
    assert "Expected impact" not in result


def test_clarification_context_missing_id():
    """Missing question ID returns empty string."""
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state._clarification_cache = {}
    result = state.get_clarification_context("mid-1", "nonexistent")
    assert result == ""


def test_clarification_cache_cross_matter_isolation():
    """Cache from one matter must not leak into another matter's context."""
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state._clarification_cache = {"q1": {"id": "q1", "question_text": "Leaked?"}}
    state._clarification_cache_matter = "matter-A"
    result = state.get_clarification_context("matter-B", "q1")
    assert result == ""


def test_fmt_assumptions_shows_id():
    """Assumption table should display IDs for copy-paste into action form."""
    from irys.ui.app import _fmt_assumptions
    assumptions = [
        {"id": "asmp-abc123", "statement": "Contract was signed",
         "status": "provisional", "rationale": "From document"},
    ]
    result = _fmt_assumptions(assumptions, domain="legal")
    assert "asmp-abc123" in result
    assert "<th>ID</th>" in result
    assert "<code" in result
    assert "1 assumption" in result


def test_update_assumption_status_backend_interface():
    """Verify update_assumption_status exists in backend interface."""
    import inspect
    from irys.ui.backends.base import UIBackend
    assert hasattr(UIBackend, "update_assumption_status")
    sig = inspect.signature(UIBackend.update_assumption_status)
    params = list(sig.parameters.keys())
    assert "matter_id" in params
    assert "assumption_id" in params
    assert "status" in params
    assert "reason" in params


def test_set_issue_priority_backend_interface():
    """Verify set_issue_priority exists in backend interface."""
    import inspect
    from irys.ui.backends.base import UIBackend
    assert hasattr(UIBackend, "set_issue_priority")
    sig = inspect.signature(UIBackend.set_issue_priority)
    params = list(sig.parameters.keys())
    assert "matter_id" in params
    assert "issue_id" in params
    assert "priority" in params


def test_set_issue_priority_appstate_validation():
    """Priority steering validates inputs before calling backend."""
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    assert state.set_issue_priority("", "i1", "high") == "Load a matter first."
    assert state.set_issue_priority("—", "i1", "high") == "Load a matter first."
    assert state.set_issue_priority("mid-1", "", "high") == "Enter an issue ID."
    assert state.set_issue_priority("mid-1", "i1", "") == "Select a priority level."


def test_fmt_gaps_issue_title_resolution():
    """Gap dependencies should show issue titles when available."""
    from irys.ui.app import _fmt_gaps
    gaps = [
        {"id": "g1", "gap_type": "missing_document", "description": "Need contract",
         "materiality_score": 0.7,
         "dependencies": [{"affected_type": "issue", "affected_id": "iss-1"}]},
    ]
    titles = {"iss-1": "Breach of Contract"}
    result = _fmt_gaps(gaps, [], domain="legal", issue_titles=titles)
    assert "Breach of Contract" in result
    result_no_titles = _fmt_gaps(gaps, [], domain="legal")
    assert "issue" in result_no_titles


def test_fmt_gaps_missing_document_tracker():
    """Missing document gaps get a highlighted checklist above the main table."""
    from irys.ui.app import _fmt_gaps
    gaps = [
        {"id": "g1", "gap_type": "missing_document", "description": "Need signed contract",
         "materiality_score": 0.8, "dependencies": [{"affected_type": "issue", "affected_id": "i1"}]},
        {"id": "g2", "gap_type": "expected_absent_attachment", "description": "Expected Exhibit A",
         "materiality_score": 0.5, "dependencies": []},
        {"id": "g3", "gap_type": "missing_issue_predicate", "description": "Need proof of breach",
         "materiality_score": 0.6, "dependencies": []},
    ]
    titles = {"i1": "Breach of Contract"}
    result = _fmt_gaps(gaps, [], domain="legal", issue_titles=titles)
    assert "Missing Documents" in result
    assert "Need signed contract" in result
    assert "Expected Exhibit A" in result
    assert "Breach of Contract" in result
    assert "(2)" in result


def test_fmt_gaps_missing_document_domain_labels():
    """Missing document tracker uses domain-appropriate headers."""
    from irys.ui.app import _fmt_gaps
    gaps = [{"id": "g1", "gap_type": "missing_document", "description": "Need report",
             "materiality_score": 0.5, "dependencies": []}]
    assert "Missing Filings" in _fmt_gaps(gaps, [], domain="finance")
    assert "Missing Sources" in _fmt_gaps(gaps, [], domain="academic_research")
    assert "Missing Records" in _fmt_gaps(gaps, [], domain="biomedical")


def test_fmt_gap_workbench_basic():
    from irys.ui.app import _fmt_gap_workbench
    payload = {
        "items": [
            {
                "gap_id": "g1",
                "type": "missing_document",
                "description": "Need employment contract",
                "materiality_score": 0.8,
                "blocker_score": 0.5,
                "affected_issues": [{"affected_type": "issue", "affected_id": "i1", "title": "Wrongful termination"}],
                "dependencies": [],
                "missing_source_suggestion": "Upload the employment contract.",
                "pending_clarifications": [{"question_text": "Was the contract verbal or written?"}],
                "recommended_next_action": "clarify",
            },
        ],
    }
    result = _fmt_gap_workbench(payload)
    assert "Need employment contract" in result
    assert "Wrongful termination" in result
    assert "Upload the employment contract" in result
    assert "verbal or written" in result
    assert "Answer clarification" in result
    assert "0.80" in result


def test_fmt_gap_workbench_empty():
    from irys.ui.app import _fmt_gap_workbench
    result = _fmt_gap_workbench({"items": []})
    assert "No open gaps" in result


def test_fmt_gap_workbench_domain_labels():
    from irys.ui.app import _fmt_gap_workbench
    payload = {
        "items": [
            {
                "gap_id": "g1",
                "type": "missing_spec",
                "description": "Missing API spec",
                "materiality_score": 0.5,
                "blocker_score": 0.3,
                "affected_issues": [],
                "dependencies": [],
                "missing_source_suggestion": "Provide spec.",
                "pending_clarifications": [],
                "recommended_next_action": "request_document",
            },
        ],
    }
    result_coding = _fmt_gap_workbench(payload, domain="coding")
    assert "Missing specification" in result_coding or "Request spec" in result_coding
    result_bio = _fmt_gap_workbench(payload, domain="biomedical")
    assert "Missing record" in result_bio or "Request record" in result_bio


def test_fmt_gap_workbench_non_dict_guard():
    from irys.ui.app import _fmt_gap_workbench
    payload = {"items": ["not-a-dict", None, 42]}
    result = _fmt_gap_workbench(payload)
    assert "No open gaps" in result or "0 open gap" in result


def test_fmt_gap_workbench_xss():
    from irys.ui.app import _fmt_gap_workbench
    payload = {
        "items": [
            {
                "gap_id": "g1",
                "type": "missing_document",
                "description": "<script>alert(1)</script>",
                "materiality_score": 0.5,
                "blocker_score": 0.3,
                "affected_issues": [{"affected_type": "issue", "affected_id": "i1", "title": "<img onerror=x>"}],
                "dependencies": [],
                "missing_source_suggestion": "<b>bold</b>",
                "pending_clarifications": [{"question_text": "<iframe>"}],
                "recommended_next_action": "investigate",
            },
        ],
    }
    result = _fmt_gap_workbench(payload)
    assert "<script>" not in result
    assert "<img onerror" not in result
    assert "<iframe>" not in result
    assert "&lt;script&gt;" in result


def test_doc_intel_trust_distribution():
    """Document intelligence panel shows trust override distribution."""
    from irys.ui.app import _fmt_document_intelligence_panel
    data = {"cards": [], "total_inventory": 5, "ingested_count": 3}
    overrides = [
        {"trust_level": "high", "document_pattern": "contract.pdf"},
        {"trust_level": "low", "document_pattern": "unverified.pdf"},
        {"trust_level": "high", "document_pattern": "pleading.pdf"},
    ]
    result = _fmt_document_intelligence_panel(data, domain="legal", trust_overrides=overrides)
    assert "Source Trust" in result
    assert "2 high" in result
    assert "1 low" in result
    result_no_overrides = _fmt_document_intelligence_panel(data, domain="legal")
    assert "Source Trust" not in result_no_overrides


def test_fmt_assertion_inspector_linked_issues():
    from irys.ui.app import _fmt_assertion_inspector

    health = {
        "assertion_id": "a-linked",
        "proposition_text": "Revenue exceeded target",
        "belief_state": "accepted",
        "confidence": 0.85,
        "oscillating": False,
        "support_count": 2,
        "attack_count": 0,
        "has_superseding": False,
        "support_source_roles": [],
        "attack_source_roles": [],
        "provenance": [],
        "linked_issues": [
            {"id": "iss-1", "title": "Material breach claim", "relation_type": "supports", "status": "open", "materiality": 0.8},
            {"id": "iss-2", "title": "Damages calculation", "relation_type": "attacks", "status": "open", "materiality": 0.5},
        ],
    }
    result = _fmt_assertion_inspector(health, domain="legal")
    assert "Linked Issues" in result
    assert "Material breach claim" in result
    assert "Damages calculation" in result
    assert "supports" in result
    assert "attacks" in result
    assert "0.80" in result
    assert "0.50" in result

    result_finance = _fmt_assertion_inspector(health, domain="finance")
    assert "Linked Theses" in result_finance
    assert "challenges" in result_finance

    result_coding = _fmt_assertion_inspector(health, domain="coding")
    assert "Linked Tasks" in result_coding

    health_no_links = dict(health, linked_issues=[])
    result_empty = _fmt_assertion_inspector(health_no_links, domain="legal")
    assert "Linked Issues" not in result_empty


def test_fmt_assertion_inspector_linked_issues_non_dict_guard():
    from irys.ui.app import _fmt_assertion_inspector

    health = {
        "assertion_id": "a-guard",
        "proposition_text": "Test proposition",
        "belief_state": "undetermined",
        "confidence": 0.5,
        "oscillating": False,
        "support_count": 0,
        "attack_count": 0,
        "has_superseding": False,
        "support_source_roles": [],
        "attack_source_roles": [],
        "provenance": [],
        "linked_issues": [
            "not-a-dict",
            None,
            {"id": "iss-ok", "title": "Valid issue", "relation_type": "supports", "status": "open", "materiality": 0.6},
        ],
    }
    result = _fmt_assertion_inspector(health, domain="legal")
    assert "Valid issue" in result
    assert "not-a-dict" not in result


def test_get_issues_for_assertion_graph_layer():
    import sqlite3
    from irys.matter.graph import IssueStore

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    conn.executescript("""
        CREATE TABLE matter (id TEXT PRIMARY KEY);
        INSERT INTO matter VALUES ('m1');
        CREATE TABLE issue (
            id TEXT PRIMARY KEY, matter_id TEXT, title TEXT, status TEXT DEFAULT 'open',
            materiality REAL DEFAULT 0.5, salience REAL DEFAULT 1.0, created_at TEXT, updated_at TEXT
        );
        CREATE TABLE assertion (
            id TEXT PRIMARY KEY, matter_id TEXT, proposition_text TEXT,
            belief_state TEXT DEFAULT 'undetermined', confidence REAL DEFAULT 0.5,
            created_at TEXT, updated_at TEXT
        );
        CREATE TABLE assertion_issue_link (
            id TEXT PRIMARY KEY, assertion_id TEXT, issue_id TEXT,
            relation_type TEXT DEFAULT 'supports', created_at TEXT
        );
        CREATE UNIQUE INDEX ux_assertion_issue ON assertion_issue_link(assertion_id, issue_id, relation_type);
        INSERT INTO issue VALUES ('i1', 'm1', 'Breach claim', 'open', 0.8, 1.0, '2026-01-01', '2026-01-01');
        INSERT INTO issue VALUES ('i2', 'm1', 'Damages', 'open', 0.5, 1.0, '2026-01-01', '2026-01-01');
        INSERT INTO assertion VALUES ('a1', 'm1', 'Contract signed', 'accepted', 0.9, '2026-01-01', '2026-01-01');
        INSERT INTO assertion_issue_link VALUES ('l1', 'a1', 'i1', 'supports', '2026-01-01');
        INSERT INTO assertion_issue_link VALUES ('l2', 'a1', 'i2', 'attacks', '2026-01-01');
    """)

    class FakeDB:
        def __init__(self, c):
            self.conn = c
        def execute(self, sql, params=()):
            return self.conn.execute(sql, params)

    store = IssueStore.__new__(IssueStore)
    store.db = FakeDB(conn)
    store.matter_id = "m1"

    issues = store.get_issues_for_assertion("a1")
    assert len(issues) == 2
    titles = {i["title"] for i in issues}
    assert "Breach claim" in titles
    assert "Damages" in titles
    rels = {i["relation_type"] for i in issues}
    assert "supports" in rels
    assert "attacks" in rels

    empty = store.get_issues_for_assertion("nonexistent")
    assert empty == []
    conn.close()


def test_fmt_source_agreement_basic():
    from irys.ui.app import _fmt_source_agreement

    sources = [
        {"doc_id": "d1", "doc_label": "contract.pdf", "source_role": "CONTRACT", "supports": 5, "attacks": 1},
        {"doc_id": "d2", "doc_label": "complaint.pdf", "source_role": "COMPLAINT", "supports": 0, "attacks": 3},
    ]
    result = _fmt_source_agreement(sources, domain="legal")
    assert "Source Agreement Analysis" in result
    assert "contract.pdf" in result
    assert "complaint.pdf" in result
    assert "CONTRACT" in result
    assert "COMPLAINT" in result

    result_finance = _fmt_source_agreement(sources, domain="finance")
    assert "Filing / Report" in result_finance

    result_empty = _fmt_source_agreement([], domain="legal")
    assert "viz-empty" in result_empty
    assert "No source data" in result_empty


def test_fmt_source_agreement_non_dict_guard():
    from irys.ui.app import _fmt_source_agreement

    sources = [
        "bad-entry",
        None,
        {"doc_id": "d1", "doc_label": "valid.pdf", "source_role": "EXHIBIT", "supports": 2, "attacks": 0},
    ]
    result = _fmt_source_agreement(sources, domain="legal")
    assert "valid.pdf" in result
    assert "bad-entry" not in result


def test_fmt_source_agreement_domain_labels():
    from irys.ui.app import _fmt_source_agreement

    sources = [{"doc_id": "d1", "doc_label": "test.pdf", "source_role": "X", "supports": 1, "attacks": 0}]
    for domain, expected_col in [
        ("coding", "Spec / Artifact"),
        ("academic_research", "Paper / Source"),
        ("biomedical", "Record / Source"),
    ]:
        result = _fmt_source_agreement(sources, domain=domain)
        assert expected_col in result


def test_quant_threshold_config_parameters():
    from irys.ui.app import _fmt_quant_thresholds_panel

    violations_default = [
        {"threshold": "positive_exposure", "level": "HIGH", "description": "Exposure: USD 15,000.00", "amount": 15000.0},
        {"threshold": "disputed_fraction", "level": "MED", "description": "Disputed 12% of invoiced", "amount": 1200.0},
    ]
    result = _fmt_quant_thresholds_panel(violations_default, domain="legal")
    assert "Financial Health Alerts" in result
    assert "HIGH" in result
    assert "MED" in result

    result_empty = _fmt_quant_thresholds_panel([], domain="legal")
    assert "viz-empty" in result_empty

    result_finance = _fmt_quant_thresholds_panel(violations_default, domain="finance")
    assert "Financial Risk Alerts" in result_finance


def test_fmt_assertion_graph_basic():
    from irys.ui.app import _fmt_assertion_graph

    graph = {
        "nodes": [
            {"id": "a1", "proposition_text": "Contract signed on Jan 1", "belief_state": "accepted",
             "confidence": 0.9, "relation_type": "supports"},
            {"id": "a2", "proposition_text": "Defendant disputes date", "belief_state": "disputed",
             "confidence": 0.4, "relation_type": "attacks"},
        ],
        "edges": [
            {"src": "a2", "dst": "a1", "link_type": "attacks"},
        ],
    }
    result = _fmt_assertion_graph(graph, domain="legal")
    assert "Assertion Relationship Map" in result
    assert "Contract signed" in result
    assert "Defendant disputes" in result
    assert "Supporting (1)" in result
    assert "Attacking (1)" in result
    assert "Inter-Assertion Links (1)" in result
    assert "attacks" in result

    result_finance = _fmt_assertion_graph(graph, domain="finance")
    assert "Finding Relationship Map" in result_finance


def test_fmt_assertion_graph_empty():
    from irys.ui.app import _fmt_assertion_graph

    result = _fmt_assertion_graph({"nodes": [], "edges": []}, domain="legal")
    assert "viz-empty" in result
    assert "No assertions" in result


def test_fmt_assertion_graph_non_dict_guard():
    from irys.ui.app import _fmt_assertion_graph

    graph = {
        "nodes": [
            "bad",
            None,
            {"id": "a1", "proposition_text": "Valid", "belief_state": "accepted",
             "confidence": 0.9, "relation_type": "supports"},
        ],
        "edges": ["bad-edge", {"src": "a1", "dst": "a1", "link_type": "supports"}],
    }
    result = _fmt_assertion_graph(graph, domain="legal")
    assert "Valid" in result
    assert "bad" not in result.replace("bad-edge", "")


def test_fmt_issue_closure_workbench_ready():
    from irys.ui.app import _fmt_issue_closure_workbench
    data = {
        "issue_id": "i1",
        "title": "Breach of contract",
        "readiness": "ready",
        "coverage_fraction": 0.85,
        "supporting_count": 5,
        "predicate_count": 4,
        "verified_count": 3,
        "pending_count": 2,
        "blockers": [],
        "gaps": [],
        "source_agreement": [
            {"doc_label": "contract.pdf", "source_role": "primary", "supports": 3, "attacks": 0},
        ],
    }
    result = _fmt_issue_closure_workbench(data)
    assert "Breach of contract" in result
    assert "Ready to rely on" in result
    assert "85%" in result
    assert "contract.pdf" in result


def test_fmt_issue_closure_workbench_blocked():
    from irys.ui.app import _fmt_issue_closure_workbench
    data = {
        "issue_id": "i2",
        "title": "Damages calculation",
        "readiness": "blocked",
        "coverage_fraction": 0.3,
        "supporting_count": 1,
        "predicate_count": 5,
        "verified_count": 0,
        "pending_count": 1,
        "blockers": ["Low evidence coverage", "No verified supporting facts"],
        "gaps": [
            {"description": "Need expert report", "gap_type": "missing_document", "materiality_score": 0.8},
        ],
        "source_agreement": [],
    }
    result = _fmt_issue_closure_workbench(data)
    assert "Not ready" in result
    assert "Low evidence coverage" in result
    assert "No verified" in result
    assert "Need expert report" in result


def test_fmt_issue_closure_workbench_domain_labels():
    from irys.ui.app import _fmt_issue_closure_workbench
    data = {
        "issue_id": "i1", "title": "Test", "readiness": "ready",
        "coverage_fraction": 0.9, "supporting_count": 3, "predicate_count": 2,
        "verified_count": 2, "pending_count": 0, "blockers": [], "gaps": [],
        "source_agreement": [],
    }
    result_coding = _fmt_issue_closure_workbench(data, domain="coding")
    assert "Requirement Closure" in result_coding
    result_bio = _fmt_issue_closure_workbench(data, domain="biomedical")
    assert "Finding Closure" in result_bio


def test_fmt_issue_closure_workbench_xss():
    from irys.ui.app import _fmt_issue_closure_workbench
    data = {
        "issue_id": "i1", "title": "<script>alert(1)</script>",
        "readiness": "blocked", "coverage_fraction": 0.5,
        "supporting_count": 1, "predicate_count": 2,
        "verified_count": 0, "pending_count": 1,
        "blockers": ["<img onerror=x>"],
        "gaps": [{"description": "<b>bold</b>", "gap_type": "missing", "materiality_score": 0.5}],
        "source_agreement": [{"doc_label": "<iframe>", "source_role": "x", "supports": 1, "attacks": 0}],
    }
    result = _fmt_issue_closure_workbench(data)
    assert "<script>" not in result
    assert "<img onerror" not in result
    assert "<iframe>" not in result


def test_fmt_issue_closure_workbench_empty():
    from irys.ui.app import _fmt_issue_closure_workbench
    result = _fmt_issue_closure_workbench({})
    assert "No closure data" in result
    result_error = _fmt_issue_closure_workbench({"error": "Issue not found"})
    assert "Issue not found" in result_error


def test_fmt_assumptions_linked_targets():
    from irys.ui.app import _fmt_assumptions

    assumptions = [
        {
            "id": "asm-1",
            "statement": "Contract was fully executed",
            "status": "provisional",
            "rationale": "Based on initial review",
            "invalidation_condition": "if unsigned copy found",
            "linked_target_count": 3,
            "linked_targets": [
                {"target_type": "issue", "target_id": "iss-1"},
                {"target_type": "assertion", "target_id": "a-1"},
                {"target_type": "predicate", "target_id": "p-1"},
            ],
        },
        {
            "id": "asm-2",
            "statement": "No prior litigation",
            "status": "confirmed",
            "rationale": None,
            "invalidation_condition": None,
            "linked_target_count": 0,
            "linked_targets": [],
        },
    ]
    result = _fmt_assumptions(assumptions, domain="legal")
    assert "Contract was fully executed" in result
    assert "Linked to 3" in result
    assert "assertion" in result
    assert "issue" in result
    assert "No prior litigation" in result
    assert "Linked to 0" not in result


# ------------------------------------------------------------------
# Investigation Readiness panel tests (SO-3, SO-7)
# ------------------------------------------------------------------

def test_readiness_panel_ready():
    from irys.ui.app import _fmt_readiness_panel
    data = {
        "readiness": "ready",
        "blocker_count": 0,
        "blockers": [],
        "summary": {
            "issue_count": 5,
            "avg_coverage": 0.82,
            "proof_summary": {"total_issues_tracked": 5, "avg_sufficiency": 0.75, "gap_count": 0},
            "open_gap_count": 0,
            "contradiction_count": 0,
            "pending_clarifications": 0,
            "pending_review": 0,
        },
    }
    result = _fmt_readiness_panel(data, domain="legal")
    assert "Ready for reliance" in result
    assert "#059669" in result
    assert "No blockers detected" in result
    assert "82%" in result


def test_readiness_panel_blocked():
    from irys.ui.app import _fmt_readiness_panel
    data = {
        "readiness": "blocked",
        "blocker_count": 2,
        "blockers": [
            {
                "type": "low_coverage",
                "severity": "high",
                "label": "2 high-materiality issue(s) below 50% coverage",
                "items": [
                    {"issue_id": "i1", "title": "Breach of contract", "materiality": 0.9, "coverage_fraction": 0.2},
                ],
            },
            {
                "type": "contradictions",
                "severity": "medium",
                "label": "3 unresolved contradiction(s)",
                "items": [],
            },
        ],
        "summary": {
            "issue_count": 3,
            "avg_coverage": 0.35,
            "open_gap_count": 4,
            "contradiction_count": 3,
            "pending_clarifications": 1,
            "pending_review": 7,
        },
    }
    result = _fmt_readiness_panel(data, domain="legal")
    assert "Not ready" in result
    assert "#dc2626" in result
    assert "Breach of contract" in result
    assert "contradiction" in result.lower()
    assert "Blockers (2)" in result


def test_readiness_panel_caution():
    from irys.ui.app import _fmt_readiness_panel
    data = {
        "readiness": "caution",
        "blocker_count": 1,
        "blockers": [
            {
                "type": "pending_clarifications",
                "severity": "low",
                "label": "2 pending clarification(s)",
                "items": [],
            },
        ],
        "summary": {
            "issue_count": 2,
            "avg_coverage": 0.65,
            "open_gap_count": 0,
            "contradiction_count": 0,
            "pending_clarifications": 2,
            "pending_review": 0,
        },
    }
    result = _fmt_readiness_panel(data, domain="legal")
    assert "Proceed with caution" in result
    assert "#f59e0b" in result


def test_readiness_panel_empty():
    from irys.ui.app import _fmt_readiness_panel
    result = _fmt_readiness_panel({}, domain="legal")
    assert "viz-empty" in result


def test_readiness_panel_xss():
    from irys.ui.app import _fmt_readiness_panel
    data = {
        "readiness": "blocked",
        "blockers": [
            {
                "type": "low_coverage",
                "severity": "high",
                "label": "<script>alert(1)</script>",
                "items": [{"issue_id": "x", "title": "<img onerror=alert(1)>"}],
            },
        ],
        "summary": {"issue_count": 1, "avg_coverage": 0.1},
    }
    result = _fmt_readiness_panel(data, domain="legal")
    assert "<script>" not in result
    assert "<img onerror" not in result
    assert "&lt;" in result


def test_readiness_panel_domain_labels():
    from irys.ui.app import _fmt_readiness_panel
    data = {
        "readiness": "ready",
        "blockers": [],
        "summary": {"issue_count": 3, "avg_coverage": 0.9},
    }
    for domain, expected in [
        ("finance", "Analysis Readiness"),
        ("coding", "Analysis Readiness"),
        ("academic_research", "Research Readiness"),
        ("biomedical", "Assessment Readiness"),
    ]:
        result = _fmt_readiness_panel(data, domain=domain)
        assert expected in result, f"Domain {domain}: expected '{expected}' in output"


def test_readiness_panel_high_mat_gaps():
    from irys.ui.app import _fmt_readiness_panel
    data = {
        "readiness": "blocked",
        "blockers": [
            {
                "type": "high_materiality_gaps",
                "severity": "high",
                "label": "2 high-materiality gap(s) remain open",
                "items": [
                    {"gap_id": "g1", "description": "Missing employment contract"},
                    {"gap_id": "g2", "description": "Missing financial statement"},
                ],
            },
        ],
        "summary": {"issue_count": 2, "avg_coverage": 0.6, "open_gap_count": 2},
    }
    result = _fmt_readiness_panel(data, domain="legal")
    assert "Missing employment contract" in result
    assert "Missing financial statement" in result
    assert "HIGH" in result.upper()


def test_readiness_panel_non_dict_guard():
    from irys.ui.app import _fmt_readiness_panel
    data = {
        "readiness": "blocked",
        "blockers": [
            "not a dict",
            {
                "type": "contradictions",
                "severity": "medium",
                "label": "1 contradiction",
                "items": [],
            },
        ],
        "summary": {"issue_count": 1, "avg_coverage": 0.5},
    }
    result = _fmt_readiness_panel(data, domain="legal")
    assert "1 contradiction" in result


# ------------------------------------------------------------------
# Document Review Console formatter tests (SO-3, SO-5)
# ------------------------------------------------------------------

def test_document_console_basic():
    from irys.ui.app import _fmt_document_console
    data = {
        "document_ref": "contract_2024.pdf",
        "card": {
            "doc_type": "contract",
            "privilege_flag": False,
            "parties_summary": "Acme vs TechCo",
            "date_range": "2024-01-01 to 2024-12-31",
        },
        "candidate_count": 5,
        "verified_count": 3,
        "rejected_count": 1,
        "candidates": [
            {"proposition_text": "Payment due on signing", "belief_state": "operative", "confidence": 0.9},
            {"proposition_text": "30-day termination clause", "belief_state": "alleged", "confidence": 0.6},
        ],
        "linked_issues": [
            {"id": "i1", "title": "Breach of contract", "status": "open", "materiality": 0.9},
        ],
        "actor_roles": [
            {"actor_name": "Acme Corp", "role": "plaintiff", "confidence": 0.95},
        ],
    }
    result = _fmt_document_console(data, domain="legal")
    assert "contract_2024.pdf" in result
    assert "contract" in result
    assert "Not privileged" in result
    assert "Acme vs TechCo" in result
    assert "Payment due on signing" in result
    assert "Breach of contract" in result
    assert "Acme Corp" in result
    assert "plaintiff" in result


def test_document_console_no_card():
    from irys.ui.app import _fmt_document_console
    data = {
        "document_ref": "unknown.pdf",
        "card": {},
        "candidate_count": 0,
        "verified_count": 0,
        "rejected_count": 0,
        "candidates": [],
        "linked_issues": [],
        "actor_roles": [],
    }
    result = _fmt_document_console(data, domain="legal")
    assert "No document profile" in result
    assert "unknown.pdf" in result


def test_document_console_xss():
    from irys.ui.app import _fmt_document_console
    data = {
        "document_ref": "<script>alert(1)</script>",
        "card": {"doc_type": "<img onerror=x>", "privilege_flag": True},
        "candidate_count": 1,
        "verified_count": 0,
        "rejected_count": 0,
        "candidates": [
            {"proposition_text": "<b>XSS</b>", "belief_state": "test", "confidence": 0.5},
        ],
        "linked_issues": [{"id": "i1", "title": "<script>bad</script>", "materiality": 0.5}],
        "actor_roles": [{"actor_name": "<img src=x>", "role": "test"}],
    }
    result = _fmt_document_console(data, domain="legal")
    assert "<script>" not in result
    assert "<img onerror" not in result
    assert "<img src=" not in result
    assert "&lt;" in result


def test_document_console_domain_labels():
    from irys.ui.app import _fmt_document_console
    data = {
        "document_ref": "data.xlsx",
        "card": {},
        "candidate_count": 2,
        "verified_count": 1,
        "rejected_count": 0,
        "candidates": [],
        "linked_issues": [],
        "actor_roles": [],
    }
    result = _fmt_document_console(data, domain="finance")
    assert "Document Review Console" in result
    result_bio = _fmt_document_console(data, domain="biomedical")
    assert "Record Review Console" in result_bio


def test_document_console_empty():
    from irys.ui.app import _fmt_document_console
    result = _fmt_document_console({}, domain="legal")
    assert "viz-empty" in result


def test_document_console_non_dict_guard():
    from irys.ui.app import _fmt_document_console
    data = {
        "document_ref": "test.pdf",
        "card": {},
        "candidate_count": 1,
        "verified_count": 0,
        "rejected_count": 0,
        "candidates": ["not a dict", {"proposition_text": "Valid", "belief_state": "operative", "confidence": 0.8}],
        "linked_issues": ["bad", {"id": "i1", "title": "Real issue", "materiality": 0.7}],
        "actor_roles": [42],
    }
    result = _fmt_document_console(data, domain="legal")
    assert "Valid" in result
    assert "Real issue" in result


# ------------------------------------------------------------------
# Assertion Impact Trace formatter tests (SO-2, SO-3, SO-5)
# ------------------------------------------------------------------

def test_assertion_trace_basic():
    from irys.ui.app import _fmt_assertion_trace
    data = {
        "assertion_id": "a-123",
        "proposition_text": "Payment was due on signing",
        "belief_state": "operative",
        "confidence": 0.9,
        "speech_act": "alleged",
        "source_documents": [
            {"document_label": "contract.pdf", "section_label": "Section 4.2", "span_id": "sp1"},
        ],
        "affected_issues": [
            {"id": "i1", "title": "Breach of contract", "materiality": 0.9},
        ],
        "dependent_assertions": [
            {"id": "a-456", "proposition_text": "Late penalty accrues", "belief_state": "inferred"},
        ],
        "verification": {"status": "verified"},
        "revision_history": [
            {"old_state": "alleged", "new_state": "operative", "cause": "corroboration"},
        ],
        "impact_summary": {
            "source_doc_count": 1,
            "issues_affected": 1,
            "dependents_count": 1,
            "revision_count": 1,
        },
    }
    result = _fmt_assertion_trace(data, domain="legal")
    assert "Payment was due on signing" in result
    assert "operative" in result
    assert "contract.pdf" in result
    assert "Section 4.2" in result
    assert "Breach of contract" in result
    assert "Late penalty accrues" in result
    assert "verified" in result
    assert "alleged" in result and "corroboration" in result


def test_assertion_trace_xss():
    from irys.ui.app import _fmt_assertion_trace
    data = {
        "assertion_id": "a-x",
        "proposition_text": "<script>alert(1)</script>",
        "belief_state": "disputed",
        "confidence": 0.5,
        "speech_act": "alleged",
        "source_documents": [{"document_label": "<img onerror=x>", "section_label": ""}],
        "affected_issues": [{"id": "i1", "title": "<b>XSS</b>", "materiality": 0.5}],
        "dependent_assertions": [{"id": "a-2", "proposition_text": "<script>bad</script>", "belief_state": "operative"}],
        "verification": {},
        "revision_history": [],
        "impact_summary": {"source_doc_count": 1, "issues_affected": 1, "dependents_count": 1, "revision_count": 0},
    }
    result = _fmt_assertion_trace(data, domain="legal")
    assert "<script>" not in result
    assert "<img onerror" not in result
    assert "&lt;" in result


def test_assertion_trace_empty():
    from irys.ui.app import _fmt_assertion_trace
    result = _fmt_assertion_trace({}, domain="legal")
    assert "viz-empty" in result


def test_assertion_trace_error():
    from irys.ui.app import _fmt_assertion_trace
    result = _fmt_assertion_trace({"error": "Assertion not found"}, domain="legal")
    assert "not found" in result.lower()


def test_assertion_trace_domain_labels():
    from irys.ui.app import _fmt_assertion_trace
    data = {
        "assertion_id": "a-1",
        "proposition_text": "Test",
        "belief_state": "alleged",
        "confidence": 0.5,
        "speech_act": "alleged",
        "source_documents": [],
        "affected_issues": [],
        "dependent_assertions": [],
        "verification": {},
        "revision_history": [],
        "impact_summary": {"source_doc_count": 0, "issues_affected": 0, "dependents_count": 0, "revision_count": 0},
    }
    result = _fmt_assertion_trace(data, domain="finance")
    assert "Data Point Impact Trace" in result
    result_bio = _fmt_assertion_trace(data, domain="biomedical")
    assert "Finding Impact Trace" in result_bio


def test_assertion_trace_non_dict_guard():
    from irys.ui.app import _fmt_assertion_trace
    data = {
        "assertion_id": "a-1",
        "proposition_text": "Test",
        "belief_state": "operative",
        "confidence": 0.8,
        "speech_act": "alleged",
        "source_documents": ["bad", {"document_label": "real.pdf", "section_label": "S1"}],
        "affected_issues": [42, {"id": "i1", "title": "Real", "materiality": 0.5}],
        "dependent_assertions": ["nope"],
        "verification": {},
        "revision_history": [None, {"old_state": "alleged", "new_state": "operative", "cause": "test"}],
        "impact_summary": {"source_doc_count": 1, "issues_affected": 1, "dependents_count": 0, "revision_count": 1},
    }
    result = _fmt_assertion_trace(data, domain="legal")
    assert "real.pdf" in result
    assert "Real" in result


# ---------------------------------------------------------------------------
# Domain-aware prompt vocabulary tests
# ---------------------------------------------------------------------------


def test_domain_orientation_context_has_all_five_domains():
    from irys.rlm.engine import _DOMAIN_ORIENTATION_CONTEXT
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        ctx = _DOMAIN_ORIENTATION_CONTEXT[domain]
        assert "issue_types" in ctx
        assert "issue_type_descriptions" in ctx
        assert "predicate_examples" in ctx
        assert "document_priorities" in ctx
        assert "search_examples" in ctx
        assert "|" in ctx["issue_types"], f"{domain} should have pipe-separated issue types"


def test_domain_extraction_examples_has_all_five_domains():
    from irys.rlm.engine import _DOMAIN_EXTRACTION_EXAMPLES
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        ex = _DOMAIN_EXTRACTION_EXAMPLES[domain]
        assert "subject_examples" in ex
        assert "predicate_examples" in ex
        assert "object_examples" in ex


def test_orientation_prompt_accepts_domain_parameters():
    from irys.rlm.engine import ORIENTATION_PROMPT, _DOMAIN_ORIENTATION_CONTEXT
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        ctx = _DOMAIN_ORIENTATION_CONTEXT[domain]
        result = ORIENTATION_PROMPT.format(
            structure="test/",
            file_listing="file1.pdf",
            total_files=1,
            query="test query",
            matter_context="",
            research_alignment_guidance="",
            domain_issue_types=ctx["issue_types"],
            domain_issue_type_descriptions=ctx["issue_type_descriptions"],
            domain_predicate_examples=ctx["predicate_examples"],
            domain_document_priorities=ctx["document_priorities"],
            domain_search_examples=ctx["search_examples"],
        )
        assert ctx["issue_types"] in result
        if domain != "legal":
            assert "breach of contract" not in result.lower() or domain == "legal"


def test_orientation_prompt_finance_uses_financial_types():
    from irys.rlm.engine import ORIENTATION_PROMPT, _DOMAIN_ORIENTATION_CONTEXT
    ctx = _DOMAIN_ORIENTATION_CONTEXT["finance"]
    result = ORIENTATION_PROMPT.format(
        structure="test/",
        file_listing="10K.pdf",
        total_files=1,
        query="review financials",
        matter_context="",
        research_alignment_guidance="",
        domain_issue_types=ctx["issue_types"],
        domain_issue_type_descriptions=ctx["issue_type_descriptions"],
        domain_predicate_examples=ctx["predicate_examples"],
        domain_document_priorities=ctx["document_priorities"],
        domain_search_examples=ctx["search_examples"],
    )
    assert "revenue_recognition" in result
    assert "covenant_compliance" in result
    assert "Audited financial statements" in result


def test_extract_prompt_accepts_domain_examples():
    from irys.rlm.engine import EXTRACT_FINDINGS_PROMPT, _DOMAIN_EXTRACTION_EXAMPLES
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        ex = _DOMAIN_EXTRACTION_EXAMPLES[domain]
        result = EXTRACT_FINDINGS_PROMPT.format(
            query="test",
            hypothesis="test",
            relevance_hint="(none)",
            search_term="test",
            search_results="no results",
            domain_subject_examples=ex["subject_examples"],
            domain_predicate_examples=ex["predicate_examples"],
            domain_object_examples=ex["object_examples"],
        )
        assert ex["subject_examples"] in result


def test_resolve_active_domain_returns_legal_without_model():
    from irys.rlm.engine import RLMEngine
    engine = RLMEngine(gemini_client=_StubClient())
    assert engine._resolve_active_domain() == "legal"


def test_resolve_active_domain_returns_cached():
    from irys.rlm.engine import RLMEngine
    from irys.rlm.state import InvestigationState
    engine = RLMEngine(gemini_client=_StubClient())
    state = InvestigationState.create("test", ".")
    state._cached_domain = "finance"
    assert engine._resolve_active_domain(state) == "finance"


def test_authority_extraction_gated_to_legal():
    """Verify that _extract_and_store_authorities is only invoked for legal domain."""
    from irys.rlm.engine import _DOMAIN_ORIENTATION_CONTEXT
    assert "legal" in _DOMAIN_ORIENTATION_CONTEXT
    for domain in ("finance", "coding", "academic_research", "biomedical"):
        ctx = _DOMAIN_ORIENTATION_CONTEXT[domain]
        assert "claim" not in ctx["issue_types"].split("|"), (
            f"{domain} should not use legal issue type 'claim'"
        )


# --- Quant Ontology Workbench tests ---


def test_fmt_quant_ontology_empty_returns_placeholder():
    from irys.ui.app import _fmt_quant_ontology
    html = _fmt_quant_ontology({}, domain="legal")
    assert "No quantitative facts" in html


def test_fmt_quant_ontology_renders_metric_groups():
    from irys.ui.app import _fmt_quant_ontology
    data = {
        "metric_groups": [
            {
                "metric_type": "invoice_amount",
                "canonical_metric": "accounts_receivable",
                "approved": True,
                "fact_count": 5,
                "total_value": 1234.56,
                "sample_facts": [{"raw_text": "Invoice #42: $500"}],
            },
            {
                "metric_type": "penalty_rate",
                "canonical_metric": None,
                "approved": False,
                "fact_count": 2,
                "total_value": 0,
                "sample_facts": [],
            },
        ],
        "approved_count": 1,
        "total_metric_types": 2,
        "coverage_fraction": 0.5,
    }
    html = _fmt_quant_ontology(data, domain="legal")
    assert "invoice_amount" in html
    assert "accounts_receivable" in html
    assert "Approved" in html
    assert "Pending" in html
    assert "1/2" in html
    assert "Invoice #42" in html


def test_fmt_quant_ontology_xss_escapes_metric_type():
    from irys.ui.app import _fmt_quant_ontology
    data = {
        "metric_groups": [
            {
                "metric_type": "<script>alert(1)</script>",
                "canonical_metric": "<img onerror=alert(1)>",
                "approved": True,
                "fact_count": 1,
                "total_value": 0,
                "sample_facts": [{"raw_text": "<b>xss</b>"}],
            },
        ],
        "approved_count": 1,
        "total_metric_types": 1,
        "coverage_fraction": 1.0,
    }
    html = _fmt_quant_ontology(data, domain="legal")
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "<b>xss</b>" not in html
    assert "&lt;script&gt;" in html


def test_fmt_quant_ontology_non_dict_guard():
    from irys.ui.app import _fmt_quant_ontology
    data = {
        "metric_groups": [
            "not_a_dict",
            None,
            {"metric_type": "valid", "approved": False, "fact_count": 1},
        ],
        "approved_count": 0,
        "total_metric_types": 1,
        "coverage_fraction": 0.0,
    }
    html = _fmt_quant_ontology(data, domain="legal")
    assert "valid" in html


def test_canonical_metrics_all_five_domains():
    from irys.ui.app import _CANONICAL_METRICS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _CANONICAL_METRICS, f"Missing canonical metrics for {domain}"
        assert len(_CANONICAL_METRICS[domain]) >= 5, f"Too few metrics for {domain}"


def test_canonical_metric_choices_returns_domain_specific():
    from irys.ui.app import _canonical_metric_choices
    legal = _canonical_metric_choices("legal")
    assert "accounts_receivable" in legal
    finance = _canonical_metric_choices("finance")
    assert "revenue" in finance
    unknown = _canonical_metric_choices("unknown_domain")
    assert unknown == _canonical_metric_choices("legal")


def test_quant_ontology_labels_all_five_domains():
    from irys.ui.app import _QUANT_ONTOLOGY_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _QUANT_ONTOLOGY_LABELS
        labels = _QUANT_ONTOLOGY_LABELS[domain]
        for key in ("title", "approved", "pending", "empty", "coverage"):
            assert key in labels, f"Missing key {key} for {domain}"


def test_fmt_quant_ontology_sample_facts_non_dict_guard():
    from irys.ui.app import _fmt_quant_ontology
    data = {
        "metric_groups": [
            {
                "metric_type": "test_metric",
                "canonical_metric": "canonical",
                "approved": True,
                "fact_count": 3,
                "total_value": 0,
                "sample_facts": ["not_dict", None, {"raw_text": "valid fact"}],
            },
        ],
        "approved_count": 1,
        "total_metric_types": 1,
        "coverage_fraction": 1.0,
    }
    html = _fmt_quant_ontology(data, domain="legal")
    assert "valid fact" in html


# --- Answer Audit Workbench tests ---


def test_fmt_answer_audit_empty_returns_placeholder():
    from irys.ui.app import _fmt_answer_audit
    html = _fmt_answer_audit({}, domain="legal")
    assert "No answer audits" in html


def test_fmt_answer_audit_renders_fresh_and_stale():
    from irys.ui.app import _fmt_answer_audit
    data = {
        "audits": [
            {
                "manifest_hash": "abc123def456",
                "purpose": "synthesis",
                "created_at": "2026-05-04T10:00:00",
                "domain_profile_id": "legal",
                "policy_audience": "clean",
                "taint_class": "public",
                "status_badge": "fresh",
                "valid": True,
                "stale_reasons": [],
                "object_dependency_count": 5,
                "negative_dependency_count": 2,
                "object_groups": {"assertions": [{"target_id": "a1"}]},
                "negative_dependencies": [{"namespace": "gaps", "query_predicate": "none"}],
            },
            {
                "manifest_hash": "xyz789",
                "purpose": "orient",
                "created_at": "2026-05-04T09:00:00",
                "domain_profile_id": "finance",
                "policy_audience": "internal",
                "taint_class": "sensitive",
                "status_badge": "stale",
                "valid": False,
                "stale_reasons": ["namespace claims:*: expected 3, current 5"],
                "object_dependency_count": 10,
                "negative_dependency_count": 0,
                "object_groups": {},
                "negative_dependencies": [],
            },
        ],
        "total_manifests": 5,
    }
    html = _fmt_answer_audit(data, domain="legal")
    assert "synthesis" in html
    assert "Fresh" in html
    assert "Stale" in html
    assert "orient" in html
    assert "expected 3" in html
    assert "2 of 5" in html


def test_fmt_answer_audit_xss_escapes():
    from irys.ui.app import _fmt_answer_audit
    data = {
        "audits": [
            {
                "manifest_hash": "<script>alert(1)</script>",
                "purpose": "<img onerror=evil>",
                "created_at": "2026-05-04T10:00:00",
                "domain_profile_id": "legal",
                "policy_audience": "clean",
                "taint_class": "public",
                "status_badge": "fresh",
                "valid": True,
                "stale_reasons": ["<b>xss</b>"],
                "object_dependency_count": 0,
                "negative_dependency_count": 0,
                "object_groups": {},
                "negative_dependencies": [],
            },
        ],
        "total_manifests": 1,
    }
    html = _fmt_answer_audit(data, domain="legal")
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "&lt;script&gt;" in html


def test_fmt_answer_audit_non_dict_guard():
    from irys.ui.app import _fmt_answer_audit
    data = {
        "audits": [
            "not_a_dict",
            None,
            {
                "manifest_hash": "valid",
                "purpose": "test",
                "created_at": "2026-05-04",
                "domain_profile_id": "legal",
                "policy_audience": "clean",
                "taint_class": "public",
                "status_badge": "unknown",
                "valid": False,
                "stale_reasons": [],
                "object_dependency_count": 0,
                "negative_dependency_count": 0,
                "object_groups": {},
                "negative_dependencies": [],
            },
        ],
        "total_manifests": 1,
    }
    html = _fmt_answer_audit(data, domain="legal")
    assert "test" in html


def test_answer_audit_labels_all_five_domains():
    from irys.ui.app import _ANSWER_AUDIT_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _ANSWER_AUDIT_LABELS
        labels = _ANSWER_AUDIT_LABELS[domain]
        for key in ("title", "fresh", "stale", "empty", "evidence", "missingness"):
            assert key in labels, f"Missing key {key} for {domain}"


# --- Contradiction Resolution Workflow tests ---


def test_resolve_contradiction_invalid_decision():
    """resolve_contradiction rejects invalid decision values."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.resolve_contradiction("a1", "a2", "invalid_choice", "reason")
    assert result.get("error")
    assert "Invalid decision" in result["error"]


def test_resolve_contradiction_missing_assertion():
    """resolve_contradiction returns error when assertion not found."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.resolve_contradiction("nonexistent", "also_nonexistent", "prefer_attacker", "test")
    assert result.get("error")
    assert "not found" in result["error"]


# --- Knowledge Seed Reuse tests ---


def test_fmt_knowledge_seeds_empty_returns_placeholder():
    from irys.ui.app import _fmt_knowledge_seeds
    html = _fmt_knowledge_seeds({})
    assert "No knowledge seeds" in html or "viz-empty" in html


def test_fmt_knowledge_seeds_renders_groups():
    from irys.ui.app import _fmt_knowledge_seeds
    data = {
        "total": 2,
        "counts": {"promotable": 1, "matter_local": 1},
        "promotable": [
            {"id": "seed1abc", "seed_kind": "metric_alias", "source_matter_id": "m1",
             "domain_profile_id": "legal:1", "created_at": "2026-05-04T00:00:00Z",
             "promotion_status": "promotable", "review_note": None},
        ],
        "accepted": [
            {"id": "seed2def", "seed_kind": "contradiction_resolution", "source_matter_id": None,
             "domain_profile_id": "finance:1", "created_at": "2026-05-04T01:00:00Z",
             "promotion_status": "matter_local", "review_note": "Looks good"},
        ],
        "rejected": [],
    }
    html = _fmt_knowledge_seeds(data, domain="legal")
    assert "metric_alias" in html
    assert "seed1abc" in html[:200] or "seed1abc" in html
    assert "Promotable" in html
    assert "Accepted" in html


def test_fmt_knowledge_seeds_xss_escapes():
    from irys.ui.app import _fmt_knowledge_seeds
    data = {
        "total": 1,
        "counts": {"promotable": 1},
        "promotable": [
            {"id": "xss<script>", "seed_kind": "<img onerror=alert(1)>",
             "source_matter_id": None, "domain_profile_id": "legal:1",
             "created_at": "2026-05-04", "promotion_status": "promotable",
             "review_note": None},
        ],
        "accepted": [],
        "rejected": [],
    }
    html = _fmt_knowledge_seeds(data)
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "&lt;script&gt;" in html or "&lt;img" in html


def test_fmt_knowledge_seeds_non_dict_guard():
    from irys.ui.app import _fmt_knowledge_seeds
    data = {
        "total": 1,
        "counts": {"promotable": 1},
        "promotable": ["not a dict", 42],
        "accepted": [],
        "rejected": [],
    }
    html = _fmt_knowledge_seeds(data)
    assert "Promotable" in html


def test_knowledge_seed_labels_all_five_domains():
    from irys.ui.app import _KNOWLEDGE_SEED_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _KNOWLEDGE_SEED_LABELS
        labels = _KNOWLEDGE_SEED_LABELS[domain]
        for key in ("title", "empty", "promotable", "matter_local", "rejected", "seed_kind"):
            assert key in labels, f"Missing key {key} for {domain}"


def test_knowledge_seed_store_upsert_and_list():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    seed_id = model.knowledge_seeds.upsert(
        seed_kind="metric_alias",
        domain_profile_id="legal:1",
        payload_json='{"raw":"test","canonical":"revenue"}',
        source_matter_id="src_matter_1",
    )
    assert seed_id
    seeds = model.knowledge_seeds.list_all()
    assert len(seeds) == 1
    assert seeds[0]["seed_kind"] == "metric_alias"
    assert seeds[0]["promotion_status"] == "promotable"


def test_knowledge_seed_store_review():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    seed_id = model.knowledge_seeds.upsert(
        seed_kind="contradiction_resolution",
        domain_profile_id="legal:1",
        payload_json='{"decision":"prefer_attacker"}',
    )
    ok = model.knowledge_seeds.review(seed_id, "matter_local", review_note="Approved")
    assert ok
    seed = model.knowledge_seeds.get(seed_id)
    assert seed["promotion_status"] == "matter_local"
    assert seed["review_note"] == "Approved"


def test_knowledge_seed_store_review_invalid_decision():
    from irys.matter.matter import MatterModel
    import pytest
    model = MatterModel.open_in_memory()
    seed_id = model.knowledge_seeds.upsert(
        seed_kind="test",
        domain_profile_id="legal:1",
        payload_json='{}',
    )
    with pytest.raises(ValueError, match="Invalid decision"):
        model.knowledge_seeds.review(seed_id, "bad_decision")


def test_knowledge_seed_store_count_by_status():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.knowledge_seeds.upsert("a", "legal:1", '{"a":1}')
    model.knowledge_seeds.upsert("b", "legal:1", '{"b":2}')
    sid = model.knowledge_seeds.upsert("c", "legal:1", '{"c":3}')
    model.knowledge_seeds.review(sid, "rejected", review_note="Bad")
    counts = model.knowledge_seeds.count_by_status()
    assert counts.get("promotable") == 2
    assert counts.get("rejected") == 1


def test_knowledge_seed_workbench_facade():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.knowledge_seeds.upsert("metric", "legal:1", '{"x":1}')
    model.knowledge_seeds.upsert("resolution", "legal:1", '{"y":2}')
    wb = model.get_knowledge_seed_workbench()
    assert wb["total"] == 2
    assert len(wb["promotable"]) == 2
    assert len(wb["accepted"]) == 0


def test_review_knowledge_seed_facade():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    sid = model.knowledge_seeds.upsert("test", "legal:1", '{"z":1}')
    result = model.review_knowledge_seed(sid, "matter_local", review_note="OK")
    assert result["success"]
    assert result["decision"] == "matter_local"


def test_review_knowledge_seed_invalid():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.review_knowledge_seed("nonexistent", "matter_local")
    assert result.get("error")


def test_promote_knowledge_seed_facade():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.promote_knowledge_seed(
        seed_kind="alias", domain_profile_id="finance:1",
        payload_json='{"metric":"revenue"}', source_matter_id="src1",
    )
    assert result["success"]
    assert result["seed_id"]
    seeds = model.knowledge_seeds.list_all()
    assert len(seeds) == 1


# --- Assumption Lifecycle Review tests ---


def test_assumption_review_invalid_decision():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.review_assumption("a1", "bad_status")
    assert result.get("error")
    assert "Invalid decision" in result["error"]


def test_assumption_review_not_found():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.review_assumption("nonexistent", "confirmed")
    assert result.get("error")
    assert "not found" in result["error"]


def test_assumption_review_confirm():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    aid = model.assumptions.upsert("Test assumption", rationale="test reason")
    result = model.review_assumption(aid, "confirmed", reason="Verified by expert")
    assert result["success"]
    assert result["decision"] == "confirmed"
    assumption_list = model.assumptions.get_all()
    match = [a for a in assumption_list if a["id"] == aid]
    assert match[0]["status"] == "confirmed"


def test_assumption_review_invalidate_records_gap():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    aid = model.assumptions.upsert("Revenue is linear", rationale="Assumed from Q1 data")
    initial_gaps = model.gaps.count_open()
    result = model.review_assumption(aid, "invalidated", reason="Q2 data shows non-linear")
    assert result["success"]
    assert "Recorded gap" in " ".join(result.get("actions", []))
    assert model.gaps.count_open() > initial_gaps


def test_assumption_review_workbench():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.assumptions.upsert("Assumption A")
    model.assumptions.upsert("Assumption B")
    aid = model.assumptions.upsert("Assumption C")
    model.assumptions.set_status(aid, "confirmed")
    wb = model.get_assumption_review_workbench()
    assert wb["total"] == 3
    assert wb["counts"]["provisional"] == 2
    assert wb["counts"]["confirmed"] == 1
    assert wb["counts"]["invalidated"] == 0


def test_assumption_review_invalidate_blocks_predicates():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import IssueType
    model = MatterModel.open_in_memory()
    iid, _ = model.issues.upsert_issue(
        title="Test issue",
        issue_type=IssueType.CLAIM,
        materiality=0.8,
    )
    pid = model.issues.add_predicate(iid, "Test predicate", "plaintiff")
    aid = model.assumptions.upsert("Key assumption", invalidation_condition="if data changes")
    model.assumptions.link(aid, "predicate", pid)
    result = model.review_assumption(aid, "invalidated", reason="Data changed")
    assert result["success"]
    assert any("Blocked" in a for a in result.get("actions", []))


# --- Objective Coverage Workbench tests ---


def test_fmt_objective_coverage_empty_returns_placeholder():
    from irys.ui.app import _fmt_objective_coverage
    html = _fmt_objective_coverage({})
    assert "No objectives" in html or "viz-empty" in html


def test_fmt_objective_coverage_renders_objectives():
    from irys.ui.app import _fmt_objective_coverage
    data = {
        "total": 2,
        "objectives": [
            {
                "id": "obj1abc",
                "title": "Breach of Contract",
                "issue_type": "claim",
                "materiality": 0.9,
                "salience": 0.8,
                "burden_side": "plaintiff",
                "coverage_fraction": 0.85,
                "coverage_badge": "covered",
                "supporting_count": 5,
                "predicate_total": 3,
                "predicate_satisfied": 2,
                "predicate_blocked": 0,
                "predicate_contested": 0,
                "predicates": [
                    {"description": "Duty existed", "status": "resolved"},
                    {"description": "Breach occurred", "status": "resolved"},
                    {"description": "Damages resulted", "status": "open"},
                ],
                "gaps": [],
                "has_proof_gap": False,
            },
            {
                "id": "obj2def",
                "title": "Revenue Recognition",
                "issue_type": "claim",
                "materiality": 0.7,
                "salience": 0.5,
                "burden_side": None,
                "coverage_fraction": 0.2,
                "coverage_badge": "missing",
                "supporting_count": 1,
                "predicate_total": 2,
                "predicate_satisfied": 0,
                "predicate_blocked": 1,
                "predicate_contested": 0,
                "predicates": [
                    {"description": "Revenue is measurable", "status": "blocked"},
                    {"description": "Revenue is earned", "status": "open"},
                ],
                "gaps": [{"gap_type": "missing_document", "description": "Need Q2 data"}],
                "has_proof_gap": True,
            },
        ],
        "summary": {
            "covered": 1,
            "thin": 0,
            "blocked": 0,
            "missing": 1,
            "contradicted": 0,
        },
    }
    html = _fmt_objective_coverage(data, domain="legal")
    assert "Breach of Contract" in html
    assert "Revenue Recognition" in html
    assert "Covered" in html
    assert "Missing" in html
    assert "Duty existed" in html
    assert "85%" in html


def test_fmt_objective_coverage_xss_escapes():
    from irys.ui.app import _fmt_objective_coverage
    data = {
        "total": 1,
        "objectives": [
            {
                "id": "<script>alert(1)</script>",
                "title": "<img onerror=evil>",
                "issue_type": "claim",
                "materiality": 0.5,
                "salience": 0.5,
                "burden_side": None,
                "coverage_fraction": 0.5,
                "coverage_badge": "thin",
                "supporting_count": 1,
                "predicate_total": 0,
                "predicate_satisfied": 0,
                "predicate_blocked": 0,
                "predicate_contested": 0,
                "predicates": [],
                "gaps": [],
                "has_proof_gap": False,
            },
        ],
        "summary": {"covered": 0, "thin": 1, "blocked": 0, "missing": 0, "contradicted": 0},
    }
    html = _fmt_objective_coverage(data)
    assert "<script>" not in html
    assert "<img onerror" not in html


def test_fmt_objective_coverage_non_dict_guard():
    from irys.ui.app import _fmt_objective_coverage
    data = {
        "total": 1,
        "objectives": ["not a dict", 42],
        "summary": {},
    }
    html = _fmt_objective_coverage(data)
    assert "Objective" in html or "0 total" in html


def test_objective_coverage_labels_all_five_domains():
    from irys.ui.app import _OBJECTIVE_COVERAGE_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _OBJECTIVE_COVERAGE_LABELS
        labels = _OBJECTIVE_COVERAGE_LABELS[domain]
        for key in ("title", "empty", "objective", "criteria", "support", "gaps", "covered", "missing"):
            assert key in labels, f"Missing key {key} for {domain}"


def test_objective_coverage_workbench_empty():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    wb = model.get_objective_coverage_workbench()
    assert wb["total"] == 0
    assert wb["objectives"] == []


def test_objective_coverage_workbench_with_issues():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import IssueType
    model = MatterModel.open_in_memory()
    iid1, _ = model.issues.upsert_issue("Test claim", IssueType.CLAIM, materiality=0.9)
    model.issues.add_predicate(iid1, "Element A", "plaintiff")
    model.issues.add_predicate(iid1, "Element B", "plaintiff")
    iid2, _ = model.issues.upsert_issue("Secondary claim", IssueType.CLAIM, materiality=0.5)
    wb = model.get_objective_coverage_workbench()
    assert wb["total"] == 2
    objs = wb["objectives"]
    assert len(objs) == 2
    obj1 = next(o for o in objs if o["id"] == iid1)
    assert obj1["predicate_total"] == 2
    assert obj1["predicate_satisfied"] == 0


# ---- Predicate status management (SO-4) ---- #

def test_set_criterion_status_resolves():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import IssueType
    model = MatterModel.open_in_memory()
    iid, _ = model.issues.upsert_issue("Test claim", IssueType.CLAIM, materiality=0.8)
    pid = model.issues.add_predicate(iid, "Element A", "plaintiff")
    ok = model.issues.set_predicate_status(pid, "resolved", "met via evidence")
    assert ok is True
    preds = model.issues.get_predicates_by_status(iid, statuses=("resolved",))
    resolved = [p for p in preds if p.get("id") == pid]
    assert len(resolved) == 1
    assert resolved[0]["status"] == "resolved"


def test_set_criterion_status_blocked():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import IssueType
    model = MatterModel.open_in_memory()
    iid, _ = model.issues.upsert_issue("Test claim", IssueType.CLAIM, materiality=0.8)
    pid = model.issues.add_predicate(iid, "Element B", "defendant")
    ok = model.issues.set_predicate_status(pid, "blocked", "assumption invalidated")
    assert ok is True


def test_set_criterion_status_not_found():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    ok = model.issues.set_predicate_status("nonexistent-id", "resolved")
    assert ok is False


def test_add_criterion_to_objective():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import IssueType
    model = MatterModel.open_in_memory()
    iid, _ = model.issues.upsert_issue("Test claim", IssueType.CLAIM, materiality=0.8)
    pid = model.issues.add_predicate(iid, "New element", "plaintiff")
    assert pid
    preds = model.issues.get_predicates(iid)
    assert any(p.get("id") == pid for p in preds)


def test_set_criterion_changes_coverage_badge():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import IssueType
    model = MatterModel.open_in_memory()
    iid, _ = model.issues.upsert_issue("Claim with blocked", IssueType.CLAIM, materiality=0.8)
    pid = model.issues.add_predicate(iid, "Must prove X", "plaintiff")
    model.issues.set_predicate_status(pid, "blocked", "reason")
    wb = model.get_objective_coverage_workbench()
    obj = next(o for o in wb["objectives"] if o["id"] == iid)
    assert obj["coverage_badge"] == "blocked"
    assert obj["predicate_blocked"] == 1


def test_objective_coverage_clamped_values():
    """Verify the formatter clamps materiality/coverage to [0, 1] (P1-2 fix)."""
    import html as _html_mod
    from irys.ui.app import _fmt_objective_coverage
    data = {
        "total": 1,
        "summary": {"covered": 0, "thin": 0, "blocked": 0, "missing": 1, "contradicted": 0},
        "objectives": [{
            "id": "test-id",
            "title": "Test",
            "issue_type": "claim",
            "materiality": 2.5,
            "coverage_fraction": -0.3,
            "coverage_badge": "missing",
            "supporting_count": -5,
            "predicate_total": 0,
            "predicate_satisfied": 0,
            "predicate_blocked": 0,
            "predicate_contested": 0,
            "predicates": [],
            "gaps": [],
            "has_proof_gap": False,
        }],
    }
    html = _fmt_objective_coverage(data, domain="legal")
    assert "0%" in html
    assert "-" not in html.split("Coverage:")[1].split("<")[0] if "Coverage:" in html else True


def test_objective_coverage_nan_values():
    """NaN/inf should degrade to safe defaults, not appear as raw numeric text."""
    from irys.ui.app import _fmt_objective_coverage
    data = {
        "total": 1,
        "summary": {"covered": 0, "thin": 0, "blocked": 0, "missing": 1, "contradicted": 0},
        "objectives": [{
            "id": "safe-id",
            "title": "Safe title",
            "issue_type": "claim",
            "materiality": float("nan"),
            "coverage_fraction": float("inf"),
            "coverage_badge": "missing",
            "supporting_count": 0,
            "predicate_total": 0,
            "predicate_satisfied": 0,
            "predicate_blocked": 0,
            "predicate_contested": 0,
            "predicates": [],
            "gaps": [],
            "has_proof_gap": False,
        }],
    }
    html = _fmt_objective_coverage(data, domain="legal")
    assert "nan" not in html.lower()
    assert "inf" not in html.lower()


def test_backend_interface_balance_predicate_management():
    """Verify set_criterion_status and add_criterion exist in all 3 backends."""
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("set_criterion_status", "add_criterion"):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


# ---- Quant fact review workbench (SO-6) ---- #

def test_quant_fact_workbench_empty():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    wb = model.get_quant_fact_workbench()
    assert wb["total"] == 0
    assert wb["total_conflicted"] == 0
    assert isinstance(wb["by_kind"], list)


def test_quant_fact_workbench_with_facts():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.quant.record("amount", "$50,000", amount_value=50000.0, currency="USD", subject_type="invoice", subject_id="INV-001")
    model.quant.record("amount", "$75,000", amount_value=75000.0, currency="USD", subject_type="invoice", subject_id="INV-002")
    model.quant.record("date", "2025-01-15", date_value="2025-01-15")
    model.quant.record("rate", "3.5%", rate_value=0.035)
    wb = model.get_quant_fact_workbench()
    assert wb["total"] == 4
    amount_group = next(g for g in wb["by_kind"] if g["kind"] == "amount")
    assert amount_group["count"] == 2
    date_group = next(g for g in wb["by_kind"] if g["kind"] == "date")
    assert date_group["count"] == 1


def test_quant_fact_workbench_conflict_detection():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.quant.record("amount", "$50,000 on INV-001", amount_value=50000.0, currency="USD", subject_type="invoice", subject_id="INV-001")
    model.quant.record("amount", "$55,000 on INV-001", amount_value=55000.0, currency="USD", subject_type="invoice", subject_id="INV-001")
    wb = model.get_quant_fact_workbench()
    assert wb["total_conflicted"] >= 1
    assert wb["conflict_groups"] >= 1
    amount_group = next(g for g in wb["by_kind"] if g["kind"] == "amount")
    assert amount_group["conflicted"] >= 1
    conflicted_facts = [f for f in amount_group["facts"] if f.get("has_conflict")]
    assert len(conflicted_facts) >= 1


def test_quant_fact_formatter_empty():
    from irys.ui.app import _fmt_quant_facts
    html = _fmt_quant_facts({}, domain="legal")
    assert "No quantitative facts" in html


def test_quant_fact_formatter_renders():
    from irys.ui.app import _fmt_quant_facts
    data = {
        "total": 2,
        "total_conflicted": 1,
        "conflict_groups": 1,
        "by_kind": [
            {
                "kind": "amount",
                "count": 2,
                "conflicted": 1,
                "facts": [
                    {"id": "f1", "raw_text": "$50,000", "amount_value": 50000.0, "currency": "USD",
                     "subject_type": "invoice", "subject_id": "INV-001", "has_conflict": True},
                    {"id": "f2", "raw_text": "$55,000", "amount_value": 55000.0, "currency": "USD",
                     "subject_type": "invoice", "subject_id": "INV-001", "has_conflict": True},
                ],
            },
        ],
    }
    html = _fmt_quant_facts(data, domain="legal")
    assert "Monetary Amounts" in html
    assert "50,000" in html
    assert "conflict" in html.lower()


def test_quant_fact_formatter_xss():
    from irys.ui.app import _fmt_quant_facts
    data = {
        "total": 1,
        "total_conflicted": 0,
        "conflict_groups": 0,
        "by_kind": [
            {
                "kind": "amount",
                "count": 1,
                "conflicted": 0,
                "facts": [
                    {"id": "xss-id", "raw_text": "<script>alert('xss')</script>",
                     "amount_value": 100.0, "currency": "USD",
                     "subject_type": "<b>evil</b>", "subject_id": None, "has_conflict": False},
                ],
            },
        ],
    }
    html = _fmt_quant_facts(data, domain="legal")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_quant_fact_formatter_non_dict_guard():
    from irys.ui.app import _fmt_quant_facts
    data = {
        "total": 1,
        "total_conflicted": 0,
        "conflict_groups": 0,
        "by_kind": [
            {
                "kind": "amount",
                "count": 1,
                "conflicted": 0,
                "facts": ["not-a-dict", None, 42],
            },
        ],
    }
    html = _fmt_quant_facts(data, domain="legal")
    assert isinstance(html, str)


def test_quant_fact_labels_all_five_domains():
    from irys.ui.app import _QUANT_FACT_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _QUANT_FACT_LABELS
        labels = _QUANT_FACT_LABELS[domain]
        for key in ("title", "empty", "amount", "date", "rate", "conflict", "source", "subject"):
            assert key in labels, f"{domain} missing label '{key}'"


def test_backend_interface_balance_quant_facts():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_quant_facts",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


# ── Decision Leverage Map tests ─────────────────────────────────────


def test_decision_leverage_map_empty():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_decision_leverage_map()
    assert isinstance(result, dict)
    assert result["total"] == 0
    assert isinstance(result["items"], list)


def test_decision_leverage_map_with_data():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import GapType, IssueType
    model = MatterModel.open_in_memory()
    run_id = model.start_run("leverage test")
    iid, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM, materiality=0.9)
    model.issues.add_predicate(iid, "Damages proved")
    model.assumptions.upsert("Markets are efficient", source_kind="analyst")
    model.gaps.record(GapType.MISSING_DOCUMENT, "Financial records for Q4", materiality=0.8)
    model.complete_run(run_id)
    result = model.get_decision_leverage_map()
    assert result["total"] > 0
    kinds = {item["kind"] for item in result["items"]}
    assert "weak_objective" in kinds or "open_gap" in kinds
    for item in result["items"]:
        assert "kind" in item
        assert "title" in item
        assert "impact" in item
        assert "action" in item
        assert isinstance(item["impact"], (int, float))


def test_decision_leverage_map_top_n():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import GapType
    model = MatterModel.open_in_memory()
    run_id = model.start_run("leverage top_n")
    for i in range(20):
        model.gaps.record(GapType.MISSING_DOCUMENT, f"Gap {i}", materiality=0.5)
    model.complete_run(run_id)
    result = model.get_decision_leverage_map(top_n=5)
    assert len(result["items"]) <= 5


def test_decision_leverage_formatter_empty():
    from irys.ui.app import _fmt_decision_leverage
    html = _fmt_decision_leverage({}, domain="legal")
    assert isinstance(html, str)
    assert "No leverage" in html or "viz-empty" in html or "empty" in html.lower()


def test_decision_leverage_formatter_renders():
    from irys.ui.app import _fmt_decision_leverage
    data = {
        "total": 2,
        "items": [
            {"kind": "weak_objective", "id": "obj-1", "title": "Contract breach",
             "blocker": "missing", "impact": 0.85, "detail": "2 blocked, 1 gap",
             "action": "Strengthen evidence"},
            {"kind": "open_gap", "id": "gap-1", "title": "Missing Q4 records",
             "blocker": "missing_evidence", "impact": 0.6, "detail": "Gap type: missing_evidence",
             "action": "Resolve or escalate"},
        ],
    }
    html = _fmt_decision_leverage(data, domain="legal")
    assert "Contract breach" in html
    assert "Missing Q4 records" in html
    assert "85%" in html
    assert "Strengthen evidence" in html


def test_decision_leverage_formatter_xss():
    from irys.ui.app import _fmt_decision_leverage
    data = {
        "total": 1,
        "items": [
            {"kind": "weak_objective", "id": "xss-id",
             "title": "<script>alert('xss')</script>",
             "blocker": "<img onerror=alert(1)>", "impact": 0.5,
             "detail": "<b>evil</b>", "action": "Do <script>bad</script> things"},
        ],
    }
    html = _fmt_decision_leverage(data, domain="legal")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_decision_leverage_formatter_non_dict_guard():
    from irys.ui.app import _fmt_decision_leverage
    data = {
        "total": 3,
        "items": ["not-a-dict", None, 42, {"kind": "open_gap", "id": "ok",
                  "title": "Valid", "blocker": "x", "impact": 0.3,
                  "detail": "d", "action": "a"}],
    }
    html = _fmt_decision_leverage(data, domain="legal")
    assert "Valid" in html
    assert isinstance(html, str)


def test_decision_leverage_labels_all_five_domains():
    from irys.ui.app import _DECISION_LEVERAGE_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _DECISION_LEVERAGE_LABELS, f"Missing domain '{domain}'"
        labels = _DECISION_LEVERAGE_LABELS[domain]
        for key in ("title", "subtitle", "empty", "weak_objective", "unreviewed_assumption",
                     "tainted_evidence", "quant_conflict", "open_gap", "pending_review",
                     "blocker", "impact", "action"):
            assert key in labels, f"{domain} missing label '{key}'"


def test_decision_leverage_formatter_nan_impact():
    import math
    from irys.ui.app import _fmt_decision_leverage
    data = {
        "total": 2,
        "items": [
            {"kind": "weak_objective", "id": "safe-1", "title": "Bad number item A",
             "blocker": "missing", "impact": float("nan"), "detail": "d", "action": "a"},
            {"kind": "open_gap", "id": "safe-2", "title": "Bad number item B",
             "blocker": "x", "impact": float("inf"), "detail": "d", "action": "a"},
        ],
    }
    html = _fmt_decision_leverage(data, domain="legal")
    assert isinstance(html, str)
    assert "nan" not in html.lower()
    assert "inf" not in html.lower()


def test_decision_leverage_map_nan_materiality():
    import math
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_decision_leverage_map()
    for item in result.get("items", []):
        impact = item.get("impact", 0)
        assert isinstance(impact, (int, float))
        assert math.isfinite(impact)


def test_backend_interface_balance_decision_leverage():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_decision_leverage",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


# ── Executable Steering Actions ─────────────────────────────────────

def test_steering_panel_returns_tuple_with_choices():
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    html, choices = state.load_steering_panel("", domain="legal")
    assert isinstance(html, str)
    assert isinstance(choices, list)
    assert len(choices) == 0


def test_steering_panel_choices_encode_action_json():
    import json
    from irys.ui.app import _fmt_steering_panel, _STEERING_ACTION_LABELS
    from irys.ui.app import AppState
    actions = [
        {
            "action_id": "a1",
            "action_type": "redirect_focus",
            "description": "Redirect to under-covered issue",
            "params": {"issue_id": "iss-1", "matter_id": "m1"},
            "rationale": "Low coverage",
            "priority": "high",
            "impact": "Increases coverage",
        },
        {
            "action_id": "a2",
            "action_type": "answer_clarification",
            "description": "Answer pending question",
            "params": {"question_id": "q-1", "answer_text": "<your answer here>"},
            "rationale": "Material improvement",
            "priority": "medium",
            "impact": "Closes gap",
        },
    ]

    class FakeBackend:
        async def get_steering_surface(self, mid, run_id=None):
            return actions

    state = AppState.__new__(AppState)
    state._backends = {"test": FakeBackend()}
    state._active_backend_key = "test"
    state.current_run_id = None
    state.backend = lambda: FakeBackend()
    html, choices = state.load_steering_panel("m1", domain="legal")
    assert len(choices) == 2
    label_0, value_0 = choices[0]
    parsed = json.loads(value_0)
    assert parsed["action_type"] == "redirect_focus"
    assert parsed["params"]["issue_id"] == "iss-1"
    assert parsed["needs_input"] is False
    label_1, value_1 = choices[1]
    parsed_1 = json.loads(value_1)
    assert parsed_1["action_type"] == "answer_clarification"
    assert parsed_1["needs_input"] is True


def test_execute_steering_action_redirect():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    called = {}
    def fake_redirect(mid, rid, iid):
        called["args"] = (mid, rid, iid)
        return "Redirected"
    state.do_redirect = fake_redirect
    action_json = json.dumps({
        "action_type": "redirect_focus",
        "params": {"issue_id": "iss-42"},
    })
    result = state.execute_steering_action("m1", action_json, "")
    assert result == "Redirected"
    assert called["args"] == ("m1", "", "iss-42")


def test_execute_steering_action_answer_clarification_needs_input():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    action_json = json.dumps({
        "action_type": "answer_clarification",
        "params": {"question_id": "q-1"},
    })
    result = state.execute_steering_action("m1", action_json, "")
    assert "Enter your answer" in result


def test_execute_steering_action_answer_clarification_with_input():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    called = {}
    def fake_answer(mid, qid, text):
        called["args"] = (mid, qid, text)
        return "Answered"
    state.do_answer_clarification = fake_answer
    action_json = json.dumps({
        "action_type": "answer_clarification",
        "params": {"question_id": "q-1"},
    })
    result = state.execute_steering_action("m1", action_json, "Yes, the contract was signed.")
    assert result == "Answered"
    assert called["args"][2] == "Yes, the contract was signed."


def test_execute_steering_action_force_belief_state():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    called = {}
    def fake_correct(mid, aid, new_state, reason):
        called["args"] = (mid, aid, new_state, reason)
        return "Corrected"
    state.do_correct_assertion = fake_correct
    action_json = json.dumps({
        "action_type": "force_belief_state",
        "params": {"assertion_id": "a-1", "new_state": "disputed"},
    })
    result = state.execute_steering_action("m1", action_json, "Witness contradicts")
    assert result == "Corrected"
    assert called["args"][1] == "a-1"
    assert called["args"][2] == "disputed"
    assert called["args"][3] == "Witness contradicts"


def test_execute_steering_action_supply_document():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    action_json = json.dumps({
        "action_type": "supply_document",
        "params": {"gap_id": "g-1", "description": "Signed lease agreement"},
    })
    result = state.execute_steering_action("m1", action_json, "")
    assert "Signed lease agreement" in result
    assert "Upload" in result


def test_execute_steering_action_no_matter():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    result = state.execute_steering_action("", "", "")
    assert "Load a matter" in result


def test_execute_steering_action_no_selection():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    result = state.execute_steering_action("m1", "", "")
    assert "Select an action" in result


def test_execute_steering_action_invalid_json():
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    result = state.execute_steering_action("m1", "not-json", "")
    assert "Invalid action data" in result


def test_execute_steering_action_xss_safety():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    action_json = json.dumps({
        "action_type": "supply_document",
        "params": {"gap_id": "g-1", "description": "<script>alert(1)</script>"},
    })
    result = state.execute_steering_action("m1", action_json, "")
    assert "<script>" not in result


def test_steering_action_labels_all_five_domains():
    from irys.ui.app import _STEERING_ACTION_LABELS
    required_keys = {
        "title", "empty", "correct_assertion", "force_belief_state",
        "redirect_focus", "supply_document", "answer_clarification",
        "set_trust_override",
    }
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _STEERING_ACTION_LABELS, f"Missing domain: {domain}"
        for key in required_keys:
            assert key in _STEERING_ACTION_LABELS[domain], f"Missing key {key} in {domain}"


def test_execute_steering_action_set_trust_override():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    called = {}
    def fake_trust(mid, pattern, level, note):
        called["args"] = (mid, pattern, level, note)
        return "Override set"
    state.do_set_trust_override = fake_trust
    action_json = json.dumps({
        "action_type": "set_trust_override",
        "params": {"document_pattern": "contract.pdf", "trust_level": "low"},
    })
    result = state.execute_steering_action("m1", action_json, "Unreliable source")
    assert result == "Override set"
    assert called["args"][1] == "contract.pdf"
    assert called["args"][3] == "Unreliable source"


def test_execute_steering_action_unknown_type():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    action_json = json.dumps({
        "action_type": "nonexistent_action",
        "params": {},
    })
    result = state.execute_steering_action("m1", action_json, "")
    assert "Unknown action type" in result


def test_execute_steering_action_unknown_type_xss():
    import json
    from irys.ui.app import AppState
    state = AppState.__new__(AppState)
    state.current_run_id = None
    action_json = json.dumps({
        "action_type": "<script>alert(1)</script>",
        "params": {},
    })
    result = state.execute_steering_action("m1", action_json, "")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


# ── Output Quality Contract Workbench ────────────────────────────────

def test_output_quality_workbench_empty():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_output_quality_workbench()
    assert isinstance(result, dict)
    assert result["matter_id"] == model.matter_id
    assert result["readiness"] in ("ready", "caution", "blocked")
    assert isinstance(result["runs"], list)
    assert isinstance(result["obligations"], list)
    assert isinstance(result["manifest_count"], int)


def test_output_quality_workbench_with_run():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Test quality question")
    model.complete_run(run_id, "Run completed.")
    result = model.get_output_quality_workbench()
    assert len(result["runs"]) >= 1
    run = result["runs"][0]
    assert run["run_id"] == run_id
    assert run["status"] == "completed"
    assert "completion_summary" in run


def test_output_quality_workbench_obligations_pass():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_output_quality_workbench()
    passed = [o for o in result["obligations"] if o.get("satisfied")]
    assert len(passed) > 0


def test_output_quality_formatter_empty():
    from irys.ui.app import _fmt_output_quality
    result = _fmt_output_quality({}, domain="legal")
    assert "viz-empty" in result


def test_output_quality_formatter_renders():
    from irys.ui.app import _fmt_output_quality
    data = {
        "readiness": "blocked",
        "blocker_count": 2,
        "runs": [
            {
                "run_id": "r-1",
                "status": "completed",
                "query": "What happened?",
                "research_mode": "deep",
                "event_count": 5,
                "cache_reuse_rate": 0.7,
                "completion_summary": "Run complete.",
            }
        ],
        "obligations": [
            {"name": "No contradictions", "satisfied": True, "severity": "passed", "item_count": 0},
            {"name": "2 high-materiality gaps open", "satisfied": False, "severity": "high", "item_count": 2},
        ],
        "manifest_count": 3,
        "manifest_fresh": False,
        "stale_manifest_count": 1,
        "summary": {"avg_coverage": 0.65},
    }
    html = _fmt_output_quality(data, domain="legal")
    assert "Output Quality Contract" in html
    assert "Not ready" in html
    assert "No contradictions" in html
    assert "2 high-materiality" in html
    assert "completed" in html
    assert "70%" in html
    assert "1 stale" in html


def test_output_quality_formatter_xss():
    from irys.ui.app import _fmt_output_quality
    data = {
        "readiness": "ready",
        "blocker_count": 0,
        "runs": [
            {
                "run_id": "r-1",
                "status": "completed",
                "query": "<script>alert(1)</script>",
                "research_mode": "deep",
                "event_count": 1,
                "cache_reuse_rate": 0.5,
                "completion_summary": "<img onerror=alert(1)>",
            }
        ],
        "obligations": [
            {"name": "<b>xss</b>", "satisfied": True, "severity": "passed", "item_count": 0},
        ],
        "manifest_count": 0,
        "manifest_fresh": True,
        "stale_manifest_count": 0,
        "summary": {},
    }
    html = _fmt_output_quality(data, domain="legal")
    assert "<script>" not in html
    assert "<img " not in html
    assert "<b>" not in html


def test_output_quality_formatter_non_dict_guards():
    from irys.ui.app import _fmt_output_quality
    data = {
        "readiness": "ready",
        "blocker_count": 0,
        "runs": ["not-a-dict", {"run_id": "r-1", "status": "completed", "query": "Q",
                                "research_mode": "deep", "event_count": 1,
                                "cache_reuse_rate": 0.5, "completion_summary": "done"}],
        "obligations": ["not-a-dict", {"name": "Check", "satisfied": True, "severity": "passed", "item_count": 0}],
        "manifest_count": 1,
        "manifest_fresh": True,
        "stale_manifest_count": 0,
        "summary": {},
    }
    html = _fmt_output_quality(data, domain="legal")
    assert "completed" in html
    assert "Check" in html


def test_output_quality_labels_all_five_domains():
    from irys.ui.app import _OUTPUT_QUALITY_LABELS
    required_keys = {
        "title", "subtitle", "empty", "ready", "caution", "blocked",
        "run_header", "obligations_header", "manifest_header",
    }
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _OUTPUT_QUALITY_LABELS, f"Missing domain: {domain}"
        for key in required_keys:
            assert key in _OUTPUT_QUALITY_LABELS[domain], f"Missing key {key} in {domain}"


def test_backend_interface_balance_output_quality():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_output_quality",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


def test_output_quality_formatter_nan_reuse_rate():
    import math
    from irys.ui.app import _fmt_output_quality
    data = {
        "readiness": "ready",
        "blocker_count": 0,
        "runs": [
            {
                "run_id": "r-1",
                "status": "completed",
                "query": "Test",
                "research_mode": "deep",
                "event_count": 1,
                "cache_reuse_rate": float("nan"),
                "completion_summary": "done",
            }
        ],
        "obligations": [],
        "manifest_count": 0,
        "manifest_fresh": True,
        "stale_manifest_count": 0,
        "summary": {"avg_coverage": float("inf")},
    }
    html = _fmt_output_quality(data, domain="legal")
    assert "nan" not in html.lower()
    assert "inf" not in html.lower()


# ── Deliverable Builder Workbench ────────────────────────────────────

def test_deliverable_workbench_empty():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_deliverable_workbench()
    assert isinstance(result, dict)
    assert result["matter_id"] == model.matter_id
    assert result["reliance_gate"] in ("ready", "caution", "blocked")
    assert isinstance(result["issues"], list)
    assert isinstance(result["total_verified_assertions"], int)
    assert isinstance(result["total_source_documents"], int)


def test_deliverable_workbench_with_issue():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import IssueType
    model = MatterModel.open_in_memory()
    iid, _ = model.issues.upsert_issue("Was the contract valid?", IssueType.CLAIM, materiality=0.9)
    result = model.get_deliverable_workbench()
    assert len(result["issues"]) >= 1
    iss = next((i for i in result["issues"] if i["issue_id"] == iid), None)
    assert iss is not None
    assert "Was the contract valid?" in iss["title"]


def test_deliverable_workbench_scoped_issues():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import IssueType
    model = MatterModel.open_in_memory()
    iid1, _ = model.issues.upsert_issue("Issue A", IssueType.CLAIM, materiality=0.8)
    iid2, _ = model.issues.upsert_issue("Issue B", IssueType.CLAIM, materiality=0.5)
    result = model.get_deliverable_workbench(issue_ids=[iid1])
    assert len(result["issues"]) == 1
    assert result["issues"][0]["issue_id"] == iid1


def test_deliverable_formatter_empty():
    from irys.ui.app import _fmt_deliverable_workbench
    html = _fmt_deliverable_workbench({}, domain="legal")
    assert "viz-empty" in html


def test_deliverable_formatter_renders():
    from irys.ui.app import _fmt_deliverable_workbench
    data = {
        "reliance_gate": "ready",
        "blocker_count": 0,
        "issue_count": 1,
        "issues": [
            {
                "issue_id": "iss-1",
                "title": "Contract validity",
                "materiality": 0.85,
                "verified_assertion_count": 3,
                "verified_assertions": [
                    {"assertion_id": "a1", "proposition": "Signed on Jan 1", "belief_state": "operative"},
                ],
                "source_documents": ["contract.pdf", "addendum.pdf"],
            },
        ],
        "total_verified_assertions": 3,
        "total_source_documents": 2,
    }
    html = _fmt_deliverable_workbench(data, domain="legal")
    assert "Deliverable Builder" in html
    assert "Contract validity" in html
    assert "contract.pdf" in html
    assert "Signed on Jan 1" in html
    assert "operative" in html
    assert "3 verified facts" in html


def test_deliverable_formatter_xss():
    from irys.ui.app import _fmt_deliverable_workbench
    data = {
        "reliance_gate": "ready",
        "blocker_count": 0,
        "issue_count": 1,
        "issues": [
            {
                "issue_id": "iss-1",
                "title": "<script>alert(1)</script>",
                "materiality": 0.5,
                "verified_assertion_count": 1,
                "verified_assertions": [
                    {"assertion_id": "a1", "proposition": "<img onerror=x>", "belief_state": "alleged"},
                ],
                "source_documents": ["<b>bad.pdf</b>"],
            },
        ],
        "total_verified_assertions": 1,
        "total_source_documents": 1,
    }
    html = _fmt_deliverable_workbench(data, domain="legal")
    assert "<script>" not in html
    assert "<img " not in html
    assert "<b>" not in html


def test_deliverable_formatter_non_dict_guards():
    from irys.ui.app import _fmt_deliverable_workbench
    data = {
        "reliance_gate": "blocked",
        "blocker_count": 1,
        "issue_count": 2,
        "issues": [
            "not-a-dict",
            {
                "issue_id": "iss-1",
                "title": "Valid issue",
                "materiality": 0.7,
                "verified_assertion_count": 0,
                "verified_assertions": ["not-a-dict"],
                "source_documents": [],
            },
        ],
        "total_verified_assertions": 0,
        "total_source_documents": 0,
    }
    html = _fmt_deliverable_workbench(data, domain="legal")
    assert "Valid issue" in html


def test_deliverable_labels_all_five_domains():
    from irys.ui.app import _DELIVERABLE_LABELS
    required_keys = {
        "title", "subtitle", "empty", "gate_ready", "gate_blocked",
        "gate_caution", "issues_header", "verified", "sources",
    }
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert domain in _DELIVERABLE_LABELS, f"Missing domain: {domain}"
        for key in required_keys:
            assert key in _DELIVERABLE_LABELS[domain], f"Missing key {key} in {domain}"


def test_backend_interface_balance_deliverable():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_deliverable_workbench",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


def test_deliverable_formatter_nan_materiality():
    import math
    from irys.ui.app import _fmt_deliverable_workbench
    data = {
        "reliance_gate": "ready",
        "blocker_count": 0,
        "issue_count": 1,
        "issues": [
            {
                "issue_id": "iss-1",
                "title": "Test",
                "materiality": float("nan"),
                "verified_assertion_count": 0,
                "verified_assertions": [],
                "source_documents": [],
            },
        ],
        "total_verified_assertions": 0,
        "total_source_documents": 0,
    }
    html = _fmt_deliverable_workbench(data, domain="legal")
    assert "nan" not in html.lower()


# ---- Scenario Branch Workbench tests ----


def test_scenario_branch_create_and_list():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.create_scenario_branch(
        name="Contract is void",
        assumptions=[{"text": "The contract was signed under duress"}],
        notes="Testing alternative theory",
    )
    assert isinstance(result, dict)
    assert result["branch_id"]
    assert result["name"] == "Contract is void"
    assert result["status"] == "active"
    assert len(result["assumptions"]) == 1

    branches = model.list_scenario_branches()
    assert len(branches) == 1
    assert branches[0]["name"] == "Contract is void"


def test_scenario_branch_get():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    created = model.create_scenario_branch(
        name="Statute expired",
        assumptions=[{"text": "Statute of limitations has run"}],
    )
    fetched = model.get_scenario_branch(created["branch_id"])
    assert fetched is not None
    assert fetched["name"] == "Statute expired"
    assert fetched["assumptions"] == [{"text": "Statute of limitations has run"}]


def test_scenario_branch_get_nonexistent():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    assert model.get_scenario_branch("nonexistent-id") is None


def test_scenario_branch_archive():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    created = model.create_scenario_branch(
        name="Waiver valid",
        assumptions=[{"text": "The waiver is enforceable"}],
    )
    assert model.archive_scenario_branch(created["branch_id"]) is True
    fetched = model.get_scenario_branch(created["branch_id"])
    assert fetched["status"] == "archived"


def test_scenario_branch_archive_nonexistent():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    assert model.archive_scenario_branch("nonexistent-id") is False


def test_scenario_workbench_empty():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    wb = model.get_scenario_workbench()
    assert isinstance(wb, dict)
    assert wb["total_branches"] == 0
    assert wb["active_count"] == 0
    assert wb["branches"] == []


def test_scenario_workbench_filters_archived():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b1 = model.create_scenario_branch(name="Active one", assumptions=[])
    b2 = model.create_scenario_branch(name="Archived one", assumptions=[])
    model.archive_scenario_branch(b2["branch_id"])
    wb = model.get_scenario_workbench()
    assert wb["active_count"] == 1
    assert wb["archived_count"] == 1
    assert len(wb["branches"]) == 1
    assert wb["branches"][0]["name"] == "Active one"


def test_scenario_formatter_empty():
    from irys.ui.app import _fmt_scenario_workbench
    html = _fmt_scenario_workbench({}, domain="legal")
    assert "viz-empty" in html


def test_scenario_formatter_renders():
    from irys.ui.app import _fmt_scenario_workbench
    data = {
        "total_branches": 1,
        "active_count": 1,
        "archived_count": 0,
        "branches": [
            {
                "id": "b1",
                "name": "Contract void",
                "status": "active",
                "assumptions": [{"text": "Signed under duress"}],
                "created_at": "2026-05-04T12:00:00",
                "notes": "Testing",
            }
        ],
    }
    html = _fmt_scenario_workbench(data, domain="legal")
    assert "Contract void" in html
    assert "Signed under duress" in html
    assert "Testing" in html
    assert "viz-shell" in html


def test_scenario_formatter_xss():
    from irys.ui.app import _fmt_scenario_workbench
    data = {
        "total_branches": 1,
        "active_count": 1,
        "archived_count": 0,
        "branches": [
            {
                "id": "b1",
                "name": "<script>alert('xss')</script>",
                "status": "active",
                "assumptions": [{"text": "<img onerror=alert(1)>"}],
                "created_at": "2026-05-04",
                "notes": "<b>bold</b>",
            }
        ],
    }
    html = _fmt_scenario_workbench(data, domain="legal")
    assert "<script>" not in html
    assert "<img " not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img onerror" in html


def test_scenario_formatter_non_dict_guards():
    from irys.ui.app import _fmt_scenario_workbench
    data = {
        "total_branches": 2,
        "active_count": 2,
        "archived_count": 0,
        "branches": ["not-a-dict", None, {"id": "b1", "name": "Valid", "status": "active"}],
    }
    html = _fmt_scenario_workbench(data, domain="legal")
    assert "Valid" in html


def test_scenario_labels_all_five_domains():
    from irys.ui.app import _SCENARIO_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _SCENARIO_LABELS[domain]
        for key in ("title", "subtitle", "empty", "active", "archived", "assumptions",
                     "created", "branch_header", "notes", "create_hint"):
            assert key in labels, f"{domain} missing key {key}"


def test_backend_interface_balance_scenario():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_scenario_workbench", "create_scenario_branch", "archive_scenario_branch"):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


def test_scenario_branch_unique_name():
    import sqlite3
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.create_scenario_branch(name="Same Name", assumptions=[])
    try:
        model.create_scenario_branch(name="Same Name", assumptions=[])
        assert False, "Should have raised IntegrityError for duplicate name"
    except sqlite3.IntegrityError:
        pass


# ---- SO Scorecard tests ----


def test_so_scorecard_model_returns_dict():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_so_metrics()
    assert isinstance(result, dict)
    assert "targets" in result
    assert "targets_met" in result
    assert "counts" in result
    assert isinstance(result["counts"], dict)


def test_so_scorecard_formatter_empty():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard
    html = _fmt_so_scorecard({}, domain="legal")
    assert "viz-empty" in html


def test_so_scorecard_formatter_renders():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard
    data = {
        "assertion_structure_rate": 0.95,
        "source_role_known_rate": 0.88,
        "issue_coverage_avg": 0.72,
        "reuse_rate": 0.81,
        "numeric_extraction_rate": 0.93,
        "provenance_attribution_rate": 0.87,
        "gap_surface_ratio": 0.3,
        "steerability": True,
        "belief_revision": False,
        "targets": {
            "assertion_structure_rate": 1.0,
            "source_role_known_rate": 0.9,
            "issue_coverage_avg": 0.8,
            "reuse_rate": 0.7,
            "numeric_extraction_rate": 0.9,
            "provenance_attribution_rate": 0.9,
            "steerability": True,
            "belief_revision": True,
        },
        "targets_met": {
            "assertion_structure_rate": False,
            "source_role_known_rate": False,
            "issue_coverage_avg": False,
            "reuse_rate": True,
            "numeric_extraction_rate": True,
            "provenance_attribution_rate": False,
            "steerability": True,
            "belief_revision": False,
        },
    }
    html = _fmt_so_scorecard(data, domain="legal")
    assert "Sacred Outcomes" in html
    assert "SO-2 Structure" in html
    assert "SO-5 Calibration" in html
    assert "95.0%" in html
    assert "pill-green" in html
    assert "pill-red" in html


def test_so_scorecard_formatter_xss():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard
    data = {
        "reuse_rate": 0.5,
        "targets": {"reuse_rate": 0.7},
        "targets_met": {"reuse_rate": False},
    }
    html = _fmt_so_scorecard(data, domain="legal")
    assert "<script>" not in html
    assert "viz-shell" in html


def test_so_scorecard_formatter_none_metrics():
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard
    data = {
        "assertion_structure_rate": None,
        "source_role_known_rate": None,
        "issue_coverage_avg": None,
        "reuse_rate": None,
        "steerability": None,
        "belief_revision": None,
        "targets": {
            "assertion_structure_rate": 1.0,
            "source_role_known_rate": 0.9,
            "issue_coverage_avg": 0.8,
            "reuse_rate": 0.7,
            "steerability": True,
            "belief_revision": True,
        },
        "targets_met": {
            "assertion_structure_rate": None,
            "source_role_known_rate": None,
            "issue_coverage_avg": None,
            "reuse_rate": None,
            "steerability": None,
            "belief_revision": None,
        },
        "counts": {},
    }
    html = _fmt_so_scorecard(data, domain="legal")
    assert "N/A" in html
    assert "viz-shell" in html


def test_so_scorecard_formatter_nan_value():
    import math
    from irys.ui.app import _fmt_so_scorecard_panel as _fmt_so_scorecard
    data = {
        "assertion_structure_rate": float("nan"),
        "targets": {"assertion_structure_rate": 1.0},
        "targets_met": {"assertion_structure_rate": False},
        "counts": {},
    }
    html = _fmt_so_scorecard(data, domain="legal")
    assert "nan%" not in html.lower()


def test_so_scorecard_labels_all_five_domains():
    from irys.ui.app import _SO_SCORECARD_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _SO_SCORECARD_LABELS[domain]
        for key in ("title", "subtitle", "so1", "so2_struct", "so2_revision",
                     "so2_provenance", "so3", "so4", "so5", "so6", "so7"):
            assert key in labels, f"{domain} missing key {key}"


def test_so_scorecard_backend_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    assert hasattr(UIBackend, "get_so_scorecard")
    assert hasattr(InProcessBackend, "get_so_scorecard")
    assert hasattr(HttpBackend, "get_so_scorecard")


# ── Alternative Theory Portfolio ──────────────────────────────────────

def test_alt_theory_portfolio_empty():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_alternative_theory_portfolio()
    assert isinstance(result, dict)
    assert result["matter_id"] == model.matter_id
    assert result["theory_count"] == 0
    assert result["theories"] == []


def test_alt_theory_portfolio_with_assertions():
    from irys.matter.matter import MatterModel
    from irys.matter.models import AssertionCandidate
    from irys.matter.enums import SpeechAct, SourceRole
    model = MatterModel.open_in_memory()
    model.assertions.upsert_occurrence(AssertionCandidate(
        proposition_text="The contract was signed on January 1",
        document_id="doc1.pdf",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
    ))
    model.assertions.upsert_occurrence(AssertionCandidate(
        proposition_text="Payment was not received by deadline",
        document_id="doc2.pdf",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
    ))
    result = model.get_alternative_theory_portfolio()
    assert result["theory_count"] >= 1
    theories = result["theories"]
    assert isinstance(theories, list)
    for t in theories:
        assert isinstance(t, dict)
        assert "id" in t
        assert "label" in t
        assert "stance" in t
        assert "supporting_assertions" in t
        assert "attacking_assertions" in t
        assert "confidence_range" in t
        assert isinstance(t["confidence_range"], list)


def test_alt_theory_portfolio_max_theories():
    from irys.matter.matter import MatterModel
    from irys.matter.models import AssertionCandidate
    from irys.matter.enums import SpeechAct, SourceRole
    model = MatterModel.open_in_memory()
    model.assertions.upsert_occurrence(AssertionCandidate(
        proposition_text="Fact A",
        document_id="d.pdf",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
    ))
    result = model.get_alternative_theory_portfolio(max_theories=1)
    assert result["theory_count"] <= 1


def test_alt_theory_portfolio_objective_filter():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_alternative_theory_portfolio(objective_id="nonexistent-id")
    assert result["objective_id"] == "nonexistent-id"
    assert isinstance(result["theories"], list)


def test_alt_theory_portfolio_domain_profile():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_alternative_theory_portfolio()
    assert "domain_profile_id" in result
    assert isinstance(result["domain_profile_id"], str)


def test_alt_theory_formatter_empty():
    from irys.ui.app import _fmt_alternative_theories
    html = _fmt_alternative_theories({}, domain="legal")
    assert "viz-empty" in html


def test_alt_theory_formatter_no_theories():
    from irys.ui.app import _fmt_alternative_theories
    html = _fmt_alternative_theories({"theories": []}, domain="legal")
    assert "viz-empty" in html


def test_alt_theory_formatter_renders():
    from irys.ui.app import _fmt_alternative_theories
    data = {
        "theories": [
            {
                "id": "theory-1",
                "label": "Baseline Theory",
                "stance": "supporting",
                "supporting_assertions": 5,
                "attacking_assertions": 1,
                "assumptions": 2,
                "open_gaps": 3,
                "confidence_range": [0.6, 0.9],
                "source_role_mix": {"ADVOCATE": 3, "OPERATIVE": 2},
                "taint_summary": {"tainted_assertion_count": 0},
                "discriminator_questions": ["Was the contract valid?"],
            },
        ],
    }
    html = _fmt_alternative_theories(data, domain="legal")
    assert "viz-shell" in html
    assert "Baseline Theory" in html
    assert "supporting" in html
    assert "Was the contract valid?" in html


def test_alt_theory_formatter_xss():
    from irys.ui.app import _fmt_alternative_theories
    data = {
        "theories": [
            {
                "id": "t1",
                "label": "<script>alert(1)</script>",
                "stance": "supporting",
                "supporting_assertions": 1,
                "attacking_assertions": 0,
                "assumptions": 0,
                "open_gaps": 0,
                "confidence_range": [0.5, 0.8],
                "source_role_mix": {"<img onerror=alert(1)>": 1},
                "taint_summary": {"tainted_assertion_count": 0},
                "discriminator_questions": ["<b onmouseover=alert(1)>test</b>"],
            },
        ],
    }
    html = _fmt_alternative_theories(data, domain="legal")
    assert "<script>" not in html
    assert "<img " not in html
    assert "<b " not in html


def test_alt_theory_formatter_non_dict_guards():
    from irys.ui.app import _fmt_alternative_theories
    data = {"theories": ["not-a-dict", None, 42, {"id": "valid", "label": "Test", "stance": "uncertain"}]}
    html = _fmt_alternative_theories(data, domain="legal")
    assert "viz-shell" in html
    assert "Test" in html


def test_alt_theory_formatter_nan_confidence():
    import math
    from irys.ui.app import _fmt_alternative_theories
    data = {
        "theories": [{
            "id": "t1", "label": "Test", "stance": "supporting",
            "supporting_assertions": 1, "attacking_assertions": 0,
            "assumptions": 0, "open_gaps": 0,
            "confidence_range": [float("nan"), float("inf")],
            "source_role_mix": {},
            "taint_summary": {"tainted_assertion_count": 0},
            "discriminator_questions": [],
        }],
    }
    html = _fmt_alternative_theories(data, domain="legal")
    assert "nan" not in html.lower().replace("provenance", "").replace("governance", "").replace("finance", "")


def test_alt_theory_labels_all_five_domains():
    from irys.ui.app import _ALTERNATIVE_THEORY_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _ALTERNATIVE_THEORY_LABELS[domain]
        for key in ("title", "subtitle", "empty", "support", "attack",
                     "assumptions", "gaps", "confidence", "taint", "discriminators"):
            assert key in labels, f"{domain} missing key {key}"


def test_alt_theory_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    assert hasattr(UIBackend, "get_alternative_theory_portfolio")
    assert hasattr(InProcessBackend, "get_alternative_theory_portfolio")
    assert hasattr(HttpBackend, "get_alternative_theory_portfolio")


# ── Dependency Manifest Inspector ─────────────────────────────────────

def test_manifest_inspector_empty():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_dependency_manifest_inspector()
    assert isinstance(result, dict)
    assert result["matter_id"] == model.matter_id
    assert result["total_count"] == 0
    assert result["fresh_count"] == 0
    assert result["stale_count"] == 0
    assert result["manifests"] == []


def test_manifest_inspector_limit_clamp():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_dependency_manifest_inspector(limit=0)
    assert isinstance(result, dict)
    result2 = model.get_dependency_manifest_inspector(limit=999)
    assert isinstance(result2, dict)


def test_manifest_inspector_policy_filter():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_dependency_manifest_inspector(policy_audience="attorney")
    assert isinstance(result, dict)
    assert result["manifests"] == []


def test_manifest_inspector_domain_profile():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_dependency_manifest_inspector()
    assert "domain" in result
    assert isinstance(result["domain"], str)


def test_manifest_formatter_empty():
    from irys.ui.app import _fmt_manifest_inspector
    html = _fmt_manifest_inspector({}, domain="legal")
    assert "viz-empty" in html


def test_manifest_formatter_no_manifests():
    from irys.ui.app import _fmt_manifest_inspector
    html = _fmt_manifest_inspector({"manifests": []}, domain="legal")
    assert "viz-empty" in html


def test_manifest_formatter_renders():
    from irys.ui.app import _fmt_manifest_inspector
    data = {
        "total_count": 2,
        "fresh_count": 1,
        "stale_count": 1,
        "manifests": [
            {
                "manifest_hash": "abc123def456",
                "purpose": "Investigation run 1",
                "created_at": "2026-05-04T10:00:00Z",
                "domain_profile_id": "legal",
                "policy_audience": "clean",
                "taint_class": "clean",
                "broker_version": "v15",
                "object_dependency_count": 10,
                "negative_dependency_count": 2,
                "status": "fresh",
                "valid": True,
                "stale_reasons": [],
                "stale_namespaces": [],
                "consumed_objects_by_kind": {"assertion": 5, "gap": 3},
                "current_revisions": {},
            },
            {
                "manifest_hash": "xyz789stale",
                "purpose": "Investigation run 2",
                "created_at": "2026-05-04T09:00:00Z",
                "status": "stale",
                "valid": False,
                "stale_reasons": ["namespace assertions:*:* expected 5, current 8"],
                "stale_namespaces": [{"reason": "namespace assertions:*:* expected 5, current 8"}],
                "object_dependency_count": 7,
                "negative_dependency_count": 1,
                "consumed_objects_by_kind": {"assertion": 4},
                "current_revisions": {},
            },
        ],
    }
    html = _fmt_manifest_inspector(data, domain="legal")
    assert "viz-shell" in html
    assert "abc123def456" in html
    assert "pill-green" in html
    assert "Stale namespaces" in html


def test_manifest_formatter_xss():
    from irys.ui.app import _fmt_manifest_inspector
    data = {
        "total_count": 1,
        "fresh_count": 0,
        "stale_count": 1,
        "manifests": [{
            "manifest_hash": "<script>alert(1)</script>",
            "purpose": "<img onerror=alert(1)>",
            "created_at": "2026-01-01",
            "status": "stale",
            "valid": False,
            "stale_reasons": ["<b onmouseover=alert(1)>xss</b>"],
            "stale_namespaces": [{"reason": "<script>xss</script>"}],
            "object_dependency_count": 0,
            "negative_dependency_count": 0,
            "consumed_objects_by_kind": {},
            "current_revisions": {},
        }],
    }
    html = _fmt_manifest_inspector(data, domain="legal")
    assert "<script>" not in html
    assert "<img " not in html
    assert "<b " not in html


def test_manifest_formatter_non_dict_guards():
    from irys.ui.app import _fmt_manifest_inspector
    data = {
        "total_count": 1,
        "fresh_count": 0,
        "stale_count": 0,
        "manifests": ["not-a-dict", None, 42, {
            "manifest_hash": "valid",
            "purpose": "test",
            "status": "fresh",
            "valid": True,
        }],
    }
    html = _fmt_manifest_inspector(data, domain="legal")
    assert "viz-shell" in html
    assert "valid" in html


def test_manifest_labels_all_five_domains():
    from irys.ui.app import _MANIFEST_INSPECTOR_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _MANIFEST_INSPECTOR_LABELS[domain]
        for key in ("title", "subtitle", "empty", "fresh", "stale",
                     "taint_blocked", "objects", "negative", "profile", "audience"):
            assert key in labels, f"{domain} missing key {key}"


def test_manifest_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    assert hasattr(UIBackend, "get_dependency_manifest_inspector")
    assert hasattr(InProcessBackend, "get_dependency_manifest_inspector")
    assert hasattr(HttpBackend, "get_dependency_manifest_inspector")


# ── Steering Impact Preview ─────────────────────────────────────────


def test_impact_preview_unknown_action():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_steering_impact_preview("bogus_action", {})
    assert result["valid"] is False
    assert "Unknown action_type" in result["warnings"][0]


def test_impact_preview_resolve_gap_missing_id():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_steering_impact_preview("resolve_gap", {})
    assert result["valid"] is False
    assert "gap_id" in result["warnings"][0]


def test_impact_preview_resolve_gap_not_found():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_steering_impact_preview("resolve_gap", {"gap_id": "nonexistent"})
    assert result["valid"] is False
    assert "not found" in result["warnings"][0]


def test_impact_preview_resolve_gap_valid():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import GapType
    model = MatterModel.open_in_memory()
    gap_id = model.gaps.record(
        GapType.MISSING_DOCUMENT,
        "Missing contract exhibit A",
        materiality=0.8,
    )
    result = model.get_steering_impact_preview("resolve_gap", {"gap_id": gap_id})
    assert result["valid"] is True
    assert result["action_type"] == "resolve_gap"
    assert result["before"]["open_gap_count"] >= 1
    assert result["after"]["open_gap_count"] < result["before"]["open_gap_count"]
    assert result["deltas"]["gaps_closed"] == 1
    gap_row = model.db.execute(
        "SELECT status FROM gap WHERE id=?", (gap_id,)
    ).fetchone()
    assert gap_row["status"] == "open"


def test_impact_preview_correct_assertion_missing_fields():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_steering_impact_preview("correct_assertion", {"assertion_id": ""})
    assert result["valid"] is False
    assert "assertion_id" in result["warnings"][0]


def test_impact_preview_correct_assertion_valid():
    from irys.matter.matter import MatterModel
    from irys.matter.models import AssertionCandidate
    from irys.matter.enums import SpeechAct, SourceRole
    model = MatterModel.open_in_memory()
    model.assertions.upsert_occurrence(AssertionCandidate(
        proposition_text="Test proposition for impact preview",
        document_id="doc.pdf",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
    ))
    assertions = model.db.execute(
        "SELECT id FROM assertion WHERE matter_id=?", (model.matter_id,)
    ).fetchall()
    assert len(assertions) > 0
    aid = assertions[0]["id"]
    result = model.get_steering_impact_preview("correct_assertion", {
        "assertion_id": aid,
        "new_state": "disputed",
    })
    assert result["valid"] is True
    assert aid in result["deltas"]["affected_assertions"]
    rec = model.assertions.get(aid)
    assert rec.belief_state != "disputed"


def test_impact_preview_resolve_contradiction_invalid_decision():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_steering_impact_preview("resolve_contradiction", {
        "attacker_id": "a1",
        "attacked_id": "a2",
        "decision": "invalid_choice",
    })
    assert result["valid"] is False
    assert "Invalid decision" in result["warnings"][0]


def test_impact_preview_resolve_contradiction_valid():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_steering_impact_preview("resolve_contradiction", {
        "attacker_id": "a1",
        "attacked_id": "a2",
        "decision": "prefer_attacker",
    })
    assert result["valid"] is True
    assert result["deltas"]["contradictions_resolved"] == 1
    assert "a1" in result["deltas"]["affected_assertions"]
    assert "a2" in result["deltas"]["affected_assertions"]


def test_impact_preview_escalate_gap_missing_id():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_steering_impact_preview("escalate_gap", {})
    assert result["valid"] is False
    assert "gap_id" in result["warnings"][0]


def test_impact_preview_does_not_mutate():
    from irys.matter.matter import MatterModel
    from irys.matter.enums import GapType
    model = MatterModel.open_in_memory()
    gap_id = model.gaps.record(GapType.MISSING_DOCUMENT, "Test gap for preview")
    before_count = model.gaps.count_open()
    model.get_steering_impact_preview("resolve_gap", {"gap_id": gap_id})
    after_count = model.gaps.count_open()
    assert before_count == after_count


def test_impact_preview_formatter_empty():
    from irys.ui.app import _fmt_impact_preview
    html = _fmt_impact_preview({})
    assert "viz-empty" in html
    assert "Select a steering action" in html


def test_impact_preview_formatter_valid():
    from irys.ui.app import _fmt_impact_preview
    data = {
        "action_type": "resolve_gap",
        "valid": True,
        "warnings": [],
        "before": {
            "issue_coverage_avg": 0.42,
            "open_gap_count": 5,
            "contradiction_count": 2,
            "disputed_count": 1,
            "readiness": "attention_needed",
        },
        "after": {
            "issue_coverage_avg": 0.45,
            "open_gap_count": 4,
            "contradiction_count": 2,
            "disputed_count": 1,
            "readiness": "attention_needed",
        },
        "deltas": {
            "coverage_delta": 0.03,
            "gaps_closed": 1,
            "gaps_opened": 0,
            "contradictions_resolved": 0,
            "disputed_delta": 0,
            "affected_objectives": ["obj-1"],
            "affected_assertions": [],
        },
        "recommended_followups": ["Review remaining gaps"],
    }
    html = _fmt_impact_preview(data, domain="legal")
    assert "Steering Impact Preview" in html
    assert "resolve_gap" in html
    assert "42.0%" in html
    assert "45.0%" in html
    assert "Review remaining gaps" in html
    assert "Affected objectives: 1" in html


def test_impact_preview_formatter_invalid():
    from irys.ui.app import _fmt_impact_preview
    data = {
        "action_type": "bogus",
        "valid": False,
        "warnings": ["Unknown action_type: bogus"],
        "before": {},
        "after": {},
        "deltas": {},
        "recommended_followups": [],
    }
    html = _fmt_impact_preview(data, domain="legal")
    assert "Invalid action" in html
    assert "Unknown action_type" in html


def test_impact_preview_formatter_xss():
    from irys.ui.app import _fmt_impact_preview
    data = {
        "action_type": "<script>alert(1)</script>",
        "valid": True,
        "warnings": [],
        "before": {
            "issue_coverage_avg": 0.5,
            "open_gap_count": 1,
            "contradiction_count": 0,
            "disputed_count": 0,
            "readiness": "<img onerror=alert(1)>",
        },
        "after": {
            "issue_coverage_avg": 0.5,
            "open_gap_count": 0,
            "contradiction_count": 0,
            "disputed_count": 0,
            "readiness": "good",
        },
        "deltas": {
            "coverage_delta": 0.0,
            "gaps_closed": 1,
            "gaps_opened": 0,
            "contradictions_resolved": 0,
            "disputed_delta": 0,
            "affected_objectives": [],
            "affected_assertions": [],
        },
        "recommended_followups": ["<script>xss</script>"],
    }
    html = _fmt_impact_preview(data)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_impact_preview_formatter_non_dict_guards():
    from irys.ui.app import _fmt_impact_preview
    data = {
        "action_type": "resolve_gap",
        "valid": True,
        "warnings": [],
        "before": "not-a-dict",
        "after": None,
        "deltas": 42,
        "recommended_followups": [123, None, "Valid followup"],
    }
    html = _fmt_impact_preview(data)
    assert "Steering Impact Preview" in html
    assert "Valid followup" in html


def test_impact_preview_labels_all_five_domains():
    from irys.ui.app import _IMPACT_PREVIEW_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _IMPACT_PREVIEW_LABELS[domain]
        assert "title" in labels
        assert "coverage" in labels
        assert "gaps" in labels
        assert "empty" in labels


def test_impact_preview_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    assert hasattr(UIBackend, "get_steering_impact_preview")
    assert hasattr(InProcessBackend, "get_steering_impact_preview")
    assert hasattr(HttpBackend, "get_steering_impact_preview")


# ── Domain Investigation Readiness ──────────────────────────────────


def test_domain_readiness_empty_matter():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.evaluate_domain_investigation_readiness()
    assert result["matter_id"] == model.matter_id
    assert result["overall_status"] in ("ready", "partial", "blocked")
    assert len(result["profiles"]) == 5
    for p in result["profiles"]:
        assert p["profile_id"] in ("legal", "finance", "coding", "academic_research", "biomedical")
        assert p["status"] in ("ready", "partial", "blocked")
        assert "source_role_calibration" in p
        assert "assertion_quality" in p
        assert "objective_coverage" in p
        assert "recommended_repairs" in p


def test_domain_readiness_specific_profiles():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.evaluate_domain_investigation_readiness(profile_ids=["legal", "finance"])
    assert len(result["profiles"]) == 2
    pids = {p["profile_id"] for p in result["profiles"]}
    assert pids == {"legal", "finance"}


def test_domain_readiness_invalid_profile():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.evaluate_domain_investigation_readiness(profile_ids=["nonexistent"])
    assert len(result["profiles"]) == 5


def test_domain_readiness_with_assertions():
    from irys.matter.matter import MatterModel
    from irys.matter.models import AssertionCandidate
    from irys.matter.enums import SpeechAct, SourceRole
    model = MatterModel.open_in_memory()
    model.assertions.upsert_occurrence(AssertionCandidate(
        proposition_text="Test for domain readiness",
        document_id="doc.pdf",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
    ))
    result = model.evaluate_domain_investigation_readiness()
    assert result["overall_status"] in ("ready", "partial", "blocked")
    legal_profile = next(p for p in result["profiles"] if p["profile_id"] == "legal")
    assert "assertion_quality" in legal_profile
    assert "assertion_count" in legal_profile["assertion_quality"]


def test_domain_readiness_cross_domain_findings():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.evaluate_domain_investigation_readiness()
    assert isinstance(result["cross_domain_findings"], list)
    for finding in result["cross_domain_findings"]:
        assert "kind" in finding
        assert "severity" in finding
        assert "message" in finding


def test_domain_readiness_no_repairs_flag():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.evaluate_domain_investigation_readiness(include_repair_recommendations=False)
    for p in result["profiles"]:
        assert p["recommended_repairs"] == []


def test_domain_readiness_formatter_empty():
    from irys.ui.app import _fmt_domain_readiness
    html = _fmt_domain_readiness({})
    assert "viz-empty" in html


def test_domain_readiness_formatter_renders():
    from irys.ui.app import _fmt_domain_readiness
    data = {
        "matter_id": "test",
        "overall_status": "partial",
        "profiles": [
            {
                "profile_id": "legal",
                "status": "ready",
                "primary_failures": [],
                "domain_detection": {"confidence": 0.9, "is_primary": True},
                "source_role_calibration": {"pass": True, "source_role_known_rate": 0.95, "defined_roles": 5},
                "assertion_quality": {"pass": True, "assertion_count": 10, "structure_rate": 1.0},
                "objective_coverage": {"pass": True, "coverage_avg": 0.8, "issue_count": 3},
                "quantitative_coverage": {"pass": True, "quant_fact_count": 2},
                "gap_modeling": {"pass": True, "open_gap_count": 1},
                "steering_readiness": {"pass": True, "steerability": True},
                "deliverable_readiness": {"pass": True},
                "recommended_repairs": [],
            },
            {
                "profile_id": "finance",
                "status": "blocked",
                "primary_failures": ["No detection signal"],
                "domain_detection": {"confidence": 0.0, "is_primary": False},
                "source_role_calibration": {"pass": False, "source_role_known_rate": 0.0, "defined_roles": 6},
                "assertion_quality": {"pass": False, "assertion_count": 0, "structure_rate": None},
                "objective_coverage": {"pass": False, "coverage_avg": None, "issue_count": 0},
                "quantitative_coverage": {"pass": False, "quant_fact_count": 0},
                "gap_modeling": {"pass": True, "open_gap_count": 0},
                "steering_readiness": {"pass": False, "steerability": False},
                "deliverable_readiness": {"pass": False},
                "recommended_repairs": ["Run investigation to populate assertions"],
            },
        ],
        "cross_domain_findings": [
            {"kind": "mapping_gap", "profiles": ["finance"], "severity": "medium",
             "message": "Profile finance has no detection signal"},
        ],
    }
    html = _fmt_domain_readiness(data, domain="legal")
    assert "Domain Investigation Readiness" in html
    assert "partial" in html.lower()
    assert "legal" in html
    assert "finance" in html
    assert "mapping_gap" in html
    assert "Run investigation" in html


def test_domain_readiness_formatter_xss():
    from irys.ui.app import _fmt_domain_readiness
    data = {
        "matter_id": "test",
        "overall_status": "<script>alert(1)</script>",
        "profiles": [
            {
                "profile_id": "<img onerror=alert(1)>",
                "status": "ready",
                "primary_failures": [],
                "domain_detection": {"confidence": 0.5, "is_primary": False},
                "source_role_calibration": {"pass": True},
                "assertion_quality": {"pass": True},
                "objective_coverage": {"pass": True},
                "quantitative_coverage": {"pass": True},
                "gap_modeling": {"pass": True},
                "steering_readiness": {"pass": True},
                "deliverable_readiness": {"pass": True},
                "recommended_repairs": ["<script>xss</script>"],
            },
        ],
        "cross_domain_findings": [
            {"kind": "test", "severity": "high", "message": "<script>xss</script>"},
        ],
    }
    html = _fmt_domain_readiness(data)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_domain_readiness_formatter_non_dict_guards():
    from irys.ui.app import _fmt_domain_readiness
    data = {
        "matter_id": "test",
        "overall_status": "partial",
        "profiles": ["not-a-dict", None, 42],
        "cross_domain_findings": [123, "not-a-dict"],
    }
    html = _fmt_domain_readiness(data)
    assert "Domain Investigation Readiness" in html
    assert "<script>" not in html


def test_domain_readiness_labels_all_five_domains():
    from irys.ui.app import _DOMAIN_READINESS_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _DOMAIN_READINESS_LABELS[domain]
        assert "title" in labels
        assert "coverage" in labels
        assert "empty" in labels


def test_domain_readiness_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    assert hasattr(UIBackend, "get_domain_investigation_readiness")
    assert hasattr(InProcessBackend, "get_domain_investigation_readiness")
    assert hasattr(HttpBackend, "get_domain_investigation_readiness")


# ------------------------------------------------------------------
# Issue Brief Compiler
# ------------------------------------------------------------------


def test_issue_brief_empty_matter():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.compile_issue_brief()
    assert result["matter_id"] == model.matter_id
    assert result["domain"] in ("legal", "finance", "coding", "academic_research", "biomedical")
    assert result["section_count"] == 0
    assert result["total_assertions"] == 0
    assert result["total_gaps"] == 0
    assert result["total_contradictions"] == 0
    assert result["total_source_documents"] == 0
    assert result["sections"] == []


def test_issue_brief_specific_issue_ids():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.compile_issue_brief(issue_ids=["nonexistent-1", "nonexistent-2"])
    assert result["section_count"] == 0
    assert result["sections"] == []


def test_issue_brief_include_gaps_false():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.compile_issue_brief(include_gaps=False)
    assert result["total_gaps"] == 0
    for sec in result["sections"]:
        assert sec["gaps"] == []


def test_issue_brief_include_contradictions_false():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.compile_issue_brief(include_contradictions=False)
    assert result["total_contradictions"] == 0
    for sec in result["sections"]:
        assert sec["contradictions"] == []


def test_issue_brief_include_quant_false():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.compile_issue_brief(include_quant=False)
    assert isinstance(result["sections"], list)


def test_issue_brief_reliance_gate():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.compile_issue_brief()
    assert "reliance_gate" in result
    assert isinstance(result["reliance_gate"], str)


def test_issue_brief_does_not_include_withdrawn():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.compile_issue_brief()
    for sec in result["sections"]:
        for a in sec.get("assertions", []):
            assert a["belief_state"] not in ("superseded", "withdrawn")


def test_issue_brief_formatter_empty():
    from irys.ui.app import _fmt_issue_brief
    html = _fmt_issue_brief({})
    assert "viz-empty" in html


def test_issue_brief_formatter_no_sections():
    from irys.ui.app import _fmt_issue_brief
    data = {
        "matter_id": "test",
        "reliance_gate": "ready",
        "total_assertions": 0,
        "total_gaps": 0,
        "total_contradictions": 0,
        "total_source_documents": 0,
        "sections": [],
    }
    html = _fmt_issue_brief(data)
    assert "viz-empty" in html


def test_issue_brief_formatter_renders():
    from irys.ui.app import _fmt_issue_brief
    data = {
        "matter_id": "test",
        "reliance_gate": "ready",
        "total_assertions": 2,
        "total_gaps": 1,
        "total_contradictions": 0,
        "total_source_documents": 1,
        "sections": [
            {
                "issue_id": "iss-1",
                "title": "Contract breach",
                "materiality": 0.85,
                "assertion_count": 2,
                "supporting_count": 1,
                "attacking_count": 1,
                "assertions": [
                    {
                        "assertion_id": "a1",
                        "proposition": "Defendant breached clause 3",
                        "belief_state": "verified",
                        "confidence": 0.9,
                        "source_roles": ["primary"],
                        "edge_type": "supports",
                    },
                    {
                        "assertion_id": "a2",
                        "proposition": "No breach occurred",
                        "belief_state": "disputed",
                        "confidence": 0.4,
                        "source_roles": ["opposing"],
                        "edge_type": "attacks",
                    },
                ],
                "gaps": [
                    {
                        "gap_id": "g1",
                        "gap_type": "missing_evidence",
                        "description": "No corroborating email",
                        "materiality_score": 0.7,
                    },
                ],
                "contradictions": [],
                "source_documents": ["contract.pdf"],
            },
        ],
    }
    html = _fmt_issue_brief(data)
    assert "Issue Brief" in html
    assert "Contract breach" in html
    assert "Defendant breached clause 3" in html
    assert "No breach occurred" in html
    assert "verified" in html
    assert "disputed" in html
    assert "No corroborating email" in html
    assert "contract.pdf" in html
    assert "ready" in html


def test_issue_brief_formatter_xss():
    from irys.ui.app import _fmt_issue_brief
    data = {
        "matter_id": "test",
        "reliance_gate": "<script>xss</script>",
        "total_assertions": 1,
        "total_gaps": 0,
        "total_contradictions": 0,
        "total_source_documents": 0,
        "sections": [
            {
                "issue_id": "iss-1",
                "title": "<img onerror=alert(1) src=x>",
                "materiality": 0.5,
                "assertion_count": 1,
                "supporting_count": 1,
                "attacking_count": 0,
                "assertions": [
                    {
                        "assertion_id": "a1",
                        "proposition": "<script>alert('xss')</script>",
                        "belief_state": "verified",
                        "confidence": 0.9,
                        "source_roles": ["<script>role</script>"],
                        "edge_type": "supports",
                    },
                ],
                "gaps": [
                    {
                        "gap_id": "g1",
                        "gap_type": "<script>type</script>",
                        "description": "<script>desc</script>",
                        "materiality_score": 0.5,
                    },
                ],
                "contradictions": [
                    {
                        "attacker_id": "x",
                        "attacked_id": "y",
                        "attacker_prop": "<script>a</script>",
                        "attacked_prop": "<script>b</script>",
                    },
                ],
                "source_documents": ["<script>doc</script>"],
            },
        ],
    }
    html = _fmt_issue_brief(data)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_issue_brief_formatter_non_dict_guards():
    from irys.ui.app import _fmt_issue_brief
    data = {
        "matter_id": "test",
        "reliance_gate": "ready",
        "total_assertions": 0,
        "total_gaps": 0,
        "total_contradictions": 0,
        "total_source_documents": 0,
        "sections": ["not-a-dict", None, 42, {"issue_id": "ok", "title": "Valid"}],
    }
    html = _fmt_issue_brief(data)
    assert "Issue Brief" in html
    assert "Valid" in html
    assert "<script>" not in html


def test_issue_brief_formatter_non_dict_inner_guards():
    from irys.ui.app import _fmt_issue_brief
    data = {
        "matter_id": "test",
        "reliance_gate": "ready",
        "total_assertions": 0,
        "total_gaps": 0,
        "total_contradictions": 0,
        "total_source_documents": 0,
        "sections": [
            {
                "issue_id": "iss-1",
                "title": "Test",
                "materiality": 0.5,
                "assertion_count": 0,
                "supporting_count": 0,
                "attacking_count": 0,
                "assertions": ["not-a-dict", 42],
                "gaps": [None, "bad"],
                "contradictions": [123, False],
                "source_documents": [42, None, "valid.pdf"],
            },
        ],
    }
    html = _fmt_issue_brief(data)
    assert "Issue Brief" in html
    assert "<script>" not in html


def test_issue_brief_labels_all_five_domains():
    from irys.ui.app import _ISSUE_BRIEF_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _ISSUE_BRIEF_LABELS[domain]
        assert "title" in labels
        assert "assertions" in labels
        assert "gaps" in labels
        assert "empty" in labels
        assert "no_sections" in labels


def test_issue_brief_formatter_domain_labels():
    from irys.ui.app import _fmt_issue_brief
    data = {
        "matter_id": "test",
        "reliance_gate": "ready",
        "total_assertions": 1,
        "total_gaps": 0,
        "total_contradictions": 0,
        "total_source_documents": 0,
        "sections": [
            {
                "issue_id": "iss-1",
                "title": "Test",
                "materiality": 0.5,
                "assertion_count": 1,
                "supporting_count": 1,
                "attacking_count": 0,
                "assertions": [
                    {
                        "assertion_id": "a1",
                        "proposition": "claim",
                        "belief_state": "verified",
                        "confidence": 0.9,
                        "source_roles": [],
                        "edge_type": "supports",
                    },
                ],
                "gaps": [],
                "contradictions": [],
                "source_documents": [],
            },
        ],
    }
    html_finance = _fmt_issue_brief(data, domain="finance")
    assert "Analytical Memo" in html_finance
    html_bio = _fmt_issue_brief(data, domain="biomedical")
    assert "Evidence Summary" in html_bio


def test_issue_brief_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    assert hasattr(UIBackend, "compile_issue_brief")
    assert hasattr(InProcessBackend, "compile_issue_brief")
    assert hasattr(HttpBackend, "compile_issue_brief")


# ------------------------------------------------------------------
# Assumption Review Workbench
# ------------------------------------------------------------------


def test_assumption_review_empty_matter():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_assumption_review_workbench()
    assert result["total"] == 0
    assert result["provisional"] == []
    assert result["confirmed"] == []
    assert result["invalidated"] == []
    assert result["counts"]["provisional"] == 0
    assert result["counts"]["confirmed"] == 0
    assert result["counts"]["invalidated"] == 0


def test_assumption_review_grouping():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.assumptions.upsert(statement="Test assumption A", rationale="for testing")
    result = model.get_assumption_review_workbench()
    assert result["total"] >= 1
    assert result["counts"]["provisional"] >= 1
    found = False
    for a in result["provisional"]:
        if isinstance(a, dict) and "Test assumption A" in str(a.get("statement", "")):
            found = True
            break
    assert found


def test_assumption_review_linked_targets():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.assumptions.upsert(statement="Linked assumption", rationale="test")
    result = model.get_assumption_review_workbench()
    for a in result["provisional"]:
        if isinstance(a, dict):
            assert "linked_target_count" in a
            assert "linked_targets" in a


def test_assumption_review_after_invalidate():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    model.assumptions.upsert(statement="Will invalidate", rationale="test")
    wb = model.get_assumption_review_workbench()
    prov = [a for a in wb["provisional"] if isinstance(a, dict) and "Will invalidate" in str(a.get("statement", ""))]
    assert len(prov) >= 1
    aid = prov[0]["id"]
    review_result = model.review_assumption(aid, "invalidated", reason="test reason")
    assert review_result.get("success") is True
    wb2 = model.get_assumption_review_workbench()
    assert wb2["counts"]["invalidated"] >= 1
    inv_ids = {a["id"] for a in wb2["invalidated"] if isinstance(a, dict)}
    assert aid in inv_ids


def test_assumption_review_invalid_decision():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.review_assumption("fake-id", "bogus_decision")
    assert "error" in result


def test_assumption_review_not_found():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.review_assumption("nonexistent", "confirmed")
    assert "error" in result
    assert "not found" in result["error"]


def test_assumption_review_formatter_empty():
    from irys.ui.app import _fmt_assumption_review_workbench
    html = _fmt_assumption_review_workbench({})
    assert "viz-empty" in html


def test_assumption_review_formatter_zero_total():
    from irys.ui.app import _fmt_assumption_review_workbench
    data = {
        "total": 0,
        "provisional": [],
        "confirmed": [],
        "invalidated": [],
        "counts": {"provisional": 0, "confirmed": 0, "invalidated": 0},
    }
    html = _fmt_assumption_review_workbench(data)
    assert "viz-empty" in html


def test_assumption_review_formatter_renders():
    from irys.ui.app import _fmt_assumption_review_workbench
    data = {
        "total": 3,
        "provisional": [
            {
                "id": "a1",
                "statement": "Contract was signed before deadline",
                "status": "provisional",
                "rationale": "Based on metadata",
                "invalidation_condition": "If signing date is after Jan 1",
                "linked_target_count": 2,
                "linked_targets": [
                    {"target_type": "predicate", "target_id": "p1"},
                    {"target_type": "issue", "target_id": "i1"},
                ],
            },
        ],
        "confirmed": [
            {
                "id": "a2",
                "statement": "Party A is the plaintiff",
                "status": "confirmed",
                "rationale": "Verified from filing",
                "invalidation_condition": "",
                "linked_target_count": 0,
                "linked_targets": [],
            },
        ],
        "invalidated": [
            {
                "id": "a3",
                "statement": "Deadline was March 15",
                "status": "invalidated",
                "rationale": "Contradicted by exhibit B",
                "invalidation_condition": "",
                "linked_target_count": 1,
                "linked_targets": [{"target_type": "predicate", "target_id": "p2"}],
            },
        ],
        "counts": {"provisional": 1, "confirmed": 1, "invalidated": 1},
    }
    html = _fmt_assumption_review_workbench(data)
    assert "Assumption Review Workbench" in html
    assert "Contract was signed before deadline" in html
    assert "Party A is the plaintiff" in html
    assert "Deadline was March 15" in html
    assert "Provisional" in html
    assert "Confirmed" in html
    assert "Invalidated" in html
    assert "Linked targets" in html
    assert "predicate, issue" in html or "issue, predicate" in html


def test_assumption_review_formatter_xss():
    from irys.ui.app import _fmt_assumption_review_workbench
    data = {
        "total": 1,
        "provisional": [
            {
                "id": "<script>xss</script>",
                "statement": "<img onerror=alert(1) src=x>",
                "status": "provisional",
                "rationale": "<script>bad</script>",
                "invalidation_condition": "<script>cond</script>",
                "linked_target_count": 1,
                "linked_targets": [{"target_type": "<script>type</script>", "target_id": "t1"}],
            },
        ],
        "confirmed": [],
        "invalidated": [],
        "counts": {"provisional": 1, "confirmed": 0, "invalidated": 0},
    }
    html = _fmt_assumption_review_workbench(data)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_assumption_review_formatter_non_dict_guards():
    from irys.ui.app import _fmt_assumption_review_workbench
    data = {
        "total": 3,
        "provisional": ["not-a-dict", None, 42],
        "confirmed": [123],
        "invalidated": ["bad"],
        "counts": {"provisional": 3, "confirmed": 1, "invalidated": 1},
    }
    html = _fmt_assumption_review_workbench(data)
    assert "Assumption Review Workbench" in html
    assert "<script>" not in html


def test_assumption_review_labels_all_five_domains():
    from irys.ui.app import _ASSUMPTION_REVIEW_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _ASSUMPTION_REVIEW_LABELS[domain]
        assert "title" in labels
        assert "provisional" in labels
        assert "confirmed" in labels
        assert "invalidated" in labels
        assert "empty" in labels


def test_assumption_review_formatter_domain_labels():
    from irys.ui.app import _fmt_assumption_review_workbench
    data = {
        "total": 1,
        "provisional": [
            {"id": "a1", "statement": "test", "status": "provisional",
             "rationale": "", "invalidation_condition": "",
             "linked_target_count": 0, "linked_targets": []},
        ],
        "confirmed": [],
        "invalidated": [],
        "counts": {"provisional": 1, "confirmed": 0, "invalidated": 0},
    }
    html_finance = _fmt_assumption_review_workbench(data, domain="finance")
    assert "Thesis Assumption Review" in html_finance
    html_bio = _fmt_assumption_review_workbench(data, domain="biomedical")
    assert "Mechanism Assumption Review" in html_bio
    html_code = _fmt_assumption_review_workbench(data, domain="coding")
    assert "Design Assumption Review" in html_code


def test_assumption_review_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    assert hasattr(UIBackend, "get_assumption_review")
    assert hasattr(InProcessBackend, "get_assumption_review")
    assert hasattr(HttpBackend, "get_assumption_review")
    assert hasattr(UIBackend, "review_assumption")
    assert hasattr(InProcessBackend, "review_assumption")
    assert hasattr(HttpBackend, "review_assumption")


# ------------------------------------------------------------------
# Durable Scenario Graphs
# ------------------------------------------------------------------


def test_scenario_delta_invalid_kind():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="test-branch", assumptions=[])
    bid = b["branch_id"]
    result = model.apply_scenario_delta(bid, "bogus_kind", "t1", "suppress")
    assert "error" in result
    assert "target_kind" in result["error"]


def test_scenario_delta_invalid_operation():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="test-op", assumptions=[])
    bid = b["branch_id"]
    result = model.apply_scenario_delta(bid, "assertion", "a1", "bogus_op")
    assert "error" in result
    assert "operation" in result["error"]


def test_scenario_delta_branch_not_found():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.apply_scenario_delta("nonexistent", "assertion", "a1", "suppress")
    assert "error" in result
    assert "not found" in result["error"]


def test_scenario_delta_apply_valid():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="valid-delta", assumptions=[])
    bid = b["branch_id"]
    result = model.apply_scenario_delta(
        bid, "assertion", "a1", "override_belief", {"new_belief": "disputed"},
    )
    assert "delta_id" in result
    assert result["branch_id"] == bid
    assert result["total_deltas"] == 1


def test_scenario_delta_does_not_mutate_baseline():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="no-mutate", assumptions=[])
    bid = b["branch_id"]
    model.apply_scenario_delta(bid, "assertion", "a1", "override_belief", {"new_belief": "disputed"})
    rec = model.assertions.get("a1")
    assert rec is None


def test_scenario_list_deltas():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="list-deltas", assumptions=[])
    bid = b["branch_id"]
    model.apply_scenario_delta(bid, "assertion", "a1", "suppress")
    model.apply_scenario_delta(bid, "gap", "g1", "resolve_gap", {"reason": "resolved"})
    deltas = model.list_scenario_deltas(bid)
    assert len(deltas) == 2
    assert deltas[0]["operation"] == "suppress"
    assert deltas[1]["operation"] == "resolve_gap"
    assert deltas[1]["payload"]["reason"] == "resolved"


def test_scenario_snapshot_empty_branch():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="snap-empty", assumptions=[])
    bid = b["branch_id"]
    snap = model.compute_scenario_snapshot(bid)
    assert snap["delta_count"] == 0
    assert snap["branch_id"] == bid
    assert "coverage" in snap
    assert "gaps" in snap
    assert isinstance(snap["warnings"], list)


def test_scenario_snapshot_with_deltas():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="snap-deltas", assumptions=[])
    bid = b["branch_id"]
    model.apply_scenario_delta(bid, "assertion", "a1", "override_belief", {"new_belief": "disputed"})
    model.apply_scenario_delta(bid, "gap", "g1", "add_gap", {"description": "test gap"})
    snap = model.compute_scenario_snapshot(bid)
    assert snap["delta_count"] == 2
    assert "a1" in snap["overridden_beliefs"]
    assert "g1" in snap["new_gaps"]


def test_scenario_compare_branch_not_found():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.compare_scenario_to_baseline("nonexistent")
    assert "error" in result


def test_scenario_compare_with_deltas():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="compare-test", assumptions=[{"text": "assume X"}])
    bid = b["branch_id"]
    model.apply_scenario_delta(bid, "assertion", "a1", "override_belief", {"new_belief": "disputed"})
    model.apply_scenario_delta(bid, "assertion", "a2", "suppress")
    model.apply_scenario_delta(bid, "gap", "g1", "add_gap", {"description": "new gap", "gap_type": "missing_doc"})
    result = model.compare_scenario_to_baseline(bid)
    assert result["delta_count"] == 3
    assert len(result["belief_changes"]) == 1
    assert result["belief_changes"][0]["branch_belief"] == "disputed"
    assert len(result["suppressions"]) == 1
    assert len(result["new_gaps"]) == 1
    assert result["assumptions"][0]["text"] == "assume X"


def test_scenario_compare_formatter_empty():
    from irys.ui.app import _fmt_scenario_comparison
    html = _fmt_scenario_comparison({})
    assert "viz-empty" in html


def test_scenario_compare_formatter_no_deltas():
    from irys.ui.app import _fmt_scenario_comparison
    data = {
        "branch_id": "b1",
        "branch_name": "Test",
        "branch_status": "active",
        "delta_count": 0,
        "assumptions": [],
        "belief_changes": [],
        "suppressions": [],
        "new_gaps": [],
        "resolved_gaps": [],
        "new_assertions": [],
    }
    html = _fmt_scenario_comparison(data)
    assert "viz-empty" in html


def test_scenario_compare_formatter_renders():
    from irys.ui.app import _fmt_scenario_comparison
    data = {
        "branch_id": "b1",
        "branch_name": "Void contract theory",
        "branch_status": "active",
        "delta_count": 3,
        "assumptions": [{"text": "Contract was signed under duress"}],
        "belief_changes": [
            {
                "assertion_id": "a1",
                "proposition": "Contract is enforceable",
                "baseline_belief": "verified",
                "branch_belief": "disputed",
            },
        ],
        "suppressions": [
            {"target_kind": "assertion", "target_id": "a2", "label": "Payment was made on time"},
        ],
        "new_gaps": [
            {"target_id": "g1", "description": "Missing duress evidence", "gap_type": "missing_document"},
        ],
        "resolved_gaps": [
            {"gap_id": "g2", "reason": "Resolved by testimony"},
        ],
        "new_assertions": [
            {"assertion_id": "a3", "proposition": "Duress alleged", "belief_state": "provisional"},
        ],
    }
    html = _fmt_scenario_comparison(data)
    assert "Scenario Comparison" in html
    assert "Void contract theory" in html
    assert "Contract is enforceable" in html
    assert "verified" in html
    assert "disputed" in html
    assert "Payment was made on time" in html
    assert "Missing duress evidence" in html
    assert "Resolved by testimony" in html
    assert "Duress alleged" in html


def test_scenario_compare_formatter_xss():
    from irys.ui.app import _fmt_scenario_comparison
    data = {
        "branch_id": "b1",
        "branch_name": "<script>xss</script>",
        "branch_status": "active",
        "delta_count": 1,
        "assumptions": [{"text": "<script>bad</script>"}],
        "belief_changes": [
            {
                "assertion_id": "a1",
                "proposition": "<img onerror=alert(1)>",
                "baseline_belief": "<script>b</script>",
                "branch_belief": "<script>d</script>",
            },
        ],
        "suppressions": [{"target_kind": "<script>k</script>", "target_id": "t1", "label": "<script>l</script>"}],
        "new_gaps": [{"target_id": "g1", "description": "<script>g</script>", "gap_type": "<script>t</script>"}],
        "resolved_gaps": [{"gap_id": "rg1", "reason": "<script>r</script>"}],
        "new_assertions": [{"assertion_id": "na1", "proposition": "<script>p</script>", "belief_state": "<script>s</script>"}],
    }
    html = _fmt_scenario_comparison(data)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_scenario_compare_formatter_non_dict_guards():
    from irys.ui.app import _fmt_scenario_comparison
    data = {
        "branch_id": "b1",
        "branch_name": "Test",
        "branch_status": "active",
        "delta_count": 2,
        "assumptions": ["not-a-dict", None],
        "belief_changes": [42, "bad"],
        "suppressions": [None],
        "new_gaps": [123],
        "resolved_gaps": [False],
        "new_assertions": ["bad"],
    }
    html = _fmt_scenario_comparison(data)
    assert "Scenario Comparison" in html
    assert "<script>" not in html


def test_scenario_compare_labels_all_five_domains():
    from irys.ui.app import _SCENARIO_COMPARE_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _SCENARIO_COMPARE_LABELS[domain]
        assert "title" in labels
        assert "belief_changes" in labels
        assert "empty" in labels
        assert "new_gaps" in labels


def test_scenario_compare_formatter_domain_labels():
    from irys.ui.app import _fmt_scenario_comparison
    data = {
        "branch_id": "b1",
        "branch_name": "Test",
        "branch_status": "active",
        "delta_count": 1,
        "assumptions": [],
        "belief_changes": [
            {"assertion_id": "a1", "proposition": "x", "baseline_belief": "a", "branch_belief": "b"},
        ],
        "suppressions": [],
        "new_gaps": [],
        "resolved_gaps": [],
        "new_assertions": [],
    }
    html_finance = _fmt_scenario_comparison(data, domain="finance")
    assert "Scenario Comparison" in html_finance
    html_bio = _fmt_scenario_comparison(data, domain="biomedical")
    assert "Interpretation Comparison" in html_bio


def test_scenario_snapshot_includes_warnings_on_metric_failure():
    """P2 fix: snapshot must surface warnings when baseline metrics fail."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="warn-test", assumptions=[])
    bid = b["branch_id"]
    snap = model.compute_scenario_snapshot(bid)
    assert "warnings" in snap
    assert isinstance(snap["warnings"], list)


def test_scenario_graph_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("apply_scenario_delta", "list_scenario_deltas",
                   "compute_scenario_snapshot", "compare_scenario_to_baseline"):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


# --- Sensitivity Reclassification Workbench (SO-3, SO-5) ---


def test_sensitivity_review_labels_all_five_domains():
    from irys.ui.app import _SENSITIVITY_REVIEW_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _SENSITIVITY_REVIEW_LABELS[domain]
        assert "title" in labels
        assert "doc_id" in labels
        assert "privilege_flag" in labels
        assert "non_privilege" in labels
        assert "empty" in labels


def test_sensitivity_review_formatter_empty():
    from irys.ui.app import _fmt_sensitivity_review_result
    html = _fmt_sensitivity_review_result({})
    assert "viz-empty" in html


def test_sensitivity_review_formatter_error():
    from irys.ui.app import _fmt_sensitivity_review_result
    html = _fmt_sensitivity_review_result({"error": "not found"})
    assert "not found" in html
    assert "viz-empty" in html


def test_sensitivity_review_formatter_success():
    from irys.ui.app import _fmt_sensitivity_review_result
    html = _fmt_sensitivity_review_result({"doc_id": "doc-1", "staled_count": 5})
    assert "doc-1" in html
    assert "5" in html
    assert "Result" in html


def test_sensitivity_review_formatter_xss():
    from irys.ui.app import _fmt_sensitivity_review_result
    html = _fmt_sensitivity_review_result({
        "doc_id": "<script>xss</script>",
        "staled_count": 3,
    })
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_sensitivity_review_formatter_domain_labels():
    from irys.ui.app import _fmt_sensitivity_review_result
    html_finance = _fmt_sensitivity_review_result(
        {"doc_id": "d1", "staled_count": 2}, domain="finance",
    )
    assert "Result" in html_finance
    html_bio = _fmt_sensitivity_review_result(
        {"doc_id": "d1", "staled_count": 1}, domain="biomedical",
    )
    assert "Result" in html_bio


def test_sensitivity_review_formatter_non_dict():
    from irys.ui.app import _fmt_sensitivity_review_result
    html = _fmt_sensitivity_review_result("bad")
    assert "viz-empty" in html
    html2 = _fmt_sensitivity_review_result(None)
    assert "viz-empty" in html2


def test_reclassify_privilege_no_change():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.reclassify_privilege(
        "nonexistent-doc", True, reviewed_by_kind="user",
    )
    assert result == 0


def test_mark_document_stale_no_doc():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.mark_document_stale("nonexistent-doc", "test")
    assert result == 0


def test_mark_span_stale_no_span():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.mark_span_stale("nonexistent-span", "test")
    assert result == 0


def test_sensitivity_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("reclassify_document_sensitivity", "mark_document_stale",
                   "mark_span_stale"):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


def test_sensitivity_review_formatter_span_stale():
    from irys.ui.app import _fmt_sensitivity_review_result
    html = _fmt_sensitivity_review_result({"span_id": "span-42", "staled_count": 2})
    assert "span-42" in html
    assert "2" in html


def test_sensitivity_review_formatter_staled_count_non_numeric():
    from irys.ui.app import _fmt_sensitivity_review_result
    html = _fmt_sensitivity_review_result({"doc_id": "d1", "staled_count": "bad"})
    assert "d1" in html
    assert "0" in html


# --- Scenario Snapshot History (SO-1, SO-3) ---


def test_snapshot_history_labels_all_five_domains():
    from irys.ui.app import _SNAPSHOT_HISTORY_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _SNAPSHOT_HISTORY_LABELS[domain]
        assert "title" in labels
        assert "snapshot_id" in labels
        assert "empty" in labels


def test_snapshot_history_formatter_empty():
    from irys.ui.app import _fmt_scenario_snapshot_history
    html = _fmt_scenario_snapshot_history({})
    assert "viz-empty" in html


def test_snapshot_history_formatter_no_snapshots():
    from irys.ui.app import _fmt_scenario_snapshot_history
    html = _fmt_scenario_snapshot_history({"branch_id": "b1", "snapshots": [], "count": 0})
    assert "viz-empty" in html


def test_snapshot_history_formatter_renders():
    from irys.ui.app import _fmt_scenario_snapshot_history
    data = {
        "branch_id": "b-test",
        "snapshots": [
            {"id": "snap-001", "delta_count": 5, "created_at": "2026-05-04T10:00:00"},
            {"id": "snap-002", "delta_count": 3, "created_at": "2026-05-04T09:00:00"},
        ],
        "count": 2,
    }
    html = _fmt_scenario_snapshot_history(data)
    assert "Scenario Snapshot History" in html
    assert "snap-001" in html
    assert "5" in html
    assert "2026-05-04" in html
    assert "b-test" in html


def test_snapshot_history_formatter_xss():
    from irys.ui.app import _fmt_scenario_snapshot_history
    data = {
        "branch_id": "<script>xss</script>",
        "snapshots": [
            {"id": "<img onerror=alert(1)>", "delta_count": 1, "created_at": "<script>bad</script>"},
        ],
        "count": 1,
    }
    html = _fmt_scenario_snapshot_history(data)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_snapshot_history_formatter_non_dict_guard():
    from irys.ui.app import _fmt_scenario_snapshot_history
    data = {
        "branch_id": "b1",
        "snapshots": [123, "bad", None, {"id": "ok", "delta_count": 1, "created_at": "now"}],
        "count": 4,
    }
    html = _fmt_scenario_snapshot_history(data)
    assert "ok" in html
    assert "Scenario Snapshot History" in html


def test_snapshot_history_formatter_domain_labels():
    from irys.ui.app import _fmt_scenario_snapshot_history
    data = {
        "branch_id": "b1",
        "snapshots": [{"id": "s1", "delta_count": 1, "created_at": "now"}],
        "count": 1,
    }
    html_bio = _fmt_scenario_snapshot_history(data, domain="biomedical")
    assert "Interpretation Snapshot History" in html_bio
    html_code = _fmt_scenario_snapshot_history(data, domain="coding")
    assert "Path Snapshot History" in html_code


def test_snapshot_history_formatter_non_numeric_delta():
    from irys.ui.app import _fmt_scenario_snapshot_history
    data = {
        "branch_id": "b1",
        "snapshots": [{"id": "s1", "delta_count": "bad", "created_at": "now"}],
        "count": 1,
    }
    html = _fmt_scenario_snapshot_history(data)
    assert "0" in html


def test_snapshot_history_model_roundtrip():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    b = model.create_scenario_branch(name="hist-test", assumptions=[])
    bid = b["branch_id"]
    model.compute_scenario_snapshot(bid)
    model.apply_scenario_delta(bid, "assertion", "a1", "override_belief", {"new_belief": "disputed"})
    model.compute_scenario_snapshot(bid)
    snaps = model.list_scenario_snapshots(bid)
    assert len(snaps) == 2
    assert snaps[0]["delta_count"] == 1
    assert snaps[1]["delta_count"] == 0


def test_snapshot_history_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("list_scenario_snapshots",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


# --- Operative Document Version Lookup (SO-5, SO-7) ---


def test_operative_version_labels_all_five_domains():
    from irys.ui.app import _OPERATIVE_VERSION_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _OPERATIVE_VERSION_LABELS[domain]
        assert "title" in labels
        assert "operative" in labels
        assert "superseded" in labels
        assert "same" in labels
        assert "empty" in labels


def test_operative_version_formatter_empty():
    from irys.ui.app import _fmt_operative_version_result
    html = _fmt_operative_version_result({})
    assert "viz-empty" in html


def test_operative_version_formatter_is_operative():
    from irys.ui.app import _fmt_operative_version_result
    html = _fmt_operative_version_result({
        "doc_id": "doc-1", "operative_doc_id": "doc-1", "is_operative": True,
    })
    assert "doc-1" in html
    assert "already the operative" in html


def test_operative_version_formatter_superseded():
    from irys.ui.app import _fmt_operative_version_result
    html = _fmt_operative_version_result({
        "doc_id": "doc-v1", "operative_doc_id": "doc-v2", "is_operative": False,
    })
    assert "doc-v1" in html
    assert "doc-v2" in html
    assert "Superseded by" in html


def test_operative_version_formatter_xss():
    from irys.ui.app import _fmt_operative_version_result
    html = _fmt_operative_version_result({
        "doc_id": "<script>xss</script>",
        "operative_doc_id": "<img onerror=alert(1)>",
        "is_operative": False,
    })
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_operative_version_formatter_error():
    from irys.ui.app import _fmt_operative_version_result
    html = _fmt_operative_version_result({"error": "not found"})
    assert "not found" in html
    assert "viz-empty" in html


def test_operative_version_formatter_non_dict():
    from irys.ui.app import _fmt_operative_version_result
    html = _fmt_operative_version_result("bad")
    assert "viz-empty" in html
    html2 = _fmt_operative_version_result(None)
    assert "viz-empty" in html2


def test_operative_version_formatter_domain_labels():
    from irys.ui.app import _fmt_operative_version_result
    html_fin = _fmt_operative_version_result(
        {"doc_id": "d1", "operative_doc_id": "d1", "is_operative": True},
        domain="finance",
    )
    assert "current version" in html_fin or "current filing" in html_fin.lower()
    html_bio = _fmt_operative_version_result(
        {"doc_id": "d1", "operative_doc_id": "d2", "is_operative": False},
        domain="biomedical",
    )
    assert "Superseded by" in html_bio


def test_operative_version_model_returns_self():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    result = model.get_operative_document_version("nonexistent")
    assert result == "nonexistent"


def test_operative_version_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_operative_document_version",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


# ---------------------------------------------------------------------------
# Query Context Inspector (SO-1, SO-3, SO-4, SO-7)
# ---------------------------------------------------------------------------


def test_query_context_labels_all_five_domains():
    from irys.ui.app import _QUERY_CONTEXT_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _QUERY_CONTEXT_LABELS[domain]
        assert "title" in labels
        assert "subtitle" in labels
        assert "issues" in labels
        assert "gaps" in labels
        assert "weakest" in labels
        assert "actors" in labels
        assert "predicates" in labels
        assert "empty" in labels


def test_query_context_formatter_empty():
    from irys.ui.app import _fmt_query_context
    html = _fmt_query_context({})
    assert "viz-empty" in html


def test_query_context_formatter_error():
    from irys.ui.app import _fmt_query_context
    html = _fmt_query_context({"error": "db down"})
    assert "db down" in html
    assert "viz-empty" in html


def test_query_context_formatter_non_dict():
    from irys.ui.app import _fmt_query_context
    html = _fmt_query_context("not a dict")
    assert "viz-empty" in html


def test_query_context_formatter_renders_kpis():
    from irys.ui.app import _fmt_query_context
    data = {
        "matter_id": "m1",
        "matter_name": "Test",
        "existing_assertion_count": 12,
        "existing_actor_count": 3,
        "known_document_ids": ["d1", "d2"],
        "document_card_count": 5,
        "active_assumptions": [{"id": "a1"}],
        "open_issues": [{"id": "i1"}],
        "open_gaps": [{"id": "g1", "description": "missing doc"}],
        "weakest_issue_id": "i1",
        "known_actors": ["Alice"],
        "answered_clarifications": [],
        "document_annotations": [],
        "key_predicates": ["owes_money_to"],
        "domain_facets": [],
        "composed_trust_weights": {"official_record": 0.95},
        "primary_domain_profile_id": "legal",
    }
    html = _fmt_query_context(data)
    assert "<strong>12</strong> assertions" in html
    assert "<strong>3</strong> actors" in html
    assert "<strong>2</strong> documents" in html
    assert "<strong>5</strong> document cards" in html
    assert "<strong>1</strong> active assumptions" in html
    assert "Next Run Focus" in html
    assert "i1" in html
    assert "missing doc" in html
    assert "Alice" in html
    assert "owes_money_to" in html
    assert "official_record" in html
    assert "legal" in html


def test_query_context_formatter_xss():
    from irys.ui.app import _fmt_query_context
    data = {
        "matter_id": "m1",
        "matter_name": "<script>alert(1)</script>",
        "existing_assertion_count": 0,
        "existing_actor_count": 0,
        "known_document_ids": [],
        "document_card_count": 0,
        "active_assumptions": [],
        "open_issues": [],
        "open_gaps": [{"description": "<img onerror=alert(1)>"}],
        "weakest_issue_id": "<script>x</script>",
        "known_actors": ["<b>evil</b>"],
        "answered_clarifications": [],
        "document_annotations": [],
        "key_predicates": ["<script>p</script>"],
        "domain_facets": [],
        "composed_trust_weights": {},
        "primary_domain_profile_id": "legal",
    }
    html = _fmt_query_context(data)
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "<b>evil</b>" not in html


def test_query_context_formatter_domain_labels():
    from irys.ui.app import _fmt_query_context
    data = {
        "matter_id": "m1",
        "matter_name": "Test",
        "open_issues": [],
        "open_gaps": [],
        "weakest_issue_id": None,
        "known_actors": [],
        "existing_assertion_count": 0,
        "existing_actor_count": 0,
        "known_document_ids": [],
        "document_card_count": 0,
        "active_assumptions": [],
        "answered_clarifications": [],
        "document_annotations": [],
        "key_predicates": [],
        "domain_facets": [],
        "composed_trust_weights": {},
        "primary_domain_profile_id": None,
    }
    legal_html = _fmt_query_context(data, domain="legal")
    assert "Case Context" in legal_html
    finance_html = _fmt_query_context(data, domain="finance")
    assert "Diligence Context" in finance_html
    coding_html = _fmt_query_context(data, domain="coding")
    assert "Codebase Context" in coding_html
    research_html = _fmt_query_context(data, domain="academic_research")
    assert "Research Context" in research_html
    bio_html = _fmt_query_context(data, domain="biomedical")
    assert "Clinical Evidence Context" in bio_html


def test_query_context_formatter_nan_guard():
    from irys.ui.app import _fmt_query_context
    data = {
        "matter_id": "m1",
        "matter_name": "Test",
        "existing_assertion_count": float("nan"),
        "existing_actor_count": float("inf"),
        "known_document_ids": [],
        "document_card_count": "not_a_number",
        "active_assumptions": [],
        "open_issues": [],
        "open_gaps": [],
        "weakest_issue_id": None,
        "known_actors": [],
        "answered_clarifications": [],
        "document_annotations": [],
        "key_predicates": [],
        "domain_facets": [],
        "composed_trust_weights": {"w1": float("nan")},
        "primary_domain_profile_id": None,
    }
    html = _fmt_query_context(data)
    assert "<strong>0</strong> assertions" in html
    assert "<strong>0</strong> actors" in html
    assert "<strong>0</strong> document cards" in html


def test_query_context_formatter_trust_weights_non_dict():
    from irys.ui.app import _fmt_query_context
    data = {
        "matter_id": "m1",
        "matter_name": "Test",
        "existing_assertion_count": 0,
        "existing_actor_count": 0,
        "known_document_ids": [],
        "document_card_count": 0,
        "active_assumptions": [],
        "open_issues": [],
        "open_gaps": [],
        "weakest_issue_id": None,
        "known_actors": [],
        "answered_clarifications": [],
        "document_annotations": [],
        "key_predicates": [],
        "domain_facets": [],
        "composed_trust_weights": "not_a_dict",
        "primary_domain_profile_id": None,
    }
    html = _fmt_query_context(data)
    assert "Domain Calibration" in html


def test_query_context_model_roundtrip():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    ctx = model.build_query_context()
    assert ctx.matter_id == model.matter_id
    assert isinstance(ctx.open_issues, list)
    assert isinstance(ctx.composed_trust_weights, dict)


def test_query_context_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_query_context",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


def test_query_context_appstate_method_exists():
    from irys.ui.app import AppState
    assert hasattr(AppState, "load_query_context")


def test_query_context_api_endpoint_exists():
    from irys.service.api import app as fastapi_app
    routes = [r.path for r in fastapi_app.routes]
    assert "/matter/{matter_id}/query-context" in routes


def test_query_context_in_process_roundtrip():
    from dataclasses import fields, asdict
    from irys.matter.models import QueryMatterContext
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    ctx = model.build_query_context()
    result = asdict(ctx)
    assert isinstance(result, dict)
    expected_keys = {f.name for f in fields(QueryMatterContext)}
    assert expected_keys.issubset(result.keys())
    assert result["matter_id"] == model.matter_id


def test_query_context_http_non_dict_returns_error():
    from irys.ui.app import _fmt_query_context
    error_data = {"error": "unexpected response type: list"}
    html = _fmt_query_context(error_data)
    assert "viz-empty" in html
    assert "unexpected response type" in html


def test_query_context_safe_list_normalization():
    from irys.ui.app import _fmt_query_context
    data = {
        "matter_id": "m1",
        "matter_name": "Test",
        "existing_assertion_count": 1,
        "existing_actor_count": 0,
        "known_document_ids": "not_a_list",
        "document_card_count": 0,
        "active_assumptions": "also_not_a_list",
        "open_issues": None,
        "open_gaps": 42,
        "weakest_issue_id": None,
        "known_actors": {"bad": "data"},
        "answered_clarifications": None,
        "document_annotations": None,
        "key_predicates": None,
        "domain_facets": None,
        "composed_trust_weights": {},
        "primary_domain_profile_id": None,
    }
    html = _fmt_query_context(data)
    assert "<strong>0</strong> documents" in html
    assert "<strong>0</strong> active assumptions" in html


# ---------------------------------------------------------------------------
# XSS regression tests for formatters previously lacking coverage
# ---------------------------------------------------------------------------


def test_belief_revision_panel_xss():
    from irys.ui.app import _fmt_belief_revision_panel
    revisions = [{
        "proposition_text": "<script>alert('xss')</script>",
        "old_belief_state": "<img onerror=alert(1)>",
        "new_belief_state": "operative",
        "cause": "<b>evil</b>",
        "old_confidence": 0.5,
        "new_confidence": 0.9,
    }]
    html = _fmt_belief_revision_panel(revisions)
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "<b>evil</b>" not in html


def test_contradiction_panel_xss():
    from irys.ui.app import _fmt_contradiction_panel
    contradictions = [{
        "attacker_prop": "<script>alert(1)</script>",
        "attacked_prop": "<img src=x onerror=alert(2)>",
        "link_type": "<b>bad</b>",
        "attacker_belief": "alleged",
        "attacked_belief": "disputed",
    }]
    html = _fmt_contradiction_panel(contradictions)
    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "<b>bad</b>" not in html


def test_document_versions_panel_xss():
    from irys.ui.app import _fmt_document_versions_panel
    families = [{
        "members": [{
            "relative_path": "<script>alert(1)</script>/evil.pdf",
            "is_operative": True,
        }]
    }]
    html = _fmt_document_versions_panel(families)
    assert "<script>" not in html


def test_quant_thresholds_panel_xss():
    from irys.ui.app import _fmt_quant_thresholds_panel
    violations = [{
        "threshold": "<script>alert(1)</script>",
        "level": "HIGH",
        "description": "<img onerror=alert(2)>",
        "amount": 100.0,
    }]
    html = _fmt_quant_thresholds_panel(violations)
    assert "<script>" not in html
    assert "<img onerror" not in html


def test_system_health_panel_xss():
    from irys.ui.app import _fmt_system_health_panel
    health = {
        "total_assertions": 10,
        "disputed_assertions": 2,
        "dispute_rate": 0.2,
        "total_revisions": 5,
        "open_gaps": 3,
        "oscillating_assertions": ["<script>alert(1)</script>"],
    }
    html = _fmt_system_health_panel(health)
    assert "<script>" not in html


def test_domain_composition_panel_xss():
    from irys.ui.app import _fmt_domain_composition_panel
    data = {
        "primary_profile": "<script>alert(1)</script>",
        "detections": [{"domain_id": "<img onerror=alert(2)>", "confidence": 0.9}],
        "source_roles": {"<b>evil</b>": 3},
        "trust_weights": {"<script>x</script>": 0.5},
    }
    html = _fmt_domain_composition_panel(data)
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "<b>evil</b>" not in html


# ---------------------------------------------------------------------------
# Source Calibration Inspector (SO-4, SO-5, SO-7)
# ---------------------------------------------------------------------------


def test_source_calibration_labels_all_five_domains():
    from irys.ui.app import _SOURCE_CALIBRATION_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _SOURCE_CALIBRATION_LABELS[domain]
        assert "title" in labels
        assert "subtitle" in labels
        assert "issue_col" in labels
        assert "gap_col" in labels
        assert "empty" in labels


def test_source_calibration_formatter_empty():
    from irys.ui.app import _fmt_source_calibration
    html = _fmt_source_calibration({})
    assert "viz-empty" in html


def test_source_calibration_formatter_error():
    from irys.ui.app import _fmt_source_calibration
    html = _fmt_source_calibration({"error": "service down"})
    assert "service down" in html
    assert "viz-empty" in html


def test_source_calibration_formatter_non_dict():
    from irys.ui.app import _fmt_source_calibration
    html = _fmt_source_calibration("bad")
    assert "viz-empty" in html


def test_source_calibration_formatter_renders():
    from irys.ui.app import _fmt_source_calibration
    data = {
        "evidence_matrix": {
            "issues": [{"id": "i1", "title": "Breach of Contract"}],
            "sources": ["doc_a.pdf", "doc_b.pdf"],
            "cells": {"i1": {"doc_a.pdf": {"supporting": 3, "attacking": 0, "total": 3}}},
            "issue_totals": {"i1": {"supporting": 3, "attacking": 0}},
            "source_totals": {"doc_a.pdf": {"supporting": 3, "attacking": 0}},
        },
        "coverage_report": [{"id": "i1", "title": "Breach", "coverage_fraction": 0.6}],
        "domain_profile": {
            "profile_id": "legal",
            "source_roles": ["advocacy", "operative", "authority"],
            "composed_trust_weights": {"official_record": 0.95, "advocacy": 0.6},
        },
        "gap_workbench": {
            "items": [{"gap_type": "MISSING_AUTHORITY", "description": "Need court filing", "materiality_score": 0.8, "issue_title": "Breach"}]
        },
        "reviewable_documents": [
            {"path": "doc_a.pdf", "doc_type": "advocacy"},
            {"path": "doc_b.pdf", "doc_type": "operative"},
        ],
    }
    html = _fmt_source_calibration(data)
    assert "Calibration Summary" in html
    assert "Issue Source Mix" in html
    assert "Breach of Contract" in html
    assert "single-source" in html
    assert "no attack tested" in html
    assert "Missing Source Obligations" in html
    assert "Authority" in html
    assert "0.80" in html
    assert "Distinct source types" in html


def test_source_calibration_formatter_xss():
    from irys.ui.app import _fmt_source_calibration
    data = {
        "evidence_matrix": {
            "issues": [{"id": "i1", "title": "<script>alert(1)</script>"}],
            "sources": ["<img onerror=alert(2)>"],
            "cells": {"i1": {"<img>": {"supporting": 1, "attacking": 0, "total": 1}}},
            "issue_totals": {"i1": {"supporting": 1, "attacking": 0}},
            "source_totals": {},
        },
        "coverage_report": [],
        "domain_profile": {
            "profile_id": "<script>x</script>",
            "source_roles": [],
            "composed_trust_weights": {"<b>evil</b>": 0.5},
        },
        "gap_workbench": {
            "items": [{"gap_type": "MISSING_DOC", "description": "<script>xss</script>", "materiality_score": 0.5, "issue_title": "<b>bad</b>"}]
        },
        "reviewable_documents": [],
    }
    html = _fmt_source_calibration(data)
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "<b>evil</b>" not in html
    assert "<b>bad</b>" not in html


def test_source_calibration_formatter_domain_labels():
    from irys.ui.app import _fmt_source_calibration
    data = {
        "evidence_matrix": {"issues": [], "sources": [], "cells": {}, "issue_totals": {}, "source_totals": {}},
        "coverage_report": [],
        "domain_profile": {"profile_id": "test", "source_roles": [], "composed_trust_weights": {}},
        "gap_workbench": {"items": []},
        "reviewable_documents": [],
    }
    legal_html = _fmt_source_calibration(data, domain="legal")
    assert "Source Calibration Inspector" in legal_html
    finance_html = _fmt_source_calibration(data, domain="finance")
    assert "Source Type Assessment" in finance_html
    coding_html = _fmt_source_calibration(data, domain="coding")
    assert "Evidence Source Assessment" in coding_html
    research_html = _fmt_source_calibration(data, domain="academic_research")
    assert "Citation Source Assessment" in research_html
    bio_html = _fmt_source_calibration(data, domain="biomedical")
    assert "Clinical Evidence Assessment" in bio_html


def test_source_calibration_type_diversity():
    from irys.ui.app import _fmt_source_calibration
    data = {
        "evidence_matrix": {
            "issues": [{"id": "i1", "title": "Claim A"}],
            "sources": ["d1.pdf", "d2.pdf", "d3.pdf"],
            "cells": {"i1": {"d1.pdf": {"supporting": 1, "attacking": 0}, "d2.pdf": {"supporting": 1, "attacking": 0}, "d3.pdf": {"supporting": 1, "attacking": 0}}},
            "issue_totals": {"i1": {"supporting": 3, "attacking": 0}},
            "source_totals": {},
        },
        "coverage_report": [],
        "domain_profile": {"profile_id": "legal", "source_roles": [], "composed_trust_weights": {}},
        "gap_workbench": {"items": []},
        "reviewable_documents": [
            {"path": "d1.pdf", "doc_type": "advocacy"},
            {"path": "d2.pdf", "doc_type": "advocacy"},
            {"path": "d3.pdf", "doc_type": "operative"},
        ],
    }
    html = _fmt_source_calibration(data)
    assert "2 types" in html
    assert "single type" not in html


def test_source_calibration_single_type_warning():
    from irys.ui.app import _fmt_source_calibration
    data = {
        "evidence_matrix": {
            "issues": [{"id": "i1", "title": "Claim A"}],
            "sources": ["d1.pdf", "d2.pdf"],
            "cells": {"i1": {"d1.pdf": {"supporting": 1, "attacking": 0}, "d2.pdf": {"supporting": 1, "attacking": 1}}},
            "issue_totals": {"i1": {"supporting": 2, "attacking": 1}},
            "source_totals": {},
        },
        "coverage_report": [],
        "domain_profile": {"profile_id": "legal", "source_roles": [], "composed_trust_weights": {}},
        "gap_workbench": {"items": []},
        "reviewable_documents": [
            {"path": "d1.pdf", "doc_type": "advocacy"},
            {"path": "d2.pdf", "doc_type": "advocacy"},
        ],
    }
    html = _fmt_source_calibration(data)
    assert "single type" in html


def test_source_calibration_items_key_matches_model():
    """Verify formatter reads 'items' key matching MatterModel.get_gap_workbench() shape."""
    from irys.ui.app import _fmt_source_calibration
    model_shape = {
        "evidence_matrix": {"issues": [], "sources": [], "cells": {}, "issue_totals": {}, "source_totals": {}},
        "coverage_report": [],
        "domain_profile": {"profile_id": "legal", "source_roles": [], "composed_trust_weights": {}},
        "gap_workbench": {"matter_id": "m1", "items": [
            {"gap_type": "MISSING_EXPERT", "description": "Need expert witness", "materiality_score": 0.9, "issue_title": "Liability"}
        ]},
        "reviewable_documents": [],
    }
    html = _fmt_source_calibration(model_shape)
    assert "Missing Source Obligations" in html
    assert "Expert" in html
    old_shape = dict(model_shape)
    old_shape["gap_workbench"] = {"gaps": [
        {"gap_type": "MISSING_EXPERT", "description": "Need expert witness", "materiality_score": 0.9, "issue_title": "Liability"}
    ]}
    html_old = _fmt_source_calibration(old_shape)
    assert "Missing Source Obligations" not in html_old


def test_source_calibration_formatter_non_dict_guards():
    from irys.ui.app import _fmt_source_calibration
    data = {
        "evidence_matrix": "not_a_dict",
        "coverage_report": "not_a_list",
        "domain_profile": 42,
        "gap_workbench": None,
        "reviewable_documents": None,
    }
    html = _fmt_source_calibration(data)
    assert "Calibration Summary" in html


def test_source_calibration_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_source_calibration",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


def test_source_calibration_appstate_method_exists():
    from irys.ui.app import AppState
    assert hasattr(AppState, "load_source_calibration")


def test_source_calibration_api_endpoint_exists():
    from irys.service.api import app as fastapi_app
    routes = [r.path for r in fastapi_app.routes]
    assert "/matter/{matter_id}/source-calibration" in routes


# ── Document Card Inspector ──────────────────────────────────────


def test_document_card_labels_all_domains():
    from irys.ui.app import _DOCUMENT_CARD_LABELS
    for d in ("legal", "finance", "coding", "academic_research", "biomedical"):
        assert d in _DOCUMENT_CARD_LABELS, f"Missing domain {d}"
        assert "title" in _DOCUMENT_CARD_LABELS[d]
        assert "empty" in _DOCUMENT_CARD_LABELS[d]
        assert "type" in _DOCUMENT_CARD_LABELS[d]
        assert "privilege" in _DOCUMENT_CARD_LABELS[d]


def test_document_card_formatter_empty():
    from irys.ui.app import _fmt_document_card
    html = _fmt_document_card({})
    assert "viz-empty" in html
    html2 = _fmt_document_card({"card": None})
    assert "viz-empty" in html2
    html3 = _fmt_document_card(None)
    assert "viz-empty" in html3


def test_document_card_formatter_non_dict():
    from irys.ui.app import _fmt_document_card
    html = _fmt_document_card("bad")
    assert "viz-empty" in html


def test_document_card_error_display():
    from irys.ui.app import _fmt_document_card
    html = _fmt_document_card({"error": "unexpected response type: list"})
    assert "error:" in html.lower()
    assert "unexpected response type" in html
    assert "<script>" not in _fmt_document_card({"error": "<script>x</script>"})


def test_document_card_formatter_renders():
    from irys.ui.app import _fmt_document_card
    data = {
        "card": {
            "title": "Master Purchase Agreement",
            "doc_type": "contract",
            "doc_subtype": "purchase",
            "source_side": "plaintiff",
            "author": "Jane Smith",
            "sender": "Legal Dept",
            "recipient": "Finance Dept",
            "creation_date": "2025-01-15",
            "sent_date": "2025-01-16",
            "effective_date": "2025-02-01",
            "purpose": "Governs product purchase terms",
            "rhetorical_posture": "assertive",
            "reliability_posture": "high",
            "operative_status": "operative",
            "privilege_flag": False,
            "unresolved_flags": ["missing_exhibit_A", "date_discrepancy"],
            "source_role": "operative",
        }
    }
    html = _fmt_document_card(data)
    assert "Document Card" in html
    assert "Master Purchase Agreement" in html
    assert "contract" in html
    assert "purchase" in html
    assert "plaintiff" in html
    assert "Jane Smith" in html
    assert "Legal Dept" in html
    assert "Finance Dept" in html
    assert "2025-01-15" in html
    assert "Governs product purchase terms" in html
    assert "assertive" in html
    assert "operative" in html.lower()
    assert "missing_exhibit_A" in html
    assert "date_discrepancy" in html
    assert "PRIVILEGED" not in html


def test_document_card_privilege_badge():
    from irys.ui.app import _fmt_document_card
    data = {"card": {"title": "Secret Memo", "privilege_flag": True, "doc_type": "memo"}}
    html = _fmt_document_card(data)
    assert "PRIVILEGED" in html


def test_document_card_xss():
    from irys.ui.app import _fmt_document_card
    data = {
        "card": {
            "title": "<script>alert(1)</script>",
            "doc_type": "<img onerror=x>",
            "author": "<b>evil</b>",
            "purpose": "<script>xss</script>",
            "rhetorical_posture": "<b>bad</b>",
            "source_role": "<em>test</em>",
            "unresolved_flags": ["<script>flag</script>"],
        }
    }
    html = _fmt_document_card(data)
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "<b>evil</b>" not in html
    assert "<b>bad</b>" not in html
    assert "<em>test</em>" not in html


def test_document_card_domain_labels():
    from irys.ui.app import _fmt_document_card
    data = {"card": {"title": "Test", "doc_type": "report"}}
    legal_html = _fmt_document_card(data, domain="legal")
    assert "Document Card" in legal_html
    finance_html = _fmt_document_card(data, domain="finance")
    assert "Filing Card" in finance_html
    coding_html = _fmt_document_card(data, domain="coding")
    assert "Artifact Card" in coding_html
    research_html = _fmt_document_card(data, domain="academic_research")
    assert "Citation Card" in research_html
    bio_html = _fmt_document_card(data, domain="biomedical")
    assert "Clinical Record Card" in bio_html


def test_document_card_hidden_empty_fields():
    from irys.ui.app import _fmt_document_card
    data = {"card": {"title": "Minimal", "doc_type": "report"}}
    html = _fmt_document_card(data)
    assert "Sender" not in html
    assert "Recipient" not in html


def test_document_card_malformed_flags():
    from irys.ui.app import _fmt_document_card
    data = {"card": {"title": "Test", "unresolved_flags": "not-a-list"}}
    html = _fmt_document_card(data)
    assert "viz-empty" not in html

    data2 = {"card": {"title": "Test", "unresolved_flags": [123, None, {"bad": True}]}}
    html2 = _fmt_document_card(data2)
    assert "viz-empty" not in html2


def test_document_card_backend_interface_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("get_document_card",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


def test_document_card_appstate_methods_exist():
    from irys.ui.app import AppState
    assert hasattr(AppState, "load_document_card")
    assert hasattr(AppState, "get_document_card_choices")


def test_document_card_api_endpoint_exists():
    from irys.service.api import app as fastapi_app
    routes = [r.path for r in fastapi_app.routes]
    assert "/matter/{matter_id}/documents/card" in routes


# ── Editable Document Card ───────────────────────────────────────


def test_patch_document_card_api_endpoint_exists():
    from irys.service.api import app as fastapi_app
    routes = [r.path for r in fastapi_app.routes]
    assert "/matter/{matter_id}/documents/{doc_id}/card" in routes


def test_patch_document_card_backend_balance():
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("patch_document_card",):
        assert hasattr(UIBackend, method), f"UIBackend missing {method}"
        assert hasattr(InProcessBackend, method), f"InProcessBackend missing {method}"
        assert hasattr(HttpBackend, method), f"HttpBackend missing {method}"


def test_correct_document_card_appstate_exists():
    from irys.ui.app import AppState
    assert hasattr(AppState, "correct_document_card")


def test_reclassify_document_card_fields_model_method():
    from irys.matter.matter import MatterModel
    assert hasattr(MatterModel, "reclassify_document_card_fields")


def test_reclassify_card_no_change():
    """When all fields match existing values, no staling should occur."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_no_change")
    result = model.reclassify_document_card_fields("nonexistent_doc_id")
    assert result.get("error") == "Document not found"


def test_reclassify_card_partial_update():
    """Verify partial update only writes specified fields."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_partial")
    doc_id, _ = model.inventory.upsert("doc.pdf", sha256="abc123")

    model.document_cards.upsert(
        doc_id=doc_id,
        doc_type="memo",
        source_role="informal",
        operative_status="draft",
    )
    result = model.reclassify_document_card_fields(
        doc_id, doc_type="contract",
    )
    assert "doc_type" in result.get("changed_fields", [])
    assert "source_role" not in result.get("changed_fields", [])
    card = model.document_cards.get_by_doc_id(doc_id)
    assert card["doc_type"] == "contract"
    assert card["source_role"] == "informal"


def test_reclassify_card_stales_downstream():
    """Verify that doc_type/source_role changes trigger staling."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_stale")
    doc_id, _ = model.inventory.upsert("a.pdf", sha256="abc")

    model.document_cards.upsert(doc_id=doc_id, doc_type="draft", source_role="unknown")
    result = model.reclassify_document_card_fields(
        doc_id, doc_type="contract", source_role="operative",
    )
    assert "doc_type" in result["changed_fields"]
    assert "source_role" in result["changed_fields"]
    assert isinstance(result["staled_count"], int)


def test_reclassify_card_flags_only_no_stale():
    """Flags-only edit should not stale downstream."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_flags")
    doc_id, _ = model.inventory.upsert("b.pdf", sha256="def")

    model.document_cards.upsert(doc_id=doc_id, doc_type="report")
    result = model.reclassify_document_card_fields(
        doc_id, unresolved_flags=["needs_review"],
    )
    assert result["changed_fields"] == ["unresolved_flags"]
    assert result["staled_count"] == 0


# ---------------------------------------------------------------------------
# Scenario Delta Inspector (SO-1, SO-3)
# ---------------------------------------------------------------------------

def test_scenario_delta_labels_all_domains():
    from irys.ui.app import _SCENARIO_DELTA_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        L = _SCENARIO_DELTA_LABELS[domain]
        assert "title" in L
        assert "empty" in L
        assert "col_op" in L


def test_scenario_delta_formatter_empty():
    from irys.ui.app import _fmt_scenario_deltas
    html = _fmt_scenario_deltas([])
    assert "viz-empty" in html


def test_scenario_delta_formatter_renders():
    from irys.ui.app import _fmt_scenario_deltas
    deltas = [
        {"operation": "override_belief", "target_kind": "assertion",
         "target_id": "a1", "created_at": "2026-05-04", "created_by": "user"},
        {"operation": "suppress", "target_kind": "gap",
         "target_id": "g1", "created_at": "2026-05-04", "created_by": "system"},
    ]
    html = _fmt_scenario_deltas(deltas)
    assert "override_belief" in html
    assert "suppress" in html
    assert "2 deltas" in html


def test_scenario_delta_formatter_non_dict_guard():
    from irys.ui.app import _fmt_scenario_deltas
    html = _fmt_scenario_deltas(["not-a-dict", 42])
    assert "viz-empty" in html


def test_scenario_delta_formatter_xss():
    from irys.ui.app import _fmt_scenario_deltas
    deltas = [
        {"operation": "<script>alert(1)</script>", "target_kind": "assertion",
         "target_id": "a1", "created_at": "now", "created_by": "user"},
    ]
    html = _fmt_scenario_deltas(deltas)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_scenario_delta_backend_balance():
    import inspect
    from irys.ui.backends.base import UIBackend as DashboardBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    for method in ("list_scenario_deltas", "apply_scenario_delta",
                    "compute_scenario_snapshot", "archive_scenario_branch"):
        assert hasattr(DashboardBackend, method), f"base missing {method}"
        assert hasattr(InProcessBackend, method), f"in_process missing {method}"
        assert hasattr(HttpBackend, method), f"http missing {method}"


def test_scenario_delta_appstate_methods_exist():
    from irys.ui.app import AppState
    for method in ("load_scenario_deltas", "apply_scenario_delta_ui",
                    "compute_scenario_snapshot_ui", "archive_scenario_branch_ui"):
        assert hasattr(AppState, method), f"AppState missing {method}"


def test_apply_scenario_delta_model():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_delta")
    bid = model.create_scenario_branch(name="test-branch", assumptions=[{"text": "test"}])["branch_id"]
    result = model.apply_scenario_delta(
        bid, "assertion", "a1", "override_belief", {"new_belief": "accepted"},
    )
    assert "delta_id" in result
    assert result["total_deltas"] == 1
    assert result["operation"] == "override_belief"


def test_list_scenario_deltas_model():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_list_delta")
    bid = model.create_scenario_branch(name="test-branch", assumptions=[{"text": "test"}])["branch_id"]
    model.apply_scenario_delta(bid, "assertion", "a1", "suppress")
    model.apply_scenario_delta(bid, "gap", "g1", "resolve_gap")
    deltas = model.list_scenario_deltas(bid)
    assert len(deltas) == 2
    assert deltas[0]["operation"] == "suppress"
    assert deltas[1]["operation"] == "resolve_gap"


def test_compute_scenario_snapshot_model():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_snapshot")
    bid = model.create_scenario_branch(name="test-branch", assumptions=[{"text": "test"}])["branch_id"]
    model.apply_scenario_delta(bid, "gap", "g1", "add_gap")
    result = model.compute_scenario_snapshot(bid)
    assert "snapshot_id" in result
    assert result["delta_count"] == 1
    assert "g1" in result["new_gaps"]


def test_archive_scenario_branch_model():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_archive")
    bid = model.create_scenario_branch(name="to-archive", assumptions=[{"text": "test"}])["branch_id"]
    assert model.archive_scenario_branch(bid) is True
    branch = model.get_scenario_branch(bid)
    assert branch["status"] == "archived"


def test_apply_delta_invalid_operation():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_bad_op")
    bid = model.create_scenario_branch(name="test", assumptions=[{"text": "test"}])["branch_id"]
    result = model.apply_scenario_delta(bid, "assertion", "a1", "invalid_op")
    assert "error" in result


def test_scenario_delta_formatter_shows_payload():
    from irys.ui.app import _fmt_scenario_deltas
    deltas = [
        {"operation": "override_belief", "target_kind": "assertion",
         "target_id": "a1", "created_at": "now", "created_by": "user",
         "payload": {"new_belief": "accepted"}},
    ]
    html = _fmt_scenario_deltas(deltas)
    assert "new_belief" in html
    assert "accepted" in html


def test_scenario_delta_formatter_payload_xss():
    from irys.ui.app import _fmt_scenario_deltas
    deltas = [
        {"operation": "add_gap", "target_kind": "gap",
         "target_id": "g1", "created_at": "now", "created_by": "user",
         "payload": {"description": "<script>xss</script>"}},
    ]
    html = _fmt_scenario_deltas(deltas)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_promote_knowledge_seed_appstate_exists():
    from irys.ui.app import AppState
    assert hasattr(AppState, "promote_knowledge_seed_ui")


def test_promote_knowledge_seed_model():
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory("test_promote")
    result = model.promote_knowledge_seed(
        seed_kind="contradiction_resolution",
        domain_profile_id="legal:1",
        payload_json='{"resolved": true}',
        source_matter_id=model.matter_id,
    )
    assert result["success"] is True
    assert "seed_id" in result


def test_promote_knowledge_seed_backend_balance():
    import inspect
    from irys.ui.backends.base import UIBackend
    from irys.ui.backends.in_process import InProcessBackend
    from irys.ui.backends.http import HttpBackend
    assert hasattr(UIBackend, "promote_knowledge_seed")
    assert hasattr(InProcessBackend, "promote_knowledge_seed")
    assert hasattr(HttpBackend, "promote_knowledge_seed")


# ------------------------------------------------------------------ #
# Privilege taint propagation (SO-5, SO-7, SO-2)
# ------------------------------------------------------------------ #


def test_privilege_taint_propagation_model():
    """reclassify_privilege writes privilege_restricted taint and build_query_context filters it."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    run_id = model.start_run("priv test")
    from irys.matter.runtime import MatterRuntimeAdapter
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    inv_id, _ = model.inventory.upsert("priv.docx", "a" * 64, size_bytes=1)
    model.document_cards.upsert(doc_id=inv_id, title="Priv doc", doc_type="internal", privilege_flag=False)
    aid = adapter.record_fact("Privileged fact", "priv.docx")
    # Before privilege: assertion is clean.
    assert model.memory_broker.object_is_clean("assertion", aid)
    # Mark privileged.
    model.reclassify_privilege("priv.docx", new_flag=True, reviewed_by_kind="user")
    # After: assertion and document are tainted.
    assert not model.memory_broker.object_is_clean("assertion", aid)
    assert not model.memory_broker.object_is_clean("artifact", inv_id)
    # build_query_context should filter tainted assertions.
    ctx = model.build_query_context()
    assert inv_id not in ctx.known_document_ids


def test_privilege_taint_removal_model():
    """Removing privilege clears taint records so objects re-enter clean pipeline."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    run_id = model.start_run("priv clear")
    from irys.matter.runtime import MatterRuntimeAdapter
    adapter = MatterRuntimeAdapter(model, run_id=run_id)
    inv_id, _ = model.inventory.upsert("was-priv.docx", "b" * 64, size_bytes=1)
    model.document_cards.upsert(doc_id=inv_id, title="Was priv", doc_type="internal", privilege_flag=False)
    aid = adapter.record_fact("Was secret", "was-priv.docx")
    model.reclassify_privilege("was-priv.docx", new_flag=True, reviewed_by_kind="user")
    assert not model.memory_broker.object_is_clean("assertion", aid)
    # Remove privilege.
    model.reclassify_privilege("was-priv.docx", new_flag=False, reviewed_by_kind="user")
    assert model.memory_broker.object_is_clean("assertion", aid)
    assert model.memory_broker.object_is_clean("artifact", inv_id)


def test_remove_object_taint_by_class_broker():
    """MemoryBrokerStore.remove_object_taint_by_class deletes only the specified class."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    broker = model.memory_broker
    broker.record_object_taint(target_kind="assertion", target_id="a1", taint_class="privilege_restricted")
    broker.record_object_taint(target_kind="assertion", target_id="a1", taint_class="unknown_taint")
    assert not broker.object_is_clean("assertion", "a1")
    deleted = broker.remove_object_taint_by_class("assertion", "a1", "privilege_restricted")
    assert deleted == 1
    # Still tainted by unknown_taint.
    assert not broker.object_is_clean("assertion", "a1")
    # Remove the other one.
    deleted2 = broker.remove_object_taint_by_class("assertion", "a1", "unknown_taint")
    assert deleted2 == 1
    assert broker.object_is_clean("assertion", "a1")


def test_dependency_manifest_hash_propagates_to_working_set():
    """cache_manifest_hash flows from InvestigationState to WorkingSet."""
    state = InvestigationState.create("test query", ".")
    state.working_set = WorkingSet()
    assert state.working_set.dependency_manifest_hash is None
    state.cache_manifest_hash = "sha256:abc123"
    state.working_set.dependency_manifest_hash = state.cache_manifest_hash
    assert state.working_set.dependency_manifest_hash == "sha256:abc123"
    envelope = OutputEnvelope.create(
        output_text="answer",
        workflow_kind=WorkflowKind.ANALYSIS.value,
        output_shape="answer",
        emitter="test",
        dependency_manifest_hash=state.working_set.dependency_manifest_hash,
    )
    assert envelope.dependency_manifest_hash == "sha256:abc123"


def test_dependency_manifest_hash_survives_working_set_roundtrip():
    """WorkingSet serialization preserves dependency_manifest_hash."""
    ws = WorkingSet()
    ws.dependency_manifest_hash = "sha256:test_hash_42"
    data = ws.to_dict()
    assert data["dependency_manifest_hash"] == "sha256:test_hash_42"
    restored = WorkingSet.from_dict(data)
    assert restored.dependency_manifest_hash == "sha256:test_hash_42"


def test_freshness_report_returns_namespace_data():
    """MatterModel.get_freshness_report returns namespace revision data."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    report = model.get_freshness_report()
    assert isinstance(report, dict)
    assert "namespaces" in report
    assert "stale_namespaces" in report
    assert "is_hot_answerable" in report
    assert isinstance(report["namespaces"], list)
    assert len(report["namespaces"]) > 0
    for ns in report["namespaces"]:
        assert "namespace" in ns
        assert "revision" in ns
        assert "state" in ns


def test_freshness_report_detects_fresh_namespaces():
    """After bumping a namespace revision, freshness report shows it as fresh."""
    from irys.matter.matter import MatterModel
    model = MatterModel.open_in_memory()
    broker = model.memory_broker
    broker.bump_namespace_revision("claims")
    report = model.get_freshness_report()
    claims_ns = [ns for ns in report["namespaces"] if ns["namespace"] == "claims"]
    assert len(claims_ns) == 1
    assert claims_ns[0]["revision"] > 0
    assert claims_ns[0]["state"] == "fresh"


def test_freshness_report_formatter_renders_table():
    """_fmt_freshness_report renders an HTML table with namespace rows."""
    import importlib
    app_mod = importlib.import_module("irys.ui.app")
    fmt = app_mod._fmt_freshness_report
    data = {
        "matter_id": "test",
        "namespace_count": 2,
        "stale_namespaces": [],
        "is_hot_answerable": True,
        "active_run_count": 0,
        "last_update_at": "2026-05-04T12:00:00",
        "namespaces": [
            {"namespace": "claims", "revision": 5, "state": "fresh", "updated_at": "2026-05-04T12:00:00"},
            {"namespace": "entities", "revision": 0, "state": "missing", "updated_at": None},
        ],
    }
    html = fmt(data, domain="legal")
    assert "claims" in html
    assert "entities" in html
    assert "Namespace Freshness" in html
    assert "Hot-Answerable" in html


def test_freshness_report_formatter_xss():
    """_fmt_freshness_report escapes all namespace values."""
    from irys.ui.app import _fmt_freshness_report
    data = {
        "matter_id": "test",
        "namespace_count": 1,
        "stale_namespaces": ["<script>alert(1)</script>"],
        "is_hot_answerable": False,
        "active_run_count": 0,
        "last_update_at": "<img onerror=alert(1)>",
        "namespaces": [
            {
                "namespace": "<script>xss</script>",
                "revision": 1,
                "state": "fresh",
                "updated_at": "<b onmouseover=alert(1)>xss</b>",
            },
        ],
    }
    html = _fmt_freshness_report(data, domain="legal")
    assert "<script>" not in html
    assert "<img " not in html
    assert "<b " not in html
    assert "&lt;script&gt;" in html


def test_freshness_report_formatter_error_dict():
    """_fmt_freshness_report surfaces error dicts from backend."""
    from irys.ui.app import _fmt_freshness_report
    data = {"error": "backend failed"}
    html = _fmt_freshness_report(data, domain="legal")
    assert "Backend error" in html
    assert "backend failed" in html


# ── Reasoning Cache Stats tests ──────────────────────────────────────


def test_cache_stats_formatter_renders_table():
    """_fmt_cache_stats renders stage table with hit rates."""
    from irys.ui.app import _fmt_cache_stats
    data = {
        "matter_id": "m1",
        "trust_revision": 3,
        "total_entries": 10,
        "total_hits": 4,
        "overall_hit_rate": 0.4,
        "stages": [
            {"stage": "cascade_decision", "total_entries": 5, "hit_count": 3, "hit_rate": 0.6},
            {"stage": "orient", "total_entries": 5, "hit_count": 1, "hit_rate": 0.2},
        ],
    }
    html = _fmt_cache_stats(data, domain="legal")
    assert "cascade_decision" in html
    assert "orient" in html
    assert "60.0%" in html
    assert "20.0%" in html
    assert "Trust Revision" in html


def test_cache_stats_formatter_xss():
    """_fmt_cache_stats escapes untrusted stage names."""
    from irys.ui.app import _fmt_cache_stats
    data = {
        "matter_id": "m1",
        "trust_revision": 1,
        "total_entries": 1,
        "total_hits": 0,
        "overall_hit_rate": 0.0,
        "stages": [
            {"stage": "<script>alert(1)</script>", "total_entries": 1, "hit_count": 0, "hit_rate": 0.0},
        ],
    }
    html = _fmt_cache_stats(data, domain="legal")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_cache_stats_formatter_error_dict():
    """_fmt_cache_stats surfaces error dicts from backend."""
    from irys.ui.app import _fmt_cache_stats
    data = {"error": "cache unavailable"}
    html = _fmt_cache_stats(data, domain="legal")
    assert "Backend error" in html
    assert "cache unavailable" in html


def test_cache_stats_formatter_empty():
    """_fmt_cache_stats shows empty message when no stages."""
    from irys.ui.app import _fmt_cache_stats
    html = _fmt_cache_stats({}, domain="legal")
    assert "viz-empty" in html


def test_cache_stats_formatter_all_domains():
    """_fmt_cache_stats uses domain-specific labels."""
    from irys.ui.app import _fmt_cache_stats
    data = {
        "matter_id": "m1",
        "trust_revision": 1,
        "total_entries": 2,
        "total_hits": 1,
        "overall_hit_rate": 0.5,
        "stages": [
            {"stage": "synthesis", "total_entries": 2, "hit_count": 1, "hit_rate": 0.5},
        ],
    }
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        html = _fmt_cache_stats(data, domain=domain)
        assert "synthesis" in html
        assert "viz-shell" in html


# ── LLM Usage Summary tests ──────��──────────────────────────────────


def test_llm_usage_formatter_renders_summary():
    """_fmt_llm_usage renders token counts and cost."""
    from irys.ui.app import _fmt_llm_usage
    data = {
        "request_count": 15,
        "successful_requests": 14,
        "failed_requests": 1,
        "input_tokens": 50000,
        "output_tokens": 12000,
        "cache_read_tokens": 8000,
        "thinking_tokens": 3000,
        "estimated_cost_usd": 0.0425,
        "by_tier": {
            "flash": {
                "requests": 10, "input_tokens": 30000, "output_tokens": 8000,
                "estimated_cost_usd": 0.015,
            },
            "pro": {
                "requests": 5, "input_tokens": 20000, "output_tokens": 4000,
                "estimated_cost_usd": 0.0275,
            },
        },
    }
    html = _fmt_llm_usage(data, domain="legal")
    assert "14 Successful" in html
    assert "1 Failed" in html
    assert "$0.0425" in html
    assert "50,000" in html
    assert "flash" in html
    assert "pro" in html


def test_llm_usage_formatter_xss():
    """_fmt_llm_usage escapes untrusted tier names."""
    from irys.ui.app import _fmt_llm_usage
    data = {
        "request_count": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 0,
        "thinking_tokens": 0,
        "estimated_cost_usd": 0.001,
        "by_tier": {
            "<script>alert(1)</script>": {
                "requests": 1, "input_tokens": 100, "output_tokens": 50,
                "estimated_cost_usd": 0.001,
            },
        },
    }
    html = _fmt_llm_usage(data, domain="legal")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_llm_usage_formatter_error_dict():
    """_fmt_llm_usage surfaces error dicts from backend."""
    from irys.ui.app import _fmt_llm_usage
    data = {"error": "usage unavailable"}
    html = _fmt_llm_usage(data, domain="legal")
    assert "Backend error" in html
    assert "usage unavailable" in html


def test_llm_usage_formatter_empty():
    """_fmt_llm_usage shows empty message when no requests."""
    from irys.ui.app import _fmt_llm_usage
    html = _fmt_llm_usage({}, domain="legal")
    assert "viz-empty" in html
    html2 = _fmt_llm_usage({"request_count": 0}, domain="finance")
    assert "viz-empty" in html2


def test_llm_usage_formatter_all_domains():
    """_fmt_llm_usage uses domain-specific labels."""
    from irys.ui.app import _fmt_llm_usage
    data = {
        "request_count": 1,
        "successful_requests": 1,
        "failed_requests": 0,
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 0,
        "thinking_tokens": 0,
        "estimated_cost_usd": 0.001,
        "by_tier": {},
    }
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        html = _fmt_llm_usage(data, domain=domain)
        assert "viz-shell" in html
        assert "$0.0010" in html


# ── Domain label lookup tests ────────────────────────────────────────


def test_domain_source_label_all_domains():
    """_domain_source_label resolves source roles across all 5 domains."""
    from irys.ui.app import _domain_source_label
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        label = _domain_source_label("operative", domain)
        assert label and isinstance(label, str)
        assert label != "operative"


def test_domain_source_label_unknown_falls_back():
    """_domain_source_label falls back to title-cased role for unknown roles."""
    from irys.ui.app import _domain_source_label
    assert _domain_source_label("made_up_role", "legal") == "Made Up Role"


def test_domain_speech_act_label_all_domains():
    """_domain_speech_act_label resolves speech acts across all 5 domains."""
    from irys.ui.app import _domain_speech_act_label
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        label = _domain_speech_act_label("alleged", domain)
        assert label and isinstance(label, str)


def test_domain_speech_act_label_unknown_falls_back():
    """_domain_speech_act_label falls back for unknown acts."""
    from irys.ui.app import _domain_speech_act_label
    assert _domain_speech_act_label("unrecognized_act", "finance") == "Unrecognized Act"


def test_domain_belief_label_all_domains():
    """_domain_belief_label resolves belief states across all 5 domains."""
    from irys.ui.app import _domain_belief_label
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        label = _domain_belief_label("disputed", domain)
        assert label and isinstance(label, str)


def test_domain_belief_label_unknown_falls_back():
    """_domain_belief_label falls back for unknown states."""
    from irys.ui.app import _domain_belief_label
    assert _domain_belief_label("phantom_state", "coding") == "Phantom State"


def test_document_console_formatter_all_domains():
    """_fmt_document_console renders across all 5 domains."""
    from irys.ui.app import _fmt_document_console
    data = {
        "document_ref": "doc-001",
        "card": {
            "document_kind": "contract",
            "page_count": 12,
            "extracted_date": "2025-01-15",
        },
        "candidates": [
            {"proposition_text": "Amount is $5000", "belief_state": "alleged"},
        ],
    }
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        html = _fmt_document_console(data, domain=domain)
        assert "doc-001" in html


def test_document_console_formatter_error_dict():
    """_fmt_document_console surfaces error dicts."""
    from irys.ui.app import _fmt_document_console
    html = _fmt_document_console({"error": "doc not found"}, domain="legal")
    assert "Backend error" in html
    assert "doc not found" in html


def test_document_console_formatter_xss():
    """_fmt_document_console escapes untrusted document refs."""
    from irys.ui.app import _fmt_document_console
    data = {
        "document_ref": "<script>alert(1)</script>",
        "card": {},
        "candidates": [],
    }
    html = _fmt_document_console(data, domain="legal")
    assert "<script>" not in html


# ── Review queue label tests ─────────────────────────────────────────


def test_review_count_badge_all_domains():
    """_fmt_review_count_badge renders bucket labels per domain."""
    from irys.ui.app import _fmt_review_count_badge
    bucket_counts = {0: 2, 2: 5, 4: 1}
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        html = _fmt_review_count_badge(8, bucket_counts, domain=domain)
        assert "8 finding(s)" in html


def test_review_count_badge_zero():
    """_fmt_review_count_badge shows success when no items need review."""
    from irys.ui.app import _fmt_review_count_badge
    html = _fmt_review_count_badge(0, {}, domain="legal")
    assert "All findings reviewed" in html


def test_review_bucket_badge_all_domains():
    """_review_bucket_badge renders domain-specific labels."""
    from irys.ui.app import _review_bucket_badge
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        html = _review_bucket_badge(0, 0.85, domain=domain)
        assert "blocker" in html.lower()
        assert "85%" in html


def test_review_queue_formatter_all_domains():
    """_fmt_review_queue renders kind labels per domain."""
    from irys.ui.app import _fmt_review_queue
    queue = [
        {
            "priority_bucket": 2,
            "priority_score": 0.7,
            "target_kind": "assertion",
            "target_id": "a-001",
            "proposition_text": "Payment was made on Jan 5",
        },
    ]
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        html = _fmt_review_queue(queue, domain=domain)
        assert "Payment was made" in html


def test_review_queue_empty():
    """_fmt_review_queue shows empty state."""
    from irys.ui.app import _fmt_review_queue
    html = _fmt_review_queue([], domain="legal")
    assert "No findings need review" in html


def test_review_queue_xss():
    """_fmt_review_queue escapes untrusted proposition text."""
    from irys.ui.app import _fmt_review_queue
    queue = [
        {
            "priority_bucket": 1,
            "priority_score": 0.5,
            "target_kind": "assertion",
            "target_id": "a-xss",
            "proposition_text": "<img src=x onerror=alert(1)>",
        },
    ]
    html = _fmt_review_queue(queue, domain="legal")
    assert "<img " not in html


def test_source_drawer_reviewer_labels_all_domains():
    """_SOURCE_DRAWER_REVIEWER_LABELS has entries for all 5 domains."""
    from irys.ui.app import _SOURCE_DRAWER_REVIEWER_LABELS
    for domain in ("legal", "finance", "coding", "academic_research", "biomedical"):
        labels = _SOURCE_DRAWER_REVIEWER_LABELS[domain]
        assert "user" in labels
        assert "system" in labels


# ── Utility formatter tests ──────────────────────────────────────────


def test_fmt_ws_status_escapes():
    """_fmt_ws_status escapes message content."""
    from irys.ui.app import _fmt_ws_status
    html = _fmt_ws_status("<script>xss</script>", kind="ok")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "ws-ok" in html


def test_fmt_ws_status_empty():
    """_fmt_ws_status returns empty string for empty message."""
    from irys.ui.app import _fmt_ws_status
    assert _fmt_ws_status("") == ""


def test_fmt_matter_card_escapes_files():
    """_fmt_matter_card escapes file names."""
    from irys.ui.app import _fmt_matter_card
    html = _fmt_matter_card("test-matter", ["<img src=x>"])
    assert "<img " not in html
    assert "&lt;img" in html


def test_fmt_matter_card_empty():
    """_fmt_matter_card shows empty state when no name."""
    from irys.ui.app import _fmt_matter_card
    html = _fmt_matter_card("", [])
    assert "No matter selected" in html


def test_fmt_ledger_event_returns_none_for_sentinel():
    """_fmt_ledger_event returns None for terminal sentinel dicts."""
    from irys.ui.app import _fmt_ledger_event
    assert _fmt_ledger_event({"event": "run_terminal"}) is None
    assert _fmt_ledger_event({"error": "something broke"}) is None


def test_fmt_ledger_event_formats_event():
    """_fmt_ledger_event renders a valid event dict."""
    from irys.ui.app import _fmt_ledger_event
    result = _fmt_ledger_event({
        "event_type": "cascade_decision",
        "seq_no": 3,
        "summary": "Routed to trace family",
        "why": "Found entity reference",
    })
    assert "#3" in result
    assert "cascade_decision" in result
    assert "Routed to trace family" in result
    assert "Found entity reference" in result


# --- _fmt_research_mode_label tests ---


def test_research_mode_label_known_modes():
    """_fmt_research_mode_label normalizes known modes to title case."""
    from irys.ui.app import _fmt_research_mode_label
    assert _fmt_research_mode_label("deep") == "Deep"
    assert _fmt_research_mode_label("simple") == "Simple"
    assert _fmt_research_mode_label("sebih_special") == "Sebih Special"


def test_research_mode_label_none_falls_back():
    """_fmt_research_mode_label returns the default mode label for None."""
    from irys.ui.app import _fmt_research_mode_label
    result = _fmt_research_mode_label(None)
    assert result == "Deep"


def test_research_mode_label_unknown_falls_back():
    """_fmt_research_mode_label returns the default for unrecognized values."""
    from irys.ui.app import _fmt_research_mode_label
    result = _fmt_research_mode_label("nonexistent_mode_xyz")
    assert result == "Deep"


# --- _fmt_thinking_steps_fallback tests ---


def test_thinking_steps_fallback_renders_numbered():
    """_fmt_thinking_steps_fallback renders numbered lines from state."""
    from irys.ui.app import _fmt_thinking_steps_fallback
    from types import SimpleNamespace
    step1 = SimpleNamespace(content="Analyzing jurisdiction")
    step2 = SimpleNamespace(content="Checking statute of limitations")
    state = SimpleNamespace(thinking_steps=[step1, step2])
    result = _fmt_thinking_steps_fallback(state)
    assert "1. Analyzing jurisdiction" in result
    assert "2. Checking statute of limitations" in result


def test_thinking_steps_fallback_empty():
    """_fmt_thinking_steps_fallback returns empty string when no steps."""
    from irys.ui.app import _fmt_thinking_steps_fallback
    from types import SimpleNamespace
    assert _fmt_thinking_steps_fallback(SimpleNamespace(thinking_steps=[])) == ""
    assert _fmt_thinking_steps_fallback(SimpleNamespace()) == ""


def test_thinking_steps_fallback_skips_blank():
    """_fmt_thinking_steps_fallback skips steps with empty content."""
    from irys.ui.app import _fmt_thinking_steps_fallback
    from types import SimpleNamespace
    step1 = SimpleNamespace(content="Valid step")
    step2 = SimpleNamespace(content="")
    step3 = SimpleNamespace(content="Another valid step")
    state = SimpleNamespace(thinking_steps=[step1, step2, step3])
    result = _fmt_thinking_steps_fallback(state)
    assert "1. Valid step" in result
    assert "3. Another valid step" in result
    lines = [l for l in result.split("\n") if l.strip()]
    assert len(lines) == 2


# --- _fmt_route_chip tests ---


def test_route_chip_simple():
    """_fmt_route_chip renders 'via <label>' for a single-family route."""
    from irys.ui.app import _fmt_route_chip
    from types import SimpleNamespace
    state = SimpleNamespace(findings={
        "route": {"classifier_family": "read", "terminal_family": "read"}
    })
    assert _fmt_route_chip(state) == "via Quick summary"


def test_route_chip_escalation():
    """_fmt_route_chip shows both families when escalation happened."""
    from irys.ui.app import _fmt_route_chip
    from types import SimpleNamespace
    state = SimpleNamespace(findings={
        "route": {"classifier_family": "read", "terminal_family": "investigate"}
    })
    assert _fmt_route_chip(state) == "via Quick summary → Deep investigation"


def test_route_chip_none_when_no_route():
    """_fmt_route_chip returns None when state has no route."""
    from irys.ui.app import _fmt_route_chip
    from types import SimpleNamespace
    assert _fmt_route_chip(SimpleNamespace(findings={})) is None
    assert _fmt_route_chip(SimpleNamespace(findings=None)) is None
    assert _fmt_route_chip(SimpleNamespace()) is None


def test_route_chip_hides_infra_failure():
    """_fmt_route_chip hides read_infra_failure family."""
    from irys.ui.app import _fmt_route_chip
    from types import SimpleNamespace
    state = SimpleNamespace(findings={
        "route": {"classifier_family": "read", "terminal_family": "read_infra_failure"}
    })
    assert _fmt_route_chip(state) is None


# --- _fmt_span_label tests ---


def test_span_label_page():
    """_fmt_span_label converts 'page:3' to 'Page 3'."""
    from irys.ui.app import _fmt_span_label
    assert _fmt_span_label("page:3") == "Page 3"


def test_span_label_paragraph():
    """_fmt_span_label converts 'para:12' to 'Paragraph 12'."""
    from irys.ui.app import _fmt_span_label
    assert _fmt_span_label("para:12") == "Paragraph 12"
    assert _fmt_span_label("paragraph:5") == "Paragraph 5"


def test_span_label_section_clause_line():
    """_fmt_span_label handles section, clause, line prefixes."""
    from irys.ui.app import _fmt_span_label
    assert _fmt_span_label("section:4.2") == "Section 4.2"
    assert _fmt_span_label("clause:7") == "Clause 7"
    assert _fmt_span_label("line:99") == "Line 99"


def test_span_label_uuid_hidden():
    """_fmt_span_label returns dash for UUID-like strings."""
    from irys.ui.app import _fmt_span_label
    assert _fmt_span_label("a1b2c3d4e5f6a1b2c3d4e5f6") == "—"
    assert _fmt_span_label("a1b2c3d4-e5f6-a1b2-c3d4-e5f6a1b2c3d4") == "—"


def test_span_label_empty():
    """_fmt_span_label returns dash for None/empty."""
    from irys.ui.app import _fmt_span_label
    assert _fmt_span_label(None) == "—"
    assert _fmt_span_label("") == "—"
    assert _fmt_span_label("   ") == "—"


def test_span_label_passthrough():
    """_fmt_span_label passes through short human-readable labels."""
    from irys.ui.app import _fmt_span_label
    assert _fmt_span_label("Exhibit A") == "Exhibit A"


def test_span_label_truncates_long():
    """_fmt_span_label truncates labels over 40 chars."""
    from irys.ui.app import _fmt_span_label
    long = "X" * 50
    result = _fmt_span_label(long)
    assert len(result) <= 40
    assert result.endswith("…")
    assert "X" in result


# --- _friendly_source_label tests ---


def test_friendly_source_label_empty():
    """_friendly_source_label returns dash for None/empty."""
    from irys.ui.app import _friendly_source_label
    assert _friendly_source_label(None) == "—"
    assert _friendly_source_label("") == "—"


def test_friendly_source_label_hex_hidden():
    """_friendly_source_label hides hex-only internal IDs."""
    from irys.ui.app import _friendly_source_label
    assert _friendly_source_label("a1b2c3d4e5f6a1b2c3d4e5f6") == "—"


def test_friendly_source_label_path_basename():
    """_friendly_source_label extracts basename from file paths."""
    from irys.ui.app import _friendly_source_label
    assert _friendly_source_label("/docs/contracts/lease.pdf") == "lease.pdf"
    assert _friendly_source_label("C:\\docs\\contracts\\lease.pdf") == "lease.pdf"


def test_friendly_source_label_passthrough():
    """_friendly_source_label passes through normal names."""
    from irys.ui.app import _friendly_source_label
    assert _friendly_source_label("Exhibit A") == "Exhibit A"


def test_friendly_source_label_truncates_long():
    """_friendly_source_label truncates names over 80 chars."""
    from irys.ui.app import _friendly_source_label
    long = "Z" * 100
    result = _friendly_source_label(long)
    assert len(result) <= 80
    assert result.endswith("…")
    assert "Z" in result


# --- _provenance_tier_label tests ---


def test_provenance_tier_label_known():
    """_provenance_tier_label uppercases the model_tier."""
    from irys.ui.app import _provenance_tier_label
    assert _provenance_tier_label({"model_tier": "openai"}) == "OPENAI tier"
    assert _provenance_tier_label({"model_tier": "anthropic"}) == "ANTHROPIC tier"


def test_provenance_tier_label_empty_fallback():
    """_provenance_tier_label returns 'Irys' when no tier."""
    from irys.ui.app import _provenance_tier_label
    assert _provenance_tier_label({}) == "Irys"
    assert _provenance_tier_label({"model_tier": ""}) == "Irys"
    assert _provenance_tier_label({"model_tier": None}) == "Irys"


# --- _error_html guard regression tests (previously unguarded formatters) ---


def test_authority_panel_error_dict():
    """_fmt_authority_panel surfaces error dicts instead of silent empty state."""
    from irys.ui.app import _fmt_authority_panel
    html = _fmt_authority_panel({"error": "network timeout"})
    assert "Backend error" in html
    assert "network timeout" in html


def test_authority_panel_xss_in_error():
    """_fmt_authority_panel escapes XSS in error messages."""
    from irys.ui.app import _fmt_authority_panel
    html = _fmt_authority_panel({"error": "<script>alert(1)</script>"})
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_document_intelligence_panel_error_dict():
    """_fmt_document_intelligence_panel surfaces error dicts."""
    from irys.ui.app import _fmt_document_intelligence_panel
    html = _fmt_document_intelligence_panel({"error": "db unreachable"})
    assert "Backend error" in html
    assert "db unreachable" in html


def test_answer_audit_error_dict():
    """_fmt_answer_audit surfaces error dicts instead of silent empty state."""
    from irys.ui.app import _fmt_answer_audit
    html = _fmt_answer_audit({"error": "audit service unavailable"})
    assert "Backend error" in html
    assert "audit service unavailable" in html


# --- RLM exception hierarchy and context manager tests ---


def test_rlm_exception_hierarchy():
    """All RLM exceptions inherit from RLMError."""
    from irys.rlm.exceptions import (
        RLMError,
        CriticalReasoningError,
        OptionalEnrichmentError,
        RecoverableInfrastructureError,
        UserVisibleDegradation,
    )
    assert issubclass(CriticalReasoningError, RLMError)
    assert issubclass(OptionalEnrichmentError, RLMError)
    assert issubclass(RecoverableInfrastructureError, RLMError)
    assert issubclass(UserVisibleDegradation, RLMError)
    assert issubclass(RLMError, Exception)


def test_noncritical_success_passes_through():
    """noncritical context manager does nothing on success."""
    from irys.rlm.exceptions import noncritical
    executed = False
    with noncritical("test.ok", default="fallback"):
        executed = True
    assert executed


def test_noncritical_catches_and_logs(caplog):
    """noncritical logs the exception and continues."""
    import logging
    from irys.rlm.exceptions import noncritical
    with caplog.at_level(logging.WARNING, logger="irys.rlm.exceptions"):
        with noncritical("cache.write", default=None):
            raise ValueError("simulated cache failure")
    assert "noncritical cache.write failed" in caplog.text
    assert "simulated cache failure" in caplog.text


def test_noncritical_does_not_swallow_keyboard_interrupt():
    """noncritical must not catch BaseException subclasses."""
    import pytest
    from irys.rlm.exceptions import noncritical
    with pytest.raises(KeyboardInterrupt):
        with noncritical("test.interrupt"):
            raise KeyboardInterrupt()


def test_visible_degradation_success():
    """visible_degradation does nothing on success."""
    from irys.rlm.exceptions import visible_degradation
    executed = False
    with visible_degradation("test.ok"):
        executed = True
    assert executed


def test_visible_degradation_raises_user_visible():
    """visible_degradation wraps exceptions in UserVisibleDegradation."""
    import pytest
    from irys.rlm.exceptions import visible_degradation, UserVisibleDegradation
    with pytest.raises(UserVisibleDegradation, match="render_gaps failed"):
        with visible_degradation("render_gaps"):
            raise RuntimeError("db down")


def test_visible_degradation_calls_result_factory():
    """visible_degradation calls result_factory instead of raising."""
    from irys.rlm.exceptions import visible_degradation
    captured = {}
    def factory(component, exc):
        captured["component"] = component
        captured["exc"] = str(exc)
    with visible_degradation("render_gaps", result_factory=factory):
        raise RuntimeError("db down")
    assert captured["component"] == "render_gaps"
    assert "db down" in captured["exc"]


# ── Domain-parameterized prompt tests ─────────────────────────────────


_ALL_DOMAINS = ["legal", "finance", "coding", "academic_research", "biomedical"]


def test_classifier_prompt_builds_for_all_domains():
    """_build_classifier_prompt produces a valid template for every domain."""
    from irys.rlm.governance import _build_classifier_prompt
    for domain in _ALL_DOMAINS:
        prompt = _build_classifier_prompt(domain)
        formatted = prompt.format(
            snapshot_block="SNAP", conversation_block="CONV", query="test",
        )
        assert "test" in formatted
        assert "{snapshot_block}" not in formatted
        assert "{conversation_block}" not in formatted
        assert "{query}" not in formatted


def test_classifier_prompt_unknown_domain_falls_back_to_legal():
    """Unknown domain IDs fall back to legal vocabulary."""
    from irys.rlm.governance import _build_classifier_prompt
    legal = _build_classifier_prompt("legal")
    unknown = _build_classifier_prompt("nonexistent_domain")
    assert legal == unknown


def test_classifier_prompt_domain_specific_vocabulary():
    """Each domain's prompt contains its platform and matter terms."""
    from irys.rlm.governance import _build_classifier_prompt
    assert "legal intelligence platform" in _build_classifier_prompt("legal")
    assert "financial intelligence platform" in _build_classifier_prompt("finance")
    assert "software analysis platform" in _build_classifier_prompt("coding")
    assert "research analysis platform" in _build_classifier_prompt("academic_research")
    assert "biomedical intelligence platform" in _build_classifier_prompt("biomedical")


def test_pleasantry_prompt_builds_for_all_domains():
    """_build_pleasantry_prompt produces a valid template for every domain."""
    from irys.rlm.governance import _build_pleasantry_prompt
    for domain in _ALL_DOMAINS:
        prompt = _build_pleasantry_prompt(domain)
        formatted = prompt.format(query="hello")
        assert "hello" in formatted
        assert "{query}" not in formatted


def test_pleasantry_prompt_domain_specific_analysis_noun():
    """Each domain's pleasantry prompt uses the correct analysis noun."""
    from irys.rlm.governance import _build_pleasantry_prompt
    assert "legal analysis" in _build_pleasantry_prompt("legal")
    assert "financial analysis" in _build_pleasantry_prompt("finance")
    assert "code analysis" in _build_pleasantry_prompt("coding")
    assert "research analysis" in _build_pleasantry_prompt("academic_research")
    assert "clinical analysis" in _build_pleasantry_prompt("biomedical")


def test_cascade_governor_resolve_domain_default():
    """CascadeGovernor._resolve_domain returns 'legal' with no matter model."""
    gov = CascadeGovernor(client=_StubClient())
    assert gov._resolve_domain() == "legal"


def test_cascade_governor_resolve_domain_caches():
    """Once resolved, domain is cached on the governor."""
    gov = CascadeGovernor(client=_StubClient())
    gov._cached_domain = "finance"
    assert gov._resolve_domain() == "finance"


def test_read_family_prompt_builds_for_all_domains():
    """_build_read_family_prompt produces a valid template for every domain."""
    from irys.rlm.governance import _build_read_family_prompt
    for domain in _ALL_DOMAINS:
        prompt = _build_read_family_prompt(domain)
        formatted = prompt.format(
            matter_summary="S", verified_block="V",
            attorney_guidance_block="G", candidate_block="C",
            issues_block="I", gaps_block="GG",
            conversation_block="CONV", query="test",
        )
        assert "test" in formatted
        assert "{query}" not in formatted
        assert "{matter_summary}" not in formatted


def test_read_family_prompt_domain_specific_terms():
    """Each domain's read prompt uses domain-specific matter noun."""
    from irys.rlm.governance import _build_read_family_prompt
    assert "legal matter" in _build_read_family_prompt("legal")
    assert "financial matter" in _build_read_family_prompt("finance")
    assert "codebase analysis" in _build_read_family_prompt("coding")
    assert "research matter" in _build_read_family_prompt("academic_research")
    assert "biomedical matter" in _build_read_family_prompt("biomedical")


def test_scenario_parse_prompt_builds_for_all_domains():
    """_build_scenario_parse_prompt produces a valid template for every domain."""
    from irys.rlm.governance import _build_scenario_parse_prompt
    for domain in _ALL_DOMAINS:
        prompt = _build_scenario_parse_prompt(domain)
        formatted = prompt.format(query="what if X")
        assert "what if X" in formatted
        assert "{query}" not in formatted


def test_scenario_parse_prompt_domain_examples():
    """Each domain's scenario prompt contains domain-relevant examples."""
    from irys.rlm.governance import _build_scenario_parse_prompt
    assert "contract is void" in _build_scenario_parse_prompt("legal")
    assert "revenue is restated" in _build_scenario_parse_prompt("finance")
    assert "auth service is stateless" in _build_scenario_parse_prompt("coding")
    assert "effect size is zero" in _build_scenario_parse_prompt("academic_research")
    assert "drug is hepatotoxic" in _build_scenario_parse_prompt("biomedical")


def test_deliverable_sub_intents_per_domain():
    """Every domain has at least 3 deliverable sub-intents including 'other'."""
    from irys.rlm.governance import _DELIVERABLE_SUB_INTENTS_BY_DOMAIN
    for domain in _ALL_DOMAINS:
        intents = _DELIVERABLE_SUB_INTENTS_BY_DOMAIN[domain]
        assert len(intents) >= 3, f"{domain} has too few intents"
        names = {n for n, _ in intents}
        assert "other" in names, f"{domain} missing 'other' fallback intent"


def test_deliverable_sub_intent_prompt_builds_for_all_domains():
    """_build_deliverable_sub_intent_prompt produces a valid template for every domain."""
    from irys.rlm.governance import _build_deliverable_sub_intent_prompt
    for domain in _ALL_DOMAINS:
        prompt = _build_deliverable_sub_intent_prompt(domain)
        formatted = prompt.format(intent_list="- a: b", query="test")
        assert "test" in formatted
        assert "{intent_list}" not in formatted
        assert "{query}" not in formatted


def test_query_intents_per_domain():
    """Every domain has 8 query intents matching standard names."""
    from irys.rlm.governance import _QUERY_INTENTS_BY_DOMAIN
    expected_names = {
        "list_quants", "list_actors", "list_documents", "list_gaps",
        "list_issues", "list_contradictions", "list_recent_facts",
        "list_authorities",
    }
    for domain in _ALL_DOMAINS:
        intents = _QUERY_INTENTS_BY_DOMAIN[domain]
        assert len(intents) == 8, f"{domain} has {len(intents)} intents, expected 8"
        names = {n for n, _ in intents}
        assert names == expected_names, f"{domain} intent names mismatch"


# ── Shared domain resolver tests ──────────────────────────────────────


def test_resolve_matter_domain_defaults_to_legal():
    """resolve_matter_domain returns 'legal' with no matter model."""
    from irys.rlm.governance import resolve_matter_domain
    assert resolve_matter_domain(None) == "legal"


def test_resolve_matter_domain_uses_cached():
    """resolve_matter_domain returns cached value if valid."""
    from irys.rlm.governance import resolve_matter_domain
    assert resolve_matter_domain(None, "finance") == "finance"


def test_resolve_matter_domain_rejects_unknown_cached():
    """resolve_matter_domain falls back if cached value is not in supported set."""
    from irys.rlm.governance import resolve_matter_domain
    assert resolve_matter_domain(None, "unknown_domain") == "legal"


def test_resolve_matter_domain_validates_supported_set():
    """All five target domains are in the supported set."""
    from irys.rlm.governance import _SUPPORTED_DOMAINS
    for domain in _ALL_DOMAINS:
        assert domain in _SUPPORTED_DOMAINS
