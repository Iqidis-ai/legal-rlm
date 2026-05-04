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
    from irys.ui.app import _fmt_so_scorecard_panel
    result = _fmt_so_scorecard_panel({})
    assert "viz-empty" in result


def test_fmt_so_scorecard_panel_none():
    from irys.ui.app import _fmt_so_scorecard_panel
    result = _fmt_so_scorecard_panel(None)
    assert "viz-empty" in result


def test_fmt_so_scorecard_panel_with_data():
    from irys.ui.app import _fmt_so_scorecard_panel
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
    from irys.ui.app import _fmt_so_scorecard_panel
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
    from irys.ui.app import _fmt_so_scorecard_panel
    so = {
        "targets": {},
        "targets_met": {},
    }
    result = _fmt_so_scorecard_panel(so, domain="finance")
    assert "Analysis Quality Scorecard" in result
    assert "Durability" in result


def test_fmt_so_scorecard_panel_missing_targets():
    from irys.ui.app import _fmt_so_scorecard_panel
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
    """HttpBackend.compute_proof_state returns {} on non-dict response."""
    from irys.ui.backends.http import HttpBackend
    backend = HttpBackend.__new__(HttpBackend)
    import asyncio

    async def _mock_post(path, body=None):
        return "unexpected string"

    backend._post = _mock_post
    result = asyncio.run(
        backend.compute_proof_state("m1")
    )
    assert result == {}


def test_http_backend_flush_pending_type_guard():
    """HttpBackend.flush_pending returns {} on non-dict response."""
    from irys.ui.backends.http import HttpBackend
    backend = HttpBackend.__new__(HttpBackend)
    import asyncio

    async def _mock_post(path, body=None):
        return [1, 2, 3]

    backend._post = _mock_post
    result = asyncio.run(
        backend.flush_pending("m1")
    )
    assert result == {}


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
    assert _fmt_percent_html(None) == "&mdash;"
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
