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
