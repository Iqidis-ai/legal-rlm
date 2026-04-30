"""Workflow primitive tests.

These are intentionally small: the first slice only proves that
objective/obligation/working-set/plan/validation state can survive
checkpoint serialization. The planner and validators can build on this
without inventing a second state channel.
"""

from irys.rlm.state import (
    InvestigationState,
    Obligation,
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
