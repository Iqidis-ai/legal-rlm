import asyncio
from dataclasses import dataclass

import pytest

from irys.matter import AssertionCandidate, MatterModel, SourceRole, SpeechAct
from irys.rlm.engine import RLMConfig, RLMEngine
from irys.rlm.governance import CascadeGovernor, infer_task_spec
from irys.rlm.state import InvestigationState


class _StubClient:
    pass


class _ReadBiasedClient:
    def __init__(self):
        self.calls = []

    async def complete(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return '{"family": "read", "confidence": 0.99, "rationale": "cheap path"}'


def _warm_matter_model() -> MatterModel:
    matter = MatterModel.open_in_memory()
    run_id = matter.start_run("seed")
    matter.record_assertion(
        AssertionCandidate(
            proposition_text="Seed fact",
            speech_act=SpeechAct.ALLEGED,
            source_role=SourceRole.ADVOCACY,
            document_id="seed.pdf",
        ),
        run_id=run_id,
    )
    matter.complete_run(run_id)
    return matter


@dataclass(frozen=True)
class FailureQueryCase:
    item_id: str
    query: str
    task_type: str
    operation: str
    answer_shape: str
    required_evidence: tuple[str, ...]


TESTER_QUERY_CASES = (
    FailureQueryCase(
        item_id="DELE-001",
        query=(
            "Give me a one-page deal summary across all three Third A&R "
            "Master Supply and Offtake Agreements: Lion, ARKS, BSR."
        ),
        task_type="multi_document_synthesis",
        operation="synthesize",
        answer_shape="multi_document_memo",
        required_evidence=("document_span", "per_document_finding", "source_summary"),
    ),
    FailureQueryCase(
        item_id="DELE-002",
        query=(
            "Compare the schedules in the Lion agreement vs the ARKS "
            "agreement and identify differences."
        ),
        task_type="document_comparison",
        operation="compare",
        answer_shape="side_by_side_comparison",
        required_evidence=("document_span", "per_document_finding"),
    ),
    FailureQueryCase(
        item_id="DELE-008",
        query=(
            "Find every reference to the Step-Out Inventory Sales Agreement "
            "across all three documents."
        ),
        task_type="cross_document_reference_search",
        operation="extract",
        answer_shape="reference_inventory",
        required_evidence=("reference_span", "document_ref"),
    ),
    FailureQueryCase(
        item_id="DELE-016",
        query=(
            "Generate a complete inventory of every defined term across all "
            "three agreements. Group by document."
        ),
        task_type="defined_term_inventory",
        operation="extract",
        answer_shape="inventory_table",
        required_evidence=("defined_term", "definition_span", "section_ref"),
    ),
    FailureQueryCase(
        item_id="DELE-020",
        query=(
            "Identify the signatories for each of the three agreements. For "
            "each: name, title, entity."
        ),
        task_type="signatory_extraction",
        operation="extract",
        answer_shape="signatory_table",
        required_evidence=("signature_block_span", "actor_role", "entity"),
    ),
    FailureQueryCase(
        item_id="DELE-013",
        query=(
            "Pull the language of Section 19.7 of the Lion agreement "
            "(default events related to ESG covenants)."
        ),
        task_type="premise_check",
        operation="verify_absence",
        answer_shape="premise_status",
        required_evidence=("search_coverage", "absence_status"),
    ),
    FailureQueryCase(
        item_id="DELE-011",
        query=(
            "These SEC-filed exhibits contain [***] redactions throughout "
            "for confidentiality. List the categories of information that "
            "appear to be redacted."
        ),
        task_type="redaction_categorization",
        operation="classify",
        answer_shape="redaction_category_table",
        required_evidence=("redaction_marker", "source_span", "category_rationale"),
    ),
    FailureQueryCase(
        item_id="DELE-014",
        query=(
            "Tell me everything I should know about Alon Refining North "
            "Dakota, LP - specifically how it interacts with the J. Aron "
            "supply structure."
        ),
        task_type="out_of_matter_check",
        operation="verify_absence",
        answer_shape="matter_membership_status",
        required_evidence=("matter_entity_search", "absence_status"),
    ),
    FailureQueryCase(
        item_id="WHIT-007",
        query=(
            "Extract Marcus Whitfield's deposition admissions and cross-reference "
            "how each was used in the summary judgment briefing."
        ),
        task_type="deposition_extraction",
        operation="extract",
        answer_shape="testimony_matrix",
        required_evidence=("testimony_span", "admission", "cross_reference"),
    ),
    FailureQueryCase(
        item_id="WHIT-014",
        query=(
            "Give me the procedural history and continuance history in this "
            "case as a timeline with source support."
        ),
        task_type="procedural_history",
        operation="extract",
        answer_shape="procedural_timeline",
        required_evidence=("timeline_event", "order_or_filing", "source_span"),
    ),
    FailureQueryCase(
        item_id="WHIT-008",
        query=(
            "What is the property at issue in this case? Confirm the address "
            "and legal description, and surface any inconsistency."
        ),
        task_type="matter_subject_identification",
        operation="identify",
        answer_shape="identity_with_conflicts",
        required_evidence=("subject_reference", "conflict_check", "source_span"),
    ),
    FailureQueryCase(
        item_id="WHIT-013",
        query=(
            "How much money is at stake? Reconcile purchase price, paid "
            "amount, outstanding balance, improvements, fees, and damages."
        ),
        task_type="quantitative_reconciliation",
        operation="reconcile",
        answer_shape="scoped_quant_table",
        required_evidence=("quant_fact", "reconciliation_scope", "source_span"),
    ),
)


@pytest.mark.parametrize("case", TESTER_QUERY_CASES, ids=lambda case: case.item_id)
def test_tester_queries_infer_expected_task_spec(case: FailureQueryCase):
    spec = infer_task_spec(case.query, "legal")

    assert spec.task_type == case.task_type
    assert spec.operation == case.operation
    assert spec.answer_shape == case.answer_shape
    assert spec.required_evidence == case.required_evidence
    assert spec.fresh_extraction_required is True
    assert spec.cached_state_allowed is False


@pytest.mark.parametrize("case", TESTER_QUERY_CASES, ids=lambda case: case.item_id)
def test_source_grounded_tester_tasks_force_investigate_contract(case: FailureQueryCase):
    spec = infer_task_spec(case.query, "legal")
    contract = CascadeGovernor._contract_for_task("investigate", spec)

    assert contract.family == "investigate"
    assert contract.output_contract["task_spec"]["task_type"] == case.task_type
    assert contract.output_contract["fresh_extraction_allowed"] is True


@pytest.mark.parametrize(
    "query",
    (
        TESTER_QUERY_CASES[0].query,
        TESTER_QUERY_CASES[-2].query,
        TESTER_QUERY_CASES[-1].query,
    ),
)
def test_source_grounded_tester_queries_bypass_read_biased_classifier(query: str):
    client = _ReadBiasedClient()
    governor = CascadeGovernor(client=client, matter_model=_warm_matter_model())

    decision = asyncio.run(governor.decide(query))

    assert decision.family == "investigate"
    assert decision.contract.family == "investigate"
    assert client.calls == []


def test_fact_cache_followup_remains_cache_eligible():
    query = (
        "Earlier I asked you for a one-page deal summary across all three "
        "agreements. Now: based on those same facts, what's the single biggest "
        "contract-level risk for J. Aron if the Big Spring refinery experiences "
        "a prolonged force majeure event?"
    )

    spec = infer_task_spec(query, "legal")

    assert spec.task_type == "legal_analysis"
    assert spec.fresh_extraction_required is False
    assert spec.cached_state_allowed is True


@pytest.mark.parametrize(
    ("query", "bad_output", "expected_failure"),
    (
        (
            "Compare the schedules in the Lion agreement vs the ARKS agreement.",
            "## What changed\n\nAssertions 75 -> 75 (+0).",
            "task_evidence_contract",
        ),
        (
            "Give me a one-page deal summary across all three Third A&R Master Supply and Offtake Agreements.",
            "Evidence on the Lion agreement is currently absent from the provided facts.",
            "task_evidence_contract",
        ),
        (
            "Find every reference to the Step-Out Inventory Sales Agreement across all three documents.",
            "## List Documents\n\n- Lion.pdf (32 pending)",
            "task_evidence_contract",
        ),
        (
            "These SEC-filed exhibits contain [***] redactions throughout. List the categories of information that appear redacted.",
            "I cannot categorize the redacted information because the operative text containing the [***] markers is completely absent.",
            "task_evidence_contract",
        ),
        (
            "Identify the signatories for each of the three agreements.",
            "## List Actors\n\n| actor | role |\n| J. Aron | unknown |",
            "task_evidence_contract",
        ),
        (
            "What is the property at issue in this case? Confirm the address and legal description.",
            "## List Documents\n\nNo documents read.",
            "task_evidence_contract",
        ),
        (
            "How much money is at stake? Reconcile purchase price, paid amount, and outstanding balance.",
            (
                "## Run Diagnostics & Safeguards\n\n"
                "## Financial Analysis\n\n"
                "Auto-generated by SO-6 threshold gate.\n\n"
                "Payment Reconciliation (USD): Invoiced $0.00 - Paid $401,252.88."
            ),
            "task_evidence_contract",
        ),
    ),
)
def test_known_bad_tester_artifacts_fail_task_contract(
    query: str,
    bad_output: str,
    expected_failure: str,
):
    spec = infer_task_spec(query, "legal")
    state = InvestigationState.create(query, ".")
    state.execution_contract = CascadeGovernor._contract_for_task("investigate", spec)
    engine = RLMEngine(gemini_client=_StubClient(), config=RLMConfig())
    engine._initialize_workflow_state(state)

    envelope = engine._emit_output(state, bad_output, emitter="test")

    failures = {
        result.validator
        for result in envelope.validation_results
        if not result.passed
    }
    assert expected_failure in failures
