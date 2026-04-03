"""Tests for IssueStore and issue-driven retrieval integration."""

import pytest
from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole, ModelLayer
from irys.matter import AssertionKind, IssueType, BeliefState
from irys.matter.enums import OriginKind
from irys.matter.runtime import MatterRuntimeAdapter


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def add_assertion(model, text, doc="doc1"):
    c = AssertionCandidate(
        proposition_text=text,
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id=doc,
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(c)
    return aid


# ---------------------------------------------------------------------------
# Issue creation
# ---------------------------------------------------------------------------

def test_upsert_issue_creates_new(model):
    issue_id, is_new = model.issues.upsert_issue(
        title="Did the defendant breach the service agreement?",
        issue_type=IssueType.CLAIM,
        materiality=0.9,
    )
    assert issue_id
    assert is_new is True
    assert model.issues.count_open() == 1


def test_upsert_issue_deduplication(model):
    """Same title (case-insensitive) → same issue."""
    id1, is_new1 = model.issues.upsert_issue(
        title="Was payment made?",
        issue_type=IssueType.CLAIM,
    )
    id2, is_new2 = model.issues.upsert_issue(
        title="was payment made?",  # different case
        issue_type=IssueType.CLAIM,
    )
    assert id1 == id2
    assert is_new1 is True
    assert is_new2 is False
    assert model.issues.count_open() == 1


def test_upsert_different_issues(model):
    id1, _ = model.issues.upsert_issue("Breach of contract?", IssueType.CLAIM)
    id2, _ = model.issues.upsert_issue("Damages amount?", IssueType.DAMAGES)
    assert id1 != id2
    assert model.issues.count_open() == 2


def test_hierarchical_issues(model):
    parent_id, _ = model.issues.upsert_issue(
        title="Liability for breach",
        issue_type=IssueType.CLAIM,
        materiality=0.9,
    )
    child_id, _ = model.issues.upsert_issue(
        title="Was the contract valid?",
        issue_type=IssueType.CONTRACT_QUESTION,
        parent_issue_id=parent_id,
        materiality=0.8,
    )
    assert parent_id != child_id
    assert model.issues.count_open() == 2


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def test_add_predicate(model):
    issue_id, _ = model.issues.upsert_issue("Breach?", IssueType.CLAIM)
    pred_id = model.issues.add_predicate(
        issue_id,
        description="Plaintiff must show defendant failed to perform under §3.2",
        burden_side="plaintiff",
    )
    assert pred_id
    predicates = model.issues.get_predicates(issue_id)
    assert len(predicates) == 1
    assert "§3.2" in predicates[0]["description"]


# ---------------------------------------------------------------------------
# Assertion-issue linking
# ---------------------------------------------------------------------------

def test_link_assertion_to_issue(model):
    issue_id, _ = model.issues.upsert_issue("Payment obligation?", IssueType.CLAIM)
    a_id = add_assertion(model, "Contract requires $50,000 payment by January 15.")

    link_id = model.issues.link_assertion(a_id, issue_id, relation_type="supports")
    assert link_id

    linked = model.issues.get_assertions_for_issue(issue_id)
    assert len(linked) == 1
    assert linked[0]["id"] == a_id
    assert linked[0]["relation_type"] == "supports"


def test_link_assertion_idempotent(model):
    issue_id, _ = model.issues.upsert_issue("Issue?", IssueType.CLAIM)
    a_id = add_assertion(model, "Some fact.")

    link1 = model.issues.link_assertion(a_id, issue_id, "supports")
    link2 = model.issues.link_assertion(a_id, issue_id, "supports")
    assert link1 == link2


def test_multiple_assertions_per_issue(model):
    issue_id, _ = model.issues.upsert_issue("Breach?", IssueType.CLAIM)

    a1 = add_assertion(model, "Contract signed January 15.", "contract.pdf")
    a2 = add_assertion(model, "Payment not received.", "email.pdf")
    a3 = add_assertion(model, "Defendant denies breach.", "answer.pdf")

    model.issues.link_assertion(a1, issue_id, "supports")
    model.issues.link_assertion(a2, issue_id, "supports")
    model.issues.link_assertion(a3, issue_id, "attacks")

    linked = model.issues.get_assertions_for_issue(issue_id)
    assert len(linked) == 3

    relations = {r["relation_type"] for r in linked}
    assert "supports" in relations
    assert "attacks" in relations


# ---------------------------------------------------------------------------
# Open issues ordering
# ---------------------------------------------------------------------------

def test_get_open_issues_ordered_by_salience_materiality(model):
    model.issues.upsert_issue("High priority", IssueType.CLAIM, materiality=0.9, salience=0.9)
    model.issues.upsert_issue("Low priority", IssueType.CLAIM, materiality=0.2, salience=0.2)
    model.issues.upsert_issue("Medium priority", IssueType.CLAIM, materiality=0.6, salience=0.6)

    issues = model.issues.get_open_issues()
    # Should be ordered high → medium → low
    scores = [i["materiality"] * i["salience"] for i in issues]
    assert scores == sorted(scores, reverse=True)


def test_get_open_issues_min_materiality_filter(model):
    model.issues.upsert_issue("Important", IssueType.CLAIM, materiality=0.8)
    model.issues.upsert_issue("Minor", IssueType.CLAIM, materiality=0.1)

    above_threshold = model.issues.get_open_issues(min_materiality=0.5)
    assert len(above_threshold) == 1
    assert above_threshold[0]["title"] == "Important"


# ---------------------------------------------------------------------------
# QueryMatterContext integration
# ---------------------------------------------------------------------------

def test_build_query_context_includes_issues(model):
    model.issues.upsert_issue("Issue A", IssueType.CLAIM, materiality=0.8, salience=0.9)
    model.issues.upsert_issue("Issue B", IssueType.DAMAGES, materiality=0.5, salience=0.5)

    ctx = model.build_query_context()
    assert len(ctx.open_issues) == 2
    assert ctx.weakest_issue_id is not None

    # Weakest issue = highest-priority issue with least evidentiary support.
    # Neither issue has supporting assertions, so coverage_fraction=0 for both.
    # Issue A: 0.8 * 0.9 * (1-0) = 0.72 > Issue B: 0.5 * 0.5 * (1-0) = 0.25
    # Issue A is the most important uncovered issue → it gets priority for retrieval.
    weakest = next(i for i in ctx.open_issues if i["id"] == ctx.weakest_issue_id)
    assert weakest["title"] == "Issue A"


def test_stats_includes_issue_count(model):
    model.issues.upsert_issue("Issue A", IssueType.CLAIM)
    stats = model.stats()
    assert stats["open_issue_count"] == 1
    assert "actor_count" in stats


# ---------------------------------------------------------------------------
# record_fact() issue linking via MatterRuntimeAdapter (SO-4)
# ---------------------------------------------------------------------------

def test_record_fact_links_assertion_to_issue(model):
    """record_fact(issue_id=...) must create an assertion→issue link (SO-4)."""
    issue_id, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM)
    run_id = model.start_run("test run")
    adapter = MatterRuntimeAdapter(model, run_id)

    a_id = adapter.record_fact(
        "Defendant failed to make payment on January 15.",
        document_id="complaint.pdf",
        issue_id=issue_id,
    )

    linked = model.issues.get_assertions_for_issue(issue_id)
    assert len(linked) == 1
    assert linked[0]["id"] == a_id
    assert linked[0]["relation_type"] == "supports"


def test_record_fact_without_issue_id_does_not_link(model):
    """record_fact() with no issue_id must not create any issue links."""
    issue_id, _ = model.issues.upsert_issue("Damages claim", IssueType.DAMAGES)
    run_id = model.start_run("test run")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.record_fact("Some extracted fact.", document_id="doc.pdf")

    linked = model.issues.get_assertions_for_issue(issue_id)
    assert len(linked) == 0


def test_record_fact_issue_link_is_idempotent(model):
    """Calling record_fact twice with same text + issue_id must not duplicate links."""
    issue_id, _ = model.issues.upsert_issue("Liability", IssueType.CLAIM)
    run_id = model.start_run("test run")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Same proposition text will upsert to the same assertion_id
    adapter.record_fact("Defendant admitted fault.", document_id="depo.pdf", issue_id=issue_id)
    adapter.record_fact("Defendant admitted fault.", document_id="depo.pdf", issue_id=issue_id)

    linked = model.issues.get_assertions_for_issue(issue_id)
    assert len(linked) == 1


# ---------------------------------------------------------------------------
# SO-4: Multi-claim evidence coverage — the key correctness test
# ---------------------------------------------------------------------------

def test_multi_claim_evidence_coverage_and_weakest_prioritization(model):
    """Given 3 active claims, the system reports per-claim coverage and
    directs retrieval focus toward the weakest-covered claim (SO-4).

    Claim A: 2 supporting assertions — well-covered
    Claim B: 1 supporting assertion — partially covered
    Claim C: 0 supporting assertions — proof gap exposed

    build_query_context() must:
    - report correct assertion counts per claim
    - set weakest_issue_id to the most important uncovered claim
    """
    claim_a, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM, materiality=0.9, salience=0.9)
    claim_b, _ = model.issues.upsert_issue("Damages amount", IssueType.DAMAGES, materiality=0.8, salience=0.8)
    claim_c, _ = model.issues.upsert_issue("Causation", IssueType.CLAIM, materiality=0.95, salience=0.95)

    # claim_a gets 2 supporting assertions
    a1 = add_assertion(model, "Contract signed and delivered.", "contract.pdf")
    a2 = add_assertion(model, "Defendant failed to perform.", "complaint.pdf")
    model.issues.link_assertion(a1, claim_a, "supports")
    model.issues.link_assertion(a2, claim_a, "supports")

    # claim_b gets 1 supporting assertion
    a3 = add_assertion(model, "Expert estimates $500,000 in lost revenue.", "expert_report.pdf")
    model.issues.link_assertion(a3, claim_b, "supports")

    # claim_c gets 0 assertions (proof gap)

    ctx = model.build_query_context()

    # Verify all 3 claims are included
    assert len(ctx.open_issues) == 3

    # Verify coverage counts via assertion lookups
    a_linked = model.issues.get_assertions_for_issue(claim_a)
    b_linked = model.issues.get_assertions_for_issue(claim_b)
    c_linked = model.issues.get_assertions_for_issue(claim_c)
    assert len(a_linked) == 2
    assert len(b_linked) == 1
    assert len(c_linked) == 0

    # Weakest issue should be claim_c (zero coverage, highest materiality*salience)
    # claim_c: 0.95 * 0.95 * (1 - 0) = 0.9025
    # claim_a: 0.9 * 0.9 * (1 - coverage>0) — coverage dampens it
    assert ctx.weakest_issue_id == claim_c, (
        "claim_c has zero evidence and highest materiality — must be prioritized for retrieval"
    )


# ---------------------------------------------------------------------------
# Issue predicates (SO-4)
# ---------------------------------------------------------------------------

def test_add_predicate_persists(model):
    """add_predicate() stores a testable element that get_predicates() returns."""
    issue_id, _ = model.issues.upsert_issue(
        title="Breach of contract",
        issue_type=IssueType.CLAIM,
    )
    pred_id = model.issues.add_predicate(
        issue_id=issue_id,
        description="Contract existence and terms",
    )
    assert pred_id

    preds = model.issues.get_predicates(issue_id)
    assert len(preds) == 1
    assert preds[0]["description"] == "Contract existence and terms"
    assert preds[0]["status"] == "open"
    assert preds[0]["issue_id"] == issue_id


def test_multiple_predicates_returned_in_order(model):
    """Multiple predicates for an issue are returned ordered by creation."""
    issue_id, _ = model.issues.upsert_issue(
        title="Breach of service agreement",
        issue_type=IssueType.CLAIM,
    )
    descs = [
        "Agreement existence and terms",
        "Defendant's obligation under agreement",
        "Defendant's failure to perform",
        "Resulting damages",
    ]
    for d in descs:
        model.issues.add_predicate(issue_id=issue_id, description=d)

    preds = model.issues.get_predicates(issue_id)
    assert len(preds) == 4
    assert [p["description"] for p in preds] == descs


def test_predicates_are_issue_scoped(model):
    """Predicates are scoped to their issue — other issues return empty."""
    id_a, _ = model.issues.upsert_issue("Issue A", IssueType.CLAIM)
    id_b, _ = model.issues.upsert_issue("Issue B", IssueType.DEFENSE)

    model.issues.add_predicate(id_a, "Only for A")

    assert len(model.issues.get_predicates(id_a)) == 1
    assert len(model.issues.get_predicates(id_b)) == 0


def test_add_predicate_idempotent(model):
    """add_predicate() with same (issue_id, description) must not create duplicate rows."""
    issue_id, _ = model.issues.upsert_issue("Claim A", IssueType.CLAIM)
    id1 = model.issues.add_predicate(issue_id, "Element one")
    id2 = model.issues.add_predicate(issue_id, "Element one")  # duplicate

    assert id1 == id2, "Second call must return the existing predicate ID"
    assert len(model.issues.get_predicates(issue_id)) == 1


def test_add_predicates_batch_idempotent(model):
    """add_predicates_batch() called twice must not duplicate rows."""
    issue_id, _ = model.issues.upsert_issue("Claim batch", IssueType.CLAIM)
    descs = ["Element A", "Element B", "Element C"]
    ids_first = model.issues.add_predicates_batch(issue_id, descs)
    ids_second = model.issues.add_predicates_batch(issue_id, descs)

    assert set(ids_first) == set(ids_second), "Second batch must return same IDs"
    assert len(model.issues.get_predicates(issue_id)) == 3


def test_get_predicates_limit(model):
    """get_predicates(limit=N) returns at most N rows (DB-level bound)."""
    issue_id, _ = model.issues.upsert_issue("Claim limit", IssueType.CLAIM)
    for i in range(5):
        model.issues.add_predicate(issue_id, f"Element {i}")

    bounded = model.issues.get_predicates(issue_id, limit=3)
    assert len(bounded) == 3

    all_preds = model.issues.get_predicates(issue_id)
    assert len(all_preds) == 5
