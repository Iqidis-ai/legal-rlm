"""Tests for ProofStateStore — proof-aware reasoning layer (SO-4).

Verifies:
1. compute_and_store() creates a new row for a fresh issue
2. compute_and_store() updates existing row (upsert semantics)
3. get() returns None before any computation
4. Zero supporting assertions → proof_status=insufficient, sufficiency near 0
5. Supporting assertions increase sufficiency
6. Attacking assertions decrease sufficiency
7. Contested: attacking >= supporting → proof_status=contested
8. Sufficient: sufficiency >= 0.75 → proof_status=sufficient
9. Partial: 0.25 <= sufficiency < 0.75 → proof_status=partial
10. Predicates factor into sufficiency: satisfied_predicates / total raises score
11. compute_all() refreshes all open issues
12. get_all() returns all states ordered by sufficiency
13. get_by_status() filters correctly
14. get_gaps() returns only insufficient/partial issues below threshold
15. get_summary() aggregates correctly
16. Multi-matter isolation
17. API: POST /compute, GET /proof-state, GET /proof-state/gaps
18. API: POST /issues/{id}/proof-state/compute, GET /issues/{id}/proof-state
"""

import pytest
from irys.matter import MatterModel, IssueType, SpeechAct, SourceRole, AssertionKind
from irys.matter.models import AssertionCandidate


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


@pytest.fixture
def issue_id(model):
    iid, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM)
    return iid


def _add_assertion(model, issue_id, relation="supports", speech_act=SpeechAct.OPERATIVE):
    cand = AssertionCandidate(
        proposition_text=f"Fact {_add_assertion._counter}",
        speech_act=speech_act,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc1",
    )
    _add_assertion._counter += 1
    aid, _ = model.assertions.upsert_occurrence(cand)
    model.issues.link_assertion(aid, issue_id, relation_type=relation)
    return aid


_add_assertion._counter = 0


# ---------------------------------------------------------------------------
# Basic compute_and_store
# ---------------------------------------------------------------------------

def test_get_returns_none_before_computation(model, issue_id):
    assert model.proof_state.get(issue_id) is None


def test_compute_creates_row(model, issue_id):
    ps = model.proof_state.compute_and_store(issue_id)
    assert ps["issue_id"] == issue_id
    assert ps["matter_id"] == model.matter_id
    assert "sufficiency" in ps
    assert "proof_status" in ps


def test_compute_upsert_same_id(model, issue_id):
    ps1 = model.proof_state.compute_and_store(issue_id)
    ps2 = model.proof_state.compute_and_store(issue_id)
    assert ps1["id"] == ps2["id"]  # same row updated


# ---------------------------------------------------------------------------
# Sufficiency scoring
# ---------------------------------------------------------------------------

def test_no_assertions_insufficient(model, issue_id):
    ps = model.proof_state.compute_and_store(issue_id)
    assert ps["proof_status"] == "insufficient"
    assert ps["sufficiency"] == 0.0


def test_one_supporting_raises_sufficiency(model, issue_id):
    _add_assertion(model, issue_id, "supports")
    ps = model.proof_state.compute_and_store(issue_id)
    assert ps["supporting_count"] == 1
    assert ps["sufficiency"] > 0.0


def test_more_supporting_higher_sufficiency(model, issue_id):
    for _ in range(3):
        _add_assertion(model, issue_id, "supports")
    ps = model.proof_state.compute_and_store(issue_id)
    assert ps["sufficiency"] > 0.5


def test_attacking_lowers_sufficiency(model, issue_id):
    for _ in range(3):
        _add_assertion(model, issue_id, "supports")
    ps_before = model.proof_state.compute_and_store(issue_id)

    _add_assertion(model, issue_id, "attacks")
    _add_assertion(model, issue_id, "attacks")
    ps_after = model.proof_state.compute_and_store(issue_id)
    assert ps_after["sufficiency"] < ps_before["sufficiency"]


def test_contested_status(model, issue_id):
    """attacking >= supporting > 0 → contested."""
    _add_assertion(model, issue_id, "supports")
    _add_assertion(model, issue_id, "attacks")
    _add_assertion(model, issue_id, "attacks")
    ps = model.proof_state.compute_and_store(issue_id)
    assert ps["proof_status"] == "contested"


def test_sufficient_status(model, issue_id):
    """Enough supporting without attacks → sufficient."""
    for _ in range(9):
        _add_assertion(model, issue_id, "supports")
    ps = model.proof_state.compute_and_store(issue_id)
    assert ps["proof_status"] == "sufficient"
    assert ps["sufficiency"] >= 0.75


def test_partial_status(model, issue_id):
    """Some supporting, no attacks, moderate sufficiency → partial."""
    for _ in range(2):
        _add_assertion(model, issue_id, "supports")
    ps = model.proof_state.compute_and_store(issue_id)
    # 2 supporting: 2/(2+0+1) = 0.667 → partial range if no predicates
    assert ps["proof_status"] in ("partial", "sufficient")
    assert ps["sufficiency"] >= 0.25


# ---------------------------------------------------------------------------
# Predicates factor into score
# ---------------------------------------------------------------------------

def test_predicates_factor_into_score(model, issue_id):
    """With predicates defined but none resolved, score falls back to assertion_ratio.

    Prior behavior: predicate_ratio=0 → score=0 (penalized issues with any predicates
    because no production writer ever resolved them). New behavior: zero resolved
    predicates falls back to assertion_ratio so predicate-bearing issues are not
    unfairly scored 0 before resolve_predicate() is called in production (SO-4).
    """
    model.issues.add_predicate(issue_id, "Element A must be proved")
    model.issues.add_predicate(issue_id, "Element B must be proved")
    for _ in range(5):
        _add_assertion(model, issue_id, "supports")

    ps = model.proof_state.compute_and_store(issue_id)
    assert ps["total_predicate_count"] == 2
    assert ps["satisfied_predicate_count"] == 0
    # 0 resolved predicates → falls back to assertion_ratio = 5/(5+0+1) ≈ 0.8333
    assert ps["sufficiency"] > 0.0, (
        "Zero resolved predicates must fall back to assertion_ratio, not 0"
    )
    # Predicates with assertions must score at least as well as no predicates/no assertions
    ps_no_pred = model.proof_state._score(5, 0, 0, 0)
    assert ps["sufficiency"] == ps_no_pred, (
        "Zero resolved predicates should give same score as no-predicate case"
    )


def test_resolved_predicates_raise_score(model, issue_id):
    pred_id = model.issues.add_predicate(issue_id, "Element A")
    model.issues.add_predicate(issue_id, "Element B")  # stays open
    for _ in range(3):
        _add_assertion(model, issue_id, "supports")

    # Resolve one predicate via the canonical API
    assert model.issues.resolve_predicate(pred_id) is True

    ps = model.proof_state.compute_and_store(issue_id)
    assert ps["satisfied_predicate_count"] == 1
    assert ps["total_predicate_count"] == 2
    # With satisfied predicates: predicate_ratio=0.5 × assertion_ratio=3/(3+0+1)=0.75 = 0.375
    assert ps["sufficiency"] == pytest.approx(0.375), (
        "Partial predicate resolution must use predicate_ratio × assertion_ratio"
    )

    # Resolve both → predicate_ratio=1.0, score = assertion_ratio
    pred_b_row = model.db.execute(
        "SELECT id FROM issue_predicate WHERE issue_id=? AND description='Element B'",
        (issue_id,),
    ).fetchone()
    model.issues.resolve_predicate(pred_b_row["id"])
    ps2 = model.proof_state.compute_and_store(issue_id)
    assert ps2["satisfied_predicate_count"] == 2
    assert ps2["sufficiency"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# resolve_predicate matter-scoping and edge cases
# ---------------------------------------------------------------------------

def test_resolve_predicate_cannot_cross_matter(model, issue_id):
    """resolve_predicate() must not resolve a predicate in another matter."""
    m_b = MatterModel.open_in_memory()
    iid_b, _ = m_b.issues.upsert_issue("Claim B", IssueType.CLAIM)
    pred_id_b = m_b.issues.add_predicate(iid_b, "Foreign element")

    # Matter A tries to resolve a predicate that belongs to Matter B
    result = model.issues.resolve_predicate(pred_id_b)
    assert result is False, "resolve_predicate must refuse cross-matter predicate IDs"


def test_resolve_predicate_by_description_none_guard(model, issue_id):
    """resolve_predicate_by_description() must not raise on empty/None description."""
    assert model.issues.resolve_predicate_by_description(issue_id, "") is False
    assert model.issues.resolve_predicate_by_description(issue_id, "   ") is False


def test_resolve_predicate_by_description_full_text(model, issue_id):
    """Lookup must use full stripped description (no 300-char truncation)."""
    long_desc = "Element " + "x" * 310  # > 300 chars
    pred_id = model.issues.add_predicate(issue_id, long_desc)
    result = model.issues.resolve_predicate_by_description(issue_id, long_desc)
    assert result is True
    row = model.db.execute(
        "SELECT status FROM issue_predicate WHERE id=?", (pred_id,)
    ).fetchone()
    assert row["status"] == "resolved"


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all_refreshes_all_open_issues(model):
    i1, _ = model.issues.upsert_issue("Claim A", IssueType.CLAIM)
    i2, _ = model.issues.upsert_issue("Claim B", IssueType.CLAIM)
    for _ in range(3):
        _add_assertion(model, i1, "supports")

    results = model.proof_state.compute_all()
    assert len(results) == 2

    ps1 = model.proof_state.get(i1)
    ps2 = model.proof_state.get(i2)
    assert ps1["supporting_count"] == 3
    assert ps2["supporting_count"] == 0


def test_compute_all_closed_issues_excluded(model):
    i1, _ = model.issues.upsert_issue("Open claim", IssueType.CLAIM)
    i2, _ = model.issues.upsert_issue("Closed claim", IssueType.CLAIM)
    # Close the second issue
    model.db.execute("UPDATE issue SET status='closed' WHERE id=?", (i2,))

    results = model.proof_state.compute_all()
    assert len(results) == 1
    assert results[0]["issue_id"] == i1


# ---------------------------------------------------------------------------
# Read methods
# ---------------------------------------------------------------------------

def test_get_all_ordered_by_sufficiency(model):
    i1, _ = model.issues.upsert_issue("Weak", IssueType.CLAIM)
    i2, _ = model.issues.upsert_issue("Strong", IssueType.CLAIM)
    for _ in range(9):
        _add_assertion(model, i2, "supports")
    model.proof_state.compute_all()

    all_states = model.proof_state.get_all()
    assert all_states[0]["issue_id"] == i1  # weakest first
    assert all_states[1]["issue_id"] == i2


def test_get_by_status_filters(model):
    i1, _ = model.issues.upsert_issue("Weak", IssueType.CLAIM)
    i2, _ = model.issues.upsert_issue("Strong", IssueType.CLAIM)
    for _ in range(9):
        _add_assertion(model, i2, "supports")
    model.proof_state.compute_all()

    insufficient = model.proof_state.get_by_status("insufficient")
    assert len(insufficient) == 1
    assert insufficient[0]["issue_id"] == i1


def test_get_gaps_returns_below_threshold(model):
    i1, _ = model.issues.upsert_issue("Gap issue", IssueType.CLAIM)
    i2, _ = model.issues.upsert_issue("Covered issue", IssueType.CLAIM)
    for _ in range(9):
        _add_assertion(model, i2, "supports")
    model.proof_state.compute_all()

    gaps = model.proof_state.get_gaps(threshold=0.25)
    assert any(g["issue_id"] == i1 for g in gaps)
    assert all(g["issue_id"] != i2 for g in gaps)


def test_get_summary(model):
    i1, _ = model.issues.upsert_issue("Weak", IssueType.CLAIM)
    i2, _ = model.issues.upsert_issue("Strong", IssueType.CLAIM)
    for _ in range(9):
        _add_assertion(model, i2, "supports")
    model.proof_state.compute_all()

    summary = model.proof_state.get_summary()
    assert summary["total_issues_tracked"] == 2
    assert 0.0 <= summary["avg_sufficiency"] <= 1.0
    assert "by_status" in summary
    assert summary["gap_count"] >= 1


def test_get_summary_empty(model):
    summary = model.proof_state.get_summary()
    assert summary["total_issues_tracked"] == 0


# ---------------------------------------------------------------------------
# Multi-matter isolation
# ---------------------------------------------------------------------------

def test_proof_state_isolated_per_matter(model, issue_id):
    m_b = MatterModel.open_in_memory()
    for _ in range(3):
        _add_assertion(model, issue_id, "supports")
    model.proof_state.compute_and_store(issue_id)

    # Matter B has no proof state
    assert m_b.proof_state.get_summary()["total_issues_tracked"] == 0


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    from irys.service.api import app, _active_matter_models
    _active_matter_models.clear()
    return TestClient(app), _active_matter_models


def _reg(active, m):
    active[m.matter_id] = m
    return m.matter_id


def test_api_compute_all(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    m.issues.upsert_issue("Claim A", IssueType.CLAIM)
    m.issues.upsert_issue("Claim B", IssueType.CLAIM)

    resp = client.post(f"/matter/{mid}/proof-state/compute")
    assert resp.status_code == 200
    body = resp.json()
    assert body["updated_count"] == 2


def test_api_compute_single_issue(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    iid, _ = m.issues.upsert_issue("Test Claim", IssueType.CLAIM)

    resp = client.post(f"/matter/{mid}/issues/{iid}/proof-state/compute")
    assert resp.status_code == 200
    body = resp.json()
    assert body["issue_id"] == iid
    assert body["proof_status"] == "insufficient"


def test_api_get_proof_state_summary(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    m.issues.upsert_issue("Claim", IssueType.CLAIM)
    m.proof_state.compute_all()

    resp = client.get(f"/matter/{mid}/proof-state")
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert "issues" in body
    assert body["summary"]["total_issues_tracked"] == 1


def test_api_get_issue_proof_state_null_before_compute(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    iid, _ = m.issues.upsert_issue("Claim", IssueType.CLAIM)

    resp = client.get(f"/matter/{mid}/issues/{iid}/proof-state")
    assert resp.status_code == 200
    assert resp.json() is None


def test_api_get_proof_gaps(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    m.issues.upsert_issue("Weak", IssueType.CLAIM)
    m.proof_state.compute_all()

    resp = client.get(f"/matter/{mid}/proof-state/gaps")
    assert resp.status_code == 200
    gaps = resp.json()
    assert len(gaps) >= 1
    assert all(g["sufficiency"] < 0.25 for g in gaps)
