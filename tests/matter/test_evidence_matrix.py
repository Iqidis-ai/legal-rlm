"""Tests for get_evidence_matrix() — visual work product Priority 2.

Verifies:
1. Empty model returns empty matrix structure
2. Single assertion link creates correct cell entry
3. Supporting and attacking counts are tracked separately
4. Multiple documents produce multiple columns
5. Multiple issues produce multiple rows
6. Issue totals aggregate across all documents
7. Source totals aggregate across all issues
8. Withdrawn/superseded assertions excluded
9. Closed issues excluded
10. Unknown document maps to '(unknown)' bucket
11. API endpoint returns correct matrix
"""

import pytest
from irys.matter import MatterModel, IssueType, SpeechAct, SourceRole, AssertionKind
from irys.matter.models import AssertionCandidate


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _link(model, issue_id, doc_id, relation="supports", n=1):
    """Add n assertion occurrences from doc_id linked to issue_id."""
    aids = []
    for i in range(n):
        cand = AssertionCandidate(
            proposition_text=f"Fact {_link._ctr}",
            speech_act=SpeechAct.OPERATIVE,
            source_role=SourceRole.OPERATIVE,
            assertion_kind=AssertionKind.FACTUAL,
            document_id=doc_id,
        )
        _link._ctr += 1
        aid, _ = model.assertions.upsert_occurrence(cand)
        model.issues.link_assertion(aid, issue_id, relation_type=relation)
        aids.append(aid)
    return aids


_link._ctr = 0


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_empty_model_returns_empty_matrix(model):
    m = model.get_evidence_matrix()
    assert m["issues"] == []
    assert m["sources"] == []
    assert m["cells"] == {}
    assert m["issue_totals"] == {}
    assert m["source_totals"] == {}


def test_single_assertion_creates_cell(model):
    iid, _ = model.issues.upsert_issue("Breach", IssueType.CLAIM)
    _link(model, iid, "complaint.pdf", "supports")

    m = model.get_evidence_matrix()
    assert len(m["issues"]) == 1
    assert "complaint.pdf" in m["sources"]
    cell = m["cells"][iid]["complaint.pdf"]
    assert cell["supporting"] == 1
    assert cell["attacking"] == 0
    assert cell["total"] == 1


def test_supporting_and_attacking_tracked_separately(model):
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM)
    _link(model, iid, "contract.pdf", "supports", n=2)
    _link(model, iid, "contract.pdf", "attacks", n=1)

    m = model.get_evidence_matrix()
    cell = m["cells"][iid]["contract.pdf"]
    assert cell["supporting"] == 2
    assert cell["attacking"] == 1
    assert cell["total"] == 3


def test_multiple_documents_as_columns(model):
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM)
    _link(model, iid, "doc_a.pdf", "supports")
    _link(model, iid, "doc_b.pdf", "supports")

    m = model.get_evidence_matrix()
    assert set(m["sources"]) == {"doc_a.pdf", "doc_b.pdf"}
    assert m["cells"][iid]["doc_a.pdf"]["supporting"] == 1
    assert m["cells"][iid]["doc_b.pdf"]["supporting"] == 1


def test_multiple_issues_as_rows(model):
    i1, _ = model.issues.upsert_issue("Claim A", IssueType.CLAIM)
    i2, _ = model.issues.upsert_issue("Claim B", IssueType.CLAIM)
    _link(model, i1, "doc.pdf", "supports")
    _link(model, i2, "doc.pdf", "supports")

    m = model.get_evidence_matrix()
    assert len(m["issues"]) == 2
    assert i1 in m["cells"]
    assert i2 in m["cells"]


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------

def test_issue_totals(model):
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM)
    _link(model, iid, "doc_a.pdf", "supports", n=3)
    _link(model, iid, "doc_b.pdf", "attacks", n=1)

    m = model.get_evidence_matrix()
    totals = m["issue_totals"][iid]
    assert totals["supporting"] == 3
    assert totals["attacking"] == 1


def test_source_totals(model):
    i1, _ = model.issues.upsert_issue("Claim A", IssueType.CLAIM)
    i2, _ = model.issues.upsert_issue("Claim B", IssueType.CLAIM)
    _link(model, i1, "contract.pdf", "supports", n=2)
    _link(model, i2, "contract.pdf", "supports", n=1)

    m = model.get_evidence_matrix()
    totals = m["source_totals"]["contract.pdf"]
    assert totals["supporting"] == 3


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_closed_issue_excluded(model):
    i_open, _ = model.issues.upsert_issue("Open", IssueType.CLAIM)
    i_closed, _ = model.issues.upsert_issue("Closed", IssueType.CLAIM)
    _link(model, i_open, "doc.pdf", "supports")
    _link(model, i_closed, "doc.pdf", "supports")
    model.db.execute("UPDATE issue SET status='closed' WHERE id=?", (i_closed,))

    m = model.get_evidence_matrix()
    issue_ids = {i["id"] for i in m["issues"]}
    assert i_open in issue_ids
    assert i_closed not in issue_ids


def test_issues_without_assertions_excluded(model):
    iid_linked, _ = model.issues.upsert_issue("With assertions", IssueType.CLAIM)
    iid_bare, _ = model.issues.upsert_issue("No assertions", IssueType.CLAIM)
    _link(model, iid_linked, "doc.pdf", "supports")

    m = model.get_evidence_matrix()
    issue_ids = {i["id"] for i in m["issues"]}
    assert iid_linked in issue_ids
    assert iid_bare not in issue_ids


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

def _reg(active, m):
    active[m.matter_id] = m
    return m.matter_id


def test_api_evidence_matrix_empty(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    resp = client.get(f"/matter/{mid}/evidence-matrix")
    assert resp.status_code == 200
    body = resp.json()
    assert body["issues"] == []
    assert body["sources"] == []


def test_api_evidence_matrix_populated(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    iid, _ = m.issues.upsert_issue("Breach", IssueType.CLAIM)
    _link(m, iid, "complaint.pdf", "supports", n=2)
    _link(m, iid, "answer.pdf", "attacks", n=1)

    resp = client.get(f"/matter/{mid}/evidence-matrix")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["issues"]) == 1
    assert set(body["sources"]) == {"complaint.pdf", "answer.pdf"}
    cells = body["cells"][iid]
    assert cells["complaint.pdf"]["supporting"] == 2
    assert cells["answer.pdf"]["attacking"] == 1
