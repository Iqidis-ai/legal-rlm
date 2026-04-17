"""P0.3: Review Queue and Verification API (SO-3).

Tests the attorney-facing review workflow:
- get_review_queue prioritizes issue-linked candidates (with proof gap
  first), then quant_fact, then authority, then everything else.
- verify_target / reject_target emit ledger audit events and recompute
  proof state for affected issues.
- bulk_verify_by_document promotes every candidate assertion sourced
  from a given document in one operation.
- Promotion to verified still requires a human reviewer — automation
  cannot bulk-verify.
"""

import pytest

from irys.matter import MatterModel
from irys.matter.enums import (
    IssueType, LedgerEventType, VerificationStatus, VerificationTargetKind,
)
from irys.matter.runtime import MatterRuntimeAdapter


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _adapter(m):
    run_id = m.start_run("review queue test")
    return MatterRuntimeAdapter(m, run_id), run_id


# ---------------------------------------------------------------------------
# Review queue: priority ordering
# ---------------------------------------------------------------------------

def test_review_queue_puts_gap_blocked_issue_assertions_first(model):
    """Priority bucket 0: assertions linked to an open issue with a
    known missing_issue_predicate gap."""
    adapter, _ = _adapter(model)
    iid_gap, _ = model.issues.upsert_issue(
        "Gap-blocked claim", IssueType.CLAIM, materiality=0.7,
    )
    iid_nogap, _ = model.issues.upsert_issue(
        "Supported claim", IssueType.CLAIM, materiality=0.9,
    )
    # Link one assertion to each issue.
    aid_gap = adapter.record_fact(
        "Gap issue support", "doc.pdf",
        issue_id=iid_gap, issue_link_type="supports",
    )
    aid_nogap = adapter.record_fact(
        "No-gap issue support", "doc.pdf",
        issue_id=iid_nogap, issue_link_type="supports",
    )
    # Mark the first issue as gap-blocked.
    from irys.matter.enums import GapType
    model.gaps.record(
        gap_type=GapType.MISSING_ISSUE_PREDICATE,
        description="Missing predicate for gap issue",
        affected_type="issue", affected_id=iid_gap,
        materiality=0.9,
    )
    q = model.get_review_queue(limit=10)
    # Both assertions appear; gap-blocked one is at bucket 0, the
    # other at bucket 1.
    by_id = {r["target_id"]: r for r in q}
    assert aid_gap in by_id
    assert aid_nogap in by_id
    assert by_id[aid_gap]["priority_bucket"] < by_id[aid_nogap]["priority_bucket"]


def test_review_queue_surfaces_quant_and_authority(model):
    """Quant facts and authorities live in buckets 2 and 3 — lower
    priority than issue-linked assertions but still visible to the
    reviewer."""
    adapter, _ = _adapter(model)
    model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    adapter.record_quant(quant_kind="amount", raw_text="$1000 invoice", amount_value=1000.0)
    model.authority.upsert(
        citation="Smith v. Jones, 1 F.3d 100 (9th Cir. 2020)",
        authority_type="case",
    )
    q = model.get_review_queue(limit=20)
    kinds = {row["target_kind"] for row in q}
    assert "quant_fact" in kinds
    assert "authority" in kinds
    # Quant precedes authority in the priority ordering.
    quants = [r for r in q if r["target_kind"] == "quant_fact"]
    auths = [r for r in q if r["target_kind"] == "authority"]
    assert quants[0]["priority_bucket"] <= auths[0]["priority_bucket"]


def test_review_queue_target_kind_filter(model):
    """Caller can narrow the queue to one kind (e.g. only assertions)."""
    adapter, _ = _adapter(model)
    adapter.record_fact("assertion A", "doc.pdf")
    adapter.record_quant(quant_kind="amount", raw_text="$5", amount_value=5.0)
    only_assertions = model.get_review_queue(
        limit=10, target_kind="assertion",
    )
    assert all(r["target_kind"] == "assertion" for r in only_assertions)
    only_quants = model.get_review_queue(
        limit=10, target_kind="quant_fact",
    )
    assert all(r["target_kind"] == "quant_fact" for r in only_quants)


# ---------------------------------------------------------------------------
# verify_target / reject_target: audit events + proof recompute
# ---------------------------------------------------------------------------

def test_verify_target_records_ledger_event_and_recomputes_proof(model):
    """Verifying a candidate assertion linked to an issue must
    emit a ledger event AND move supporting_count from candidate
    lane to verified lane in the coverage report."""
    adapter, run_id = _adapter(model)
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    aid = adapter.record_fact(
        "Some fact", "doc.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    # Also verify the companion edge so the verified lane is reachable.
    edge = model.db.execute(
        """SELECT id FROM evidence_edge
           WHERE matter_id=? AND source_id=? AND target_id=?""",
        (model.matter_id, aid, iid),
    ).fetchone()
    model.verify_target(
        "assertion", aid,
        reviewed_by_kind="user", reviewed_by_id="reviewer1",
        run_id=run_id,
    )
    model.verify_target(
        "evidence_edge", edge["id"],
        reviewed_by_kind="user", reviewed_by_id="reviewer1",
        run_id=run_id,
    )
    # Coverage report reflects the new verified lane count.
    report = model.get_issue_coverage_report(policy_audience="internal")
    entry = next(r for r in report if r["id"] == iid)
    assert entry["verified_supporting_count"] == 1
    # Ledger has two promotion events linked to the right targets.
    events = model.ledger.get_events(run_id)
    verified_events = [
        e for e in events
        if "Verified" in (e.get("summary") or "")
    ]
    assert len(verified_events) >= 2
    assert any(aid in (e.get("changed_object_id") or "") for e in verified_events)


def test_reject_target_drops_support_and_logs(model):
    """Rejecting a candidate supporting assertion must drop it from
    the coverage report (stale+rejected filter) AND append a ledger
    event."""
    adapter, run_id = _adapter(model)
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    aid = adapter.record_fact(
        "Bad fact", "doc.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    model.reject_target(
        "assertion", aid,
        reviewed_by_kind="user", reviewed_by_id="reviewer1",
        rejection_reason="fabricated",
        run_id=run_id,
    )
    report = model.get_issue_coverage_report(policy_audience="internal")
    entry = next(r for r in report if r["id"] == iid)
    assert entry["supporting_count"] == 0
    assert entry["verified_supporting_count"] == 0
    events = model.ledger.get_events(run_id)
    assert any(
        "Rejected" in (e.get("summary") or "")
        and aid in (e.get("changed_object_id") or "")
        for e in events
    )


def test_reject_requires_rejection_reason(model):
    """Rejection with empty reason must raise."""
    adapter, _ = _adapter(model)
    aid = adapter.record_fact("some fact", "doc.pdf")
    with pytest.raises(ValueError):
        model.reject_target(
            "assertion", aid,
            reviewed_by_kind="user", reviewed_by_id="r",
            rejection_reason="",
        )


def test_bulk_verify_by_document_promotes_all_candidates(model):
    """bulk_verify_by_document must promote every candidate
    assertion sourced from the named document in one operation."""
    adapter, run_id = _adapter(model)
    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.7)
    aids = [
        adapter.record_fact(
            f"Doc fact {i}", "docA.pdf",
            issue_id=iid, issue_link_type="supports",
        )
        for i in range(3)
    ]
    # One fact from a different document — must NOT be verified.
    aid_other = adapter.record_fact("Other fact", "docB.pdf")
    ids = model.bulk_verify_by_document(
        "docA.pdf",
        reviewed_by_kind="user", reviewed_by_id="reviewer1",
        run_id=run_id,
    )
    assert len(ids) == 3
    for aid in aids:
        vs = model.verification.get("assertion", aid)
        assert vs is not None and vs["status"] == "verified"
    other_vs = model.verification.get("assertion", aid_other)
    assert other_vs is not None
    assert other_vs["status"] == "candidate", (
        "bulk-verify must not touch assertions outside the document scope"
    )


def test_bulk_verify_rejects_automation_reviewer(model):
    """Promotion to verified — including via bulk — must require a
    human reviewer."""
    adapter, _ = _adapter(model)
    adapter.record_fact("a fact", "docA.pdf")
    with pytest.raises(ValueError):
        model.bulk_verify_by_document(
            "docA.pdf",
            reviewed_by_kind="system", reviewed_by_id=None,
        )
