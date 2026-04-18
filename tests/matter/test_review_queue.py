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


def test_review_queue_surfaces_predicates(model):
    """Codex P0.3 review fix #1: candidate issue_predicate rows must
    land in bucket 3, ordered by parent issue's materiality × salience,
    not in the generic bucket 6 with score 0."""
    from irys.matter.enums import IssueType

    iid, _ = model.issues.upsert_issue(
        "Breach", IssueType.CLAIM, materiality=0.9, salience=0.8,
    )
    pid = model.issues.add_predicate(iid, "Element: duty")
    q = model.get_review_queue(limit=20)
    pred_rows = [r for r in q if r["target_kind"] == "issue_predicate"]
    assert len(pred_rows) >= 1
    found = next(r for r in pred_rows if r["target_id"] == pid)
    assert found["priority_bucket"] == 3
    assert found["priority_score"] > 0.0
    # Predicate text surfaces as context.
    assert found["predicate_description"] == "Element: duty"


def test_review_queue_surfaces_contradicted_assertions(model):
    """Codex P0.3 review fix #1: assertions that appear on either
    side of an attacks/contradicts link get bucket 1 priority, so
    reviewers see live conflicts near the top of the queue."""
    from irys.matter.enums import AssertionLinkType

    adapter, _ = _adapter(model)
    aid_a = adapter.record_fact("Defendant signed contract.", "doc.pdf")
    aid_b = adapter.record_fact("Defendant never signed.", "doc.pdf")
    model.assertions.link(aid_a, aid_b, AssertionLinkType.CONTRADICTS)

    q = model.get_review_queue(limit=20)
    by_id = {r["target_id"]: r for r in q}
    assert aid_a in by_id and aid_b in by_id
    # Contradicted assertions land in bucket 1 (or 0 if also gap-blocked).
    assert by_id[aid_a]["priority_bucket"] <= 1
    assert by_id[aid_b]["priority_bucket"] <= 1


def test_review_scope_propagates_into_verification_row(model):
    """Codex P0.3 review fix #2: review_scope on verify_target must
    land in the verification_state row, not be silently swallowed."""
    adapter, _ = _adapter(model)
    aid = adapter.record_fact("Fact to record-truth verify", "doc.pdf")
    model.verify_target(
        "assertion", aid,
        reviewed_by_kind="attorney", reviewed_by_id="a1",
        review_scope="record_truth",
    )
    vs = model.verification.get("assertion", aid)
    assert vs["review_scope"] == "record_truth"


def test_reject_authority_triggers_issues_helper(model):
    """Codex P0.3 review fix #3: rejecting non-assertion targets
    (authority) still passes through _issues_affected_by_target
    without crashing. Coverage recompute is a no-op when the
    optional authority_issue_link table is absent, but the rejection
    itself must succeed and emit a verification_event."""
    adapter, run_id = _adapter(model)
    aid = model.authority.upsert("Smith v. Jones, 1 F.3d 100")[0]
    model.reject_target(
        "authority", aid,
        reviewed_by_kind="attorney", reviewed_by_id="a1",
        rejection_reason="wrong jurisdiction",
        run_id=run_id,
    )
    vs = model.verification.get("authority", aid)
    assert vs is not None
    assert vs["status"] == "rejected"


def test_reject_quant_fact_traces_through_to_issue(model):
    """Codex P0.3 review fix #3: a quant_fact is attributable to a
    parent assertion; rejecting the quant must trace back to the
    assertion's issues for proof recompute."""
    from irys.matter.enums import IssueType

    iid, _ = model.issues.upsert_issue("Damages", IssueType.CLAIM, materiality=0.7)
    adapter, run_id = _adapter(model)
    aid = adapter.record_fact(
        "Invoice total $1000", "invoice.pdf",
        issue_id=iid, issue_link_type="supports",
    )
    edge = model.db.execute(
        "SELECT id FROM evidence_edge WHERE source_id=? AND target_id=?",
        (aid, iid),
    ).fetchone()
    from irys.matter.enums import VerificationTargetKind
    model.verification.verify(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind="user", reviewed_by_id="r1",
    )
    model.verification.verify(
        VerificationTargetKind.EVIDENCE_EDGE, edge["id"],
        reviewed_by_kind="user", reviewed_by_id="r1",
    )
    # Record a quant_fact tied to this assertion.
    qid = adapter.record_quant(
        quant_kind="amount", raw_text="$1000 total",
        amount_value=1000.0, assertion_id=aid,
    )
    # Rejecting the quant should traverse quant → parent assertion →
    # affected issues and recompute proof state. The issue the
    # assertion supports is returned by _issues_affected_by_target.
    affected = model._issues_affected_by_target("quant_fact", qid)
    assert iid in affected


def test_bulk_verify_by_document_matches_basename_and_normalized(model):
    """Codex P0.3 review fix #2: document_ref matching must succeed
    for basename and slash-normalized paths, not just the exact raw
    stored string."""
    adapter, run_id = _adapter(model)
    # Record a fact whose occurrence stores a Windows-style path.
    adapter.record_fact("Claim text", "contracts/msa.pdf")
    # Caller passes the basename.
    ids = model.bulk_verify_by_document(
        "msa.pdf",
        reviewed_by_kind="user", reviewed_by_id="r1",
    )
    assert len(ids) == 1


def test_bulk_verify_by_span_promotes_span_scoped(model):
    """Codex P0.3 AC #3 coverage: span-set bulk variant must verify
    every candidate assertion whose occurrence carries the given
    span_id, and leave other assertions alone."""
    adapter, run_id = _adapter(model)
    # Each (document_id, span_id) is a unique occurrence slot — a
    # second write to the same slot overwrites the first. Use
    # different docs to get two distinct occurrences sharing a span_id.
    adapter.record_fact(
        "Fact A in signature span", "docA.pdf", span_id="sig-block-1",
    )
    adapter.record_fact(
        "Fact B in signature span", "docB.pdf", span_id="sig-block-1",
    )
    adapter.record_fact(
        "Fact C elsewhere", "docC.pdf", span_id="other-span",
    )
    ids = model.bulk_verify_by_span(
        "sig-block-1",
        reviewed_by_kind="user", reviewed_by_id="r1",
        run_id=run_id,
    )
    assert len(ids) == 2
    # Non-matching span must not have been promoted.
    other_aid = model.db.execute(
        "SELECT ao.assertion_id FROM assertion_occurrence ao WHERE ao.span_id='other-span'",
    ).fetchone()["assertion_id"]
    vs = model.verification.get("assertion", other_aid)
    assert vs is None or vs["status"] == "candidate"


def test_get_verification_events_returns_history(model):
    """Codex P0.3 AC #5 + fix #3: verification_event rows must be
    reachable via a matter-level API, not just the store-local
    list_events(). Every transition must be auditable."""
    adapter, run_id = _adapter(model)
    aid = adapter.record_fact("verified claim", "doc.pdf")
    model.verify_target(
        "assertion", aid,
        reviewed_by_kind="attorney", reviewed_by_id="a1",
        review_note="reviewed on 2026-04-17",
    )
    events = model.get_verification_events("assertion", aid)
    # candidate seed + verified transition = at least 2 events.
    assert len(events) >= 2
    statuses = [e["new_status"] for e in events]
    assert "verified" in statuses
    assert "candidate" in statuses


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
