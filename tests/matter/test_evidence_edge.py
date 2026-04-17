"""Unit tests for EvidenceStore (MVP.3, SO-4).

Covers the upsert idempotency, link_assertion integration, backfill
idempotency, and the proof-state substrate switch between evidence_edge
and the legacy assertion_issue_link.
"""

import uuid
from datetime import datetime, timezone

import pytest

from irys.matter import (
    AssertionKind,
    MatterModel,
    ModelLayer,
    SourceRole,
    SpeechAct,
)
from irys.matter.enums import (
    EvidenceOriginKind,
    EvidenceRelationType,
    IssueType,
    OriginKind,
    ReviewedByKind,
    VerificationTargetKind,
)
from irys.matter.models import AssertionCandidate


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _mkassertion(model, text="fact", doc="d.pdf"):
    cand = AssertionCandidate(
        proposition_text=text,
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        model_layer=ModelLayer.RECORD,
        document_id=doc,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    return aid


def _insert_legacy_link(model, aid, iid, relation_type="supports"):
    """Write an assertion_issue_link row directly, bypassing link_assertion's
    MVP.3 evidence_edge sidecar. Used to simulate pre-migration state."""
    now = datetime.now(timezone.utc).isoformat()
    model.db.execute(
        """INSERT OR IGNORE INTO assertion_issue_link
           (id, assertion_id, issue_id, relation_type, created_at)
           VALUES (?,?,?,?,?)""",
        (uuid.uuid4().hex, aid, iid, relation_type, now),
    )


def test_upsert_edge_idempotent(model):
    aid = _mkassertion(model)
    iid, _ = model.issues.upsert_issue("I", IssueType.CLAIM)
    eid1, new1 = model.evidence.upsert_edge(
        source_kind="assertion", source_id=aid,
        target_kind="issue", target_id=iid,
        relation_type=EvidenceRelationType.SUPPORTS,
    )
    eid2, new2 = model.evidence.upsert_edge(
        source_kind="assertion", source_id=aid,
        target_kind="issue", target_id=iid,
        relation_type=EvidenceRelationType.SUPPORTS,
    )
    assert new1 is True and new2 is False
    assert eid1 == eid2
    rows = model.evidence.list_edges_for_target("issue", iid)
    assert len(rows) == 1


def test_link_assertion_creates_evidence_edge(model):
    aid = _mkassertion(model)
    iid, _ = model.issues.upsert_issue("I", IssueType.CLAIM)
    model.issues.link_assertion(aid, iid, "supports")
    edges = model.evidence.list_edges_for_target("issue", iid)
    assert len(edges) == 1
    assert edges[0]["source_id"] == aid
    assert edges[0]["relation_type"] == "supports"
    assert edges[0]["origin_kind"] == "system_inferred"
    # Re-link must not inflate.
    model.issues.link_assertion(aid, iid, "supports")
    assert len(model.evidence.list_edges_for_target("issue", iid)) == 1


def test_link_assertion_seeds_candidate_verification_for_edge(model):
    aid = _mkassertion(model)
    iid, _ = model.issues.upsert_issue("I", IssueType.CLAIM)
    model.issues.link_assertion(aid, iid, "supports")
    eid = model.evidence.list_edges_for_target("issue", iid)[0]["id"]
    vs = model.verification.get(VerificationTargetKind.EVIDENCE_EDGE, eid)
    assert vs is not None
    assert vs["status"] == "candidate"


def test_backfill_from_legacy_links_is_idempotent(model):
    aid1 = _mkassertion(model, "a1")
    aid2 = _mkassertion(model, "a2")
    iid, _ = model.issues.upsert_issue("I", IssueType.CLAIM)
    _insert_legacy_link(model, aid1, iid)
    _insert_legacy_link(model, aid2, iid)
    new_first = model.evidence.backfill_from_legacy_links()
    new_second = model.evidence.backfill_from_legacy_links()
    assert new_first == 2
    assert new_second == 0
    edges = model.evidence.list_edges_for_target("issue", iid)
    assert len(edges) == 2
    for e in edges:
        assert e["source_identity_status"] == "missing_occurrence_span"
        assert e["origin_kind"] == "legacy_backfill"
        assert e["backfill_source"] == "assertion_issue_link"


def test_proof_state_prefers_evidence_edge_when_present(model):
    aid = _mkassertion(model, "supporting fact")
    iid, _ = model.issues.upsert_issue("I", IssueType.CLAIM, materiality=0.7)
    model.issues.link_assertion(aid, iid, "supports")
    # Edge exists — delete the legacy link and the substrate switch must
    # continue to report the edge as supporting. If the edge branch were
    # broken, supporting_count would drop to zero because the legacy
    # branch no longer has the row.
    model.db.execute(
        "DELETE FROM assertion_issue_link WHERE assertion_id=? AND issue_id=?",
        (aid, iid),
    )
    model.proof_state.compute_and_store(iid)
    ps = model.proof_state.get(iid)
    assert ps["supporting_count"] == 1, (
        "evidence-edge branch must still see the supporting assertion when "
        "the legacy link has been removed"
    )


def test_proof_state_falls_back_to_assertion_issue_link_when_no_edges_exist(model):
    aid = _mkassertion(model)
    iid, _ = model.issues.upsert_issue("I", IssueType.CLAIM, materiality=0.7)
    _insert_legacy_link(model, aid, iid, "supports")  # legacy only
    model.proof_state.compute_and_store(iid)
    ps = model.proof_state.get(iid)
    assert ps["supporting_count"] == 1


def test_proof_state_does_not_double_count_when_both_tables_have_same_relation(model):
    aid = _mkassertion(model)
    iid, _ = model.issues.upsert_issue("I", IssueType.CLAIM, materiality=0.7)
    # link_assertion writes both legacy + edge via upsert_edge
    model.issues.link_assertion(aid, iid, "supports")
    model.proof_state.compute_and_store(iid)
    ps = model.proof_state.get(iid)
    assert ps["supporting_count"] == 1, (
        "edge + legacy row for the same assertion must count once, not twice"
    )


def test_proof_substrate_no_split_brain_between_coverage_and_proof_state(model):
    """MVP.3 post-audit: adversarial #4 reproduced a split-brain bug where
    one edge-backed support plus one legacy-only support yielded
    proof_state.supporting_count=1 while get_issue_coverage_report returned
    2 for the same issue. Both substrates must now agree: edge-first with
    legacy fallback, applied identically to proof_state and to
    get_issue_coverage_report."""
    aid_edge = _mkassertion(model, "edge-supported fact", "edge.pdf")
    aid_legacy = _mkassertion(model, "legacy-only fact", "legacy.pdf")
    iid, _ = model.issues.upsert_issue("Split brain claim", IssueType.CLAIM, materiality=0.7)

    # One assertion linked through link_assertion (creates edge + legacy)
    model.issues.link_assertion(aid_edge, iid, "supports")
    # A second assertion linked through legacy-only insert (no edge)
    _insert_legacy_link(model, aid_legacy, iid, "supports")

    model.proof_state.compute_and_store(iid)
    ps = model.proof_state.get(iid)
    report = next(r for r in model.get_issue_coverage_report() if r["id"] == iid)

    # Both substrates must report the same picture. Edge-first wins: since
    # at least one edge exists, the legacy-only support for this issue
    # does not count. proof_state and coverage report must AGREE.
    assert ps["supporting_count"] == report["supporting_count"], (
        f"split-brain regression: proof_state.supporting_count="
        f"{ps['supporting_count']} but coverage_report.supporting_count="
        f"{report['supporting_count']}"
    )


def test_detect_proof_gaps_honors_edge_backed_support(model):
    """_detect_proof_gaps must not open a gap on an issue that is supported
    via evidence_edge. Adversarial #4 flagged this as the PR.2 pattern
    repeating: gap detection read legacy only, so an edge-only support
    would still trigger a missing_issue_predicate gap."""
    from irys.rlm.engine import RLMEngine

    aid = _mkassertion(model)
    iid, _ = model.issues.upsert_issue(
        "Edge-supported claim", IssueType.CLAIM, materiality=0.9
    )
    # Create the edge without the legacy row.
    model.evidence.upsert_edge(
        source_kind="assertion", source_id=aid,
        target_kind="issue", target_id=iid,
        relation_type=EvidenceRelationType.SUPPORTS,
    )
    # Confirm the legacy table is empty for this issue.
    legacy_count = model.db.execute(
        "SELECT COUNT(*) FROM assertion_issue_link WHERE issue_id=?", (iid,)
    ).fetchone()[0]
    assert legacy_count == 0

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    engine._detect_proof_gaps()

    gaps = model.gaps.open_gaps(min_materiality=0.0)
    for g in gaps:
        for d in g.get("dependencies") or []:
            if d.get("affected_type") == "issue" and d.get("affected_id") == iid:
                raise AssertionError(
                    "_detect_proof_gaps must not open a missing_issue_predicate "
                    "gap on an issue that has edge-backed support"
                )


def test_evidence_edge_verification_status_materializes_from_verification_state(model):
    aid = _mkassertion(model)
    iid, _ = model.issues.upsert_issue("I", IssueType.CLAIM)
    model.issues.link_assertion(aid, iid, "supports")
    eid = model.evidence.list_edges_for_target("issue", iid)[0]["id"]
    # Reject via verification API — the mirror must update edge status.
    model.verification.reject(
        VerificationTargetKind.EVIDENCE_EDGE, eid,
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
        rejection_reason="test",
    )
    row = model.db.execute(
        "SELECT verification_status FROM evidence_edge WHERE id=?", (eid,),
    ).fetchone()
    assert row["verification_status"] == "rejected"
