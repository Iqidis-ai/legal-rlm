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
