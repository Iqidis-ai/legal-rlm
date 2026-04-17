"""Unit tests for VerificationStateStore (MVP.2, SO-2).

Covers the invariants MVP.2 acceptance criteria require: candidate default
for AI-derived targets, human-only promotion to verified, rejection with
reason, non-downgrade on re-candidate, version/event trail.
"""

import pytest

from irys.matter import MatterModel
from irys.matter.enums import (
    ReviewScope,
    ReviewedByKind,
    VerificationStatus,
    VerificationTargetKind,
)


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def test_candidate_idempotent_does_not_create_duplicates(model):
    vid1 = model.verification.candidate(
        VerificationTargetKind.ASSERTION, "a1", ai_confidence=0.7
    )
    vid2 = model.verification.candidate(
        VerificationTargetKind.ASSERTION, "a1", ai_confidence=0.9
    )
    assert vid1 == vid2
    rows = model.db.execute(
        "SELECT COUNT(*) FROM verification_state WHERE target_id=?", ("a1",)
    ).fetchone()[0]
    assert rows == 1


def test_candidate_does_not_downgrade_verified_or_rejected(model):
    model.verification.candidate(VerificationTargetKind.ASSERTION, "a1")
    model.verification.verify(
        VerificationTargetKind.ASSERTION, "a1",
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
    )
    # Re-calling candidate must not revert status to candidate.
    model.verification.candidate(VerificationTargetKind.ASSERTION, "a1")
    row = model.verification.get(VerificationTargetKind.ASSERTION, "a1")
    assert row["status"] == "verified"

    model.verification.reject(
        VerificationTargetKind.ASSERTION, "a2",
        reviewed_by_kind=ReviewedByKind.USER,
        rejection_reason="no source support",
    )
    model.verification.candidate(VerificationTargetKind.ASSERTION, "a2")
    row = model.verification.get(VerificationTargetKind.ASSERTION, "a2")
    assert row["status"] == "rejected"


def test_verify_rejects_non_human_reviewers(model):
    with pytest.raises(ValueError, match="system"):
        model.verification.verify(
            VerificationTargetKind.ASSERTION, "a1",
            reviewed_by_kind=ReviewedByKind.SYSTEM,
        )
    with pytest.raises(ValueError, match="import"):
        model.verification.verify(
            VerificationTargetKind.ASSERTION, "a1",
            reviewed_by_kind=ReviewedByKind.IMPORT,
        )


def test_reject_requires_non_empty_reason(model):
    with pytest.raises(ValueError, match="rejection_reason"):
        model.verification.reject(
            VerificationTargetKind.ASSERTION, "a1",
            reviewed_by_kind=ReviewedByKind.USER,
            rejection_reason="",
        )
    with pytest.raises(ValueError, match="rejection_reason"):
        model.verification.reject(
            VerificationTargetKind.ASSERTION, "a1",
            reviewed_by_kind=ReviewedByKind.USER,
            rejection_reason="   ",
        )


def test_verify_increments_version_and_appends_event(model):
    model.verification.candidate(VerificationTargetKind.ASSERTION, "a1")
    model.verification.verify(
        VerificationTargetKind.ASSERTION, "a1",
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
        review_scope=ReviewScope.RECORD_TRUTH,
        review_note="clause is unambiguous",
    )
    row = model.verification.get(VerificationTargetKind.ASSERTION, "a1")
    assert row["version"] == 2
    assert row["review_scope"] == "record_truth"
    assert row["review_note"] == "clause is unambiguous"
    events = model.verification.list_events(
        VerificationTargetKind.ASSERTION, "a1"
    )
    assert len(events) == 2  # initial candidate + verify
    # events are ordered newest-first
    assert events[0]["new_status"] == "verified"
    assert events[0]["old_status"] == "candidate"
    assert events[1]["new_status"] == "candidate"


def test_rejection_reason_persists(model):
    model.verification.reject(
        VerificationTargetKind.ASSERTION, "a1",
        reviewed_by_kind=ReviewedByKind.USER,
        rejection_reason="contradicted by operative contract",
    )
    row = model.verification.get(VerificationTargetKind.ASSERTION, "a1")
    assert row["status"] == "rejected"
    assert row["rejection_reason"] == "contradicted by operative contract"


def test_list_by_status_filters_correctly(model):
    model.verification.candidate(VerificationTargetKind.ASSERTION, "cand1")
    model.verification.candidate(VerificationTargetKind.ASSERTION, "cand2")
    model.verification.verify(
        VerificationTargetKind.ASSERTION, "ver1",
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
    )

    candidates = model.verification.list_by_status(VerificationStatus.CANDIDATE)
    verifieds = model.verification.list_by_status(VerificationStatus.VERIFIED)
    assert {r["target_id"] for r in candidates} == {"cand1", "cand2"}
    assert {r["target_id"] for r in verifieds} == {"ver1"}

    # target_kind filter narrows further
    cand_issue = model.verification.list_by_status(
        VerificationStatus.CANDIDATE, target_kind=VerificationTargetKind.ISSUE_PREDICATE
    )
    assert cand_issue == []


def test_assertion_upsert_creates_candidate_row():
    """MVP.2 AC #2: AI extraction paths create verification rows as candidate."""
    from irys.matter.models import AssertionCandidate
    from irys.matter import SpeechAct, SourceRole, ModelLayer, AssertionKind
    from irys.matter.enums import OriginKind

    m = MatterModel.open_in_memory()
    cand = AssertionCandidate(
        proposition_text="Test proposition",
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
        assertion_kind=AssertionKind.FACTUAL,
        model_layer=ModelLayer.RECORD,
        document_id="contract.pdf",
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, is_new = m.assertions.upsert_occurrence(cand)
    assert is_new
    row = m.verification.get(VerificationTargetKind.ASSERTION, aid)
    assert row is not None
    assert row["status"] == "candidate"
    assert row["ai_confidence"] is not None  # populated from initial belief confidence


def test_rejected_assertions_excluded_from_proof_substrate():
    """MVP.2 AC #4: rejected targets are excluded from proof and clean synthesis.

    This is the substrate-level test the second adversarial audit flagged as
    missing — MVP.2 must change what ProofStateStore.compute_and_store,
    get_issue_coverage_report, and _detect_proof_gaps actually read, not
    just patch consumers. Seeds one supporting assertion, rejects it, and
    asserts all three substrate consumers treat the issue as unsupported.
    """
    from irys.matter.enums import IssueType, OriginKind
    from irys.matter.models import AssertionCandidate
    from irys.matter import SpeechAct, SourceRole, ModelLayer, AssertionKind
    from irys.rlm.engine import RLMEngine

    m = MatterModel.open_in_memory()
    iid, _ = m.issues.upsert_issue(
        "High-materiality claim", IssueType.CLAIM, materiality=0.9
    )
    cand = AssertionCandidate(
        proposition_text="Defendant performed obligation",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        model_layer=ModelLayer.RECORD,
        document_id="contract.pdf",
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = m.assertions.upsert_occurrence(cand)
    m.issues.link_assertion(aid, iid, relation_type="supports")

    # Baseline: before rejection, the support shows up everywhere.
    m.proof_state.compute_and_store(iid)
    ps_before = m.proof_state.get(iid)
    report_before = next(r for r in m.get_issue_coverage_report() if r["id"] == iid)
    assert ps_before["supporting_count"] == 1
    assert report_before["supporting_count"] == 1

    # Reject the assertion.
    m.verification.reject(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
        rejection_reason="contradicted by operative MSA",
    )

    # After rejection: proof_state must see zero supports.
    m.proof_state.compute_and_store(iid)
    ps_after = m.proof_state.get(iid)
    assert ps_after["supporting_count"] == 0, (
        "ProofStateStore.compute_and_store must exclude rejected assertions"
    )

    # Coverage report must see zero supports.
    report_after = next(r for r in m.get_issue_coverage_report() if r["id"] == iid)
    assert report_after["supporting_count"] == 0, (
        "get_issue_coverage_report must exclude rejected assertions"
    )
    assert report_after["verified_supporting_count"] == 0
    assert report_after["candidate_supporting_count"] == 0

    # _detect_proof_gaps must now treat the issue as unsupported.
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = m
    engine._detect_proof_gaps()
    gaps = m.gaps.open_gaps(min_materiality=0.0)
    assert any(
        g.get("gap_type") == "missing_issue_predicate"
        and any(
            d.get("affected_type") == "issue" and d.get("affected_id") == iid
            for d in (g.get("dependencies") or [])
        )
        for g in gaps
    ), "_detect_proof_gaps must open a missing_issue_predicate gap once the only support is rejected"


def test_predicate_add_creates_candidate_verification(model):
    """MVP.2 AC #2: issue_predicate additions create candidate verification rows."""
    from irys.matter.enums import IssueType

    iid, _ = model.issues.upsert_issue("Test claim", IssueType.CLAIM)
    pid = model.issues.add_predicate(iid, "Defendant owed a duty")
    row = model.verification.get(VerificationTargetKind.ISSUE_PREDICATE, pid)
    assert row is not None
    assert row["status"] == "candidate"


def test_quant_record_creates_candidate_verification(model):
    """MVP.2 AC #2: quant_fact additions create candidate verification rows."""
    qid = model.quant.record(quant_kind="amount", raw_text="Invoice $100", amount_value=100.0, currency="USD")
    row = model.verification.get(VerificationTargetKind.QUANT_FACT, qid)
    assert row is not None
    assert row["status"] == "candidate"


def test_authority_upsert_creates_candidate_verification(model):
    """MVP.2 AC #2: authority inserts create candidate verification rows."""
    auth_id, is_new = model.authority.upsert("Smith v. Jones, 1 F.3d 100 (9th Cir. 2020)")
    assert is_new
    row = model.verification.get(VerificationTargetKind.AUTHORITY, auth_id)
    assert row is not None
    assert row["status"] == "candidate"


def test_attacking_count_excludes_rejected_assertions(model):
    """get_issue_coverage_report's attacking_count must filter rejected too."""
    from irys.matter.enums import IssueType, OriginKind
    from irys.matter.models import AssertionCandidate
    from irys.matter import SpeechAct, SourceRole, ModelLayer, AssertionKind

    iid, _ = model.issues.upsert_issue("Claim", IssueType.CLAIM, materiality=0.9)
    cand = AssertionCandidate(
        proposition_text="Attacks the claim",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        model_layer=ModelLayer.RECORD,
        document_id="doc.pdf",
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    model.issues.link_assertion(aid, iid, relation_type="attacks")

    report = next(r for r in model.get_issue_coverage_report() if r["id"] == iid)
    assert report["attacking_count"] == 1

    # Reject the attacker — attacking_count must drop.
    model.verification.reject(
        VerificationTargetKind.ASSERTION, aid,
        reviewed_by_kind=ReviewedByKind.ATTORNEY,
        rejection_reason="incorrect attribution",
    )
    report = next(r for r in model.get_issue_coverage_report() if r["id"] == iid)
    assert report["attacking_count"] == 0, (
        "rejected attacking assertion must not inflate attacking_count"
    )
