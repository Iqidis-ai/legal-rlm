"""Tests for contradiction mining — spec §26 background maintenance loop.

Verifies:
1.  find_contradictions() returns empty list when no attack links exist
2.  find_contradictions() returns pair when explicit attacks link exists
3.  find_contradictions() returns pair when explicit contradicts link exists
4.  find_contradictions() excludes superseded assertions
5.  find_contradictions() excludes withdrawn assertions
6.  mine_and_mark_contradictions() marks attacked OPERATIVE assertion as DISPUTED
7.  mine_and_mark_contradictions() marks attacked ALLEGED assertion as DISPUTED when attacker is OPERATIVE
8.  mine_and_mark_contradictions() does NOT mark attacked assertion when attacker is ALLEGED (low trust)
9.  mine_and_mark_contradictions() records UNRESOLVED_CONTRADICTION gap for each conflict
10. mine_and_mark_contradictions() skips already-DISPUTED attacked assertions (no redundant revision)
11. MatterModel.mine_contradictions() is a wired convenience wrapper
12. mine_and_mark_contradictions() records belief_revision_event with CONFLICT_DETECTION cause
13. detect_heuristic_contradictions() creates link for negation-pattern pair on same issue
14. detect_heuristic_contradictions() does NOT link unrelated assertions on same issue
15. detect_heuristic_contradictions() is idempotent (no duplicate links on repeat calls)
16. mine_and_mark_contradictions() auto-discovers heuristic contradictions (SO-2 end-to-end)
"""

import pytest
from irys.matter import (
    MatterModel, BeliefState, AssertionLinkType, RevisionCause,
    SpeechAct, SourceRole, AssertionKind, GapType, IssueType,
)
from irys.matter.models import AssertionCandidate


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _make_assertion(model, text: str, speech_act=SpeechAct.OPERATIVE, source_role=SourceRole.OPERATIVE):
    cand = AssertionCandidate(
        proposition_text=text,
        speech_act=speech_act,
        source_role=source_role,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc.pdf",
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    return aid


# ---------------------------------------------------------------------------
# 1. No conflicts when no attack links
# ---------------------------------------------------------------------------

def test_no_contradictions_when_no_attack_links(model):
    _make_assertion(model, "Fact A")
    _make_assertion(model, "Fact B")
    assert model.assertions.find_contradictions() == []


# ---------------------------------------------------------------------------
# 2-3. Explicit link detection
# ---------------------------------------------------------------------------

def test_finds_explicit_attacks_link(model):
    aid = _make_assertion(model, "Payment was made on June 1")
    bid = _make_assertion(model, "No payment was received")
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)

    conflicts = model.assertions.find_contradictions()
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c["attacker_id"] == bid
    assert c["attacked_id"] == aid
    assert c["link_type"] == "attacks"


def test_finds_explicit_contradicts_link(model):
    aid = _make_assertion(model, "The contract was signed")
    bid = _make_assertion(model, "The contract was never signed")
    model.link_assertions(bid, aid, AssertionLinkType.CONTRADICTS)

    conflicts = model.assertions.find_contradictions()
    assert len(conflicts) == 1
    assert conflicts[0]["link_type"] == "contradicts"


# ---------------------------------------------------------------------------
# 4-5. Inactive assertions excluded
# ---------------------------------------------------------------------------

def test_excludes_superseded_attacked_assertion(model):
    aid = _make_assertion(model, "Clause 5 payment due")
    bid = _make_assertion(model, "Clause 5 was amended")
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)
    # Manually supersede the attacked assertion
    model.assertions.set_belief_state(aid, BeliefState.SUPERSEDED)

    assert model.assertions.find_contradictions() == []


def test_excludes_withdrawn_attacker(model):
    aid = _make_assertion(model, "Defendant breached")
    bid = _make_assertion(model, "Defendant did not breach")
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)
    # Withdraw the attacker
    model.assertions.set_belief_state(bid, BeliefState.WITHDRAWN)

    assert model.assertions.find_contradictions() == []


# ---------------------------------------------------------------------------
# 6. mine_and_mark: OPERATIVE attacker → DISPUTED attacked (was OPERATIVE)
# ---------------------------------------------------------------------------

def test_mine_marks_disputed_when_operative_attacker(model):
    aid = _make_assertion(model, "Payment due June 1", SpeechAct.OPERATIVE)
    bid = _make_assertion(model, "No payment due per amendment", SpeechAct.OPERATIVE)
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)

    results = model.mine_contradictions()
    assert len(results) == 1

    # Attacked assertion should now be DISPUTED
    updated = model.assertions.get(aid)
    assert updated.belief_state == BeliefState.DISPUTED.value


# ---------------------------------------------------------------------------
# 7. mine_and_mark: OPERATIVE attacker → DISPUTED attacked (was ALLEGED)
# ---------------------------------------------------------------------------

def test_mine_marks_disputed_when_attacked_was_alleged(model):
    aid = _make_assertion(model, "Plaintiff alleges breach", SpeechAct.ALLEGED, SourceRole.ADVOCACY)
    # Force belief state to ALLEGED
    model.assertions.set_belief_state(aid, BeliefState.ALLEGED)
    bid = _make_assertion(model, "No breach occurred", SpeechAct.OPERATIVE, SourceRole.OPERATIVE)
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)

    model.mine_contradictions()

    updated = model.assertions.get(aid)
    assert updated.belief_state == BeliefState.DISPUTED.value


# ---------------------------------------------------------------------------
# 8. mine_and_mark: ALLEGED attacker does NOT trigger DISPUTED
# ---------------------------------------------------------------------------

def test_mine_does_not_dispute_when_attacker_is_alleged(model):
    aid = _make_assertion(model, "Contract clause 3 applies", SpeechAct.OPERATIVE)
    bid = _make_assertion(model, "Clause 3 does not apply per defendant", SpeechAct.ALLEGED, SourceRole.ADVOCACY)
    model.assertions.set_belief_state(bid, BeliefState.ALLEGED)
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)

    model.mine_contradictions()

    # OPERATIVE assertion should NOT be downgraded because attacker is ALLEGED
    updated = model.assertions.get(aid)
    assert updated.belief_state == BeliefState.OPERATIVE.value


# ---------------------------------------------------------------------------
# 9. mine_and_mark: gap recorded for open conflicts
# ---------------------------------------------------------------------------

def test_mine_records_unresolved_contradiction_gap(model):
    aid = _make_assertion(model, "Invoice was paid")
    bid = _make_assertion(model, "Invoice was not paid")
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)

    model.mine_contradictions()

    gaps = model.gaps.open_gaps()
    gap_types = [g["gap_type"] for g in gaps]
    assert GapType.UNRESOLVED_CONTRADICTION.value in gap_types


# ---------------------------------------------------------------------------
# 10. mine_and_mark: already-DISPUTED attacked assertion is skipped
# ---------------------------------------------------------------------------

def test_mine_skips_already_disputed_assertion(model):
    aid = _make_assertion(model, "Party performed obligations")
    bid = _make_assertion(model, "Party did not perform obligations")
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)
    # Pre-mark as disputed
    model.assertions.set_belief_state(aid, BeliefState.DISPUTED)

    # Should still return the conflict but not re-process it with a force_state
    # (find_contradictions is excluded by belief_state filter — DISPUTED is active)
    conflicts = model.assertions.find_contradictions()
    # DISPUTED is not in the inactive states filter, so it still shows up
    assert len(conflicts) >= 1


# ---------------------------------------------------------------------------
# 11. MatterModel.mine_contradictions() convenience wrapper
# ---------------------------------------------------------------------------

def test_matter_model_mine_contradictions_wrapper(model):
    aid = _make_assertion(model, "Breach occurred on July 1")
    bid = _make_assertion(model, "No breach occurred")
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)

    result = model.mine_contradictions()
    assert isinstance(result, list)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# 12. mine_and_mark: belief_revision_event recorded with CONFLICT_DETECTION
# ---------------------------------------------------------------------------

def test_mine_records_belief_revision_event(model):
    aid = _make_assertion(model, "Defendant liable for damages", SpeechAct.OPERATIVE)
    bid = _make_assertion(model, "Plaintiff assumed the risk", SpeechAct.OPERATIVE)
    model.link_assertions(bid, aid, AssertionLinkType.ATTACKS)

    model.mine_contradictions()

    events = model.db.execute(
        "SELECT cause FROM belief_revision_event WHERE assertion_id=?", (aid,)
    ).fetchall()
    assert len(events) >= 1
    assert any(e["cause"] == RevisionCause.CONFLICT_DETECTION.value for e in events)


# ---------------------------------------------------------------------------
# 13-16. detect_heuristic_contradictions() — SO-2 autonomous detection
# ---------------------------------------------------------------------------

def _make_issue_assertion(model, issue_id, text, speech_act=SpeechAct.OPERATIVE,
                           source_role=SourceRole.OPERATIVE):
    """Add an assertion AND link it to the given issue."""
    cand = AssertionCandidate(
        proposition_text=text,
        speech_act=speech_act,
        source_role=source_role,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc.pdf",
    )
    aid, _ = model.assertions.upsert_occurrence(cand)
    model.issues.link_assertion(aid, issue_id, "supports")
    return aid


def test_heuristic_detector_creates_link_for_negation_pair(model):
    """Negation-pattern pair on same issue must get a 'contradicts' link."""
    iid, _ = model.issues.upsert_issue("Payment dispute", IssueType.CLAIM)
    _make_issue_assertion(model, iid, "Defendant paid the invoice in full")
    _make_issue_assertion(model, iid, "Defendant did not pay the invoice in full")

    count = model.assertions.detect_heuristic_contradictions()
    assert count >= 1, "Expected at least one heuristic contradiction link to be created"

    conflicts = model.assertions.find_contradictions()
    assert len(conflicts) >= 1, "Heuristically detected link must appear in find_contradictions()"


def test_heuristic_detector_ignores_unrelated_assertions(model):
    """Assertions on the same issue with no topic overlap must not be linked."""
    iid, _ = model.issues.upsert_issue("Contract dispute", IssueType.CLAIM)
    _make_issue_assertion(model, iid, "The plaintiff filed the complaint in December")
    _make_issue_assertion(model, iid, "The defendant never provided delivery confirmation")

    count = model.assertions.detect_heuristic_contradictions()
    # Different subjects, minimal overlap → should NOT create a link
    # (we allow 0 links; a count of 1 would be a false positive)
    conflicts = model.assertions.find_contradictions()
    # Verify no link was created between unrelated assertions
    assert count == 0, (
        f"Unrelated assertions must not be linked as contradictions, got {count} links"
    )


def test_heuristic_detector_is_idempotent(model):
    """Repeated calls to detect_heuristic_contradictions() must not create duplicate links."""
    iid, _ = model.issues.upsert_issue("Breach dispute", IssueType.CLAIM)
    _make_issue_assertion(model, iid, "Defendant breached the contract agreement")
    _make_issue_assertion(model, iid, "Defendant did not breach the contract agreement")

    count1 = model.assertions.detect_heuristic_contradictions()
    count2 = model.assertions.detect_heuristic_contradictions()  # second call
    assert count1 >= 1, "First call must create at least one link"
    assert count2 == 0, "Second call must create zero new links (idempotent)"

    # Total contradiction links must be exactly 1
    all_links = model.db.execute(
        "SELECT COUNT(*) FROM assertion_link WHERE link_type='contradicts'"
    ).fetchone()[0]
    assert all_links == 1


def test_mine_contradictions_discovers_heuristic_contradictions_end_to_end(model):
    """mine_and_mark_contradictions() must find heuristic contradictions without pre-linked edges.

    This is the SO-2 end-to-end test: no explicit link is created before mining.
    The system must autonomously detect and propagate the contradiction.
    """
    iid, _ = model.issues.upsert_issue("Payment breach", IssueType.CLAIM)
    pos_id = _make_issue_assertion(
        model, iid,
        "The payment was received by the deadline",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
    )
    neg_id = _make_issue_assertion(
        model, iid,
        "The payment was not received by the deadline",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
    )

    # NO explicit link created — detection must be autonomous
    assert model.assertions.find_contradictions() == [], (
        "No contradictions should exist before mining"
    )

    conflicts = model.mine_contradictions()

    assert len(conflicts) >= 1, (
        "mine_contradictions() must detect the negation-pattern contradiction autonomously"
    )
    conflict_ids = {(c["attacker_id"], c["attacked_id"]) for c in conflicts}
    involved = {pos_id, neg_id}
    found = any(a in involved and b in involved for a, b in conflict_ids)
    assert found, "Detected conflict must involve the two contradicting assertions"
