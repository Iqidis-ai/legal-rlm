"""Critical correctness tests for belief revision.

Test 2 (Codex): Belief revision propagation over a dependency chain B -> A -> C.
When B is attacked or user-corrected, A and C both change state and
belief_revision_event rows record the exact transitions.
"""

import pytest
from irys.matter import (
    MatterModel, AssertionCandidate, SpeechAct, SourceRole,
    ModelLayer, AssertionKind, AssertionLinkType, BeliefState,
    RevisionCause,
)
from irys.matter.enums import OriginKind


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def add(model, text, doc="doc1", speech_act=SpeechAct.EXTRACTED):
    c = AssertionCandidate(
        proposition_text=text,
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id=doc,
        speech_act=speech_act,
        source_role=SourceRole.UNKNOWN,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(c)
    return aid


# ---------------------------------------------------------------------------
# Test 2: Chain B -> A -> C — revising B propagates to A and C
# ---------------------------------------------------------------------------

def test_revision_propagates_through_dependency_chain(model):
    """
    B supports A; A supports C.
    When B is forced to DISPUTED, A and C should also transition.
    belief_revision_event rows must be written for each change.
    """
    b_id = add(model, "The contract was signed by both parties.")
    a_id = add(model, "The contract is valid and operative.")
    c_id = add(model, "The payment obligation under the contract is enforceable.")

    # B supports A; A supports C
    model.assertions.link(b_id, a_id, AssertionLinkType.SUPPORTS)
    model.assertions.link(a_id, c_id, AssertionLinkType.SUPPORTS)

    # Set initial states
    model.assertions.set_belief_state(b_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.set_belief_state(a_id, BeliefState.OPERATIVE, 0.85)
    model.assertions.set_belief_state(c_id, BeliefState.OPERATIVE, 0.8)

    # Force B to DISPUTED (e.g., user correction: "the signature is disputed")
    result = model.belief.force_state(
        assertion_id=b_id,
        new_state=BeliefState.DISPUTED,
        new_confidence=0.3,
        cause=RevisionCause.USER_CORRECTION,
        note="Signature authenticity disputed in deposition",
    )

    assert result.old_belief_state == BeliefState.OPERATIVE
    assert result.new_belief_state == BeliefState.DISPUTED

    # A and C should have been propagated to
    a_record = model.assertions.get(a_id)
    c_record = model.assertions.get(c_id)

    # A is supported by DISPUTED B → A should be DISPUTED or UNKNOWN
    assert a_record.belief_state != BeliefState.OPERATIVE.value, \
        f"A should not remain OPERATIVE when its support B is DISPUTED, got {a_record.belief_state}"

    # C is supported by A (which is now non-OPERATIVE) → C should change too
    assert c_record.belief_state != BeliefState.OPERATIVE.value, \
        f"C should not remain OPERATIVE when its support chain is disrupted, got {c_record.belief_state}"

    # Verify belief_revision_event rows were written for B
    events = model.db.execute(
        "SELECT * FROM belief_revision_event WHERE assertion_id=? ORDER BY created_at",
        (b_id,),
    ).fetchall()
    assert len(events) >= 1
    assert events[-1]["new_belief_state"] == BeliefState.DISPUTED.value
    assert events[-1]["cause"] == RevisionCause.USER_CORRECTION.value


def test_revision_events_record_exact_transitions(model):
    """Each belief_revision_event must record old and new states."""
    a_id = add(model, "Delivery occurred on January 15.")
    model.assertions.set_belief_state(a_id, BeliefState.OPERATIVE, 0.9)

    model.belief.force_state(
        assertion_id=a_id,
        new_state=BeliefState.DISPUTED,
        new_confidence=0.2,
        cause=RevisionCause.CONFLICT_DETECTION,
    )

    events = model.db.execute(
        "SELECT old_belief_state, new_belief_state, cause FROM belief_revision_event WHERE assertion_id=?",
        (a_id,),
    ).fetchall()

    assert len(events) >= 1
    last = events[-1]
    assert last["old_belief_state"] == BeliefState.OPERATIVE.value
    assert last["new_belief_state"] == BeliefState.DISPUTED.value
    assert last["cause"] == RevisionCause.CONFLICT_DETECTION.value


def test_no_revision_when_state_unchanged(model):
    """If computed state equals current state, no event should be written."""
    a_id = add(model, "Something ambiguous happened.")
    # Default state is UNKNOWN, no links → revision should not change anything

    results = model.belief.apply(
        seed_assertion_ids=[a_id],
        cause=RevisionCause.NEW_EVIDENCE,
    )

    # No change → no revision events
    changed = [r for r in results if r.old_belief_state != r.new_belief_state]
    assert len(changed) == 0


def test_superseded_is_terminal(model):
    """Superseded assertions must not be un-superseded by graph revision."""
    a_id = add(model, "Original contract term.")
    model.assertions.set_belief_state(a_id, BeliefState.SUPERSEDED, 0.1)

    b_id = add(model, "Supporting fact for original term.")
    model.assertions.set_belief_state(b_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.link(b_id, a_id, AssertionLinkType.SUPPORTS)

    # Revision with strong support should NOT un-supersede
    model.belief.apply([b_id, a_id], cause=RevisionCause.NEW_EVIDENCE)

    a_record = model.assertions.get(a_id)
    assert a_record.belief_state == BeliefState.SUPERSEDED.value, \
        "Superseded assertion must not be un-superseded by belief revision"


def test_user_correction_via_matter_model(model):
    """MatterModel.correct_assertion wraps force_state correctly."""
    a_id = add(model, "Payment was received.", doc="invoice.pdf")
    model.assertions.set_belief_state(a_id, BeliefState.ALLEGED, 0.5)

    result = model.correct_assertion(
        assertion_id=a_id,
        new_state=BeliefState.OPERATIVE,
        note="Confirmed by bank statement",
    )

    assert result.new_belief_state == BeliefState.OPERATIVE
    record = model.assertions.get(a_id)
    assert record.belief_state == BeliefState.OPERATIVE.value


def test_bfs_stops_at_max_hops(model):
    """Revision must not run forever on long chains."""
    from irys.matter.belief_revision import BeliefRevisionEngine

    original_max = BeliefRevisionEngine.MAX_HOPS
    BeliefRevisionEngine.MAX_HOPS = 2  # artificially low

    try:
        # Build chain of length 5
        ids = [add(model, f"Proposition {i}.") for i in range(6)]
        for i in range(5):
            model.assertions.link(ids[i], ids[i + 1], AssertionLinkType.SUPPORTS)
            model.assertions.set_belief_state(ids[i], BeliefState.OPERATIVE, 0.9)
        model.assertions.set_belief_state(ids[5], BeliefState.OPERATIVE, 0.9)

        # Force ids[0] to DISPUTED
        model.belief.force_state(
            ids[0], BeliefState.DISPUTED, 0.2, RevisionCause.USER_CORRECTION
        )

        # ids[3], ids[4], ids[5] should NOT have been revised (beyond MAX_HOPS=2)
        for far_id in ids[3:]:
            record = model.assertions.get(far_id)
            assert record.belief_state == BeliefState.OPERATIVE.value, \
                f"Should not have revised beyond MAX_HOPS, but {far_id} was revised"
    finally:
        BeliefRevisionEngine.MAX_HOPS = original_max


def test_both_support_and_attack_results_in_disputed(model):
    """An assertion with both active supports and active attacks must become DISPUTED.

    This is the classic conflicting-evidence case in truth maintenance: a claim
    that has at least one solid supporter AND at least one active attacker cannot
    be resolved — it must be DISPUTED.  This path is exercised by SO-6 quant
    conflicts: two assertions about the same invoice amount become mutually
    contradicting (each attacks the other), but each also has its own supporting
    document as OPERATIVE context.
    """
    # The central assertion whose belief state we're testing
    central_id = add(model, "Invoice total is $50,000.")

    # A solid supporter: this assertion supports the central claim
    supporter_id = add(model, "Invoice document says $50,000.")
    model.assertions.set_belief_state(supporter_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.link(supporter_id, central_id, AssertionLinkType.SUPPORTS)

    # An active attacker: this assertion attacks the central claim
    attacker_id = add(model, "Counter-document says $55,000.")
    model.assertions.set_belief_state(attacker_id, BeliefState.ALLEGED, 0.7)
    model.assertions.link(attacker_id, central_id, AssertionLinkType.CONTRADICTS)

    # Trigger revision on the central assertion
    result = model.belief.apply(
        seed_assertion_ids=[central_id],
        cause=RevisionCause.CONFLICT_DETECTION,
    )

    central_record = model.assertions.get(central_id)
    assert central_record.belief_state == BeliefState.DISPUTED.value, (
        f"Both support+attack must result in DISPUTED; got {central_record.belief_state}"
    )


def test_all_supporters_disputed_collapses_to_unknown(model):
    """When ALL of an assertion's supporters are DISPUTED, support base collapses to UNKNOWN.

    In _compute_belief_state(), if support_states is non-empty but none qualify as
    strong_supports (because all supporters are DISPUTED/UNKNOWN/WITHDRAWN/SUPERSEDED),
    the assertion reverts to UNKNOWN.  This prevents an assertion from remaining
    OPERATIVE when all its evidence has been challenged.

    Example: 'Contract is enforceable' (C) was supported by 'Contract was duly signed' (B).
    If B is disputed (forgery alleged), C has no solid backing and must drop to UNKNOWN.
    """
    c_id = add(model, "Contract is fully enforceable.")
    b_id = add(model, "Contract was duly signed by both parties.")

    # B supports C; B starts OPERATIVE
    model.assertions.set_belief_state(b_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.link(b_id, c_id, AssertionLinkType.SUPPORTS)

    # Mark B as DISPUTED (forgery claim)
    model.assertions.set_belief_state(b_id, BeliefState.DISPUTED, 0.3)

    # Trigger revision on C — its only supporter is now DISPUTED → should become UNKNOWN
    model.belief.apply(
        seed_assertion_ids=[c_id],
        cause=RevisionCause.CONFLICT_DETECTION,
    )

    c_record = model.assertions.get(c_id)
    assert c_record.belief_state == BeliefState.UNKNOWN.value, (
        f"Assertion whose only supporter is DISPUTED must collapse to UNKNOWN; "
        f"got {c_record.belief_state}"
    )


def test_bfs_does_not_loop_on_circular_dependency(model):
    """BFS revision must not loop infinitely when assertions form a mutual-support cycle.

    The BFS uses a 'visited' set to avoid re-processing the same assertion_id.
    If A supports B and B supports A (circular), the apply() call must terminate.
    This is an invariant of the BFS implementation — not just a MAX_HOPS check.
    """
    a_id = add(model, "Fact A — mutually supports B.")
    b_id = add(model, "Fact B — mutually supports A.")

    # Circular support: A → B and B → A
    model.assertions.link(a_id, b_id, AssertionLinkType.SUPPORTS)
    model.assertions.link(b_id, a_id, AssertionLinkType.SUPPORTS)

    model.assertions.set_belief_state(a_id, BeliefState.OPERATIVE, 0.8)
    model.assertions.set_belief_state(b_id, BeliefState.OPERATIVE, 0.8)

    # This must not hang or stack-overflow
    results = model.belief.apply(
        seed_assertion_ids=[a_id],
        cause=RevisionCause.NEW_EVIDENCE,
    )

    # Both assertions should still be accessible (no crash)
    assert model.assertions.get(a_id) is not None
    assert model.assertions.get(b_id) is not None


def test_withdrawn_attacker_does_not_trigger_disputed(model):
    """A WITHDRAWN attacker must not count as an active attack (SO-2 INERT filtering).

    In _compute_belief_state(), WITHDRAWN is in _INERT so withdrawn attackers are
    excluded from active_attacks.  Once a contradicting claim is withdrawn (e.g.
    the opposing party retracts a position), the previously-challenged assertion
    must recover — it must not remain DISPUTED due to an inert challenge.
    """
    central_id = add(model, "Payment of $50,000 was made on January 15.")
    model.assertions.set_belief_state(central_id, BeliefState.OPERATIVE, 0.85)

    # An attacker that was subsequently WITHDRAWN (retracted claim)
    withdrawn_attacker_id = add(model, "Payment was never made (withdrawn by counsel).")
    model.assertions.set_belief_state(withdrawn_attacker_id, BeliefState.WITHDRAWN, 0.0)
    model.assertions.link(withdrawn_attacker_id, central_id, AssertionLinkType.CONTRADICTS)

    # Trigger revision — the WITHDRAWN attacker must NOT cause DISPUTED
    model.belief.apply(
        seed_assertion_ids=[central_id],
        cause=RevisionCause.NEW_EVIDENCE,
    )

    central_record = model.assertions.get(central_id)
    assert central_record.belief_state != BeliefState.DISPUTED.value, (
        f"WITHDRAWN attacker must not trigger DISPUTED; got {central_record.belief_state}"
    )
    assert central_record.belief_state == BeliefState.OPERATIVE.value, (
        "Assertion with only WITHDRAWN attackers must remain OPERATIVE (INERT filtering)"
    )


def test_revision_event_stores_note_and_run_id():
    """belief_revision_event must store the note and run_id for traceability (SO-3).

    When a user corrects an assertion, they provide a note explaining WHY.
    That note must survive to the belief_revision_event table so it can be
    surfaced in the reasoning trail.  The run_id must also be stored so
    the correction can be traced back to the specific investigation session.
    """
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Note persistence test")

    a_id = add(model, "Delivery occurred on time.")
    model.assertions.set_belief_state(a_id, BeliefState.ALLEGED, 0.5)

    model.belief.force_state(
        assertion_id=a_id,
        new_state=BeliefState.DISPUTED,
        new_confidence=0.2,
        cause=RevisionCause.USER_CORRECTION,
        run_id=run_id,
        note="Client confirmed delivery was actually 3 days late per shipping receipt.",
    )

    events = model.db.execute(
        "SELECT note, run_id FROM belief_revision_event WHERE assertion_id=?",
        (a_id,),
    ).fetchall()
    assert len(events) >= 1
    last = events[-1]
    assert last["note"] is not None and "3 days late" in last["note"], (
        "User correction note must be stored in belief_revision_event for traceability (SO-3)"
    )
    assert last["run_id"] == run_id, (
        "run_id must be stored in belief_revision_event so corrections are traceable to their run (SO-3)"
    )
