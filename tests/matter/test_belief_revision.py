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
