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
from irys.matter.belief_revision import _compute_belief_state, _SOURCE_TRUST


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


def test_bfs_stops_at_max_work(model):
    """Revision must not run forever — MAX_WORK caps total node-visits."""
    from irys.matter.belief_revision import BeliefRevisionEngine

    original_max = BeliefRevisionEngine.MAX_WORK
    BeliefRevisionEngine.MAX_WORK = 2  # artificially low: only 2 node-visits allowed

    try:
        # Build chain of length 5: ids[0] → ids[1] → ids[2] → ids[3] → ids[4] → ids[5]
        ids = [add(model, f"Proposition {i}.") for i in range(6)]
        for i in range(5):
            model.assertions.link(ids[i], ids[i + 1], AssertionLinkType.SUPPORTS)
            model.assertions.set_belief_state(ids[i], BeliefState.OPERATIVE, 0.9)
        model.assertions.set_belief_state(ids[5], BeliefState.OPERATIVE, 0.9)

        # Force ids[0] to DISPUTED — propagation starts from ids[1] (the dependent)
        model.belief.force_state(
            ids[0], BeliefState.DISPUTED, 0.2, RevisionCause.USER_CORRECTION
        )

        # With MAX_WORK=2: ids[1] (work=1) and ids[2] (work=2) may be processed.
        # ids[3], ids[4], ids[5] must NOT have been revised (work budget exhausted).
        for far_id in ids[3:]:
            record = model.assertions.get(far_id)
            assert record.belief_state == BeliefState.OPERATIVE.value, \
                f"Should not have revised beyond MAX_WORK=2, but {far_id} was revised"
    finally:
        BeliefRevisionEngine.MAX_WORK = original_max


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
    """Revision must not loop infinitely when assertions form a mutual-support cycle.

    The fixpoint engine uses MAX_WORK to cap total node-visits.  If A supports B
    and B supports A (circular), the apply() call must terminate because each
    re-enqueue only happens on state change, and belief states converge to a fixed
    point (no further changes) well within the MAX_WORK budget.
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


def test_alleged_supporter_does_not_elevate_to_inferred(model):
    """An ALLEGED supporter must not elevate a UNKNOWN assertion to INFERRED (SO-5).

    ALLEGED means "asserted by an advocacy source" — a party's claim, not an operative fact.
    Elevating an assertion to INFERRED solely on the basis of an ALLEGED supporter would
    amplify advocacy material (e.g. complaint allegations) to a higher epistemic status
    than they deserve.  Only OPERATIVE supporters must trigger INFERRED — this is the
    core SO-5 source calibration invariant in the belief revision engine.
    """
    central_id = add(model, "Defendant failed to pay $100,000.")

    # Add an ALLEGED supporter (e.g., a complaint allegation)
    alleged_id = add(model, "Complaint says defendant owes $100,000.", speech_act=SpeechAct.ALLEGED)
    model.assertions.set_belief_state(alleged_id, BeliefState.ALLEGED, 0.7)
    model.assertions.link(alleged_id, central_id, AssertionLinkType.SUPPORTS)

    # Trigger revision on central — ALLEGED supporter should not elevate to INFERRED
    model.belief.apply(
        seed_assertion_ids=[central_id],
        cause=RevisionCause.NEW_EVIDENCE,
    )

    central_record = model.assertions.get(central_id)
    assert central_record.belief_state != BeliefState.INFERRED.value, (
        f"ALLEGED supporter must not elevate assertion to INFERRED — "
        f"only OPERATIVE supporters cause INFERRED state (SO-5). Got: {central_record.belief_state}"
    )
    assert central_record.belief_state != BeliefState.OPERATIVE.value, (
        "ALLEGED supporter must not make assertion OPERATIVE (SO-5 source calibration)"
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


def test_corroborates_link_supports_belief_revision(model):
    """A CORROBORATES link with OPERATIVE corroborator must elevate to INFERRED (SO-2).

    CORROBORATES is a weaker form of support — 'this fact is consistent with / bolsters
    the central claim.'  Like SUPPORTS, it must be included in the belief revision's
    supporter set so an operative corroborating fact can elevate the central assertion.

    This verifies end-to-end: get_supports() includes CORROBORATES → _compute_belief_state()
    sees the OPERATIVE corroborator → central claim becomes INFERRED.
    """
    central_id = add(model, "Payment of $50,000 was received on January 15.")

    # OPERATIVE corroborator from a different document
    corroborator_id = add(model, "Bank wire confirmation shows $50k transfer on Jan 15.",
                          doc="bank_statement.pdf", speech_act=SpeechAct.OPERATIVE)
    model.assertions.set_belief_state(corroborator_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.link(corroborator_id, central_id, AssertionLinkType.CORROBORATES)

    # Trigger belief revision on central
    model.belief.apply(
        seed_assertion_ids=[central_id],
        cause=RevisionCause.NEW_EVIDENCE,
    )

    central_record = model.assertions.get(central_id)
    # CORROBORATES from OPERATIVE source must cause INFERRED (not remain UNKNOWN)
    assert central_record.belief_state == BeliefState.INFERRED.value, (
        f"OPERATIVE corroborator must elevate central assertion to INFERRED (SO-2); "
        f"got {central_record.belief_state}"
    )


# ---------------------------------------------------------------------------
# Trust-weighted belief revision (SO-5 + SO-2)
# ---------------------------------------------------------------------------

def test_advocacy_attacker_lower_confidence_penalty_than_operative_attacker():
    """Advocacy-source attackers must reduce confidence less than operative-source attackers (SO-5).

    The state transition (→ DISPUTED) is the same — presence of any active attack triggers
    DISPUTED regardless of source trust.  But the magnitude of the confidence hit reflects
    source trust: advocacy attacker (weight 0.3) leaves higher confidence than an operative
    attacker (weight 1.0) hitting the same assertion.
    """
    # Advocacy-source single attacker
    state_adv, conf_adv = _compute_belief_state(
        BeliefState.OPERATIVE,
        support_states=[],
        attack_states=[BeliefState.ALLEGED],
        support_source_roles=[],
        attack_source_roles=["advocacy"],
    )
    # Operative-source single attacker
    state_op, conf_op = _compute_belief_state(
        BeliefState.OPERATIVE,
        support_states=[],
        attack_states=[BeliefState.OPERATIVE],
        support_source_roles=[],
        attack_source_roles=["operative"],
    )

    assert state_adv == BeliefState.DISPUTED, "Any active attacker must trigger DISPUTED state"
    assert state_op == BeliefState.DISPUTED, "Any active attacker must trigger DISPUTED state"

    # Advocacy attacker (weight 0.3) → confidence = max(0.1, 0.5 - 0.1*0.3) = 0.47
    # Operative attacker (weight 1.0) → confidence = max(0.1, 0.5 - 0.1*1.0) = 0.4
    assert conf_adv > conf_op, (
        f"Advocacy attacker must leave higher confidence ({conf_adv}) than "
        f"operative attacker ({conf_op}) — SO-5 trust calibration"
    )
    assert abs(conf_adv - 0.47) < 1e-9, f"Expected 0.47, got {conf_adv}"
    assert abs(conf_op - 0.4) < 1e-9, f"Expected 0.4, got {conf_op}"


def test_operative_source_operative_supporter_confidence_higher_than_advocacy_source():
    """Operative-source OPERATIVE supporter must yield higher INFERRED confidence than advocacy-source (SO-5).

    Two assertions both have OPERATIVE belief state, but one comes from an operative
    document (signed contract) while the other comes from advocacy material (complaint).
    The operative-source supporter must produce higher confidence for the supported assertion.
    """
    # OPERATIVE supporter from operative document (weight 1.0)
    state_op, conf_op = _compute_belief_state(
        BeliefState.UNKNOWN,
        support_states=[BeliefState.OPERATIVE],
        attack_states=[],
        support_source_roles=["operative"],
        attack_source_roles=[],
    )
    # OPERATIVE supporter from advocacy document (weight 0.3)
    state_adv, conf_adv = _compute_belief_state(
        BeliefState.UNKNOWN,
        support_states=[BeliefState.OPERATIVE],
        attack_states=[],
        support_source_roles=["advocacy"],
        attack_source_roles=[],
    )

    assert state_op == BeliefState.INFERRED, "Operative supporter must produce INFERRED state"
    assert state_adv == BeliefState.INFERRED, "OPERATIVE belief state still triggers INFERRED regardless of source role"

    assert conf_op > conf_adv, (
        f"Operative-source supporter ({conf_op}) must produce higher INFERRED confidence "
        f"than advocacy-source ({conf_adv}) — SO-5 source trust calibration"
    )
    assert abs(conf_op - 0.6) < 1e-9, f"Expected 0.6, got {conf_op}"
    assert abs(conf_adv - 0.53) < 1e-9, f"Expected 0.53, got {conf_adv}"


def test_trust_weights_backward_compatible_when_roles_absent():
    """When source_roles are None, _compute_belief_state must behave identically to the pre-trust version.

    All callers that don't pass source_roles get weight 1.0 (same as before SO-5 trust weighting).
    This ensures no regressions for code paths that haven't been updated to pass roles yet.
    """
    # With roles=None (backward compat)
    state_none, conf_none = _compute_belief_state(
        BeliefState.OPERATIVE,
        support_states=[],
        attack_states=[BeliefState.ALLEGED],
        support_source_roles=None,
        attack_source_roles=None,
    )
    # With explicit weight=1.0 (operative role)
    state_one, conf_one = _compute_belief_state(
        BeliefState.OPERATIVE,
        support_states=[],
        attack_states=[BeliefState.ALLEGED],
        support_source_roles=[],
        attack_source_roles=["operative"],
    )

    assert state_none == state_one == BeliefState.DISPUTED
    assert conf_none == conf_one, (
        f"None roles (weight=1.0 default) must match explicit operative roles; "
        f"got none={conf_none}, one={conf_one}"
    )


def test_source_trust_weights_table_completeness():
    """All commonly used source_role values must appear in _SOURCE_TRUST (SO-5 completeness)."""
    required_roles = {"operative", "authoritative", "procedural", "informal", "unknown", "draft", "advocacy", "post_hoc"}
    missing = required_roles - set(_SOURCE_TRUST.keys())
    assert not missing, f"Missing source roles in _SOURCE_TRUST: {missing}"

    # High-trust roles must all be >= 0.7
    for role in ("operative", "authoritative", "procedural"):
        assert _SOURCE_TRUST[role] >= 0.7, f"Expected {role} weight >= 0.7, got {_SOURCE_TRUST[role]}"

    # Low-trust roles must all be <= 0.4
    for role in ("advocacy", "post_hoc", "draft"):
        assert _SOURCE_TRUST[role] <= 0.4, f"Expected {role} weight <= 0.4, got {_SOURCE_TRUST[role]}"


def test_get_neighbor_belief_states_returns_source_roles(model):
    """get_neighbor_belief_states must return source_roles alongside belief states (SO-5 wiring)."""
    central_id = add(model, "The agreement was executed on June 1.")

    supporter_id = add(model, "Contract signed June 1.", doc="contract.pdf",
                       speech_act=SpeechAct.OPERATIVE)
    model.assertions.set_belief_state(supporter_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.link(supporter_id, central_id, AssertionLinkType.SUPPORTS)

    neighbors = model.assertions.get_neighbor_belief_states(central_id)

    assert "support_source_roles" in neighbors, "get_neighbor_belief_states must include support_source_roles"
    assert "attack_source_roles" in neighbors, "get_neighbor_belief_states must include attack_source_roles"
    assert len(neighbors["support_source_roles"]) == len(neighbors["support_states"]), (
        "support_source_roles and support_states must be parallel lists of equal length"
    )
    assert len(neighbors["attack_source_roles"]) == len(neighbors["attack_states"]), (
        "attack_source_roles and attack_states must be parallel lists of equal length"
    )


def test_document_trust_override_low_reduces_supporter_weight(model):
    """A 'low' document trust override must reduce supporter confidence (SO-3 → SO-2 integration).

    When a user marks a document as low-trust, assertions from that document must
    behave as 'advocacy'-weighted in belief revision — even if their stored source_role
    is 'operative'.  This tests that trust overrides flow through get_neighbor_belief_states()
    into _compute_belief_state() confidence output.
    """
    central_id = add(model, "The payment was made in full on time.")

    # Supporter from a document the user will later flag as low-trust
    supporter_id = add(model, "Payment receipt says paid in full.", doc="disputed_receipt.pdf",
                       speech_act=SpeechAct.OPERATIVE)
    model.assertions.set_belief_state(supporter_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.link(supporter_id, central_id, AssertionLinkType.SUPPORTS)

    # Without trust override: supporter has source_role='unknown' → weight 0.5
    # Confidence for INFERRED = min(0.9, 0.5 + 0.1 * 0.5) = 0.55
    model.belief.apply([central_id], cause=RevisionCause.NEW_EVIDENCE)
    record_before = model.assertions.get(central_id)
    state_before = record_before.belief_state
    conf_before = record_before.confidence

    # User marks the receipt document as low-trust (SO-3 steering)
    model.trust_overrides.set("disputed_receipt.pdf", "low", note="Disputed by opposing party")

    # Reset central assertion state so revision runs fresh
    model.assertions.set_belief_state(central_id, BeliefState.UNKNOWN, 0.5)

    # After trust override: supporter behaves as 'advocacy' weight 0.3
    # Confidence for INFERRED = min(0.9, 0.5 + 0.1 * 0.3) = 0.53
    model.belief.apply([central_id], cause=RevisionCause.NEW_EVIDENCE)
    record_after = model.assertions.get(central_id)

    # State should still be INFERRED (OPERATIVE support still promotes — trust only affects confidence)
    assert record_after.belief_state == BeliefState.INFERRED.value, (
        f"State must be INFERRED even with low-trust override; got {record_after.belief_state}"
    )
    # Confidence must be lower after the low-trust override
    assert record_after.confidence < conf_before, (
        f"Low-trust override must reduce confidence: before={conf_before}, after={record_after.confidence}"
    )


def test_matter_model_set_trust_override_triggers_belief_revision(model):
    """MatterModel.set_trust_override() must automatically propagate through dependents (SO-3 → SO-2).

    When a user marks a document as low-trust:
    1. The trust override is persisted.
    2. Belief revision is triggered on assertions from that document (seeds).
    3. The BFS propagates to their dependents (e.g., central_id that the doc's assertion supports).
    4. The dependent's confidence AUTOMATICALLY changes — no manual belief.apply() needed.

    This verifies the full SO-3 → SO-2 integration loop.
    """
    central_id = add(model, "The payment obligation is confirmed.")

    # Supporter from internal_memo.pdf — source_role='unknown' → weight 0.5 (before override)
    supporter_id = add(model, "Memo confirms payment obligation.",
                       doc="internal_memo.pdf", speech_act=SpeechAct.OPERATIVE)
    model.assertions.set_belief_state(supporter_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.link(supporter_id, central_id, AssertionLinkType.SUPPORTS)

    # Apply initial belief revision to establish baseline
    # central_id: INFERRED, confidence = min(0.9, 0.5 + 0.1 * 0.5) = 0.55 (unknown weight)
    model.belief.apply([central_id], cause=RevisionCause.NEW_EVIDENCE)
    conf_before = model.assertions.get(central_id).confidence

    # Use the high-level method — persists override AND triggers belief.apply([supporter_id])
    # BFS seeds on supporter_id → propagates to central_id (dependent via SUPPORTS link)
    # central_id revised with new trust weight: advocacy (0.3)
    # confidence = min(0.9, 0.5 + 0.1 * 0.3) = 0.53
    model.set_trust_override("internal_memo.pdf", "low", note="Authored by interested party")

    # Override is persisted
    override = model.trust_overrides.get("internal_memo.pdf")
    assert override == "low", "Trust override must be persisted by set_trust_override()"

    # Dependent (central_id) confidence must have dropped AUTOMATICALLY
    # — no additional belief.apply() call needed
    conf_after = model.assertions.get(central_id).confidence
    assert conf_after < conf_before, (
        f"set_trust_override() must automatically propagate to dependents: "
        f"before={conf_before}, after={conf_after} (expected < {conf_before})"
    )


def test_document_trust_override_high_increases_attacker_weight(model):
    """A 'high' document trust override must increase attacker confidence impact (SO-3 → SO-2).

    When a user marks a document as high-trust, assertions from that document must
    behave as 'operative'-weighted in belief revision — even if their stored source_role
    was inferred as a lower-trust role.
    """
    central_id = add(model, "The contract clause was waived.")

    # Attacker from a document the user will flag as high-trust
    attacker_id = add(model, "Waiver was never executed per signed amendment.",
                      doc="court_order.pdf", speech_act=SpeechAct.ALLEGED)
    model.assertions.set_belief_state(attacker_id, BeliefState.ALLEGED, 0.7)
    model.assertions.link(attacker_id, central_id, AssertionLinkType.ATTACKS)
    model.assertions.set_belief_state(central_id, BeliefState.OPERATIVE, 0.85)

    # Without trust override: attacker source_role='unknown' weight=0.5
    # Confidence = max(0.1, 0.5 - 0.1 * 0.5) = 0.45
    model.belief.apply([central_id], cause=RevisionCause.NEW_EVIDENCE)
    record_before = model.assertions.get(central_id)
    conf_before = record_before.confidence

    # User marks the court order as high-trust (weight 1.0)
    model.trust_overrides.set("court_order.pdf", "high", note="Signed court order")

    # Reset
    model.assertions.set_belief_state(central_id, BeliefState.OPERATIVE, 0.85)

    # After trust override: attacker behaves as 'operative' weight=1.0
    # Confidence = max(0.1, 0.5 - 0.1 * 1.0) = 0.4
    model.belief.apply([central_id], cause=RevisionCause.NEW_EVIDENCE)
    record_after = model.assertions.get(central_id)

    assert record_after.belief_state == BeliefState.DISPUTED.value, (
        f"High-trust attacker must still produce DISPUTED; got {record_after.belief_state}"
    )
    assert record_after.confidence < conf_before, (
        f"High-trust override must increase attacker weight → lower confidence: "
        f"before={conf_before}, after={record_after.confidence}"
    )


def test_both_sides_confidence_is_balance_aware():
    """Both-sides-present DISPUTED confidence must reflect trust-weighted balance (SO-5).

    Equal weights → confidence = 0.4 (midpoint of [0.3, 0.5))
    Support-dominant (operative support vs advocacy attack) → confidence closer to 0.5
    Attack-dominant (operative attack vs advocacy support) → confidence closer to 0.3
    """
    # Equal trust: operative support (1.0) vs operative attack (1.0) → 0.3 + 0.2 * 0.5 = 0.4
    _, conf_equal = _compute_belief_state(
        BeliefState.OPERATIVE,
        support_states=[BeliefState.OPERATIVE],
        attack_states=[BeliefState.ALLEGED],
        support_source_roles=["operative"],
        attack_source_roles=["operative"],
    )

    # Support-dominant: operative support (1.0) vs advocacy attack (0.3)
    # sup_frac = 1.0 / 1.3 ≈ 0.769; confidence = 0.3 + 0.2 * 0.769 ≈ 0.4538
    _, conf_support_dom = _compute_belief_state(
        BeliefState.OPERATIVE,
        support_states=[BeliefState.OPERATIVE],
        attack_states=[BeliefState.ALLEGED],
        support_source_roles=["operative"],
        attack_source_roles=["advocacy"],
    )

    # Attack-dominant: advocacy support (0.3) vs operative attack (1.0)
    # sup_frac = 0.3 / 1.3 ≈ 0.231; confidence = 0.3 + 0.2 * 0.231 ≈ 0.3462
    _, conf_attack_dom = _compute_belief_state(
        BeliefState.OPERATIVE,
        support_states=[BeliefState.ALLEGED],  # solid: ALLEGED is not in _UNDERMINING
        attack_states=[BeliefState.OPERATIVE],
        support_source_roles=["advocacy"],
        attack_source_roles=["operative"],
    )

    assert conf_support_dom > conf_equal > conf_attack_dom, (
        f"Both-sides confidence must reflect trust balance: "
        f"support-dominant ({conf_support_dom}) > equal ({conf_equal}) > attack-dominant ({conf_attack_dom})"
    )
    assert abs(conf_equal - 0.4) < 1e-4, f"Equal trust must yield 0.4; got {conf_equal}"


def test_mixed_attacker_weights_accumulate_correctly():
    """Effective attack weight must accumulate across multiple attackers with different trust levels.

    Three attackers: operative (1.0) + advocacy (0.3) + draft (0.4) = 1.7 total weight.
    Expected confidence = max(0.1, 0.5 - 0.1 * 1.7) = max(0.1, 0.33) = 0.33.
    All produce DISPUTED state.
    """
    state, confidence = _compute_belief_state(
        BeliefState.OPERATIVE,
        support_states=[],
        attack_states=[BeliefState.OPERATIVE, BeliefState.ALLEGED, BeliefState.ALLEGED],
        support_source_roles=[],
        attack_source_roles=["operative", "advocacy", "draft"],
    )

    assert state == BeliefState.DISPUTED, "Multiple attackers must produce DISPUTED state"
    expected = max(0.1, 0.5 - 0.1 * (1.0 + 0.3 + 0.4))
    assert abs(confidence - expected) < 1e-9, (
        f"Expected confidence {expected} for mixed-trust attackers, got {confidence}"
    )


def test_multiple_operative_supporters_accumulate_confidence():
    """Two operative-source OPERATIVE supporters must produce higher INFERRED confidence than one.

    single supporter: confidence = min(0.9, 0.5 + 0.1 * 1.0) = 0.6
    two supporters:   confidence = min(0.9, 0.5 + 0.1 * 2.0) = 0.7
    """
    _, conf_one = _compute_belief_state(
        BeliefState.UNKNOWN,
        support_states=[BeliefState.OPERATIVE],
        attack_states=[],
        support_source_roles=["operative"],
        attack_source_roles=[],
    )
    _, conf_two = _compute_belief_state(
        BeliefState.UNKNOWN,
        support_states=[BeliefState.OPERATIVE, BeliefState.OPERATIVE],
        attack_states=[],
        support_source_roles=["operative", "operative"],
        attack_source_roles=[],
    )

    assert conf_two > conf_one, (
        f"Two operative supporters ({conf_two}) must produce higher confidence than one ({conf_one})"
    )
    assert abs(conf_one - 0.6) < 1e-9, f"Expected 0.6, got {conf_one}"
    assert abs(conf_two - 0.7) < 1e-9, f"Expected 0.7, got {conf_two}"
