"""Tests for user-steerable reasoning ledger and interruptibility (SO-5).

Verifies:
1. Stop signal propagates from ledger → adapter.is_stop_requested()
2. Ledger events record step-by-step reasoning in ordered sequence
3. MatterModel.correct_assertion() provides user correction entry point
4. Ledger event sequence is monotonically increasing
5. run_session status transitions: running → completed / failed
"""

import pytest
from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole
from irys.matter import ModelLayer, AssertionKind, BeliefState, LedgerEventType
from irys.matter.enums import OriginKind, RevisionCause
from irys.matter.runtime import MatterRuntimeAdapter, NullMatterAdapter


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def add_assertion(model, text, doc="doc1"):
    c = AssertionCandidate(
        proposition_text=text,
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id=doc,
        speech_act=SpeechAct.EXTRACTED,
        source_role=SourceRole.UNKNOWN,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model.assertions.upsert_occurrence(c)
    return aid


# ---------------------------------------------------------------------------
# Stop propagation
# ---------------------------------------------------------------------------

def test_stop_propagates_from_ledger_to_adapter(model):
    run_id = model.start_run("Interruptible run")
    adapter = MatterRuntimeAdapter(model, run_id)

    assert not adapter.is_stop_requested()
    assert not model.ledger.is_stop_requested(run_id)

    # User requests stop via adapter
    adapter.request_stop()

    assert adapter.is_stop_requested()
    assert model.ledger.is_stop_requested(run_id)


def test_stop_logged_as_ledger_event(model):
    run_id = model.start_run("Stop event test")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.request_stop()

    events = model.ledger.get_events(run_id)
    user_interrupt_events = [
        e for e in events
        if e["event_type"] == LedgerEventType.USER_INTERRUPTED.value
    ]
    assert len(user_interrupt_events) >= 1


def test_null_adapter_stop_is_always_false():
    adapter = NullMatterAdapter()
    adapter.request_stop()  # must not raise
    assert adapter.is_stop_requested() is False  # always False for NullAdapter


# ---------------------------------------------------------------------------
# Ledger event ordering
# ---------------------------------------------------------------------------

def test_ledger_events_monotonic_sequence(model):
    run_id = model.start_run("Sequence test")
    adapter = MatterRuntimeAdapter(model, run_id)

    for i in range(10):
        adapter.log_step(f"Step {i}", why=f"Reason {i}")

    events = model.ledger.get_events(run_id)
    seq_nos = [e["seq_no"] for e in events]
    assert seq_nos == sorted(seq_nos), "Sequence numbers must be monotonically increasing"
    # seq_no=0 is the run_started event
    assert seq_nos[0] == 0


def test_ledger_run_started_event_first(model):
    run_id = model.start_run("Ordering test")
    events = model.ledger.get_events(run_id)
    assert events[0]["event_type"] == LedgerEventType.RUN_STARTED.value


def test_ledger_assertion_added_events(model):
    run_id = model.start_run("Assertion event test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.record_fact("Payment was due January 15.", document_id="contract.pdf")
    adapter.record_fact("Payment was not made.", document_id="email.pdf")

    events = model.ledger.get_events(run_id)
    assertion_events = [
        e for e in events
        if e["event_type"] == LedgerEventType.ASSERTION_ADDED.value
    ]
    assert len(assertion_events) == 2


def test_ledger_conflict_detected_event(model):
    run_id = model.start_run("Conflict test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.log_conflict("Contradiction: 'payment received' conflicts with 'no payment made'")

    events = model.ledger.get_events(run_id)
    conflict_events = [
        e for e in events
        if e["event_type"] == LedgerEventType.CONFLICT_DETECTED.value
    ]
    assert len(conflict_events) == 1
    assert "Contradiction" in conflict_events[0]["summary"]


def test_ledger_gap_identified_event(model):
    run_id = model.start_run("Gap test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.log_gap("Missing: signed amendment to contract", gap_id=None)

    events = model.ledger.get_events(run_id)
    gap_events = [
        e for e in events
        if e["event_type"] == LedgerEventType.GAP_IDENTIFIED.value
    ]
    assert len(gap_events) == 1


# ---------------------------------------------------------------------------
# Run session lifecycle
# ---------------------------------------------------------------------------

def test_run_complete_transition(model):
    run_id = model.start_run("Complete lifecycle test")
    run = model.ledger.get_run(run_id)
    assert run.status == "running"

    model.complete_run(run_id)
    run = model.ledger.get_run(run_id)
    assert run.status == "completed"
    assert run.completed_at is not None


def test_run_fail_transition(model):
    run_id = model.start_run("Failure lifecycle test")
    model.fail_run(run_id, "Gemini API timeout")
    run = model.ledger.get_run(run_id)
    assert run.status == "failed"


def test_recent_runs(model):
    for i in range(3):
        run_id = model.start_run(f"Query {i}")
        model.complete_run(run_id)

    recent = model.ledger.recent_runs(limit=5)
    assert len(recent) == 3
    assert all(r["status"] == "completed" for r in recent)


# ---------------------------------------------------------------------------
# User correction entry point
# ---------------------------------------------------------------------------

def test_correct_assertion_via_matter_model(model):
    """correct_assertion() is the user steering mechanism for belief state."""
    a_id = add_assertion(model, "Defendant received payment on time.", "invoice.pdf")
    model.assertions.set_belief_state(a_id, BeliefState.ALLEGED, 0.5)

    result = model.correct_assertion(
        assertion_id=a_id,
        new_state=BeliefState.OPERATIVE,
        note="Confirmed by bank wire transfer record",
    )

    assert result.old_belief_state == BeliefState.ALLEGED
    assert result.new_belief_state == BeliefState.OPERATIVE

    record = model.assertions.get(a_id)
    assert record.belief_state == BeliefState.OPERATIVE.value


def test_correct_assertion_writes_revision_event(model):
    a_id = add_assertion(model, "Payment was timely.")
    model.assertions.set_belief_state(a_id, BeliefState.ALLEGED, 0.5)

    model.correct_assertion(a_id, BeliefState.DISPUTED, note="Disputed in deposition")

    events = model.db.execute(
        "SELECT * FROM belief_revision_event WHERE assertion_id=? ORDER BY created_at",
        (a_id,),
    ).fetchall()
    assert len(events) >= 1
    assert events[-1]["cause"] == RevisionCause.USER_CORRECTION.value
    assert events[-1]["new_belief_state"] == BeliefState.DISPUTED.value
    assert "deposition" in (events[-1]["note"] or "")
