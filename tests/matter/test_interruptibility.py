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
from irys.matter.enums import OriginKind, RevisionCause, GapType, IssueType
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


def test_record_gap_persists_to_gap_store_and_ledger(model):
    """record_gap() must write both a gap row and a ledger event (SO-7)."""
    run_id = model.start_run("Gap store test")
    adapter = MatterRuntimeAdapter(model, run_id)

    gap_id = adapter.record_gap(
        description="No documents found for: 'signed amendment'",
        gap_type=GapType.MISSING_DOCUMENT,
        expected_artifact="signed amendment",
        materiality=0.7,
    )

    assert gap_id

    # Gap persisted in store
    open_gaps = model.gaps.open_gaps(min_materiality=0.5)
    assert len(open_gaps) == 1
    assert open_gaps[0]["id"] == gap_id
    assert open_gaps[0]["gap_type"] == GapType.MISSING_DOCUMENT.value

    # Ledger event recorded
    events = model.ledger.get_events(run_id)
    gap_events = [e for e in events if e["event_type"] == LedgerEventType.GAP_IDENTIFIED.value]
    assert len(gap_events) == 1
    assert gap_events[0]["changed_object_id"] == gap_id


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


def test_run_interrupt_transition(model):
    """interrupt_run() sets status=interrupted and records a ledger event."""
    run_id = model.start_run("Interrupt lifecycle test")
    model.interrupt_run(run_id)
    run = model.ledger.get_run(run_id)
    assert run.status == "interrupted"
    assert run.completed_at is not None

    events = model.ledger.get_events(run_id)
    event_types = [e["event_type"] for e in events]
    assert "user_interrupted" in event_types


def test_interrupt_distinct_from_complete_and_fail(model):
    """All three terminal statuses are distinct."""
    r1 = model.start_run("complete test")
    model.complete_run(r1)
    r2 = model.start_run("fail test")
    model.fail_run(r2, "error")
    r3 = model.start_run("interrupt test")
    model.interrupt_run(r3)

    statuses = {model.ledger.get_run(rid).status for rid in [r1, r2, r3]}
    assert statuses == {"completed", "failed", "interrupted"}


def test_recent_runs(model):
    for i in range(3):
        run_id = model.start_run(f"Query {i}")
        model.complete_run(run_id)

    recent = model.ledger.recent_runs(limit=5)
    assert len(recent) == 3
    assert all(r["status"] == "completed" for r in recent)


# ---------------------------------------------------------------------------
# Redirect (SO-3)
# ---------------------------------------------------------------------------

def test_redirect_request_propagates_to_adapter(model):
    """request_redirect() signals via ledger; is_redirect_requested() returns True."""
    run_id = model.start_run("Redirect test")
    adapter = MatterRuntimeAdapter(model, run_id)

    assert not adapter.is_redirect_requested()

    issue_id, _ = model.issues.upsert_issue("Payment obligation", IssueType.CLAIM)
    adapter.request_redirect(issue_id)

    assert adapter.is_redirect_requested()
    assert adapter.get_redirect_issue_id() == issue_id


def test_clear_redirect_resets_flag(model):
    """clear_redirect() must allow is_redirect_requested() to return False again."""
    run_id = model.start_run("Redirect clear test")
    adapter = MatterRuntimeAdapter(model, run_id)

    issue_id, _ = model.issues.upsert_issue("Damages", IssueType.DAMAGES)
    adapter.request_redirect(issue_id)
    assert adapter.is_redirect_requested()

    adapter.clear_redirect()
    assert not adapter.is_redirect_requested()


def test_null_adapter_redirect_is_always_false():
    adapter = NullMatterAdapter()
    adapter.request_redirect("some-issue-id")  # must not raise
    assert adapter.is_redirect_requested() is False
    assert adapter.get_redirect_issue_id() is None


def test_issue_store_get_issue_by_id(model):
    """IssueStore.get_issue(id) must return a dict with 'title' key."""
    issue_id, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM)
    row = model.issues.get_issue(issue_id)
    assert row is not None
    assert row["id"] == issue_id
    assert row["title"] == "Breach of contract"


def test_issue_store_get_issue_returns_none_for_unknown(model):
    """get_issue() with unknown id must return None, not raise."""
    assert model.issues.get_issue("nonexistent-id") is None


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
