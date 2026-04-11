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


def test_stop_flag_cached_in_memory(model):
    """After request_stop(), is_stop_requested() uses the in-memory flag, not DB.

    Tests that the cached fast path is set by request_stop() and remains True
    on repeated calls — critical for performance when called on every lead/doc boundary.
    """
    run_id = model.start_run("Cache test run")
    adapter = MatterRuntimeAdapter(model, run_id)

    assert adapter._stop_flag is False
    assert not adapter.is_stop_requested()

    adapter.request_stop()

    # In-memory flag must be set immediately
    assert adapter._stop_flag is True
    # Public method must also return True (via fast path)
    assert adapter.is_stop_requested()
    # Calling many times must not raise (regression: would OOM if it hit DB each time)
    for _ in range(1000):
        assert adapter.is_stop_requested()


def test_stop_cross_process_detected_on_next_check(model):
    """Cross-process stop (DB written without going through request_stop()) must be
    detected on the next is_stop_requested() call.

    This verifies the 'detects stop between every pair of consecutive check points'
    contract: if an API call on another thread writes stop_requested=True to the DB,
    the engine running on this adapter must see it at the very next stop check.
    """
    run_id = model.start_run("Cross-process stop test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Simulate a cross-process stop by writing directly to the DB (bypassing request_stop)
    model.ledger.request_stop(run_id)

    # is_stop_requested() must detect it on the NEXT call — no caching of negative result
    assert adapter.is_stop_requested() is True
    assert adapter._stop_flag is True  # must also set in-memory flag


def test_null_adapter_stop_is_always_false():
    adapter = NullMatterAdapter()
    adapter.request_stop()  # must not raise
    assert adapter.is_stop_requested() is False  # always False for NullAdapter


# ---------------------------------------------------------------------------
# Ledger event ordering
# ---------------------------------------------------------------------------

def test_record_assertion_link_unknown_type_logs_warning(model):
    """Unknown link types must log a SYSTEM_WARNING ledger event, not silently disappear.

    Silent drops hide dependency graph gaps (SO-2 reliability).
    """
    from irys.matter.enums import LedgerEventType
    run_id = model.start_run("link warning test")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid1 = adapter.record_fact("Fact A.", "docA.pdf")
    aid2 = adapter.record_fact("Fact B.", "docB.pdf")

    # Provide an invalid link type — should log a warning, not raise
    adapter.record_assertion_link(aid1, aid2, "definitely_not_a_real_link_type")

    events = model.ledger.get_events(run_id)
    warnings = [e for e in events if e["event_type"] == LedgerEventType.SYSTEM_WARNING.value]
    assert len(warnings) >= 1, "Unknown link type must produce a SYSTEM_WARNING ledger entry"
    assert "link_type" in warnings[0]["summary"] or "definitely_not_a_real_link_type" in warnings[0]["summary"]


def test_record_assertion_link_valid_type_creates_link(model):
    """record_assertion_link() with a valid link type must create the assertion link (SO-2).

    The adapter's record_assertion_link() is how the engine wires semantic
    relationships between extracted assertions.  A SUPPORTS link must be
    retrievable via get_supports() — the dependency graph must actually contain
    the link, not just acknowledge the call.
    """
    run_id = model.start_run("valid link test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Create two facts
    src_id = adapter.record_fact("Contract requires 30-day notice.", "contract.pdf")
    dst_id = adapter.record_fact("Plaintiff gave only 10-day notice.", "complaint.pdf")

    # Wire the src as supporting (or in this case, contradicting) dst
    adapter.record_assertion_link(src_id, dst_id, "contradicts")

    # The link must appear in the dependency graph
    attackers = model.assertions.get_attackers(dst_id)
    assert src_id in attackers, (
        "record_assertion_link('contradicts') must create an attackable edge in the assertion graph (SO-2)"
    )


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


def test_correct_assertion_propagates_to_downstream_dependents(model):
    """correct_assertion() must propagate through the dependency graph (SO-2).

    SO-2 test contract: 'A user correction changes not just that assertion but
    also downstream conclusions that depended on it. The dependency graph is
    traversed and updated.'

    This is the end-to-end test through the user-facing correct_assertion() entry
    point — not just the internal force_state() BFS.
    """
    from irys.matter.enums import AssertionLinkType

    # A supports B: if A is withdrawn, B should no longer be INFERRED/OPERATIVE
    a_id = add_assertion(model, "The contract was fully executed.", "contract.pdf")
    b_id = add_assertion(model, "The payment obligation is enforceable.", "memo.pdf")

    model.assertions.set_belief_state(a_id, BeliefState.OPERATIVE, 0.9)
    model.assertions.set_belief_state(b_id, BeliefState.OPERATIVE, 0.8)
    model.assertions.link(a_id, b_id, AssertionLinkType.SUPPORTS)

    # User corrects A to WITHDRAWN (e.g., "actually this is a draft, not the executed version")
    result = model.correct_assertion(
        assertion_id=a_id,
        new_state=BeliefState.WITHDRAWN,
        note="Document is a draft, not the executed contract",
    )

    assert result.new_belief_state == BeliefState.WITHDRAWN

    # B must have been revised — it cannot remain OPERATIVE when its only support is WITHDRAWN
    b_record = model.assertions.get(b_id)
    assert b_record.belief_state != BeliefState.OPERATIVE.value, (
        "correct_assertion() must propagate through the dependency graph — "
        "B's belief state must change when its support A is WITHDRAWN (SO-2)"
    )


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


def test_correct_assertion_writes_assertion_revision_rows(model):
    """correct_assertion() writes immutable field-diff rows to assertion_revision (Q4 HIGH)."""
    import json
    a_id = add_assertion(model, "Invoice was delivered.")
    model.assertions.set_belief_state(a_id, BeliefState.ALLEGED, 0.4)

    result = model.correct_assertion(a_id, BeliefState.OPERATIVE, note="Confirmed by affidavit")

    rows = model.db.execute(
        "SELECT * FROM assertion_revision WHERE assertion_id=? ORDER BY created_at",
        (a_id,),
    ).fetchall()
    assert len(rows) >= 1, "assertion_revision must have at least one row after correction"
    fields_changed = {r["changed_field"] for r in rows}
    assert "belief_state" in fields_changed, "belief_state change must be recorded"
    bs_row = next(r for r in rows if r["changed_field"] == "belief_state")
    assert json.loads(bs_row["old_value_json"]) == BeliefState.ALLEGED.value
    assert json.loads(bs_row["new_value_json"]) == BeliefState.OPERATIVE.value
    assert bs_row["actor_kind"] == "user"
    assert bs_row["cause"] == RevisionCause.USER_CORRECTION.value
    assert not result.propagation_truncated


def test_bfs_propagation_writes_system_revision_rows(model):
    """BFS-driven belief revision writes assertion_revision rows with actor_kind='system'."""
    from irys.matter.enums import AssertionLinkType
    # support edge: b depends on a
    a_id = add_assertion(model, "Contract was signed by both parties.")
    b_id = add_assertion(model, "Contract is binding.")
    model.assertions.link(a_id, b_id, AssertionLinkType.SUPPORTS)
    model.assertions.set_belief_state(a_id, BeliefState.OPERATIVE, 0.9)

    # Force a to disputed — should propagate to b
    model.correct_assertion(a_id, BeliefState.DISPUTED)

    sys_rows = model.db.execute(
        "SELECT * FROM assertion_revision WHERE assertion_id=? AND actor_kind='system'",
        (b_id,),
    ).fetchall()
    # b should have at least one system revision row from BFS propagation
    assert len(sys_rows) >= 1, "BFS propagation must write system revision rows"


def test_upsert_occurrence_upgrade_writes_revision_rows(model):
    """upsert_occurrence() upgrade path writes assertion_revision rows (Q4 HIGH)."""
    import json
    from irys.matter.enums import SpeechAct, SourceRole, OriginKind
    from irys.matter.models import AssertionCandidate

    # First ingest at low confidence
    low_candidate = AssertionCandidate(
        proposition_text="Defendant breached the agreement.",
        model_layer=model.assertions.db.execute(
            "SELECT model_layer FROM assertion LIMIT 0"
        ),
    )
    # Use record_assertion which calls upsert_occurrence
    from irys.matter.enums import ModelLayer, AssertionKind
    c1 = AssertionCandidate(
        proposition_text="Defendant breached the agreement.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="complaint.pdf",
        source_role=SourceRole.ADVOCACY,
        source_side="plaintiff",
        speech_act=SpeechAct.ALLEGED,
        origin_kind=OriginKind.EXTRACTED,
    )
    a_id, _ = model.record_assertion(c1)

    # Second ingest at higher confidence (operative source, same speaker scope
    # so both map to same claim_key under claim identity v2)
    c2 = AssertionCandidate(
        proposition_text="Defendant breached the agreement.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="signed_agreement.pdf",
        source_role=SourceRole.OPERATIVE,
        source_side="plaintiff",
        speech_act=SpeechAct.OPERATIVE,
        origin_kind=OriginKind.EXTRACTED,
    )
    model.record_assertion(c2)

    rev_rows = model.db.execute(
        "SELECT * FROM assertion_revision WHERE assertion_id=? AND cause='occurrence_upgrade'",
        (a_id,),
    ).fetchall()
    assert len(rev_rows) >= 1, "occurrence_upgrade must write assertion_revision rows"
    fields = {r["changed_field"] for r in rev_rows}
    assert "confidence" in fields

# ---------------------------------------------------------------------------
# _gather_with_cancellation — true in-flight cancellation (SO-3)
# ---------------------------------------------------------------------------

import asyncio
import pytest


@pytest.mark.asyncio
async def test_gather_with_cancellation_no_stop():
    """_gather_with_cancellation completes all tasks when no stop is requested."""
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.rlm.state import InvestigationState

    engine = RLMEngine.__new__(RLMEngine)
    engine._CANCEL_POLL_SECS = 0.05

    async def _fast_task(n):
        await asyncio.sleep(0)
        return n * 2

    tasks = [asyncio.create_task(_fast_task(i)) for i in range(4)]

    # State with no adapter — fallback path
    state = InvestigationState(id="t", query="q", repository_path="/tmp")
    results = await engine._gather_with_cancellation(state, tasks)

    assert results == [0, 2, 4, 6], f"All tasks must complete: {results}"


@pytest.mark.asyncio
async def test_gather_with_cancellation_cancels_on_stop(model):
    """_gather_with_cancellation cancels in-flight tasks when stop is requested."""
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.rlm.state import InvestigationState
    from irys.matter.runtime import MatterRuntimeAdapter

    engine = RLMEngine.__new__(RLMEngine)
    engine._CANCEL_POLL_SECS = 0.05  # Fast poll for test

    run_id = model.start_run("cancel test")
    adapter = MatterRuntimeAdapter(model, run_id)
    state = InvestigationState(id="t", query="q", repository_path="/tmp")
    state._matter_adapter = adapter

    completed = []

    async def _slow_task(n):
        await asyncio.sleep(10)  # Long sleep — should be cancelled
        completed.append(n)
        return n

    tasks = [asyncio.create_task(_slow_task(i)) for i in range(3)]

    # Request stop before starting the gather
    adapter.request_stop()

    results = await engine._gather_with_cancellation(state, tasks)

    # All tasks should be cancelled (None), none should have completed
    assert all(r is None for r in results), (
        f"Cancelled tasks must return None: {results}"
    )
    assert completed == [], "Long-sleeping tasks must not complete after cancellation"


@pytest.mark.asyncio
async def test_gather_with_cancellation_partial_completion(model):
    """Tasks that complete before stop is requested return their results normally."""
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.rlm.state import InvestigationState
    from irys.matter.runtime import MatterRuntimeAdapter

    engine = RLMEngine.__new__(RLMEngine)
    engine._CANCEL_POLL_SECS = 0.05

    run_id = model.start_run("partial cancel test")
    adapter = MatterRuntimeAdapter(model, run_id)
    state = InvestigationState(id="t", query="q", repository_path="/tmp")
    state._matter_adapter = adapter

    async def _instant_task(n):
        await asyncio.sleep(0)
        return n

    async def _slow_task(n):
        await asyncio.sleep(10)
        return n

    # Mix of fast (completes before stop) and slow (cancelled) tasks
    fast = asyncio.create_task(_instant_task(42))
    slow = asyncio.create_task(_slow_task(99))

    # Let fast task complete first
    await asyncio.sleep(0.01)

    adapter.request_stop()
    results = await engine._gather_with_cancellation(state, [fast, slow])

    assert results[0] == 42, "Fast task that completed before stop must keep its result"
    assert results[1] is None, "Slow task cancelled after stop must keep its result"


@pytest.mark.asyncio
async def test_resume_investigation_applies_follow_up_query(tmp_path, monkeypatch):
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.rlm.state import InvestigationState

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    checkpoint_path = tmp_path / "resume-checkpoint.json"

    state = InvestigationState.create("Original query", str(repo_dir))
    state.status = "running"
    state.research_mode = "deep"
    state.query_classification = {"complexity": "medium"}
    state.findings["final_output"] = "stale output from interrupted run"
    state.save_checkpoint(checkpoint_path)

    engine = RLMEngine(gemini_client=object(), config=RLMConfig(enable_matter_model=False))

    async def _fake_loop(state_obj, repo):
        state_obj.findings["loop_seen_query"] = state_obj.query

    async def _fake_verify(state_obj, repo):
        return None

    async def _fake_synthesize(state_obj):
        state_obj.findings["final_output"] = "fresh output"

    monkeypatch.setattr(engine, "_investigate_loop", _fake_loop)
    monkeypatch.setattr(engine, "_verify_citations", _fake_verify)
    monkeypatch.setattr(engine, "_synthesize", _fake_synthesize)

    resumed = await engine.resume_investigation(
        checkpoint_path,
        follow_up_query="Focus on termination liability",
        research_mode="simple",
    )

    assert resumed.query == "Focus on termination liability"
    assert resumed.research_mode == "simple"
    assert resumed.findings["continued_from_query"] == "Original query"
    assert resumed.findings["follow_up_query"] == "Focus on termination liability"
    assert resumed.findings["query_history"] == ["Original query"]
    assert resumed.findings["continued_from_research_mode"] == "deep"
    assert resumed.findings["research_mode_override"] == "simple"
    assert resumed.findings["loop_seen_query"] == "Focus on termination liability"
    assert resumed.findings["final_output"] == "fresh output"
    assert any(
        lead.source == "follow_up_query" and lead.search_term == "Focus on termination liability"
        for lead in resumed.leads
    )


# ---------------------------------------------------------------------------
# SO-3 structured steering surface
# ---------------------------------------------------------------------------

def test_steering_surface_empty_on_clean_matter(model):
    """get_ledger_steering_surface() returns empty list when matter has no conflicts/gaps."""
    actions = model.get_ledger_steering_surface()
    assert isinstance(actions, list)
    # May return correct_assertion suggestions for any assertions in disputed/unknown state;
    # with empty matter there are none, so the list should be empty.
    assert actions == []


def test_steering_surface_suggests_redirect_for_low_coverage_issue(model):
    """get_ledger_steering_surface() returns redirect_focus action for uncovered issue."""
    from irys.matter.enums import IssueType

    issue_id, _ = model.issues.upsert_issue(
        title="Breach of contract claim",
        issue_type=IssueType.CLAIM,
        materiality=0.9,
    )
    # No assertions linked → coverage = 0

    actions = model.get_ledger_steering_surface()
    redirect_actions = [a for a in actions if a["action_type"] == "redirect_focus"]
    assert len(redirect_actions) >= 1, "Expected redirect_focus action for 0% covered issue"

    action = redirect_actions[0]
    assert action["params"]["issue_id"] == issue_id
    assert action["priority"] in ("high", "medium")
    assert "description" in action
    assert "rationale" in action
    assert "impact" in action


def test_steering_surface_suggests_correct_for_disputed_assertion(model):
    """get_ledger_steering_surface() returns correct_assertion for disputed assertions."""
    from irys.matter.enums import AssertionLinkType

    a1 = add_assertion(model, "Defendant breached clause 4.2", doc="complaint.pdf")
    a2 = add_assertion(model, "Defendant did not breach clause 4.2", doc="answer.pdf")
    # Link a1 attacks a2 → a2 becomes disputed
    model.assertions.link(a1, a2, AssertionLinkType.ATTACKS)
    model.assertions.set_belief_state(a2, BeliefState.DISPUTED)

    actions = model.get_ledger_steering_surface()
    correct_actions = [a for a in actions if a["action_type"] in ("correct_assertion", "force_belief_state")]
    assert len(correct_actions) >= 1, "Expected steering action for disputed assertion"


def test_steering_surface_suggests_answer_for_pending_clarification(model):
    """get_ledger_steering_surface() returns answer_clarification for pending questions."""
    q_id = model.clarifications.add_question(
        question_text="Is the signed amendment dated before the alleged breach?",
        why_it_matters="The amendment date determines which obligations apply.",
        expected_impact="Resolves the operative-version gap for the contract.",
    )

    actions = model.get_ledger_steering_surface()
    clarify_actions = [a for a in actions if a["action_type"] == "answer_clarification"]
    assert len(clarify_actions) >= 1, "Expected answer_clarification action for pending question"

    action = clarify_actions[0]
    assert action["params"]["question_id"] == q_id
    assert "answer_text" in action["params"]
    assert action["priority"] == "medium"


def test_steering_surface_suggests_supply_document_for_missing_doc_gap(model):
    """get_ledger_steering_surface() returns supply_document for high-materiality missing docs."""
    from irys.matter.enums import GapType as GT

    gap_id = model.record_gap(
        gap_type=GT.MISSING_DOCUMENT,
        description="Signed amendment #3 referenced but not produced",
        materiality=0.9,
    )

    actions = model.get_ledger_steering_surface()
    supply_actions = [a for a in actions if a["action_type"] == "supply_document"]
    assert len(supply_actions) >= 1, "Expected supply_document action for missing doc gap"

    action = supply_actions[0]
    assert action["priority"] in ("high", "medium")
    assert "Signed amendment" in action["description"]


def test_steering_surface_returns_sorted_by_priority(model):
    """get_ledger_steering_surface() returns actions sorted high → medium → low."""
    from irys.matter.enums import IssueType, GapType as GT

    # Create a low-coverage issue (medium/high priority)
    model.issues.upsert_issue(
        title="Damages computation",
        issue_type=IssueType.DAMAGES,
        materiality=0.7,
    )
    # Add a disputed assertion (medium priority)
    a1 = add_assertion(model, "Plaintiff claimed $500k")
    a2 = add_assertion(model, "Plaintiff claimed only $200k")
    model.assertions.set_belief_state(a1, BeliefState.DISPUTED)

    actions = model.get_ledger_steering_surface()
    if len(actions) >= 2:
        priorities = [a["priority"] for a in actions]
        order = {"high": 0, "medium": 1, "low": 2}
        scores = [order[p] for p in priorities]
        assert scores == sorted(scores), f"Actions not sorted by priority: {priorities}"


def test_steering_surface_respects_limit(model):
    """get_ledger_steering_surface(limit=N) returns at most N actions."""
    # Add many disputed assertions
    for i in range(10):
        a = add_assertion(model, f"Disputed fact {i}")
        model.assertions.set_belief_state(a, BeliefState.DISPUTED)

    actions = model.get_ledger_steering_surface(limit=3)
    assert len(actions) <= 3


def test_steering_surface_action_shape(model):
    """Every action in get_ledger_steering_surface() has required fields."""
    a = add_assertion(model, "Some disputed claim")
    model.assertions.set_belief_state(a, BeliefState.DISPUTED)

    actions = model.get_ledger_steering_surface()
    for action in actions:
        assert "action_id" in action
        assert "action_type" in action
        assert "description" in action
        assert "params" in action
        assert isinstance(action["params"], dict)
        assert "rationale" in action
        assert "priority" in action
        assert action["priority"] in ("high", "medium", "low")
        assert "impact" in action
