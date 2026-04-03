"""Integration tests for the MatterModel facade and reasoning ledger.

Test 5 (Codex): Engine bridge with enable_matter_model=False preserves
current InvestigationResult behavior; with it enabled, the same run also
writes a run_session, ledger events, and assertions.
"""

import pytest
from irys.matter import (
    MatterModel, AssertionCandidate, SpeechAct, SourceRole,
    ModelLayer, AssertionKind, LedgerEventType, GapType,
    RevisionCause, BeliefState,
)
from irys.matter.enums import OriginKind
from irys.matter.runtime import MatterRuntimeAdapter, NullMatterAdapter


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# MatterModel lifecycle
# ---------------------------------------------------------------------------

def test_open_in_memory_creates_matter(model):
    row = model.db.execute(
        "SELECT id FROM matter WHERE id=?", (model.matter_id,)
    ).fetchone()
    assert row is not None


def test_stats_initial(model):
    stats = model.stats()
    assert stats["assertion_count"] == 0
    assert stats["open_gap_count"] == 0


def test_run_lifecycle(model):
    run_id = model.start_run("What are the payment obligations?")
    assert run_id

    run = model.ledger.get_run(run_id)
    assert run.status == "running"

    # Events should include run_started
    events = model.ledger.get_events(run_id)
    assert any(e["event_type"] == LedgerEventType.RUN_STARTED.value for e in events)

    model.complete_run(run_id)
    run = model.ledger.get_run(run_id)
    assert run.status == "completed"
    assert run.completed_at is not None


def test_run_fail(model):
    run_id = model.start_run("Test query")
    model.fail_run(run_id, "Gemini API error")
    run = model.ledger.get_run(run_id)
    assert run.status == "failed"


def test_ledger_append_and_retrieve(model):
    run_id = model.start_run("Test ledger")
    model.ledger.append_event(
        run_id=run_id,
        event_type=LedgerEventType.ASSERTION_ADDED,
        summary="New assertion: contract requires payment",
        why="Found in contract.pdf §3.1",
    )
    events = model.ledger.get_events(run_id)
    added_events = [e for e in events if e["event_type"] == LedgerEventType.ASSERTION_ADDED.value]
    assert len(added_events) == 1
    assert "contract" in added_events[0]["summary"]


def test_ledger_sequence_numbers_are_ordered(model):
    run_id = model.start_run("Sequence test")
    for i in range(5):
        model.ledger.append_event(
            run_id=run_id,
            event_type=LedgerEventType.BRANCH_SELECTED,
            summary=f"Step {i}",
        )
    events = model.ledger.get_events(run_id)
    seq_nos = [e["seq_no"] for e in events]
    assert seq_nos == sorted(seq_nos)
    assert seq_nos[0] == 0


def test_stop_requested_flag(model):
    run_id = model.start_run("Interruptible run")
    assert not model.ledger.is_stop_requested(run_id)
    model.ledger.request_stop(run_id)
    assert model.ledger.is_stop_requested(run_id)


# ---------------------------------------------------------------------------
# Gap store
# ---------------------------------------------------------------------------

def test_gap_recorded_and_retrieved(model):
    gap_id = model.record_gap(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Signed amendment referenced in §4.2 but not present in repository",
        expected_artifact="Amendment #1 to Service Agreement",
        materiality=0.9,
    )
    assert gap_id

    open_gaps = model.gaps.open_gaps(min_materiality=0.5)
    assert len(open_gaps) == 1
    assert open_gaps[0]["gap_type"] == GapType.MISSING_DOCUMENT.value


# ---------------------------------------------------------------------------
# Test 5: NullMatterAdapter is a safe no-op
# ---------------------------------------------------------------------------

def test_null_adapter_is_noop():
    adapter = NullMatterAdapter()
    assert adapter.get_context() is None
    assert adapter.record_fact("Some fact", "doc1") == ""
    assert adapter.flush_revisions() == 0
    assert adapter.is_stop_requested() is False
    adapter.request_stop()  # must not raise
    adapter.log_step("step", "why")  # must not raise


def test_matter_runtime_adapter_records_facts(model):
    run_id = model.start_run("Adapter test")
    adapter = MatterRuntimeAdapter(model, run_id)

    fact_id = adapter.record_fact(
        proposition_text="The contract was executed on January 15, 2024.",
        document_id="contract.pdf",
        source_role=SourceRole.OPERATIVE,
        speech_act=SpeechAct.OPERATIVE,
    )
    assert fact_id

    # Should be in the assertion store
    assert model.assertions.count() == 1
    assert model.assertions.get(fact_id) is not None


def test_matter_runtime_adapter_flush_revisions(model):
    run_id = model.start_run("Revision flush test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.record_fact("Fact A.", document_id="doc1")
    adapter.record_fact("Fact B.", document_id="doc2")

    # flush should not raise even with no link structure
    count = adapter.flush_revisions()
    assert isinstance(count, int)


def test_matter_runtime_adapter_stop_requested(model):
    run_id = model.start_run("Stop test")
    adapter = MatterRuntimeAdapter(model, run_id)

    assert not adapter.is_stop_requested()
    adapter.request_stop()
    assert adapter.is_stop_requested()


# ---------------------------------------------------------------------------
# Build query context
# ---------------------------------------------------------------------------

def test_build_query_context_empty_matter(model):
    ctx = model.build_query_context()
    assert ctx.matter_id == model.matter_id
    assert ctx.existing_assertion_count == 0
    assert ctx.open_gaps == []


def test_build_query_context_with_data(model):
    # Add some assertions and a gap
    for text in ["Fact A.", "Fact B.", "Fact C."]:
        c = AssertionCandidate(
            proposition_text=text,
            model_layer=ModelLayer.RECORD,
            assertion_kind=AssertionKind.FACTUAL,
            document_id="doc1",
            speech_act=SpeechAct.EXTRACTED,
            source_role=SourceRole.UNKNOWN,
            origin_kind=OriginKind.EXTRACTED,
        )
        model.assertions.upsert_occurrence(c)

    model.record_gap(GapType.MISSING_DOCUMENT, "Missing schedule A", materiality=0.8)

    ctx = model.build_query_context()
    assert ctx.existing_assertion_count == 3
    assert len(ctx.open_gaps) == 1


# ---------------------------------------------------------------------------
# Gap store deduplication (SO-7: idempotent gap recording)
# ---------------------------------------------------------------------------

def test_gap_record_idempotent_same_description(model):
    """Calling record() twice with the same gap_type+description returns the same gap_id."""
    id1 = model.gaps.record(GapType.MISSING_DOCUMENT, "Amendment #1 not found")
    id2 = model.gaps.record(GapType.MISSING_DOCUMENT, "Amendment #1 not found")
    assert id1 == id2
    assert len(model.gaps.open_gaps(min_materiality=0.0)) == 1


def test_gap_record_idempotent_normalizes_whitespace(model):
    """Trailing spaces and case differences collapse to the same gap."""
    id1 = model.gaps.record(GapType.MISSING_DOCUMENT, "  Amendment #1 not found  ")
    id2 = model.gaps.record(GapType.MISSING_DOCUMENT, "Amendment #1 Not Found")
    assert id1 == id2


def test_gap_record_different_types_not_deduplicated(model):
    """Same description with different gap_type creates distinct gaps."""
    id1 = model.gaps.record(GapType.MISSING_DOCUMENT, "Amendment #1 not found")
    id2 = model.gaps.record(GapType.MISSING_ISSUE_PREDICATE, "Amendment #1 not found")
    assert id1 != id2
    assert len(model.gaps.open_gaps(min_materiality=0.0)) == 2


def test_gap_record_reopen_closed_gap(model):
    """Re-detecting a gap that was closed reopens it rather than creating a duplicate."""
    gap_id = model.gaps.record(GapType.MISSING_DOCUMENT, "Exhibit A not attached")
    # Close it
    model.db.execute("UPDATE gap SET status='resolved' WHERE id=?", (gap_id,))
    assert len(model.gaps.open_gaps(min_materiality=0.0)) == 0

    # Re-detect the same gap
    reopened_id = model.gaps.record(GapType.MISSING_DOCUMENT, "Exhibit A not attached")
    assert reopened_id == gap_id
    assert len(model.gaps.open_gaps(min_materiality=0.0)) == 1


def test_gap_record_many_idempotent(model):
    """record_many() returns existing ids when specs duplicate existing gaps."""
    specs = [
        {"gap_type": GapType.MISSING_DOCUMENT, "description": "Missing invoice #42"},
        {"gap_type": GapType.MISSING_DOCUMENT, "description": "Missing payment receipt"},
    ]
    ids_first = model.gaps.record_many(specs)
    ids_second = model.gaps.record_many(specs)
    assert ids_first == ids_second
    assert len(model.gaps.open_gaps(min_materiality=0.0)) == 2
