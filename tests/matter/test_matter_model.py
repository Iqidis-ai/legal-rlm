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


def test_stats_reflects_current_state(model):
    """stats() must accurately reflect the current assertion and gap counts (SO-1/SO-7).

    stats() is used to show the user how much intelligence has been accumulated
    (SO-1) and how many gaps remain (SO-7).  If it returns stale or incorrect
    counts, the user sees a misleading picture of matter model completeness.
    """
    # Add assertions
    from irys.matter.runtime import MatterRuntimeAdapter
    run_id = model.start_run("stats test")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_fact("Fact A.", "doc1.pdf")
    adapter.record_fact("Fact B.", "doc2.pdf")

    # Add a gap
    model.record_gap(GapType.MISSING_DOCUMENT, "Missing exhibit", materiality=0.7)

    stats = model.stats()
    assert stats["assertion_count"] == 2, (
        "stats() must return the correct assertion count after recording facts"
    )
    assert stats["open_gap_count"] == 1, (
        "stats() must reflect the number of open gaps in the matter model (SO-7)"
    )


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


def test_open_gaps_min_materiality_filters_low_materiality_gaps(model):
    """open_gaps(min_materiality=X) must exclude gaps whose materiality is below X (SO-7).

    High-materiality gaps are those whose absence materially affects the analysis.
    Low-materiality gaps (e.g. minor CC emails) should not surface in most contexts.
    The filter must correctly exclude low-materiality gaps so the system only
    surfaces actionable missingness.
    """
    # High-materiality gap (0.9) — should appear with min_materiality=0.5
    model.record_gap(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Signed amendment not found",
        materiality=0.9,
    )
    # Low-materiality gap (0.1) — should NOT appear with min_materiality=0.5
    model.record_gap(
        gap_type=GapType.MISSING_DOCUMENT,
        description="CC email chain not found",
        materiality=0.1,
    )

    high_gaps = model.gaps.open_gaps(min_materiality=0.5)
    all_gaps = model.gaps.open_gaps(min_materiality=0.0)

    assert len(all_gaps) == 2, "Both gaps must be stored"
    assert len(high_gaps) == 1, (
        "open_gaps(min_materiality=0.5) must exclude the gap with materiality=0.1"
    )
    assert high_gaps[0]["description"] == "Signed amendment not found"


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


def test_build_query_context_excludes_low_materiality_gaps(model):
    """build_query_context() must not surface low-materiality gaps in the orientation context (SO-7).

    Low-materiality gaps (< 0.3) are stored for completeness but must not
    clutter the orientation prompt.  Only actionable gaps (materiality >= 0.3)
    should appear in ctx.open_gaps so the LLM focuses on material missingness
    and not minor peripheral absences.
    """
    # High-materiality gap — should appear in context
    model.record_gap(GapType.MISSING_DOCUMENT, "Signed amendment not found", materiality=0.9)
    # Low-materiality gap — must NOT appear in context
    model.record_gap(GapType.MISSING_DOCUMENT, "Minor CC email not found", materiality=0.1)

    ctx = model.build_query_context()
    assert len(ctx.open_gaps) == 1, (
        "build_query_context() must exclude gaps below the materiality threshold "
        "(SO-7: only actionable missingness surfaces in orientation)"
    )
    assert ctx.open_gaps[0]["description"] == "Signed amendment not found"


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


def test_gap_record_many_intra_batch_dedup(model):
    """record_many() with duplicate specs in one call returns the same id for each duplicate."""
    specs = [
        {"gap_type": GapType.MISSING_DOCUMENT, "description": "Same gap"},
        {"gap_type": GapType.MISSING_DOCUMENT, "description": "Same gap"},  # duplicate
    ]
    ids = model.gaps.record_many(specs)
    assert len(ids) == 2, "Return list length must match input specs length"
    assert ids[0] == ids[1], "Intra-batch duplicates must return the same gap_id"
    assert len(model.gaps.open_gaps(min_materiality=0.0)) == 1, "Only one gap should be created"


# ---------------------------------------------------------------------------
# SO-1: Durable persistence across sessions (SQLite file backend)
# ---------------------------------------------------------------------------

def test_matter_model_persists_assertions_across_reopens(tmp_path):
    """Opening a model, recording assertions, closing and reopening must preserve data (SO-1).

    This is the core SO-1 invariant: the system must never rediscover stable
    structure from scratch.  Assertions written in session 1 must survive
    into session 2 when the model is reopened from disk.
    """
    # --- Session 1: write ---
    model1 = MatterModel.open(tmp_path, matter_name="Acme v Tech")
    cand = AssertionCandidate(
        proposition_text="Defendant failed to deliver by the deadline.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="complaint.pdf",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid, _ = model1.assertions.upsert_occurrence(cand)
    del model1  # Explicitly close / GC the model

    # --- Session 2: read ---
    model2 = MatterModel.open(tmp_path)
    record = model2.assertions.get(aid)
    assert record is not None, (
        "Assertion must be readable after reopening the model from disk (SO-1)"
    )
    assert record.proposition_text == "Defendant failed to deliver by the deadline."
    # Confirm it's the same matter (same matter_id stored on disk)
    stats = model2.stats()
    assert stats["assertion_count"] == 1


# ---------------------------------------------------------------------------
# SO-7: open_gaps() includes dependencies (gap → issue/assertion link)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SO-1: build_query_context() populates known_document_ids
# ---------------------------------------------------------------------------

def test_build_query_context_populates_known_document_ids(model):
    """build_query_context() must populate known_document_ids from assertion occurrences (SO-1).

    This prevents redundant re-reading of documents already in the matter model.
    The engine injects known_document_ids into the orientation prompt so the LLM
    can exclude already-indexed documents from new searches — the core SO-1
    'no redundant recompute' guarantee at the document level.
    """
    from irys.matter.runtime import MatterRuntimeAdapter

    run_id = model.start_run("doc ids test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.record_fact("Contract requires payment by Jan 15.", document_id="contract.pdf")
    adapter.record_fact("Breach occurred on Jan 16.", document_id="complaint.pdf")
    adapter.record_fact("Defendant denies breach.", document_id="answer.pdf")

    ctx = model.build_query_context()

    assert "contract.pdf" in ctx.known_document_ids, (
        "Documents that contributed assertions must appear in known_document_ids (SO-1)"
    )
    assert "complaint.pdf" in ctx.known_document_ids
    assert "answer.pdf" in ctx.known_document_ids


def test_build_query_context_known_document_ids_empty_with_no_assertions(model):
    """known_document_ids must be empty when no assertions have been recorded."""
    ctx = model.build_query_context()
    assert ctx.known_document_ids == []


def test_build_query_context_populates_known_actors(model):
    """build_query_context() must populate known_actors from the actor store (SO-5).

    The engine injects known_actors into prompts so the LLM can reference
    established party identities by canonical name rather than guessing from
    raw document text.
    """
    model.actors.upsert_actor("Acme Corporation", actor_type="company")
    model.actors.upsert_actor("John Smith", actor_type="person")

    ctx = model.build_query_context()

    assert "Acme Corporation" in ctx.known_actors, (
        "Recorded actors must appear in known_actors in query context (SO-5)"
    )
    assert "John Smith" in ctx.known_actors


def test_open_gaps_includes_dependencies(model):
    """open_gaps() must include a 'dependencies' key with linked entity info (SO-7).

    The core SO-7 value is: the system identifies *which conclusions depend on*
    the missing document or predicate.  open_gaps() pre-fetches gap_link rows
    and attaches them under 'dependencies' so callers (e.g. _build_gap_summary)
    can surface the Affects: line without a second query.
    """
    from irys.matter.enums import GapType, IssueType

    issue_id, _ = model.issues.upsert_issue(
        title="Damages calculation", issue_type=IssueType.DAMAGES, materiality=0.9
    )

    gap_id = model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Expert damages report referenced but not found",
        materiality=0.8,
        affected_type="issue",
        affected_id=issue_id,
    )

    gaps = model.gaps.open_gaps(min_materiality=0.0)
    assert len(gaps) == 1

    gap = gaps[0]
    assert "dependencies" in gap, "open_gaps() must include a 'dependencies' key (SO-7)"
    deps = gap["dependencies"]
    assert len(deps) == 1, f"Expected 1 dependency, got {deps}"
    assert deps[0]["affected_type"] == "issue"
    assert deps[0]["affected_id"] == issue_id


def test_open_gaps_no_deps_returns_empty_list(model):
    """Gaps without any gap_links must have dependencies=[] (not absent key)."""
    from irys.matter.enums import GapType

    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Exhibit A not provided",
        materiality=0.6,
    )

    gaps = model.gaps.open_gaps(min_materiality=0.0)
    assert gaps[0]["dependencies"] == [], (
        "Unlinked gap must have dependencies=[] (empty list, not missing key)"
    )


def test_matter_model_hot_path_survives_reopen(tmp_path):
    """A document marked as ingested must still appear ingested on next open (SO-1 hot-path)."""
    model1 = MatterModel.open(tmp_path)
    doc_id, _ = model1.inventory.upsert(
        relative_path="contract.pdf",
        sha256="abc123",
        size_bytes=1024,
        file_type="pdf",
    )
    model1.inventory.mark_ingested(doc_id)
    del model1

    model2 = MatterModel.open(tmp_path)
    assert model2.inventory.is_ingested("contract.pdf"), (
        "is_ingested() must return True for a doc marked ingested in a prior session (SO-1)"
    )


def test_build_query_context_populates_existing_actor_count(model):
    """build_query_context() must include the actor count in the context (SO-5).

    The engine uses existing_actor_count to surface 'N known parties' in orientation
    so the LLM knows how populated the actor register is before extracting more actors.
    """
    assert model.build_query_context().existing_actor_count == 0

    model.actors.upsert_actor("Plaintiff Corp", actor_type="company")
    model.actors.upsert_actor("Defendant LLC", actor_type="company")

    ctx = model.build_query_context()
    assert ctx.existing_actor_count == 2, (
        "existing_actor_count must reflect the number of actors in the store (SO-5)"
    )
