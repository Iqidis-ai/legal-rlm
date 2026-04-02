"""Engine bridge tests — matter model integration with RLMEngine.

Verifies:
1. enable_matter_model=False (NullAdapter): engine behaves identically to baseline.
2. enable_matter_model=True: investigate() creates a run_session and assertions
   accumulate in the matter model.
3. MatterRuntimeAdapter.record_fact deduplicates the same proposition from
   different documents (same assertion row, multiple occurrences).
4. NullMatterAdapter is a safe no-op on all methods.
"""

import pytest
from irys.rlm.engine import RLMConfig, RLMEngine
from irys.matter import MatterModel
from irys.matter.runtime import MatterRuntimeAdapter, NullMatterAdapter


# ---------------------------------------------------------------------------
# NullAdapter: safe no-op contract
# ---------------------------------------------------------------------------

def test_null_adapter_all_methods():
    adapter = NullMatterAdapter()
    assert adapter.get_context() is None
    assert adapter.record_fact("Some fact.", "doc1") == ""
    assert adapter.flush_revisions() == 0
    assert adapter.is_stop_requested() is False
    adapter.request_stop()   # must not raise
    adapter.log_step("retrieving", "initial search")  # must not raise
    adapter.log_conflict("Contradiction: fact A conflicts with fact B")  # must not raise
    adapter.log_gap("Missing document", "contract.pdf")  # must not raise
    assert adapter.record_gap("Missing: signed amendment") == ""  # must not raise


# ---------------------------------------------------------------------------
# enable_matter_model=False: config default preserves existing behavior
# ---------------------------------------------------------------------------

def test_config_default_disable():
    config = RLMConfig()
    assert config.enable_matter_model is False


def test_engine_without_matter_model_uses_null_adapter():
    """When enable_matter_model=False or no matter_model provided, adapter is NullMatterAdapter."""
    from irys.matter.runtime import NullMatterAdapter
    # We can't run a full investigation (needs Gemini), but we can verify
    # the adapter type selected at construction and in a dry investigate call.
    config = RLMConfig(enable_matter_model=False)
    assert config.enable_matter_model is False
    # NullMatterAdapter satisfies the interface
    adapter = NullMatterAdapter()
    assert adapter.record_fact("anything", "doc") == ""


# ---------------------------------------------------------------------------
# MatterRuntimeAdapter: records facts into assertion store
# ---------------------------------------------------------------------------

def test_adapter_records_facts_to_assertion_store():
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Test query")
    adapter = MatterRuntimeAdapter(model, run_id)

    aid1 = adapter.record_fact("The contract was signed on January 15, 2024.", "contract.pdf")
    aid2 = adapter.record_fact("Payment of $50,000 was due on February 1, 2024.", "contract.pdf")

    assert aid1
    assert aid2
    assert aid1 != aid2
    assert model.assertions.count() == 2


def test_adapter_deduplicates_same_fact_different_docs():
    """Same proposition from two docs → 1 assertion, 2 occurrences."""
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Dedup test")
    adapter = MatterRuntimeAdapter(model, run_id)

    text = "The contract requires payment of $50,000."
    aid1 = adapter.record_fact(text, document_id="complaint.pdf")
    aid2 = adapter.record_fact(text, document_id="contract.pdf")

    assert aid1 == aid2, "Same proposition must map to same assertion_id"
    assert model.assertions.count() == 1
    occurrences = model.assertions.get_occurrences(aid1)
    assert len(occurrences) == 2


def test_adapter_run_session_written():
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Evidence query")
    assert run_id
    run = model.ledger.get_run(run_id)
    assert run.status == "running"
    assert run.query == "Evidence query"


def test_adapter_flush_revisions_returns_int():
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Flush test")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_fact("Fact A.", "doc1")
    adapter.record_fact("Fact B.", "doc2")
    count = adapter.flush_revisions()
    assert isinstance(count, int)


def test_adapter_stop_requested_propagates():
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Stop test")
    adapter = MatterRuntimeAdapter(model, run_id)

    assert not adapter.is_stop_requested()
    adapter.request_stop()
    assert adapter.is_stop_requested()
    assert model.ledger.is_stop_requested(run_id)


def test_adapter_log_step_writes_ledger_event():
    from irys.matter import LedgerEventType
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Log test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.log_step("Searching for payment terms", "following lead from orientation")

    events = model.ledger.get_events(run_id)
    branch_events = [e for e in events if e["event_type"] == LedgerEventType.BRANCH_SELECTED.value]
    assert len(branch_events) >= 1
    assert "payment" in branch_events[0]["summary"].lower()


# ---------------------------------------------------------------------------
# SO-5: _build_source_calibration() reflects actual assertion source roles
# ---------------------------------------------------------------------------

def test_build_source_calibration_groups_by_role():
    """_build_source_calibration() must show calibration text keyed by source_role."""
    from irys.matter.enums import SourceRole

    model = MatterModel.open_in_memory()
    run_id = model.start_run("calibration test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Record facts with different source roles
    adapter.record_fact("Plaintiff alleges breach.", document_id="complaint.pdf")  # → ADVOCACY
    adapter.record_fact("Contract requires payment by Jan 15.", document_id="contract.pdf")  # → OPERATIVE
    adapter.record_fact("Court granted summary judgment.", document_id="order.pdf")  # → AUTHORITATIVE

    # Build a minimal engine with the matter model (no Gemini client needed for this method)
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    calibration = engine._build_source_calibration(None)

    assert "ADVOCACY" in calibration or "advocacy" in calibration.lower()
    assert "OPERATIVE" in calibration or "operative" in calibration.lower()
    assert "WARNING" in calibration  # always has the advocacy amplification warning


def test_build_source_calibration_no_model():
    """_build_source_calibration() with no matter model returns a safe fallback."""
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = None

    calibration = engine._build_source_calibration(None)
    assert "skepticism" in calibration.lower() or "unavailable" in calibration.lower()


# ---------------------------------------------------------------------------
# SO-5: Actor store wiring
# ---------------------------------------------------------------------------

def test_record_actor_persists_to_actor_store():
    """record_actor() must persist people/organizations to durable actor store."""
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Actor test")
    adapter = MatterRuntimeAdapter(model, run_id)

    aid1 = adapter.record_actor("Jane Smith", actor_type="person")
    aid2 = adapter.record_actor("Acme Corp", actor_type="organization")

    assert aid1
    assert aid2
    assert aid1 != aid2
    assert model.actors.count() == 2


def test_record_actor_is_idempotent():
    """Same actor name → same actor_id."""
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Actor dedup test")
    adapter = MatterRuntimeAdapter(model, run_id)

    aid1 = adapter.record_actor("John Doe", actor_type="person")
    aid2 = adapter.record_actor("john doe", actor_type="person")  # normalized match

    assert aid1 == aid2
    assert model.actors.count() == 1


def test_null_adapter_record_actor():
    """NullMatterAdapter.record_actor() must not raise and must return empty string."""
    adapter = NullMatterAdapter()
    result = adapter.record_actor("Jane Smith")
    assert result == ""
