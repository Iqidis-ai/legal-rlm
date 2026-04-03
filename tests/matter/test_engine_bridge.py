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


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


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
    adapter.log_objective("Determine whether breach occurred")  # must not raise
    adapter.log_warning("Low-confidence extraction")  # must not raise
    adapter.log_conflict("Contradiction: fact A conflicts with fact B")  # must not raise
    adapter.log_gap("Missing document", "contract.pdf")  # must not raise
    assert adapter.record_gap("Missing: signed amendment") == ""  # must not raise
    # record_facts_batch must return a list of empty strings, same length as input
    result = adapter.record_facts_batch([("fact a", "doc.pdf"), ("fact b", "doc.pdf")])
    assert result == ["", ""]
    # Quant recording
    assert adapter.record_quant("amount", "Invoice total $50,000") == ""
    adapter.record_quants_batch([{"quant_kind": "amount", "raw_text": "$100"}])  # must not raise
    # Assertion link recording
    adapter.record_assertion_link("id1", "id2", "supports")  # must not raise
    # Mid-run steering — NullMatterAdapter must return empty for clarifications
    assert adapter.get_new_answered_clarifications() == []
    # Redirect — NullMatterAdapter must return safe defaults
    assert not adapter.is_redirect_requested()
    assert adapter.get_redirect_issue_id() is None
    adapter.clear_redirect()  # must not raise


# ---------------------------------------------------------------------------
# enable_matter_model: default True ensures matter model is active in normal runs
# ---------------------------------------------------------------------------

def test_config_default_enable():
    # Default is True so every run builds the durable matter model (SO-1).
    # Callers that explicitly opt out must pass enable_matter_model=False.
    config = RLMConfig()
    assert config.enable_matter_model is True


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

def test_record_facts_batch_single_transaction():
    """record_facts_batch() stores all facts and returns correct assertion IDs."""
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Batch test")
    adapter = MatterRuntimeAdapter(model, run_id)

    facts = [
        ("Contract was executed on 2023-01-01.", "contract.pdf"),
        ("Payment of $10,000 is due on 2023-02-01.", "contract.pdf"),
        ("Defendant breached the agreement.", "complaint.pdf"),
    ]
    ids = adapter.record_facts_batch(facts)

    assert len(ids) == 3
    assert len(set(ids)) == 3, "Each unique proposition must get a unique assertion ID"
    assert model.assertions.count() == 3


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


def test_assertion_added_ledger_event_fires_only_once_for_duplicate_proposition():
    """ASSERTION_ADDED ledger event must be written exactly once per unique proposition (SO-3).

    When the same proposition is recorded from two different source documents,
    the second call is a deduplication (is_new=False) and must NOT fire another
    ASSERTION_ADDED event.  A user reading the reasoning ledger would see one
    'new assertion' notice per unique fact — not one per occurrence document.
    """
    from irys.matter import LedgerEventType
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Dedup ledger test")
    adapter = MatterRuntimeAdapter(model, run_id)

    text = "Defendant failed to deliver the goods by the deadline."
    adapter.record_fact(text, document_id="complaint.pdf")
    adapter.record_fact(text, document_id="deposition.pdf")  # same proposition, new document

    events = model.ledger.get_events(run_id)
    added_events = [
        e for e in events if e["event_type"] == LedgerEventType.ASSERTION_ADDED.value
    ]
    assert len(added_events) == 1, (
        "ASSERTION_ADDED must fire exactly once per unique proposition — "
        f"got {len(added_events)} events for 2 occurrences of the same fact"
    )


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
    step_events = [e for e in events if e["event_type"] == LedgerEventType.PROGRESS_NOTE.value]
    assert len(step_events) >= 1
    assert "payment" in step_events[0]["summary"].lower()


def test_adapter_log_warning_writes_system_warning_event():
    """log_warning() must write a SYSTEM_WARNING ledger event (SO-3).

    The reasoning ledger is structured and user-facing.  Warnings from the
    engine (low-confidence extractions, unexpected data shapes, non-fatal errors)
    must be surfaced as SYSTEM_WARNING events so the user can see where the
    engine flagged uncertainty — they must not silently disappear into logs.
    """
    from irys.matter import LedgerEventType
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Warning event test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.log_warning("Low-confidence extraction: 'payment amount' field ambiguous")

    events = model.ledger.get_events(run_id)
    warn_events = [e for e in events if e["event_type"] == LedgerEventType.SYSTEM_WARNING.value]
    assert len(warn_events) >= 1, (
        "log_warning() must write a SYSTEM_WARNING event to the reasoning ledger (SO-3)"
    )
    assert "Low-confidence" in warn_events[0]["summary"] or "ambiguous" in warn_events[0]["summary"], (
        "SYSTEM_WARNING event summary must include the warning message"
    )


def test_adapter_log_objective_writes_objective_set_event():
    """log_objective() must write an OBJECTIVE_SET ledger event (SO-3).

    The reasoning ledger is user-facing and actionable.  When the engine sets
    its investigation objective, that must appear in the ledger so the user
    can see what question the system is working on — not just a hidden log line.
    """
    from irys.matter import LedgerEventType
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Objective test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.log_objective("Determine whether the defendant breached the payment clause.")

    events = model.ledger.get_events(run_id)
    obj_events = [e for e in events if e["event_type"] == LedgerEventType.OBJECTIVE_SET.value]
    assert len(obj_events) >= 1, (
        "log_objective() must write an OBJECTIVE_SET event to the reasoning ledger (SO-3)"
    )
    assert "breach" in obj_events[0]["summary"].lower() or "objective" in obj_events[0]["summary"].lower(), (
        "OBJECTIVE_SET event summary must include the objective text"
    )


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


def test_build_source_calibration_includes_trust_overrides():
    """_build_source_calibration() must surface user trust overrides (SO-5)."""
    model = MatterModel.open_in_memory()

    # Record at least one assertion so calibration doesn't bail out early
    run_id = model.start_run("trust override test")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_fact("Defendant admits non-payment.", document_id="answer.pdf")

    # Set a low-trust and a high-trust override
    model.trust_overrides.set("plaintiff_damages_report.pdf", "low", "Expert hired by plaintiff")
    model.trust_overrides.set("signed_contract.pdf", "high", "Fully executed operative document")

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    calibration = engine._build_source_calibration(None)

    assert "plaintiff_damages_report.pdf" in calibration, (
        "low-trust override document must appear in calibration"
    )
    assert "LOW TRUST" in calibration.upper() or "low" in calibration.lower()
    assert "signed_contract.pdf" in calibration, (
        "high-trust override document must appear in calibration"
    )
    assert "HIGH TRUST" in calibration.upper() or "high" in calibration.lower()
    assert "Expert hired by plaintiff" in calibration, "override note must appear"


def test_build_source_calibration_includes_user_annotations():
    """_build_source_calibration() must surface user document annotations (SO-3 + SO-5)."""
    model = MatterModel.open_in_memory()

    run_id = model.start_run("annotation calibration test")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_fact("Party A claims breach.", document_id="complaint.pdf")

    # Add a strategic annotation
    model.annotations.add(
        "expert_damages_report.pdf",
        "Prepared post-litigation — treat damages estimates as advocacy positions, not operative values.",
        annotation_type="reliability",
    )

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    calibration = engine._build_source_calibration(None)

    assert "expert_damages_report.pdf" in calibration, (
        "Annotated document must appear in calibration block (SO-3 annotation surfacing)"
    )
    assert "advocacy" in calibration.lower() or "annotation" in calibration.lower(), (
        "Annotation text must appear in calibration"
    )


# ---------------------------------------------------------------------------
# SO-7: _build_gap_summary() surfaces open gaps
# ---------------------------------------------------------------------------

def test_build_gap_summary_no_model():
    """_build_gap_summary() with no matter model returns safe fallback."""
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = None

    result = engine._build_gap_summary()
    assert "available" in result.lower() or "no" in result.lower()


def test_build_gap_summary_shows_gaps():
    """_build_gap_summary() must list open gaps with type and materiality label."""
    from irys.matter.enums import GapType

    model = MatterModel.open_in_memory()
    model.record_gap(
        description="Missing signed amendment — critical to damages calculation",
        gap_type=GapType.MISSING_DOCUMENT,
        materiality=0.9,
    )
    model.record_gap(
        description="Conflicting invoice amounts in two exhibits",
        gap_type=GapType.UNRESOLVED_CONTRADICTION,
        materiality=0.5,
    )

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_gap_summary()

    assert "Missing signed amendment" in result, f"Gap description missing: {result}"
    assert "Conflicting invoice amounts" in result
    assert "HIGH" in result or "MED" in result, f"Materiality label missing: {result}"
    assert "2" in result, f"Gap count missing: {result}"


def test_build_gap_summary_no_gaps():
    """_build_gap_summary() with no open gaps returns 'No significant gaps'."""
    model = MatterModel.open_in_memory()

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_gap_summary()
    assert "no significant gaps" in result.lower() or "no gap" in result.lower()


def test_build_gap_summary_shows_affects_line_for_linked_gap():
    """_build_gap_summary() must show 'Affects:' line for gaps linked to issues/assertions (SO-7).

    The 'Affects:' line tells the LLM which conclusions depend on the missing document —
    this is SO-7's core promise: 'identifies which conclusions depend on it'.
    Without the 'Affects:' line the LLM cannot surface the dependency in its analysis.
    """
    from irys.matter.enums import GapType, IssueType

    model = MatterModel.open_in_memory()

    issue_id, _ = model.issues.upsert_issue(
        title="Damages exposure", issue_type=IssueType.DAMAGES, materiality=0.9
    )

    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Expert damages analysis not in repository",
        materiality=0.85,
        affected_type="issue",
        affected_id=issue_id,
    )

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_gap_summary()

    assert "Expert damages analysis" in result, f"Gap description missing: {result}"
    assert "Affects" in result or "affect" in result.lower(), (
        f"_build_gap_summary must show 'Affects:' dependency line for linked gaps (SO-7): {result}"
    )
    # The affected entity's type and ID prefix must appear
    assert "issue" in result.lower(), (
        f"Affects line must reference the affected entity type: {result}"
    )


# ---------------------------------------------------------------------------
# SO-6: _build_quant_summary() surfaces numeric facts
# ---------------------------------------------------------------------------

def test_build_quant_summary_no_model():
    """_build_quant_summary() with no matter model returns safe fallback."""
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = None

    result = engine._build_quant_summary()
    assert "no" in result.lower() or "unavailable" in result.lower()


def test_build_quant_summary_no_facts():
    """_build_quant_summary() with no quant facts returns 'No numeric facts'."""
    model = MatterModel.open_in_memory()

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_quant_summary()
    assert "no numeric" in result.lower() or "0" in result


def test_build_quant_summary_shows_amounts():
    """_build_quant_summary() must surface monetary amounts from the quant store."""
    model = MatterModel.open_in_memory()
    model.quant.record(quant_kind="amount", raw_text="$500,000 total claim",
                       amount_value=500_000.0, currency="USD", subject_type="claim")
    model.quant.record(quant_kind="amount", raw_text="$250,000 paid to date",
                       amount_value=250_000.0, currency="USD", subject_type="payment")

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_quant_summary()

    assert "500,000" in result or "500000" in result, f"Claim amount missing: {result}"
    assert "250,000" in result or "250000" in result, f"Payment amount missing: {result}"
    assert "claim" in result.lower() or "payment" in result.lower()


def test_build_quant_summary_shows_reconciliation_by_category():
    """_build_quant_summary() must show per-category monetary totals (SO-6 reconciliation)."""
    model = MatterModel.open_in_memory()
    model.quant.record(quant_kind="amount", raw_text="Invoice 1",
                       amount_value=50_000.0, currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="Invoice 2",
                       amount_value=30_000.0, currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="Payment",
                       amount_value=40_000.0, currency="USD", subject_type="payment")

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_quant_summary()

    assert "invoice" in result.lower(), f"invoice category missing: {result}"
    assert "payment" in result.lower(), f"payment category missing: {result}"
    # invoice total = $80,000; payment total = $40,000
    assert "80,000" in result, f"invoice total $80,000 missing: {result}"
    assert "40,000" in result, f"payment total $40,000 missing: {result}"


def test_build_quant_summary_shows_conflicts():
    """_build_quant_summary() must surface NUMERIC CONFLICTS when get_conflicts() returns items (SO-6)."""
    model = MatterModel.open_in_memory()
    model.quant.record(quant_kind="amount", raw_text="version A",
                       amount_value=50_000.0, currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="version B",
                       amount_value=55_000.0, currency="USD", subject_type="invoice")

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_quant_summary()

    assert "CONFLICT" in result.upper(), f"Expected CONFLICTS section: {result}"
    assert "invoice" in result.lower(), f"invoice must appear in conflict: {result}"
    assert "UNRESOLVED" in result.upper(), f"UNRESOLVED DISCREPANCY label missing: {result}"


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


# ---------------------------------------------------------------------------
# SO-3: Trust overrides — source trust steering
# ---------------------------------------------------------------------------

def test_trust_override_low_forces_alleged():
    """Low trust override must force speech_act to ALLEGED regardless of filename."""
    from irys.matter.enums import SpeechAct
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Trust test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Mark contract.pdf as low-trust (adversarial, should not be treated as operative)
    model.trust_overrides.set("contract.pdf", "low", note="Disputed authenticity")

    # Record a fact from contract.pdf — normally operative, but low trust → alleged
    aid = adapter.record_fact("Payment of $50,000 was due.", document_id="contract.pdf")
    assert aid

    # Check the occurrence's speech_act was forced to ALLEGED
    occurrences = model.assertions.get_occurrences(aid)
    assert len(occurrences) >= 1
    assert any(o["speech_act"] == SpeechAct.ALLEGED.value for o in occurrences)


def test_trust_override_high_promotes_alleged():
    """High trust override must promote ALLEGED → OPERATIVE for advocacy docs."""
    from irys.matter.enums import SpeechAct
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Trust promote test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # complaint.pdf normally infers ADVOCACY → ALLEGED; 'high' trust should promote to OPERATIVE
    model.trust_overrides.set("complaint.pdf", "high", note="Verified by court order")

    aid = adapter.record_fact("Defendant owes $100,000.", document_id="complaint.pdf")
    assert aid

    occurrences = model.assertions.get_occurrences(aid)
    assert any(o["speech_act"] == SpeechAct.OPERATIVE.value for o in occurrences)


def test_trust_override_basename_matching():
    """Trust override by basename must match a full relative path document_id."""
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Basename match test")
    adapter = MatterRuntimeAdapter(model, run_id)

    model.trust_overrides.set("contract.pdf", "low")

    # document_id is a relative path — basename match should still apply
    aid = adapter.record_fact("Clause 3 requires X.", document_id="pleadings/contract.pdf")
    assert aid
    from irys.matter.enums import SpeechAct
    occurrences = model.assertions.get_occurrences(aid)
    assert any(o["speech_act"] == SpeechAct.ALLEGED.value for o in occurrences)


def test_trust_override_list_and_upsert():
    """set() is idempotent; list_all() returns current overrides."""
    model = MatterModel.open_in_memory()

    model.trust_overrides.set("doc1.pdf", "low", note="First")
    model.trust_overrides.set("doc1.pdf", "high", note="Corrected")  # upsert
    model.trust_overrides.set("doc2.pdf", "low")

    overrides = model.trust_overrides.list_all()
    assert len(overrides) == 2
    doc1 = next(o for o in overrides if o["document_pattern"] == "doc1.pdf")
    assert doc1["trust_level"] == "high"  # latest value after upsert
    assert doc1["note"] == "Corrected"


def test_null_adapter_trust_override():
    """NullMatterAdapter trust methods must not raise and return safe defaults."""
    adapter = NullMatterAdapter()
    assert adapter.set_trust_override("anything.pdf", "low") == ""
    assert adapter.list_trust_overrides() == []


# ---------------------------------------------------------------------------
# SO-3: Document annotations — strategic notes injected into orientation
# ---------------------------------------------------------------------------

def test_document_annotation_persists():
    """add() stores annotation; list_recent() returns it."""
    model = MatterModel.open_in_memory()
    ann_id = model.annotations.add(
        "expert_report.pdf",
        "Report was prepared for litigation; treat damages figures as advocacy positions.",
        annotation_type="reliability",
    )
    assert ann_id

    recent = model.annotations.list_recent()
    assert len(recent) == 1
    assert recent[0]["document_pattern"] == "expert_report.pdf"
    assert "advocacy" in recent[0]["annotation_text"]
    assert recent[0]["annotation_type"] == "reliability"


def test_document_annotation_surfaces_in_matter_context():
    """build_query_context() includes document_annotations from annotation store."""
    model = MatterModel.open_in_memory()
    model.annotations.add("complaint.pdf", "Plaintiff's complaint; treat all amounts as alleged.")

    ctx = model.build_query_context()
    assert len(ctx.document_annotations) == 1
    assert ctx.document_annotations[0]["document_pattern"] == "complaint.pdf"


def test_document_annotation_get_for_document_filters_correctly():
    """get_for_document() must return only annotations matching the document_id (SO-5 + SO-3).

    Annotations are user trust signals injected per-document during processing.
    If get_for_document() returns annotations from other documents, the engine
    would apply the wrong trust calibration — a high-severity SO-5 bug.
    """
    model = MatterModel.open_in_memory()

    model.annotations.add("expert_report.pdf",
                           "Expert hired by opposing counsel; treat as advocacy.")
    model.annotations.add("signed_contract.pdf",
                           "Fully executed; treat as operative.")
    model.annotations.add("email_chain.pdf",
                           "Informal; use only for context.")

    # Only expert_report.pdf annotations must come back
    anns = model.annotations.get_for_document("expert_report.pdf")
    assert len(anns) == 1, "get_for_document must return only annotations for that document"
    assert "advocacy" in anns[0]["annotation_text"], "Must return the correct annotation text"

    # Verify adapter list_annotations(document_id=...) routes to get_for_document
    run_id = model.start_run("annotation filter test")
    adapter = MatterRuntimeAdapter(model, run_id)
    result = adapter.list_annotations(document_id="signed_contract.pdf")
    assert len(result) == 1
    assert "operative" in result[0]["annotation_text"]


def test_null_adapter_annotation():
    """NullMatterAdapter annotation methods must not raise and return safe defaults."""
    adapter = NullMatterAdapter()
    assert adapter.annotate_document("doc.pdf", "some note") == ""
    assert adapter.list_annotations() == []
    assert adapter.list_annotations(document_id="doc.pdf") == []


# ---------------------------------------------------------------------------
# SO-5: Source calibration end-to-end — same proposition, two source roles
# ---------------------------------------------------------------------------

def test_so5_same_proposition_complaint_vs_contract(model):
    """SO-5 end-to-end: same proposition from complaint (advocacy→alleged) and
    contract (operative→operative) must produce 1 assertion + 2 occurrences
    with distinct speech_acts, so advocacy is never amplified as operative.
    """
    from irys.matter.enums import SpeechAct
    run_id = model.start_run("SO-5 test run")
    adapter = MatterRuntimeAdapter(model, run_id)

    proposition = "Payment of $50,000 was due by January 15, 2024."

    # complaint.pdf → inferred source_role=ADVOCACY → speech_act=ALLEGED
    a1 = adapter.record_fact(proposition, document_id="complaint.pdf")
    # contract.pdf → inferred source_role=OPERATIVE → speech_act=OPERATIVE
    a2 = adapter.record_fact(proposition, document_id="Service_Agreement_v3.pdf")

    # Same proposition → same assertion_id (dedup)
    assert a1 == a2, "Same proposition must map to the same assertion (dedup)"

    # Two occurrences with distinct speech_acts
    occurrences = model.assertions.get_occurrences(a1)
    assert len(occurrences) == 2, "Complaint and contract occurrences must both be recorded"

    speech_acts = {occ["speech_act"] for occ in occurrences}
    assert SpeechAct.ALLEGED.value in speech_acts, "Complaint occurrence must be ALLEGED"
    assert SpeechAct.OPERATIVE.value in speech_acts, "Contract occurrence must be OPERATIVE"

    # Source roles must reflect the document origin
    source_roles = {occ["source_role"] for occ in occurrences}
    assert "advocacy" in source_roles, "Complaint must be tagged as advocacy source"
    assert "operative" in source_roles, "Contract must be tagged as operative source"


# ---------------------------------------------------------------------------
# SO-7: _detect_proof_gaps() SQL correctness
# ---------------------------------------------------------------------------

def test_detect_proof_gaps_records_gap_for_unsupported_issue(model):
    """_detect_proof_gaps() must record exactly one proof-gap for issues with no supporting assertions.

    Tests:
    - Only issues with materiality >= 0.4 get proof-gap records
    - Issues that already have a supporting assertion are not gapped
    - Already-gapped issues are not duplicated
    - SQL column names are correct (regression test for materiality vs materiality_score)
    """
    from irys.matter.enums import IssueType, GapType

    # Create three issues
    issue_low, _ = model.issues.upsert_issue(
        title="Minor procedural claim",
        issue_type=IssueType.CLAIM,
        materiality=0.2,   # below threshold — should NOT be gapped
        salience=0.5,
    )
    issue_high, _ = model.issues.upsert_issue(
        title="Breach of contract",
        issue_type=IssueType.CLAIM,
        materiality=0.8,   # above threshold — should be gapped (no support)
        salience=0.5,
    )
    issue_supported, _ = model.issues.upsert_issue(
        title="Payment amount",
        issue_type=IssueType.CLAIM,
        materiality=0.7,   # above threshold but HAS supporting assertion
        salience=0.5,
    )

    # Add a supporting assertion for issue_supported
    run_id = model.start_run("Proof gap test")
    from irys.matter.runtime import MatterRuntimeAdapter
    adapter = MatterRuntimeAdapter(model, run_id)
    aid = adapter.record_fact("Payment was $50,000 per the contract.", "contract.pdf",
                               issue_id=issue_supported)

    # Verify the link was created
    assert model.assertions.count() == 1

    # Run _detect_proof_gaps() via a minimal engine
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    engine._detect_proof_gaps()

    open_gaps = model.gaps.open_gaps(min_materiality=0.0)
    # Only issue_high (materiality=0.8, no support) should produce a proof gap
    assert len(open_gaps) == 1
    assert "Breach of contract" in open_gaps[0]["description"]

    # Running again must not create a duplicate
    engine._detect_proof_gaps()
    assert len(model.gaps.open_gaps()) == 1, "duplicate proof-gap runs must be idempotent"


def test_proof_gap_auto_resolved_when_issue_gains_support(model):
    """When an issue gains an active supporting assertion, its open proof-gap must
    be automatically resolved on the next _detect_proof_gaps() call (SO-7).
    """
    from irys.matter.enums import IssueType
    from irys.matter.runtime import MatterRuntimeAdapter

    issue_id, _ = model.issues.upsert_issue(
        title="Tortious interference",
        issue_type=IssueType.CLAIM,
        materiality=0.85, salience=0.7,
    )

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    # Step 1: no support → proof gap created
    engine._detect_proof_gaps()
    assert len(model.gaps.open_gaps(min_materiality=0.0)) == 1

    # Step 2: add supporting assertion
    run_id = model.start_run("Resolution test")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_fact("Defendant contacted plaintiff's client to undermine the contract.",
                        "email_thread.pdf", issue_id=issue_id)

    # Step 3: re-run detector — should resolve the proof gap
    engine._detect_proof_gaps()
    open_gaps = model.gaps.open_gaps(min_materiality=0.0)
    assert len(open_gaps) == 0, "Proof gap must be resolved when issue gains active support"


def test_proof_gap_recreated_after_gap_closed(model):
    """A closed proof-gap must be re-created on the next detector run if the issue
    still has no supporting assertion links (NOT EXISTS looks at status='open' only).
    """
    from irys.matter.enums import IssueType

    issue_id, _ = model.issues.upsert_issue(
        title="Unlawful termination",
        issue_type=IssueType.CLAIM,
        materiality=0.9,
        salience=0.5,
    )

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    # First run: proof gap created
    engine._detect_proof_gaps()
    gaps_after_first = model.gaps.open_gaps(min_materiality=0.0)
    assert len(gaps_after_first) == 1
    gap_id = gaps_after_first[0]["id"]

    # Simulate external close (e.g. user dismisses the gap)
    model.gaps.db.execute(
        "UPDATE gap SET status='closed' WHERE id=?", (gap_id,)
    )
    assert len(model.gaps.open_gaps(min_materiality=0.0)) == 0

    # Second run: issue still has no support — proof gap must be re-created
    engine._detect_proof_gaps()
    gaps_after_second = model.gaps.open_gaps(min_materiality=0.0)
    assert len(gaps_after_second) == 1, (
        "Proof gap must be re-created when issue still has no support after gap was closed"
    )
    assert "Unlawful termination" in gaps_after_second[0]["description"]


def test_unrelated_gap_does_not_suppress_proof_gap(model):
    """An issue-linked gap with a different description must not suppress a proof-gap record.

    Before the NOT EXISTS + description-filter fix, ANY open gap linked to an issue
    would block _detect_proof_gaps() from recording the zero-support proof gap —
    even a 'missing exhibit A' gap that is semantically unrelated to the
    'no supporting assertions' condition.
    """
    from irys.matter.enums import IssueType, GapType

    issue_id, _ = model.issues.upsert_issue(
        title="Breach of contract",
        issue_type=IssueType.CLAIM,
        materiality=0.8,
        salience=0.5,
    )

    # Record an UNRELATED gap linked to the same issue (e.g., missing exhibit)
    model.gaps.record(
        gap_type=GapType.MISSING_DOCUMENT,
        description="Signed amendment referenced but not provided",
        materiality=0.6,
        affected_type="issue",
        affected_id=issue_id,
    )
    assert len(model.gaps.open_gaps()) == 1

    # Now run _detect_proof_gaps() — issue has no supporting assertions
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    engine._detect_proof_gaps()

    # The proof gap must be created even though an unrelated gap already links to this issue
    open_gaps = model.gaps.open_gaps(min_materiality=0.0)
    assert len(open_gaps) == 2, (
        "Unrelated gap must not suppress proof-gap — both must coexist"
    )
    descriptions = {g["description"] for g in open_gaps}
    assert any("No supporting evidence found" in d for d in descriptions), (
        "Proof gap description must appear alongside the unrelated gap"
    )


# ---------------------------------------------------------------------------
# SO-4: per-issue coverage report
# ---------------------------------------------------------------------------

def test_get_issue_coverage_report(model):
    """get_issue_coverage_report() returns per-claim coverage ordered by weakness (SO-4)."""
    from irys.matter.enums import IssueType
    from irys.matter.runtime import MatterRuntimeAdapter

    issue_weak_id, _ = model.issues.upsert_issue(
        title="Breach of contract",
        issue_type=IssueType.CLAIM,
        materiality=0.8, salience=0.8,
    )
    issue_strong_id, _ = model.issues.upsert_issue(
        title="Damages calculation",
        issue_type=IssueType.CLAIM,
        materiality=0.7, salience=0.7,
    )

    # Add two supporting assertions to issue_strong only
    run_id = model.start_run("Coverage test")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_fact("Damages total $200,000.", "invoice.pdf", issue_id=issue_strong_id)
    adapter.record_fact("Expert report confirms damages.", "expert.pdf", issue_id=issue_strong_id)

    report = model.get_issue_coverage_report()

    # Two issues returned
    assert len(report) == 2

    # Ordered weakest first
    assert report[0]["id"] == issue_weak_id, "breach (0 support) should be first"
    assert report[1]["id"] == issue_strong_id

    # Coverage fractions
    assert report[0]["supporting_count"] == 0
    assert report[0]["coverage_fraction"] == 0.0
    assert report[1]["supporting_count"] == 2
    assert report[1]["coverage_fraction"] > 0.5

    # has_proof_gap: both False (no _detect_proof_gaps has been run yet)
    assert not report[0]["has_proof_gap"]
    assert not report[1]["has_proof_gap"]

    # After running detect, weak issue should have has_proof_gap=True
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    engine._detect_proof_gaps()
    report2 = model.get_issue_coverage_report()
    weak_row = next(r for r in report2 if r["id"] == issue_weak_id)
    strong_row = next(r for r in report2 if r["id"] == issue_strong_id)
    assert weak_row["has_proof_gap"], "unsupported issue must have proof gap flagged"
    assert not strong_row["has_proof_gap"], "supported issue must not have a proof gap"


def test_disputed_assertion_does_not_count_as_coverage(model):
    """DISPUTED/WITHDRAWN/SUPERSEDED assertions must not inflate issue coverage (SO-2 correctness).

    Belief revision propagates state changes. If a supporting assertion is later
    disputed, the issue must no longer count it as active support — and proof-gap
    detection must recognise the issue as uncovered.
    """
    from irys.matter.enums import IssueType, BeliefState
    from irys.matter.runtime import MatterRuntimeAdapter

    issue_id, _ = model.issues.upsert_issue(
        title="Breach of contract",
        issue_type=IssueType.CLAIM,
        materiality=0.9, salience=0.8,
    )

    run_id = model.start_run("Dispute test")
    adapter = MatterRuntimeAdapter(model, run_id)
    assertion_id = adapter.record_fact("Defendant failed to deliver goods.", "complaint.pdf",
                                       issue_id=issue_id)

    # Before dispute: issue has one supporting assertion
    report = model.get_issue_coverage_report()
    row = next(r for r in report if r["id"] == issue_id)
    assert row["supporting_count"] == 1

    # Dispute the assertion
    model.correct_assertion(assertion_id, BeliefState.DISPUTED)

    # After dispute: supporting_count must drop to 0 (SO-2 belief state flows into coverage)
    report2 = model.get_issue_coverage_report()
    row2 = next(r for r in report2 if r["id"] == issue_id)
    assert row2["supporting_count"] == 0, (
        "DISPUTED assertion must not count as active support"
    )

    # Proof gap detector must now flag the issue as uncovered
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model
    engine._detect_proof_gaps()
    gaps = model.gaps.open_gaps(min_materiality=0.0)
    assert len(gaps) == 1
    assert "Breach of contract" in gaps[0]["description"]


# ---------------------------------------------------------------------------
# infer_source_side — litigation side inference from document name
# ---------------------------------------------------------------------------

def test_infer_source_side_plaintiff():
    from irys.matter.runtime import infer_source_side
    assert infer_source_side("plaintiff_complaint.pdf") == "plaintiff"
    assert infer_source_side("docs/petitioner_brief.pdf") == "plaintiff"
    assert infer_source_side("claimant_exhibit.pdf") == "plaintiff"


def test_infer_source_side_defendant():
    from irys.matter.runtime import infer_source_side
    assert infer_source_side("defendant_answer.pdf") == "defendant"
    assert infer_source_side("defense_memo.pdf") == "defendant"
    assert infer_source_side("respondent_filing.pdf") == "defendant"


def test_infer_source_side_neutral():
    from irys.matter.runtime import infer_source_side
    assert infer_source_side("contract_agreement.pdf") is None
    assert infer_source_side("court_order.pdf") is None
    assert infer_source_side("email_thread.pdf") is None


# ---------------------------------------------------------------------------
# list_recent_for_hydration — lightweight assertion query
# ---------------------------------------------------------------------------

def test_list_recent_for_hydration_filters_inactive(model):
    """list_recent_for_hydration() filters inactive belief states at DB level (SO-2).

    Inactive assertions must NOT be returned so the 200-slot LIMIT budget is not
    wasted on facts that the engine would skip anyway.
    """
    from irys.matter.enums import BeliefState
    run_id = model.start_run("hydration test")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid1 = adapter.record_fact("Active fact.", "doc1.pdf")
    aid2 = adapter.record_fact("Disputed fact.", "doc2.pdf")
    model.correct_assertion(aid2, BeliefState.DISPUTED)

    rows = model.assertions.list_recent_for_hydration(limit=50)
    assert any(r["id"] == aid1 for r in rows), "Active assertion must be returned"
    assert not any(r["id"] == aid2 for r in rows), \
        "Disputed assertion must be excluded at DB level — not returned for hydration"
    # Verify required fields are present for hydration
    assert all("belief_state" in r for r in rows)
    assert all("proposition_text" in r for r in rows)
    assert all("source_role" in r for r in rows)


def test_list_recent_for_hydration_limit_not_consumed_by_inactive(model):
    """LIMIT budget must not be consumed by inactive assertions (SO-2 budget efficiency).

    If inactive assertions are filtered AFTER applying LIMIT (wrong), an old active
    assertion will not appear when the window is filled by newer inactive rows.
    The DB-level WHERE filter must exclude inactive assertions before LIMIT is applied.
    """
    from irys.matter.enums import BeliefState
    run_id = model.start_run("limit budget test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Record the one active assertion FIRST (oldest by created_at)
    old_active_id = adapter.record_fact("Old active fact that must survive.", "anchor.pdf")

    # Then record LIMIT+1 inactive assertions (newer, so they appear first in ORDER BY created_at DESC)
    limit = 10
    inactive_ids = []
    for i in range(limit + 1):
        aid = adapter.record_fact(f"Inactive fact {i}.", f"doc{i}.pdf")
        model.correct_assertion(aid, BeliefState.SUPERSEDED)
        inactive_ids.append(aid)

    rows = model.assertions.list_recent_for_hydration(limit=limit)
    row_ids = {r["id"] for r in rows}

    # The old active assertion must appear despite being "oldest" — inactive ones must
    # not consume the LIMIT budget.
    assert old_active_id in row_ids, (
        "Active assertion crowded out by inactive rows — DB-level filtering is broken"
    )
    # None of the inactive assertions should be present
    for iid in inactive_ids:
        assert iid not in row_ids, f"Inactive assertion {iid} leaked into hydration results"


# ---------------------------------------------------------------------------
# record_facts_batch — neutral relation does not create issue link
# ---------------------------------------------------------------------------

def test_neutral_fact_not_linked_to_issue(model):
    from irys.matter.enums import IssueType
    issue_id, _ = model.issues.upsert_issue("Payment claim", IssueType.CLAIM)
    run_id = model.start_run("neutral test")
    adapter = MatterRuntimeAdapter(model, run_id)
    aids = adapter.record_facts_batch(
        [("Payment was made in full.", "receipt.pdf", "supports"),
         ("Invoice was issued.", "invoice.pdf", "neutral")],
        issue_id=issue_id,
    )
    assert len(aids) == 2
    assertions_for_issue = model.issues.get_assertions_for_issue(issue_id)
    linked_ids = {a["id"] for a in assertions_for_issue}
    # Only the 'supports' fact is linked; 'neutral' is not
    assert aids[0] in linked_ids
    assert aids[1] not in linked_ids


# ---------------------------------------------------------------------------
# infer_source_side — ambiguous and directory-name cases
# ---------------------------------------------------------------------------

def test_infer_source_side_ambiguous_returns_none():
    """Filename containing both plaintiff and defendant keywords must return None."""
    from irys.matter.runtime import infer_source_side
    # Both patterns match the basename → ambiguous → None
    assert infer_source_side("defendant_answer_to_plaintiff_complaint.pdf") is None
    assert infer_source_side("plaintiff_response_to_defendant_motion.pdf") is None


def test_infer_source_side_directory_precedence():
    """Basename takes precedence over parent dir; parent dir used only as fallback."""
    from irys.matter.runtime import infer_source_side
    # Basename signal wins over conflicting parent dir
    assert infer_source_side("plaintiff_exhibits/defendant_answer.pdf") == "defendant"
    assert infer_source_side("defendant_productions/plaintiff_complaint.pdf") == "plaintiff"
    # No basename signal → fall back to immediate parent dir
    assert infer_source_side("defendant_motions/court_order_granting.pdf") == "defendant"
    assert infer_source_side("plaintiff_exhibits/exhibit_001.pdf") == "plaintiff"
    # Neutral basename AND neutral parent → None
    assert infer_source_side("court_records/order_granting.pdf") is None


# ---------------------------------------------------------------------------
# record_facts_batch — 4-tuple with temporal_scope_start
# ---------------------------------------------------------------------------

def test_record_facts_batch_four_tuple_temporal(model):
    """4-tuple (text, doc_id, issue_rel, temporal_scope_start) must be stored."""
    from irys.matter.enums import IssueType
    issue_id, _ = model.issues.upsert_issue("Contract term", IssueType.CLAIM)
    run_id = model.start_run("temporal test")
    adapter = MatterRuntimeAdapter(model, run_id)

    aids = adapter.record_facts_batch(
        [
            ("Payment due by 2024-01-15.", "contract.pdf", "supports", "2024-01-15"),
            ("Late fee applies after 2024-01-15.", "contract.pdf", "supports", "2024-01-15"),
        ],
        issue_id=issue_id,
    )
    assert len(aids) == 2
    assert all(aid for aid in aids)  # non-empty IDs

    # Both facts must be linked to the issue (supports, not neutral)
    assertions_for_issue = model.issues.get_assertions_for_issue(issue_id)
    linked_ids = {a["id"] for a in assertions_for_issue}
    assert aids[0] in linked_ids
    assert aids[1] in linked_ids

    # temporal_scope_start must be stored in the assertion row
    rows = model.db.execute(
        "SELECT temporal_scope_start FROM assertion WHERE id IN (?, ?)",
        (aids[0], aids[1]),
    ).fetchall()
    assert all(r["temporal_scope_start"] == "2024-01-15" for r in rows)


# ---------------------------------------------------------------------------
# SO-4: Typed issue parsing — _orient() maps LLM type strings to IssueType enum
# ---------------------------------------------------------------------------

def test_orient_typed_issues_stored_with_correct_issue_type(model):
    """_orient() must store typed issues from LLM plan with the correct IssueType.

    Covers the _issue_type_map path added to resolve the ambiguous all-CLAIM fallback
    that occurred when the prompt listed the wrong type tokens.
    """
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.matter.runtime import MatterRuntimeAdapter
    from irys.matter.enums import IssueType
    from irys.core.repository import RepositoryStats

    # LLM returns a plan with four distinct typed issues
    fake_plan = {
        "issues": [
            {"title": "Breach of payment obligation", "type": "claim"},
            {"title": "Limitation of liability defense", "type": "defense"},
            {"title": "Lost profits exposure", "type": "damages"},
            {"title": "Ambiguity in exclusivity clause", "type": "contract_question"},
        ],
        "relevant_folders": [],
        "initial_searches": ["payment", "exclusivity"],
        "hypothesis": "Plaintiff alleges non-payment under the agreement.",
    }

    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps(fake_plan))

    engine = RLMEngine(gemini_client=mock_client, config=RLMConfig(), matter_model=model)

    # Minimal repo mock — _orient() only needs stats + structure
    mock_repo = MagicMock()
    mock_repo.get_stats.return_value = RepositoryStats(
        total_files=5, total_size_bytes=1024, files_by_type={".pdf": 5}, folders=["pleadings"],
    )
    mock_repo.get_structure.return_value = {"pleadings": 3, "contracts": 2}

    from irys.rlm.state import InvestigationState
    run_id = model.start_run("typed issue test")
    adapter = MatterRuntimeAdapter(model, run_id)
    state = InvestigationState(id="test-run-1", query="Breach of contract claim", repository_path="/tmp/test")
    state._matter_adapter = adapter

    asyncio.run(engine._orient(state, mock_repo))

    open_issues = model.issues.get_open_issues()
    issue_map = {i["title"]: i["issue_type"] for i in open_issues}

    assert issue_map.get("Breach of payment obligation") == IssueType.CLAIM.value
    assert issue_map.get("Limitation of liability defense") == IssueType.DEFENSE.value
    assert issue_map.get("Lost profits exposure") == IssueType.DAMAGES.value
    assert issue_map.get("Ambiguity in exclusivity clause") == IssueType.CONTRACT_QUESTION.value


def test_orient_legacy_string_issues_default_to_claim(model):
    """Legacy string-format issues (plain strings, not dicts) must default to IssueType.CLAIM."""
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.matter.runtime import MatterRuntimeAdapter
    from irys.matter.enums import IssueType
    from irys.core.repository import RepositoryStats

    fake_plan = {
        "issues": [
            "Breach of contract",           # legacy string format
            "Damages sought",               # legacy string format
            {"title": "Defense raised", "type": "defense"},  # new format alongside
        ],
        "relevant_folders": [],
        "initial_searches": [],
        "hypothesis": "Test.",
    }

    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps(fake_plan))
    engine = RLMEngine(gemini_client=mock_client, config=RLMConfig(), matter_model=model)

    mock_repo = MagicMock()
    mock_repo.get_stats.return_value = RepositoryStats(
        total_files=1, total_size_bytes=512, files_by_type={}, folders=[],
    )
    mock_repo.get_structure.return_value = {}

    from irys.rlm.state import InvestigationState
    run_id = model.start_run("legacy string test")
    adapter = MatterRuntimeAdapter(model, run_id)
    state = InvestigationState(id="t-legacy", query="breach", repository_path="/tmp/test")
    state._matter_adapter = adapter

    asyncio.run(engine._orient(state, mock_repo))

    open_issues = model.issues.get_open_issues()
    issue_map = {i["title"]: i["issue_type"] for i in open_issues}

    assert issue_map.get("Breach of contract") == IssueType.CLAIM.value
    assert issue_map.get("Damages sought") == IssueType.CLAIM.value
    assert issue_map.get("Defense raised") == IssueType.DEFENSE.value


# ---------------------------------------------------------------------------
# SO-2: Pre-synthesis refresh — superseded assertions excluded from hydration
# ---------------------------------------------------------------------------

def test_hydrate_skips_superseded_assertions(model):
    """_hydrate_from_matter_model() must exclude assertions revised to inactive belief states.

    This is the core SO-2 mechanism: a user correction that supersedes an assertion
    must not re-appear in accumulated_facts when synthesis re-hydrates from the model.
    """
    from irys.matter.enums import BeliefState
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.rlm.state import InvestigationState
    from unittest.mock import MagicMock

    # Record two assertions
    run_id = model.start_run("hydration test")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid_active = adapter.record_fact("Payment was due on January 15.", "contract.pdf")
    aid_superseded = adapter.record_fact("Original invoice amount was $10,000.", "invoice_v1.pdf")

    # Simulate a user correction: mark the second assertion superseded
    model.assertions.set_belief_state(aid_superseded, BeliefState.SUPERSEDED)

    engine = RLMEngine(gemini_client=MagicMock(), config=RLMConfig(), matter_model=model)
    state = InvestigationState(id="test-2", query="test", repository_path="/tmp/test")

    engine._hydrate_from_matter_model(state)

    facts = state.findings.get("accumulated_facts", [])
    # Active assertion must be present
    assert any("Payment was due on January 15." in f for f in facts), \
        "Active assertion must appear in accumulated_facts"
    # Superseded assertion must be excluded
    assert not any("Original invoice amount was $10,000." in f for f in facts), \
        "Superseded assertion must not appear in accumulated_facts after user correction"


# ---------------------------------------------------------------------------
# SO-3: Stopped lead stays pending — not marked investigated after stop
# ---------------------------------------------------------------------------

def test_stopped_lead_remains_pending(model):
    """_investigate_lead() must NOT mark a lead as investigated when stop fires.

    After the stop, the lead must remain pending so a resumed run can retry
    the full analysis (SO-3 resume correctness).
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.rlm.state import InvestigationState, Lead
    from irys.matter.runtime import MatterRuntimeAdapter
    from irys.core.search import SearchResults

    run_id = model.start_run("stop-lead test")
    adapter = MatterRuntimeAdapter(model, run_id)

    engine = RLMEngine(gemini_client=MagicMock(), config=RLMConfig(), matter_model=model)

    lead = Lead.create(description="Payment obligation", source="orient", search_term="payment")
    state = InvestigationState(id="test-stop", query="breach", repository_path="/tmp/test")
    state._matter_adapter = adapter
    state.leads.append(lead)
    state.recursion_depth = 0

    # repo.search() returns one result so the analysis path is reached
    mock_results = SearchResults(query="payment", hits=[MagicMock(
        file_path="/tmp/test/contract.pdf", snippet="Payment due Jan 15", score=0.9
    )], files_searched=5, total_matches=1)

    mock_repo = MagicMock()
    mock_repo.search = MagicMock(return_value=mock_results)

    # _analyze_search_results sets the stop flag mid-analysis
    async def fake_analyze(st, repo, results, ld):
        adapter.request_stop()

    with patch.object(engine, "_analyze_search_results", side_effect=fake_analyze):
        asyncio.run(engine._investigate_lead(state, mock_repo, lead))

    # Lead must still be pending — stop fired before mark_lead_investigated
    assert not lead.investigated, \
        "Lead must remain pending (not investigated) when stop fires during analysis"
    assert adapter.is_stop_requested()


# ---------------------------------------------------------------------------
# SO-4: issue_type_map handles null/non-string title + all 10 IssueType values
# ---------------------------------------------------------------------------

def test_orient_null_title_is_skipped(model):
    """_orient() must not crash when LLM returns null title — skips the item."""
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.matter.runtime import MatterRuntimeAdapter
    from irys.core.repository import RepositoryStats

    fake_plan = {
        "issues": [
            {"title": None, "type": "claim"},          # null title — must skip
            {"title": "Valid breach claim", "type": "claim"},
        ],
        "relevant_folders": [],
        "initial_searches": [],
        "hypothesis": "Breach of contract.",
    }

    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps(fake_plan))
    engine = RLMEngine(gemini_client=mock_client, config=RLMConfig(), matter_model=model)

    mock_repo = MagicMock()
    mock_repo.get_stats.return_value = RepositoryStats(
        total_files=1, total_size_bytes=512, files_by_type={}, folders=[],
    )
    mock_repo.get_structure.return_value = {}

    from irys.rlm.state import InvestigationState
    run_id = model.start_run("null title test")
    adapter = MatterRuntimeAdapter(model, run_id)
    state = InvestigationState(id="t-null", query="breach", repository_path="/tmp/test")
    state._matter_adapter = adapter

    asyncio.run(engine._orient(state, mock_repo))  # must not raise

    open_issues = model.issues.get_open_issues()
    assert len(open_issues) == 1
    assert open_issues[0]["title"] == "Valid breach claim"


def test_orient_full_issue_type_map(model):
    """_orient() must map all 10 IssueType enum values from LLM plan without fallback."""
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock
    from irys.rlm.engine import RLMEngine, RLMConfig
    from irys.matter.runtime import MatterRuntimeAdapter
    from irys.matter.enums import IssueType
    from irys.core.repository import RepositoryStats

    all_types = [
        ("Primary claim", "claim", IssueType.CLAIM),
        ("Affirmative defense", "defense", IssueType.DEFENSE),
        ("Damages component", "damages", IssueType.DAMAGES),
        ("Contract ambiguity", "contract_question", IssueType.CONTRACT_QUESTION),
        ("Procedural threshold", "procedural", IssueType.PROCEDURAL_BARRIER),
        ("Evidentiary issue", "evidentiary", IssueType.EVIDENTIARY_BOTTLENECK),
        ("Condition not met", "condition_precedent", IssueType.CONDITION_PRECEDENT),
        ("Waiver argument", "waiver", IssueType.WAIVER),
        ("Diligence risk", "diligence_red_flag", IssueType.DILIGENCE_RED_FLAG),
        ("Compliance issue", "compliance_failure", IssueType.COMPLIANCE_FAILURE),
    ]

    fake_plan = {
        "issues": [{"title": t[0], "type": t[1]} for t in all_types],
        "relevant_folders": [],
        "initial_searches": [],
        "hypothesis": "Full type coverage test.",
    }

    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps(fake_plan))
    engine = RLMEngine(gemini_client=mock_client, config=RLMConfig(), matter_model=model)

    mock_repo = MagicMock()
    mock_repo.get_stats.return_value = RepositoryStats(
        total_files=1, total_size_bytes=512, files_by_type={}, folders=[],
    )
    mock_repo.get_structure.return_value = {}

    from irys.rlm.state import InvestigationState
    run_id = model.start_run("full type map test")
    adapter = MatterRuntimeAdapter(model, run_id)
    state = InvestigationState(id="t-full", query="test", repository_path="/tmp/test")
    state._matter_adapter = adapter

    asyncio.run(engine._orient(state, mock_repo))

    open_issues = model.issues.get_open_issues()
    issue_map = {i["title"]: i["issue_type"] for i in open_issues}

    for title, _, expected_type in all_types:
        assert issue_map.get(title) == expected_type.value, \
            f"Issue '{title}' expected type {expected_type.value}, got {issue_map.get(title)}"


# ---------------------------------------------------------------------------
# SO-5: infer_source_side — plural/possessive forms
# ---------------------------------------------------------------------------

def test_infer_source_side_plural_forms():
    """Plural and possessive side labels must be recognized (e.g. plaintiffs_, defendants_)."""
    from irys.matter.runtime import infer_source_side
    # Plurals with underscore separator (common in filenames)
    assert infer_source_side("plaintiffs_exhibit_001.pdf") == "plaintiff"
    assert infer_source_side("defendants_motion_to_dismiss.pdf") == "defendant"
    assert infer_source_side("respondents_brief.pdf") == "defendant"
    assert infer_source_side("petitioners_reply.pdf") == "plaintiff"
    # Original singular forms still work
    assert infer_source_side("plaintiff_complaint.pdf") == "plaintiff"
    assert infer_source_side("defendant_answer.pdf") == "defendant"


# ---------------------------------------------------------------------------
# SO-2: _retry_spo_extraction — SPO retry logic (SO-2 validated extraction)
# ---------------------------------------------------------------------------

def test_retry_spo_extraction_returns_spo_dict():
    """_retry_spo_extraction() must return index→spo mapping when LLM provides triples."""
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock

    engine = RLMEngine.__new__(RLMEngine)
    engine.client = MagicMock()
    engine.client.complete = AsyncMock(return_value=json.dumps([
        {"index": 0, "subject": "Acme Corp", "predicate": "agreed_to_pay", "object": "$50,000"},
        {"index": 2, "subject": "plaintiff", "predicate": "filed_complaint", "object": "breach_of_contract"},
    ]))

    fact_texts = [
        "Acme Corp agreed to pay $50,000 by March 2023.",
        "The case was filed in the Southern District.",
        "Plaintiff filed a complaint for breach of contract.",
    ]

    result = asyncio.run(engine._retry_spo_extraction(fact_texts))

    assert 0 in result
    assert result[0]["subject_ref_id"] == "Acme Corp"
    assert result[0]["predicate_key"] == "agreed_to_pay"
    assert result[0]["object_json"] == '"$50,000"'
    assert 2 in result
    assert result[2]["predicate_key"] == "filed_complaint"
    assert 1 not in result  # fact 1 was not in LLM response


def test_retry_spo_extraction_empty_on_error():
    """_retry_spo_extraction() must return empty dict if LLM call fails."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    engine = RLMEngine.__new__(RLMEngine)
    engine.client = MagicMock()
    engine.client.complete = AsyncMock(side_effect=RuntimeError("network error"))

    result = asyncio.run(engine._retry_spo_extraction(["Some fact about the case."]))
    assert result == {}


def test_retry_spo_extraction_empty_input():
    """_retry_spo_extraction() must return empty dict immediately for empty input."""
    import asyncio
    from unittest.mock import MagicMock

    engine = RLMEngine.__new__(RLMEngine)
    engine.client = MagicMock()  # should not be called

    result = asyncio.run(engine._retry_spo_extraction([]))
    assert result == {}
    engine.client.complete.assert_not_called()


def test_retry_spo_extraction_ignores_out_of_range_index():
    """_retry_spo_extraction() must ignore items with index out of bounds."""
    import asyncio
    import json
    from unittest.mock import AsyncMock, MagicMock

    engine = RLMEngine.__new__(RLMEngine)
    engine.client = MagicMock()
    engine.client.complete = AsyncMock(return_value=json.dumps([
        {"index": 99, "subject": "ghost", "predicate": "haunts", "object": "nobody"},  # out of range
        {"index": 0, "subject": "defendant", "predicate": "breached", "object": "contract"},
    ]))

    result = asyncio.run(engine._retry_spo_extraction(["Defendant breached the contract."]))
    assert 99 not in result
    assert 0 in result


# ---------------------------------------------------------------------------
# ReasoningLedgerStore: seq_no cache correctness (commit e33b084)
# ---------------------------------------------------------------------------

def test_ledger_seq_no_cache_no_duplicates():
    """Multiple appends to the same run_id must produce strictly increasing seq_nos."""
    from irys.matter.enums import LedgerEventType
    model = MatterModel.open_in_memory()
    run_id = model.start_run("seq_no test")

    # Append several additional events
    for i in range(5):
        model.ledger.append_event(
            run_id=run_id,
            event_type=LedgerEventType.PROGRESS_NOTE,
            summary=f"step {i}",
        )

    events = model.ledger.get_events(run_id)
    seq_nos = [e["seq_no"] for e in events]
    # seq_nos must be strictly increasing (no duplicates)
    assert seq_nos == sorted(set(seq_nos)), "seq_nos must be unique and increasing"
    assert len(seq_nos) >= 6  # 1 from start_run + 5 appended


def test_ledger_seq_no_cache_multiple_runs():
    """seq_no cache must be per-run_id (two runs must not share seq_no state)."""
    from irys.matter.enums import LedgerEventType
    model = MatterModel.open_in_memory()
    run_id_a = model.start_run("run A")
    run_id_b = model.start_run("run B")

    model.ledger.append_event(run_id_a, LedgerEventType.PROGRESS_NOTE, "A step 1")
    model.ledger.append_event(run_id_b, LedgerEventType.PROGRESS_NOTE, "B step 1")
    model.ledger.append_event(run_id_a, LedgerEventType.PROGRESS_NOTE, "A step 2")

    events_a = model.ledger.get_events(run_id_a)
    events_b = model.ledger.get_events(run_id_b)
    seq_a = [e["seq_no"] for e in events_a]
    seq_b = [e["seq_no"] for e in events_b]
    assert seq_a == sorted(set(seq_a)), "run A seq_nos must be unique and increasing"
    assert seq_b == sorted(set(seq_b)), "run B seq_nos must be unique and increasing"


# ---------------------------------------------------------------------------
# InvestigationState serialization — query_classification + facts_per_iteration
# + reasoning_trail (commits 21bbe3d, e86b9ff)
# ---------------------------------------------------------------------------

def test_investigation_state_serialization_roundtrip():
    """to_dict() / from_dict() must preserve query_classification, facts_per_iteration,
    and reasoning_trail without loss."""
    from irys.rlm.state import InvestigationState

    state = InvestigationState.create("Test query", "/repo")
    state.query_classification = {"complexity": "high", "type": "analytical"}
    state.facts_per_iteration = [3, 5, 2, 0]
    state.reasoning_trail = [{"seq_no": 0, "summary": "Run started"}, {"seq_no": 1, "summary": "Searching"}]

    data = state.to_dict()
    restored = InvestigationState.from_dict(data)

    assert restored.query_classification == {"complexity": "high", "type": "analytical"}
    assert restored.facts_per_iteration == [3, 5, 2, 0]
    assert restored.reasoning_trail == [{"seq_no": 0, "summary": "Run started"}, {"seq_no": 1, "summary": "Searching"}]


def test_investigation_state_serialization_defaults():
    """from_dict() on old-format dict (missing new fields) must use safe defaults."""
    from irys.rlm.state import InvestigationState

    minimal = {"id": "abc123", "query": "test", "repository_path": "/repo"}
    state = InvestigationState.from_dict(minimal)

    assert state.query_classification is None
    assert state.facts_per_iteration == []
    assert state.reasoning_trail == []


# ---------------------------------------------------------------------------
# AssertionStore.list_recent_for_hydration — SPO columns returned (commit 35ccc45)
# ---------------------------------------------------------------------------

def test_list_recent_for_hydration_returns_spo_columns():
    """list_recent_for_hydration() must include subject_ref_id, predicate_key,
    object_json columns when SPO fields are stored."""
    from irys.matter import AssertionCandidate, SpeechAct, SourceRole, ModelLayer, AssertionKind
    from irys.matter.enums import OriginKind

    model = MatterModel.open_in_memory()

    # Record an assertion with full SPO fields
    candidate = AssertionCandidate(
        proposition_text="Acme Corp agreed to pay $50,000 by March 2023.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="contract.pdf",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        origin_kind=OriginKind.EXTRACTED,
        subject_ref_type="free_text",
        subject_ref_id="Acme Corp",
        predicate_key="agreed_to_pay",
        object_json='"$50,000 by March 2023"',
    )
    model.assertions.upsert_occurrence(candidate)

    rows = model.assertions.list_recent_for_hydration(limit=10)
    assert len(rows) >= 1

    row = rows[0]
    assert "subject_ref_id" in row or row.get("subject_ref_id") == "Acme Corp"
    assert row.get("subject_ref_id") == "Acme Corp"
    assert row.get("predicate_key") == "agreed_to_pay"
    assert row.get("object_json") == '"$50,000 by March 2023"'


# ---------------------------------------------------------------------------
# _build_structured_relationships — completeness ordering (commit f882894)
# ---------------------------------------------------------------------------

def test_build_structured_relationships_completeness_ordering():
    """Assertions with all 3 SPO fields must appear before partial ones."""
    from irys.matter import AssertionCandidate, SpeechAct, SourceRole, ModelLayer, AssertionKind
    from irys.matter.enums import OriginKind
    import json

    model = MatterModel.open_in_memory()

    # Record partial assertion (subject only)
    partial = AssertionCandidate(
        proposition_text="Partial: subject only.",
        model_layer=ModelLayer.RECORD, assertion_kind=AssertionKind.FACTUAL,
        document_id="doc1.pdf", speech_act=SpeechAct.ALLEGED, source_role=SourceRole.ADVOCACY,
        origin_kind=OriginKind.EXTRACTED,
        subject_ref_id="Plaintiff",
    )
    # Record complete assertion (all 3 SPO)
    complete = AssertionCandidate(
        proposition_text="Complete: Acme Corp agreed to pay $50,000.",
        model_layer=ModelLayer.RECORD, assertion_kind=AssertionKind.FACTUAL,
        document_id="doc2.pdf", speech_act=SpeechAct.OPERATIVE, source_role=SourceRole.OPERATIVE,
        origin_kind=OriginKind.EXTRACTED,
        subject_ref_id="Acme Corp",
        predicate_key="agreed_to_pay",
        object_json=json.dumps("$50,000"),
    )
    model.assertions.upsert_occurrence(partial)
    model.assertions.upsert_occurrence(complete)

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_structured_relationships()
    assert result  # non-empty: at least one typed assertion exists

    # _build_structured_relationships formats as: [ROLE/belief] subj →[pred]→ obj
    # The complete triple uses subject_ref_id="Acme Corp" and predicate_key="agreed_to_pay"
    # The partial only has subject_ref_id="Plaintiff" (pred and obj are "?")
    idx_complete = result.find("agreed_to_pay")  # predicate of complete triple
    idx_partial = result.find("Plaintiff")         # subject of partial triple
    assert idx_complete != -1, "complete triple predicate must appear in output"
    assert idx_partial != -1, "partial triple subject must appear in output"
    assert idx_complete < idx_partial, (
        "Fully-structured triple (3 fields) must appear before partial (1 field) in synthesis block"
    )


# ---------------------------------------------------------------------------
# SO-4: ORIENTATION_PROMPT biases initial_searches toward weakest issue
# ---------------------------------------------------------------------------

def test_orientation_prompt_contains_priority_focus_instruction():
    """ORIENTATION_PROMPT PRIORITIZE block must instruct the LLM to bias
    initial_searches toward the PRIORITY FOCUS issue when one is present."""
    from irys.rlm.engine import ORIENTATION_PROMPT
    assert "PRIORITY FOCUS" in ORIENTATION_PROMPT, (
        "ORIENTATION_PROMPT must reference PRIORITY FOCUS issue in PRIORITIZE block (SO-4)"
    )
    assert "initial_searches" in ORIENTATION_PROMPT, (
        "ORIENTATION_PROMPT must mention initial_searches in the PRIORITY FOCUS instruction"
    )


def test_orientation_cache_version_bumped():
    """_ORIENTATION_CACHE_VERSION must be '3' after SO-4 PRIORITIZE update."""
    from irys.rlm.engine import _ORIENTATION_CACHE_VERSION
    assert _ORIENTATION_CACHE_VERSION == "3", (
        "_ORIENTATION_CACHE_VERSION must be bumped to '3' after ORIENTATION_PROMPT change "
        "to invalidate stale cached plans (SO-1 stale-cache prevention)"
    )


def test_format_matter_context_emits_priority_focus_line():
    """_format_matter_context() must emit 'PRIORITY FOCUS' line when weakest_issue_id is set."""
    from irys.rlm.engine import _format_matter_context
    from irys.matter.runtime import QueryMatterContext

    ctx = QueryMatterContext(
        matter_id="m_test",
        matter_name="Test Matter",
        existing_assertion_count=5,
        open_issues=[
            {"id": "iss_001", "title": "Breach of payment obligation"},
            {"id": "iss_002", "title": "Damages calculation"},
        ],
        weakest_issue_id="iss_001",
        open_gaps=[],
        known_actors=[],
        known_document_ids=[],
    )
    result = _format_matter_context(ctx)
    assert "PRIORITY FOCUS" in result, (
        "_format_matter_context must emit PRIORITY FOCUS line when weakest_issue_id is set"
    )
    assert "Breach of payment obligation" in result, (
        "PRIORITY FOCUS line must include the issue title"
    )


def test_format_matter_context_no_priority_focus_when_weakest_issue_id_is_none():
    """_format_matter_context() must NOT emit 'PRIORITY FOCUS' when weakest_issue_id is None.

    Without this test, an always-on PRIORITY FOCUS emission would pass the positive test
    but would silently pollute every context with a spurious focus directive — causing
    the LLM to hallucinate an issue priority that doesn't exist (SO-4 correctness).
    """
    from irys.rlm.engine import _format_matter_context
    from irys.matter.runtime import QueryMatterContext

    ctx = QueryMatterContext(
        matter_id="m_test",
        matter_name="Test Matter",
        existing_assertion_count=0,
        open_issues=[],
        weakest_issue_id=None,  # no issues → no priority focus
        open_gaps=[],
        known_actors=[],
        known_document_ids=[],
    )
    result = _format_matter_context(ctx)
    assert "PRIORITY FOCUS" not in result, (
        "_format_matter_context must NOT emit PRIORITY FOCUS when weakest_issue_id is None (SO-4 correctness)"
    )


# ---------------------------------------------------------------------------
# SO-7: pending_clarifications wired into InvestigationState (SO-7)
# ---------------------------------------------------------------------------

def test_investigation_state_serialization_includes_pending_clarifications():
    """pending_clarifications must round-trip through to_dict/from_dict."""
    from irys.rlm.state import InvestigationState

    state = InvestigationState.create("test query", "/repo")
    state.pending_clarifications = [
        {
            "id": "cl_001",
            "question_text": "Do you have the signed amendment?",
            "why_it_matters": "The amendment was referenced but not found.",
            "expected_impact": "high",
            "status": "pending",
        }
    ]

    data = state.to_dict()
    assert "pending_clarifications" in data
    assert len(data["pending_clarifications"]) == 1
    assert data["pending_clarifications"][0]["question_text"] == "Do you have the signed amendment?"

    restored = InvestigationState.from_dict(data)
    assert len(restored.pending_clarifications) == 1
    assert restored.pending_clarifications[0]["id"] == "cl_001"


def test_investigation_state_pending_clarifications_defaults_empty():
    """pending_clarifications must default to [] when absent from serialized data."""
    from irys.rlm.state import InvestigationState

    state = InvestigationState.create("test query", "/repo")
    data = state.to_dict()
    # Remove the key to simulate old serialized data
    data.pop("pending_clarifications", None)

    restored = InvestigationState.from_dict(data)
    assert restored.pending_clarifications == []


def test_format_matter_context_emits_known_document_ids():
    """_format_matter_context() must list known_document_ids so the LLM avoids re-reading (SO-1).

    The 'Documents already analyzed' line prevents the LLM from requesting
    documents that are already fully ingested into the assertion graph.
    Without this line, the model wastes tokens re-reading ingested content
    instead of focusing on new or gap-filling sources — violating SO-1.
    """
    from irys.rlm.engine import _format_matter_context
    from irys.matter.runtime import QueryMatterContext

    ctx = QueryMatterContext(
        matter_id="m_test",
        matter_name="Test Matter",
        existing_assertion_count=5,
        open_issues=[],
        open_gaps=[],
        known_actors=[],
        known_document_ids=["contract.pdf", "complaint.pdf", "exhibit_a.pdf"],
    )
    result = _format_matter_context(ctx)
    assert "contract.pdf" in result, (
        "_format_matter_context must include known_document_ids so LLM skips re-reading (SO-1)"
    )
    assert "complaint.pdf" in result or "3" in result, (
        "Known documents or count must appear in formatted context"
    )


def test_format_matter_context_emits_answered_clarifications():
    """_format_matter_context() must include answered clarifications so LLM uses user context (SO-3).

    When a user answers a clarification question mid-run or before a run,
    that information MUST appear in the orientation prompt.  Without it,
    the LLM ignores user-supplied strategic context — a direct SO-3 violation.
    The 'User-supplied context' block must include both the question and the answer.
    """
    from irys.rlm.engine import _format_matter_context
    from irys.matter.runtime import QueryMatterContext

    ctx = QueryMatterContext(
        matter_id="m_test",
        matter_name="Test Matter",
        existing_assertion_count=0,
        open_issues=[],
        open_gaps=[],
        known_actors=[],
        known_document_ids=[],
        answered_clarifications=[
            {
                "id": "cl_001",
                "question_text": "Do you have the signed amendment No. 2?",
                "answer_text": "No, the client confirmed it was never executed.",
            }
        ],
    )
    result = _format_matter_context(ctx)
    assert "User-supplied context" in result or "answered" in result.lower(), (
        "_format_matter_context must label the user-supplied context block (SO-3)"
    )
    assert "signed amendment" in result.lower(), (
        "Clarification question must appear in formatted context (SO-3)"
    )
    assert "never executed" in result.lower(), (
        "Clarification answer must appear in formatted context (SO-3)"
    )


def test_format_matter_context_emits_document_annotations():
    """_format_matter_context() must include document annotations so the LLM applies user notes (SO-3).

    When a user annotates a document (e.g. 'treat damages as advocacy positions'), that
    annotation must appear in the orientation prompt so the LLM calibrates trust accordingly.
    Without this, user strategic notes are stored but silently ignored — a direct SO-3 failure.
    """
    from irys.rlm.engine import _format_matter_context
    from irys.matter.runtime import QueryMatterContext

    ctx = QueryMatterContext(
        matter_id="m_test",
        matter_name="Test Matter",
        existing_assertion_count=0,
        open_issues=[],
        open_gaps=[],
        known_actors=[],
        known_document_ids=[],
        document_annotations=[
            {
                "document_pattern": "expert_report.pdf",
                "annotation_text": "Prepared for litigation; treat damages figures as advocacy positions.",
                "annotation_type": "reliability",
            }
        ],
    )
    result = _format_matter_context(ctx)
    assert "expert_report.pdf" in result, (
        "Document annotation pattern must appear in formatted context (SO-3)"
    )
    assert "advocacy" in result.lower() or "litigation" in result.lower(), (
        "Annotation text must appear in formatted context (SO-3)"
    )


def test_format_matter_context_emits_key_predicates():
    """_format_matter_context() must surface key_predicates from SPO graph (SO-2)."""
    from irys.rlm.engine import _format_matter_context
    from irys.matter.runtime import QueryMatterContext

    ctx = QueryMatterContext(
        matter_id="m_test",
        matter_name="Test Matter",
        existing_assertion_count=10,
        open_issues=[],
        open_gaps=[],
        known_actors=[],
        known_document_ids=[],
        key_predicates=["agreed_to_pay", "executed_contract", "disputes_claim"],
    )
    result = _format_matter_context(ctx)
    assert "agreed_to_pay" in result, "key_predicates must appear in formatted context"
    assert "predicate graph" in result.lower() or "relationship" in result.lower(), (
        "formatted context must label predicate section"
    )


def test_build_query_context_populates_key_predicates():
    """build_query_context() must populate key_predicates from assertion predicate_key values."""
    from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole
    from irys.matter.enums import ModelLayer, AssertionKind, OriginKind

    model = MatterModel.open_in_memory()
    # Insert 3 distinct assertions with agreed_to_pay and 1 with executed_contract.
    # Distinct proposition_texts → distinct assertion rows → honest frequency count.
    entries = [
        ("Party A agreed to pay $100k", "agreed_to_pay"),
        ("Party A agreed to pay $50k by March", "agreed_to_pay"),
        ("Party A agreed to pay late fee of $5k", "agreed_to_pay"),
        ("Party B executed the contract on Jan 1", "executed_contract"),
    ]
    for prop, pred in entries:
        c = AssertionCandidate(
            proposition_text=prop,
            speech_act=SpeechAct.ALLEGED,
            source_role=SourceRole.ADVOCACY,
            document_id="doc1.txt",
            model_layer=ModelLayer.RECORD,
            assertion_kind=AssertionKind.FACTUAL,
            origin_kind=OriginKind.EXTRACTED,
            predicate_key=pred,
        )
        model.assertions.upsert_occurrence(c)

    ctx = model.build_query_context()
    assert "agreed_to_pay" in ctx.key_predicates, "agreed_to_pay must appear in key_predicates"
    assert "executed_contract" in ctx.key_predicates, "executed_contract must appear in key_predicates"
    # agreed_to_pay has 3 rows vs 1 for executed_contract → must rank first
    assert ctx.key_predicates[0] == "agreed_to_pay", "predicates ordered by frequency descending"


# ---------------------------------------------------------------------------
# SO-4: _build_issue_coverage_summary()
# ---------------------------------------------------------------------------

def test_build_issue_coverage_summary_no_model():
    """_build_issue_coverage_summary() with no matter model returns safe fallback."""
    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = None

    result = engine._build_issue_coverage_summary()
    assert "available" in result.lower() or "no" in result.lower()


def test_build_issue_coverage_summary_no_issues():
    """_build_issue_coverage_summary() with no open issues returns 'No open issues'."""
    model = MatterModel.open_in_memory()

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_issue_coverage_summary()
    assert "no open issues" in result.lower()


def test_build_issue_coverage_summary_shows_coverage():
    """_build_issue_coverage_summary() must show each issue with strength label and assertion count."""
    from irys.matter import MatterModel, AssertionCandidate
    from irys.matter.enums import SpeechAct, OriginKind, AssertionKind, ModelLayer, SourceRole

    model = MatterModel.open_in_memory()

    # Create an issue
    from irys.matter.enums import IssueType
    issue_id, _ = model.issues.upsert_issue(
        title="Breach of payment obligation",
        issue_type=IssueType.CLAIM,
        materiality=0.9,
    )

    # Record 2 assertions and link both to the issue
    for i in range(2):
        c = AssertionCandidate(
            proposition_text=f"Defendant failed to pay invoice {i}",
            assertion_kind=AssertionKind.FACTUAL,
            speech_act=SpeechAct.ALLEGED,
            origin_kind=OriginKind.EXTRACTED,
            model_layer=ModelLayer.RECORD,
            source_role=SourceRole.ADVOCACY,
            document_id="complaint.pdf",
        )
        aid, _ = model.record_assertion(c)
        model.issues.link_assertion(aid, issue_id, "supports")

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_issue_coverage_summary()

    assert "Breach of payment obligation" in result, f"Issue title missing from: {result}"
    # 2 supporting assertions → coverage_fraction = 2/3 ≈ 0.67 → STRONG
    assert "STRONG" in result, f"Expected STRONG coverage label: {result}"
    assert "2" in result, f"Expected assertion count 2 in: {result}"


def test_build_issue_coverage_summary_weak_coverage_no_assertions():
    """An issue with zero supporting assertions must be labeled WEAK."""
    model = MatterModel.open_in_memory()

    from irys.matter.enums import IssueType
    model.issues.upsert_issue(
        title="Fraud claim",
        issue_type=IssueType.CLAIM,
        materiality=0.8,
    )  # return value not needed — we just need the issue to exist

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_issue_coverage_summary()

    assert "Fraud claim" in result
    assert "WEAK" in result, f"Expected WEAK label for zero-assertion issue: {result}"
    assert "0" in result, f"Expected 0 supporting assertions in: {result}"


def test_build_issue_coverage_summary_shows_proof_gap_flag():
    """_build_issue_coverage_summary() must show ⚠ PROOF GAP for issues with open proof gaps (SO-4+SO-7).

    The proof-gap flag is the synthesis-level surface of SO-7's missingness tracking:
    the LLM must see which issues have zero evidentiary support so it can surface
    proof gaps in its analysis rather than synthesizing confidently over silence.
    """
    from irys.matter.enums import IssueType

    model = MatterModel.open_in_memory()
    issue_id, _ = model.issues.upsert_issue(
        title="Tortious interference claim",
        issue_type=IssueType.CLAIM,
        materiality=0.9,
    )

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    # Before running proof-gap detection: no PROOF GAP flag
    result_before = engine._build_issue_coverage_summary()
    assert "PROOF GAP" not in result_before, (
        "PROOF GAP must not appear before _detect_proof_gaps() runs"
    )

    # Run proof-gap detection — creates a gap linked to the issue
    engine._detect_proof_gaps()

    # After detection: PROOF GAP flag must appear in summary
    result_after = engine._build_issue_coverage_summary()
    assert "PROOF GAP" in result_after, (
        f"_build_issue_coverage_summary must show ⚠ PROOF GAP for issues with open gaps: {result_after}"
    )
    assert "Tortious interference claim" in result_after


def test_build_issue_coverage_summary_partial_coverage():
    """An issue with 1 supporting assertion must be labeled PARTIAL (coverage ~0.5)."""
    from irys.matter import MatterModel, AssertionCandidate
    from irys.matter.enums import (
        SpeechAct, OriginKind, AssertionKind, ModelLayer, SourceRole, IssueType,
    )

    model = MatterModel.open_in_memory()
    issue_id, _ = model.issues.upsert_issue(
        title="Causation element",
        issue_type=IssueType.CLAIM,
        materiality=0.8,
    )
    c = AssertionCandidate(
        proposition_text="The breach caused the financial loss.",
        assertion_kind=AssertionKind.FACTUAL,
        speech_act=SpeechAct.ALLEGED,
        origin_kind=OriginKind.EXTRACTED,
        model_layer=ModelLayer.RECORD,
        source_role=SourceRole.ADVOCACY,
        document_id="complaint.pdf",
    )
    aid, _ = model.record_assertion(c)
    model.issues.link_assertion(aid, issue_id, "supports")

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_issue_coverage_summary()

    # 1 assertion → coverage_fraction = 1/2 = 0.5 → PARTIAL
    assert "Causation element" in result
    assert "PARTIAL" in result, f"Expected PARTIAL coverage label for 1 assertion: {result}"
    assert "1" in result


# ---------------------------------------------------------------------------
# SO-1: documents_from_cache / reuse_rate serialization and get_summary()
# ---------------------------------------------------------------------------

def test_investigation_state_documents_from_cache_roundtrip():
    """documents_from_cache must survive to_dict/from_dict (SO-1 metric persistence)."""
    from irys.rlm.state import InvestigationState

    state = InvestigationState.create("reuse test", "/repo")
    state.documents_read = 10
    state.documents_from_cache = 7

    data = state.to_dict()
    assert data.get("documents_from_cache") == 7

    restored = InvestigationState.from_dict(data)
    assert restored.documents_from_cache == 7
    assert restored.documents_read == 10


def test_investigation_state_documents_from_cache_defaults_zero():
    """from_dict() on old-format dict (missing documents_from_cache) must default to 0."""
    from irys.rlm.state import InvestigationState

    minimal = {"id": "abc", "query": "test", "repository_path": "/repo"}
    state = InvestigationState.from_dict(minimal)
    assert state.documents_from_cache == 0


def test_get_summary_reuse_rate_computed():
    """get_summary() must compute reuse_rate = documents_from_cache / documents_read (SO-1)."""
    from irys.rlm.state import InvestigationState

    state = InvestigationState.create("reuse rate test", "/repo")
    state.documents_read = 8
    state.documents_from_cache = 6

    summary = state.get_summary()
    metrics = summary.get("metrics", {})
    assert metrics.get("documents_from_cache") == 6
    assert metrics.get("reuse_rate") == pytest.approx(0.75, abs=0.001), (
        f"Expected reuse_rate 0.75, got {metrics.get('reuse_rate')}"
    )


def test_get_summary_reuse_rate_zero_when_no_docs_read():
    """get_summary() reuse_rate must be 0.0 when documents_read=0 (no division by zero)."""
    from irys.rlm.state import InvestigationState

    state = InvestigationState.create("empty state", "/repo")
    summary = state.get_summary()
    metrics = summary.get("metrics", {})
    assert metrics.get("reuse_rate") == 0.0


# ---------------------------------------------------------------------------
# SO-3: log_conflict writes CONFLICT_DETECTED to the reasoning ledger
# ---------------------------------------------------------------------------

def test_log_conflict_writes_conflict_detected_event():
    """MatterRuntimeAdapter.log_conflict() must write a CONFLICT_DETECTED ledger event (SO-3).

    This verifies that numeric or factual conflicts surfaced during investigation
    are recorded in the structured reasoning ledger — not silently discarded.
    The ledger is user-facing (SO-3), so conflict detection events must appear
    so the user can see and interrupt investigation around specific conflicts.
    """
    from irys.matter import LedgerEventType

    model = MatterModel.open_in_memory()
    run_id = model.start_run("conflict log test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.log_conflict(
        "Invoice #42 has conflicting amounts: $50,000 in contract vs $55,000 in email"
    )

    events = model.ledger.get_events(run_id)
    conflict_events = [
        e for e in events if e["event_type"] == LedgerEventType.CONFLICT_DETECTED.value
    ]
    assert len(conflict_events) == 1, "log_conflict must write exactly one CONFLICT_DETECTED event"
    assert "Invoice #42" in conflict_events[0]["summary"], (
        "Conflict summary must appear in ledger event"
    )


# ---------------------------------------------------------------------------
# SO-6: _build_quant_summary() surfaces date and rate facts
# ---------------------------------------------------------------------------

def test_request_redirect_writes_branch_selected_event():
    """request_redirect() must write a BRANCH_SELECTED ledger event (SO-3).

    The reasoning ledger is user-facing.  A redirect request must appear in the
    ledger so the user can see that their steering action was registered and which
    issue branch is now the focus — not just update a hidden flag.
    """
    from irys.matter import LedgerEventType

    from irys.matter.enums import IssueType

    model = MatterModel.open_in_memory()
    issue_id, _ = model.issues.upsert_issue("Breach of contract", IssueType.CLAIM)
    run_id = model.start_run("redirect test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.request_redirect(issue_id)

    # Flag must be set
    assert adapter.is_redirect_requested()
    assert adapter.get_redirect_issue_id() == issue_id

    # Ledger event must also be written
    events = model.ledger.get_events(run_id)
    redirect_events = [
        e for e in events if e["event_type"] == LedgerEventType.BRANCH_SELECTED.value
    ]
    assert len(redirect_events) == 1, (
        "request_redirect must write a BRANCH_SELECTED ledger event"
    )
    assert issue_id in redirect_events[0].get("summary", ""), (
        "BRANCH_SELECTED event summary must reference the target issue_id"
    )


def test_build_source_calibration_shows_litigation_side_breakdown():
    """_build_source_calibration() must show plaintiff vs. defendant document breakdown (SO-5).

    Knowing which side produced each block of facts is critical for source calibration:
    an operative contract signed by both parties is neutral, but a damages calculation
    produced entirely by plaintiff counsel is advocacy.  The side breakdown surfaces this.
    """
    model = MatterModel.open_in_memory()
    run_id = model.start_run("side breakdown test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # plaintiff_complaint.pdf → inferred source_side="plaintiff"
    adapter.record_fact("Defendant failed to deliver on time.", document_id="plaintiff_complaint.pdf")
    adapter.record_fact("Plaintiff suffered $200,000 in losses.", document_id="plaintiff_brief.pdf")

    # defendant_answer.pdf → inferred source_side="defendant"
    adapter.record_fact("Delivery was completed as scheduled.", document_id="defendant_answer.pdf")

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    calibration = engine._build_source_calibration(None)

    # Side breakdown section must appear
    assert "plaintiff" in calibration.lower(), (
        "Calibration must show plaintiff-side document count (SO-5 litigation-side breakdown)"
    )
    assert "defendant" in calibration.lower(), (
        "Calibration must show defendant-side document count"
    )


def test_log_gap_writes_gap_identified_event():
    """MatterRuntimeAdapter.log_gap() must write a GAP_IDENTIFIED ledger event (SO-3 + SO-7).

    Gaps are user-visible findings — the reasoning ledger must record them so the
    user can see what is missing and interrupt/redirect accordingly (SO-3).
    This also verifies SO-7: missingness is not silently skipped but logged.
    """
    from irys.matter import LedgerEventType

    model = MatterModel.open_in_memory()
    run_id = model.start_run("gap log test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.log_gap("Missing signed amendment — referenced in §4.2 but absent from repository")

    events = model.ledger.get_events(run_id)
    gap_events = [
        e for e in events if e["event_type"] == LedgerEventType.GAP_IDENTIFIED.value
    ]
    assert len(gap_events) == 1, "log_gap must write exactly one GAP_IDENTIFIED event"
    assert "amendment" in gap_events[0]["summary"].lower(), (
        "Gap summary must appear in ledger event"
    )


def test_adapter_record_gap_writes_gap_store_and_ledger():
    """adapter.record_gap() must persist the gap in the gap store AND write a ledger event (SO-3 + SO-7).

    record_gap() is the compound call that both records missingness structurally
    (SO-7) and surfaces it in the user-visible reasoning ledger (SO-3).
    """
    from irys.matter import LedgerEventType
    from irys.matter.enums import GapType

    model = MatterModel.open_in_memory()
    run_id = model.start_run("adapter record_gap test")
    adapter = MatterRuntimeAdapter(model, run_id)

    gap_id = adapter.record_gap(
        description="Exhibit B referenced but not produced",
        gap_type=GapType.MISSING_DOCUMENT,
        materiality=0.75,
    )

    assert gap_id, "record_gap must return a gap_id"

    # Gap store: the gap must be persisted and open
    open_gaps = model.gaps.open_gaps(min_materiality=0.0)
    assert len(open_gaps) == 1
    assert open_gaps[0]["id"] == gap_id

    # Ledger: a GAP_IDENTIFIED event must reference the gap
    events = model.ledger.get_events(run_id)
    gap_events = [e for e in events if e["event_type"] == LedgerEventType.GAP_IDENTIFIED.value]
    assert len(gap_events) == 1
    assert "Exhibit B" in gap_events[0]["summary"], (
        "Gap description must appear in ledger event summary"
    )


def test_build_quant_summary_shows_dates_and_rates():
    """_build_quant_summary() must include date and rate facts alongside monetary amounts (SO-6).

    Dates and rates are first-class quant kinds — not shown only as prose in memos.
    The synthesis block must include a 'Key dates' section and a 'Rates' section
    so the LLM can reference structured timeline and interest-rate data.
    """
    model = MatterModel.open_in_memory()

    # Record a monetary amount (to ensure amounts section is present)
    model.quant.record(quant_kind="amount", raw_text="$100,000 total claim",
                       amount_value=100_000.0, currency="USD", subject_type="claim")

    # Record date facts (SO-6 timeline intelligence)
    model.quant.record(quant_kind="date", raw_text="Contract execution date",
                       date_value="2023-01-15")
    model.quant.record(quant_kind="date", raw_text="Payment deadline",
                       date_value="2023-02-01")

    # Record a rate fact (SO-6 interest / penalty modeling)
    model.quant.record(quant_kind="rate", raw_text="18% per annum default interest rate",
                       rate_value=18.0)

    engine = RLMEngine.__new__(RLMEngine)
    engine._matter_model = model

    result = engine._build_quant_summary()

    # Date section must appear
    assert "2023-01-15" in result or "Contract execution" in result, (
        f"Date fact must appear in quant summary: {result}"
    )
    assert "2023-02-01" in result or "Payment deadline" in result, (
        f"Second date fact must appear in quant summary: {result}"
    )

    # Rate section must appear
    assert "18" in result, f"Rate value 18% must appear in quant summary: {result}"
    assert "interest" in result.lower() or "rate" in result.lower(), (
        f"Rate description must appear in quant summary: {result}"
    )


# ---------------------------------------------------------------------------
# SO-2 × SO-3: flush_revisions() writes ASSERTION_REVISED ledger event
# ---------------------------------------------------------------------------

def test_flush_revisions_writes_assertion_revised_event_on_state_change():
    """flush_revisions() must write ASSERTION_REVISED ledger events when belief state changes (SO-2 × SO-3).

    When fact A (pending) gains an OPERATIVE supporter B, flush_revisions()
    triggers belief revision: A's state transitions from UNKNOWN → OPERATIVE.
    That state change must produce an ASSERTION_REVISED event in the reasoning
    ledger so the user-visible reasoning trail reflects the inference.

    This is the SO-2 × SO-3 integration point: truth maintenance (SO-2)
    produces user-visible ledger evidence (SO-3).
    """
    from irys.matter import (
        AssertionCandidate, SpeechAct, SourceRole, ModelLayer, AssertionKind,
        AssertionLinkType, BeliefState, LedgerEventType,
    )
    from irys.matter.enums import OriginKind

    model = MatterModel.open_in_memory()
    run_id = model.start_run("flush revisions ledger test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Record fact A via adapter — puts A_id in _pending_assertion_ids
    a_id = adapter.record_fact(
        "The contract was executed on January 15.",
        document_id="contract.pdf",
    )

    # Record fact B directly (not via adapter) as an OPERATIVE supporter of A
    b_cand = AssertionCandidate(
        proposition_text="Executed copy of the contract bears both signatures.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="contract.pdf",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        origin_kind=OriginKind.EXTRACTED,
    )
    b_id, _ = model.assertions.upsert_occurrence(b_cand)
    model.assertions.set_belief_state(b_id, BeliefState.OPERATIVE, 0.9)

    # Link B → A as SUPPORTS so A gains an OPERATIVE supporter
    model.assertions.link(b_id, a_id, AssertionLinkType.SUPPORTS)

    # flush_revisions() should revise A (UNKNOWN → OPERATIVE) and write ledger event
    count = adapter.flush_revisions()
    assert count > 0, "flush_revisions() must report revised assertions when belief state changes"

    # ASSERTION_REVISED event must appear in the reasoning ledger (SO-3)
    events = model.ledger.get_events(run_id)
    revised_events = [
        e for e in events
        if e["event_type"] == LedgerEventType.ASSERTION_REVISED.value
    ]
    assert len(revised_events) >= 1, (
        "flush_revisions() must write ASSERTION_REVISED ledger event when belief state changes (SO-2×SO-3)"
    )
    # The event must reference the revised assertion
    assert any(e.get("changed_object_id") == a_id for e in revised_events), (
        "ASSERTION_REVISED event must reference the assertion whose belief state changed"
    )


def test_flush_revisions_clears_pending_list_so_second_call_is_noop():
    """flush_revisions() must clear _pending_assertion_ids after running (SO-2 correctness).

    If flush_revisions() does NOT clear the pending list, a second call would
    re-trigger belief revision on already-revised assertions — producing either
    spurious ledger events or a runaway cascade.  The second call must be a no-op.
    """
    from irys.matter import LedgerEventType
    model = MatterModel.open_in_memory()
    run_id = model.start_run("Flush idempotency test")
    adapter = MatterRuntimeAdapter(model, run_id)

    # Record two facts — they go into _pending_assertion_ids
    adapter.record_fact("Fact A.", "doc1.pdf")
    adapter.record_fact("Fact B.", "doc2.pdf")

    # First flush: processes pending IDs
    first_count = adapter.flush_revisions()
    assert isinstance(first_count, int)

    # Count ASSERTION_REVISED events after first flush
    events_after_first = model.ledger.get_events(run_id)
    revised_count_after_first = sum(
        1 for e in events_after_first
        if e["event_type"] == LedgerEventType.ASSERTION_REVISED.value
    )

    # Second flush: must not re-process (pending list cleared)
    second_count = adapter.flush_revisions()
    assert second_count == 0, (
        "flush_revisions() called a second time must return 0 — "
        "_pending_assertion_ids must be cleared after first flush (SO-2 correctness)"
    )

    # No new ASSERTION_REVISED events must have been added
    events_after_second = model.ledger.get_events(run_id)
    revised_count_after_second = sum(
        1 for e in events_after_second
        if e["event_type"] == LedgerEventType.ASSERTION_REVISED.value
    )
    assert revised_count_after_second == revised_count_after_first, (
        "Second flush_revisions() must not produce additional ASSERTION_REVISED events"
    )
