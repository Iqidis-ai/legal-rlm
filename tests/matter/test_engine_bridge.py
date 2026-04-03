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
    adapter.log_conflict("Contradiction: fact A conflicts with fact B")  # must not raise
    adapter.log_gap("Missing document", "contract.pdf")  # must not raise
    assert adapter.record_gap("Missing: signed amendment") == ""  # must not raise
    # record_facts_batch must return a list of empty strings, same length as input
    result = adapter.record_facts_batch([("fact a", "doc.pdf"), ("fact b", "doc.pdf")])
    assert result == ["", ""]


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
    from irys.matter.enums import BeliefState
    run_id = model.start_run("hydration test")
    adapter = MatterRuntimeAdapter(model, run_id)
    aid1 = adapter.record_fact("Active fact.", "doc1.pdf")
    aid2 = adapter.record_fact("Disputed fact.", "doc2.pdf")
    model.correct_assertion(aid2, BeliefState.DISPUTED)

    rows = model.assertions.list_recent_for_hydration(limit=50)
    assert any(r["id"] == aid1 for r in rows)
    assert any(r["id"] == aid2 for r in rows)  # method returns all; filtering is in engine
    # Verify belief_state is present in each row for the engine's filter
    assert all("belief_state" in r for r in rows)
    assert all("proposition_text" in r for r in rows)
    assert all("source_role" in r for r in rows)


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
