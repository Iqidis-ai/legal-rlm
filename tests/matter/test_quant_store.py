"""Tests for QuantStore and SO-6 quantitative intelligence wiring.

Verifies:
1. QuantStore.record() persists numeric facts
2. get_amounts() filters by kind and sorts by value
3. get_by_kind() retrieves by quant_kind
4. record_quant() via MatterRuntimeAdapter persists to QuantStore
5. NullMatterAdapter.record_quant() is a safe no-op
"""

import pytest
from irys.matter import MatterModel, QuantStore
from irys.matter.runtime import MatterRuntimeAdapter, NullMatterAdapter


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# QuantStore basic contract
# ---------------------------------------------------------------------------

def test_record_amount(model):
    qf_id = model.quant.record(
        quant_kind="amount",
        raw_text="$50,000 — payment due under Section 3",
        amount_value=50000.0,
        currency="USD",
    )
    assert qf_id
    assert model.quant.count() == 1


def test_get_amounts_returns_amounts(model):
    model.quant.record(quant_kind="amount", raw_text="$10,000", amount_value=10000.0, currency="USD")
    model.quant.record(quant_kind="amount", raw_text="$75,000", amount_value=75000.0, currency="USD")
    model.quant.record(quant_kind="date", raw_text="January 15, 2024")

    amounts = model.quant.get_amounts()
    assert len(amounts) == 2
    # Sorted by value descending
    assert amounts[0]["amount_value"] == 75000.0


def test_get_amounts_min_filter(model):
    model.quant.record(quant_kind="amount", raw_text="$500", amount_value=500.0, currency="USD")
    model.quant.record(quant_kind="amount", raw_text="$1,000,000", amount_value=1_000_000.0, currency="USD")

    large = model.quant.get_amounts(min_value=10_000.0)
    assert len(large) == 1
    assert large[0]["amount_value"] == 1_000_000.0


def test_get_by_kind(model):
    model.quant.record(quant_kind="date", raw_text="February 1, 2024 — payment deadline")
    model.quant.record(quant_kind="date", raw_text="March 15, 2024 — contract expiry")
    model.quant.record(quant_kind="amount", raw_text="$25,000")

    dates = model.quant.get_by_kind("date")
    assert len(dates) == 2
    assert all(d["quant_kind"] == "date" for d in dates)


def test_record_rate(model):
    qf_id = model.quant.record(
        quant_kind="rate",
        raw_text="18% per annum — default interest rate",
        rate_value=18.0,
        unit="percent_annual",
    )
    assert qf_id
    rates = model.quant.get_by_kind("rate")
    assert len(rates) == 1
    assert rates[0]["rate_value"] == 18.0


# ---------------------------------------------------------------------------
# MatterRuntimeAdapter.record_quant()
# ---------------------------------------------------------------------------

def test_adapter_record_quant_persists(model):
    run_id = model.start_run("Payment dispute test")
    adapter = MatterRuntimeAdapter(model, run_id)

    qf_id = adapter.record_quant(
        quant_kind="amount",
        raw_text="$100,000 — total damages claimed",
        amount_value=100_000.0,
        currency="USD",
    )

    assert qf_id
    assert model.quant.count() == 1


def test_null_adapter_record_quant():
    """NullMatterAdapter.record_quant() must not raise and return empty string."""
    adapter = NullMatterAdapter()
    result = adapter.record_quant("amount", "$50,000", amount_value=50000.0)
    assert result == ""


# ---------------------------------------------------------------------------
# SO-6 reconciliation and conflict detection
# ---------------------------------------------------------------------------

def test_reconcile_by_subject_groups_by_type(model):
    """reconcile_by_subject() must sum amounts per subject_type."""
    model.quant.record(quant_kind="amount", raw_text="inv1", amount_value=50_000.0,
                       currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="inv2", amount_value=30_000.0,
                       currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="pmt1", amount_value=40_000.0,
                       currency="USD", subject_type="payment")

    rec = model.quant.reconcile_by_subject("USD")
    assert "invoice" in rec
    assert "payment" in rec
    assert rec["invoice"]["total"] == 80_000.0
    assert rec["invoice"]["count"] == 2
    assert rec["payment"]["total"] == 40_000.0


def test_reconcile_by_subject_excludes_wrong_currency(model):
    """reconcile_by_subject() must not mix currencies."""
    model.quant.record(quant_kind="amount", raw_text="usd_inv", amount_value=10_000.0,
                       currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="eur_pmt", amount_value=9_000.0,
                       currency="EUR", subject_type="payment")

    rec_usd = model.quant.reconcile_by_subject("USD")
    assert "invoice" in rec_usd
    assert "payment" not in rec_usd


def test_get_conflicts_detects_same_subject_different_values(model):
    """get_conflicts() must flag when same subject_type has multiple distinct amounts."""
    model.quant.record(quant_kind="amount", raw_text="version A", amount_value=50_000.0,
                       currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="version B", amount_value=55_000.0,
                       currency="USD", subject_type="invoice")

    conflicts = model.quant.get_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0]["subject_type"] == "invoice"
    assert 50_000.0 in conflicts[0]["values"]
    assert 55_000.0 in conflicts[0]["values"]


def test_get_conflicts_no_conflict_when_values_agree(model):
    """get_conflicts() must return empty when all amounts for a subject agree."""
    model.quant.record(quant_kind="amount", raw_text="invoice", amount_value=50_000.0,
                       currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="invoice copy", amount_value=50_000.0,
                       currency="USD", subject_type="invoice")

    assert model.quant.get_conflicts() == []


def test_get_conflicts_no_conflict_distinct_subject_ids(model):
    """Two invoices with different subject_ids must NOT conflict even if amounts differ.

    With subject_id-aware grouping, each distinct (subject_type, subject_id, currency)
    tuple is its own bucket. Different invoices legitimately have different amounts.
    """
    model.quant.record(quant_kind="amount", raw_text="inv-1 amount", amount_value=50_000.0,
                       currency="USD", subject_type="invoice", subject_id="Invoice #1042")
    model.quant.record(quant_kind="amount", raw_text="inv-2 amount", amount_value=75_000.0,
                       currency="USD", subject_type="invoice", subject_id="Invoice #2017")

    assert model.quant.get_conflicts() == [], (
        "Distinct invoices with different subject_ids should not conflict"
    )


def test_get_conflicts_same_subject_id_different_amounts(model):
    """Same invoice recorded twice with different amounts IS a conflict (data error)."""
    model.quant.record(quant_kind="amount", raw_text="inv copy A", amount_value=50_000.0,
                       currency="USD", subject_type="invoice", subject_id="Invoice #1042")
    model.quant.record(quant_kind="amount", raw_text="inv copy B", amount_value=55_000.0,
                       currency="USD", subject_type="invoice", subject_id="Invoice #1042")

    conflicts = model.quant.get_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0]["subject_id"] == "Invoice #1042"
    assert 50_000.0 in conflicts[0]["values"]
    assert 55_000.0 in conflicts[0]["values"]


def test_adapter_record_quant_passes_subject_type(model):
    """record_quant() subject_type must be stored and queryable."""
    run_id = model.start_run("Reconciliation test")
    adapter = MatterRuntimeAdapter(model, run_id)

    adapter.record_quant(
        quant_kind="amount",
        raw_text="$75,000 invoice total",
        amount_value=75_000.0,
        currency="USD",
        subject_type="invoice",
    )

    rec = model.quant.reconcile_by_subject("USD")
    assert rec.get("invoice", {}).get("total") == 75_000.0


# ---------------------------------------------------------------------------
# MatterModel.detect_quant_conflicts() integration
# ---------------------------------------------------------------------------

def test_detect_quant_conflicts_creates_gap(model):
    """detect_quant_conflicts() must record UNRESOLVED_CONTRADICTION gap for each conflict."""
    model.quant.record(quant_kind="amount", raw_text="$50k — invoice A",
                       amount_value=50_000.0, currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="$55k — invoice B",
                       amount_value=55_000.0, currency="USD", subject_type="invoice")

    gap_ids = model.detect_quant_conflicts()
    assert len(gap_ids) == 1

    open_gaps = model.gaps.open_gaps(min_materiality=0.0)
    assert any(g["id"] == gap_ids[0] for g in open_gaps)
    assert any("invoice" in g["description"] for g in open_gaps)


def test_detect_quant_conflicts_is_idempotent(model):
    """Calling detect_quant_conflicts() twice must not create duplicate gaps."""
    model.quant.record(quant_kind="amount", raw_text="$50k", amount_value=50_000.0,
                       currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="$60k", amount_value=60_000.0,
                       currency="USD", subject_type="invoice")

    model.detect_quant_conflicts()
    model.detect_quant_conflicts()

    open_gaps = model.gaps.open_gaps(min_materiality=0.0)
    invoice_gaps = [g for g in open_gaps if "invoice" in g.get("description", "")]
    assert len(invoice_gaps) == 1  # not duplicated


def test_detect_quant_conflicts_no_conflicts_returns_empty(model):
    """detect_quant_conflicts() must return [] when no conflicts exist."""
    model.quant.record(quant_kind="amount", raw_text="$50k", amount_value=50_000.0,
                       currency="USD", subject_type="invoice")
    assert model.detect_quant_conflicts() == []


def test_detect_quant_conflicts_propagates_disputed_to_assertions(model):
    """SO-6→SO-2: detect_quant_conflicts() must wire contradicts links and mark
    the linked assertions as DISPUTED via BeliefRevisionEngine (truth maintenance).
    """
    from irys.matter.models import AssertionCandidate
    from irys.matter.enums import AssertionKind, SpeechAct, OriginKind

    # Record two assertions — one per conflicting quant fact
    cand_a = AssertionCandidate(
        proposition_text="Invoice #X totals $50,000",
        assertion_kind=AssertionKind.QUANTITATIVE,
        speech_act=SpeechAct.ALLEGED,
        origin_kind=OriginKind.EXTRACTED,
    )
    cand_b = AssertionCandidate(
        proposition_text="Invoice #X totals $55,000",
        assertion_kind=AssertionKind.QUANTITATIVE,
        speech_act=SpeechAct.ALLEGED,
        origin_kind=OriginKind.EXTRACTED,
    )
    aid_a, _ = model.record_assertion(cand_a)
    aid_b, _ = model.record_assertion(cand_b)

    # Link both quant facts to their respective assertions
    model.quant.record(
        quant_kind="amount", raw_text="$50k invoice X",
        amount_value=50_000.0, currency="USD",
        subject_type="invoice", subject_id="Invoice #X",
        assertion_id=aid_a,
    )
    model.quant.record(
        quant_kind="amount", raw_text="$55k invoice X",
        amount_value=55_000.0, currency="USD",
        subject_type="invoice", subject_id="Invoice #X",
        assertion_id=aid_b,
    )

    model.detect_quant_conflicts()

    # Both assertions must now be DISPUTED (belief revision propagated)
    rec_a = model.assertions.get(aid_a)
    rec_b = model.assertions.get(aid_b)
    assert rec_a is not None
    assert rec_b is not None
    assert rec_a.belief_state == "disputed", f"Expected disputed, got {rec_a.belief_state}"
    assert rec_b.belief_state == "disputed", f"Expected disputed, got {rec_b.belief_state}"


# ---------------------------------------------------------------------------
# SO-6: date and rate quant kinds are readable (not write-only)
# ---------------------------------------------------------------------------

def test_get_by_kind_date_returns_dates(model):
    """Date quant facts must be retrievable via get_by_kind('date') (SO-6)."""
    model.quant.record(quant_kind="date", raw_text="January 15, 2024",
                       date_value="2024-01-15")
    model.quant.record(quant_kind="date", raw_text="March 3, 2024",
                       date_value="2024-03-03")
    model.quant.record(quant_kind="amount", raw_text="$10,000", amount_value=10000.0)

    dates = model.quant.get_by_kind("date")
    assert len(dates) == 2
    assert all(d["quant_kind"] == "date" for d in dates)
    # Must be sorted chronologically
    assert dates[0]["date_value"] == "2024-01-15"
    assert dates[1]["date_value"] == "2024-03-03"


def test_get_by_kind_rate_returns_rates(model):
    """Rate quant facts must be retrievable via get_by_kind('rate') (SO-6)."""
    model.quant.record(quant_kind="rate", raw_text="8% annual interest rate",
                       rate_value=8.0)
    model.quant.record(quant_kind="rate", raw_text="1.5% monthly penalty",
                       rate_value=1.5)

    rates = model.quant.get_by_kind("rate")
    assert len(rates) == 2
    assert all(r["quant_kind"] == "rate" for r in rates)


def test_get_by_kind_empty_when_no_facts(model):
    """get_by_kind() returns [] when no facts of that kind exist."""
    assert model.quant.get_by_kind("date") == []
    assert model.quant.get_by_kind("rate") == []


def test_get_by_kind_limit(model):
    """get_by_kind(limit=N) returns at most N rows (DB-level bound)."""
    for i in range(5):
        model.quant.record(quant_kind="date", raw_text=f"Date {i}",
                           date_value=f"2024-0{i+1}-01")
    assert len(model.quant.get_by_kind("date", limit=3)) == 3
    assert len(model.quant.get_by_kind("date")) == 5


def test_get_by_kind_date_none_date_value(model):
    """Date fact with null date_value must still be retrievable (raw_text fallback)."""
    model.quant.record(quant_kind="date", raw_text="sometime in late 2023",
                       date_value=None)
    dates = model.quant.get_by_kind("date")
    assert len(dates) == 1
    assert dates[0]["date_value"] is None


# ---------------------------------------------------------------------------
# SO-6: record_quants_batch() batch ingestion
# ---------------------------------------------------------------------------

def test_adapter_record_quants_batch_persists_all_records(model):
    """record_quants_batch() must persist all specs in a single atomic write (SO-6).

    The batch API is the high-throughput path for quantitative extraction — when
    the LLM extracts 10 numeric facts in one pass, the engine calls record_quants_batch()
    once rather than 10 individual record_quant() calls.  All records must appear
    in QuantStore, not just the first one.
    """
    run_id = model.start_run("Batch quant test")
    adapter = MatterRuntimeAdapter(model, run_id)

    specs = [
        {"quant_kind": "amount", "raw_text": "$50,000 invoice total",
         "amount_value": 50_000.0, "currency": "USD", "subject_type": "invoice"},
        {"quant_kind": "amount", "raw_text": "$30,000 payment received",
         "amount_value": 30_000.0, "currency": "USD", "subject_type": "payment"},
        {"quant_kind": "rate", "raw_text": "8% annual interest",
         "rate_value": 8.0},
    ]
    adapter.record_quants_batch(specs)

    assert model.quant.count() == 3, (
        "record_quants_batch() must persist all 3 specs — batch write must not silently drop records"
    )
    amounts = model.quant.get_by_kind("amount")
    assert len(amounts) == 2
    rates = model.quant.get_by_kind("rate")
    assert len(rates) == 1


def test_adapter_record_quants_batch_seeds_verification_and_provenance(model):
    """Adversarial audit #6 regression: the batched path must seed
    verification_state + provenance_event on every newly-inserted row,
    matching the single-row record_quant() contract. The engine's real
    extraction path only uses the batch API, so MVP.2 + P0.1 are broken
    if this substrate seeding is missing."""
    run_id = model.start_run("Batch quant substrate")
    adapter = MatterRuntimeAdapter(model, run_id)
    adapter.record_quants_batch(
        [
            {"quant_kind": "amount", "raw_text": "$1 first",
             "amount_value": 1.0, "currency": "USD"},
            {"quant_kind": "amount", "raw_text": "$2 second",
             "amount_value": 2.0, "currency": "USD"},
        ],
        document_id="ledger.pdf",
    )
    all_quants = model.db.execute(
        "SELECT id FROM quant_fact WHERE matter_id=?", (model.matter_id,),
    ).fetchall()
    assert len(all_quants) == 2
    for row in all_quants:
        qid = row["id"]
        vs = model.verification.get("quant_fact", qid)
        assert vs is not None, f"quant_fact {qid} missing verification_state"
        assert vs["status"] == "candidate"
        events = model.get_provenance("quant_fact", qid)
        assert len(events) == 1, f"quant_fact {qid} missing provenance"
        assert events[0]["event_kind"] == "quant_record_batch"
        assert events[0]["source_document_ref"] == "ledger.pdf"
        assert events[0]["run_id"] == run_id


# ---------------------------------------------------------------------------
# SO-6: MatterModel.reconcile() payment reconciliation
# ---------------------------------------------------------------------------

def test_matter_model_reconcile_shows_payment_exposure(model):
    """model.reconcile() must expose invoiced vs. paid totals from quant facts (SO-6).

    SO-6 test contract: 'Given documents with payment histories, the system can
    produce a reconciliation showing what was invoiced, what was paid, what is
    disputed, and what the claimed exposure is.'

    reconcile() groups by subject_type so invoice and payment totals are
    independently retrievable — the caller computes exposure = invoiced - paid.
    """
    # Record three invoices and two payments
    model.quant.record(quant_kind="amount", raw_text="Invoice #1: $80,000",
                       amount_value=80_000.0, currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="Invoice #2: $20,000",
                       amount_value=20_000.0, currency="USD", subject_type="invoice")
    model.quant.record(quant_kind="amount", raw_text="Payment Mar 15: $60,000",
                       amount_value=60_000.0, currency="USD", subject_type="payment")
    model.quant.record(quant_kind="amount", raw_text="Payment Apr 1: $15,000",
                       amount_value=15_000.0, currency="USD", subject_type="payment")

    rec = model.reconcile(currency="USD")

    assert "invoice" in rec, "Reconciliation must include invoice totals (SO-6)"
    assert "payment" in rec, "Reconciliation must include payment totals (SO-6)"

    invoiced = rec["invoice"]["total"]
    paid = rec["payment"]["total"]
    assert invoiced == 100_000.0, f"Expected $100k invoiced, got {invoiced}"
    assert paid == 75_000.0, f"Expected $75k paid, got {paid}"

    # Exposure = invoiced - paid (what SO-6 calls 'claimed exposure')
    exposure = invoiced - paid
    assert exposure == 25_000.0, (
        "reconcile() must support computing exposure = invoiced - paid (SO-6 payment reconciliation)"
    )


# ---------------------------------------------------------------------------
# SO-6: compute_thresholds() — hard threshold detection and gap creation
# ---------------------------------------------------------------------------

def _add_quant(model, subject_type, amount, currency="USD", subject_id=None):
    """Helper: add a quant_fact of kind=amount."""
    model.quant.record(
        quant_kind="amount",
        raw_text=f"{subject_type} {amount} {currency}",
        subject_type=subject_type,
        subject_id=subject_id,
        amount_value=amount,
        currency=currency,
    )


def test_compute_thresholds_positive_exposure_creates_gap():
    """Positive exposure (invoiced > paid) must create a gap."""
    model = MatterModel.open_in_memory()
    _add_quant(model, "invoice", 50_000.0)
    _add_quant(model, "payment", 30_000.0)

    violations = model.compute_quant_thresholds()

    assert len(violations) >= 1, "Positive exposure must produce at least one violation"
    exposure_violation = next(
        (v for v in violations if v["threshold"] == "positive_exposure"), None
    )
    assert exposure_violation is not None, "Must have a 'positive_exposure' threshold violation"
    assert exposure_violation["amount"] == pytest.approx(20_000.0, abs=0.01)
    assert exposure_violation["level"] in ("HIGH", "MED")

    gaps = model.gaps.open_gaps()
    assert any("exposure" in g["description"].lower() for g in gaps), (
        "compute_thresholds() must record an exposure gap in the gap store"
    )


def test_compute_thresholds_no_violation_when_paid_in_full():
    """No threshold violation when invoiced == paid."""
    model = MatterModel.open_in_memory()
    _add_quant(model, "invoice", 100_000.0)
    _add_quant(model, "payment", 100_000.0)

    violations = model.compute_quant_thresholds()
    exposure_violations = [v for v in violations if v["threshold"] == "positive_exposure"]
    assert len(exposure_violations) == 0, "Zero exposure must not produce exposure violation"


def test_compute_thresholds_numeric_conflict_creates_gap():
    """Numeric conflict for same subject must create an UNRESOLVED_CONTRADICTION gap."""
    model = MatterModel.open_in_memory()
    # Two different amounts for the same invoice subject_id
    _add_quant(model, "invoice", 50_000.0, subject_id="INV-001")
    _add_quant(model, "invoice", 60_000.0, subject_id="INV-001")

    violations = model.compute_quant_thresholds()
    conflict_violations = [v for v in violations if v["threshold"] == "numeric_conflict"]
    assert len(conflict_violations) >= 1, (
        "Conflicting amounts for same subject_id must produce a numeric_conflict violation"
    )

    gaps = model.gaps.open_gaps()
    assert any("conflict" in g["description"].lower() or "INV-001" in g["description"]
               for g in gaps), "Numeric conflict must create a gap record"


def test_compute_thresholds_is_idempotent():
    """Repeated calls must not create duplicate gap records."""
    model = MatterModel.open_in_memory()
    _add_quant(model, "invoice", 25_000.0)
    _add_quant(model, "payment", 10_000.0)

    model.compute_quant_thresholds()
    model.compute_quant_thresholds()
    model.compute_quant_thresholds()

    gaps = model.gaps.open_gaps()
    exposure_gaps = [g for g in gaps if "exposure" in g["description"].lower()]
    assert len(exposure_gaps) == 1, (
        "Repeated compute_thresholds() calls must not create duplicate gap records"
    )
