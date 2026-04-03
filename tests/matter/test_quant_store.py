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
