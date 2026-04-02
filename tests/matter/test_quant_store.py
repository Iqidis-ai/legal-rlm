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
