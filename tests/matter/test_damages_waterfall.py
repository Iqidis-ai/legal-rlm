"""Tests for get_damages_waterfall() — visual work product Priority 2.

Verifies:
1. Empty model returns empty list
2. Single amount produces one waterfall entry
3. Amounts grouped by subject_type
4. claimed_amount is the sum of all amounts in the group
5. Waterfall ordered by claimed_amount descending
6. Null subject_type groups as '(uncategorised)'
7. source_count reflects number of quant_fact entries
8. Conflict detection: multiple distinct amounts with spread > 20%
9. No conflict when all amounts are equal
10. Currency filter: only matching currency included
11. API endpoint returns correct waterfall
"""

import pytest
from irys.matter import MatterModel


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_empty_model_returns_empty_list(model):
    assert model.get_damages_waterfall() == []


def test_single_amount_entry(model):
    model.quant.record(
        quant_kind="amount",
        raw_text="Lost profits: $500,000",
        amount_value=500_000.0,
        currency="USD",
        subject_type="lost_profits",
    )
    waterfall = model.get_damages_waterfall()
    assert len(waterfall) == 1
    assert waterfall[0]["component"] == "lost_profits"
    assert waterfall[0]["claimed_amount"] == 500_000.0


def test_amounts_grouped_by_subject_type(model):
    model.quant.record(
        quant_kind="amount", raw_text="Wage loss A", amount_value=10_000.0,
        currency="USD", subject_type="wage_loss", subject_id="w1",
    )
    model.quant.record(
        quant_kind="amount", raw_text="Wage loss B", amount_value=5_000.0,
        currency="USD", subject_type="wage_loss", subject_id="w2",
    )
    model.quant.record(
        quant_kind="amount", raw_text="Medical bills", amount_value=20_000.0,
        currency="USD", subject_type="medical",
    )

    waterfall = model.get_damages_waterfall()
    assert len(waterfall) == 2
    components = {e["component"]: e for e in waterfall}
    assert components["wage_loss"]["claimed_amount"] == 15_000.0
    assert components["medical"]["claimed_amount"] == 20_000.0


def test_waterfall_ordered_by_claimed_amount_descending(model):
    model.quant.record(
        quant_kind="amount", raw_text="Small", amount_value=1_000.0,
        currency="USD", subject_type="small",
    )
    model.quant.record(
        quant_kind="amount", raw_text="Large", amount_value=1_000_000.0,
        currency="USD", subject_type="large",
    )
    model.quant.record(
        quant_kind="amount", raw_text="Medium", amount_value=50_000.0,
        currency="USD", subject_type="medium",
    )

    waterfall = model.get_damages_waterfall()
    amounts = [e["claimed_amount"] for e in waterfall]
    assert amounts == sorted(amounts, reverse=True)


def test_null_subject_type_grouped_as_uncategorised(model):
    model.quant.record(
        quant_kind="amount", raw_text="Unknown damages $10,000",
        amount_value=10_000.0, currency="USD",
    )
    waterfall = model.get_damages_waterfall()
    assert waterfall[0]["component"] == "(uncategorised)"


def test_source_count(model):
    for i in range(3):
        model.quant.record(
            quant_kind="amount",
            raw_text=f"Invoice #{i}: $5,000",
            amount_value=5_000.0,
            currency="USD",
            subject_type="invoice",
            subject_id=f"inv_{i}",
        )
    waterfall = model.get_damages_waterfall()
    assert waterfall[0]["source_count"] == 3


def test_waterfall_preserves_all_source_entries_with_grounding_fields(model):
    q1 = model.quant.record(
        quant_kind="amount",
        raw_text="Invoice 1001 says $15,000",
        amount_value=15_000.0,
        currency="USD",
        subject_type="invoice",
        subject_id="inv_1001",
        span_id="span_1",
    )
    q2 = model.quant.record(
        quant_kind="amount",
        raw_text="Follow-up spreadsheet says $15,500",
        amount_value=15_500.0,
        currency="USD",
        subject_type="invoice",
        subject_id="inv_1002",
        span_id="span_2",
    )

    waterfall = model.get_damages_waterfall()
    entry = waterfall[0]

    assert len(entry["amounts"]) == 2
    assert {row["quant_fact_id"] for row in entry["amounts"]} == {q1, q2}
    assert {row["span_id"] for row in entry["amounts"]} == {"span_1", "span_2"}
    assert all("assertion_id" in row for row in entry["amounts"])
    assert {row["raw_text"] for row in entry["amounts"]} == {
        "Invoice 1001 says $15,000",
        "Follow-up spreadsheet says $15,500",
    }


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def test_conflict_detected_when_spread_exceeds_20_pct(model):
    model.quant.record(
        quant_kind="amount", raw_text="High estimate $100k",
        amount_value=100_000.0, currency="USD", subject_type="exposure",
        subject_id="e1",
    )
    model.quant.record(
        quant_kind="amount", raw_text="Low estimate $50k",
        amount_value=50_000.0, currency="USD", subject_type="exposure",
        subject_id="e2",
    )

    waterfall = model.get_damages_waterfall()
    entry = waterfall[0]
    # Spread: 50k / 100k = 50% > 20% → conflict
    assert len(entry["conflicts"]) >= 2


def test_no_conflict_when_amounts_equal(model):
    model.quant.record(
        quant_kind="amount", raw_text="A $50k", amount_value=50_000.0,
        currency="USD", subject_type="fee", subject_id="f1",
    )
    model.quant.record(
        quant_kind="amount", raw_text="B $50k", amount_value=50_000.0,
        currency="USD", subject_type="fee", subject_id="f2",
    )

    waterfall = model.get_damages_waterfall()
    assert waterfall[0]["conflicts"] == []


def test_small_spread_no_conflict(model):
    """Spread < 20% → no conflict."""
    model.quant.record(
        quant_kind="amount", raw_text="Est A", amount_value=100_000.0,
        currency="USD", subject_type="est", subject_id="a",
    )
    model.quant.record(
        quant_kind="amount", raw_text="Est B", amount_value=95_000.0,
        currency="USD", subject_type="est", subject_id="b",
    )
    # Spread: 5k / 100k = 5% < 20%
    waterfall = model.get_damages_waterfall()
    assert waterfall[0]["conflicts"] == []


# ---------------------------------------------------------------------------
# Currency filter
# ---------------------------------------------------------------------------

def test_currency_filter_includes_only_matching(model):
    model.quant.record(
        quant_kind="amount", raw_text="USD claim", amount_value=100.0,
        currency="USD", subject_type="claim",
    )
    model.quant.record(
        quant_kind="amount", raw_text="EUR claim", amount_value=200.0,
        currency="EUR", subject_type="claim",
    )

    usd_waterfall = model.get_damages_waterfall(currency="USD")
    eur_waterfall = model.get_damages_waterfall(currency="EUR")
    assert usd_waterfall[0]["claimed_amount"] == 100.0
    assert eur_waterfall[0]["claimed_amount"] == 200.0


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

def _reg(active, m):
    active[m.matter_id] = m
    return m.matter_id


def test_api_damages_waterfall_empty(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    resp = client.get(f"/matter/{mid}/damages-waterfall")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_damages_waterfall_returns_entries(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    m.quant.record(
        quant_kind="amount", raw_text="Lost profits $250,000",
        amount_value=250_000.0, currency="USD", subject_type="lost_profits",
    )
    m.quant.record(
        quant_kind="amount", raw_text="Attorney fees $50,000",
        amount_value=50_000.0, currency="USD", subject_type="attorney_fees",
    )

    resp = client.get(f"/matter/{mid}/damages-waterfall")
    assert resp.status_code == 200
    waterfall = resp.json()
    assert len(waterfall) == 2
    # Ordered: lost_profits first (larger)
    assert waterfall[0]["component"] == "lost_profits"
    assert waterfall[0]["claimed_amount"] == 250_000.0
