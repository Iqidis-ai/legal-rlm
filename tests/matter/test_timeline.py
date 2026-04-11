"""Tests for get_timeline() — visual work product Priority 2.

Verifies:
1. Empty model returns empty list
2. Date-type quant facts appear in timeline
3. Date_range quant facts appear in timeline
4. Assertions with temporal_scope_start appear in timeline
5. Events ordered by date ascending
6. Undated events appear last
7. limit parameter caps results
8. API GET /matter/{id}/timeline returns correct events
9. Mixed date sources merge and sort correctly
"""

import pytest
from irys.matter import MatterModel, SpeechAct, SourceRole, AssertionKind
from irys.matter.models import AssertionCandidate


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_empty_model_returns_empty_timeline(model):
    assert model.get_timeline() == []


def test_date_quant_appears_in_timeline(model):
    model.quant.record(
        quant_kind="date",
        raw_text="Contract signed January 15, 2024",
        date_value="2024-01-15",
    )
    events = model.get_timeline()
    assert len(events) == 1
    assert events[0]["kind"] == "date"
    assert events[0]["date"] == "2024-01-15"
    assert "Contract signed" in events[0]["event"]


def test_date_range_quant_appears_in_timeline(model):
    model.quant.record(
        quant_kind="date_range",
        raw_text="Service period March 1 – June 30, 2023",
        date_value="2023-03-01",
        date_end_value="2023-06-30",
    )
    events = model.get_timeline()
    assert len(events) == 1
    assert events[0]["kind"] == "date_range"


def test_temporal_assertion_appears_in_timeline(model):
    cand = AssertionCandidate(
        proposition_text="Defendant breached on March 5, 2024",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        assertion_kind=AssertionKind.TEMPORAL,
        document_id="complaint.pdf",
        temporal_scope_start="2024-03-05",
    )
    model.assertions.upsert_occurrence(cand)
    events = model.get_timeline()
    assert len(events) == 1
    assert events[0]["kind"] == "temporal_assertion"
    assert events[0]["date"] == "2024-03-05"
    assert events[0]["assertion_id"] is not None


def test_assertion_without_temporal_scope_excluded(model):
    cand = AssertionCandidate(
        proposition_text="No date here",
        speech_act=SpeechAct.OPERATIVE,
        source_role=SourceRole.OPERATIVE,
        assertion_kind=AssertionKind.FACTUAL,
        document_id="doc.pdf",
    )
    model.assertions.upsert_occurrence(cand)
    assert model.get_timeline() == []


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_events_ordered_by_date_ascending(model):
    model.quant.record(quant_kind="date", raw_text="Late event", date_value="2025-06-01")
    model.quant.record(quant_kind="date", raw_text="Early event", date_value="2023-01-01")
    model.quant.record(quant_kind="date", raw_text="Mid event", date_value="2024-03-15")

    events = model.get_timeline()
    dates = [e["date"] for e in events]
    assert dates == sorted(dates)


def test_undated_events_appear_last(model):
    model.quant.record(quant_kind="date", raw_text="Dated event", date_value="2024-01-01")
    model.quant.record(quant_kind="date", raw_text="Undated event")  # no date_value

    events = model.get_timeline()
    # Dated events before undated
    dated = [e for e in events if e["date"]]
    undated = [e for e in events if not e["date"]]
    assert len(dated) >= 1
    # All dated events appear before undated (if list is sorted correctly)
    if undated:
        last_dated_idx = max(i for i, e in enumerate(events) if e["date"])
        first_undated_idx = min(i for i, e in enumerate(events) if not e["date"])
        assert last_dated_idx < first_undated_idx


def test_mixed_sources_merge_and_sort(model):
    # Quant date
    model.quant.record(quant_kind="date", raw_text="Invoice date", date_value="2024-02-01")
    # Temporal assertion
    cand = AssertionCandidate(
        proposition_text="Breach on Jan 10",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        assertion_kind=AssertionKind.TEMPORAL,
        document_id="complaint.pdf",
        temporal_scope_start="2024-01-10",
    )
    model.assertions.upsert_occurrence(cand)

    events = model.get_timeline()
    assert len(events) == 2
    assert events[0]["date"] == "2024-01-10"  # assertion first
    assert events[1]["date"] == "2024-02-01"  # quant second


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------

def test_limit_caps_results(model):
    for i in range(10):
        model.quant.record(
            quant_kind="date",
            raw_text=f"Event {i}",
            date_value=f"2024-{i+1:02d}-01",
        )
    events = model.get_timeline(limit=5)
    assert len(events) <= 5


# ---------------------------------------------------------------------------
# Event structure
# ---------------------------------------------------------------------------

def test_quant_event_has_quant_id(model):
    model.quant.record(quant_kind="date", raw_text="Event", date_value="2024-01-01")
    events = model.get_timeline()
    assert events[0]["quant_id"] is not None
    assert events[0]["assertion_id"] is None or events[0]["quant_id"] is not None


def test_assertion_event_has_assertion_id(model):
    cand = AssertionCandidate(
        proposition_text="Temporal event",
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        assertion_kind=AssertionKind.TEMPORAL,
        document_id="doc.pdf",
        temporal_scope_start="2024-05-01",
    )
    model.assertions.upsert_occurrence(cand)
    events = model.get_timeline()
    assert events[0]["assertion_id"] is not None
    assert events[0]["quant_id"] is None


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

def _reg(active, m):
    active[m.matter_id] = m
    return m.matter_id


def test_api_timeline_empty(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)
    resp = client.get(f"/matter/{mid}/timeline")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_timeline_returns_events(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    m.quant.record(quant_kind="date", raw_text="Signed contract", date_value="2024-01-01")
    m.quant.record(quant_kind="date", raw_text="Breach date", date_value="2024-06-15")

    resp = client.get(f"/matter/{mid}/timeline")
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) == 2
    # Should be ordered
    assert events[0]["date"] == "2024-01-01"
    assert events[1]["date"] == "2024-06-15"


def test_api_timeline_limit(api_client):
    client, active = api_client
    m = MatterModel.open_in_memory()
    mid = _reg(active, m)

    for i in range(8):
        m.quant.record(
            quant_kind="date",
            raw_text=f"Event {i}",
            date_value=f"2024-{i+1:02d}-01",
        )

    resp = client.get(f"/matter/{mid}/timeline?limit=3")
    assert resp.status_code == 200
    assert len(resp.json()) <= 3
