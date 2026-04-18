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


def test_display_date_tolerates_null_date_parts():
    from irys.ui.app import _display_date

    assert _display_date("2024-null-15", "day") == "2024-null-15"
    assert _display_date("null", "day") == "Undated"


def test_llm_token_breakdown_keeps_input_and_output_separate():
    from irys.ui.app import _llm_token_breakdown

    text = _llm_token_breakdown({
        "input_tokens": 1200,
        "cache_read_tokens": 300,
        "tool_use_prompt_tokens": 25,
        "thinking_tokens": 80,
        "output_tokens": 400,
        "total_processed_tokens": 2005,
    })

    assert text == "1,200 input / 300 cache / 25 tool / 80 thinking / 400 output"


def test_llm_analytics_panel_drops_aggregate_prompt_and_total_columns():
    from irys.ui.app import _fmt_llm_analytics_panel

    summary = {
        "request_count": 4,
        "input_tokens": 1200,
        "cache_read_tokens": 300,
        "tool_use_prompt_tokens": 25,
        "thinking_tokens": 80,
        "output_tokens": 400,
        "total_processed_tokens": 2005,
        "estimated_cost_usd": 0.00166,
    }

    html = _fmt_llm_analytics_panel(summary, calls=[], breakdown=None, anomalies=None)

    assert "Total work" not in html
    assert "<th>Prompt</th>" not in html
    assert "<th>Total</th>" not in html
    assert "<th>Model</th>" not in html
    assert ">Input<" in html
    assert ">Output<" in html


def test_source_drawer_uses_tier_not_model_name():
    from irys.ui.app import _fmt_source_drawer

    html = _fmt_source_drawer(
        "assertion",
        "a1",
        [{
            "source_document_ref": "complaint.pdf",
            "source_span_id": "section:4.2",
            "source_span_status": "present",
            "created_at": "2026-04-18T12:00:00Z",
            "model_id": "gemini-2.5-flash-lite",
            "model_tier": "LITE",
        }],
        [],
    )

    assert "gemini-2.5-flash-lite" not in html
    assert "LITE tier" in html


def test_in_process_backend_provenance_redacts_model_id():
    import asyncio
    from irys.ui.backends.in_process import InProcessBackend

    class _FakeModel:
        def get_provenance(self, target_kind, target_id, limit=50):
            assert target_kind == "assertion"
            assert target_id == "a1"
            assert limit == 50
            return [{
                "model_id": "gemini-2.5-flash-lite",
                "model_tier": "LITE",
                "source_document_ref": "complaint.pdf",
            }]

    backend = InProcessBackend(api_key="test")
    backend._get_matter_model = lambda _matter_id: _FakeModel()  # type: ignore[method-assign]

    rows = asyncio.run(backend.get_provenance("m1", "assertion", "a1"))

    assert rows == [{
        "model_tier": "LITE",
        "source_document_ref": "complaint.pdf",
    }]
