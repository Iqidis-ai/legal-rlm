"""Tests for extraction resilience — Phase 1 (BUG-1782324002896)."""
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.rlm.decisions import _salvage_truncated_json, extract_facts
from irys.rlm.engine import InvestigationCache, RLMEngine
from irys.rlm.state import InvestigationState
from irys.core.models import ModelTier


# ---------------------------------------------------------------------------
# _salvage_truncated_json
# ---------------------------------------------------------------------------

def test_salvage_clean_json_passthrough():
    text = '{"facts": ["Net 30"], "quotes": []}'
    result = _salvage_truncated_json(text)
    assert result is not None
    assert result["facts"] == ["Net 30"]


def test_salvage_truncated_mid_array():
    # Simulates LITE truncation mid-second item — first item survives
    text = '{"facts": ["Net 30 payment terms", "3% annual escalat'
    result = _salvage_truncated_json(text)
    assert result is not None
    assert "facts" in result
    assert result["facts"] == ["Net 30 payment terms"]


def test_salvage_no_facts_key_returns_none():
    # No facts key — not useful to salvage
    text = '{"quotes": ["some quote"'
    result = _salvage_truncated_json(text)
    assert result is None


def test_salvage_empty_string_returns_none():
    assert _salvage_truncated_json("") is None


def test_salvage_no_brace_returns_none():
    assert _salvage_truncated_json("model error: rate limit exceeded") is None


def test_salvage_fully_closed_facts_preserved():
    # Both facts closed; quotes truncated mid-entry — facts still returned
    text = '{"facts": ["Net 30", "3% escalation"], "quotes": ["partial quo'
    result = _salvage_truncated_json(text)
    assert result is not None
    assert result["facts"] == ["Net 30", "3% escalation"]


# ---------------------------------------------------------------------------
# extract_facts — new signature and status returns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_facts_ok_clean_parse():
    client = AsyncMock()
    client.complete = AsyncMock(return_value='{"facts": ["Net 30"], "quotes": []}')
    facts, status = await extract_facts("query", "file.pdf", "content", client)
    assert status == "ok"
    assert facts["facts"] == ["Net 30"]


@pytest.mark.asyncio
async def test_extract_facts_ok_empty_facts_is_not_an_error():
    """Clean parse with facts:[] is ok — never triggers retry."""
    client = AsyncMock()
    client.complete = AsyncMock(return_value='{"facts": [], "quotes": []}')
    facts, status = await extract_facts("query", "file.pdf", "content", client)
    assert status == "ok"
    assert facts["facts"] == []


@pytest.mark.asyncio
async def test_extract_facts_parse_failed():
    client = AsyncMock()
    client.complete = AsyncMock(return_value="not valid json at all")
    facts, status = await extract_facts("query", "file.pdf", "content", client)
    assert status == "parse_failed"
    assert facts == {"facts": [], "quotes": [], "references": []}


@pytest.mark.asyncio
async def test_extract_facts_api_error_returns_status():
    """503 / timeout / network error → api_error, never raises."""
    client = AsyncMock()
    client.complete = AsyncMock(side_effect=Exception("503 Service Unavailable"))
    facts, status = await extract_facts("query", "file.pdf", "content", client)
    assert status == "api_error"
    assert facts == {}


@pytest.mark.asyncio
async def test_extract_facts_salvage_on_final_attempt():
    """allow_salvage=True + truncated JSON → salvaged status."""
    client = AsyncMock()
    # Truncated mid-second fact — first fact survives salvage
    client.complete = AsyncMock(
        return_value='{"facts": ["Net 30 payment terms", "3% annual escalat'
    )
    facts, status = await extract_facts(
        "query", "file.pdf", "content", client,
        tier=ModelTier.FLASH, allow_salvage=True,
    )
    assert status == "salvaged"
    assert len(facts.get("facts", [])) >= 1


@pytest.mark.asyncio
async def test_extract_facts_no_salvage_on_attempt_1():
    """allow_salvage=False — parse_failed returned even for salvageable text."""
    client = AsyncMock()
    client.complete = AsyncMock(
        return_value='{"facts": ["Net 30 payment terms", "3% annual escalat'
    )
    facts, status = await extract_facts(
        "query", "file.pdf", "content", client,
        tier=ModelTier.LITE, allow_salvage=False,
    )
    assert status == "parse_failed"


@pytest.mark.asyncio
async def test_extract_facts_api_error_no_salvage():
    """api_error on final attempt → api_error (no text to salvage from)."""
    client = AsyncMock()
    client.complete = AsyncMock(side_effect=Exception("timeout"))
    facts, status = await extract_facts(
        "query", "file.pdf", "content", client,
        tier=ModelTier.FLASH, allow_salvage=True,
    )
    assert status == "api_error"
    assert facts == {}


@pytest.mark.asyncio
async def test_extract_facts_uses_flash_tier_when_specified():
    """tier=FLASH is passed through to client.complete."""
    client = AsyncMock()
    client.complete = AsyncMock(return_value='{"facts": [], "quotes": []}')
    await extract_facts("q", "f.pdf", "c", client, tier=ModelTier.FLASH)
    call_kwargs = client.complete.call_args
    assert call_kwargs.kwargs.get("tier") == ModelTier.FLASH or \
           (call_kwargs.args and ModelTier.FLASH in call_kwargs.args)


# ---------------------------------------------------------------------------
# InvestigationCache — extraction attempt tracking
# ---------------------------------------------------------------------------

def test_extraction_budget_starts_full():
    cache = InvestigationCache()
    assert cache.extraction_budget_remaining("doc::scope1") is True


def test_extraction_budget_exhausted_after_max():
    cache = InvestigationCache()
    cache.record_extraction_attempt("doc::scope1")   # attempt 1
    assert cache.extraction_budget_remaining("doc::scope1") is True
    cache.record_extraction_attempt("doc::scope1")   # attempt 2
    assert cache.extraction_budget_remaining("doc::scope1") is False


def test_extraction_attempts_are_per_scope_key():
    cache = InvestigationCache()
    cache.record_extraction_attempt("doc1::scope")
    cache.record_extraction_attempt("doc1::scope")
    # doc2 untouched
    assert cache.extraction_budget_remaining("doc2::scope") is True


def test_extraction_attempts_independent_from_read_failures():
    """Read failures must not consume extraction retry budget."""
    cache = InvestigationCache()
    cache.record_read_failure()
    cache.record_read_failure()
    cache.record_read_failure()
    assert cache.extraction_budget_remaining("doc::scope") is True


# ---------------------------------------------------------------------------
# RLMEngine static methods — failure recording and formatting
# ---------------------------------------------------------------------------

def _make_state() -> InvestigationState:
    return InvestigationState(
        id="test01",
        query="What are the payment terms?",
        repository_path="/tmp/test",
    )


def test_record_extraction_failure_writes_to_findings():
    state = _make_state()
    RLMEngine._record_extraction_failure(
        state, "scope1", "draft_msa.pdf", 62477,
        "parse_failed", 0, 2, ["LITE", "FLASH"],
    )
    assert "extraction_failures" in state.findings
    rec = state.findings["extraction_failures"]["scope1"]
    assert rec["status"] == "parse_failed"
    assert rec["filename"] == "draft_msa.pdf"
    assert rec["chars_read"] == 62477
    assert rec["attempts_used"] == 2
    assert rec["tiers_used"] == ["LITE", "FLASH"]


def test_record_extraction_failure_overwrites_same_scope_key():
    state = _make_state()
    RLMEngine._record_extraction_failure(
        state, "scope1", "draft_msa.pdf", 62477,
        "parse_failed", 0, 1, ["LITE"],
    )
    RLMEngine._record_extraction_failure(
        state, "scope1", "draft_msa.pdf", 62477,
        "api_error", 0, 2, ["LITE", "FLASH"],
    )
    assert state.findings["extraction_failures"]["scope1"]["status"] == "api_error"
    assert len(state.findings["extraction_failures"]) == 1  # no duplicates


def test_record_extraction_failure_salvaged_records_count():
    state = _make_state()
    RLMEngine._record_extraction_failure(
        state, "scope1", "contract.pdf", 40000,
        "salvaged", 3, 2, ["LITE", "FLASH"],
    )
    assert state.findings["extraction_failures"]["scope1"]["salvaged_count"] == 3


def test_format_extraction_failures_empty_returns_empty_string():
    assert RLMEngine._format_extraction_failures({}) == ""


def test_format_extraction_failures_parse_failed():
    failures = {
        "scope1": {
            "filename": "draft_msa.pdf",
            "chars_read": 62477,
            "status": "parse_failed",
            "salvaged_count": 0,
            "attempts_used": 2,
            "tiers_used": ["LITE", "FLASH"],
        }
    }
    result = RLMEngine._format_extraction_failures(failures)
    assert "⚠" in result
    assert "draft_msa.pdf" in result
    assert "parse_failed" in result
    assert "no salvage" in result
    assert "unverified" in result


def test_format_extraction_failures_salvaged_shows_fact_count():
    failures = {
        "scope1": {
            "filename": "contract.pdf",
            "chars_read": 40000,
            "status": "salvaged",
            "salvaged_count": 3,
            "attempts_used": 2,
            "tiers_used": ["LITE", "FLASH"],
        }
    }
    result = RLMEngine._format_extraction_failures(failures)
    assert "partial salvage=3 facts" in result


def test_format_extraction_failures_api_error():
    failures = {
        "scope1": {
            "filename": "exhibit_a.pdf",
            "chars_read": 30000,
            "status": "api_error",
            "salvaged_count": 0,
            "attempts_used": 2,
            "tiers_used": ["LITE", "FLASH"],
        }
    }
    result = RLMEngine._format_extraction_failures(failures)
    assert "api_error" in result
    assert "exhibit_a.pdf" in result


# ---------------------------------------------------------------------------
# End-to-end: retry loop + failure recording + evidence injection
# ---------------------------------------------------------------------------

def test_failure_section_omitted_when_no_failures():
    """_format_extraction_failures returns '' when dict is empty."""
    result = RLMEngine._format_extraction_failures({})
    assert result == ""


def test_failure_section_present_when_failures_exist():
    """Failures injected by _record_extraction_failure appear in formatted output."""
    state = _make_state()
    RLMEngine._record_extraction_failure(
        state, "draft_msa.pdf::full",
        "draft_msa.pdf", 62477,
        "parse_failed", 0, 2, ["LITE", "FLASH"],
    )
    section = RLMEngine._format_extraction_failures(
        state.findings.get("extraction_failures", {})
    )
    assert "⚠" in section
    assert "draft_msa.pdf" in section
    assert "parse_failed" in section


def test_failure_record_not_duplicated_on_second_call():
    """Writing the same scope_key twice results in exactly one record."""
    state = _make_state()
    RLMEngine._record_extraction_failure(
        state, "scope1", "f.pdf", 1000, "parse_failed", 0, 1, ["LITE"]
    )
    RLMEngine._record_extraction_failure(
        state, "scope1", "f.pdf", 1000, "parse_failed", 0, 2, ["LITE", "FLASH"]
    )
    assert len(state.findings["extraction_failures"]) == 1
    assert state.findings["extraction_failures"]["scope1"]["attempts_used"] == 2


def test_salvaged_status_included_in_failure_record():
    """Salvaged results are still recorded so harness can flag degraded data."""
    state = _make_state()
    RLMEngine._record_extraction_failure(
        state, "scope1", "contract.pdf", 40000,
        "salvaged", 2, 2, ["LITE", "FLASH"],
    )
    rec = state.findings["extraction_failures"]["scope1"]
    assert rec["status"] == "salvaged"
    assert rec["salvaged_count"] == 2


# ---------------------------------------------------------------------------
# Regression: create_summaries call site — extract_facts tuple unpack
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_facts_tuple_unpack_at_summary_call_site():
    """extract_facts returns (dict, status) — callers must unpack both values."""
    client = AsyncMock()
    client.complete = AsyncMock(return_value='{"facts": ["clause 1"], "quotes": []}')
    result = await extract_facts("Summarize this document", "doc.pdf", "content", client)
    # Must be a 2-tuple — callers like create_summaries use: extraction, _ = ...
    assert isinstance(result, tuple), "extract_facts must return (dict, status) tuple"
    assert len(result) == 2
    facts_dict, status = result
    assert isinstance(facts_dict, dict)
    assert status == "ok"
    assert facts_dict.get("facts") == ["clause 1"]
