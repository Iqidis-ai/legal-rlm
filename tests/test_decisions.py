"""Tests for decisions.py functions and engine helpers.

Network calls are mocked. No real LLM or file I/O occurs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Task 3 tests ────────────────────────────────────────────────────────────

class TestStructuredFactsTracking:
    """Verify _add_current_facts populates structured_facts alongside accumulated_facts."""

    @pytest.mark.asyncio
    async def test_structured_facts_populated_on_new_fact(self):
        from irys.rlm.engine import RLMEngine, RLMConfig
        from irys.rlm.state import InvestigationState

        engine = RLMEngine.__new__(RLMEngine)
        engine.config = RLMConfig()
        engine._telemetry = None
        engine.on_citation = None
        engine.fact_store = None

        state = InvestigationState(id="m1", query="test query", repository_path="")

        await engine._add_current_facts(
            state,
            facts=["The contract was signed on 2020-01-15.", "The parties agreed to 12% interest."],
            source_doc="contract.pdf",
            origin="prefix_read",
        )

        structured = state.findings.get("structured_facts", [])
        assert len(structured) == 2
        assert structured[0]["text"] == "The contract was signed on 2020-01-15."
        assert structured[0]["source"] == "contract.pdf"
        assert structured[1]["source"] == "contract.pdf"

    @pytest.mark.asyncio
    async def test_duplicate_fact_not_added_to_structured_facts(self):
        from irys.rlm.engine import RLMEngine, RLMConfig
        from irys.rlm.state import InvestigationState

        engine = RLMEngine.__new__(RLMEngine)
        engine.config = RLMConfig()
        engine._telemetry = None
        engine.on_citation = None
        engine.fact_store = None

        state = InvestigationState(id="m1", query="test query", repository_path="")

        await engine._add_current_facts(state, ["Same fact."], source_doc="doc1.pdf", origin="prefix_read")
        await engine._add_current_facts(state, ["Same fact."], source_doc="doc2.pdf", origin="prefix_read")

        structured = state.findings.get("structured_facts", [])
        assert len(structured) == 1, "Duplicate fact should not appear twice in structured_facts"


# ── Task 4 tests ────────────────────────────────────────────────────────────

class TestDetectContradictions:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_contradictions(self):
        from irys.rlm.decisions import detect_contradictions

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value='[]')

        result = await detect_contradictions(
            new_facts=[{"text": "X was signed on Jan 1.", "source": "a.pdf"}],
            recent_window=[{"text": "Y was signed on Jan 2.", "source": "b.pdf"}],
            client=mock_client,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_contradiction_dicts(self):
        from irys.rlm.decisions import detect_contradictions

        mock_client = MagicMock()
        contradiction_json = '''[
            {
                "statement1": "X was signed Jan 1",
                "source1": "a.pdf",
                "statement2": "X was signed Jan 5",
                "source2": "b.pdf",
                "contradiction_type": "factual",
                "severity": "high",
                "notes": "Conflicting dates for same event"
            }
        ]'''
        mock_client.complete = AsyncMock(return_value=contradiction_json)

        result = await detect_contradictions(
            new_facts=[{"text": "X was signed Jan 1", "source": "a.pdf"}],
            recent_window=[{"text": "X was signed Jan 5", "source": "b.pdf"}],
            client=mock_client,
        )
        assert len(result) == 1
        c = result[0]
        assert c["contradiction_type"] == "factual"
        assert c["severity"] == "high"
        assert "source1" in c and "source2" in c

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_malformed_llm_response(self):
        from irys.rlm.decisions import detect_contradictions

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="NOT JSON AT ALL")

        result = await detect_contradictions(
            new_facts=[{"text": "fact", "source": "x.pdf"}],
            recent_window=[],
            client=mock_client,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_llm_call_when_no_new_facts(self):
        from irys.rlm.decisions import detect_contradictions

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="[]")

        result = await detect_contradictions(
            new_facts=[],
            recent_window=[{"text": "existing fact", "source": "b.pdf"}],
            client=mock_client,
        )
        assert result == []
        mock_client.complete.assert_not_called()
