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

        state = InvestigationState(id="m1", query="test query", repository_path="")

        await engine._add_current_facts(state, ["Same fact."], source_doc="doc1.pdf", origin="prefix_read")
        await engine._add_current_facts(state, ["Same fact."], source_doc="doc2.pdf", origin="prefix_read")

        structured = state.findings.get("structured_facts", [])
        assert len(structured) == 1, "Duplicate fact should not appear twice in structured_facts"
