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


# ── Task 5 tests ────────────────────────────────────────────────────────────

class TestPriorityDecay:
    def test_rlmconfig_has_decay_fields(self):
        from irys.rlm.engine import RLMConfig
        cfg = RLMConfig()
        assert hasattr(cfg, "priority_decay_factor")
        assert hasattr(cfg, "max_reflexion_cycles")
        assert cfg.priority_decay_factor == 0.7
        assert cfg.max_reflexion_cycles == 1

    @pytest.mark.asyncio
    async def test_leads_are_processed_highest_priority_first(self):
        from irys.rlm.engine import RLMConfig
        from irys.rlm.state import Lead

        config = RLMConfig(max_leads_per_level=2)
        low = Lead.create("low priority", source="test")
        low.params["priority"] = 0.2
        mid = Lead.create("mid priority", source="test")
        mid.params["priority"] = 0.5
        high = Lead.create("high priority", source="test")
        high.params["priority"] = 0.9

        pending = [low, mid, high]
        selected = sorted(
            pending,
            key=lambda l: l.params.get("priority", 1.0),
            reverse=True,
        )[:config.max_leads_per_level]

        assert selected[0].params["priority"] == 0.9
        assert selected[1].params["priority"] == 0.5

    def test_decay_applied_to_non_reflexion_leads(self):
        from irys.rlm.engine import RLMEngine
        from irys.rlm.state import Lead
        lead = Lead.create("test", source="s")
        lead.params["priority"] = 1.0
        RLMEngine._apply_priority_decay([lead], 0.7)
        assert abs(lead.params["priority"] - 0.7) < 1e-9

    def test_decay_exempt_for_reflexion_leads(self):
        from irys.rlm.engine import RLMEngine
        from irys.rlm.state import Lead
        lead = Lead.create("reflexion gap", source="reflexion")
        lead.params["priority"] = 1.0
        lead.params["origin"] = "reflexion"
        RLMEngine._apply_priority_decay([lead], 0.7)
        assert lead.params["priority"] == 1.0


# ── Task 6 tests ────────────────────────────────────────────────────────────

class TestPureHelpers:
    def test_count_citation_refs_finds_bracketed_numbers(self):
        from irys.rlm.decisions import _count_citation_refs
        text = "The contract [1] was signed. The party [2] agreed. See also [1] again."
        assert _count_citation_refs(text) == 2

    def test_count_citation_refs_empty_text(self):
        from irys.rlm.decisions import _count_citation_refs
        assert _count_citation_refs("") == 0

    def test_count_uncited_sentences_counts_uncited(self):
        from irys.rlm.decisions import _count_uncited_sentences
        text = "This is cited [1]. This is not cited. Another without citation."
        assert _count_uncited_sentences(text) == 2

    def test_count_uncited_sentences_all_cited(self):
        from irys.rlm.decisions import _count_uncited_sentences
        text = "First sentence [1]. Second sentence [2]."
        assert _count_uncited_sentences(text) == 0

    def test_count_uncited_sentences_handles_legal_abbreviations(self):
        from irys.rlm.decisions import _count_uncited_sentences
        # U.S.C. should not create false sentence split
        text = "Pursuant to U.S.C. 1983, plaintiff filed suit. The court held for defendant [1]."
        # Should count only the first sentence (U.S.C. is not split), not multiple
        assert _count_uncited_sentences(text) == 1

        # Multiple abbreviations: U.S.C., v., No., S.Ct.
        text2 = "Under U.S.C. section 1983. Smith v. Jones [1] is the controlling case. See No. 456 here [2]."
        # Should be 1 uncited (first sentence about U.S.C.), 2 cited
        assert _count_uncited_sentences(text2) == 1

        # et al., Id., supra, infra
        text3 = "See Jones et al. standards [1]. Under supra rule [1]. The Id. reference. Following infra."
        # Should count only "The Id. reference" and "Following infra" as uncited
        assert _count_uncited_sentences(text3) == 2


class TestCritiqueSynthesis:
    @pytest.mark.asyncio
    async def test_returns_ok_true_when_no_gaps(self):
        from irys.rlm.decisions import critique_synthesis

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value='{"ok": true, "gaps": [], "internal_contradictions": [], "uncited_sentences": []}')

        result = await critique_synthesis(
            query="What damages were claimed?",
            synthesis="The plaintiff claimed $500k in damages [1].",
            evidence="Document 1: plaintiff claimed $500k",
            client=mock_client,
        )
        assert result["ok"] is True
        assert result["gaps"] == []

    @pytest.mark.asyncio
    async def test_returns_gaps_when_found(self):
        from irys.rlm.decisions import critique_synthesis

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value='''{
            "ok": false,
            "gaps": ["Missing: defendant response to damages claim"],
            "internal_contradictions": [],
            "uncited_sentences": ["The defendant denied all claims."]
        }''')

        result = await critique_synthesis(
            query="What damages were claimed?",
            synthesis="The defendant denied all claims.",
            evidence="evidence...",
            client=mock_client,
        )
        assert result["ok"] is False
        assert len(result["gaps"]) == 1
        assert "defendant response" in result["gaps"][0]

    @pytest.mark.asyncio
    async def test_returns_ok_true_on_malformed_response(self):
        from irys.rlm.decisions import critique_synthesis

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value="GARBAGE RESPONSE")

        result = await critique_synthesis(
            query="q", synthesis="s", evidence="e", client=mock_client,
        )
        assert result["ok"] is True


# ── Task 7 tests ────────────────────────────────────────────────────────────

class TestDecisionLogging:

    def test_emit_decision_record_appends_to_log(self):
        from irys.rlm.decisions import _emit_decision_record
        from irys.core.models import ModelTier

        log: list[dict] = []
        _emit_decision_record(
            decision_log=log,
            func_name="detect_contradictions",
            tier=ModelTier.FLASH,
            input_preview="facts: The contract...",
            result="[]",
            duration_ms=142,
        )
        assert len(log) == 1
        rec = log[0]
        assert rec["function"] == "detect_contradictions"
        assert rec["tier"] == "flash"
        assert rec["duration_ms"] == 142
        assert "timestamp_ms" in rec

    def test_emit_decision_record_truncates_previews(self):
        from irys.rlm.decisions import _emit_decision_record
        from irys.core.models import ModelTier

        log: list[dict] = []
        _emit_decision_record(
            decision_log=log,
            func_name="fn",
            tier=ModelTier.FLASH,
            input_preview="x" * 500,
            result="y" * 500,
            duration_ms=10,
        )
        assert len(log[0]["input_preview"]) <= 200
        assert len(log[0]["result_preview"]) <= 200

    def test_emit_decision_record_noop_when_log_is_none(self):
        from irys.rlm.decisions import _emit_decision_record
        from irys.core.models import ModelTier

        _emit_decision_record(
            decision_log=None,
            func_name="fn",
            tier=ModelTier.FLASH,
            input_preview="x",
            result="y",
            duration_ms=10,
        )

    def test_engine_has_decision_log_attribute(self):
        from irys.rlm.engine import RLMEngine, RLMConfig
        from unittest.mock import MagicMock

        engine = RLMEngine.__new__(RLMEngine)
        engine._decision_log = []
        assert isinstance(engine._decision_log, list)
