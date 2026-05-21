# tests/test_fact_store.py
"""Tests for FactStore helpers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestFormatForLlmProportionalBudget:
    """format_for_llm distributes budget proportionally across sources."""

    def _make_store(self, facts_by_source: dict) -> "FactStore":
        from irys.core.fact_store import FactStore, StoredFact
        import tempfile, os
        tmp = tempfile.mkdtemp()
        store = FactStore(Path(tmp))
        store._loaded = True
        store._facts = []
        for source, fact_texts in facts_by_source.items():
            for text in fact_texts:
                store._facts.append(StoredFact(fact=text, source=source))
        return store

    def test_single_source_gets_full_budget(self):
        """One source should receive the entire budget."""
        from irys.core.fact_store import FactStore
        store = self._make_store({"doc_a.pdf": [f"Fact {i} " * 10 for i in range(20)]})
        result = store.format_for_llm(max_chars=500)
        assert "[doc_a.pdf]" in result

    def test_three_sources_proportional(self):
        """With 3 equal sources each should appear in the output."""
        store = self._make_store({
            "Delek ARKS.pdf":     [f"fact text here {i}" for i in range(19)],
            "Delek BSR.pdf":      [f"fact text here {i}" for i in range(18)],
            "Delek Standard.pdf": [f"fact text here {i}" for i in range(19)],
        })
        result = store.format_for_llm(max_chars=3000)
        assert "[Delek ARKS.pdf]" in result
        assert "[Delek BSR.pdf]" in result
        assert "[Delek Standard.pdf]" in result

    def test_no_source_exceeds_budget(self):
        """No single source should crowd out smaller sources."""
        store = self._make_store({
            "A.pdf": ["A fact " * 20 for _ in range(50)],
            "B.pdf": ["B fact " * 20 for _ in range(5)],
            "C.pdf": ["C fact " * 20 for _ in range(5)],
        })
        result = store.format_for_llm(max_chars=3000)
        assert "[B.pdf]" in result
        assert "[C.pdf]" in result

    def test_empty_store_returns_empty(self):
        store = self._make_store({})
        assert store.format_for_llm() == ""

    def test_returns_header(self):
        store = self._make_store({"doc.pdf": ["A fact about something."]})
        result = store.format_for_llm()
        assert "CACHED FACTS" in result

    def test_single_fact_included(self):
        store = self._make_store({"doc.pdf": ["The contract value is $2.5M."]})
        result = store.format_for_llm()
        assert "The contract value is $2.5M." in result


class TestFactCompletenessInstruction:
    """P_EXTRACT_FACTS must contain the FACT COMPLETENESS instruction block.
    P_ANALYZE_RESULTS must NOT — it works from search snippets and cannot see surrounding lines.
    """

    def test_fact_completeness_present_in_extract_facts(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "FACT COMPLETENESS" in P_EXTRACT_FACTS

    def test_surrounding_sentence_instruction_present(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "sentence immediately before" in P_EXTRACT_FACTS
        assert "sentence immediately after" in P_EXTRACT_FACTS

    def test_three_sentence_cap_present(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "3 sentences" in P_EXTRACT_FACTS

    def test_quotes_exemption_present(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert 'Do NOT apply this to "quotes"' in P_EXTRACT_FACTS

    def test_fact_completeness_absent_from_analyze_results(self):
        """Search snippet path must NOT get this instruction — model cannot see surrounding lines."""
        from irys.rlm.prompts import P_ANALYZE_RESULTS
        assert "FACT COMPLETENESS" not in P_ANALYZE_RESULTS
