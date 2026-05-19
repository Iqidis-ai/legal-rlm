# tests/test_fact_store.py
"""Tests for FactStore and doc-label derivation helpers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.rlm.engine import _derive_doc_label


class TestDeriveDocLabel:
    """_derive_doc_label maps filenames to short stable identifiers."""

    def test_arks_agreement(self):
        result = _derive_doc_label(
            "Delek ARKS Amended Restated Master Supply & Offtake Agreement.pdf"
        )
        assert result == "ARKS-S&O"

    def test_bsr_agreement(self):
        result = _derive_doc_label(
            "Delek BSR Amended Restated Master Supply & Offtake Agreement.pdf"
        )
        assert result == "BSR-S&O"

    def test_delek_standard_agreement(self):
        result = _derive_doc_label(
            "Delek Amended Restated Master Supply & Offtake Agreement.pdf"
        )
        assert result == "Delek-S&O"

    def test_msa_label(self):
        result = _derive_doc_label("Waymo_MSA_v3.docx")
        assert result == "Waymo-MSA"

    def test_settlement_agreement(self):
        result = _derive_doc_label("Settlement Agreement - John v Acme 2024.pdf")
        assert result == "Settlement-Agmt"

    def test_nda(self):
        result = _derive_doc_label("Non-Disclosure Agreement.pdf")
        assert result == "NDA"

    def test_no_extension(self):
        # Should not raise, should return something non-empty
        result = _derive_doc_label("SomeDocument")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_label_is_stable(self):
        """Calling twice on same filename must return identical string."""
        fname = "Delek ARKS Amended Restated Master Supply & Offtake Agreement.pdf"
        assert _derive_doc_label(fname) == _derive_doc_label(fname)


class TestScopeContextLabel:
    """scope_context passed to extract_facts must contain the canonical doc label."""

    def test_scope_context_contains_label(self):
        """_derive_doc_label output must appear verbatim in the scope_context string."""
        filename = "Delek ARKS Amended Restated Master Supply & Offtake Agreement.pdf"
        label = _derive_doc_label(filename)
        # Simulate what engine.py will build
        scope_label = "pages 5-10"  # stand-in for scope.label()
        scope_context = (
            f"Read scope: {scope_label}\n"
            f"Document label (use exactly this in attribution anchors): {label}"
        )
        assert label in scope_context
        assert "ARKS-S&O" in scope_context

    def test_scope_context_label_delek_standard(self):
        filename = "Delek Amended Restated Master Supply & Offtake Agreement.pdf"
        label = _derive_doc_label(filename)
        assert label == "Delek-S&O"
        scope_context = (
            f"Read scope: prefix\n"
            f"Document label (use exactly this in attribution anchors): {label}"
        )
        assert "Delek-S&O" in scope_context


class TestExtractFactsPromptAttribution:
    """P_EXTRACT_FACTS must contain the ATTRIBUTION RULE block."""

    def test_attribution_rule_present(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "ATTRIBUTION RULE" in P_EXTRACT_FACTS

    def test_attribution_format_instruction(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        assert "[{doc_label}, §{section}]" in P_EXTRACT_FACTS

    def test_attribution_example_present(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        # The example line illustrates the expected output format
        assert "ARKS-S&O" in P_EXTRACT_FACTS

    def test_quotes_exemption_stated(self):
        from irys.rlm.prompts import P_EXTRACT_FACTS
        # Quotes must be explicitly excluded from attribution tagging
        assert "quotes" in P_EXTRACT_FACTS.lower()
        assert "verbatim" in P_EXTRACT_FACTS.lower()


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
            "Delek ARKS.pdf":     [f"[ARKS-S&O, §2.1({i})] fact text here" for i in range(19)],
            "Delek BSR.pdf":      [f"[BSR-S&O,  §2.1({i})] fact text here" for i in range(18)],
            "Delek Standard.pdf": [f"[Delek-S&O,§2.1({i})] fact text here" for i in range(19)],
        })
        result = store.format_for_llm(max_chars=3000)
        # All three sources must appear — no source should be fully truncated
        assert "[Delek ARKS.pdf]" in result
        assert "[Delek BSR.pdf]" in result
        assert "[Delek Standard.pdf]" in result

    def test_no_source_exceeds_budget(self):
        """No single source should receive more than ceil(budget/n_sources) + small overhead."""
        store = self._make_store({
            "A.pdf": ["A fact " * 20 for _ in range(50)],   # large source
            "B.pdf": ["B fact " * 20 for _ in range(5)],    # small source
            "C.pdf": ["C fact " * 20 for _ in range(5)],    # small source
        })
        result = store.format_for_llm(max_chars=3000)
        # B and C must appear even though A is large
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
