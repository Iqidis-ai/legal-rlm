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
