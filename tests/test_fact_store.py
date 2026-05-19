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
