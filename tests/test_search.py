"""Tests for DocumentSearch."""

import pytest
from pathlib import Path

from irys.core.search import DocumentSearch, SearchHit, SearchResults
from irys.core.reader import DocumentReader


class TestDocumentSearch:
    """Tests for DocumentSearch."""

    def test_search_single_file(self, tmp_path):
        """Test searching a single file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line one\nLine two with keyword\nLine three")

        search = DocumentSearch()
        results = search.search("keyword", [test_file])

        assert results.total_matches == 1
        assert results.hits[0].match_text == "Line two with keyword"

    def test_search_multiple_files(self, tmp_path):
        """Test searching multiple files."""
        (tmp_path / "a.txt").write_text("Contract agreement terms")
        (tmp_path / "b.txt").write_text("No matches here")
        (tmp_path / "c.txt").write_text("Another agreement document")

        files = list(tmp_path.glob("*.txt"))
        search = DocumentSearch()
        results = search.search("agreement", files)

        assert results.total_matches == 2

    def test_search_case_insensitive(self, tmp_path):
        """Test case insensitive search."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("CONTRACT terms\ncontract details\nCONTRACT end")

        search = DocumentSearch()
        results = search.search("contract", [test_file], case_sensitive=False)

        assert results.total_matches == 3

    def test_search_case_sensitive(self, tmp_path):
        """Test case sensitive search."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("CONTRACT terms\ncontract details\nCONTRACT end")

        search = DocumentSearch()
        results = search.search("contract", [test_file], case_sensitive=True)

        assert results.total_matches == 1

    def test_search_regex(self, tmp_path):
        """Test regex search."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("$1,000 damages\n$50,000 settlement\nno amount here")

        search = DocumentSearch()
        results = search.search(r"\$[\d,]+", [test_file], regex=True)

        assert results.total_matches == 2

    def test_search_context(self, tmp_path):
        """Test context lines around match."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nMATCH here\nLine 4\nLine 5")

        search = DocumentSearch()
        results = search.search("MATCH", [test_file], context_lines=2)

        hit = results.hits[0]
        assert "Line 2" in hit.context_before
        assert "Line 4" in hit.context_after

    def test_search_multi_terms(self, tmp_path):
        """Test multi-term search."""
        (tmp_path / "a.txt").write_text("breach of contract")
        (tmp_path / "b.txt").write_text("contract signed")
        (tmp_path / "c.txt").write_text("breach occurred")

        files = list(tmp_path.glob("*.txt"))
        search = DocumentSearch()

        # OR search
        results = search.search_multi(["breach", "contract"], files, require_all=False)
        assert len(results.hits) >= 3

        # AND search (files containing both)
        results = search.search_multi(["breach", "contract"], files, require_all=True)
        assert any("a.txt" in h.file_path for h in results.hits)


class TestSearchHit:
    """Tests for SearchHit."""

    def test_citation_format(self):
        """Test citation string format."""
        hit = SearchHit(
            file_path="/path/to/doc.pdf",
            filename="doc.pdf",
            page_num=5,
            line_num=10,
            match_text="Some matched text",
        )

        assert hit.citation == "doc.pdf, p. 5"

    def test_context_format(self):
        """Test context formatting."""
        hit = SearchHit(
            file_path="/path/to/doc.pdf",
            filename="doc.pdf",
            page_num=1,
            line_num=5,
            match_text="The matched line",
            context_before=["Line before"],
            context_after=["Line after"],
        )

        context = hit.context
        assert "Line before" in context
        assert ">>> The matched line" in context
        assert "Line after" in context
