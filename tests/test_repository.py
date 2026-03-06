"""Tests for MatterRepository."""

import pytest
from pathlib import Path
import tempfile
import os

from irys.core.repository import MatterRepository
from irys.core.reader import DocumentReader


class TestDocumentReader:
    """Tests for DocumentReader."""

    def test_read_txt(self, tmp_path):
        """Test reading plain text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is a test document.\nWith multiple lines.")

        reader = DocumentReader()
        doc = reader.read(txt_file)

        assert doc.filename == "test.txt"
        assert doc.file_type == "txt"
        assert "test document" in doc.full_text
        assert doc.page_count == 1

    def test_can_read(self):
        """Test file type detection."""
        reader = DocumentReader()

        assert reader.can_read(Path("test.pdf"))
        assert reader.can_read(Path("test.docx"))
        assert reader.can_read(Path("test.txt"))
        assert not reader.can_read(Path("test.xlsx"))
        assert not reader.can_read(Path("test.jpg"))


class TestMatterRepository:
    """Tests for MatterRepository."""

    def test_list_files(self, tmp_path):
        """Test listing files in repository."""
        # Create test structure
        (tmp_path / "contracts").mkdir()
        (tmp_path / "contracts" / "agreement.txt").write_text("Agreement content")
        (tmp_path / "memos").mkdir()
        (tmp_path / "memos" / "memo1.txt").write_text("Memo content")

        repo = MatterRepository(tmp_path)
        files = repo.list_files()

        assert len(files) == 2
        filenames = [f.filename for f in files]
        assert "agreement.txt" in filenames
        assert "memo1.txt" in filenames

    def test_get_structure(self, tmp_path):
        """Test getting folder structure."""
        (tmp_path / "contracts").mkdir()
        (tmp_path / "contracts" / "a.txt").write_text("a")
        (tmp_path / "contracts" / "b.txt").write_text("b")
        (tmp_path / "memos").mkdir()
        (tmp_path / "memos" / "c.txt").write_text("c")

        repo = MatterRepository(tmp_path)
        structure = repo.get_structure()

        assert structure["contracts"] == 2
        assert structure["memos"] == 1

    def test_search(self, tmp_path):
        """Test searching across documents."""
        (tmp_path / "doc1.txt").write_text("The quick brown fox jumps over the lazy dog.")
        (tmp_path / "doc2.txt").write_text("A lazy cat sleeps all day.")

        repo = MatterRepository(tmp_path)
        results = repo.search("lazy")

        assert results.total_matches == 2
        assert results.files_searched == 2

    def test_read(self, tmp_path):
        """Test reading a document."""
        (tmp_path / "test.txt").write_text("Test content here.")

        repo = MatterRepository(tmp_path)
        doc = repo.read("test.txt")

        assert "Test content" in doc.full_text

    def test_get_stats(self, tmp_path):
        """Test getting repository stats."""
        (tmp_path / "a.txt").write_text("content")
        (tmp_path / "b.txt").write_text("more content")

        repo = MatterRepository(tmp_path)
        stats = repo.get_stats()

        assert stats.total_files == 2
        assert ".txt" in stats.files_by_type

    def test_filename_mapping_preserves_url_and_mime(self, tmp_path):
        """Test filename mapping metadata lookup for URL-backed documents."""
        (tmp_path / "file_a.pdf").write_text("a")
        (tmp_path / "file_b.pdf").write_text("b")
        (tmp_path / "_filename_mapping.json").write_text(
            '{\n'
            '  "file_a.pdf": {"display_name": "same-name.pdf", "url": "https://example.com/a", "mime": "application/pdf"},\n'
            '  "file_b.pdf": {"display_name": "same-name_1.pdf", "url": "https://example.com/b", "mime": "application/pdf"}\n'
            '}'
        )

        repo = MatterRepository(tmp_path)

        files = repo.list_files()
        filenames = [f.filename for f in files]

        assert "same-name.pdf" in filenames
        assert "same-name_1.pdf" in filenames
        assert repo.get_document_url("same-name.pdf") == "https://example.com/a"
        assert repo.get_document_url("file_b.pdf") == "https://example.com/b"
        assert repo.get_document_mime("same-name_1.pdf") == "application/pdf"
