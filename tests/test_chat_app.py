"""Tests for ChatApp file-handling helpers."""
from types import SimpleNamespace

import pytest

from irys.ui.chat_app import ChatApp


@pytest.fixture
def app() -> ChatApp:
    """A ChatApp instance is enough — these helpers don't touch session state."""
    return ChatApp(api_key="test-key")


def _gradio_file(name: str, orig_name: str | None = None, path: str | None = None):
    """Mimic the duck-typed file objects Gradio hands to upload callbacks.

    Real Gradio objects expose `.name` (on-disk path) and `.orig_name`
    (original filename, possibly with subdirectories under file_count="directory").
    """
    obj = SimpleNamespace(name=name)
    if orig_name is not None:
        obj.orig_name = orig_name
    if path is not None:
        obj.path = path
    return obj


class TestGetDisplayRelpath:
    def test_string_input_returns_leaf_name(self, app):
        assert app._get_display_relpath("/tmp/upload/report.pdf") == "report.pdf"

    def test_string_with_no_extension_returns_none(self, app):
        # No '.' in leaf -> can't trust it as a real filename
        assert app._get_display_relpath("/tmp/upload/abcdef") is None

    def test_string_hash_filename_returns_none(self, app):
        # 32+ hex chars with extension is treated as a Gradio temp hash
        hash_name = "/tmp/upload/" + "a" * 40 + ".pdf"
        assert app._get_display_relpath(hash_name) is None

    def test_orig_name_preserves_subdirectories(self, app):
        f = _gradio_file(name="/tmp/abc.pdf", orig_name="MatterA/Pleadings/answer.pdf")
        assert app._get_display_relpath(f) == "MatterA/Pleadings/answer.pdf"

    def test_orig_name_sanitizes_traversal(self, app):
        f = _gradio_file(name="/tmp/abc.pdf", orig_name="../escape/file.pdf")
        assert app._get_display_relpath(f) == "escape/file.pdf"

    def test_orig_name_strips_absolute_prefix(self, app):
        f = _gradio_file(name="/tmp/abc.pdf", orig_name="/abs/path/file.pdf")
        assert app._get_display_relpath(f) == "abs/path/file.pdf"

    def test_orig_name_with_backslashes_normalised(self, app):
        f = _gradio_file(name="/tmp/abc.pdf", orig_name=r"sub\nest\file.pdf")
        assert app._get_display_relpath(f) == "sub/nest/file.pdf"

    def test_hash_leaf_in_orig_name_falls_through_to_path(self, app):
        # orig_name leaf looks like a Gradio temp hash, but .path has a real name
        hash_orig = "a" * 40 + ".pdf"
        f = _gradio_file(
            name="/tmp/abc.pdf",
            orig_name=hash_orig,
            path="/tmp/uploads/real_file.pdf",
        )
        assert app._get_display_relpath(f) == "real_file.pdf"

    def test_no_orig_name_falls_back_to_actual_leaf(self, app):
        f = _gradio_file(name="/tmp/uploads/contract.docx")
        assert app._get_display_relpath(f) == "contract.docx"

    def test_no_useful_info_returns_none(self, app):
        # No orig_name, no extension on disk name, no .path -> can't recover
        f = _gradio_file(name="/tmp/uploads/" + "b" * 40)
        assert app._get_display_relpath(f) is None
