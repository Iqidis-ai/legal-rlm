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


class TestBuildFolderTreeMarkdown:
    def test_empty_list_returns_empty(self, app):
        assert app._build_folder_tree_markdown([]) == ""

    def test_none_returns_empty(self, app):
        assert app._build_folder_tree_markdown(None) == ""

    def test_flat_files_only(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="report.pdf"),
            _gradio_file(name="/tmp/b.pdf", orig_name="contract.docx"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "- contract.docx" in result
        assert "- report.pdf" in result
        assert "**" not in result

    def test_single_nested_folder(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="Pleadings/answer.pdf"),
            _gradio_file(name="/tmp/b.pdf", orig_name="Pleadings/complaint.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "**Pleadings/**" in result
        assert "2 files" in result
        assert "  - answer.pdf" in result
        assert "  - complaint.pdf" in result

    def test_multiple_top_level_folders_sorted(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="Zebra/z.pdf"),
            _gradio_file(name="/tmp/b.pdf", orig_name="Alpha/a.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        lines = result.splitlines()
        alpha_idx = next(i for i, l in enumerate(lines) if "Alpha" in l)
        zebra_idx = next(i for i, l in enumerate(lines) if "Zebra" in l)
        assert alpha_idx < zebra_idx

    def test_deep_nesting(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="L1/L2/L3/deep.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "**L1/**" in result
        assert "  - **L2/**" in result
        assert "    - **L3/**" in result
        assert "      - deep.pdf" in result

    def test_unnamed_files_grouped(self, app):
        files = [
            _gradio_file(name="/tmp/" + "f" * 40),
            _gradio_file(name="/tmp/a.pdf", orig_name="real.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "(unnamed)" in result
        assert "1 file)" in result
        assert "- real.pdf" in result

    def test_traversal_segments_sanitized(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="../escape/../../file.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert ".." not in result
        assert "file.pdf" in result

    def test_mixed_root_and_nested(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="root_file.pdf"),
            _gradio_file(name="/tmp/b.pdf", orig_name="Folder/nested.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "(root)" in result
        assert "root_file.pdf" in result
        assert "Folder" in result
        assert "nested.pdf" in result

    def test_folder_file_count_accurate(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="Docs/a.pdf"),
            _gradio_file(name="/tmp/b.pdf", orig_name="Docs/b.pdf"),
            _gradio_file(name="/tmp/c.pdf", orig_name="Docs/sub/c.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "3 files" in result

    def test_markdown_link_injection_escaped(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="[malicious](javascript:alert(1)).pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "\\[" in result
        assert "\\(" in result
        assert "[malicious](" not in result

    def test_markdown_image_injection_escaped(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="![steal](http://evil.com).pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "\\!" in result
        assert "\\[" in result

    def test_markdown_link_in_folder_name_escaped(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="[click](evil)/file.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "\\[click\\]\\(evil\\)" in result

    def test_unicode_filenames(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="文件夹/文件.pdf"),
        ]
        result = app._build_folder_tree_markdown(files)
        assert "文件.pdf" in result
        assert "文件夹" in result

    def test_deep_nesting_does_not_crash(self, app):
        deep_path = "/".join([f"d{i}" for i in range(100)]) + "/file.pdf"
        files = [_gradio_file(name="/tmp/a.pdf", orig_name=deep_path)]
        result = app._build_folder_tree_markdown(files)
        assert "truncated" in result
        assert "file.pdf" not in result
