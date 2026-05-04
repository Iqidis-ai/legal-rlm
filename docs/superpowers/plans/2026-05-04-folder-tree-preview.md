# Folder Tree Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a read-only Markdown tree preview of the uploaded folder hierarchy beneath the "Upload Folder" tab in the chat app, so users can confirm nested folders were captured.

**Architecture:** Two changes inside `src/irys/ui/chat_app.py` plus a new test file. (1) Lift the nested `get_original_relpath` inside `_extract_files_from_upload` into a method `_get_display_relpath(file_obj) -> str | None` so the tree-builder can reuse the same `orig_name`/`_sanitize_relpath` pipeline. (2) Add a method `_build_folder_tree_markdown(uploaded_files) -> str` that groups paths into a sorted tree (folders before files at each level) and returns indented Markdown. (3) Add a `gr.Markdown` component inside the "Upload Folder" tab and wire `folder_upload.change` to update it.

**Tech Stack:** Python 3.12, Gradio 6.x, pytest. Tests use `types.SimpleNamespace` for fake Gradio file objects (matches existing `tests/test_usage_stats.py` style).

Spec: `docs/superpowers/specs/2026-05-04-folder-tree-preview-design.md`.

---

## File Structure

- **Modify** `src/irys/ui/chat_app.py`:
  - Lines 261-380 (`_extract_files_from_upload`): lift `get_original_relpath` into a class method `_get_display_relpath`; update call sites.
  - Add new method `_build_folder_tree_markdown` directly after `_extract_files_from_upload`.
  - Lines 922-931 (Upload Folder TabItem): add new `gr.Markdown` component.
  - Inside `create_app` after both folder components are defined: add `folder_upload.change(...)` wiring.
- **Create** `tests/test_chat_app.py`: covers `_get_display_relpath` and `_build_folder_tree_markdown`.

---

## Task 1: Extract `_get_display_relpath` helper with tests

**Files:**
- Create: `tests/test_chat_app.py`
- Modify: `src/irys/ui/chat_app.py:261-380`

This is a no-behavior-change refactor: the lifted method returns the same sanitized display relpath that `get_original_relpath` returns today (its first tuple element). The existing `_extract_files_from_upload` keeps its own `actual_name` computation inline so the new helper stays focused on the relpath only.

- [ ] **Step 1: Create the test file with fake-Gradio fixture and failing tests**

Create `tests/test_chat_app.py` with this content:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail with `AttributeError`**

Run: `cd /home/ubuntu/legal-rlm && python3 -m pytest tests/test_chat_app.py -v`
Expected: All `TestGetDisplayRelpath` tests fail with `AttributeError: 'ChatApp' object has no attribute '_get_display_relpath'`.

- [ ] **Step 3: Add `_get_display_relpath` as a method on ChatApp**

Open `src/irys/ui/chat_app.py`. Immediately *before* the `_extract_files_from_upload` method (which currently starts at line 261), insert this new method. Keep it `_get_display_relpath` so it's clearly internal.

```python
    @staticmethod
    def _is_hash_filename(name: str) -> bool:
        base = Path(name).stem
        return len(base) >= 32 and all(c in '0123456789abcdef' for c in base.lower())

    @staticmethod
    def _sanitize_relpath(path: str) -> str:
        """Drop empty / `.` / `..` / absolute segments from a relpath."""
        parts: list[str] = []
        for raw in path.replace('\\', '/').split('/'):
            part = raw.strip()
            if not part or part in ('.', '..'):
                continue
            parts.append(part)
        return '/'.join(parts)

    def _get_display_relpath(self, file_obj) -> Optional[str]:
        """Return the sanitized display relpath for a Gradio upload object.

        Preserves any subdirectory structure Gradio supplies in ``orig_name``
        (the case under ``file_count='directory'``). Returns ``None`` when
        we can't derive a usable filename — callers should fall back to a
        generated name (e.g. ``document_{idx}.{ext}``).
        """
        if isinstance(file_obj, str):
            actual_name = Path(file_obj).name
            if '.' in actual_name and not self._is_hash_filename(actual_name):
                return actual_name
            return None

        orig = getattr(file_obj, 'orig_name', None)
        if orig:
            relpath = self._sanitize_relpath(str(orig))
            leaf = Path(relpath).name if relpath else ''
            if leaf and not self._is_hash_filename(leaf):
                return relpath or leaf

        if hasattr(file_obj, 'path') and file_obj.path:
            path_name = Path(file_obj.path).name
            if '.' in path_name and not self._is_hash_filename(path_name):
                return path_name

        actual_name = (
            Path(file_obj.name).name if hasattr(file_obj, 'name') else str(file_obj)
        )
        if '.' in actual_name and not self._is_hash_filename(actual_name):
            return actual_name

        return None
```

Note: `Path` and `Optional` are already imported at the top of `chat_app.py` — verify with `grep -n "^from pathlib\|^from typing" src/irys/ui/chat_app.py` before editing.

- [ ] **Step 4: Run the new tests to confirm they pass**

Run: `cd /home/ubuntu/legal-rlm && python3 -m pytest tests/test_chat_app.py::TestGetDisplayRelpath -v`
Expected: All 10 tests pass.

- [ ] **Step 5: Update `_extract_files_from_upload` to call the new helper**

In `src/irys/ui/chat_app.py`, replace the body of `_extract_files_from_upload` (currently lines 261-380) so that it delegates path-extraction to `_get_display_relpath` and computes `actual_name` inline. The `_detect_extension_from_path` nested helper stays nested (it's only used here).

Replace the existing method (lines 261-380) with:

```python
    def _extract_files_from_upload(
        self,
        uploaded_files: list,
    ) -> tuple[list[tuple[str, str, str]], str | None]:
        """Extract files from a Gradio upload, preserving folder structure.

        Returns paths (not bytes) so downstream consumers can stream from
        disk. The original ``orig_name`` from Gradio is treated as a possibly
        nested relative path (e.g. ``subfolder/file.pdf``); we sanitize it to
        drop traversal segments but otherwise preserve subdirectory layout.

        Returns:
            Tuple of (list of (display_name, source_path, actual_filename), error or None)
        """
        if not uploaded_files:
            return [], None

        def _detect_extension_from_path(path: Path) -> str:
            """Sniff the first KB of the file to guess an extension."""
            try:
                with open(path, 'rb') as fh:
                    head = fh.read(1024)
            except Exception:
                return ''
            if head.startswith(b'%PDF'):
                return '.pdf'
            if head.startswith(b'PK\x03\x04'):
                return '.docx'
            if head.startswith(b'\xd0\xcf\x11\xe0'):
                return '.doc'
            if head.startswith(b'{\\rtf'):
                return '.rtf'
            try:
                head.decode('utf-8')
                return '.txt'
            except UnicodeDecodeError:
                return ''

        files: list[tuple[str, str, str]] = []
        for idx, f in enumerate(uploaded_files):
            try:
                source_path = (
                    Path(f.name) if hasattr(f, 'name') else Path(str(f))
                )
                if isinstance(f, str):
                    actual_name = Path(f).name
                else:
                    actual_name = (
                        Path(f.name).name if hasattr(f, 'name') else str(f)
                    )

                orig_relpath = self._get_display_relpath(f)

                if orig_relpath:
                    display_relpath = orig_relpath
                else:
                    ext = _detect_extension_from_path(source_path)
                    display_relpath = (
                        f"document_{idx + 1}{ext}" if ext else f"document_{idx + 1}"
                    )
                    logger.warning(
                        f"Could not get original filename for {actual_name}, "
                        f"using generated name: {display_relpath}"
                    )

                if any(part.startswith('.') for part in Path(display_relpath).parts):
                    logger.debug(f"Skipping hidden file: {display_relpath}")
                    continue

                files.append((display_relpath, str(source_path), actual_name))
                logger.debug(
                    f"Extracted file: display={display_relpath}, actual={actual_name}"
                )
            except Exception as e:
                logger.error(f"Error extracting file at index {idx}: {e}")
                continue

        if not files:
            return [], "No valid files could be extracted from upload"

        return files, None
```

Note: the `try`/`except` around the loop body and the final `if not files` check at the end of the original method are preserved — verify by reading lines 380-420 of the current file to confirm the exact tail of the existing method, and keep behavior identical there.

- [ ] **Step 6: Run the full test suite to confirm no regression**

Run: `cd /home/ubuntu/legal-rlm && python3 -m pytest tests/ -v`
Expected: All tests (existing + new) pass. If anything fails in `tests/test_usage_stats.py`, `tests/matter/`, or `tests/service/`, the refactor introduced a bug — re-read the tail of the original `_extract_files_from_upload` and reconcile.

- [ ] **Step 7: Commit**

```bash
git add tests/test_chat_app.py src/irys/ui/chat_app.py
git commit -m "$(cat <<'EOF'
Extract _get_display_relpath helper from upload extraction

Committed by Devansh
EOF
)"
```

---

## Task 2: Implement `_build_folder_tree_markdown` (TDD)

**Files:**
- Modify: `src/irys/ui/chat_app.py` (add method directly after `_extract_files_from_upload`)
- Modify: `tests/test_chat_app.py` (append new test class)

- [ ] **Step 1: Append failing tests for the tree builder**

Append this to the end of `tests/test_chat_app.py`:

```python
class TestBuildFolderTreeMarkdown:
    def test_empty_input_returns_empty_string(self, app):
        assert app._build_folder_tree_markdown([]) == ""
        assert app._build_folder_tree_markdown(None) == ""

    def test_flat_files_only_no_headers(self, app):
        files = [
            _gradio_file(name="/tmp/a.pdf", orig_name="b.pdf"),
            _gradio_file(name="/tmp/c.pdf", orig_name="a.pdf"),
        ]
        out = app._build_folder_tree_markdown(files)
        # Sorted alphabetical, no folder headers, no "(root)" header
        assert out == "- a.pdf\n- b.pdf"

    def test_single_nested_folder(self, app):
        files = [
            _gradio_file(name="/tmp/x.pdf", orig_name="MatterA/answer.pdf"),
            _gradio_file(name="/tmp/y.pdf", orig_name="MatterA/complaint.pdf"),
        ]
        out = app._build_folder_tree_markdown(files)
        assert out == (
            "**📁 MatterA** (2 files)\n"
            "  - answer.pdf\n"
            "  - complaint.pdf"
        )

    def test_multiple_top_level_folders_sorted_with_counts(self, app):
        files = [
            _gradio_file(name="/tmp/1", orig_name="MatterB/notes.txt"),
            _gradio_file(name="/tmp/2", orig_name="MatterA/p1.pdf"),
            _gradio_file(name="/tmp/3", orig_name="MatterA/p2.pdf"),
        ]
        out = app._build_folder_tree_markdown(files)
        # MatterA before MatterB (case-insensitive alphabetical), counts correct
        assert out == (
            "**📁 MatterA** (2 files)\n"
            "  - p1.pdf\n"
            "  - p2.pdf\n"
            "**📁 MatterB** (1 file)\n"
            "  - notes.txt"
        )

    def test_deep_nesting_indents_correctly(self, app):
        files = [
            _gradio_file(name="/tmp/1", orig_name="A/B/C/deep.pdf"),
            _gradio_file(name="/tmp/2", orig_name="A/B/mid.pdf"),
            _gradio_file(name="/tmp/3", orig_name="A/top.pdf"),
        ]
        out = app._build_folder_tree_markdown(files)
        # Folders before files at each level; indent is two spaces per depth.
        # Depth 1 (under top-level header) starts at indent "  ".
        assert out == (
            "**📁 A** (3 files)\n"
            "  - 📁 B\n"
            "    - 📁 C\n"
            "      - deep.pdf\n"
            "    - mid.pdf\n"
            "  - top.pdf"
        )

    def test_hash_named_files_grouped_under_unnamed(self, app):
        # orig_name leaf is a hash AND no path/name fallback gives a real name
        hash_orig = "f" * 40 + ".pdf"
        files = [
            _gradio_file(name="/tmp/" + "f" * 40, orig_name=hash_orig),
            _gradio_file(name="/tmp/x.pdf", orig_name="real/file.pdf"),
        ]
        out = app._build_folder_tree_markdown(files)
        assert out == (
            "**📁 real** (1 file)\n"
            "  - file.pdf\n"
            "**📄 (unnamed)** (1 file)"
        )

    def test_traversal_and_absolute_paths_sanitized(self, app):
        # `..` segments are dropped entirely; absolute paths just lose the
        # leading slash (the rest of the path is preserved as a relpath).
        # Both files here share the "escape" parent after `..` is dropped.
        files = [
            _gradio_file(name="/tmp/1", orig_name="../escape/a.pdf"),
            _gradio_file(name="/tmp/2", orig_name="../escape/b.pdf"),
        ]
        out = app._build_folder_tree_markdown(files)
        assert out == (
            "**📁 escape** (2 files)\n"
            "  - a.pdf\n"
            "  - b.pdf"
        )

    def test_mixed_root_and_nested_uses_root_header(self, app):
        files = [
            _gradio_file(name="/tmp/1", orig_name="readme.txt"),
            _gradio_file(name="/tmp/2", orig_name="MatterA/p1.pdf"),
        ]
        out = app._build_folder_tree_markdown(files)
        # Root files are headed with "(root)" because nested folders also exist;
        # root section comes before folder sections.
        assert out == (
            "**📄 (root)** (1 file)\n"
            "- readme.txt\n"
            "**📁 MatterA** (1 file)\n"
            "  - p1.pdf"
        )
```

- [ ] **Step 2: Run tests to confirm they all fail**

Run: `cd /home/ubuntu/legal-rlm && python3 -m pytest tests/test_chat_app.py::TestBuildFolderTreeMarkdown -v`
Expected: All 8 tests fail with `AttributeError: 'ChatApp' object has no attribute '_build_folder_tree_markdown'`.

- [ ] **Step 3: Implement `_build_folder_tree_markdown`**

In `src/irys/ui/chat_app.py`, insert this method *immediately after* the closing of `_extract_files_from_upload`:

```python
    def _build_folder_tree_markdown(self, uploaded_files: list) -> str:
        """Render an indented Markdown tree of uploaded folder contents.

        Reuses ``_get_display_relpath`` so the preview matches exactly what
        ``_extract_files_from_upload`` would emit. Folders are sorted before
        files at each level (case-insensitive alphabetical). Files whose
        relpath cannot be derived are surfaced under an ``(unnamed)`` group
        so the user can still see they were captured.
        """
        if not uploaded_files:
            return ""

        relpaths: list[str] = []
        unnamed_count = 0
        for f in uploaded_files:
            rp = self._get_display_relpath(f)
            if rp:
                relpaths.append(rp)
            else:
                unnamed_count += 1

        # Partition top-level: bare files vs first-folder buckets
        root_files: list[str] = []
        folders: dict[str, list[list[str]]] = {}
        for rp in relpaths:
            parts = rp.split('/')
            if len(parts) == 1:
                root_files.append(parts[0])
            else:
                folders.setdefault(parts[0], []).append(parts[1:])

        has_nesting = bool(folders) or unnamed_count > 0

        def _plural(n: int) -> str:
            return "file" if n == 1 else "files"

        def _count_files(parts_list: list[list[str]]) -> int:
            return len(parts_list)

        def _render_subtree(
            parts_list: list[list[str]],
            indent: int,
            out: list[str],
        ) -> None:
            indent_str = "  " * indent
            sub_files: list[str] = []
            sub_folders: dict[str, list[list[str]]] = {}
            for parts in parts_list:
                if len(parts) == 1:
                    sub_files.append(parts[0])
                else:
                    sub_folders.setdefault(parts[0], []).append(parts[1:])
            for name in sorted(sub_folders.keys(), key=str.lower):
                out.append(f"{indent_str}- 📁 {name}")
                _render_subtree(sub_folders[name], indent + 1, out)
            for name in sorted(sub_files, key=str.lower):
                out.append(f"{indent_str}- {name}")

        lines: list[str] = []

        if not has_nesting:
            for name in sorted(root_files, key=str.lower):
                lines.append(f"- {name}")
            return "\n".join(lines)

        if root_files:
            lines.append(
                f"**📄 (root)** ({len(root_files)} {_plural(len(root_files))})"
            )
            for name in sorted(root_files, key=str.lower):
                lines.append(f"- {name}")

        for folder_name in sorted(folders.keys(), key=str.lower):
            sub_paths = folders[folder_name]
            count = _count_files(sub_paths)
            lines.append(
                f"**📁 {folder_name}** ({count} {_plural(count)})"
            )
            _render_subtree(sub_paths, indent=1, out=lines)

        if unnamed_count > 0:
            lines.append(
                f"**📄 (unnamed)** ({unnamed_count} {_plural(unnamed_count)})"
            )

        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `cd /home/ubuntu/legal-rlm && python3 -m pytest tests/test_chat_app.py -v`
Expected: All 18 tests pass (10 from Task 1 + 8 new).

- [ ] **Step 5: Run full suite to confirm no regression**

Run: `cd /home/ubuntu/legal-rlm && python3 -m pytest tests/ -v`
Expected: full suite green.

- [ ] **Step 6: Commit**

```bash
git add tests/test_chat_app.py src/irys/ui/chat_app.py
git commit -m "$(cat <<'EOF'
Add folder tree markdown builder for upload preview

Committed by Devansh
EOF
)"
```

---

## Task 3: Wire UI component and `.change` event

**Files:**
- Modify: `src/irys/ui/chat_app.py:922-931` (Upload Folder TabItem)
- Modify: `src/irys/ui/chat_app.py` (event wiring inside `create_app`, after both folder components and `app` exist)

This task has no automated tests — Gradio components don't have a clean unit-testing path here. The verification step is manual: launch the app, upload a nested folder, and confirm the tree appears.

- [ ] **Step 1: Add the `gr.Markdown` preview component inside the Upload Folder tab**

Find the existing block at `src/irys/ui/chat_app.py:922-931`:

```python
                        with gr.TabItem("Upload Folder"):
                            folder_upload = gr.File(
                                label="Upload Document Folder",
                                file_count="directory",
                                type="filepath",
                            )
                            gr.Markdown(
                                "*Select a folder to upload all documents including subfolders. "
                                "Folder structure will be preserved.*",
                            )
```

Replace it with:

```python
                        with gr.TabItem("Upload Folder"):
                            folder_upload = gr.File(
                                label="Upload Document Folder",
                                file_count="directory",
                                type="filepath",
                            )
                            gr.Markdown(
                                "*Select a folder to upload all documents including subfolders. "
                                "Folder structure will be preserved.*",
                            )
                            folder_tree_md = gr.Markdown(
                                value="",
                                label="Folder Structure",
                            )
```

- [ ] **Step 2: Wire the `.change` event**

The wiring must go *inside* `create_app`, after `folder_upload` and `folder_tree_md` are both defined and after `app` (the `ChatApp` instance) is in scope. Find an existing `.change` / `.click` block on `folder_upload` or other components in `create_app` to confirm the right block — typically near where `submit_btn.click(...)` and similar handlers are wired.

Search for an anchor:

```bash
grep -n "submit_btn.click\|folder_upload\.\|file_upload\." src/irys/ui/chat_app.py
```

Add this line in that wiring section (immediately after the `folder_upload` / `folder_tree_md` are both in scope — i.e. after the `with gr.Blocks(...)` body has finished defining components but before `return demo`):

```python
            folder_upload.change(
                fn=app._build_folder_tree_markdown,
                inputs=folder_upload,
                outputs=folder_tree_md,
            )
```

Indentation: match the indent of the surrounding `submit_btn.click(...)` / similar event-wiring lines in the same scope.

- [ ] **Step 3: Smoke-test the change is syntactically valid**

Run: `cd /home/ubuntu/legal-rlm && python3 -c "from irys.ui.chat_app import create_app; print('import OK')"`
Expected: `import OK`. Any syntax/indentation error surfaces here.

- [ ] **Step 4: Manual UI verification**

The smoke test only proves it imports. Functional verification needs the running app. From the project memory, the local server start command is in `memory/project_servers.md` — read that file (`/home/ubuntu/.claude/projects/-home-ubuntu-legal-rlm/memory/project_servers.md`) for the correct command, or fall back to:

```bash
cd /home/ubuntu/legal-rlm && python3 -m irys.ui.chat_app
```

In the browser:
1. Open the chat app and click the **Upload Folder** tab.
2. Select a folder containing at least one nested subfolder (e.g. `Root/SubA/file1.pdf`, `Root/SubB/file2.pdf`).
3. Confirm the new Markdown area below the picker renders a tree like:
   ```
   📁 Root (2 files)
     - 📁 SubA
       - file1.pdf
     - 📁 SubB
       - file2.pdf
   ```
4. Clear the picker and confirm the tree clears with it.
5. Switch to the **Upload Files** tab and confirm nothing changed there.

If the tree renders but folders are unsorted or counts are wrong, that's a bug in `_build_folder_tree_markdown` — fix in Task 2 territory. If the tree never renders, the `.change` wiring isn't hit — re-check the indent and that `folder_upload` and `folder_tree_md` are in the same Gradio context.

- [ ] **Step 5: Commit**

```bash
git add src/irys/ui/chat_app.py
git commit -m "$(cat <<'EOF'
Render folder tree preview under Upload Folder tab

Committed by Devansh
EOF
)"
```

---

## Done criteria

- [ ] All 18 unit tests in `tests/test_chat_app.py` pass.
- [ ] Full test suite (`python3 -m pytest tests/ -v`) is green.
- [ ] Manual UI test (Task 3, Step 4) shows the tree updating live as the folder selection changes.
- [ ] Three commits on `SebihSpecial` matching the per-task messages above.
