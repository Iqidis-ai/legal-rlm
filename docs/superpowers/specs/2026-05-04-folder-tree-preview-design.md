# Folder Tree Preview for Upload Folder Tab

## Problem

The chat app's "Upload Folder" tab uses Gradio's `gr.File(file_count="directory")` to capture nested folders. The relative paths from `orig_name` are preserved in the data layer (`_extract_files_from_upload` at `src/irys/ui/chat_app.py:261`) and round-trip correctly to S3. But Gradio's built-in file widget renders only a flat list of leaf filenames, so the user has no visual confirmation that the folder structure was captured. For 400-file matter uploads this is disorienting — users can't tell whether subfolders made it.

## Goal

Show a read-only tree preview of the uploaded folder hierarchy inside the "Upload Folder" tab, immediately below the existing `folder_upload` widget. The preview updates whenever the file selection changes.

Non-goals: changing the picker widget itself, adding interactivity (collapse/expand, file removal), or touching the "Upload Files" tab (which has no nesting to display).

## Design

### UI placement

A new `gr.Markdown` component (`folder_tree_md`) is added inside the "Upload Folder" `TabItem` at `chat_app.py:922-931`, directly after `folder_upload` and the existing helper Markdown. Initial value is the empty string — Markdown renders blank, so no separate hide logic is needed.

### Rendering

The tree is rendered as a Markdown indented bullet list:

```
**📁 MatterA** (12 files)
  - 📁 Pleadings
    - answer.pdf
    - complaint.pdf
  - 📁 Discovery
    - interrogatories.docx
**📁 MatterB** (3 files)
  - notes.txt
```

Rules:
- Folders sorted before files within each level; alphabetical (case-insensitive) within each group.
- Each top-level folder gets a bold header line with its file count.
- Root-level files (no parent folder): if any nested folders also exist, root files are grouped under a `**📄 (root)** (N files)` header before the folder headers; if there are no nested folders at all, the output is just a flat bullet list with no header.
- Two-space indent per level.

### Helper function

A new private helper alongside `_extract_files_from_upload`:

```python
def _build_folder_tree_markdown(self, uploaded_files: list) -> str:
    """Render an indented Markdown tree of uploaded folder contents.

    Reuses orig_name extraction + _sanitize_relpath logic from
    _extract_files_from_upload, then groups paths into a tree and
    renders folders-before-files at each level.
    """
```

It must reuse the same path-extraction and sanitization logic that `_extract_files_from_upload` uses, so the tree reflects exactly what will be uploaded (hash-named fallbacks included). The cleanest way is to extract a small shared helper `_get_display_relpath(file_obj) -> str | None` that both functions call, returning the sanitized display relpath or `None` if it can't be determined. Files where the helper returns `None` are listed under a synthetic `(unnamed)` group at the bottom so the user still sees they were captured.

Empty input returns the empty string.

### Event wiring

In `create_app`, after both components exist:

```python
folder_upload.change(
    fn=app._build_folder_tree_markdown,
    inputs=folder_upload,
    outputs=folder_tree_md,
)
```

This fires on every selection change — including when the user clears the picker — so the tree always matches the current selection. Pure function on a small input list, so no perf concern at the 400-file scale already supported.

### Scope guard

Only the folder tab is modified. The "Upload Files" tab is left untouched. No changes to `_extract_files_from_upload`, S3 upload paths, repository code, or anything downstream of the picker.

## Testing

Unit tests on `_build_folder_tree_markdown` (or on the shared `_get_display_relpath` helper plus a thin tree-builder), covering:

1. Empty list → empty string.
2. Flat files only (no subfolders) → bullet list, no folder headers.
3. Single nested folder → one folder header + indented files.
4. Multiple top-level folders → multiple headers, sorted, with per-folder counts.
5. Deep nesting (3+ levels) → correct indentation at each level.
6. Hash-named leaf with no usable `orig_name` → routed to `(unnamed)` group, doesn't crash.
7. Path containing `..` / absolute segments → sanitized away (uses existing `_sanitize_relpath`).
8. Mixed root-level and nested files → both render correctly.

Tests use Gradio-style file mock objects (objects with `name` and `orig_name` attrs) to mirror what the real picker hands to the callback.

## File changes

- `src/irys/ui/chat_app.py`:
  - Extract `_get_display_relpath` helper from `_extract_files_from_upload`.
  - Add `_build_folder_tree_markdown` method.
  - Add `folder_tree_md` component inside the "Upload Folder" tab.
  - Wire `folder_upload.change` event.
- New test file `tests/test_chat_app.py` (top-level alongside `tests/test_usage_stats.py`; there's no existing `tests/ui/` directory and the chat-app module isn't currently covered by tests).

## Risk

Low. The change is additive and confined to one tab of one screen; no existing data flow is touched. The main risk is event-handler exceptions blanking the preview on edge inputs, mitigated by tests on hash names, sanitization, and empty input.
