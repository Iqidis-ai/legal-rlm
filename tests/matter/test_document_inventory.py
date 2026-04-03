"""Tests for DocumentInventoryStore (SO-1 cold/hot split gate).

Verifies:
1. upsert() creates new row, returns (doc_id, is_new=True)
2. upsert() on duplicate relative_path returns (existing_id, is_new=False)
3. is_ingested() returns False for pending docs, True after mark_ingested()
4. mark_ingested() sets ingest_status='complete'
5. get_ingested_paths() returns only complete docs
6. sha256 collision path: upsert with same hash but different path stays stable
7. count() reflects total inventory rows
8. Ephemeral absolute paths normalize to stable repo-relative keys (SO-1 hot-path reuse)
"""

import pytest
from pathlib import Path
from irys.matter import MatterModel


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


# ---------------------------------------------------------------------------
# upsert() basic contract
# ---------------------------------------------------------------------------

def test_upsert_new_document(model):
    """First upsert returns is_new=True and a valid doc_id."""
    doc_id, is_new = model.inventory.upsert(
        relative_path="Contract_v1.pdf",
        sha256="a" * 64,
        size_bytes=1024,
    )
    assert is_new is True
    assert isinstance(doc_id, str) and len(doc_id) > 0


def test_upsert_duplicate_path_returns_existing(model):
    """Second upsert of the same relative_path returns same doc_id, is_new=False."""
    doc_id1, is_new1 = model.inventory.upsert("Contract_v1.pdf", "a" * 64)
    doc_id2, is_new2 = model.inventory.upsert("Contract_v1.pdf", "a" * 64)
    assert is_new1 is True
    assert is_new2 is False
    assert doc_id1 == doc_id2


def test_upsert_different_paths_are_distinct(model):
    """Different paths produce different rows."""
    id1, _ = model.inventory.upsert("Doc_A.pdf", "a" * 64)
    id2, _ = model.inventory.upsert("Doc_B.pdf", "b" * 64)
    assert id1 != id2


# ---------------------------------------------------------------------------
# mark_ingested() / is_ingested()
# ---------------------------------------------------------------------------

def test_is_ingested_false_before_mark(model):
    """Document is not ingested right after upsert (ingest_status='pending')."""
    model.inventory.upsert("Brief.pdf", "c" * 64)
    assert model.inventory.is_ingested("Brief.pdf") is False


def test_mark_ingested_sets_complete(model):
    """mark_ingested() causes is_ingested() to return True."""
    doc_id, _ = model.inventory.upsert("Brief.pdf", "c" * 64)
    model.inventory.mark_ingested(doc_id)
    assert model.inventory.is_ingested("Brief.pdf") is True


def test_is_ingested_unknown_path(model):
    """is_ingested() returns False for unknown path."""
    assert model.inventory.is_ingested("nonexistent.pdf") is False


# ---------------------------------------------------------------------------
# get_ingested_paths()
# ---------------------------------------------------------------------------

def test_get_ingested_paths_empty(model):
    id1, _ = model.inventory.upsert("A.pdf", "a" * 64)
    model.inventory.upsert("B.pdf", "b" * 64)
    # Neither is marked ingested yet
    assert model.inventory.get_ingested_paths() == []


def test_get_ingested_paths_includes_only_complete(model):
    id1, _ = model.inventory.upsert("A.pdf", "a" * 64)
    id2, _ = model.inventory.upsert("B.pdf", "b" * 64)
    model.inventory.mark_ingested(id1)
    paths = model.inventory.get_ingested_paths()
    assert "A.pdf" in paths
    assert "B.pdf" not in paths


# ---------------------------------------------------------------------------
# count()
# ---------------------------------------------------------------------------

def test_count(model):
    assert model.inventory.count() == 0
    model.inventory.upsert("X.pdf", "x" * 64)
    assert model.inventory.count() == 1
    model.inventory.upsert("Y.pdf", "y" * 64)
    assert model.inventory.count() == 2
    # Duplicate upsert should not increment count
    model.inventory.upsert("X.pdf", "x" * 64)
    assert model.inventory.count() == 2


# ---------------------------------------------------------------------------
# Partial ingest (pending row + re-upsert)
# ---------------------------------------------------------------------------

def test_partial_ingest_not_skipped(model):
    """A document with ingest_status='pending' (started but not finished) is NOT hot-pathed."""
    doc_id, is_new = model.inventory.upsert("Partial.pdf", "p" * 64)
    # Do NOT call mark_ingested — simulate incomplete prior run
    assert model.inventory.is_ingested("Partial.pdf") is False
    # Re-upsert returns the same id but is_ingested stays False
    doc_id2, is_new2 = model.inventory.upsert("Partial.pdf", "p" * 64)
    assert doc_id == doc_id2
    assert model.inventory.is_ingested("Partial.pdf") is False


# ---------------------------------------------------------------------------
# HIGH-1: basename collision — distinct paths must not alias
# ---------------------------------------------------------------------------

def test_same_basename_different_dirs_are_distinct(model):
    """contracts/msa.pdf and exhibits/msa.pdf must never share a row.

    This catches the HIGH-1 finding: if the inventory key is doc.filename (path.name),
    the second file silently aliases to the first and can be hot-skipped incorrectly.
    """
    id1, new1 = model.inventory.upsert("contracts/msa.pdf", "a" * 64)
    id2, new2 = model.inventory.upsert("exhibits/msa.pdf", "b" * 64)
    assert id1 != id2, "documents with the same basename but different paths must be distinct"
    assert new1 is True
    assert new2 is True
    # Marking one ingested must not affect the other
    model.inventory.mark_ingested(id1)
    assert model.inventory.is_ingested("contracts/msa.pdf") is True
    assert model.inventory.is_ingested("exhibits/msa.pdf") is False


# ---------------------------------------------------------------------------
# HIGH-2: sha256 mismatch — content change must force cold re-ingest
# ---------------------------------------------------------------------------

def test_sha256_change_resets_ingest_status(model):
    """If a path's content changes, ingest_status must be reset to 'pending'.

    This catches the HIGH-2 finding: without sha256 comparison, a changed file
    at the same path keeps ingest_status='complete' and stale assertions are reused.
    """
    doc_id, _ = model.inventory.upsert("contract.pdf", "a" * 64)
    model.inventory.mark_ingested(doc_id)
    assert model.inventory.is_ingested("contract.pdf") is True

    # Simulate file content change — same path, different sha256
    doc_id2, is_new2 = model.inventory.upsert("contract.pdf", "b" * 64)
    assert doc_id2 == doc_id, "same path must return same row id"
    assert is_new2 is False
    # Status must be reset so the engine runs a fresh cold ingest
    assert model.inventory.is_ingested("contract.pdf") is False, (
        "changed sha256 must reset ingest_status to 'pending'"
    )


# ---------------------------------------------------------------------------
# HIGH-3: ephemeral absolute paths — inventory key must be repo-relative
# ---------------------------------------------------------------------------

def test_relative_to_base_path_produces_stable_key(model):
    """Simulates the engine's _rel_path normalization across two 'runs'.

    Before the fix, engine.py set _rel_path = file_path (the raw SearchHit
    absolute path, e.g. /tmp/run1/contracts/msa.pdf). On a second run the
    temp dir changes (/tmp/run2/...) so the key never matches — hot path
    (SO-1 reuse) never activates.

    The fix: _rel_path = str(_fp.relative_to(repo.base_path))
    This test verifies that two absolute paths rooted at different base dirs
    but referring to the same relative file produce the same inventory key.
    """
    base_run1 = Path("/tmp/run1")
    base_run2 = Path("/tmp/run2")
    abs_path_run1 = base_run1 / "contracts" / "msa.pdf"
    abs_path_run2 = base_run2 / "contracts" / "msa.pdf"

    # Simulate what the engine does: normalize to repo-relative key
    rel_key_run1 = str(abs_path_run1.relative_to(base_run1))
    rel_key_run2 = str(abs_path_run2.relative_to(base_run2))

    assert rel_key_run1 == rel_key_run2, (
        "same file in different temp dirs must produce the same inventory key"
    )
    assert rel_key_run1 == str(Path("contracts") / "msa.pdf")

    # Verify the stable key activates hot path across simulated runs
    sha = "c" * 64
    doc_id_run1, is_new = model.inventory.upsert(rel_key_run1, sha)
    model.inventory.mark_ingested(doc_id_run1)
    assert model.inventory.is_ingested(rel_key_run1) is True

    # Second run uses a different absolute path but the SAME relative key
    doc_id_run2, is_new2 = model.inventory.upsert(rel_key_run2, sha)
    assert doc_id_run2 == doc_id_run1, "same relative key must find existing row"
    assert is_new2 is False
    assert model.inventory.is_ingested(rel_key_run2) is True, (
        "hot path must activate on second run — ephemeral base_path must not break reuse"
    )


# ---------------------------------------------------------------------------
# MEDIUM: sha256 collision — status reset must succeed even when new hash
#         conflicts with ux_inventory_hash from another row
# ---------------------------------------------------------------------------

def test_sha256_change_resets_status_when_another_path_has_same_hash(model):
    """ingest_status must be reset to 'pending' when a path's content changes,
    even when another path in the same matter already has the new sha256.

    ux_inventory_hash was removed in v6 so same-content different-path files
    each have independent rows.  Content change detection is per-path only.
    """
    # Another file already has the hash we're about to 'update' to
    model.inventory.upsert("exhibits/archive.pdf", "b" * 64)

    # This file starts as complete with a different hash
    doc_id, _ = model.inventory.upsert("contracts/msa.pdf", "a" * 64)
    model.inventory.mark_ingested(doc_id)
    assert model.inventory.is_ingested("contracts/msa.pdf") is True

    # Content changes to the same hash as exhibits/archive.pdf
    # The sha256 update will conflict — but status MUST still be reset
    doc_id2, is_new2 = model.inventory.upsert("contracts/msa.pdf", "b" * 64)
    assert doc_id2 == doc_id
    assert is_new2 is False
    assert model.inventory.is_ingested("contracts/msa.pdf") is False, (
        "status must be reset to 'pending' when sha256 changes at same path"
    )
    # exhibits/archive.pdf is unaffected — separate row for same content at different path
    assert model.inventory.is_ingested("exhibits/archive.pdf") is False


# ---------------------------------------------------------------------------
# MEDIUM: duplicate-content alternate paths tracked independently (v6: no ux_inventory_hash)
# ---------------------------------------------------------------------------

def test_same_content_different_paths_have_independent_rows(model):
    """Same sha256 at two different paths must produce two independent rows.

    Before v6, ux_inventory_hash caused INSERT OR IGNORE to collapse the second
    path onto the first row, making is_ingested(second_path) always return False.
    After v6, each (matter_id, relative_path) pair has its own row.
    """
    sha = "d" * 64
    id1, new1 = model.inventory.upsert("contracts/msa.pdf", sha)
    id2, new2 = model.inventory.upsert("backup/msa.pdf", sha)

    assert id1 != id2, "same sha256 at different paths must produce separate rows"
    assert new1 is True
    assert new2 is True

    # Ingesting one must not affect the other
    model.inventory.mark_ingested(id1)
    assert model.inventory.is_ingested("contracts/msa.pdf") is True
    assert model.inventory.is_ingested("backup/msa.pdf") is False, (
        "alternate path with same content must track independently after v6"
    )
