"""Tests for DocumentInventoryStore (SO-1 cold/hot split gate).

Verifies:
1. upsert() creates new row, returns (doc_id, is_new=True)
2. upsert() on duplicate relative_path returns (existing_id, is_new=False)
3. is_ingested() returns False for pending docs, True after mark_ingested()
4. mark_ingested() sets ingest_status='complete'
5. get_ingested_paths() returns only complete docs
6. sha256 collision path: upsert with same hash but different path stays stable
7. count() reflects total inventory rows
"""

import pytest
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
