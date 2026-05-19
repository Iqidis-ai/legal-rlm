"""Tests for the v2 SQLite-backed FactStore."""
import sys
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.core.fact_store import FactStore, StoredFact


def make_store() -> tuple[FactStore, Path]:
    """Create an isolated FactStore in a temp directory."""
    tmp = Path(tempfile.mkdtemp())
    store = FactStore(tmp)
    return store, tmp


class TestSchema:
    """DB is created with the correct tables and indices."""

    def test_db_file_created_on_init(self):
        store, tmp = make_store()
        assert (tmp / ".irys" / "facts.db").exists()

    def test_facts_table_exists(self):
        store, _ = make_store()
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "facts" in table_names
        assert "source_synopses" in table_names
        assert "fact_stubs" in table_names

    def test_fts_tables_exist(self):
        store, _ = make_store()
        tables = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "fact_fts" in table_names
        assert "synopsis_fts" in table_names

    def test_wal_mode_enabled(self):
        store, _ = make_store()
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self):
        store, _ = make_store()
        fk = store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_source_synopses_has_integer_pk(self):
        """source_synopses must have INTEGER PRIMARY KEY for synopsis_fts rowid stability."""
        store, _ = make_store()
        info = store._conn.execute("PRAGMA table_info(source_synopses)").fetchall()
        cols = {row[1]: row[2] for row in info}
        assert "id" in cols
        assert cols["id"].upper() == "INTEGER"


class TestAddFactsFromExtraction:
    """add_facts_from_extraction inserts facts and returns content_hashes."""

    def _make_scope(self, targeted: bool = False):
        class FakeScope:
            is_targeted = targeted
        return FakeScope()

    def test_returns_content_hashes(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=True)
        hashes = store.add_facts_from_extraction(
            ["Contract value is $2.5M", "Governing law is Texas"],
            source="Agreement.pdf",
            scope=scope,
        )
        assert len(hashes) == 2
        assert all(len(h) == 64 for h in hashes)

    def test_targeted_scope_sets_scope_type(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=True)
        hashes = store.add_facts_from_extraction(
            ["Contract value is $2.5M"],
            source="Agreement.pdf",
            scope=scope,
        )
        row = store._conn.execute(
            "SELECT scope_type FROM facts WHERE content_hash = ?", (hashes[0],)
        ).fetchone()
        assert row["scope_type"] == "targeted"

    def test_prefix_scope_sets_scope_type(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=False)
        hashes = store.add_facts_from_extraction(
            ["Contract value is $2.5M"],
            source="Agreement.pdf",
            scope=scope,
        )
        row = store._conn.execute(
            "SELECT scope_type FROM facts WHERE content_hash = ?", (hashes[0],)
        ).fetchone()
        assert row["scope_type"] == "prefix"

    def test_duplicate_insert_returns_existing_hash(self):
        store, _ = make_store()
        scope = self._make_scope()
        h1 = store.add_facts_from_extraction(["Fact A"], source="doc.pdf", scope=scope)
        h2 = store.add_facts_from_extraction(["Fact A"], source="doc.pdf", scope=scope)
        assert h1 == h2
        count = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        assert count == 1

    def test_same_text_different_source_both_stored(self):
        store, _ = make_store()
        scope = self._make_scope()
        h1 = store.add_facts_from_extraction(["Clause 2.1(g) applies"], source="ARKS.pdf", scope=scope)
        h2 = store.add_facts_from_extraction(["Clause 2.1(g) applies"], source="BSR.pdf", scope=scope)
        assert h1 != h2
        assert store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 2

    def test_synopsis_created_lazily(self):
        store, _ = make_store()
        scope = self._make_scope()
        store.add_facts_from_extraction(
            [f"Fact {i}" for i in range(5)],
            source="Contract.pdf",
            scope=scope,
        )
        synopsis = store.get_synopsis("Contract.pdf")
        assert synopsis is not None
        assert "Contract.pdf" in synopsis

    def test_re_extraction_bumps_importance(self):
        store, _ = make_store()
        scope = self._make_scope(targeted=True)
        h = store.add_facts_from_extraction(["Re-extracted fact"], source="d.pdf", scope=scope)
        before = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (h[0],)
        ).fetchone()["importance"]
        store.add_facts_from_extraction(["Re-extracted fact"], source="d.pdf", scope=scope)
        after = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (h[0],)
        ).fetchone()["importance"]
        assert after == before + 5
