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
