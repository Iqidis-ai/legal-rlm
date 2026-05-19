"""Tests for the v2 SQLite-backed FactStore."""
import sys
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime, timezone

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


class TestImportanceLifecycle:
    """on_search_hit, on_re_extraction, tick_decay, archive_cold_facts."""

    def _insert_fact(self, store, text="Test fact", source="doc.pdf",
                     importance=50.0, tier="draft", scope_type="targeted"):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        content_hash = StoredFact.compute_hash(text, source)
        store._conn.execute(
            """INSERT OR IGNORE INTO facts
               (fact, source, extracted, scope_type, importance, recency_updated, tier, content_hash)
               VALUES (?,?,?,?,?,?,?,?)""",
            (text, source, now, scope_type, importance, now, tier, content_hash),
        )
        store._conn.commit()
        return content_hash

    def test_on_search_hit_increments_importance(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=50.0)
        store.on_search_hit(h)
        imp = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (h,)
        ).fetchone()["importance"]
        assert imp == 53.0

    def test_on_search_hit_caps_at_100(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=99.0)
        store.on_search_hit(h)
        imp = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (h,)
        ).fetchone()["importance"]
        assert imp == 100.0

    def test_on_search_hit_promotes_tier(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=63.0, tier="draft")
        store.on_search_hit(h)   # 63 + 3 = 66 >= 65 -> validated
        tier = store._conn.execute(
            "SELECT tier FROM facts WHERE content_hash=?", (h,)
        ).fetchone()["tier"]
        assert tier == "validated"

    def test_tick_decay_reduces_importance(self):
        store, _ = make_store()
        content_hash = StoredFact.compute_hash("Old fact", "doc.pdf")
        store._conn.execute(
            """INSERT INTO facts
               (fact, source, extracted, scope_type, importance, recency_updated, tier, content_hash)
               VALUES ('Old fact','doc.pdf','2026-04-01','snippet',80.0,'2026-04-01','validated',?)""",
            (content_hash,),
        )
        store._conn.commit()
        updated = store.tick_decay()
        assert updated >= 1
        imp = store._conn.execute(
            "SELECT importance FROM facts WHERE content_hash=?", (content_hash,)
        ).fetchone()["importance"]
        assert imp < 80.0

    def test_archive_cold_facts_moves_to_stubs(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=20.0, tier="draft")
        archived = store.archive_cold_facts()
        assert archived == 1
        assert store._conn.execute(
            "SELECT id FROM facts WHERE content_hash=?", (h,)
        ).fetchone() is None
        stub = store._conn.execute(
            "SELECT stub_summary FROM fact_stubs WHERE content_hash=?", (h,)
        ).fetchone()
        assert stub is not None

    def test_archive_cold_facts_skips_validated(self):
        store, _ = make_store()
        h = self._insert_fact(store, importance=20.0, tier="validated")
        archived = store.archive_cold_facts()
        assert archived == 0
        assert store._conn.execute(
            "SELECT id FROM facts WHERE content_hash=?", (h,)
        ).fetchone() is not None
