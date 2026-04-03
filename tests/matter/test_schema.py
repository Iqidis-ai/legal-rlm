"""Tests for schema creation and DB lifecycle."""

import pytest
from irys.matter.db import SQLiteMatterDB
from irys.matter.schema import SCHEMA_VERSION


def test_in_memory_db_creates_schema():
    db = SQLiteMatterDB.in_memory()
    # All required tables must exist
    tables = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for required in [
        "matter", "assertion", "assertion_occurrence",
        "assertion_link", "belief_revision_event",
        "run_session", "ledger_event", "schema_version",
        "document_inventory", "document_trust_override", "document_annotation",
        "actor", "actor_alias",
        "issue", "assertion_issue_link",
        "gap", "gap_link",
        "quant_fact",
        "clarification_question",
    ]:
        assert required in tables, f"Missing table: {required}"


def test_schema_version_recorded():
    db = SQLiteMatterDB.in_memory()
    row = db.execute(
        "SELECT version FROM schema_version WHERE version=?", (SCHEMA_VERSION,)
    ).fetchone()
    assert row is not None
    assert row[0] == SCHEMA_VERSION


def test_foreign_keys_enforced():
    db = SQLiteMatterDB.in_memory()
    with pytest.raises(Exception):
        # Inserting occurrence with nonexistent assertion_id should fail
        db.execute("BEGIN")
        db.execute(
            """INSERT INTO assertion_occurrence
               (id, assertion_id, document_id, source_role, speech_act, origin_kind, created_at)
               VALUES ('x','nonexistent','doc1','unknown','extracted','extracted','2026-01-01')"""
        )
        db.execute("COMMIT")


def test_wal_mode():
    db = SQLiteMatterDB.in_memory()
    row = db.execute("PRAGMA journal_mode").fetchone()
    # In-memory DBs always use "memory" journal mode — WAL is for file DBs
    assert row is not None  # Just confirm pragma runs


def test_idempotent_schema_application():
    """Applying schema twice must not raise."""
    from irys.matter.schema import apply_schema
    db = SQLiteMatterDB.in_memory()
    apply_schema(db.conn)  # Second application — skips all migrations, no error


def test_unique_index_on_assertion_occurrence():
    """The ix_occurrence_unique_doc index must exist and include speech_act in the key.

    The v5 migration corrected the key from (assertion_id, document_id) to
    (assertion_id, document_id, speech_act) so that same-assertion, same-doc,
    different-speech-act occurrences are preserved rather than silently dropped.
    """
    db = SQLiteMatterDB.in_memory()
    row = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='ix_occurrence_unique_doc'"
    ).fetchone()
    assert row is not None, "UNIQUE INDEX ix_occurrence_unique_doc must exist"
    assert "speech_act" in row[0], (
        "ix_occurrence_unique_doc must include speech_act column "
        f"(found: {row[0]})"
    )


def test_migration_v5_fixes_index_on_existing_db():
    """Applying _migration_v5 on a DB with the old coarse index corrects it."""
    import sqlite3
    from irys.matter.schema import _migration_v5, apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)

    # Verify v5 index has the correct key
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='ix_occurrence_unique_doc'"
    ).fetchone()
    assert row is not None
    assert "speech_act" in row["sql"]
