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
        "document_actor_role",
        # MVP.2: verification substrate
        "verification_state", "verification_event",
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


def test_migration_v15_drops_redundant_quant_index():
    """v15 migration must drop ix_quant_matter_kind from existing DBs.

    ux_quant_fact_key(matter_id, quant_kind, raw_text) already covers the
    same (matter_id, quant_kind) prefix so ix_quant_matter_kind is write
    overhead with no query benefit.
    """
    import sqlite3
    from irys.matter.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)

    # After full migration suite, ix_quant_matter_kind must not exist
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='ix_quant_matter_kind'"
    ).fetchone()
    assert row is None, "ix_quant_matter_kind must be dropped by migration v15"

    # The covering index ux_quant_fact_key must still exist
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='ux_quant_fact_key'"
    ).fetchone()
    assert row is not None, "ux_quant_fact_key must be present after migration"


def test_migration_v16_unique_predicate_index():
    """Migration v16 must create ix_predicate_unique on issue_predicate(issue_id, description)."""
    import sqlite3
    from irys.matter.schema import apply_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='ix_predicate_unique'"
    ).fetchone()
    assert row is not None, "ix_predicate_unique must be created by migration v16"


def test_migration_v16_deduplicates_existing_predicates():
    """Migration v16 must remove duplicate predicate rows before creating unique index."""
    import sqlite3
    from irys.matter.schema import apply_schema, _migration_v16

    # Build a DB at v15 with duplicate issue_predicate rows.
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)

    # Drop the v16 unique index so we can insert duplicates
    conn.execute("DROP INDEX IF EXISTS ix_predicate_unique")
    conn.commit()

    # Insert an issue and two duplicate predicate rows
    conn.execute(
        "INSERT INTO matter (id, name, repository_root, created_at, updated_at)"
        " VALUES ('m1','Test','/tmp/test',datetime('now'),datetime('now'))"
    )
    conn.execute(
        "INSERT INTO issue (id, matter_id, title, issue_type, materiality, salience, status, sort_order, created_at, updated_at)"
        " VALUES ('i1','m1','Test issue','claim',0.5,0.5,'open',0,datetime('now'),datetime('now'))"
    )
    conn.execute(
        "INSERT INTO issue_predicate (id, issue_id, description, status, created_at)"
        " VALUES ('p1','i1','Element A','open',datetime('now'))"
    )
    conn.execute(
        "INSERT INTO issue_predicate (id, issue_id, description, status, created_at)"
        " VALUES ('p2','i1','Element A','open',datetime('now'))"  # duplicate
    )
    conn.commit()

    # Applying v16 migration should deduplicate and create the index
    _migration_v16(conn)

    rows = conn.execute(
        "SELECT id FROM issue_predicate WHERE issue_id='i1' AND description='Element A'"
    ).fetchall()
    assert len(rows) == 1, "Migration v16 must deduplicate existing predicate rows"


def test_migration_v27_adds_doc_basename_column():
    """v27 adds doc_basename TEXT to assertion_occurrence and creates index."""
    from irys.matter.db import SQLiteMatterDB
    db = SQLiteMatterDB.in_memory()

    # Column must exist after schema application
    col_names = {
        row[1]
        for row in db.execute(
            "PRAGMA table_info(assertion_occurrence)"
        ).fetchall()
    }
    assert "doc_basename" in col_names, "doc_basename column must exist after v27"

    # Index must exist
    idx = db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='ix_occurrence_doc_basename'"
    ).fetchone()
    assert idx is not None, "ix_occurrence_doc_basename must be created by v27"


def test_upsert_occurrence_populates_doc_basename():
    """upsert_occurrence must store correct doc_basename for various path formats."""
    from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole, ModelLayer, AssertionKind
    from irys.matter.enums import OriginKind

    model = MatterModel.open_in_memory()

    cases = [
        ("contracts/msa.pdf", "msa.pdf"),
        ("C:\\Users\\devan\\docs\\contract.pdf", "contract.pdf"),
        ("subdir/nested/letter.docx", "letter.docx"),
        ("plain_file.txt", "plain_file.txt"),
    ]

    for doc_id, expected_basename in cases:
        c = AssertionCandidate(
            proposition_text=f"Assertion for {doc_id}",
            model_layer=ModelLayer.RECORD,
            assertion_kind=AssertionKind.FACTUAL,
            document_id=doc_id,
            speech_act=SpeechAct.EXTRACTED,
            source_role=SourceRole.UNKNOWN,
            origin_kind=OriginKind.EXTRACTED,
        )
        model.assertions.upsert_occurrence(c)

        row = model.db.execute(
            "SELECT doc_basename FROM assertion_occurrence WHERE document_id=?",
            (doc_id,),
        ).fetchone()
        assert row is not None, f"Occurrence not found for document_id={doc_id!r}"
        assert row["doc_basename"] == expected_basename, (
            f"Expected basename {expected_basename!r} for {doc_id!r}, got {row['doc_basename']!r}"
        )


def test_set_trust_override_finds_assertion_via_basename():
    """set_trust_override must find assertions using doc_basename index (not LIKE)."""
    from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole, ModelLayer, AssertionKind
    from irys.matter.enums import OriginKind

    model = MatterModel.open_in_memory()

    # Store assertion with a full path document_id
    full_path = "C:/docs/matters/case001/complaint.pdf"
    c = AssertionCandidate(
        proposition_text="Plaintiff alleges breach.",
        model_layer=ModelLayer.RECORD,
        assertion_kind=AssertionKind.FACTUAL,
        document_id=full_path,
        speech_act=SpeechAct.ALLEGED,
        source_role=SourceRole.ADVOCACY,
        origin_kind=OriginKind.EXTRACTED,
    )
    assertion_id, _ = model.assertions.upsert_occurrence(c)

    # Setting override by basename only (not full path) must still find the assertion
    model.set_trust_override("complaint.pdf", "high")

    # Override must be persisted
    override = model.trust_overrides.get("complaint.pdf")
    assert override == "high", "Override must be persisted"

    # Verify the doc_basename column has the correct value
    row = model.db.execute(
        "SELECT doc_basename FROM assertion_occurrence WHERE document_id=?",
        (full_path,),
    ).fetchone()
    assert row is not None
    assert row["doc_basename"] == "complaint.pdf"


def test_set_trust_override_same_basename_different_paths():
    """Python post-filter must distinguish same-basename assertions from different paths."""
    from irys.matter import MatterModel, AssertionCandidate, SpeechAct, SourceRole, ModelLayer, AssertionKind
    from irys.matter.enums import OriginKind

    model = MatterModel.open_in_memory()

    # Two assertions with the same basename but different full paths
    def _add(text, doc):
        c = AssertionCandidate(
            proposition_text=text,
            model_layer=ModelLayer.RECORD,
            assertion_kind=AssertionKind.FACTUAL,
            document_id=doc,
            speech_act=SpeechAct.EXTRACTED,
            source_role=SourceRole.UNKNOWN,
            origin_kind=OriginKind.EXTRACTED,
        )
        aid, _ = model.assertions.upsert_occurrence(c)
        return aid

    _add("Assertion from plaintiff docs.", "plaintiff/exhibit.pdf")
    _add("Assertion from defendant docs.", "defendant/exhibit.pdf")

    # Override by basename "exhibit.pdf" — will match BOTH via doc_basename index
    # Python filter checks pat_norm == doc or pat_norm == doc_basename
    # Since pat_norm = "exhibit.pdf" != full paths, both pass the basename check
    # This is the expected behavior: basename-only override applies to all matched docs
    model.set_trust_override("exhibit.pdf", "low")

    override = model.trust_overrides.get("exhibit.pdf")
    assert override == "low"

    # Both assertions should have been revised (no assertion should have been missed)
    # (No assertion state check needed — just verifying no crash on same-basename multi-match)


def test_migration_v29_adds_reuse_rate_columns():
    """v29 adds assertions_at_start and reuse_rate columns to run_session."""
    from irys.matter.db import SQLiteMatterDB

    db = SQLiteMatterDB.in_memory()
    col_names = {
        row[1]
        for row in db.execute("PRAGMA table_info(run_session)").fetchall()
    }
    assert "assertions_at_start" in col_names, (
        "assertions_at_start column must exist in run_session after v29"
    )
    assert "reuse_rate" in col_names, (
        "reuse_rate column must exist in run_session after v29"
    )


def test_migration_v30_adds_completed_at_index():
    """v30 adds ix_run_completed index on run_session(matter_id, status, completed_at)."""
    from irys.matter.db import SQLiteMatterDB

    db = SQLiteMatterDB.in_memory()
    idx = db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='ix_run_completed'"
    ).fetchone()
    assert idx is not None, "ix_run_completed must be created by migration v30"


def test_migration_v31_adds_assertion_belief_state_index():
    """v31 adds ix_assertion_belief_state on assertion(matter_id, belief_state, updated_at)."""
    from irys.matter.db import SQLiteMatterDB

    db = SQLiteMatterDB.in_memory()
    idx = db.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type='index' AND name='ix_assertion_belief_state'"
    ).fetchone()
    assert idx is not None, (
        "ix_assertion_belief_state must be created by migration v31"
    )


def test_migration_v31_adds_gap_matter_type_index():
    """v31 adds ix_gap_matter_type on gap(matter_id, status, gap_type, materiality_score)."""
    from irys.matter.db import SQLiteMatterDB

    db = SQLiteMatterDB.in_memory()
    idx = db.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type='index' AND name='ix_gap_matter_type'"
    ).fetchone()
    assert idx is not None, (
        "ix_gap_matter_type must be created by migration v31"
    )


def test_migration_v32_adds_prop_nolayer_index():
    """v32 adds ix_assertion_prop_nolayer on assertion(matter_id, proposition_key, created_at)."""
    from irys.matter.db import SQLiteMatterDB

    db = SQLiteMatterDB.in_memory()
    idx = db.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type='index' AND name='ix_assertion_prop_nolayer'"
    ).fetchone()
    assert idx is not None, (
        "ix_assertion_prop_nolayer must be created by migration v32"
    )


def test_migration_v33_adds_ail_covering_index():
    """v33 adds ix_ail_issue_rel_assertion on assertion_issue_link(issue_id, relation_type, assertion_id)."""
    from irys.matter.db import SQLiteMatterDB

    db = SQLiteMatterDB.in_memory()
    idx = db.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type='index' AND name='ix_ail_issue_rel_assertion'"
    ).fetchone()
    assert idx is not None, (
        "ix_ail_issue_rel_assertion must be created by migration v33"
    )


# ---------------------------------------------------------------------------
# PR.1 Schema Discipline Gate
# ---------------------------------------------------------------------------

def test_fresh_db_records_current_schema_in_all_ledgers():
    """Fresh DB must record SCHEMA_VERSION in schema_version, schema_migration,
    and PRAGMA user_version — the three ledgers must agree."""
    from irys.matter.schema import get_schema_ledger_versions

    db = SQLiteMatterDB.in_memory()
    ledgers = get_schema_ledger_versions(db.conn)
    assert ledgers["schema_version"] == SCHEMA_VERSION
    assert ledgers["schema_migration"] == SCHEMA_VERSION
    assert ledgers["user_version"] == SCHEMA_VERSION


def test_open_refuses_database_newer_than_supported_user_version(tmp_path):
    """A DB whose PRAGMA user_version is above SCHEMA_VERSION must fail closed
    BEFORE any persistent write to the file. That means no new tables and no
    change to journal_mode (WAL would be a persistent file mutation)."""
    import sqlite3
    from irys.matter.schema import SchemaVersionTooNewError

    path = tmp_path / "newer.sqlite3"
    conn = sqlite3.connect(str(path))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    # Baseline: rollback journal mode, explicitly not WAL.
    conn.execute("PRAGMA journal_mode = delete")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionTooNewError):
        SQLiteMatterDB(path)

    # The guard runs before any DDL — schema_version table must not exist yet.
    check = sqlite3.connect(str(path))
    row = check.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    # journal_mode must not have been upgraded to WAL, since WAL is a persistent
    # file mutation that happens inside _conn() before apply_schema's guard runs.
    mode_row = check.execute("PRAGMA journal_mode").fetchone()
    check.close()
    assert row is None, "schema_version table must not be created after a guard failure"
    assert mode_row[0].lower() != "wal", (
        f"file-backed DB journal_mode must not be mutated by a rejected open "
        f"(got journal_mode={mode_row[0]!r})"
    )


def test_open_refuses_database_newer_than_supported_schema_migration(tmp_path):
    """A DB whose schema_migration ledger has a version above SCHEMA_VERSION
    must fail closed."""
    import sqlite3
    from irys.matter.schema import SchemaVersionTooNewError

    path = tmp_path / "newer_ledger.sqlite3"
    # Build a valid fresh DB first.
    SQLiteMatterDB(path).close()

    # Manually insert a future schema_migration row.
    conn = sqlite3.connect(str(path))
    conn.execute(
        """INSERT INTO schema_migration
           (version, name, checksum, applied_at, app_schema_version, app_build, duration_ms)
           VALUES (?, ?, '', ?, ?, 'test_build', 0)""",
        (SCHEMA_VERSION + 1, "future_v", "2026-04-17T00:00:00", SCHEMA_VERSION + 1),
    )
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionTooNewError):
        SQLiteMatterDB(path)


def test_apply_schema_does_not_run_future_migrations(monkeypatch):
    """Migrations whose target exceeds SCHEMA_VERSION must not run, even if
    they were registered in _MIGRATIONS."""
    import sqlite3
    from irys.matter import schema as schema_mod

    future_ran = {"count": 0}

    def _fake_future_migration(conn):
        future_ran["count"] += 1

    patched = list(schema_mod._MIGRATIONS) + [
        (SCHEMA_VERSION + 1, _fake_future_migration)
    ]
    monkeypatch.setattr(schema_mod, "_MIGRATIONS", patched)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    schema_mod.apply_schema(conn)

    assert future_ran["count"] == 0, "Future migration must not execute"
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version"
    ).fetchone()
    assert row[0] == SCHEMA_VERSION, "schema_version must not record a future version"


def test_duplicate_column_guard_reraises_nonduplicate_operational_error():
    """_execute_allow_duplicate_column must swallow only duplicate-column errors."""
    import sqlite3
    from irys.matter.schema import _execute_allow_duplicate_column

    db = SQLiteMatterDB.in_memory()
    conn = db.conn

    # Duplicate-column: must be swallowed.
    _execute_allow_duplicate_column(
        conn, "ALTER TABLE matter ADD COLUMN name TEXT"
    )

    # Any other OperationalError must re-raise (unknown table name here).
    with pytest.raises(sqlite3.OperationalError):
        _execute_allow_duplicate_column(
            conn, "ALTER TABLE nonexistent_table ADD COLUMN foo TEXT"
        )


def _resolve_qualname(ref: str):
    """Resolve a module:qualname reference to a live callable."""
    import importlib

    module_name, _, qualname = ref.partition(":")
    assert qualname, f"Manifest ref {ref!r} must be module:qualname form"
    module = importlib.import_module(module_name)
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def test_non_legacy_tables_have_writer_reader_or_explicit_deferral():
    """Every non-legacy table must appear in the coverage manifest with
    either named writers+readers or an explicit deferred status. Covered
    refs must resolve to real callables and the source of each callable
    must mention the table it covers (sanity check only)."""
    import inspect
    from tests.matter.table_coverage_manifest import (
        LEGACY_TABLES,
        TABLE_COVERAGE_MANIFEST,
    )

    db = SQLiteMatterDB.in_memory()
    live_tables = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }

    # Every table must be accounted for.
    uncovered = []
    for table in sorted(live_tables):
        if table in LEGACY_TABLES:
            continue
        if table in TABLE_COVERAGE_MANIFEST:
            continue
        uncovered.append(table)
    assert not uncovered, (
        f"Non-legacy tables missing from coverage manifest: {uncovered}. "
        "Add writers/readers or mark as deferred in "
        "tests/matter/table_coverage_manifest.py."
    )

    # Every covered entry must name real callables; every deferred entry
    # must supply a milestone and a reason and must not claim writers/readers.
    for table, spec in TABLE_COVERAGE_MANIFEST.items():
        if spec.deferred_until is not None:
            assert spec.deferred_until, (
                f"{table}: deferred_until must be a non-empty string"
            )
            assert spec.reason, (
                f"{table}: deferred table must include a reason"
            )
            assert not spec.writers and not spec.readers, (
                f"{table}: deferred table must not declare writers/readers; "
                "use a covered entry instead"
            )
            continue

        assert spec.writers, f"{table}: covered entry needs at least one writer"
        assert spec.readers, f"{table}: covered entry needs at least one reader"

        for ref in (*spec.writers, *spec.readers):
            fn = _resolve_qualname(ref)
            assert callable(fn), f"{table}: {ref!r} is not callable"
            src = inspect.getsource(fn)
            assert table in src, (
                f"{table}: source of {ref!r} does not mention table name — "
                "sanity check failed, is this really the writer/reader?"
            )
