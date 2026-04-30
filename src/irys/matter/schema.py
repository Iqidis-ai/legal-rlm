"""SQLite schema DDL and migration runner for the matter model.

One DB per repository at repository/.irys/matter.sqlite3.
WAL mode, foreign_keys=ON, STRICT tables, JSON1, FTS5.
"""

import sqlite3

SCHEMA_VERSION = 61

# Human-readable names for the schema_migration ledger, keyed by version.
# Versions not listed here record as legacy_v<N>.
_MIGRATION_NAMES: dict[int, str] = {
    49: "schema_discipline",
    50: "verification_state",
    51: "seed_verification_state",
    52: "evidence_edge_mvp3_columns",
    53: "backfill_evidence_edge_from_legacy",
    54: "issue_predicate_template_metadata",
    55: "provenance_event_and_llm_call_hashes",
    56: "matter_trust_revision",
    57: "content_policy_audit",
    58: "llm_thinking_telemetry",
    59: "memory_broker_substrate",
    60: "object_taint_profile_binding",
    61: "object_taint_profile_scoped_uniqueness",
}


class SchemaVersionTooNewError(RuntimeError):
    """Raised when a database was written by a newer build of the code."""

    def __init__(self, db_version: int, supported_version: int):
        self.db_version = db_version
        self.supported_version = supported_version
        super().__init__(
            f"Database schema version {db_version} is newer than supported "
            f"version {supported_version}. Upgrade the application or open "
            f"with a newer build. Refusing to proceed."
        )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def get_schema_ledger_versions(conn: sqlite3.Connection) -> dict[str, int]:
    """Return the highest version recorded in each schema ledger.

    Reads from schema_version, schema_migration (if present), and
    PRAGMA user_version. Missing ledgers report 0. No writes.
    """
    sv_max = 0
    if _table_exists(conn, "schema_version"):
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        ).fetchone()
        sv_max = int(row[0] or 0)

    sm_max = 0
    if _table_exists(conn, "schema_migration"):
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migration"
        ).fetchone()
        sm_max = int(row[0] or 0)

    uv_row = conn.execute("PRAGMA user_version").fetchone()
    uv = int(uv_row[0] or 0) if uv_row else 0

    return {
        "schema_version": sv_max,
        "schema_migration": sm_max,
        "user_version": uv,
    }


def get_recorded_schema_version(conn: sqlite3.Connection) -> int:
    """Return the highest version recorded across all ledgers."""
    ledgers = get_schema_ledger_versions(conn)
    return max(ledgers.values()) if ledgers else 0


def _record_schema_version(
    conn: sqlite3.Connection, version: int, applied_at: str
) -> None:
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, applied_at),
    )


def _record_schema_migration(
    conn: sqlite3.Connection, version: int, applied_at: str, name: str
) -> None:
    """Record a migration in the schema_migration ledger if the table exists."""
    if not _table_exists(conn, "schema_migration"):
        return
    conn.execute(
        """INSERT OR IGNORE INTO schema_migration
           (version, name, checksum, applied_at, app_schema_version, app_build, duration_ms)
           VALUES (?, ?, '', ?, ?, 'python_runner', 0)""",
        (version, name, applied_at, SCHEMA_VERSION),
    )


def _execute_allow_duplicate_column(conn: sqlite3.Connection, sql: str) -> None:
    """Execute DDL that may add a column already present. Re-raise any other error.

    SQLite raises OperationalError('duplicate column name: ...') when ALTER TABLE
    ADD COLUMN targets a column that already exists. Every other OperationalError
    is a real migration defect and must surface.
    """
    try:
        conn.execute(sql)
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def assert_file_db_version_compatible(db_path) -> None:
    """Probe a file-backed database for schema version compatibility before
    any writable connection is opened against it.

    Opening a writable connection and setting PRAGMA journal_mode=WAL is a
    persistent file mutation, so the version guard must run against a plain
    probe connection before that happens. Non-existent files are treated as
    a fresh-create case and skipped. In-memory DBs do not call this.
    """
    from pathlib import Path

    path = Path(db_path)
    if not path.exists():
        return
    probe = sqlite3.connect(str(path))
    try:
        db_version = get_recorded_schema_version(probe)
    finally:
        probe.close()
    if db_version > SCHEMA_VERSION:
        raise SchemaVersionTooNewError(db_version, SCHEMA_VERSION)

# Core tables built first (the "2-hour task" subset per Codex design gate)
_DDL_CORE = """
CREATE TABLE IF NOT EXISTS matter (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    repository_root TEXT NOT NULL,
    forum       TEXT,
    posture     TEXT,
    governing_law TEXT,
    maturity    TEXT NOT NULL DEFAULT 'initial',
    -- P0.4 Trust Invalidation Lite: monotonically-increasing
    -- revision counter. Bumped every time a trigger fires
    -- (document hash change, human rejection, span replacement,
    -- privilege reclassification). Downstream caches include this
    -- in their key so a stale cache hit cannot reappear after
    -- upstream support was invalidated.
    trust_revision INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_matter_repo
    ON matter(repository_root);

CREATE TABLE IF NOT EXISTS assertion (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    proposition_key TEXT NOT NULL,
    claim_key       TEXT,
    identity_version TEXT NOT NULL DEFAULT 'legacy_text_v1',
    proposition_text TEXT NOT NULL,
    model_layer     TEXT NOT NULL,
    assertion_kind  TEXT NOT NULL,
    polarity        TEXT NOT NULL DEFAULT 'affirmed',
    canonical_subject_key TEXT,
    subject_ref_type TEXT,
    subject_ref_id   TEXT,
    predicate_key    TEXT,
    canonical_object_key TEXT,
    object_json      TEXT,
    temporal_scope_start TEXT,
    temporal_scope_end   TEXT,
    temporal_identity_key TEXT NOT NULL DEFAULT 'atemporal',
    speaker_scope_key TEXT,
    canonicalization_confidence REAL NOT NULL DEFAULT 0.0,
    belief_state    TEXT NOT NULL DEFAULT 'unknown',
    confidence      REAL NOT NULL DEFAULT 0.5,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_assertion_prop_legacy
    ON assertion(matter_id, model_layer, proposition_key, canonicalization_confidence DESC, created_at ASC);

CREATE UNIQUE INDEX IF NOT EXISTS ux_assertion_claim_key
    ON assertion(matter_id, model_layer, claim_key)
    WHERE claim_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_assertion_subject
    ON assertion(subject_ref_type, subject_ref_id, predicate_key);

CREATE INDEX IF NOT EXISTS ix_assertion_identity_lookup
    ON assertion(
        matter_id,
        subject_ref_type,
        subject_ref_id,
        predicate_key,
        canonical_object_key,
        polarity,
        temporal_identity_key
    );

CREATE TABLE IF NOT EXISTS assertion_occurrence (
    id              TEXT PRIMARY KEY,
    assertion_id    TEXT NOT NULL REFERENCES assertion(id),
    document_id     TEXT NOT NULL,
    document_inventory_id TEXT REFERENCES document_inventory(id),
    doc_basename    TEXT,
    raw_text        TEXT,
    span_id         TEXT,
    speaker_actor_id TEXT,
    source_role     TEXT NOT NULL DEFAULT 'unknown',
    source_side     TEXT,
    speech_act      TEXT NOT NULL,
    origin_kind     TEXT NOT NULL DEFAULT 'extracted',
    subject_ref_type TEXT,
    subject_ref_id   TEXT,
    predicate_key    TEXT,
    object_json      TEXT,
    temporal_scope_start TEXT,
    temporal_scope_end   TEXT,
    polarity        TEXT NOT NULL DEFAULT 'affirmed',
    speaker_scope_key TEXT,
    claim_key_candidate TEXT,
    resolution_strategy TEXT,
    extraction_confidence REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_occurrence_assertion
    ON assertion_occurrence(assertion_id, speech_act);

CREATE INDEX IF NOT EXISTS ix_occurrence_assertion_created
    ON assertion_occurrence(assertion_id, created_at);

CREATE INDEX IF NOT EXISTS ix_occurrence_document
    ON assertion_occurrence(document_id, span_id);

CREATE INDEX IF NOT EXISTS ix_occurrence_doc_basename
    ON assertion_occurrence(doc_basename);

CREATE INDEX IF NOT EXISTS ix_occurrence_doc_ref
    ON assertion_occurrence(document_inventory_id, created_at);

CREATE INDEX IF NOT EXISTS ix_occurrence_claim_candidate
    ON assertion_occurrence(claim_key_candidate);

CREATE INDEX IF NOT EXISTS ix_assertion_matter_created
    ON assertion(matter_id, created_at DESC);

-- Supports build_query_context() predicate-frequency query:
-- SELECT predicate_key, COUNT(*) WHERE matter_id=? AND predicate_key IS NOT NULL GROUP BY predicate_key
-- Without this index the query degrades to a full assertion table scan as assertion count grows.
CREATE INDEX IF NOT EXISTS ix_assertion_matter_predicate
    ON assertion(matter_id, predicate_key)
    WHERE predicate_key IS NOT NULL;

-- Prevents duplicate occurrences from concurrent runs ingesting the same document
-- with the same speech-act classification. Including speech_act allows the same
-- assertion to legitimately appear multiple times in one document when attributed
-- differently (e.g., alleged in the complaint, admitted in the answer).
-- Widened in v12 to include span identity so same (assertion, doc, speech_act)
-- can have multiple occurrences when they come from different source spans.
CREATE UNIQUE INDEX IF NOT EXISTS ix_occurrence_unique_doc
    ON assertion_occurrence(
        assertion_id,
        COALESCE(document_inventory_id, document_id),
        speech_act,
        COALESCE(span_id, '')
    );

CREATE TABLE IF NOT EXISTS assertion_link (
    id              TEXT PRIMARY KEY,
    src_assertion_id TEXT NOT NULL REFERENCES assertion(id),
    dst_assertion_id TEXT NOT NULL REFERENCES assertion(id),
    link_type       TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_assertion_link
    ON assertion_link(src_assertion_id, dst_assertion_id, link_type);

CREATE INDEX IF NOT EXISTS ix_link_dst
    ON assertion_link(dst_assertion_id, link_type);

CREATE TABLE IF NOT EXISTS belief_revision_event (
    id              TEXT PRIMARY KEY,
    assertion_id    TEXT NOT NULL REFERENCES assertion(id),
    run_id          TEXT,
    cause           TEXT NOT NULL,
    old_belief_state TEXT NOT NULL,
    new_belief_state TEXT NOT NULL,
    old_confidence  REAL,
    new_confidence  REAL,
    note            TEXT,
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_revision_assertion
    ON belief_revision_event(assertion_id, created_at);

-- Immutable field-level audit log for assertion mutations (Q4 HIGH, SO-2).
-- One row per changed field per correction batch.
-- batch_id groups all field changes from one correction call.
-- actor_kind: 'user' (direct correction) or 'system' (BFS propagation, occurrence upgrade).
-- cause: 'user_correction' | 'belief_revision' | 'occurrence_upgrade'
CREATE TABLE IF NOT EXISTS assertion_revision (
    id              TEXT PRIMARY KEY,
    batch_id        TEXT NOT NULL,
    assertion_id    TEXT NOT NULL REFERENCES assertion(id),
    changed_field   TEXT NOT NULL,
    old_value_json  TEXT,
    new_value_json  TEXT,
    actor_kind      TEXT NOT NULL,
    actor_ref       TEXT,
    cause           TEXT NOT NULL,
    run_id          TEXT,
    note            TEXT,
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_assertion_revision_assertion
    ON assertion_revision(assertion_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_assertion_revision_lock
    ON assertion_revision(assertion_id, new_value_json, created_at DESC, actor_kind)
    WHERE changed_field = 'belief_state';

CREATE TABLE IF NOT EXISTS run_session (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    query           TEXT NOT NULL,
    objective       TEXT,
    active_branch_issue_id TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    stop_requested  INTEGER NOT NULL DEFAULT 0,
    redirect_requested INTEGER NOT NULL DEFAULT 0,
    next_action     TEXT,
    operation_type  TEXT NOT NULL DEFAULT 'query',
    trigger         TEXT NOT NULL DEFAULT 'user',
    research_mode   TEXT NOT NULL DEFAULT 'deep',
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    resumed_from    TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS ix_run_matter
    ON run_session(matter_id, status, started_at);

CREATE TABLE IF NOT EXISTS ledger_event (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES run_session(id),
    seq_no              INTEGER NOT NULL,
    event_type          TEXT NOT NULL,
    why                 TEXT,
    summary             TEXT NOT NULL,
    branch_issue_id     TEXT,
    changed_object_type TEXT,
    changed_object_id   TEXT,
    snapshot_json       TEXT,
    created_at          TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_ledger_seq
    ON ledger_event(run_id, seq_no);

CREATE INDEX IF NOT EXISTS ix_ledger_object
    ON ledger_event(changed_object_type, changed_object_id);
"""

# Remaining stores (built after core, in next implementation units)
_DDL_REPOSITORY = """
CREATE TABLE IF NOT EXISTS document_inventory (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    relative_path   TEXT NOT NULL,
    storage_uri     TEXT,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    file_type       TEXT,
    modified_at     TEXT,
    discovered_at   TEXT NOT NULL,
    ingest_status   TEXT NOT NULL DEFAULT 'pending',
    parse_status    TEXT NOT NULL DEFAULT 'pending',
    duplicate_cluster TEXT,
    family_id       TEXT,
    version_chain_id TEXT,
    salience_score  REAL NOT NULL DEFAULT 0.5,
    last_read_at    TEXT,
    maintenance_status TEXT NOT NULL DEFAULT 'pending',
    profiled_at     TEXT,
    last_maintained_at TEXT
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_inventory_path
    ON document_inventory(matter_id, relative_path);

-- ux_inventory_hash deliberately omitted: same-content different-path files must have
-- separate rows so that is_ingested(path) is consistent with upsert(path).  Content
-- identity is checked per-path via the sha256 column, not across paths.

CREATE INDEX IF NOT EXISTS ix_inventory_salience
    ON document_inventory(salience_score DESC, last_read_at);

CREATE TABLE IF NOT EXISTS document_relation (
    id          TEXT PRIMARY KEY,
    source_doc_id TEXT NOT NULL REFERENCES document_inventory(id),
    target_doc_id TEXT NOT NULL REFERENCES document_inventory(id),
    relation_type TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_doc_relation_src
    ON document_relation(source_doc_id, relation_type);

CREATE INDEX IF NOT EXISTS ix_doc_relation_dst
    ON document_relation(target_doc_id, relation_type);
"""

_DDL_DOCUMENT_CARDS = """
CREATE TABLE IF NOT EXISTS document_card (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL UNIQUE REFERENCES document_inventory(id),
    title           TEXT,
    doc_type        TEXT,
    doc_subtype     TEXT,
    source_side     TEXT,
    author          TEXT,
    sender          TEXT,
    recipient       TEXT,
    creation_date   TEXT,
    sent_date       TEXT,
    effective_date  TEXT,
    discovery_date  TEXT,
    purpose         TEXT,
    rhetorical_posture TEXT,
    reliability_posture TEXT,
    operative_status TEXT NOT NULL DEFAULT 'unknown',
    privilege_flag  INTEGER NOT NULL DEFAULT 0,
    unresolved_flags TEXT,
    source_role     TEXT,
    signatories_json TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_card_type
    ON document_card(doc_type, doc_subtype);

CREATE INDEX IF NOT EXISTS ix_card_operative
    ON document_card(source_side, operative_status, effective_date);

CREATE INDEX IF NOT EXISTS ix_card_source_role
    ON document_card(source_role);
"""

_DDL_SPANS = """
CREATE TABLE IF NOT EXISTS span (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES document_inventory(id),
    span_type       TEXT NOT NULL,
    page_start      INTEGER,
    page_end        INTEGER,
    line_start      INTEGER,
    line_end        INTEGER,
    char_start      INTEGER,
    char_end        INTEGER,
    section_ref     TEXT,
    clause_ref      TEXT,
    parent_span_id  TEXT REFERENCES span(id),
    ordinal_in_doc  INTEGER,
    text_hash       TEXT NOT NULL,
    span_text       TEXT NOT NULL,
    parser_version  TEXT,
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_span_doc
    ON span(document_id, span_type, ordinal_in_doc);

CREATE UNIQUE INDEX IF NOT EXISTS ux_span_location
    ON span(document_id, text_hash, span_type, page_start, char_start);
"""

_DDL_ACTORS = """
CREATE TABLE IF NOT EXISTS actor (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    canonical_name  TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    actor_type      TEXT NOT NULL,
    home_side       TEXT,
    agenda_notes    TEXT,
    reliability_notes TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_actor_name
    ON actor(matter_id, normalized_name);

CREATE TABLE IF NOT EXISTS actor_alias (
    id          TEXT PRIMARY KEY,
    actor_id    TEXT NOT NULL REFERENCES actor(id),
    alias_text  TEXT NOT NULL,
    alias_type  TEXT NOT NULL DEFAULT 'name',
    created_at  TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_actor_alias
    ON actor_alias(actor_id, alias_text);

CREATE INDEX IF NOT EXISTS ix_actor_alias_text
    ON actor_alias(alias_text);

CREATE TABLE IF NOT EXISTS actor_affiliation (
    id          TEXT PRIMARY KEY,
    actor_id    TEXT NOT NULL REFERENCES actor(id),
    org_actor_id TEXT NOT NULL REFERENCES actor(id),
    role        TEXT,
    start_date  TEXT,
    end_date    TEXT,
    created_at  TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS document_actor_role (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES document_inventory(id),
    actor_id    TEXT NOT NULL REFERENCES actor(id),
    role_type   TEXT NOT NULL,
    raw_name    TEXT,
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_doc_actor_role_doc
    ON document_actor_role(doc_id, role_type);

CREATE INDEX IF NOT EXISTS ix_doc_actor_role_actor
    ON document_actor_role(actor_id, role_type);

CREATE UNIQUE INDEX IF NOT EXISTS ux_doc_actor_role
    ON document_actor_role(doc_id, actor_id, role_type);
"""

_DDL_ISSUES = """
CREATE TABLE IF NOT EXISTS issue (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    parent_issue_id TEXT REFERENCES issue(id),
    title           TEXT NOT NULL,
    issue_type      TEXT NOT NULL,
    burden_side     TEXT,
    materiality     REAL NOT NULL DEFAULT 0.5,
    salience        REAL NOT NULL DEFAULT 0.5,
    status          TEXT NOT NULL DEFAULT 'open',
    sort_order      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_issue_tree
    ON issue(parent_issue_id, sort_order);

CREATE TABLE IF NOT EXISTS issue_predicate (
    id          TEXT PRIMARY KEY,
    issue_id    TEXT NOT NULL REFERENCES issue(id),
    description TEXT NOT NULL,
    burden_side TEXT,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_predicate_issue
    ON issue_predicate(issue_id, status);

CREATE UNIQUE INDEX IF NOT EXISTS ix_predicate_unique
    ON issue_predicate(issue_id, description);

CREATE TABLE IF NOT EXISTS assertion_issue_link (
    id              TEXT PRIMARY KEY,
    assertion_id    TEXT NOT NULL REFERENCES assertion(id),
    issue_id        TEXT NOT NULL REFERENCES issue(id),
    relation_type   TEXT NOT NULL DEFAULT 'supports',
    created_at      TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_assertion_issue
    ON assertion_issue_link(assertion_id, issue_id, relation_type);

CREATE INDEX IF NOT EXISTS ix_issue_assertions
    ON assertion_issue_link(issue_id, relation_type);
"""

_DDL_GAPS = """
CREATE TABLE IF NOT EXISTS gap (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    gap_type        TEXT NOT NULL,
    description     TEXT NOT NULL,
    expected_artifact TEXT,
    materiality_score REAL NOT NULL DEFAULT 0.5,
    blocker_score   REAL NOT NULL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'open',
    resolution_note TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_gap_open
    ON gap(gap_type, status, materiality_score DESC);

CREATE TABLE IF NOT EXISTS gap_link (
    id              TEXT PRIMARY KEY,
    gap_id          TEXT NOT NULL REFERENCES gap(id),
    affected_type   TEXT NOT NULL,
    affected_id     TEXT NOT NULL,
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_gap_link_affected
    ON gap_link(affected_type, affected_id);
CREATE INDEX IF NOT EXISTS ix_gap_link_gap_id
    ON gap_link(gap_id);
"""

_DDL_ASSUMPTIONS = """
CREATE TABLE IF NOT EXISTS assumption (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    statement       TEXT NOT NULL,
    rationale       TEXT,
    invalidation_condition TEXT,
    source_kind     TEXT NOT NULL DEFAULT 'system',
    status          TEXT NOT NULL DEFAULT 'provisional',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_assumption_status
    ON assumption(status, source_kind);

CREATE TABLE IF NOT EXISTS assumption_link (
    id              TEXT PRIMARY KEY,
    assumption_id   TEXT NOT NULL REFERENCES assumption(id),
    target_type     TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_assumption_link_target
    ON assumption_link(target_type, target_id);
"""

_DDL_EVIDENCE = """
CREATE TABLE IF NOT EXISTS evidence_link (
    id              TEXT PRIMARY KEY,
    target_type     TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    relation_type   TEXT NOT NULL,
    proof_weight    REAL NOT NULL DEFAULT 0.5,
    source_diversity INTEGER NOT NULL DEFAULT 1,
    auth_status     TEXT,
    admissibility_status TEXT,
    vulnerability_json TEXT,
    note            TEXT,
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_evidence_target
    ON evidence_link(target_type, target_id, relation_type, proof_weight DESC);

CREATE INDEX IF NOT EXISTS ix_evidence_source
    ON evidence_link(source_type, source_id);

CREATE TABLE IF NOT EXISTS evidence_edge (
    id                          TEXT PRIMARY KEY,
    matter_id                   TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
    source_kind                 TEXT NOT NULL,
    source_id                   TEXT NOT NULL,
    source_document_inventory_id TEXT REFERENCES document_inventory(id),
    source_span_id              TEXT,
    source_occurrence_id        TEXT,
    target_kind                 TEXT NOT NULL,
    target_id                   TEXT NOT NULL,
    relation_type               TEXT NOT NULL,
    proof_weight                REAL NOT NULL DEFAULT 0.5,
    source_confidence           REAL NOT NULL DEFAULT 1.0,
    admissibility_status        TEXT,
    vulnerability_json          TEXT,
    note                        TEXT,
    verification_status         TEXT NOT NULL DEFAULT 'candidate'
        CHECK (verification_status IN ('candidate','verified','rejected','stale')),
    independence_factor         REAL NOT NULL DEFAULT 1.0,
    backfill_source             TEXT,
    source_identity_status      TEXT NOT NULL DEFAULT 'unknown',
    origin_kind                 TEXT NOT NULL DEFAULT 'legacy_backfill'
        CHECK (origin_kind IN ('ai_extracted','attorney_annotated','system_inferred','imported','legacy_backfill')),
    active                      INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
    effective_weight            REAL,
    independence_cluster_id     TEXT,
    created_at                  TEXT NOT NULL,
    updated_at                  TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_evidence_edge
    ON evidence_edge(matter_id, source_kind, source_id, target_kind, target_id, relation_type);

CREATE INDEX IF NOT EXISTS ix_evidence_edge_target
    ON evidence_edge(matter_id, target_kind, target_id, relation_type, proof_weight DESC);

CREATE INDEX IF NOT EXISTS ix_evidence_edge_source
    ON evidence_edge(matter_id, source_kind, source_id);
"""

_DDL_QUANT = """
CREATE TABLE IF NOT EXISTS quant_fact (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    quant_kind      TEXT NOT NULL,
    amount_value    REAL,
    date_value      TEXT,
    date_end_value  TEXT,
    rate_value      REAL,
    currency        TEXT,
    unit            TEXT,
    raw_text        TEXT NOT NULL,
    subject_type    TEXT,
    subject_id      TEXT,
    span_id         TEXT,
    assertion_id    TEXT REFERENCES assertion(id),
    date_precision  TEXT,
    quant_dedup_key TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_quant_fact_key
    ON quant_fact(matter_id, quant_dedup_key);

CREATE INDEX IF NOT EXISTS ix_quant_kind
    ON quant_fact(quant_kind, date_value);

CREATE INDEX IF NOT EXISTS ix_quant_subject
    ON quant_fact(subject_type, subject_id);

"""

_DDL_CLARIFICATION = """
CREATE TABLE IF NOT EXISTS clarification_question (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    gap_id          TEXT REFERENCES gap(id),
    run_id          TEXT REFERENCES run_session(id),
    question_text   TEXT NOT NULL,
    why_it_matters  TEXT,
    expected_impact TEXT,
    answer_text     TEXT,
    answered_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_clarification_matter_status
    ON clarification_question(matter_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_clarification_matter_text
    ON clarification_question(matter_id, question_text);

CREATE INDEX IF NOT EXISTS ix_clarification_matter_answered
    ON clarification_question(matter_id, answered_at DESC);
"""

_DDL_REASONING_CACHE = """
CREATE TABLE IF NOT EXISTS reasoning_cache (
    id          TEXT PRIMARY KEY,
    matter_id   TEXT NOT NULL REFERENCES matter(id),
    stage       TEXT NOT NULL,
    cache_key   TEXT NOT NULL,
    plan_json   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    last_hit_at TEXT
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_reasoning_cache
    ON reasoning_cache(matter_id, stage, cache_key);
"""

_DDL_TRUST_OVERRIDE = """
CREATE TABLE IF NOT EXISTS document_trust_override (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    document_pattern TEXT NOT NULL,
    trust_level     TEXT NOT NULL,
    note            TEXT,
    created_at      TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_trust_override
    ON document_trust_override(matter_id, document_pattern);
"""

_DDL_DOCUMENT_ANNOTATION = """
CREATE TABLE IF NOT EXISTS document_annotation (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    document_pattern TEXT NOT NULL,
    annotation_text TEXT NOT NULL,
    annotation_type TEXT NOT NULL DEFAULT 'strategic',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_annotation_matter
    ON document_annotation(matter_id, annotation_type, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_annotation_pattern
    ON document_annotation(matter_id, document_pattern, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_annotation_recent
    ON document_annotation(matter_id, created_at DESC);
"""

_DDL_AUTHORITY = """
CREATE TABLE IF NOT EXISTS authority (
    id               TEXT PRIMARY KEY,
    matter_id        TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
    authority_type   TEXT NOT NULL DEFAULT 'case',
    citation         TEXT NOT NULL,
    name             TEXT,
    jurisdiction     TEXT,
    decided_at       TEXT,
    holdings         TEXT,
    key_rules        TEXT,
    weight           TEXT NOT NULL DEFAULT 'persuasive',
    precedential_rank INTEGER NOT NULL DEFAULT 0,
    applicability    TEXT,
    source_doc_id    TEXT,
    source_span_id   TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_authority_citation
    ON authority(matter_id, citation);

CREATE INDEX IF NOT EXISTS ix_authority_type
    ON authority(matter_id, authority_type);

CREATE INDEX IF NOT EXISTS ix_authority_weight
    ON authority(matter_id, weight);

CREATE TABLE IF NOT EXISTS authority_issue_link (
    authority_id  TEXT NOT NULL REFERENCES authority(id) ON DELETE CASCADE,
    issue_id      TEXT NOT NULL REFERENCES issue(id) ON DELETE CASCADE,
    relevance     TEXT NOT NULL DEFAULT 'supporting',
    created_at    TEXT NOT NULL,
    PRIMARY KEY (authority_id, issue_id)
) STRICT;

CREATE INDEX IF NOT EXISTS ix_authority_issue_link_issue
    ON authority_issue_link(issue_id);
"""

_DDL_PROOF_STATE = """
CREATE TABLE IF NOT EXISTS proof_state (
    id                          TEXT PRIMARY KEY,
    matter_id                   TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
    issue_id                    TEXT NOT NULL REFERENCES issue(id) ON DELETE CASCADE,
    sufficiency                 REAL NOT NULL DEFAULT 0.0,
    supporting_count            INTEGER NOT NULL DEFAULT 0,
    attacking_count             INTEGER NOT NULL DEFAULT 0,
    total_predicate_count       INTEGER NOT NULL DEFAULT 0,
    satisfied_predicate_count   INTEGER NOT NULL DEFAULT 0,
    proof_status                TEXT NOT NULL DEFAULT 'insufficient',
    support_score               REAL NOT NULL DEFAULT 0.0,
    attack_score                REAL NOT NULL DEFAULT 0.0,
    coverage_version            TEXT NOT NULL DEFAULT 'proof_v2',
    notes                       TEXT,
    computed_at                 TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_proof_state_issue
    ON proof_state(matter_id, issue_id);

CREATE INDEX IF NOT EXISTS ix_proof_state_status
    ON proof_state(matter_id, proof_status);
"""

_DDL_DECISION_CONTEXT = """
CREATE TABLE IF NOT EXISTS decision_context (
    id                   TEXT PRIMARY KEY,
    matter_id            TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
    decision_maker_type  TEXT,
    decision_maker_name  TEXT,
    objective            TEXT,
    strategic_notes      TEXT,
    scope_narrow         INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_decision_context_matter
    ON decision_context(matter_id);
"""

_DDL_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL
) STRICT;
"""

_DDL_PENDING_PROPAGATION = """
CREATE TABLE IF NOT EXISTS pending_propagation (
    id           TEXT PRIMARY KEY,
    matter_id    TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
    assertion_id TEXT NOT NULL,
    cause        TEXT NOT NULL,
    orig_run_id  TEXT,
    queue        TEXT NOT NULL,
    enqueued_at  TEXT NOT NULL,
    UNIQUE(matter_id, assertion_id, queue)
) STRICT;

CREATE INDEX IF NOT EXISTS ix_pending_propagation_matter
    ON pending_propagation(matter_id, queue);
"""

# Full DDL in apply order
ALL_DDL = [
    _DDL_SCHEMA_VERSION,
    _DDL_CORE,
    _DDL_REPOSITORY,
    _DDL_DOCUMENT_CARDS,
    _DDL_SPANS,
    _DDL_ACTORS,
    _DDL_ISSUES,
    _DDL_GAPS,
    _DDL_ASSUMPTIONS,
    _DDL_EVIDENCE,
    _DDL_QUANT,
    _DDL_PENDING_PROPAGATION,
]

def _migration_v1(conn) -> None:
    """Initial schema: create all tables and indexes."""
    for block in ALL_DDL:
        for stmt in block.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)


def _migration_v2(conn) -> None:
    """Add missing query-critical indexes identified in Tier 2 scaling review.

    Fixes O(n) scans on: gap open/count, issue title dedup, issue open list,
    run_session recent_runs, actor list, assertion_link dependents lookup.
    """
    stmts = [
        # assertion_link: src+type lookup for get_dependents()
        # Current ux_assertion_link has dst between src and type — suboptimal.
        "CREATE INDEX IF NOT EXISTS ix_link_src_type "
        "ON assertion_link(src_assertion_id, link_type, dst_assertion_id)",

        # gap: matter+status filter for open_gaps() / count_open()
        # Current ix_gap_open starts with gap_type, not matter_id.
        "CREATE INDEX IF NOT EXISTS ix_gap_matter_status "
        "ON gap(matter_id, status, materiality_score DESC)",

        # issue: matter+status filter for get_open_issues()
        # Current ix_issue_tree is (parent_issue_id, sort_order) — wrong filter.
        "CREATE INDEX IF NOT EXISTS ix_issue_matter_status "
        "ON issue(matter_id, status, salience DESC, materiality DESC)",

        # issue: expression index for LOWER(title) dedup in upsert_issue()
        "CREATE INDEX IF NOT EXISTS ix_issue_title_norm "
        "ON issue(matter_id, LOWER(title))",

        # run_session: matter+time for recent_runs() ORDER BY started_at DESC
        # Current ix_run_matter has status between matter_id and started_at.
        "CREATE INDEX IF NOT EXISTS ix_run_matter_time "
        "ON run_session(matter_id, started_at DESC)",

        # actor: matter+canonical_name for list_actors() ORDER BY canonical_name
        # ux_actor_name covers matter_id+normalized_name but not canonical_name sort.
        "CREATE INDEX IF NOT EXISTS ix_actor_matter_name "
        "ON actor(matter_id, canonical_name)",
    ]
    for stmt in stmts:
        conn.execute(stmt)


def _migration_v3(conn) -> None:
    """Fix assertion identity collapse — include model_layer in the unique key.

    Previously ux_assertion_prop was (matter_id, proposition_key), which meant
    the same text proposition in different reasoning layers silently merged into
    one canonical assertion. This migration widens the key so record/reality/
    proof/legal/decision-context assertions can coexist as separate objects.

    For DBs with existing data: rows with the same (matter_id, proposition_key)
    but different model_layer will now be distinct. Rows with the same
    (matter_id, model_layer, proposition_key) are deduplicated — the first
    row encountered is kept, others are deleted before the index is rebuilt.
    """
    # Remove duplicate rows that would violate the new unique constraint,
    # keeping the earliest created_at row per (matter_id, model_layer, proposition_key).
    conn.execute("""
        DELETE FROM assertion WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM assertion
            GROUP BY matter_id, model_layer, proposition_key
        )
    """)
    # Drop old single-key index and create layer-aware replacement.
    conn.execute("DROP INDEX IF EXISTS ux_assertion_prop")
    conn.execute(
        "CREATE UNIQUE INDEX ux_assertion_prop "
        "ON assertion(matter_id, model_layer, proposition_key)"
    )


def _migration_v4(conn) -> None:
    """Add clarification_question table (clarification engine, SO-7/SO-3)."""
    for block in [_DDL_CLARIFICATION]:
        for stmt in block.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)


def _migration_v5(conn) -> None:
    """Fix ix_occurrence_unique_doc: widen key to include speech_act.

    The v4 schema introduced ix_occurrence_unique_doc on (assertion_id, document_id),
    which is too coarse — it silently drops legitimate second occurrences of the same
    assertion within one document when attributed with a different speech act (e.g.,
    alleged vs admitted). The correct key is (assertion_id, document_id, speech_act).

    Also ensures the index exists on databases that predate the v4 schema extension
    (databases that skipped the index entirely due to SCHEMA_VERSION lagging).
    """
    # DROP is idempotent via IF EXISTS; handles both the too-coarse v4 variant
    # and the case where the index was never created on older databases.
    conn.execute("DROP INDEX IF EXISTS ix_occurrence_unique_doc")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_occurrence_unique_doc"
        " ON assertion_occurrence(assertion_id, document_id, speech_act)"
    )


def _migration_v6(conn) -> None:
    """Drop ux_inventory_hash to allow same-content different-path files to coexist.

    ux_inventory_hash on (matter_id, sha256) caused two problems:
    1. INSERT OR IGNORE silently collapsed same-content alternate paths onto the
       same inventory row, making is_ingested(alt_path) always return False.
    2. sha256 UPDATE on content change failed when the new hash already existed on
       another row, requiring an error-swallowing fallback.

    Removing this index gives each (matter_id, relative_path) pair an independent row.
    Content-change detection still works via the sha256 column comparison in upsert().
    """
    conn.execute("DROP INDEX IF EXISTS ux_inventory_hash")


def _migration_v7(conn) -> None:
    """Add gap_id index on gap_link for efficient per-gap link lookups.

    generate_clarifications_from_gaps() queries gap_link by gap_id.
    The existing ix_gap_link_affected index is on (affected_type, affected_id),
    not gap_id, so each lookup was a full scan of gap_link.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_gap_link_gap_id ON gap_link(gap_id)"
    )


def _migration_v8(conn) -> None:
    """Add reasoning_cache table for SO-1 hot-path orientation reuse.

    Caches expensive LLM orientation plans by (matter_id, stage, cache_key)
    so warm runs can skip the FLASH model call when the query and repo structure
    are unchanged.
    """
    for stmt in _DDL_REASONING_CACHE.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _migration_v10(conn) -> None:
    """Add document_annotation table for SO-3 user document annotation.

    Allows users to attach persistent strategic notes to document patterns
    (e.g., "this expert report tends to overstate damages"). Notes are injected
    into the orientation prompt so the engine uses them when building its plan.
    """
    for stmt in _DDL_DOCUMENT_ANNOTATION.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _migration_v9(conn) -> None:
    """Add document_trust_override table for SO-3 user trust steering.

    Allows users to mark specific documents as low-trust (forcing ALLEGED speech act)
    or high-trust (promoting ALLEGED → OPERATIVE) so source calibration respects
    user domain knowledge rather than relying solely on filename heuristics.
    """
    for stmt in _DDL_TRUST_OVERRIDE.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _migration_v11(conn) -> None:
    """Add performance indexes for list_recent() and document_annotation queries.

    These indexes support:
    - ix_occurrence_assertion_created: correlated subqueries in list_recent() for
      primary_document_id/source_role/speech_act chronological ordering
    - ix_assertion_matter_created: fast matter-scoped paging in list_recent()
    - ix_annotation_pattern: get_for_document() filter on matter_id + document_pattern
    - ix_annotation_recent: list_recent() order by created_at within a matter
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_occurrence_assertion_created "
        "ON assertion_occurrence(assertion_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_assertion_matter_created "
        "ON assertion(matter_id, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_annotation_pattern "
        "ON document_annotation(matter_id, document_pattern, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_annotation_recent "
        "ON document_annotation(matter_id, created_at DESC)"
    )


def _migration_v12(conn) -> None:
    """Widen ix_occurrence_unique_doc to include span_id.

    The v5 key (assertion_id, document_id, speech_act) silently drops a second
    occurrence of the same assertion from the same document + speech_act when it
    comes from a different source span. Including span identity preserves provenance.
    COALESCE(span_id, '') treats NULL spans as a single bucket (backward-compatible
    with existing rows that have no span).
    """
    conn.execute("DROP INDEX IF EXISTS ix_occurrence_unique_doc")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_occurrence_unique_doc"
        " ON assertion_occurrence(assertion_id, document_id, speech_act, COALESCE(span_id, ''))"
    )


def _migration_v13(conn) -> None:
    """Add UNIQUE constraint on quant_fact(matter_id, quant_kind, raw_text).

    Enables INSERT OR IGNORE idempotency in QuantStore.record() and
    QuantStore.record_many(), eliminating the SELECT-then-INSERT pattern
    and enabling bulk inserts with executemany().

    First removes any pre-existing duplicate rows (keeping the earliest by
    rowid per key group) so that CREATE UNIQUE INDEX does not fail on
    databases that accumulated duplicates via the old SELECT-then-INSERT
    race pattern.

    Both the DELETE and the CREATE UNIQUE INDEX run inside a single explicit
    transaction so that if the process dies between them the migration is
    either fully applied or fully not applied (recoverable on next open).
    SQLiteMatterDB uses isolation_level=None (autocommit), so we manage the
    transaction explicitly here.
    """
    conn.execute("BEGIN")
    try:
        conn.execute(
            """DELETE FROM quant_fact
               WHERE rowid NOT IN (
                   SELECT MIN(rowid)
                   FROM quant_fact
                   GROUP BY matter_id, quant_kind, raw_text
               )"""
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_quant_fact_key"
            " ON quant_fact(matter_id, quant_kind, raw_text)"
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _migration_v14(conn) -> None:
    """Add composite index ix_quant_matter_kind on (matter_id, quant_kind).

    The conflict-detection query in get_conflicts() filters by matter_id AND
    quant_kind='amount' before grouping. Without a covering index on both
    columns, SQLite must scan all quant_fact rows for the matter. This index
    makes the aggregation O(amount-facts-in-matter) instead of O(all-quant-facts).
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_quant_matter_kind"
        " ON quant_fact(matter_id, quant_kind)"
    )


def _migration_v15(conn) -> None:
    """Drop redundant ix_quant_matter_kind index.

    ux_quant_fact_key(matter_id, quant_kind, raw_text) already provides an index
    with (matter_id, quant_kind) as the leading two columns. The separate
    ix_quant_matter_kind adds write amplification on every quant_fact insert
    without improving selectivity for the conflict-detection query.
    """
    conn.execute("DROP INDEX IF EXISTS ix_quant_matter_kind")


def _migration_v17(conn) -> None:
    """Add covering index for get_by_kind(matter_id, quant_kind) with date ordering.

    get_by_kind() filters by (matter_id, quant_kind) and orders by (date_value, created_at).
    The existing ix_quant_kind(quant_kind, date_value) lacks matter_id as a leading column,
    so SQLite must sort the filtered set rather than using index order.
    ix_quant_matter_kind_date(matter_id, quant_kind, date_value, created_at) allows the DB
    to satisfy both the filter and the ORDER BY from a single index scan.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_quant_matter_kind_date"
        " ON quant_fact(matter_id, quant_kind, date_value, created_at)"
    )


def _migration_v16(conn) -> None:
    """Add UNIQUE constraint on issue_predicate(issue_id, description).

    Enables INSERT OR IGNORE idempotency in add_predicate() / add_predicates_batch(),
    preventing duplicate predicate rows on warm-run cache hits.

    Deduplicates any pre-existing duplicate rows (keeping the earliest by rowid per
    key group) before creating the index, mirroring the safe pattern from v13.

    Uses SAVEPOINT instead of BEGIN so the migration is safe when called
    within an outer transaction (nested-transaction safe).
    """
    conn.execute("SAVEPOINT _v16")
    try:
        conn.execute(
            """DELETE FROM issue_predicate
               WHERE rowid NOT IN (
                   SELECT MIN(rowid)
                   FROM issue_predicate
                   GROUP BY issue_id, description
               )"""
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_predicate_unique"
            " ON issue_predicate(issue_id, description)"
        )
        conn.execute("RELEASE _v16")
    except Exception:
        conn.execute("ROLLBACK TO _v16")
        conn.execute("RELEASE _v16")  # always release to close the savepoint
        raise


def _migration_v18(conn) -> None:
    """Add description_key column and lookup index for idempotent gap inserts.

    Without this, every engine run that detects the same structural gap inserts a new
    duplicate row, causing the open-gap list to fill with identical entries and the
    synthesis gap summary to show the same gap N times.

    description_key = sha256(gap_type + ':' + normalized_description)[:32].
    The index on (matter_id, gap_type, description_key) allows record() to quickly
    find an existing gap with the same description before deciding to INSERT vs. reopen.
    A closed gap that is re-detected is reopened rather than duplicated.

    Does NOT use a UNIQUE index because a closed gap and a new open gap with the same
    description would conflict; instead the application layer handles dedup logic.
    """
    import hashlib
    conn.execute("SAVEPOINT _v18")
    try:
        _execute_allow_duplicate_column(
            conn, "ALTER TABLE gap ADD COLUMN description_key TEXT"
        )

        # Back-fill existing rows using executemany (avoids N×individual UPDATEs)
        rows = conn.execute("SELECT id, gap_type, description FROM gap WHERE description_key IS NULL").fetchall()
        if rows:
            updates = []
            for row in rows:
                key = hashlib.sha256(
                    f"{row[1]}:{(row[2] or '').lower().strip()}".encode()
                ).hexdigest()[:32]
                updates.append((key, row[0]))
            conn.executemany("UPDATE gap SET description_key=? WHERE id=?", updates)

        # Index for fast lookup by matter + type + key, with created_at trailing to
        # cover ORDER BY created_at DESC in record() (non-unique — allows closed dupes)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_gap_dedup"
            " ON gap(matter_id, gap_type, description_key, created_at)"
        )
        conn.execute("RELEASE _v18")
    except Exception:
        conn.execute("ROLLBACK TO _v18")
        conn.execute("RELEASE _v18")
        raise


def _migration_v19(conn) -> None:
    """Add quant_dedup_key column and rebuild ux_quant_fact_key to include subject_id.

    The old unique index on (matter_id, quant_kind, raw_text) would collide for
    distinct numeric facts sharing the same raw text (e.g. "Invoice Amount: $50,000"
    appearing in two different invoices). This caused SO-6 coverage gaps where the
    second occurrence was silently dropped.

    The new dedup key is:
        sha256(quant_kind + ':' + (subject_id or '') + ':' + raw_text[:200])[:32]

    This incorporates subject_id so facts with different identifiers ("Invoice #1042"
    vs "Invoice #2001") are treated as distinct even if their raw text matches.
    """
    import hashlib
    conn.execute("SAVEPOINT _v19")
    try:
        _execute_allow_duplicate_column(
            conn,
            "ALTER TABLE quant_fact ADD COLUMN quant_dedup_key TEXT NOT NULL DEFAULT ''",
        )

        # Back-fill existing rows in chunks of 500 to bound memory use and lock window.
        _CHUNK = 500
        while True:
            rows = conn.execute(
                "SELECT id, quant_kind, subject_id, raw_text FROM quant_fact"
                " WHERE quant_dedup_key = '' LIMIT ?",
                (_CHUNK,),
            ).fetchall()
            if not rows:
                break
            updates = []
            for row in rows:
                payload = f"{row[1]}:{row[2] or ''}:{(row[3] or '')[:200]}"
                key = hashlib.sha256(payload.encode()).hexdigest()[:32]
                updates.append((key, row[0]))
            conn.executemany("UPDATE quant_fact SET quant_dedup_key=? WHERE id=?", updates)

        # Drop old raw_text-based unique index and replace with dedup_key index
        conn.execute("DROP INDEX IF EXISTS ux_quant_fact_key")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_quant_fact_key"
            " ON quant_fact(matter_id, quant_dedup_key)"
        )
        conn.execute("RELEASE _v19")
    except Exception:
        conn.execute("ROLLBACK TO _v19")
        conn.execute("RELEASE _v19")
        raise


def _migration_v20(conn) -> None:
    """Add (matter_id, predicate_key) partial index for build_query_context() predicate query.

    build_query_context() executes:
        SELECT predicate_key, COUNT(*) WHERE matter_id=? AND predicate_key IS NOT NULL
        GROUP BY predicate_key ORDER BY cnt DESC LIMIT 20

    Without a (matter_id, predicate_key) index this degrades to a full assertion table
    scan as assertion count grows (identified as MEDIUM by Tier 1 Performance review).
    The partial index (WHERE predicate_key IS NOT NULL) excludes non-SPO assertions
    and is ~50–80% smaller than a full index at typical matter sizes.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_assertion_matter_predicate"
        " ON assertion(matter_id, predicate_key)"
        " WHERE predicate_key IS NOT NULL"
    )


def _migration_v21(conn) -> None:
    """Add decision_context table for decision-context overlays (Priority 1).

    One row per matter (unique index on matter_id).  Stores the decision-maker
    type, objective, and strategic notes that influence synthesis framing without
    touching the canonical record model.
    """
    for stmt in _DDL_DECISION_CONTEXT.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _migration_v22(conn) -> None:
    """Add authority and authority_issue_link tables (legal research layer).

    authority: one row per unique citation per matter.  Stores structured
    metadata about legal authorities (cases, statutes, regulations, rules,
    secondary sources) used in the matter analysis.

    authority_issue_link: many-to-many relationship between authorities and
    issues, with a relevance label (supporting/attacking/neutral).
    """
    for stmt in _DDL_AUTHORITY.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _migration_v23(conn) -> None:
    """Add proof_state table (proof-aware reasoning layer).

    proof_state: one row per issue per matter.  Stores a computed snapshot of
    each issue's proof coverage: sufficiency score, assertion counts,
    predicate satisfaction ratio, and a categorical proof_status.
    """
    for stmt in _DDL_PROOF_STATE.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _migration_v24(conn) -> None:
    """Add trust columns to proof_state (SO-5 enforcement upgrade).

    Moves trust metadata from JSON notes blob to dedicated indexed columns so
    trust enforcement is a first-class structural property, not advisory text:
    - trust_weighted_support: source-trust-weighted sum of supporting evidence
    - trust_weighted_attack:  source-trust-weighted sum of attacking evidence
    - advocacy_only:          1 when all supporting assertions are low-trust sources

    Existing rows get DEFAULT 0 / 0.0; they will be refreshed by the next
    compute_and_store() call.  The notes column is retained for forward
    compatibility but trust values are now read from the typed columns.
    """
    conn.execute(
        "ALTER TABLE proof_state ADD COLUMN trust_weighted_support REAL NOT NULL DEFAULT 0.0"
    )
    conn.execute(
        "ALTER TABLE proof_state ADD COLUMN trust_weighted_attack REAL NOT NULL DEFAULT 0.0"
    )
    conn.execute(
        "ALTER TABLE proof_state ADD COLUMN advocacy_only INTEGER NOT NULL DEFAULT 0"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_proof_state_advocacy"
        " ON proof_state(matter_id, advocacy_only)"
    )


def _migration_v30(conn) -> None:
    """Add completed_at index on run_session for get_so_metrics() reuse_rate query.

    get_so_metrics() queries:
        SELECT reuse_rate FROM run_session
        WHERE matter_id=? AND status='completed' AND reuse_rate IS NOT NULL
        ORDER BY completed_at DESC LIMIT 5

    The existing ix_run_matter index covers (matter_id, status, started_at) but
    does not include completed_at, so the ORDER BY requires an in-memory sort.
    This covering index eliminates the sort by making completed_at a key column.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_run_completed"
        " ON run_session(matter_id, status, completed_at DESC)"
    )


def _migration_v29(conn) -> None:
    """Add reuse-rate tracking columns to run_session (SO-1 measurability).

    Adds two columns so the system can record and surface the fraction of
    matter-model assertions that were read from the persistent store versus
    freshly extracted during each investigation run:

      assertions_at_start INTEGER — assertion count snapshotted before the run
      reuse_rate          REAL    — assertions_at_start / assertions_at_end,
                                    computed when the run completes

    A reuse_rate of 1.0 means no new assertions were created (full reuse);
    0.0 means the matter was empty at start (first run).  The target from
    docs/PROJECT_CONTEXT.md is > 0.70 on repeated queries over a stable matter.
    """
    conn.execute(
        "ALTER TABLE run_session ADD COLUMN assertions_at_start INTEGER"
    )
    conn.execute(
        "ALTER TABLE run_session ADD COLUMN reuse_rate REAL"
    )


def _migration_v28(conn) -> None:
    """Add covering index on assertion_link for get_neighbor_belief_states() CTE.

    The `linked` subquery in get_neighbor_belief_states() runs:
        SELECT link_type, src_assertion_id
        FROM assertion_link
        WHERE dst_assertion_id = ? AND link_type IN (...)

    Without this index the query performs a full scan filtered by dst_assertion_id
    via the existing ix_link_src/ix_link_dst indexes, then re-reads the heap for
    link_type.  A covering index on (dst_assertion_id, link_type, src_assertion_id)
    eliminates the heap fetch entirely — all needed columns are in the index leaf.

    This is the highest-ROI index from the Tier 2 scaling review: the `linked`
    subquery is the hot inner loop of every BFS step in BeliefRevisionEngine.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_link_dst_covering"
        " ON assertion_link(dst_assertion_id, link_type, src_assertion_id)"
    )


def _migration_v27(conn) -> None:
    """Add doc_basename column to assertion_occurrence for indexed basename lookup.

    set_trust_override() previously used leading-wildcard LIKE ('%/basename')
    to find assertions by document basename — not sargable on the document_id
    index.  This migration adds a pre-computed, indexed doc_basename column so
    the query can use an equality check instead.

    Existing rows are backfilled by normalizing document_id to forward slashes
    and extracting the basename in Python (no SQLite BASENAME() function exists).
    """
    import pathlib as _pathlib
    _execute_allow_duplicate_column(
        conn,
        "ALTER TABLE assertion_occurrence ADD COLUMN doc_basename TEXT",
    )
    rows = conn.execute(
        "SELECT id, document_id FROM assertion_occurrence WHERE document_id IS NOT NULL"
    ).fetchall()
    for row in rows:
        doc_norm = row[1].replace("\\\\", "/").replace("\\", "/")
        basename = _pathlib.Path(doc_norm).name
        conn.execute(
            "UPDATE assertion_occurrence SET doc_basename=? WHERE id=?",
            (basename, row[0]),
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_occurrence_doc_basename"
        " ON assertion_occurrence(doc_basename)"
    )


def _migration_v26(conn) -> None:
    """Add leading link_type index on assertion_link for find_contradictions().

    find_contradictions() filters assertion_link by link_type IN
    ('attacks', 'contradicts') before joining assertion rows. The existing
    indexes have link_type as a secondary column; a leading link_type index
    allows direct range scan on the type without touching every row.

    This is a low-overhead addition since assertion_link is a narrow table
    and the index covers a small cardinality column.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_link_type"
        " ON assertion_link(link_type, src_assertion_id, dst_assertion_id)"
    )


def _migration_v25(conn) -> None:
    """Add UNIQUE constraint to document_relation (idempotency fix).

    link_documents() used INSERT OR IGNORE but document_relation had no
    UNIQUE constraint, so duplicate (source, target, relation_type) triples
    were silently inserted on repeated runs.  This adds the constraint.

    SQLite does not support ADD CONSTRAINT, so we create a UNIQUE INDEX
    which has the same enforcement effect.
    """
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_doc_relation_triple"
        " ON document_relation(source_doc_id, target_doc_id, relation_type)"
    )


def _migration_v31(conn) -> None:
    """Add hot-path indexes from Tier 2 scaling review Q5 (10k docs / 100k assertions).

    Two indexes targeting scan-heavy queries identified in the Tier 2 milestone review:

    1. ix_assertion_belief_state — get_ledger_steering_surface() disputed/unknown query:
           SELECT id, proposition_text, belief_state
           FROM assertion
           WHERE matter_id=? AND belief_state IN ('disputed','unknown')
           ORDER BY updated_at DESC LIMIT 5
       Without this index the query scans all assertions for the matter, filters post-scan
       by belief_state (low selectivity on large matters), then sorts on updated_at.
       The (matter_id, belief_state, updated_at DESC) composite lets the planner use
       a range scan on matter_id + equality on belief_state with pre-sorted updated_at.

    2. ix_gap_matter_type — extends the existing ix_gap_matter_status coverage:
           SELECT * FROM gap WHERE matter_id=? AND status='open'
           AND materiality_score >= ? ORDER BY materiality_score DESC
       ix_gap_matter_status covers (matter_id, status, materiality_score DESC) for the
       open_gaps() query.  This new index adds gap_type as a key column so future
       queries that filter by both status and gap_type (e.g. steering surface
       supply_document classification) get full composite selectivity rather than
       filtering gap_type post-scan.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_assertion_belief_state"
        " ON assertion(matter_id, belief_state, updated_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_gap_matter_type"
        " ON gap(matter_id, status, gap_type, materiality_score DESC)"
    )


def _migration_v32(conn) -> None:
    """Add seekable index for the no-layer get_by_proposition() query path.

    AssertionStore.get_by_proposition(text, model_layer=None) now issues:
        SELECT * FROM assertion
        WHERE matter_id=? AND proposition_key=?
        ORDER BY created_at ASC LIMIT 1

    The existing ux_assertion_prop is (matter_id, model_layer, proposition_key) —
    model_layer in the middle means the prefix (matter_id, proposition_key) is not
    directly seekable from that index.  ix_assertion_matter_created covers
    (matter_id, created_at DESC) but cannot efficiently filter proposition_key.

    A composite (matter_id, proposition_key, created_at) index lets the planner
    seek to (matter_id, proposition_key) and then return the first row in created_at
    order without a full-matter scan, making the no-layer lookup O(log N) at any
    matter size.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_assertion_prop_nolayer"
        " ON assertion(matter_id, proposition_key, created_at)"
    )


def _migration_v34(conn) -> None:
    """Add assertion_revision table for immutable field-level audit log (Q4 HIGH, SO-2).

    Tracks every mutation to assertion.belief_state, confidence, and proposition_text
    with old/new JSON values, actor kind, cause, and batch grouping. Additive only —
    no existing table is modified. Existing DBs receive the new table and index; no
    backfill is possible for pre-v34 mutations since the old values were not recorded.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS assertion_revision (
            id              TEXT PRIMARY KEY,
            batch_id        TEXT NOT NULL,
            assertion_id    TEXT NOT NULL REFERENCES assertion(id),
            changed_field   TEXT NOT NULL,
            old_value_json  TEXT,
            new_value_json  TEXT,
            actor_kind      TEXT NOT NULL,
            actor_ref       TEXT,
            cause           TEXT NOT NULL,
            run_id          TEXT,
            note            TEXT,
            created_at      TEXT NOT NULL
        ) STRICT"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_assertion_revision_assertion"
        " ON assertion_revision(assertion_id, created_at DESC)"
    )


def _migration_v33(conn) -> None:
    """Add covering index on assertion_issue_link for the SO-4 weighted coverage queries.

    Both get_issue_coverage_report() and the _investigate_context() weakest-issue selector
    join assertion_issue_link by (issue_id, relation_type) and then access assertion_id.

    The existing ix_issue_assertions(issue_id, relation_type) does not include assertion_id,
    so SQLite must return to the heap for each matched link to retrieve assertion_id before
    doing the PK lookup on assertion. At scale (10k+ links per matter), this is one heap
    fetch per link.

    A covering (issue_id, relation_type, assertion_id) index eliminates the heap fetch
    for the join step: the assertion_id value is read directly from the index leaf, and
    the subsequent PK lookup on assertion(id) is the only remaining heap access.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_ail_issue_rel_assertion"
        " ON assertion_issue_link(issue_id, relation_type, assertion_id)"
    )


def _migration_v35(conn) -> None:  # noqa: ARG001
    """Reserved no-op: v35 was previously issued with a duplicate index (ix_assertion_prop_key
    identical to v32's ix_assertion_prop_nolayer) and reverted. The version slot is kept here
    as a no-op so that any database that was touched by that short-lived build (schema_version=35)
    remains valid and future real migrations start at v36.
    """


def _migration_v37(conn) -> None:
    """Add answered_at index on clarification_question for get_answered(limit=N) perf.

    get_answered() orders by answered_at DESC; without this index SQLite must
    scan and sort the full answered set before applying LIMIT.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_clarification_matter_answered"
        " ON clarification_question(matter_id, answered_at DESC)"
    )


def _migration_v36(conn) -> None:
    """Add SO-1 real reuse telemetry columns to run_session.

    llm_calls_avoided: number of LLM calls skipped because durable state was reused
        (orientation cache hit, search-analysis cache hit, inventory hot-path skip,
        synthesis cache hit).
    llm_calls_required: number of LLM calls that could not be avoided (cache misses,
        cold runs, doc reads not already ingested).

    True reuse rate = llm_calls_avoided / (llm_calls_avoided + llm_calls_required).
    This replaces the proxy metric (assertions_at_start / assertions_at_end or
    documents_from_cache / documents_read) with a direct measurement of LLM avoidance.
    """
    conn.execute("ALTER TABLE run_session ADD COLUMN llm_calls_avoided INTEGER")
    conn.execute("ALTER TABLE run_session ADD COLUMN llm_calls_required INTEGER")


def _migration_v38(conn) -> None:
    """Add partial covering index for the supersession user-lock query (Tier 2 r3 LOW).

    The lock query in _revise_one() filters assertion_revision on:
      assertion_id=? AND changed_field='belief_state' AND new_value_json=?
    ORDER BY created_at DESC LIMIT 1

    The existing ix_assertion_revision_assertion(assertion_id, created_at DESC) only
    supports the assertion_id seek + sort; changed_field and new_value_json are post-
    filters. Under high correction volume on a single assertion, this degrades to
    scanning all that assertion's revision history rows.

    The partial index (WHERE changed_field='belief_state') covers only belief_state rows,
    reducing index size. Combined with (assertion_id, new_value_json, created_at DESC),
    the lock query becomes a 2-column seek + LIMIT 1 scan — O(log N) regardless of
    how many lock rows exist for a given assertion.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_assertion_revision_lock"
        " ON assertion_revision(assertion_id, new_value_json, created_at DESC)"
        " WHERE changed_field = 'belief_state'"
    )


def _migration_v39(conn) -> None:
    """Add actor_kind to ix_assertion_revision_lock to make it a covering index.

    The lock query in _revise_one() selects actor_kind:
      SELECT actor_kind FROM assertion_revision
      WHERE assertion_id=? AND changed_field='belief_state' AND new_value_json=?
      ORDER BY created_at DESC LIMIT 1

    The v38 index (assertion_id, new_value_json, created_at DESC) does not include
    actor_kind, forcing one heap fetch per LIMIT 1 probe.  Adding actor_kind after
    created_at DESC makes the index covering: the seek+sort returns actor_kind
    directly without a table lookup.
    """
    # Wrap DROP+CREATE atomically using a SAVEPOINT so this works in both autocommit
    # mode (isolation_level=None) and when apply_schema() is called inside an outer
    # transaction.  SQLite cannot nest BEGIN, but SAVEPOINT/RELEASE is always safe.
    conn.execute("SAVEPOINT ix_rebuild_v39")
    try:
        conn.execute("DROP INDEX IF EXISTS ix_assertion_revision_lock")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_assertion_revision_lock"
            " ON assertion_revision(assertion_id, new_value_json, created_at DESC, actor_kind)"
            " WHERE changed_field = 'belief_state'"
        )
        conn.execute("RELEASE ix_rebuild_v39")
    except Exception:
        conn.execute("ROLLBACK TO ix_rebuild_v39")
        conn.execute("RELEASE ix_rebuild_v39")
        raise


def _migration_v42(conn) -> None:
    """Add cold-path maintenance columns + document_actor_role table.

    Priority 1: split cold-path into query-agnostic profiling + issue-specific
    extraction. document_inventory gets maintenance_status/profiled_at/last_maintained_at
    to track profiling state. document_card gets source_role + signatories_json.
    document_actor_role links actors to documents with role types.
    """
    import sqlite3 as _sqlite3
    # document_inventory: maintenance columns
    for col_def in [
        "ALTER TABLE document_inventory ADD COLUMN maintenance_status TEXT NOT NULL DEFAULT 'pending'",
        "ALTER TABLE document_inventory ADD COLUMN profiled_at TEXT",
        "ALTER TABLE document_inventory ADD COLUMN last_maintained_at TEXT",
    ]:
        try:
            conn.execute(col_def)
        except _sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    # document_card: source_role + signatories
    for col_def in [
        "ALTER TABLE document_card ADD COLUMN source_role TEXT",
        "ALTER TABLE document_card ADD COLUMN signatories_json TEXT",
    ]:
        try:
            conn.execute(col_def)
        except _sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_card_source_role ON document_card(source_role)"
    )
    # document_actor_role table
    conn.execute(
        """CREATE TABLE IF NOT EXISTS document_actor_role (
               id          TEXT PRIMARY KEY,
               doc_id      TEXT NOT NULL REFERENCES document_inventory(id),
               actor_id    TEXT NOT NULL REFERENCES actor(id),
               role_type   TEXT NOT NULL,
               raw_name    TEXT,
               confidence  REAL NOT NULL DEFAULT 1.0,
               created_at  TEXT NOT NULL
           ) STRICT"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_doc_actor_role_doc"
        " ON document_actor_role(doc_id, role_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_doc_actor_role_actor"
        " ON document_actor_role(actor_id, role_type)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_doc_actor_role"
        " ON document_actor_role(doc_id, actor_id, role_type)"
    )


def _migration_v41(conn) -> None:
    """Add resumed_from to run_session for resume lineage tracking (r84 HIGH).

    _resolve_active_run_id() previously used a count heuristic (exactly one
    running non-utility run in the matter) which could target an unrelated run.
    This column stores the original interrupted run_id that a resume was started
    from, enabling an exact lineage query instead of a count assumption.

    Idempotent: fresh DBs already have the column in the CREATE TABLE DDL;
    SQLite does not support ADD COLUMN IF NOT EXISTS so we swallow the duplicate.
    """
    import sqlite3 as _sqlite3
    try:
        conn.execute("ALTER TABLE run_session ADD COLUMN resumed_from TEXT")
    except _sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def _migration_v40(conn) -> None:
    """Add pending_propagation table for durable correction/evidence queue persistence.

    In-memory _correction_pending and _evidence_pending dicts are lost on process
    restart, stranding partial BFS propagation with no recovery path (adv#029 SO-1 HIGH).
    This table persists each enqueued assertion so MatterModel.__init__ can reconstruct
    both queues on open, enabling crash-safe deferred belief revision replay.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS pending_propagation (
               id           TEXT PRIMARY KEY,
               matter_id    TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
               assertion_id TEXT NOT NULL,
               cause        TEXT NOT NULL,
               orig_run_id  TEXT,
               queue        TEXT NOT NULL,
               enqueued_at  TEXT NOT NULL,
               UNIQUE(matter_id, assertion_id, queue)
           ) STRICT"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_pending_propagation_matter"
        " ON pending_propagation(matter_id, queue)"
    )


def _migration_v43(conn) -> None:
    """Phase 1 correctness floor: authority ranking + run session operation typing.

    1. Add precedential_rank to authority for numeric ordering (replaces lexical weight sort).
    2. Add operation_type and trigger to run_session so flush/lint/ingest sessions
       are distinguishable from user query runs.
    """
    for stmt in [
        "ALTER TABLE authority ADD COLUMN precedential_rank INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE run_session ADD COLUMN operation_type TEXT NOT NULL DEFAULT 'query'",
        "ALTER TABLE run_session ADD COLUMN trigger TEXT NOT NULL DEFAULT 'user'",
    ]:
        _execute_allow_duplicate_column(conn, stmt)
    conn.commit()


def _migration_v44(conn) -> None:
    """Phase 2 claim identity v2: add identity columns to assertion and occurrence tables.

    1. Assertion table: claim_key, identity_version, polarity, canonical_subject_key,
       canonical_object_key, temporal_identity_key, speaker_scope_key, canonicalization_confidence.
    2. Assertion_occurrence table: document_inventory_id, raw_text, SPO parse fields,
       polarity, speaker_scope_key, claim_key_candidate, resolution_strategy, extraction_confidence.
    3. Replace ux_assertion_prop unique index with legacy covering index + claim_key unique index.
    4. Widen ix_occurrence_unique_doc to prefer document_inventory_id over document_id.
    5. Backfill document_inventory_id and occurrence parse fields from parent assertions.
    6. Python-side claim identity backfill: resolve claim_key for every existing occurrence
       and split assertions whose occurrences disagree on claim_key.
    """
    from datetime import datetime, timezone
    from .models import AssertionCandidate, ClaimIdentity
    from .enums import ModelLayer, AssertionKind, SpeechAct, SourceRole, OriginKind

    # --- Step 1: ALTER TABLE assertion ---
    assertion_alters = [
        "ALTER TABLE assertion ADD COLUMN claim_key TEXT",
        "ALTER TABLE assertion ADD COLUMN identity_version TEXT NOT NULL DEFAULT 'legacy_text_v1'",
        "ALTER TABLE assertion ADD COLUMN polarity TEXT NOT NULL DEFAULT 'affirmed'",
        "ALTER TABLE assertion ADD COLUMN canonical_subject_key TEXT",
        "ALTER TABLE assertion ADD COLUMN canonical_object_key TEXT",
        "ALTER TABLE assertion ADD COLUMN temporal_identity_key TEXT NOT NULL DEFAULT 'atemporal'",
        "ALTER TABLE assertion ADD COLUMN speaker_scope_key TEXT",
        "ALTER TABLE assertion ADD COLUMN canonicalization_confidence REAL NOT NULL DEFAULT 0.0",
    ]
    for stmt in assertion_alters:
        _execute_allow_duplicate_column(conn, stmt)

    # --- Step 2: ALTER TABLE assertion_occurrence ---
    occurrence_alters = [
        "ALTER TABLE assertion_occurrence ADD COLUMN document_inventory_id TEXT REFERENCES document_inventory(id)",
        "ALTER TABLE assertion_occurrence ADD COLUMN raw_text TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN subject_ref_type TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN subject_ref_id TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN predicate_key TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN object_json TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN temporal_scope_start TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN temporal_scope_end TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN polarity TEXT NOT NULL DEFAULT 'affirmed'",
        "ALTER TABLE assertion_occurrence ADD COLUMN speaker_scope_key TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN claim_key_candidate TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN resolution_strategy TEXT",
        "ALTER TABLE assertion_occurrence ADD COLUMN extraction_confidence REAL NOT NULL DEFAULT 0.0",
    ]
    for stmt in occurrence_alters:
        _execute_allow_duplicate_column(conn, stmt)

    # --- Step 3: Replace ux_assertion_prop with legacy index + claim_key unique ---
    conn.execute("DROP INDEX IF EXISTS ux_assertion_prop")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_assertion_prop_legacy"
        " ON assertion(matter_id, model_layer, proposition_key,"
        " canonicalization_confidence DESC, created_at ASC)"
    )

    # --- Step 4: Widen occurrence unique index ---
    conn.execute("DROP INDEX IF EXISTS ix_occurrence_unique_doc")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_occurrence_unique_doc"
        " ON assertion_occurrence("
        "     assertion_id,"
        "     COALESCE(document_inventory_id, document_id),"
        "     speech_act,"
        "     COALESCE(span_id, '')"
        " )"
    )

    # New occurrence indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_occurrence_doc_ref"
        " ON assertion_occurrence(document_inventory_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_occurrence_claim_candidate"
        " ON assertion_occurrence(claim_key_candidate)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_assertion_identity_lookup"
        " ON assertion("
        "     matter_id,"
        "     subject_ref_type,"
        "     subject_ref_id,"
        "     predicate_key,"
        "     canonical_object_key,"
        "     polarity,"
        "     temporal_identity_key"
        " )"
    )

    # --- Step 5: Backfill document_inventory_id from relative_path ---
    conn.execute("""
        UPDATE assertion_occurrence
        SET document_inventory_id = (
            SELECT di.id
            FROM assertion a
            JOIN document_inventory di
              ON di.matter_id = a.matter_id
             AND REPLACE(di.relative_path, char(92), '/') = REPLACE(assertion_occurrence.document_id, char(92), '/')
            WHERE a.id = assertion_occurrence.assertion_id
            LIMIT 1
        )
        WHERE document_inventory_id IS NULL
    """)

    # Backfill occurrence parse fields from parent assertion
    conn.execute("""
        UPDATE assertion_occurrence
        SET raw_text = COALESCE(raw_text, (SELECT proposition_text FROM assertion WHERE assertion.id = assertion_occurrence.assertion_id)),
            subject_ref_type = COALESCE(subject_ref_type, (SELECT subject_ref_type FROM assertion WHERE assertion.id = assertion_occurrence.assertion_id)),
            subject_ref_id = COALESCE(subject_ref_id, (SELECT subject_ref_id FROM assertion WHERE assertion.id = assertion_occurrence.assertion_id)),
            predicate_key = COALESCE(predicate_key, (SELECT predicate_key FROM assertion WHERE assertion.id = assertion_occurrence.assertion_id)),
            object_json = COALESCE(object_json, (SELECT object_json FROM assertion WHERE assertion.id = assertion_occurrence.assertion_id)),
            temporal_scope_start = COALESCE(temporal_scope_start, (SELECT temporal_scope_start FROM assertion WHERE assertion.id = assertion_occurrence.assertion_id)),
            temporal_scope_end = COALESCE(temporal_scope_end, (SELECT temporal_scope_end FROM assertion WHERE assertion.id = assertion_occurrence.assertion_id)),
            polarity = COALESCE(polarity, 'affirmed')
    """)

    conn.commit()

    # --- Step 6: Python-side claim identity backfill ---
    import uuid
    import logging
    _log = logging.getLogger(__name__)

    rows = conn.execute("""
        SELECT ao.id AS occ_id, ao.assertion_id,
               ao.document_id, ao.speech_act, ao.speaker_actor_id,
               ao.source_role, ao.source_side, ao.origin_kind,
               COALESCE(ao.subject_ref_type, a.subject_ref_type) AS subject_ref_type,
               COALESCE(ao.subject_ref_id, a.subject_ref_id) AS subject_ref_id,
               COALESCE(ao.predicate_key, a.predicate_key) AS predicate_key,
               COALESCE(ao.object_json, a.object_json) AS object_json,
               COALESCE(ao.temporal_scope_start, a.temporal_scope_start) AS temporal_scope_start,
               COALESCE(ao.temporal_scope_end, a.temporal_scope_end) AS temporal_scope_end,
               COALESCE(ao.raw_text, a.proposition_text) AS raw_text,
               a.proposition_text, a.model_layer, a.assertion_kind
        FROM assertion_occurrence ao
        JOIN assertion a ON a.id = ao.assertion_id
    """).fetchall()

    # Resolve claim identity for each occurrence.
    # The previous implementation wrapped this loop in a broad
    # except-Exception-as-log-warn block and continued past failures. Per PR.1
    # schema discipline, migration failures must surface and abort so the
    # schema_migration ledger cannot record a partial backfill as applied.
    # If a single occurrence row is malformed, the entire migration aborts and
    # can be fixed by data repair before re-running.
    occ_identities = {}  # occ_id -> ClaimIdentity
    for r in rows:
        cand = AssertionCandidate(
            proposition_text=r["proposition_text"],
            model_layer=ModelLayer(r["model_layer"]),
            assertion_kind=AssertionKind(r["assertion_kind"]),
            document_id=r["document_id"] or "",
            raw_text=r["raw_text"],
            speaker_actor_id=r["speaker_actor_id"],
            source_role=SourceRole(r["source_role"]) if r["source_role"] else SourceRole.UNKNOWN,
            source_side=r["source_side"],
            speech_act=SpeechAct(r["speech_act"]) if r["speech_act"] else SpeechAct.EXTRACTED,
            origin_kind=OriginKind(r["origin_kind"]) if r["origin_kind"] else OriginKind.EXTRACTED,
            subject_ref_type=r["subject_ref_type"],
            subject_ref_id=r["subject_ref_id"],
            predicate_key=r["predicate_key"],
            object_json=r["object_json"],
            temporal_scope_start=r["temporal_scope_start"],
            temporal_scope_end=r["temporal_scope_end"],
        )
        identity = cand.resolve_claim_identity()
        occ_identities[r["occ_id"]] = identity

        # Update occurrence with resolved identity
        conn.execute(
            """UPDATE assertion_occurrence
               SET claim_key_candidate=?, resolution_strategy=?,
                   speaker_scope_key=COALESCE(speaker_scope_key, ?),
                   extraction_confidence=?
               WHERE id=?""",
            (
                identity.claim_key,
                identity.resolution_strategy,
                identity.speaker_scope_key,
                identity.canonicalization_confidence,
                r["occ_id"],
            ),
        )

    conn.commit()

    # Group occurrences by assertion_id
    from collections import defaultdict
    assertion_groups = defaultdict(list)  # assertion_id -> [(occ_id, ClaimIdentity)]
    for r in rows:
        occ_id = r["occ_id"]
        if occ_id in occ_identities:
            assertion_groups[r["assertion_id"]].append((occ_id, occ_identities[occ_id]))

    now = datetime.now(timezone.utc).isoformat()

    for assertion_id, occ_list in assertion_groups.items():
        # Group by claim_key
        by_claim = defaultdict(list)
        for occ_id, identity in occ_list:
            by_claim[identity.claim_key].append((occ_id, identity))

        if len(by_claim) <= 1:
            # All occurrences agree — update assertion in place
            if occ_list:
                _, identity = occ_list[0]
                conn.execute(
                    """UPDATE assertion
                       SET claim_key=?, identity_version='claim_v2',
                           polarity=?, canonical_subject_key=?,
                           canonical_object_key=?,
                           temporal_identity_key=?, speaker_scope_key=?,
                           canonicalization_confidence=?, updated_at=?
                       WHERE id=?""",
                    (
                        identity.claim_key, identity.polarity,
                        identity.canonical_subject_key,
                        identity.canonical_object_key,
                        identity.temporal_identity_key,
                        identity.speaker_scope_key,
                        identity.canonicalization_confidence,
                        now, assertion_id,
                    ),
                )
        else:
            # Multiple claim keys — keep largest group on existing row, split rest
            sorted_groups = sorted(by_claim.items(), key=lambda x: -len(x[1]))
            # Largest group stays on the original assertion
            keep_key, keep_occs = sorted_groups[0]
            _, keep_identity = keep_occs[0]
            conn.execute(
                """UPDATE assertion
                   SET claim_key=?, identity_version='claim_v2',
                       polarity=?, canonical_subject_key=?,
                       canonical_object_key=?,
                       temporal_identity_key=?, speaker_scope_key=?,
                       canonicalization_confidence=?, updated_at=?
                   WHERE id=?""",
                (
                    keep_identity.claim_key, keep_identity.polarity,
                    keep_identity.canonical_subject_key,
                    keep_identity.canonical_object_key,
                    keep_identity.temporal_identity_key,
                    keep_identity.speaker_scope_key,
                    keep_identity.canonicalization_confidence,
                    now, assertion_id,
                ),
            )

            # Split remaining groups into new assertion rows
            for split_key, split_occs in sorted_groups[1:]:
                new_id = uuid.uuid4().hex
                _, split_identity = split_occs[0]
                conn.execute(
                    """INSERT INTO assertion (
                           id, matter_id, proposition_key, claim_key, identity_version,
                           proposition_text, model_layer, assertion_kind, polarity,
                           canonical_subject_key, subject_ref_type, subject_ref_id,
                           predicate_key, canonical_object_key, object_json,
                           temporal_scope_start, temporal_scope_end,
                           temporal_identity_key, speaker_scope_key,
                           canonicalization_confidence,
                           belief_state, confidence, created_at, updated_at
                       )
                       SELECT
                           ?, matter_id, proposition_key, ?, 'claim_v2',
                           proposition_text, model_layer, assertion_kind, ?,
                           ?, subject_ref_type, subject_ref_id,
                           predicate_key, ?, object_json,
                           temporal_scope_start, temporal_scope_end,
                           ?, ?,
                           ?,
                           belief_state, confidence, created_at, ?
                       FROM assertion WHERE id=?""",
                    (
                        new_id, split_identity.claim_key,
                        split_identity.polarity,
                        split_identity.canonical_subject_key,
                        split_identity.canonical_object_key,
                        split_identity.temporal_identity_key,
                        split_identity.speaker_scope_key,
                        split_identity.canonicalization_confidence,
                        now, assertion_id,
                    ),
                )

                # Move occurrences to the new assertion
                split_occ_ids = [oid for oid, _ in split_occs]
                placeholders = ",".join("?" for _ in split_occ_ids)
                conn.execute(
                    f"UPDATE assertion_occurrence SET assertion_id=? WHERE id IN ({placeholders})",
                    [new_id] + split_occ_ids,
                )

                # Clone issue links conservatively
                conn.execute(
                    """INSERT OR IGNORE INTO assertion_issue_link
                       (id, assertion_id, issue_id, relation_type, created_at)
                       SELECT lower(hex(randomblob(16))), ?, issue_id, relation_type, created_at
                       FROM assertion_issue_link
                       WHERE assertion_id=?""",
                    (new_id, assertion_id),
                )

                _log.info(
                    "v44 split assertion=%s new=%s claim_key=%s occs=%d",
                    assertion_id, new_id, split_key, len(split_occs),
                )

    conn.commit()

    # Create the claim_key unique index after backfill
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_assertion_claim_key"
        " ON assertion(matter_id, model_layer, claim_key)"
        " WHERE claim_key IS NOT NULL"
    )
    conn.commit()


def _migration_v45(conn) -> None:
    """Phase 3: evidence_edge table + proof_state columns for unified proof metric.

    evidence_edge: provenance-rich evidence links between any two objects
    (assertion→issue, work_product→issue, etc.) with source confidence,
    admissibility tracking, and vulnerability notes.

    proof_state additions:
    - support_score / attack_score: raw weighted sums (pre-formula) so
      consumers can inspect components without re-deriving from edges.
    - coverage_version: tracks which formula version produced the snapshot
      so stale rows can be detected after formula changes.
    """
    # --- evidence_edge table (DDL already in _DDL_EVIDENCE for fresh DBs).
    # Every statement in _DDL_EVIDENCE already uses CREATE TABLE/INDEX IF NOT
    # EXISTS, so any error surfaced here is a real migration defect, not a
    # benign "already exists" case. Broad swallow removed.
    for stmt in _DDL_EVIDENCE.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)

    # --- proof_state new columns ---
    for alter in (
        "ALTER TABLE proof_state ADD COLUMN support_score REAL NOT NULL DEFAULT 0.0",
        "ALTER TABLE proof_state ADD COLUMN attack_score REAL NOT NULL DEFAULT 0.0",
        "ALTER TABLE proof_state ADD COLUMN coverage_version TEXT NOT NULL DEFAULT 'proof_v2'",
    ):
        _execute_allow_duplicate_column(conn, alter)

    conn.commit()


def _migration_v46(conn) -> None:
    """Add durable LLM usage tracking.

    1. llm_call: one row per Gemini API request with tokens, cost, latency, and outcome.
    2. run_session aggregates: cheap totals for UI/status queries.
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS llm_call (
               id                  TEXT PRIMARY KEY,
               matter_id           TEXT NOT NULL REFERENCES matter(id),
               run_id              TEXT REFERENCES run_session(id),
               model_tier          TEXT NOT NULL,
               model_id            TEXT NOT NULL,
               usage_label         TEXT,
               input_tokens        INTEGER NOT NULL DEFAULT 0,
               cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
               output_tokens       INTEGER NOT NULL DEFAULT 0,
               total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
               estimated_cost_usd  REAL NOT NULL DEFAULT 0.0,
               latency_ms          INTEGER NOT NULL DEFAULT 0,
               success             INTEGER NOT NULL DEFAULT 1,
               error_kind          TEXT,
               created_at          TEXT NOT NULL
           ) STRICT"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_llm_call_matter"
        " ON llm_call(matter_id, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_llm_call_run"
        " ON llm_call(run_id, created_at DESC)"
    )
    for alter in (
        "ALTER TABLE run_session ADD COLUMN llm_input_tokens INTEGER",
        "ALTER TABLE run_session ADD COLUMN llm_cache_read_tokens INTEGER",
        "ALTER TABLE run_session ADD COLUMN llm_output_tokens INTEGER",
        "ALTER TABLE run_session ADD COLUMN llm_request_count INTEGER",
        "ALTER TABLE run_session ADD COLUMN llm_estimated_cost_usd REAL",
    ):
        _execute_allow_duplicate_column(conn, alter)
    conn.commit()


def _migration_v47(conn) -> None:
    """Add research_mode to run_session for explicit budget/audit tracking."""
    _execute_allow_duplicate_column(
        conn,
        "ALTER TABLE run_session ADD COLUMN research_mode TEXT NOT NULL DEFAULT 'deep'",
    )
    conn.commit()


def _migration_v48(conn) -> None:
    """Add date_precision column to quant_fact for timeline display fidelity."""
    _execute_allow_duplicate_column(
        conn, "ALTER TABLE quant_fact ADD COLUMN date_precision TEXT"
    )
    conn.commit()


def _migration_v49(conn) -> None:
    """Introduce schema_migration ledger and migration_backfill_job queue.

    The schema_migration table is the canonical migration ledger going forward;
    the older schema_version table is preserved as a compatibility ledger. Any
    rows in schema_version are backfilled into schema_migration as legacy_v<N>.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration (
            version            INTEGER PRIMARY KEY,
            name               TEXT NOT NULL,
            checksum           TEXT NOT NULL DEFAULT '',
            applied_at         TEXT NOT NULL,
            app_schema_version INTEGER NOT NULL,
            app_build          TEXT,
            duration_ms        INTEGER NOT NULL DEFAULT 0
        ) STRICT
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migration
            (version, name, checksum, applied_at, app_schema_version, app_build, duration_ms)
        SELECT
            version,
            'legacy_v' || version,
            'legacy',
            applied_at,
            version,
            'legacy_python_runner',
            0
        FROM schema_version
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS migration_backfill_job (
            id                  TEXT PRIMARY KEY,
            matter_id           TEXT REFERENCES matter(id) ON DELETE CASCADE,
            job_name            TEXT NOT NULL,
            status              TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'complete', 'failed', 'skipped')),
            priority            INTEGER NOT NULL DEFAULT 100,
            total_items         INTEGER NOT NULL DEFAULT 0,
            completed_items     INTEGER NOT NULL DEFAULT 0,
            estimated_tokens    INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd  REAL NOT NULL DEFAULT 0.0,
            error               TEXT,
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            started_at          TEXT,
            completed_at        TEXT,
            UNIQUE(matter_id, job_name)
        ) STRICT
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_backfill_job_status
            ON migration_backfill_job(status, priority, created_at)
        """
    )
    conn.commit()


def _migration_v50(conn) -> None:
    """MVP.2: add verification_state + verification_event tables (SO-2).

    Canonical human-review substrate for all AI-derived matter intelligence.
    target_kind is the frozen VerificationTargetKind vocabulary; status is
    the minimal candidate/verified/rejected/stale state machine.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_state (
            id                TEXT PRIMARY KEY,
            matter_id         TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
            target_kind       TEXT NOT NULL CHECK (target_kind IN (
                'assertion','assertion_occurrence','issue_predicate','evidence_edge',
                'quant_fact','authority','document_card','privilege_classification',
                'gap','dispute','dispute_position','timeline_event','deadline',
                'authority_treatment','actor_relationship','defined_term',
                'causation_edge','theory','artifact','artifact_manifest_item'
            )),
            target_id         TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'candidate'
                CHECK (status IN ('candidate','verified','rejected','stale')),
            ai_confidence     REAL CHECK (ai_confidence IS NULL OR (ai_confidence >= 0.0 AND ai_confidence <= 1.0)),
            reviewed_by_kind  TEXT CHECK (reviewed_by_kind IS NULL OR reviewed_by_kind IN ('user','attorney','system','import')),
            reviewed_by_id    TEXT,
            reviewed_at       TEXT,
            review_scope      TEXT NOT NULL DEFAULT 'extraction_correct'
                CHECK (review_scope IN (
                    'extraction_correct','record_truth','inference','legal_conclusion',
                    'truth_override','internal_privileged','clean_output',
                    'privilege_classification','dispute_resolution','artifact_policy'
                )),
            review_scope_json TEXT,
            review_note       TEXT,
            rejection_reason  TEXT,
            stale_reason      TEXT,
            version           INTEGER NOT NULL DEFAULT 1,
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(matter_id, target_kind, target_id)
        ) STRICT
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_verification_status"
        " ON verification_state(matter_id, status, target_kind, updated_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_verification_target"
        " ON verification_state(target_kind, target_id)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verification_event (
            id               TEXT PRIMARY KEY,
            matter_id        TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
            verification_id  TEXT REFERENCES verification_state(id) ON DELETE SET NULL,
            target_kind      TEXT NOT NULL,
            target_id        TEXT NOT NULL,
            old_status       TEXT,
            new_status       TEXT NOT NULL CHECK (new_status IN ('candidate','verified','rejected','stale')),
            reviewed_by_kind TEXT NOT NULL CHECK (reviewed_by_kind IN ('user','attorney','system','import')),
            reviewed_by_id   TEXT,
            review_scope     TEXT NOT NULL,
            rejection_reason TEXT,
            run_id           TEXT REFERENCES run_session(id),
            cause            TEXT NOT NULL,
            note             TEXT,
            old_version      INTEGER,
            new_version      INTEGER,
            created_at       TEXT NOT NULL DEFAULT (datetime('now'))
        ) STRICT
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_verification_event_target"
        " ON verification_event(matter_id, target_kind, target_id, created_at DESC)"
    )
    conn.commit()


def _migration_v51(conn) -> None:
    """MVP.2: seed candidate verification rows for pre-existing intelligence.

    Every existing assertion, assertion_occurrence, issue_predicate,
    quant_fact, authority, and document_card gets a candidate row so
    the unique constraint on (matter_id, target_kind, target_id) is
    populated without auto-upgrading any row to verified.
    """
    conn.execute(
        """INSERT OR IGNORE INTO verification_state
            (id, matter_id, target_kind, target_id, status, ai_confidence,
             created_at, updated_at)
           SELECT lower(hex(randomblob(16))), matter_id, 'assertion', id,
                  'candidate', confidence, datetime('now'), datetime('now')
           FROM assertion"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO verification_state
            (id, matter_id, target_kind, target_id, status, ai_confidence,
             created_at, updated_at)
           SELECT lower(hex(randomblob(16))), a.matter_id, 'assertion_occurrence',
                  ao.id, 'candidate', ao.extraction_confidence,
                  datetime('now'), datetime('now')
           FROM assertion_occurrence ao
           JOIN assertion a ON a.id = ao.assertion_id"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO verification_state
            (id, matter_id, target_kind, target_id, status,
             created_at, updated_at)
           SELECT lower(hex(randomblob(16))), i.matter_id, 'issue_predicate',
                  ip.id, 'candidate', datetime('now'), datetime('now')
           FROM issue_predicate ip
           JOIN issue i ON i.id = ip.issue_id"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO verification_state
            (id, matter_id, target_kind, target_id, status,
             created_at, updated_at)
           SELECT lower(hex(randomblob(16))), matter_id, 'quant_fact', id,
                  'candidate', datetime('now'), datetime('now')
           FROM quant_fact"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO verification_state
            (id, matter_id, target_kind, target_id, status,
             created_at, updated_at)
           SELECT lower(hex(randomblob(16))), matter_id, 'authority', id,
                  'candidate', datetime('now'), datetime('now')
           FROM authority"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO verification_state
            (id, matter_id, target_kind, target_id, status,
             created_at, updated_at)
           SELECT lower(hex(randomblob(16))), di.matter_id, 'document_card',
                  dc.id, 'candidate', datetime('now'), datetime('now')
           FROM document_card dc
           JOIN document_inventory di ON di.id = dc.doc_id"""
    )
    conn.commit()


def _migration_v52(conn) -> None:
    """MVP.3: add evidence-edge columns needed for the proof substrate switch.

    The base evidence_edge table exists since v45 but only carries the
    shape needed for legacy evidence_link parity. MVP.3 adds the columns
    the proof-edge-first substrate requires. All ALTER TABLE ADD COLUMN
    calls go through _execute_allow_duplicate_column so reapplying is safe.
    """
    for alter in (
        "ALTER TABLE evidence_edge ADD COLUMN verification_status TEXT NOT NULL DEFAULT 'candidate'",
        "ALTER TABLE evidence_edge ADD COLUMN independence_factor REAL NOT NULL DEFAULT 1.0",
        "ALTER TABLE evidence_edge ADD COLUMN backfill_source TEXT",
        "ALTER TABLE evidence_edge ADD COLUMN source_identity_status TEXT NOT NULL DEFAULT 'unknown'",
        "ALTER TABLE evidence_edge ADD COLUMN origin_kind TEXT NOT NULL DEFAULT 'legacy_backfill'",
        "ALTER TABLE evidence_edge ADD COLUMN active INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE evidence_edge ADD COLUMN effective_weight REAL",
        "ALTER TABLE evidence_edge ADD COLUMN independence_cluster_id TEXT",
    ):
        _execute_allow_duplicate_column(conn, alter)
    # Fill effective_weight for any pre-existing rows so the MVP.3 proof
    # substrate can read effective_weight without a NULL check.
    conn.execute(
        "UPDATE evidence_edge SET effective_weight = COALESCE(effective_weight, proof_weight)"
    )
    conn.commit()


def _migration_v53(conn) -> None:
    """MVP.3: backfill one evidence_edge per legacy assertion_issue_link row.

    Uses the existing unique key (matter_id, source_kind, source_id,
    target_kind, target_id, relation_type) for idempotent INSERT OR IGNORE.
    Occurrence/span identity is deliberately NULL — current link APIs do
    not preserve it, and MVP.3 AC explicitly flags this as a documented
    limitation.

    Also seeds verification_state candidate rows for every inserted edge,
    matching the MVP.2 substrate contract.
    """
    conn.execute(
        """INSERT OR IGNORE INTO evidence_edge
            (id, matter_id,
             source_kind, source_id,
             source_document_inventory_id, source_span_id, source_occurrence_id,
             target_kind, target_id, relation_type,
             proof_weight, source_confidence, admissibility_status,
             vulnerability_json, note,
             verification_status, independence_factor, backfill_source,
             source_identity_status, origin_kind, active,
             effective_weight, independence_cluster_id,
             created_at, updated_at)
           SELECT
               lower(hex(randomblob(16))), i.matter_id,
               'assertion', ail.assertion_id,
               NULL, NULL, NULL,
               'issue', ail.issue_id, ail.relation_type,
               0.5, COALESCE(a.confidence, 0.5), NULL,
               NULL,
               'Backfilled from assertion_issue_link; occurrence/span identity unavailable in legacy link.',
               'candidate', 1.0, 'assertion_issue_link',
               'missing_occurrence_span', 'legacy_backfill', 1,
               0.5, NULL,
               COALESCE(ail.created_at, datetime('now')), datetime('now')
           FROM assertion_issue_link ail
           JOIN assertion a ON a.id = ail.assertion_id
           JOIN issue i ON i.id = ail.issue_id
           WHERE ail.relation_type IN ('supports','establishes','attacks','negates')"""
    )
    # Seed verification_state candidate rows for every edge that now exists
    # on target_kind='evidence_edge'. INSERT OR IGNORE keeps reruns safe.
    conn.execute(
        """INSERT OR IGNORE INTO verification_state
            (id, matter_id, target_kind, target_id, status,
             review_scope, version, created_at, updated_at)
           SELECT lower(hex(randomblob(16))), matter_id, 'evidence_edge', id,
                  'candidate', 'inference', 1, datetime('now'), datetime('now')
           FROM evidence_edge"""
    )
    conn.commit()


def _migration_v54(conn) -> None:
    """MVP.5: add template/element metadata to issue_predicate.

    Each predicate can trace back to an IssueTemplate + element key so
    the proof substrate can compute per-element sufficiency and surface
    missing elements as gaps. Columns:
    - template_id: reference string to a TemplateRegistry entry
    - template_version: semantic version of the template applied
    - element_key: canonical identifier within the template (e.g.
      'formation', 'damages')
    - element_order: display ordering within the issue
    """
    for alter in (
        "ALTER TABLE issue_predicate ADD COLUMN template_id TEXT",
        "ALTER TABLE issue_predicate ADD COLUMN template_version TEXT",
        "ALTER TABLE issue_predicate ADD COLUMN element_key TEXT",
        "ALTER TABLE issue_predicate ADD COLUMN element_order INTEGER NOT NULL DEFAULT 0",
    ):
        _execute_allow_duplicate_column(conn, alter)
    # Unique index keeps (issue_id, template_id, element_key) idempotent
    # so apply_template() can safely re-run without duplicating predicates.
    # WHERE clause excludes manually-added predicates (template_id is NULL)
    # because they have no template to uniquify against.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_issue_predicate_template "
        "ON issue_predicate(issue_id, template_id, element_key) "
        "WHERE template_id IS NOT NULL AND element_key IS NOT NULL"
    )
    conn.commit()


def _migration_v55(conn) -> None:
    """P0.1 Provenance Lite: add append-only provenance_event table and
    extend llm_call with prompt/response SHA256 hashes (SO-2).

    provenance_event captures every AI-derived object write with:
    - matter_id, target_kind, target_id (polymorphic pointer to the
      intelligence object being attributed)
    - event_kind (e.g. 'assertion_extraction', 'edge_write',
      'quant_record', 'card_profile', 'authority_upsert')
    - writer_name (e.g. 'AssertionStore.upsert_occurrence')
    - run_id, model_id, prompt_version, extractor_version,
      llm_call_id — everything needed to identify WHICH AI call produced
      the row.
    - prompt_hash, response_hash — stable digests for reproducibility.
    - source_document_ref / source_document_inventory_id / source_span_id
      — the primary source; source_span_status explicitly records
      'missing' when span identity is unavailable per P0.1 AC #4.

    Schema is append-only: no updates, no deletes. Callers query by
    (target_kind, target_id) to reconstruct an object's AI provenance.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provenance_event (
            id                           TEXT PRIMARY KEY,
            matter_id                    TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
            target_kind                  TEXT NOT NULL,
            target_id                    TEXT NOT NULL,
            event_kind                   TEXT NOT NULL,
            writer_name                  TEXT NOT NULL,
            run_id                       TEXT REFERENCES run_session(id),
            model_id                     TEXT,
            model_tier                   TEXT,
            prompt_version               TEXT,
            extractor_version            TEXT,
            llm_call_id                  TEXT,
            prompt_hash                  TEXT,
            response_hash                TEXT,
            source_document_ref          TEXT,
            source_document_inventory_id TEXT REFERENCES document_inventory(id),
            source_span_id               TEXT,
            source_span_status           TEXT NOT NULL DEFAULT 'unknown'
                CHECK (source_span_status IN ('present','missing','not_applicable','unknown')),
            note                         TEXT,
            created_at                   TEXT NOT NULL DEFAULT (datetime('now'))
        ) STRICT
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_provenance_target"
        " ON provenance_event(matter_id, target_kind, target_id, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_provenance_llm_call"
        " ON provenance_event(llm_call_id)"
    )
    # Extend llm_call with prompt/response SHA256 hashes so the client
    # can persist the digest that provenance_event references.
    for alter in (
        "ALTER TABLE llm_call ADD COLUMN prompt_hash TEXT",
        "ALTER TABLE llm_call ADD COLUMN response_hash TEXT",
    ):
        _execute_allow_duplicate_column(conn, alter)
    conn.commit()


def _migration_v56(conn) -> None:
    """P0.4 Trust Invalidation Lite: add matter.trust_revision so
    downstream caches can include it in their key. Any invalidation
    trigger (document hash change, human rejection, span replacement,
    privilege reclassification) bumps this counter; a stale cache
    hit cannot reappear because its key no longer matches.

    Legacy rows default to 0. Callers never read trust_revision
    before bumping — the cache bypass is keyed on the bumped value.
    """
    _execute_allow_duplicate_column(
        conn,
        "ALTER TABLE matter ADD COLUMN trust_revision INTEGER NOT NULL DEFAULT 0",
    )
    conn.commit()


def _migration_v57(conn) -> None:
    """P0.5 Content Policy MVI: append-only audit log of every
    content-policy decision the guard emits (SO-5).

    Writers never mutate past rows. One row per decision lets an
    audit trace, for a given target, every time a surface asked
    "may this enter clean output for purpose X?" and what the guard
    returned.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS content_policy_audit (
            id             TEXT PRIMARY KEY,
            matter_id      TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
            purpose        TEXT NOT NULL,
            policy_audience TEXT NOT NULL,
            target_kind    TEXT NOT NULL,
            target_id      TEXT NOT NULL,
            action         TEXT NOT NULL
                CHECK (action IN ('allow','block','withhold')),
            reason_code    TEXT NOT NULL,
            trust_bucket   TEXT NOT NULL,
            privilege_flag INTEGER,
            note           TEXT,
            created_at     TEXT NOT NULL DEFAULT (datetime('now'))
        ) STRICT
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_content_policy_audit_target"
        " ON content_policy_audit(matter_id, target_kind, target_id, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_content_policy_audit_purpose"
        " ON content_policy_audit(matter_id, purpose, created_at DESC)"
    )
    conn.commit()


def _migration_v58(conn) -> None:
    """Persist Gemini thinking/tool token telemetry for analytics surfaces."""
    for alter in (
        "ALTER TABLE llm_call ADD COLUMN tool_use_prompt_tokens INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE llm_call ADD COLUMN thinking_tokens INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE run_session ADD COLUMN llm_tool_use_prompt_tokens INTEGER",
        "ALTER TABLE run_session ADD COLUMN llm_thinking_tokens INTEGER",
        "ALTER TABLE run_session ADD COLUMN llm_total_processed_tokens INTEGER",
    ):
        _execute_allow_duplicate_column(conn, alter)
    conn.commit()


def _migration_v59(conn) -> None:
    """Add memory-broker substrate tables for freshness, taint, and profiles."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS namespace_revision (
            id           TEXT PRIMARY KEY,
            matter_id    TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
            namespace    TEXT NOT NULL,
            target_kind  TEXT NOT NULL,
            target_id    TEXT NOT NULL,
            revision     INTEGER NOT NULL DEFAULT 0,
            updated_at   TEXT NOT NULL,
            UNIQUE(matter_id, namespace, target_kind, target_id)
        ) STRICT"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_namespace_revision_lookup"
        " ON namespace_revision(matter_id, namespace, target_kind, target_id)"
    )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS object_taint (
            id                  TEXT PRIMARY KEY,
            matter_id           TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
            target_kind         TEXT NOT NULL,
            target_id           TEXT NOT NULL,
            taint_class         TEXT NOT NULL,
            domain_profile_id   TEXT NOT NULL DEFAULT '',
            domain_profile_version INTEGER NOT NULL DEFAULT 0,
            profile_mapping_hash TEXT NOT NULL DEFAULT '',
            source_packet_id    TEXT NOT NULL DEFAULT '',
            provenance_event_id TEXT NOT NULL DEFAULT '',
            policy_decision_id  TEXT,
            derivation_reason   TEXT,
            created_at          TEXT NOT NULL,
            UNIQUE(
                matter_id, target_kind, target_id, taint_class,
                domain_profile_id, domain_profile_version, profile_mapping_hash,
                source_packet_id, provenance_event_id
            )
        ) STRICT"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_object_taint_target"
        " ON object_taint(matter_id, target_kind, target_id, taint_class)"
    )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS domain_profile (
            id                  TEXT PRIMARY KEY,
            matter_id           TEXT REFERENCES matter(id) ON DELETE CASCADE,
            profile_id          TEXT NOT NULL,
            profile_version     INTEGER NOT NULL,
            profile_kind        TEXT NOT NULL,
            profile_json        TEXT NOT NULL,
            mapping_hash        TEXT NOT NULL,
            status              TEXT NOT NULL DEFAULT 'current',
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL,
            UNIQUE(matter_id, profile_id, profile_version)
        ) STRICT"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_domain_profile_current"
        " ON domain_profile(matter_id, profile_id, status)"
    )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS profile_mapping (
            id                              TEXT PRIMARY KEY,
            matter_id                       TEXT REFERENCES matter(id) ON DELETE CASCADE,
            source_domain_profile_id        TEXT NOT NULL,
            source_domain_profile_version   INTEGER NOT NULL,
            target_domain_profile_id        TEXT NOT NULL,
            target_domain_profile_version   INTEGER NOT NULL,
            source_mapping_hash             TEXT NOT NULL,
            target_mapping_hash             TEXT NOT NULL,
            target_kind                     TEXT NOT NULL,
            target_namespace                TEXT NOT NULL,
            compatibility_status            TEXT NOT NULL,
            required_transform_id           TEXT,
            reviewer_id                     TEXT,
            created_at                      TEXT NOT NULL,
            UNIQUE(
                matter_id,
                source_domain_profile_id,
                source_domain_profile_version,
                target_domain_profile_id,
                target_domain_profile_version,
                target_kind,
                target_namespace
            )
        ) STRICT"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_profile_mapping_lookup"
        " ON profile_mapping(matter_id, target_domain_profile_id, target_kind, target_namespace)"
    )
    conn.commit()


def _migration_v60(conn) -> None:
    """Bind object taint rows to domain-profile and profile-mapping state."""
    for alter in (
        "ALTER TABLE object_taint ADD COLUMN domain_profile_id TEXT",
        "ALTER TABLE object_taint ADD COLUMN domain_profile_version INTEGER",
        "ALTER TABLE object_taint ADD COLUMN profile_mapping_hash TEXT",
    ):
        _execute_allow_duplicate_column(conn, alter)
    conn.commit()


def _migration_v61(conn) -> None:
    """Scope object_taint uniqueness by domain profile and mapping hash."""
    conn.execute("DROP INDEX IF EXISTS ix_object_taint_target")
    conn.execute("ALTER TABLE object_taint RENAME TO object_taint_old")
    conn.execute(
        """CREATE TABLE object_taint (
            id                  TEXT PRIMARY KEY,
            matter_id           TEXT NOT NULL REFERENCES matter(id) ON DELETE CASCADE,
            target_kind         TEXT NOT NULL,
            target_id           TEXT NOT NULL,
            taint_class         TEXT NOT NULL,
            domain_profile_id   TEXT NOT NULL DEFAULT '',
            domain_profile_version INTEGER NOT NULL DEFAULT 0,
            profile_mapping_hash TEXT NOT NULL DEFAULT '',
            source_packet_id    TEXT NOT NULL DEFAULT '',
            provenance_event_id TEXT NOT NULL DEFAULT '',
            policy_decision_id  TEXT,
            derivation_reason   TEXT,
            created_at          TEXT NOT NULL,
            UNIQUE(
                matter_id, target_kind, target_id, taint_class,
                domain_profile_id, domain_profile_version, profile_mapping_hash,
                source_packet_id, provenance_event_id
            )
        ) STRICT"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO object_taint (
               id, matter_id, target_kind, target_id, taint_class,
               domain_profile_id, domain_profile_version, profile_mapping_hash,
               source_packet_id, provenance_event_id, policy_decision_id,
               derivation_reason, created_at
           )
           SELECT
               id, matter_id, target_kind, target_id, taint_class,
               COALESCE(domain_profile_id, ''),
               COALESCE(domain_profile_version, 0),
               COALESCE(profile_mapping_hash, ''),
               COALESCE(source_packet_id, ''),
               COALESCE(provenance_event_id, ''),
               policy_decision_id,
               derivation_reason,
               created_at
           FROM object_taint_old"""
    )
    conn.execute("DROP TABLE object_taint_old")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_object_taint_target"
        " ON object_taint(matter_id, target_kind, target_id, taint_class)"
    )
    conn.commit()


# Ordered migrations: (target_version, callable).
# Each migration brings the DB from (target_version - 1) to target_version.
# Never remove or reorder entries — append new ones for future changes.
_MIGRATIONS: list[tuple[int, object]] = [
    (1, _migration_v1),
    (2, _migration_v2),
    (3, _migration_v3),
    (4, _migration_v4),
    (5, _migration_v5),
    (6, _migration_v6),
    (7, _migration_v7),
    (8, _migration_v8),
    (9, _migration_v9),
    (10, _migration_v10),
    (11, _migration_v11),
    (12, _migration_v12),
    (13, _migration_v13),
    (14, _migration_v14),
    (15, _migration_v15),
    (16, _migration_v16),
    (17, _migration_v17),
    (18, _migration_v18),
    (19, _migration_v19),
    (20, _migration_v20),
    (21, _migration_v21),
    (22, _migration_v22),
    (23, _migration_v23),
    (24, _migration_v24),
    (25, _migration_v25),
    (26, _migration_v26),
    (27, _migration_v27),
    (28, _migration_v28),
    (29, _migration_v29),
    (30, _migration_v30),
    (31, _migration_v31),
    (32, _migration_v32),
    (33, _migration_v33),
    (34, _migration_v34),
    (35, _migration_v35),
    (36, _migration_v36),
    (37, _migration_v37),
    (38, _migration_v38),
    (39, _migration_v39),
    (40, _migration_v40),
    (41, _migration_v41),
    (42, _migration_v42),
    (43, _migration_v43),
    (44, _migration_v44),
    (45, _migration_v45),
    (46, _migration_v46),
    (47, _migration_v47),
    (48, _migration_v48),
    (49, _migration_v49),
    (50, _migration_v50),
    (51, _migration_v51),
    (52, _migration_v52),
    (53, _migration_v53),
    (54, _migration_v54),
    (55, _migration_v55),
    (56, _migration_v56),
    (57, _migration_v57),
    (58, _migration_v58),
    (59, _migration_v59),
    (60, _migration_v60),
    (61, _migration_v61),
]


def apply_schema(conn) -> None:
    """Run pending migrations to bring the DB to SCHEMA_VERSION.

    Safe to call on both fresh DBs (runs all migrations) and existing DBs
    (skips already-applied migrations). Idempotent.

    Refuses to proceed if the database records any version higher than the
    declared SCHEMA_VERSION — the code would not know how to read a future
    schema, and silently downgrading is worse than failing loudly.
    """
    from datetime import datetime, timezone

    # Step 1: read-only version check. No writes until the guard passes.
    db_version = get_recorded_schema_version(conn)
    if db_version > SCHEMA_VERSION:
        raise SchemaVersionTooNewError(db_version, SCHEMA_VERSION)

    # Step 2: ensure schema_version ledger table exists before any writes.
    for stmt in _DDL_SCHEMA_VERSION.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()

    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current_version: int = row[0] if row and row[0] is not None else 0

    # Step 3: apply migrations up to but never past SCHEMA_VERSION.
    for to_version, migration_fn in _MIGRATIONS:
        if to_version > SCHEMA_VERSION:
            break
        if current_version < to_version:
            migration_fn(conn)  # type: ignore[operator]
            applied_at = datetime.now(timezone.utc).isoformat()
            _record_schema_version(conn, to_version, applied_at)
            _record_schema_migration(
                conn,
                to_version,
                applied_at,
                _MIGRATION_NAMES.get(to_version, f"legacy_v{to_version}"),
            )
            conn.execute(f"PRAGMA user_version = {int(to_version)}")
            conn.commit()
            current_version = to_version

    # Ensure user_version is aligned even on already-current DBs.
    conn.execute(f"PRAGMA user_version = {int(current_version)}")
    conn.commit()
