"""SQLite schema DDL and migration runner for the matter model.

One DB per repository at repository/.irys/matter.sqlite3.
WAL mode, foreign_keys=ON, STRICT tables, JSON1, FTS5.
"""

SCHEMA_VERSION = 26

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
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_matter_repo
    ON matter(repository_root);

CREATE TABLE IF NOT EXISTS assertion (
    id              TEXT PRIMARY KEY,
    matter_id       TEXT NOT NULL REFERENCES matter(id),
    proposition_key TEXT NOT NULL,
    proposition_text TEXT NOT NULL,
    model_layer     TEXT NOT NULL,
    assertion_kind  TEXT NOT NULL,
    subject_ref_type TEXT,
    subject_ref_id   TEXT,
    predicate_key    TEXT,
    object_json      TEXT,
    temporal_scope_start TEXT,
    temporal_scope_end   TEXT,
    belief_state    TEXT NOT NULL DEFAULT 'unknown',
    confidence      REAL NOT NULL DEFAULT 0.5,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_assertion_prop
    ON assertion(matter_id, model_layer, proposition_key);

CREATE INDEX IF NOT EXISTS ix_assertion_subject
    ON assertion(subject_ref_type, subject_ref_id, predicate_key);

CREATE TABLE IF NOT EXISTS assertion_occurrence (
    id              TEXT PRIMARY KEY,
    assertion_id    TEXT NOT NULL REFERENCES assertion(id),
    document_id     TEXT NOT NULL,
    span_id         TEXT,
    speaker_actor_id TEXT,
    source_role     TEXT NOT NULL DEFAULT 'unknown',
    source_side     TEXT,
    speech_act      TEXT NOT NULL,
    origin_kind     TEXT NOT NULL DEFAULT 'extracted',
    created_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_occurrence_assertion
    ON assertion_occurrence(assertion_id, speech_act);

CREATE INDEX IF NOT EXISTS ix_occurrence_assertion_created
    ON assertion_occurrence(assertion_id, created_at);

CREATE INDEX IF NOT EXISTS ix_occurrence_document
    ON assertion_occurrence(document_id, span_id);

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
    ON assertion_occurrence(assertion_id, document_id, speech_act, COALESCE(span_id, ''));

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
    started_at      TEXT NOT NULL,
    completed_at    TEXT
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
    last_read_at    TEXT
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
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_card_type
    ON document_card(doc_type, doc_subtype);

CREATE INDEX IF NOT EXISTS ix_card_operative
    ON document_card(source_side, operative_status, effective_date);
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
        try:
            conn.execute("ALTER TABLE gap ADD COLUMN description_key TEXT")
        except Exception:
            pass  # column already exists

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
        try:
            conn.execute("ALTER TABLE quant_fact ADD COLUMN quant_dedup_key TEXT NOT NULL DEFAULT ''")
        except Exception:
            pass  # column already exists

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
]


def apply_schema(conn) -> None:
    """Run pending migrations to bring the DB to SCHEMA_VERSION.

    Safe to call on both fresh DBs (runs all migrations) and existing DBs
    (skips already-applied migrations). Idempotent.
    """
    from datetime import datetime, timezone

    # Ensure schema_version table exists before reading it.
    for stmt in _DDL_SCHEMA_VERSION.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()

    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current_version: int = row[0] if row and row[0] is not None else 0

    for to_version, migration_fn in _MIGRATIONS:
        if current_version < to_version:
            migration_fn(conn)  # type: ignore[operator]
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (to_version, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            current_version = to_version
