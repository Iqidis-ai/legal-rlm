Recommend approach 1: add an append-only `assertion_revision` table and keep `assertion.id` as the stable graph node. That fits the current architecture in [schema.py:26](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L26), [graph.py:79](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L79), [matter.py:217](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L217), and [belief_revision.py:344](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L344) without forcing a rewrite of the FK graph already hanging off `assertion.id` through occurrences, links, issue links, revision events, and quant facts in [schema.py:51](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L51), [schema.py:92](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L92), [schema.py:107](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L107), [schema.py:344](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L344), and [schema.py:458](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L458).

**Comparison**
1. `assertion_revision` table. DDL is additive: one new append-only table plus one index. `correct_assertion()` stays the canonical entrypoint; it logs field diffs, mutates the canonical row, then lets BFS propagate. Read-path impact is near zero because hydration/synthesis still read `assertion` directly. Main risk is missing a mutation path, so all `UPDATE assertion ...` sites must be centralized behind one audited helper.
2. Bitemporal valid-time rows. A real version of this needs at least `assertion_root_id`, `valid_from`, `valid_to`, and a rebuilt active-row unique index; just adding `valid_from/valid_to` is not enough. It also collides conceptually with the fact-time fields already present on the row (`temporal_scope_start/end`), so it mixes transaction-time audit with legal valid-time semantics. Query impact is high because every current read must resolve “current row” first. Migration from v33 is intrusive and FK-heavy.
3. Append-only assertions with `supersedes`. This sounds clean, but in this codebase it breaks the dedupe invariant and forces relinking or head-resolution across the graph. Existing reasoning already uses `supersedes` as a semantic belief-revision edge in [graph.py:257](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L257) and [belief_revision.py:61](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L61), so using it as the storage-version mechanism overloads two meanings. Query impact is worst because issue/proof queries join link tables to concrete assertion ids and already exclude superseded rows, for example [graph.py:3264](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L3264).

**Recommended Shape**
```sql
CREATE TABLE IF NOT EXISTS assertion_revision (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    assertion_id TEXT NOT NULL REFERENCES assertion(id),
    changed_field TEXT NOT NULL,
    old_value_json TEXT,
    new_value_json TEXT,
    actor_kind TEXT NOT NULL,   -- user | system
    actor_ref TEXT,             -- nullable until caller identity exists
    cause TEXT NOT NULL,        -- user_correction | belief_revision | occurrence_upgrade
    run_id TEXT,
    note TEXT,
    created_at TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_assertion_revision_assertion
    ON assertion_revision(assertion_id, created_at DESC);
```

- Capture on every correction batch: `batch_id`, `assertion_id`, `actor_kind`, nullable `actor_ref`, `cause`, `run_id`, `note`, `created_at`.
- Capture per changed field: `changed_field`, `old_value_json`, `new_value_json`. Use JSON values, not plain text, so `null`, numbers, and structured fields round-trip cleanly.
- Track `proposition_text` separately from `belief_state`. If text changes, also log the derived `proposition_key` change in the same `batch_id`; if the new key collides with an existing `(matter_id, model_layer, proposition_key)`, reject and require an explicit merge/supersede workflow instead of silently rekeying.
- Keep `belief_revision_event` exactly as the BFS ledger for state propagation in [schema.py:107](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L107), [belief_revision.py:321](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L321), and [belief_revision.py:368](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L368). The new table is the immutable field-audit log; the old table remains the causal truth-maintenance log.
- Change `correct_assertion()` so it creates one `batch_id`, writes direct field-diff rows, applies the root mutation, and only then triggers BFS for belief-state/confidence changes. Text-only corrections do not need BFS.
- Route `_revise_one()` and `force_state()` through one audited assertion-update helper instead of raw `set_belief_state()`, so downstream BFS changes also get immutable field history.
- `record_assertion()` does not need a new public API, but the existing upgrade branch inside `AssertionStore.upsert_occurrence()` in [graph.py:79](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L79) should reuse the same helper when it mutates an existing assertion’s `belief_state` or `confidence`; otherwise that path remains a silent overwrite.
- Migration from v33 should be `v34`, additive only, with no table rebuild and no required backfill. Optional backfill from `belief_revision_event` is possible for historical belief-state/confidence changes only; overwritten historical text values cannot be reconstructed and should not be guessed.

This is the lowest-complexity design that gives durable legal auditability without breaking the current assertion graph, dedupe semantics, or hot-path queries.