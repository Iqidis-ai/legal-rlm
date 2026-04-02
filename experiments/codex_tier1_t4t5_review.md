# Codex/Manual Tier 1 Review — T4 + T5

Date: 2026-04-02
Scope: src/irys/rlm/engine.py (T4 bridge), src/irys/matter/graph.py (ActorStore),
       src/irys/matter/runtime.py (infer_source_role, record_fact), tests/matter/

## Findings

### MEDIUM — Missing index on actor_alias(alias_text) for get_by_alias lookups
- `get_by_alias()` queries `WHERE aa.alias_text=?` but `ux_actor_alias` only covers
  `(actor_id, alias_text)`. Full table scan for large alias sets.
- **Fix applied:** Added `ix_actor_alias_text ON actor_alias(alias_text)`.

### LOW — Source-role DRAFT priority corrected
- 'Contract_Draft_v2.docx' matched OPERATIVE before DRAFT in original pattern order.
- **Fix applied:** DRAFT pattern moved to first position in _SOURCE_ROLE_PATTERNS.

### LOW — run_id None path in investigate() is safe
- Guard `if run_id is not None:` on complete_run/fail_run prevents calls when
  enable_matter_model=False or matter_model=None.

### DESIGN NOTE — actor_alias has no cross-actor uniqueness constraint
- Two different actors CAN share the same alias_text (e.g. both have alias "Acme").
  `get_by_alias` returns the first DB match, which is indeterminate.
- Acceptable for current scope. Not a defect given controlled alias creation.

### CORRECTNESS — ActorStore.upsert_actor is thread-safe ✓
- Thread-local SQLite connections + WAL mode; no shared mutable state.

## Status
All issues addressed. 70/70 tests passing. T5 approved for commit.
