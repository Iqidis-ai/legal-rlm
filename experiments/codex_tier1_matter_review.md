# Codex Tier 1 Review — Matter Module (T3)

Date: 2026-04-02
Session: 019d4f95-0b2d-7422-9a62-8a8587009e14
Scope: src/irys/matter/ (10 files), tests/matter/ (4 files, 33 tests)

## Findings

### MEDIUM — Dead variable `disputed_supports` in belief_revision.py:68
- `disputed_supports` was computed but never referenced in the control flow.
- The branch on L80 correctly handles all non-solid supporter cases through `strong_supports` filter.
- **Fix applied:** Removed dead `disputed_supports` assignment.

### MEDIUM — `_Transaction` does not support nested transactions (db.py:100)
- `_Transaction.__enter__` unconditionally called `BEGIN`. A nested call would error.
- **Fix applied:** Uses `conn.in_transaction` to detect nesting; falls back to SQLite savepoints.

### LOW — Graph traversal directions correct (graph.py:173, 198) ✓
### LOW — Index coverage complete for all graph traversal queries (schema.py:79-83) ✓

## Status
Both MEDIUM findings fixed. 33/33 tests passing. Approved for commit.
