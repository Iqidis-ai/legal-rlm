**Findings**
r7 Codex session verified both r6 perf fixes are present in current code:
- ix_clarification_matter_answered index: SCHEMA_VERSION=37, DDL at schema.py:518, migration v37 at schema.py:1433 ✓
- engine._orient() _stats param: present at engine.py:1132; investigate() passes _stats=stats at engine.py:1046 ✓

**Additional MEDIUM found by session**: engine.py:1175 calls `get_answered()` and `open_gaps()` without limits
for orientation cache fingerprint — full-table scans on large matters.

**Fix applied (commit bc66624)**: `get_answered(limit=100)` and `open_gaps(limit=100)` in fingerprint computation.

**Status**: FAIL (one new MEDIUM found and fixed). Proceeding to r8 for clean verification.
