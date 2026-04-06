Reviewed against the current code paths and schema. Mandatory `CLAUDE.md` constraints are accounted for.

- [C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/.claude/CLAUDE.md](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/.claude/CLAUDE.md)

### HIGH
- None identified from these changes.

### MEDIUM
- `complete_run` performs an extra `SELECT` for `assertions_at_start` even though it is already known at run start and stored in the same `run_session` row. This adds an unnecessary round-trip on every completion path. Consider threading the snapshot forward (`start_run` → `complete_run`) and using an in-memory fallback map keyed by `run_id`, with DB fallback when missing.  
  - [src/irys/matter/matter.py:148](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:148)  
  - [src/irys/matter/reasoning.py:140](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py:140)

- `get_so_metrics()` uses `ORDER BY completed_at DESC` on `run_session` while schema has `[matter_id, status, started_at]` indexing, not `completed_at`, so this can require a sort for potentially large run histories. It is correct semantically, but can become slower as runs scale. Add `run_session(matter_id, status, completed_at DESC)` (or equivalent partial index) if this metric is called often.  
  - [src/irys/matter/matter.py:1291](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:1291)  
  - [src/irys/matter/schema.py:137](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py:137)

- `get_ledger_steering_surface()` is not N+1, but it is a multi-query fan-out: 1 query in `find_contradictions` + 3 inside `get_issue_coverage_surface` + 2 in `open_gaps` + 1 pending clarifications + 1 direct disputed query = 8 total per call. On large matters this is material if called per UI poll.  
  - [src/irys/matter/matter.py:1072](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:1072)  
  - [src/irys/matter/matter.py:478](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:478)  
  - [src/irys/matter/graph.py:548](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py:548)  
  - [src/irys/matter/graph.py:978](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py:978)  
  - [src/irys/matter/graph.py:1550](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py:1550)

- In that same flow, large-cardinality tables are read without `LIMIT` before Python slicing (notably pending clarifications/open gaps). This increases work and memory as scale grows. Add targeted DB-side limits for the steering endpoint (e.g., top-N only).  
  - [src/irys/matter/graph.py:978](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py:978)  
  - [src/irys/matter/graph.py:1550](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py:1550)

### LOW
- Importing `uuid` inside `get_ledger_steering_surface()` is a minor style/perf nit (cached import cost is low, but not ideal in hot paths). Move to module-level import.  
  - [src/irys/matter/matter.py:1072](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:1072)

- `get_ledger_steering_surface(run_id=...)` accepts `run_id` but does not use it, so steering cannot be scoped to the active run and misses a potential short-circuit optimization.  
  - [src/irys/matter/matter.py:1072](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:1072)

If you want, I can implement a minimal patch set for the MEDIUM items (pass-through snapshot + run_session `completed_at` index + query limits in steering path) without changing public behavior.