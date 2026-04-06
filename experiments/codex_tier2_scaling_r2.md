FAIL — 3 HIGH, 3 MEDIUM, 1 LOW found. HIGH/MEDIUM fixes applied.

**Findings:**

HIGH: find_possible_duplicates() O(N²) — at 10k actors: 50M pair checks before sorting.
Fix applied: sorted-prefix scan O(N·k); actors sorted by normalized_name so same-prefix
actors are always adjacent. Added limit=100 param. Commit 9101136.

HIGH: Belief revision completeness-bounded at 2000 visits — truncation is acceptable
steady-state, but large reachable subgraphs on 100k-assertion matters will leave downstream
beliefs stale by design. Prior fix (64ef34f) ensures flush_revisions() batches seeds so
no seeds are lost. The BFS propagation cap itself (2000) is architectural — making
propagation resumable/async is the full fix (deferred, logged in backlog).

HIGH: InProcessBackend not production-scalable — unbounded repo→MatterModel cache,
SQLite connection accumulation per matter, ReasoningLedgerStore._seq_cache never shrinks.
Architectural note: do NOT use InProcessBackend in production; use HttpBackend + service.
Noted in STATUS.md.

MEDIUM: AppState global state (current_matter_id, current_run_id, _irys_ref, shared engine
_matter_model) can be overwritten by concurrent sessions. Known architectural limitation for
single-user dev tool. Noted in STATUS.md.

MEDIUM: list_recent(limit=50) aggregated GROUP BY across ALL assertions before LIMIT —
cost grew with total matter size, not visible rows.
Fix applied: CTE bounds assertion ID set first, then joins occurrences only for those IDs.
Commit 9101136.

MEDIUM: get_ledger_steering_surface() called find_contradictions() without limit — loaded
all conflict links then sliced [:5] in Python.
Fix applied: limit=5 pushed to SQL via find_contradictions(limit=5). Commit 9101136.

LOW: get_issue_coverage_report() + get_so_metrics() are full-matter aggregations but
indexed and set-based; materialized summary table would help at very large scale.
Logged in backlog.
