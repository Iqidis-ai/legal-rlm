**Findings**
1. MEDIUM: `build_query_context()` still fetches all actors, then slices in Python. [`matter.py:487`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L487) uses `self.actors.list_actors()[:10]`, but [`list_actors()`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1259) runs an unbounded `SELECT *`. Downstream only uses a small prefix in the prompt (`ctx.known_actors[:8]`) at [`engine.py:197`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L197). This is a remaining hot-path full materialization on every run start.

2. MEDIUM: `build_query_context()` still fetches all answered clarifications. [`matter.py:553`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L553) calls `self.clarifications.get_answered()`, and [`get_answered()`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1741) is unbounded. The orientation prompt only renders the first three answers at [`engine.py:204`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py#L204). Same pattern as the fixed pending-clarification path, but still unbounded here.

No remaining HIGH issues stood out in the reviewed UI/matter paths.

**Fix verification**
- Fix 1 verified: `get_ledger_steering_surface()` now uses `open_gaps(..., limit=20)` at [`matter.py:1348`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1348) and `get_pending(limit=3)` at [`matter.py:1381`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1381). Both are DB-side limited in [`graph.py:1115`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1115) and [`graph.py:1730`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1730).
- Fix 2 verified: `QueryMatterContext`/`build_query_context()` now uses `open_gaps(..., limit=10)` at [`matter.py:482`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L482), with DB-side limiting in [`graph.py:1115`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1115).
- Fix 3 verified: `list_clarifications()` now has `limit: int = 20` and passes it through to `get_pending(limit=limit)` at [`in_process.py:222`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L222), backed by the DB-side limit in [`graph.py:1730`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1730).

Conclusion: **FAIL**.

Static review only; I did not run runtime profiling or benchmarks.