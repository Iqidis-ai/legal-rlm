**Findings**
No HIGH findings. 4 MEDIUM findings remain.

- `MEDIUM` The `_run_async()` change is not a clean performance win for the current UI because it serializes all in-process backend work onto one background loop thread. [`app.py:25`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L25) [`app.py:35`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L35) [`app.py:188`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L188) [`in_process.py:88`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L88) [`in_process.py:211`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L211)  
  The in-process backend methods are `async` in signature but do synchronous DB/model work and do not yield, so `run_coroutine_threadsafe(..., _ASYNC_LOOP)` runs them one-at-a-time. The parent revision used a `ThreadPoolExecutor(max_workers=4)`, so concurrent panel refreshes now lose parallelism even though per-call loop startup overhead is gone.

- `MEDIUM` The Gaps tab still materializes the full open-gap set and only slices afterward. [`app.py:393`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L393) [`in_process.py:211`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L211) [`graph.py:1115`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1115)  
  `AppState.load_gaps()` calls `list_gaps()` with the default cap, but `InProcessBackend.list_gaps()` still does `open_gaps()[:limit]`. That keeps the exact full-scan/full-link-fetch behavior that commit `2b0f3a0` fixed for overview.

- `MEDIUM` Overview still fetches all pending clarifications and slices to 5 in Python. [`api.py:2158`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2158) [`in_process.py:111`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L111) [`graph.py:1730`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1730)  
  `ClarificationStore.get_pending()` has no limit parameter, so overview latency still scales with total pending clarifications even though the UI only renders 5.

- `MEDIUM` `generate_clarifications_from_gaps(top_n=...)` does not use the new `open_gaps(limit=...)` path. [`matter.py:712`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L712) [`matter.py:724`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L724) [`graph.py:1115`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1115)  
  It still loads all qualifying gaps, re-sorts them, and then keeps `top_n`. This leaves avoidable O(all qualifying gaps) work in the end-of-run path.

**Verification**
- Fix 1, `_run_async()`: functionally correct from sync Gradio callbacks, and it does remove per-call event-loop creation/teardown. I do not sign it off as a net performance improvement because, in this app, it also collapses read concurrency to one thread.
- Fix 2, coverage dedup in overview: correct and real win. [`api.py:2124`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2124) [`api.py:2134`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2134) [`in_process.py:91`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L91) [`in_process.py:99`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L99) [`matter.py:1465`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1465)  
  Overview now computes `get_issue_coverage_report()` once and reuses it for both `weakest_issues` and `get_so_metrics(...)`.
- Fix 3, `open_gaps(limit=...)`: correct and real win where used. [`graph.py:1126`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1126) [`graph.py:1135`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L1135) [`api.py:2151`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2151) [`in_process.py:106`](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L106)  
  The limit is pushed into SQL and the `gap_link` fetch is restricted to returned IDs only, so overview no longer pays for all matching gaps.

**Result**
FAIL.

Runtime benchmarking/test execution was not possible here because the shell policy blocks Python execution, so this is a static performance review based on the current code, the target commit diff, and query/work analysis.