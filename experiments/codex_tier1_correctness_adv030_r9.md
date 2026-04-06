CLEAN.

1. The new predicate is valid SQLite in both updates: `[reasoning.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L201)` and `[reasoning.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L220)`. `NOT IN (...)` is legal in `UPDATE ... WHERE`, and the explicit `objective IS NULL OR ...` is the right way to preserve `NULL` objectives under SQLite’s three-valued logic.

2. It does not break the legitimate engine steering path. The runtime adapter forwards `self.run_id` in `[runtime.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py#L684)` and `[runtime.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py#L720)`, and those are normal investigation runs, not `manual_flush` / `background_flush`. Utility adapters are only created for flush helpers in `[api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1547)` and `[api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1920)`; I did not find a production path from those helpers into `request_stop()` or `request_redirect()`.

3. The stop/redirect utility-run chain is now closed across current callers:
   - raw ledger layer rejects utility runs: `[reasoning.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L201)`, `[reasoning.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L220)`
   - matter-level stop selector excludes them: `[api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1366)`
   - run-scoped API stop/redirect reject them: `[api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1402)`, `[api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2446)`
   - in-process backend rejects them: `[in_process.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L82)`, `[in_process.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L253)`
   - UI fallback selectors exclude them: `[app.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L292)`, `[app.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/app.py#L471)`
   - model-level run-id guards already exclude them for correction/trust flows: `[matter.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L503)`, `[matter.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L644)`

Static review only; I did not run tests.