Not CLEAN.

1. [belief_revision.py:234](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py:234)  
MEDIUM: MAX_WORK truncation still returns a partial revision result with no caller-visible indicator.  
You fixed the silent behavior by logging, but runtime callers still get `RevisionResult[]` as if revision completed. In production, this can leave downstream beliefs stale without any API-level signal (except logs). The previous issue is reduced, but not eliminated in terms of correctness observability.

2. [graph.py:528](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py:528)  
MEDIUM: `get_by_proposition()` without `model_layer` can return any matching proposition across layers because there is no ordering/deterministic tie-break.  
Given legal assertions are explicitly partitioned by layer, returning an arbitrary layer can inject cross-layer leakage if callers forget/omit `model_layer`.

Notes:
- I could not find `MatterModel._investigate_context`; closest current method is `build_query_context()` in `matter.py`, and no additional high/medium issue stood out there.
- `get_issue_coverage_report()` did not expose a fresh high/medium in this pass beyond the two points above.