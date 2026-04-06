PASS

Verified with `rg` in [in_process.py:9](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L9) and [in_process.py:104](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py#L104). The change is exactly:

- `import heapq`
- `heapq.nsmallest(5, coverage_report, key=lambda r: float(r.get("coverage_fraction", 0.0)))`

This is correctness-preserving relative to `sorted(coverage_report, key=...)[:5]`:
- The `key=` function is identical.
- `heapq.nsmallest(5, ...)` returns the same 5 lowest-key items in ascending order as `sorted(... )[:5]`.
- Upstream, `coverage_fraction` is a bounded numeric value from [_coverage_fraction()`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L587) and the report is already sorted ascending in [matter.py:737](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L737).
- The service overview path still uses the original `sorted(... )[:5]` with the same key in [api.py:2161](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2161), which matches this backend semantically.

Quick full-file scan of [in_process.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py) found no new correctness issues not previously flagged.