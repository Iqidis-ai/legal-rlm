MEDIUM found and fixed.

**MEDIUM: Overview panel fetches unused data on every load**

Findings (Codex r9 perf static review):
1. `in_process.py:103` — `model.ledger.recent_runs(limit=5)` fetched and returned as `"recent_runs"` list, but `_fmt_overview()` never consumes it. DB query on every overview load.
2. `api.py:2154-2158` — same `recent_runs(limit=5)` fetch in overview endpoint, unused by formatter.
3. `api.py:2182-2192` — `source_role_summary` GROUP BY query across all assertion_occurrences for this matter. Not used by `_fmt_overview()`. Added as informational but never consumed.

**Fixes applied:**
- Removed `recent = model.ledger.recent_runs(limit=5)` from `in_process.py:get_overview()`; removed `"recent_runs"` and `"source_role_summary"` keys from return dict.
- Removed same from `api.py:get_overview()` endpoint: dropped `recent` block, `source_summary` GROUP BY block, and both keys from return dict.

**PASS verifications (carried forward from session):**
- PASS: `get_open_issues(order_by_score=False)` at graph.py — skips expression sort when formatter re-sorts anyway.
- PASS: `get_issue_coverage_report()` passes `order_by_score=False` at matter.py.
- PASS: Clarifications and gaps accept + forward `limit` at api.py and http.py.

All fixes committed. 725/725 tests passing.
