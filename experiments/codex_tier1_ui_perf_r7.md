FAIL

- Medium: HTTP gaps/clarifications service endpoints ignore `limit` param —
  `api.py:1339/1696` called store methods unbounded even when HTTP backend sent limit.
- Medium: `get_open_issues()` uses `ORDER BY (salience * materiality) DESC` expression sort;
  `get_issue_coverage_report()` re-sorts by coverage_fraction anyway — wasted O(n log n) work.

**Verification (confirmed correct):**
- schema v37 ix_clarification_matter_answered: present at schema.py:518 ✓
- engine._orient() _stats param + stats dedup: present at engine.py:1046/1132/1144 ✓
- orientation fingerprint get_answered(limit=100) + open_gaps(limit=100): present ✓

**All MEDIUMs fixed in commit 8c05d21:**
- service get_pending_clarifications() + get_matter_gaps() accept limit param
- get_open_issues(order_by_score=False) skips sort for get_issue_coverage_report()

Static review only.
