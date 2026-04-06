Not `CLEAN`.

## CORRECTNESS

HIGH
- `complete_run()` is still not fully matter-scoped on write.
  - The DB fallback read is scoped correctly: `SELECT assertions_at_start FROM run_session WHERE id=? AND matter_id=?` in `complete_run()` [matter.py:167](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:167).
  - But `self.ledger.complete_run(run_id, summary, reuse_rate)` hands control to `ReasoningLedgerStore.complete_run()`, which updates by `run_session.id` only, with no `matter_id` guard [matter.py:174](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:174) + [reasoning.py:155](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py:155).
  - This can update an in-scope run row from another matter if a foreign `run_id` is passed.

MEDIUM
- `_run_snapshots` can grow unbounded in long-lived processes.
  - Snapshot entries are added in `start_run()` [matter.py:152](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:152).
  - They are only popped in `complete_run()` [matter.py:164](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:164), not in `fail_run()`/`interrupt_run()`.
  - A run that fails, is interrupted, or is orphaned before completion can leave a stale entry.
  - Suggested fix: centralize cleanup in one method (e.g., a `finalize_run` helper) and call from complete/fail/interrupt.

LOW
- `run_id` is intentionally documented as reserved/not used in `get_ledger_steering_surface()` [matter.py:1089](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:1089). This is not a correctness defect by itself.

## PERFORMANCE

LOW
- The v30 migration/index appears correct for the `get_so_metrics()` reuse-rate query:
  - Index exists on `(matter_id, status, completed_at DESC)` via `ix_run_completed` [schema.py:1176](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py:1176).
  - Query uses `ORDER BY completed_at DESC LIMIT 5` with matching leading keys [matter.py:1414](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:1414), so this should avoid the previous sort-heavy path.

## Requested-point check

- Are `_run_snapshots` memory bounds safe? No (MEDIUM issue above).
- Is DB fallback in `complete_run()` correctly scoped? Partially: read-scope is correct, write-scope is not.
- Does v30 migration benefit `ORDER BY completed_at DESC`? Yes.
- Any remaining bare `except` in `get_ledger_steering_surface()`? No; they are now `except Exception` with `_log.warning(..., exc_info=True)` [matter.py:1154](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:1154).
- New issues introduced by fixes? Yes: the two above (HIGH+MEDIUM).