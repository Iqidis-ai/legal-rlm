**Review**

CLEAN

No performance findings in commit `836de294a93e0729c7f9bd5eda388ace14efd7b8`.

`fetchall()` does change the lookup from “return first row” to `O(k)` materialization of currently `running` steerable runs for the matter in [src/irys/ui/backends/in_process.py:153](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py:153), but `k` is only the active-run set, not historical `run_session` rows. The query is still well-shaped for the existing `(matter_id, status, started_at)` index in [src/irys/matter/schema.py:164](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py:164), so the main filtering remains index-backed; the `objective` predicate is just a residual filter over that small candidate set.

Given that `_resolve_active_run_id()` is only used on stop/redirect paths in [src/irys/ui/backends/in_process.py:164](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py:164) and [src/irys/ui/backends/in_process.py:388](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py:388), not in a hot loop, I do not see a Tier 1 performance concern here. Residual note: there is no hard schema cap on concurrent running rows, so `LIMIT 2` would be a tighter micro-optimization if desired, but I would not block this commit on that.