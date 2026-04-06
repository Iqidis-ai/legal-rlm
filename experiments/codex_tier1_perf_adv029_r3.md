CLEAN.

Repo-root `CLAUDE.md` is not present in this workspace; I reviewed [`.claude/CLAUDE.md`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/.claude/CLAUDE.md) and the `adv#029` changes in `185d882`, `a5b521c`, and `a4772d1`.

1. The lock-held `INSERT OR IGNORE` in [`matter.py:278`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:278) and [`matter.py:334`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:334) is not a meaningful throughput concern for normal correction flow. Those methods only run when BFS truncates, which is already the slow path, and SQLite is single-writer anyway. The Python lock mostly preserves the first-write-wins queue invariant; it is not the dominant cost.

2. `drain_correction_pending` / `drain_evidence_pending` stopping DB deletion is correct and better for crash safety. The delete-after-replay path through [`runtime.py:440`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:440) and [`matter.py:386`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:386) is cheap and properly chunked.

3. `delete_pending_propagation_db()` doing chunked `DELETE`s by `assertion_id` is correct.

4. `peek_correction_pending_ids` / `peek_evidence_pending_ids` in [`matter.py:308`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:308) and [`matter.py:366`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:366) only hold the lock briefly to snapshot keys. That is fine.

5. The standalone flush endpoint’s extra `start_run()` / `complete_run()` in [`api.py:1514`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py:1514), [`matter.py:188`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:188), and [`matter.py:201`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:201) is negligible.

6. The only potentially noticeable added cost is the end-of-flush proof-state recompute in [`runtime.py:596`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:596) calling [`graph.py:3370`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py:3370). On very large `_revised_ids` sets, yes, this can be expensive because each affected issue still pays the full `compute_and_store()` query bundle. But it is already reasonably bounded: revised assertions are deduped, issue lookup is chunked, trust overrides are preloaded once, and writes are grouped in one transaction. That makes it a targeted, proportional cost, not a performance defect.

Overall impact is low for normal correction flow. Large deferred flushes will now spend extra time recomputing `proof_state`, and that will be the dominant added latency, but it is expected work to keep issue state coherent and I do not see a severity-worthy performance issue here. No benchmarks were run.