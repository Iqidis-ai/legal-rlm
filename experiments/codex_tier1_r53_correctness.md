**Findings**
- `LOW` [api.py:143](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L143): `6956b30` changed `_sync_running_matter_ids` from `set[str]` to `dict[str, int]`, but `_cleanup_loop()` still does `set | _sync_running_matter_ids`. In Python, `set | dict` raises `TypeError`, so the orphan-eviction pass now aborts each sweep after the expired-job loop. The direct impact is stale/rehydrated models no longer being evicted; the refcount logic itself is otherwise sound.

**Answers**
1. Refcount increment/decrement in [api.py:1057](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1057), [api.py:1134](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1134), [api.py:1384](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1384), and [api.py:1435](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1435) is correct. Increment happens before the awaited investigation, decrement happens once in the outer `finally`, and the refcount fixes the old same-matter overlap bug.

2. `sync_matter_id` / `urls_matter_id` are guaranteed in scope because they are initialized to `None` before `try` at [api.py:979](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L979) and [api.py:1350](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1350). I do not see a double-decrement path: the inner unpin was removed, and the outer `finally` is the only remaining unpin site.

3. The new expired-job-loop check at [api.py:128](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L128) is in the right place. That is exactly where an expired async job could previously evict a model still needed by a sync run. Separate issue: line 143 still treats the refcount dict like a set.

4. `_rehydration_locks.pop()` on the two success paths at [api.py:324](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L324) and [api.py:331](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L331) looks correct. In both cases the model is already in `_active_matter_models`, and queued waiters already hold the old `Lock` object reference.

5. `_active_sync_requests` at [api.py:847](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L847), [api.py:972](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L972), and [api.py:1344](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1344) is correct for this process-local in-memory design. The `global` declarations are present, the check/increment happens before any `await`, and decrement is in `finally`. The only caveat is the same one the whole service already has: it is per-process, not cross-worker.

Overall: `LOW`.

Static review only. I did not run tests, and I did not find targeted regression coverage for the new refcount/cleanup paths.