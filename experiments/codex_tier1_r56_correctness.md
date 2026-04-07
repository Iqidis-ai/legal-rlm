**Findings**
1. HIGH: The new steerability metric still produces false positives because it counts any `run_session` for the matter, not just steerable investigation runs. The query in [matter.py:1996](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1996) will count utility flush runs, but those same runs are explicitly treated elsewhere as non-steerable: stop/redirect ignore `manual_flush` and `background_flush` in [reasoning.py:204](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L204), [reasoning.py:229](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L229), and [service/api.py:2718](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2718). Those non-steerable runs are still created under the same matter in [service/api.py:1691](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1691), [service/api.py:1693](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1693), and [service/api.py:2144](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L2144). A matter with only flush history will now report `steerability=True`, which is still manufactured compliance.
2. MEDIUM: `COUNT(*) > 0` is not durable evidence that steerability capability was available for the counted run. The `run_session` schema has no capability/version/source marker in [schema.py:150](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L150), and local git history shows `run_session` existed in commit `8304f47` before the engine stop wiring landed later (`c9124e6` and follow-ups). Old rows therefore cannot be distinguished from genuinely steerable runs.

**Answers**
1. The SQL is syntactically fine and matter-scoped by `matter_id`, but it is not correctly scoped for SO-3. It also does not match its own comment in [matter.py:1988](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1988), which says “completed” session while the query counts every status.
2. Yes. `run_session` can exist for a matter even when the interruptible investigation engine was not the path in use. `manual_flush` and `background_flush` are the concrete examples in this codebase.
3. No. `run_session count > 0` only proves that some row exists. It does not prove a steerable investigation ran, and legacy rows are not distinguishable.
4. The `COUNT(*)` query itself is safe. In SQLite it returns one non-null row even for zero matches, so the realistic `None` path is only an exception/schema problem, which the code already maps to `None`. The `sr_row else False` branch is basically defensive dead code.

**Rating**

HIGH

Static review only; I did not run tests.