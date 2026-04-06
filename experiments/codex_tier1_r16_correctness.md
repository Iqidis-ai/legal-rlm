FAIL → FIXED (commit dc9b592, 733 tests)

**MEDIUM #1 (r16):** User-lock race condition — `_revise_one()` read `assertion_revision` for user-lock check BEFORE acquiring the write lock (BEGIN IMMEDIATE). A concurrent `force_state(USER_CORRECTION, same-state, same-confidence)` could commit the lock row after the pre-tx read but before our write transaction. The OCC guard only compares `assertion.belief_state/confidence` — a same-state/same-confidence correction leaves no delta there, so OCC passed and the stale "no lock" decision was committed.
- Root cause: user-lock predicate not rechecked in-tx; OCC ignores audit-only changes.
- Fix: moved user-lock re-read inside the write transaction (after OCC check). Now protected by BEGIN IMMEDIATE: any concurrent `force_state` that committed before our write lock is visible; any that starts after is blocked. [belief_revision.py:586-601](/src/irys/matter/belief_revision.py#L586)

**MEDIUM #2 (r16):** Priority ordering in the `ORDER BY CASE speech_act` expression is not purely consistent with `_initial_belief_state()` confidence values — `alleged/argued` rank at priority 3 while `ELSE` (EXTRACTED, DENIED, etc.) rank at 0, but `_initial_belief_state()` returns UNKNOWN 0.5 for EXTRACTED and ALLEGED 0.3 for alleged. If confidence were the ordering key, EXTRACTED would win over ALLEGED.
- Root cause: duplicated hand-maintained priority logic inlined into SQL to avoid circular import (belief_revision.py → graph.py already exists; adding the reverse would create a cycle).
- Resolution: the ordering is intentionally about legal informativeness (specificity of speech-act classification), not starting confidence. ALLEGED is more semantically rich than EXTRACTED even though confidence is lower — an explicit allegation tells us more about legal character than an unclassified extraction. The priority correctly reflects this. Added code comment documenting the intentional deviation and circular-import constraint. No code change.
- LOW sub-issue: recovery fallback for unrecognized speech acts (ELSE) used UNKNOWN 0.3, but `_initial_belief_state()` returns UNKNOWN 0.5 for the same default case. Fixed: aligned to 0.5. [belief_revision.py:529-531](/src/irys/matter/belief_revision.py#L529)

**Other checks (all PASS):**
1. `elif cause==USER_CORRECTION` — no duplicate belief_state rows on state-change corrections; elif skipped when first branch fires. [belief_revision.py:708-725]
2. CASE priority order for explicit buckets consistent with `_initial_belief_state()`. Inconsistency was in the ELSE fallback confidence (fixed) and the alleged/ELSE ordering (intentional, documented). [belief_revision.py:504]
3. BFS-vs-BFS race on same SUPERSEDED node — OCC correctly aborts the second writer. BFS-vs-user race was MEDIUM #1 (fixed). [belief_revision.py:584-586]
4. No-op assertion_revision row (old==new) — no constraint/trigger issues; `detect_oscillation()` correctly skips adjacent duplicates. [schema.py:128] [graph.py:314]
5. BFS double-processing of SUPERSEDED node — `in_queue` prevents duplicate enqueuing; second visit sees recovered state, `_recovery_case=False`. [belief_revision.py:473-477]

Note: `cause=USER_CORRECTION` propagated BFS rows keep the same cause but log `actor_kind='system'` at L615/L765. The user-lock query correctly filters on `actor_kind='user'`, not on cause. Confirmed correct.

2 new tests: `test_user_lock_not_missed_by_concurrent_same_state_correction` (simulates race by writing lock row directly before BFS); `test_user_lock_preserved_when_confidence_also_changes` (corrected to pass explicit `confidence=0.05` to exercise the actual confidence-change path).
Total new tests since adversarial #027: 9.
