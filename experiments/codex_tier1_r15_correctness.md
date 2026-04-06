FAIL → FIXED (commit 41a756a, 732 tests)

**HIGH (Codex r15):** User-lock no-op row not written when same-state USER_CORRECTION also changes confidence.
- Root cause: `if not _fs_rev_rows` guard was False when confidence row was present, so no `belief_state` row written.
- Fix: changed to `elif cause == USER_CORRECTION` — belief_state intent row always written when state unchanged, regardless of confidence change.

**MEDIUM (Codex r15):** Recovery baseline used earliest occurrence (created_at ASC), not most authoritative.
- Root cause: a proposition first seen as ALLEGED then upgraded to OPERATIVE would recover to ALLEGED.
- Fix: `ORDER BY CASE speech_act WHEN 'operative' THEN 9 ... END DESC LIMIT 1` — picks highest-authority speech act across all occurrences.

**All r15 checks confirmed PASS:**
1. `_compute_belief_state()` with `superseding_states=[WITHDRAWN]` and `current_state=SUPERSEDED` passed directly: falls through to `return current_state, current_confidence` = SUPERSEDED. Standalone callers must use `_revise_one()` for recovery; direct calls use legacy path. Acceptable — documented behavior.
2. `_occ_row is None` fallback to UNKNOWN: defensive, correct. Invariant: assertion always has ≥1 occurrence (FK enforced, inserted in same transaction).
3. `detect_oscillation()` not triggered by no-op S→S rows: confirmed, adjacent duplicates don't trip the A→B→A detector.
4. `assertion_id` binding correct: no cross-assertion contamination.
5. Missing cases now covered: same-state+confidence lock, multi-occurrence recovery baseline.

2 new tests: `test_user_lock_preserved_when_confidence_also_changes`, `test_multi_occurrence_recovery_uses_best_speech_act`.
Total new tests since adversarial #027: 7.
