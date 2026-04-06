PASS

All three Tier 2 r3 fixes verified clean.

**1. PRAGMA busy_timeout=5000 — PASS.**
- Applied via `conn.execute("PRAGMA busy_timeout=5000")` on file-based connections only (in-memory test connections use a separate shared path and are unaffected). [db.py:60-67]
- Interacts correctly with BEGIN IMMEDIATE: when another writer holds the write lock, SQLite now retries internally for up to 5 s before raising OperationalError, rather than failing immediately. This is the intended behavior.
- Does not mask starvation: 5 s is a per-acquisition timeout, not a per-session queue depth limit. Writers still fail after 5 s if the lock is held the whole time — they don't queue silently forever.
- WAL checkpoint: busy_timeout also applies to checkpoint lock acquisition. 5 s is sufficient for normal checkpoint waits. No adverse interaction.

**2. _SPEECH_ACT_RECOVERY_PRIORITY module-level constant — PASS.**
- Defined at module level [belief_revision.py:60-65]. All 11 keys match the _initial_belief_state() speech_act mapping exactly.
- Referenced correctly in _revise_one() via `_SPEECH_ACT_RECOVERY_PRIORITY.get(r["speech_act"] or "", 0)`. The `or ""` handles None speech_act rows; default 0 routes to `_initial_belief_state(SpeechAct(None))` → ValueError → UNKNOWN, 0.5 fallback. ✓

**3. Schema v38 partial covering index — PASS.**
- `assertion_revision` table is defined in _DDL_CORE at line 128, BEFORE both the existing index (L143) and the new index (L146–148). `CREATE INDEX IF NOT EXISTS` in DDL applies to fresh DBs; migration v38 applies to existing DBs at v37 — no conflict. [schema.py:128-148]
- The lock query: `WHERE assertion_id=? AND changed_field='belief_state' AND new_value_json=? ORDER BY created_at DESC LIMIT 1`
  - Partial index WHERE `changed_field='belief_state'` eliminates all non-belief_state rows from the index.
  - Index columns `(assertion_id, new_value_json, created_at DESC)` support a 2-column seek + ordering + LIMIT 1 — O(log N) regardless of lock-row history.
  - LOW: `actor_kind` (the projected column) is not in the index → one heap fetch for the LIMIT 1 result. With LIMIT 1 this is O(1) and not a concern at current scale. A future optimization would add `actor_kind` to the index columns.

**4. HIGH #1 (bulk withdrawal truncation) — stale, not permanently wrong.**
- Nodes left in the BFS queue when truncation fires are in a stale SUPERSEDED state, not an incorrect state. Their stale-ness is correctly flagged: `propagation_truncated=True` in `RevisionResult`; SYSTEM_WARNING logged with specifics (budget, OCC exhaustion). [belief_revision.py:357-380]
- Next relevant trigger (upstream state change, user correction, `flush_revisions()`) re-enqueues them and they converge correctly.
- Not a new regression: BFS budget ceiling is a pre-existing architectural limitation, now documented.

No HIGH or MEDIUM findings. 733 tests continue to pass.
