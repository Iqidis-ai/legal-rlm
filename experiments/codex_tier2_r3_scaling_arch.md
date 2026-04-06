FAIL → FIXED (commit below, 733 tests)
MEDIUM — 1 finding. No HIGH.

**MEDIUM: Inlined speech-act priority CASE is unnecessary — no circular import exists.**
- `belief_revision.py` already imports `AssertionStore` from `graph.py`. `graph.py` imports nothing from `belief_revision.py`. There is no circular dependency, so the comment "Inlined from _initial_belief_state() in graph.py to avoid circular import" is incorrect.
- Root cause: the inline was written defensively but the import graph was not verified. Result: two parallel priority mappings (the CASE expression and `_initial_belief_state()`) that have already drifted once (ELSE confidence was 0.3 vs 0.5 in `_initial_belief_state()`, fixed in r16 LOW).
- Risk: next time a SpeechAct member is added or `_initial_belief_state()` is updated, the CASE must be manually updated too, and the update may be missed.
- Fix: import `_initial_belief_state` and `SpeechAct` from `graph.py`/`enums.py`; fetch all speech_acts for the assertion; pick the highest-priority one in Python using `_initial_belief_state()`. The SQL CASE expression can be replaced with `SELECT speech_act FROM assertion_occurrence WHERE assertion_id=?` + Python sort.
  [belief_revision.py:493-533] [graph.py:32-64]

**LOW findings:**

1. Recovery fan-out at scale — bounded by work budget, not a new risk.
   - If one assertion supersedes 1,000 others and is withdrawn, BFS processes up to `_effective_max_work = min(2000, seeds*3)` nodes. Each recovering node runs 2 extra queries (user-lock + occurrence baseline). For 1,000 superseded nodes: 2,000 extra queries on the recovery path. These are fast indexed reads (assertion_revision and assertion_occurrence both have assertion_id-prefixed indexes).
   - In legal matter practice, one-to-many supersession at this scale is unusual (an amendment supersedes specific provisions, not thousands). The BFS truncation mechanism correctly handles pathological cases.
   - Acceptable at current scale.

2. BEGIN IMMEDIATE write-lock contention — negligible for single-matter investigations.
   - The user-lock re-read inside BEGIN IMMEDIATE adds one fast indexed read to the critical section (~100μs). For single-matter investigations (the common path), only one BFS runs at a time. For concurrent multi-matter on the same SQLite file, writers serialize — but queries are sub-millisecond, so the expected wait is low. SQLite WAL mode allows concurrent readers while a writer holds the lock.
   - Not a bottleneck at current scale. Note for future: multi-matter parallel workloads should use separate DB files per matter (MatterModel.open_at() with matter-scoped paths), not a single shared SQLite instance.

3. assertion_revision no-op row accumulation — storage and query cost negligible.
   - The user-lock query uses `ORDER BY created_at DESC LIMIT 1` against `ix_assertion_revision_assertion(assertion_id, created_at DESC)`. Even 10,000 no-op rows for one assertion would still resolve in O(log N) time (index seek to the most recent row). Storage: ~200 bytes/row × 10,000 = ~2MB — immaterial.

**Architecture / TMS theory:**

4. Recovery policy soundness — correct for JTMS-like dependency tracking.
   - "If every superseder of B is now WITHDRAWN or SUPERSEDED itself, B recovers to its speech-act baseline" is a sound JTMS-derived policy. It models the legal intuition: "the thing that superseded you is no longer operative, so you are no longer superseded by it."
   - The speech-act baseline correctly represents B's initial epistemic state, not a user-defined override. Using the most-authoritative occurrence (r15 MEDIUM fix) is the right policy: a proposition first seen as ALLEGED and later observed as OPERATIVE in a contract should recover to OPERATIVE, not ALLEGED.
   - Edge cases correctly handled: (a) user-forced SUPERSEDED without superseding link → terminal; (b) user explicitly locked SUPERSEDED → user intent respected; (c) mixed inert/active superseders → recovery blocked as long as any active superseder exists.
   - One policy gap (pre-existing, not introduced by this milestone): if B was superseded by A, and A was itself superseded by C (chain: C→A→B), and C is withdrawn — does B recover? Current code: `_revise_one(A)` sees C is WITHDRAWN → A may recover → `_revise_one(B)` sees A recovered → B may recover. The fixpoint BFS handles this correctly through iterative re-enqueuing. **PASS.**

5. Long-term architecture note: the `_initial_belief_state()` function encodes the speech-act-to-belief-state mapping as a canonical source of truth. The recovery path SHOULD use it. As the assertion type model grows (new SpeechAct values for legal domain specifics), having one authoritative mapping reduces maintenance surface. The MEDIUM fix above addresses this.
