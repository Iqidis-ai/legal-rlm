FAIL → FIXED (commit 5c0bab8, 730 tests)

**HIGH (Codex r14 finding):** Supersession recovery reset to UNKNOWN instead of speech-act baseline; user-forced SUPERSEDED wrongly recovered when inert superseding link exists.

**Root causes:**
1. Recovery code overrode `old_state = UNKNOWN` (or speech-act state) — but the fast-path comparison `new_state == old_state` used the OVERRIDDEN value. Since `new_state (UNKNOWN) == old_state (UNKNOWN)`, fast-path returned None with no DB write. DB stayed SUPERSEDED.
2. When user calls `correct_assertion(B, SUPERSEDED)` on an already-SUPERSEDED assertion, `force_state()` writes no revision row (no delta). User-lock query then found only the BFS system row, not user intent → allowed recovery.

**Fixes (commit 5c0bab8):**

1. `_revise_one()`: separate `_compute_state`/`_compute_conf` (computation input) from `old_state`/`old_confidence` (actual DB state for fast-path and OCC). DB state correctly detected as changed (SUPERSEDED → UNKNOWN/OPERATIVE/etc.).

2. Speech-act baseline: first occurrence `speech_act` looked up for recovery state. OPERATIVE → OPERATIVE, ADMITTED → ADMITTED, ALLEGED → ALLEGED, EXTRACTED → UNKNOWN. Contract clause superseded by voided amendment now recovers to OPERATIVE, not UNKNOWN.

3. User-lock: in `force_state()`, when no state/confidence delta AND `cause=USER_CORRECTION`, write a no-op revision row (`old_value_json == new_value_json`). Supersession recovery detects this `actor_kind='user'` row and skips recovery when user explicitly chose SUPERSEDED.

**All Codex r14 checks:**
- (a) No BFS loop — confirmed in Codex session, unchanged.
- (b) WITHDRAWN early-exit before superseding logic — unchanged, still correct.
- (c) Backward compat — `superseding_states=None` path unchanged.
- (d) SPO upgrade transactionally safe — confirmed in Codex session, unchanged.
- (e) ADMITTED/RESOLVED in _PROMOTING — consistent with corroborates/supports, unchanged.
- (f) Edge cases now covered: OPERATIVE recovery (new test), user-lock (new test).

5 new tests total since adversarial #027: supersession recovery (EXTRACTED baseline), OPERATIVE recovery, user-lock preservation, ADMITTED promotion, RESOLVED promotion.
