PASS

Verified post-916323e state. Three r12 fixes confirmed correct in current code:

**1. Per-call stop event (HIGH #1 fix — shared-event reuse race)**
`stream_investigation()` line 348: `self._stop_event = threading.Event()` creates a fresh event per call.
- `AppState.__init__` initializes a placeholder at line 267 (correct).
- Each `stream_investigation()` call overwrites it with a new event before the thread starts.
- `stop_investigation()` always sets `self._stop_event`, which now always points to the current call's event.
- No cross-call interference possible. CLEAN.

**2. Correction refresh 4-tuple with issues_md (HIGH #3 fix)**
`_correct_and_refresh()` at line 728 returns `(correction_result, assertions_md, issues_md, overview_md)`.
- `issues_md` is in scope at line 747 (defined earlier in the Assertions tab layout at line ~686).
- `correction_btn.click()` outputs list includes `issues_md` at index 2.
- Belief revision propagation now refreshes assertions + issues + overview panels. CLEAN.

**3. stop_investigation() run_id capture order (bug introduced in 5dacd8f, fixed in 916323e)**
- Bug (5dacd8f): `self.current_run_id = None` appeared BEFORE `run_id = self.current_run_id`,
  so `run_id` was always None; `request_stop()` fell through to DB fallback on every stop.
- Fix (916323e): `run_id = self.current_run_id` at line 457 runs FIRST inside the try block;
  `self.current_run_id = None` is moved to line 470 after `request_stop()` completes.
- DB fallback at lines 459–465 covers the early-stop race (Stop pressed before first step).
- `self._stop_event.set()` at line 449 fires before any DB work, covering the no-run-session case. CLEAN.

No new issues found in full file scan of app.py (relevant sections) and in_process.py.

Prior r12 PASS (heapq.nsmallest) remains correct — `import heapq` at in_process.py:9,
`heapq.nsmallest(5, ...)` at line 105 unchanged.
