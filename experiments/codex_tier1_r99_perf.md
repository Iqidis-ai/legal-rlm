**CLEAN** (manual review — Codex credits exhausted until 2026-04-08)

Commit `2274830`: one `append_event()` call added to `InProcessBackend.redirect_run()`.

**(1) Hot path** — `redirect_run()` is user-triggered (button click), not called in a loop.
A single additional SQLite write is negligible on this path.

**(2) Lock contention** — no new locks introduced. `append_event()` uses the same
`_append_event()` / `reasoning.py` DB path as all other ledger writes. No regression.

**(3) Other perf concerns** — none. The local import `from irys.matter.enums import LedgerEventType`
is a one-time module load cached by Python's import system after first use.

No findings.
