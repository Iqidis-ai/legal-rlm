Not CLEAN.

- `complete_run()` in `ReasoningLedgerStore` is correctly matter-scoped: it updates with `WHERE id=? AND matter_id=?`.
  - [src/irys/matter/reasoning.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L155)

- `_run_snapshots` is popped on both `fail_run()` and `interrupt_run()` in `MatterModel`, so the in-memory snapshot leak is fixed.
  - [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L177)
  - [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L181)

Remaining MEDIUM issue in scope:

- `ReasoningLedgerStore.fail_run()` and `ReasoningLedgerStore.interrupt_run()` still update `run_session` by `id` only (no `matter_id` predicate), so a mismatched `run_id` can affect a run from another matter.
  - [src/irys/matter/reasoning.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L170)
  - [src/irys/matter/reasoning.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L184)

