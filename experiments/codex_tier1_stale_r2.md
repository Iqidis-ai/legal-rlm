FLAG

- HIGH: `_revise_one()` still has a pre-transaction no-op short-circuit at [belief_revision.py:324](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L324). The assertion row and neighbor states are read before the transaction, and if that stale snapshot looks unchanged, the method returns before entering [belief_revision.py:329](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L329). A concurrent writer can therefore change the row in that window and bypass the new in-tx `_intx_old_*` logic entirely. So the original race is not fully closed in `_revise_one()`. `force_state()` does not have this specific gap.

- MEDIUM: `BEGIN IMMEDIATE` only helps for top-level calls. In nested use, `write_transaction()` falls back to a savepoint because `_Transaction.__enter__()` checks `in_transaction` first and does `SAVEPOINT` instead of `BEGIN IMMEDIATE` at [db.py:130](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/db.py#L130) through [db.py:139](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/db.py#L139). That means an outer deferred transaction can still carry the old WAL snapshot problem; the inner belief-revision write does not actually acquire the write lock early in that case.

- MEDIUM: `RevisionResult` still returns the pre-tx `old_*` snapshot at [belief_revision.py:378](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L378) and [belief_revision.py:458](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L458), while the committed audit trail now uses in-tx values. That object is surfaced to callers via [api.py:1773](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py#L1773) and used for ledger messaging at [runtime.py:453](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py#L453). Under concurrency, callers can observe a transition that does not match what was actually committed.

Answers to your questions:

1. No. The in-tx diff checks fix the persisted diff/event logic once the code reaches the transaction, but `_revise_one()` still has the stale pre-tx early-return path.

2. Top-level `BEGIN IMMEDIATE` is correct and does address the WAL upgrade problem. The nested/savepoint edge case is still open for the reason above.

3. I do not see a separate new HIGH/MEDIUM regression caused by the patch beyond these residual/incomplete-fix issues.

4. No, not as currently used. It would only be acceptable if `RevisionResult` were explicitly defined as a pre-compute snapshot, but current API and ledger call sites treat it like the committed transition.

I also did not find test coverage for nested `write_transaction()` / savepoint behavior.