**Findings**
- Medium: `flush_revisions()` still does not return a true count of unique revised assertions. It sums `len(_cr_results)` and `len(results)` in both loops ([runtime.py:478](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:478), [runtime.py:505](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:505)), but `BeliefRevisionEngine.apply()` appends one `RevisionResult` per write event ([belief_revision.py:345](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py:345)), and `_revise_one()` can return a result even when `belief_state` is unchanged and only confidence changes ([belief_revision.py:202](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py:202), [belief_revision.py:206](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py:206), [belief_revision.py:609](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py:609)). So the commit fixes the old `0` return for correction-only flushes, but the returned number is still only correct if the intended metric is “RevisionResult rows,” not “unique assertions whose belief state changed.”

**Answers**
1. The change is correct for the narrow bug it targets: correction-only flushes now return a nonzero count instead of hardcoded `0` ([runtime.py:456](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:456), [runtime.py:481](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:481)).  
It can still return a wrong count if you mean either:
   - unique assertion IDs revised, or
   - state-change revisions only.  
The current total counts revision results, which can include confidence-only updates and repeated revisions of the same assertion within one flush.

2. Yes. The ledger-event pattern is now the same, but behavior still differs in three ways:
   - correction-pending uses `RevisionCause.USER_CORRECTION` plus note `"deferred correction retry"` ([runtime.py:462](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:462), [runtime.py:464](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:464)); new-evidence uses `NEW_EVIDENCE` with no note ([runtime.py:490](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:490)).
   - correction-pending is drained from a deduping `set` ([matter.py:86](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:86), [matter.py:248](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py:248)); new-evidence comes from an adapter-local `list` that appends on every `record_fact()` call ([runtime.py:301](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:301), [runtime.py:484](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py:484)).
   - correction-pending runs first, so deferred user-correction propagation is applied before fresh new-evidence seeds.

3. Tier 1 is not fully CLEAN. Remaining correctness issue:
   - the `flush_revisions()` return value is still semantically loose/overcounting for “revised assertions,” as above.  
Other than that, I did not find a new commit-local state-propagation bug in `43cb6af`.

I could not run the targeted pytest checks here because command execution for `pytest` was blocked by the environment policy, so this review is based on static analysis.