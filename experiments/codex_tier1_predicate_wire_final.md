FLAG

1. Yes. The gate now uses `analysis.get("key_facts")` at [engine.py:2033](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py:2033), so the prior unbound-local on `facts_to_add` is fixed.

2. Not CLEAN. One remaining `MEDIUM` issue:
   The resolution guard at [engine.py:2033](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py:2033) now checks raw `analysis["key_facts"]`, but the actual persisted fact set is built later by filtering into `facts_to_add` at [engine.py:1903](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py:1903) and then recording via `record_facts_batch(...)` at [engine.py:1981](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py:1981). If `key_facts` is truthy but malformed/skipped or otherwise yields no persisted facts, predicates can still be resolved with no supporting fact stored from this pass.

No remaining `HIGH` issue is visible in this block, but that `MEDIUM` means the block is still flagged.