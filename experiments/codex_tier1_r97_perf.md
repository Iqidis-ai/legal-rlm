CLEAN.

The new cache evictions in [reasoning.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L181), [reasoning.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L204), and [reasoning.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py#L219) correctly bound `_seq_cache` growth on the normal complete/fail/interrupt paths. I traced the investigate, resume, stop, and redirect flows and did not find a seq-ordering regression or new hot-path cost beyond the expected one-time `MAX(seq_no)` rehydrate if an interrupted run later receives another event.

I did not run tests.