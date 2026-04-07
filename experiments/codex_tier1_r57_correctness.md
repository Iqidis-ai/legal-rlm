**Findings**
1. MEDIUM: `steerability` can still false-positive on historical non-flush runs. [`get_so_metrics()`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py) now excludes flush objectives at [src/irys/matter/matter.py:1996](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py), which is the right fix for this commit, but it still treats any older non-flush `run_session` as proof of SO-3. Repo history shows `run_session` existed before end-to-end steering was actually wired: base table in `8304f47`, stop polling added later in `c9124e6`, redirect wiring in `81573f6`, and REST stop endpoint in `ddbc483`. Those old rows are not distinguishable in current schema, so the earlier MEDIUM concern remains.

2. LOW: `belief_revision` is still a manufactured-compliance proxy. The target says “corrections propagate” at [src/irys/matter/matter.py:1934](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py), but the metric is satisfied by any `belief_revision_event` at [src/irys/matter/matter.py:2013](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py). That includes automatic causes like `new_evidence`, `conflict_detection`, and `trust_override` from [src/irys/matter/enums.py:102](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/enums.py), so a matter can report success without a user correction path ever being exercised.

**Answers**
1. The SQL `WHERE` clause in `8c553c4` is correct. It matches the same guard used by [`request_stop()` / `request_redirect()`](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py) at [src/irys/matter/reasoning.py:214](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py) and [src/irys/matter/reasoning.py:235](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/reasoning.py). The `objective IS NULL` part is necessary because `NOT IN (...)` alone would drop `NULL` objectives.

2. I did not find any other current non-steerable run types. The only explicit utility `run_session.objective` values written in the tree are `manual_flush` and `background_flush` at [src/irys/service/api.py:1693](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py), [src/irys/service/api.py:2144](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py), and [src/irys/ui/backends/in_process.py:28](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/ui/backends/in_process.py).

3. Yes, the old-row MEDIUM concern can still produce false positives.

4. Remaining manufactured-compliance issues in `get_so_metrics()` are:
- historical non-flush `run_session` rows still counting toward `steerability`
- `belief_revision` counting any revision event, not specifically user-correction propagation

Rating: **MEDIUM**

Static review only; I did not run tests.