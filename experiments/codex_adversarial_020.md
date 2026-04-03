### HOSTILE AUDIT (SebihSpecial, schema v32)

## Verdicts

1. **SO-1 (Durable model + reuse):** **PARTIAL**  
`Matter` run persistence and document cache reuse are real (`Matter.start_run`, `Matter.complete_run`, `MatterRuntimeAdapter._deep_read_document` hot-path `is_ingested`, `document_inventory`, [matter.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py), [graph.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py)).  
But `reuse_rate` is a **count ratio** (`assertions_at_start / assertions_at_end`) in run metrics, not a measured reuse-behavior metric; it can be gamed by assertion churn without actual cache reuse. This is a classic “metric theater” signal for SO-1.

2. **SO-2 (Fixpoint BFS belief revision):** **PARTIAL**  
`BeliefRevisionEngine.apply()` runs a deque fixpoint pass and propagates dependents ([belief_revision.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py)), so there is real revision logic.  
The hard stop `MAX_WORK = 500` can terminate revision silently with only a warning log; on legal graphs with dense dependencies, stale belief states can remain without any run-level failure signal. This is exactly the “appears to converge, but may not” problem.

3. **SO-3 (User-steerable reasoning):** **PASS**  
`request_stop`, `request_redirect`, and `clarification` correction paths are persisted and polled during reasoning, then acted on in the loop ([runtime.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/runtime.py), [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py), [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py)).  
Not a full pass-quality scorecard because this can still be interrupted mid-pipeline, but behavior is genuinely wired, not a stub.

4. **SO-4 (Predicate-aware `coverage_fraction` = proof sufficiency):** **FAIL**  
`coverage_fraction` is computed as support-count over predicate-count (`supports`/`predicates`, with a zero-denominator fallback), i.e., a **counting heuristic** with no proof-strength or contradiction-quality gating ([matter.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py)).  
That does not represent sufficiency of proof, only volume-of-links coverage.

5. **SO-5 (Source-aware intelligence / role calibration):** **PARTIAL**  
Source role and trust are actually consumed in belief calculations and overrides are applied (`effective_source_role` + trust weighting), so it is not a dead feature ([graph.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py), [belief_revision.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py)).  
However, there is no visible adaptive calibration loop grounded in outcomes; roles are applied, not continually calibrated, so SO-5’s stronger interpretation is only partially met.

6. **SO-6 (Quantitative intelligence):** **PARTIAL**  
The quant subsystem exists and has reconcile/conflict functions, but the evidence trail is mostly data-layer centered and I do not see guaranteed downstream reasoning behavior shown as mandatory in the audited paths ([graph.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py)).  
Stored numeric facts are real, but not clearly part of the core proof/decision loop.

7. **SO-7 (Explicit missingness modeling):** **PASS**  
Gaps are persisted, deduped, reopened/resolved, and connected to assertion links, and missingness influences proof-gap detection ([graph.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py), [engine.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/rlm/engine.py), [api.py](/C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/service/api.py)).

## Highest-risk self-deception patterns

1. `SO-2`: the system can claim truth-maintenance yet silently stop propagation at `MAX_WORK`, leaving potentially inconsistent states.  
2. `SO-4`: coverage is labeled as sufficiency but is only count-based.  
3. `SO-1`: “reuse” success is inferred from assertion counts, not from actual reused intelligence or savings.

## Additional red flags

- `Matter.get_so_metrics()` returns `steerability = True` unconditionally, which overstates capability if steering is unavailable in specific run states or paths.  
- Several trust and revision branches include exceptions (`ValueError` in belief-state vector role checks) without a user-visible recovery path in the same flow.