## Architecture Theorist

- **HIGH — Q1: Layer separation is mostly present at write time, but broken at read time in some paths.**
  - Store-level dedupe now includes layer (`(matter_id, model_layer, proposition_key)`), so same text can exist across layers without collision (dedupe in one layer only).  
  - But lookup by text in `get_by_proposition` ignores layer and can return the wrong layered node for downstream logic. That weakens layer semantics and can leak record/proof/reality facts across tiers.
  - Fix: add `model_layer` to proposition lookup APIs and propagate explicit layer filtering at all semantic call sites; keep layer-only queries explicit.  
  - References: [src/irys/matter/schema.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L46), [src/irys/matter/graph.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L99), [src/irys/matter/graph.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L531)

- **HIGH — Q2: BFS revision is not a true convergence engine; it can produce stale states in cyclic/competing-evidence graphs.**
  - `BeliefRevisionEngine.apply()` marks nodes visited once and does not re-enqueue after a first visit, so a node can miss a second-pass recomputation in convergent or cyclic dependency structures.
  - `MAX_HOPS=10` also hard-truncates propagation chains; deep legal argument graphs can exceed this.
  - Fix: switch to event-driven requeue-on-change with change-set convergence until fixpoint (or explicit SCC traversal with iterative relaxation), and only cap by max total work/radius with configurable guardrails.
  - References: [src/irys/matter/belief_revision.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L95), [src/irys/matter/belief_revision.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L114), [src/irys/matter/belief_revision.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L136), [src/irys/matter/belief_revision.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/belief_revision.py#L144)

- **MEDIUM — Q3: `coverage_fraction = count / (count + 1)` is a heuristic, not a strong SO-4 proof metric.**
  - It is monotone and simple, but it saturates quickly (one assertion => 0.5), ignores expected predicate cardinality per issue, ignores attack pressure in coverage, and ignores trust/strength asymmetry.
  - It can overstate “coverage progress” for evidence that is not lawfully sufficient.
  - Fix: define coverage as a function of required predicates/claim elements and evidence quality (weighted support vs attack), e.g., `sufficient_predicate_ratio * trust_weighted_support_ratio`.
  - References: [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L495), [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L544)

- **HIGH — Q4 (architectural gap): No immutable provenance/version model for legal facts after revision.**
  - Current core writes mutate `assertion.belief_state` in place; no versioned fact lineages or validity windows per proposition are modeled.
  - That blocks robust legal auditability and adversarial review (what was true before a correction, who/when changed it, and why from which source).
  - Fix: introduce immutable assertion revision records (or valid-time slices) and keep derived belief as materialized projection over immutable base facts.  
  - References: [src/irys/matter/graph.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L140), [src/irys/matter/graph.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L254)

## Scaling Expert

- **MEDIUM — Q5 remaining hot-spot queries at 10k docs / 100k assertions (with v28-v30 indexes already in place).**
  - `get_issue_coverage_report()` still does a full coverage aggregate across `assertion_issue_link` + `assertion` + `issue` and then a proof-gap join; no `issue` or `assertion.belief_state` composite indexes (potentially scan-heavy in larger matters).  
  - `find_contradictions()` performs a full join of all attack/contradiction edges to assertion pairs; edge density grows faster than assertions.
  - `open_gaps()` uses 2 query pattern (`gap` + subquery + `gap_link`) and repeatedly scans links for candidate gaps; lacking `(matter_id, status, gap_type)` on `gap` can reduce selectivity.
  - `disputed assertion` query in steering has no `(matter_id, belief_state, updated_at)` index; scans can rise with assertion count.
  - References: [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L495), [src/irys/matter/graph.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L548), [src/irys/matter/graph.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L978), [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1260), [src/irys/matter/schema.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L46)

- **LOW — Q6: `get_ledger_steering_surface()` currently runs 7 DB queries per call (not 8), but many are repeated on each UI refresh.**
  - Breakdown: `find_contradictions` (1), `get_issue_coverage_report` (2), `open_gaps` (2), `clarifications.get_pending` (1), disputed assertions query (1) = 7.
  - Not blocking yet, but this is a repeated hot path; cache steering snapshots per run and invalidate by delta events to avoid repeated recompute.
  - References: [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1089), [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1161), [src/irys/matter/graph.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/graph.py#L548), [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L1260)

- **MEDIUM — Q7: `_run_snapshots` is not robust for multi-user long-lived concurrency.**
  - In-memory dict is fine for short-lived single-threaded usage, but it is not synchronized and can leak entries if runs never call completion/fail/interrupt paths.
  - In clustered deployments, each worker has separate memory state; recovery is uneven. It is also not run-scoped-safe under concurrent request access.
  - Fix: persist snapshot state in DB row (`run_session`) as authoritative source and keep in-memory as optional cache; add TTL/housekeeping for orphaned starts.
  - References: [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L79), [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L146), [src/irys/matter/matter.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/matter.py#L156), [src/irys/matter/schema.py](C:/Users/devan/OneDrive/Desktop/Projects/legal-rlm/src/irys/matter/schema.py#L1164)