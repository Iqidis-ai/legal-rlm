Audit #019 result (hostile):

## Verdicts

- **SO-1 — PASS**
  - The matter model is real, persisted, and re-used across runs, not just declared in docs.
  - Evidence: schema objects exist for matter/assertion/proof/events and are indexed for reuse (`.../src/irys/matter/schema.py:11`, `.../src/irys/matter/schema.py:26`, `.../src/irys/matter/schema.py:444`).
  - Evidence: run start/stop persists into matter model and engine hydration actually reads prior assertions (`.../src/irys/rlm/engine.py:1608`, `.../src/irys/matter/graph.py:489`).
  - Evidence: persisted document inventory is read to avoid re-processing old documents (`.../src/irys/rlm/engine.py` deep-read path, `.../src/irys/matter/schema.py:163`).
  - Evidence: matter context is built from stored counts/open-issues/gaps at run time (`.../src/irys/matter/matter.py:364`, `.../src/irys/rlm/engine.py:1113`).

- **SO-2 — PARTIAL**
  - Propagation exists and is triggered on edits/trust overrides, but it is not a fully general truth-maintenance calculus.
  - Evidence: typed assertions + occurrences are materially stored (`.../src/irys/matter/schema.py:26`, `.../src/irys/matter/schema.py:51`).
  - Evidence: revision engine applies to dependents, recomputes belief states, and records revision events (`.../src/irys/matter/belief_revision.py:162`, `.../src/irys/matter/belief_revision.py:225`, `.../src/irys/matter/matter.py:184`, `.../src/irys/matter/matter.py:211`).
  - Evidence: dependency traversal is directional and bounded by engine logic (support/attack/corroborates/supersede edges, neighbor state weighting) (`.../src/irys/matter/graph.py:242`, `.../src/irys/matter/graph.py:307`, `.../src/irys/matter/graph.py:419`).
  - **Risk:** “correct truth-maintenance semantics” appears to be approximated (weighted rule updates, hop-limited revision), so this is behaviorally present but likely not complete logical closure.

- **SO-3 — PARTIAL**
  - User steering exists and can be acted on, but only at bounded engine checkpoints rather than immediate hard interrupts.
  - Evidence: stop/redirect fields in run session and persistence methods exist (`.../src/irys/matter/schema.py:123`, `.../src/irys/matter/reasoning.py:167`).
  - Evidence: API endpoints create stop/redirect requests and events (`.../src/irys/service/api.py:1344`, `.../src/irys/service/api.py:1372`).
  - Evidence: engine loop polls these flags and branches into redirect handling / cancellation paths (`.../src/irys/rlm/engine.py:876`, `.../src/irys/rlm/engine.py:1393`, `.../src/irys/rlm/engine.py:1562`, `.../src/irys/rlm/engine.py:1689`, `.../src/irys/rlm/engine.py:1794`).
  - **Risk:** if “interrupt now” is interpreted as preemptive cancellation, implementation does not provide that; behavior is deferred.

- **SO-4 — PASS**
  - Issue coverage data is not dead schema; it is consumed to shape retrieval/generation behavior.
  - Evidence: query context explicitly builds issue coverage/open-issue signals from persisted model (`.../src/irys/matter/matter.py:378`, `.../src/irys/matter/matter.py:423`, `.../src/irys/matter/matter.py:462`).
  - Evidence: engine uses coverage map for lead targeting and issue-focused prioritization (`.../src/irys/rlm/engine.py:3511`, `.../src/irys/rlm/engine.py:3561`).
  - Evidence: proof-gap detection/persistence uses issue weakness to alter next-step behavior (`.../src/irys/rlm/engine.py:4533`, `.../src/irys/rlm/engine.py:4539`).
  - Evidence: issue/gap APIs return those computed outputs (so the model is both used and exposed) (`.../src/irys/service/api.py:1576`, `.../src/irys/service/api.py:1694`).

- **SO-5 — PASS**
  - Source-aware behavior is wired through data and revision/proof scoring paths.
  - Evidence: source role is persisted on assertion occurrences (`.../src/irys/matter/schema.py:57`).
  - Evidence: trust-source weighting and neighbor belief derivation explicitly read that role (`.../src/irys/matter/belief_revision.py:36`, `.../src/irys/matter/graph.py:390`, `.../src/irys/matter/graph.py:412`).

- **SO-6 — PARTIAL**
  - Numbers are stored structurally and used in computation paths, but usage appears selective and thresholded rather than globally coupled.
  - Evidence: `quant_fact` schema exists with structured fields and dedupe/hash fields (`.../src/irys/matter/schema.py:444`, `.../src/irys/matter/schema.py:463`).
  - Evidence: extraction writes structured numeric facts in batches from deep-read (`.../src/irys/rlm/engine.py:2359`).
  - Evidence: conflicts are detected and emitted into matter/gate/pipeline structures (`.../src/irys/matter/matter.py:621`, `.../src/irys/rlm/engine.py:3376`).
  - **Risk:** numeric model is used at certain gates and summaries, but not yet clear as a full symbolic/calculation graph across all reasoning paths; appears feature-rich but partially enforced.

- **SO-7 — PASS**
  - Missingness is represented as first-class model state and surfaced into behavior/output.
  - Evidence: gap + gap_link schema is implemented with status/materiality and linkage (`.../src/irys/matter/schema.py:360`).
  - Evidence: gaps are recorded, opened/resolved, and translated to clarifications (`.../src/irys/matter/matter.py:346`, `.../src/irys/matter/matter.py:551`, `.../src/irys/matter/matter.py:532`).
  - Evidence: open gaps drive synthesis/runtime flow and are included in responses (`.../src/irys/rlm/engine.py:1053`, `.../src/irys/rlm/engine.py:4539`, `.../src/irys/service/models.py:79`).

## Reconsider / high-risk areas (not pass/soft)
1. **SO-2 truth maintenance depth and semantics** need explicit justification in ADR/spec: current behavior is principled but bounded/heuristic, so adversarial dependency chains can exceed intended semantics without full logical explosion handling.
2. **SO-3 interrupt semantics** are cooperative, not hard preemption; if UX expects immediate cancellation, current code can appear “responsive” while still finishing current chunk.
3. **SO-6** should explicitly document where numeric outputs are hard-gated into final recommendations so it cannot be misread as full quantitative reasoning everywhere.