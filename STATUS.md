# Project Status

Last updated: 2026-04-03 (post-adversarial-audit #020 + SO-4 weighted coverage fix)
Branch: SebihSpecial

---

## Current Phase

**Phase 6: Intelligence Substrate Hardening** — All Priority 0/1/2 components are built.
Active work is deepening correctness, adversarial audit hardening, and closing
architectural gaps discovered through Tier 1 / adversarial Codex reviews.

---

## Sacred Outcomes (SO-1 through SO-7)

| Outcome | Status | Notes |
|---------|--------|-------|
| SO-1: Durable Matter Model | **PARTIAL** | Audit #019: reuse opportunistic, no >70% gate. Schema v29 adds measurable reuse_rate to run_session; target in get_so_metrics() |
| SO-2: Typed Assertion Graph + Truth Maintenance | **PARTIAL** | Audit #021: correct_assertion() now triggers proof_state recompute (stale issue steering fixed). MAX_WORK truncation surfaces in ledger. Remaining: correction doesn't re-trigger loop reprioritization in real-time |
| SO-3: User-Steerable Reasoning | **PASS** | Audit #021 PASS — stop/redirect/correction/clarification/annotations/trust all wired |
| SO-4: Issue-Driven Architecture | **PARTIAL** | Audit #021: proof_state.sufficiency overlay removed (was bypassing weighted coverage). Bare-string facts no longer auto-credited as supports. Remaining: issue_predicate.resolved writer absent in production |
| SO-5: Source-Aware Intelligence | **PARTIAL** | Audit #021: multi-source ambiguity collapsed in output (complaint allegation vs contract clause — strongest role wins, ambiguity lost) |
| SO-6: Quantitative Intelligence | **PARTIAL** | Audit #021: quant is a sidecar subsystem, not core retrieval/proof backbone. Real gates + reconciliation, but not mandatory in all proof paths |
| SO-7: Missingness Modeled | **PASS** | Audit #021 PASS — gaps recorded, proof gaps detected, high-materiality gaps generate clarifications |

---

## What Is Built

### Priority 0 — Intelligence Substrate

| Component | Status | Schema / Notes |
|-----------|--------|----------------|
| Persistent matter model + canonical stores | **COMPLETE** | graph.py; MatterModel.open_in_memory() / open_at() |
| Matter registry | **COMPLETE** | MatterStore |
| Repository inventory store | **COMPLETE** | DocumentInventoryStore; SHA256 keyed; ingest_status tracking |
| Document card store | **COMPLETE** | DocumentInventoryStore with metadata + source_role |
| Span store | **COMPLETE** | span table; citation_grounding in assertions |
| Actor/contact store | **COMPLETE** | ActorStore; alias resolution; merge_actors(); find_possible_duplicates() |
| Typed assertion graph | **COMPLETE** | assertion + assertion_occurrence + assertion_link tables; v26 ix_link_type index |
| Evidence store (support/attack/corroboration) | **COMPLETE** | assertion_link; AssertionLinkType enum |
| Issue model / structured issue tree | **COMPLETE** | IssueStore; issue_predicate; issue_element; assertion_issue_link |
| Assumption store | **COMPLETE** | AssumptionStore |
| Gap store (structured missingness) | **COMPLETE** | GapStore; gap recording in contradiction mining |
| Quant store (amounts, dates, formulas) | **COMPLETE** | QuantStore; compute_thresholds(); reconcile_payment_chain() |
| Authority store | **COMPLETE** | AuthorityStore; schema v22 |
| Decision-context store | **COMPLETE** | DecisionContextStore; schema v21 |
| Work-product store | **COMPLETE** | WorkProductStore |
| Reasoning ledger (structured, user-facing) | **COMPLETE** | ReasoningLedger; run tracking; belief_revision_event |
| Belief revision / truth maintenance | **COMPLETE** | BeliefRevisionEngine; BFS propagation; trust-weighted confidence (SO-5) |
| User steering + interruptibility | **COMPLETE** | InterruptibleRun; correct_assertion(); force_state(); annotation store |
| Clarification engine | **COMPLETE** | ClarificationEngine; targeted gap questions |
| Source-role / agenda modeling | **COMPLETE** | source_role in assertion_occurrence; ProofStateStore.SOURCE_TRUST; advocacy gate |
| Repository intelligence | **COMPLETE** | detect_document_version_chains(); get_operative_version(); version families |

### Priority 1 — Higher-Order Reasoning

| Component | Status | Notes |
|-----------|--------|-------|
| Decision-context overlays | **COMPLETE** | schema v21; _build_decision_context_block() in engine |
| Legal research layer | **COMPLETE** | AuthorityStore; schema v22 |
| Quantitative intelligence | **COMPLETE** | QuantStore; schema v23; damages waterfall |
| Proof-aware reasoning | **COMPLETE** | ProofStateStore; schema v23; _enforce_quant_threshold_gate() |
| Adversarial / source-calibration reasoning | **COMPLETE** | _enforce_advocacy_gate(); trust-weighted belief revision |
| Attention allocation | **COMPLETE** | AttentionAllocationStore |
| Background maintenance loops | **COMPLETE** | mine_contradictions(); detect_version_chains() wired automatically |

### Priority 2 — Presentation Surfaces

| Component | Status | Notes |
|-----------|--------|-------|
| Timelines from structured state | **COMPLETE** | TimelineView |
| Issue-evidence matrices | **COMPLETE** | EvidenceMatrix |
| Damages waterfalls | **COMPLETE** | DamagesWaterfall; reconcile_payment_chain() |
| Communication maps | **COMPLETE** | CommunicationMap |
| Actor resolution API | **COMPLETE** | merge_actors(); resolve_by_name(); REST endpoints |
| Multi-matter isolation | **COMPLETE** | MatterIsolation; matter_id scoping verified |

---

## Adversarial Audit History

| Audit | Status | Key Findings |
|-------|--------|--------------|
| #011–#014 | PASS/PARTIAL cycle | Progressive SO coverage |
| #015 | SO-1/2/3/4/5/7 PASS; SO-6 PARTIAL | Hard quant gate added |
| #016 | SO-1/2/3/4/7 PASS; SO-5/6 PARTIAL | Hard advocacy + quant gates added |
| #017 | SO-1–SO-6 PASS; SO-7 PARTIAL → **FIXED** | Gap count now shows total vs filtered |
| #018 | SO-1/5/7 PASS; SO-2/3/4/6 PARTIAL | SO-5 FAIL fixed; incremental backlog for others |
| #019 | SO-2/4/5/6/7 PASS; SO-1/3 PARTIAL | SO-1 downgraded (no reuse gate); SO-3 ledger actionability gap |
| #020 | SO-3/7 PASS; SO-1/2/5/6 PARTIAL; SO-4 FAIL → FIXED | coverage_fraction was count heuristic; fixed to belief-state-weighted |
| #021 | SO-3/7 PASS; SO-1/2/4/5/6 PARTIAL | proof_state override removed; correct_assertion now refreshes proof_state; bare-string inflation fixed |

**Tier 1 reviews:** Correctness CLEAN (SO-4, SO-2 fixes; #021 fixes in progress). Performance CLEAN (schema v33 covering index).

---

## Active Work

### JUST COMPLETED — Adversarial audit #021 fixes

**SO-4 PARTIAL (audit #021) — fixes:**
- `_get_issue_coverage_map()` no longer replaces belief-state-weighted `coverage_fraction` with count-based `proof_state.sufficiency`. Only metadata overlays (advocacy_only, contested, proof_status) taken from proof_state.
- Bare-string facts no longer auto-credited as `supports` when issue-targeted. Now always `neutral`, consistent with dict-fact fallback — prevents unclassified facts from inflating coverage.

**SO-2 PARTIAL (audit #021) — fix:**
- `correct_assertion()` now triggers targeted `proof_state.compute_and_store()` recompute for all issues linked to the corrected assertion and its propagated dependents. Loop steering no longer uses stale proof_state after user correction.

**SO-2 PARTIAL (audit #020) — MAX_WORK truncation signal:**
- `BeliefRevisionEngine.apply()` writes `SYSTEM_WARNING` ledger event when MAX_WORK hit.
- Routes through canonical `ledger.append_event()` — preserves `_seq_cache` to prevent seq_no collision.
- `BeliefRevisionEngine` receives ledger at construction from `MatterModel`.

**Schema v33 (Tier 1 Performance MEDIUM):**
- `ix_ail_issue_rel_assertion ON assertion_issue_link(issue_id, relation_type, assertion_id)` — covering index for SO-4 support queries.

HEAD: e79181d

---

## Known Blockers

None active.

---

## Key Metrics (Current)

- Tests passing: 717 / 717
- Schema version: v33
- SO-3, SO-7: PASS; SO-1, SO-2, SO-4, SO-5, SO-6: PARTIAL
- Tier 1 correctness: **CLEAN** (SO-4 fix; SO-2 ledger routing; #021 fixes — review in progress)
- Tier 1 performance: **CLEAN** (schema v33 covering index on assertion_issue_link)
- Adversarial audit #021: DONE — proof_state override removed; correct_assertion refreshes proof_state; bare-string inflation fixed

## Architectural Backlog (Tier 2 HIGH remaining)

- **Q4 HIGH**: No immutable provenance/version model for legal facts after revision — requires schema redesign (assertion revision records or valid-time slices); needs Codex design gate
- **Q6 LOW**: get_ledger_steering_surface() steering snapshot caching — not blocking
- **Q7 MEDIUM**: _run_snapshots dict not persistent for multi-worker/clustered — not blocking for single-worker
- **SO-4 PARTIAL**: issue_predicate.status='resolved' writer absent in production (only in tests) — predicates always 0% satisfied; predicate-aware coverage formula never reaches full benefit
- **SO-5 PARTIAL**: Multi-source ambiguity collapsed (complaint vs contract same proposition → strongest role wins, ambiguity lost in output)
- **SO-2 PARTIAL**: Corrections propagate to proof_state now, but don't trigger real-time loop re-prioritization in the same iteration
- **SO-1 PARTIAL**: reuse_rate metric is assertion-count ratio, not measured reuse behavior
- **SO-6 PARTIAL**: Quant is sidecar subsystem, not core retrieval/proof backbone
- **Q4 HIGH**: No immutable provenance/version model — schema redesign needed (assertion revision records or valid-time slices)
