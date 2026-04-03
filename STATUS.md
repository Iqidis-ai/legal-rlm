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
| SO-2: Typed Assertion Graph + Truth Maintenance | **PARTIAL** | Audit #020: MAX_WORK=500 can silently truncate revision; partial-result documented + warning logged |
| SO-3: User-Steerable Reasoning | **PASS** | Audit #020 PASS — steering mechanisms wired; get_ledger_steering_surface() structured |
| SO-4: Issue-Driven Architecture | **PASS** | Audit #020 FAIL fixed: coverage_fraction now belief-state-weighted (operative=1.0, alleged=0.5, other=0.3) |
| SO-5: Source-Aware Intelligence | **PARTIAL** | Audit #020: roles applied but no adaptive calibration loop grounded in outcomes |
| SO-6: Quantitative Intelligence | **PARTIAL** | Audit #020: quant subsystem real but not clearly mandatory in core proof/decision loop |
| SO-7: Missingness Modeled | **PASS** | Audit #020 PASS — gaps persisted, deduped, reopened/resolved, connected to assertions |

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

**Tier 1 reviews:** Correctness CLEAN (SO-4 changes). Performance CLEAN (manual analysis — single-query aggregations, no O(issues) round trips).

---

## Active Work

### JUST COMPLETED — Adversarial audit #020 SO-4 FAIL fix

**SO-4 FAIL (audit #020) → FIXED:**
- `coverage_fraction` was raw assertion count ratio — no proof-strength weighting
- Fixed: belief-state-weighted coverage in both `get_issue_coverage_report()` and `_investigate_context()` weakest-issue selector
- Weights: operative/admitted/resolved=1.0, alleged/argued/inferred=0.5, other active=0.3
- Excluded: disputed/withdrawn/superseded
- `supporting_count` field preserves raw integer count; `coverage_fraction` uses weighted computation
- Removed phantom 'partial' belief-state from SQL weight buckets (BeliefState has no PARTIAL)
- Tier 1 Correctness CLEAN; Tier 1 Performance CLEAN (manual)
- Tests: 715/715

### Previously completed — Tier 2 Architecture+Scaling milestone review fixes (prior session)

Q1 HIGH: layer-aware proposition lookup; Q2 HIGH: fixpoint BFS; Q3 MEDIUM: predicate-aware coverage;
Q5 MEDIUM: schema v31 indexes; schema v32 no-layer proposition index.

HEAD: d342e8a

---

## Known Blockers

None active.

---

## Key Metrics (Current)

- Tests passing: 715 / 715
- Schema version: v32
- SO-3, SO-4, SO-7: PASS; SO-1, SO-2, SO-5, SO-6: PARTIAL
- Tier 1 correctness: **CLEAN** (SO-4 fix)
- Tier 1 performance: **CLEAN** (manual analysis)
- Adversarial audit #020: DONE — SO-4 FAIL fixed; SO-3/7 upgraded to PASS
- Next adversarial audit (#021): OVERDUE — many Codex sessions since #020

## Architectural Backlog (Tier 2 HIGH remaining)

- **Q4 HIGH**: No immutable provenance/version model for legal facts after revision — requires schema redesign (assertion revision records or valid-time slices); needs Codex design gate
- **Q6 LOW**: get_ledger_steering_surface() steering snapshot caching — not blocking
- **Q7 MEDIUM**: _run_snapshots dict not persistent for multi-worker/clustered — not blocking for single-worker
- **SO-2 PARTIAL**: MAX_WORK=500 silent truncation; warning logged but run-level failure not raised — decision needed: raise vs continue-with-warning
- **SO-5 PARTIAL**: Source roles applied but no adaptive calibration loop grounded in outcomes
- **SO-6 PARTIAL**: Quant subsystem real but not clearly mandatory in core proof/decision loop
- **SO-1 PARTIAL**: reuse_rate metric is assertion-count ratio, not measured reuse behavior
