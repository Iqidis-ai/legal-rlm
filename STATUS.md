# Project Status

Last updated: 2026-04-03 (post-adversarial-audit #019 + schema v29)
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
| SO-2: Typed Assertion Graph + Truth Maintenance | **PASS** | Audit #019 PASS — BFS propagation confirmed; belief revision chain tested |
| SO-3: User-Steerable Reasoning | **PARTIAL** | Audit #019: steering mechanisms PASS; ledger actionability gap. Added get_ledger_steering_surface() — structured action affordances |
| SO-4: Issue-Driven Architecture | **PASS** | Audit #019 PASS — live issue-coverage weakness drives lead scoring; per-claim reporting confirmed |
| SO-5: Source-Aware Intelligence | **PASS** | Tier 1 r8 CLEAN; structural violation gate; per-title hedge check |
| SO-6: Quantitative Intelligence | **PASS** | Audit #019 PASS — QuantStore read and used; reconcile_payment_chain() flows to user |
| SO-7: Missingness Modeled | **PASS** | Total gap count surfaced in synthesis; gap recording isolated |

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

**Tier 1 reviews:** Correctness r8 CLEAN. Performance r3 CLEAN.

---

## Active Work

### JUST COMPLETED — Adversarial Audit #019 + SO-1/SO-3 PARTIAL fixes

**Audit #019 results:** SO-2/4/5/6/7 all PASS. SO-1 downgraded (reuse opportunistic, no measurable threshold).
SO-3 PARTIAL maintained (steering mechanisms work; ledger surface was raw event logs).

**SO-1 fix (schema v29):**
- `run_session` gains `assertions_at_start INTEGER` + `reuse_rate REAL` columns
- `start_run()` snapshots assertion count; `complete_run()` computes `at_start / at_end`
- `get_so_metrics()` averages reuse_rate over 5 recent completed runs; target 0.70 in targets/targets_met
- 7 new tests verify snapshot, computation, and metric aggregation

**SO-3 fix — `get_ledger_steering_surface()`:**
- New `MatterModel.get_ledger_steering_surface(run_id, limit)` method
- Derives structured steering actions from: active conflicts, low-coverage issues,
  high-materiality missing-doc gaps, pending clarifications, disputed/unknown assertions
- Each action: `action_type`, `description`, `params` (ready to pass to API), `rationale`, `priority`, `impact`
- Sorted high → medium → low priority; 8 new tests

Previous session: SO-5 Tier 1 loop (8 correctness rounds + 3 performance rounds) CLEAN.

HEAD: ccbfbb5

---

## Known Blockers

None active.

---

## Key Metrics (Current)

- Tests passing: 708 / 708
- Schema version: v30
- SO-2, SO-4, SO-5, SO-6, SO-7: PASS; SO-1, SO-3: PARTIAL (adversarial #019, partially addressed)
- Tier 1 correctness: **CLEAN** (r4 on SO-1/SO-3 changes; r8 total on this session's work)
- Tier 1 performance: **CLEAN** (v30 index, in-memory snapshot)
- Adversarial audit #019: DONE — SO-2/4/6 upgraded to PASS; SO-1 downgraded; SO-3 PARTIAL
- Next adversarial audit (#020): due after ~3 more Codex sessions
