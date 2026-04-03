# Project Status

Last updated: 2026-04-03
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
| SO-1: Durable Matter Model | **PASS** | DocumentInventoryStore; operative version enforced; SHA256-keyed re-ingest |
| SO-2: Typed Assertion Graph + Truth Maintenance | **PASS** | Trust-weighted belief revision; split try/except in mining loop; detect_heuristic_contradictions() |
| SO-3: User-Steerable Reasoning | **PASS** | force_state(), correct_assertion(), document trust overrides, annotation store |
| SO-4: Issue-Driven Architecture | **PASS** | _enrich_search_term_with_issue_context() + _build_issue_focus_block() in analysis prompt |
| SO-5: Source-Aware Intelligence | **PASS** | _enforce_advocacy_gate() hard post-synthesis gate; trust-weighted confidence in belief revision |
| SO-6: Quantitative Intelligence | **PASS** | _enforce_quant_threshold_gate() content-based; QuantStore.compute_thresholds() |
| SO-7: Missingness Modeled | **PASS** | GapStore; gap recording isolated from force_state failures |

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
| #017 | Due in ~4-5 Codex sessions | — |

**Tier 1 reviews:** #016–#018: issues fixed; #019: 3 LOW findings (all fixed)

---

## Active Work

### JUST COMPLETED — Trust-Weighted Belief Revision (SO-2, SO-5)

Advocacy-source attackers (weight 0.3) now inflict less confidence damage than
operative-source attackers (weight 1.0). The state transition logic is unchanged
(any active attack → DISPUTED) — only confidence magnitude is trust-weighted.

`get_neighbor_belief_states()` now fetches best source_role per neighbor via
correlated subquery. `_compute_belief_state()` accepts `support_source_roles`
and `attack_source_roles`; backward-compatible (None → weight=1.0).

### QUEUED — Document Trust Override → Belief Revision Integration

User-set document trust overrides (DocumentTrustOverrideStore) currently flow
to the LLM prompt but NOT to the structured belief revision system. When a user
marks a document pattern as "low" trust, assertions from that document should
be treated as advocacy-weight (0.3) in belief revision, not their original
source_role. Requires Codex design gate before implementation.

---

## Known Blockers

Codex (GPT-5.3-Codex-Spark) rate-limited until 1:19 PM EDT on 2026-04-03.
Tier 1 Correctness + Performance reviews for trust-weighted belief revision
queued; will run immediately on rate limit reset.

---

## Key Metrics (Current)

- Tests passing: 670 / 670
- Schema version: v26
- Commits since session start: 5
- All 7 Sacred Outcomes: PASS
- Tier 1 reviews pending: 2 (queued, awaiting Codex rate limit reset)
- Adversarial audit #017: due in ~4 Codex sessions
