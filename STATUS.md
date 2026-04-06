# Project Status

Last updated: 2026-04-05 (Tier 1 CLEAN r4: Q4/SO-2 stale pre-state fully closed; OCC + write_transaction())
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
| SO-1: Durable Matter Model | **PASS** | Audit #022 PASS — hot path genuinely reuses persisted state via run-start context, hydration, and caches |
| SO-2: Typed Assertion Graph + Truth Maintenance | **PARTIAL** | Audit #022: belief revision + proof_state recompute real. SPO retry now covers partial coverage. assertion_revision table (v34) closes Q4 HIGH. Remaining: success signal now exposes propagation_truncated; review SO-2 coverage rate |
| SO-3: User-Steerable Reasoning | **PASS** | Audit #022 PASS — stop/redirect/correction/clarification/annotations/trust all wired |
| SO-4: Issue-Driven Architecture | **PARTIAL** | Audit #022: weighted coverage used by loop. Predicate resolution now wired in production (ANALYZE_FINDINGS_PROMPT step 5 → resolve_predicate_by_description). Remaining: _focus_issue_id attribution is heuristic |
| SO-5: Source-Aware Intelligence | **PASS** | Audit #022 PASS — source-role classification, trust overrides, advocacy-only gating are real behavioral changes |
| SO-6: Quantitative Intelligence | **PASS** | Audit #022 PASS — numeric facts structured, persisted, reconciled, conflict-checked, force-surfaced |
| SO-7: Missingness Modeled | **PASS** | Audit #022 PASS — gaps tied to issues/assertions, surfaced in synthesis, converted to clarifications |
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
| #022 | **SO-1/3/5/6/7 PASS; SO-2/4 PARTIAL** | 5 PASSes. Predicate resolution wired in production. SO-4 remaining: _focus_issue_id attribution heuristic |

**Tier 1 reviews:** CLEAN — predicate resolver matter-scoped + atomic, correct_assertion batched, SO-4 predicate production wiring with allowlist + gating, _orient() field normalization + _parse_json_safe non-dict guard, Q4/SO-2 stale pre-state (4 rounds: write_transaction()/BEGIN IMMEDIATE, in-tx diff checks, pre-tx early-exit removed, OCC conflict abort), all Tier 1 HIGH/MEDIUM resolved.

---

## Active Work

### JUST COMPLETED — Tier 1 CLEAN r4: Q4/SO-2 stale pre-state (2026-04-05)

**Stale pre-state fix — 4 rounds to CLEAN:**
- **Round 1 (b8714e7):** Re-read `belief_state`/`confidence` inside transaction for `old_value_json` in `assertion_revision` — wrong audit values under concurrent write.
- **Round 2 (dc0f61b):** `write_transaction()` / `BEGIN IMMEDIATE` added to `SQLiteMatterDB` — prevents WAL SQLITE_BUSY_SNAPSHOT under concurrent writers. Both `_revise_one()` and `force_state()` use it. In-tx diff checks use `_intx_old_state`/`_intx_old_conf` (not stale pre-tx values) for both change detection AND `old_value_json`.
- **Round 3 (83841e5):** Removed pre-tx early-exit from `_revise_one()` — the stale no-change check was bypassing the write_transaction entirely. No-change now detected INSIDE the transaction via `if not _rev_rows: return None`. `RevisionResult.old_belief_state`/`old_confidence` updated to use in-tx values so callers (api.py, runtime.py) see the committed transition.
- **Round 4 (79aa097 + 9b6387a):** OCC check added in `_revise_one()`: if in-tx state diverged from pre-tx snapshot (concurrent writer committed), abort rather than overwrite with stale BFS target. Float tolerance aligned (`>= 0.001` across OCC + diff detection).

**assertion_revision table (schema v34, 2026-04-03):**
- New append-only `assertion_revision` table: field-level audit log for every mutation to assertion.belief_state, confidence
- `write_revision_rows()` in AssertionStore; `force_state()` writes `actor_kind=user`; `_revise_one()` writes `actor_kind=system`; `upsert_occurrence()` writes `cause=occurrence_upgrade`
- `RevisionResult.propagation_truncated: bool`; `_apply_with_truncation()` private BFS driver
- Schema v34, additive migration; 4 new tests

**SPO retry partial coverage (2026-04-03):**
- Retry condition: `_spo_count < len(facts_to_add)` (was `== 0`)
- Merge fix: `retry.get(i) if spo is None else spo` — preserves primary-extraction SPO

HEAD: 9b6387a — Tests: 725/725

### PREVIOUSLY COMPLETED — Tier 1 CLEAN: _orient() field normalization + _parse_json_safe

**Tier 1 Correctness CLEAN confirmed (2026-04-03):**
- `_parse_json_safe()`: `isinstance(result, dict)` guard after `json.loads()` — null/[]/string root returns defaults safely
- `_orient()` hypothesis: `isinstance(_hyp, str)` normalization before assignment — non-string truthy values never reach `[:200]` slice
- `_orient()` issues: `isinstance(..., list)` guard before storage AND iteration at L1216/1217/1241/1243
- `_orient()` initial_searches: `isinstance(..., list)` guard before slice and iteration at L1296/1297/1299
- `_ORIENTATION_CACHE_VERSION` bumped to "4" to invalidate stale cached plans
- `_raw_idx_to_issue_id` mapping: raw LLM issue_idx → issue_id (fixes attribution when filtered issues shrink)
- HEAD: 7b9f440 — Tests: 721/721

### PREVIOUSLY COMPLETED — Adversarial Audit #022 + SO-4 Predicate Production Wiring

**Audit #022 results: 5 PASSes (SO-1/3/5/6/7)**
- SO-1 PASS: hot path genuinely reuses persisted state (run-start context, hydration, caches)
- SO-5 PASS: source-role classification + advocacy gating are real behavioral changes
- SO-6 PASS: numeric facts structured, persisted, reconciled, conflict-checked
- SO-7 PASS: gaps tied to issues/assertions, surfaced in synthesis, converted to clarifications

**SO-4 predicate production wiring (post-audit #022):**
- `ANALYZE_FINDINGS_PROMPT` now includes step 5 asking LLM to identify which Issue Focus elements are established by the extracted facts
- `resolve_predicate_by_description()` called for each satisfied predicate after fact recording
- Allowlist guards: only predicates shown in Issue Focus block (limit=2) are eligible
- Gate: only resolves when `any(_search_assertion_ids)` — supporting facts actually persisted
- Cache key includes predicate hash so changing predicates invalidates cached analysis
- `_build_issue_focus_block()` now returns `(block, pred_descs)` tuple eliminating duplicate DB read

**Tier 1 loop CLEAN:** 6 Codex sessions (correctness + performance) all clean after fixes.

HEAD: 7b9f440
Tests: 721 / 721

### PREVIOUSLY COMPLETED — Tier 1 Correctness CLEAN

**Predicate resolver correctness (Tier 1 HIGH/MEDIUM from predicate review):**
- `resolve_predicate()` matter-scoped: WHERE clause now JOINs through issue.matter_id — prevents cross-matter predicate resolution.
- `resolve_predicate_by_description()` rewritten: single atomic UPDATE with correlated subquery (eliminates TOCTOU), None/empty guard, full-text description match (removes 300-char truncation mismatch).
- `correct_assertion()` batching: chunked iteration over affected assertion IDs (900/batch) so ALL affected issues are discovered, not just the first 900.
- `_score()` docstring updated to document zero-resolved-predicates fallback.
- 3 new tests: cross-matter guard, None/empty guard, full-text description resolution.

**SO-5 multi-source ambiguity fix:**
- `list_recent_for_hydration()` returns `source_roles_csv` via GROUP_CONCAT(DISTINCT).
- Engine display shows `MULTI-SOURCE[OPERATIVE,ADVOCACY]` when same proposition spans docs of different source roles.

HEAD: b7ade5f
Tests: 721 / 721

### PREVIOUSLY COMPLETED — Adversarial audit #021 fixes

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

- Tests passing: 725 / 725
- Schema version: v34
- SO-1/3/5/6/7: **PASS**; SO-2/4: **PARTIAL** (SO-2 improving)
- Tier 1 Q4/SO-2 stale pre-state: **CLEAN** (4 rounds, r4 confirmed 2026-04-05)
- Adversarial audit #022: DONE — 5 PASSes, SO-2/4 PARTIAL
- Q4 HIGH: CLOSED — assertion_revision table (schema v34) + stale pre-state OCC fix
- Next: Adversarial audit #023 (OVERDUE — 9+ Codex sessions since #022); SO-4 attribution design gate

## Architectural Backlog (Tier 2 HIGH remaining)

- **Q4 HIGH**: ~~No immutable provenance/version model~~ CLOSED — assertion_revision table (schema v34) provides field-level audit log for all mutations
- **Q6 LOW**: get_ledger_steering_surface() steering snapshot caching — not blocking
- **Q7 MEDIUM**: _run_snapshots dict not persistent for multi-worker/clustered — not blocking for single-worker
- **SO-4 PARTIAL**: ~~issue_predicate.status='resolved' writer absent in production~~ CLOSED — predicate resolution wired in production via ANALYZE_FINDINGS_PROMPT step 5 + allowlist guard (commit a07c0bf)
- **SO-2 PARTIAL**: SPO retry now covers partial batches; success signal exposes propagation_truncated. Remaining: measure actual SPO coverage rate in production runs
- **SO-4 PARTIAL**: _focus_issue_id attribution still heuristic for loop-phase leads; initial_searches now use LLM-provided issue_idx but fallback is still weakest_id
- **SO-2 PARTIAL**: Corrections propagate to proof_state now, but don't trigger real-time loop re-prioritization in the same iteration
- **SO-1 PARTIAL**: reuse_rate metric is assertion-count ratio, not measured reuse behavior
- **SO-6 PARTIAL**: Quant is sidecar subsystem, not core retrieval/proof backbone
- **Q4 HIGH**: CLOSED — assertion_revision table (schema v34)
