# Project Status

Last updated: 2026-04-06 (Tier 1 UI CLEAN declared r10; all MEDIUMs fixed; schema v37; 725 tests pass)
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
| #023 | **SO-1/3/5/6/7 PASS; SO-2/4 PARTIAL** | 5 PASSes. 3 HIGHs fixed: OCC silent loss, flat fact ingress (SPO threshold >= 1), SO-4 attribution (biased pool). Post-audit Tier 1 r5/r6: schema v35 no-op, OCC retry-self (not dependents), pre-tx fast-path, bare-idx counter, seedness preserved across OCC retries |
| #024 | **NEEDS MAJOR REWORK (backend ≠ UI)** | All 6 SOs FAIL/PARTIAL in UI: assertion IDs hidden (SO-2), steering not wired (SO-3), issue IDs hidden (SO-4), source role mis-rendered (SO-5), Quant panel missing (SO-6). **FIXED:** assertion IDs + issue IDs in tables; get_steering_surface() + get_quant_summary() wired to new Tab 6 Quant; BeliefState validation; redirect validation. Post-audit Tier 1 r6: redirect no-validation, backend type guard, answered_at index (v37), repo stats dedup |

**Tier 1 UI reviews: CLEAN (r10 PASS, 2026-04-06).** All HIGH/MEDIUM fixed across r7–r10:
- r7: UIBackend abstract methods + panel exception propagation
- r8: Full IDs in tables; HttpBackend correct routes; do_redirect() error check; run_investigation_thread() encapsulation; /steering-surface endpoint added to service
- r9 correctness: All 6 r8 fix points verified clean
- r9 perf: Removed unused recent_runs list fetch + source_role_summary GROUP BY from overview
- r10: Quant panel key mismatch (total_invoiced→invoiced etc); removed false issue_id contract from list_assertions; removed exception masking in HttpBackend.get_quant_summary()

**Tier 2 r2 Scaling + Architecture reviews: FAIL (HIGHs fixed, 2026-04-06):**
- Scaling HIGH: `find_possible_duplicates()` O(N²) → sorted-prefix O(N·k) + limit=100 (commit 9101136)
- Scaling MEDIUM: `list_recent()` O(N) before LIMIT → CTE bounds ID set first (commit 9101136)
- Scaling MEDIUM: `find_contradictions()` no limit → limit=5 pushed to SQL in steering surface (commit 9101136)
- Arch HIGH: `redirect_focus` actions missing `matter_id`+`run_id` → now embedded; get_steering_surface() threads run_id through full stack (commit 9101136)
- Scaling HIGH (architectural): InProcessBackend not production-scalable (unbounded cache, SQLite connections) — architectural note; use HttpBackend+service in production
- Arch HIGH (deferred): redirect is iteration-bound, not interrupt-grade — stop is sub-second, redirect latency = full current-batch duration
- Arch MEDIUM (deferred): UIBackend.start_investigation() contract incoherent; backend abstraction lacks live-run/streaming contract
- Tier 2 r2 flush_revisions HIGH (commit 64ef34f): seed batching prevents loss at >2000 seeds

**Adversarial #025 FAIL → ALL 4 HIGHs FIXED (2026-04-06, commit e033914):**
- HIGH #1 (SO-3 stop): `threading.Event _stop_event` closes early-stop race; set in `stop_investigation()`, checked in `run_investigation_thread()` before `irys.investigate()` starts
- HIGH #2 (SO-3 steering actionable): `load_gaps()` returns `(text, top_redirect_issue_id)` tuple; `refresh_gaps_btn` auto-populates redirect form from top steering recommendation
- HIGH #3 (SO-2 correction refresh): `_correct_and_refresh()` wrapper refreshes `assertions_md` + `overview_md` after successful correction — belief state changes visible immediately
- HIGH #4 (SO-5 multi-source): `_fmt_assertions()` now uses `source_roles` list → `MULTI-SOURCE[...]` when assertion spans multiple source types
- MEDIUM #6 (errors surfaced): steering surface errors now shown as `⚠️` instead of silent empty string
- Tier 1 r11 PASS (725 tests, no regressions)

**Adversarial #026 PASS (2026-04-06, commit 5d13c82):** All #025 fixes verified solid.
- CONFIRMED: stop_event lifecycle clean; correction→refresh DB chain correct; load_gaps tuple safe; SO-3/SO-2 end-to-end
- 2 pre-existing LOWs: _get_matter_model() linear scan; sorted() vs heapq.nsmallest in overview

**Tier 1 r11 PASS (725 tests, 2026-04-06).**

**Tier 2 r2 HIGH fix (2026-04-06, commit 64ef34f):**
- `flush_revisions()` now batches seeds in groups of `MAX_WORK // 2` (250). Previously,
  large flushes (>2000 seeds) could silently skip high-index seeds because the BFS
  frontier budget was exhausted — those seeds were then cleared and permanently lost.

**Tier 2 implementations (2026-04-06, from Tier 2 Scaling+Architecture review):**
1. SO-4 semantic attribution gate: `_build_issue_profiles()` + `_best_semantic_issue()` Jaccard gate;
   unannotated initial_searches + SPO graph leads now validated against issue content before round-robin
2. SO-2 frontier-aware BFS budget: `_effective_max_work = min(2000, seeds*3)` when `seeds*2 > MAX_WORK`
3. SO-2 assertion_revision read path: `detect_oscillation()` detects A→B→A cycles; emits SYSTEM_WARNING
4. SO-1 real reuse telemetry: schema v36 adds `llm_calls_avoided`/`llm_calls_required` to run_session;
   `true_reuse_rate = avoided / (avoided + required)` replaces assertion-count proxy in run summary

---

## Active Work

### JUST COMPLETED — adversarial #024 product surface fixes + Tier 1 r6 (2026-04-06)

**Adversarial #024 fixes (commit cebc76e):**
- Assertion IDs + issue IDs added to Assertions/Issues tables (SO-2/4)
- `get_steering_surface()` + `get_quant_summary()` added to UIBackend base, InProcessBackend, HttpBackend
- `load_gaps()` wired to steering surface; new `load_quant()` + Tab 6 Quant panel (SO-3/6)
- `_fmt_steering()` + `_fmt_quant()` formatters; `primary_source_role`/`primary_speech_act` fallback (SO-5)

**Tier 1 r6 fixes (commit 17fef25):**
- HIGH: `_run_thread` type guard — fails cleanly if backend lacks `_get_irys()`
- MEDIUM: `InProcessBackend.redirect_run()` now validates run existence + status + issue existence
- MEDIUM perf: schema v37 `ix_clarification_matter_answered` index eliminates full-table sort
- MEDIUM perf: `engine._orient()` accepts `_stats` to reuse pre-computed stats from `investigate()`

Tier 1 r7 review running (background). HEAD: 17fef25 — Tests: 725/725

### PREVIOUSLY COMPLETED — Tier 1+2 MEDIUM fixes (2026-04-06, commit 1cddd8e)

Two MEDIUM bugs found by Tier 1 review of Tier 2 implementations — both fixed:

**MEDIUM 1 — Semantic gate _profile_pool NameError (engine.py):**
- `_profile_pool` was referenced in initial_searches loop and SPO fallback but never defined
- Fix: `_profile_pool = list(_issue_profiles.keys())` added after `_issue_profiles` build
- SPO lead fallback changed from `_biased_pool` to `_profile_pool`
- Stale issue IDs that failed profile lookup are now excluded from round-robin fallback

**MEDIUM 2 — complete_run() not idempotent (reasoning.py):**
- SQL always SET `llm_calls_avoided=?` / `llm_calls_required=?` — replay with None overwrote stored values
- Fix: SQL uses `COALESCE(?, llm_calls_avoided)` / `COALESCE(?, llm_calls_required)`
- First call stores real values; recovery/replay calls with None preserve them

HEAD: 1cddd8e — Tests: 725/725

### PREVIOUSLY COMPLETED — Tier 1 CLEAN r6: post-audit #023 Tier 1 MEDIUM/LOW (2026-04-06)

**r5 fixes (commit f2dfaea):**
1. **schema.py**: SCHEMA_VERSION restored to 35 with no-op _migration_v35 — monotonic lineage preserved; any DB at v35 stays valid.
2. **belief_revision.py**: pre-tx fast-path added — skips BEGIN IMMEDIATE when pre-tx snapshot shows no state change needed (safe: no-op has no side effects to OCC-protect).
3. **belief_revision.py**: OCC abort now re-enqueues X itself (not dependents), with _OCC_MAX_RETRIES=3 cap — prevents BFS fan-out explosion under concurrent contention. occ_exhausted_count tracked.
4. **belief_revision.py**: seed deduplication before building pending deque.
5. **engine.py**: _bare_idx counter (not _idx) for unannotated initial_searches — annotated entries no longer shift round-robin rotation. weakest_id always included in _biased_pool even when not in _orient_issue_ids.

**r6 fix (commit 5517351):**
6. **belief_revision.py**: seeds_remaining.discard() moved after non-aborted pass — seedness preserved across OCC retries so seed fan-out still fires on successful retry. occ_exhausted_count contributes to truncated=True with specific warning.

**r6 LOW fix (commit 856e2f7):**
7. **belief_revision.py**: truncation warning message enumerates specific causes (MAX_WORK and/or OCC exhaustion).

HEAD: 856e2f7 — Tests: 725/725

### PREVIOUSLY COMPLETED — Adversarial Audit #023 HIGH fixes (2026-04-05)

**Audit #023 results:** SO-1/3/5/6/7 PASS; SO-2/4 PARTIAL. THREE HIGH findings, all fixed:

1. **SO-2 HIGH: OCC silent propagation loss** — `_revise_one()` returned `None` on OCC conflict; BFS pruned downstream subtree silently. Fix: signature → `(result, occ_aborted: bool)`; BFS re-enqueues dependents when `occ_aborted=True`. (commit 6586399)

2. **SO-2 HIGH: Flat fact ingress** — SPO retry threshold `>= 3` left single/two-fact batches without SPO. Fix: lower to `>= 1` in both search and deep-read paths. (commit 26125f9)

3. **SO-4 HIGH: Attribution contamination** — partial fix: coverage-biased round-robin replaces all-to-weakest_id. Full fix: bare_idx counter (r5) + weakest_id always in pool. (commit cb26e78, then f2dfaea)

HEAD (before r5): 26125f9 — Tests: 725/725

### PREVIOUSLY COMPLETED — Tier 1 CLEAN r4: Q4/SO-2 stale pre-state (2026-04-05)

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
- SO-1/3/5/6/7: **PASS**; SO-2/4: **PARTIAL** (improving post-#023)
- Tier 1 Q4/SO-2 stale pre-state: **CLEAN** (4 rounds, r4 confirmed 2026-04-05)
- Adversarial audit #023: DONE — same PASSes (SO-1/3/5/6/7); 3 HIGH findings; all 3 fixed (see below)
- Next: Tier 1 CLEAN on audit #023 fixes; SO-4 attribution remaining (coverage contamination)

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
