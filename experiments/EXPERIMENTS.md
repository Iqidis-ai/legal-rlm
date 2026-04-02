# Experiments

Reverse chronological. All entries must have a corresponding entry in `ledger.jsonl`.
Only Codex-validated conclusions are recorded as findings.

---

## EXP-008 — Service API Matter Model Endpoints + SO-6 Reconciliation (2026-04-02)

**Status:** COMPLETE
**Git commits:** ddbc483 → 78e906a (3 commits)
**Purpose:** Expose matter model intelligence via REST API (SO-3 user steerability); implement SO-6 numeric conflict detection and payment reconciliation query layer.

**Changes shipped:**
1. **Matter model REST endpoints** (SO-3) — 8 new endpoints:
   - `GET /matter/{id}` — stats (assertions, gaps, issues, actors, quant, clarifications)
   - `GET /matter/{id}/runs` — recent investigation run history
   - `GET /matter/{id}/runs/{run_id}/events` — full reasoning ledger event sequence
   - `GET /matter/{id}/clarifications` — pending clarification questions
   - `POST /matter/{id}/stop` — signal engine stop via `request_stop()` (sets DB flag read by engine on next iteration)
   - `POST /matter/{id}/runs/{run_id}/redirect` — redirect active investigation to target issue
   - `POST /matter/{id}/clarifications/{qid}/answer` — answer clarification question
   - `GET /matter/{id}/reconcile` — payment reconciliation summary + conflict list
2. **Matter model registry** — `_active_matter_models: dict[str, MatterModel]` in service/api.py; `_wire_matter_model()` helper pre-creates and registers the model before `irys.investigate()` is called so stop/redirect signals reach the live engine
3. **ServiceConfig.enable_matter_model** — env var `IRYS_ENABLE_MATTER_MODEL`; all three background investigation tasks (`_run_investigation`, `_run_upload_investigation`, `_run_urls_investigation`) now wire matter model when enabled
4. **QuantStore.get_conflicts()** — finds amount facts with same subject_type+currency but different values; returns conflict groups with value lists (SO-6 numeric conflict detection)
5. **QuantStore.reconcile_by_subject()** — groups amount totals by subject_type+currency; invoice vs payment reconciliation
6. **MatterModel.detect_quant_conflicts()** — calls `get_conflicts()`, records each conflict as UNRESOLVED_CONTRADICTION gap (materiality 0.8); idempotent
7. **MatterModel.reconcile()** — thin facade over `reconcile_by_subject()`
8. **DEEP_READ_PROMPT: numeric_facts subject field** — added `subject` field ("invoice" | "payment" | "fee" | "damages" | "balance" | "rate" | "deposit" | "penalty" | "other") to numeric_facts LLM schema
9. **engine.py**: passes `subject_type` from `nf.get("subject")` on quant extraction; calls `detect_quant_conflicts()` at run completion alongside `generate_clarifications_from_gaps()`
10. **MatterRuntimeAdapter.record_quant()**: added `subject_type` parameter
11. **8 new tests** — reconcile_by_subject, get_conflicts, detect_quant_conflicts idempotency, adapter subject_type wiring

**What we learned:**
- The stop endpoint design requires `request_stop()` (sets DB flag), NOT `interrupt_run()` (which is the terminal state the engine sets after actually stopping)
- Pre-creating the matter model before `irys.investigate()` is the only way to register it in the service registry early enough for stop/redirect to work during active runs
- `subject_type=None` in existing quant records means reconciliation silently loses data — prompting the LLM for a subject field is essential
- Conflict detection is cheapest at run-end (not per-document) since we need all amounts extracted first

**Test count:** 134 passing (up from 126)

**What remains (Priority 1):**
- Adversarial Audit #2 now overdue — all Priority-0 SOs present and Priority-1 work started
- Tier 1 Codex review pending (review of EXP-007 + EXP-008 changes)
- SO-6 quant reconciliation surface in synthesis output (currently computed but not shown in investigation output)
- SO-2 belief revision propagation end-to-end test with assertion link graph

---

## EXP-007 — All Priority-0 SOs Complete + SO-6 Initial + Service Wiring (2026-04-02)

**Status:** COMPLETE
**Git commits:** 81573f6 → 8bac432 (12 commits)
**Purpose:** Close all remaining Priority-0 gaps and begin Priority-1 work (SO-6 quantitative intelligence).

**Changes shipped:**
1. **SO-3 redirect bug fix** — redirect title lookup searched `state.findings["issues"]` (string list from LLM) for dict with `.id == UUID`; always failed silently; now resolves title via `IssueStore.get_issue(id)` directly
2. **Actor store wired** (SO-5) — entity extraction in `_deep_read_document` now persists people/companies to `ActorStore.upsert_actor()`; `adapter.record_actor()` added
3. **Actor + doc context in orientation** (SO-1/SO-5) — `build_query_context()` now populates `known_actors` (top 10) and `known_document_ids` (up to 20 from assertion_occurrence); both shown in orientation prompt
4. **Matter model in public API** (SO-1) — `IrysConfig.enable_matter_model` flag; `Irys` class opens `MatterModel.open(repository)` and injects into engine per-repository (keyed by resolved path)
5. **LLM-driven gap detection** (SO-7) — `_deep_read_document` now checks `connections` from analysis against `repo.list_files()`; referenced docs not found in repo → `MISSING_DOCUMENT` gap + search lead
6. **Clarification engine** (SO-7/SO-3) — `ClarificationStore` with add/answer/get_pending/get_answered; `MatterModel.generate_clarifications_from_gaps()` auto-generates questions for high-materiality gaps at run end; answered clarifications injected into orientation context; schema v4 migration
7. **QuantStore** (SO-6) — `QuantStore.record()` persists structured numeric facts; `DEEP_READ_PROMPT` updated with `numeric_facts` array field; engine extracts and persists per-document; `adapter.record_quant()` added
8. **stats() expanded** — includes `quant_fact_count` and `pending_clarifications`
9. **IssueStore.get_issue(id)** — single issue lookup by UUID (was missing; needed for redirect fix)

**What we learned:**
- `state.findings["issues"]` is a list of LLM-generated strings, never dicts — the redirect UUID lookup was dead code from the start
- Actor store was fully built but never called; single-line wiring in `_deep_read_document` activated it
- The clarification engine only requires ~100 lines: a simple Q&A store + materiality threshold + gap iteration
- SO-6 quant extraction is now live but quality depends entirely on LLM; will need calibration on real documents

**What remains (Priority 1):**
- Service API matter model endpoints (stop/redirect/clarify via REST) — SO-3 not yet accessible from API layer
- Quant reconciliation queries (invoice vs payment matching) — data structure exists but no query layer
- SO-6 numeric conflict detection
- Adversarial Audit #2 now warranted — Priority 0 items are all present

---

## EXP-006 — Tier 1 Review Fixes + SO-3/SO-4/SO-5/SO-7 Depth (2026-04-02)

**Status:** COMPLETE
**Git commits:** 9cb8d72 → 33ed9d2 (14 commits)
**Purpose:** Fix bugs found by Tier 1 correctness review (bz547kj24) and systematically
close all remaining partial SO gaps from Adversarial Audit #1.

**Changes shipped:**
1. **Checkpoint serialization bug** — `Lead.search_term` and `Lead.focus_issue_id` were missing
   from `to_dict()`/`from_dict()` in state.py; silently dropped on checkpoint save/resume
2. **Reasoning ledger depth** (SO-3) — `adapter.log_step()` was only called in one orientation
   branch; now logged at: orientation complete, each iteration, key facts recorded, synthesis phase
3. **Issue-to-assertion linking** (SO-4) — `record_fact()` accepts `issue_id`; threaded through
   both `_analyze_search_results` AND `_batch_deep_read`/`_deep_read_document`
4. **Source-role calibration in synthesis** (SO-5) — `_build_source_calibration()` queries
   assertion counts by source_role; synthesis prompt now has a CRITICAL calibration block warning
   LLM not to amplify advocacy material as established facts
5. **Gap store populated by engine** (SO-7) — `adapter.record_gap()` added to adapter (dual-writes
   gap table + ledger event); engine records `MISSING_DOCUMENT` gaps when: (a) search returns no
   hits for a focus-issue lead, (b) deep read fails with an exception
6. **NullMatterAdapter.record_gap()** — missing method would have caused AttributeError on
   non-matter-model runs
7. **resume_investigation() adapter wiring** — `state._matter_adapter` was never set on resumed
   state; fixed: now starts a new run_session, wires adapter, handles complete/fail lifecycle
8. **Gap descriptions in orientation** — `_format_matter_context()` now shows up to 3 gap
   descriptions (not just count) so LLM knows specifically what's missing

**What we learned:**
- Checkpoint serialization is a silent data-loss category: fields added to a dataclass must
  also be added to to_dict/from_dict or they vanish on resume without any error
- `resume_investigation()` was missing adapter wiring since day 1 — resumed runs had zero
  matter model writes and no stop propagation, silently
- Source calibration required just one DB query (GROUP BY source_role); the structured assertion
  data was already there, just not surfaced to the synthesis LLM
- NullMatterAdapter interface completeness is a recurring issue: every new adapter method needs
  a no-op counterpart

**What remains:**
- Redirect (SO-3) — pure scaffolding, `redirect_requested` field exists but nothing triggers it
- Numeric extraction (SO-6) — not started
- Gap store: only populated on search misses and read failures; no LLM-driven gap detection

---

## EXP-005 — Engine-Substrate Wiring Sprint (2026-04-02)

**Status:** COMPLETE
**Git commits:** 414c3e2 → 6977d4c (7 commits)
**Purpose:** Close the engine-substrate disconnect flagged by Adversarial Audit #1 (all 5 SOs rated PARTIAL due to engine never reading the matter model stores during live runs).

**Changes shipped:**
1. **SO-5 bug fix** — `document_id = results.query` → `results.top(1)[0].filename` (source-role inference was running on search strings)
2. **Migration runner** — `apply_schema()` replaced with versioned `_MIGRATIONS` list; CREATE-IF-NOT-EXISTS replay removed
3. **Migration 2: 6 missing indexes** — gap/matter_status, issue/matter_status, issue/LOWER(title), run/matter_time, actor/matter_name, link/src_type
4. **weakest_issue_id stability** — `get_open_issues()` now has `id ASC` tiebreaker; `build_query_context()` `min()` key includes id
5. **depends_on direction fix** — removed from `get_dependents()` propagation (was propagating toward prerequisite, not dependent)
6. **SO-1/SO-4 wiring** — `_orient()` reads `adapter.get_context()`, formats `QueryMatterContext` into orientation prompt; weakest issue injected with explicit "PRIORITY FOCUS" instruction; `Lead.search_term` preserves raw terms from LLM output bypassing `_extract_search_term()` token collapse
7. **SO-3 interrupt lifecycle** — `_investigate_lead()` exits early without marking lead investigated; `interrupt_run()` added to ReasoningLedgerStore + MatterModel; post-loop stop check skips verify/synthesis; `state.interrupt()` sets proper status
8. **Migration 3: assertion identity** — unique index rebuilt as `(matter_id, model_layer, proposition_key)` so same-text propositions in different reasoning layers coexist; upsert lookup updated to include `model_layer`

**What we learned (Codex-validated):**
- SO-1/SO-4: Prompt-only injection is not sufficient without preserving `Lead.search_term`; `_extract_search_term()` collapses multi-word queries to a single token, losing all issue-focused search intent
- SO-3: Stop must skip synthesis entirely (not partial synthesis) per Codex design gate; run_session must transition to `interrupted` not `completed`
- Assertion identity: the (matter_id, proposition_key) unique key was the single highest-risk design choice — it collapsed all five reasoning layers
- Migration system was prerequisite for all schema changes; existing `apply_schema()` couldn't handle any non-additive changes

**What remains (superseded by EXP-006):**
- `adapter.log_step()` called in only one orientation branch — reasoning ledger is thin ✓ FIXED
- Issue-to-assertion linking not yet happening in live engine (SO-4) ✓ FIXED
- Redirect (SO-3) is pure scaffolding
- Source role still not influencing synthesis weighting (SO-5)
- Numeric extraction (SO-6) not started
- Gap store not populated by engine (SO-7)

---

## EXP-001 — Static Structural Baseline (2026-04-02)

**Status:** COMPLETE (static analysis — no API key available for live run)
**Git commit:** 61de459
**Purpose:** Establish pre-refactor structural baseline for the current pipeline.

**Key findings:**
- `engine.py`: 2069 lines, `state.py`: 1801 lines — large, tightly coupled
- Facts stored as **flat strings** in `findings["accumulated_facts"]` via `add_fact()` — no typing, no speech-act classification, no support/attack links. This is the exact target of SO-2.
- **Zero persistence** — `InvestigationState` is fully ephemeral. Everything lost after each run. Reuse rate: 0%.
- Issues stored as `list[str]` in `findings["issues"]` from `_orient()` — no structured issue tree.
- Actor/entity store: per-run `dict[str, Entity]`, lost after run. No alias resolution, no role assignment beyond filename heuristics.
- Source-role modeling: `EVIDENCE_SOURCE_WEIGHTS` dict maps filename keywords to weights. Complaint and order both score high (~0.85-0.95) — no distinction between advocacy and authoritative sources.
- Test files: 6 test files with 87 tests described in ROUNDTABLE_PROGRESS.md are **not committed**. Only `test_simple.py` exists.
- No belief revision, no assumption tracking, no gap store, no reasoning ledger.

**What we learned:** The refactor scope is exactly as the gap analysis predicted. The `_investigate_loop` in engine.py is the primary integration point for matter model writes. The `InvestigationState.add_fact()` → `findings["accumulated_facts"]` pattern is the specific code to replace with typed assertions.

---

## EXP-000 — Ledger Initialization (2026-04-02)

**Status:** INIT
**Purpose:** No experiments have been run yet. Prior roundtable cycles were implementation
work, not controlled experiments with measurable outcomes vs. baselines.
**What we learned:** N/A — baseline state.
**Next:** First real experiment will be a baseline measurement of the current pipeline
(EXP-001) to establish ground truth before the matter model refactor begins.

---

## Planned Experiments (not yet run)

### EXP-001: Current Pipeline Baseline
**Purpose:** Measure the current recursive pipeline on a standardized legal matter corpus
to establish baseline metrics before Phase 5 refactoring.
**Metrics to capture:** query latency, token cost per query, assertion count, citation
accuracy, issue coverage %, reuse rate (expected: 0% since no persistence).
**Why it matters:** Without a baseline, we cannot claim the matter model refactor is an
improvement. Every subsequent experiment compares against this.

### EXP-002: SQLite Matter Model Throughput
**Purpose:** Validate A-007 — does SQLite handle target scale (100k assertions, 10k documents)
within latency budget?
**Metrics to capture:** write throughput (assertions/sec), read latency for graph traversal,
total storage size per matter.
**Why it matters:** Architecture decision D-004 depends on this. If SQLite fails, we need
PostgreSQL before building anything else on top of it.

### EXP-003: Source-Role Classification Accuracy
**Purpose:** Validate A-006 — can document type + metadata → source role be inferred with
>85% accuracy?
**Setup:** Hand-label 100 documents from test corpus with ground-truth source roles.
Run inference. Measure precision/recall per role type.
**Why it matters:** Source-role modeling is a Priority 0 capability (SO-5). If inference
accuracy is low, the architecture must include a human-annotation pathway.
