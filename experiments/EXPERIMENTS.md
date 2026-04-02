# Experiments

Reverse chronological. All entries must have a corresponding entry in `ledger.jsonl`.
Only Codex-validated conclusions are recorded as findings.

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

**What remains:**
- `adapter.log_step()` called in only one orientation branch — reasoning ledger is thin
- Redirect (SO-3) is pure scaffolding
- Issue-to-assertion linking not yet happening in live engine (SO-4)
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
