# AR Repo — Cowork Context

> **How to use this file**
> This is the briefing document for every Cowork session on this repo. Read it at the start of each session before doing anything else. It encodes current branch state, active workstreams, open experiments, decision gates, and deferred items. Update it after standups, after experiments produce results, and after merges.
>
> **What Cowork can do autonomously:** Draft docs, update this file, generate experiment specs, update the weight/boost tables, produce PR descriptions from experiment results pasted in.
> **What always needs approval:** Any Slack or external message. Any merge decision. Any code change.

---

## 1. Repo and system overview

**System:** Irys — AI-powered legal fact-gathering and synthesis platform. Multi-phase pipeline: orientation → lead generation → proof state → gap detection → synthesis.

**Core problem being solved:** The echo chamber problem. The agent currently treats a well-written advocacy brief as equivalent to a court order. The classifier + search ranking changes fix this by assigning epistemic authority to every document and using it to surface authoritative sources first.

**Three-tier model stack:** LITE (`gemini-2.5-flash-lite`) for high-volume extraction and search analysis. FLASH (`gemini-3-flash-preview`) for planning and routing. PRO (`gemini-3.1-pro-preview`) for final synthesis. Fallback chain: Gemini → Vertex AI → fallback model on 503/timeout.

**Key files:**
- `src/irys/core/fact_store.py` — SQLite + FTS5 FactStore, EvidencePacker, BM25/RRF retrieval, importance lifecycle
- `src/irys/rlm/engine.py` — RLMEngine, investigation loop, RLMConfig, async SQLite wiring
- `src/irys/rlm/decisions.py` — all LLM decision functions by tier
- `src/irys/rlm/prompts.py` — all prompt templates
- `src/irys/service/inline_citation_service.py` — post-synthesis citation injection
- `src/irys/core/search.py` — DocumentSearch, smart_search(), density scorer
- `v2-dataset/waymo_dataset/classifier_experiments.py` — epistemic classifier (standalone, not integrated)
- `scripts/eval_smart_search_mrr.py` — end-to-end MRR eval harness

---

## 2. Branch state

| Branch | Status | Notes |
|---|---|---|
| `feat/context-store-v2` | **Current HEAD. Local only, not pushed.** | SQLite FactStore replacement + async fix + citation injection fix. See Section 3. |
| `feat/rlm-improvements` | Remote. Production target. | Current production branch. Density scoring not applied. |
| `feat/search+` | PR open. Ready to merge. | Density scoring on OR-fallback only (12 lines). Prerequisite for density-upgrade branch. |
| `search/density-ranking-upgrade` | Local only, not pushed. MRR 0.164 | 3-commit density scoring chain + MRR harness. On top of `feat/search+`. |
| `feat/inline` | Local. | Inline citation work. |
| `feat/bias-reduction` | Local. | Classifier/bias workstream. |
| `multi-modal` | Local. | Multimodal (image/OCR) detection. |

---

## 3. feat/context-store-v2 — what this branch did

### 3.1 Core feature: SQLite FactStore (replacing JSONL)

`src/irys/core/fact_store.py` was fully rewritten. Key changes:

**Schema:** SQLite + FTS5 (Porter stemmer). Tables: `facts`, `fact_fts`, `source_synopses`, `synopsis_fts`, `fact_stubs`. WAL mode on. Content-hash deduplication via SHA-256(`fact\x00source`).

**Retrieval:** Three-lane RRF fusion — per-fact BM25 (weight 0.6), synopsis-expanded BM25 (weight 0.3), importance sweep (weight 0.1). Compound rescoring after fusion: BM25 relevance (60%), importance (25%), recency (15%).

**Importance lifecycle:** Four tiers (`draft` → `hot` → `warm` → `cold` → archived). Hooks: `add_facts_from_extraction` (+50 base, +5 on re-extraction), `on_search_hit` (+3), `on_re_extraction` (+5), `tick_decay` (×0.995^days_idle). `archive_cold_facts` moves importance < 35 to `fact_stubs`.

**EvidencePacker:** 30% per-source token cap prevents single-source domination in synthesis context. Density scoring: `score / log(token_estimate)`.

**S3 round-trip:** `load()`/`save()` via `facts.ndjson` for cross-session persistence. Backward compatible with old JSONL via `migrate_from_jsonl()`.

**`stats()` + CLI migration:** `FactStoreStats` dataclass. `migrate_from_jsonl` CLI script for upgrading existing `.irys/facts.jsonl` stores.

### 3.2 Prompt change: FACT COMPLETENESS instruction

`P_EXTRACT_FACTS` now instructs the LITE model to include 1–2 sentences of surrounding context around each core claim (max 3 sentences total). Targets legal documents where qualifications and carve-outs appear in adjacent sentences.

### 3.3 Critical fix: async event-loop blocking (this session)

All hot-path SQLite calls were being made synchronously from async context. Fixed by wrapping in `asyncio.to_thread`:

| Call site | Method | Location |
|---|---|---|
| `_read_document` | `add_facts_from_extraction` | `engine.py:2648` |
| `_direct_answer` | `pack_evidence` | `engine.py:803` |
| `_assess_and_create_plan` | `pack_evidence` | `engine.py:1204` |
| `_build_evidence_context` | `pack_evidence` | `engine.py:2210` |
| `_synthesize` (post-synthesis bump) | `on_search_hit` loop | `engine.py:2864` |

The `on_search_hit` loop was also replaced with `on_search_hit_batch(hashes)` — a single `executemany` + one `COMMIT` instead of one `UPDATE + COMMIT` per hash. `on_search_hit_batch` was added to `fact_store.py`.

Load/save/tick_decay/archive paths were already correctly wrapped in prior commits.

### 3.4 Citation injection fix (this session)

`inline_citation_service.py`: `all_ids` (the valid-ID set passed to `_validate_response`) was built from all `citations`, but `_sanitize_citations` silently drops citations with no usable content. Those IDs were never sent to the LLM but were in `all_ids`, meaning the validator used a superset of what the LLM saw. Fixed: `all_ids` is now built from `sanitized_for_ids` — only the IDs actually sent to the LLM.

**Before fix:** `fed=10 matched=0 unmatched=10` — injection failed every run, returned original answer.
**After fix:** `fed=6 matched=6 unmatched=0` — 100% match rate.

### 3.5 Integration test: test_dele004_context_store_v2.py

New standalone integration test using the full DELE-004 3-document Delek S&O corpus. Checks: SQLite write+read (via `pack_evidence` log), event-loop health (no blocking errors), citation injection success (`[[cite:N]]` markers present, no raw UUIDs), cross-document coverage (ARKS/BSR/Lion Oil all cited), answer quality (key phrases present). **Last run: 14/14 passed, 122s.**

### 3.6 Reverted work

Three commits adding `_derive_doc_label` / `ATTRIBUTION RULE` for document-anchored fact strings were reverted after test failures. The underlying problem (facts extracted without document anchor) is partially addressed by the FACT COMPLETENESS instruction. Full attribution anchoring is deferred — see Section 7.

---

## 4. Epistemic classifier — current state

**What it is:** Metadata-only classifier. Zero LLM calls. Runs once per document. Assigns `epistemic_category` + `authority_weight` to every document.

**Validation status:** 99.6% corpus coverage on 7,052-doc Waymo corpus. SME-reviewed May 1, 2026 (Christian Brown). See Section 5 for the authoritative weight table.

**Integration status: NOT integrated.** Classifier lives at `v2-dataset/waymo_dataset/classifier_experiments.py`. It is not called anywhere in the main pipeline. `StoredFact` does not have `epistemic_category` or `authority_weight` fields. Search scoring does not apply stance boost. Integration follows the merge sequence in Section 6.

**Two integration points (pending):**
1. **Search pre-ranking** — stance boost as a scoring signal in `search.py`. Formula: `score = 0.35 × term_coverage + 0.25 × match_density + 0.40 × epistemic_stance_boost`
2. **Fact metadata enrichment** — `epistemic_category`, `authority_weight` on `StoredFact` (currently absent)

---

## 5. Authoritative weight + stance boost table (post-SME review, 2026-05-01)

> Source of truth. Update here first whenever weights change.

| Weight | Category | Stance boost | Status | Notes |
|---|---|---|---|---|
| 10.0 | `authority_court_substantive` | 1.50× | Existing | SJ, PI, dispositive motions, judgments, contempt findings |
| 9.0 | `case_law_external` | 1.45× | **New** | Binding/persuasive precedent via CourtListener/Tavily. Must attach at ingestion in `external_search.py` — cannot be assigned from filename |
| 9.0 | `prior_judgment_or_award` | 1.45× | **New** | Prior judgments for collateral estoppel; arbitration awards; appellate mandates |
| 8.5 | `authority_court_procedural_substantive` | 1.35× | **Split from existing** | Compel, sanctions (FRCP 11/26/37), protective orders, privilege rulings, MIL rulings, Daubert orders |
| 8.0 | `statute_or_regulation` | 1.25× | **New** | USC, CFR, model rules |
| 8.0 | `administrative_record` | 1.25× | **New** | Certified administrative record in agency-action matters |
| 7.5 | `expert_report_court_appointed` | 1.22× | **New** | Rule 706 court-appointed experts; special masters; technical advisors |
| 7.0 | `settlement_agreement` | 1.15× | **New** | Executed settlement agreements; consent decrees |
| 6.0 | `expert_report_independent` | 1.20× | Existing | Rule 26(a)(2) disclosed experts. Subject to Daubert. |
| 5.0 | `evidence` | 1.10× | Existing | Exhibits, deposition transcripts, contracts in record |
| 5.0 | `stipulation_joint` | 1.10× | **New** | Joint stipulations; agreed orders; joint CMC statements |
| 5.0 | `authority_court_administrative` | 1.05× | **Split from existing** | Scheduling orders, calendar orders, conference notices |
| 4.5 | `discovery_response_sworn` | 1.10× | **New** | Interrogatory responses (sworn); RFA responses; deposition designations |
| 4.0 | `declaration_witness` | 1.00× | Existing | Third-party fact witnesses under penalty of perjury |
| 4.0 | `complaint` | 1.00× | Existing | Pleadings |
| 2.5 | `declaration_party` | 0.90× | Existing | Sworn but self-serving |
| 2.0 | `advocacy_plaintiff` | 0.90× | Existing | Known-party advocacy floor |
| 2.0 | `advocacy_defendant` | 0.90× | Existing | Same |
| 2.0 | `settlement_communication` | 0.90× | **New** | Demand letters, mediation positions, FRE 408-protected comms |
| 2.0 | `UNCLASSIFIED` | 0.85× | **Lowered from 3.0** | Default should be conservative |
| 1.5 | `advocacy_unknown` | 0.80× | **Lowered from 2.0** | Unknown provenance weaker than known attribution |
| 0.0 | `EXCLUDE` | 0.00× | Existing | Service certificates, summonses, docket sheets |

**Dominance gap short-circuit: DEFERRED.** Precision at current threshold: 33%. Do not ship.

---

## 6. Open experiments — closing the search + classifier branch

### Experiment 1 — Classify the 33-doc Waymo eval set (PREREQUISITE)
**What:** Run `classifier_experiments.py` with the post-SME weight table (Section 5) against all 33 PDFs in `v2-dataset/`.
**Gate:** Manual verification of the procedural split. Every scheduling order → `authority_court_administrative`. Every compel/sanctions/protective/MIL order → `authority_court_procedural_substantive`.
**Output:** Label + weight for each of the 33 docs. Stored alongside MRR run outputs in `v2-dataset/`.

### Experiment 2 — MRR with stance boost vs density-only baseline (MERGE GATE)
**What:** Wire stance boost into `search/density-ranking-upgrade`. Run `scripts/eval_smart_search_mrr.py`.
**Baseline:** 0.164 (density-only). 11/20 hard negative separation.
**Pass criterion:** Stance boost MRR > 0.164 AND ≥ 11/20 hard negative separation.
**Fail path:** If no signal over density alone, diagnose before proceeding.

### Parallel — `case_law_external` ingestion path
Confirm where `epistemic_category` and `authority_weight` attach to `LegalCase` / `WebSearchResult` in `external_search.py` at ingestion time. Not blocking Exp 2, blocking full merge.

### Parallel — `_CHARS_PER_PAGE = 3000` calibration
Sample DOCX files from CITIOM corpus. Compute actual chars per logical page. Currently unvalidated. Not blocking Exp 2.

---

## 7. Merge sequence

```
Step 1  Merge feat/context-store-v2 into feat/rlm-improvements   ← READY (this branch)
        14/14 integration tests pass. Async fix + citation injection fix confirmed.
        Remove .venv/ from tracked files before merging (git rm -r --cached .venv/).
Step 2  Merge feat/search+ into feat/rlm-improvements            ← ready, no dependencies
Step 3  Fix timeout=0 on assess_small_repo                       ← independent bug
Step 4  Run Experiment 1 (classify eval set, verify procedural split)
Step 5  Run Experiment 2 (MRR gate)
        ├─ Pass → continue
        └─ Fail → diagnose, do not proceed
Step 6  Resolve case_law_external ingestion path
Step 7  _CHARS_PER_PAGE decision (empirical or log-and-defer)
Step 8  Push search/density-ranking-upgrade, open PR against feat/rlm-improvements
Step 9  Merge
Step 10 Classifier integration (search pre-ranking first, fact metadata second)
        ← StoredFact needs epistemic_category + authority_weight fields (currently absent)
        ← Requires Langfuse live before synthesis-input change ships
```

**Deferred from this branch (pick up in a follow-on):**
- `on_re_extraction` is defined but never called. Wire in `_read_document` when a scope is seen in `cache.extracted_scopes`, or document the intentional omission.
- `_TIER_PROMOTE`/`_TIER_DEMOTE` class dicts defined but `_check_tier` uses hard-coded literals. Either delete or use them.
- `tick_decay` is an O(n) Python loop. Replace with a single SQL `UPDATE ... SET importance = importance * (0.995 ** ...)`.
- Synopsis built from only 3 facts (80 chars each). Use first 10 facts at 120 chars each for better BM25 q2 recall.
- `EvidencePacker` leaves budget unused when a source is sparse. Consider a second top-up pass.
- Attribution anchoring (`_derive_doc_label` / `ATTRIBUTION RULE`) was reverted after test failures. The FACT COMPLETENESS instruction is a partial substitute. Full context-anchored extraction design is at `docs/superpowers/specs/2026-05-18-context-anchored-fact-extraction-design.md`.

---

## 8. Active workstreams (as of 2026-05-22)

| Workstream | Owner | Status | Blocking on |
|---|---|---|---|
| context-store-v2 (SQLite FactStore + async fix + citation fix) | Sudarsh | **Complete. Ready to merge.** | .venv cleanup before merge |
| Bias correction (classifier + search integration) | Sudarsh | In progress — Experiments 1+2 pending | Experiments 1+2 |
| search/density-ranking-upgrade | Sudarsh | Local only. Waiting on search+ merge. | feat/search+ merge |
| Langfuse integration / observability | Arpit | Integrated (see feat/harness-fixes-observe, merged) | — |
| Attribution anchoring (context-anchored fact extraction) | Deferred | Reverted from this branch | New design pass needed |
| Index/batching for background maintenance | Not started | Design needed | Classifier must ship first |

---

## 9. Deferred items

| Item | Reason deferred | Trigger to revisit |
|---|---|---|
| `on_re_extraction` never called | Not wired in engine.py | When importance lifecycle tuning begins |
| `tick_decay` O(n) Python loop | Low priority while store is small | When matters exceed ~5,000 cached facts |
| Synopsis built from 3 facts | Degrades BM25 q2 recall on large documents | Next FactStore tuning pass |
| EvidencePacker unused-budget gap | Complex fix, minor impact at current scale | When source count per matter grows |
| Attribution anchoring (reverted) | Test failures — design needs rework | `docs/superpowers/specs/2026-05-18-context-anchored-fact-extraction-design.md` |
| Dominance gap short-circuit | 33% precision — silently skips FLASH analysis on wrong queries | When precision improves on new corpus |
| Flat hit caps (`max_hits_per_file`) | Penalises short docs (court orders, 1-page notices) | When length-aware or type-aware cap design exists |
| OR-fallback stopword filtering | Full NL queries pull 3-5K hits; MRR ceiling ~0.16 vs ~0.40 formula-isolated | Separate query-handling workstream |
| `_CHARS_PER_PAGE = 3000` calibration | Needs CITIOM DOCX empirical sampling | When CITIOM accessible |
| Classifier integration into main pipeline | StoredFact schema change + Langfuse prerequisite | After Step 9 merge |
| Citation provenance verification | Not yet built | DELE-013: fabricated Reuters URLs cited on a correct answer — wrong source on a legal product is worse than a wrong answer. Every cited source must exist in the retrieved fact set. |
| Synthesis prompt updates | Downstream of classifier integration | After classifier + Langfuse both live |
| Hybrid Rule 26(a)(2)(C) witnesses | Detectable only by parsing disclosure docs | Future classifier refinement pass |

---

## 10. Observability architecture

Two non-overlapping capture layers (Layer 2 integrated on `feat/harness-fixes-observe`, merged).

**Layer 1 — Annotation stream + enrichment signals → internal telemetry tables**
SSE annotation stream events: `rlm-lead-started`, `rlm-lead-update`, `rlm-checkpoint`, `rlm-synthesis-complete`. Enrichment signals to add (Track B, owner: Sudarsh):
- I7: Search analysis summary — hits_total, hits_selected, hit_selection_ratio, decisive_docs_pinned, child_reads_spawned.
- D10 extension: facts_delta — net new facts per document read, not cumulative.
- D12 extension: checkpoint enrichment — should_replan, citations_count, new_leads_added.
- I6: Cache skip events — emit when a doc or search is skipped due to cache hit.

**Layer 2 — LLM call inputs/outputs → Langfuse** ✓ Integrated.
`GeminiClient.complete()` wrapped. Every generation tagged with `investigation_id` via Python contextvars. Call types tagged: `extract_facts`, `analyze_search`, `checkpoint`, `synthesize`, `citation_injection`.

**Metadata constraint:** Langfuse metadata capped at 200 chars per value. Rule: counts, booleans, tier names, short IDs → Langfuse metadata. Full text, reasoning strings, fact lists → internal telemetry tables only.

**Payload ceiling:** Langfuse Cloud 5MB per request / 4.5MB ingestion hard limit. Complex multi-document traces may approach this. Measure actual trace size in production.

---

## 11. How to update this file

**After a standup:** Paste the standup summary and ask Cowork to update Section 8 (workstreams) and append new deferred items to Section 9.

**After an experiment completes:** Paste results and ask Cowork to update Section 6 (mark complete, update MRR numbers) and Section 7 (advance merge sequence pointer).

**After a merge:** Ask Cowork to update Section 2 (branch state) and Section 7.

**After SME input:** Paste the relevant memo section and ask Cowork to update Section 5 (weight table). All weight changes go here first before touching any code.

**Never update the merge sequence (Section 7) without an experiment result or explicit decision.**

---

*Last updated: 2026-05-22. Source: feat/context-store-v2 branch completion, integration test results, code review findings.*
