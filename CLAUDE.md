# CLAUDE.md

## What this repo is

Irys is an AI-powered legal document analysis system. Given a legal question and a folder of documents, it runs a multi-phase investigation—searching, reading, and extracting facts—then synthesizes a polished answer with inline citations. "Epistemic attribution" is the design goal of assigning every document an authority weight based on its legal role (court order vs advocacy brief vs sworn declaration), so that more authoritative sources are ranked and weighted more heavily than self-serving documents. The classifier that assigns these weights is validated but not yet integrated into the main pipeline.

## Repo structure

```
src/irys/rlm/engine.py          — RLMEngine: main investigation loop, RLMConfig, small/large repo routing
src/irys/rlm/decisions.py       — All LLM decision functions, organized by LITE/FLASH/PRO tier
src/irys/rlm/prompts.py         — All prompt templates used by decisions layer
src/irys/rlm/state.py           — InvestigationState, Citation, Lead, ThinkingStep, Entity dataclasses
src/irys/rlm/research_agent.py  — ResearchAgent: tool-calling external search loop
src/irys/core/search.py         — DocumentSearch, smart_search(), density scoring
src/irys/core/fact_store.py     — FactStore, StoredFact, JSONL persistence (per-repository)
src/irys/core/models.py         — GeminiClient, ModelTier, MODEL_CONFIGS, system prompts
src/irys/core/reader.py         — PDF, DOCX, TXT, MHT document text extraction
src/irys/core/repository.py     — MatterRepository: file discovery, metadata, small-repo check
src/irys/core/external_search.py — ExternalSearchManager: CourtListener + Tavily wrappers
src/irys/api.py                 — Irys high-level Python API, IrysConfig, InvestigationResult
src/irys/service/api.py         — FastAPI REST service (S3/cloud path)
src/irys/service/inline_citation_service.py — Post-synthesis inline citation injection
src/irys/ui/chat_app.py         — Gradio multi-turn chat UI
v2-dataset/waymo_dataset/classifier_experiments.py — Epistemic classifier (standalone, not integrated)
scripts/eval_smart_search_mrr.py — End-to-end MRR eval harness (search/density-ranking-upgrade only)
```

## How to run

```bash
# Required in .env:
GEMINI_API_KEY=xxx

# Optional:
COURTLISTENER_API_TOKEN=xxx   # case law search
TAVILY_API_KEY=xxx            # web search
VERTEXAI_CREDENTIALS_B64=xxx  # Vertex AI fallback (base64-encoded service account JSON)
S3_BUCKET=xxx                 # S3-backed fact/checkpoint storage

# Gradio UI (port 7862, all interfaces)
python run_ui.py

# FastAPI REST server (port 8000)
python run_server.py
```

## Three-tier model stack

All tiers use `temperature=0` for deterministic output. Fallback chain runs Gemini → Vertex AI → fallback model on 503/timeout.

| Tier | Primary model | Fallback | Tasks |
|---|---|---|---|
| **LITE** | `gemini-2.5-flash-lite` | `gemini-3.1-flash-lite-preview` | Fact extraction, search hit selection, file picking, sufficiency checks, trigger extraction |
| **FLASH** | `gemini-3-flash-preview` | `gemini-2.5-flash` | Planning, unified assessment, routing decisions, case law/web result analysis |
| **PRO** | `gemini-3.1-pro-preview` | `gemini-2.5-pro` (then `gemini-2.5-flash`) | Final synthesis only — always uses PRO system prompt regardless of model |

## Investigation pipeline — phase order

1. **Routing** — `investigate()`: if `repo.is_small_repo`, go to `_direct_answer()`; otherwise go to full RLM path.
2. **Assess and plan** — `_assess_and_create_plan()` → `decisions.assess_and_plan()` (FLASH): one call combines complexity classification, cached-fact sufficiency check, and lead/search-term generation. For small repos, `_direct_answer()` calls `decisions.assess_small_repo()` (FLASH) instead.
3. **Investigation loop** — `_investigate_loop()`: iterative reads and searches; each lead calls `decisions.extract_facts()` (LITE); facts accumulate in `InvestigationState` and `FactStore`; external research triggers accumulate in `state.external_triggers`.
4. **External research** (large-repo path, post-loop) — `_run_external_research_post_loop()`: gated by `decisions.should_research_externally()` (LITE); if needed, runs `_run_research_agent()` → `ResearchAgent`.
5. **Synthesis** — `_synthesize()` → `decisions.synthesize()`: PRO model for complex queries, FLASH model with PRO system prompt for simple queries.

## Search scoring — current formula

`smart_search()` tries exact phrase match first. If no hits and query has multiple words, falls back to OR search across individual terms (> 2 chars), then deduplicates by `(file, page, line)` and accumulates `match_count`.

Density scoring runs on **both** branches (exact and OR-fallback):

```
hit.score = hit.match_count / divisor

where divisor =
  doc.page_count            (PDFs with page_count > 1)
  max(1.0, total_chars / 3000)  (all other formats: DOCX, TXT, MD, MHT, DOC)
```

`_CHARS_PER_PAGE = 3000` is an unvalidated constant. CONTEXT.md flags it as needing empirical calibration against the CITIOM DOCX corpus before production use on mixed-format matters.

**Stance boost is not yet wired in.** The scoring formula in CONTEXT.md (`0.35 × term_coverage + 0.25 × match_density + 0.40 × epistemic_stance_boost`) is the design target, but `SearchHit.score` currently carries only match density. Stance boost integration is blocked on Experiments 1 and 2 (see CONTEXT.md Section 5).

## Epistemic classifier — integration status

**What it classifies:** Assigns `epistemic_category` (e.g., `authority_court_substantive`, `advocacy_plaintiff`, `evidence`) and `authority_weight` (0.0–10.0) to documents using filename/metadata regex rules — zero LLM calls.

**Where it lives now:** Standalone experiment file at `v2-dataset/waymo_dataset/classifier_experiments.py`. It is not imported or called anywhere in the main pipeline (`engine.py`, `search.py`, `fact_store.py`, `external_search.py`).

**What is NOT yet integrated:**
- Classifier is not called during document ingestion or search
- `StoredFact` does not have `epistemic_category` or `authority_weight` fields
- `search.py` scoring does not apply stance boost
- `external_search.py` does not attach `epistemic_category = "case_law_external"` at ingestion

**Validation status:** 99.6% corpus coverage on 7,052-doc Waymo corpus. SME-reviewed May 1, 2026 (weight table in CONTEXT.md Section 4). Integration follows the merge sequence in CONTEXT.md Section 6.

## StoredFact schema

From `fact_store.py`:

```python
@dataclass
class StoredFact:
    fact: str                        # fact text
    source: str                      # filename
    page: Optional[int] = None
    quote: Optional[str] = None      # verbatim quote if available
    category: Optional[str] = None   # financial, timeline, entity, etc. (NOT epistemic category)
    extracted: str = ""              # ISO date string
    query_context: Optional[str] = None
```

`epistemic_category` and `authority_weight` do **not** exist on `StoredFact`. They are pending classifier integration (CONTEXT.md Section 6, Step 9).

## Key config values

From `RLMConfig` in `engine.py`:

| Field | Default | Effect |
|---|---|---|
| `max_depth` | 3 | Max recursion depth |
| `max_iterations` | 10 | Hard iteration cap on investigation loop |
| `max_leads_per_level` | 3 | Leads created per planning cycle |
| `excerpt_chars_simple` | 8000 | Document read limit for simple queries |
| `excerpt_chars_complex` | 40000 | Document read limit for complex queries |
| `parallel_reads` | 3 | Concurrent document reads |
| `early_exit_facts` | 5 | Exit loop early when this many facts accumulated |
| `max_research_turns` | 4 | Hard cap on ResearchAgent decide_next_action calls |
| `max_research_actions_per_turn` | 6 | Parallel tool calls per research turn |
| `research_tool_timeout_s` | 45.0 | Per-tool timeout in ResearchAgent |

## Active branches

| Branch | State | MRR | What it is |
|---|---|---|---|
| `search/density-ranking-upgrade` | **Current HEAD. Local only, not pushed.** | 0.164 | 3-commit density scoring chain (both branches scored) + MRR harness. Top of merge queue. |
| `feat/search+` | PR open. Ready to merge. | 0.167 | Density scoring on OR-fallback only (12 lines). Prerequisite for the density-upgrade branch. |
| `feat/rlm-improvements` | Remote. Production target. | 0.073 | Current production branch. Density scoring not yet applied. |
| `feat/inline` | Local. | — | Inline citation work. |
| `feat/bias-reduction` | Local. | — | Related to classifier/bias workstream. |
| `multi-modal` | Local. | — | Multimodal (image/OCR) detection work. |
| Remote: `feat/caselaw-agentic`, `feat/db-integration`, `feat/ocr-integration` | Remote only. | — | Workstream branches, not active locally. |

## Do not touch without reading CONTEXT.md first

Before making changes to `search.py`, `classifier_experiments.py`, or `fact_store.py`, read CONTEXT.md — these files have active experiment gates and a validated merge sequence.

---
