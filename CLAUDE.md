# Irys RLM - Project Instructions

## CRITICAL: Git Commit Rules

**COMMIT AFTER EVERY LOGICAL CHANGE. NO EXCEPTIONS.**

A "logical change" is ONE of the following:
- Fix a single bug
- Add a single function
- Update a single prompt
- Add a single test
- Fix imports in one file
- Any change that can be described in ONE sentence

**DO NOT:**
- Bundle multiple fixes into one commit
- Wait until "everything works" to commit
- Make commits with 500+ lines of changes
- Commit multiple unrelated changes together

**Commit message format:**
```
<short description of the ONE thing changed>

Committed by Devansh
```

**Example good commits:**
- "Fix citation text truncation - remove 200 char limit"
- "Add on_citation callback for external sources"
- "Filter out template-style queries with brackets"

**Example BAD commits:**
- "Various fixes and improvements" (too vague)
- "External search integration with multiple fixes" (too many things)

**WHY THIS MATTERS:**
- Easy to revert if something breaks
- Clear history of what changed when
- Proper code review possible
- Isolate bugs to specific changes

---

## Overview
Irys RLM (Recursive Language Model) is a legal document analysis system that investigates queries against document repositories using multi-tier LLM reasoning.

## Architecture

```
src/irys/
├── __init__.py           # Public API exports
├── api.py                # High-level API (Irys class, IrysConfig)
├── core/
│   ├── models.py         # GeminiClient (LITE/FLASH/PRO tiers, rate limiting)
│   ├── repository.py     # MatterRepository (file discovery, search)
│   ├── reader.py         # Document reading (PDF, DOCX, TXT, MHT)
│   ├── search.py         # DocumentSearch (text search, smart_search)
│   ├── cache.py          # LRUCache, DiskCache, ResponseCache
│   ├── clustering.py     # TF-IDF document clustering
│   ├── external_search.py # CourtListener + Tavily external search
│   └── utils.py          # Retry logic, telemetry, validation, config
├── rlm/
│   ├── engine.py         # RLMEngine (main investigation loop)
│   ├── decisions.py      # LLM decision functions (organized by tier)
│   ├── prompts.py        # Prompt templates (organized by tier)
│   ├── state.py          # InvestigationState, Citation, Lead, Entity
│   └── templates.py      # Investigation templates (contract, litigation, etc.)
├── service/
│   ├── api.py            # FastAPI REST API for S3/cloud deployment
│   ├── config.py         # ServiceConfig (env-based configuration)
│   ├── models.py         # Pydantic request/response schemas
│   └── s3_repository.py  # S3Repository (download, cleanup)
├── output/
│   └── formatters.py     # Markdown, HTML, JSON, PlainText formatters
└── ui/
    └── app.py            # Gradio web UI with real-time streaming
```

## Model Tiers

- **LITE** (gemini-2.5-flash-lite): Quick decisions, file picking, sufficiency checks
- **FLASH** (gemini-2.5-flash): Analysis, planning, fact extraction, simple synthesis
- **PRO** (gemini-2.5-pro): Complex synthesis only

## Test Repository
- **Location**: `C:\Users\devan\Downloads\CITIOM v Gulfstream\documents`
- **Case**: CITIOM v Gulfstream (aircraft 192-month inspection dispute)
- **Evaluation Set**: `legal_queries.json` (100 queries across 6 categories)

## Evaluation Categories
1. `factual_extraction` (20 queries) - Simple fact lookups
2. `multi_document_synthesis` (20 queries) - Cross-document analysis
3. `timeline_construction` (15 queries) - Chronological analysis
4. `contradiction_detection` (15 queries) - Finding inconsistencies
5. `legal_analysis` (15 queries) - Legal reasoning
6. `evidence_assessment` (15 queries) - Evidence evaluation

## Environment

```bash
# Required
GEMINI_API_KEY=xxx  # In .env file, never hardcode

# Install
pip install -r requirements.txt

# Run UI
python run_ui.py
```

## Improvement Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. BASELINE: Run eval set, record metrics                  │
├─────────────────────────────────────────────────────────────┤
│  2. ANALYZE: Identify failure patterns                      │
│     - Which query types fail?                               │
│     - Search not finding docs?                              │
│     - Synthesis weak?                                       │
├─────────────────────────────────────────────────────────────┤
│  3. HYPOTHESIZE: Root cause analysis                        │
│     - Search layer issues (PDF indexing, folder nav)        │
│     - LLM decision issues (prompts, model selection)        │
│     - Caching/efficiency issues                             │
├─────────────────────────────────────────────────────────────┤
│  4. IMPLEMENT: Make targeted fix                            │
│     - Edit src/irys/rlm/engine.py                           │
│     - Edit src/irys/rlm/decisions.py                        │
│     - Edit src/irys/core/search.py                          │
├─────────────────────────────────────────────────────────────┤
│  5. VALIDATE: Test the fix                                  │
├─────────────────────────────────────────────────────────────┤
│  6. COMPARE: Run full eval, compare to baseline             │
└─────────────────────────────────────────────────────────────┘
```

## Key Files for Modifications

| Area | File | Purpose |
|------|------|---------|
| Search | `src/irys/core/search.py` | Document search logic, smart_search with OR fallback |
| Engine | `src/irys/rlm/engine.py` | Main investigation loop (Plan → Investigate → Synthesize) |
| Decisions | `src/irys/rlm/decisions.py` | All LLM decision functions organized by model tier |
| Prompts | `src/irys/rlm/prompts.py` | All prompt templates organized by tier |
| Reader | `src/irys/core/reader.py` | PDF/DOCX/TXT/MHT extraction |
| External | `src/irys/core/external_search.py` | CourtListener (case law) + Tavily (web search) |
| State | `src/irys/rlm/state.py` | Citation, Lead, Entity, Timeline tracking |
| API | `src/irys/api.py` | High-level Python API (Irys class) |
| REST API | `src/irys/service/api.py` | FastAPI REST endpoints for S3/cloud |

## Entry Points

| File | Purpose |
|------|---------|
| `run_ui.py` | Launch Gradio UI (port 7862) |
| `run_server.py` | Launch FastAPI REST server (port 8000) |
| `quick_test.py` | Quick local testing script |

## External Search Integration

The engine supports external search sources for enriching investigations:
- **CourtListener**: Legal case law search (requires API token)
- **Tavily**: Web search for current information (requires API key)

Enable via environment variables:
```bash
COURTLISTENER_API_TOKEN=xxx  # Optional: case law search
TAVILY_API_KEY=xxx           # Optional: web search
```

## RLMConfig Defaults

Current defaults in `engine.py`:
- `max_depth`: 3 (reduced from 5 for efficiency)
- `max_iterations`: 10 (hard limit)
- `max_leads_per_level`: 5
- `min_lead_priority`: 0.3
- `parallel_reads`: 5
