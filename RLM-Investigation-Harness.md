# RLM Investigation Harness — Complete Flow Reference

## 1. Key Primitives

### InvestigationState (`state.py`)
The central "working memory" that persists across the entire investigation. All primitives live here.

| Primitive | Type | Purpose |
|---|---|---|
| **Lead** | `Lead` dataclass | A unit of work to execute — either `"Read document: <path>"` or `"Search for: <term>"`. Has `id`, `description`, `source`, `investigated` flag, `lead_type` (`read`/`search`/`caselaw`/`web`), optional `parent_lead_id` for tracking spawned leads. Deduped via word-overlap (>0.8 = duplicate). |
| **Fact** | `str` in `findings["accumulated_facts"]` | A single extracted piece of evidence from a document. Stored as plain strings in a list. Deduped via word-overlap (>0.7 = duplicate). Added by `state.add_fact()` / `state.add_facts()`. |
| **Citation** | `Citation` dataclass | A traceable reference — `document`, `page`, `text`, `context`, `relevance`, `url`, `mime`. Deduped by normalized (document + page + first 100 chars of text). Created by `state.add_citation()`. |
| **Entity** | `Entity` dataclass | Person, company, date, amount extracted from docs. Keyed by lowercase name, tracks `mentions` count and `sources`. |
| **TimelineEvent** | `TimelineEvent` dataclass | Date-anchored events with parsing support. |
| **Contradiction** | `Contradiction` dataclass | Two conflicting statements from different sources with severity rating. |
| **ExternalTrigger** | `dict[str, set]` | Accumulated signals from documents: `jurisdictions`, `regulations_statutes`, `legal_doctrines`, `industry_standards`, `case_references`. Used to decide if external search is needed. |
| **ThinkingStep** | `ThinkingStep` dataclass | Trace/log entry with `StepType` enum — emitted to UI via SSE. Not evidence, just observability. |
| **InvestigationCache** | `InvestigationCache` dataclass (engine-local) | Tracks `extracted_docs` (set of filepaths), `searched_terms` (set), `irrelevant_docs` (set). Prevents re-reading docs and re-running similar searches. |

### Findings Dict (`state.findings`)
A free-form dict accumulating structured data throughout investigation:

| Key | Set By | Used By |
|---|---|---|
| `accumulated_facts` | `_read_document`, `_analyze_results_consolidated` | `_synthesize`, `checkpoint` |
| `initial_plan` | `_assess_and_create_plan`, `_create_plan` | `_investigate_loop` (checkpoint reference), `_execute_external_searches` |
| `issues` | `_assess_and_create_plan` | `_analyze_results_consolidated` (key_issues for ranking) |
| `small_repo_content` | `_direct_answer` | `_synthesize` (passed as pinned content) |
| `pinned_documents` | `_analyze_results_consolidated` (DECISIVE docs) | `_synthesize` → `_load_pinned_documents` |
| `external_research` | `_execute_external_searches` | `_synthesize` → `_format_external_research` |
| `final_output` | `_synthesize` | Returned to caller as the answer |
| `answered_from_cache` | `_assess_and_create_plan`, `_direct_answer` | `_synthesize` (skip doc-read guard) |

---

## 2. Entry Point

```
Irys.investigate(query, repository, seed_facts?, seed_citations?, context?)
  └── RLMEngine.investigate(query, repository_path, ...)
```

The `Irys` class (`api.py`) wraps `RLMEngine`. It creates a `GeminiClient`, `RLMConfig`, and callbacks (`on_step`, `on_citation`, `on_fact`, `on_progress`), then calls `engine.investigate()`.

### Seeding
Prior-session data can be injected:
- `seed_facts` → `state.add_facts()` before any investigation
- `seed_citations` → `state.add_citation()` for each prior citation

### Context Object
Optional `InvestigationContext` carries:
- `conversation_history` — prior Q&A turns (truncated to 300K/200K chars for planning/synthesis)
- `planning_instructions` — guidance injected into planning prompt
- `output_instructions` — guidance injected into synthesis prompt
- `output_system_instructions` — appended to PRO system prompt at synthesis

---

## 3. The Two Paths

```
investigate()
  ├── repo.is_small_repo? → _direct_answer()     [Small Repo Path]
  └── else               → Full RLM Pipeline      [Large Repo Path]
```

### 3A. Small Repo Path: `_direct_answer()`

**All documents fit in context.** Steps:

1. **Load all content** — `repo.get_all_content()` → `state.findings["small_repo_content"]`
2. **Extract facts per doc** — If fact store is empty, iterates every file through `_read_document()`, each wrapped in its own lead lifecycle (`LEAD_STARTED` → read → `LEAD_DONE`)
3. **Check cached facts** — `fact_store.get_relevant(query)` → formatted for LLM
4. **`decisions.assess_small_repo()`** — Single FLASH call determines:
   - `can_answer_from_facts` — skip doc analysis entirely
   - `complexity` — `"simple"` or `"complex"` → chooses FLASH vs PRO synthesis
   - `can_answer_from_docs` — if false, identifies `gap` and generates `case_law_searches`/`web_searches`
5. **External search** (conditional) — Only if `can_answer_from_docs=false`:
   - `_execute_external_searches()` → round 1
   - `decisions.check_search_sufficiency()` → evaluates if gap filled
   - Optional round 2 if critical gap remains
6. **`_add_external_citations()`** → adds case law/web citations to state
7. **`_synthesize(state, is_simple)`** → final answer

### 3B. Large Repo Path: Full RLM Pipeline

Three-phase architecture:

```
Phase 1: _assess_and_create_plan()   → Plan + Leads
Phase 2: _investigate_loop()          → Read/Search/Extract iteratively
Phase 3: _synthesize()                → Final answer
```

---

## 4. Phase 1 — Planning: `_assess_and_create_plan()`

### LLM Call: `decisions.assess_and_plan()` [FLASH]

**Input:** query, file list (top 50 files with names/sizes/types), cached facts
**Output JSON:**

```json
{
  "can_answer_from_facts": bool,
  "relevant_facts": [...],
  "complexity": "simple" | "complex",
  "key_issues": [...],
  "priority_files": ["file1.pdf", "file2.docx"],
  "search_terms": ["term1", "term2"],
  "success_criteria": "...",
  "reasoning": "..."
}
```

**If `can_answer_from_facts=true`:** Skip Phase 2 entirely → jump to `_synthesize()`.

**Otherwise:** Creates leads:
1. **Priority file leads** — `"Read document: <filename>"` (up to 3) — created FIRST
2. **Search term leads** — `"Search for: <term>"` (up to 3) — created AFTER reads
3. **Fallback** — If no leads, calls `decisions.extract_search_terms()` [LITE]

Emits `StepType.PLAN` event with all leads, strategy, success criteria.

---

## 5. Phase 2 — Investigation Loop: `_investigate_loop()`

The core agentic loop. Runs up to `max_iterations` (default 10).

```
while iteration < max_iterations:
    1. Get pending leads (FIFO)
    2. Take top N leads (max_leads_per_level, default 3)
    3. Process leads in PARALLEL (asyncio.gather)
    4. CHECKPOINT — sufficiency + replan
    5. Dynamic external search check
    6. Save checkpoint (if configured)
```

### 5.1 Lead Processing: `_investigate_lead()`

Every lead forks into one of two behaviors:

#### A. Read Lead (`"Read document: <path>"`)

Guard checks:
- `cache.has_extracted(filepath)` → skip (already read)
- `cache.is_irrelevant(filepath)` → skip (previously judged irrelevant)
- `state.recursion_depth > max_depth` → skip

Emit: `LEAD_STARTED` → calls `_read_document()` → `LEAD_DONE`

#### B. Search Lead (`"Search for: <term>"`)

Guard checks:
- `cache.is_similar_search(term)` → skip (word overlap > 0.6 with prior search)
- `state.recursion_depth > max_depth` → skip

Steps:
1. `cache.add_search(term)` — register the search
2. `repo.smart_search(term)` — text search with OR-fallback
3. If no hits → mark investigated, done
4. Emit `lead.update("matches", ...)` — match count and doc breakdown
5. **`_analyze_results_consolidated()`** — the key decision point

### 5.2 Consolidated Search Analysis: `_analyze_results_consolidated()`

#### LLM Call: `decisions.analyze_search()` [FLASH]

**Input:** query, key_issues, search results (up to 30 hits formatted), already_read list
**Output JSON:**

```json
{
  "relevant_hit_numbers": [1, 3, 7],
  "facts": ["fact1", "fact2"],
  "ranked_documents": [
    {"file": "contract.pdf", "score": 95, "criticality": "DECISIVE"},
    {"file": "email.pdf", "score": 60, "criticality": "SUPPORTING"},
    {"file": "invoice.pdf", "score": 10, "criticality": "IRRELEVANT"}
  ],
  "read_deeper": ["contract.pdf"],
  "additional_searches": ["breach notification"]
}
```

**What happens with the output:**

| Field | Action |
|---|---|
| `facts` | `state.add_facts(facts)` — accumulated immediately |
| `relevant_hits` | Top 3 → `state.add_citation()` with doc URL/mime. Triggers `on_citation` callback. |
| `ranked_documents` | **DECISIVE** → added to `state.findings["pinned_documents"]` (loaded at synthesis). **IRRELEVANT** → `cache.mark_irrelevant()`. Others ranked for reading. |
| `read_deeper` | Up to 2 → new `"Read document:"` leads with `parent_lead_id` tracking |
| `additional_searches` | Up to 1 → new `"Search for:"` leads with `parent_lead_id` tracking |
| Top ranked files (non-IRRELEVANT, non-extracted) | `_batch_read()` — parallel read of up to `parallel_reads` (default 5) docs |

### 5.3 Document Reading: `_read_document()`

#### LLM Call: `decisions.extract_facts()` [FLASH]

**Input:** query, filename, document content (excerpt, 10K–35K chars based on query complexity)
**Output JSON:**

```json
{
  "facts": ["The contract was signed on 2024-01-15", "Rent is $6,900/month"],
  "quotes": [{"text": "...", "page": 3, "relevance": "key clause"}],
  "insights": "This is the primary lease agreement",
  "gaps": "No termination clause found",
  "next_steps": "Check for amendments",
  "external_triggers": {
    "jurisdictions": ["California"],
    "regulations_statutes": ["CCP 585(b)"],
    "legal_doctrines": ["unlawful detainer"]
  }
}
```

**What happens with the output:**

| Field | Action |
|---|---|
| `facts` | `state.add_facts(facts)` + `fact_store.add_facts_from_extraction()` (persistent) |
| `quotes` | Up to 2 → `state.add_citation()` with page reference. Triggers `on_citation`. |
| `external_triggers` | `state.add_triggers(triggers)` — accumulated for later external search decision |
| `insights`, `gaps` | Emitted as `lead.update("insight", ...)` for UI observability |

### 5.4 Checkpoint: Sufficiency + Replan

Runs after **every** iteration (if facts ≥ `early_exit_facts` OR iteration > 1).

#### LLM Call: `decisions.checkpoint()` [LITE]

**Input:** query, formatted findings summary, plan reasoning, cached facts from `fact_store`
**Output JSON:**

```json
{
  "sufficient": false,
  "should_replan": true,
  "progress_assessment": "Found lease terms but missing breach notification details",
  "new_search_terms": ["notice to quit", "breach letter"],
  "files_to_check": ["Notice.pdf"]
}
```

**Decision logic:**

| Condition | Action |
|---|---|
| `sufficient=true` AND `docs_read > 0` | **BREAK** — exit loop, proceed to synthesis |
| `sufficient=true` AND `docs_read == 0` | **CONTINUE** — ignore (logically impossible to be sufficient with 0 docs) |
| `should_replan=true` | Add new leads: `new_search_terms` (if not similar to prior), `files_to_check` (if not extracted/irrelevant) |

### 5.5 Dynamic External Search (mid-loop)

After each iteration's checkpoint, checks if accumulated `external_triggers` warrant external research.

**Guard:** `state.documents_read >= 2` AND `state.has_external_triggers(min_triggers=2)`

#### LLM Call: `decisions.generate_external_queries()` [LITE]

**Input:** query, accumulated facts, entities, trigger summary
**Output:** `case_law_queries: [...]`, `web_queries: [...]`

Filters out already-executed queries (supports tiered/iterative searching). Remaining queries executed via:
- `_execute_external_searches()` → CourtListener (case law) + Tavily (web)
- `decisions.analyze_external()` [FLASH] → analyzes results into precedents, standards, combined framework
- Results → `state.add_citation()` for each case/web result

---

## 6. Phase 3 — Synthesis: `_synthesize()`

Combines ALL accumulated evidence into the final answer.

### Sources gathered:

| Source | How it's built |
|---|---|
| **Local evidence** | `state.findings["accumulated_facts"]` — top 20 facts formatted as bullets |
| **Pinned content** | `_load_pinned_documents()` — full text of DECISIVE docs (100K budget, 30K/doc max) |
| **External research** | `_format_external_research()` — case law + web formatted sections |

### Model Selection:
- **Simple query** (`complexity="simple"` from planning) → FLASH model with PRO system prompt
- **Complex query** → PRO model

#### LLM Call: `decisions.synthesize()` [PRO or FLASH]

**Input:** query, evidence (fact bullets), external_research (case law + web), pinned_content, context (conversation history + output instructions)
**Output:** Final markdown answer string → `state.findings["final_output"]`

Emits `SYNTHESIS_STARTED` then `SYNTHESIS_COMPLETE` with stats (output length, duration, docs read, facts used, citation count).

---

## 7. Complete LLM Call Chain

```
Phase 1 - Planning:
  ├── decisions.assess_and_plan()          [FLASH]  — plan + leads
  └── decisions.extract_search_terms()     [LITE]   — fallback if no leads

Phase 2 - Investigation Loop (per iteration):
  ├── decisions.extract_facts()            [FLASH]  — per document read
  ├── decisions.analyze_search()           [FLASH]  — per search lead (consolidated)
  ├── decisions.checkpoint()               [LITE]   — sufficiency + replan
  ├── decisions.generate_external_queries() [LITE]  — dynamic external trigger
  └── decisions.analyze_external()         [FLASH]  — external results analysis

Phase 3 - Synthesis:
  └── decisions.synthesize()               [PRO/FLASH] — final answer

Small Repo Path:
  ├── decisions.assess_small_repo()        [FLASH]  — quick assessment
  ├── decisions.extract_facts()            [FLASH]  — per document
  ├── decisions.check_search_sufficiency() [LITE]   — external gap check
  └── decisions.synthesize()               [PRO/FLASH] — final answer
```

---

## 8. SSE Event Flow (emitted to UI via callbacks)

Events emitted in order for a typical investigation:

```
INVESTIGATION_STARTED    — query, doc count, repository
  THINKING               — "Analyzing N files..."
  PLAN                   — leads[], strategy, success_criteria
  THINKING               — "Iteration 1 — processing N leads"
    LEAD_STARTED         — lead.id, type, description
      lead.update("matches")    — search hit count + doc breakdown
      lead.update("reading")    — doc being read
      lead.update("fact")       — each extracted fact
      lead.update("triggers")   — external triggers found
      lead.update("insight")    — learned / gaps / next_steps
      lead.update("ranking")    — doc criticality ranking
      lead.update("spawned")    — child lead created
    LEAD_DONE            — lead.id
  CHECKPOINT             — sufficient/insufficient, facts, docs_read, reasoning
  REPLAN                 — new leads added (if should_replan)
  THINKING               — "Iteration 2 — processing N leads"
  ...repeat...
  SYNTHESIS_STARTED      — fact_count, citation_count, model tier
  SYNTHESIS_COMPLETE     — output_length, duration_ms, docs_read, facts_used
```

---

## 9. Data Flow Diagram

```
Query ──► assess_and_plan() ──► Leads (read/search)
                                    │
              ┌─────────────────────┤
              ▼                     ▼
         Read Lead             Search Lead
              │                     │
              ▼                     ▼
       extract_facts()        smart_search()
              │                     │
     ┌────────┤                     ▼
     ▼        ▼              analyze_search()
   Facts   Citations               │
     │        │         ┌──────────┼──────────┐
     │        │         ▼          ▼          ▼
     │        │      Facts    Citations   New Leads
     │        │         │         │       (spawned)
     │        │         │         │          │
     ▼        ▼         ▼         ▼          │
  ┌──────────────────────────────────────────┘
  │         InvestigationState
  │  (accumulated_facts, citations, leads,
  │   entities, triggers, pinned_documents)
  │                     │
  │              checkpoint()
  │              ├── sufficient? ──► BREAK
  │              └── replan? ──► New Leads ──► back to loop
  │                     │
  │         generate_external_queries()
  │              │
  │         analyze_external()
  │              │
  │              ▼
  │       External Citations
  │              │
  ▼              ▼
  └──────► synthesize()
                 │
                 ▼
          Final Output (markdown)
```

Emits `StepType.PLAN` event with all leads, strategy, success criteria.

---

## 5. Phase 2 — Investigation Loop: `_investigate_loop()`

The core agentic loop. Runs up to `max_iterations` (default 10).

```
while iteration < max_iterations:
    1. Get pending leads (FIFO)
    2. Take top N leads (max_leads_per_level, default 3)
    3. Process leads in PARALLEL (asyncio.gather)
    4. CHECKPOINT — sufficiency + replan
    5. Dynamic external search check
    6. Save checkpoint (if configured)
```
