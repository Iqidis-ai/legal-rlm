# External Research Deep Dive
## How Irys RLM Performs External Legal Research

This document provides a comprehensive explanation of how external research (CourtListener case law + Tavily web search) is triggered, executed, analyzed, and fed back into the investigation loop across all API paths.

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [API Entry Points](#2-api-entry-points)
3. [Core Primitives](#3-core-primitives)
4. [Small Repository Path](#4-small-repository-path)
5. [Large Repository Path](#5-large-repository-path)
6. [The `_execute_external_searches` Function](#6-the-_execute_external_searches-function)
7. [How Results Feed Back Into the Agentic Loop](#7-how-results-feed-back-into-the-agentic-loop)
8. [SSE Streaming Events for External Research](#8-sse-streaming-events-for-external-research)
9. [Configuration and Guards](#9-configuration-and-guards)

---

## 1. System Overview

External research in Irys is **not a default step** — it is triggered conditionally based on LLM assessment. The core philosophy mirrors how a real lawyer works:

1. **Read the documents first** — understand what you have.
2. **Identify gaps** — what legal questions cannot be answered from the documents alone?
3. **Search externally only for those gaps** — case law precedents, regulations, statutes.
4. **Weave external findings into the synthesis** — produce a richer, legally-grounded answer.

There are two external research APIs:
- **CourtListener** (`https://www.courtlistener.com/api/rest/v4/search`) — U.S. case law opinions
- **Tavily** (`https://api.tavily.com/search`) — AI-native web search (regulations, news, standards)

Both are managed by `ExternalSearchManager` in `src/irys/core/external_search.py`.

---

## 2. API Entry Points

All three investigation endpoints eventually call `engine.RLMEngine.investigate()`. External research behavior is identical regardless of which endpoint is used.

| Endpoint | Mode | File |
|---|---|---|
| `POST /investigate/urls/stream` | SSE streaming, real-time events | `service/api.py` |
| `POST /investigate/urls/sync` | Blocking, returns full result | `service/api.py` |
| `POST /investigate` | Async job (S3 prefix) | `service/api.py` |

**Request structure (`S3UrlsInvestigateRequest`):**
```json
{
  "query": "What legal standards apply to aircraft inspection delays?",
  "s3_urls": ["https://bucket.s3.amazonaws.com/doc1.pdf", "..."],
  "session_id": "optional-session-id",
  "context": {
    "conversation_history": [...],
    "planning_instructions": "Focus on FAA regulations",
    "output_instructions": "Respond in English"
  }
}
```

**Flow before `engine.investigate()` is called:**
1. URLs downloaded to temp directory via `S3Repository.download_urls_to_temp(job_id, s3_urls)`
2. Session facts/citations loaded: `_load_session(config, session_id)` → `seed_facts[]`, `seed_citations[]`
3. `irys.investigate(query, repository=temp_dir, seed_facts, seed_citations, context)` called
4. This calls `engine.investigate(query, repository_path, seed_facts, seed_citations, context)`

---

## 3. Core Primitives

### 3.1 `_external_research` Dictionary
The engine accumulates all external findings in a single instance variable, reset at the start of each investigation:
```python
self._external_research = {
    "case_law": [],        # list[dict] — LegalCase.to_dict() objects
    "web": [],             # list[dict] — Tavily result objects
    "web_answer": None,    # str | None — Tavily's AI-generated summary
    "analysis": {          # dict — Gemini FLASH analysis of all results
        "case_law": {
            "key_precedents": [],
            "legal_standards": [],
            "summary": ""
        },
        "web": {
            "regulations": [],
            "standards": [],
            "summary": ""
        },
        "combined": ""     # str — combined legal framework
    }
}
```

### 3.2 `LegalCase` Dataclass
```python
@dataclass
class LegalCase:
    id: str
    case_name: str
    court: str
    date_filed: Optional[str]
    citation: Optional[str]        # e.g. "123 F.3d 456"
    docket_number: Optional[str]
    opinion_text: Optional[str]    # Full text (truncated to 2000 chars in to_dict())
    url: Optional[str]             # e.g. "https://www.courtlistener.com/opinion/..."
    snippet: Optional[str]         # Search result snippet (highlighted text)
```

### 3.3 `WebSearchResult` Dataclass
```python
@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str           # Snippet or extracted content
    score: Optional[float] # Relevance score from Tavily
    published_date: Optional[str]
```

### 3.4 `ExternalSearchManager`
Unified facade over CourtListener and Tavily:
```python
manager = ExternalSearchManager(
    courtlistener_token=os.environ["COURTLISTENER_API_TOKEN"],  # Optional
    tavily_api_key=os.environ["TAVILY_API_KEY"],                # Required for web
)
```

### 3.5 `InvestigationState.external_triggers`
A categorized set of strings accumulated during document reading:
```python
state.external_triggers = {
    "case_law_refs":    set(),  # Case names, citations found in docs
    "regulatory_refs":  set(),  # FAA regs, UCC sections, etc.
    "statutes":         set(),  # Statutory references
}
```
Populated by `_read_document()` → `extract_from_document()` which returns an `external_triggers` dict.

---

## 4. Small Repository Path

**Trigger condition:** `repo.is_small_repo == True` (total chars fit in a single LLM context window)

### 4.1 Entry Point: `_direct_answer(state, repo)`

```
engine.investigate()
    └── repo.is_small_repo == True
        └── _direct_answer(state, repo)
```

### 4.2 Step-by-Step Flow

**Step 1 — Load all content**
```python
all_content = repo.get_all_content()  
# → single string: all docs concatenated
# e.g. "=== contract.pdf ===\n...\n=== amendment.pdf ===\n..."
state.documents_read = len(repo.list_files())
state.findings["small_repo_content"] = all_content
```

**Step 1.5 — Extract facts per document (if fact store empty)**
For each document, `_read_document()` is called, which calls `decisions.extract_from_document()` (Gemini FLASH). This populates `state.findings["accumulated_facts"]` and `state.external_triggers`.

**Step 1.6 — Get cached facts**
```python
relevant_facts = fact_store.get_relevant(state.query)
cached_facts_str = fact_store.format_for_llm(relevant_facts)
# → pre-formatted string for LLM prompt injection
```

**Step 2 — Unified assessment (THE KEY DECISION)**

```python
assessment = await decisions.assess_small_repo(
    query=state.query,
    content=all_content,           # All doc content
    client=self.client,
    cached_facts=cached_facts_str, # Fact sheet from prior runs
    context=self._context,         # Conversation history, instructions
    active_step=t_step,            # Telemetry
)
```

**Gemini FLASH call** using `P_ASSESS_SMALL_REPO` prompt.

**Returns:**
```json
{
  "can_answer_from_facts": false,
  "relevant_facts": [],
  "complexity": "complex",
  "can_answer_from_docs": false,
  "gap": "Need case law on FAA 14 CFR 91.409 inspection interval disputes",
  "case_law_searches": [
    "FAA 14 CFR 91.409 inspection interval aircraft",
    "breach of contract aircraft maintenance delay damages"
  ],
  "web_searches": [
    "FAA advisory circular aircraft inspection requirements 2023"
  ],
  "reasoning": "Documents contain the contract terms but no applicable legal standards..."
}
```

**Decision tree after assessment:**
```
can_answer_from_facts == True → synthesize immediately from facts
         ↓ else
can_answer_from_docs == True  → synthesize from documents only (no external search)
         ↓ else
can_answer_from_docs == False → RUN EXTERNAL SEARCH
```

**Step 3 — Execute external searches (if needed)**

Calls `_execute_external_searches(state)` with `case_law_searches` and `web_searches` from assessment.
See [Section 6](#6-the-_execute_external_searches-function) for full detail.

**Step 4 — Sufficiency check (after round 1 results)**

```python
results_summary = self._format_results_summary()
# → "CASE LAW:\n- Smith v. Jones (123 F.3d 456): ..."
# → "WEB RESULTS:\n- FAA Advisory Circular 91-101: ..."

sufficiency = await decisions.check_search_sufficiency(
    query=state.query,
    original_gap=gap,              # gap string from assessment
    results_summary=results_summary,
    client=self.client,
)
```

**Gemini FLASH call** using `P_CHECK_SEARCH_SUFFICIENCY` prompt.

**Returns:**
```json
{
  "sufficient": false,
  "additional_search": "FAA enforcement actions aircraft inspection violations",
  "remaining_gap": "No enforcement precedent found yet"
}
```

If `sufficient == False`, a **round 2 search** is triggered with `additional_search` as the query, routed to case law or web based on keyword detection.

**Step 5 — Synthesis**

After all searches complete, `_synthesize(state, is_simple)` is called.
See [Section 7](#7-how-results-feed-back-into-the-agentic-loop).

---

## 5. Large Repository Path

**Trigger condition:** `repo.is_small_repo == False`

```
engine.investigate()
    └── repo.is_small_repo == False
        ├── _assess_and_create_plan(state, repo)    ← planning only
        ├── _investigate_loop(state, repo, cache)   ← reads docs, triggers external search
        └── _synthesize(state, is_simple)
```

### 5.1 Phase 1 — Planning (`_assess_and_create_plan`)

Gemini FLASH call with `P_ASSESS_AND_PLAN` prompt.

**Input:**
```python
assessment = await decisions.assess_and_plan(
    query=state.query,
    file_list=file_list_str,     # "- contract.pdf (245KB, pdf)\n- exhibit_A.pdf ..."
    total_files=stats.total_files,
    client=self.client,
    cached_facts=cached_facts_str,
    context=self._context,
)
```

**Returns:**
```json
{
  "complexity": "complex",
  "can_answer_from_facts": false,
  "priority_files": ["Main_Contract.pdf", "Inspection_Report.pdf"],
  "search_terms": ["192-month inspection", "gulfstream aircraft delay"],
  "key_issues": ["inspection interval compliance", "breach of contract damages"],
  "success_criteria": "Determine whether CITIOM's claim of delayed inspection is valid",
  "case_law_searches": [],
  "web_searches": [],
  "reasoning": "..."
}
```

> **Note:** `case_law_searches` and `web_searches` in the planning phase are almost always empty. External search is deliberately deferred until after document reading reveals concrete triggers. This avoids over-searching with generic terms.

Creates leads from `priority_files` (read leads) and `search_terms` (search leads).

### 5.2 Phase 2 — Investigation Loop (`_investigate_loop`)

Iterates up to `max_iterations` (default: 10), processing `max_leads_per_level` (default: 3) leads per iteration.

**Per-iteration steps:**

**a) Process leads in parallel:**
```python
tasks = [self._investigate_lead(state, repo, lead, cache) for lead in leads_to_process]
await asyncio.gather(*tasks, return_exceptions=True)
```

For each "Read document" lead, `_read_document()` calls `decisions.extract_from_document()` (FLASH):
```json
{
  "facts": ["The 192-month inspection was due on 2021-03-15"],
  "quotes": [{"text": "...", "page": 12, "relevance": "Direct evidence of delay"}],
  "external_triggers": {
    "case_law_refs": ["breach of contract damages aviation"],
    "regulatory_refs": ["14 CFR 91.409"],
    "statutes": []
  },
  "insights": "Document confirms inspection was delayed by 47 days",
  "gaps": "No explanation for delay found in this document",
  "next_steps": "Check maintenance log for explanation"
}
```

Triggers are accumulated into `state.external_triggers` via `state.add_triggers(triggers)`.

**b) Checkpoint (sufficiency + replanning):**
```python
checkpoint_result = await decisions.checkpoint(
    query=state.query,
    findings=findings_summary,
    plan=plan_summary,
    client=self.client,
    cached_facts=cached_facts_str,
)
```
Returns: `{sufficient, should_replan, new_search_terms[], files_to_check[], progress_assessment}`

If `sufficient == True` and `docs_read > 0` → break out of loop.

**c) Dynamic external search trigger check:**
```python
new_queries = await self._check_if_external_needed(state, executed_external_queries)
```

This function:
1. Checks `state.documents_read >= 2` and `state.has_external_triggers(min_triggers=2)`
2. Gets `state.get_trigger_summary()` → comma-separated trigger string
3. Calls `decisions.generate_external_queries()` (LITE model):

```python
result = await decisions.generate_external_queries(
    query=state.query,
    facts=facts[:15],              # Up to 15 extracted facts
    entities=entities[:10],        # Entity names identified
    client=self.client,
    triggers=triggers,             # Trigger summary string
)
```

**Returns:**
```json
{
  "case_law_queries": ["aircraft inspection delay breach of contract FAA"],
  "web_queries": ["14 CFR 91.409 inspection interval requirements"],
  "reasoning": "Documents reference FAA regulations and contract breach..."
}
```

4. Filters out already-executed queries (deduplication via `executed_external_queries` set)
5. If new queries exist → calls `_execute_external_searches(state, case_law_queries, web_queries)`

### 5.3 Phase 3 — Synthesis

Same as small repo — see [Section 7](#7-how-results-feed-back-into-the-agentic-loop).

---

## 6. The `_execute_external_searches` Function

This is the workhorse that actually calls the external APIs. Called from both small repo (`_direct_answer`) and large repo (`_investigate_loop`).

**Signature:**
```python
async def _execute_external_searches(
    self,
    state: InvestigationState,
    case_law_queries: list[str] = None,  # Explicit queries, or read from state.findings
    web_queries: list[str] = None,
)
```

If `case_law_queries` and `web_queries` are `None`, reads from `state.findings["initial_plan"]`.

**Config limits applied:**
```python
case_law_queries = case_law_queries[:self.config.max_case_law_queries]  # default: 5
web_queries = web_queries[:self.config.max_web_queries]                 # default: 5
```

### 6.1 Parallel Execution (default mode)

```python
tasks = [search_case_law(q) for q in case_law_queries] + 
        [search_web(q) for q in web_queries]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**`search_case_law(query)` — CourtListener:**
```
POST https://www.courtlistener.com/api/rest/v4/search/
    ?q={query}&type=o&order_by=score+desc
Headers: Authorization: Token {COURTLISTENER_API_TOKEN}

Response: {
  "results": [
    {
      "id": 1234567,
      "caseName": "Smith v. Jones",
      "court": "ca9",
      "dateFiled": "2019-03-15",
      "citation": ["987 F.3d 654"],
      "docketNumber": "19-cv-12345",
      "snippet": "The court held that...",
      "absolute_url": "/opinion/1234567/smith-v-jones/"
    }, ...
  ]
}
```

Returns `list[LegalCase]` → converted to `list[dict]` via `LegalCase.to_dict()`.
Max results: `self.config.max_case_law_results` (default: 5) per query.

**`search_web(query)` — Tavily:**
```
POST https://api.tavily.com/search
{
  "api_key": "...",
  "query": "{query}",
  "search_depth": "basic",
  "max_results": 5,
  "include_answer": true,
  "include_usage": true
}

Response: {
  "query": "...",
  "answer": "FAA requires...",       ← AI-generated summary
  "results": [
    {
      "title": "FAA Advisory Circular AC 91-101",
      "url": "https://www.faa.gov/...",
      "content": "This circular establishes...",
      "score": 0.98,
      "published_date": "2022-01-15"
    }, ...
  ],
  "usage": {"credits": 1}
}
```

### 6.2 Lead Lifecycle for Each Query

Each query (case law or web) gets its own **Lead** object, emitting SSE events:
```
lead.started  → {lead_id, type: "caselaw"|"web", description: "CaseLaw: {query}"}
lead.update   → {lead_id, kind: "external_results", data: {source, count, items[]}}
lead.done     → {lead_id}
```

Results accumulate into `self._external_research`:
```python
self._external_research["case_law"].extend(cases)   # list[dict]
self._external_research["web"].extend(web_results)   # list[dict]
self._external_research["web_answer"] = data["answer"]  # str
```

### 6.3 Post-Search Analysis (Gemini FLASH)

After all search tasks complete, a **consolidated analysis call** is made:

```python
analysis = await decisions.analyze_external(
    query=state.query,
    case_law_results=case_law_text,  # Formatted top-5 case law results
    web_results=web_text,            # Formatted top-5 web results
    client=self.client,
)
```

**Gemini FLASH call** using `P_ANALYZE_EXTERNAL` prompt.

**Returns:**
```json
{
  "key_precedents": ["Smith v. Jones establishes that..."],
  "legal_standards": ["FAA requires inspection within 12 calendar months"],
  "regulations": ["14 CFR 91.409 - Annual inspection requirement"],
  "regulatory_standards": ["AC 91-101 provides guidance on..."],
  "combined_framework": "The applicable legal framework consists of...",
  "summary": "Case law and regulations establish a clear duty to..."
}
```

Stored in `self._external_research["analysis"]`.

### 6.4 Citation Creation

After analysis, `_add_external_citations(state)` commits all results to `state.citations`:

**Case law citations:**
```python
state.add_citation(
    document="[Case Law] Smith v. Jones",
    page=None,
    text=case.get("snippet") or case.get("opinion_text") or "",
    context=f"Citation: 987 F.3d 654 | Court: ca9",
    relevance="External case law research",
    url="https://www.courtlistener.com/opinion/...",
    source_type="case_law",
)
```

**Web citations:**
```python
state.add_citation(
    document="[Web] FAA Advisory Circular AC 91-101",
    page=None,
    text=result.get("content") or "",
    context=f"URL: https://www.faa.gov/...",
    relevance="External regulatory research",
    url="https://www.faa.gov/...",
    source_type="web",
)
```

Deduplication: case law by `case_name`, web by `url`.

---

## 7. How Results Feed Back Into the Agentic Loop

### 7.1 How External Research Affects Synthesis

`_synthesize(state, is_simple)` receives four source streams:

| Source | Content | Used As |
|---|---|---|
| `evidence` | `state.findings["accumulated_facts"]` (up to 20 facts) | Bullet list of extracted facts |
| `external_research` | `_format_external_research()` output | Case law + web text block |
| `pinned_content` | All doc content (small repo) or DECISIVE docs (large repo) | Full document context |
| `context` | Conversation history + instructions | Injected into prompt |

**`_format_external_research()` output structure:**
```
=== CASE LAW (CourtListener) ===
- **Smith v. Jones** (987 F.3d 654)
  Court: ca9 | Date: 2019-03-15
  Snippet: The court held that delays in aircraft inspection...

**Legal Standards Identified:** FAA regulations require annual inspection within 12 calendar months

=== REGULATIONS/STANDARDS (Web) ===
- **FAA Advisory Circular AC 91-101**
  URL: https://www.faa.gov/...
  Content: This circular establishes guidance on inspection intervals...

**Regulatory Context:** The FAA has issued specific guidance on...
```

**Final synthesis call:**
```python
response = await decisions.synthesize(
    query=state.query,
    evidence=evidence,                    # Extracted facts as bullets
    external_research=formatted_text,     # Case law + web block
    pinned_content=pinned_content,        # Full document content
    client=self.client,
    tier=ModelTier.FLASH if is_simple else ModelTier.PRO,
    context=self._context,
)
```

The Gemini model (FLASH or PRO) receives all four sources in a single prompt and produces the final answer string.

### 7.2 Large Repo Loop: External Research Does NOT Pause the Loop

In the large repo path, external searches are triggered **at the end of each iteration** inside `_investigate_loop`. The loop does NOT pause and wait for external results before continuing to the next iteration. However, since `asyncio.gather` is used within `_execute_external_searches`, all queries for a given trigger batch run in parallel and complete before the next iteration's checkpoint check.

### 7.3 Tiered Search (Deduplication Across Iterations)

In the large repo path, an `executed_external_queries` set tracks all queries that have already been executed. The `_check_if_external_needed` function filters out these queries to ensure each external search is unique across iterations. This allows progressive refinement: first iteration might search "FAA inspection requirements", second iteration (with more facts) might add "aircraft maintenance contract breach damages".

---

## 8. SSE Streaming Events for External Research

For the `/investigate/urls/stream` endpoint, external research emits these SSE events in order:

```
event: lead.started
data: {"lead_id": "...", "type": "caselaw", "description": "CaseLaw: FAA inspection delay"}

event: lead.update
data: {"lead_id": "...", "kind": "external_results", "data": {
  "source": "caselaw",
  "count": 3,
  "items": [
    {"name": "Smith v. Jones", "citation": "987 F.3d 654", 
     "snippet": "The court held...", "url": "https://..."}
  ]
}}

event: lead.done
data: {"lead_id": "..."}

event: lead.started
data: {"lead_id": "...", "type": "web", "description": "Web: FAA advisory circular"}

event: lead.update
data: {"lead_id": "...", "kind": "external_results", "data": {
  "source": "web",
  "count": 5,
  "items": [{"name": "FAA AC 91-101", "snippet": "...", "url": "https://faa.gov/..."}]
}}

event: lead.done
data: {"lead_id": "..."}

event: lead.started
data: {"lead_id": "...", "type": "search", "description": "External research analysis"}

event: lead.update
data: {"lead_id": "...", "kind": "analysis", "data": {
  "summary": "Case law and regulations establish...",
  "key_precedents": [...],
  "regulations": [...],
  "legal_standards": [...],
  "combined_framework": "..."
}}

event: lead.done
data: {"lead_id": "..."}
```

After synthesis completes:
```
event: synthesis.complete
data: {"output": "## Legal Analysis\n\n...", "citations": [...]}
```

---

## 9. Configuration and Guards

### Guards that prevent external search from running:

| Guard | Location | Effect |
|---|---|---|
| `config.enable_external_search == False` | `RLMConfig` | Disables entirely |
| `self.external_search is None` | Engine init | No `ExternalSearchManager` → no calls |
| `TAVILY_API_KEY` not set | `TavilyClient.__init__` | Web searches fail silently |
| `can_answer_from_docs == True` | `_direct_answer` | Skips external search for small repo |
| `state.documents_read < 2` | `_check_if_external_needed` | Won't trigger until 2+ docs read |
| `state.has_external_triggers(min_triggers=2) == False` | `_check_if_external_needed` | Won't trigger without document-derived triggers |
| Query deduplication | `executed_external_queries` set | Prevents re-running same queries |

### Key `RLMConfig` settings:
```python
enable_external_search: bool = True
max_case_law_results: int = 5    # Results per CourtListener query
max_web_results: int = 5         # Results per Tavily query
max_case_law_queries: int = 5    # Max CourtListener queries per investigation
max_web_queries: int = 5         # Max Tavily queries per investigation
parallel_external_searches: bool = True  # asyncio.gather vs sequential
```
