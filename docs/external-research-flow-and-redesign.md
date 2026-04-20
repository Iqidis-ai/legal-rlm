# External Research — Current Flow & Redesign Scratchpad

> **Purpose:** Detailed flow diagram of the current external research system, followed by a blank section to brainstorm and agree on a new, more agentic design before touching code.

---

## PART 1: CURRENT FLOW

### High-Level Decision Tree

```
POST /investigate/urls/stream  (or /sync or /investigate)
  │
  ├── Download docs to temp dir (S3Repository.download_urls_to_temp)
  │
  └── engine.investigate(query, repository_path, seed_facts, seed_citations, context)
        │
        ├─[is_small_repo == True]──────────────────────────────────────────────────────┐
        │                                                                              │
        │  _direct_answer(state, repo)                                                 │
        │  ├── STEP 1: repo.get_all_content() → all_content: str                       │
        │  ├── STEP 1.5: _read_document() × N docs  [only if fact_store is empty]      │
        │  │   └── decisions.extract_facts(query, filename, content)  [GEMINI LITE]    │
        │  │       prompt: P_EXTRACT_FACTS                                              │
        │  │       → {                                                                  │
        │  │           facts[],              → state.add_facts() → accumulated_facts    │
        │  │           quotes[{text,page,    → state.add_citation() (top 2 only)        │
        │  │                  relevance}],                                              │
        │  │           references[],         → NOT used (lead spawning disabled)        │
        │  │           insights,  ┐                                                      │
        │  │           gaps,      ├→ ONE lead.update("insight") with payload:           │
        │  │           next_steps ┘   {learned: insights, gaps: gaps,                  │
        │  │                           next_steps: next_steps}                          │
        │  │           external_triggers: {  → state.add_triggers()                    │
        │  │             jurisdictions[],                                               │
        │  │             regulations_statutes[],                                        │
        │  │             legal_doctrines[],                                             │
        │  │             industry_standards[],                                          │
        │  │             case_references[]                                              │
        │  │           }                                                                │
        │  │         }                                                                  │
        │  ├── STEP 1.6: fact_store.get_relevant(query) → cached_facts_str             │
        │  │   NOTE: query param is IGNORED — returns ALL stored facts (up to 50)     │
        │  │   the LLM can short-circuit if the question was answered before.          │
        │  │                                                                           |
        │  ├── STEP 2: [GEMINI FLASH] assess_small_repo(query, all_content,            │
        │  │           cached_facts_str, context)                                      │
        │  │           → {can_answer_from_facts, complexity, can_answer_from_docs,     │
        │  │              gap, case_law_searches[], web_searches[], reasoning}         │
        │  │                                                                           │
        │  │  State updates from assessment:                                           │
        │  │    self._is_simple_query        ← complexity == "simple"                 │
        │  │    state.query_classification   ← {type, complexity, llm_classified}     │
        │  │    [if can_answer_from_facts]                                             │
        │  │      state.findings["accumulated_facts"] = assessment["relevant_facts"]  │
        │  │      state.findings["answered_from_cache"] = True  → synthesize, return  │
        │  │    [else] local vars: can_answer_from_docs, gap extracted for use below  │
        │  │                                                                           │
        │  ├─[can_answer_from_facts==True]──────────────────────────── synthesize ─────┤
        │  ├─[can_answer_from_docs==True]───────────────────────────── synthesize ─────┤
        │  └─[can_answer_from_docs==False]                                             │
        │       │                                                                      │
        │       │  ROUND 1:                                                            │
        │       │  case_law_queries = assessment["case_law_searches"]                 │
        │       │  web_queries      = assessment["web_searches"]                      │
        │       │  state.findings["initial_plan"] = {                                 │
        │       │      "case_law_searches": case_law_queries,  ← handoff mechanism    │
        │       │      "web_searches":      web_queries,       ← handoff mechanism    │
        │       │  }                                                                   │
        │       │  _execute_external_searches(state)  ← called with no explicit args  │
        │       │    internally reads: state.findings["initial_plan"]["case_law_..."]  │
        │       │    → extends self._external_research["case_law"] and ["web"]        │
        │       │    → state.findings["external_research"] = self._external_research  │
        │       │    → state.citations += case law + web entries                      │
        │       │                                                                      │
        │       │  SUFFICIENCY CHECK (only if results exist AND gap non-empty):       │
        │       │  results_summary = _format_results_summary()  [reads self._ext...]  │
        │       │    → "CASE LAW:\n- {name} ({citation}): {snippet[:200]}...\n..."    │
        │       │    → "WEB RESULTS:\n- {title}: {content[:200]}...\n..."             │
        │       │  [GEMINI FLASH] check_search_sufficiency(query, gap, results_summary)│
        │       │    → {sufficient, additional_search, remaining_gap}                  │
        │       │                                                                      │
        │       │  ROUND 2 (only if sufficient==False and additional_search non-empty):│
        │       │  routing heuristic (keyword-based, NOT an LLM call):                │
        │       │    is_case_law = any word in ["case","v.","vs","court",             │
        │       │                   "ruling","precedent"] in additional_search.lower() │
        │       │  state.findings["initial_plan"] OVERWRITTEN with:                   │
        │       │    {"case_law_searches": [additional_search] if is_case_law else [], │
        │       │     "web_searches":      [] if is_case_law else [additional_search]} │
        │       │  _execute_external_searches(state)  ← reads overwritten initial_plan│
        │       │    self._external_research EXTENDED (not reset)                     │
        │       │    analyze_external re-runs on ALL accumulated results (both rounds) │
        │                                                                              │
        │  synthesize ──────────────────────────────────────────────────────────────── ┤
        │                                                                              │
        └─[is_small_repo == False]────────────────────────────────────────────────────┐│
                                                                                      ││
           [GEMINI FLASH] assess_and_plan(query, file_list, total_files,              ││
                          cached_facts_str, context)                                  ││
           → {complexity, can_answer_from_facts, priority_files[],                   ││
              search_terms[], key_issues[], success_criteria,                         ││
              case_law_searches:[], web_searches:[]}    ← almost always empty         ││
                                                                                      ││
           _investigate_loop(state, repo, cache)   [max 10 iterations]               ││
           │                                                                          ││
           │  per iteration:                                                          ││
           │  ├── asyncio.gather(_investigate_lead × N leads)                          ││
           │  │                                                                       ││
           │  │   ── "Search for: {term}" lead ──────────────────────────────────    ││
           │  │   │  repo.smart_search(term) → SearchResults (hit list)              ││
           │  │   │  [GEMINI FLASH] analyze_search(query, key_issues, hits,          ││
           │  │   │                                already_read)                     ││
           │  │   │  → {relevant_hit_numbers[], facts[], citations[],                ││
           │  │   │     ranked_documents[{file,score,criticality}],                  ││
           │  │   │     additional_searches[], read_deeper[]}                        ││
           │  │   │  → DECISIVE docs pinned → state.findings["pinned_documents"]     ││
           │  │   │  → new "Read document" leads spawned from read_deeper[]          ││
           │  │   │  → new "Search for" leads from additional_searches[]             ││
           │  │   │  → batch_read(ranked_docs, parallel=3) calls _read_document      ││
           │  │                                                                       ││
           │  │   ── "Read document: {path}" lead ─────────────────────────────────  ││
           │  │      repo.read_async(file_path) → doc content (with OCR if needed)   ││
           │  │      decisions.extract_facts(query, filename, content) [GEMINI LITE]  ││
           │  │      prompt: P_EXTRACT_FACTS                                          ││
           │  │      → {                                                              ││
           │  │          facts[],              → state.add_facts() + fact_store       ││
           │  │          quotes[{text,page,    → state.add_citation() (top 2 only)    ││
           │  │                 relevance}],                                          ││
           │  │          references[],         → NOT used (lead spawning disabled)    ││
           │  │          insights,  ┐                                                   ││
           │  │          gaps,      ├→ ONE lead.update("insight") payload:             ││
           │  │          next_steps ┘   {learned, gaps, next_steps}                   ││
           │  │          fires only if (insights OR gaps) has content;                ││
           │  │          next_steps alone does NOT trigger emission                   ││
           │  │          external_triggers: {  → state.add_triggers()                 ││
           │  │            jurisdictions[],      ↳ state.external_triggers dict       ││
           │  │            regulations_statutes[],                                    ││
           │  │            legal_doctrines[],                                         ││
           │  │            industry_standards[],                                      ││
           │  │            case_references[]                                          ││
           │  │          }                                                            ││
           │  │        }                                                              ││
           │  │                                                                       ││
           │  ├── [GEMINI FLASH] checkpoint(query, findings, plan, cached_facts)      ││
           │  │   [only runs if facts_count >= 5 OR iteration > 1]                   ││
           │  │   → {sufficient, should_replan, progress_assessment,                 ││
           │  │      new_search_terms[], files_to_check[]}                           ││
           │  │   ├─[sufficient==True && docs_read>0]──── BREAK loop ────────────────┘│
           │  │   └─[should_replan==True]── add new "Search for" + "Read" leads       │
           │  │                             emits replan event                         │
           │  │                                                                        │
           │  ├── _check_if_external_needed(state, executed_queries)                  │
           │  │    ├── Guard: state.documents_read >= 2  [else → skip]                │
           │  │    ├── Guard: state.has_external_triggers(min_triggers=2)  [else→skip]│
           │  │    ├── triggers = state.get_trigger_summary()                         │
           │  │    │   → comma-separated string of all accumulated trigger values      │
           │  │    ├── [GEMINI LITE] generate_external_queries(query, facts[:15],     │
           │  │    │         entities[:10], triggers)                                 │
           │  │    │   → {case_law_queries[], web_queries[], reasoning}               │
           │  │    └── filter: remove queries already in executed_queries set          │
           │  │         └─[new queries exist]                                          │
           │  │              _execute_external_searches(state,                         │
           │  │                  case_law_queries, web_queries)                        │
           │  │              executed_queries.update(new_case_law + new_web)           │
           │  │                                                                        │
           │  └── (optional) _save_checkpoint() if checkpoint_dir configured           │
           │                                                                            │
           │  ◄─────────────────── loop back to next iteration ────────────────────── │
           │  loop exits when:                                                          │
           │    • sufficient==True && docs_read>0  (BREAK above)                       │
           │    • no pending leads remain                                               │
           │    • max_iterations (10) reached                                           │
           │                                                                            │
           _synthesize(state, is_simple)  ← called AFTER the loop exits ──────────── ┘
```

---

### `_execute_external_searches` — Detailed Internal Flow

> **Key fact about external_triggers:** The categories accumulated in `state.external_triggers`
> (jurisdictions, regulations_statutes, legal_doctrines, industry_standards, case_references)
> do NOT flow into this function. They were consumed UPSTREAM by `_check_if_external_needed`
> → `state.get_trigger_summary()` → `generate_external_queries()` (LITE) to produce plain text
> query strings. By the time `_execute_external_searches` is called, it only sees those strings.

```
INPUTS TO _execute_external_searches:
  state                — carries leads[], findings{}, citations[] for mutation
  case_law_queries[]   — plain text strings (from assess_small_repo OR generate_external_queries)
  web_queries[]        — plain text strings (same sources)
  [if both None]       → falls back to state.findings["initial_plan"]["case_law_searches/web_searches"]

ENGINE INSTANCE STATE (accumulator, NOT reset between rounds, only extended):
  self._external_research = {
    "case_law": [],      ← extended each call (never reset mid-investigation)
    "web":      [],      ← extended each call
    "web_answer": str,   ← overwritten by last Tavily answer
    "analysis": {        ← overwritten by fresh analyze_external after each round
      "case_law": {key_precedents[], legal_standards[], summary},
      "web":      {regulations[], standards[], summary},
      "combined": str
    }
  }

──────────────────────────────────────────────────────────
EXECUTION:
──────────────────────────────────────────────────────────

  1. Apply limits: case_law_queries[:max_case_law_queries(5)]
                   web_queries[:max_web_queries(5)]

  2. asyncio.gather( all case law tasks + all web tasks ) [parallel by default]

     search_case_law(query):
       GET https://www.courtlistener.com/api/rest/v4/search/
           ?q={query}&type=o&order_by=score+desc
       Headers: Authorization: Token {COURTLISTENER_API_TOKEN}  [optional]
       Response body: {results: [{id, caseName, court, dateFiled,
                                   citation[], docketNumber, snippet,
                                   absolute_url}, ...]}
       Returns: list[dict] where each dict has:
         {id, case_name, court, date_filed, citation,
          docket_number, opinion_text[:2000], url, snippet}
       → self._external_research["case_law"].extend(these dicts)

     search_web(query):
       POST https://api.tavily.com/search
       Body: {api_key, query, search_depth:"basic", max_results:5,
              include_answer:true, include_usage:true}
       Response body: {answer: str, results: [{title, url, content,
                        score, published_date}], usage: {credits: N}}
       → self._external_research["web"].extend(results[])
       → self._external_research["web_answer"] = answer
       → credits × $0.008 tracked as cost_usd in telemetry

  3. Per query result: create Lead in state.leads
       lead.started  → {lead_id, type:"caselaw"|"web", description:"CaseLaw: {query}"}
       lead.update("external_results") → {source, count, items[{name,citation,snippet,url}]}
       lead.done     → {lead_id}

  4. [GEMINI FLASH] analyze_external(query, case_law_text, web_text)
       Input: top-5 case law formatted + top-5 web formatted (plain text blocks)
       → {key_precedents[], legal_standards[], regulations[],
          regulatory_standards[], combined_framework, summary}
       → self._external_research["analysis"]["case_law"] = {key_precedents, legal_standards, summary}
       → self._external_research["analysis"]["web"]      = {regulations, standards, summary}
       → self._external_research["analysis"]["combined"] = combined_framework
       Emitted: create dedicated "External research analysis" lead
         lead.started → lead.update("analysis", {summary,key_precedents,
                         regulations,legal_standards,combined_framework}) → lead.done

──────────────────────────────────────────────────────────
STATE MUTATIONS AFTER EACH CALL:
──────────────────────────────────────────────────────────

  state.findings["external_research"] = self._external_research
    ↑ snapshot for reference/inspection only — synthesis does NOT read from here

  state.citations += case law + web entries  [via _add_external_citations]
    Case law: {document:"[Case Law] {case_name}", text:snippet,
               context:"Citation: X | Court: Y", url:courtlistener_url,
               source_type:"case_law"}  — deduplicated by case_name
    Web:      {document:"[Web] {title}", text:content,
               context:"URL: {url}", url:url,
               source_type:"web"}  — deduplicated by url
    ↑ used by UI citation overlay — NOT passed into synthesis LLM call directly

  self._external_research  ← this is what synthesis actually reads
    (extended across multiple rounds; analyze_external re-runs on FULL
     accumulated results after each round, so analysis reflects everything)
```

---

### Synthesis Input Streams

> **Read path:** `_synthesize` reads from `self._external_research` (engine instance var)
> via `_format_external_research()`. It does NOT read from `state.findings["external_research"]`
> (that mirror is for inspection only) and does NOT receive `state.citations` objects.
> Citations are a parallel output for the UI, not for the synthesis LLM.

```
_synthesize(state, is_simple)
  │
  ├── evidence          = "\n".join(accumulated_facts[:20])
  │
  ├── external_research = _format_external_research()
  │     reads: self._external_research  (engine instance var, all rounds combined)
  │     returns: {"case_law": str, "web": str}
  │
  │     "case_law" block built as:
  │       top-5 self._external_research["case_law"] entries formatted as:
  │         "- **{case_name}** ({citation})
  │            Court: {court} | Date: {date_filed}
  │            Snippet: {snippet[:300]}..."
  │       + appended: "**Legal Standards Identified:** {analysis.case_law.summary}"
  │
  │     "web" block built as:
  │       prepended: "**Summary:** {web_answer}"   ← Tavily AI answer if present
  │       + top-5 self._external_research["web"] entries:
  │           "- **{title}**
  │              URL: {url}
  │              Content: {content[:300]}..."
  │       + appended: "**Regulatory Context:** {analysis.web.summary}"
  │
  ├── pinned_content
  │   ├─[small_repo]── all_content (the full repo string from STEP 1)
  │   └─[large_repo]── DECISIVE pinned docs (100k total budget, 30k per doc max)
  │
  └── [GEMINI FLASH or PRO] decisions.synthesize(
          query, evidence, external_research={"case_law":str,"web":str},
          pinned_content, client, tier, context)
      → final answer string → state.findings["final_output"]
      tier: FLASH if is_simple==True, PRO otherwise
```

---

### LLM Calls Summary

| Step | Function | Model | Input | Output |
|------|----------|-------|-------|--------|
| Small repo assessment | `assess_small_repo` | FLASH | query + all_content + cached_facts + context | can_answer_from_facts/docs, complexity, case_law_searches[], web_searches[], gap |
| Sufficiency check | `check_search_sufficiency` | FLASH | query + gap + results_summary | sufficient, additional_search, remaining_gap |
| Large repo planning | `assess_and_plan` | FLASH | query + file_list + cached_facts + context | complexity, priority_files[], search_terms[], key_issues[] |
| Document extraction | `extract_facts` | LITE | query + filename + content (truncated to 10k/35k chars for simple/complex) | facts[], quotes[], references[], insights, gaps, next_steps, external_triggers{jurisdictions, regulations_statutes, legal_doctrines, industry_standards, case_references} |
| Checkpoint | `checkpoint` | FLASH | query + findings + plan + cached_facts | sufficient, should_replan, new_search_terms[], files_to_check[] |
| Query generation | `generate_external_queries` | LITE | query + facts[] + entities[] + triggers | case_law_queries[], web_queries[] |
| External analysis | `analyze_external` | FLASH | query + case_law_text + web_text | key_precedents[], regulations[], combined_framework, summary |
| Synthesis | `synthesize` | FLASH/PRO | query + evidence + external_research + pinned_content + context | final answer string |

---

### External Search Data Flow: Field Mapping

```
CourtListener API Response field  →  LegalCase field  →  Citation field
─────────────────────────────────────────────────────────────────────────
caseName                          →  case_name        →  document (prefixed "[Case Law]")
snippet                           →  snippet          →  text
citation[0]                       →  citation         →  context ("Citation: X | Court: Y")
court                             →  court            →  context
dateFiled                         →  date_filed       →  (not in citation, used in synthesis)
absolute_url                      →  url              →  url
─────────────────────────────────────────────────────────────────────────
Tavily Response field             →  _external_research["web"]  →  Citation field
─────────────────────────────────────────────────────────────────────────
results[].title                   →  web[].title      →  document (prefixed "[Web]")
results[].content                 →  web[].content    →  text
results[].url                     →  web[].url        →  url, context ("URL: X")
results[].score                   →  web[].score      →  (not in citation)
answer                            →  web_answer       →  prepended to web synthesis block
usage.credits                     →  (telemetry only) →  cost_usd = credits × $0.008
```

---

## PART 2: REDESIGN — NEW AGENTIC RESEARCH FLOW

> **Goal:** Replace the current pre-defined, two-phase research approach with a fully agentic model where the investigation loop can dynamically call any available research tool at any time — more like how a lawyer actually works.

### Current Limitations to Address

1. **Pre-determined search types:** The current system only supports case law (CourtListener) + web (Tavily). Adding a new API (e.g., citation lookup, PACER dockets, semantic scholar) requires code changes.
2. **Two-stage trigger latency:** In large repos, external search can only start after 2+ documents are read AND 2+ triggers are accumulated — it cannot search early even when the query obviously needs external law.
3. **Batch-only:** All queries for a given trigger batch run together. The agent cannot iteratively refine: "I found case X — now look up case X's citing cases."
4. **Analysis is post-hoc:** `analyze_external` runs after all searches complete. The agent cannot decide mid-search to abandon one thread and pursue another.
5. **No URL follow-through:** CourtListener returns opinion URLs, but the agent never fetches full opinion text (Tavily extract is available but unused in main flow).
6. **Hard-wired decision logic:** Whether to search case law vs web is determined by keyword heuristics (`is_case_law = "case" in query.lower()`), not by LLM judgment.

---

### Proposed New Design: Tool-Calling Research Agent

**Core idea:** Give the LLM a set of callable tools and let it decide when and how to use them, with the human-readable investigation loop still providing the outer structure.

#### Available Tools (proposed)

| Tool | API | Description |
|------|-----|-------------|
| `search_case_law` | CourtListener `/search?type=o` | Keyword + semantic search over opinions |
| `search_dockets` | CourtListener `/search?type=r` | Find PACER dockets and filings |
| `get_opinion_full` | CourtListener `/opinions/{id}/` | Fetch full opinion text by ID |
| `search_citations` | CourtListener `/search?type=o&q=citing:{id}` | Find cases that cite a known case |
| `search_web` | Tavily `/search` | General web search |
| `extract_url` | Tavily `/extract` | Fetch and extract a specific URL |
| `search_statutes` | Law.cornell.edu / Tavily filtered | Statutory text lookup |
| *(future)* `search_irac` | Custom | Find cases with same IRAC structure |

#### Proposed Flow

```
[TO BE DESIGNED]

High-level concept:
  - The investigation loop provides a "research budget" (N tool calls allowed)
  - After reading docs and accumulating facts, the LLM is given:
      * The query
      * Accumulated facts and triggers  
      * Available tools (with descriptions and schemas)
      * Prior tool calls and results
  - The LLM decides: "call tool X with params Y" or "done with research"
  - Results are fed back, LLM makes next decision
  - This continues until LLM declares sufficient or budget exhausted
  - All results feed into synthesis as before
```

#### Key Design Questions to Resolve

1. **Where does the tool-calling loop live?**
   - Inside `_investigate_loop` (replacing `_check_if_external_needed`)?
   - As a separate `_research_phase` that runs after doc reading?
   - Interleaved with doc reading (parallel to existing leads)?

2. **Which model drives tool selection?**
   - LITE (fast, cheap) — but may make poor tool choices
   - FLASH (balanced) — current choice for most decisions
   - PRO (powerful) — reserved for synthesis only
   - Or: LITE to decide *whether* to research, FLASH to decide *what* to search

3. **How many tool-call iterations?**
   - Hard budget (e.g., max 5 tool calls per investigation)?
   - Soft budget (LLM declares done)?
   - Per-source limits maintained?

4. **How do we handle tool results that suggest follow-up?**
   - Case X's snippet mentions Case Y → follow up with `get_opinion_full(case_Y_id)`?
   - Opinion cites a regulation → follow up with `search_statutes`?

5. **How does this interact with the existing fact accumulation?**
   - Should tool results inject directly into `accumulated_facts`?
   - Or remain as a separate `external_research` block?

6. **Citation management:**
   - Same `_add_external_citations` approach?
   - Or richer citation metadata (chain of reasoning for why this case was found)?

7. **Streaming behavior:**
   - Each tool call should emit `lead.started/update/done` events
   - How do we represent iterative tool chains in SSE?

---

*[Use this section to sketch the new flow and agree on approach before coding]*
