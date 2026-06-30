# External Research Redesign — Implementation Plan

> Companion docs: `external-research-flow-and-redesign.md` (current flow), `courtlistener-api-reference.md` (API details).

## 1. Goal

Replace the two-phase keyword-routed pipeline with a tool-calling research agent. The agent picks from a typed tool registry (CourtListener search / citation-lookup / cluster / opinion / cites-traversal + Tavily search/extract) and may batch multiple tool calls per turn. Results commit to state **immediately** so synthesis never depends on agent coherence. Frontend contract is preserved (new `lead.update` kinds are additive).

## 2. Scope

**Replaced:**
- `_check_if_external_needed` (trigger-count gate)
- `generate_external_queries` (LITE query generator)
- `check_search_sufficiency` (FLASH second-round gate)
- Keyword routing in `_execute_external_searches` (`"case" in query.lower()`)

**Kept (unchanged contracts):**
- `state.citations` shape + `source_type="case_law"|"web"`
- `self._external_research["case_law" | "web" | "analysis"]` slots (additive fields only)
- `_format_external_research` → synthesis feed
- `_add_external_citations` dedup semantics
- `lead.started / update / done` SSE lifecycle

## 3. New / Changed Files

| File | Change |
|---|---|
| `core/external_search.py` | Add `lookup_citations`, `get_cluster`, refactor `get_opinion` (uses `fields=` for perf); add filter params to `search_opinions` (court, caseName, status, citeCount, dateFiled, order_by, cites). All tools normalize to `CaseLawResult` dict (§7 of API ref). |
| `core/research_tools.py` *(NEW)* | Tool registry: `{name, schema, execute, summarize_for_log}`. Single source of truth for the agent. |
| `rlm/research_agent.py` *(NEW)* | The loop. Builds rolling log, calls `decide_next_action`, dispatches actions in parallel, commits to state, emits SSE, runs final brief. |
| `rlm/decisions.py` | +`should_research_externally` (LITE), +`decide_next_action` (FLASH). Rename `analyze_external` → `build_research_brief` (same output schema). Delete `generate_external_queries`, `check_search_sufficiency`. |
| `rlm/prompts.py` | +`P_SHOULD_RESEARCH`, +`P_DECIDE_NEXT_ACTION`, +`P_BUILD_BRIEF`. Delete `P_GENERATE_EXTERNAL_QUERIES`, `P_CHECK_SUFFICIENCY`. |
| `rlm/engine.py` | Both small+large paths converge to `research_agent.run(state, research_context)`. `_execute_external_searches` becomes a thin shim (kept for any tests that still call it) or is deleted. |
| `docs/streaming-event-format.md` | Document new `lead.update` kinds: `tool_call`, `opinion_fetched`, `citations_validated`, `citing_cases`. |

## 4. Entry Points — Which LLM Calls Trigger Research

Research is always entered with a **`research_context`** struct. Two upstream LLM calls produce one.

### 4.1 Small-repo path
**Trigger:** `assess_small_repo` (FLASH) returns `can_answer_from_docs=false`.
**Schema change:** delete `case_law_searches[]` and `web_searches[]` from its output schema. Keep `complexity, can_answer_from_facts, can_answer_from_docs, gap, reasoning, relevant_facts`.

```python
research_context = {
  "gap":          assessment["gap"],
  "reasoning":    assessment["reasoning"],
  "cached_facts": state.findings["accumulated_facts"][:15],
  "triggers":     {},                    # none accumulated in small-repo path
  "source_path":  "small_repo",
}
```

### 4.2 Large-repo path
**Trigger:** end of `_investigate_loop`, after doc reading completes.
**Gate (NEW):** one **LITE** call — `should_research_externally(query, facts[:15], triggers)` → `{needed: bool, reason: str}`. If `needed=false`, skip the agent entirely. This replaces both the `min_triggers>=2` heuristic *and* the `generate_external_queries` LITE call.

```python
research_context = {
  "gap":          gate["reason"],
  "reasoning":    "Doc extraction surfaced legal/jurisdictional triggers",
  "cached_facts": state.findings["accumulated_facts"][:15],
  "triggers":     state.get_trigger_summary(),   # jurisdictions, regs, doctrines, ...
  "source_path":  "large_repo",
}
```

Both paths then call `research_agent.run(state, research_context)` — identical downstream.

## 5. LLM Calls Inventory

| # | Call | Tier | Where | Input (key fields) | Output (key fields) | Repeats |
|---|---|---|---|---|---|---|
| 1 | `assess_small_repo` | FLASH | small-repo entry (schema trimmed) | `query, all_content, cached_facts, context` | `complexity, can_answer_from_facts, can_answer_from_docs, gap, reasoning, relevant_facts` | 1× |
| 2 | `should_research_externally` | LITE | large-repo gate *(NEW)* | `query, facts[:15], triggers_summary` | `needed: bool, reason: str` | 1× |
| 3 | `decide_next_action` | FLASH | agent loop, **every turn** *(NEW)* | `query, research_context, tool_schemas, research_log` | `reasoning, actions:[{tool,args}], done_after_this` | ≤6× |
| 4 | `build_research_brief` | FLASH | agent end (renamed `analyze_external`, same output) | `query, normalized_case_law[:100], web[:30]` | `key_precedents[], legal_standards[], regulations[], combined_framework, summary` | 1× |
| 5 | `synthesize` | FLASH/PRO | after agent (unchanged) | `query, evidence, external_research, pinned_content` | final answer string | 1× |

**Net change per investigation:** `-1` LITE (`generate_external_queries` deleted) `-1` FLASH (`check_search_sufficiency` deleted) `+1` LITE (new gate) `+N` FLASH (new decide loop, N≤6) `+0` FLASH (brief is rename). Worst case: `+N` FLASH turns. Typical case (validation-style queries): `N=1`.

## 6. The Agent Loop — How LLM Calls Are Made Inside Research

```
research_agent.run(state, research_context):
    log = ResearchLog()                   # rolling structured summary, NOT a transcript
    budget = config.max_research_turns    # default 6
    while budget > 0:
        decision = await decide_next_action(
            query           = state.query,
            research_context= research_context,
            tool_schemas    = REGISTRY.schemas_for_prompt(),
            research_log    = log.render(),    # compact text, ~1–2k tokens max
        )
        if decision["done_after_this"] or not decision["actions"]:
            break
        results = await asyncio.gather(*[
            REGISTRY[a["tool"]].execute(**a["args"]) for a in decision["actions"]
        ], return_exceptions=True)
        for action, result in zip(decision["actions"], results):
            commit_result(state, action, result)   # see §7
            log.append(action, REGISTRY[action["tool"]].summarize_for_log(result))
        budget -= 1

    brief = await build_research_brief(
        query=state.query,
        case_law=self._external_research["case_law"][:100],
        web=self._external_research["web"][:30],
    )
    self._external_research["analysis"] = brief     # same slot as today
    emit_lead_update("External research analysis", kind="analysis", data=brief)
```

**Key properties:**
- `ResearchLog.render()` produces a compact, structured summary (tool + args + 1-line result summary + count). **Reasoning from prior turns is discarded** — the agent doesn't see its own monologue, only outcomes. Keeps the prompt bounded regardless of turn count.
- `decide_next_action` returns strict JSON. Parse-failure → fall through to brief phase (defensive).
- `actions[]` can contain N tools in parallel. No hard cap; if the agent emits 20 actions, we still execute them (with a soft-warning log).
- `commit_result` mutates state **before** the next turn is planned, so if the agent crashes or exhausts budget mid-flight, nothing is lost.

## 7. State Writes & SSE Emissions Per Tool

Every tool's `commit_result` does three things: appends to `self._external_research[...]`, calls `state.add_citation(...)`, and emits a `lead.update`. Mapping:

| Tool | `_external_research` write | `add_citation` | `lead.update(kind=...)` |
|---|---|---|---|
| `search_opinions` | extend `["case_law"]` with N normalized dicts (`source_tool="search_opinions"`) | one per new cluster, `source_type="case_law"` | `external_results` `{source:"caselaw", count, items[{name,citation,snippet,url}]}` |
| `lookup_citations` | extend `["case_law"]` with resolved clusters (`source_tool="lookup_citations"`, `validated_for_input=<cite>`); append raw response to `["citation_matches"]` *(new slot)* | one per new cluster | `citations_validated` `{resolved_count, unresolved_count, items[{input, cluster_id, case_name, citation}]}` |
| `get_opinion` | write to `["opinions"][id]` *(new slot)* with full text; merge into matching `["case_law"]` entry (populates `opinion_text`) | update existing Citation text if cluster already cited; otherwise add | `opinion_fetched` `{case_name, citation, id, char_count, url}` |
| `get_cluster` | merge metadata into existing `["case_law"]` entry; create one if not present | one if not already cited | `external_results` `{source:"caselaw", count:1, items:[...]}` |
| `find_citing_cases` (alias for `search_opinions` w/ `cites:<id>`) | extend `["case_law"]`; append IDs to `["citing_cases"][source_id]` *(new slot)* | one per new cluster | `citing_cases` `{source_case, count, items[...]}` |
| `web_search` | extend `["web"]`; set `["web_answer"]` if Tavily returned one | one per new result, `source_type="web"` | `external_results` `{source:"web", count, items[{title,snippet,url}]}` |
| `fetch_url` | extend `["web"]` with the extracted page as a pseudo-result | one, `source_type="web"` | `external_results` `{source:"web", count:1, items:[{title,url,content_preview}]}` |

**Every turn, before execution:** one `lead.update(kind="tool_call")` with `{turn, reasoning, actions:[{tool,args}]}` so the UI can show the agent's decision before results land.

All new `_external_research` slots (`citation_matches`, `opinions`, `citing_cases`) are additive and inspection-only — synthesis still reads only `case_law`, `web`, and `analysis`.

## 8. Sample Agent Run (Worked Example)

**Query:** *"Validate these Texas Supreme Court cases and summarize their holdings: Trevino v. State (991 S.W.2d 849), Formosa Plastics v. Presidio (960 S.W.2d 41), and tell me if Obergefell applies to commercial contracts."*

### Entry
Small-repo path. `assess_small_repo` returns:
```json
{"complexity":"complex","can_answer_from_docs":false,
 "gap":"Need to verify three cited authorities and reason about Obergefell's scope in commercial context",
 "reasoning":"Two citations given; one case named without citation; cross-domain question",
 "relevant_facts":[]}
```

`research_context` built → `research_agent.run()` invoked. Budget=6.

### Turn 1 — `decide_next_action` (FLASH)
Input: query + context + tool schemas + empty log. Output:
```json
{
  "reasoning":"Three cites given — batch into one lookup_citations call; keyword search on S.W. numbers would produce noise.",
  "actions":[{"tool":"lookup_citations","args":{"text":"991 S.W.2d 849; 960 S.W.2d 41; 576 U.S. 644"}}],
  "done_after_this":false
}
```

Emitted: `lead.update(kind="tool_call", data={turn:1, reasoning, actions})`.
Executed: **1 HTTP call**, returns 3 resolved clusters.
Committed: 3 entries into `_external_research["case_law"]`, 3 into `["citation_matches"]`, 3 `Citation` rows (`source_type="case_law"`, `relevance="Validated via citation lookup"`). Emits `citations_validated` + 3 `citation` SSE events.

`ResearchLog` after turn 1:
```
T1 lookup_citations(text="991 S.W.2d 849; ...") →
   Resolved 3/3: Trevino v. State (991 S.W.2d 849, Tex. Crim. App. 1999);
   Formosa Plastics v. Presidio (960 S.W.2d 41, Tex. 1998);
   Obergefell v. Hodges (576 U.S. 644, SCOTUS 2015)
```

### Turn 2 — `decide_next_action`
To answer "does Obergefell apply to commercial contracts," agent fetches Obergefell's opinion text and pulls Texas commercial-contract cases:
```json
{
  "reasoning":"Need Obergefell scope language + Tex. commercial-contract precedent.",
  "actions":[
    {"tool":"get_opinion","args":{"cluster_id":2812209,"prefer":"lead-opinion"}},
    {"tool":"search_opinions","args":{"q":"Obergefell commercial contract","court":"tex","status":"published","order_by":"citeCount desc","max_results":5}}
  ],
  "done_after_this":false
}
```

Executed in parallel (`asyncio.gather`). Commits: `_external_research["opinions"][12345]` full text; matching `case_law` entry updated with `opinion_text`; up to 5 new case_law entries from the search. Emits `opinion_fetched` + `external_results`.

### Turn 3 — `decide_next_action`
```json
{"reasoning":"Validated 3, fetched Obergefell, found Tex. commercial precedent. Sufficient.","actions":[],"done_after_this":true}
```
Loop exits.

### Brief (`build_research_brief`, FLASH)
Stored at `_external_research["analysis"]`:
```json
{"key_precedents":["Obergefell v. Hodges — state-action marriage recognition; limited commercial scope",
                   "Formosa Plastics — Texas fraudulent-inducement standard",
                   "Trevino v. State — ..."],
 "legal_standards":["..."],"regulations":[],
 "combined_framework":"...","summary":"Obergefell's holding is confined to ..."}
```
Emitted: `lead.update(kind="analysis")`.

### Synthesis (unchanged)
`_synthesize` reads `_external_research["case_law"]` (6 cases), `["web"]` (empty), `["analysis"]` (brief) → final answer. `state.citations` holds 6+2 entries for the UI overlay.

**Totals:** 2 FLASH decide + 1 FLASH brief + 1 FLASH/PRO synthesis. HTTP: 1 citation-lookup + 1 get_opinion + 1 search_opinions = **3 external calls**.
*Today's system on the same query:* 3 keyword `search_opinions` (one per cite, with `S.W.` noise), 1 Tavily, 1 `analyze_external`, often 1 sufficiency + round-2 ≈ **6 external calls + worse recall**.

## 9. Testing Plan

### 9.1 Unit — tools (mocked HTTP)
`tests/core/test_external_search_v2.py`: one test per tool verifying request URL/params, response normalization to `CaseLawResult`, and error paths (4xx, timeout, 429 with `wait_util`). Fixtures: recorded JSON for `search_opinions`, `lookup_citations` (each status 200/404/400/300/429), `get_opinion`, `get_cluster`, `cites:<id>`, Tavily search + extract.

### 9.2 Agent loop (mocked `decide_next_action` + mocked tools)
`tests/rlm/test_research_agent.py`:
- **Citation-only validation:** canned turn1→batched lookup, turn2→done. Assert 1 HTTP call, 3 citations added, 1 brief emitted.
- **Mixed name + citation:** turn1 returns parallel `lookup_citations` + N× `search_opinions`. Assert parallel dispatch (mock counter).
- **Budget exhaustion:** decide always returns `done=false` with repeating actions. Assert loop exits at `max_research_turns`, brief still runs.
- **Parse failure:** malformed JSON. Assert graceful fallthrough to brief with whatever accumulated; no exception.
- **Zero-hit → Tavily fallback:** turn1 search empty → turn2 web_search. Assert web citation added.

### 9.3 Gate
`tests/rlm/test_should_research_externally.py`: yes-case (contract-law w/ triggers), no-case (pure factual extraction).

### 9.4 SSE contract
`tests/service/test_sse_external_research.py`: run agent with mocks, capture events, assert order: `lead.started → tool_call → (external_results | citations_validated | opinion_fetched)+ → analysis → lead.done`. Shape-validate against `docs/streaming-event-format.md`.

### 9.5 End-to-end (real APIs, gated by env)
`tests/e2e/test_research_agent_live.py`, marked `@pytest.mark.live`, skipped unless `COURTLISTENER_API_TOKEN` set:
- **V1 — case-name-only:** 5 Texas case names → expect 5 parallel `search_opinions` in one turn, 5 citations.
- **V2 — citation-only:** 5 S.W. cites → expect **1** `lookup_citations` call, 5 citations.
- **V3 — CITIOM full investigation** against `C:\Users\devan\Downloads\CITIOM v Gulfstream\documents` — agent chooses tools itself; assert brief non-empty, citations span `case_law` + `web`.

### 9.6 Regression guard
Snapshot one SSE stream as `tests/fixtures/sse_research_v2.jsonl`. Any change to event shape/order breaks the test until snapshot is updated deliberately.

## 10. Open Items / Deferred

- **Chunking for `lookup_citations`:** API caps at 64K chars / 250 cites / 60 valid cites/min. Add auto-chunking + rate-limit backoff only if CITIOM fixtures need it.
- **Per-investigation cache:** keyed by `(tool, normalized_args)` to dedupe repeated calls. Phase-1 add.
- **PACER/docket-entry endpoints:** `docket-entries`, `recap-documents`, `parties`, `attorneys` are gated to select users — skip until confirmed access.
- **Semantic search (`semantic=true`):** expose as arg on `search_opinions`; let the agent choose.
- **Citation-graph UI:** `citing_cases` + `opinions_cited` accumulate enough data to power a precedent tree later. Out of scope here.
