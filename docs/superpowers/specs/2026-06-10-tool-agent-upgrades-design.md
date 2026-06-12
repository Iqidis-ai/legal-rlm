# Irys RLM — Tool & Agent Upgrades Design Spec

**Date:** 2026-06-10  
**Status:** Draft  
**Scope:** A-external, B, C, D — excludes A-local (local doc tools in registry, separate plan)

---

## Through-line

Every layer of this upgrade makes the agent more honest about what it knows, what it
doesn't, and where it's uncertain. Tool descriptions stop being terse; contradictions
get surfaced instead of silently coexisting; synthesis critiques itself before returning;
decisions become auditable. Nothing in this spec changes `state.py` or any database
schema — all new per-lead metadata goes into `lead.params` (already freeform) and all
new state metadata goes into `state.findings` (already freeform).

---

## Implementation Order

```
Phase 1 → A-external   Isolated. No dependencies. Immediate agent quality improvement.
Phase 2 → D            Cross-cutting, low risk, purely additive. Makes 3 and 4 measurable.
Phase 3 → B            Prompt + engine changes. Contradiction flagging powers C's reflexion.
Phase 4 → C            Builds on B. Highest risk; D's logging lets you validate it.
Phase 5 → A-local      Separate plan. Not blocked by 1–4.
```

---

## Phase 1 — A-external: Tool Layer

### Files changed
- `src/irys/core/external_search.py` — add `get_cluster_validity()`
- `src/irys/core/research_tools.py` — rewrite descriptions, enrich `ToolContext`, fix
  `fetch_url` title, surface Tavily score, add two new tools

### 1.1 ToolContext enrichment

Add `query: str = ""` and `gap: str = ""` to `ToolContext`. Both are sourced from
`ResearchContext` at the point where `ResearchAgent` constructs the context object.
Every tool executor receives the investigation query and the specific gap being
investigated, enabling intelligent defaults without requiring the LLM to re-encode
jurisdiction and date range manually in every call.

```python
@dataclass
class ToolContext:
    external_search: ExternalSearchManager
    telemetry_step: Any = None
    query: str = ""       # full investigation query
    gap: str = ""         # specific research gap this agent turn is addressing
```

`ResearchAgent.run()` sets both fields from `context.gap` and `state.query` when
constructing `ToolContext` before each turn.

### 1.2 Tool description rewrites

Every tool description is rewritten to include:
1. A concrete example of a well-formed call
2. Explicit "do NOT use when" guidance
3. Chaining guidance — which tool to call next with results from this one

**`search_opinions`**

```
Keyword and fielded search across CourtListener case-law opinions.

Use when: locating cases by topic, doctrine, court, date range, or cite count.
Supports operators: caseName:("Obergefell"), cites:2812209, citeCount:[50 TO *],
  court:scotus, filed_after:2010-01-01.

Example: {"q": "aircraft inspection 192-month", "court": "ca9",
          "filed_after": "2000-01-01", "max_results": 5}

Do NOT use when: you already have citation strings — use lookup_citations instead.
Do NOT use when: you need full opinion text — chain to get_opinion after this call.

Chaining: pass cluster_id from results to get_opinion, get_cluster, or
check_case_validity. Pass opinion_id to find_citing_cases.
```

**`lookup_citations`**

```
Resolve every citation string inside a text blob to CourtListener clusters in one call
(Eyecite-powered). Batches many citations together.

Use when: you have explicit reporter citations extracted from documents
  (e.g. "261 S.W.3d 316; 42 U.S.C. § 1983").

Example: {"text": "See Smith v. Jones, 261 S.W.3d 316 (Tex. 2008); also
          Acme Corp v. Beta Inc., 512 F.3d 44 (9th Cir. 2007)."}

Do NOT use when: you have a topic but no citation strings — use search_opinions instead.

Chaining: use cluster_id from resolved results with get_opinion or check_case_validity.
```

**`get_opinion`**

```
Fetch the full text of a single opinion. Prefer cluster_id over opinion_id.
Use `prefer` to target dissents or concurrences ("lead-opinion", "dissent",
"concurrence").

Example: {"cluster_id": 12345, "prefer": "lead-opinion"}

Do NOT use when: you only need case metadata (date, court, citation count) —
  use get_cluster instead. Full text is expensive; only fetch when you need to
  quote or analyse the reasoning.

Chaining: run check_case_validity on the same cluster_id to confirm this opinion
  is still good law before citing it.
```

**`get_cluster`**

```
Fetch case-level metadata for a cluster: case name, court, date, citation string,
precedential status, and citation count. Does NOT return opinion text.

Example: {"cluster_id": 12345}

Do NOT use when: you need the reasoning or holding text — use get_opinion instead.

Use when: you need metadata only, or as a fast validity pre-check before fetching
  the full opinion. Prefer check_case_validity if validity is the primary question.
```

**`find_citing_cases`**

```
Forward-citation traversal: lists opinions that cite the given opinion.
Use to gauge authority, find subsequent treatment, or locate recent cases applying
the same doctrine.

Example: {"opinion_id": 98765, "filed_after": "2015-01-01", "max_results": 10}

Do NOT use when: you want to find what this case cites (backward citations) —
  use get_opinion and extract citations from the text, then resolve with
  lookup_citations.

Chaining: run check_case_validity on citing cases with high cite counts to confirm
  they are still good law.
```

**`check_case_validity`** *(new)*

```
Checks whether a case is still good law by fetching cluster metadata and extracting
precedential status, blocked flag, and citation count.

Returns: {precedential_status, blocked, citation_count, validity_summary}

Example: {"cluster_id": 12345}

Do NOT use when: you need the full opinion text — use get_opinion instead.
Do NOT use when: you only need to find cases on a topic — use search_opinions.

Use before citing any case in a final synthesis to confirm it has not been blocked
or depublished.
```

**`search_statutes`** *(new)*

```
Searches for statutory and regulatory text across Cornell LII, eCFR, and
regulations.gov. Use for primary statutory authority, regulatory requirements,
and federal rules.

Example: {"query": "14 CFR Part 91 aircraft inspection requirements",
          "jurisdiction": "federal"}

Do NOT use when: you need case law — use search_opinions instead.
Do NOT use when: the question is purely factual and unlikely to have statutory
  grounding.

Chaining: use fetch_url on high-relevance results to extract the full statutory text.
```

**`web_search`**

```
General web search via Tavily. Use for current information, news, secondary sources,
regulatory guidance documents, or when primary legal sources are insufficient.

Example: {"query": "CITIOM Gulfstream 192-month inspection NTSB report",
          "search_depth": "advanced", "max_results": 5}

Do NOT use when: you need case law — use search_opinions instead.
Do NOT use when: you need statutory text — use search_statutes instead.

Chaining: use fetch_url to extract full text from high-relevance results.
```

**`fetch_url`**

```
Extract the full text of a single web page. Use when web_search or search_statutes
returns a URL whose snippet is insufficient and you need the complete document.

Example: {"url": "https://www.ecfr.gov/current/title-14/part-91/section-91.409",
          "extract_depth": "advanced"}

Do NOT use when: you do not yet know which URL to fetch — use web_search or
  search_statutes first to identify candidates.

Note: paywalled pages will return partial content or fail. Try extract_depth
  "advanced" on failure before giving up.
```

### 1.3 New tool: `check_case_validity`

Add `get_cluster_validity()` to `CourtListenerClient` in `external_search.py`.
The method calls `get_cluster()` (already implemented) and extracts:

```python
{
    "precedential_status": cluster.get("precedential_status"),
    "blocked": cluster.get("blocked", False),
    "date_blocked": cluster.get("date_blocked"),
    "citation_count": cluster.get("citation_count", 0),
    "validity_summary": str  # human-readable one-liner derived from above
}
```

`_execute_check_case_validity` in `research_tools.py` calls this method and returns a
`ToolResult` with `update_kind="validity_check"`.

### 1.4 New tool: `search_statutes`

`_execute_search_statutes` calls `tav.search()` with a pre-scoped domain list:

```python
STATUTE_DOMAINS = [
    "law.cornell.edu",
    "ecfr.gov",
    "regulations.gov",
    "uscode.house.gov",
    "govinfo.gov",
]
```

`jurisdiction` parameter maps to an additional query prefix:
- `"federal"` → no prefix change
- `"texas"` → prepends `"Texas"` to query
- Any other string → prepended as-is

Returns `ToolResult` with `update_kind="external_results"`, `source="statutes"`.

### 1.5 Tavily score and fetch_url title

**Tavily score:** `update_data` in `_execute_web_search` includes
`"items"` entries with `"score": e["score"]` so the agent can rank results.

**fetch_url title:** Use Tavily's returned `title` field if present:
```python
"title": ex.title or url   # was: "title": url
```

---

## Phase 2 — D: Observability

### Files changed
- `src/irys/rlm/decisions.py` — add `_emit_decision_record()` helper, call from each
  decision function
- No changes to `state.py`

### 2.1 DecisionRecord

Each decision function in `decisions.py` appends a record to
`state.findings["decision_log"]` (initialised to `[]` on first write):

```python
{
    "decision_type": str,       # "plan", "fact_extraction", "sufficiency",
                                # "external_trigger", "synthesis", "reflexion_critique",
                                # "plan_pruning"
    "reasoning_summary": str,   # ≤ 100 chars extracted from LLM reasoning field
    "outcome": Any,             # structured result (bool, list, dict)
    "model_tier": str,          # "LITE" | "FLASH" | "PRO"
    "duration_ms": int,
    "timestamp": str,           # ISO
}
```

`_emit_decision_record(state, record)` is a one-function helper that does the
`state.findings.setdefault("decision_log", []).append(record)` safely.

### 2.2 Citation groundedness check

After final synthesis, a post-processing pass checks each sentence in the output
for citation markers. Sentences without a backing citation ID are collected into
`state.findings["ungrounded_claims"]`. This list is passed into the output metadata
and feeds the "citation correctness" eval metric directly.

The pass is deterministic (regex + citation ID lookup), not an LLM call.

---

## Phase 3 — B: Memory / State

### Files changed
- `src/irys/rlm/decisions.py` — enhance fact extraction prompt, add contradiction check
- `src/irys/rlm/engine.py` — lead-close convention writing `lead.findings` and
  `lead.params["citation_ids"]`

### 3.1 Contradiction flagging

The fact extraction prompt in `decisions.py` (`extract_facts` function) is extended
to include the current list of known facts from `state.findings` and asks:

> "For each new fact, check whether it contradicts any existing fact. If so, return it
> in the `contradictions` array with the conflicting existing fact and its source."

The function then calls `state.add_contradiction()` for each returned contradiction.
The subagent surfaces both facts with full provenance. It does not resolve or
suppress either side. Synthesis receives both facts plus the `Contradiction` record
and must address the discrepancy explicitly.

### 3.2 Lead compression

When the engine marks a lead as complete (setting `lead.investigated = True`), it
also writes:

```python
lead.findings = conclusion_sentence          # one-sentence summary of what was found
lead.params["citation_ids"] = [c.id for c in relevant_citations]
```

The synthesis prompt receives closed leads in compressed form:
`"{lead.description}: {lead.findings} [citations: {citation_ids}]"` rather than the
full step trace. Full traces remain in `state.thinking_steps` for the checkpoint and
telemetry record; they just don't bloat the synthesis context.

---

## Phase 4 — C: Loop Quality

### Files changed
- `src/irys/rlm/decisions.py` — add `assess_lead_relevance()` (LITE),
  `critique_synthesis()` (FLASH)
- `src/irys/rlm/engine.py` — priority decay in lead selection, reflexion pass in
  synthesis path, reflexion cycle limit
- `src/irys/api.py` / `RLMConfig` — add `max_reflexion_cycles: int = 1`

### 4.1 Lead priority decay

`lead.params["priority"]` is initialised to `1.0` when a lead is created. After each
lead batch completes, `assess_lead_relevance()` — a LITE-tier call — receives the
list of remaining unprocessed leads and the current evidence summary, and returns a
dict of `{lead_id: new_priority}`. The engine updates each lead's
`lead.params["priority"]` accordingly.

The engine's lead selection loop orders by `lead.params.get("priority", 1.0)`
descending. Leads are never deleted or marked pruned — they decay low enough that
the sufficiency checkpoint fires before they are reached. The hub's plan stays intact.

**Reflexion leads** are created with `lead.params["priority"] = 1.0` so the
checkpoint respects them before exiting.

### 4.2 Reflexion with self-consistency

After initial synthesis, before returning the result, `engine.py` calls
`critique_synthesis()` — a FLASH-tier function in `decisions.py`.

**Input:**
- Draft synthesis text
- `state.citations` (all found citations)
- `state.contradictions` (all flagged contradictions)
- Original query

**Output (structured):**
```python
{
    "ok": bool,                      # True → no re-investigation needed
    "gaps": list[str],               # topics not addressed → become new leads
    "internal_contradictions": list[str],  # synthesis contradicts itself
    "unsupported_claims": list[str], # claims with no citation backing
}
```

**Engine behaviour on the output:**

| Field | Action |
|---|---|
| `ok = True` | Proceed to final output |
| `gaps` non-empty | Create one `Lead` per gap with `priority=1.0`, re-enter investigation loop |
| `internal_contradictions` non-empty | Embed list into final synthesis prompt: model must address them explicitly |
| `unsupported_claims` non-empty | Embed list into final synthesis prompt: model marks claims as uncertain |

The structured critique is stored in `state.findings["reflexion_critique"]` for the
decision log and eval harness.

### 4.3 Reflexion cycle limit

`RLMConfig` gains `max_reflexion_cycles: int = 1`. The engine tracks
`reflexion_cycles_used: int` in the run (local variable, not in state). After one
reflexion-triggered re-investigation cycle, `critique_synthesis()` is not called
again — final synthesis runs regardless. This prevents unbounded re-entry loops.

### 4.4 Checkpoint interaction

The existing checkpoint logic is unchanged. The one addition: before the checkpoint
evaluates sufficiency, the engine checks whether any leads with
`lead.params.get("priority", 1.0) == 1.0` remain unprocessed. If reflexion leads
are pending, the checkpoint defers exit until they are processed or their priority
decays below the `min_lead_priority` threshold.

---

## Infra delta summary

| File | Nature of change |
|---|---|
| `src/irys/core/external_search.py` | Add `get_cluster_validity()` method |
| `src/irys/core/research_tools.py` | 7 rewritten descriptions, 2 new tools, ToolContext fields, fetch_url title, Tavily score |
| `src/irys/rlm/decisions.py` | `_emit_decision_record()`, enhanced fact extraction prompt, `assess_lead_relevance()` (LITE), `critique_synthesis()` (FLASH) |
| `src/irys/rlm/engine.py` | Priority-ordered lead selection, lead-close compression, reflexion pass, cycle limit, checkpoint interaction |
| `src/irys/api.py` | `max_reflexion_cycles: int = 1` in `RLMConfig` |
| `src/irys/rlm/state.py` | **Zero changes** |
| Database / checkpoint schema | **Zero changes** |

---

## Out of scope

- A-local: local doc tools in registry — separate plan
- Maker-checker pattern — separate architectural plan
- Cross-run self-consistency — eval harness concern, not loop concern
- Lifecycle hook system — separate architectural plan
- Any changes to `state.py` or checkpoint schema

---

## Evaluation

Use the existing 100-query eval set as a regression gate after each phase.
Key metrics that become computable after Phase 2 (D):

| Metric | Source after D |
|---|---|
| Checkpoint accuracy | `decision_log` entries of type `sufficiency` vs ground truth |
| External research precision | reflexion_critique `ok` rate on external-triggered leads |
| Citation correctness | `ungrounded_claims` list length per query |
| Leads per useful fact | lead count vs fact count from `decision_log` |
| Cost per answer | token counts from `decision_log` model tier entries |
