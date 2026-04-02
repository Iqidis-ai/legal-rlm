# SSE Streaming Event Format — Frontend Reference

**Endpoint:** `POST /investigate/urls/stream`
**Protocol:** Server-Sent Events (SSE)
**Content-Type:** `text/event-stream`

Each event is delivered as:
```
event: <event_type>
data: <json_payload>

```

---

## Event Lifecycle Overview

```
investigation.started
plan
  lead.started (lead_1)
    lead.update (kind: reading)
    lead.update (kind: fact)
    lead.update (kind: fact)
    lead.update (kind: insight)
  lead.done (lead_1)
  lead.started (lead_2)
    lead.update (kind: matches)
    lead.update (kind: fact)
    lead.update (kind: ranking)
    lead.update (kind: spawned -> lead_3)
  lead.done (lead_2)
  lead.started (lead_3, parent: lead_2)
    lead.update (kind: reading)
    lead.update (kind: fact)
  lead.done (lead_3)
progress
checkpoint
replan (if insufficient)
  lead.started (lead_4) ...
  lead.done (lead_4)
checkpoint
synthesis.started
synthesis.complete
complete
```

Key invariants:
- Every `lead.started` has exactly one matching `lead.done` or `lead.error`.
- `lead.update` events always appear between their lead's `started` and `done`.
- `lead.update` events reference their `lead_id` — use this to group them under the correct lead in the UI.
- `citation` events fire independently (not inside lead lifecycle). They may appear at any time.
- `progress` events fire at boundaries only (after `lead.done`, `checkpoint`, `synthesis.complete`), not after every step.

---

## Event Types

### `investigation.started`

Fires once at the very beginning of an investigation.

| Field | Type | Description |
|---|---|---|
| `query` | `string` | The user's investigation query |
| `document_count` | `number` | Number of documents in the repository |
| `repository` | `string` | Repository name (folder name) |

```json
{
  "query": "What are the payment terms in the contract?",
  "document_count": 12,
  "repository": "CITIOM-v-Gulfstream"
}
```

---

### `plan`

Fires after the planning phase completes. Contains the full investigation plan with all leads.

| Field | Type | Description |
|---|---|---|
| `leads` | `array` | List of planned leads (see Lead object below) |
| `success_criteria` | `string` | What constitutes a sufficient answer |
| `key_issues` | `string[]` | Key issues identified in the query |
| `strategy` | `string` | LLM's reasoning about how to investigate |
| `iteration` | `number` | Plan iteration (1 for initial, 2+ for replans) |

**Lead object** (in `leads` array):

| Field | Type | Description |
|---|---|---|
| `id` | `string` | 8-char UUID prefix (e.g. `"a1b2c3d4"`) |
| `type` | `string` | `"read"` \| `"search"` \| `"caselaw"` \| `"web"` |
| `description` | `string` | Human-readable lead description |

```json
{
  "leads": [
    {"id": "a1b2c3d4", "type": "read", "description": "Read document: Contract_Agreement.pdf"},
    {"id": "e5f6a7b8", "type": "search", "description": "Search for: payment terms invoice schedule"},
    {"id": "c9d0e1f2", "type": "search", "description": "Search for: penalty late payment interest rate"}
  ],
  "success_criteria": "Identify all payment terms, due dates, penalties, and related clauses",
  "key_issues": ["payment schedule", "late payment penalties", "invoice requirements"],
  "strategy": "Start by reading the main contract, then search for specific payment-related terms across supporting documents",
  "iteration": 1
}
```

---

### `lead.started`

Fires when a lead begins execution.

| Field | Type | Description |
|---|---|---|
| `lead_id` | `string` | 8-char UUID prefix matching a lead from the `plan` event |
| `type` | `string` | `"read"` \| `"search"` \| `"caselaw"` \| `"web"` |
| `description` | `string` | Human-readable lead description |
| `parent_lead_id` | `string \| null` | ID of the lead that spawned this one, or `null` for top-level leads |

```json
{
  "lead_id": "e5f6a7b8",
  "type": "search",
  "description": "Search for: payment terms invoice schedule",
  "parent_lead_id": null
}
```

Spawned lead example (created during another lead's execution):
```json
{
  "lead_id": "f3a4b5c6",
  "type": "read",
  "description": "Read document: Addendum_Payment_Schedule.pdf",
  "parent_lead_id": "e5f6a7b8"
}
```

---

### `lead.update`

Fires during lead execution with real-time deltas. This is the most frequent event — use it to build live UI.

| Field | Type | Description |
|---|---|---|
| `lead_id` | `string` | Which lead this update belongs to |
| `kind` | `string` | Update kind (see table below) |
| `data` | `object` | Kind-specific payload (see below) |

#### Update Kinds

| `kind` | When | `data` payload |
|---|---|---|
| `matches` | After a search lead executes | `{query, match_count, docs}` |
| `fact` | When a fact is extracted | `{fact, source_doc}` |
| `ranking` | When a document is ranked | `{doc, criticality}` |
| `reading` | When a document read begins | `{doc}` |
| `insight` | After extraction, LLM's analysis | `{learned, gaps, next_steps}` |
| `spawned` | When a lead creates a child lead | `{new_lead_id, type, description}` |
| `external_results` | After external search returns | `{source, count, items}` |
| `analysis` | After external results analyzed | `{summary, key_precedents, regulations, ...}` |
| `triggers` | When external research triggers found | `{count, triggers}` |

#### `kind: "matches"`

Search lead found results in the repository.

```json
{
  "lead_id": "e5f6a7b8",
  "kind": "matches",
  "data": {
    "query": "payment terms invoice schedule",
    "match_count": 47,
    "docs": [
      {"name": "Contract_Agreement.pdf", "hit_count": 32},
      {"name": "Addendum_2023.pdf", "hit_count": 15}
    ]
  }
}
```

Zero matches:
```json
{
  "lead_id": "e5f6a7b8",
  "kind": "matches",
  "data": {
    "query": "force majeure clause",
    "match_count": 0,
    "docs": []
  }
}
```

#### `kind: "fact"`

A fact extracted from a document. Streamed one at a time as they're found.

```json
{
  "lead_id": "a1b2c3d4",
  "kind": "fact",
  "data": {
    "fact": "Payment is due within 30 days of invoice date per Section 4.2",
    "source_doc": "Contract_Agreement.pdf"
  }
}
```

Note: `source_doc` may be a filename (from `_read_document`) or a search query string (from `_analyze_results_consolidated` where facts are extracted from search hit context).

#### `kind: "ranking"`

LLM ranked a document's relevance to the investigation.

| `criticality` values | Meaning |
|---|---|
| `"DECISIVE"` | Core document, will be loaded in full for synthesis |
| `"CRITICAL"` | Very relevant, should be read |
| `"SUPPORTING"` | Somewhat relevant |
| `"IRRELEVANT"` | Not relevant, will be skipped |

```json
{
  "lead_id": "e5f6a7b8",
  "kind": "ranking",
  "data": {
    "doc": "Contract_Agreement.pdf",
    "criticality": "DECISIVE"
  }
}
```

#### `kind: "reading"`

A document read has begun within a lead.

```json
{
  "lead_id": "a1b2c3d4",
  "kind": "reading",
  "data": {
    "doc": "Contract_Agreement.pdf"
  }
}
```

#### `kind: "insight"`

LLM's analysis of what was learned from a document, what gaps remain, and suggested next steps. Fields may be `null` if not applicable.

```json
{
  "lead_id": "a1b2c3d4",
  "kind": "insight",
  "data": {
    "learned": "Contract specifies net-30 payment terms with 1.5% monthly late fee",
    "gaps": "No information about early payment discounts or payment method requirements",
    "next_steps": "Search for addendum or amendments that may modify payment terms"
  }
}
```

#### `kind: "spawned"`

A lead created a child lead during its execution (e.g., analysis recommended reading a document or doing another search). The spawned lead will get its own `lead.started` event later.

```json
{
  "lead_id": "e5f6a7b8",
  "kind": "spawned",
  "data": {
    "new_lead_id": "f3a4b5c6",
    "type": "read",
    "description": "Read document: Addendum_Payment_Schedule.pdf"
  }
}
```

#### `kind: "external_results"`

External search (case law or web) returned results.

```json
{
  "lead_id": "g7h8i9j0",
  "kind": "external_results",
  "data": {
    "source": "caselaw",
    "count": 3,
    "items": [
      {
        "name": "Smith v. Jones Corp",
        "citation": "456 F.3d 789 (5th Cir. 2023)",
        "snippet": "The court held that net-30 payment terms are enforceable..."
      },
      {
        "name": "ABC Inc. v. DEF LLC",
        "citation": "123 S.W.3d 456 (Tex. App. 2022)",
        "snippet": "Late payment penalties of 1.5% per month were deemed reasonable..."
      }
    ]
  }
}
```

Web search variant:
```json
{
  "lead_id": "k1l2m3n4",
  "kind": "external_results",
  "data": {
    "source": "web",
    "count": 2,
    "items": [
      {
        "name": "Texas Prompt Payment Act - Overview",
        "snippet": "Under Texas law, government entities must pay contractors within 30 days..."
      }
    ]
  }
}
```

#### `kind: "analysis"`

Consolidated analysis of external research results.

```json
{
  "lead_id": "x1y2z3a4",
  "kind": "analysis",
  "data": {
    "summary": "Texas law generally enforces contractual payment terms as written",
    "key_precedents": [
      "Smith v. Jones: Net-30 terms enforceable",
      "ABC v. DEF: 1.5% monthly late fee is reasonable"
    ],
    "regulations": [
      "Texas Prompt Payment Act (Gov't Code Ch. 2251)"
    ],
    "legal_standards": [
      "Contractual late fees must not be punitive"
    ],
    "combined_framework": "Payment terms in commercial contracts are enforceable under Texas UCC, subject to unconscionability defense"
  }
}
```

#### `kind: "triggers"`

External research triggers discovered in a document (jurisdictions, regulations, legal doctrines that might need external verification).

```json
{
  "lead_id": "a1b2c3d4",
  "kind": "triggers",
  "data": {
    "count": 3,
    "triggers": [
      "jurisdictions: Texas",
      "regulations_statutes: UCC Article 2",
      "legal_doctrines: unconscionability"
    ]
  }
}
```

---

### `lead.done`

Fires when a lead completes successfully.

| Field | Type | Description |
|---|---|---|
| `lead_id` | `string` | Which lead finished |
| `duration_ms` | `number` | How long the lead took in milliseconds |

```json
{
  "lead_id": "e5f6a7b8",
  "duration_ms": 8700
}
```

---

### `lead.error`

Fires when a lead fails. The lead will NOT get a `lead.done` event.

| Field | Type | Description |
|---|---|---|
| `lead_id` | `string` | Which lead failed |
| `error` | `string` | Error message |

```json
{
  "lead_id": "a1b2c3d4",
  "error": "Failed to read Contract.pdf: PDF extraction error"
}
```

---

### `checkpoint`

Fires after an investigation iteration to report whether enough evidence has been gathered.

| Field | Type | Description |
|---|---|---|
| `decision` | `string` | `"sufficient"` \| `"insufficient"` |
| `total_facts` | `number` | Total facts accumulated so far |
| `docs_read` | `number` | Total documents read so far |
| `reasoning` | `string` | LLM's reasoning about progress |

```json
{
  "decision": "insufficient",
  "total_facts": 4,
  "docs_read": 2,
  "reasoning": "Found basic payment terms but missing information about penalty enforcement and dispute resolution procedures"
}
```

```json
{
  "decision": "sufficient",
  "total_facts": 12,
  "docs_read": 5,
  "reasoning": "Comprehensive coverage of payment terms, penalties, dispute resolution, and relevant case law"
}
```

---

### `replan`

Fires when the checkpoint determines more investigation is needed and new leads are added.

| Field | Type | Description |
|---|---|---|
| `new_leads` | `array` | Newly added leads (same Lead object format as `plan`) |
| `iteration` | `number` | Which iteration this replan is for (2, 3, ...) |

```json
{
  "new_leads": [
    {"id": "p1q2r3s4", "type": "search", "description": "Search for: dispute resolution arbitration clause"},
    {"id": "t5u6v7w8", "type": "read", "description": "Read document: Exhibit_B_Penalties.pdf"}
  ],
  "iteration": 2
}
```

---

### `synthesis.started`

Fires when the final synthesis (report generation) begins.

| Field | Type | Description |
|---|---|---|
| `fact_count` | `number` | Number of facts being synthesized |
| `citation_count` | `number` | Number of citations collected |
| `case_law_count` | `number` | Number of case law results available |
| `web_count` | `number` | Number of web search results available |
| `model` | `string` | `"FLASH"` (simple queries) or `"PRO"` (complex queries) |

```json
{
  "fact_count": 12,
  "citation_count": 8,
  "case_law_count": 3,
  "web_count": 2,
  "model": "PRO"
}
```

---

### `synthesis.complete`

Fires when synthesis finishes. The full output will follow in the `complete` event.

| Field | Type | Description |
|---|---|---|
| `output_length` | `number` | Character count of the generated analysis |
| `duration_ms` | `number` | Synthesis duration in milliseconds |
| `docs_read` | `number` | Total documents read during investigation |
| `facts_used` | `number` | Total facts used in synthesis |
| `citations` | `number` | Total citations in the output |

```json
{
  "output_length": 4936,
  "duration_ms": 38000,
  "docs_read": 5,
  "facts_used": 12,
  "citations": 8
}
```

---

### `step` (legacy)

Fires for step types that don't map to the new hierarchical events (e.g. `thinking`, `finding`). These are transitional and may carry `visible: false` in their details — use this flag to decide whether to show them in the UI.

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Step ID |
| `step_type` | `string` | `"thinking"` \| `"search"` \| `"reading"` \| `"finding"` \| `"error"` |
| `content` | `string` | Human-readable step description |
| `details` | `object \| null` | Additional structured data. Check `details.visible` |
| `depth` | `number` | Recursion depth (0 = top level) |
| `timestamp` | `string` | ISO 8601 timestamp |
| `duration_ms` | `number \| null` | Step duration if available |

```json
{
  "id": "abc12345",
  "step_type": "thinking",
  "content": "Found 5 potentially relevant cached facts",
  "details": {"visible": true},
  "depth": 0,
  "timestamp": "2026-03-31T10:15:30.123456",
  "duration_ms": null
}
```

**`details.visible`:** When `false`, the step is a debug/internal event. The frontend should still store it (for debugging/trace views) but should NOT display it in the primary investigation timeline. When `true` or absent, display normally.

---

### `citation`

Fires when a citation is found. Independent of the lead lifecycle — can appear at any point.

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Citation ID |
| `document` | `string` | Source document path or name. Prefixed with `[Case Law]` or `[Web]` for external sources |
| `page` | `number \| null` | Page number (null for external sources) |
| `text` | `string` | The cited text / quote |
| `context` | `string` | Surrounding context or metadata |
| `relevance` | `string` | Why this citation is relevant |
| `timestamp` | `string` | ISO 8601 timestamp |
| `url` | `string \| null` | Document URL (S3 presigned URL or external URL) |
| `mime` | `string \| null` | MIME type of the source document |

Local document citation:
```json
{
  "id": "cit_a1b2",
  "document": "Contract_Agreement.pdf",
  "page": 4,
  "text": "Payment shall be due and payable within thirty (30) days of receipt of invoice.",
  "context": "Section 4.2 - Payment Terms",
  "relevance": "Direct quote defining payment timeline",
  "timestamp": "2026-03-31T10:15:45.000000",
  "url": "https://iqidis-artifact.s3.amazonaws.com/...",
  "mime": "application/pdf"
}
```

External case law citation:
```json
{
  "id": "cit_c3d4",
  "document": "[Case Law] Smith v. Jones Corp",
  "page": null,
  "text": "The court held that net-30 payment terms are standard and enforceable in commercial contracts...",
  "context": "Citation: 456 F.3d 789 (5th Cir. 2023) | Court: Fifth Circuit",
  "relevance": "External case law research",
  "url": "https://www.courtlistener.com/opinion/...",
  "mime": null
}
```

---

### `progress`

Fires at iteration boundaries only (not after every step). Use for overall progress indicators.

| Field | Type | Description |
|---|---|---|
| `status` | `string` | Current investigation status |
| `elapsed_seconds` | `number` | Time since investigation started |
| `documents_read` | `number` | Total documents read |
| `searches_performed` | `number` | Total searches performed |
| `citations` | `number` | Total citations found |
| `leads_investigated` | `number` | Leads completed |
| `leads_pending` | `number` | Leads remaining |
| `facts_accumulated` | `number` | Total facts extracted |
| `entities_found` | `number` | Named entities discovered |

```json
{
  "status": "investigating",
  "elapsed_seconds": 15.3,
  "documents_read": 3,
  "searches_performed": 2,
  "citations": 5,
  "leads_investigated": 3,
  "leads_pending": 1,
  "facts_accumulated": 8,
  "entities_found": 4
}
```

---

### `complete`

Terminal event. Contains the full investigation result. Always the last meaningful event before the stream closes.

| Field | Type | Description |
|---|---|---|
| `query` | `string` | Original query |
| `analysis` | `string` | Full markdown analysis output |
| `citations` | `array` | All citations (same format as `citation` events) |
| `entities` | `array` | Named entities found |
| `facts` | `string[]` | All accumulated facts |
| `documents_processed` | `number` | Total documents read |
| `duration_seconds` | `number` | Total investigation time |
| `session_id` | `string \| null` | Session ID for follow-up queries |

```json
{
  "query": "What are the payment terms?",
  "analysis": "## Payment Terms Analysis\n\nBased on review of 5 documents...",
  "citations": [ ... ],
  "entities": [ ... ],
  "facts": [
    "Payment is due within 30 days of invoice (Section 4.2)",
    "Late payment incurs 1.5% monthly interest (Section 4.3)"
  ],
  "documents_processed": 5,
  "duration_seconds": 45.2,
  "session_id": "sess_abc123"
}
```

---

### `error`

Fatal error that terminates the investigation. May appear instead of `complete`.

| Field | Type | Description |
|---|---|---|
| `error` | `string` | Error message |

```json
{
  "error": "Failed to download documents from S3: Access Denied"
}
```

---

## Lead Type Reference

| Type | Description | Created by |
|---|---|---|
| `read` | Read a specific document | Planning phase, or spawned from search analysis |
| `search` | Search the repository for terms | Planning phase, checkpoint replan, or spawned from analysis |
| `caselaw` | External case law search (CourtListener) | External search phase |
| `web` | External web search (Tavily) | External search phase |

Lead types are derived from the lead description prefix:
- `"Read document: ..."` → `read`
- `"Search for: ..."` → `search`
- `"CaseLaw: ..."` → `caselaw`
- `"Web: ..."` → `web`

---

## Parent-Child Lead Tree

Leads can spawn child leads during execution. Use `parent_lead_id` from `lead.started` to build a tree:

```
lead_1 (search: "payment terms")        parent: null
  lead_3 (read: "Contract.pdf")          parent: lead_1   (spawned via lead.update kind=spawned)
  lead_4 (search: "late fees")           parent: lead_1   (spawned via lead.update kind=spawned)
lead_2 (read: "Agreement.pdf")          parent: null
lead_5 (caselaw: "payment enforcement") parent: null      (from external search phase)
```

Top-level leads (from the initial plan) always have `parent_lead_id: null`. Spawned leads reference their parent.

---

## Suggested Frontend UI Mapping

| Event | UI Action |
|---|---|
| `investigation.started` | Show investigation header with query and doc count |
| `plan` | Show lead list / investigation plan |
| `lead.started` | Add lead to active leads panel, show spinner |
| `lead.update` (matches) | Show search result count under lead |
| `lead.update` (fact) | Append fact to lead's fact list |
| `lead.update` (ranking) | Show document relevance badge |
| `lead.update` (reading) | Show "Reading document..." under lead |
| `lead.update` (insight) | Show learned/gaps summary |
| `lead.update` (spawned) | Show "spawned new lead" indicator, draw tree connection |
| `lead.update` (external_results) | Show external search results |
| `lead.update` (analysis) | Show external research summary |
| `lead.done` | Mark lead complete, show duration |
| `lead.error` | Mark lead failed with error message |
| `checkpoint` | Show sufficiency status bar |
| `replan` | Add new leads to plan, indicate iteration |
| `synthesis.started` | Show "Generating report..." with source counts |
| `synthesis.complete` | Show synthesis stats |
| `citation` | Add to citations panel |
| `progress` | Update global progress counters |
| `complete` | Show final analysis, mark investigation done |
| `error` | Show error state |
| `step` (visible=true) | Show in secondary activity log |
| `step` (visible=false) | Store but don't display (available in debug view) |
