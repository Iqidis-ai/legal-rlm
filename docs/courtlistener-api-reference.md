# CourtListener v4 API Reference (for Research Agent)

> Scoped to what the research agent actually uses. For exhaustive docs, see https://www.courtlistener.com/help/api/rest/.

**Base URL:** `https://www.courtlistener.com/api/rest/v4/`
**Auth:** `Authorization: Token <COURTLISTENER_API_TOKEN>` (recommended — 5,000 req/hr authenticated).
**Versions:** v4.3 is current. All endpoints below are v4.

---

## 1. Data Model

```
Court ──< Docket ──< Cluster ──< Opinion
                         └──< Citation (many parallel cites per cluster)
```

- **Court** (`/courts/{id}/`): cacheable; identifies jurisdiction. IDs like `scotus`, `ca9`, `nysd`, `tex`, `texapp`.
- **Docket** (`/dockets/{id}/`): case-level metadata — `docket_number`, `court_id`, `date_filed`, `date_terminated`, `case_name`, `nature_of_suit`. Lists `clusters[]` URLs but not entries/parties inline.
- **Cluster** (`/clusters/{id}/`): groups opinions of a single decision (lead + dissent + concurrence). Holds `case_name`, parallel `citations[]`, `sub_opinions[]`, `date_filed`, `judges`. **Cluster IDs are what appear in CourtListener opinion URLs** (`/opinion/{cluster_id}/slug/`), not opinion IDs.
- **Opinion** (`/opinions/{id}/`): single authored text. Prefer `html_with_citations` (pre-linked) over `plain_text`. Has `type` (`lead-opinion`, `dissent`, `concurrence-opinion`, `combined-opinion`, …), `opinions_cited[]` (outbound cites), `cluster` FK.

---

## 2. Endpoints We Use

### 2.1 Search — `GET /search/`
Powered by the Citegeist search engine (BM25 + optional semantic). **Not** a database query — does not support OPTIONS/Django filters; it uses search operators in `q`.

Key params:
- `q` — query string (supports fielded/boolean operators; see §3)
- `type` — `o` (opinion clusters, default), `r` (RECAP dockets w/ nested docs), `d` (dockets), `oa` (oral arg), `p` (judges)
- `semantic=true` — use Citegeist semantic search (case law only)
- `order_by` — `score desc` (default), `dateFiled desc|asc`, `citeCount desc`, etc.
- `highlight=on` — return highlighted snippets with `<mark>` (off by default for perf)
- `court` — court ID filter; also `filed_after` / `filed_before` (ISO-8601)

Response: `{count, next, previous, results:[{cluster_id, caseName, citation[], court_id, dateFiled, docketNumber, docket_id, citeCount, opinions[{id, type, snippet, author_id, ...}], absolute_url, ...}]}`.

### 2.2 Citation Lookup — `POST /citation-lookup/`
Parses a text blob via Eyecite, looks up every valid citation against CourtListener's 18M-citation index. **The single most efficient tool we have** for validating known citations.

Inputs (form-encoded, pick one):
- `text=<blob>` — up to **64,000 chars** per request (hard cap).
- `volume=`, `reporter=`, `page=` — look up a single citation by parts.

Per-citation response object:
```json
{
  "citation": "576 U.S. 644",
  "normalized_citations": ["576 U.S. 644"],
  "start_index": 22, "end_index": 34,
  "status": 200,          // 200=OK, 404=not found, 400=unknown reporter, 300=ambiguous, 429=over 250/request
  "error_message": "",
  "clusters": [ /* full Cluster objects: id, case_name, citations[], date_filed, docket, sub_opinions[], absolute_url */ ]
}
```

**Limits:** 250 citations per request (251st+ return 429); 60 **valid** citations per minute throttle; 64K text cap. Does **not** match statutes, `id.`, `supra`, law journals.

### 2.3 Clusters — `GET /clusters/{id}/`
Resolve a cluster by ID (e.g. from a URL `/opinion/2812209/obergefell-v-hodges/`). Returns rich metadata incl. `sub_opinions[]` URLs, parallel `citations[]`, `date_filed`, `judges`, `syllabus`. Redirected clusters return 301/410 — follow them.

### 2.4 Opinions — `GET /opinions/{id}/` or `GET /opinions/?filters`
Full opinion text. **Always use `fields=` or `omit=`** for perf — opinion text is huge.

Preferred text field: `html_with_citations` (has hyperlinked cites). Fallbacks: `plain_text`, `xml_harvard`, `html`, `html_columbia`, `html_lawbox`.

Useful filters (RelatedFilter syntax):
- `?cluster__docket__court=scotus` — opinions in a given court
- `?cluster__docket__docket_number=23A994&cluster__docket__court=scotus` — by docket number
- `?cited_opinion=32239` — opinions that cite opinion 32239 (for citation-network traversal)

### 2.5 Dockets — `GET /dockets/?filters`
Direct docket lookup. Key filters:
- `?docket_number=1:16-cv-00745&court=dcd`
- `?court=scotus&date_filed__gte=2020-01-01`
- `?court__jurisdiction=F` (federal) with `!` for exclusion: `?court__jurisdiction!=F`

### 2.6 Courts — `GET /courts/`
Cacheable. Lookup by ID to resolve `court_id` → full name / jurisdiction. 3,358 jurisdictions available.

---

## 3. Search Operators (for `q=`)

Boolean: `AND` `OR` `NOT` `-` `%` — phrase `"..."` — grouping `()` — wildcard `*` `?` `!` — fuzzy `term~`, proximity `"a b"~50` — range `[1939 TO 1945]`.

Fielded queries (camelCase on search API, snake_case in DB APIs):

| Field | Use |
|---|---|
| `caseName:"Obergefell v. Hodges"` | case name match |
| `court_id:ca9` | jurisdiction |
| `status:published` | precedential status (also `unpublished`, `errata`, …) |
| `citation:"576 U.S. 644"` | parallel citation match |
| `citeCount:[100 TO *]` | cases cited 100+ times |
| `dateFiled:[2020-01-01 TO 2024-12-31]` | date range |
| `judge:"Posner"` | judge full-text |
| `cites:<id>` | **cases citing opinion `<id>`** (citation-network forward traversal) |
| `related:<id>` | opinions most similar to opinion `<id>` |
| `docketNumber:23A994` | docket number |

---

## 4. Filtering / Pagination Conventions (non-search APIs)

- Django-style `field__gt=`, `field__gte=`, `field__range=500,1000`, `field__startswith=`.
- **RelatedFilter** joins across APIs: `cluster__docket__court=scotus`.
- **Exclusion:** prefix with `!`: `court__jurisdiction!=F`.
- **Ordering:** `order_by=-date_modified,-date_created`; provide secondary field to break ties deterministically.
- **Deep pagination:** only with `order_by=id|date_created|date_modified`; use `next`/`previous` cursors, not `page`.
- **Count only:** `count=on` returns just `{count: N}` cheaply.
- **Field selection:** `?fields=id,case_name` or `?omit=html,plain_text` — critical for opinion endpoints.

---

## 5. Rate Limits & Etiquette

- Authenticated: **5,000 requests/hour** across all endpoints.
- Citation lookup: **60 valid citations/min**, 250 per request, 64K chars per request.
- Weekly maintenance: Thursday 21:00–23:59 PT.

---

## 6. Query Pattern → Tool Mapping (Expected Agent Behavior)

These are the canonical patterns the research agent must recognize. The query text is representative, not literal.

### 6.1 Case-name-only validation
> "Validate these 5 Texas cases: Trevino v. State, Formosa Plastics v. Presidio, Kroger v. Persley, City of Keller v. Wilson, Halliburton v. KBR"

- Citation lookup won't help (no volume/reporter/page).
- Agent issues **N parallel** `search_opinions` calls, one per case name, with `caseName:"..."` + `court_id:tex*` filter where inferable.
- Each result's top cluster → `Citation(source_type="case_law")`.

### 6.2 Citation-only validation
> "Validate these: 991 S.W.2d 849, 960 S.W.2d 41, 261 S.W.3d 316, 168 S.W.3d 802, 80 S.W.3d 566"

- **One** `lookup_citations` call with all five citations concatenated into a single text blob.
- All five resolved clusters → five `Citation` entries. Done in one turn.
- **Do not** fan out to five `search_opinions` calls — keyword search on `"S.W."` matches noise.

### 6.3 Mixed validation
> "Validate Obergefell (576 U.S. 644) and Roe v. Wade"

- One `lookup_citations` for the Obergefell cite, one `search_opinions` for Roe v. Wade — executed in parallel in the same turn.

### 6.4 Issue / precedent research
> "What's the standard for contract unconscionability in Texas?"

- `search_opinions` with `q="unconscionability" court_id:tex* status:published` + `order_by=citeCount desc` to surface leading precedents.
- If top hit is pivotal, follow up with `get_opinion(id)` (use `html_with_citations`, `omit=html,plain_text,xml_harvard`) for full reasoning.
- Optionally, `search?q=cites:<lead_id>` to map progeny.

### 6.5 Docket / procedural-posture lookup
> "What's the status of docket 1:16-cv-00745 in D.D.C.?"

- `search_dockets` (or `GET /dockets/?docket_number=1:16-cv-00745&court=dcd`). Returns case metadata, not opinions.

### 6.6 Full-text fetch for a cited case
> Document mentions "as held in 576 U.S. 644 …"

- `lookup_citations(text=<quote containing cite>)` → get cluster ID.
- If agent wants the reasoning, follow with `get_opinion(sub_opinions[0])` — single extra call, typed as `lead-opinion` preferred.

### 6.7 Cases-citing-X (precedential weight)
> "What cases have cited Obergefell v. Hodges?"

- Cluster → lead opinion ID (via `cluster.sub_opinions[0]`).
- `search?q=cites:<opinion_id>&type=o&order_by=dateFiled desc` → list of citing opinions with `dateFiled`, `caseName`, `court_id`.

### 6.8 No hits in CourtListener
> Statute lookup, obscure case, secondary source

- Agent falls back to `web_search` (Tavily) and optionally `fetch_url` (Tavily extract) for the specific page. Results still land in `self._external_research["web"]` and `state.citations(source_type="web")`.

---

## 7. Implications for Our Normalization Layer

Every tool's output gets normalized into one of two shapes before landing in `self._external_research`:

**Case-law shape** (used for `search_opinions`, `lookup_citations` clusters, `get_opinion`, `find_citing_cases`):
```python
{
  "id": str,                    # cluster ID if available, else opinion ID
  "case_name": str,
  "citation": str | None,       # first parallel citation
  "all_citations": list[str],   # full citations[] for completeness
  "court": str,
  "date_filed": str | None,
  "docket_number": str | None,
  "snippet": str | None,        # search snippet OR lookup surrounding text
  "opinion_text": str | None,   # only populated if get_opinion was called
  "url": str,                   # courtlistener.com absolute URL
  "source_tool": str,           # "search_opinions" | "lookup_citations" | "get_opinion" | "find_citing_cases"
  "validated_for_input": str | None,  # set iff from lookup_citations
}
```

**Web shape** (Tavily search + extract): unchanged from current `{title, url, content, score, published_date}`.

These are the only two shapes synthesis ever sees. `_add_external_citations` and `_format_external_research` require no changes beyond reading the two new optional fields.
