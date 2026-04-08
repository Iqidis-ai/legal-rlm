Here is the concrete, implementation-ready patch plan in the same order of precedence.  
I’m anchoring to current code lines and methods.

1) [src/irys/matter/graph.py:2709](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\matter\graph.py:2709) — add document-intelligence persistence primitives (MVP layer)

Add new stores after `DocumentInventoryStore` (around line 2709).

```python
class DocumentCardStore:
    def __init__(self, db: SQLiteMatterDB, matter_id: str) -> None: ...
    def upsert(
        self,
        doc_id: str,
        *,
        title: str | None = None,
        doc_type: str | None = None,
        doc_subtype: str | None = None,
        source_side: str | None = None,
        author: str | None = None,
        sender: str | None = None,
        recipient: str | None = None,
        creation_date: str | None = None,
        sent_date: str | None = None,
        effective_date: str | None = None,
        discovery_date: str | None = None,
        purpose: str | None = None,
        rhetorical_posture: str | None = None,
        reliability_posture: str | None = None,
        operative_status: str = "unknown",
        privilege_flag: bool = False,
        unresolved_flags: list[str] | None = None,
    ) -> str: ...
    def get_by_doc_id(self, doc_id: str) -> dict | None: ...
    def get_by_path(self, relative_path: str) -> dict | None: ...
    def list_candidates(
        self,
        doc_types: list[str] | None = None,
        unresolved_only: bool = False,
        issue_id: str | None = None,
        limit: int = 200,
    ) -> list[dict]: ...
```

```python
class SpanStore:
    def __init__(self, db: SQLiteMatterDB, matter_id: str) -> None: ...
    def upsert(
        self,
        document_id: str,
        span_type: str,
        span_text: str,
        *,
        page_start: int | None = None,
        page_end: int | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
        section_ref: str | None = None,
        clause_ref: str | None = None,
        parent_span_id: str | None = None,
        ordinal_in_doc: int | None = None,
        text_hash: str | None = None,
    ) -> str: ...
    def list_by_document(self, document_id: str, span_type: str | None = None, limit: int = 200) -> list[dict]: ...
```

```python
# Minimal extension to existing class at [src/irys/matter/graph.py:2722](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\matter\graph.py:2722)
def set_salience(self, doc_id: str, salience_score: float) -> None: ...
def get_doc_row(self, doc_id: str) -> dict | None: ...   # if needed by callers beyond existing usage
```

Key logic:
- `DocumentCardStore.upsert` uses `INSERT OR REPLACE` on `document_card.doc_id` and JSON-encodes `unresolved_flags`.
- `SpanStore.upsert` computes `text_hash` with sha256 when missing.
- `list_candidates` should join `document_inventory` so ranking can be based on salience/recency/path/doc_type.
- No schema migration needed in first pass; `document_card` and `span` already exist in [schema.py](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\matter\schema.py:237).

2) [src/irys/matter/matter.py:24](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\matter\matter.py:24), [58](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\matter\matter.py:58) — wire memory stores into MatterModel facade

Modify imports and constructor, then add matter-level memory helpers.

```python
from .graph import (
    ...,
    DocumentCardStore, SpanStore,
)

class MatterModel:
    def __init__(...):
        ...
        self.document_cards = DocumentCardStore(db, matter_id)
        self.spans = SpanStore(db, matter_id)
```

```python
def upsert_document_intelligence(
    self,
    relative_path: str,
    analysis: dict,
    focus_issue_id: str | None = None,
    file_type: str | None = None,
) -> str: ...
def get_document_card(self, relative_path: str | None = None, doc_id: str | None = None) -> dict | None: ...
def list_search_seed_docs(
    self,
    issue_id: str | None = None,
    query: str | None = None,
    doc_types: list[str] | None = None,
    include_related_versions: bool = False,
    limit: int = 120,
) -> list[str]: ...
def add_doc_span(self, doc_id: str, payload: dict) -> str: ...
```

```python
def build_query_context(self) -> QueryMatterContext:
    # include top document-card signals in the context payload (newest/most unresolved, stale, high salience)
```

Key logic:
- `upsert_document_intelligence` maps LLM output into `document_card`, updates `document_inventory` salience, and reuses `relative_path` for stable lookup.
- `list_search_seed_docs` pulls candidate paths from `document_cards` + `document_inventory` and includes version-head preference via `inventory.get_operative_version`.
- `build_query_context` should expose doc-memory summary so search/search planning is already issue- and context-aware before first lead.

3) [src/irys/core/repository.py:85](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\core\repository.py:85), [239](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\core\repository.py:239), [285](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\core\repository.py:285) — eliminate full-walk waste; enable candidate-limited retrieval

```python
def list_files(
    self,
    pattern: str = "**/*",
    file_types: Optional[list[str]] = None,
    use_cache: bool = True,
) -> list[FileInfo]: ...
def _ensure_file_cache(self, pattern: str = "**/*", file_types: list[str] | None = None) -> list[FileInfo]: ...
def list_paths(self, folder: str | None = None, file_types: list[str] | None = None, use_cache: bool = True) -> list[str]: ...
def search(
    self,
    query: str,
    folder: Optional[str] = None,
    file_types: Optional[list[str]] = None,
    regex: bool = False,
    case_sensitive: bool = False,
    context_lines: int = 2,
    max_workers: Optional[int] = None,
    file_paths: list[str | Path] | None = None,
) -> SearchResults: ...
def search_multi(
    self,
    queries: list[str],
    folder: Optional[str] = None,
    require_all: bool = False,
    file_paths: list[str | Path] | None = None,
) -> SearchResults: ...
```

Key logic:
- `_file_cache` is currently unused; use it for repeated candidate retrieval in leads.
- When `file_paths` is provided, skip `list_files()` entirely for that call.
- Keep backward compatibility: existing callers keep working unchanged.
- Keep `DocumentSearch` cache behavior as-is (repo shares `_doc_cache` already).

4) [src/irys/core/search.py:75](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\core\search.py:75) — make query expansion usable by the engine (and issue-aware)

```python
def expand_query(
    query: str,
    max_expansions: int = 3,
    context_terms: list[str] | None = None,
) -> list[str]: ...
```

Key logic:
- Keep existing synonym expansion and add `context_terms` fusion (append top 1–2 legal-context terms from active predicates).
- Return de-duplicated, bounded list (`max_expansions + 1` base + context variants), preserving deterministic ordering.
- Engine will call this directly from `_build_lead_queries` (next step); no need for grep-style ad hoc query variants elsewhere.

5) [src/irys/rlm/engine.py:1620](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\rlm\engine.py:1620), [1926](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\rlm\engine.py:1926), [2008](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\rlm\engine.py:2008) — refactor lead-driven search into staged, memory-aware planning

```python
def _build_lead_queries(
    self,
    lead: Lead,
    focus_issue_id: Optional[str],
    max_queries: int = 8,
) -> list[str]: ...
def _candidate_files_for_lead(
    self,
    state: InvestigationState,
    repo: MatterRepository,
    lead: Lead,
    max_files: int = 120,
) -> list[str]: ...
def _select_deep_read_targets(
    self,
    state: InvestigationState,
    results: SearchResults,
    lead: Lead,
    focus_issue_id: Optional[str],
) -> list[str]: ...
```

Modify:
- `async def _investigate_loop(...)` (`engine.py:1620`): at each iteration call `state.apply_feedback_to_leads()` before lead scoring (w/ low cost), then proceed with existing coverage-weighted partition.
- `async def _investigate_lead(...)` (`engine.py:1926`): replace direct `repo.search(...)` with:
  - `_build_lead_queries(...)`
  - `_candidate_files_for_lead(...)` from memory (`MatterModel.document_cards`, `document_inventory`, plus current issue gap context)
  - `repo.search_multi(queries, file_paths=candidates)` first; fallback to full search only when hits are below threshold.
- `async def _analyze_search_results(...)` (`engine.py:2008`): after extraction, call `_select_deep_read_targets(...)` (not direct top-by-score by file) so deep reads prioritize:
  - issue-linked evidence gaps (`_get_issue_coverage_map`, `state.feedback`)
  - new/unread documents
  - higher salience/doc-type priority
  - unresolved document flags from cards.

6) [src/irys/rlm/engine.py:2414](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\rlm\engine.py:2414), [2507](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\rlm\engine.py:2507), [2540](C:\Users\devan\OneDrive\Desktop\Projects\legal-rlm\src\irys\rlm\engine.py:2540) — make deep-read write back memory and avoid wasted re-reads

```python
def _persist_document_intelligence(
    self,
    state: InvestigationState,
    repo: MatterRepository,
    doc_id: str,
    file_path: str,
    analysis: dict,
    focus_issue_id: str | None,
) -> None: ...
def _extract_spans_from_analysis(
    self,
    doc_id: str,
    analysis: dict,
    file_path: str,
    source_role: str | None = None,
) -> list[dict]: ...
```

Apply minimal edits:
- `async def _ingest_documents(...)` (`engine.py:2414`):
  - do not force full-repo hot pass every run.
  - ingest new/changed + top-priority unread candidates first, bounded by `parallel_reads * k` (e.g., 3x parallel budget), using `MatterModel.list_search_seed_docs(...)`.
  - continue if weak issue coverage remains.
- `async def _batch_deep_read(...)`:
  - keep semaphore but use deduped candidate order already from `_select_deep_read_targets`.
- `async def _deep_read_document(...)` (`engine.py:2540`):
  - add `finally` to `state._reading_in_progress.discard(_rel_path)` (currently only in some early-fail branches).
  - on successful parse, call `_persist_document_intelligence(...)` to:
    - `document_cards.upsert(...)`
    - write spans (`span` rows for quotes, key clauses, evidence anchors, contradiction markers)
    - update `document_inventory.set_salience(...)` and operative relations when version/citation evidence suggests supersession/contradiction.
  - on hot path, if doc has non-stale card and unchanged hash, count as cache hit as today but still update card-derived signal in state/context.
  - keep existing behavior for gap recording and assertion recording unchanged.

If you want, I can turn this into a ready-to-apply diff grouped by file in the same sequence.