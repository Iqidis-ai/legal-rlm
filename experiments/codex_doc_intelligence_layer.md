Implemented as a design with `read -> extract -> targeted follow-up`, no repo-size branching.

### 1) Document Intelligence Layer contracts (exact surface)

#### `src/irys/matter/schema.py`
Keep current tables and extend schema.

```sql
-- Document-level intelligence (augment existing document_card)
ALTER TABLE document_card
ADD COLUMN role_in_matter TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE document_card
ADD COLUMN key_parties_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE document_card
ADD COLUMN key_clauses_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE document_card
ADD COLUMN card_summary TEXT;
ALTER TABLE document_card
ADD COLUMN extraction_model TEXT;
ALTER TABLE document_card
ADD COLUMN extraction_prompt_version TEXT;
ALTER TABLE document_card
ADD COLUMN extraction_confidence REAL NOT NULL DEFAULT 1.0;
ALTER TABLE document_card
ADD COLUMN stale_reason TEXT;
CREATE INDEX IF NOT EXISTS idx_document_card_role_in_matter
ON document_card(role_in_matter);
CREATE INDEX IF NOT EXISTS idx_document_card_updated_at
ON document_card(updated_at);

-- Relationship map + contradiction edges (upgrade existing document_relation)
ALTER TABLE document_relation
ADD COLUMN evidence_assertion_id INTEGER;
ALTER TABLE document_relation
ADD COLUMN evidence_span_id INTEGER;
ALTER TABLE document_relation
ADD COLUMN evidence_excerpt TEXT;
ALTER TABLE document_relation
ADD COLUMN detected_by TEXT;
ALTER TABLE document_relation
ADD COLUMN evidence_run_id TEXT;
CREATE INDEX IF NOT EXISTS idx_document_relation_type
ON document_relation(relation_type);
CREATE INDEX IF NOT EXISTS idx_document_relation_source
ON document_relation(source_doc_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_document_relation_target
ON document_relation(target_doc_id, relation_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_relation_dedup
ON document_relation(source_doc_id, target_doc_id, relation_type, COALESCE(evidence_assertion_id, -1));

-- Optional explicit stale marker support for re-indexing control
ALTER TABLE document_inventory
ADD COLUMN reindex_reason TEXT;
CREATE INDEX IF NOT EXISTS idx_document_inventory_ingest_status
ON document_inventory(ingest_status);
```

Use `document_relation` as canonical map with relation types:
`references`, `supersedes`, `modifies`, `contradicts`, `incorporates`, `amends`, `defines`, `revokes`.

---

### 2) Method/API signatures to add/adjust

#### `src/irys/matter/graph.py`
```python
# Document intelligence cards
def upsert_document_card(
    self,
    doc_id: int,
    title: str | None = None,
    doc_type: str | None = None,
    doc_subtype: str | None = None,
    source_side: str | None = None,
    role_in_matter: str = "unknown",
    parties: list[dict] | None = None,
    key_clauses: list[dict] | None = None,
    metadata: dict[str, Any] | None = None,
    extraction_meta: dict[str, Any] | None = None
) -> int

def get_document_card(self, doc_id: int) -> dict | None
def list_document_cards(
    self,
    matter_id: int,
    doc_ids: list[int] | None = None,
    roles: list[str] | None = None,
    doc_types: list[str] | None = None
) -> list[dict]

# Relationship map
def upsert_document_relationship(
    self,
    source_doc_id: int,
    target_doc_id: int,
    relation_type: str,
    confidence: float = 1.0,
    evidence_assertion_id: int | None = None,
    evidence_span_id: int | None = None,
    evidence_excerpt: str | None = None,
    detected_by: str = "engine"
) -> int

def list_document_relationships(
    self,
    doc_id: int | None = None,
    relation_types: list[str] | None = None
) -> list[dict]

def delete_document_relationship(
    self,
    source_doc_id: int,
    target_doc_id: int,
    relation_type: str
) -> int

# Derived contradiction materialization + cleanup
def materialize_cross_document_contradictions(
    self,
    matter_id: int,
    issue_id: str | None = None,
    min_confidence: float = 0.55
) -> list[dict]

def purge_doc_derived_artifacts(self, doc_id: int) -> int
```

#### `src/irys/matter/matter.py`
```python
def refresh_document_inventory_from_repo(self, repo_root: str) -> dict[str, list[dict]]
def upsert_document_card(self, file_path: str, card_payload: dict, extraction_meta: dict | None = None) -> int
def get_document_intelligence_cards(self, issue_id: str | None = None, doc_ids: list[int] | None = None) -> list[dict]
def get_document_relationship_graph(
    self,
    relation_types: list[str] | None = None,
    doc_ids: list[int] | None = None
) -> list[dict]
def mine_document_level_contradictions(
    self,
    issue_id: str | None = None,
    min_confidence: float = 0.55
) -> list[dict]
def rebuild_document_for_file_change(self, file_path: str, force: bool = False) -> int
```

#### `src/irys/matter/runtime.py`
```python
def get_context(self) -> QueryMatterContext
def list_annotations(self, document_id=None)
def annotate_document_intelligence(
    self,
    document_pattern: str,
    annotation_text: str,
    annotation_type: str = "intelligence",
    metadata: dict | None = None
)
def list_document_relationships(self, document_id: int | None = None, relation_types: list[str] | None = None)
```
You can keep existing `annotate_document` and expose intelligence annotation as typed variant via `annotation_type="intelligence"` or equivalent.

#### `src/irys/rlm/engine.py`
Keep existing signatures; add helper methods and invert flow inside these existing methods:

```python
def _orient(self, state: InvestigationState, repo: MatterRepository, _stats=None)
def _deep_read_document(
    self,
    state: InvestigationState,
    repo: MatterRepository,
    file_path: str,
    focus_issue_id: Optional[str] = None
)
```

Suggested helpers in same file:
```python
def _scan_documents_for_change(self, repo: MatterRepository, file_paths: list[str]) -> list[dict]
def _deep_read_document_card(self, state: InvestigationState, repo: MatterRepository, file_path: str, force: bool = False) -> int
def _extract_document_relationships(self, state: InvestigationState, repo: MatterRepository, doc_id: int, payload: dict) -> list[int]
def _targeted_followups_after_read(self, state: InvestigationState, repo: MatterRepository, doc_context: dict) -> None
def _refresh_document_contradictions(self, state: InvestigationState, focus_issue_id: Optional[str] = None) -> None
```

---

### 3) `_orient` inversion design (exact behavior change)

1. Inventory pass
- Enumerate repository files (same set each run).
- For each file, compute metadata+hash and call `DocumentInventoryStore` (`upsert` + `get_doc_row`) to classify:
  - `new`
  - `changed` (`sha256` changed)
  - `unchanged`.
- Mark stale as changed if parse/ingest status is not complete.

2. Read-first intelligence pass
- For every `new|changed|force` file:
  - call `_deep_read_document(file_path, focus_issue_id=None, force=True)` immediately.
- For unchanged files:
  - do not skip if missing/empty `document_card`.
  - if card missing, read anyway.
- No branch for small/large repo paths; one pass for all.

3. Build document intelligence context
- Aggregate from `document_card`, version status (`document_inventory`), and `document_relation`.
- Add trust-influenced weighting from `trust_overrides`.
- Add current annotations from `DocumentAnnotationStore`.
- Include contradiction map built from prior assertions only as context.

4. Run targeted search/followups
- Only after document cards are materialized:
  - run follow-up search for explicit unresolved issues, missing references, and weak-confidence relation/conflict checks.
  - open docs from follow-up results are passed directly back to `_deep_read_document` (targeted reads, not global broad scan).

---

### 4) `_deep_read_document` inverted implementation

1. Pre-read state
- Resolve `doc_id` and current `document_inventory` row.
- Compute hash/size and force refresh if changed or stale.
- If unchanged and card exists and ingested, still allow fast path with no reparse unless caller passes `force=True`.

2. Read + extract
- Read file content once.
- Extract structured payload with:
  - `doc_card` (what IS the doc, doc subtype, role-in-matter)
  - `parties` (normalized entities)
  - `key_clauses` (clause text + span)
  - `document_relationships` (`references/supersedes/modifies/contradicts`)
  - standard existing extraction fields (`key_facts`, `facts`, `connections`, `leads`, etc.)

3. Persist intelligence
- Upsert `document_card`.
- Upsert document assertions with current mechanisms (`record_facts_batch`).
- Upsert `document_relation` edges from extracted relationships and from assertion-based contradictions.
- Attach relation evidence using assertion/span ids where available.
- `mark_ingested` only after successful persistence.

4. Reindex hygiene
- On changed doc with prior assertions/relations, call purge/rebuild for that doc-derived layer before upsert (or supersede strategy via validity stamps).
- Refresh `trust override` impact (if overridden trust lowers to ignore/highlight in context).

---

### 5) Cross-document contradiction detection strategy

1. Use existing assertion contradiction miner:
- `AssertionStore.find_contradictions` / `mine_and_mark_contradictions` remain source of truth.

2. Document-level projection:
- Convert each assertion-level contradiction `(assertion_a, assertion_b)` into `document_relation` edge:
  - `source_doc_id = doc(assertion_a)`
  - `target_doc_id = doc(assertion_b)`
  - `relation_type = 'contradicts'`
  - `confidence` from contradiction score/evidence strength.

3. Return format:
- Query API should return grouped contradictions by pair of docs with highest-confidence evidence excerpts and issue IDs.

---

### 6) File change detection and re-indexing order

1. compute hash/size/mtime per file from filesystem.
2. compare with stored `document_inventory.sha256` and `size_bytes`.
3. if changed:
   - mark reindex reason on inventory (or force flag in engine state),
   - purge derived artifacts for that doc,
   - re-read and fully regenerate card + relations + assertions from that file.
4. for unchanged and untouched:
   - no full extraction.
5. for parse failures:
   - create a gap record; retry on next oriented run automatically.

---

### 7) Integration points (required, existing store-first)

- `DocumentInventoryStore`: canonical source of file lifecycle and change detection.
- Assertions store: fact/relationship graph + contradiction mining.
- `TrustOverrideStore`: apply trust multipliers when selecting docs for follow-up and when ranking contradictions.
- `DocumentAnnotationStore`: store analyst annotations against document cards and relation edges when needed.
- `DocumentRelation`: stores explicit document map and contradiction map, no duplicate custom model.

---

### 8) Exact implementation order

1. Extend schema DDL in `src/irys/matter/schema.py` (cards + relationship metadata columns).
2. Add/adjust graph APIs in `src/irys/matter/graph.py`.
3. Add `MatterModel` facade methods in `src/irys/matter/matter.py` and runtime exposures in `src/irys/matter/runtime.py`.
4. Rework `_orient` in `src/irys/rlm/engine.py` to “read-first -> contextual follow-up.”
5. Rework `_deep_read_document` in `src/irys/rlm/engine.py` to produce document card + relationship edges first, then assertions.
6. Add contradiction projection into `document_relation` in the same run.
7. Wire user-intelligence annotation path and expose retrieval in context.
8. Add migration + backward-compatible reads for old rows (all new columns defaulted, no breaking reads).