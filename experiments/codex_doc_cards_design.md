Here’s the grounded assessment from the current code.

## 1) How to search/read legal documents with modern agentic behavior

Current loop is mostly “full-scan by lead, then read matching docs,” which is costly:

- Lead execution is centralized in ` _investigate_lead` and always goes through repository search as a broad retrieval step: `[src/irys/rlm/engine.py:1926](src/irys/rlm/engine.py:1926)` → `[src/irys/rlm/engine.py:1967](src/irys/rlm/engine.py:1967)`.
- `core/search.py` is line-based regex matching over raw text (`search` + `_search_file`), with no persisted section memory or candidate filtering: `[src/irys/core/search.py:219](src/irys/core/search.py:219)` and `[src/irys/core/search.py:295](src/irys/core/search.py:295)`.
- `core/repository.py` rebuilds file lists per search call, so repeated lead scans repeatedly touch the repo index: `[src/irys/core/repository.py:239](src/irys/core/repository.py:239)`.

A better pattern for this codebase is an iterative frontier pipeline:

1. Build a short “investigation frontier” from unresolved/weak issues and contradiction candidates, not a fixed linear lead order.
2. Run staged retrieval:
   - Stage A: issue-targeted terms enriched by issue context (`_enrich_search_term_with_issue_context`) for high-confidence docs only.
   - Stage B: expanded terms only if frontier remains under-answered.
3. Read only high-value spans/sections first, not entire files.
4. Use negative evidence immediately (no-hit leads and low-yield docs should suppress similar future leads).

You already have scaffolding for part of this:
- `build_query_context()` gives issue state/context and recent assertions: `[src/irys/matter/matter.py:896](src/irys/matter/matter.py:896)`.
- Issue-focused query enrichment exists: `[src/irys/rlm/engine.py:3667](src/irys/rlm/engine.py:3667)` and `[src/irys/rlm/engine.py:3609](src/irys/rlm/engine.py:3609)`.
- But frontier/backtracking behavior is not yet enforcing a strict candidate-first strategy end-to-end.

## 2) How persistent memory changes search/retrieval

Persistent memory is already in place structurally; it is underused in retrieval decisions:

- Inventory dedupe/versioning gives “what’s already known” signals: `[src/irys/matter/graph.py:2726](src/irys/matter/graph.py:2726)` and `[src/irys/matter/graph.py:2788](src/irys/matter/graph.py:2788)`.
- Inventory schema has long-tail fields (salience/read state/version/family) suitable for retrieval bias: `[src/irys/matter/schema.py:191](src/irys/matter/schema.py:191)`.
- `document_card` and `span` tables already exist but are not wired into document selection/read routing yet: `[src/irys/matter/schema.py:237](src/irys/matter/schema.py:237)` and `[src/irys/matter/schema.py:268](src/irys/matter/schema.py:268)`.

This matters because with memory you can stop scanning repeatedly:
- Use per-document salience/read count to prioritize unseen or historically useful docs.
- Use version history + family fields to avoid re-reading superseded copies.
- Use spans to re-open only untouched/uncertain sections instead of full files.
- Use relationship signals (`document_relation`) to route from a document that mentions another as likely relevant: `[src/irys/matter/schema.py:220](src/irys/matter/schema.py:220)`.

Also, feedback signal exists but is not wired into lead ordering yet:
- term boosting/demotion + apply_feedback exists in state, but not in lead selection path: `[src/irys/rlm/state.py:1416](src/irys/rlm/state.py:1416)`.

## 3) Specific high-impact changes (with line anchors, ordered by impact)

1) Make leads candidate-first instead of full-repo-first  
   - Add file-list candidate filters before search; derive candidates from issue frontier, unresolved assertions, recent misses, and document salience.
   - Entry point: `[src/irys/rlm/engine.py:1926](src/irys/rlm/engine.py:1926)`, `[src/irys/rlm/engine.py:1967](src/irys/rlm/engine.py:1967)`.
   - Repository side currently only supports broad search: `[src/irys/core/repository.py:239](src/irys/core/repository.py:239)`.

2) Use staged search expansion with existing query expansion function  
   - `expand_query` exists in `core/search.py` but is currently not part of the lead path.
   - Wire it into `_investigate_lead` as optional stage-2 only after weak/no-hit leads.
   - Anchors: `[src/irys/core/search.py:75](src/irys/core/search.py:75)` and `[src/irys/rlm/engine.py:1962](src/irys/rlm/engine.py:1962)`.

3) Persist and reuse section-level reads  
   - Use `span` rows for read coverage + re-read avoidance.
   - Add persistence layer methods around document inventory graph and call them from `_deep_read_document` / `_batch_deep_read`.
   - Read layer code currently deep-reads without section memory: `[src/irys/rlm/engine.py:2507](src/irys/rlm/engine.py:2507)` and `[src/irys/rlm/engine.py:2540](src/irys/rlm/engine.py:2540)`.
   - Schema support already exists: `[src/irys/matter/schema.py:268](src/irys/matter/schema.py:268)`.

4) Close-loop feedback into lead selection now  
   - Apply term boost/demotion from `state` to `_investigate_loop` lead ordering before calling `_investigate_lead`.
   - State feedback hooks exist but not applied: `[src/irys/rlm/state.py:1416](src/rlm/state.py:1416)`; loop orchestration is `[src/irys/rlm/engine.py:1620](src/irys/rlm/engine.py:1620)` and lead scheduling around `[src/irys/rlm/engine.py:1710](src/irys/rlm/engine.py:1710)`.

5) Add document cards as retrieval memory artifacts  
   - `DocumentInventoryStore` is robust, but no card accessors are implemented though table exists.
   - Add store methods and write cards during deep-read after role/title/issue-type inference.
   - Current schema has card fields: `[src/irys/matter/schema.py:237](src/irys/matter/schema.py:237)`; graph methods currently around inventory version/link only: `[src/irys/matter/graph.py:2726](src/irys/matter/graph.py:2726)` onward.

6) Fix long-lived in-progress dedupe leak in deep read  
   - `_reading_in_progress` is discarded in one error path but not guaranteed on success paths, which can undercut anti-redundancy in the same run.
   - Add `finally` cleanup around deep-read transaction.
   - Touch points: set/add around `[src/irys/rlm/engine.py:2569](src/irys/rlm/engine.py:2569)` and cleanup currently seen in exception branch `[src/irys/rlm/engine.py:3020](src/irys/rlm/engine.py:3020)`.

If useful, I can produce a concrete patch plan with exact method signatures and the minimal edits per file in the order above.