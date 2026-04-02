# Decision Log

Decisions are recorded here permanently. When a decision is superseded, mark it SUPERSEDED
and link to the new decision. Never delete entries.

---

## D-001: Use Google Gemini as Primary Model Provider

**Date:** Pre-2026-04-02 (during initial roundtable cycles)
**Status:** Active

**Decision:** Use Google Gemini with a 3-tier strategy: LITE (gemini-2.5-flash-lite) for
bulk reading, FLASH (gemini-2.5-flash) for analysis and routing, PRO (gemini-2.5-pro) for
final synthesis.

**Rationale:** Gemini's large context window is well-suited to legal document ingestion.
The tiering strategy balances cost and quality — cheap models handle volume, expensive models
handle synthesis.

**Trade-offs:** Locks to Google's API. ANTHROPIC_API_KEY is kept as optional for comparison
testing.

**Revisit if:** Gemini pricing changes materially, context window requirements exceed model
limits, or a better provider emerges for legal document tasks.

---

## D-002: Python Package Architecture with Async API

**Date:** Pre-2026-04-02
**Status:** Active

**Decision:** Structure Irys as an installable Python package (pyproject.toml) with an async
API as the canonical interface and sync wrappers for convenience.

**Rationale:** Async is necessary for parallel document processing and concurrent lead
investigation. Package structure enables clean imports and versioned distribution.

**Trade-offs:** Async complexity. Sync wrappers add some overhead.

---

## D-003: Recursive Investigation Pattern (Orient → Investigate Loop → Synthesize)

**Date:** Pre-2026-04-02
**Status:** Active — will be EXTENDED (not replaced) by matter model substrate

**Decision:** The core investigation uses a recursive pattern: Phase 1 orient (plan), Phase 2
iterative investigation with parallel lead processing, Phase 2.5 citation verification,
Phase 3 synthesis.

**Rationale:** Effective for complex multi-hop legal questions where initial search results
reveal follow-on leads.

**Trade-offs:** Expensive for simple queries. Current implementation is ephemeral — knowledge
from each run is lost.

**Planned extension (Phase 5):** The investigation loop will be modified to read from and
write to the persistent matter model, transforming it from a one-shot pipeline into an
incremental model update process.

---

## D-004: SQLite as Persistent Store for Matter Model

**Date:** 2026-04-02
**Status:** Active — confirmed by Codex design gate 2026-04-02 (session 019d4f71)

**Decision (proposed):** Use SQLite as the persistence layer for all canonical stores
(matter registry, document cards, actor store, assertion graph, issue model, etc.).

**Rationale:** Zero infrastructure overhead, file-portable, supports complex queries,
works in both local and embedded modes. For a single-matter single-user system, SQLite
is sufficient. Can be swapped for PostgreSQL later if multi-user or multi-matter at scale
is required.

**Alternatives considered:**
- JSON files: Too slow for graph traversal at scale, no ACID
- PostgreSQL: Requires running server, overkill for current scale
- DuckDB: Good for analytics but less mature for OLTP workloads
- Neo4j: Great for graph queries but heavy infrastructure for a legal tool

**Revisit if:** Multi-tenant requirements emerge or assertion graph queries require
graph-native storage.

---

## D-005: Matter Model First, UI Second

**Date:** 2026-04-02
**Status:** Active

**Decision:** All development priority goes to the intelligence substrate (persistent matter
model, typed assertion graph, issue model) before any UI or presentation improvements.

**Rationale:** Per IDEAL_PRODUCT_SPEC.md §36, Priority 0 is the intelligence substrate.
Visual polish before structured intelligence creates attractive but brittle product behavior.
The Gradio UI exists and is sufficient for current testing.

**Non-goals until substrate is complete:** charting, timelines rendered from prose, visual
improvements, advanced presentation surfaces.

---

## D-006: Swarm Build as Governance Framework

**Date:** 2026-04-02
**Status:** Active

**Decision:** All autonomous engineering on this project operates under Swarm Build governance.
Codex is architectural authority. Every non-trivial implementation goes through a Design Gate.
Every meaningful block of work goes through a PR Gate. All experiments are logged.

**Rationale:** Legal intelligence at this complexity requires disciplined governance. Without
it, autonomous work will drift toward whatever is easiest to code, which is consistently not
what the product needs.
