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

## D-007: Single-Provider Model Architecture (NANO via routing, not new vendor)

**Date:** 2026-04-03
**Status:** Active — extends D-001

**Decision:** Irys RLM will remain single-provider for production matter-content traffic.
NANO is a routing budget on `gemini-2.5-flash-lite` (shorter prompts, narrower output),
not a second vendor. The system maximizes ROI through Google Batch API for cold-path
NANO/LITE ingestion and selective Gemini context caching of hot reusable prefixes
(system instructions, matter summaries, issue context).

**Rationale:** At 1,000 queries/day, adding a second NANO provider (Together AI, Groq)
changes total-system cost by at most $3–5/month while materially increasing routing
complexity, testing surface, failure modes, and data-governance burden for tasks that feed
the durable matter model (SO-1, SO-2, SO-5, SO-7). The 90% Gemini cache-read discount
erases the second-provider advantage with as few as 3,600 cached tokens per NANO call.
Data sovereignty: raw matter content must stay on Google (paid API, no training use by
default); Together requires explicit ZDR configuration; Groq adds a second processor.

**Implementation plan:**
1. Add `ModelTier.NANO` as a routing budget concept on `gemini-2.5-flash-lite`
   (max_output_tokens=2048, for triage/classification calls)
2. Add `use_batch=True` flag to `GeminiClient.complete()` for cold-path ingestion
3. Add `cache_key` param for reusable prefix caching via Gemini context caching API
4. Route engine triage tasks (doc type, actor spotting) to NANO tier

**Revisit if:** Vertex AI migration is complete and a second provider can prove materially
lower total cost without handling raw matter text.

**Source:** Codex Design Gate review 2026-04-03 (codex_design_gate_model_tiers.txt).

---

## D-007: Scaling Architecture — Deferred Items from Tier 2 Review (r50)

**Date:** 2026-04-06
**Status:** Active / Deferred

**Context:** Tier 2 Scaling review (codex_tier2_r50_scaling.md) identified three architectural
scaling risks beyond the current single-matter dev scope:

1. **O(corpus) rehydration** — `_try_rehydrate_matter_model` linearly scans all corpus
   directories on every cold miss. Fix: persist a `matter_registry` table mapping `matter_id`
   to `db_path` so rehydration is O(1).

2. **Multiple live MatterModel instances** — `_wire_matter_model` always opens a fresh model
   and overwrites the registry. Post-eviction rehydration can create a second instance,
   splitting coordination state (flush lock, pending queues, run snapshots). Fix: enforce
   one live in-process owner per matter_id with ref-counting or an acquire-or-wait pattern.

3. **Background flush thread count unbound** — with 100+ active matters and frequent
   truncation, per-model non-daemon flush threads accumulate with no service-wide cap.
   Fix: use a shared bounded thread pool for all flush work.

**Decision:** Defer items 1–3 until multi-matter production use is needed. Current use is
single-matter / small concurrent job count where these risks are not material. Log here for
visibility and to prevent re-discovery.

**Source:** codex_tier2_r50_scaling.md, Tier 2 review 2026-04-06.

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
