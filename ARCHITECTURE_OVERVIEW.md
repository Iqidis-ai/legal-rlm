# Irys RLM — System Architecture Overview

> Structured for visual diagram generation. Each section maps to a diagram layer or swimlane.

---

## 1. Entry Points (How Queries Enter the System)

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                             │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  Gradio UI   │  │  FastAPI REST     │  │  Python API       │  │
│  │  (app.py /   │  │  (service/api.py) │  │  (api.py → Irys)  │  │
│  │  chat_app.py)│  │                   │  │                   │  │
│  │  Port 7862   │  │  Port 8000        │  │  Direct import    │  │
│  │              │  │                   │  │                   │  │
│  │ Local: folder│  │ /investigate      │  │ irys.investigate() │  │
│  │ Cloud: upload│  │ /upload/invest    │  │ investigate_sync() │  │
│  │ Real-time    │  │ /search           │  │ quick_search()     │  │
│  │ streaming    │  │ /upload/search    │  │ quick_summarize()  │  │
│  └──────┬───────┘  └────────┬─────────┘  └─────────┬─────────┘  │
│         │                   │                      │            │
│         └───────────────────┼──────────────────────┘            │
│                             ▼                                   │
│                    ┌────────────────┐                            │
│                    │  Irys Class    │                            │
│                    │  (api.py)      │                            │
│                    │                │                            │
│                    │ • Validates    │                            │
│                    │   query/path   │                            │
│                    │ • Inits engine │                            │
│                    │ • Formats      │                            │
│                    │   output       │                            │
│                    └───────┬────────┘                            │
│                            ▼                                    │
│                    ┌────────────────┐                            │
│                    │  RLMEngine     │                            │
│                    │  (engine.py)   │                            │
│                    └────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

**Storage modes (REST/UI):**
- `IRYS_STORAGE_MODE=local` → Folder browser, files on disk
- `IRYS_STORAGE_MODE=s3` → File upload, S3Repository streams to S3, downloads to temp for processing

---

## 2. Model Tier Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GEMINI MODEL TIERS                                 │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  LITE  ─  gemini-2.5-flash-lite                                   │  │
│  │  Role: Workhorse (high volume, low cost)                          │  │
│  │  Cost: $0.10/1M input, $0.40/1M output                           │  │
│  │  Max tokens: 16,384 │ Temp: 0.0                                   │  │
│  │                                                                   │  │
│  │  Tasks:                                                           │  │
│  │   • classify_query_complexity (simple vs complex routing)         │  │
│  │   • pick_files (select relevant files from list)                  │  │
│  │   • extract_facts (document → structured facts)                   │  │
│  │   • is_sufficient (enough evidence?)                              │  │
│  │   • checkpoint (sufficiency + replan combined)                    │  │
│  │   • extract_search_terms (query → search terms)                   │  │
│  │   • analyze_search_results (search hits → relevance)              │  │
│  │   • citation verification                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  FLASH  ─  gemini-3-flash-preview (fallback: gemini-2.5-flash)    │  │
│  │  Role: Intelligent tasks (planning, routing, analysis)            │  │
│  │  Cost: $0.50/1M input, $3.00/1M output                           │  │
│  │  Max tokens: 32,768 │ Temp: 0.0                                   │  │
│  │  Fallback: auto-switches to gemini-2.5-flash on 503 errors        │  │
│  │                                                                   │  │
│  │  Tasks:                                                           │  │
│  │   • create_plan (investigation strategy from file list)           │  │
│  │   • assess_and_plan (unified assessment + planning)               │  │
│  │   • assess_small_repo (direct answer assessment)                  │  │
│  │   • check_if_external_needed (external search decision)           │  │
│  │   • simple query synthesis (FLASH model + PRO system prompt)      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  PRO  ─  gemini-2.5-pro                                           │  │
│  │  Role: Final synthesis (polished legal output)                    │  │
│  │  Cost: $1.25/1M input, $5.00/1M output                           │  │
│  │  Max tokens: 65,536 │ Temp: 0.0                                   │  │
│  │                                                                   │  │
│  │  Tasks:                                                           │  │
│  │   • synthesize (final legal memorandum — complex queries)         │  │
│  │   • System prompt: "Named Partner at elite law firm"              │  │
│  │                                                                   │  │
│  │  Note: Simple queries use FLASH model but PRO system prompt       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Cross-cutting: GeminiClient                                            │
│   • Rate limiter: token-bucket (60 RPM, burst 10)                       │
│   • ResponseCache: LRU (500 entries, 1hr TTL, SHA-256 keyed)            │
│   • Timeout: 120s default, retry up to 3x                               │
│   • UsageStats: per-tier token/cost tracking                             │
│   • Batch: parallel prompts via asyncio.Semaphore                        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Tier Selection Decision Tree:**
```
Query arrives
  │
  ├─ Small repo (< threshold)?
  │   └─ YES → _direct_answer: assess_small_repo [FLASH]
  │               └─ Synthesis: FLASH (simple) or PRO (complex)
  │
  └─ NO → Full investigation
      ├─ Phase 1: assess_and_plan [FLASH] → complexity + plan
      │    └─ Can answer from cached facts? → skip to synthesis
      ├─ Phase 2: _investigate_loop
      │    ├─ extract_facts [LITE] per document (parallel)
      │    ├─ checkpoint [LITE] per iteration
      │    ├─ analyze_search_results [LITE] per search
      │    └─ check_if_external_needed [FLASH] (triggered mid-loop)
      └─ Phase 3: synthesize [PRO or FLASH based on complexity]
```

---

## 3. Investigation Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      INVESTIGATION PHASES                                │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════     │
│  PHASE 0: ROUTING                                                       │
│  ═══════════════════════════════════════════════════════════════════     │
│                                                                         │
│  Query + Repository                                                     │
│       │                                                                 │
│       ├─ Load FactStore (cached facts from prior investigations)        │
│       │   └─ If facts exist → "📚 Loaded N cached facts"               │
│       │                                                                 │
│       ├─ Is small repo? (repo.is_small_repo)                            │
│       │   ├─ YES → _direct_answer (load all docs into context)          │
│       │   └─ NO  → Full RLM investigation                               │
│       │                                                                 │
│  ═══════════════════════════════════════════════════════════════════     │
│  PHASE 1: ASSESSMENT & PLANNING  [FLASH model]                          │
│  ═══════════════════════════════════════════════════════════════════     │
│                                                                         │
│  assess_and_plan() — Single unified LLM call:                           │
│       │                                                                 │
│       ├─ INPUT: query, file list (top 50), cached facts                 │
│       │                                                                 │
│       ├─ OUTPUT:                                                        │
│       │   ├─ can_answer_from_facts: bool (skip to synthesis?)           │
│       │   ├─ complexity: "simple" | "complex"                           │
│       │   ├─ priority_files: [top 3 files to read first]               │
│       │   ├─ search_terms: [top 3 search queries]                      │
│       │   └─ key_issues: [what to investigate]                          │
│       │                                                                 │
│       ├─ If can_answer_from_facts → Jump to Phase 3 (synthesis)         │
│       │                                                                 │
│       └─ Generate initial Leads:                                        │
│           ├─ "Read document: <filepath>" (from priority_files)          │
│           └─ "Search for: <term>" (from search_terms)                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════     │
│  PHASE 2: INVESTIGATION LOOP  [LITE model primarily]                    │
│  ═══════════════════════════════════════════════════════════════════     │
│                                                                         │
│  while iteration < max_iterations (10):                                 │
│       │                                                                 │
│       ├─ 1. GET PENDING LEADS                                           │
│       │   └─ Take up to max_leads_per_level (3) leads                  │
│       │   └─ No leads? → break                                         │
│       │                                                                 │
│       ├─ 2. PROCESS LEADS IN PARALLEL  [asyncio.gather]                 │
│       │   │                                                             │
│       │   ├─ Lead: "Read document: X"                                   │
│       │   │   ├─ Skip if already extracted or irrelevant (cache)        │
│       │   │   ├─ repo.read(filepath) → DocumentContent                  │
│       │   │   ├─ extract_facts [LITE] → facts, entities, citations      │
│       │   │   ├─ Accumulate facts → state.findings["accumulated_facts"] │
│       │   │   ├─ Accumulate entities → state.entities                   │
│       │   │   ├─ Add citations → state.citations                        │
│       │   │   ├─ Store in FactStore (persistent cache)                  │
│       │   │   └─ Generate new leads for deeper reading                  │
│       │   │                                                             │
│       │   └─ Lead: "Search for: Y"                                      │
│       │       ├─ Skip if similar search already done (cache)            │
│       │       ├─ repo.smart_search(term) → SearchResults                │
│       │       │   └─ Exact match first → OR fallback if none            │
│       │       ├─ analyze_search_results [LITE] → relevant hits          │
│       │       ├─ Add citations from relevant hits                       │
│       │       └─ Generate new "Read document" leads                     │
│       │                                                                 │
│       ├─ 3. CHECKPOINT  [LITE]  (every iteration with 2+ facts)        │
│       │   ├─ Input: query + current findings + plan + cached facts      │
│       │   ├─ Output:                                                    │
│       │   │   ├─ sufficient: bool → break if true & docs_read > 0      │
│       │   │   ├─ should_replan: bool → add new leads                   │
│       │   │   └─ new_search_terms, files_to_check                      │
│       │   └─ If 0 docs read but "sufficient" → continue (bug guard)    │
│       │                                                                 │
│       └─ 4. DYNAMIC EXTERNAL SEARCH  [FLASH]  (if enabled)             │
│           ├─ check_if_external_needed → case_law_queries, web_queries   │
│           ├─ Triggered MID-LOOP (not upfront—better specificity)        │
│           ├─ Tiered: can run multiple iterations with new queries       │
│           └─ Sources:                                                   │
│               ├─ CourtListener → case law opinions, dockets             │
│               └─ Tavily → web search (legal-only mode available)        │
│                                                                         │
│  Early termination triggers:                                            │
│   • No more pending leads                                               │
│   • Checkpoint says "sufficient" (with docs_read > 0)                   │
│   • max_iterations reached                                              │
│   • early_exit_facts threshold (5 facts by default)                     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════     │
│  PHASE 3: FINAL SYNTHESIS  [PRO or FLASH model]                         │
│  ═══════════════════════════════════════════════════════════════════     │
│                                                                         │
│  synthesize():                                                          │
│       │                                                                 │
│       ├─ INPUT:                                                         │
│       │   ├─ query                                                      │
│       │   ├─ accumulated evidence (facts, citations)                    │
│       │   ├─ external research (case law + web)                         │
│       │   └─ pinned documents (decisive content)                        │
│       │                                                                 │
│       ├─ TIER SELECTION:                                                │
│       │   ├─ Simple query + use_flash_for_simple → FLASH model          │
│       │   └─ Complex query → PRO model                                  │
│       │   └─ ALWAYS uses PRO system prompt regardless of model          │
│       │                                                                 │
│       └─ OUTPUT: Polished legal memorandum                              │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════     │
│  POST-SYNTHESIS                                                         │
│  ═══════════════════════════════════════════════════════════════════     │
│       ├─ Save new facts to FactStore (persistent)                       │
│       ├─ Save learnings to repository metadata                          │
│       ├─ Format output: Markdown/HTML/JSON/PlainText                    │
│       └─ state.complete() → return InvestigationState                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. InvestigationState — Data Flow Through the System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     InvestigationState (state.py)                        │
│                     "Working memory across recursive calls"              │
│                                                                         │
│  IDENTITY                                                               │
│  ├─ id: str (uuid[:8])                                                  │
│  ├─ query: str                                                          │
│  └─ repository_path: str                                                │
│                                                                         │
│  ACCUMULATED KNOWLEDGE  ←── grows throughout investigation              │
│  ├─ citations: [Citation]        # document, page, text, context,       │
│  │                                 relevance (deduped by text+page)     │
│  ├─ leads: [Lead]                # description, source, investigated,   │
│  │                                 findings (deduped by word overlap)   │
│  ├─ entities: {name → Entity}    # type, sources, mentions, context     │
│  ├─ cross_references: [CrossRef] # source_doc → target_doc              │
│  ├─ timeline: [TimelineEvent]    # date, description, documents, type   │
│  ├─ contradictions: [Contradict] # doc1, doc2, topic, description       │
│  ├─ findings: dict               # accumulated_facts, small_repo_content│
│  │                                 answered_from_cache, initial_plan,   │
│  │                                 had_read_failures                    │
│  └─ hypothesis: str (optional)                                          │
│                                                                         │
│  EXTERNAL RESEARCH TRIGGERS  ←── accumulated from document analysis     │
│  ├─ jurisdictions: set                                                  │
│  ├─ regulations_statutes: set                                           │
│  ├─ legal_doctrines: set                                                │
│  ├─ industry_standards: set                                             │
│  └─ case_references: set                                                │
│                                                                         │
│  PROGRESS TRACKING                                                      │
│  ├─ thinking_steps: [ThinkingStep]  # type, content, timestamp, depth   │
│  │   StepTypes: THINKING, SEARCH, READING, FINDING, REPLAN,             │
│  │              VERIFY, SYNTHESIS, ERROR                                 │
│  ├─ query_classification: dict  # type, complexity, llm_classified      │
│  ├─ documents_read: int                                                 │
│  ├─ searches_performed: int                                             │
│  ├─ recursion_depth / max_depth_reached: int                            │
│  ├─ api_calls / estimated_tokens: int                                   │
│  └─ status: initialized → investigating → completed | failed            │
│                                                                         │
│  SERIALIZATION: to_dict() / from_dict() / save_checkpoint / load        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. System Components Map


```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SYSTEM COMPONENTS                                  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ DOCUMENT LAYER                                                    │  │
│  │                                                                   │  │
│  │  MatterRepository (repository.py)                                 │  │
│  │   • File discovery (PDF, DOCX, TXT, MHT/MHTML)                   │  │
│  │   • is_small_repo: < 100K chars → direct answer path              │  │
│  │   • get_all_content(): load entire repo into context              │  │
│  │   • read(filepath) → DocumentContent (via DocumentReader)         │  │
│  │   • smart_search(term) → SearchResults (via DocumentSearch)       │  │
│  │   • get_stats() → file counts, sizes, types                       │  │
│  │   • add_learning() / get_learnings() → persistent metadata        │  │
│  │   • Filename mapping (display ↔ actual) for S3/upload mode        │  │
│  │                                                                   │  │
│  │  DocumentReader (reader.py)                                       │  │
│  │   • PDF: PyMuPDF (fitz) → page-by-page extraction                │  │
│  │   • DOCX: python-docx → paragraphs + tables                      │  │
│  │   • TXT: direct read with encoding detection                      │  │
│  │   • MHT/MHTML: HTML parsing with BeautifulSoup                   │  │
│  │   • Returns: DocumentContent (pages[], total_chars, full_text)    │  │
│  │   • Page-level granularity for citation tracking                  │  │
│  │                                                                   │  │
│  │  DocumentSearch (search.py)                                       │  │
│  │   • search(): grep-style across files (regex or plain text)       │  │
│  │   • smart_search(): exact phrase first → OR fallback on terms     │  │
│  │   • Thread-safe document cache (max 100 docs)                     │  │
│  │   • Parallel search via ThreadPoolExecutor (max 10 workers)       │  │
│  │   • Returns: SearchResults (hits[], files_searched, total_matches)│  │
│  │   • No scoring — raw matches for LLM to evaluate                 │  │
│  │                                                                   │  │
│  │  DocumentClusterer (clustering.py)                                │  │
│  │   • TF-IDF based document similarity                              │  │
│  │   • Groups related documents into DocumentCluster objects         │  │
│  │   • Auto-names clusters from top keywords                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ LLM LAYER                                                         │  │
│  │                                                                   │  │
│  │  GeminiClient (models.py)                                         │  │
│  │   • Three-tier model dispatch (LITE/FLASH/PRO)                    │  │
│  │   • complete(prompt, tier) → response string                      │  │
│  │   • complete_json(prompt, tier) → parsed JSON                     │  │
│  │   • batch(prompts, tier) → parallel via asyncio.Semaphore         │  │
│  │   • RateLimiter: token-bucket (60 RPM, burst 10)                  │  │
│  │   • ResponseCache integration (optional)                          │  │
│  │   • FLASH fallback: gemini-3-flash-preview → gemini-2.5-flash     │  │
│  │   • UsageStats: per-tier token counting & cost estimation         │  │
│  │   • Retry: up to 3x with exponential backoff                      │  │
│  │                                                                   │  │
│  │  decisions.py (LLM decision functions by tier)                    │  │
│  │   LITE: classify_query_complexity, pick_files, extract_facts,     │  │
│  │         is_sufficient, checkpoint, extract_search_terms,          │  │
│  │         analyze_search_results, citation verification             │  │
│  │   FLASH: create_plan, assess_and_plan, assess_small_repo,        │  │
│  │          check_if_external_needed                                 │  │
│  │   PRO: synthesize (final legal memorandum)                        │  │
│  │                                                                   │  │
│  │  prompts.py (prompt templates organized by tier)                  │  │
│  │   • Each decision function has a corresponding prompt template    │  │
│  │   • PRO synthesis uses "Named Partner at elite law firm" persona  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ ORCHESTRATION LAYER                                               │  │
│  │                                                                   │  │
│  │  RLMEngine (engine.py)                                            │  │
│  │   • investigate(query, repo) → InvestigationState                 │  │
│  │   • Routes: small repo → _direct_answer, large → full loop       │  │
│  │   • _assess_and_create_plan() → initial leads + complexity       │  │
│  │   • _investigate_loop() → iterative lead processing              │  │
│  │   • _synthesize() → final output via PRO/FLASH                   │  │
│  │   • Callbacks: on_step(StepType, content), on_progress(%)        │  │
│  │   • InvestigationCache: track extracted docs, search terms        │  │
│  │                                                                   │  │
│  │  InvestigationState (state.py)                                    │  │
│  │   • Complete working memory across all phases                     │  │
│  │   • Checkpoint serialization (save/load to disk)                  │  │
│  │                                                                   │  │
│  │  FactStore (fact_store.py)                                        │  │
│  │   • JSONL persistence: {repo}/.irys/facts.jsonl                   │  │
│  │   • StoredFact: fact + source + page + quote + category           │  │
│  │   • add_facts_from_extraction() ← integrates with extract_facts  │  │
│  │   • get_relevant(query) → sorted facts for LLM context           │  │
│  │   • format_for_llm() → formatted string (max 15K chars)          │  │
│  │   • Deduplication: normalized text + same source = skip           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ EXTERNAL SEARCH LAYER                                             │  │
│  │                                                                   │  │
│  │  ExternalSearchManager (external_search.py)                       │  │
│  │   • Unified interface for all external APIs                       │  │
│  │   • Manages session lifecycle (aiohttp)                           │  │
│  │                                                                   │  │
│  │  CourtListenerClient                                              │  │
│  │   • Case law search (opinions, dockets)                           │  │
│  │   • REST API v4: courtlistener.com                                │  │
│  │   • Auth: optional API token for higher rate limits               │  │
│  │   • Returns: LegalCase (name, court, date, opinion_text, URL)     │  │
│  │                                                                   │  │
│  │  TavilyClient                                                     │  │
│  │   • AI-native web search (optimized for LLM/RAG)                  │  │
│  │   • URL content extraction                                        │  │
│  │   • Returns: WebSearchResult (title, URL, content, score)         │  │
│  │   • Free tier: 1,000 credits/month                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ CACHING & PERFORMANCE LAYER                                       │  │
│  │                                                                   │  │
│  │  ResponseCache (cache.py)                                         │  │
│  │   • Wraps LRUCache<str> (500 entries, 1hr TTL)                    │  │
│  │   • Key: SHA-256(model + prompt)                                  │  │
│  │   • Integrated into GeminiClient (optional)                       │  │
│  │   • Avoids redundant LLM calls for identical prompts              │  │
│  │                                                                   │  │
│  │  LRUCache<T> (cache.py)                                           │  │
│  │   • Generic in-memory cache with TTL support                      │  │
│  │   • Eviction: least recently used                                 │  │
│  │   • Used by: ResponseCache, DocumentSearch                        │  │
│  │                                                                   │  │
│  │  DiskCache (cache.py)                                             │  │
│  │   • Persistent JSON file cache with size limits                   │  │
│  │   • Key: SHA-256 hash → filename                                  │  │
│  │   • Index file for metadata + TTL tracking                        │  │
│  │   • Auto-eviction at 80% of max_size_mb                           │  │
│  │                                                                   │  │
│  │  InvestigationCache (engine.py)                                   │  │
│  │   • Per-investigation dedup: extracted_docs, searched_terms        │  │
│  │   • Prevents re-reading same documents                            │  │
│  │   • Prevents re-running similar searches                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ OUTPUT LAYER                                                      │  │
│  │                                                                   │  │
│  │  OutputFormatter protocol (formatters.py)                         │  │
│  │   • MarkdownFormatter → # Investigation Report + citations        │  │
│  │   • HTMLFormatter → styled HTML document with sections            │  │
│  │   • JSONFormatter → state.to_dict() serialization                 │  │
│  │   • PlainTextFormatter → ASCII report with separators             │  │
│  │                                                                   │  │
│  │  All formatters receive InvestigationState and extract:           │  │
│  │   • Confidence score, hypothesis, key findings                    │  │
│  │   • Citations with page references                                │  │
│  │   • Entity list, timeline events, contradictions                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Component Relationships & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMPONENT CONNECTIVITY                             │
│                                                                         │
│  Query                                                                  │
│    │                                                                    │
│    ▼                                                                    │
│  [Gradio UI / FastAPI REST / Python API]                                │
│    │                                                                    │
│    ▼                                                                    │
│  Irys (api.py)                                                          │
│    ├─ creates → GeminiClient (models.py)                                │
│    ├─ creates → ResponseCache (cache.py) ─── attached to GeminiClient   │
│    ├─ creates → RLMEngine (engine.py) with RLMConfig                    │
│    └─ creates → OutputFormatter (formatters.py)                         │
│                                                                         │
│  RLMEngine                                                              │
│    ├─ receives → MatterRepository (from Irys)                           │
│    ├─ creates → InvestigationState (state.py)                           │
│    ├─ creates → FactStore (fact_store.py) ─── loads from .irys/         │
│    ├─ creates → ExternalSearchManager (if API keys available)           │
│    ├─ creates → InvestigationCache (per-investigation)                  │
│    │                                                                    │
│    ├─ ROUTING DECISION:                                                 │
│    │   ├─ small repo → _direct_answer()                                 │
│    │   │   ├─ calls → repo.get_all_content()                            │
│    │   │   ├─ calls → decisions.assess_small_repo [FLASH]               │
│    │   │   ├─ optional → ExternalSearchManager.search_*()               │
│    │   │   └─ calls → _synthesize() [PRO or FLASH]                      │
│    │   │                                                                │
│    │   └─ large repo → full investigation                               │
│    │       ├─ calls → decisions.assess_and_plan [FLASH]                 │
│    │       ├─ calls → _investigate_loop()                               │
│    │       │   ├─ calls → repo.read() → DocumentContent                 │
│    │       │   ├─ calls → decisions.extract_facts [LITE]                │
│    │       │   ├─ calls → repo.smart_search() → SearchResults           │
│    │       │   │           └─ DocumentSearch.smart_search()              │
│    │       │   │               └─ DocumentReader.read_file()             │
│    │       │   ├─ calls → decisions.checkpoint [LITE]                   │
│    │       │   ├─ calls → decisions.check_if_external_needed [FLASH]    │
│    │       │   └─ calls → ExternalSearchManager.search_*()              │
│    │       └─ calls → _synthesize() [PRO or FLASH]                      │
│    │                                                                    │
│    └─ POST: saves facts → FactStore → .irys/facts.jsonl                 │
│                                                                         │
│  MatterRepository                                                       │
│    ├─ owns → DocumentReader (reader.py)                                 │
│    ├─ owns → DocumentSearch (search.py)                                 │
│    └─ delegates read/search to these components                         │
│                                                                         │
│  GeminiClient                                                           │
│    ├─ checks → ResponseCache.get(prompt, model) before API call         │
│    ├─ checks → RateLimiter.acquire() before API call                    │
│    ├─ calls → Gemini API (google.genai)                                 │
│    ├─ stores → ResponseCache.set(prompt, model, response) after call    │
│    └─ tracks → UsageStats per tier                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Cross-layer data flow summary:**
```
Query → Irys → RLMEngine → [assess complexity via FLASH]
                                    │
                          ┌─────────┴──────────┐
                          ▼                    ▼
                    Small Repo            Large Repo
                    (< 100K chars)        (≥ 100K chars)
                          │                    │
                    Load all docs        Plan + Investigate Loop
                    Assess [FLASH]       ├─ Read docs [LITE]
                          │              ├─ Search [LITE]
                          │              ├─ Checkpoint [LITE]
                          │              └─ External? [FLASH]
                          │                    │
                          └─────────┬──────────┘
                                    ▼
                              Synthesize
                          ┌─────────┴──────────┐
                          ▼                    ▼
                    Simple → FLASH        Complex → PRO
                    (+ PRO prompt)        (full PRO model)
                          │                    │
                          └─────────┬──────────┘
                                    ▼
                           InvestigationState
                           ├─ final_output (text)
                           ├─ citations []
                           ├─ entities {}
                           ├─ timeline []
                           └─ contradictions []
                                    │
                                    ▼
                             OutputFormatter
                          (MD / HTML / JSON / Text)
```

---

## 7. Configuration Reference

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CONFIGURATION                                      │
│                                                                         │
│  RLMConfig (engine.py) — Engine behavior                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  max_depth:                3     (recursive depth limit)        │    │
│  │  max_iterations:          10     (hard loop limit)              │    │
│  │  max_leads_per_level:      3     (leads processed per cycle)    │    │
│  │  parallel_reads:           3     (concurrent doc reads)         │    │
│  │  max_documents_per_search: 5     (results per search)           │    │
│  │  excerpt_chars_simple:   8000    (excerpt size, simple queries)  │    │
│  │  excerpt_chars_complex: 40000    (excerpt size, complex queries) │    │
│  │  early_exit_facts:         5     (sufficient facts threshold)   │    │
│  │  skip_similar_searches: true     (dedup search terms)           │    │
│  │  use_flash_for_simple:  true     (FLASH for simple synthesis)   │    │
│  │  enable_external_search: true    (CourtListener + Tavily)       │    │
│  │  max_case_law_results:     5     (per query)                    │    │
│  │  max_web_results:          5     (per query)                    │    │
│  │  max_case_law_queries:     5     (total queries)                │    │
│  │  max_web_queries:          5     (total queries)                │    │
│  │  parallel_external: true         (parallel external searches)   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  GeminiClient defaults (models.py)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  timeout:              120s                                      │    │
│  │  requests_per_minute:   60   (rate limiter)                      │    │
│  │  burst_size:            10   (token bucket burst)                │    │
│  │  max_retries:            3   (with exponential backoff)          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Environment variables                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  GEMINI_API_KEY           (required) Gemini API key              │    │
│  │  COURTLISTENER_API_TOKEN  (optional) CourtListener API token     │    │
│  │  TAVILY_API_KEY           (optional) Tavily API key              │    │
│  │  IRYS_STORAGE_MODE        local | s3 (default: local)            │    │
│  │  AWS_S3_BUCKET            S3 bucket name (for s3 mode)           │    │
│  │  AWS_ACCESS_KEY_ID        AWS credentials (for s3 mode)          │    │
│  │  AWS_SECRET_ACCESS_KEY    AWS credentials (for s3 mode)          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```