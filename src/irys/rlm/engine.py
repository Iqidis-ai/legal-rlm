"""RLM Engine - Recursive Language Model investigation engine.

This is the core of the system. It implements:
1. Iterative refinement with data-driven replanning
2. Recursive investigation of leads
3. Parallel document processing
4. Tiered model usage (Lite -> Flash -> Pro)
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any, AsyncIterator
from pathlib import Path
import asyncio
import json
import logging

from ..core.models import GeminiClient, ModelTier
from ..core.repository import MatterRepository
from ..core.search import SearchResults
from .state import InvestigationState, StepType, ThinkingStep, Citation, Lead, classify_query

logger = logging.getLogger(__name__)


@dataclass
class RLMConfig:
    """Configuration for RLM engine."""
    max_depth: int = 5
    max_leads_per_level: int = 5
    max_documents_per_search: int = 10
    min_lead_priority: float = 0.3
    excerpt_chars: int = 8000
    parallel_reads: int = 5
    checkpoint_dir: Optional[str] = None  # Directory for checkpoints
    checkpoint_interval: int = 5  # Save checkpoint every N iterations
    adaptive_depth: bool = True  # Adjust depth based on complexity
    min_depth: int = 2  # Minimum depth even for simple queries
    depth_citation_threshold: int = 15  # Stop early if enough citations
    max_iterations: int = 20  # Maximum investigation loop iterations
    enable_matter_model: bool = False  # When True, persist facts to SQLite matter model


# System prompts for different stages
ORIENTATION_PROMPT = """You are an expert legal analyst conducting due diligence on a document repository.

Repository Structure:
{structure}

Total files: {total_files}

User Query: {query}
{matter_context}
Your task is to create a strategic research plan. Think like an experienced litigator or investigator.

Consider:
1. What are the CORE legal issues that need to be established?
2. Which document types are MOST LIKELY to contain direct evidence? (e.g., contracts for terms, emails for intent, financials for damages)
3. What SPECIFIC search terms will find relevant passages? Include legal terms, party names, key dates, and transaction-specific language.
4. What is your preliminary hypothesis based on the query structure?

PRIORITIZE:
- Primary source documents (contracts, pleadings) over secondary (correspondence)
- Documents with dates matching key events
- Files mentioning specific parties or amounts

Respond in JSON format:
{{
    "issues": ["issue1", "issue2", ...],
    "relevant_folders": ["folder1", "folder2", ...],
    "initial_searches": ["term1", "term2", ...],
    "search_rationale": "Why these search terms will find relevant evidence",
    "document_priority": ["most important doc type", "second most important", ...],
    "hypothesis": "Your initial hypothesis based on query analysis"
}}
"""


def _format_matter_context(ctx) -> str:
    """Format a QueryMatterContext into a prompt-injectable string.

    Returns empty string when ctx is None (null adapter path).
    """
    if ctx is None:
        return ""
    lines = ["\nExisting Matter Intelligence (read from durable store):"]
    if ctx.existing_assertion_count:
        lines.append(f"- Known facts already recorded: {ctx.existing_assertion_count}")
    if ctx.open_issues:
        issue_titles = [i.get("title", "") for i in ctx.open_issues[:5]]
        lines.append(f"- Open legal issues: {', '.join(t for t in issue_titles if t)}")
    if ctx.weakest_issue_id:
        weakest_titles = [i.get("title", "") for i in ctx.open_issues
                          if i.get("id") == ctx.weakest_issue_id]
        if weakest_titles:
            lines.append(
                f"- PRIORITY FOCUS: Issue with least evidence — '{weakest_titles[0]}'. "
                "Generate search leads that specifically target this issue."
            )
    if ctx.open_gaps:
        lines.append(f"- Known gaps / missing information ({len(ctx.open_gaps)} total):")
        for gap in ctx.open_gaps[:3]:
            desc = gap.get("description", "")[:100]
            lines.append(f"  * {desc}")
    if ctx.known_actors:
        lines.append(f"- Key parties already identified: {', '.join(ctx.known_actors[:8])}")
    if ctx.known_document_ids:
        lines.append(f"- Documents already analyzed ({len(ctx.known_document_ids)} total): "
                     + ", ".join(ctx.known_document_ids[:5]))
    if ctx.answered_clarifications:
        lines.append(f"- User-supplied context ({len(ctx.answered_clarifications)} answers):")
        for cl in ctx.answered_clarifications[:3]:
            q = (cl.get("question_text") or "")[:80]
            a = (cl.get("answer_text") or "")[:120]
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")
    lines.append("")
    return "\n".join(lines)

ANALYZE_FINDINGS_PROMPT = """You are a senior legal analyst extracting evidence from search results.

Query: {query}
Current Hypothesis: {hypothesis}

Search Results for "{search_term}":
{search_results}

ANALYZE THESE RESULTS CAREFULLY:

1. KEY FACTS: Extract ONLY the 10 most important specific facts (STRICT LIMIT: 10 maximum):
   - Format each fact as: {"fact": "...", "source_file": "filename_if_determinable"}
   - source_file should be the filename from the search results where the fact appears
   - Directly relevant to the query
   - Supported by the document text
   - Include dates, amounts, party names where found
   - Keep each fact text under 100 characters

2. NEW LEADS: Identify specific avenues to investigate:
   - Referenced documents that should be examined
   - Named individuals who should be researched
   - Dates/events mentioned that need context
   - Cross-references to other documents
   - Potential contradictions to verify

3. HYPOTHESIS EVALUATION:
   - Does this evidence SUPPORT or CONTRADICT our hypothesis?
   - What gaps remain in our understanding?

4. NEXT SEARCHES: Suggest terms that will:
   - Corroborate findings from multiple sources
   - Fill gaps in the evidence
   - Find contradictory evidence (for completeness)

Respond in COMPACT JSON (keep under 3000 chars):
{{
    "key_facts": [{{"fact": "fact text", "source_file": "filename.pdf"}}, ...],
    "new_leads": [{{"desc": "...", "priority": 0.8}}],
    "hypothesis_update": "string or null",
    "next_searches": ["term1", "term2"]
}}
"""

DEEP_READ_PROMPT = """You are an expert legal analyst performing detailed document review.

Document: {filename}
Page Range: {page_range}

Content:
{content}

Query Context: {query}
Current Investigation Focus: {focus}

CONDUCT A FOCUSED LEGAL ANALYSIS. IMPORTANT: Keep response under 4000 characters total.

1. KEY FACTS (STRICT LIMIT: 15 maximum facts): Extract facts that are:
   - Directly relevant to the query/focus
   - Specific (include dates, amounts, names)
   - Keep each fact under 100 characters

2. CRITICAL QUOTES (STRICT LIMIT: 3 maximum): Identify the most important passages:
   - Direct admissions or acknowledgments
   - Terms that define obligations or rights
   - Statements of fact that support/contradict claims
   - Language that creates legal obligations

3. ENTITIES: Extract with role/context:
   - People: name, role, significance
   - Companies: name, relationship to parties
   - Dates: date, what happened, significance
   - Amounts: value, context, what it represents

4. NUMERIC FACTS (SO-6 — extract ALL monetary amounts, dates, rates, counts):
   For each number, provide a structured object:
   - kind: "amount" | "date" | "rate" | "balance" | "count"
   - subject: one-word subject type — "invoice" | "payment" | "fee" | "damages" | "balance" | "rate" | "deposit" | "penalty" | "other"
   - raw: exact text from document
   - value: numeric value if parseable (null otherwise)
   - currency: "USD" etc. for amounts (null if not monetary)
   - context: brief label of what this number represents (max 60 chars)

5. DOCUMENT RELATIONSHIPS:
   - References to other documents (attachments, exhibits)
   - Prior agreements or communications mentioned
   - Events that require corroboration elsewhere

6. FACT RELATIONSHIPS (SO-2 — up to 5 most important): Identify logical relationships
   BETWEEN the key_facts you listed above, using their 0-based indices.
   Relation types: "supports" (A reinforces B), "attacks" (A undermines B),
   "contradicts" (A directly conflicts with B), "corroborates" (A independently confirms B),
   "supersedes" (A replaces B as the authoritative statement).

7. RED FLAGS & CONCERNS:
   - Ambiguous or potentially misleading language
   - Missing expected provisions
   - Contradictions within the document
   - Issues requiring legal interpretation

Respond in COMPACT JSON (STRICT: under 4000 chars total):
{{
    "key_facts": [{{"fact": "...", "page": N}}],
    "quotes": [{{"text": "...", "page": N}}],
    "entities": {{"people": ["name1"], "dates": ["date1"], "amounts": ["$X"], "companies": ["co1"]}},
    "numeric_facts": [{{"kind": "amount", "subject": "invoice", "raw": "$50,000", "value": 50000, "currency": "USD", "context": "payment due"}}],
    "fact_relationships": [{{"from_idx": 0, "to_idx": 2, "relation": "supports"}}],
    "connections": ["doc reference 1"],
    "concerns": ["issue 1"]
}}
"""

SYNTHESIS_PROMPT = """You are a senior partner at a law firm drafting a legal memorandum.

Original Query: {query}

Investigation Summary:
- Documents analyzed: {docs_analyzed}
- Searches performed: {searches}
- Citations collected: {citation_count}
- Maximum investigation depth: {max_depth}

Working Hypothesis: {hypothesis}

Source Calibration (CRITICAL — read before analyzing facts):
{source_calibration}

Quantitative Summary (SO-6 — extracted monetary amounts):
{quant_summary}

Key Entities Identified:
{entities}

Evidence Gathered:
{findings}

Documentary Citations:
{citations}

PREPARE A COMPREHENSIVE LEGAL MEMORANDUM:

## Executive Summary
Provide a 2-3 sentence direct answer to the query. Lead with the conclusion.

## Factual Background
Chronological narrative of relevant events established by the evidence.
Cite sources: [Document Name, p. X]

## Analysis

### Key Findings
- Finding 1 with citation [Source]
- Finding 2 with citation [Source]
(Prioritize VERIFIED citations)

### Supporting Evidence
Detail the strongest evidence supporting conclusions.

### Contradictions or Concerns
Note any conflicting evidence or unresolved issues.

### Evidence Strength Assessment
Rate overall evidence as: Strong / Moderate / Weak
Explain basis for rating.

## Entities & Relationships
Key parties and their roles established by evidence.

## Gaps & Limitations
- What evidence was NOT found
- Areas needing further investigation
- Limitations of available documents

## Recommendations
1. Immediate actions based on findings
2. Additional investigation needed
3. Risk mitigation steps

---
Write in formal legal memorandum style. Be precise and cite everything.
Mark unverified citations with [UNVERIFIED].
Do not speculate beyond what evidence supports.
"""

SUMMARIZE_DOCUMENT_PROMPT = """You are a legal analyst creating a comprehensive document summary for litigation support.

Document: {filename}
Detected Type: {doc_type}

Content:
{content}

CREATE A LITIGATION-READY SUMMARY:

1. DOCUMENT CLASSIFICATION:
   - Type: contract, pleading, correspondence, discovery, financial, corporate, other
   - Purpose: What is this document meant to accomplish?
   - Significance: Why would this matter in litigation?

2. PARTIES & SIGNATORIES:
   - All named parties and their roles
   - Who signed/authored this document?
   - Third parties mentioned

3. KEY DATES:
   - Document date
   - Effective dates
   - Deadlines mentioned
   - Events referenced

4. SUBSTANTIVE CONTENT:
   For Contracts: key terms, obligations, conditions, termination clauses
   For Pleadings: claims, defenses, relief sought
   For Correspondence: subject matter, requests made, commitments
   For Financial: amounts, accounts, transactions

5. MONETARY AMOUNTS:
   - All dollar figures with context
   - Payment terms
   - Damages claimed

6. RED FLAGS:
   - Ambiguous provisions
   - Unusual terms
   - Potential liability issues
   - Missing expected content

Respond in JSON format:
{{
    "summary": "2-3 sentence executive summary",
    "document_type": "contract/pleading/correspondence/discovery/financial/corporate/other",
    "document_date": "YYYY-MM-DD or null",
    "parties": [{{"name": "...", "role": "..."}}],
    "signatories": ["name1", "name2"],
    "key_dates": [{{"date": "...", "event": "...", "significance": "..."}}],
    "key_terms": [{{"term": "...", "page": N, "significance": "..."}}],
    "amounts": [{{"value": "...", "context": "...", "page": N}}],
    "concerns": [{{"issue": "...", "severity": "high/medium/low"}}],
    "cross_references": ["documents mentioned or referenced"]
}}
"""

SUMMARIZE_COLLECTION_PROMPT = """You are a senior litigation analyst creating a matter overview from a document collection.

Documents Being Summarized:
{document_list}

Individual Summaries:
{summaries}

CREATE A COMPREHENSIVE MATTER ANALYSIS:

1. MATTER OVERVIEW:
   - What is this case/transaction about?
   - Key dispute or purpose
   - Current status/stage

2. PARTY ANALYSIS:
   - All parties and their roles
   - Relationships between parties
   - Key individuals and their significance

3. CHRONOLOGY:
   - Construct a timeline of events
   - Identify cause-and-effect relationships
   - Note date gaps or inconsistencies

4. DOCUMENT ECOSYSTEM:
   - How do these documents relate to each other?
   - Which documents reference others?
   - What is the chain of custody/communication?

5. KEY THEMES:
   - Major legal issues present
   - Recurring topics across documents
   - Points of agreement and dispute

6. EVIDENTIARY ASSESSMENT:
   - What is well-documented vs. poorly documented?
   - Strength of documentary evidence
   - Critical missing documents

7. STRATEGIC OBSERVATIONS:
   - Potential strengths
   - Potential weaknesses
   - Areas requiring immediate attention

Respond in JSON format:
{{
    "collection_summary": "3-5 sentence executive overview",
    "matter_type": "contract dispute/tort/corporate/regulatory/other",
    "parties": [{{"name": "...", "role": "...", "key_documents": ["doc1"]}}],
    "timeline": [{{"date": "...", "event": "...", "source": "...", "significance": "..."}}],
    "themes": [{{"theme": "...", "relevant_docs": ["doc1", "doc2"], "assessment": "..."}}],
    "document_relationships": [{{"from": "doc1", "to": "doc2", "relationship": "references/amends/responds_to/contradicts"}}],
    "evidence_strength": {{
        "well_documented": ["topic1", "topic2"],
        "poorly_documented": ["topic3"],
        "missing": ["expected document type"]
    }},
    "strategic_notes": ["observation1", "observation2"]
}}
"""

# Additional specialized prompts for enhanced analysis

ENTITY_EXTRACTION_PROMPT = """You are a legal analyst extracting entities from document text.

Document: {filename}
Text Excerpt:
{text}

Extract ALL entities with their context and significance:

1. PEOPLE:
   - Names (formal and informal references)
   - Titles/Roles
   - Affiliations
   - Actions attributed to them

2. ORGANIZATIONS:
   - Company names (including d/b/a and subsidiaries)
   - Government agencies
   - Law firms
   - Other entities

3. DATES & TIMEFRAMES:
   - Specific dates
   - Date ranges
   - Relative timeframes ("30 days after...")

4. MONETARY VALUES:
   - Dollar amounts
   - Percentages
   - Financial metrics

5. LOCATIONS:
   - Addresses
   - Jurisdictions
   - Venues

6. LEGAL TERMS:
   - Case citations
   - Statute references
   - Defined terms from agreements

Respond in JSON format:
{{
    "people": [{{"name": "...", "role": "...", "context": "...", "mentions": N}}],
    "organizations": [{{"name": "...", "type": "...", "relationship": "..."}}],
    "dates": [{{"date": "...", "context": "...", "type": "specific/deadline/effective"}}],
    "amounts": [{{"value": "...", "context": "...", "type": "payment/damages/fee"}}],
    "locations": [{{"place": "...", "type": "address/jurisdiction/venue"}}],
    "legal_refs": [{{"citation": "...", "type": "case/statute/contract_term"}}]
}}
"""

CONTRADICTION_DETECTION_PROMPT = """You are a legal analyst identifying contradictions and inconsistencies.

Document 1: {doc1_name}
Statement: "{statement1}"
Context: {context1}

Document 2: {doc2_name}
Statement: "{statement2}"
Context: {context2}

ANALYZE FOR CONTRADICTIONS:

1. Are these statements contradictory? Consider:
   - Direct factual contradictions
   - Inconsistent timelines
   - Conflicting obligations
   - Different characterizations of same event

2. Severity Assessment:
   - HIGH: Material contradiction affecting core claims
   - MEDIUM: Significant inconsistency requiring explanation
   - LOW: Minor discrepancy, possibly reconcilable

3. Possible Explanations:
   - Could both statements be true in context?
   - Is this a drafting error vs. substantive conflict?
   - Does timing explain the difference?

Respond in JSON format:
{{
    "is_contradiction": true/false,
    "contradiction_type": "factual/temporal/characterization/obligation/none",
    "severity": "high/medium/low/none",
    "explanation": "Why these contradict or don't",
    "reconciliation_possible": true/false,
    "reconciliation_theory": "How these could both be true (if applicable)",
    "legal_significance": "Why this matters for the matter",
    "follow_up_needed": ["additional verification steps"]
}}
"""

TIMELINE_EXTRACTION_PROMPT = """You are a legal analyst constructing a chronology from documents.

Documents Analyzed:
{document_list}

Events Found:
{events}

CONSTRUCT A LEGAL CHRONOLOGY:

1. Order events by date (earliest to latest)
2. Identify causal relationships between events
3. Note gaps in the timeline
4. Flag conflicting dates for the same event
5. Highlight deadline-critical events

For each event, assess:
- Certainty of date (exact vs. approximate)
- Source reliability
- Legal significance
- Relationship to other events

Respond in JSON format:
{{
    "chronology": [
        {{
            "date": "YYYY-MM-DD",
            "date_certainty": "exact/approximate/inferred",
            "event": "description",
            "source_doc": "filename",
            "source_page": N,
            "legal_significance": "why this matters",
            "related_events": ["event_ids that connect"],
            "is_deadline": true/false
        }}
    ],
    "timeline_gaps": [
        {{"period": "from - to", "what_might_be_missing": "..."}}
    ],
    "date_conflicts": [
        {{"event": "...", "date1": "...", "source1": "...", "date2": "...", "source2": "..."}}
    ],
    "key_periods": [
        {{"period": "from - to", "description": "what happened", "significance": "..."}}
    ]
}}
"""

EVIDENCE_ASSESSMENT_PROMPT = """You are a senior litigator assessing the strength of evidence.

Claim Being Assessed: {claim}

Supporting Evidence:
{supporting_evidence}

Contradicting Evidence:
{contradicting_evidence}

ASSESS EVIDENCE STRENGTH:

1. DIRECT vs. CIRCUMSTANTIAL
   - What evidence directly proves the claim?
   - What is circumstantial?

2. PRIMARY vs. SECONDARY SOURCES
   - Contracts, signed documents = primary
   - Emails, notes = secondary
   - Testimony, recollections = tertiary

3. CORROBORATION
   - Is evidence corroborated by multiple sources?
   - Any single-source critical facts?

4. AUTHENTICATION POTENTIAL
   - Can this evidence be authenticated?
   - Who would authenticate it?

5. HEARSAY ISSUES
   - What statements are hearsay?
   - Any exceptions applicable?

6. OVERALL ASSESSMENT
   - Rate claim as: Strongly Supported / Moderately Supported / Weakly Supported / Contradicted

Respond in JSON format:
{{
    "claim": "{claim}",
    "evidence_classification": {{
        "direct": ["evidence1", "evidence2"],
        "circumstantial": ["evidence3"],
        "primary_sources": ["doc1"],
        "secondary_sources": ["doc2"],
        "hearsay_concerns": ["statement1"]
    }},
    "corroboration_level": "high/medium/low/none",
    "authentication_assessment": "easily authenticated/challengeable/problematic",
    "overall_strength": "strong/moderate/weak/contradicted",
    "strength_score": 0-100,
    "reasoning": "detailed explanation",
    "vulnerabilities": ["weakness1", "weakness2"],
    "strengthening_opportunities": ["what additional evidence would help"]
}}
"""


class RLMEngine:
    """
    Recursive Language Model investigation engine.

    Implements the adaptive research loop:
    1. Orient - understand repository and form hypothesis
    2. Search - find relevant documents
    3. Analyze - extract findings and leads
    4. Recurse - investigate leads depth-first
    5. Synthesize - produce final output
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        config: Optional[RLMConfig] = None,
        on_step: Optional[Callable[[ThinkingStep], None]] = None,
        on_citation: Optional[Callable[[Citation], None]] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        matter_model=None,  # Optional[MatterModel] — injected when enable_matter_model=True
    ):
        self.client = gemini_client
        self.config = config or RLMConfig()
        self.on_step = on_step
        self.on_citation = on_citation
        self.on_progress = on_progress
        self._matter_model = matter_model
        # Per-investigation semaphore to limit concurrent CPU-intensive operations.
        # Recreated when doc_count changes so the limit stays calibrated.
        # Never reset to None mid-investigation — that would corrupt concurrent waiters.
        self._operation_semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_doc_count: int = -1  # sentinel: semaphore not yet calibrated
        self._doc_count: int = 0  # Track document count for adaptive behavior

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the operation semaphore.

        Recreates if _doc_count has changed since the semaphore was last created
        (e.g. a new investigation on a different-sized repo) but never nulls out
        an existing semaphore while coroutines may be waiting on it.
        """
        if (self._operation_semaphore is None
                or self._semaphore_doc_count != self._doc_count):
            # Small repos (<=5 docs): max 2 concurrent ops
            # Medium repos (6-20 docs): max 3 concurrent ops
            # Large repos (>20 docs): max 5 concurrent ops
            if self._doc_count <= 5:
                max_concurrent = 2
            elif self._doc_count <= 20:
                max_concurrent = 3
            else:
                max_concurrent = 5
            self._operation_semaphore = asyncio.Semaphore(max_concurrent)
            self._semaphore_doc_count = self._doc_count
        return self._operation_semaphore

    def _adapt_config_for_repo_size(self, doc_count: int):
        """Adjust config parameters based on repository size."""
        self._doc_count = doc_count

        if doc_count <= 5:
            # Small repos: reduce parallelism significantly
            self.config.max_leads_per_level = min(self.config.max_leads_per_level, 2)
            self.config.parallel_reads = min(self.config.parallel_reads, 2)
            self.config.max_iterations = min(self.config.max_iterations, 8)
            self.config.depth_citation_threshold = min(self.config.depth_citation_threshold, 8)
        elif doc_count <= 20:
            # Medium repos: moderate reduction
            self.config.max_leads_per_level = min(self.config.max_leads_per_level, 3)
            self.config.parallel_reads = min(self.config.parallel_reads, 3)
            self.config.max_iterations = min(self.config.max_iterations, 12)
        # Large repos: use default config

    async def investigate(
        self,
        query: str,
        repository_path: str | Path,
    ) -> InvestigationState:
        """
        Run full recursive investigation.

        Args:
            query: The legal question to investigate
            repository_path: Path to document repository

        Returns:
            InvestigationState with all findings, citations, thinking trace
        """
        from ..matter.runtime import MatterRuntimeAdapter, NullMatterAdapter

        repo = MatterRepository(repository_path)
        # Always store the resolved absolute path so state.repository_path is stable
        # regardless of CWD changes (e.g., FastAPI background tasks).
        state = InvestigationState.create(query, str(repo.base_path))

        # Adapt configuration based on repository size.
        # _doc_count update triggers semaphore recreation in _get_semaphore() so
        # the concurrency limit stays calibrated without nulling out mid-flight waiters.
        stats = repo.get_stats()
        self._adapt_config_for_repo_size(stats.total_files)

        # Build matter adapter — real or null depending on config + injected model
        if self.config.enable_matter_model and self._matter_model is not None:
            run_id = self._matter_model.start_run(query)
            matter_adapter = MatterRuntimeAdapter(self._matter_model, run_id)
        else:
            run_id = None
            matter_adapter = NullMatterAdapter()

        state._matter_adapter = matter_adapter

        # Classify the query
        state.query_classification = classify_query(query)
        self._emit_step(
            state,
            StepType.THINKING,
            f"Query classified as {state.query_classification['type']} (complexity: {state.query_classification['complexity']}/5)",
        )

        try:
            # Phase 1: Orientation
            await self._orient(state, repo)

            # Phase 2: Iterative investigation loop
            await self._investigate_loop(state, repo)

            # If user stopped the run, skip verify/synthesis and mark interrupted.
            # Partial facts/citations/leads are preserved as-is for resume.
            _adapter = getattr(state, "_matter_adapter", None)
            if _adapter is not None and _adapter.is_stop_requested():
                self._emit_step(state, StepType.THINKING, "Stopped by user — partial state preserved")
                state.interrupt()
                if run_id is not None:
                    self._matter_model.interrupt_run(run_id)
                return state

            # Phase 2.5: Verify citations
            await self._verify_citations(state, repo)

            # Phase 3: Final synthesis
            await self._synthesize(state)

            state.complete()
            if run_id is not None:
                self._matter_model.complete_run(run_id)
                # Detect numeric conflicts → gaps (SO-6 + SO-7)
                self._matter_model.detect_quant_conflicts()
                # Generate clarification questions from open gaps (SO-7)
                self._matter_model.generate_clarifications_from_gaps(
                    run_id=run_id,
                    top_n=3,
                    min_materiality=0.5,
                )

        except Exception as e:
            state.fail(str(e))
            if run_id is not None:
                self._matter_model.fail_run(run_id, str(e))
            raise

        return state

    async def _orient(self, state: InvestigationState, repo: MatterRepository):
        """Phase 1: Understand repository and form initial hypothesis."""
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is not None and _adapter.is_stop_requested():
            return  # Stop was requested before orientation even started
        self._emit_step(state, StepType.THINKING, "Analyzing repository structure...")

        # Get repository overview
        stats = repo.get_stats()
        structure = repo.get_structure()

        structure_str = "\n".join(f"  {folder}: {count} files" for folder, count in structure.items())

        # Read persisted matter state — activates SO-1 (reuse) and SO-4 (issue-driven)
        adapter = getattr(state, "_matter_adapter", None)
        matter_ctx = adapter.get_context() if adapter is not None else None

        # Seed InvestigationState with facts already in the matter model (SO-1 hot reuse)
        if matter_ctx is not None and matter_ctx.existing_assertion_count > 0:
            self._hydrate_from_matter_model(state)

        prompt = ORIENTATION_PROMPT.format(
            structure=structure_str,
            total_files=stats.total_files,
            query=state.query,
            matter_context=_format_matter_context(matter_ctx),
        )

        # Orientation cache key: sha256 of normalized query + total file count.
        # total_files is a cheap repo-change proxy: adding/removing documents
        # invalidates the cache so stale plans are not reused.
        import hashlib as _hashlib
        _orient_key = _hashlib.sha256(
            f"{state.query.lower().strip()}\n{stats.total_files}".encode()
        ).hexdigest()

        _plan_defaults = {
            "issues": [],
            "relevant_folders": [],
            "initial_searches": [],
            "hypothesis": "Investigating query across available documents",
        }

        # On warm runs (prior issues in matter model) check reasoning cache first
        # to avoid repeating the FLASH orientation call (SO-1 hot path).
        plan = None
        if (matter_ctx is not None
                and matter_ctx.open_issues
                and self._matter_model is not None):
            plan = self._matter_model.cache.get("orient", _orient_key)

        if plan is None:
            # Cache miss or cold run: call LLM
            response = await self.client.complete(prompt, tier=ModelTier.FLASH)
            plan = self._parse_json_safe(response, _plan_defaults)
            # Persist for future warm runs
            if self._matter_model is not None:
                self._matter_model.cache.put("orient", _orient_key, plan)
        else:
            self._emit_step(
                state, StepType.THINKING, "Orientation cache hit — reusing prior plan"
            )

        state.hypothesis = plan.get("hypothesis")
        state.findings["issues"] = plan.get("issues", [])
        state.findings["initial_plan"] = plan

        # Record issues in matter model if enabled; collect new IDs so initial leads can
        # be linked to freshly-created issues even on the first run (SO-4 backbone fix).
        adapter = getattr(state, "_matter_adapter", None)
        _new_issue_ids: list[str] = []
        if adapter is not None and self._matter_model is not None:
            from ..matter.enums import IssueType
            for issue_title in plan.get("issues", []):
                if isinstance(issue_title, str) and issue_title.strip():
                    issue_id, _ = self._matter_model.issues.upsert_issue(
                        title=issue_title.strip(),
                        issue_type=IssueType.CLAIM,
                        salience=0.7,
                    )
                    _new_issue_ids.append(issue_id)
                    adapter.log_step(
                        f"Issue identified: {issue_title[:100]}",
                        why="From orientation analysis",
                    )

        # Create initial leads from plan — preserve raw search terms to bypass
        # _extract_search_term() token collapse (SO-4 issue-focused search).
        # Use weakest prior-run issue if it exists; otherwise rotate through freshly
        # created issues so run-1 facts are linked to issues from the start.
        weakest_id = matter_ctx.weakest_issue_id if matter_ctx else None
        _issue_pool = _new_issue_ids  # fallback pool: distribute leads across new issues
        _initial_searches = [s for s in plan.get("initial_searches", [])[:5]
                             if isinstance(s, str) and s.strip()]
        for _idx, search_term in enumerate(_initial_searches):
            # Assign focus_issue_id: prefer weakest from prior run, else rotate new issues
            if weakest_id:
                _focus_id = weakest_id
            elif _issue_pool:
                _focus_id = _issue_pool[_idx % len(_issue_pool)]
            else:
                _focus_id = None
            priority = 0.9 if (_focus_id and _idx == 0) else 0.8
            state.add_lead(
                description=f"Search for: {search_term}",
                source="initial_plan",
                priority=priority,
                search_term=search_term.strip(),
                focus_issue_id=_focus_id,
            )

        # If no valid searches were produced (either planner returned none or all were
        # filtered as blank/non-string), fall back: weakest issue title → query tokens.
        # Use _initial_searches (post-filter) so sanitized-empty plans hit this branch.
        if not _initial_searches:
            _fallback_issue_id: Optional[str] = None
            if matter_ctx and matter_ctx.weakest_issue_id and matter_ctx.open_issues:
                weakest_issues = [i for i in matter_ctx.open_issues
                                  if i.get("id") == matter_ctx.weakest_issue_id]
                if weakest_issues:
                    # Issue found — use its title as search term and link to it
                    fallback_term = weakest_issues[0]["title"]
                    _fallback_issue_id = matter_ctx.weakest_issue_id
                else:
                    # weakest_issue_id not in open_issues — generic query, first new issue
                    fallback_term = state.query
                    _fallback_issue_id = _issue_pool[0] if _issue_pool else None
            else:
                fallback_term = state.query
                _fallback_issue_id = _issue_pool[0] if _issue_pool else None
            state.add_lead(
                description=f"Search for key terms in query",
                source="fallback",
                priority=0.8,
                search_term=fallback_term,
                focus_issue_id=_fallback_issue_id,
            )

        self._emit_step(
            state,
            StepType.THINKING,
            f"Hypothesis: {state.hypothesis}",
            details=plan,
        )

        # Log orientation summary to reasoning ledger (SO-3 user visibility)
        adapter = getattr(state, "_matter_adapter", None)
        if adapter is not None:
            issues_found = plan.get("issues", [])
            searches_planned = plan.get("initial_searches", [])
            adapter.log_step(
                f"Orientation complete: {len(issues_found)} issues, {len(searches_planned)} search leads",
                why=f"Hypothesis: {(state.hypothesis or '')[:200]}",
            )

    async def _investigate_loop(self, state: InvestigationState, repo: MatterRepository):
        """Phase 2: Iterative investigation with recursive lead following."""
        iteration = 0
        max_iterations = self.config.max_iterations  # Configurable limit

        while iteration < max_iterations:
            # Check user stop request before each iteration
            adapter = getattr(state, "_matter_adapter", None)
            if adapter is not None and adapter.is_stop_requested():
                self._emit_step(
                    state,
                    StepType.THINKING,
                    "Investigation paused: user requested stop",
                )
                break

            # Track facts before this iteration for diminishing returns check
            facts_before = len(state.findings.get("accumulated_facts", []))

            pending_leads = state.get_pending_leads()

            if not pending_leads:
                self._emit_step(state, StepType.THINKING, "No more leads to investigate")
                break

            # Take top leads up to limit
            leads_to_process = [
                lead for lead in pending_leads[:self.config.max_leads_per_level]
                if lead.priority >= self.config.min_lead_priority
            ]

            # Skip low priority leads
            for lead in pending_leads[:self.config.max_leads_per_level]:
                if lead.priority < self.config.min_lead_priority:
                    state.mark_lead_investigated(lead.id, "Skipped - low priority")

            if not leads_to_process:
                iteration += 1
                continue

            self._emit_step(
                state,
                StepType.THINKING,
                f"Investigating {len(leads_to_process)} leads in parallel (iteration {iteration + 1})",
            )

            # Log iteration start to reasoning ledger
            if adapter is not None:
                lead_summaries = ", ".join(
                    (l.search_term or l.description)[:60] for l in leads_to_process[:3]
                )
                adapter.log_step(
                    f"Iteration {iteration + 1}: investigating {len(leads_to_process)} leads",
                    why=f"Leads: {lead_summaries}",
                )

            # Process leads in parallel
            tasks = [
                self._investigate_lead(state, repo, lead)
                for lead in leads_to_process
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Lead investigation failed: {leads_to_process[i].description}: {result}")

            # Track facts added this iteration for diminishing returns
            facts_after = len(state.findings.get("accumulated_facts", []))
            facts_added = facts_after - facts_before
            state.facts_per_iteration.append(facts_added)

            # Flush belief revision for any new assertions added this iteration
            if adapter is not None:
                adapter.flush_revisions()

            iteration += 1

            # Check for user redirect request — inject high-priority lead for target issue (SO-3)
            if adapter is not None and adapter.is_redirect_requested():
                redirect_issue_id = adapter.get_redirect_issue_id()
                adapter.clear_redirect()
                if redirect_issue_id is not None:
                    # Resolve issue title from matter model (state.findings["issues"]
                    # stores string titles, not dicts — must use the issue store)
                    issue_title = redirect_issue_id[:40]  # safe fallback
                    if self._matter_model is not None:
                        issue_row = self._matter_model.issues.get_issue(redirect_issue_id)
                        if issue_row:
                            issue_title = issue_row.get("title", redirect_issue_id[:40])
                    state.add_lead(
                        description=f"Redirect focus: investigate '{issue_title}'",
                        source="user_redirect",
                        priority=0.95,
                        search_term=issue_title,
                        focus_issue_id=redirect_issue_id,
                    )
                    adapter.log_step(
                        f"Redirected investigation to issue: '{issue_title}'",
                        why="User redirect request",
                    )
                    self._emit_step(
                        state,
                        StepType.REPLAN,
                        f"Investigation redirected to: '{issue_title}'",
                    )

            # Reprioritize leads based on accumulated context
            if iteration % 2 == 0:  # Every other iteration
                state.reprioritize_leads()

            # Save checkpoint periodically
            if self.config.checkpoint_dir and iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint(state, iteration)

            # Check if we should continue (adaptive termination)
            should_continue, termination_reason = self._should_continue_investigation(state)
            if not should_continue:
                self._emit_step(
                    state,
                    StepType.THINKING,
                    f"Early termination: {termination_reason}",
                )
                break

    def _hydrate_from_matter_model(self, state: InvestigationState) -> None:
        """Seed InvestigationState with the top assertions from the persistent matter model.

        Called at the start of _orient() when the matter model has existing data.
        This prevents the engine from "re-discovering" facts already in the assertion
        graph, satisfying SO-1 (durable matter model — hot path reads from store).

        Only loads facts into accumulated_facts; the dedup gate in state.add_facts()
        prevents duplicates if the engine independently re-extracts the same text.
        """
        if self._matter_model is None:
            return
        try:
            recent = self._matter_model.assertions.list_recent(limit=30)
        except Exception:
            return
        if not recent:
            return

        loaded = 0
        _strip_role_prefix = __import__("re").compile(r'^\[[A-Z_]+\]\s*').sub
        for row in recent:
            prop = row.get("proposition_text", "")
            if not prop:
                continue
            source_role = row.get("source_role") or "unknown"
            label = source_role.upper()
            # Strip any existing [ROLE] prefix to prevent double-labeling legacy rows
            prop_clean = _strip_role_prefix('', prop)
            state.add_facts([f"[{label}] {prop_clean}"])
            loaded += 1

        if loaded:
            self._emit_step(
                state,
                StepType.THINKING,
                f"Hydrated {loaded} facts from prior matter model run (SO-1 reuse)",
            )

    async def _investigate_lead(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        lead: Lead,
    ):
        """Investigate a single lead - may spawn sub-investigations."""
        # Respect stop requests before doing any expensive work.
        # This is checked here (not only in _investigate_loop) because all leads
        # in a batch are launched via asyncio.gather before the loop stop check runs.
        # Do NOT mark the lead investigated — leave it pending so resume can retry it.
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is not None and _adapter.is_stop_requested():
            return

        # Acquire semaphore to limit concurrent heavy operations
        async with self._get_semaphore():
            state.recursion_depth += 1
            state.max_depth_reached = max(state.max_depth_reached, state.recursion_depth)

            try:
                effective_depth = self._calculate_effective_depth(state)
                if state.recursion_depth > effective_depth:
                    state.mark_lead_investigated(lead.id, f"Max depth ({effective_depth}) reached")
                    return

                self._emit_step(state, StepType.SEARCH, f"Investigating: {lead.description}")

                # Use preserved raw search term if available (SO-4); else extract from description.
                # Raw terms avoid token collapse from _extract_search_term for issue-focused leads.
                search_term = lead.search_term or self._extract_search_term(lead.description)

                # Perform search (scale workers based on doc count)
                max_workers = min(4, max(1, self._doc_count))
                results = repo.search(search_term, context_lines=3, max_workers=max_workers)
                state.searches_performed += 1

                if not results.hits:
                    state.mark_lead_investigated(lead.id, "No results found")
                    # Record as a gap if this lead was targeting a specific issue (SO-7)
                    _adp = getattr(state, "_matter_adapter", None)
                    if _adp is not None and lead.focus_issue_id is not None:
                        from ..matter.enums import GapType
                        _adp.record_gap(
                            description=f"No documents found for search: '{search_term}'",
                            gap_type=GapType.MISSING_DOCUMENT,
                            expected_artifact=search_term,
                            materiality=0.4,
                            affected_type="issue",
                            affected_id=lead.focus_issue_id,
                        )
                    return

                self._emit_step(
                    state,
                    StepType.FINDING,
                    f"Found {len(results.hits)} matches in {results.files_searched} files",
                )

                # Analyze top results
                await self._analyze_search_results(state, repo, results, lead)

                state.mark_lead_investigated(lead.id, f"Found {len(results.hits)} matches")

            finally:
                state.recursion_depth -= 1

    async def _analyze_search_results(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        results: SearchResults,
        lead: Lead,
    ):
        """Analyze search results and extract findings/leads."""
        # Format results for LLM
        results_text = self._format_search_results(results)

        prompt = ANALYZE_FINDINGS_PROMPT.format(
            query=state.query,
            hypothesis=state.hypothesis or "No hypothesis yet",
            search_term=results.query,
            search_results=results_text,
        )

        # Use FLASH for analysis
        response = await self.client.complete(prompt, tier=ModelTier.FLASH)

        # Parse with safe defaults
        analysis = self._parse_json_safe(response, {
            "key_facts": [],
            "new_leads": [],
            "hypothesis_update": None,
            "next_searches": [],
        })

        # Store key facts with per-fact source attribution (SO-5 provenance fix).
        # Facts from the LLM may be bare strings (legacy) or dicts with "fact" and
        # optional "source_file" keys. We use the per-fact source_file when present
        # so each fact is labeled and recorded against its actual source document
        # rather than always being attributed to the single top search hit.
        if analysis.get("key_facts"):
            from ..matter.runtime import infer_source_role as _infer_role

            # Build a name→(relative_path, SearchHit) lookup for all top hits
            _hit_by_name: dict[str, object] = {}
            for _h in results.top(5):
                _hit_by_name[_h.filename] = _h
                _hit_by_name[_h.filename.lower()] = _h

            # Fallback: top-hit doc_id and source role for facts with no source_file
            _top_hits = results.top(1)
            _fallback_hit = _top_hits[0] if _top_hits else None
            _fallback_src_label = (
                _infer_role(_fallback_hit.filename).value.upper()
                if _fallback_hit else "UNKNOWN"
            )
            if _fallback_hit:
                _fb_fp = Path(_fallback_hit.file_path)
                try:
                    _fallback_doc_id = str(_fb_fp.relative_to(repo.base_path))
                except ValueError:
                    _fallback_doc_id = _fallback_hit.filename
            else:
                _fallback_doc_id = "unknown"

            facts_to_add: list[tuple[str, str, str]] = []  # (text, src_label, doc_id)
            for fact_item in analysis["key_facts"]:
                if isinstance(fact_item, str):
                    facts_to_add.append((fact_item, _fallback_src_label, _fallback_doc_id))
                elif isinstance(fact_item, dict) and "fact" in fact_item:
                    fact_text = fact_item["fact"]
                    src_file = fact_item.get("source_file") or ""
                    # Try to resolve source_file to a known hit
                    hit = _hit_by_name.get(src_file) or _hit_by_name.get(src_file.lower())
                    if hit is not None:
                        src_label = _infer_role(hit.filename).value.upper()
                        try:
                            doc_id = str(Path(hit.file_path).relative_to(repo.base_path))
                        except ValueError:
                            doc_id = hit.filename
                    else:
                        src_label = _fallback_src_label
                        doc_id = _fallback_doc_id
                    facts_to_add.append((fact_text, src_label, doc_id))

            # Add to state with per-fact source-role prefix (SO-5)
            state.add_facts([f"[{lbl}] {txt}" for txt, lbl, _ in facts_to_add])

            # Record into matter model with correct per-fact doc_id
            adapter = getattr(state, "_matter_adapter", None)
            if adapter is not None:
                issue_id = lead.focus_issue_id if lead is not None else None
                for fact_text, _lbl, doc_id in facts_to_add:
                    adapter.record_fact(fact_text, document_id=doc_id, issue_id=issue_id)
                if facts_to_add:
                    unique_docs = {d for _, _, d in facts_to_add}
                    adapter.log_step(
                        f"Recorded {len(facts_to_add)} facts from search: {results.query[:60]}",
                        why=f"Sources: {', '.join(sorted(unique_docs)[:3])}",
                    )

        # Update hypothesis if changed
        if analysis.get("hypothesis_update"):
            state.hypothesis = analysis["hypothesis_update"]
            self._emit_step(
                state,
                StepType.REPLAN,
                f"Hypothesis updated: {state.hypothesis}",
            )

        # Add citations from top hits
        for hit in results.top(5):
            citation = state.add_citation(
                document=hit.filename,
                page=hit.page_num,
                text=hit.match_text[:200],
                context=hit.context[:500],
                relevance=f"Found via search: {results.query}",
            )
            if self.on_citation:
                self.on_citation(citation)

        # Add new leads (handle both "description" and compact "desc" formats).
        # Propagate focus_issue_id so issue focus does not decay on follow-on leads (SO-4).
        _follow_on_issue_id = lead.focus_issue_id if lead is not None else None
        for lead_data in analysis.get("new_leads", [])[:3]:
            if isinstance(lead_data, dict):
                desc = lead_data.get("description") or lead_data.get("desc")
                if desc:
                    state.add_lead(
                        description=desc,
                        source=f"Analysis of '{results.query}'",
                        priority=lead_data.get("priority", 0.5),
                        focus_issue_id=_follow_on_issue_id,
                    )

        # Deep read top documents in parallel
        top_files = list(results.by_file().keys())[:self.config.parallel_reads]
        if top_files:
            focus_issue_id = lead.focus_issue_id if lead is not None else None
            await self._batch_deep_read(state, repo, top_files, focus_issue_id=focus_issue_id)

    async def _batch_deep_read(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_paths: list[str],
        focus_issue_id: Optional[str] = None,
    ):
        """Process multiple documents with controlled parallelism."""
        if not file_paths:
            return

        self._emit_step(
            state,
            StepType.READING,
            f"Deep reading {len(file_paths)} documents",
        )

        # Use semaphore to limit concurrent document processing
        # This prevents CPU overload from too many parallel PDF parses + LLM calls
        read_semaphore = asyncio.Semaphore(self.config.parallel_reads)

        async def limited_read(fp: str):
            async with read_semaphore:
                return await self._deep_read_document(state, repo, fp, focus_issue_id=focus_issue_id)

        tasks = [limited_read(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Document read failed: {file_paths[i]}: {result}")

    async def _deep_read_document(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_path: str,
        focus_issue_id: Optional[str] = None,
    ):
        """Perform deep analysis of a document."""
        self._emit_step(state, StepType.READING, f"Deep reading: {Path(file_path).name}")

        _rel_path: Optional[str] = None  # set before _reading_in_progress; used in except
        try:
            # Cold/hot split (SO-1): check inventory BEFORE the expensive repo.read()
            # so hot-path documents skip PDF parsing entirely, not just LLM calls.
            _mm = self._matter_model
            _inventory_doc_id: Optional[str] = None
            _fp = Path(file_path)
            # Normalize to a repo-relative path so the inventory key is STABLE across runs.
            # file_path may be an absolute temp path (e.g. /tmp/abc123/contracts/msa.pdf)
            # which changes every run, defeating hot-path reuse (SO-1).
            # relative_to() gives us contracts/msa.pdf — a durable key.
            try:
                _rel_path = str(_fp.relative_to(repo.base_path))
            except ValueError:
                # file_path is outside base_path (e.g., external/S3 URI) — use as-is
                _rel_path = file_path

            # Within-run dedup: multiple parallel leads can surface the same top file.
            # Once a coroutine reaches the cold path for a file, mark it in-flight so
            # other coroutines skip it (asyncio is single-threaded; check+add is atomic).
            if _rel_path in state._reading_in_progress:
                return
            state._reading_in_progress.add(_rel_path)

            if _mm is not None:
                import hashlib as _hl
                try:
                    # Compute sha256 BEFORE the hot-path check so that content changes
                    # at the same path are always detected via upsert's mismatch logic.
                    # Reading raw bytes is cheap (no PDF parsing); we avoid that with repo.read().
                    _abs_fp = (Path(repo.base_path) / file_path) if not _fp.is_absolute() else _fp
                    _raw = _abs_fp.read_bytes()
                    _sha = _hl.sha256(_raw).hexdigest()
                    _inv_id, _ = _mm.inventory.upsert(
                        relative_path=_rel_path,
                        sha256=_sha,
                        size_bytes=len(_raw),
                        file_type=_fp.suffix.lstrip(".") or None,
                    )
                    _inventory_doc_id = _inv_id

                    if _mm.inventory.is_ingested(_rel_path):
                        # HOT PATH: sha256 verified current; already fully ingested in a prior run
                        state.documents_read += 1
                        self._emit_step(
                            state, StepType.READING,
                            f"Hot path (already ingested): {_fp.name}",
                        )
                        return
                except Exception as _inv_err:
                    logger.warning(
                        "Inventory failure for %s (proceeding to cold path): %s",
                        _rel_path, _inv_err,
                    )

            # COLD PATH: full document parsing + LLM analysis
            doc = repo.read(file_path)

            state.documents_read += 1

            # Use excerpt for analysis
            content = doc.get_excerpt(self.config.excerpt_chars)

            prompt = DEEP_READ_PROMPT.format(
                filename=doc.filename,
                page_range=f"1-{doc.page_count}",
                content=content,
                query=state.query,
                focus=state.hypothesis or state.query,
            )

            # Use LITE for bulk reading
            response = await self.client.complete(prompt, tier=ModelTier.LITE)

            analysis = self._parse_json_safe(response, {
                "key_facts": [],
                "quotes": [],
                "entities": {"people": [], "companies": [], "dates": [], "amounts": []},
                "numeric_facts": [],
                "fact_relationships": [],
                "connections": [],
                "concerns": [],
            })

            # Add quotes as citations
            for quote in analysis.get("quotes", [])[:3]:
                if isinstance(quote, dict) and "text" in quote:
                    citation = state.add_citation(
                        document=doc.filename,
                        page=quote.get("page"),
                        text=quote["text"][:300],
                        context="",
                        relevance=quote.get("relevance", "Direct quote"),
                    )
                    if self.on_citation:
                        self.on_citation(citation)

            # Store findings with deduplication
            # key_facts can be strings or dicts with "fact" key
            # Always initialize these so numeric_facts / gap grounding below can reference them
            # even when key_facts is empty.
            facts_to_add: list[str] = []
            _recorded_ids: list[str] = []
            if analysis.get("key_facts"):
                for fact_item in analysis["key_facts"]:
                    if isinstance(fact_item, str):
                        facts_to_add.append(fact_item)
                    elif isinstance(fact_item, dict) and "fact" in fact_item:
                        facts_to_add.append(fact_item["fact"])
                # Prefix each fact with its source role (SO-5 per-fact calibration)
                from ..matter.runtime import infer_source_role as _infer_role
                _src_label = _infer_role(doc.filename).value.upper()
                state.add_facts([f"[{_src_label}] {f}" for f in facts_to_add])
                # Also record into matter model if enabled; pass issue_id if from targeted lead
                adapter = getattr(state, "_matter_adapter", None)
                if adapter is not None:
                    for fact_text in facts_to_add:
                        # Use _rel_path (repo-relative, stable) not doc.filename (basename only)
                        # so same-basename files in different dirs don't alias in assertion_occurrence
                        _aid = adapter.record_fact(fact_text, document_id=_rel_path, issue_id=focus_issue_id)
                        _recorded_ids.append(_aid)

                    # Build assertion dependency graph from LLM-identified relationships (SO-2)
                    # Uses 0-based indices into facts_to_add / _recorded_ids
                    _rels = analysis.get("fact_relationships") or []
                    for _rel in _rels[:5]:  # cap to 5 edges per document
                        if not isinstance(_rel, dict):
                            continue
                        _fi = _rel.get("from_idx")
                        _ti = _rel.get("to_idx")
                        _rt = _rel.get("relation", "")
                        if (isinstance(_fi, int) and isinstance(_ti, int)
                                and 0 <= _fi < len(_recorded_ids)
                                and 0 <= _ti < len(_recorded_ids)
                                and _fi != _ti
                                # Same proposition text deduplicates to same assertion_id
                                and _recorded_ids[_fi] != _recorded_ids[_ti]):
                            adapter.record_assertion_link(
                                _recorded_ids[_fi], _recorded_ids[_ti], _rt
                            )

            # Extract and store structured numeric facts (SO-6).
            # Ground each quant fact to an assertion_id by searching the already-recorded
            # key_facts for the raw numeric text. This gives provenance for reconciliation.
            if analysis.get("numeric_facts"):
                _adp = getattr(state, "_matter_adapter", None)
                if _adp is not None:
                    for nf in analysis["numeric_facts"][:20]:  # limit to avoid noise
                        if not isinstance(nf, dict):
                            continue
                        kind = nf.get("kind", "amount")
                        raw = nf.get("raw", "")
                        if not raw:
                            continue
                        value = nf.get("value")
                        amount = float(value) if kind == "amount" and value is not None else None
                        rate = float(value) if kind == "rate" and value is not None else None
                        date_val = raw if kind == "date" else None
                        # Ground to the first key_fact assertion that mentions this raw value
                        _nf_assertion_id: Optional[str] = None
                        _raw_lower = raw.lower()
                        for _ft, _fa in zip(facts_to_add, _recorded_ids):
                            if _raw_lower and _raw_lower in _ft.lower():
                                _nf_assertion_id = _fa
                                break
                        _adp.record_quant(
                            quant_kind=kind,
                            raw_text=f"{raw} — {nf.get('context', '')}",
                            amount_value=amount,
                            currency=nf.get("currency"),
                            date_value=date_val,
                            rate_value=rate,
                            subject_type=nf.get("subject"),
                            assertion_id=_nf_assertion_id,
                        )

            # Extract and store entities
            if analysis.get("entities"):
                state.add_entities_from_analysis(analysis["entities"], doc.filename)
                # Persist people and companies to durable actor store (SO-5 actor resolution)
                adapter = getattr(state, "_matter_adapter", None)
                if adapter is not None:
                    entities = analysis["entities"]
                    for name in entities.get("people", []):
                        if isinstance(name, str) and name.strip():
                            adapter.record_actor(name.strip(), actor_type="person")
                    for name in entities.get("companies", []):
                        if isinstance(name, str) and name.strip():
                            adapter.record_actor(name.strip(), actor_type="organization")

            # Add leads for mentioned entities/connections
            for concern in analysis.get("concerns", [])[:2]:
                if isinstance(concern, str):
                    state.add_lead(
                        description=f"Investigate concern: {concern}",
                        source=doc.filename,
                        priority=0.6,
                    )

            # LLM-driven gap detection: check referenced documents against repo (SO-7)
            connections = analysis.get("connections", [])
            if connections:
                # Build a quick lookup of known filenames (lowercase)
                known_names = {
                    f.filename.lower()
                    for f in repo.list_files()
                }
                _adp = getattr(state, "_matter_adapter", None)
                for ref in connections[:5]:  # limit to avoid noise
                    if not isinstance(ref, str) or not ref.strip():
                        continue
                    ref = ref.strip()
                    # Create a search lead so the pipeline tries to find it
                    state.add_lead(
                        description=f"Find referenced document: {ref}",
                        source=doc.filename,
                        priority=0.65,
                        search_term=ref,
                        focus_issue_id=focus_issue_id,
                    )
                    # If no existing filename contains ALL significant words from the
                    # reference, record as a gap.  Requiring ALL words (not just any one)
                    # prevents "Amendment No. 2" from falsely matching an unrelated file
                    # that happens to contain the word "amendment".
                    ref_lower = ref.lower()
                    ref_words = [w for w in ref_lower.split() if len(w) > 3]
                    found_in_repo = bool(ref_words) and any(
                        all(word in fname for word in ref_words)
                        for fname in known_names
                    )
                    if not found_in_repo and _adp is not None and ref_words:
                        from ..matter.enums import GapType
                        # Link gap to an issue if available; otherwise link to the first
                        # assertion extracted from this document (SO-7 impact grounding).
                        # A referenced-but-absent document is most likely to affect the
                        # assertions from the document that cites it.
                        if focus_issue_id:
                            _gap_aff_type: Optional[str] = "issue"
                            _gap_aff_id: Optional[str] = focus_issue_id
                        elif _recorded_ids:
                            _gap_aff_type = "assertion"
                            _gap_aff_id = _recorded_ids[0]
                        else:
                            _gap_aff_type = None
                            _gap_aff_id = None
                        _adp.record_gap(
                            description=f"Referenced document not found in repository: '{ref}' (mentioned in {Path(file_path).name})",
                            gap_type=GapType.MISSING_DOCUMENT,
                            expected_artifact=ref,
                            materiality=0.5,
                            affected_type=_gap_aff_type,
                            affected_id=_gap_aff_id,
                        )

            # Mark document as fully ingested so future runs take the hot path (SO-1)
            if _mm is not None and _inventory_doc_id is not None:
                try:
                    _mm.inventory.mark_ingested(_inventory_doc_id)
                except Exception:
                    pass  # non-fatal

        except Exception as e:
            self._emit_step(state, StepType.ERROR, f"Failed to read {file_path}: {e}")
            # Remove from in-progress so a subsequent lead can retry on transient failures.
            # Permanent failures (corrupted file) will re-fail and re-record the gap below.
            if _rel_path is not None:
                state._reading_in_progress.discard(_rel_path)
            # Record as gap: document exists in search index but could not be read (SO-7)
            _adp = getattr(state, "_matter_adapter", None)
            if _adp is not None:
                from ..matter.enums import GapType
                _adp.record_gap(
                    description=f"Document read failed: {Path(file_path).name} — {str(e)[:150]}",
                    gap_type=GapType.MISSING_DOCUMENT,
                    expected_artifact=str(file_path),
                    materiality=0.3,
                    affected_type="issue" if focus_issue_id else None,
                    affected_id=focus_issue_id,
                )

    async def _verify_citations(self, state: InvestigationState, repo: MatterRepository):
        """Verify citations by checking if quoted text exists in documents."""
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is not None and _adapter.is_stop_requested():
            return  # Skip expensive verification if user stopped the run
        unverified = state.get_unverified_citations()
        if not unverified:
            return

        self._emit_step(state, StepType.VERIFY, f"Verifying {len(unverified)} citations...")

        verified_count = 0
        unverified_count = 0

        for citation in unverified:
            try:
                # Try to find the document
                doc = repo.read(citation.document)

                # Normalize text for comparison (lowercase, collapse whitespace)
                citation_text = " ".join(citation.text.lower().split())
                doc_text = " ".join(doc.full_text.lower().split())

                # Check if a significant portion of the citation exists in the document
                # Use first 50 chars for matching (handles truncation)
                search_text = citation_text[:50]

                if search_text in doc_text:
                    citation.verified = True
                    citation.verification_note = "Text found in document"
                    verified_count += 1
                else:
                    # Try fuzzy match - look for any 20-char substring
                    found = False
                    for i in range(0, min(len(citation_text) - 20, 100), 10):
                        chunk = citation_text[i:i+20]
                        if chunk in doc_text:
                            citation.verified = True
                            citation.verification_note = "Partial text match found"
                            verified_count += 1
                            found = True
                            break

                    if not found:
                        citation.verified = False
                        citation.verification_note = "Text not found in document"
                        unverified_count += 1

            except Exception as e:
                citation.verified = False
                citation.verification_note = f"Could not verify: {e}"
                unverified_count += 1

        stats = state.get_verification_stats()
        self._emit_step(
            state,
            StepType.VERIFY,
            f"Verification complete: {stats['verified']} verified, {stats['unverified']} unverified",
        )

    async def _synthesize(self, state: InvestigationState):
        """Phase 3: Final synthesis using Pro model."""
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is not None and _adapter.is_stop_requested():
            return  # Skip synthesis if user stopped the run
        self._emit_step(state, StepType.SYNTHESIS, "Synthesizing final analysis...")

        # Log synthesis entry to reasoning ledger (SO-3)
        adapter = getattr(state, "_matter_adapter", None)
        if adapter is not None:
            facts = state.findings.get("accumulated_facts", [])
            adapter.log_step(
                f"Synthesis phase: {state.documents_read} docs, "
                f"{state.searches_performed} searches, {len(facts)} facts",
                why="All leads exhausted or investigation complete",
            )

        # Compile all findings
        facts = state.findings.get("accumulated_facts", [])
        findings_text = "\n".join(f"• {fact}" for fact in facts[:30])

        # Get citations with verification status
        citations_text = state.get_citations_formatted()

        # Get entity summary
        entities_text = state.get_entities_formatted()

        # Build source-role calibration from matter model (SO-5)
        source_calibration = self._build_source_calibration(state)

        # Build quantitative reconciliation summary (SO-6)
        quant_summary = self._build_quant_summary()

        prompt = SYNTHESIS_PROMPT.format(
            query=state.query,
            docs_analyzed=state.documents_read,
            searches=state.searches_performed,
            citation_count=len(state.citations),
            max_depth=state.max_depth_reached,
            hypothesis=state.hypothesis or "No specific hypothesis formed",
            source_calibration=source_calibration,
            quant_summary=quant_summary,
            entities=entities_text or "No entities identified",
            findings=findings_text or "No specific findings accumulated",
            citations=citations_text or "No citations collected",
        )

        # Use PRO for final synthesis
        response = await self.client.complete(prompt, tier=ModelTier.PRO)

        state.findings["final_output"] = response
        self._emit_step(state, StepType.SYNTHESIS, "Analysis complete")

    def _build_source_calibration(self, state: InvestigationState) -> str:
        """
        Build a source-role calibration block for the synthesis prompt (SO-5).

        Queries the matter model for assertion counts grouped by source_role so
        the LLM knows which facts came from advocacy sources (complaints, briefs)
        vs. operative sources (contracts, orders) before synthesizing.
        """
        if self._matter_model is None:
            return "No source-role data available — treat all facts with appropriate skepticism."

        try:
            rows = self._matter_model.db.execute(
                """SELECT ao.source_role, COUNT(DISTINCT a.id) AS cnt
                   FROM assertion a
                   JOIN assertion_occurrence ao ON ao.assertion_id = a.id
                   WHERE a.matter_id = ?
                   GROUP BY ao.source_role
                   ORDER BY cnt DESC""",
                (self._matter_model.matter_id,),
            ).fetchall()
        except Exception:
            return "Source-role data unavailable."

        if not rows:
            return "No assertions recorded in matter model yet."

        # Role descriptions used to calibrate LLM trust
        _role_labels = {
            "advocacy": "ADVOCACY (alleged/argued — do NOT treat as established facts)",
            "operative": "OPERATIVE (signed documents, orders — treat as established)",
            "authoritative": "AUTHORITATIVE (statutes, case law — treat as controlling)",
            "procedural": "PROCEDURAL (court filings, notices — established procedurally)",
            "informal": "INFORMAL (emails, notes — corroborative only)",
            "draft": "DRAFT (unexecuted — treat as proposed, not operative)",
            "post_hoc": "POST-HOC EXPLANATORY (created after events — limited weight)",
            "unknown": "UNKNOWN SOURCE ROLE — verify before relying",
        }

        lines = ["The following facts were extracted from documents with these source roles:"]
        for row in rows:
            role = row["source_role"] if row["source_role"] else "unknown"
            label = _role_labels.get(role, f"{role.upper()} — calibrate appropriately")
            lines.append(f"  • {row['cnt']} assertions from {label}")
        lines.append(
            "\nWARNING: Facts from ADVOCACY sources represent one party's position, not "
            "established truth. Do not amplify advocacy material as if it were operative fact."
        )
        return "\n".join(lines)

    def _build_quant_summary(self) -> str:
        """
        Build a quantitative reconciliation block for the synthesis prompt (SO-6).

        Shows extracted monetary totals by subject_type and flags any detected conflicts
        so the LLM can include numeric analysis in the synthesis memo.
        """
        if self._matter_model is None:
            return "No quantitative data extracted."

        try:
            total_count = self._matter_model.quant.count()
            if total_count == 0:
                return "No numeric facts extracted from documents."

            reconciliation = self._matter_model.reconcile()
            conflicts = self._matter_model.quant.get_conflicts()
        except Exception:
            return "Quantitative data unavailable."

        lines = [f"Extracted {total_count} numeric facts."]

        if reconciliation:
            lines.append("Monetary amounts by category (USD unless noted):")
            for subject, data in sorted(reconciliation.items(), key=lambda x: x[1]["total"], reverse=True):
                lines.append(f"  • {subject}: ${data['total']:,.2f} ({data['count']} entries)")

        if conflicts:
            lines.append("NUMERIC CONFLICTS DETECTED (same category, different amounts):")
            for c in conflicts[:3]:
                subject = c.get("subject_type", "unknown")
                currency = c.get("currency", "")
                values = [f"${v:,.2f}" for v in (c.get("values") or [])[:4]]
                lines.append(f"  ⚠ {subject} ({currency}): {', '.join(values)} — UNRESOLVED DISCREPANCY")

        return "\n".join(lines)

    # ==========================================================================
    # Units 21-25: Advanced Analysis Methods
    # ==========================================================================

    async def extract_entities_from_text(
        self,
        text: str,
        filename: str = "unknown",
    ) -> dict:
        """
        Extract entities from text using LLM analysis.

        Args:
            text: Text to analyze
            filename: Source filename for context

        Returns:
            Dict with categorized entities
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            filename=filename,
            text=text[:5000],  # Limit text size
        )

        response = await self.client.complete(prompt, tier=ModelTier.LITE)

        defaults = {
            "people": [],
            "organizations": [],
            "dates": [],
            "amounts": [],
            "locations": [],
            "legal_refs": [],
        }

        return self._parse_json_safe(response, defaults)

    async def detect_contradiction(
        self,
        doc1_name: str,
        statement1: str,
        context1: str,
        doc2_name: str,
        statement2: str,
        context2: str,
    ) -> dict:
        """
        Analyze two statements for potential contradictions.

        Args:
            doc1_name: Name of first document
            statement1: First statement
            context1: Context around first statement
            doc2_name: Name of second document
            statement2: Second statement
            context2: Context around second statement

        Returns:
            Dict with contradiction analysis
        """
        prompt = CONTRADICTION_DETECTION_PROMPT.format(
            doc1_name=doc1_name,
            statement1=statement1,
            context1=context1,
            doc2_name=doc2_name,
            statement2=statement2,
            context2=context2,
        )

        response = await self.client.complete(prompt, tier=ModelTier.FLASH)

        defaults = {
            "is_contradiction": False,
            "contradiction_type": "none",
            "severity": "none",
            "explanation": "Unable to analyze",
            "reconciliation_possible": True,
            "reconciliation_theory": None,
            "legal_significance": "Unknown",
            "follow_up_needed": [],
        }

        return self._parse_json_safe(response, defaults)

    async def build_timeline(
        self,
        state: InvestigationState,
    ) -> dict:
        """
        Build a timeline from accumulated evidence in investigation state.

        Args:
            state: InvestigationState with findings

        Returns:
            Dict with chronology and analysis
        """
        # Collect all date-related information from state
        events = []

        # From timeline events in state
        for event in state.timeline:
            events.append({
                "date": event.date_str,
                "event": event.description,
                "source": event.source_doc,
            })

        # From entity dates
        for key, entity in state.entities.items():
            if entity.entity_type == "date":
                events.append({
                    "date": entity.name,
                    "event": entity.context or "Date mentioned",
                    "source": entity.sources[0] if entity.sources else "unknown",
                })

        document_list = "\n".join(f"- {doc}" for doc in set(
            e.get("source", "unknown") for e in events
        ))

        events_text = "\n".join(
            f"- {e.get('date', 'unknown')}: {e.get('event', 'unknown')} (from {e.get('source', 'unknown')})"
            for e in events
        )

        prompt = TIMELINE_EXTRACTION_PROMPT.format(
            document_list=document_list or "No documents",
            events=events_text or "No events found",
        )

        response = await self.client.complete(prompt, tier=ModelTier.FLASH)

        defaults = {
            "chronology": [],
            "timeline_gaps": [],
            "date_conflicts": [],
            "key_periods": [],
        }

        return self._parse_json_safe(response, defaults)

    async def assess_claim_evidence(
        self,
        claim: str,
        state: InvestigationState,
    ) -> dict:
        """
        Assess the strength of evidence for a specific claim.

        Args:
            claim: The claim to assess
            state: InvestigationState with evidence

        Returns:
            Dict with evidence assessment
        """
        # Gather supporting and contradicting evidence
        supporting = []
        contradicting = []

        # Search facts for relevant evidence
        facts = state.findings.get("accumulated_facts", [])
        for fact in facts:
            # Simple relevance check
            if any(word in fact.lower() for word in claim.lower().split()[:5]):
                supporting.append(fact)

        # Check for contradictions
        for contradiction in state.contradictions:
            if claim.lower() in contradiction.statement1.lower() or claim.lower() in contradiction.statement2.lower():
                contradicting.append(f"{contradiction.statement1} vs {contradiction.statement2}")

        prompt = EVIDENCE_ASSESSMENT_PROMPT.format(
            claim=claim,
            supporting_evidence="\n".join(f"- {e}" for e in supporting[:10]) or "No direct supporting evidence found",
            contradicting_evidence="\n".join(f"- {e}" for e in contradicting[:5]) or "No contradicting evidence found",
        )

        response = await self.client.complete(prompt, tier=ModelTier.FLASH)

        defaults = {
            "claim": claim,
            "evidence_classification": {
                "direct": [],
                "circumstantial": [],
                "primary_sources": [],
                "secondary_sources": [],
                "hearsay_concerns": [],
            },
            "corroboration_level": "unknown",
            "authentication_assessment": "unknown",
            "overall_strength": "unknown",
            "strength_score": 0,
            "reasoning": "Unable to assess",
            "vulnerabilities": [],
            "strengthening_opportunities": [],
        }

        return self._parse_json_safe(response, defaults)

    # ==========================================================================
    # Units 26-30: Utility Improvements
    # ==========================================================================

    def estimate_completion(self, state: InvestigationState) -> dict:
        """
        Estimate investigation completion percentage and remaining work.

        Args:
            state: Current investigation state

        Returns:
            Dict with completion estimates
        """
        # Calculate based on multiple factors
        factors = {}

        # Leads completion
        total_leads = len(state.leads)
        investigated_leads = len([l for l in state.leads if l.status == "investigated"])
        factors["leads_complete"] = (investigated_leads / max(total_leads, 1)) * 100

        # Depth progress
        factors["depth_progress"] = (state.max_depth_reached / self.config.max_depth) * 100

        # Citation coverage
        target_citations = 10  # Minimum target
        factors["citation_coverage"] = min(100, (len(state.citations) / target_citations) * 100)

        # Document coverage
        target_docs = 5  # Minimum target
        factors["doc_coverage"] = min(100, (state.documents_read / target_docs) * 100)

        # Confidence
        confidence = state.get_confidence_score()
        factors["confidence"] = confidence["score"]

        # Weighted average
        weights = {
            "leads_complete": 0.3,
            "depth_progress": 0.1,
            "citation_coverage": 0.25,
            "doc_coverage": 0.15,
            "confidence": 0.2,
        }

        overall = sum(factors[k] * weights[k] for k in weights)

        # Estimate remaining
        pending_leads = len(state.get_pending_leads())
        estimated_remaining_iterations = min(pending_leads, 10)

        return {
            "overall_progress": round(overall, 1),
            "factors": factors,
            "pending_leads": pending_leads,
            "estimated_remaining_iterations": estimated_remaining_iterations,
            "is_complete": overall >= 90 or (
                confidence["score"] >= 80 and len(state.citations) >= 10
            ),
        }

    def get_investigation_summary(self, state: InvestigationState) -> dict:
        """
        Generate a summary of the current investigation state.

        Args:
            state: Investigation state

        Returns:
            Dict with investigation summary
        """
        return {
            "id": state.id,
            "query": state.query,
            "status": state.status,
            "started_at": state.started_at,
            "documents_read": state.documents_read,
            "searches_performed": state.searches_performed,
            "citations_found": len(state.citations),
            "verified_citations": len([c for c in state.citations if c.verified]),
            "entities_extracted": len(state.entities),
            "leads_total": len(state.leads),
            "leads_pending": len(state.get_pending_leads()),
            "leads_investigated": len([l for l in state.leads if l.status == "investigated"]),
            "max_depth_reached": state.max_depth_reached,
            "contradictions_found": len(state.contradictions),
            "timeline_events": len(state.timeline),
            "confidence": state.get_confidence_score(),
            "completion": self.estimate_completion(state),
        }

    def _emit_step(
        self,
        state: InvestigationState,
        step_type: StepType,
        content: str,
        details: Optional[dict] = None,
    ):
        """Emit a thinking step and call callback."""
        step = state.add_step(step_type, content, details)
        if self.on_step:
            self.on_step(step)
        # Also emit progress update
        self._emit_progress(state)

    def _emit_progress(self, state: InvestigationState):
        """Emit progress update."""
        if self.on_progress:
            self.on_progress(state.get_progress())

    def _calculate_effective_depth(self, state: InvestigationState) -> int:
        """Calculate effective max depth based on investigation progress."""
        if not self.config.adaptive_depth:
            return self.config.max_depth

        base_depth = self.config.max_depth

        # Reduce depth if we have many citations already
        if len(state.citations) >= self.config.depth_citation_threshold:
            return max(self.config.min_depth, base_depth - 2)

        # Reduce depth if confidence is high
        confidence = state.get_confidence_score()
        if confidence["score"] >= 70:
            return max(self.config.min_depth, base_depth - 1)

        # Increase depth if we have few leads
        pending_leads = len(state.get_pending_leads())
        if pending_leads > 10:
            return min(base_depth + 1, 7)  # Cap at 7

        return base_depth

    def _should_continue_investigation(self, state: InvestigationState) -> tuple[bool, str]:
        """Determine if investigation should continue.

        Uses four criteria:
        1. Repository size - small repos terminate faster
        2. Query complexity - simpler queries terminate faster
        3. Diminishing returns - stop if recent iterations add few new facts
        4. Verified citations - keep this requirement (per user preference)

        Returns:
            (should_continue, reason) - reason explains why we stopped/continue
        """
        # Always continue if minimum criteria not met
        if state.max_depth_reached < self.config.min_depth:
            return True, "Building minimum evidence base"

        confidence = state.get_confidence_score()

        # Get query complexity thresholds based on query type
        query_type = state.query_classification.get("type", "unknown") if state.query_classification else "unknown"
        complexity = state.query_classification.get("complexity", 3) if state.query_classification else 3

        # Define thresholds per query type (confidence_threshold, min_citations)
        thresholds = {
            "factual": (60, 5),      # Simple fact lookup - terminate quickly
            "procedural": (65, 6),   # Process/timeline questions
            "analytical": (75, 8),   # Deeper analysis needed
            "comparative": (80, 10), # Need multiple perspectives
            "evaluative": (85, 12),  # Most thorough investigation
            "unknown": (70, 7),      # Default middle ground
        }

        conf_threshold, min_citations = thresholds.get(query_type, (70, 7))

        # Adjust thresholds based on complexity (1-5 scale)
        # Lower complexity = lower threshold, higher complexity = higher threshold
        complexity_adjustment = (complexity - 3) * 5  # -10 to +10 adjustment
        conf_threshold = max(50, min(90, conf_threshold + complexity_adjustment))

        # NEW: Adjust thresholds for small document sets
        # With fewer documents, we need fewer citations and can terminate earlier
        if self._doc_count <= 5:
            min_citations = min(min_citations, max(2, self._doc_count))
            conf_threshold = max(40, conf_threshold - 15)
        elif self._doc_count <= 10:
            min_citations = min(min_citations, self._doc_count)
            conf_threshold = max(45, conf_threshold - 10)

        # Check 1: Query complexity-aware confidence check
        if confidence["score"] >= conf_threshold and len(state.citations) >= min_citations:
            return False, f"Sufficient evidence for {query_type} query (confidence: {confidence['score']:.0f}%, {len(state.citations)} citations)"

        # Check 2: For small repos, terminate if we've read all documents
        if self._doc_count > 0 and state.documents_read >= self._doc_count:
            if len(state.citations) >= 1:  # At least some evidence found
                return False, f"All {self._doc_count} documents processed"

        # Check 3: Diminishing returns - stop if last 2 iterations added < 3 facts each
        # For small repos, be more aggressive (< 2 facts)
        fact_threshold = 2 if self._doc_count <= 5 else 3
        if len(state.facts_per_iteration) >= 2:
            recent_facts = state.facts_per_iteration[-2:]
            if all(f < fact_threshold for f in recent_facts):
                # Diminishing returns detected - but only stop if we have SOME evidence
                min_citations_for_stop = 2 if self._doc_count <= 5 else 3
                if len(state.citations) >= min_citations_for_stop and confidence["score"] >= 40:
                    return False, f"Diminishing returns (last 2 iterations: {recent_facts[0]}, {recent_facts[1]} new facts)"

        # Check 4: Extreme diminishing returns - 3 iterations with 0-1 facts each
        if len(state.facts_per_iteration) >= 3:
            recent_facts = state.facts_per_iteration[-3:]
            if all(f <= 1 for f in recent_facts):
                # Very low productivity - stop regardless
                return False, f"Very low productivity (last 3 iterations: {recent_facts} new facts each)"

        # Check if we have pending leads worth investigating
        pending = state.get_pending_leads()
        high_priority = [l for l in pending if l.priority >= 0.5]
        if not high_priority:
            return False, "No high-priority leads remaining"

        return True, f"Continuing investigation ({len(high_priority)} leads, confidence: {confidence['score']:.0f}%)"

    def _extract_search_term(self, lead_description: str) -> str:
        """Extract a SINGLE high-value search term from lead description.

        The search system uses literal string matching, not boolean operators.
        We must return a single term that's likely to find relevant documents.
        """
        import re

        # Remove common prefixes
        prefixes = ["Search for:", "Investigate:", "Find:", "Look for:"]
        term = lead_description
        for prefix in prefixes:
            if term.startswith(prefix):
                term = term[len(prefix):].strip()
                break

        # Remove boolean operators and quotes
        term = re.sub(r'\bAND\b|\bOR\b|\bNOT\b', ' ', term, flags=re.IGNORECASE)
        term = re.sub(r'[\'\"()]', ' ', term)
        term = re.sub(r'\s+', ' ', term).strip()

        # Extract all meaningful words
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into',
                     'about', 'which', 'when', 'where', 'what', 'how', 'who', 'why',
                     'any', 'all', 'each', 'between', 'related', 'regarding', 'concerning'}

        words = [w for w in term.split()
                 if len(w) > 2 and w.lower() not in stopwords and not w.startswith('$')]

        if not words:
            # Fallback: use any word over 3 chars
            words = [w for w in term.split() if len(w) > 3]

        if not words:
            return term[:50]  # Last resort

        # Priority: specific terms > generic terms
        # Look for entity-like words (capitalized, numbers, specific patterns)
        priority_words = []

        # 1. Specific dollar amounts or numbers
        for w in words:
            if re.match(r'^\$?[\d,.]+[MKBmkb]?$', w):  # $3.1M, 192, etc.
                priority_words.append(w)

        # 2. Proper nouns / entity names (capitalized words)
        for w in words:
            if w[0].isupper() and len(w) > 2:
                priority_words.append(w)

        # 3. Legal-specific terms
        legal_terms = {'contract', 'agreement', 'breach', 'damages', 'liability',
                       'warranty', 'negligence', 'fraud', 'misrepresentation',
                       'estimate', 'inspection', 'maintenance', 'invoice', 'payment'}
        for w in words:
            if w.lower() in legal_terms:
                priority_words.append(w)

        # Use the highest priority word found, or fall back to first meaningful word
        if priority_words:
            return priority_words[0]

        return words[0]

    def _format_search_results(self, results: SearchResults, max_hits: int = 10) -> str:
        """Format search results for LLM consumption."""
        lines = []
        for hit in results.top(max_hits):
            lines.append(f"File: {hit.filename} (page {hit.page_num})")
            lines.append(f"Match: {hit.match_text}")
            if hit.context_before:
                lines.append(f"Context before: {' '.join(hit.context_before)}")
            if hit.context_after:
                lines.append(f"Context after: {' '.join(hit.context_after)}")
            lines.append("---")
        return "\n".join(lines)

    def _parse_json_safe(self, text: str, defaults: dict) -> dict:
        """Parse JSON from LLM response with safe fallback to defaults."""
        try:
            if not text or not text.strip():
                logger.warning("Empty LLM response, using defaults")
                return defaults

            json_str = self._extract_json(text)
            if not json_str or json_str == text and "{" not in text:
                logger.warning(f"No JSON found in response (len={len(text)}), using defaults")
                return defaults

            result = json.loads(json_str)
            # Merge with defaults for any missing keys
            for key, value in defaults.items():
                if key not in result:
                    result[key] = value
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse failed: {e}, response preview: {text[:200] if text else 'empty'}")
            return defaults

    def _save_checkpoint(self, state: InvestigationState, iteration: int):
        """Save investigation checkpoint."""
        if not self.config.checkpoint_dir:
            return

        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{state.id}_iter{iteration}.json"
        state.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Also save latest checkpoint reference
        latest_path = Path(self.config.checkpoint_dir) / f"latest_{state.id}.json"
        state.save_checkpoint(latest_path)

    async def resume_investigation(
        self,
        checkpoint_path: str | Path,
    ) -> InvestigationState:
        """
        Resume investigation from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            InvestigationState with completed investigation
        """
        state = InvestigationState.load_checkpoint(checkpoint_path)
        repo = MatterRepository(state.repository_path)

        self._emit_step(state, StepType.THINKING, f"Resuming investigation from checkpoint")

        # Wire matter adapter so resumed runs get ledger entries + stop propagation
        from ..matter.runtime import MatterRuntimeAdapter, NullMatterAdapter
        run_id = None
        if self.config.enable_matter_model and self._matter_model is not None:
            run_id = self._matter_model.start_run(f"Resume: {state.query[:120]}")
            state._matter_adapter = MatterRuntimeAdapter(self._matter_model, run_id)
        else:
            state._matter_adapter = NullMatterAdapter()

        try:
            # Continue investigation loop if not complete
            if state.status not in ("completed", "failed"):
                await self._investigate_loop(state, repo)
                await self._verify_citations(state, repo)
                await self._synthesize(state)
                state.complete()
                if run_id is not None:
                    self._matter_model.complete_run(run_id)

        except Exception as e:
            state.fail(str(e))
            if run_id is not None:
                self._matter_model.fail_run(run_id, str(e))
            raise

        return state

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        # Try to find JSON block with proper closing
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
            # No closing ``` - try to extract JSON directly from after ```json
            remaining = text[start:].strip()
            if remaining.startswith("{"):
                return self._extract_raw_json(remaining)

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find raw JSON
        return self._extract_raw_json(text)

    def _extract_raw_json(self, text: str) -> str:
        """Extract raw JSON object from text."""
        if "{" not in text:
            return text

        start = text.find("{")
        # Find matching closing brace
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

        # If no matching brace found, try to return partial JSON up to end
        # This handles truncated responses
        return text[start:]

    async def decompose_query(self, query: str) -> list[dict]:
        """
        Decompose a compound query into sub-queries.

        Args:
            query: The potentially compound legal query

        Returns:
            List of sub-queries with metadata:
            [{"query": "...", "priority": 0-1, "depends_on": None or query_id}]
        """
        prompt = f"""You are a legal research assistant. Analyze this query and determine if it should be broken into sub-queries.

Query: {query}

If this is a simple query, return it as-is.
If this is a compound query (multiple questions, comparisons, or multi-part analysis), break it into logical sub-queries.

Consider:
1. Are there multiple distinct questions?
2. Is there a comparison between different things?
3. Are there dependent questions (one must be answered before another)?

Respond in JSON format:
{{
    "is_compound": true/false,
    "sub_queries": [
        {{"query": "...", "priority": 0.0-1.0, "depends_on": null or index}},
        ...
    ],
    "reasoning": "Why you chose to split or not split"
}}

If not compound, return the original query as a single sub_query with priority 1.0."""

        response = await self.client.complete(prompt, tier=ModelTier.FLASH)

        defaults = {
            "is_compound": False,
            "sub_queries": [{"query": query, "priority": 1.0, "depends_on": None}],
            "reasoning": "Single query"
        }

        result = self._parse_json_safe(response, defaults)

        # Ensure we always have at least the original query
        if not result.get("sub_queries"):
            result["sub_queries"] = [{"query": query, "priority": 1.0, "depends_on": None}]

        return result["sub_queries"]

    async def investigate_multi(
        self,
        queries: list[str],
        repository_path: str | Path,
        parallel: bool = True,
    ) -> dict[str, InvestigationState]:
        """
        Investigate multiple queries against the same repository.

        Args:
            queries: List of queries to investigate
            repository_path: Path to document repository
            parallel: If True, run investigations in parallel

        Returns:
            Dict mapping query to InvestigationState
        """
        if parallel:
            # Run all investigations in parallel
            tasks = [
                self.investigate(query, repository_path)
                for query in queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            return {
                query: result if not isinstance(result, Exception) else self._create_failed_state(query, str(result), repository_path)
                for query, result in zip(queries, results)
            }
        else:
            # Run sequentially
            results = {}
            for query in queries:
                try:
                    results[query] = await self.investigate(query, repository_path)
                except Exception as e:
                    results[query] = self._create_failed_state(query, str(e), repository_path)
            return results

    def _create_failed_state(self, query: str, error: str, repository_path: str | Path) -> InvestigationState:
        """Create a failed investigation state."""
        state = InvestigationState.create(query, str(Path(repository_path).resolve()))
        state.fail(error)
        return state

    async def investigate_compound(
        self,
        query: str,
        repository_path: str | Path,
    ) -> dict[str, Any]:
        """
        Investigate a potentially compound query.

        This method:
        1. Decomposes the query if compound
        2. Runs sub-investigations
        3. Merges results into a unified response

        Args:
            query: The legal query (may be compound)
            repository_path: Path to document repository

        Returns:
            Dict with merged results and individual states
        """
        # Decompose query
        sub_queries = await self.decompose_query(query)

        # Separate independent and dependent queries
        independent = [sq for sq in sub_queries if sq.get("depends_on") is None]
        dependent = [sq for sq in sub_queries if sq.get("depends_on") is not None]

        # Run independent queries in parallel
        independent_queries = [sq["query"] for sq in independent]
        results = await self.investigate_multi(independent_queries, repository_path, parallel=True)

        # Run dependent queries sequentially with context
        for dep_query in dependent:
            dep_on = dep_query.get("depends_on")
            if dep_on is not None and dep_on < len(independent_queries):
                # Add context from dependency
                parent_query = independent_queries[dep_on]
                parent_state = results.get(parent_query)
                if parent_state and parent_state.status == "completed":
                    # Enrich query with parent findings
                    enriched_query = f"{dep_query['query']}\n\nContext from previous analysis:\n{parent_state.hypothesis or ''}"
                    results[dep_query["query"]] = await self.investigate(enriched_query, repository_path)
                else:
                    results[dep_query["query"]] = await self.investigate(dep_query["query"], repository_path)
            else:
                results[dep_query["query"]] = await self.investigate(dep_query["query"], repository_path)

        # Merge results
        merged = self._merge_investigation_results(query, results)

        return {
            "original_query": query,
            "sub_queries": [sq["query"] for sq in sub_queries],
            "individual_results": results,
            "merged": merged,
        }

    def _merge_investigation_results(
        self,
        original_query: str,
        results: dict[str, InvestigationState],
    ) -> dict[str, Any]:
        """Merge results from multiple investigations."""
        merged = {
            "total_documents_read": 0,
            "total_citations": 0,
            "all_citations": [],
            "all_entities": {},
            "all_facts": [],
            "all_hypotheses": [],
            "combined_confidence": 0,
        }

        for query, state in results.items():
            if state.status != "completed":
                continue

            merged["total_documents_read"] += state.documents_read
            merged["total_citations"] += len(state.citations)
            merged["all_citations"].extend(state.citations)

            # Merge entities
            for key, entity in state.entities.items():
                if key in merged["all_entities"]:
                    merged["all_entities"][key].mentions += entity.mentions
                    merged["all_entities"][key].sources.extend(entity.sources)
                else:
                    merged["all_entities"][key] = entity

            # Collect facts
            merged["all_facts"].extend(state.findings.get("accumulated_facts", []))

            # Collect hypotheses
            if state.hypothesis:
                merged["all_hypotheses"].append({
                    "query": query,
                    "hypothesis": state.hypothesis,
                })

            # Accumulate confidence
            confidence = state.get_confidence_score()
            merged["combined_confidence"] += confidence["score"]

        # Average confidence
        num_completed = sum(1 for s in results.values() if s.status == "completed")
        if num_completed > 0:
            merged["combined_confidence"] = merged["combined_confidence"] / num_completed

        # Deduplicate facts
        seen_facts = set()
        unique_facts = []
        for fact in merged["all_facts"]:
            fact_normalized = " ".join(fact.lower().split())[:100]
            if fact_normalized not in seen_facts:
                seen_facts.add(fact_normalized)
                unique_facts.append(fact)
        merged["all_facts"] = unique_facts

        return merged

    async def summarize_document(
        self,
        file_path: Path,
        repository: Optional[MatterRepository] = None,
    ) -> dict[str, Any]:
        """
        Create a structured summary of a single document.

        Args:
            file_path: Path to document
            repository: Optional repository for reading document

        Returns:
            Dict with summary information
        """
        if repository:
            doc = repository.read(str(file_path))
        else:
            from ..core.repository import MatterRepository
            temp_repo = MatterRepository(file_path.parent)
            doc = temp_repo.read(str(file_path))

        # Determine document type from filename
        filename_lower = file_path.name.lower()
        doc_type = "other"
        type_keywords = {
            "contract": "contract",
            "agreement": "contract",
            "complaint": "pleading",
            "motion": "pleading",
            "letter": "correspondence",
            "email": "correspondence",
            "memo": "correspondence",
        }
        for keyword, dtype in type_keywords.items():
            if keyword in filename_lower:
                doc_type = dtype
                break

        # Truncate content for prompt
        content = doc.full_text[:self.config.excerpt_chars]

        prompt = SUMMARIZE_DOCUMENT_PROMPT.format(
            filename=file_path.name,
            doc_type=doc_type,
            content=content,
        )

        response = await self.client.complete(prompt, tier=ModelTier.FLASH)

        defaults = {
            "summary": "Unable to generate summary",
            "document_type": doc_type,
            "parties": [],
            "key_dates": [],
            "key_terms": [],
            "amounts": [],
            "concerns": [],
        }

        result = self._parse_json_safe(response, defaults)
        result["filename"] = file_path.name
        result["file_path"] = str(file_path)

        return result

    async def summarize_documents(
        self,
        file_paths: list[Path],
        repository: Optional[MatterRepository] = None,
    ) -> dict[str, Any]:
        """
        Create summaries for multiple documents and a collection summary.

        Args:
            file_paths: List of document paths
            repository: Optional repository

        Returns:
            Dict with individual and collection summaries
        """
        # Generate individual summaries in parallel
        tasks = [
            self.summarize_document(fp, repository)
            for fp in file_paths
        ]
        individual_summaries = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        valid_summaries = [
            s for s in individual_summaries
            if not isinstance(s, Exception)
        ]

        if not valid_summaries:
            return {
                "individual_summaries": [],
                "collection_summary": None,
                "error": "No documents could be summarized",
            }

        # Generate collection summary
        document_list = "\n".join(
            f"- {s['filename']}: {s.get('document_type', 'unknown')}"
            for s in valid_summaries
        )

        summaries_text = "\n\n".join(
            f"### {s['filename']}\n{s.get('summary', 'No summary')}"
            for s in valid_summaries
        )

        prompt = SUMMARIZE_COLLECTION_PROMPT.format(
            document_list=document_list,
            summaries=summaries_text,
        )

        response = await self.client.complete(prompt, tier=ModelTier.FLASH)

        defaults = {
            "collection_summary": "Unable to generate collection summary",
            "parties": [],
            "timeline": [],
            "themes": [],
            "document_relationships": [],
            "gaps": [],
        }

        collection_summary = self._parse_json_safe(response, defaults)

        return {
            "individual_summaries": valid_summaries,
            "collection_summary": collection_summary,
            "document_count": len(valid_summaries),
        }
