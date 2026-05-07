"""RLM Engine - Recursive Language Model investigation engine.

OPTIMIZED VERSION:
- Document extraction cache (don't re-extract same doc)
- Search term deduplication (skip similar searches)
- Smart model selection (FLASH for simple queries)
- Early sufficiency checks (after first iteration)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any
from pathlib import Path
import asyncio
import logging
import re
import time

from ..core.models import GeminiClient, ModelTier
from ..core.repository import MatterRepository
from ..core.search import SearchResults
from ..core.external_search import ExternalSearchManager
from ..core.fact_store import FactStore
from ..core.telemetry import InvestigationTelemetry, StepOperation
from ..core.tracing import TracingProvider, TracingContext, NoOpProvider, SpanHandle
from .state import InvestigationState, StepType, ThinkingStep, Citation, Lead, classify_query
from . import decisions
from .research_agent import (
    ResearchAgent,
    ResearchAgentConfig,
    ResearchContext,
    ResearchEmitter,
)

logger = logging.getLogger(__name__)


def _fmt_list(items: list, max_items: int = 10, max_len: int = 200) -> str:
    """Format a list for display: 'item1, item2, item3...'"""
    if not items:
        return "(none)"
    # Handle items that might be dicts or other types
    display = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            # Try common keys
            s = item.get("description") or item.get("name") or item.get("file") or str(item)
        else:
            s = str(item)
        display.append(s[:max_len] if len(s) > max_len else s)
    suffix = f"... (+{len(items) - max_items} more)" if len(items) > max_items else ""
    return ", ".join(display) + suffix


@dataclass
class RLMConfig:
    """Configuration for RLM engine."""
    max_depth: int = 3  # Reduced from 5
    max_leads_per_level: int = 3  # Reduced from 5
    max_documents_per_search: int = 5  # Reduced from 10
    # Dynamic excerpt limits based on query complexity
    excerpt_chars_simple: int = 8000  # Fast path for simple queries
    excerpt_chars_complex: int = 40000  # Full coverage for complex queries
    parallel_reads: int = 3  # Reduced from 5
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 5
    max_iterations: int = 10  # Reduced from 20
    # New optimization settings
    early_exit_facts: int = 5  # Exit early if we have this many facts
    skip_similar_searches: bool = True
    use_flash_for_simple: bool = True
    # External search settings
    enable_external_search: bool = True
    max_case_law_results: int = 5  # Results per query
    max_web_results: int = 5       # Results per query
    max_case_law_queries: int = 5  # Max queries to run
    max_web_queries: int = 5       # Max queries to run
    parallel_external_searches: bool = True  # Run queries in parallel
    # Research-agent settings (replace legacy keyword-routed external search)
    max_research_turns: int = 4           # Hard cap on decide_next_action calls
    max_research_actions_per_turn: int = 6  # Soft cap on parallel tool calls per turn
    research_tool_timeout_s: float = 45.0
    research_turn_timeout_s: float = 90.0
    # S3 settings (all optional; local disk used if not set)
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_checkpoint_prefix: Optional[str] = None  # e.g. "matters/case-123/checkpoints"
    s3_facts_prefix: Optional[str] = None       # e.g. "matters/case-123/facts"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


@dataclass
class InvestigationCache:
    """Cache to avoid redundant work."""
    extracted_docs: set = field(default_factory=set)  # Docs we've extracted facts from
    searched_terms: set = field(default_factory=set)  # Search terms we've used
    irrelevant_docs: set = field(default_factory=set)  # Docs marked IRRELEVANT by LLM
    consecutive_read_failures: int = 0  # Track consecutive read failures
    total_read_failures: int = 0  # Track total read failures
    MAX_CONSECUTIVE_FAILURES: int = 5  # Abort after this many consecutive failures

    def record_read_success(self):
        """Record a successful document read."""
        self.consecutive_read_failures = 0

    def record_read_failure(self) -> bool:
        """Record a failed document read. Returns True if should abort."""
        self.consecutive_read_failures += 1
        self.total_read_failures += 1
        return self.consecutive_read_failures >= self.MAX_CONSECUTIVE_FAILURES

    def has_extracted(self, filepath: str) -> bool:
        """Check if we've already extracted facts from this doc."""
        return filepath in self.extracted_docs

    def mark_extracted(self, filepath: str):
        """Mark a doc as extracted."""
        self.extracted_docs.add(filepath)

    def mark_irrelevant(self, filepath: str):
        """Mark a doc as irrelevant (skip in future ranking)."""
        self.irrelevant_docs.add(filepath)

    def is_irrelevant(self, filepath: str) -> bool:
        """Check if doc was marked irrelevant."""
        return filepath in self.irrelevant_docs

    def is_similar_search(self, term: str) -> bool:
        """Check if we've done a similar search."""
        term_lower = term.lower()
        term_words = set(term_lower.split())

        for existing in self.searched_terms:
            existing_words = set(existing.lower().split())
            # If >50% word overlap, consider it similar
            if term_words and existing_words:
                overlap = len(term_words & existing_words) / min(len(term_words), len(existing_words))
                if overlap > 0.5:
                    return True
        return False

    def add_search(self, term: str):
        """Record a search term."""
        self.searched_terms.add(term.lower())


class RLMEngine:
    """
    Recursive Language Model investigation engine.

    OPTIMIZED loop:
    1. Plan - create investigation strategy
    2. Execute - search and read documents (with caching)
    3. Check - LLM decides if sufficient (check early!)
    4. Synthesize - use appropriate model based on query type
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        config: Optional[RLMConfig] = None,
        on_step: Optional[Callable[[ThinkingStep], None]] = None,
        on_citation: Optional[Callable[[Citation], None]] = None,
        on_fact: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        tracing_provider: Optional[TracingProvider] = None,
    ):
        self.client = gemini_client
        self.config = config or RLMConfig()
        self.on_step = on_step
        self.on_citation = on_citation
        self.on_fact = on_fact
        self.on_progress = on_progress
        self._tracing_provider = tracing_provider or NoOpProvider()
        # Initialize external search manager (enabled by default)
        self.external_search = ExternalSearchManager() if self.config.enable_external_search else None
        self._external_research: dict = {}  # Store external research results
        self._context: Optional[Any] = None  # Investigation context (set during investigate())
        self.repo: Optional[MatterRepository] = None  # Set during investigate()
        self.fact_store: Optional[FactStore] = None  # Set during investigate()
        self._telemetry: Optional[InvestigationTelemetry] = None  # Set during investigate()
        self._trace_ctx: Optional[TracingContext] = None  # Set during investigate()
        # Lead lifecycle tracking
        self._lead_start_times: dict[str, float] = {}

    async def _emit_lead_started(self, state: InvestigationState, lead: Lead):
        """Emit lead.started event and track timing."""
        self._lead_start_times[lead.id] = time.monotonic()
        lead.started_at = datetime.now()
        await self._emit_step_async(
            state, StepType.LEAD_STARTED, f"Lead: {lead.description}",
            details={
                "lead_id": lead.id,
                "type": lead.lead_type,
                "description": lead.description,
                "parent_lead_id": lead.parent_lead_id,
            },
        )

    async def _emit_lead_update(self, state: InvestigationState, lead_id: str, kind: str, data: dict):
        """Emit lead.update event with structured data."""
        await self._emit_step_async(
            state, StepType.LEAD_UPDATE, f"Lead update: {kind}",
            details={"lead_id": lead_id, "kind": kind, "data": data},
        )

    async def _emit_lead_done(self, state: InvestigationState, lead_id: str):
        """Emit lead.done event with duration."""
        duration_ms = 0
        if lead_id in self._lead_start_times:
            duration_ms = int((time.monotonic() - self._lead_start_times.pop(lead_id)) * 1000)
        # Find and update the lead's finished_at
        for lead in state.leads:
            if lead.id == lead_id:
                lead.finished_at = datetime.now()
                break
        await self._emit_step_async(
            state, StepType.LEAD_DONE, f"Lead done ({duration_ms}ms)",
            details={"lead_id": lead_id, "duration_ms": duration_ms},
        )

    async def _emit_lead_error(self, state: InvestigationState, lead_id: str, error: str):
        """Emit lead.error event."""
        await self._emit_step_async(
            state, StepType.LEAD_ERROR, f"Lead failed: {error}",
            details={"lead_id": lead_id, "error": error},
        )

    async def investigate(
        self,
        query: str,
        repository_path: str | Path,
        seed_facts: Optional[list[str]] = None,
        seed_citations: Optional[list[dict]] = None,
        context: Optional[Any] = None,
        message_id: Optional[str] = None,
        user_id: Optional[str] = None,
        setup_duration_ms: int = 0,
    ) -> InvestigationState:
        """Run full recursive investigation.

        Args:
            query: The investigation question
            repository_path: Path to document repository
            seed_facts: Prior-session facts to seed
            seed_citations: Prior-session citations to seed
            context: Optional InvestigationContext with:
                - conversation_history: Prior Q&A for context
                - planning_instructions: Guidance for planning phase
                - output_instructions: Guidance for synthesis (e.g., language)
        """
        repo = MatterRepository(repository_path)
        self.repo = repo  # Store for methods that need repo access (e.g., _load_pinned_documents)

        # Pre-warm _doc_cache with OCR'd content and set _metadata accurately.
        # Must happen before any access to repo.metadata or repo.is_small_repo so
        # those properties never fall through to the sync _compute_metadata() path,
        # which would cache fitz-only (pre-OCR) content and poison the cache.
        await repo._compute_metadata_async()

        self._external_research = {"case_law": [], "web": [], "analysis": {}}  # Reset with proper structure

        self._context = context  # Store context for use in decision functions
        state = InvestigationState.create(query, str(repository_path))

        # Initialize per-investigation telemetry
        self._telemetry = InvestigationTelemetry(message_id=message_id, user_id=user_id)
        self._telemetry.setup_duration_ms = setup_duration_ms

        # Initialize tracing (Langfuse or NoOp)
        trace_handle = self._tracing_provider.start_trace(
            trace_id=state.id,
            name=f"investigation:{state.id}",
            metadata={
                "query": query,
                "repository": str(repository_path),
                "message_id": message_id,
                "user_id": user_id,
            },
        )
        self._trace_ctx = TracingContext(self._tracing_provider, trace_handle)

        # Load fact store for this repository (S3-backed when configured)
        s3_facts_config = None
        if self.config.s3_bucket and self.config.s3_facts_prefix:
            s3_facts_config = {
                "bucket": self.config.s3_bucket,
                "region": self.config.s3_region,
                "prefix": self.config.s3_facts_prefix,
                "aws_access_key_id": self.config.aws_access_key_id,
                "aws_secret_access_key": self.config.aws_secret_access_key,
            }
        self.fact_store = FactStore(Path(repository_path), s3_config=s3_facts_config)
        facts_loaded = await asyncio.to_thread(self.fact_store.load)

        # Emit fact store status to UI trace
        if facts_loaded > 0:
            await self._emit_step_async(
                state,
                StepType.FINDING,
                f"Loaded {facts_loaded} cached facts from previous investigations",
            )
        else:
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"No cached facts found - starting fresh (will save at {self.fact_store.facts_file})",
                visible=False,
            )

        cache = InvestigationCache()

        # Seed prior session data if provided
        # Run off the event loop via to_thread so /health stays responsive
        if seed_facts:
            added = await asyncio.to_thread(state.add_facts, seed_facts)
            logger.info(f"Seeded {added}/{len(seed_facts)} prior-session facts")
        if seed_citations:
            for c in seed_citations:
                state.add_citation(
                    document=c.get("document", ""),
                    page=c.get("page"),
                    text=c.get("text", ""),
                    context=c.get("context", ""),
                    relevance=c.get("relevance", "prior session"),
                    url=c.get("url"),
                    mime=c.get("mime"),
                )
            logger.info(f"Seeded {len(seed_citations)} prior-session citations")

        # Get repo info for informative step message
        repo_name = Path(repository_path).name
        file_count = len(repo.list_files())
        total_chars = repo.metadata.total_chars if repo.metadata else 0

        try:
            # Emit investigation.started event
            await self._emit_step_async(
                state, StepType.INVESTIGATION_STARTED, "Starting investigation",
                details={
                    "query": query,
                    "document_count": file_count,
                    "repository": repo_name,
                },
            )

            # Check if small repository first - uses unified assessment (includes complexity)
            if repo.is_small_repo:
                await self._emit_step_async(
                    state, StepType.THINKING,
                    f"Starting: \"{query[:60]}{'...' if len(query) > 60 else ''}\" on {repo_name} ({file_count} files, {total_chars:,} chars) → small repo mode",
                    visible=False,
                )
                # _direct_answer does its own unified assessment (complexity + external search decision)
                await self._direct_answer(state, repo)
            else:
                # Full RLM investigation for large repositories
                # Phase 1: Unified assessment and planning (includes complexity + fact sheet check)
                assessment = await self._assess_and_create_plan(state, repo)

                # Check if we can answer directly from cached facts
                if assessment.get("can_answer_from_facts", False):
                    relevant_facts = assessment.get("relevant_facts", [])
                    if relevant_facts:
                        await self._emit_step_async(
                            state,
                            StepType.FINDING,
                            f"Can answer from {len(relevant_facts)} cached facts - skipping document investigation",
                        )
                        state.findings["accumulated_facts"] = relevant_facts
                        state.findings["answered_from_cache"] = True
                        is_simple = assessment.get("complexity") == "simple"
                        await self._synthesize(state, is_simple)
                        # Skip the rest - we're done
                        state.complete()
                        return state

                # Extract complexity for synthesis tier selection
                is_simple = assessment.get("complexity") == "simple"
                self._is_simple_query = is_simple

                state.query_classification = {
                    "type": "simple" if is_simple else "complex",
                    "complexity": 2 if is_simple else 4,
                    "llm_classified": True,
                }

                await self._emit_step_async(
                    state, StepType.THINKING,
                    f"Starting: \"{query[:60]}{'...' if len(query) > 60 else ''}\" on {repo_name} ({file_count} files, {total_chars:,} chars) → {'FLASH' if is_simple else 'PRO'} synthesis",
                    visible=False,
                )

                # Phase 2: Investigation loop — reads documents, extracts facts,
                # accumulates research triggers. External research has moved out of
                # the loop body and now runs once after the loop (below).
                await self._investigate_loop(state, repo, cache)

                # Phase 2b: External research agent (runs at most once per
                # investigation). Gated by LITE should_research_externally.
                await self._run_external_research_post_loop(state)

                # Phase 3: Final synthesis
                await self._synthesize(state, is_simple)

                # Save learnings from this query for future reference
                if repo.metadata and state.findings.get("accumulated_facts"):
                    key_facts = state.findings["accumulated_facts"][:3]
                    if key_facts:
                        repo.add_learning(query, "; ".join(key_facts))

            state.complete()

        except Exception as e:
            state.fail(str(e))
            raise
        finally:
            # Save fact store to persist extracted facts for future queries
            if self.fact_store:
                fact_count = len(self.fact_store)
                self._emit_step(
                    state,
                    StepType.THINKING,
                    f"DEBUG: fact_store has {fact_count} facts, _facts list: {len(self.fact_store._facts)}",
                    visible=False,
                )
                if fact_count > 0:
                    try:
                        saved = await asyncio.to_thread(self.fact_store.save)
                        self._emit_step(
                            state,
                            StepType.FINDING,
                            f"Saved {saved} facts to {self.fact_store.facts_file}",
                            visible=False,
                        )
                    except Exception as e:
                        self._emit_step(
                            state,
                            StepType.THINKING,
                            f"ERROR saving facts: {e}",
                            visible=False,
                        )
                else:
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"No new facts extracted this session",
                        visible=False,
                    )

            # Clean up external search sessions
            if self.external_search:
                try:
                    await self.external_search.close()
                except Exception:
                    pass

            # Delete S3 checkpoints after successful completion
            if state.status == "completed":
                self._delete_s3_checkpoints(state.id)

            # Finalize telemetry and emit structured log
            if self._telemetry:
                telemetry_status = state.status or "unknown"
                summary = self._telemetry.finalize(status=telemetry_status)
                state.telemetry_summary = summary.to_dict()
                logger.info(
                    "investigation_complete: id=%s status=%s duration_ms=%d cost_usd=%.6f steps=%d",
                    summary.investigation_id,
                    summary.status,
                    summary.total_duration_ms,
                    summary.total_cost_usd,
                    summary.total_steps,
                    extra={"telemetry": summary.to_dict()},
                )

                # Persist telemetry to DB (fire-and-forget)
                try:
                    from ..db.config import get_database_config
                    get_database_config()
                    asyncio.create_task(_persist_telemetry(summary))
                except ValueError as e:
                    logger.warning("Database not configured - telemetry not persisted: %s", e)

        return state

    def get_trace_ctx(self) -> Optional[TracingContext]:
        """Return the current trace context (if any) for external use.

        Called by Irys.investigate() to pass trace_ctx to post-processing
        steps (e.g. citation injection) before finalize_trace() is called.
        """
        return self._trace_ctx

    def finalize_trace(self, state: InvestigationState) -> None:
        """End the Langfuse trace and flush.  Called by Irys after all
        post-processing (citation injection etc.) is complete."""
        if self._trace_ctx:
            try:
                status = "error" if state.status == "failed" else "ok"
                final_output = state.findings.get("final_output", "")
                self._tracing_provider.end_trace(
                    self._trace_ctx.span_handle,
                    metadata={"status": state.status, "answer_length": len(final_output)},
                    status=status,
                )
                self._tracing_provider.flush()
            except Exception as e:
                logger.warning("Failed to finalize trace: %s", e)
            self._trace_ctx = None

    async def _direct_answer(self, state: InvestigationState, repo: MatterRepository):
        """
        Direct answer mode for small repositories.

        Intelligent flow that only searches externally when genuinely needed:
        0. Check cached facts - maybe we can answer without reading docs
        1. Load all documents (small enough to fit in context)
        2. Unified assessment (FLASH): complexity + external search decision
        3. If external search needed: execute specific searches
        4. If searched: check sufficiency, only do round 2 if critical gap remains
        5. Synthesize with all available information

        This replaces the old trigger-based approach which over-searched.
        """
        # Step 1: Load all content directly
        all_content = repo.get_all_content()
        state.documents_read = len(repo.list_files())
        state.findings["small_repo_content"] = all_content

        # Step 1.5: Extract facts from each document if fact store is empty
        # Wrap each doc read in lead lifecycle
        cache = InvestigationCache()
        if self.fact_store and len(self.fact_store) == 0:
            for doc in repo.list_files():
                read_lead = Lead.create(f"Read document: {doc.filename}", source="direct_answer")
                state.leads.append(read_lead)
                await self._emit_lead_started(state, read_lead)
                await self._read_document(state, repo, doc.path, cache, lead_id=read_lead.id)
                await self._emit_lead_done(state, read_lead.id)

        # Step 1.6: Get cached facts for this query
        cached_facts_str = ""
        if self.fact_store and len(self.fact_store) > 0:
            relevant_facts = self.fact_store.get_relevant(state.query)
            if relevant_facts:
                cached_facts_str = self.fact_store.format_for_llm(relevant_facts)
                await self._emit_step_async(
                    state,
                    StepType.THINKING,
                    f"Found {len(relevant_facts)} potentially relevant cached facts",
                )

        # Step 2: Unified assessment - determines complexity AND external search need

        t_step = self._telemetry.begin_step("assess_small_repo", "planning") if self._telemetry else None
        assessment = await decisions.assess_small_repo(
            query=state.query,
            content=all_content,
            client=self.client,
            cached_facts=cached_facts_str,
            context=self._context,
            active_step=t_step,
            trace_ctx=self._trace_ctx,
        )
        if t_step:
            self._telemetry.end_step(t_step)

        # Store complexity for synthesis tier selection
        is_simple = assessment.get("complexity") == "simple"
        self._is_simple_query = is_simple
        state.query_classification = {
            "type": "simple" if is_simple else "complex",
            "complexity": 2 if is_simple else 4,
            "llm_classified": True,
        }

        # Check if we can answer directly from cached facts
        can_answer_from_facts = assessment.get("can_answer_from_facts", False)
        relevant_facts_used = assessment.get("relevant_facts", [])

        if can_answer_from_facts and relevant_facts_used:
            await self._emit_step_async(
                state,
                StepType.FINDING,
                f"Can answer from {len(relevant_facts_used)} cached facts - skipping document analysis",
            )
            # Store the relevant facts as evidence for synthesis
            state.findings["accumulated_facts"] = relevant_facts_used
            state.findings["answered_from_cache"] = True
            # Proceed directly to synthesis
            await self._synthesize(state, is_simple)
            return

        can_answer_from_docs = assessment.get("can_answer_from_docs", True)
        gap = assessment.get("gap", "")

        # Emit plan event for direct answer assessment
        await self._emit_step_async(
            state, StepType.PLAN, "Assessment complete",
            details={
                "leads": [{"id": l.id, "type": l.lead_type, "description": l.description} for l in state.leads],
                "success_criteria": "",
                "key_issues": [],
                "strategy": f"{'Quick answer' if is_simple else 'In-depth analysis'} — "
                           f"{'reviewing available documents' if can_answer_from_docs else f'also searching legal databases for: {gap}'}",
                "iteration": 1,
            },
        )

        # Step 3: Research agent (replaces legacy two-phase external-search flow)
        if not can_answer_from_docs and self.config.enable_external_search and self.external_search:
            research_context = ResearchContext(
                gap=gap or "",
                reasoning=assessment.get("reasoning", "") or "",
                cached_facts=state.findings.get("accumulated_facts", [])[:15],
                triggers_summary="",
                source_path="small_repo",
            )
            await self._run_research_agent(state, research_context)
        elif can_answer_from_docs:
            await self._emit_step_async(state, StepType.THINKING, "Proceeding with documents only (no external search needed)", visible=False)

        # Step 5: Synthesize (citations already added inside _execute_external_searches)
        state.findings["mode"] = "direct_answer"
        await self._synthesize(state, is_simple)

    def _format_results_summary(self) -> str:
        """Format external research results for sufficiency check."""
        parts = []

        case_law = self._external_research.get("case_law", [])
        if case_law:
            case_summaries = []
            for c in case_law[:5]:
                name = c.get("case_name", "Unknown")
                citation = c.get("citation", "N/A")
                snippet = (c.get("snippet") or c.get("opinion_text") or "")[:200]
                case_summaries.append(f"- {name} ({citation}): {snippet}...")
            parts.append("CASE LAW:\n" + "\n".join(case_summaries))

        web = self._external_research.get("web", [])
        if web:
            web_summaries = []
            for r in web[:5]:
                title = r.get("title", "Untitled")
                content = (r.get("content") or "")[:200]
                web_summaries.append(f"- {title}: {content}...")
            parts.append("WEB RESULTS:\n" + "\n".join(web_summaries))

        return "\n\n".join(parts) if parts else "No results found."

    def _add_external_citations(self, state: InvestigationState):
        """Commit all external search results directly to state.citations.

        Deduplicates by case_name (case law) and url (web).  Called from
        _execute_external_searches() so all search rounds are captured.
        InlineCitationService handles downstream selection and trimming.
        """
        if not self._external_research:
            return

        seen_cases = {c.document for c in state.citations if c.source_type == "case_law"}
        for case in self._external_research.get("case_law", []):
            doc_name = f"[Case Law] {case.get('case_name', 'Unknown Case')}"
            if doc_name not in seen_cases:
                citation = state.add_citation(
                    document=doc_name,
                    page=None,
                    text=case.get('snippet', '') or case.get('opinion_text', '') or '',
                    context=f"Citation: {case.get('citation', 'N/A')} | Court: {case.get('court', 'N/A')}",
                    relevance="External case law research",
                    url=case.get('url'),
                    mime=case.get('mime'),
                    source_type="case_law",
                )
                if citation and self.on_citation:
                    self.on_citation(citation)
                seen_cases.add(doc_name)

        seen_web = {c.url for c in state.citations if c.source_type == "web"}
        for result in self._external_research.get("web", []):
            if result.get("url") not in seen_web:
                citation = state.add_citation(
                    document=f"[Web] {result.get('title', 'Unknown Source')}",
                    page=None,
                    text=result.get('content', '') or '',
                    context=f"URL: {result.get('url', 'N/A')}",
                    relevance="External regulatory research",
                    url=result.get('url'),
                    mime=result.get('mime'),
                    source_type="web",
                )
                if citation and self.on_citation:
                    self.on_citation(citation)
                seen_web.add(result.get("url"))

    def _format_external_research(self) -> dict[str, str]:
        """Format external research results for synthesis prompts.

        Returns:
            Dict with 'case_law' and 'web' keys containing formatted text
        """
        result = {"case_law": "", "web": ""}

        if not self._external_research:
            return result

        # Format case law results
        case_law = self._external_research.get("case_law", [])
        if case_law:
            case_lines = []
            for c in case_law[:100]:
                snippet = c.get('snippet') or c.get('opinion_text') or 'No snippet available'
                case_lines.append(
                    f"- **{c.get('case_name', 'Unknown')}** ({c.get('citation') or 'No citation'})\n"
                    f"  Court: {c.get('court', 'Unknown')} | Date: {c.get('date_filed', 'Unknown')}\n"
                    f"  Snippet: {snippet[:300]}..."
                )
            result["case_law"] = "\n\n".join(case_lines)

            # Include analysis if available
            analysis = self._external_research.get("analysis", {}).get("case_law", {})
            if analysis:
                summary = analysis.get("summary", "")
                if summary:
                    result["case_law"] += f"\n\n**Legal Standards Identified:** {summary}"

        # Format web results
        web = self._external_research.get("web", [])
        if web:
            web_lines = []
            for r in web[:30]:
                web_lines.append(
                    f"- **{r.get('title', 'Untitled')}**\n"
                    f"  URL: {r.get('url', '')}\n"
                    f"  Content: {r.get('content', '')[:300]}..."
                )
            result["web"] = "\n\n".join(web_lines)

            # Include Tavily's AI answer if available
            if self._external_research.get("web_answer"):
                result["web"] = f"**Summary:** {self._external_research['web_answer']}\n\n" + result["web"]

            # Include analysis if available
            analysis = self._external_research.get("analysis", {}).get("web", {})
            if analysis:
                summary = analysis.get("summary", "")
                if summary:
                    result["web"] += f"\n\n**Regulatory Context:** {summary}"

        return result


    async def _run_research_agent(
        self,
        state: InvestigationState,
        context: ResearchContext,
    ) -> None:
        """Run the tool-calling research agent once, persisting everything to state/store.

        Replaces the legacy keyword-routed two-phase external-search flow
        (`_execute_external_searches` + `_check_if_external_needed` +
        `check_search_sufficiency`). Used by both small-repo and large-repo
        paths — caller builds the :class:`ResearchContext`.
        """
        if not self.external_search or not self.config.enable_external_search:
            return

        emitter = ResearchEmitter(
            emit_lead_started=self._emit_lead_started,
            emit_lead_update=self._emit_lead_update,
            emit_lead_done=self._emit_lead_done,
            on_citation=self.on_citation,
        )
        agent_cfg = ResearchAgentConfig(
            max_turns=self.config.max_research_turns,
            max_actions_per_turn=self.config.max_research_actions_per_turn,
            per_tool_timeout_s=self.config.research_tool_timeout_s,
            turn_timeout_s=self.config.research_turn_timeout_s,
        )
        agent = ResearchAgent(
            client=self.client,
            external_search=self.external_search,
            emitter=emitter,
            external_research_store=self._external_research,
            config=agent_cfg,
            telemetry=self._telemetry,
            trace_ctx=self._trace_ctx,
        )
        try:
            await agent.run(state, context)
        except Exception as e:
            logger.warning("research agent crashed (context=%s): %s", context.source_path, e)

    async def _run_external_research_post_loop(self, state: InvestigationState) -> None:
        """Large-repo path: gate with LITE, then run the research agent once.

        Called after ``_investigate_loop`` exits so document facts + triggers
        are already accumulated. The LITE gate decides whether any external
        research should run at all.
        """
        if not self.external_search or not self.config.enable_external_search:
            return

        facts = state.findings.get("accumulated_facts", [])[:15]
        triggers_summary = ""
        if hasattr(state, "get_trigger_summary"):
            try:
                triggers_summary = state.get_trigger_summary() or ""
            except Exception:
                triggers_summary = ""

        t_gate = self._telemetry.begin_step("should_research_externally", "investigation_loop") if self._telemetry else None
        try:
            gate = await decisions.should_research_externally(
                query=state.query,
                facts=facts,
                triggers_summary=triggers_summary,
                client=self.client,
                active_step=t_gate,
                trace_ctx=self._trace_ctx,
            )
        except Exception as e:
            logger.warning("should_research_externally failed: %s", e)
            gate = {"needed": False, "reason": f"gate_error: {e}"}
        finally:
            if t_gate:
                self._telemetry.end_step(t_gate)

        if not gate.get("needed"):
            await self._emit_step_async(
                state, StepType.THINKING,
                f"External research skipped: {gate.get('reason', '')}",
                visible=False,
            )
            return

        context = ResearchContext(
            gap=gate.get("reason", ""),
            reasoning="Document extraction surfaced legal/regulatory triggers.",
            cached_facts=facts,
            triggers_summary=triggers_summary,
            source_path="large_repo",
        )
        await self._run_research_agent(state, context)

    async def _create_plan(self, state: InvestigationState, repo: MatterRepository):
        """Phase 1: Create investigation plan using LLM."""
        stats = repo.get_stats()

        await self._emit_step_async(state, StepType.THINKING, f"Analyzing {stats.total_files} files...")
        file_list = repo.get_file_list()

        # Format file list for LLM - show filenames so it can prioritize
        file_list_str = "\n".join(
            f"  - {f['filename']} ({f['size_kb']}KB, {f['type']})"
            for f in file_list[:50]  # Limit to 50 files for context
        )
        if len(file_list) > 50:
            file_list_str += f"\n  ... and {len(file_list) - 50} more files"

        # Use decisions layer for planning
        plan = await decisions.create_plan(
            query=state.query,
            file_list=file_list_str,
            total_files=stats.total_files,
            client=self.client,
            trace_ctx=self._trace_ctx,
        )

        state.hypothesis = plan.get("success_criteria", "Investigating query")
        state.findings["issues"] = plan.get("key_issues", [])
        state.findings["initial_plan"] = plan

        # Emit the reasoning (show LLM's thinking process)
        reasoning = plan.get("reasoning", "")
        challenges = plan.get("potential_challenges", "")
        key_issues = plan.get("key_issues", [])

        # PRIORITY: Create leads for priority files FIRST (read before searching)
        # Resolve display names to actual on-disk paths (LLM sees display names
        # but files may be hash-named on disk)
        name_to_path = {f["filename"]: f["path"] for f in file_list}
        priority_files = plan.get("priority_files", [])
        for filepath in priority_files[:3]:  # Limit to top 3 priority files
            if isinstance(filepath, str):
                resolved = name_to_path.get(filepath)
                if not resolved:
                    # Partial match: LLM may truncate or approximate filenames
                    for display_name, path in name_to_path.items():
                        if display_name.startswith(filepath.split("...")[0].rstrip(". ")):
                            resolved = path
                            break
                state.add_lead(
                    f"Read document: {resolved or filepath}",
                    source="initial_plan",
                )

        # Then create leads from search terms
        for term in plan.get("search_terms", [])[:3]:
            if isinstance(term, str):
                state.add_lead(f"Search for: {term}", source="initial_plan")

        # Fallback if no leads
        if not state.leads:
            terms = await decisions.extract_search_terms(state.query, self.client, trace_ctx=self._trace_ctx)
            for term in terms[:2]:
                state.add_lead(f"Search for: {term}", source="fallback")

        # Emit structured plan event
        await self._emit_step_async(
            state, StepType.PLAN,
            f"Investigation plan — {len(state.leads)} leads",
            details={
                "leads": [{"id": l.id, "type": l.lead_type, "description": l.description} for l in state.leads],
                "success_criteria": plan.get("success_criteria", ""),
                "key_issues": plan.get("key_issues", []),
                "strategy": plan.get("reasoning", ""),
                "iteration": 1,
            },
        )

    async def _assess_and_create_plan(
        self,
        state: InvestigationState,
        repo: MatterRepository,
    ) -> dict:
        """Unified assessment and planning for large repositories.

        Combines complexity classification and planning into one LLM call.
        Also checks if cached facts can answer the query.

        Returns:
            Assessment dict with can_answer_from_facts, complexity, plan details
        """
        stats = repo.get_stats()

        await self._emit_step_async(state, StepType.THINKING, f"Analyzing {stats.total_files} files...")
        file_list = repo.get_file_list()

        # Format file list for LLM - show filenames so it can prioritize
        file_list_str = "\n".join(
            f"  - {f['filename']} ({f['size_kb']}KB, {f['type']})"
            for f in file_list[:50]  # Limit to 50 files for context
        )
        if len(file_list) > 50:
            file_list_str += f"\n  ... and {len(file_list) - 50} more files"

        # Get cached facts for this query
        cached_facts_str = ""
        if self.fact_store and len(self.fact_store) > 0:
            relevant_facts = self.fact_store.get_relevant(state.query)
            if relevant_facts:
                cached_facts_str = self.fact_store.format_for_llm(relevant_facts)
                await self._emit_step_async(
                    state,
                    StepType.THINKING,
                    f"Found {len(relevant_facts)} potentially relevant cached facts",
                )

        # Use unified assess_and_plan
        t_step = self._telemetry.begin_step("planning", "planning") if self._telemetry else None
        assessment = await decisions.assess_and_plan(
            query=state.query,
            file_list=file_list_str,
            total_files=stats.total_files,
            client=self.client,
            cached_facts=cached_facts_str,
            context=self._context,
            active_step=t_step,
            trace_ctx=self._trace_ctx,
        )
        if t_step:
            self._telemetry.end_step(t_step)

        # If can answer from facts, return early (caller handles synthesis)
        if assessment.get("can_answer_from_facts", False):
            return assessment

        # Store plan info in state (same as _create_plan)
        state.hypothesis = assessment.get("success_criteria", "Investigating query")
        state.findings["issues"] = assessment.get("key_issues", [])
        state.findings["initial_plan"] = assessment

        # PRIORITY: Create leads for priority files FIRST (read before searching)
        priority_files = assessment.get("priority_files", [])
        for filepath in priority_files[:3]:  # Limit to top 3 priority files
            if isinstance(filepath, str):
                state.add_lead(f"Read document: {filepath}", source="initial_plan")

        # Then create leads from search terms
        for term in assessment.get("search_terms", [])[:3]:
            if isinstance(term, str):
                state.add_lead(f"Search for: {term}", source="initial_plan")

        # Fallback if no leads
        if not state.leads:
            terms = await decisions.extract_search_terms(state.query, self.client, trace_ctx=self._trace_ctx)
            for term in terms[:2]:
                state.add_lead(f"Search for: {term}", source="fallback")

        # Emit structured plan event
        await self._emit_step_async(
            state, StepType.PLAN,
            f"Investigation plan — {len(state.leads)} leads",
            details={
                "leads": [{"id": l.id, "type": l.lead_type, "description": l.description} for l in state.leads],
                "success_criteria": assessment.get("success_criteria", ""),
                "key_issues": assessment.get("key_issues", []),
                "strategy": assessment.get("reasoning", ""),
                "iteration": 1,
            },
        )

        return assessment

    async def _execute_external_searches(
        self,
        state: InvestigationState,
        case_law_queries: list[str] = None,
        web_queries: list[str] = None,
    ):
        """Execute external searches (case law, web) with parallel query support.

        This mimics how a real lawyer works:
        1. First, read the repository documents
        2. Based on what's found, identify if external research is needed
        3. Only then search case law or web for specific legal questions

        External searches are TOOLS, not mandatory steps:
        - Case law: Use when legal precedent questions arise from the documents
        - Web search: Use when regulations/standards need verification

        Supports:
        - Configurable query limits (max_case_law_queries, max_web_queries)
        - Parallel execution (parallel_external_searches=True)
        - Tiered/iterative queries (can be called multiple times with new queries)
        """
        # Get queries from plan if not provided directly
        if case_law_queries is None and web_queries is None:
            plan = state.findings.get("initial_plan", {})
            case_law_queries = plan.get("case_law_searches", [])
            web_queries = plan.get("web_searches", [])

        # Only proceed if we have queries
        if not case_law_queries and not web_queries:
            self._emit_step(
                state,
                StepType.THINKING,
                "No external queries generated - local documents should suffice",
            )
            return

        # Initialize or extend external research storage
        if not hasattr(self, '_external_research') or self._external_research is None:
            self._external_research = {"case_law": [], "web": [], "analysis": {}}

        # Apply configurable limits
        case_law_queries = case_law_queries[:self.config.max_case_law_queries] if case_law_queries else []
        web_queries = web_queries[:self.config.max_web_queries] if web_queries else []

        total_queries = len(case_law_queries) + len(web_queries)
        self._emit_step(
            state, StepType.SEARCH,
            f"Executing {total_queries} external searches ({len(case_law_queries)} case law, {len(web_queries)} web)",
            visible=False,
        )

        import time as _time

        # Create a telemetry step for the entire external search batch
        t_step_ext = self._telemetry.begin_step("external_search", "investigation_loop") if self._telemetry else None

        # Helper for case law search
        async def search_case_law(query: str) -> tuple[str, list]:
            t0 = _time.monotonic()
            try:
                cases = await self.external_search.search_case_law(
                    query,
                    max_results=self.config.max_case_law_results
                )
                cases = cases or []
                if t_step_ext:
                    t_step_ext.add_operation(StepOperation(
                        type="ext_search",
                        latency_ms=int((_time.monotonic() - t0) * 1000),
                        service="courtlistener",
                        query=query,
                        result_count=len(cases),
                    ))
                return query, cases
            except Exception as e:
                logger.warning(f"Case law search failed for '{query}': {e}")
                return query, []

        # Helper for web search
        # Tavily pricing: 1 credit = $0.008 (basic=1 credit, advanced=2 credits)
        _TAVILY_USD_PER_CREDIT = 0.008

        async def search_web(query: str) -> tuple[str, dict]:
            t0 = _time.monotonic()
            try:
                result_data = await self.external_search.search_web(
                    query,
                    max_results=self.config.max_web_results
                )
                result_data = result_data or {}
                web_results = result_data.get("results", [])
                usage_raw = result_data.get("usage")
                credits_used = (usage_raw.get("credits", 0) if isinstance(usage_raw, dict) else 0)
                cost_usd = credits_used * _TAVILY_USD_PER_CREDIT
                if t_step_ext:
                    t_step_ext.add_operation(StepOperation(
                        type="ext_search",
                        latency_ms=int((_time.monotonic() - t0) * 1000),
                        service="tavily",
                        query=query,
                        result_count=len(web_results),
                        usage_raw=usage_raw,
                        cost_usd=cost_usd,
                    ))
                return query, result_data
            except Exception as e:
                logger.warning(f"Web search failed for '{query}': {e}")
                return query, {}

        # Execute searches (parallel or sequential)
        if self.config.parallel_external_searches and self.external_search:
            # Parallel execution with asyncio.gather
            tasks = []
            if case_law_queries:
                tasks.extend([search_case_law(q) for q in case_law_queries])
            if web_queries:
                tasks.extend([search_web(q) for q in web_queries])

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results — each query is its own lead
            case_law_count = len(case_law_queries)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"External search task failed: {result}")
                    continue

                query, data = result
                if i < case_law_count:
                    # Case law result — create lead
                    ext_lead = Lead.create(f"CaseLaw: {query}", source="external_search")
                    ext_lead.lead_type = "caselaw"
                    state.leads.append(ext_lead)
                    await self._emit_lead_started(state, ext_lead)
                    if data:
                        self._external_research["case_law"].extend(data)
                        items = [{"name": c.get("case_name", "Unknown"), "citation": c.get("citation", ""), "snippet": (c.get("snippet") or "")[:200], "url": c.get("url", "")} for c in data]
                        await self._emit_lead_update(state, ext_lead.id, "external_results", {
                            "source": "caselaw", "count": len(data), "items": items,
                        })
                    await self._emit_lead_done(state, ext_lead.id)
                else:
                    # Web result — create lead
                    ext_lead = Lead.create(f"Web: {query}", source="external_search")
                    ext_lead.lead_type = "web"
                    state.leads.append(ext_lead)
                    await self._emit_lead_started(state, ext_lead)
                    web_results = data.get("results", [])
                    if web_results:
                        self._external_research["web"].extend(web_results)
                        items = [{"type": "web", "name": r.get("title", "Untitled"), "title": r.get("title", "Untitled"), "snippet": (r.get("content") or "")[:200], "url": r.get("url", "")} for r in web_results]
                        await self._emit_lead_update(state, ext_lead.id, "external_results", {
                            "source": "web", "count": len(web_results), "items": items,
                        })
                    if data.get("answer"):
                        self._external_research["web_answer"] = data["answer"]
                    await self._emit_lead_done(state, ext_lead.id)
        else:
            # Sequential execution (fallback)
            if case_law_queries and self.external_search:
                for query in case_law_queries:
                    ext_lead = Lead.create(f"CaseLaw: {query}", source="external_search")
                    ext_lead.lead_type = "caselaw"
                    state.leads.append(ext_lead)
                    await self._emit_lead_started(state, ext_lead)
                    _, cases = await search_case_law(query)
                    if cases:
                        self._external_research["case_law"].extend(cases)
                        items = [{"name": c.get("case_name", "Unknown"), "citation": c.get("citation", ""), "snippet": (c.get("snippet") or "")[:200], "url": c.get("url", "")} for c in cases]
                        await self._emit_lead_update(state, ext_lead.id, "external_results", {
                            "source": "caselaw", "count": len(cases), "items": items,
                        })
                    await self._emit_lead_done(state, ext_lead.id)

            if web_queries and self.external_search:
                for query in web_queries:
                    ext_lead = Lead.create(f"Web: {query}", source="external_search")
                    ext_lead.lead_type = "web"
                    state.leads.append(ext_lead)
                    await self._emit_lead_started(state, ext_lead)
                    _, result_data = await search_web(query)
                    web_results = result_data.get("results", [])
                    if web_results:
                        self._external_research["web"].extend(web_results)
                        items = [{"name": r.get("title", "Untitled"), "snippet": (r.get("content") or "")[:200], "url": r.get("url", "")} for r in web_results]
                        await self._emit_lead_update(state, ext_lead.id, "external_results", {
                            "source": "web", "count": len(web_results), "items": items,
                        })
                    if result_data.get("answer"):
                        self._external_research["web_answer"] = result_data["answer"]
                    await self._emit_lead_done(state, ext_lead.id)

        # Finalize external search telemetry step
        if t_step_ext:
            self._telemetry.end_step(t_step_ext)

        # CONSOLIDATED: Analyze all external results in one call
        if self._external_research.get("case_law") or self._external_research.get("web"):
            # Format case law results
            case_law_text = ""
            if self._external_research.get("case_law"):
                case_law_text = "\n\n".join([
                    f"**{c.get('case_name', 'Unknown')}** ({c.get('citation') or 'No citation'})\n"
                    f"Court: {c.get('court', 'Unknown')}\nDate: {c.get('date_filed', 'Unknown')}\n"
                    f"Snippet: {(c.get('snippet') or c.get('opinion_text', ''))[:500] if c.get('snippet') or c.get('opinion_text') else 'No summary'}"
                    for c in self._external_research["case_law"][:100]
                ])

            # Format web results
            web_text = ""
            if self._external_research.get("web"):
                web_text = "\n\n".join([
                    f"**{r.get('title', 'Untitled')}**\nURL: {r.get('url', '')}\n"
                    f"Content: {r.get('content', 'No content')[:500]}"
                    for r in self._external_research["web"][:30]
                ])

            # Single consolidated call replaces analyze_case_law_results + analyze_web_results
            t_step_ae = self._telemetry.begin_step("analyze_external", "investigation_loop") if self._telemetry else None
            analysis = await decisions.analyze_external(
                query=state.query,
                case_law_results=case_law_text,
                web_results=web_text,
                client=self.client,
                active_step=t_step_ae,
                trace_ctx=self._trace_ctx,
            )
            if t_step_ae:
                self._telemetry.end_step(t_step_ae)

            # Store analysis in both locations for backwards compatibility
            self._external_research["analysis"]["case_law"] = {
                "key_precedents": analysis.get("key_precedents", []),
                "legal_standards": analysis.get("legal_standards", []),
                "summary": analysis.get("summary", ""),
            }
            self._external_research["analysis"]["web"] = {
                "regulations": analysis.get("regulations", []),
                "standards": analysis.get("regulatory_standards", []),
                "summary": analysis.get("summary", ""),
            }
            self._external_research["analysis"]["combined"] = analysis.get("combined_framework", "")

            # Emit external analysis as a lead.update on a dedicated analysis lead
            ext_analysis_lead = Lead.create("External research analysis", source="external_search")
            ext_analysis_lead.lead_type = "search"
            state.leads.append(ext_analysis_lead)
            await self._emit_lead_started(state, ext_analysis_lead)
            await self._emit_lead_update(state, ext_analysis_lead.id, "analysis", {
                "summary": analysis.get("summary", ""),
                "key_precedents": analysis.get("key_precedents", []),
                "regulations": analysis.get("regulations", []),
                "legal_standards": analysis.get("legal_standards", []),
                "combined_framework": analysis.get("combined_framework", ""),
            })
            await self._emit_lead_done(state, ext_analysis_lead.id)

        # Store in state findings for reference
        state.findings["external_research"] = self._external_research

        # Commit external results directly to state.citations
        self._add_external_citations(state)

    async def _investigate_loop(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        cache: InvestigationCache,
    ):
        """Phase 2: Iterative investigation with continuous recalibration.

        Mimics how a lawyer works:
        1. Follow leads, read documents, extract facts
        2. Continuously assess: Do we have enough? Should we change approach?
        3. Dynamically decide if external research is needed based on what we find
        4. Stop when we have sufficient evidence
        """
        iteration = 0
        executed_external_queries = set()  # Track executed queries for tiered search

        while iteration < self.config.max_iterations:
            state.recursion_depth = iteration + 1

            pending_leads = state.get_pending_leads()
            if not pending_leads:
                await self._emit_step_async(state, StepType.THINKING, "No more leads to investigate")
                break

            # Take leads to process
            leads_to_process = pending_leads[:self.config.max_leads_per_level]

            lead_lines = [f"  - {l.description} (source: {l.source})" for l in leads_to_process]
            remaining = len(pending_leads) - len(leads_to_process)
            remaining_str = f" ({remaining} more leads queued)" if remaining > 0 else ""
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"Iteration {iteration + 1} — processing {len(leads_to_process)} leads{remaining_str}:\n" + "\n".join(lead_lines),
            )

            # Process leads in parallel
            tasks = [self._investigate_lead(state, repo, lead, cache) for lead in leads_to_process]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Log read failures but DON'T abort - keep trying other documents
            if cache.consecutive_read_failures >= cache.MAX_CONSECUTIVE_FAILURES:
                await self._emit_step_async(
                    state,
                    StepType.THINKING,
                    f"Note: {cache.consecutive_read_failures} consecutive read failures (some files may be in subdirectories). Continuing with other documents...",
                    visible=False,
                )
                # Reset counter to allow more attempts - don't abort investigation
                cache.consecutive_read_failures = 0
                state.findings["had_read_failures"] = True

            iteration += 1
            facts_count = len(state.findings.get("accumulated_facts", []))
            findings_summary = self._format_findings(state)
            plan_summary = state.findings.get("initial_plan", {}).get("reasoning", "")

            # === CONSOLIDATED CHECKPOINT ===
            # Single LLM call replaces is_sufficient + should_replan
            if facts_count >= self.config.early_exit_facts or iteration > 1:
                # Get cached facts for checkpoint evaluation
                cached_facts_str = ""
                if self.fact_store and len(self.fact_store) > 0:
                    relevant_facts = self.fact_store.get_relevant(state.query)
                    if relevant_facts:
                        cached_facts_str = self.fact_store.format_for_llm(relevant_facts)

                t_step_ck = self._telemetry.begin_step("sufficiency_check", "investigation_loop") if self._telemetry else None
                checkpoint_result = await decisions.checkpoint(
                    query=state.query,
                    findings=findings_summary,
                    plan=plan_summary,
                    client=self.client,
                    cached_facts=cached_facts_str,
                    active_step=t_step_ck,
                    trace_ctx=self._trace_ctx,
                )
                if t_step_ck:
                    self._telemetry.end_step(t_step_ck)

                # Check sufficiency - MUST have read at least 1 document
                # Facts from 0 docs is logically impossible for document investigation
                docs_read = state.documents_read
                progress = checkpoint_result.get("progress_assessment", "")
                is_sufficient = checkpoint_result.get("sufficient", False)
                should_replan = checkpoint_result.get("should_replan", False)

                await self._emit_step_async(
                    state, StepType.CHECKPOINT, "Sufficiency check",
                    details={
                        "decision": "sufficient" if (is_sufficient and docs_read > 0) else "insufficient",
                        "total_facts": facts_count,
                        "docs_read": docs_read,
                        "reasoning": progress,
                    },
                )
                if is_sufficient and docs_read > 0:
                    break
                elif is_sufficient and docs_read == 0:
                    logger.warning("Checkpoint claims sufficient but 0 docs read — continuing")

                # Handle replanning if needed
                if should_replan and pending_leads:
                    new_search_terms = checkpoint_result.get("new_search_terms", [])[:3]
                    files_to_check = checkpoint_result.get("files_to_check", [])[:2]

                    # Add new leads from checkpoint
                    newly_added_leads = []
                    for term in new_search_terms:
                        if not cache.is_similar_search(term):
                            new_lead = state.add_lead(f"Search for: {term}", source="checkpoint")
                            if new_lead:
                                newly_added_leads.append(new_lead)
                    for filepath in files_to_check:
                        if not cache.has_extracted(filepath) and not cache.is_irrelevant(filepath):
                            new_lead = state.add_lead(f"Read document: {filepath}", source="checkpoint")
                            if new_lead:
                                newly_added_leads.append(new_lead)

                    if newly_added_leads:
                        await self._emit_step_async(
                            state, StepType.REPLAN, f"Replan — added {len(newly_added_leads)} new leads",
                            details={
                                "new_leads": [{"id": l.id, "type": l.lead_type, "description": l.description} for l in newly_added_leads],
                                "iteration": iteration + 1,
                            },
                        )

            # 3. External research is no longer triggered per iteration; it
            # runs once after the investigate loop finishes (see
            # _run_external_research_post_loop). Kept here for documentation.

            # Save checkpoint periodically
            if self.config.checkpoint_dir and iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint(state, iteration)

    async def _check_if_external_needed(
        self,
        state: InvestigationState,
        executed_queries: set[str] = None,
    ) -> dict | None:
        """Check if accumulated triggers suggest we need external legal research.

        Uses trigger-based approach:
        1. Check if meaningful triggers have been accumulated from documents
        2. If yes, use LITE to generate specific queries using those triggers
        3. Filter out already-executed queries for tiered search support
        4. If no new queries, return None

        Args:
            state: Current investigation state
            executed_queries: Set of already-executed query strings to skip

        Returns:
            Dict with case_law_queries and web_queries, or None if no new queries needed.
        """
        executed_queries = executed_queries or set()

        # Must have read some documents first
        if state.documents_read < 2:
            return None

        # Check if we have meaningful triggers from document analysis
        if not state.has_external_triggers(min_triggers=2):
            # No triggers accumulated - skip external search entirely
            return None

        facts = state.findings.get("accumulated_facts", [])
        entities = [e.name for e in state.entities.values()][:10] if state.entities else []
        triggers = state.get_trigger_summary()

        # Parse triggers into a list for better formatting
        trigger_list = [t.strip() for t in triggers.split(",") if t.strip()]
        self._emit_step(
            state,
            StepType.THINKING,
            f"Checking external search need — {len(trigger_list)} triggers accumulated from {state.documents_read} docs:\n"
            + "\n".join(f"  - {t}" for t in trigger_list),
        )

        # Generate specific queries using accumulated triggers
        t_step_eq = self._telemetry.begin_step("generate_external_queries", "investigation_loop") if self._telemetry else None
        result = await decisions.generate_external_queries(
            query=state.query,
            facts=facts,
            entities=entities,
            client=self.client,
            triggers=triggers,
            active_step=t_step_eq,
            trace_ctx=self._trace_ctx,
        )
        if t_step_eq:
            self._telemetry.end_step(t_step_eq)

        case_law_queries = result.get("case_law_queries", [])
        web_queries = result.get("web_queries", [])

        # Filter out already-executed queries (for tiered search support)
        new_case_law = [q for q in case_law_queries if q not in executed_queries]
        new_web = [q for q in web_queries if q not in executed_queries]

        # If any NEW queries were generated, return them
        if new_case_law or new_web:
            # Also store in plan for reference
            if "initial_plan" not in state.findings:
                state.findings["initial_plan"] = {}

            # Accumulate queries (don't replace)
            existing_case_law = state.findings["initial_plan"].get("case_law_searches", [])
            existing_web = state.findings["initial_plan"].get("web_searches", [])
            state.findings["initial_plan"]["case_law_searches"] = list(set(existing_case_law + case_law_queries))
            state.findings["initial_plan"]["web_searches"] = list(set(existing_web + web_queries))

            query_lines = []
            for q in new_case_law:
                query_lines.append(f"  Case law: \"{q}\"")
            for q in new_web:
                query_lines.append(f"  Web: \"{q}\"")
            # Also show what was filtered out
            filtered_case = [q for q in case_law_queries if q in executed_queries]
            filtered_web = [q for q in web_queries if q in executed_queries]
            if filtered_case or filtered_web:
                query_lines.append(f"  (Filtered {len(filtered_case) + len(filtered_web)} already-executed queries)")
            self._emit_step(
                state,
                StepType.THINKING,
                f"Generated {len(new_case_law) + len(new_web)} new external queries:\n" + "\n".join(query_lines),
            )
            return {"case_law_queries": new_case_law, "web_queries": new_web}

        return None

    async def _investigate_lead(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        lead: Lead,
        cache: InvestigationCache,
    ):
        """Investigate a single lead with caching."""
        if state.recursion_depth > self.config.max_depth:
            state.mark_lead_investigated(lead.id, "Max depth reached")
            return

        if state.recursion_depth > state.max_depth_reached:
            state.max_depth_reached = state.recursion_depth

        # Determine if this is a search or read lead
        if lead.description.startswith("Read document:"):
            filepath = lead.description.replace("Read document:", "").strip()

            # OPTIMIZATION: Skip if already extracted or marked irrelevant
            if cache.has_extracted(filepath):
                await self._emit_step_async(state, StepType.THINKING, f"Skip read (already extracted): {filepath}", visible=False)
                state.mark_lead_investigated(lead.id, "Already extracted")
                return
            if cache.is_irrelevant(filepath):
                await self._emit_step_async(state, StepType.THINKING, f"Skip read (marked irrelevant): {filepath}", visible=False)
                state.mark_lead_investigated(lead.id, "Marked irrelevant")
                return

            await self._emit_lead_started(state, lead)
            try:
                await self._read_document(state, repo, filepath, cache, lead_id=lead.id)
                state.mark_lead_investigated(lead.id, "Document read")
            except Exception as e:
                await self._emit_lead_error(state, lead.id, str(e))
                raise
            await self._emit_lead_done(state, lead.id)
        else:
            # Extract search term
            search_term = self._extract_search_term(lead.description)

            # OPTIMIZATION: Skip similar searches
            if self.config.skip_similar_searches and cache.is_similar_search(search_term):
                await self._emit_step_async(state, StepType.THINKING, f"Skip search (similar already done): \"{search_term}\"", visible=False)
                state.mark_lead_investigated(lead.id, f"Similar search already done")
                return

            await self._emit_lead_started(state, lead)
            try:
                cache.add_search(search_term)

                # Perform search (using smart_search for OR fallback)
                # Run in thread to avoid blocking the asyncio event loop —
                # large repositories can produce 10k+ hits, and the sync
                # search/rank work would starve heartbeat delivery.
                results = await asyncio.to_thread(repo.smart_search, search_term, context_lines=2)
                state.searches_performed += 1

                if not results.hits:
                    await self._emit_lead_update(state, lead.id, "matches", {
                        "query": search_term, "match_count": 0, "docs": [],
                    })
                    state.mark_lead_investigated(lead.id, "No results found")
                    await self._emit_lead_done(state, lead.id)
                    return

                # Build match details
                doc_hit_counts = {}
                for hit in results.hits:
                    name = Path(hit.file_path).name
                    doc_hit_counts[name] = doc_hit_counts.get(name, 0) + 1
                await self._emit_lead_update(state, lead.id, "matches", {
                    "query": search_term,
                    "match_count": len(results.hits),
                    "docs": [{"name": name, "hit_count": count} for name, count in doc_hit_counts.items()],
                })

                # CONSOLIDATED: Single analyze_search call replaces pick_relevant_hits + analyze_results + prioritize_documents
                await self._analyze_results_consolidated(state, repo, results, cache, lead_id=lead.id)

                state.mark_lead_investigated(lead.id, f"Found {len(results.hits)} matches")
            except Exception as e:
                await self._emit_lead_error(state, lead.id, str(e))
                raise
            await self._emit_lead_done(state, lead.id)

    async def _analyze_results_consolidated(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        results: SearchResults,
        cache: InvestigationCache,
        lead_id: Optional[str] = None,
    ):
        """Consolidated search analysis - single FLASH call replaces 3 separate calls."""
        key_issues = state.findings.get("issues", [])
        already_read = list(cache.extracted_docs)

        # Single consolidated call: pick_relevant_hits + analyze_results + prioritize_documents
        t_step_as = self._telemetry.begin_step("analyze_search", "investigation_loop") if self._telemetry else None
        analysis = await decisions.analyze_search(
            query=state.query,
            key_issues=key_issues,
            results=results,
            already_read=already_read,
            client=self.client,
            active_step=t_step_as,
            trace_ctx=self._trace_ctx,
        )
        if t_step_as:
            self._telemetry.end_step(t_step_as)

        # Store facts and emit per-fact updates
        facts = analysis.get("facts", [])
        state.add_facts(facts)
        if lead_id:
            for f in facts:
                await self._emit_lead_update(state, lead_id, "fact", {"fact": f, "source_doc": results.query})

        # Emit rankings
        ranked_docs = analysis.get("ranked_documents", [])
        if lead_id:
            for doc in ranked_docs:
                filepath = doc.get("file") if isinstance(doc, dict) else doc
                crit = doc.get("criticality", "?") if isinstance(doc, dict) else "?"
                await self._emit_lead_update(state, lead_id, "ranking", {"doc": filepath, "criticality": crit})

        read_deeper = analysis.get("read_deeper", [])[:2]
        additional_searches = analysis.get("additional_searches", [])[:1]

        # Add citations from relevant hits
        relevant_hits = analysis.get("relevant_hits", [])
        for hit in relevant_hits[:3]:
            # Get URL for the document if available
            doc_url = repo.get_document_url(hit.filename) if hasattr(repo, 'get_document_url') else None
            doc_mime = repo.get_document_mime(hit.filename) if hasattr(repo, 'get_document_mime') else None
            citation = state.add_citation(
                document=hit.file_path,
                page=hit.page_num,
                text=hit.match_text,
                context=hit.context,
                relevance=f"Found via search: {results.query}",
                url=doc_url,
                mime=doc_mime,
            )
            if citation and self.on_citation:
                self.on_citation(citation)

        # Add leads for docs to read deeper — with parent tracking
        for filepath in read_deeper:
            if isinstance(filepath, str) and not cache.has_extracted(filepath):
                new_lead = state.add_lead(f"Read document: {filepath}", source="analysis", parent_lead_id=lead_id)
                if new_lead and lead_id:
                    await self._emit_lead_update(state, lead_id, "spawned", {
                        "new_lead_id": new_lead.id, "type": new_lead.lead_type, "description": new_lead.description,
                    })

        # Add additional search leads — with parent tracking
        for term in additional_searches:
            if isinstance(term, str) and not cache.is_similar_search(term):
                new_lead = state.add_lead(f"Search for: {term}", source="analysis", parent_lead_id=lead_id)
                if new_lead and lead_id:
                    await self._emit_lead_update(state, lead_id, "spawned", {
                        "new_lead_id": new_lead.id, "type": new_lead.lead_type, "description": new_lead.description,
                    })

        # Get prioritized files from the consolidated analysis
        ranked_docs = analysis.get("ranked_documents", [])

        # Filter to unread, non-irrelevant files and extract top ones
        # Use set to deduplicate and preserve order
        top_files = []
        seen_files = set()
        for doc in ranked_docs:
            filepath = doc.get("file") if isinstance(doc, dict) else doc
            criticality = doc.get("criticality", "SUPPORTING") if isinstance(doc, dict) else "SUPPORTING"

            # Normalize criticality to uppercase for case-insensitive comparison
            criticality = criticality.upper() if isinstance(criticality, str) else "SUPPORTING"

            # Skip irrelevant documents and mark them in cache
            if criticality == "IRRELEVANT":
                if filepath:
                    cache.mark_irrelevant(filepath)
                continue

            # Skip duplicates, already-extracted, and previously marked irrelevant files
            if not filepath or filepath in seen_files or cache.has_extracted(filepath) or cache.is_irrelevant(filepath):
                continue

            seen_files.add(filepath)
            top_files.append(filepath)

            # Mark DECISIVE documents for potential pinning
            if criticality == "DECISIVE":
                if "pinned_documents" not in state.findings:
                    state.findings["pinned_documents"] = []
                if filepath not in state.findings["pinned_documents"]:
                    state.findings["pinned_documents"].append(filepath)

        await self._batch_read(state, repo, top_files[:self.config.parallel_reads], cache, lead_id=lead_id)

    async def _batch_read(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_paths: list[str],
        cache: InvestigationCache,
        lead_id: Optional[str] = None,
    ):
        """Read multiple documents in parallel."""
        # Filter out already extracted docs
        to_read = [fp for fp in file_paths if not cache.has_extracted(fp)]

        if not to_read:
            return

        doc_names = [Path(fp).name for fp in to_read]
        self._emit_step(
            state,
            StepType.READING,
            f"Reading: {_fmt_list(doc_names, 3, 30)}",
        )

        tasks = [self._read_document(state, repo, fp, cache, lead_id=lead_id) for fp in to_read]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _read_document(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_path: str,
        cache: InvestigationCache,
        lead_id: Optional[str] = None,
    ) -> bool:
        """Read and extract facts from a document.

        Returns:
            True if document was read successfully, False otherwise.
        """
        # OPTIMIZATION: Skip if already extracted
        if cache.has_extracted(file_path):
            logger.debug(f"Skipping already extracted: {file_path}")
            return True  # Already extracted = success

        filename = Path(file_path).name

        # Emit reading update
        if lead_id:
            await self._emit_lead_update(state, lead_id, "reading", {"doc": filename})
        else:
            await self._emit_step_async(state, StepType.READING, f"Reading: {filename}")

        try:
            doc, ocr_meta = await repo.read_async(file_path)
            state.documents_read += 1
            cache.mark_extracted(file_path)  # Mark as extracted
            cache.record_read_success()  # Reset consecutive failure counter

            # Attach OCR telemetry if Mistral was called for this file
            if ocr_meta is not None and self._telemetry:
                t_step_ocr = self._telemetry.begin_step("document_read_ocr", "investigation_loop")
                t_step_ocr.add_operation(StepOperation(
                    type="ocr",
                    latency_ms=ocr_meta.latency_ms,
                    service="mistral-ocr",
                    file_name=ocr_meta.file_name,
                    file_type=ocr_meta.file_type,
                    page_count=ocr_meta.page_count,
                    timed_out=ocr_meta.timed_out,
                    cost_usd=0.0,
                ))
                self._telemetry.end_step(t_step_ocr)

            # Dynamic excerpt limit based on query complexity
            excerpt_limit = (
                self.config.excerpt_chars_simple
                if getattr(self, '_is_simple_query', False)
                else self.config.excerpt_chars_complex
            )
            content = doc.get_excerpt(excerpt_limit)

            # Dynamic extraction limit (matching excerpt)
            extraction_limit = 10000 if getattr(self, '_is_simple_query', False) else 35000

            # Use decisions layer to extract facts
            t_step_ef = self._telemetry.begin_step("extract_facts", "investigation_loop") if self._telemetry else None
            extraction = await decisions.extract_facts(
                query=state.query,
                filename=doc.filename,
                content=content,
                client=self.client,
                max_content_chars=extraction_limit,
                active_step=t_step_ef,
                trace_ctx=self._trace_ctx,
            )
            if t_step_ef:
                self._telemetry.end_step(t_step_ef)

            # Store facts and emit per-fact updates
            facts = extraction.get("facts", [])
            state.add_facts(facts)
            if lead_id:
                for f in facts:
                    await self._emit_lead_update(state, lead_id, "fact", {"fact": f, "source_doc": doc.filename})
            elif facts:
                self._emit_step(state, StepType.FINDING, f"Extracted {len(facts)} facts from {doc.filename}")

            # Save facts to persistent store for future queries
            if self.fact_store:
                new_facts = self.fact_store.add_facts_from_extraction(
                    extraction=extraction,
                    source_filename=doc.filename,
                    query_context=state.query,
                )
                self._emit_step(
                    state, StepType.THINKING,
                    f"Added {new_facts} facts from {doc.filename} (store total: {len(self.fact_store)})",
                    visible=False,
                )

            # Accumulate external research triggers
            triggers = extraction.get("external_triggers", {})
            if triggers:
                added = state.add_triggers(triggers)
                if added > 0:
                    trigger_list = []
                    for category, items in triggers.items():
                        if isinstance(items, list) and items:
                            for item in items:
                                trigger_list.append(f"{category}: {item}")
                    if lead_id:
                        await self._emit_lead_update(state, lead_id, "triggers", {
                            "count": added, "triggers": trigger_list,
                        })

            # Emit insights from the extraction
            insights = extraction.get("insights", "")
            gaps = extraction.get("gaps", "")
            next_steps_text = extraction.get("next_steps", "")
            if lead_id and (insights or gaps):
                await self._emit_lead_update(state, lead_id, "insight", {
                    "learned": insights or None,
                    "gaps": gaps or None,
                    "next_steps": next_steps_text or None,
                })
            elif insights or gaps:
                insight_msg = f"From {doc.filename}:"
                if insights:
                    insight_msg += f" LEARNED: {insights}"
                if gaps:
                    insight_msg += f" GAPS: {gaps}"
                self._emit_step(state, StepType.THINKING, insight_msg)

            # Add citations from quotes (limit to 2)
            quotes = extraction.get("quotes", [])[:2]
            for quote in quotes:
                if isinstance(quote, dict) and "text" in quote:
                    page = quote.get("page")
                    relevance = quote.get("relevance", "Direct quote")
                    doc_url = repo.get_document_url(doc.filename) if hasattr(repo, 'get_document_url') else None
                    doc_mime = repo.get_document_mime(doc.filename) if hasattr(repo, 'get_document_mime') else None
                    citation = state.add_citation(
                        document=doc.path,
                        page=page,
                        text=quote["text"],
                        context="",
                        relevance=relevance,
                        url=doc_url,
                        mime=doc_mime,
                    )
                    if citation and self.on_citation:
                        self.on_citation(citation)

            # Skip adding reference leads - reduces iteration depth
            return True  # Success

        except Exception as e:
            self._emit_step(state, StepType.ERROR, f"Failed to read {file_path}: {e}")
            cache.record_read_failure()
            return False  # Failure

    async def _synthesize(self, state: InvestigationState, is_simple: bool = False):
        """Phase 3: Final synthesis using ALL sources.

        Sources:
        1. Local document evidence (facts extracted during investigation)
        2. Case law search (CourtListener)
        3. Web search (Tavily - regulations/standards)
        4. Pinned DECISIVE documents OR small repo full content
        """
        # CRITICAL: Block synthesis if no documents were read
        # Exception: Allow if answering from cached facts (we intentionally skipped reading)
        small_repo_content = state.findings.get("small_repo_content")
        answered_from_cache = state.findings.get("answered_from_cache", False)

        # Donot block synthesis when no documents were read
        # if state.documents_read == 0 and not small_repo_content and not answered_from_cache:
        #     self._emit_step(
        #         state,
        #         StepType.ERROR,
        #         "Cannot synthesize: 0 documents were successfully read. Check document paths and access.",
        #     )
        #     error_msg = (
        #         "**Investigation Failed**\n\n"
        #         "Unable to read any documents from the repository. This may indicate:\n"
        #         "- Documents were not downloaded correctly\n"
        #         "- File paths do not match between search index and storage\n"
        #         "- Files were cleaned up before investigation completed\n\n"
        #         f"Total read attempts that failed: multiple\n"
        #         f"Query: {state.query}"
        #     )
        #     state.findings["final_output"] = error_msg
        #     return

        # Note if there were read failures (for caveat in output)
        if state.findings.get("had_read_failures"):
            state.findings["read_failure_caveat"] = (
                f"Note: Some documents were inaccessible during investigation. "
                f"Analysis is based on {state.documents_read} successfully read documents."
            )

        # Count sources for informative message
        facts = state.findings.get("accumulated_facts", [])
        case_law_count = len(self._external_research.get("case_law", []))
        web_count = len(self._external_research.get("web", []))
        pinned_count = len(state.findings.get("pinned_documents", []))
        small_repo = bool(small_repo_content)

        source_parts = [f"{len(facts)} facts"]
        if case_law_count:
            source_parts.append(f"{case_law_count} cases")
        if web_count:
            source_parts.append(f"{web_count} web")
        if pinned_count and not small_repo:
            source_parts.append(f"{pinned_count} decisive docs")
        elif small_repo:
            source_parts.append("all docs (small repo)")

        synth_tier = "FLASH" if (is_simple and self.config.use_flash_for_simple) else "PRO"
        synth_start_time = time.monotonic()
        await self._emit_step_async(
            state, StepType.SYNTHESIS_STARTED, "Synthesizing",
            details={
                "fact_count": len(facts),
                "citation_count": len(state.citations),
                "case_law_count": case_law_count,
                "web_count": web_count,
                "model": synth_tier,
            },
        )

        # SOURCE 1: Local document evidence
        facts = state.findings.get("accumulated_facts", [])
        evidence = "\n".join(f"- {fact}" for fact in facts[:20])
        # Note: Citations tracked in state for UI/downstream, not passed to synthesis

        # SOURCE 4: Pinned content - either from small repo (all docs) or DECISIVE docs
        # (small_repo_content already fetched above for the guard check)
        if small_repo_content:
            # Small repo mode - all content already loaded
            pinned_content = f"=== ALL REPOSITORY DOCUMENTS ===\n{small_repo_content}"
            logger.info(f"Using small repo content: {len(small_repo_content)} chars")
        else:
            # Large repo mode - load DECISIVE pinned documents
            pinned_content = await self._load_pinned_documents(state)

        # SOURCE 3: External research (case law + web)
        external_formatted = self._format_external_research()
        case_law_text = external_formatted.get("case_law", "No case law found")
        web_text = external_formatted.get("web", "No web results found")

        # Track sources used
        sources_used = ["local_documents", "case_law", "web_search"]
        if pinned_content:
            sources_used.append("decisive_documents")
        state.findings["sources_used"] = sources_used

        # Choose tier: FLASH for simple queries, PRO for complex
        # FLASH model + PRO system prompt for simple synthesis
        from irys.core.models import ModelTier
        synthesis_tier = ModelTier.FLASH if (is_simple and self.config.use_flash_for_simple) else ModelTier.PRO
        if synthesis_tier == ModelTier.FLASH:
            logger.info("Using FLASH model with PRO system prompt for simple query synthesis")

        # Synthesize with materials only - PRO system prompt handles the rest
        t_step_syn = self._telemetry.begin_step("synthesize", "synthesis") if self._telemetry else None
        response = await decisions.synthesize(
            query=state.query,
            evidence=evidence,
            external_research=f"=== CASE LAW (CourtListener) ===\n{case_law_text}\n\n=== REGULATIONS/STANDARDS (Web) ===\n{web_text}",
            pinned_content=pinned_content,
            client=self.client,
            tier=synthesis_tier,
            context=self._context,
            active_step=t_step_syn,
            trace_ctx=self._trace_ctx,
        )
        if t_step_syn:
            self._telemetry.end_step(t_step_syn)

        state.findings["final_output"] = response

        output_len = len(response)
        total_citations = len(state.citations)
        total_facts = len(state.findings.get("accumulated_facts", []))
        synth_duration_ms = int((time.monotonic() - synth_start_time) * 1000)
        await self._emit_step_async(
            state, StepType.SYNTHESIS_COMPLETE, "Synthesis complete",
            details={
                "output_length": output_len,
                "duration_ms": synth_duration_ms,
                "docs_read": state.documents_read,
                "facts_used": total_facts,
                "citations": total_citations,
            },
        )

    async def _load_pinned_documents(self, state: InvestigationState) -> str:
        """Load content from DECISIVE pinned documents for synthesis.

        Respects a 100k character budget with max 30k per document.
        Returns formatted content string or empty string if none.
        """
        pinned_docs = state.findings.get("pinned_documents", [])
        if not pinned_docs:
            return ""

        # Safety check - repo must be set
        if not self.repo:
            logger.warning("Cannot load pinned documents - repo not initialized")
            return ""

        TOTAL_BUDGET = 100_000  # 100k total budget
        MAX_PER_DOC = 30_000   # Max 30k per document
        HEADER_OVERHEAD = 50   # Approximate overhead for "=== DECISIVE: filename ===" header

        pinned_content_parts = []
        budget_remaining = TOTAL_BUDGET
        docs_loaded = 0
        seen_files = set()  # Deduplicate pinned docs

        for filepath in pinned_docs:
            # Skip duplicates
            if filepath in seen_files:
                continue
            seen_files.add(filepath)

            if budget_remaining <= HEADER_OVERHEAD:
                logger.info(f"Pinned document budget exhausted, skipping remaining docs")
                break

            try:
                doc, _ = await self.repo.read_async(filepath)
                if doc:
                    # Get excerpt respecting both per-doc and remaining budget limits
                    # Account for header overhead in budget
                    max_chars = min(MAX_PER_DOC, budget_remaining - HEADER_OVERHEAD)
                    if max_chars <= 0:
                        break
                    excerpt = doc.get_excerpt(max_chars)

                    if excerpt:
                        filename = doc.filename or filepath.split("/")[-1].split("\\")[-1]
                        header = f"\n=== DECISIVE: {filename} ===\n"
                        pinned_content_parts.append(f"{header}{excerpt}")
                        budget_remaining -= (len(header) + len(excerpt))
                        docs_loaded += 1
                        logger.info(f"Loaded pinned document: {filename} ({len(excerpt)} chars)")
            except Exception as e:
                logger.info(f"WARNING: Failed to load pinned document {filepath}: {e}")

        if pinned_content_parts:
            total_chars = sum(len(p) for p in pinned_content_parts)
            doc_names = [Path(f).name for f in list(seen_files)[:docs_loaded]]
            self._emit_step(
                state,
                StepType.FINDING,
                f"Decisive docs ({total_chars:,} chars): {_fmt_list(doc_names, 3, 30)}"
            )

        return "".join(pinned_content_parts)

    def _emit_step(
        self,
        state: InvestigationState,
        step_type: StepType,
        content: str,
        details: Optional[dict] = None,
        visible: bool = True,
    ):
        """Emit a thinking step and call callback.

        Args:
            visible: Whether step should be shown in user-facing UI.
                     Hidden steps are for developer/debug purposes only.
        """
        # Merge visible flag into details
        step_details = details.copy() if details else {}
        step_details["visible"] = visible

        step = state.add_step(step_type, content, step_details)
        if self.on_step:
            self.on_step(step)
        # Progress only emitted at boundaries (lead.done, checkpoint, synthesis.complete)
        if step_type in (StepType.INVESTIGATION_STARTED, StepType.LEAD_DONE, StepType.CHECKPOINT, StepType.SYNTHESIS_COMPLETE):
            self._emit_progress(state)

    async def _emit_step_async(
        self,
        state: InvestigationState,
        step_type: StepType,
        content: str,
        details: Optional[dict] = None,
        visible: bool = True,
    ):
        """Emit a thinking step with async yield for streaming.

        Use this in hot paths where multiple steps emit between awaits,
        to allow the event loop to process queued events for streaming.

        Args:
            visible: Whether step should be shown in user-facing UI.
        """
        self._emit_step(state, step_type, content, details, visible)
        await asyncio.sleep(0)  # Yield to event loop for streaming

    def _emit_progress(self, state: InvestigationState):
        """Emit progress update."""
        if self.on_progress:
            self.on_progress(state.get_progress())

    def _extract_search_term(self, lead_description: str) -> str:
        """Extract search term from lead description."""
        import re

        # Remove common prefixes
        prefixes = ["Search for:", "Investigate:", "Find:", "Look for:"]
        term = lead_description
        for prefix in prefixes:
            if term.startswith(prefix):
                term = term[len(prefix):].strip()
                break

        # Clean up
        term = re.sub(r'\bAND\b|\bOR\b|\bNOT\b', ' ', term, flags=re.IGNORECASE)
        term = re.sub(r'[\'\"()]', ' ', term)
        term = re.sub(r'\s+', ' ', term).strip()

        return term[:50] if term else "contract"

    def _format_findings(self, state: InvestigationState) -> str:
        """Format current findings for LLM."""
        facts = state.findings.get("accumulated_facts", [])
        citations_count = len(state.citations)
        docs_read = state.documents_read

        lines = [
            f"Documents read: {docs_read}",
            f"Citations found: {citations_count}",
            "",
            "Key facts found:",
        ]

        for fact in facts[:15]:  # Limit facts in summary
            lines.append(f"- {fact}")

        if not facts:
            lines.append("- No facts extracted yet")

        return "\n".join(lines)

    def _get_s3_client(self):
        """Build a boto3 S3 client from RLMConfig S3 settings."""
        import boto3
        kwargs = {"region_name": self.config.s3_region}
        if self.config.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.config.aws_access_key_id
        if self.config.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
        return boto3.client("s3", **kwargs)

    def _save_checkpoint(self, state: InvestigationState, iteration: int):
        """Save investigation checkpoint to S3 or local disk."""
        if self.config.s3_bucket and self.config.s3_checkpoint_prefix:
            try:
                s3 = self._get_s3_client()
                key = f"{self.config.s3_checkpoint_prefix}/checkpoint_{state.id}_iter{iteration}.json"
                state.save_checkpoint_to_s3(s3, self.config.s3_bucket, key)
                logger.info(f"Saved checkpoint to S3: {key}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint to S3: {e}")
            return

        if not self.config.checkpoint_dir:
            return
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{state.id}_iter{iteration}.json"
        state.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _delete_s3_checkpoints(self, state_id: str):
        """Delete all S3 checkpoints for a given state_id after successful completion."""
        if not (self.config.s3_bucket and self.config.s3_checkpoint_prefix):
            return
        try:
            s3 = self._get_s3_client()
            prefix = f"{self.config.s3_checkpoint_prefix}/checkpoint_{state_id}_"
            paginator = s3.get_paginator("list_objects_v2")
            keys_to_delete = []
            for page in paginator.paginate(Bucket=self.config.s3_bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys_to_delete.append({"Key": obj["Key"]})
            if keys_to_delete:
                s3.delete_objects(
                    Bucket=self.config.s3_bucket,
                    Delete={"Objects": keys_to_delete},
                )
                logger.info(f"Deleted {len(keys_to_delete)} S3 checkpoints for state {state_id}")
        except Exception as e:
            logger.warning(f"Failed to delete S3 checkpoints for {state_id}: {e}")

    async def resume_investigation(
        self,
        checkpoint_path: Optional[str | Path] = None,
        s3_key: Optional[str] = None,
    ) -> InvestigationState:
        """Resume investigation from checkpoint (local path or S3 key)."""
        if s3_key and self.config.s3_bucket:
            s3 = self._get_s3_client()
            state = InvestigationState.load_checkpoint_from_s3(s3, self.config.s3_bucket, s3_key)
        else:
            state = InvestigationState.load_checkpoint(checkpoint_path)
        repo = MatterRepository(state.repository_path)
        cache = InvestigationCache()

        self._emit_step(state, StepType.THINKING, "Resuming investigation from checkpoint")

        try:
            if state.status not in ("completed", "failed"):
                await self._investigate_loop(state, repo, cache)
                await self._run_external_research_post_loop(state)
                await self._synthesize(state)
                state.complete()

        except Exception as e:
            state.fail(str(e))
            raise

        return state

    async def summarize_documents(
        self,
        file_paths: list[Path],
        repository: Optional[MatterRepository] = None,
    ) -> dict[str, Any]:
        """Create summaries for multiple documents."""
        if not file_paths:
            return {"individual_summaries": [], "collection_summary": None}

        summaries = []
        for fp in file_paths:
            try:
                if repository:
                    doc, _ = await repository.read_async(str(fp))
                else:
                    temp_repo = MatterRepository(fp.parent)
                    doc, _ = await temp_repo.read_async(str(fp))

                content = doc.get_excerpt(self.config.excerpt_chars_complex)

                extraction = await decisions.extract_facts(
                    query="Summarize this document",
                    filename=doc.filename,
                    content=content,
                    client=self.client,
                    trace_ctx=self._trace_ctx,
                )

                summaries.append({
                    "filename": doc.filename,
                    "facts": extraction.get("facts", []),
                    "quotes": extraction.get("quotes", []),
                    "references": extraction.get("references", []),
                })

            except Exception as e:
                logger.error(f"Failed to summarize {fp}: {e}")

        return {
            "individual_summaries": summaries,
            "document_count": len(summaries),
        }


# ---------------------------------------------------------------------------
# Telemetry persistence (fire-and-forget, never blocks investigation)
# ---------------------------------------------------------------------------

def _persist_telemetry_sync(summary) -> None:
    """Synchronous DB write for telemetry. Runs in a thread to avoid blocking the event loop."""
    from ..db.session import session_scope
    from ..db.models.investigation_log import (
        InvestigationLog,
        InvestigationStepModel,
        InvestigationOperation,
    )
    from datetime import datetime, timezone
    from uuid import uuid4

    summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else summary

    def _parse_dt(iso_str):
        if isinstance(iso_str, datetime):
            return iso_str
        return datetime.fromisoformat(iso_str)

    with session_scope() as session:
        # 1. Insert investigation log
        log = InvestigationLog(
            id=summary_dict["investigation_id"],
            message_id=summary_dict.get("message_id"),
            user_id=summary_dict.get("user_id"),
            started_at=_parse_dt(summary_dict["started_at"]),
            completed_at=_parse_dt(summary_dict["completed_at"]),
            status=summary_dict["status"],
            total_duration_ms=summary_dict.get("total_duration_ms"),
            total_cost_usd=summary_dict.get("total_cost_usd"),
            total_steps=summary_dict.get("total_steps"),
            phase_breakdown=summary_dict.get("phase_breakdown"),
        )
        session.add(log)

        # 2. Insert steps first, then flush so FK references exist
        step_ops = []  # collect (step_id, op_dict) pairs
        for step_dict in summary_dict.get("steps", []):
            step_id = str(uuid4())
            step = InvestigationStepModel(
                id=step_id,
                investigation_id=summary_dict["investigation_id"],
                seq=step_dict["seq"],
                step_name=step_dict["step_name"],
                phase=step_dict["phase"],
                started_at=_parse_dt(step_dict["started_at"]),
                step_latency_ms=step_dict.get("step_latency_ms"),
            )
            session.add(step)
            for op_dict in step_dict.get("operations", []):
                step_ops.append((step_id, op_dict))

        # Flush log + steps so FK constraints are satisfied
        session.flush()

        # 3. Insert operations
        for step_id, op_dict in step_ops:
            details = {k: v for k, v in op_dict.items()
                       if k not in ("type", "started_at", "latency_ms")}
            op = InvestigationOperation(
                id=str(uuid4()),
                step_id=step_id,
                investigation_id=summary_dict["investigation_id"],
                type=op_dict["type"],
                started_at=_parse_dt(op_dict["started_at"]),
                latency_ms=op_dict.get("latency_ms"),
                details=details,
            )
            session.add(op)

    logger.info("Telemetry persisted to DB: %s", summary_dict["investigation_id"])


async def _persist_telemetry(summary) -> None:
    """Write telemetry to DB in a background thread. Logs warning on failure, never raises."""
    try:
        await asyncio.to_thread(_persist_telemetry_sync, summary)
    except Exception as e:
        logger.warning("Failed to persist telemetry to DB: %s", e)
