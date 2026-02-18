"""RLM Engine - Recursive Language Model investigation engine.

OPTIMIZED VERSION:
- Document extraction cache (don't re-extract same doc)
- Search term deduplication (skip similar searches)
- Smart model selection (FLASH for simple queries)
- Early sufficiency checks (after first iteration)
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from pathlib import Path
import asyncio
import logging
import re

from ..core.models import GeminiClient, ModelTier
from ..core.repository import MatterRepository
from ..core.search import SearchResults
from ..core.external_search import ExternalSearchManager
from ..core.fact_store import FactStore
from .state import InvestigationState, StepType, ThinkingStep, Citation, Lead, classify_query
from . import decisions

logger = logging.getLogger(__name__)


def _fmt_list(items: list, max_items: int = 3, max_len: int = 50) -> str:
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
    ):
        self.client = gemini_client
        self.config = config or RLMConfig()
        self.on_step = on_step
        self.on_citation = on_citation
        self.on_fact = on_fact
        self.on_progress = on_progress
        # Initialize external search manager (enabled by default)
        self.external_search = ExternalSearchManager() if self.config.enable_external_search else None
        self._external_research: dict = {}  # Store external research results
        self.repo: Optional[MatterRepository] = None  # Set during investigate()
        self.fact_store: Optional[FactStore] = None  # Set during investigate()

    async def investigate(
        self,
        query: str,
        repository_path: str | Path,
        seed_facts: Optional[list[str]] = None,
        seed_citations: Optional[list[dict]] = None,
    ) -> InvestigationState:
        """Run full recursive investigation."""
        repo = MatterRepository(repository_path)
        self.repo = repo  # Store for methods that need repo access (e.g., _load_pinned_documents)
        self._external_research = {"case_law": [], "web": [], "analysis": {}}  # Reset with proper structure
        state = InvestigationState.create(query, str(repository_path))

        # Load fact store for this repository
        self.fact_store = FactStore(Path(repository_path))
        facts_loaded = self.fact_store.load()

        # Emit fact store status to UI trace
        if facts_loaded > 0:
            await self._emit_step_async(
                state,
                StepType.FINDING,
                f"📚 Loaded {facts_loaded} cached facts from previous investigations",
            )
        else:
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"📚 No cached facts found - starting fresh (will save at {self.fact_store.facts_file})",
            )

        cache = InvestigationCache()

        # Seed prior session data if provided
        if seed_facts:
            added = state.add_facts(seed_facts)
            logger.info(f"Seeded {added}/{len(seed_facts)} prior-session facts")
        if seed_citations:
            for c in seed_citations:
                state.add_citation(
                    document=c.get("document", ""),
                    page=c.get("page"),
                    text=c.get("text", ""),
                    context=c.get("context", ""),
                    relevance=c.get("relevance", "prior session"),
                )
            logger.info(f"Seeded {len(seed_citations)} prior-session citations")

        # Get repo info for informative step message
        repo_name = Path(repository_path).name
        file_count = len(repo.list_files())
        total_chars = repo.metadata.total_chars if repo.metadata else 0

        try:
            # Check if small repository first - uses unified assessment (includes complexity)
            if repo.is_small_repo:
                await self._emit_step_async(
                    state,
                    StepType.THINKING,
                    f"Starting: \"{query[:60]}{'...' if len(query) > 60 else ''}\" on {repo_name} ({file_count} files, {total_chars:,} chars) → small repo mode",
                )
                file_names = [f.filename for f in repo.list_files()[:5]]
                await self._emit_step_async(
                    state,
                    StepType.THINKING,
                    f"Small repo mode: {_fmt_list(file_names, 4, 30)}",
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
                    state,
                    StepType.THINKING,
                    f"Starting: \"{query[:60]}{'...' if len(query) > 60 else ''}\" on {repo_name} ({file_count} files, {total_chars:,} chars) → {'FLASH' if is_simple else 'PRO'} synthesis",
                )

                # Phase 2: Investigation loop with continuous recalibration
                # - Reads documents, extracts facts, accumulates research triggers
                # - Continuously checks: sufficient? need to replan? need external research?
                # - External search is triggered MID-LOOP after triggers are accumulated
                #   (not upfront with generic terms - that produced irrelevant results)
                await self._investigate_loop(state, repo, cache)

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
                    f"📚 DEBUG: fact_store has {fact_count} facts, _facts list: {len(self.fact_store._facts)}",
                )
                if fact_count > 0:
                    try:
                        saved = self.fact_store.save()
                        self._emit_step(
                            state,
                            StepType.FINDING,
                            f"📚 Saved {saved} facts to {self.fact_store.facts_file}",
                        )
                    except Exception as e:
                        self._emit_step(
                            state,
                            StepType.THINKING,
                            f"📚 ERROR saving facts: {e}",
                        )
                else:
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"📚 No new facts extracted this session",
                    )

            # Clean up external search sessions
            if self.external_search:
                try:
                    await self.external_search.close()
                except Exception:
                    pass

        return state

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
        file_names = [f.filename for f in repo.list_files()]
        await self._emit_step_async(state, StepType.READING, f"Loading: {_fmt_list(file_names, 4, 25)}")

        # Step 1: Load all content directly
        all_content = repo.get_all_content()
        state.documents_read = len(repo.list_files())
        state.findings["small_repo_content"] = all_content

        await self._emit_step_async(
            state,
            StepType.FINDING,
            f"Loaded {state.documents_read} documents ({len(all_content):,} chars total)",
        )

        # Step 1.5: Extract facts from each document if fact store is empty
        # Reuse _read_document which already handles extraction and fact storage
        cache = InvestigationCache()
        if self.fact_store and len(self.fact_store) == 0:
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"Extracting facts from {len(repo.list_files())} documents for future reference...",
            )
            for doc in repo.list_files():
                await self._read_document(state, repo, doc.path, cache)

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
        await self._emit_step_async(state, StepType.THINKING, "Assessing query against documents...")

        assessment = await decisions.assess_small_repo(
            query=state.query,
            content=all_content,
            client=self.client,
            cached_facts=cached_facts_str,
        )

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

        await self._emit_step_async(
            state,
            StepType.THINKING,
            f"Assessment: {'SIMPLE' if is_simple else 'COMPLEX'} synthesis, "
            f"{'can answer from docs' if can_answer_from_docs else f'needs external: {gap[:50]}...'}",
        )

        # Step 3: External search only if assessment says we need it
        if not can_answer_from_docs and self.config.enable_external_search and self.external_search:
            case_law_queries = assessment.get("case_law_searches", [])
            web_queries = assessment.get("web_searches", [])

            if case_law_queries or web_queries:
                # Log what we're searching for
                queries_preview = []
                if case_law_queries:
                    queries_preview.append(f"CaseLaw[{_fmt_list(case_law_queries, 2, 30)}]")
                if web_queries:
                    queries_preview.append(f"Web[{_fmt_list(web_queries, 2, 30)}]")
                await self._emit_step_async(
                    state,
                    StepType.THINKING,
                    f"External search: {' | '.join(queries_preview)}",
                )

                # Execute round 1
                state.findings["initial_plan"] = {
                    "case_law_searches": case_law_queries,
                    "web_searches": web_queries,
                }
                await self._execute_external_searches(state)

                # Step 4: Check if we need more (only if we got results)
                has_results = bool(
                    self._external_research.get("case_law") or
                    self._external_research.get("web")
                )

                if has_results and gap:
                    # Summarize what we found for sufficiency check
                    results_summary = self._format_results_summary()

                    sufficiency = await decisions.check_search_sufficiency(
                        query=state.query,
                        original_gap=gap,
                        results_summary=results_summary,
                        client=self.client,
                    )

                    if not sufficiency.get("sufficient", True):
                        additional_search = sufficiency.get("additional_search", "")
                        if additional_search:
                            await self._emit_step_async(
                                state,
                                StepType.THINKING,
                                f"Gap remains, additional search: {additional_search[:50]}...",
                            )

                            # Determine if it's case law or web based on content
                            is_case_law = any(
                                term in additional_search.lower()
                                for term in ["case", "v.", "vs", "court", "ruling", "precedent"]
                            )

                            state.findings["initial_plan"] = {
                                "case_law_searches": [additional_search] if is_case_law else [],
                                "web_searches": [] if is_case_law else [additional_search],
                            }
                            await self._execute_external_searches(state)
                    else:
                        await self._emit_step_async(state, StepType.THINKING, "External results sufficient")
                elif not has_results:
                    await self._emit_step_async(state, StepType.THINKING, "No external results found")
        elif can_answer_from_docs:
            await self._emit_step_async(state, StepType.THINKING, "Proceeding with documents only (no external search needed)")

        # Step 5: Add citations and synthesize
        self._add_external_citations(state)
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
        """Add citations for external research results."""
        if not self._external_research:
            return

        # Add citations for case law
        for case in self._external_research.get("case_law", [])[:5]:
            citation = state.add_citation(
                document=f"[Case Law] {case.get('case_name', 'Unknown Case')}",
                page=None,
                text=case.get('snippet', '') or '',
                context=f"Citation: {case.get('citation', 'N/A')} | Court: {case.get('court', 'N/A')}",
                relevance="External case law research",
            )
            if citation and self.on_citation:
                self.on_citation(citation)

        # Add citations for web results
        for result in self._external_research.get("web", [])[:5]:
            citation = state.add_citation(
                document=f"[Web] {result.get('title', 'Unknown Source')}",
                page=None,
                text=result.get('content', '') or '',
                context=f"URL: {result.get('url', 'N/A')}",
                relevance="External regulatory research",
            )
            if citation and self.on_citation:
                self.on_citation(citation)

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
            for c in case_law[:5]:
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
            for r in web[:5]:
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
        )

        state.hypothesis = plan.get("success_criteria", "Investigating query")
        state.findings["issues"] = plan.get("key_issues", [])
        state.findings["initial_plan"] = plan

        # Emit the reasoning (show LLM's thinking process)
        reasoning = plan.get("reasoning", "")
        challenges = plan.get("potential_challenges", "")
        key_issues = plan.get("key_issues", [])

        if reasoning:
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"STRATEGY: {reasoning}",
            )
        if key_issues:
            issues_str = ", ".join(key_issues[:3])
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"KEY ISSUES: {issues_str}",
            )
        if challenges:
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"CHALLENGES: {challenges}",
            )

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
            terms = await decisions.extract_search_terms(state.query, self.client)
            for term in terms[:2]:
                state.add_lead(f"Search for: {term}", source="fallback")

        # Show actual lead descriptions in step message
        lead_descriptions = [l.description for l in state.leads]
        await self._emit_step_async(
            state,
            StepType.THINKING,
            f"Plan: {_fmt_list(lead_descriptions, 3, 45)}",
            details=plan,
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
        assessment = await decisions.assess_and_plan(
            query=state.query,
            file_list=file_list_str,
            total_files=stats.total_files,
            client=self.client,
            cached_facts=cached_facts_str,
        )

        # If can answer from facts, return early (caller handles synthesis)
        if assessment.get("can_answer_from_facts", False):
            return assessment

        # Store plan info in state (same as _create_plan)
        state.hypothesis = assessment.get("success_criteria", "Investigating query")
        state.findings["issues"] = assessment.get("key_issues", [])
        state.findings["initial_plan"] = assessment

        # Emit the reasoning (show LLM's thinking process)
        reasoning = assessment.get("reasoning", "")
        key_issues = assessment.get("key_issues", [])

        if reasoning:
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"STRATEGY: {reasoning}",
            )
        if key_issues:
            issues_str = ", ".join(key_issues[:3])
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"KEY ISSUES: {issues_str}",
            )

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
            terms = await decisions.extract_search_terms(state.query, self.client)
            for term in terms[:2]:
                state.add_lead(f"Search for: {term}", source="fallback")

        # Show actual lead descriptions in step message
        lead_descriptions = [l.description for l in state.leads]
        await self._emit_step_async(
            state,
            StepType.THINKING,
            f"Plan: {_fmt_list(lead_descriptions, 3, 45)}",
            details=assessment,
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
        queries_preview = []
        if case_law_queries:
            queries_preview.append(f"CaseLaw: {_fmt_list(case_law_queries, 2, 35)}")
        if web_queries:
            queries_preview.append(f"Web: {_fmt_list(web_queries, 2, 35)}")
        self._emit_step(
            state,
            StepType.THINKING,
            f"External search ({total_queries}): {' | '.join(queries_preview)}",
        )

        # Helper for case law search
        async def search_case_law(query: str) -> tuple[str, list]:
            try:
                cases = await self.external_search.search_case_law(
                    query,
                    max_results=self.config.max_case_law_results
                )
                return query, cases or []
            except Exception as e:
                logger.warning(f"Case law search failed for '{query}': {e}")
                return query, []

        # Helper for web search
        async def search_web(query: str) -> tuple[str, dict]:
            try:
                result_data = await self.external_search.search_web(
                    query,
                    max_results=self.config.max_web_results
                )
                return query, result_data or {}
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

            # Process results
            case_law_count = len(case_law_queries)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"External search task failed: {result}")
                    continue

                query, data = result
                if i < case_law_count:
                    # Case law result
                    if data:
                        self._external_research["case_law"].extend(data)
                        case_names = [c.get("case_name", "Unknown") for c in data]
                        self._emit_step(
                            state,
                            StepType.FINDING,
                            f"CaseLaw ({len(data)}): {_fmt_list(case_names, 2, 40)}",
                        )
                else:
                    # Web result
                    web_results = data.get("results", [])
                    if web_results:
                        self._external_research["web"].extend(web_results)
                        titles = [r.get("title", "Untitled") for r in web_results]
                        self._emit_step(
                            state,
                            StepType.FINDING,
                            f"Web ({len(web_results)}): {_fmt_list(titles, 2, 40)}",
                        )
                    if data.get("answer"):
                        self._external_research["web_answer"] = data["answer"]
        else:
            # Sequential execution (fallback)
            if case_law_queries and self.external_search:
                for query in case_law_queries:
                    _, cases = await search_case_law(query)
                    if cases:
                        self._external_research["case_law"].extend(cases)
                        case_names = [c.get("case_name", "Unknown") for c in cases]
                        self._emit_step(
                            state,
                            StepType.FINDING,
                            f"CaseLaw ({len(cases)}): {_fmt_list(case_names, 2, 40)}",
                        )

            if web_queries and self.external_search:
                for query in web_queries:
                    _, result_data = await search_web(query)
                    web_results = result_data.get("results", [])
                    if web_results:
                        self._external_research["web"].extend(web_results)
                        titles = [r.get("title", "Untitled") for r in web_results]
                        self._emit_step(
                            state,
                            StepType.FINDING,
                            f"Web ({len(web_results)}): {_fmt_list(titles, 2, 40)}",
                        )
                    if result_data.get("answer"):
                        self._external_research["web_answer"] = result_data["answer"]

        # CONSOLIDATED: Analyze all external results in one call
        if self._external_research.get("case_law") or self._external_research.get("web"):
            # Format case law results
            case_law_text = ""
            if self._external_research.get("case_law"):
                case_law_text = "\n\n".join([
                    f"**{c.get('case_name', 'Unknown')}** ({c.get('citation') or 'No citation'})\n"
                    f"Court: {c.get('court', 'Unknown')}\nDate: {c.get('date_filed', 'Unknown')}\n"
                    f"Snippet: {(c.get('snippet') or c.get('opinion_text', ''))[:500] if c.get('snippet') or c.get('opinion_text') else 'No summary'}"
                    for c in self._external_research["case_law"][:5]
                ])

            # Format web results
            web_text = ""
            if self._external_research.get("web"):
                web_text = "\n\n".join([
                    f"**{r.get('title', 'Untitled')}**\nURL: {r.get('url', '')}\n"
                    f"Content: {r.get('content', 'No content')[:500]}"
                    for r in self._external_research["web"][:5]
                ])

            # Single consolidated call replaces analyze_case_law_results + analyze_web_results
            analysis = await decisions.analyze_external(
                query=state.query,
                case_law_results=case_law_text,
                web_results=web_text,
                client=self.client,
            )

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

        # Store in state findings for reference
        state.findings["external_research"] = self._external_research

        # Add citations for external sources (so they appear in Citations tab)
        if self._external_research.get("case_law"):
            for case in self._external_research["case_law"][:5]:
                citation = state.add_citation(
                    document=f"[Case Law] {case.get('case_name', 'Unknown Case')}",
                    page=None,
                    text=case.get('snippet', '') or case.get('opinion_text', '') or '',
                    context=f"Citation: {case.get('citation', 'N/A')} | Court: {case.get('court', 'N/A')}",
                    relevance="External case law research",
                )
                if citation and self.on_citation:
                    self.on_citation(citation)

        if self._external_research.get("web"):
            for result in self._external_research["web"][:5]:
                citation = state.add_citation(
                    document=f"[Web] {result.get('title', 'Unknown Source')}",
                    page=None,
                    text=result.get('content', '') or '',
                    context=f"URL: {result.get('url', 'N/A')}",
                    relevance="External regulatory research",
                )
                if citation and self.on_citation:
                    self.on_citation(citation)

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

            lead_descriptions = [l.description for l in leads_to_process]
            await self._emit_step_async(
                state,
                StepType.THINKING,
                f"Iteration {iteration + 1}: {_fmt_list(lead_descriptions, 3, 40)}",
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

                checkpoint_result = await decisions.checkpoint(
                    query=state.query,
                    findings=findings_summary,
                    plan=plan_summary,
                    client=self.client,
                    cached_facts=cached_facts_str,
                )

                # Check sufficiency - MUST have read at least 1 document
                # Facts from 0 docs is logically impossible for document investigation
                docs_read = state.documents_read
                if checkpoint_result.get("sufficient") and docs_read > 0:
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"Sufficient: {facts_count} facts from {docs_read} docs - proceeding to synthesis",
                    )
                    break
                elif checkpoint_result.get("sufficient") and docs_read == 0:
                    # LLM claims sufficient but no docs read - this is a bug state
                    self._emit_step(
                        state,
                        StepType.ERROR,
                        f"Checkpoint claims sufficient but 0 documents read - continuing investigation",
                    )

                # Handle replanning if needed
                if checkpoint_result.get("should_replan") and pending_leads:
                    progress = checkpoint_result.get("progress_assessment", "")
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"Recalibrating... {progress[:50]}",
                    )

                    # Add new leads from checkpoint
                    new_leads_added = 0
                    for term in checkpoint_result.get("new_search_terms", [])[:3]:
                        if not cache.is_similar_search(term):
                            state.add_lead(f"Search for: {term}", source="checkpoint")
                            new_leads_added += 1
                    for filepath in checkpoint_result.get("files_to_check", [])[:2]:
                        if not cache.has_extracted(filepath) and not cache.is_irrelevant(filepath):
                            state.add_lead(f"Read document: {filepath}", source="checkpoint")
                            new_leads_added += 1

                    if new_leads_added > 0:
                        self._emit_step(
                            state,
                            StepType.THINKING,
                            f"Added {new_leads_added} new leads from checkpoint",
                        )

            # 3. Dynamic external search - trigger if we discover we need it
            # (e.g., found references to case law, regulations, state-specific rules)
            # Supports tiered queries - can run multiple iterations with new queries
            if self.config.enable_external_search and self.external_search:
                # Check if findings suggest we need external research
                new_queries = await self._check_if_external_needed(state, executed_external_queries)
                if new_queries:
                    new_case_law = new_queries.get("case_law_queries", [])
                    new_web = new_queries.get("web_queries", [])
                    total_new = len(new_case_law) + len(new_web)

                    if total_new > 0:
                        self._emit_step(
                            state,
                            StepType.THINKING,
                            f"Running {total_new} new external search{'es' if total_new > 1 else ''}...",
                        )
                        await self._execute_external_searches(
                            state,
                            case_law_queries=new_case_law,
                            web_queries=new_web,
                        )
                        # Track executed queries
                        executed_external_queries.update(new_case_law)
                        executed_external_queries.update(new_web)

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
            f"Triggers: {_fmt_list(trigger_list, 4, 30)}",
        )

        # Generate specific queries using accumulated triggers
        result = await decisions.generate_external_queries(
            query=state.query,
            facts=facts,
            entities=entities,
            client=self.client,
            triggers=triggers,
        )

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

            queries_preview = []
            if new_case_law:
                queries_preview.append(f"CaseLaw: {_fmt_list(new_case_law, 2, 30)}")
            if new_web:
                queries_preview.append(f"Web: {_fmt_list(new_web, 2, 30)}")
            self._emit_step(
                state,
                StepType.THINKING,
                f"New queries: {' | '.join(queries_preview)}",
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
                state.mark_lead_investigated(lead.id, "Already extracted")
                return
            if cache.is_irrelevant(filepath):
                state.mark_lead_investigated(lead.id, "Marked irrelevant")
                return

            await self._read_document(state, repo, filepath, cache)
            state.mark_lead_investigated(lead.id, "Document read")
        else:
            # Extract search term
            search_term = self._extract_search_term(lead.description)

            # OPTIMIZATION: Skip similar searches
            if self.config.skip_similar_searches and cache.is_similar_search(search_term):
                state.mark_lead_investigated(lead.id, f"Similar search already done")
                logger.info(f"Skipping similar search: {search_term}")
                return

            cache.add_search(search_term)
            await self._emit_step_async(state, StepType.SEARCH, f"Searching: {search_term}")

            # Perform search (using smart_search for OR fallback)
            results = repo.smart_search(search_term, context_lines=2)
            state.searches_performed += 1

            if not results.hits:
                state.mark_lead_investigated(lead.id, "No results found")
                return

            # Show matching document names
            doc_names = list(set(Path(hit.file_path).name for hit in results.hits[:5]))
            await self._emit_step_async(
                state,
                StepType.FINDING,
                f"Found {len(results.hits)} matches in: {_fmt_list(doc_names, 3, 30)}",
            )

            # CONSOLIDATED: Single analyze_search call replaces pick_relevant_hits + analyze_results + prioritize_documents
            await self._analyze_results_consolidated(state, repo, results, cache)

            state.mark_lead_investigated(lead.id, f"Found {len(results.hits)} matches")

    async def _analyze_results_consolidated(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        results: SearchResults,
        cache: InvestigationCache,
    ):
        """Consolidated search analysis - single FLASH call replaces 3 separate calls."""
        key_issues = state.findings.get("issues", [])
        already_read = list(cache.extracted_docs)

        # Single consolidated call: pick_relevant_hits + analyze_results + prioritize_documents
        analysis = await decisions.analyze_search(
            query=state.query,
            key_issues=key_issues,
            results=results,
            already_read=already_read,
            client=self.client,
        )

        # Store facts and emit callbacks
        facts = analysis.get("facts", [])
        state.add_facts(facts)
        if self.on_fact:
            for fact in facts:
                self.on_fact(fact)

        # Add citations from relevant hits
        relevant_hits = analysis.get("relevant_hits", [])
        for hit in relevant_hits[:3]:
            citation = state.add_citation(
                document=hit.file_path,
                page=hit.page_num,
                text=hit.match_text,
                context=hit.context,
                relevance=f"Found via search: {results.query}",
            )
            if citation and self.on_citation:
                self.on_citation(citation)

        # Add leads for docs to read deeper
        for filepath in analysis.get("read_deeper", [])[:2]:
            if isinstance(filepath, str) and not cache.has_extracted(filepath):
                state.add_lead(f"Read document: {filepath}", source="analysis")

        # Add additional search leads
        for term in analysis.get("additional_searches", [])[:1]:
            if isinstance(term, str) and not cache.is_similar_search(term):
                state.add_lead(f"Search for: {term}", source="analysis")

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

        await self._batch_read(state, repo, top_files[:self.config.parallel_reads], cache)

    async def _batch_read(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_paths: list[str],
        cache: InvestigationCache,
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

        tasks = [self._read_document(state, repo, fp, cache) for fp in to_read]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _read_document(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_path: str,
        cache: InvestigationCache,
    ) -> bool:
        """Read and extract facts from a document.

        Returns:
            True if document was read successfully, False otherwise.
        """
        # OPTIMIZATION: Skip if already extracted
        if cache.has_extracted(file_path):
            logger.debug(f"Skipping already extracted: {file_path}")
            return True  # Already extracted = success

        await self._emit_step_async(state, StepType.READING, f"Reading: {Path(file_path).name}")

        try:
            doc = repo.read(file_path)
            state.documents_read += 1
            cache.mark_extracted(file_path)  # Mark as extracted
            cache.record_read_success()  # Reset consecutive failure counter

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
            extraction = await decisions.extract_facts(
                query=state.query,
                filename=doc.filename,
                content=content,
                client=self.client,
                max_content_chars=extraction_limit,
            )

            # Store facts
            facts = extraction.get("facts", [])
            if facts:
                self._emit_step(
                    state,
                    StepType.FINDING,
                    f"Facts ({len(facts)}) from {doc.filename}: {_fmt_list(facts, 2, 60)}",
                )
            state.add_facts(facts)
            if self.on_fact:
                for fact in facts:
                    self.on_fact(fact)

            # Save facts to persistent store for future queries
            if self.fact_store:
                new_facts = self.fact_store.add_facts_from_extraction(
                    extraction=extraction,
                    source_filename=doc.filename,
                    query_context=state.query,
                )
                self._emit_step(
                    state,
                    StepType.THINKING,
                    f"📚 Added {new_facts} facts from {doc.filename} (store total: {len(self.fact_store)})",
                )

            # Accumulate external research triggers
            triggers = extraction.get("external_triggers", {})
            if triggers:
                added = state.add_triggers(triggers)
                if added > 0:
                    # Collect all trigger values for display
                    trigger_values = []
                    for trigger_list in triggers.values():
                        if isinstance(trigger_list, list):
                            trigger_values.extend(trigger_list)
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"Triggers from {doc.filename}: {_fmt_list(trigger_values, 3, 30)}",
                    )

            # Emit insights from the extraction (show LLM's thinking)
            insights = extraction.get("insights", "")
            gaps = extraction.get("gaps", "")
            next_steps = extraction.get("next_steps", "")

            if insights or gaps:
                insight_msg = f"From {doc.filename}:\n"
                if insights:
                    insight_msg += f"  LEARNED: {insights}\n"
                if gaps:
                    insight_msg += f"  GAPS: {gaps}\n"
                if next_steps:
                    insight_msg += f"  NEXT: {next_steps}"
                self._emit_step(state, StepType.THINKING, insight_msg.strip())

            # Add citations from quotes (limit to 2)
            for quote in extraction.get("quotes", [])[:2]:
                if isinstance(quote, dict) and "text" in quote:
                    citation = state.add_citation(
                        document=doc.path,
                        page=quote.get("page"),
                        text=quote["text"],
                        context="",
                        relevance=quote.get("relevance", "Direct quote"),
                    )
                    if citation and self.on_citation:
                        self.on_citation(citation)

            # Skip adding reference leads - reduces iteration depth
            return True  # Success

        except Exception as e:
            self._emit_step(state, StepType.ERROR, f"Failed to read {file_path}: {e}")
            cache.record_read_failure()
            # Don't emit error here - the loop will handle it and continue
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
        if state.documents_read == 0 and not small_repo_content and not answered_from_cache:
            self._emit_step(
                state,
                StepType.ERROR,
                "Cannot synthesize: 0 documents were successfully read. Check document paths and access.",
            )
            error_msg = (
                "**Investigation Failed**\n\n"
                "Unable to read any documents from the repository. This may indicate:\n"
                "- Documents were not downloaded correctly\n"
                "- File paths do not match between search index and storage\n"
                "- Files were cleaned up before investigation completed\n\n"
                f"Total read attempts that failed: multiple\n"
                f"Query: {state.query}"
            )
            state.findings["final_output"] = error_msg
            return

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

        await self._emit_step_async(state, StepType.SYNTHESIS, f"Synthesizing: {', '.join(source_parts)}")

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
        response = await decisions.synthesize(
            query=state.query,
            evidence=evidence,
            external_research=f"=== CASE LAW (CourtListener) ===\n{case_law_text}\n\n=== REGULATIONS/STANDARDS (Web) ===\n{web_text}",
            pinned_content=pinned_content,
            client=self.client,
            tier=synthesis_tier,
        )

        state.findings["final_output"] = response
        output_len = len(response)
        await self._emit_step_async(state, StepType.SYNTHESIS, f"Complete: {output_len:,} chars synthesized from {state.documents_read} docs")

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
                doc = self.repo.read(filepath)
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
                logger.warning(f"Failed to load pinned document {filepath}: {e}")

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
    ):
        """Emit a thinking step and call callback."""
        step = state.add_step(step_type, content, details)
        if self.on_step:
            self.on_step(step)
        self._emit_progress(state)

    async def _emit_step_async(
        self,
        state: InvestigationState,
        step_type: StepType,
        content: str,
        details: Optional[dict] = None,
    ):
        """Emit a thinking step with async yield for streaming.

        Use this in hot paths where multiple steps emit between awaits,
        to allow the event loop to process queued events for streaming.
        """
        self._emit_step(state, step_type, content, details)
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

    def _save_checkpoint(self, state: InvestigationState, iteration: int):
        """Save investigation checkpoint."""
        if not self.config.checkpoint_dir:
            return

        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{state.id}_iter{iteration}.json"
        state.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    async def resume_investigation(
        self,
        checkpoint_path: str | Path,
    ) -> InvestigationState:
        """Resume investigation from checkpoint."""
        state = InvestigationState.load_checkpoint(checkpoint_path)
        repo = MatterRepository(state.repository_path)
        cache = InvestigationCache()

        self._emit_step(state, StepType.THINKING, "Resuming investigation from checkpoint")

        try:
            if state.status not in ("completed", "failed"):
                await self._investigate_loop(state, repo, cache)
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
                    doc = repository.read(str(fp))
                else:
                    temp_repo = MatterRepository(fp.parent)
                    doc = temp_repo.read(str(fp))

                content = doc.get_excerpt(self.config.excerpt_chars_complex)

                extraction = await decisions.extract_facts(
                    query="Summarize this document",
                    filename=doc.filename,
                    content=content,
                    client=self.client,
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
