"""RLM Engine - Recursive Language Model investigation engine.

This is the core of the system. It implements:
1. Iterative refinement with data-driven replanning
2. Recursive investigation of leads
3. Parallel document processing
4. Tiered model usage (Lite -> Flash -> Pro)
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any, AsyncIterator
from pathlib import Path
import asyncio
import json
import logging
import sqlite3

from ..core.models import GeminiClient, ModelTier
from ..core.repository import MatterRepository
from ..core.search import SearchResults, SearchHit
from ..core.utils import jaccard_similarity as _jaccard_similarity
from ..matter.enums import SourceRole as _SourceRole
from ..matter.runtime import infer_source_role as _infer_source_role
from .governance import resolve_matter_domain as _resolve_matter_domain
from .state import (
    InvestigationState,
    StepType,
    ThinkingStep,
    Citation,
    Lead,
    Obligation,
    OutputEnvelope,
    ResearchMode,
    RunObjective,
    ValidationResult,
    normalize_research_mode,
    WorkflowKind,
    WorkingSet,
)

# SO-5: module-level map from LLM-returned doc_source_role strings to SourceRole enums.
# Built automatically from enum values so it never drifts when new roles are added.
# UNKNOWN is excluded (LLM "unknown" stays as UNKNOWN via .get() default below).
_CONTENT_ROLE_MAP: dict[str, "_SourceRole"] = {
    role.value: role for role in _SourceRole if role != _SourceRole.UNKNOWN
}
# Alias: LLM may return "post_hoc_explanatory" (Python name) vs "post_hoc" (enum value).
_CONTENT_ROLE_MAP["post_hoc_explanatory"] = _SourceRole.POST_HOC_EXPLANATORY

logger = logging.getLogger(__name__)


# ── Date normalisation (SO-6 timeline reliability) ──────────────────────────
import re as _re_date
from dateutil import parser as _dateutil_parser

_QUARTER_MAP = {"q1": "01-01", "q2": "04-01", "q3": "07-01", "q4": "10-01"}
_ISO_DATE_RE = _re_date.compile(r"^\d{4}-\d{2}-\d{2}$")
_QUARTER_RE = _re_date.compile(r"^[Qq]([1-4])\s*(\d{4})$")
_QUARTER_RE2 = _re_date.compile(r"^(\d{4})\s*[Qq]([1-4])$")
_YEAR_ONLY_RE = _re_date.compile(r"^(\d{4})$")
_MONTH_YEAR_RE = _re_date.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{4})$",
    _re_date.IGNORECASE,
)
_MONTH_NUM = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _normalize_date(raw: str, llm_value: Any = None, llm_precision: str | None = None) -> tuple[str | None, str]:
    """Normalise a date string to ISO YYYY-MM-DD and return (iso_date, precision).

    First trusts LLM-provided value/precision if they look valid, then falls back
    to heuristic parsing of *raw*.  Returns (None, "unknown") on total failure.

    Precision values: "day", "month", "quarter", "year", "unknown".
    """
    # 1. Try LLM-provided ISO value first.
    if isinstance(llm_value, str) and _ISO_DATE_RE.match(llm_value.strip()):
        iso = llm_value.strip()
        precision = llm_precision if llm_precision in ("day", "month", "quarter", "year") else "day"
        return iso, precision

    text = (raw or "").strip()
    if not text:
        return None, "unknown"

    # 2. Already ISO?
    if _ISO_DATE_RE.match(text):
        return text, llm_precision if llm_precision in ("day", "month", "quarter", "year") else "day"

    # 3. Quarter patterns: "Q3 2023" or "2023 Q3"
    m = _QUARTER_RE.match(text)
    if m:
        return f"{m.group(2)}-{_QUARTER_MAP['q' + m.group(1)]}", "quarter"
    m = _QUARTER_RE2.match(text)
    if m:
        return f"{m.group(1)}-{_QUARTER_MAP['q' + m.group(2)]}", "quarter"

    # 4. Year only: "2023"
    m = _YEAR_ONLY_RE.match(text)
    if m:
        return f"{m.group(1)}-01-01", "year"

    # 5. Month+year: "March 2023", "Mar 2023", "Nov. 2024"
    m = _MONTH_YEAR_RE.match(text)
    if m:
        mon = _MONTH_NUM.get(m.group(1)[:3].lower())
        if mon:
            return f"{m.group(2)}-{mon}-01", "month"

    # 6. General dateutil parse (handles "January 15, 2024", "15/01/2024", etc.)
    try:
        dt = _dateutil_parser.parse(text, fuzzy=False)
        return dt.strftime("%Y-%m-%d"), llm_precision if llm_precision in ("day", "month", "quarter", "year") else "day"
    except (ValueError, OverflowError):
        pass

    # 7. Fuzzy parse as last resort (handles "signed on March 5, 2023").
    # Guard: dateutil fills missing month/day with today's values, producing
    # misleading results like "sometime in 2023" → "2023-04-11".  Reject when
    # the parsed month+day equal today's (likely defaulted, not from the text).
    try:
        from datetime import date as _date_cls
        dt = _dateutil_parser.parse(text, fuzzy=True)
        _today = _date_cls.today()
        _looks_defaulted = (dt.month == _today.month and dt.day == _today.day
                            and dt.year != _today.year)
        if not _looks_defaulted:
            return dt.strftime("%Y-%m-%d"), llm_precision if llm_precision in ("day", "month", "quarter", "year") else "day"
    except (ValueError, OverflowError):
        pass

    return None, "unknown"


# Pre-validated assertion link types. Checked against LLM-supplied relation strings
# before calling adapter.record_assertion_link() to prevent repeated log_warning() DB
# writes when the LLM returns an unsupported relation throughout a run.
_VALID_ASSERTION_LINK_TYPES: frozenset[str] = frozenset(
    {"supports", "attacks", "depends_on", "supersedes", "contradicts", "corroborates"}
)

# ── SO-5 advocacy gate: module-level compiled patterns ────────────────────────
# Hoisted from _enforce_advocacy_gate() so they are compiled once at import time,
# not on each synthesis call.
import re as _re_engine

# Heading boundary regex per level: matches \n followed by 1..level '#' chars + space.
_ADVOCACY_HDR_RE: "dict[int, _re_engine.Pattern[str]]" = {
    lvl: _re_engine.compile(r'\n#{1,' + str(lvl) + r'} ') for lvl in (2, 3)
}

# List-item starter pattern (applied to each stripped line in semantic unit builder).
_ADVOCACY_LIST_PAT: "_re_engine.Pattern[str]" = _re_engine.compile(
    r'^(?:[-*+•]|\d+[.)]|[a-zA-Z][.)])\s'
)

# Section header patterns: (header_str, search_regex) pairs including colon variant.
# Each pattern is anchored to line-start; case-insensitive.
_ADVOCACY_SECTIONS: "list[tuple[str, _re_engine.Pattern[str]]]" = []
for _hdr in (
    "### Key Findings",
    "## Key Findings",
    "## Factual Background",
    "### Factual Background",
):
    _hdr_lower = _hdr.lower()
    for _search in (_hdr_lower, _hdr_lower.rstrip(':') + ':'):
        _ADVOCACY_SECTIONS.append((
            _hdr,
            _re_engine.compile(r'(?:^|\n)' + _re_engine.escape(_search), _re_engine.IGNORECASE),
        ))
del _hdr, _hdr_lower, _search  # clean up loop variables at module scope

# Advisory marker presence pattern.
_ADVOCACY_MARKER_NAME = "Source Calibration Advisory"
_ADVOCACY_MARKER_PAT: "_re_engine.Pattern[str]" = _re_engine.compile(
    r'(?:^|\n)#{2,3} ' + _re_engine.escape(_ADVOCACY_MARKER_NAME) + r':?[ \t]*(?:\n|$)',
    _re_engine.IGNORECASE,
)
# ── end SO-5 advocacy gate constants ─────────────────────────────────────────


# MVP.6: the default allowlist of optional synthesis sections. New
# store summaries default OFF — register their keys here only after an
# explicit decision to inject them into every synthesis prompt. Coverage
# and gaps bypass this list and are always included (PR.3 contract).
_DEFAULT_OPTIONAL_SECTIONS: frozenset[str] = frozenset({
    "source_calibration",
    "decision_context",
    "entities",
    "relationships",
    "quantitative",
    "citations",
})


@dataclass(frozen=True)
class PacketBudget:
    """MVP.6 hard prompt-budget guardrails (SO-1).

    Every expensive prompt section has an explicit cap so new durable
    stores cannot silently inflate every call the engine makes. Caps are
    in estimated tokens (len(text)//4), matching GeminiClient.

    Defaults are intentionally conservative for the first rollout:
    - coverage_tokens / gap_tokens: preserved from PR.3 so existing
      synthesis behavior does not regress
    - orientation_tokens: caps the durable matter_context block inside
      the orient prompt; repo file listing stays uncapped
    - per_optional_section_tokens: per-section cap for source
      calibration, entities, relationships, quantitative, citations,
      decision_context
    - synthesis_total_tokens: whole-packet cap enforced after per-section
      caps so mandatory sections cannot be crowded out
    """

    coverage_tokens: int = 0  # 0 = unlimited (use model's full context)
    gap_tokens: int = 0
    orientation_tokens: int = 0
    per_optional_section_tokens: int = 0
    synthesis_total_tokens: int = 0  # No cap — let the model use its full context window


@dataclass
class RLMConfig:
    """Configuration for RLM engine."""
    max_depth: int = 8
    max_leads_per_level: int = 12
    max_documents_per_search: int = 10
    min_lead_priority: float = 0.3
    excerpt_chars: int = 64000
    parallel_reads: int = 5
    max_initial_deep_read_documents: int = 20
    checkpoint_dir: Optional[str] = None  # Directory for checkpoints
    checkpoint_interval: int = 5  # Save checkpoint every N iterations
    adaptive_depth: bool = True  # Adjust depth based on complexity
    min_depth: int = 2  # Minimum depth even for simple queries
    depth_citation_threshold: int = 15  # Stop early if enough citations
    max_iterations: int = 20  # Maximum investigation loop iterations
    enable_matter_model: bool = True  # When True, persist facts to SQLite matter model
    synthesis_pro_timeout: float = 180.0  # Final synthesis PRO call timeout.
    synthesis_fallback_timeout: float = 120.0  # FLASH fallback timeout after PRO timeout.
    # MVP.6: prompt-budget guardrails. Immutable so tests that swap it
    # via dataclasses.replace get fresh caps without side effects.
    packet_budget: PacketBudget = field(default_factory=PacketBudget)


@dataclass(frozen=True)
class ResearchBudgetProfile:
    """Effective per-run investigation budget for a research mode."""

    mode: str
    max_depth: int
    min_depth: int
    max_iterations: int
    depth_citation_threshold: int
    confidence_threshold: int
    min_citations: int
    diminishing_returns_fact_threshold: int
    diminishing_returns_min_citations: int
    diminishing_returns_min_confidence: int
    very_low_productivity_max_facts: int


# System prompts for different stages
RESEARCH_ALIGNMENT_GUIDANCE = """
USER-OBJECTIVE ALIGNMENT:
- Keep the investigation anchored to the user's actual objective and requested direction.
- Prioritize the lines of inquiry most likely to answer the user's question or support the requested work product.
- Do not let one interesting tangent, repeated search hit, or advocacy framing pull the investigation away from the user's objective.
- Apply pragmatic judgment when choosing what to pursue next, including risk exposure, timing, commercial realities, and domain-specific constraints.
- If the record points in a different direction than the user's apparent assumption, surface that clearly, but still organize the research around answering the user's question.
- Favor the highest-yield next step, not the most intellectually interesting one.
""".strip()

ORIENTATION_PROMPT = """You are an expert analyst conducting due diligence on a document repository.

Repository Structure:
{structure}

Document Listing:
{file_listing}

Total files: {total_files}

User Query: {query}
{matter_context}
Your task is to create a COMPREHENSIVE and EXHAUSTIVE research plan. Think like a senior partner conducting final due diligence review — you must identify EVERY material issue, not just the top 3-5.

{research_alignment_guidance}

CRITICAL INSTRUCTION — EXHAUSTIVE COVERAGE:
Do NOT produce a high-level summary of 3-5 major themes. You must identify EVERY specific provision, deviation, requirement, risk, obligation, and material detail that the query asks about. If analyzing a contract, list EVERY material term — not just "key financial terms" but each specific one (interest rate, commitment fee, leverage covenants, financial reporting, events of default, change of control, assignment restrictions, etc.). If comparing documents, identify EVERY point of difference, not just the most obvious ones.

DOCUMENT COMPARISON INSTRUCTION (when comparing two or more documents — e.g., markup vs. original, competing proposals, redline analysis):
You MUST create a separate issue for EACH of these provision categories (skip only if the document type clearly does not contain such provisions):
- Interest rate / SOFR floor / margin grid / rate mechanics
- Commitment fees / upfront fees / unused fees
- EACH financial covenant separately (leverage ratio, FCCR, interest coverage, etc.)
- EBITDA definition and add-back caps (non-recurring, synergy, pro forma adjustments)
- Permitted acquisition baskets (individual, aggregate, pro forma compliance)
- Restricted payments / distributions / dividend restrictions
- Events of default (cross-default thresholds, payment defaults, covenant defaults)
- Change of control definition and thresholds
- Assignment / transfer / participation restrictions
- Mandatory prepayment / excess cash flow sweep mechanics
- Negative covenants (anti-layering, most-favored-nation, additional indebtedness)
- Reinvestment period / asset sale proceeds
- MAE / MAC definition and qualifiers
- Reporting requirements and information covenants
- Representations and warranties scope
- Extension options / maturity / amortization
For each category, generate a search targeting the SAME provision in BOTH documents (e.g., search for "SOFR" or "interest rate" to find the term in both the original and the markup). This ensures provision-by-provision comparison, not just sampling of the most obvious differences.
Generate at least 12-15 initial_searches for comparison tasks.

Consider:
1. READ THE DOCUMENT LISTING CAREFULLY. File names reveal what each document IS (e.g., "Master_Service_Agreement.pdf" is a contract, "Complaint_Filed_2024.pdf" is a pleading, "Invoice_March.xlsx" is financial). Use file names to identify the MOST IMPORTANT documents.
2. What are ALL the issues that need to be established? List 12-20 issues for document comparison tasks, 8-15 for other tasks.
3. For each issue, what SPECIFIC provisions, sections, or data points must be found?
4. Which specific documents from the listing are MOST LIKELY to contain direct evidence? Name them explicitly in your search terms.
5. What SPECIFIC search terms will find relevant passages? Generate 12-15 targeted searches covering different aspects of the query. Use party names, document-specific terms, financial terms, defined terms, and key phrases you expect to find IN those documents.
6. What is your preliminary hypothesis based on the query and the document names?

PRIORITIZE:
{domain_document_priorities}
- Documents with dates matching key events
- Files mentioning specific parties or amounts
- Existing Matter Intelligence about open gaps, missing evidence, and weakly supported issues
- If a PRIORITY FOCUS issue is listed in Existing Matter Intelligence, direct the first 2-3 `initial_searches` specifically toward that issue before broadening to general exploration
- QUANTITATIVE DATA: specific dollar amounts, percentages, dates, thresholds, ratios
- DEFINED TERMS: capitalized terms that have specific contractual meaning
- CROSS-REFERENCES: provisions that reference other sections or documents

Respond in JSON format:
{{
    "issues": [
        {{
            "title": "issue description",
            "type": "{domain_issue_types}",
            "predicates": ["testable element 1", "testable element 2"]
        }}
    ],
    "relevant_folders": ["folder1", "folder2", ...],
    "initial_searches": [{{"term": "search term", "issue_idx": 0}}, {{"term": "term2", "issue_idx": 1}}, ...],
    "search_rationale": "Why these search terms will find relevant evidence",
    "document_priority": ["most important doc type", "second most important", ...],
    "target_documents": ["exact_filename_1.pdf", "exact_filename_2.docx"],
    "hypothesis": "Your initial hypothesis based on query analysis"
}}

Generate 12-15 initial_searches minimum. Each search should target a DIFFERENT aspect of the query.

For target_documents: list the EXACT filenames from the Document Listing above that you
believe are the highest-value retrieval targets. These should be specific files, not types.
Maximum 10 filenames. These will be used as durable retrieval targets throughout the investigation.

{domain_issue_type_descriptions}

For each issue, include 2-4 "predicates": the specific testable elements that must be
established to prove or defeat that issue ({domain_predicate_examples}).
Predicates drive targeted document search — make them concrete and searchable.

For initial_searches: each entry must include "term" (the search string) and "issue_idx"
(0-based index into the issues array above identifying which issue this search targets).
This enables the system to link discovered facts to the correct issue.

IMPORTANT — search term format:
Each "term" must be a simple literal phrase or exact filename that grep can match.
DO NOT use boolean operators (AND, OR, NOT), quotes as search syntax, or wildcards.
{domain_search_examples}
Bad: "breach AND contract", "\"termination\" OR \"cancellation\""
"""

# Bump this version string whenever ORIENTATION_PROMPT structure changes.
# Including it in the cache key ensures old cached plans (which may lack
# new fields like "predicates") are automatically invalidated after a
# prompt update (SO-1 stale-cache prevention).
_ORIENTATION_CACHE_VERSION = "12"


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
        lines.append(f"- Open issues: {', '.join(t for t in issue_titles if t)}")
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
    if getattr(ctx, "key_predicates", None):
        preds_display = ", ".join(ctx.key_predicates[:8])
        lines.append(
            f"- Known relationship types (predicate graph): {preds_display} — "
            "use these to target searches (e.g. '<party> {predicate} <object>')"
        )
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
    if getattr(ctx, "document_annotations", None):
        lines.append(f"- Document annotations from user ({len(ctx.document_annotations)} notes):")
        for ann in ctx.document_annotations[:5]:
            doc = (ann.get("document_pattern") or "")
            txt = (ann.get("annotation_text") or "")[:150]
            ann_type = (ann.get("annotation_type") or "strategic").upper()
            lines.append(f"  [{ann_type}] {doc}: {txt}")
    # Gap 3: surface active assumptions so orientation/synthesis can caveat conclusions
    if getattr(ctx, "active_assumptions", None):
        lines.append(f"- Active assumptions ({len(ctx.active_assumptions)} provisional):")
        for asm in ctx.active_assumptions[:8]:
            stmt = (asm.get("statement") or "")[:120]
            cond = asm.get("invalidation_condition") or ""
            line = f"  * ASSUMED: {stmt}"
            if cond:
                line += f" [invalidated if: {cond[:80]}]"
            lines.append(line)
        lines.append(
            "  → Conclusions depending on these assumptions must be qualified. "
            "If evidence contradicts an assumption, flag the conflict."
        )
    lines.append("")
    return "\n".join(lines)

def _conversation_history_digest(conversation_history: list[dict[str, str]] | None) -> str:
    """Stable digest input for cache keys when prior visible turns matter."""
    if not conversation_history:
        return ""
    parts: list[str] = []
    for turn in conversation_history:
        q = str(turn.get("query") or "").strip()
        a = str(turn.get("answer") or "").strip()
        if q:
            parts.append(f"U:{q}")
        if a:
            parts.append(f"A:{a}")
    return "\n".join(parts)

# Two-stage search analysis (hot-loop cost cut). Stage 1 is pure
# extraction over the big search_results input — cheap LITE call that
# always fires. Stage 2 is reasoning over Stage 1's compact output —
# FLASH call that only fires when Stage 1 actually found signal.
# Combined output matches the legacy ANALYZE_FINDINGS_PROMPT shape so
# downstream code is unchanged.

EXTRACT_FINDINGS_PROMPT = """You extract verifiable facts from search results. Pure extraction — do NOT judge evidentiary weight, contradiction, or hypothesis support (another pass handles that). Your only bias is relevance to the investigation below.

Investigation context (for relevance only — use this to decide which facts to pick, not to evaluate them):
- Query: {query}
- Current hypothesis: {hypothesis}
- Priority focus: {relevance_hint}

Search Results for "{search_term}":
{search_results}

1. KEY_FACTS (max 10): the most concrete, specific facts that are ALSO plausibly relevant to the investigation context above. Prefer facts with dates, amounts, party names, and facts that name entities or events in the query or hypothesis.
   Format each fact as: {{"fact": "under-100-char text", "source_file": "filename_if_determinable", "subject": "entity", "predicate": "snake_case_verb", "object": "value_or_target"}}
   - source_file: file identifier exactly as it appears in the results (may be "filename.pdf" or "folder/filename.pdf")
   - subject / predicate / object: REQUIRED except for purely procedural facts with no entity relationship (omit all three then)
   - subject examples: {domain_subject_examples}
   - predicate examples: {domain_predicate_examples}
   - object examples: {domain_object_examples}
   - Include dates, amounts, party names when present

2. MENTIONED_LEADS: referenced documents, named individuals, dates, or cross-references that look worth investigating next. List only what's mentioned; no priority.
   Format: [{{"desc": "under-120-char description of what to investigate"}}]

3. MENTIONED_SEARCHES: follow-up search terms that naturally appear. Simple literal phrases — no boolean operators, no quoted sub-expressions.
   Format: ["term1", "term2"]

Respond in COMPACT JSON (under 2500 chars):
{{
  "key_facts": [...],
  "mentioned_leads": [...],
  "mentioned_searches": [...]
}}
"""

REASON_FINDINGS_PROMPT = """You are a senior analyst reasoning over pre-extracted facts. You do not re-read source documents — everything you need is below.

Investigation query: {query}
Current hypothesis: {hypothesis}
{issue_focus}
{research_alignment_guidance}

Pre-extracted facts (index → fact):
{facts_block}

Pre-extracted leads (index → description):
{leads_block}

Pre-extracted candidate searches:
{searches_block}

REASON:

1. For each fact, classify its relation to the current hypothesis: "supports" / "attacks" / "neutral".
   Format: "fact_issue_relations": [{{"fact_idx": 0, "relation": "supports"}}, ...]

2. Relationships between facts in this batch (optional — only clear ones):
   Format: "fact_relationships": [{{"from_idx": 0, "to_idx": 1, "relation": "corroborates|contradicts|supersedes|supports"}}]

3. Update the working hypothesis ONLY if these facts meaningfully change it, otherwise null.
   Format: "hypothesis_update": "one-sentence update, or null"

4. PREDICATES (only when an Issue Focus block appears above):
   "predicates_satisfied": list the VERBATIM "Element to prove" text that the extracted facts CLEARLY and DIRECTLY establish.
   "predicates_contested": list the VERBATIM element text where the extracted facts support BOTH sides (evidence of conflict).
   Empty arrays if no Issue Focus or no elements are clearly established/contested.

5. Rank the leads by investigation priority (0.0–1.0):
   Format: "lead_priorities": [{{"lead_idx": 0, "priority": 0.8}}]

6. Rank the candidate searches (0.0–1.0). You may drop duds or add up to 2 new terms you think would help fill evidence gaps:
   Format: "next_search_priorities": [{{"term": "phrase", "priority": 0.9}}]

Respond in COMPACT JSON only (under 2000 chars):
{{
  "fact_issue_relations": [...],
  "fact_relationships": [...],
  "hypothesis_update": null,
  "predicates_satisfied": [],
  "predicates_contested": [],
  "lead_priorities": [...],
  "next_search_priorities": [...]
}}
"""

# Pre-computed template hashes for search-analysis cache versioning.
# _ANALYZE_PROMPT_VER covers the merged two-stage contract — any edit
# to either prompt invalidates all cached analyses from the old shape.
import hashlib as _hashlib
_ANALYZE_PROMPT_VER = _hashlib.sha256(
    (EXTRACT_FINDINGS_PROMPT + "\n--stage2--\n" + REASON_FINDINGS_PROMPT).encode()
).hexdigest()[:12]
del _hashlib  # avoid polluting module namespace

DEEP_READ_PROMPT = """You are an expert analyst performing detailed document review.

Document: {filename}
Page Range: {page_range}

Content:
{content}

Query Context: {query}
Current Investigation Focus: {focus}
{domain_vocabulary}
{mna_section}
CONDUCT A THOROUGH ANALYSIS. Extract ALL relevant information — do not truncate or omit details.

EXHAUSTIVE EXTRACTION RULE: When a document contains a table, list, schedule, or enumeration
with N items (e.g., 9 geographic markets, 11 facilities, 15 contracts, 8 provisions), you
MUST extract data for ALL N items — not just the top 3-5. Extracting only the most prominent
examples while skipping the rest is a critical failure. Count the items and verify your
extraction is complete.

1. KEY FACTS (extract ALL relevant facts — no artificial limit): Extract facts that are:
   - Directly relevant to the query/focus
   - Specific (include EXACT section numbers, clause references, dollar amounts, percentages, thresholds, defined terms, time periods)
   - Keep each fact under 250 characters — include section numbers, specific values, and comparison points
   - Format each fact as: {{"fact": "...", "page": N, "issue_relation": "supports|attacks|neutral", "effective_date": "YYYY-MM-DD or null", "subject": "entity_name", "predicate": "snake_case_verb", "object": "value_or_target"}}
   - issue_relation: whether the fact SUPPORTS the investigation focus, ATTACKS/undermines it, or is NEUTRAL
   - effective_date: ISO date when this fact became effective/occurred (null if not temporally scoped)
   - subject: entity performing the action — REQUIRED; provide best-effort
   - predicate: verb/action in snake_case — REQUIRED
   - object: what the predicate applies to — REQUIRED; include the key value
   - Omit subject/predicate/object ONLY when the fact has no entity relationship (purely procedural)
{domain_deep_read_examples}

2. CRITICAL QUOTES (extract up to 25 most important passages): Identify the most important passages:
   - Direct admissions or acknowledgments
   - Terms that define obligations or rights
   - Statements of fact that support/contradict claims
   - Language that creates binding obligations

3. ENTITIES: Extract with role/context:
   - People: name, role, significance
   - Companies: name, relationship to parties
   - Dates: date, what happened, significance
   - Amounts: value, context, what it represents
   - Products/technology/systems: product lines, algorithms, licensed technology, ERP/software platforms, model names, version numbers
   - Securities/equity: RSUs, options, share counts, share prices, vesting/acceleration quantities

4. NUMERIC FACTS (SO-6 — extract ALL monetary amounts, dates, rates, counts):
   For each number, provide a structured object:
   - kind: "amount" | "date" | "date_range" | "rate" | "balance" | "count"
   - subject: one-word subject type — {domain_numeric_subjects}
   - subject_id: specific identifier if present (e.g. {domain_numeric_subject_id_example}, null if none)
   - raw: exact text from document (preserve original wording)
   - value: for amounts/rates/counts: numeric value. For dates: ALWAYS use ISO format YYYY-MM-DD (e.g. "2023-03-15"). For date_range: use "YYYY-MM-DD/YYYY-MM-DD". If only month is known use first of month (e.g. "2023-03-01"). If only year, use "2023-01-01". If only quarter, use first day of quarter (Q1="01-01", Q2="04-01", Q3="07-01", Q4="10-01").
   - date_precision: REQUIRED for kind="date" or "date_range": "day" | "month" | "quarter" | "year" (how precise the original date is)
   - currency: "USD" etc. for amounts (null if not monetary)
   - context: brief label of what this number represents (max 60 chars)
   - page: page number where this number appears (integer, null if unknown)
   - assertion_idx: 0-based index into key_facts of the fact this number comes from (null if none)

5. DOCUMENT RELATIONSHIPS:
   - References to other documents (attachments, exhibits)
   - Prior agreements or communications mentioned
   - Events that require corroboration elsewhere

6. FACT RELATIONSHIPS (SO-2 — up to 10 most important): Identify logical relationships
   BETWEEN the key_facts you listed above, using their 0-based indices.
   Relation types: "supports" (A reinforces B), "attacks" (A undermines B),
   "contradicts" (A directly conflicts with B), "corroborates" (A independently confirms B),
   "supersedes" (A replaces B as the authoritative statement).

7. RED FLAGS & CONCERNS:
   - Ambiguous or potentially misleading language
   - Missing expected provisions
   - Contradictions within the document
   - Issues requiring interpretation or expert judgment
   - ADVERSE EVIDENCE: Any statement, admission, or language that could be used adversarially
     (e.g., anticompetitive intent, awareness of deficiencies, retaliatory motive, willful
     noncompliance, or pricing power). For each: extract the EXACT quote, speaker/author,
     section/page reference, and a one-line explanation of why it is adverse.

8. DOC SOURCE ROLE (SO-5 — classify this document by its content, NOT its filename):
   Choose exactly one of: advocacy, operative, authoritative, procedural, informal, draft, post_hoc, unknown
   - advocacy: position papers, briefs, proposals authored to advance a party's interest
   - operative: signed agreements, executed contracts, official orders with binding effect
   - authoritative: statutes, regulations, standards, published guidelines, peer-reviewed findings
   - procedural: filings, applications, process documents, compliance submissions
   - informal: emails, messages, notes, chats, internal memos, correspondence
   - draft: unsigned or unapproved versions — not yet operative
   - post_hoc: analysis, reports, expert opinions written after the events to explain or assess
   - unknown: cannot determine from document content alone

9. DOCUMENT CARD (classify this document for the matter model):
   - doc_type: broad category — "contract", "filing", "correspondence", "invoice", "order", "memo", "report", "notice", "exhibit", "other"
   - doc_subtype: specific subtype — e.g. "services_agreement", "demand_letter", "email_chain", "expert_report"
   - title: document title or best descriptive label (e.g. "Master Services Agreement between Acme and Beta Corp")
   - author: primary author name if identifiable (null if unknown)
   - sender: sender if correspondence (null otherwise)
   - recipient: recipient if correspondence (null otherwise)
   - creation_date: ISO date of creation/execution if stated (null if unknown)
   - effective_date: ISO date when terms take effect (null if not applicable)
   - operative_status: "operative" (binding/in-force), "superseded", "draft", "expired", "disputed", "unknown"
   - purpose: one-sentence description of what this document does (max 80 chars)
   - rhetorical_posture: "neutral", "adversarial", "cooperative", "protective", "informational"
   - unresolved_flags: list of open questions about this document, e.g. ["missing signature page", "references Amendment 3 not in file"]

Respond in JSON (be thorough — include ALL relevant provisions, section numbers, and defined terms):
{{
    "key_facts": [{{"fact": "...", "page": N, "issue_relation": "supports", "effective_date": "2023-03-15", "subject": "Party A", "predicate": "agreed_to_pay", "object": "50000 USD"}}],
    "quotes": [{{"text": "...", "page": N}}],
    "entities": {{"people": ["name1"], "dates": ["date1"], "amounts": ["$X"], "companies": ["co1"]}},
    "numeric_facts": [{{"kind": "amount", "subject": "invoice", "subject_id": "Invoice #1042", "raw": "$50,000", "value": 50000, "currency": "USD", "context": "payment due", "page": 3, "assertion_idx": 2}}, {{"kind": "date", "subject": "payment", "subject_id": null, "raw": "March 2023", "value": "2023-03-01", "date_precision": "month", "context": "payment due date", "page": 1, "assertion_idx": 0}}],
    "fact_relationships": [{{"from_idx": 0, "to_idx": 2, "relation": "supports"}}],
    "connections": ["doc reference 1"],
    "concerns": ["issue 1"],
    "doc_source_role": "advocacy|operative|authoritative|procedural|informal|draft|post_hoc|unknown",
    "doc_type": "contract|filing|correspondence|invoice|order|memo|report|notice|exhibit|other",
    "doc_subtype": "specific_subtype_here",
    "title": "Descriptive document title",
    "author": "Author Name or null",
    "sender": "Sender or null",
    "recipient": "Recipient or null",
    "creation_date": "YYYY-MM-DD or null",
    "effective_date": "YYYY-MM-DD or null",
    "operative_status": "operative|superseded|draft|expired|disputed|unknown",
    "purpose": "One-sentence description of what this document does",
    "rhetorical_posture": "neutral|adversarial|cooperative|protective|informational",
    "unresolved_flags": ["any open questions about this document"],
    {transaction_context_schema}"contract_card": {{
        "contract_name": "Full contract/agreement name",
        "counterparty": "Other party name",
        "assignment_clause": "Section X.Y — exact language or null",
        "change_of_control_definition": "Section X.Y — definition text and thresholds, or 'ABSENT'",
        "exact_trigger_language": ["short exact phrases that control trigger analysis"],
        "consent_requirements": "Prior written consent / not unreasonably withheld / etc., or null",
        "timing_windows": ["notification: X days", "cure period: Y days", "termination notice: Z days"],
        "consent_timing_sequence": "conditional sequence such as consent within X days after closing then termination on Y days notice",
        "termination_rights": "Who can terminate, under what conditions, with what notice",
        "event_of_default_consequences": "Event of Default / acceleration / prepayment consequences, or null",
        "carve_outs": ["carve-out 1 with specific conditions", "carve-out 2"],
        "dependency_relationships": [
            {{"component": "embedded/licensed product or technology",
              "host_product": "product line or system that uses it",
              "relationship_type": "embedded_in|licensed_to|integrated_with|depends_on",
              "revenue_attribution": "dollar amount and period if stated, or null",
              "source_section": "Section X.Y or Schedule X",
              "contract_or_vendor": "licensor/vendor name or contract governing this dependency, or null"}}
        ],
        "unreviewed_dependency_contracts": ["third-party product/software/license contracts mentioned but not reviewed"],
        "schedule_entries": [
            {{"schedule_ref": "Schedule X.XX",
              "row_index": 0,
              "target_label": "counterparty or contract name",
              "metric": "revenue|commitment|obligation|other",
              "value": "dollar amount or description",
              "period": "TTM|annual|quarterly|as-of-date or null",
              "linked_contract": "contract name if identifiable, or null"}}
        ],
        "revenue_exposure": "dollar amount and percentage if calculable, or null",
        "financial_operands": ["specific operands for calculations: TTM revenue, drawn debt, RSU count, share price, coverage limit"],
        "coverage_limits": ["per-occurrence / aggregate / tail limits"],
        "post_closing_coverage_gaps": ["coverage gap for successor, parent, affiliate, or post-closing new products"],
        "downstream_indirect_risks": [{{"contract_name": "specific contract",
            "provision_section": "Section X.Y",
            "trigger_language": "direct or indirect ownership/control language",
            "affected_actor_role": "acquirer|target|parent",
            "re_trigger_scenario": "description of what future event could re-trigger"}}],
        "risk_rating_candidate": "Critical|High|Moderate|Low with one-sentence reason",
        "action_items": ["pre-closing consent, waiver, amendment, payoff, replacement policy, review missing dependency"],
        "missing_expected_provisions": ["provision type expected but absent"]
    }},
    "provision_comparisons": [
        {{"provision": "provision name (e.g. Interest Rate Floor, Leverage Ratio)",
          "value": "exact value from this document (number, percentage, threshold)",
          "section_ref": "Section X.Y or clause reference",
          "source_role": "original|markup|playbook|commitment_letter|credit_memo",
          "value_type": "threshold|cap|rate|period|presence|basket|trigger"}}
    ],
    "regulatory_data": [
        {{"category": "market_share|hhi|hot_doc|barrier|remedy|timeline|jurisdiction|overlap|synergy|accretion|valuation",
          "entity": "company or market name",
          "value": "exact data point (number, percentage, or quote)",
          "source_detail": "page/slide/section reference",
          "significance": "brief note on why this matters"}}
    ],
    "adverse_evidence": [
        {{"quote": "exact verbatim quote from the document",
          "speaker": "name/role of the person who said/wrote it",
          "section_ref": "section, slide, or page reference",
          "adverse_theory": "one-line explanation of why this is problematic"}}
    ],
    "extraction_completeness": [
        {{"group": "name of the table/list/schedule (e.g. 'MSA market shares', 'covenant grid')",
          "items_in_source": "integer: how many items are in the source table/list",
          "items_extracted": "integer: how many you actually extracted",
          "complete": true}}
    ]
}}
"""

_MNA_DEEP_READ_SECTION = """
M&A / CHANGE-OF-CONTROL RELEVANCE EXPANSION:
If the query involves change of control, merger, acquisition, assignment, or material-contract review, treat these as relevant even when the phrase "change of control" is absent:
- assignment, transfer, delegation, deemed assignment, assignment by operation of law
- merger, consolidation, successor, assignee, affiliate transfer
- direct or indirect ownership/control, ultimate ownership/control
- consent, prior written consent, approval, not unreasonably withheld
- termination rights, acceleration, run-off coverage, pricing/buy-out formula, carve-out
- early termination fee, prepayment, event of default, cure period

For leases, Section-style assignment/transfer clauses are CoC-relevant when tenant ownership/control changes are deemed assignments, landlord consent is required, a consent standard applies, landlord may terminate, or a fee/rent formula applies.

Every matching provision MUST be extracted as a key_fact with exact section number, trigger family, consent requirement, consequence, and any fee/timing formula.

M&A MATERIAL-CONTRACT MUST-CAPTURE DETAILS:
When reviewing acquisition, merger, change-of-control, assignment, or material-contract diligence documents:
- Preserve legally operative phrases verbatim when short, especially "whether by operation of law or otherwise", "direct or indirect", and "ultimate ownership or control".
- Treat schedules, exhibits, tables, side letters, declarations pages, and pricing/revenue schedules as first-class evidence; do not stop at the main body of the agreement.
- CRITICAL: Every row in a schedule or table that contains a party, contract, product, metric, amount, threshold, or period MUST be extracted as a schedule_entries[] item in the contract_card. Do not summarize schedules — enumerate each row.
- CRITICAL: When a schedule lists counterparty-specific revenue (e.g., "approximately $X million in trailing twelve-month revenue attributable to this agreement"), extract that as the ACTUAL revenue operand — it takes precedence over minimum purchase commitments stated in the agreement body.
- Separate actual counterparty/product TTM revenue from minimum purchase commitments, facility commitments, sample calculations, and limits. The schedule/disclosure figure is the real revenue; the body's minimum commitment is a floor, not actual revenue.
- For credit agreements, extract both facility/commitment size and current outstanding/drawn amount; label which amount is the mandatory prepayment exposure.
- For default provisions, state whether a Change of Control is an Event of Default and extract acceleration, termination of commitments, and prepayment consequences.
- For consent/termination mechanics, preserve conditional timing chains (e.g. consent not obtained within X days after closing -> termination on Y days' notice).
- For carve-outs, extract every condition and threshold, then extract facts needed to test whether the condition is satisfied.
- For equity awards, extract unvested award count, exchange ratio, per-share value, rollover/continuing-vesting treatment, and any automatic acceleration conflict.
- For insurance, extract run-off/tail mechanics, aggregate limits, successor/new-product exclusions, and post-closing go-forward coverage gaps.
- For embedded products, technology, and ERP/software systems, extract each dependency as a dependency_relationships[] item with component, host_product, relationship_type, revenue_attribution if available, and contract_or_vendor.
- When the document reveals the transaction structure (target, acquirer, merger subsidiary, parent), populate transaction_context. Extract actor names only from transaction/disclosure language, not from generic contract role labels.
"""

_COMPARISON_DEEP_READ_SECTION = """
COMPARISON/MARKUP ANALYSIS — EXHAUSTIVE PROVISION-LEVEL EXTRACTION:
This is a document comparison task. For EVERY provision below, extract the EXACT
values from THIS document. Do not paraphrase or approximate — use the exact numbers,
percentages, thresholds, and defined terms as written.

TRACKED CHANGES / MARKUP MARKERS:
This document may contain tracked changes represented as:
- [DELETED: text] — text that was REMOVED from the original
- [ADDED: text] — text that was INSERTED as a proposed change
When you see these markers:
- The [DELETED: ...] text is the ORIGINAL value → extract with source_role "original"
- The [ADDED: ...] text is the MARKUP/PROPOSED value → extract with source_role "markup"
- Text WITHOUT markers is UNCHANGED from the original
- A provision with ONLY [ADDED: ...] and no [DELETED: ...] is a NEW provision added by the markup
- A provision with ONLY [DELETED: ...] and no [ADDED: ...] is a provision REMOVED by the markup
You MUST scan the ENTIRE document for ALL [DELETED:] and [ADDED:] markers. Each one
represents a change that must be captured as a provision_comparison entry.

DOCUMENT FORMAT GUIDANCE:
- If this is a REDLINE/MARKUP document with [DELETED:]/[ADDED:] markers or visual
  formatting (strikethroughs/underline), extract BOTH the original and changed values.
- If this is a CLEAN original document, extract all values with source_role "original".
- If this is a BORROWER'S markup, values that differ from the original/commitment
  letter are the PROPOSED changes. Extract both the baseline and the proposal.
- If this is a COMMITMENT LETTER, its terms are the baseline — extract as "commitment_letter".
- If this is a CREDIT MEMO, its terms are internal lender analysis — extract as "credit_memo".

CRITICAL: Extract EVERY provision that differs between documents, not just the ones
listed below. The list below is a minimum — if you find additional provisions with
specific values, extract those too. If a provision has a GRID or TIER structure,
extract EVERY tier/step-down with its exact breakpoint and value.

CRITICAL PROVISIONS TO EXTRACT (with expected value types):
1. Interest Rate Floor: exact basis points (e.g. 0.00%, 0.75%)
2. Applicable Margin / Spread: each leverage-tier rate (e.g. ≤2.50x→200bps, >3.25x→300bps)
3. Commitment Fee: flat rate OR each tier with leverage breakpoints
4. Financial Covenants — FCCR: exact minimum ratio (e.g. 1.25x, 1.35x)
5. Financial Covenants — Leverage Ratio: each step-down date and threshold
6. Financial Covenants — Revolver Testing Threshold: % of commitment and dollar amount
7. Financial Covenants — Testing Holiday: number of quarters, conditions
8. EBITDA Add-backs: non-recurring cap ($/year AND $/lifetime), synergy cap (% AND months),
   EACH individually named add-back category (business interruption, restructuring,
   business optimization, purchase accounting, non-recurring losses, etc.)
9. Cash Netting Cap: exact dollar amount for leverage calculation AND for ECF sweep separately
10. Permitted Acquisitions: individual basket ($), aggregate basket ($), pro forma cushion (x),
    pro forma compliance requirement (present/absent, when springing covenant not in effect)
11. Restricted Payments: hard dollar cap ($), ECF percentage, pro forma leverage test threshold,
    builder basket leverage test threshold
12. ECF Sweep: percentage at EACH leverage tier (step-downs vs flat), de minimis threshold ($),
    ECF definition deductions (list any catch-all deductions added)
13. Cross-Default Threshold: exact dollar amount
14. Change of Control: exact ownership percentage trigger, key person triggers
15. MAE/MAC Definition: presence of "taken as a whole", "material" qualifiers in sub-clauses
16. Extension Options: number of extensions, duration, fee, notice period
17. Anti-Layering Covenant: present/absent, specific restrictions
18. MFN (Most Favored Nation): present/absent, scope, margin adjustment trigger (bps), sunset
19. Reinvestment Period: exact number of days for asset sale reinvestment
20. Reporting Requirements: financial statement delivery deadlines
21. Prepayment Provisions: soft call period, prepayment premium %, repricing protection
22. Equity Cure: methodology (EBITDA addback vs debt reduction), consecutive cure permission,
    lifetime cap (number of cures), cure period (business days), over-cure limitation
23. Incremental Facilities: free-and-clear basket ($), ratio-based incurrence test (x),
    junior lien permission, Disqualified Lender restriction for incremental lenders
24. Investment/RP Baskets: general investment basket ($), general RP basket ($/%),
    Available Equity Amount basket (capped/uncapped, leverage test), Similar Business definition
25. IP/Asset Transfers: transfers to unrestricted subsidiaries or non-Loan Party subsidiaries,
    fair value requirements, J. Crew-style IP transfer baskets
26. Voting/Assignment: Serta-style priming provisions, open market purchase provisions,
    CLO/DQ lender carveouts for eligible assignees
27. Governing Law: jurisdiction (New York, Delaware, etc.)

MANDATORY: For EACH provision, create TWO entries in provision_comparisons when you can
identify both the original and the changed value — one with source_role "original" and
one with source_role "markup". This is critical for generating the deviation analysis.

PROVISION COMPLETENESS CHECK: After extracting all provisions, review the list above
(items 1-27) and confirm you checked EACH ONE. For provisions that exist in the document
but have NO change, you may skip them. For provisions where you found a change (including
additions or deletions), you MUST extract both original and markup values. If you are
uncertain whether a provision changed, extract it — false positives are better than misses.

For EACH provision found, create a key_fact with:
- The EXACT value (not "changed" or "modified" — the actual number)
- The section/clause reference
- Whether this is an original/baseline value or a markup/proposed value

If this document is a NEGOTIATION PLAYBOOK, extract for each provision:
- Preferred position (ideal value)
- Acceptable fallback (compromise value)
- Hard no threshold (walk-away value)

Include a "provision_comparisons" array in your JSON response with structured rows:
"provision_comparisons": [
    {"provision": "name", "value": "exact value from this document", "section_ref": "Section X.Y",
     "source_role": "original|markup|playbook|commitment_letter|credit_memo",
     "value_type": "threshold|cap|rate|period|presence|basket|trigger"}
]
"""

_REGULATORY_DEEP_READ_SECTION = """
REGULATORY/ANTITRUST ANALYSIS — EXHAUSTIVE DATA EXTRACTION:
This is a regulatory analysis task. Extract ALL quantitative and factual data needed
for competitive effects analysis, HHI calculations, and enforcement assessment.

CRITICAL: You must extract data for EVERY geographic market, EVERY product market, and
EVERY entity mentioned in the document — not just the most prominent examples. If the
document discusses 9 MSAs, extract data for all 9. If it mentions 4 product markets,
extract data for all 4. Partial extraction is a critical failure.

CRITICAL DATA CATEGORIES:
1. Market Shares: company name, share percentage, geographic market (MSA), product market, source, date
   → Extract for EVERY geographic market mentioned, even if data is less detailed for some
2. Market Definition: EVERY product market boundary (bulk gases, packaged gases, specialty, CO2, etc.),
   EVERY geographic market (each MSA/region), substitutability analysis
3. HHI Components: pre-merger and post-merger HHI, delta, shares for EACH competitor in EACH market
   → If the document has HHI data for multiple markets, extract ALL of them
4. Hot Documents: EVERY internal quote showing competitive harm awareness, anticompetitive intent,
   pricing power, market elimination, or customer conversion plans. Include:
   - Exact quote text (verbatim, not paraphrased)
   - Speaker/author name
   - Document section or slide number
   - WHY this language is problematic (e.g., "suggests anticompetitive motive")
5. Maverick/Disruptive Competitor Evidence: specific pricing, entry timing, customer diversion,
   margin compression (with exact basis points or dollar amounts)
6. Barriers to Entry: type, height, timeframe, and the parties' own admissions about barriers
7. Customer Overlap: customer name, share of purchases, diversion ratio, CRM data about alternatives
8. Divestiture/Remedy Data: EVERY candidate asset/facility by name, location, revenue, buyer qualifications
9. Efficiency Claims: type, magnitude, whether merger-specific, and whether verifiable
   → Flag "synergies" that are actually price increases (e.g., "pricing optimization" = eliminating competition)
10. Deal Timeline: EVERY date (signing, HSR filing, waiting period, outside date, extensions, fund terms)
11. Internal Strategy Documents: EVERY quote about competitive strategy, pricing, market positioning,
    customer conversion, or facility consolidation — each with section reference and speaker
12. Contractual Provisions: divestiture caps, ASU/asset exclusions, breakup fees, termination triggers,
    hell-or-high-water obligations — with exact dollar amounts and conditions

13. Legal Framework: governing statute (e.g., Clayton Act Section 7, Sherman Act Section 1),
    structural presumption thresholds (HHI >1800 AND delta >200 per 2023 Merger Guidelines),
    reviewing agency (FTC vs DOJ), HSR filing thresholds, size-of-person test amounts
14. Defenses: efficiency defense (merger-specific, verifiable, cognizable?), failing firm
    defense (profitable? other buyers?), ease-of-entry defense (strong? weak?),
    buyer power defense. For each, state whether available on these facts.
15. Remedy Analysis: fix-it-first vs consent decree tradeoffs, structural vs behavioral,
    ASU/production asset divestiture vs distribution-only, potential buyers by name,
    buyer adequacy concerns (scale, financial capacity, operational capability)
16. Timeline/Procedure: initial 30-day waiting period, Second Request likelihood,
    compliance timeline estimate, outside date adequacy (specific dates), fund term
    pressures, integration milestones that conflict with regulatory timeline

HOT DOCUMENT IDENTIFICATION — CRITICAL:
"Hot documents" are internal communications whose language could be used adversarially
by regulators to demonstrate anticompetitive intent or harm. Specifically flag:
- Board presentations mentioning "eliminating" competitors, "pricing optimization",
  "restoring pricing levels", or "removing independent competitors"
- Strategy memos about market consolidation, capacity reduction, or customer conversion
- Emails linking deal timing to specific competitive contracts or bids
- Internal acknowledgments of high barriers to entry, margin compression from competitors,
  or characterizations of targets as "maverick" or "disruptive"
- Integration plans mentioning facility closures or "rationalization" in overlap markets
- Financial projections labeled as "synergies" that are actually price increases
Each hot document MUST be extracted as BOTH a key_fact AND a regulatory_data entry with
category "hot_doc", including the EXACT verbatim quote, speaker/author, and section/slide.

For EACH data point, create a key_fact with:
- The EXACT number, percentage, or quote (not summaries)
- The specific page, slide, or section reference
- The speaker/author if identifiable
- Tag: [REGULATORY:CATEGORY] prefix (e.g., [REGULATORY:MARKET_SHARE], [REGULATORY:HOT_DOC])

Include a "regulatory_data" array in your JSON response:
"regulatory_data": [
    {"category": "market_share|hhi|hot_doc|barrier|remedy|timeline|jurisdiction|overlap|synergy|accretion|valuation|framework|defense",
     "entity": "company or market name", "value": "exact data point",
     "source_detail": "page/slide/section", "significance": "brief note"}
]
"""

_DOMAIN_DEEP_READ_VOCABULARY = {
    "legal": (
        "DOMAIN CONTEXT: Legal matter analysis.\n"
        "- doc_type priorities: contract, filing, correspondence, order, memo, exhibit, report, invoice, notice\n"
        "- Source roles: advocacy (briefs, motions), operative (executed agreements, orders), "
        "authoritative (statutes, regulations), procedural (filings, applications)\n"
        "- Key predicates: agreed_to_pay, breached_obligation, executed_contract, filed_motion, "
        "disputes_claim, owes_damages, failed_to_perform, warranted_condition, "
        "deems_assignment, prohibits_assignment, requires_prior_consent, sets_consent_standard, "
        "grants_termination_right, accelerates_obligation, defines_change_of_control, "
        "creates_runoff_coverage, excludes_successor_coverage, sets_buyout_formula, "
        "sets_early_termination_fee, triggers_prepayment, triggers_event_of_default\n"
        "- Numeric focus: damages, payment amounts, contract values, deadlines, limitation periods, "
        "RSU/option counts, share prices, revenue figures, EBITDA multiples, termination fees, coverage limits"
    ),
    "finance": (
        "DOMAIN CONTEXT: Financial analysis.\n"
        "- doc_type priorities: report (10-K, 10-Q, annual report), correspondence (analyst note, "
        "earnings call transcript), filing (SEC filing, regulatory submission), memo (investment memo, "
        "credit memo), invoice, notice (guidance update, earnings warning)\n"
        "- Source roles: advocacy (management commentary, investor presentations), operative (loan "
        "agreements, indentures), authoritative (auditor opinions, regulatory standards), "
        "informal (analyst estimates, market commentary)\n"
        "- Key predicates: reported_revenue, recognized_expense, disclosed_risk, restated_earnings, "
        "breached_covenant, exceeded_guidance, downgraded_rating, missed_estimate\n"
        "- Numeric focus: revenue, EBITDA, margins, debt/equity ratios, guidance ranges, EPS"
    ),
    "coding": (
        "DOMAIN CONTEXT: Software engineering analysis.\n"
        "- doc_type priorities: report (design doc, architecture doc, test report), filing (PR, issue, "
        "RFC), correspondence (code review, commit message), memo (decision record, postmortem), "
        "notice (deprecation notice, security advisory)\n"
        "- Source roles: advocacy (proposal, RFC), operative (merged PR, release notes), "
        "authoritative (specification, standard, official documentation), "
        "informal (commit message, chat, code comment)\n"
        "- Key predicates: introduced_bug, fixed_issue, deprecated_api, changed_behavior, "
        "added_dependency, removed_feature, violated_constraint, passed_test\n"
        "- Numeric focus: latency, error rates, test coverage, LOC, version numbers, SLA thresholds"
    ),
    "academic_research": (
        "DOMAIN CONTEXT: Academic research analysis.\n"
        "- doc_type priorities: report (journal article, conference paper, thesis), filing (grant "
        "proposal, IRB submission), memo (research note, lab notebook), notice (retraction, "
        "correction, erratum), correspondence (peer review, editorial decision)\n"
        "- Source roles: advocacy (grant proposal, position paper), operative (published findings, "
        "accepted methodology), authoritative (peer-reviewed journal, systematic review, meta-analysis), "
        "post_hoc (commentary, retrospective analysis)\n"
        "- Key predicates: demonstrated_effect, found_no_significance, replicated_finding, "
        "contradicted_hypothesis, measured_outcome, controlled_for_variable\n"
        "- Numeric focus: p-values, effect sizes, confidence intervals, sample sizes, R-squared"
    ),
    "biomedical": (
        "DOMAIN CONTEXT: Biomedical / clinical analysis.\n"
        "- doc_type priorities: report (clinical trial report, case study, lab results), filing (FDA "
        "submission, IND application, NDA), memo (clinical protocol, investigator brochure), "
        "correspondence (FDA letter, DSMB report), notice (safety alert, label change)\n"
        "- Source roles: advocacy (sponsor materials, marketing claims), operative (FDA-approved label, "
        "clinical protocol), authoritative (clinical guideline, systematic review, Phase III results), "
        "post_hoc (case report, retrospective analysis, real-world evidence)\n"
        "- Key predicates: demonstrated_efficacy, showed_adverse_event, met_primary_endpoint, "
        "failed_safety_threshold, exceeded_non_inferiority_margin, achieved_response_rate\n"
        "- Numeric focus: hazard ratios, odds ratios, NNT, p-values, survival rates, dosage, AE frequency"
    ),
}

_DOMAIN_DEEP_READ_EXAMPLES: dict[str, dict[str, str]] = {
    "legal": {
        "deep_read_examples": (
            '   - subject examples: "plaintiff", "defendant", "contracting_party"\n'
            '   - predicate examples: "agreed_to_pay", "executed_contract", "filed_motion"\n'
            '   - object examples: "50000 USD by March 2023", "the services agreement"'
        ),
        "numeric_subjects": '"invoice" | "payment" | "fee" | "damages" | "balance" | "rate" | "deposit" | "penalty" | "revenue" | "ttm_revenue" | "credit_drawn" | "facility_commitment" | "coverage_limit" | "rsu_count" | "share_price" | "ebitda" | "buyout_multiple" | "termination_fee" | "purchase_commitment" | "other"',
        "numeric_subject_id_example": '"Invoice #1042", "Payment #3", "Counterparty A TTM revenue", "Revolving Loans outstanding", "Executive unvested RSUs"',
    },
    "finance": {
        "deep_read_examples": (
            '   - subject examples: "Company_X", "auditor", "management"\n'
            '   - predicate examples: "reported_revenue", "restated_earnings", "breached_covenant"\n'
            '   - object examples: "12.5M USD", "the credit facility", "Q3 2024"'
        ),
        "numeric_subjects": '"revenue" | "expense" | "asset" | "liability" | "ratio" | "margin" | "rate" | "share_price" | "other"',
        "numeric_subject_id_example": '"Revenue FY2024", "Debt Facility #2"',
    },
    "coding": {
        "deep_read_examples": (
            '   - subject examples: "auth_service", "UserController", "CI_pipeline"\n'
            '   - predicate examples: "introduced_bug", "deprecated_api", "changed_behavior"\n'
            '   - object examples: "v2.3.1", "login endpoint", "rate limit threshold"'
        ),
        "numeric_subjects": '"latency" | "error_rate" | "coverage" | "version" | "threshold" | "count" | "size" | "duration" | "other"',
        "numeric_subject_id_example": '"PR #1042", "Issue #567"',
    },
    "academic_research": {
        "deep_read_examples": (
            '   - subject examples: "treatment_group", "Smith_et_al_2023", "variable_X"\n'
            '   - predicate examples: "demonstrated_effect", "controlled_for", "found_no_significance"\n'
            '   - object examples: "p=0.003", "n=1200", "depression score"'
        ),
        "numeric_subjects": '"sample_size" | "effect_size" | "p_value" | "confidence_interval" | "correlation" | "mean" | "variance" | "duration" | "other"',
        "numeric_subject_id_example": '"Study A", "Experiment 3"',
    },
    "biomedical": {
        "deep_read_examples": (
            '   - subject examples: "Drug_X", "treatment_arm", "FDA"\n'
            '   - predicate examples: "demonstrated_efficacy", "showed_adverse_event", "met_primary_endpoint"\n'
            '   - object examples: "HR=0.72", "Grade 3 AE", "overall survival endpoint"'
        ),
        "numeric_subjects": '"hazard_ratio" | "odds_ratio" | "nnt" | "dosage" | "ae_rate" | "survival_rate" | "p_value" | "sample_size" | "other"',
        "numeric_subject_id_example": '"Trial NCT0001", "Cohort B"',
    },
}

_DOMAIN_ORIENTATION_CONTEXT: dict[str, dict[str, str]] = {
    "legal": {
        "issue_types": "claim|defense|exposure|interpretation|procedural|evidentiary|condition_precedent|waiver|diligence_red_flag|compliance_failure",
        "issue_type_descriptions": (
            "Issue types: claim=a primary assertion or position, defense=a counterargument or defense, "
            "exposure=a risk, liability, or cost component, interpretation=a disputed interpretation of terms, "
            "procedural=a procedural barrier or threshold issue, evidentiary=an evidentiary bottleneck, "
            "condition_precedent=a condition that must be satisfied, waiver=a waiver or estoppel defense, "
            "diligence_red_flag=a due-diligence risk item, compliance_failure=a regulatory or policy violation."
        ),
        "predicate_examples": (
            'e.g., for breach of contract: ["contract existence and terms", '
            '"defendant\'s obligation", "failure to perform", "resulting damages"]'
        ),
        "document_priorities": (
            "- Primary source documents (contracts, pleadings, agreements) over secondary (correspondence)\n"
            "- Documents whose filenames suggest they contain key evidence for the query"
        ),
        "search_examples": 'Good: "breach of contract", "Invoice_March.xlsx", "termination clause"',
    },
    "finance": {
        "issue_types": "revenue_recognition|risk_exposure|covenant_compliance|valuation_dispute|disclosure_gap|audit_finding|forecast_deviation|regulatory_violation|related_party_transaction|going_concern",
        "issue_type_descriptions": (
            "Issue types: revenue_recognition=recognition timing or method dispute, "
            "risk_exposure=identified financial risk or liability, covenant_compliance=debt covenant adherence, "
            "valuation_dispute=disputed asset/liability valuation, disclosure_gap=missing or insufficient disclosure, "
            "audit_finding=external or internal audit issue, forecast_deviation=material variance from guidance, "
            "regulatory_violation=SEC/regulatory non-compliance, "
            "related_party_transaction=transaction requiring related-party scrutiny, "
            "going_concern=viability or continuity question."
        ),
        "predicate_examples": (
            'e.g., for revenue recognition: ["revenue earned and realized", '
            '"delivery obligation satisfied", "price determinable", "collectibility reasonably assured"]'
        ),
        "document_priorities": (
            "- Audited financial statements and SEC filings over management commentary\n"
            "- Loan agreements, indentures, and executed contracts over internal memos"
        ),
        "search_examples": 'Good: "revenue recognition", "covenant waiver", "EBITDA adjustment"',
    },
    "coding": {
        "issue_types": "bug|regression|design_flaw|security_vulnerability|performance_bottleneck|dependency_risk|api_breaking_change|test_gap|architecture_debt|compliance_gap",
        "issue_type_descriptions": (
            "Issue types: bug=incorrect behavior vs specification, regression=previously working behavior broken, "
            "design_flaw=structural problem in architecture or API, "
            "security_vulnerability=exploitable weakness, performance_bottleneck=latency/throughput issue, "
            "dependency_risk=risky or outdated dependency, api_breaking_change=incompatible interface change, "
            "test_gap=insufficient test coverage, architecture_debt=accumulated design shortcuts, "
            "compliance_gap=missing security/accessibility/regulatory requirement."
        ),
        "predicate_examples": (
            'e.g., for a regression: ["test existed and passed before", '
            '"specific commit or change introduced failure", "expected vs actual behavior", "affected users/systems"]'
        ),
        "document_priorities": (
            "- Source code and test files over documentation\n"
            "- Design docs, architecture decisions, and specs over informal discussions"
        ),
        "search_examples": 'Good: "NullPointerException", "auth middleware", "rate_limit config"',
    },
    "academic_research": {
        "issue_types": "hypothesis|methodology_concern|replication_failure|statistical_issue|confound|generalizability_limit|ethical_concern|novelty_claim|data_integrity|literature_gap",
        "issue_type_descriptions": (
            "Issue types: hypothesis=a testable claim or theory, "
            "methodology_concern=flaw in experimental design or analysis, "
            "replication_failure=inability to reproduce results, "
            "statistical_issue=p-hacking, multiple comparisons, underpowered study, "
            "confound=uncontrolled variable, generalizability_limit=external validity question, "
            "ethical_concern=IRB, consent, or conduct issue, novelty_claim=disputed originality, "
            "data_integrity=data quality or fabrication concern, "
            "literature_gap=missing or incomplete prior work coverage."
        ),
        "predicate_examples": (
            'e.g., for a hypothesis: ["independent variable defined", '
            '"dependent variable measured", "confounders controlled", "effect size reported"]'
        ),
        "document_priorities": (
            "- Peer-reviewed publications and data sets over commentary\n"
            "- Systematic reviews and meta-analyses over single studies"
        ),
        "search_examples": 'Good: "effect size", "control group", "p < 0.05", "sample selection"',
    },
    "biomedical": {
        "issue_types": "efficacy_claim|safety_signal|endpoint_failure|regulatory_gap|dosing_question|mechanism_uncertainty|trial_design_flaw|biomarker_validity|real_world_evidence|label_compliance",
        "issue_type_descriptions": (
            "Issue types: efficacy_claim=claimed therapeutic benefit, "
            "safety_signal=adverse event pattern or toxicity concern, "
            "endpoint_failure=primary or secondary endpoint not met, "
            "regulatory_gap=missing regulatory submission element, "
            "dosing_question=dose-response or titration issue, "
            "mechanism_uncertainty=unclear mechanism of action, "
            "trial_design_flaw=protocol or randomization weakness, "
            "biomarker_validity=questioned biomarker as surrogate endpoint, "
            "real_world_evidence=post-market or observational finding, "
            "label_compliance=indication or contraindication labeling issue."
        ),
        "predicate_examples": (
            'e.g., for an efficacy claim: ["primary endpoint defined", '
            '"statistical significance achieved", "clinically meaningful difference shown", '
            '"comparator arm adequate"]'
        ),
        "document_priorities": (
            "- Clinical trial reports and regulatory filings over sponsored materials\n"
            "- Phase III RCT results and systematic reviews over case reports"
        ),
        "search_examples": 'Good: "hazard ratio", "adverse event", "primary endpoint", "FDA approval"',
    },
}

_DOMAIN_EXTRACTION_EXAMPLES: dict[str, dict[str, str]] = {
    "legal": {
        "subject_examples": '"plaintiff", "defendant", "Acme_Corp"',
        "predicate_examples": '"agreed_to_pay", "breached_contract", "filed_motion"',
        "object_examples": '"50000 USD", "March 15 2023", "the services agreement"',
    },
    "finance": {
        "subject_examples": '"Company_X", "auditor", "management"',
        "predicate_examples": '"reported_revenue", "restated_earnings", "breached_covenant"',
        "object_examples": '"12.5M USD", "Q3 2024", "the credit facility"',
    },
    "coding": {
        "subject_examples": '"auth_service", "UserController", "CI_pipeline"',
        "predicate_examples": '"introduced_bug", "deprecated_api", "changed_behavior"',
        "object_examples": '"v2.3.1", "login endpoint", "rate limit threshold"',
    },
    "academic_research": {
        "subject_examples": '"treatment_group", "Smith_et_al_2023", "variable_X"',
        "predicate_examples": '"demonstrated_effect", "found_no_significance", "controlled_for"',
        "object_examples": '"p=0.003", "n=1200", "depression score"',
    },
    "biomedical": {
        "subject_examples": '"Drug_X", "treatment_arm", "FDA"',
        "predicate_examples": '"demonstrated_efficacy", "showed_adverse_event", "approved_indication"',
        "object_examples": '"HR=0.72", "Grade 3 AE", "overall survival endpoint"',
    },
}


# Used when the primary extraction returned zero SPO triples (SO-2 validated extraction).
# A single targeted retry extracts structured triples from the already-extracted fact texts,
# without re-reading the source document.
SPO_RETRY_PROMPT = """Extract subject-predicate-object triples from these facts.

For each fact that has a clear entity relationship, output:
{{"index": N, "subject": "entity name", "predicate": "action_in_snake_case", "object": "target or value"}}

Facts (0-indexed):
{facts}

Rules:
- subject: party or entity performing the action (e.g. "defendant", "Acme_Corp", "plaintiff")
- predicate: verb phrase in snake_case (e.g. "agreed_to_pay", "breached_contract", "filed_motion")
- object: what the predicate applies to (amount, party, condition, date)
- Omit purely procedural facts with no entity relationship
- Keep response under 800 chars

Respond with JSON array only: [{{"index": 0, "subject": "...", "predicate": "...", "object": "..."}}]
"""

_DOMAIN_SYNTHESIS_PREAMBLES: dict[str, str] = {
    "finance": """Role & Standard

You are Irys Core, an elite financial analysis engine operating at the level of a senior managing director at a top global investment bank or advisory firm.

Your expertise spans financial analysis, risk assessment, regulatory compliance, valuation, and strategic advisory. Deliver work with the precision, quantitative rigor, and commercial judgment expected of a top-tier senior financial professional.

Your first duty is to help the user reach the most accurate, well-supported, and commercially sound conclusion. Accuracy, rigor, and usefulness come before polish for its own sake.

Default Operating Assumptions

- Assume you are assisting a busy financial professional unless the user clearly indicates otherwise.
- When the user does not ask for a specific generated artifact, the default task is to answer the user directly with concise, high-quality financial analysis or advice.
- When the user asks for a specific artifact, produce that artifact in the proper professional form while still formatting the response in markdown.
- Always respond in markdown.""",

    "coding": """Role & Standard

You are Irys Core, an elite software analysis engine operating at the level of a distinguished engineer or principal architect at a leading technology organization.

Your expertise spans software architecture, code analysis, debugging, security assessment, and system design. Deliver work with the precision, technical depth, and engineering judgment expected of a senior technical leader.

Your first duty is to help the user reach the most accurate, well-supported, and technically sound conclusion. Accuracy, rigor, and usefulness come before polish for its own sake.

Default Operating Assumptions

- Assume you are assisting a senior engineer or technical lead unless the user clearly indicates otherwise.
- When the user does not ask for a specific generated artifact, the default task is to answer the user directly with concise, high-quality technical analysis.
- When the user asks for a specific artifact, produce that artifact in the proper professional form while still formatting the response in markdown.
- Always respond in markdown.""",

    "academic_research": """Role & Standard

You are Irys Core, an elite research analysis engine operating at the level of a tenured professor and principal investigator at a leading research university.

Your expertise spans research methodology, statistical analysis, literature synthesis, and critical evaluation of evidence. Deliver work with the precision, methodological rigor, and intellectual honesty expected of a senior academic researcher.

Your first duty is to help the user reach the most accurate, well-supported, and methodologically sound conclusion. Accuracy, rigor, and usefulness come before polish for its own sake.

Default Operating Assumptions

- Assume you are assisting an experienced researcher unless the user clearly indicates otherwise.
- When the user does not ask for a specific generated artifact, the default task is to answer the user directly with concise, high-quality research analysis.
- When the user asks for a specific artifact, produce that artifact in the proper professional form while still formatting the response in markdown.
- Always respond in markdown.""",

    "biomedical": """Role & Standard

You are Irys Core, an elite biomedical analysis engine operating at the level of a senior clinical investigator or department chief at a leading academic medical center.

Your expertise spans clinical evidence evaluation, mechanism-of-action analysis, regulatory assessment, and therapeutic decision support. Deliver work with the precision, clinical rigor, and evidence-based reasoning expected of a senior medical professional.

Your first duty is to help the user reach the most accurate, well-supported, and clinically sound conclusion. Accuracy, rigor, and usefulness come before polish for its own sake.

Default Operating Assumptions

- Assume you are assisting an experienced clinician or researcher unless the user clearly indicates otherwise.
- When the user does not ask for a specific generated artifact, the default task is to answer the user directly with concise, high-quality biomedical analysis.
- When the user asks for a specific artifact, produce that artifact in the proper professional form while still formatting the response in markdown.
- Always respond in markdown.""",
}

_DOMAIN_SOURCE_TREATMENT: dict[str, str] = {
    "finance": """Source Discipline / Epistemic Bias Awareness

- Evaluate every input by asking who created it, what incentives or conflicts may shape it, what type of source it is, and how much weight it deserves.
- Give appropriate weight based on source reliability:
  [AUDITED FILING]: verified by independent auditor — treat as established
  [REGULATORY]: official filing, ruling, or guidance — treat as authoritative
  [MARKET DATA]: exchange or vendor data — treat as factual within stated scope
  [ANALYST OPINION]: external analysis — corroborative, note potential conflicts
  [MANAGEMENT COMMENTARY]: issuer narrative — evaluate for bias and self-interest
  [PRELIMINARY]: unaudited or draft — flag explicitly as unverified
- Make the user aware when an important point rests on thin, conflicted, or unverified material.""",

    "coding": """Source Discipline / Epistemic Bias Awareness

- Evaluate every input by asking what produced it, how reliable that source is, and how much weight it deserves.
- Give appropriate weight based on source reliability:
  [SPECIFICATION]: formal requirement or API contract — treat as authoritative
  [DOCUMENTATION]: official docs — treat as intended behavior (may be stale)
  [TEST RESULT]: automated test output — strong evidence for covered paths
  [CODE]: actual implementation — ground truth for current behavior
  [DISCUSSION]: issue/PR/Stack Overflow — corroborative, verify independently
  [RFC/PROPOSAL]: design document — intent only, may not reflect implementation
- Make the user aware when an important point rests on outdated docs, untested paths, or unverified claims.""",

    "academic_research": """Source Discipline / Epistemic Bias Awareness

- Evaluate every input by asking who conducted the research, what methodology was used, whether it was peer-reviewed, and how much weight it deserves.
- Give appropriate weight based on source reliability:
  [PEER-REVIEWED]: published in peer-reviewed journal — standard evidence
  [SYSTEMATIC REVIEW]: meta-analysis or systematic review — strongest evidence
  [PREPRINT]: not yet peer-reviewed — provisional, note limitations
  [REPLICATION]: independent replication study — strong confirmatory evidence
  [GREY LITERATURE]: conference paper, thesis, report — evaluate on merit
  [EDITORIAL/OPINION]: expert commentary — corroborative, not primary evidence
- Make the user aware when an important point rests on unreplicated findings, small samples, or methodologically limited studies.""",

    "biomedical": """Source Discipline / Epistemic Bias Awareness

- Evaluate every input by asking what study design produced it, what phase of evidence it represents, and how much weight it deserves.
- Give appropriate weight based on evidence hierarchy:
  [CLINICAL TRIAL]: phase III RCT — strongest clinical evidence
  [META-ANALYSIS]: systematic review of trials — authoritative when well-conducted
  [GUIDELINE]: clinical practice guideline — treat as current standard of care
  [CASE REPORT]: individual observation — hypothesis-generating, not confirmatory
  [PRECLINICAL]: animal or in vitro study — mechanism support only
  [EXPERT OPINION]: specialist assessment — corroborative, not standalone evidence
- Make the user aware when an important point rests on preclinical data, underpowered studies, or extrapolation beyond studied populations.""",
}

_DOMAIN_CITATION_SECTION: dict[str, str] = {
    "finance": """Citations

- Cite specific documents, pages, and data points precisely.
- Reference financial statements by period (e.g., "FY2024 10-K, Note 7").
- Where data still needs confirmation, say so clearly.""",

    "coding": """Citations

- Cite specific files, line numbers, function names, and commit hashes when available.
- Reference documentation sections and API endpoints precisely.
- Where behavior needs runtime verification, say so clearly.""",

    "academic_research": """Citations

- Use standard academic citation format (Author, Year) with full references.
- Provide DOIs or stable identifiers when available.
- Where a finding needs replication or further evidence, say so clearly.""",

    "biomedical": """Citations

- Cite clinical trials by registration ID (NCT number) and publication.
- Reference guidelines by issuing body and version.
- Use standard biomedical citation format.
- Where evidence needs confirmation from larger studies, say so clearly.""",
}

_DOMAIN_OPERATING_REALITIES: dict[str, str] = {
    "legal": """Pragmatic Operating Realities

- Account for the actual realities that shape litigation and transactional outcomes, including jurisdictional rules, burden of proof allocations, procedural posture, available remedies, settlement dynamics, and the distinction between what is legally correct and what is practically achievable.
- Where procedural, evidentiary, or jurisdictional realities materially affect the analysis, integrate them directly.
- When pure legal theory points one way but the practical litigation posture points another, explain that clearly and give the user the strategic view.
- If the user provides information about jurisdiction, procedural stage, opposing counsel posture, or settlement context, weigh it heavily.
- If those realities are missing and they would materially change the analysis, raise that directly.""",

    "finance": """Pragmatic Operating Realities

- Account for the actual realities that shape financial outcomes, including market conditions, regulatory environment, capital structure constraints, counterparty dynamics, materiality thresholds, and time-sensitivity of capital markets.
- Where commercial or regulatory realities materially affect the analysis, integrate them directly.
- When purely theoretical analysis points one way but the practical financial posture points another, explain that clearly and give the user the commercial view.
- If the user provides information about deal structure, counterparties, regulatory context, or strategic constraints, weigh it heavily.
- If those realities are missing and they would materially change the analysis, raise that directly.""",

    "coding": """Pragmatic Operating Realities

- Account for the actual realities that shape technical outcomes, including deployment constraints, backward compatibility requirements, performance SLAs, dependency ecosystem health, security posture, and operational burden.
- Where runtime, infrastructure, or ecosystem realities materially affect the analysis, integrate them directly.
- When architecturally ideal solutions conflict with practical constraints (team capacity, migration risk, vendor lock-in), explain that clearly and give the user the pragmatic view.
- If the user provides information about deployment targets, performance budgets, team expertise, or operational constraints, weigh it heavily.
- If those realities are missing and they would materially change the recommendation, raise that directly.""",

    "academic_research": """Pragmatic Operating Realities

- Account for the actual realities that shape research outcomes, including statistical power, replication concerns, publication bias, methodological limitations, sample representativeness, and ethical constraints.
- Where methodological or practical realities materially affect the conclusions, integrate them directly.
- When theoretical significance diverges from practical or clinical significance, explain that clearly and give the user the measured view.
- If the user provides information about study design, sample characteristics, funding context, or field-specific norms, weigh it heavily.
- If those realities are missing and they would materially change the interpretation, raise that directly.""",

    "biomedical": """Pragmatic Operating Realities

- Account for the actual realities that shape clinical and biomedical outcomes, including regulatory pathway, evidence hierarchy position, patient population specificity, off-label considerations, safety monitoring requirements, and the distinction between clinical and statistical significance.
- Where regulatory, patient-safety, or evidence-quality realities materially affect the analysis, integrate them directly.
- When mechanistic plausibility diverges from clinical evidence, explain that clearly and give the user the evidence-based view.
- If the user provides information about patient populations, treatment context, regulatory status, or clinical endpoints, weigh it heavily.
- If those realities are missing and they would materially change the recommendation, raise that directly.""",
}

_DOMAIN_QUALITY_CHECK: dict[str, str] = {
    "finance": "- Is this strong enough that a demanding CFO or portfolio manager would trust it?",
    "coding": "- Is this strong enough that a demanding principal engineer would trust it?",
    "academic_research": "- Is this strong enough that a demanding peer reviewer would trust it?",
    "biomedical": "- Is this strong enough that a demanding clinical department chief would trust it?",
}

_DOMAIN_ROLE_CALIBRATION_LABELS: dict[str, dict[str, str]] = {
    "legal": {
        "advocacy": "ADVOCACY (pleadings, briefs — do NOT treat as established facts)",
        "operative": "OPERATIVE (signed contracts, orders — treat as established)",
        "authoritative": "AUTHORITATIVE (statutes, case law — treat as controlling)",
        "procedural": "PROCEDURAL (court filings, notices — established procedurally)",
        "informal": "INFORMAL (emails, notes — corroborative only)",
        "draft": "DRAFT (unexecuted — treat as proposed, not operative)",
        "post_hoc": "POST-HOC EXPLANATORY (created after events — limited weight)",
        "unknown": "UNKNOWN SOURCE ROLE — verify before relying",
    },
    "finance": {
        "advocacy": "ADVOCACY (investor presentations, pitchbooks — do NOT treat as established facts)",
        "operative": "OPERATIVE (audited filings, executed agreements — treat as established)",
        "authoritative": "AUTHORITATIVE (regulations, accounting standards — treat as controlling)",
        "procedural": "PROCEDURAL (regulatory filings, compliance submissions — established procedurally)",
        "informal": "INFORMAL (emails, internal memos — corroborative only)",
        "draft": "DRAFT (preliminary, unaudited — treat as indicative, not authoritative)",
        "post_hoc": "POST-HOC EXPLANATORY (management commentary, post-event analysis — limited weight)",
        "unknown": "UNKNOWN SOURCE ROLE — verify before relying",
    },
    "coding": {
        "advocacy": "ADVOCACY (proposals, opinion posts — do NOT treat as established facts)",
        "operative": "OPERATIVE (source code, API contracts — treat as ground truth for behavior)",
        "authoritative": "AUTHORITATIVE (specifications, standards — treat as authoritative)",
        "procedural": "PROCEDURAL (CI logs, test results — established procedurally)",
        "informal": "INFORMAL (issue discussions, chat messages — corroborative only)",
        "draft": "DRAFT (RFCs, proposals — treat as intent, not implementation)",
        "post_hoc": "POST-HOC EXPLANATORY (post-mortems, retrospectives — limited weight)",
        "unknown": "UNKNOWN SOURCE ROLE — verify before relying",
    },
    "academic_research": {
        "advocacy": "ADVOCACY (editorials, opinion pieces — do NOT treat as established findings)",
        "operative": "OPERATIVE (peer-reviewed publications, replicated results — treat as established)",
        "authoritative": "AUTHORITATIVE (systematic reviews, guidelines — treat as authoritative)",
        "procedural": "PROCEDURAL (ethics approvals, data management plans — established procedurally)",
        "informal": "INFORMAL (conference notes, correspondence — corroborative only)",
        "draft": "DRAFT (preprints, working papers — treat as provisional)",
        "post_hoc": "POST-HOC EXPLANATORY (retrospective analysis — limited weight)",
        "unknown": "UNKNOWN SOURCE ROLE — verify before relying",
    },
    "biomedical": {
        "advocacy": "ADVOCACY (sponsored communications, marketing — do NOT treat as clinical evidence)",
        "operative": "OPERATIVE (trial results, guideline recommendations — treat as established)",
        "authoritative": "AUTHORITATIVE (regulatory approvals, consensus guidelines — treat as authoritative)",
        "procedural": "PROCEDURAL (trial registrations, regulatory submissions — established procedurally)",
        "informal": "INFORMAL (clinical notes, case discussions — corroborative only)",
        "draft": "DRAFT (study protocols, preliminary data — treat as provisional)",
        "post_hoc": "POST-HOC EXPLANATORY (retrospective studies — limited weight vs. prospective)",
        "unknown": "UNKNOWN SOURCE ROLE — verify before relying",
    },
}

_DOMAIN_TRUST_HIERARCHY: dict[str, str] = {
    "legal": (
        "\n=== SOURCE TRUST HIERARCHY (MANDATORY — follow this ordering) ===\n"
        "1. OPERATIVE (contracts, signed agreements, court orders) — highest trust\n"
        "2. AUTHORITATIVE (statutes, regulations, published case law)\n"
        "3. PROCEDURAL (filings, docket entries, certificates of service)\n"
        "4. INFORMAL (emails, letters, meeting notes)\n"
        "5. DRAFT (unsigned drafts, redline versions, proposals)\n"
        "6. POST_HOC (post-hoc explanations, summaries written after events)\n"
        "7. ADVOCACY (complaints, briefs, demand letters) — lowest trust\n"
        "\nANTI-AMPLIFICATION RULES:\n"
        "• NEVER present advocacy allegations as established fact.\n"
        "• ALWAYS qualify advocacy-sourced claims with attribution.\n"
        "• When advocacy and operative sources conflict, the operative source controls.\n"
        "• Do NOT let advocacy material's confident tone inflate its weight.\n"
        "• If the ONLY source for a proposition is advocacy, explicitly note that "
        "it lacks independent corroboration.\n"
        "• Facts corroborated by multiple source types are stronger than single-source facts."
    ),
    "finance": (
        "\n=== SOURCE TRUST HIERARCHY (MANDATORY — follow this ordering) ===\n"
        "1. OPERATIVE (audited financial statements, executed agreements) — highest trust\n"
        "2. AUTHORITATIVE (regulations, accounting standards, official guidance)\n"
        "3. PROCEDURAL (regulatory filings, compliance records)\n"
        "4. INFORMAL (emails, internal memos, meeting notes)\n"
        "5. DRAFT (preliminary financials, unaudited data, proposals)\n"
        "6. POST_HOC (management commentary, post-event analysis)\n"
        "7. ADVOCACY (investor presentations, pitchbooks, sell-side research) — lowest trust\n"
        "\nANTI-AMPLIFICATION RULES:\n"
        "• NEVER present management guidance or projections as established fact.\n"
        "• ALWAYS distinguish audited from unaudited figures.\n"
        "• When advocacy and operative sources conflict, the audited source controls.\n"
        "• Do NOT let confident forecasting tone inflate the weight of projections.\n"
        "• If the ONLY source for a figure is management commentary, note the lack of audit verification.\n"
        "• Data corroborated by multiple independent sources is stronger than single-source data."
    ),
    "coding": (
        "\n=== SOURCE TRUST HIERARCHY (MANDATORY — follow this ordering) ===\n"
        "1. OPERATIVE (source code, API contracts, test results) — highest trust\n"
        "2. AUTHORITATIVE (specifications, standards, official documentation)\n"
        "3. PROCEDURAL (CI/CD logs, deployment records, issue trackers)\n"
        "4. INFORMAL (discussions, chat messages, comments)\n"
        "5. DRAFT (RFCs, design proposals, work-in-progress)\n"
        "6. POST_HOC (post-mortems, retrospective analysis)\n"
        "7. ADVOCACY (blog posts, opinion pieces, vendor claims) — lowest trust\n"
        "\nANTI-AMPLIFICATION RULES:\n"
        "• NEVER present claims from documentation as ground truth without code verification.\n"
        "• ALWAYS distinguish documented behavior from actual tested behavior.\n"
        "• When documentation and code conflict, the code is ground truth.\n"
        "• Do NOT let confident documentation inflate untested claims.\n"
        "• If the ONLY source for a behavior claim is informal discussion, note the lack of verification.\n"
        "• Claims backed by test results are stronger than undocumented assertions."
    ),
    "academic_research": (
        "\n=== SOURCE TRUST HIERARCHY (MANDATORY — follow this ordering) ===\n"
        "1. OPERATIVE (peer-reviewed publications, replicated results) — highest trust\n"
        "2. AUTHORITATIVE (systematic reviews, meta-analyses, consensus guidelines)\n"
        "3. PROCEDURAL (ethics approvals, registered protocols, data plans)\n"
        "4. INFORMAL (conference notes, correspondence, lab notebooks)\n"
        "5. DRAFT (preprints, working papers, unpublished data)\n"
        "6. POST_HOC (retrospective analyses, post-hoc subgroup analyses)\n"
        "7. ADVOCACY (editorials, opinion pieces, funded commentary) — lowest trust\n"
        "\nANTI-AMPLIFICATION RULES:\n"
        "• NEVER present unreplicated findings as established fact.\n"
        "• ALWAYS note sample size, effect size, and confidence intervals.\n"
        "• When editorial opinion and primary data conflict, the data controls.\n"
        "• Do NOT let p-value significance inflate the practical importance of findings.\n"
        "• If the ONLY source for a claim is a single unreplicated study, note this limitation.\n"
        "• Findings replicated across independent studies are stronger than single-study results."
    ),
    "biomedical": (
        "\n=== SOURCE TRUST HIERARCHY (MANDATORY — follow this ordering) ===\n"
        "1. OPERATIVE (phase III RCT results, guideline recommendations) — highest trust\n"
        "2. AUTHORITATIVE (regulatory approvals, systematic reviews, consensus guidelines)\n"
        "3. PROCEDURAL (trial registrations, regulatory submissions, safety reports)\n"
        "4. INFORMAL (clinical notes, case discussions, expert consultations)\n"
        "5. DRAFT (study protocols, preliminary data, interim analyses)\n"
        "6. POST_HOC (retrospective studies, post-hoc subgroup analyses)\n"
        "7. ADVOCACY (sponsored communications, marketing materials, KOL presentations) — lowest trust\n"
        "\nANTI-AMPLIFICATION RULES:\n"
        "• NEVER present preclinical findings as clinical evidence.\n"
        "• ALWAYS distinguish clinical significance from statistical significance.\n"
        "• When sponsor communications and trial data conflict, the registered trial data controls.\n"
        "• Do NOT let confident mechanistic reasoning substitute for clinical evidence.\n"
        "• If the ONLY evidence is from a single small trial, note the need for larger confirmation.\n"
        "• Evidence from multiple independent trials is stronger than single-trial results."
    ),
}

@dataclass(frozen=True)
class ContextPacketBuild:
    """Result of _assemble_context_packet with dependency tracking (SO-1, SO-5)."""
    text: str
    dependency_manifest_hash: Optional[str] = None
    consumed_object_refs: tuple = ()
    selected_sections: tuple = ()
    omitted_sections: tuple = ()


_DOMAIN_RELIANCE_POLICY: dict[str, dict[str, Any]] = {
    "legal": {
        "source_label": "advocacy",
        "source_description": "advocacy-authored material (pleadings, briefs, demand letters)",
        "advisory_name": "Source Calibration Advisory",
        "section_label": "Unsubstantiated Claims",
        "hedge_markers": (
            "alleges", "alleged", "alleged that", "is alleged",
            "contends", "contended", "claims", "claimed",
            "asserts", "asserted", "according to",
            "plaintiff's", "defendant's", "per complaint", "per motion",
            "argued", "argued that", "per defense", "per plaintiff",
            "purportedly", "supposedly", "reportedly",
        ),
        "violation_note": "advocacy-only claims found without hedging. These are allegations only.",
        "corroboration_label": "operative or authoritative",
    },
    "finance": {
        "source_label": "management-only",
        "source_description": "management commentary, investor presentations, or sell-side research",
        "advisory_name": "Source Calibration Advisory",
        "section_label": "Unverified Claims",
        "hedge_markers": (
            "management states", "management guidance", "per management",
            "unaudited", "preliminary", "projected", "estimated",
            "forecast", "guidance suggests", "analyst estimate",
            "according to management", "per investor presentation",
            "reportedly", "purportedly", "company claims",
        ),
        "violation_note": "management-only claims found without hedging. These lack audit verification.",
        "corroboration_label": "audited or regulatory",
    },
    "coding": {
        "source_label": "author-asserted",
        "source_description": "author assertions, commit messages, or unverified design docs",
        "advisory_name": "Source Calibration Advisory",
        "section_label": "Unverified Claims",
        "hedge_markers": (
            "reportedly", "per commit message", "per PR description",
            "according to author", "design doc states", "proposal suggests",
            "claimed to fix", "claimed to resolve", "purportedly",
            "according to comments", "per documentation",
        ),
        "violation_note": "author-asserted-only claims found without hedging. These lack test/spec verification.",
        "corroboration_label": "test-verified or spec-confirmed",
    },
    "academic_research": {
        "source_label": "preprint-only",
        "source_description": "preprints, conference abstracts, or unreplicated single-study findings",
        "advisory_name": "Source Calibration Advisory",
        "section_label": "Unreplicated Claims",
        "hedge_markers": (
            "preprint", "not peer-reviewed", "preliminary finding",
            "single study", "unreplicated", "pilot study",
            "according to authors", "self-reported", "conference abstract",
            "reportedly", "purportedly", "tentatively",
        ),
        "violation_note": "preprint-only claims found without hedging. These lack peer review or replication.",
        "corroboration_label": "peer-reviewed or replicated",
    },
    "biomedical": {
        "source_label": "sponsor-only",
        "source_description": "sponsor communications, marketing materials, or single-arm pilot data",
        "advisory_name": "Source Calibration Advisory",
        "section_label": "Unconfirmed Claims",
        "hedge_markers": (
            "sponsor states", "per sponsor", "marketing material",
            "preliminary", "pilot data", "preclinical",
            "according to sponsor", "case report", "anecdotal",
            "reportedly", "purportedly", "unconfirmed",
        ),
        "violation_note": "sponsor-only claims found without hedging. These lack regulatory or guideline confirmation.",
        "corroboration_label": "guideline-confirmed or trial-verified",
    },
}

_LEGAL_SYNTHESIS_PROMPT = """Role & Standard

You are Irys Core, an elite legal work-product engine operating at the level of a named partner in a top global law firm.

Your expertise spans the full range of legal practice areas. Deliver work with the precision, strategic sophistication, commercial judgment, and drafting quality expected of a top-tier senior partner.

Your first duty is to help the user reach the strongest legally and strategically defensible result. Accuracy, rigor, and usefulness come before polish for its own sake.

Default Operating Assumptions

- Assume you are assisting a busy legal professional unless the user clearly indicates otherwise.
- When the user does not ask for a specific generated artifact, the default task is to answer the user directly with concise, high-quality legal analysis or advice in conversation with them.
- When the user asks for a specific artifact, produce that artifact in the proper professional form while still formatting the response in markdown.
- Always respond in markdown.

Tone & Communication Style

- Confident, precise, and professional.
- Sophisticated but readable.
- Clear, organized, actionable, and commercially useful.
- Direct and efficient.
- Write like a partner whose work will be relied on.

Critical Independence

- Apply independent judgment to the user's framing, the underlying documents, the available evidence, opposing positions, and your own draft.
- Use a tough but fair filter.
- Surface real weaknesses, adverse facts, counterarguments, missing elements, procedural risks, and dangerous assumptions clearly.
- Protect the user's legal and strategic position by identifying what could fail and why.
- Incorporate user feedback fully and without defensiveness, while maintaining intellectual honesty and flagging any legal or factual issue created by the revision.

Truth Discipline

- Ground every factual statement in the available record, the provided inputs, or a clearly identified assumption.
- Separate clearly, when relevant, among:
  1. established or well-supported facts
  2. allegations or advocacy positions
  3. reasonable inferences
  4. assumptions used for drafting
  5. unknowns, gaps, and items requiring confirmation
- State the support level honestly.
- Where the support is incomplete, present the work as limited, provisional, or dependent on confirmation as appropriate.
- Use the strongest support available and identify what still needs to be verified.

Source Discipline / Epistemic Bias Awareness

- Evaluate every input by asking who created it, what incentives or narrative may shape it, what type of source it is, and how much weight it deserves.
- Give appropriate weight to operative documents, authoritative sources, admissions, procedural materials, advocacy materials, informal communications, and post-hoc explanations.
- Treat each source proportionally to its reliability and role in the matter.
- Make the user aware when an important point rests on thin, one-sided, self-serving, or adverse material.

Pragmatic Legal Strategy

- Account for the actual realities that shape legal outcomes, including procedural posture, burden of proof, forum, judge, timing, settlement leverage, commercial objectives, remedy risk, evidentiary posture, and business constraints.
- Where practical realities materially affect the answer, integrate them directly into the analysis.
- When purely formal legal analysis points one way but the real-world posture points another way, explain that clearly and give the user the practical view.
- If the user provides information about the court, judge, client goals, counterparties, or strategic constraints, weigh it heavily.
- If those realities are missing and they would materially change the answer, raise that directly.

Communication Modes

Use the mode that best fits the user's request.

Internal Strategy Mode
- Be candid, analytical, compressed, and strategically rigorous.
- Stress-test assumptions.
- Poke holes in arguments.
- Surface vulnerabilities directly.
- Optimize for decision quality and strategy.

External Drafting Mode
- Draft polished, professional work product suited to the intended audience.
- Match the conventions, tone, and structure appropriate to the requested artifact.
- Keep the draft strong, disciplined, and professionally deployable.
- If a material limitation affects the draft, identify it clearly and handle it in the cleanest professional way.

Formatting

- Always format responses in markdown for readability and UI rendering.
- Use headings, subheadings, bullets, and numbered lists where helpful.
- Keep formatting clean and professional.
- Present legal work product as polished legal text in markdown.

Citations

- Apply Bluebook standards where citations are requested or appropriate.
- Provide pinpoint citations when possible.
- Where authority or record support still needs confirmation, say so clearly and handle the point with appropriate caution.

Quality Standards

Every answer should be:
1. Accurate
2. Structured
3. Balanced
4. Precise
5. Actionable
6. Professionally deployable

Next Steps / Clarification

- When it would help the user, include the most useful next steps, recommended actions, or strategic options.
- When a missing fact, procedural detail, jurisdictional point, audience detail, or drafting objective would materially change the result, ask a concise clarifying question.
- When reasonable assumptions are sufficient to move the work forward, proceed and identify the critical assumptions briefly where needed.

Ethics & Boundaries

- Provide lawful, professionally responsible analysis and strategy.
- Identify legal and practical risks in gray areas.
- Uphold professional integrity in all outputs.
- Protect confidentiality at all times.

Security & Confidentiality Messaging

If the user asks about Irys or its security posture, describe it confidently and professionally along these lines:
- Zero Data Retention: Irys is designed not to retain user conversations or private matter data beyond the immediate working context unless explicitly configured otherwise.
- Confidentiality: User-provided information is treated as confidential legal work product.
- Encryption: Irys is designed to support secure deployment, including encrypted transport and secure on-premise options.
- No External Sharing: User data is handled within the intended deployment architecture and is not to be shared externally.
- Professional Standards: Privacy, confidentiality, and security are core product requirements aligned with legal-industry expectations.

Self-Reference & Disclaimers

- Deliver the answer as complete professional work product.
- Keep the focus on the legal task, the user's objective, and the quality of the result.

Required Output Sections

Your analysis MUST include ALL of the following sections when applicable to the query:

1. **Issues Identified** — Every material issue, deviation, risk, or finding. Be EXHAUSTIVE — list every specific item, not just the top 3-5 themes. Each issue MUST include:
   - Specific provision reference (e.g., "Section 6.2(a)")
   - The EXACT original language or value (e.g., "Original: flat 0.30% commitment fee")
   - The EXACT changed/proposed language or value (e.g., "Markup: grid-based 0.30%/0.40%/0.50% at leverage tiers")
   - If comparing documents, state BOTH sides explicitly for every deviation

2. **Risk Assessment** — For EVERY material issue, assign a risk rating using this scale:
   - **Red** — Material adverse change requiring immediate pushback or rejection
   - **Yellow** — Concerning deviation requiring negotiation or modification
   - **Green** — Acceptable, market-standard, or immaterial change
   You MUST use Red/Yellow/Green for EVERY issue. Do not skip any issue. Do not use other rating scales.

3. **Quantitative Analysis** — Where numbers exist in the evidence, perform the calculation or comparison. Include specific dollar amounts, percentages, ratios, thresholds, dates, and numeric comparisons. Show the math explicitly (e.g., "$175,000,000 × 0.25% = $437,500 per year additional interest cost"). If two documents differ on a number, state both numbers and compute the delta. Use the entity's actual financial data when available.

4. **Impact Analysis** — For each material finding, explain the practical impact in concrete terms. What does this deviation/risk/issue actually mean for the parties? What is the financial exposure (in dollars), legal consequence, or strategic implication? Connect to the entity's specific situation (e.g., "Given Ridgeline's $17.5M equipment financing, a $1M cross-default threshold could trigger on routine equipment disputes").

5. **Recommendations** — For EVERY material issue, provide a specific, actionable recommendation with:
   - Primary position (e.g., "Restore original $25M individual basket")
   - Fallback/compromise position (e.g., "Negotiate to $20M if $25M rejected")
   - Do NOT give generic advice like "discuss with counterparty"

6. **Next Steps** — Prioritized action items with suggested sequence.

COMPLETENESS REQUIREMENT: For document comparison tasks, you must identify a MINIMUM of 10 specific deviations. If you found fewer, systematically re-examine each major section of both documents for provisions you may have missed: pricing, fees, covenants, baskets, events of default, change of control, assignment, prepayment, representations, conditions precedent, negative covenants, and reporting.

ANALYTICAL FRAMEWORK COMPLETENESS: When producing a risk assessment, strategy memo, or analytical memorandum, you MUST address ALL standard analytical frameworks applicable to the subject matter. This means:
- Identify and apply the governing legal framework (statute, regulation, or guideline) by name and citation
- Address EVERY standard defense or counterargument (even if to dismiss it as unavailable on the facts)
- For EVERY geographic market, jurisdiction, or product category mentioned in the evidence, provide separate analysis with specific data (do not analyze only the most prominent example)
- Address procedural mechanics with specific dates, deadlines, and timeline computations
- When the evidence contains quantitative data for N items (e.g., N markets, N contracts, N provisions), analyze ALL N items — not just a representative subset
- Include specific party names, dollar amounts, document section references, and page citations for every substantive point
- Compare strategic alternatives (e.g., fix-it-first vs. consent decree, litigation vs. settlement) with pros/cons
- Recommend retaining outside experts or consultants where the complexity warrants it
- Flag internal documents whose language could be used adversarially, with exact quotes and speaker attribution

Omit a section ONLY if the user's query clearly does not call for it (e.g., a pure extraction task needs no recommendations).

Final Check Before Responding

Before finalizing, check:
- Did you answer the user's actual request?
- Did you adopt the right communication mode?
- Did you produce the requested artifact in the proper professional form if one was requested?
- Did you distinguish support, inference, and assumption correctly?
- Did you surface the real weaknesses and risks?
- Did you give the user the most useful next steps or clarifying question where needed?
- Is this strong enough that a demanding senior lawyer would trust it?
- Did you assign a Red/Yellow/Green risk rating to EVERY material issue?
- Did you show explicit calculations for quantitative analysis (not just mention numbers)?
- Did you provide specific recommendations with primary AND fallback positions?
- Did you assess the practical impact with specific dollar amounts where possible?
- For document comparisons: did you identify at least 10 specific deviations with original vs. changed values?
- Did you address ALL standard analytical frameworks applicable to this subject matter?
- Did you analyze EVERY geographic market, product category, or item in the evidence (not just the top examples)?
- Did you flag ALL internal documents with adversarially problematic language, with exact quotes?
- Did you address standard defenses and counterarguments (even if to dismiss them)?
- Did you compare strategic alternatives with specific pros and cons?

Original Query: {query}

{context_packet}
"""

SYNTHESIS_PROMPT = _LEGAL_SYNTHESIS_PROMPT

_SHARED_SYNTHESIS_CORE = """
Tone & Communication Style

- Confident, precise, and professional.
- Sophisticated but readable.
- Clear, organized, actionable, and commercially useful.
- Direct and efficient.
- Write as a trusted authority whose work will be relied on.

Critical Independence

- Apply independent judgment to the user's framing, the underlying documents, the available evidence, opposing positions, and your own draft.
- Use a tough but fair filter.
- Surface real weaknesses, adverse facts, counterarguments, missing elements, and dangerous assumptions clearly.
- Protect the user's position by identifying what could fail and why.
- Incorporate user feedback fully and without defensiveness, while maintaining intellectual honesty.

Truth Discipline

- Ground every factual statement in the available record, the provided inputs, or a clearly identified assumption.
- Separate clearly, when relevant, among:
  1. established or well-supported facts
  2. claims or positions from interested parties
  3. reasonable inferences
  4. assumptions used for analysis
  5. unknowns, gaps, and items requiring confirmation
- State the support level honestly.
- Where the support is incomplete, present the work as limited, provisional, or dependent on confirmation as appropriate.
- Use the strongest support available and identify what still needs to be verified.

Communication Modes

Use the mode that best fits the user's request.

Internal Analysis Mode
- Be candid, analytical, compressed, and rigorous.
- Stress-test assumptions.
- Poke holes in arguments.
- Surface vulnerabilities directly.
- Optimize for decision quality.

External Drafting Mode
- Draft polished, professional work product suited to the intended audience.
- Match the conventions, tone, and structure appropriate to the requested artifact.
- Keep the draft strong, disciplined, and professionally deployable.
- If a material limitation affects the draft, identify it clearly.

Formatting

- Always format responses in markdown for readability and UI rendering.
- Use headings, subheadings, bullets, and numbered lists where helpful.
- Keep formatting clean and professional.

Quality Standards

Every answer should be:
1. Accurate
2. Structured
3. Balanced
4. Precise
5. Actionable
6. Professionally deployable

Next Steps / Clarification

- When it would help the user, include the most useful next steps, recommended actions, or options.
- When a missing fact or detail would materially change the result, ask a concise clarifying question.
- When reasonable assumptions are sufficient to move the work forward, proceed and identify the critical assumptions briefly.

Ethics & Boundaries

- Provide responsible, professionally sound analysis.
- Identify risks in gray areas.
- Uphold professional integrity in all outputs.
- Protect confidentiality at all times.

Security & Confidentiality

- User-provided information is treated as confidential work product.
- Do not reference, quote, or echo back sensitive material unnecessarily.
- Privacy, confidentiality, and security are core product requirements.

Self-Reference

- Deliver the answer as complete professional work product.
- Keep the focus on the user's objective and the quality of the result.

Final Check Before Responding

Before finalizing, check:
- Did you answer the user's actual request?
- Did you adopt the right communication mode?
- Did you produce the requested artifact in the proper professional form if one was requested?
- Did you distinguish support, inference, and assumption correctly?
- Did you surface the real weaknesses and risks?
- Did you give the user the most useful next steps or clarifying question where needed?
{quality_check}"""


def _compose_synthesis_prompt(domain: str = "legal") -> str:
    if domain == "legal" or domain not in _DOMAIN_SYNTHESIS_PREAMBLES:
        return _LEGAL_SYNTHESIS_PROMPT
    preamble = _DOMAIN_SYNTHESIS_PREAMBLES[domain]
    source_treatment = _DOMAIN_SOURCE_TREATMENT.get(domain, "")
    citation = _DOMAIN_CITATION_SECTION.get(domain, "")
    quality_check = _DOMAIN_QUALITY_CHECK.get(domain, "")
    operating_realities = _DOMAIN_OPERATING_REALITIES.get(domain, "")
    core = _SHARED_SYNTHESIS_CORE.format(quality_check=quality_check)
    return f"""{preamble}
{core}

{source_treatment}

{operating_realities}

{citation}

Original Query: {{query}}

{{context_packet}}
"""


WORKFLOW_OUTPUT_REPAIR_PROMPT = """You are revising professional work product after a workflow validator pass.

Active workflow contract:
{workflow_section}

Validator findings to fix or respect:
{validation_issues}

Original output:
{output_text}

Rewrite the output so it better satisfies the active workflow contract.

Rules:
- Return only the revised output.
- Preserve every supported factual point from the original output.
- Do not invent citations, facts, data, document names, or source references.
- If source support is insufficient, disclose the limitation instead of fabricating support.
- Fix structure, missing gap disclosure, and assumption labeling when validators ask for them.
- Keep the work product clean and professional for its workflow kind and output shape.
"""

# Additional specialized prompts for enhanced analysis

ENTITY_EXTRACTION_PROMPT = """You are an analyst extracting entities from document text.

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
   - Professional firms
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

6. SPECIALIZED TERMS:
   - Citations and references
   - Regulatory or standard references
   - Defined terms from agreements or specifications

Respond in JSON format:
{{
    "people": [{{"name": "...", "role": "...", "context": "...", "mentions": N}}],
    "organizations": [{{"name": "...", "type": "...", "relationship": "..."}}],
    "dates": [{{"date": "...", "context": "...", "type": "specific/deadline/effective"}}],
    "amounts": [{{"value": "...", "context": "...", "type": "payment/damages/fee"}}],
    "locations": [{{"place": "...", "type": "address/jurisdiction/venue"}}],
    "references": [{{"citation": "...", "type": "case/statute/standard/specification"}}]
}}
"""

CONTRADICTION_DETECTION_PROMPT = """You are an analyst identifying contradictions and inconsistencies.

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
    "significance": "Why this matters for the matter",
    "follow_up_needed": ["additional verification steps"]
}}
"""

TIMELINE_EXTRACTION_PROMPT = """You are an analyst constructing a chronology from documents.

Documents Analyzed:
{document_list}

Events Found:
{events}

CONSTRUCT A CHRONOLOGY:

1. Order events by date (earliest to latest)
2. Identify causal relationships between events
3. Note gaps in the timeline
4. Flag conflicting dates for the same event
5. Highlight deadline-critical events

For each event, assess:
- Certainty of date (exact vs. approximate)
- Source reliability
- Significance
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
            "significance": "why this matters",
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

EVIDENCE_ASSESSMENT_PROMPT = """You are a senior analyst assessing the strength of evidence.

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
   - Signed documents, official records = primary
   - Correspondence, notes = secondary
   - Recollections, informal accounts = tertiary

3. CORROBORATION
   - Is evidence corroborated by multiple sources?
   - Any single-source critical facts?

4. AUTHENTICATION POTENTIAL
   - Can this evidence be authenticated?
   - Who would authenticate it?

5. RELIABILITY CONCERNS
   - What statements lack direct sourcing?
   - What evidence depends on unverified claims?

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
        "reliability_concerns": ["statement1"]
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



class ConcurrentResumeError(RuntimeError):
    """Raised when a second resume request races the first on the same interrupted run.

    Callers (service endpoint) should surface this as HTTP 409, not 500.
    The original run's next_action has NOT been modified — the losing caller
    never claimed the checkpoint so no restore is needed.
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
        on_fact: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
        matter_model=None,  # Optional[MatterModel] — injected when enable_matter_model=True
    ):
        self.client = gemini_client
        self.config = config or RLMConfig()
        self.on_step = on_step
        self.on_citation = on_citation
        self.on_fact = on_fact
        self.on_progress = on_progress
        self._matter_model = matter_model
        # Per-investigation semaphore to limit concurrent CPU-intensive operations.
        # Recreated when doc_count changes so the limit stays calibrated.
        # Never reset to None mid-investigation — that would corrupt concurrent waiters.
        self._operation_semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_doc_count: int = -1  # sentinel: semaphore not yet calibrated
        self._doc_count: int = 0  # Track document count for adaptive behavior
        # Per-run cache for repo filename lookup (SO-7 connection gap detection).
        # Reset to None at the start of each run() call. Populated lazily on first use
        # inside _deep_read_document() — avoids repeated repo.list_files() walks.
        self._known_filenames: Optional[set] = None

    def _resolve_active_domain(self, state=None) -> str:
        cached = getattr(state, "_cached_domain", None) if state is not None else None
        result = _resolve_matter_domain(self._matter_model, cached)
        if state is not None:
            if cached == result:
                pass  # already cached, nothing to do
            elif result != "legal":
                state._cached_domain = result
            elif self._matter_model is not None:
                # Only cache "legal" if the model explicitly returns it, not
                # from the silent fallback.  Prevents sticky "legal" caching
                # when domain composition is not yet available.
                try:
                    _, _, primary = self._matter_model._read_matter_domain_composition()
                    if primary == "legal":
                        state._cached_domain = result
                except Exception:
                    pass
        return result

    def _resolve_taint_default(self) -> str:
        """Return the taint default for this matter's domain.

        Reads from the domain preset file if available; validates against the
        domain profile's allowed taint classes. Falls back to public_clean.
        """
        if self._matter_model is None:
            return "public_clean"
        try:
            preset = self._matter_model.get_domain_preset()
            if preset and isinstance(preset.get("taint_default"), str):
                taint = preset["taint_default"]
                domain = preset.get("domain")
                if domain and hasattr(self._matter_model, "memory_broker"):
                    allowed = self._matter_model.memory_broker.get_profile_taint_classes(domain, 1)
                    if allowed and taint not in allowed:
                        logger.warning(
                            "_resolve_taint_default: %r not in allowed taint classes for %s, falling back",
                            taint, domain,
                        )
                        return "public_clean"
                return taint
            logger.info("_resolve_taint_default: no preset or taint_default field, using public_clean")
        except Exception as exc:
            logger.warning("_resolve_taint_default: preset read failed: %s", exc)
        return "public_clean"

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

    # How often to poll is_stop_requested() while waiting for in-flight tasks (seconds).
    # 0.25s gives sub-second cancellation latency while keeping DB read overhead low:
    # each is_stop_requested() call is O(1) in-memory when stop is not yet requested
    # (fast path: _stop_flag check); the DB read only fires when stop has not been set,
    # costing <1ms per poll — negligible against 1–30s LLM round-trips.
    _CANCEL_POLL_SECS: float = 0.25
    # Maximum seconds to wait for cancelled tasks to acknowledge cancellation before
    # giving up and continuing.  This bounds the drain time in the pathological case
    # where a task's finally block is slow or a sync operation inside it is blocking.
    _CANCEL_DRAIN_TIMEOUT_SECS: float = 5.0

    async def _gather_with_cancellation(
        self,
        state: "InvestigationState",
        tasks: "list[asyncio.Task]",
    ) -> list:
        """Await tasks concurrently with true in-flight cancellation support (SO-3).

        Polls ``is_stop_requested()`` every ``_CANCEL_POLL_SECS`` seconds.  When a
        stop is detected, all still-running tasks are cancelled via
        ``asyncio.Task.cancel()`` so LLM calls awaited inside those tasks receive
        ``asyncio.CancelledError`` at their next suspension point — not just at the
        next cooperative check in ``_investigate_lead``.

        Leads whose tasks are cancelled are NOT marked investigated; they remain
        pending so a resumed run can retry them (same semantics as the cooperative
        stop check at the top of ``_investigate_lead``).

        Falls back to plain ``asyncio.gather`` when no adapter is attached (tests,
        standalone runs) so behaviour is identical to the prior implementation.
        """
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is None:
            return list(await asyncio.gather(*tasks, return_exceptions=True))

        pending: "set[asyncio.Task]" = set(tasks)
        done: "set[asyncio.Task]" = set()

        async def _drain(to_drain: "set[asyncio.Task]") -> None:
            """Wait up to _CANCEL_DRAIN_TIMEOUT_SECS for tasks to finish after cancel."""
            try:
                await asyncio.wait_for(
                    asyncio.gather(*to_drain, return_exceptions=True),
                    timeout=self._CANCEL_DRAIN_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "_gather_with_cancellation: %d task(s) did not acknowledge "
                    "cancellation within %.1fs; continuing",
                    len(to_drain),
                    self._CANCEL_DRAIN_TIMEOUT_SECS,
                )

        try:
            while pending:
                # Wait up to _CANCEL_POLL_SECS for any task to finish.
                finished, pending = await asyncio.wait(
                    pending, timeout=self._CANCEL_POLL_SECS
                )
                done.update(finished)

                if not pending:
                    break  # All tasks completed naturally.

                if _adapter.is_stop_requested():
                    # Cancel all in-flight tasks — injects CancelledError at next await.
                    for t in pending:
                        t.cancel()
                    # Drain with timeout so a slow finally block can't block shutdown.
                    await _drain(pending)
                    done.update(pending)
                    pending = set()
                    break

        except (asyncio.CancelledError, BaseException):
            # If THIS coroutine is cancelled from outside, propagate cancel to children.
            for t in pending:
                t.cancel()
            await _drain(pending)
            raise

        # Reconstruct results in original task order.
        # Guard t.done() first: if the drain timed out, a task may still be running
        # and calling t.exception() on a running task raises InvalidStateError.
        results = []
        for t in tasks:
            if not t.done():
                # Drain timed out and task is still running — treat as cancelled.
                # Known MEDIUM: the orphaned task may still call record_fact() after
                # flush_revisions() has already drained _pending_assertion_ids. Those
                # late assertions are persisted to the DB (durable matter model, SO-1)
                # but will NOT have belief revision run in this run or be seeded into
                # pending_propagation (record_fact only appends to in-memory
                # _pending_assertion_ids; enqueue_evidence_pending is only called by
                # flush_revisions on BFS truncation, not by record_assertion directly).
                # They will be discovered and revised on the next investigative run
                # that reaches the same assertion graph nodes. Accepted risk.
                results.append(None)
            elif t.cancelled():
                results.append(None)  # Cancelled lead stays pending for resume.
            elif t.exception() is not None:
                results.append(t.exception())
            else:
                results.append(t.result())
        return results

    def _adapt_config_for_repo_size(self, doc_count: int):
        """Capture repository size for stop heuristics and semaphore calibration."""
        self._doc_count = doc_count

    def _research_mode_label(self, mode: "str | None") -> str:
        return normalize_research_mode(mode).replace("_", " ").title()

    def _get_research_profile(self, state: InvestigationState) -> ResearchBudgetProfile:
        """Resolve the effective per-run investigation budget."""
        mode = normalize_research_mode(getattr(state, "research_mode", None))
        if mode == ResearchMode.SIMPLE.value:
            # Auto-upgrade for extraction/inventory tasks that need more depth
            if self._is_extraction_task(getattr(state, "query", "")):
                return ResearchBudgetProfile(
                    mode=mode,
                    max_depth=min(self.config.max_depth, 5),
                    min_depth=3,
                    max_iterations=min(self.config.max_iterations, 20),
                    depth_citation_threshold=min(self.config.depth_citation_threshold, 15),
                    confidence_threshold=75,
                    min_citations=8,
                    diminishing_returns_fact_threshold=5,
                    diminishing_returns_min_citations=4,
                    diminishing_returns_min_confidence=45,
                    very_low_productivity_max_facts=2,
                )
            return ResearchBudgetProfile(
                mode=mode,
                max_depth=min(self.config.max_depth, 4),
                min_depth=min(self.config.max_depth, 2),
                max_iterations=min(self.config.max_iterations, 15),
                depth_citation_threshold=min(self.config.depth_citation_threshold, 12),
                confidence_threshold=65,
                min_citations=6,
                diminishing_returns_fact_threshold=4,
                diminishing_returns_min_citations=3,
                diminishing_returns_min_confidence=40,
                very_low_productivity_max_facts=1,
            )
        if mode == ResearchMode.SEBIH_SPECIAL.value:
            return ResearchBudgetProfile(
                mode=mode,
                max_depth=max(self.config.max_depth, 8),
                min_depth=max(self.config.min_depth, 3),
                max_iterations=max(self.config.max_iterations, 30),
                depth_citation_threshold=max(self.config.depth_citation_threshold, 20),
                confidence_threshold=82,
                min_citations=12,
                diminishing_returns_fact_threshold=2,
                diminishing_returns_min_citations=4,
                diminishing_returns_min_confidence=55,
                very_low_productivity_max_facts=0,
            )
        return ResearchBudgetProfile(
            mode=ResearchMode.DEEP.value,
            max_depth=self.config.max_depth,
            min_depth=self.config.min_depth,
            max_iterations=self.config.max_iterations,
            depth_citation_threshold=self.config.depth_citation_threshold,
            confidence_threshold=70,
            min_citations=7,
            diminishing_returns_fact_threshold=3,
            diminishing_returns_min_citations=3,
            diminishing_returns_min_confidence=40,
            very_low_productivity_max_facts=1,
        )

    def _initialize_workflow_state(self, state: InvestigationState) -> None:
        """Seed checkpoint-safe workflow state from the active contract."""
        contract = getattr(state, "execution_contract", None)
        output_contract = dict(getattr(contract, "output_contract", {}) or {})
        workflow_kind = str(
            getattr(contract, "workflow_kind", None) or WorkflowKind.ANALYSIS.value
        )
        output_shape = str(output_contract.get("output_shape") or "investigation_memo")

        if state.run_objective is None:
            state.run_objective = RunObjective.create(
                user_goal=state.query,
                output_shape=output_shape,
                workflow_kind=workflow_kind,
                policy_audience="clean",
                success_criteria=self._workflow_success_criteria(contract),
                source_query=state.query,
            )
        if not state.workflow_obligations:
            state.workflow_obligations = self._workflow_obligations(contract)
        if state.working_set is None:
            state.working_set = WorkingSet()

    @staticmethod
    def _workflow_success_criteria(contract: Any) -> list[str]:
        output_contract = dict(getattr(contract, "output_contract", {}) or {})
        task_spec = dict(output_contract.get("task_spec") or {})
        criteria: list[str] = []
        if task_spec:
            task_type = str(task_spec.get("task_type") or "typed_task")
            answer_shape = str(task_spec.get("answer_shape") or "answer")
            criteria.append(f"satisfy task '{task_type}' as {answer_shape}")
            required = task_spec.get("required_evidence") or []
            if required:
                criteria.append(
                    "ground the answer in required evidence objects: "
                    + ", ".join(str(item) for item in required[:8])
                )
            if task_spec.get("fresh_extraction_required"):
                criteria.append(
                    "do not rely on cached summaries alone when source-grounded "
                    "extraction or validation is required"
                )
            if task_spec.get("operation") == "verify_absence":
                criteria.append(
                    "distinguish searched-not-found, false-premise-likely, "
                    "out-of-matter, and source-missing statuses"
                )
        if output_contract.get("must_ground_in_existing_state"):
            criteria.append("use only policy-eligible existing matter state")
        if output_contract.get("fresh_extraction_allowed"):
            criteria.append("advance issue coverage with fresh evidence when needed")
        if (
            output_contract.get("requires_citations")
            or getattr(contract, "citation_floor", 0) > 0
        ):
            criteria.append("ground material claims in cited source artifacts")
        if output_contract.get("requires_gap_section"):
            criteria.append("disclose material proof gaps and missing inputs")
        if output_contract.get("requires_template"):
            criteria.append("follow the selected work-product template")
        if output_contract.get("must_label_assumptions"):
            criteria.append("label temporary assumptions separately from matter facts")
        if output_contract.get("requires_output_validator"):
            criteria.append("pass workflow validators or surface blocking review issues")
        return criteria or ["satisfy the user goal using policy-eligible matter state"]

    @staticmethod
    def _workflow_obligations(contract: Any) -> list[Obligation]:
        output_contract = dict(getattr(contract, "output_contract", {}) or {})
        task_spec = dict(output_contract.get("task_spec") or {})
        obligations: list[Obligation] = []

        def add(
            description: str,
            obligation_type: str,
            validator: str | None = None,
        ) -> None:
            obligations.append(
                Obligation.create(
                    description=description,
                    obligation_type=obligation_type,
                    validator=validator,
                )
            )

        if task_spec:
            task_type = str(task_spec.get("task_type") or "typed_task")
            add(
                f"Answer must satisfy the typed task contract: {task_type}.",
                "task_contract",
                "task_evidence_contract",
            )
            add(
                "Final answer must not contradict the task evidence manifest.",
                "synthesis_alignment",
                "trace_output_alignment",
            )
            required = task_spec.get("required_evidence") or []
            if required:
                add(
                    "Answer must use or disclose absence of required evidence "
                    f"objects: {', '.join(str(item) for item in required[:8])}.",
                    "evidence_contract",
                    "required_evidence_objects",
                )
            if task_spec.get("operation") == "verify_absence":
                add(
                    "Absence checks must label the result as searched-not-found, "
                    "false-premise-likely, out-of-matter, or source-missing.",
                    "missingness",
                    "absence_status",
                )
        if output_contract.get("must_ground_in_existing_state"):
            add(
                "Output must not claim fresh document work occurred on this turn.",
                "state_grounding",
                "read_answerability",
            )
        if (
            output_contract.get("requires_citations")
            or getattr(contract, "citation_floor", 0) > 0
        ):
            add(
                "Material factual claims require source support.",
                "citation",
                "citation_floor",
            )
        if output_contract.get("requires_gap_section"):
            add(
                "Open material gaps must be disclosed instead of papered over.",
                "missingness",
                "gap_disclosure",
            )
        if output_contract.get("requires_template"):
            add(
                "Draft output must be organized by the selected work-product template.",
                "template",
                "draft_template",
            )
        if output_contract.get("must_label_assumptions"):
            add(
                "Temporary assumptions must be labeled apart from stored matter facts.",
                "assumption",
                "assumption_labeling",
            )
        if output_contract.get("requires_review_before_service"):
            add(
                "Draft is not service-ready until a human review gate passes.",
                "review",
                "human_review_required",
            )
        # --- Obligation ledger MVP validators (Codex round 2) ---
        # These fire based on task_spec to catch common benchmark failures:
        # missing issue coverage, missing risk ratings, thin comparisons.
        task_type = str(task_spec.get("task_type") or "") if task_spec else ""
        if task_type == "document_comparison":
            add(
                "Document comparison must identify at least 10 specific deviations "
                "with original vs. changed values.",
                "comparison_coverage",
                "comparison_min_deviations",
            )
        if task_type in {
            "document_comparison", "multi_document_synthesis",
            "quantitative_reconciliation", "portfolio_review",
        }:
            add(
                "Every material issue must have a Red/Yellow/Green risk rating.",
                "risk_rating",
                "risk_rating_per_issue",
            )
        if task_spec and task_spec.get("required_evidence"):
            add(
                "Every planned issue or provision category must have a cited finding "
                "or explicit missingness label.",
                "issue_coverage",
                "issue_coverage_matrix",
            )
        if task_type == "quantitative_reconciliation":
            add(
                "Every required numeric operand must have value, source, and "
                "calculation status.",
                "numeric_coverage",
                "numeric_operand_coverage",
            )
        # Target document coverage: always check if orientation produced target docs
        target_docs = list(output_contract.get("target_documents") or [])
        if target_docs:
            add(
                f"Every target document must be addressed: "
                f"{', '.join(str(d) for d in target_docs[:6])}.",
                "target_document_coverage",
                "target_document_coverage",
            )

        if not obligations:
            add(
                "Output must satisfy the active route contract.",
                "contract",
                "route_contract",
            )
        return obligations

    def _sync_workflow_obligations_from_contract(
        self, state: InvestigationState,
    ) -> None:
        """Refresh obligations after orientation enriches the execution contract."""
        contract = getattr(state, "execution_contract", None)
        if contract is None:
            return
        existing = {
            item.validator or item.obligation_type
            for item in state.workflow_obligations
        }
        for obligation in self._workflow_obligations(contract):
            key = obligation.validator or obligation.obligation_type
            if key not in existing:
                state.workflow_obligations.append(obligation)
                existing.add(key)

    @staticmethod
    def _is_extraction_task(query: str) -> bool:
        """Detect tasks requiring cross-document analysis, calculations, and deep investigation."""
        q = query.lower()
        extraction_signals = (
            "extract", "extraction", "all provisions", "every provision",
            "comprehensive", "inventory", "diligence", "identify all",
            "review all", "each contract", "every contract",
            "change of control", "change-of-control",
            "analyze", "analyse", "compare", "comparison", "markup",
            "redline", "deviation", "counterparty", "review",
            "assess", "assessment", "evaluate", "draft",
            "hsr", "antitrust", "merger", "acquisition",
            "credit facility", "credit agreement", "term sheet",
            "loan agreement", "commitment letter",
            "risk", "strategy", "compliance", "regulatory",
        )
        return sum(1 for s in extraction_signals if s in q) >= 1

    def _build_extraction_instructions(self, query: str) -> str:
        """Generate task-specific synthesis instructions for extraction, analysis, and comparison tasks."""
        q = query.lower()
        extraction_signals = (
            "extract", "extraction", "comprehensive", "all provisions",
            "identify all", "review all", "every contract", "each contract",
            "risk assessment", "change of control", "change-of-control",
        )
        analysis_signals = (
            "analyze", "analyse", "compare", "comparison", "markup",
            "redline", "deviation", "counterparty", "assess", "evaluate",
            "draft", "strategy", "review", "antitrust", "hsr", "merger",
            "credit facility", "credit agreement", "term sheet",
            "loan agreement", "commitment letter",
        )
        is_extraction = any(signal in q for signal in extraction_signals)
        is_analysis = any(signal in q for signal in analysis_signals)
        if not is_extraction and not is_analysis:
            return ""
        base = (
            "EXTRACTION TASK INSTRUCTIONS (MANDATORY):\n"
            "This is a comprehensive extraction task. Your output MUST:\n"
            "1. Address EVERY document/contract in the repository — do not omit any\n"
            "2. For each provision found, cite the EXACT section number (e.g., Section 14.2, Section 8.01(j))\n"
            "3. Quote key definitional language verbatim where it defines thresholds or triggers\n"
            "4. Calculate and state dollar exposures and revenue percentages where data permits\n"
            "5. Assign a risk rating (Critical/High/Moderate/Low) to EVERY contract\n"
            "6. Flag ABSENCE of expected provisions (e.g., no cure period, no explicit CoC definition)\n"
            "7. Identify cross-contract inconsistencies (e.g., different threshold definitions)\n"
            "8. Note timing requirements and deadlines for consent/notification obligations\n"
            "9. Identify the transaction structure and analyze how it interacts with each provision\n"
            "10. Include actionable pre-closing recommendations and post-closing obligations\n"
            "11. For any buy-out or pricing mechanisms, calculate the actual dollar amount\n"
            "12. Note downstream/indirect risks (e.g., future ownership changes re-triggering provisions)\n"
            "13. Do not use headline/facility/minimum figures when a narrower legal operand is available "
            "(e.g., drawn debt rather than commitment, actual TTM revenue rather than minimum purchase commitment, "
            "RSU-specific per-share value rather than unrelated transaction headline value)\n"
            "\n"
            "Structure: Organize by contract/agreement. For each, provide:\n"
            "- Contract name and parties\n"
            "- Relevant provisions with section numbers\n"
            "- Triggers, thresholds, and definitions (quoted)\n"
            "- Consequences (termination, acceleration, consent requirements)\n"
            "- Financial exposure (calculated where possible)\n"
            "- Risk rating with justification\n"
            "- Missing/absent provisions that would normally be expected\n"
            "\n"
            "End with: Cross-document analysis, timing/sequencing issues, and prioritized action items.\n"
        )
        if self._is_mna_change_control_task(query):
            base += (
                "\nFor M&A / change-of-control material contract reports, the final answer must explicitly cover these "
                "recurring diligence checks when the supporting evidence appears:\n"
                "- Supply agreements/MSAs: exact anti-assignment language, including operation-of-law wording; "
                "reverse-triangular-merger ambiguity; UCC Section 2-210 assignment/delegation analysis; "
                "counterparty-specific TTM revenue concentration.\n"
                "- Credit agreements: both Change of Control definitions, Event of Default status under default provisions, "
                "mandatory prepayment timing, automatic commitment termination, and drawn/outstanding debt exposure.\n"
                "- JV/equity arrangements: all carve-out conditions, revenue thresholds, management-retention conditions, "
                "successor revenue comparisons, ownership percentages, buy-out formulas, and calculate buy-out price "
                "using the stated multiple and EBITDA. Distinguish ownership percentage from voting-equity CoC threshold.\n"
                "- Technology and product licenses: embedded product lines, product-line revenue exposure, no-cure "
                "termination rights, consent discretion, and dependent ERP/software systems.\n"
                "- Customer/supply agreements with indirect language: direct/indirect ultimate ownership wording, "
                "future acquirer-parent ownership re-trigger risk, and conditional post-closing consent/termination timing.\n"
                "- Executive/equity documents: single-trigger acceleration, rollover or continued-vesting conflicts, "
                "unvested award count, exchange ratio, per-share value, and acceleration cost.\n"
                "- Real estate leases: prior written consent, deemed-assignment language, consent standards, landlord "
                "termination/recapture rights, fees, and whether any timeline is missing.\n"
                "- Insurance policies: automatic run-off conversion, aggregate limits, successor/new-product exclusions, "
                "tail options, and replacement go-forward coverage action items.\n"
                "- Unreviewed dependencies: flag named ERP, enterprise software, license, or mission-critical system "
                "dependencies as separate contracts to review for CoC/assignment risk.\n"
            )
        if is_analysis and not is_extraction:
            base = (
                "ANALYSIS TASK INSTRUCTIONS (MANDATORY):\n"
                "This is a comprehensive analysis/comparison task. Your output MUST:\n"
                "1. Identify EVERY material issue, deviation, risk, or finding — not just the top 3-5\n"
                "2. For each finding, cite the EXACT section, page, slide, or paragraph reference\n"
                "3. Include SPECIFIC numbers: dollar amounts, percentages, thresholds, dates, ratios\n"
                "4. PERFORM CALCULATIONS where the data supports them — do not merely state inputs\n"
                "   Examples: HHI = sum of squared market shares × 10,000; cost impact = principal × rate change;\n"
                "   covenant headroom = current ratio - threshold; revenue at risk = amount × probability\n"
                "5. For comparison tasks: state BOTH the original value and the changed value for each deviation\n"
                "6. Assign risk ratings (Red/Yellow/Green or Critical/High/Moderate/Low) to EACH issue\n"
                "7. Flag specific documents, emails, memos, and presentations by name as evidence\n"
                "8. Identify absences — provisions, analyses, or data points that SHOULD be present but are missing\n"
                "9. Include specific, actionable recommendations with dollar amounts and timelines\n"
                "10. Cross-reference findings across documents — connect evidence from different sources\n"
                "11. Use ACTUAL financial data from the documents for quantitative analysis, not hypotheticals\n"
                "12. For each party/entity, state their specific role, stake, and exposure\n"
                "\n"
                "QUANTITATIVE RIGOR:\n"
                "- When market shares are available, compute HHI and delta-HHI\n"
                "- When financial terms change, compute the dollar impact on the actual facility/transaction size\n"
                "- When covenant thresholds change, compute headroom against actual performance metrics\n"
                "- When multiple scenarios exist, model the range of outcomes with specific numbers\n"
                "- State your arithmetic explicitly (e.g., '$175M × 0.25% = $437,500/year additional cost')\n"
                "\n"
            )
        return base

    @staticmethod
    def _flatten_contract_card(card: dict, filename: str) -> list[str]:
        """Flatten a contract_card dict into fact strings for synthesis."""
        if not card:
            return []
        lines: list[str] = []
        prefix = f"[CONTRACT_CARD] {filename}"
        for field in (
            "assignment_clause", "change_of_control_definition",
            "consent_requirements", "consent_timing_sequence",
            "termination_rights", "event_of_default_consequences",
            "revenue_exposure", "risk_rating_candidate",
        ):
            val = card.get(field)
            if val and val != "null" and val != "ABSENT":
                lines.append(f"{prefix} | {field}: {val}")
            elif val == "ABSENT":
                lines.append(f"{prefix} | {field}: ABSENT (not found in document)")
        for field in (
            "timing_windows", "carve_outs", "exact_trigger_language",
            "unreviewed_dependency_contracts",
            "financial_operands", "coverage_limits", "post_closing_coverage_gaps",
            "action_items", "missing_expected_provisions",
        ):
            val = card.get(field)
            if isinstance(val, list):
                for item in val:
                    if item and isinstance(item, str):
                        lines.append(f"{prefix} | {field}: {item}")
            elif isinstance(val, str) and val and val not in {"null", "ABSENT"}:
                lines.append(f"{prefix} | {field}: {val}")
        # Legacy flat field — still accept for backward compatibility
        flat_deps = card.get("product_or_system_dependencies") or []
        if isinstance(flat_deps, list):
            for item in flat_deps:
                if item and isinstance(item, str):
                    lines.append(f"{prefix} | product_dependency: {item}")
        # Structured dependency relationships
        dep_rels = card.get("dependency_relationships") or []
        if isinstance(dep_rels, list):
            for dep in dep_rels:
                if isinstance(dep, dict):
                    comp = dep.get("component") or ""
                    host = dep.get("host_product") or ""
                    rel = dep.get("relationship_type") or ""
                    rev = dep.get("revenue_attribution") or ""
                    desc = f"{comp} {rel} {host}".strip()
                    if rev:
                        desc += f" (revenue: {rev})"
                    if desc:
                        lines.append(f"{prefix} | dependency_relationship: {desc}")
                elif isinstance(dep, str) and dep:
                    lines.append(f"{prefix} | dependency_relationship: {dep}")
        # Structured schedule entries
        sched = card.get("schedule_entries") or []
        if isinstance(sched, list):
            for entry in sched:
                if isinstance(entry, dict):
                    ref = entry.get("schedule_ref") or ""
                    target = entry.get("target_label") or ""
                    val = entry.get("value") or ""
                    period = entry.get("period") or ""
                    desc = f"{ref} {target}: {val}".strip()
                    if period:
                        desc += f" ({period})"
                    if desc.strip(": "):
                        lines.append(f"{prefix} | schedule_entry: {desc}")
                elif isinstance(entry, str) and entry:
                    lines.append(f"{prefix} | schedule_entry: {entry}")
        # Structured downstream risks (accept both dict and string)
        downstream = card.get("downstream_indirect_risks") or []
        if isinstance(downstream, list):
            for risk in downstream:
                if isinstance(risk, dict):
                    cname = risk.get("contract_name") or ""
                    actor = risk.get("affected_actor_role") or ""
                    scenario = risk.get("re_trigger_scenario") or ""
                    desc = f"{cname}: {scenario}".strip(": ")
                    if actor:
                        desc += f" [affects {actor}]"
                    if desc:
                        lines.append(f"{prefix} | downstream_risk: {desc}")
                elif isinstance(risk, str) and risk:
                    lines.append(f"{prefix} | downstream_risk: {risk}")
        return lines

    @staticmethod
    def _is_mna_change_control_task(query: str) -> bool:
        """Detect M&A diligence tasks where deterministic contract tools help."""
        q = (query or "").lower()
        return any(
            signal in q
            for signal in (
                "change of control",
                "change-of-control",
                "coc",
                "acquisition",
                "merger",
                "material contract",
                "material-contract",
                "required consent",
                "assignment",
            )
        )

    @staticmethod
    def _get_checklist_issues(query: str) -> list[tuple[str, list[str]]]:
        """Return deterministic issue slots for comparison/regulatory tasks.

        Each entry is (title, [predicates]). These ensure the coverage planner
        has issue objects for provision categories even if orientation missed them.
        """
        q = (query or "").lower()
        issues: list[tuple[str, list[str]]] = []
        _is_comparison = any(w in q for w in (
            "markup", "redline", "compare", "deviation", "counterparty",
            "credit facility", "term sheet", "loan",
        ))
        _is_regulatory = any(w in q for w in (
            "antitrust", "hsr", "merger review", "competition",
            "regulatory strategy",
        ))
        if _is_comparison:
            issues.extend([
                ("Interest rate and SOFR floor analysis", ["SOFR floor", "margin grid", "rate mechanics"]),
                ("Commitment fee structure", ["commitment fee", "unused fee", "facility fee"]),
                ("Financial covenant analysis — leverage ratio", ["leverage ratio", "total net leverage", "step-down schedule"]),
                ("Financial covenant analysis — FCCR", ["fixed charge coverage", "FCCR threshold", "FCCR testing"]),
                ("EBITDA definition and add-back caps", ["EBITDA add-backs", "non-recurring cap", "pro forma adjustments"]),
                ("Synergy add-back provisions", ["synergy cap", "realization period", "pro forma EBITDA"]),
                ("Permitted acquisition baskets", ["individual acquisition basket", "aggregate basket", "pro forma compliance"]),
                ("Restricted payments and distributions", ["restricted payments", "distribution cap", "leverage test for distributions"]),
                ("Excess cash flow sweep mechanics", ["ECF sweep", "step-down", "mandatory prepayment"]),
                ("Events of default — cross-default threshold", ["cross-default", "threshold amount", "judgment default"]),
                ("Change of control definition and threshold", ["change of control trigger", "ownership threshold", "sponsor"]),
                ("Anti-layering and MFN provisions", ["anti-layering", "most favored nation", "MFN cushion"]),
                ("Reinvestment period for asset sales", ["reinvestment period", "asset sale proceeds", "mandatory prepayment"]),
                ("MAE/MAC definition and qualifiers", ["material adverse effect", "taken as a whole", "MAE carve-outs"]),
                ("Extension options and maturity", ["extension option", "maturity date", "amortization"]),
                ("Reporting requirements", ["financial reporting", "compliance certificate", "annual audited"]),
            ])
        if _is_regulatory:
            issues.extend([
                ("Relevant product market definition", ["product markets", "bulk atmospheric", "packaged gases", "specialty gases"]),
                ("Geographic market definition and overlap", ["geographic market", "MSA", "local market", "overlap states"]),
                ("Market concentration and HHI analysis", ["market shares", "HHI calculation", "structural presumption", "delta HHI"]),
                ("Hot document identification", ["internal emails", "board presentations", "pricing language", "eliminates competition"]),
                ("Maverick competitor analysis", ["maverick", "pricing disruptor", "competitive significance"]),
                ("Barriers to entry analysis", ["entry barriers", "ASU construction cost", "timeline for new entry"]),
                ("Customer overlap and dual-sourcing", ["dual-source customers", "customer overlap", "win/loss data"]),
                ("Divestiture and remedy analysis", ["divestiture candidates", "remedy buyers", "divestiture cap"]),
                ("FTC enforcement precedent", ["prior enforcement", "blocked merger", "consent decree"]),
                ("Efficiency and failing firm defenses", ["efficiency defense", "failing firm", "procompetitive"]),
                ("HSR filing mechanics and timeline", ["filing threshold", "waiting period", "Second Request", "outside date"]),
                ("Deal timeline and regulatory risk", ["outside date", "extension", "reverse breakup fee", "timing risk"]),
            ])
        return issues

    @staticmethod
    def _compact_text(text: str, limit: int = 900) -> str:
        compact = _re_engine.sub(r"\s+", " ", text or "").strip()
        return compact[:limit].rstrip()

    @staticmethod
    def _split_fact_lines(text: str) -> list[str]:
        return [
            line.strip()
            for line in _re_engine.split(r"[\r\n;]+", text or "")
            if line and line.strip()
        ]

    @staticmethod
    def _money_amounts_in_millions(text: str) -> list[float]:
        """Extract dollar-like amounts from text and normalize to millions."""
        amounts: list[float] = []
        pattern = _re_engine.compile(
            r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
            r"(billion|bn|million|mm|m|thousand|k)?",
            _re_engine.IGNORECASE,
        )
        for match in pattern.finditer(text or ""):
            raw_num = match.group(1)
            suffix = (match.group(2) or "").lower()
            try:
                value = float(raw_num.replace(",", ""))
            except ValueError:
                continue
            if suffix in {"billion", "bn"}:
                amounts.append(value * 1000.0)
            elif suffix in {"million", "mm", "m"}:
                amounts.append(value)
            elif suffix in {"thousand", "k"}:
                amounts.append(value / 1000.0)
            elif "," in raw_num or value >= 100000:
                amounts.append(value / 1000000.0)
        return amounts

    @classmethod
    def _first_amount_millions_near(
        cls,
        source: str,
        required_terms: tuple[str, ...],
        preferred_terms: tuple[str, ...] = (),
        excluded_terms: tuple[str, ...] = (),
    ) -> Optional[float]:
        """Find the first money amount in a line matching required/preferred terms."""
        lines = cls._split_fact_lines(source)
        for require_preferred in (bool(preferred_terms), False):
            for line in lines:
                lower = line.lower()
                if not all(term.lower() in lower for term in required_terms):
                    continue
                if any(term.lower() in lower for term in excluded_terms):
                    continue
                if require_preferred and not any(
                    term.lower() in lower for term in preferred_terms
                ):
                    continue
                pattern = _re_engine.compile(
                    r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
                    r"(billion|bn|million|mm|m|thousand|k)?",
                    _re_engine.IGNORECASE,
                )
                positioned: list[tuple[int, float]] = []
                for match in pattern.finditer(line):
                    amounts = cls._money_amounts_in_millions(match.group(0))
                    if amounts:
                        positioned.append((match.start(), amounts[0]))
                if positioned:
                    anchor_positions = [
                        lower.find(term.lower())
                        for term in required_terms
                        if lower.find(term.lower()) >= 0
                    ]
                    anchor = max(anchor_positions) if anchor_positions else 0
                    after_anchor = [v for pos, v in positioned if pos >= anchor]
                    return after_anchor[0] if after_anchor else positioned[0][1]
            if not preferred_terms:
                break
        return None

    @staticmethod
    def _format_millions(value: float) -> str:
        if abs(value - round(value)) < 0.05:
            return f"${value:,.0f}M"
        return f"${value:,.1f}M"

    def _contract_card_provision_reinforcement(
        self, card: dict, filename: str, content: str, query: str,
    ) -> list[str]:
        """General-purpose provision reinforcement from contract_card data.

        Scans the document text for legally operative patterns (assignment
        clauses, event-of-default language, timing chains, carve-outs, etc.)
        and emits synthesis-visible facts anchored in the actual extracted text.
        Works on ANY legal document — no hardcoded names, sections, or amounts.
        """
        if not self._is_extraction_task(query):
            return []
        if not card:
            return []

        lower = _re_engine.sub(r"\s+", " ", content or "").strip().lower()
        facts: list[str] = []
        seen: set[str] = set()

        def add(fact: str) -> None:
            fact = self._compact_text(fact, 1200)
            key = fact.lower()
            if fact and key not in seen:
                seen.add(key)
                facts.append(fact)

        prefix = f"[PROVISION_LOCK] {filename}"

        # 1. Anti-assignment / operation-of-law language detection
        assignment_clause = card.get("assignment_clause") or ""
        if assignment_clause and assignment_clause not in ("null", "ABSENT"):
            if "operation of law" in lower:
                add(f"{prefix} | Assignment clause includes 'operation of law' language: {assignment_clause[:300]}")
            if "direct or indirect" in lower and ("ownership" in lower or "control" in lower):
                add(f"{prefix} | Assignment/CoC includes direct/indirect ownership/control language: {assignment_clause[:300]}")

        # 2. CoC definition presence/absence
        coc_def = card.get("change_of_control_definition") or ""
        if coc_def == "ABSENT":
            add(f"{prefix} | No explicit Change of Control definition found; "
                "anti-assignment provisions serve as analogous CoC-sensitive clauses.")
        elif coc_def and coc_def != "null":
            add(f"{prefix} | Change of Control definition: {coc_def[:400]}")

        # 3. Event of default consequences
        eod = card.get("event_of_default_consequences") or ""
        if eod and eod not in ("null", "ABSENT"):
            add(f"{prefix} | Event of Default consequences: {eod[:400]}")

        # 4. Consent timing sequence
        timing_seq = card.get("consent_timing_sequence") or ""
        if timing_seq and timing_seq != "null":
            add(f"{prefix} | Consent/termination timing sequence: {timing_seq[:400]}")

        # 5. Financial operands — emit each for synthesis visibility
        fin_operands = card.get("financial_operands") or []
        if isinstance(fin_operands, list):
            for operand in fin_operands:
                if operand and isinstance(operand, str):
                    add(f"{prefix} | Financial operand: {operand[:300]}")

        # 6. Carve-out conditions with thresholds
        carve_outs = card.get("carve_outs") or []
        if isinstance(carve_outs, list):
            for co in carve_outs:
                if co and isinstance(co, str):
                    add(f"{prefix} | Carve-out condition: {co[:400]}")

        # 7. Downstream/indirect risks (accepts both dicts and strings)
        downstream = card.get("downstream_indirect_risks") or []
        if isinstance(downstream, list):
            for risk in downstream:
                if isinstance(risk, dict):
                    cname = risk.get("contract_name") or prefix
                    actor = risk.get("affected_actor_role") or "acquirer"
                    scenario = risk.get("re_trigger_scenario") or ""
                    section = risk.get("provision_section") or ""
                    desc = f"{cname}"
                    if section:
                        desc += f" ({section})"
                    desc += f": {scenario}" if scenario else ""
                    desc += f" [affects {actor}]"
                    add(f"{prefix} | Downstream risk: {desc[:400]}")
                elif risk and isinstance(risk, str):
                    add(f"{prefix} | Downstream risk: {risk[:300]}")

        # 8. Unreviewed dependency contracts
        unreviewed = card.get("unreviewed_dependency_contracts") or []
        if isinstance(unreviewed, list):
            for dep in unreviewed:
                if dep and isinstance(dep, str):
                    add(f"{prefix} | Unreviewed dependency: {dep[:200]} — "
                        "should be reviewed for CoC/assignment/consent/termination risk.")

        # 9. Post-closing coverage gaps
        gaps = card.get("post_closing_coverage_gaps") or []
        if isinstance(gaps, list):
            for gap in gaps:
                if gap and isinstance(gap, str):
                    add(f"{prefix} | Post-closing coverage gap: {gap[:300]}")

        # 10. Missing expected provisions
        missing = card.get("missing_expected_provisions") or []
        if isinstance(missing, list):
            for m in missing:
                if m and isinstance(m, str):
                    add(f"{prefix} | Missing expected provision: {m[:200]}")

        # 11-12: M&A-specific structural analysis (gated)
        if self._is_mna_change_control_task(query):
            if ("merger sub" in lower or "merger subsidiary" in lower) and (
                "merge with and into" in lower or "merged with and into" in lower
            ):
                add(f"{prefix} | Transaction structure detected: reverse triangular merger "
                    "(target survives as subsidiary). Entity survival means anti-assignment "
                    "clauses may not be triggered but this is jurisdiction-dependent.")
            doc_subtype = (card.get("doc_subtype") or "").lower()
            if any(t in doc_subtype for t in ("supply", "services", "msa", "goods")):
                if "assign" in lower:
                    add(f"{prefix} | UCC Section 2-210 may apply: distinguish assignment "
                        "of rights from delegation of duties for this supply/services agreement.")

        return facts

    @staticmethod
    def _has_any_text(corpus: str, needles: tuple[str, ...]) -> bool:
        c = (corpus or "").lower()
        return any(n.lower() in c for n in needles)

    @staticmethod
    def _build_operand_lock_lines(corpus: str) -> list[str]:
        """Build general-purpose operand-lock instructions from extracted evidence.

        Scans the corpus for patterns indicating financial operands, timing
        sequences, and structural provisions, then emits disambiguation
        instructions. Works on ANY document set — no hardcoded names or values.
        """
        c = corpus or ""
        cl = c.lower()
        lines: list[str] = []

        # 1. Detect drawn-vs-committed credit disambiguation
        has_revolving = "revolving" in cl or "drawn" in cl or "outstanding" in cl
        has_commitment = "commitment" in cl or "facility" in cl or "maximum" in cl
        if has_revolving and has_commitment and "credit" in cl:
            drawn_amt = RLMEngine._first_amount_millions_near(
                c, ("drawn",), preferred_terms=("outstanding", "revolving")
            ) or RLMEngine._first_amount_millions_near(
                c, ("outstanding",), preferred_terms=("revolving", "loans")
            )
            commit_amt = RLMEngine._first_amount_millions_near(
                c, ("commitment",), preferred_terms=("aggregate", "facility", "maximum")
            )
            if drawn_amt and commit_amt and drawn_amt != commit_amt:
                lines.append(
                    f"Credit facility: use the drawn/outstanding amount "
                    f"({RLMEngine._format_millions(drawn_amt)}) as the actual exposure, "
                    f"not the commitment/facility maximum ({RLMEngine._format_millions(commit_amt)}). "
                    "Connect drawn amount to mandatory prepayment consequences."
                )

        # 2. Detect revenue concentration calculations
        # Company TTM must come from a line mentioning total/company/consolidated/
        # estimated revenue — NOT from counterparty-specific or minimum-commitment lines.
        company_ttm = RLMEngine._first_amount_millions_near(
            c, ("revenue",),
            preferred_terms=("total", "company", "consolidated", "estimated", "reported", "borrower"),
            excluded_terms=(
                "attributable", "minimum", "commitment", "counterparty",
                "historical range", "exposure", "operand",
            ),
        )
        if company_ttm and company_ttm > 50:
            # Find counterparty-specific revenues ONLY from schedule disclosures
            # or "attributable" patterns — these are actual TTM, not minimums.
            counterparty_revenues: list[tuple[str, float]] = []
            for line in RLMEngine._split_fact_lines(c):
                ll = line.lower()
                has_schedule = "schedule" in ll or "attributable" in ll
                has_revenue = "revenue" in ll or "sales" in ll
                if has_revenue and has_schedule and ("ttm" in ll or "trailing" in ll):
                    amt = RLMEngine._first_amount_millions_near(
                        line, ("revenue",), preferred_terms=("attributable", "trailing", "ttm")
                    )
                    if amt and amt != company_ttm and 0.5 < amt < company_ttm:
                        counterparty_revenues.append((line[:80], amt))
            for label, amt in counterparty_revenues[:5]:
                pct = amt / company_ttm * 100.0
                lines.append(
                    f"Revenue concentration: {RLMEngine._format_millions(amt)} / "
                    f"{RLMEngine._format_millions(company_ttm)} = {pct:.1f}%. "
                    "Use actual TTM revenue, not minimum purchase commitments."
                )

        # 3. Detect RSU/equity acceleration calculations
        if "rsu" in cl or "restricted stock" in cl or "unvested" in cl:
            rsu_count = RLMEngine._first_amount_millions_near(
                c, ("unvested",), preferred_terms=("rsu", "restricted", "shares")
            )
            share_price = RLMEngine._first_amount_millions_near(
                c, ("per-share", "per share", "share price"),
                preferred_terms=("implied", "transaction", "exchange")
            )
            if rsu_count is None:
                # Try alternate patterns for counts (not in millions)
                count_pattern = _re_engine.compile(r"(\d{1,3}(?:,\d{3})*)\s*(?:unvested|rsu)", _re_engine.IGNORECASE)
                m = count_pattern.search(c)
                if m:
                    try:
                        rsu_count_raw = int(m.group(1).replace(",", ""))
                        if rsu_count_raw > 0:
                            lines.append(
                                f"RSU/equity acceleration: use the exact unvested count "
                                f"({rsu_count_raw:,}) from the agreement. Multiply by the "
                                "per-share transaction/implied value (not unrelated headline values)."
                            )
                    except ValueError:
                        pass

        # 4. Detect buy-out/pricing multiples
        if "buy-out" in cl or "buyout" in cl or "purchase price" in cl:
            multiple_pattern = _re_engine.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(?:ttm\s*)?ebitda", _re_engine.IGNORECASE)
            m = multiple_pattern.search(c)
            if m:
                multiple = float(m.group(1))
                lines.append(
                    f"Buy-out pricing: apply the {multiple}x EBITDA multiple from the "
                    "agreement to calculate the actual buy-out price. "
                    f"{'Flag potential below-market pricing risk (multiple ≤5.0x). ' if multiple <= 5.0 else ''}"
                    "Do not confuse this multiple with other valuation metrics."
                )

        # 5. Detect timing-chain sequences
        timing_pattern = _re_engine.compile(
            r"(?:within|not\s+(?:later\s+than|obtained\s+within))\s+"
            r"(\w+)\s*\((\d+)\)\s*(?:days?|business\s+days?)",
            _re_engine.IGNORECASE,
        )
        timing_matches = list(timing_pattern.finditer(c))
        if len(timing_matches) >= 2:
            lines.append(
                "Cross-contract timing: multiple consent/notification deadlines detected. "
                "Include ALL timing windows in sequencing analysis and flag conflicts."
            )

        # 6. Detect operation-of-law / reverse merger structural issues
        if "operation of law" in cl and ("reverse" in cl or "merger sub" in cl or "survives" in cl):
            lines.append(
                "Structural analysis: for each contract with 'operation of law' anti-assignment "
                "language, separately analyze whether entity survival in a reverse merger avoids "
                "the trigger (jurisdiction-dependent). Apply UCC § 2-210 for supply/goods agreements."
            )

        # 7. Detect indirect/downstream re-trigger risk
        if ("direct or indirect" in cl or "indirect" in cl) and (
            "ownership" in cl or "control" in cl
        ) and "ultimate" in cl:
            lines.append(
                "Downstream risk: for contracts with 'direct or indirect' ultimate ownership/control "
                "language, flag that future changes in the acquirer's own ownership could re-trigger "
                "the provision."
            )

        # 8. Detect event of default that should be reported separately from prepayment
        if "event of default" in cl and ("prepay" in cl or "prepayment" in cl):
            lines.append(
                "Credit agreement: if Change of Control is both an Event of Default AND triggers "
                "mandatory prepayment, report these as SEPARATE consequences with distinct sections."
            )

        # 9. Detect unreviewed software/ERP dependencies
        if ("erp" in cl or "platform" in cl or "software" in cl) and (
            "mission-critical" in cl or "manufacturing" in cl or "inventory" in cl
        ):
            lines.append(
                "Software/ERP dependency: flag any named platform/software system as a potentially "
                "unreviewed CoC/assignment dependency whose loss could impair operations."
            )

        return lines

    def _build_mna_coc_completion_checklist(self, query: str, corpus: str) -> str:
        if not self._is_extraction_task(query):
            return ""
        lines = self._build_operand_lock_lines(corpus)
        if not lines:
            return ""
        output = [
            "OPERAND LOCK / COMPLETION CHECKLIST (MANDATORY):",
            "The following instructions are derived from extracted evidence. "
            "Address each in the final report. Do not drop operands or substitute wrong figures.",
        ]
        output.extend(f"- {line}" for line in lines)
        return "\n".join(output)

    @staticmethod
    def _derived_finding_conflicts_with_operand_locks(finding: str, corpus: str) -> bool:
        """General operand-conflict detection.

        If the corpus contains a PROVISION_LOCK or Financial operand line with
        a specific amount, and the derived finding references the same topic
        but uses a DIFFERENT amount that looks like a common confusion (e.g.,
        facility max instead of drawn amount), reject it.
        """
        f = (finding or "").lower()
        c = (corpus or "").lower()
        if not f:
            return False

        # Extract amounts from the finding
        finding_amounts = RLMEngine._money_amounts_in_millions(f)
        if not finding_amounts:
            return False

        # Look for PROVISION_LOCK financial operand lines in corpus
        lock_lines = [
            line for line in RLMEngine._split_fact_lines(c)
            if "provision_lock" in line.lower() and "financial operand" in line.lower()
        ]

        for lock_line in lock_lines:
            lock_amounts = RLMEngine._money_amounts_in_millions(lock_line)
            if not lock_amounts:
                continue
            # If finding mentions a topic word from the lock but uses a different amount
            lock_words = set(lock_line.lower().split()) - {
                "provision_lock", "financial", "operand", "|", "the", "a", "an",
                "of", "in", "for", "and", "or", "to", "is", "at",
            }
            topic_overlap = sum(1 for w in lock_words if w in f and len(w) > 3)
            if topic_overlap >= 2:
                # Check if finding uses a different amount than the lock
                for f_amt in finding_amounts:
                    for l_amt in lock_amounts:
                        if l_amt > 0 and abs(f_amt - l_amt) / l_amt > 0.1:
                            # Finding uses a substantially different amount on same topic
                            if f_amt not in lock_amounts:
                                return True
        return False

    def _derive_mna_coc_findings(self, query: str, facts: list[str], corpus: str) -> list[str]:
        if not self._is_extraction_task(query):
            return []
        return [f"[DERIVED] {line}" for line in self._build_operand_lock_lines(corpus)]

    # ------------------------------------------------------------------
    # Structural cross-document linking (graph-backed operand resolution)
    # ------------------------------------------------------------------

    _OPERAND_ROLE_PATTERNS: "list[tuple[str, tuple[str, ...]]]" = [
        # Company-level TTM revenue (denominator) — must match BEFORE counterparty
        ("company_total_ttm_revenue", ("total", "ttm", "revenue")),
        ("company_total_ttm_revenue", ("company", "ttm", "revenue")),
        ("company_total_ttm_revenue", ("company", "trailing", "revenue")),
        ("company_total_ttm_revenue", ("consolidated", "revenue")),
        ("company_total_ttm_revenue", ("total", "trailing", "revenue")),
        # Counterparty-specific TTM revenue (numerator)
        ("actual_counterparty_ttm_revenue", ("ttm", "attributable")),
        ("actual_counterparty_ttm_revenue", ("trailing", "revenue", "attributable")),
        ("actual_counterparty_ttm_revenue", ("revenue", "attributable")),
        ("actual_counterparty_ttm_revenue", ("ttm revenue", "attributable")),
        # Product-line revenue (for dependency chain resolution)
        ("actual_product_line_ttm_revenue", ("product line", "revenue")),
        ("actual_product_line_ttm_revenue", ("product", "ttm", "revenue")),
        ("actual_product_line_ttm_revenue", ("product", "trailing", "revenue")),
        # Minimum purchase (the wrong numerator for revenue exposure)
        ("minimum_purchase_commitment", ("minimum", "purchase")),
        ("minimum_purchase_commitment", ("minimum", "commitment")),
        ("minimum_purchase_commitment", ("annual", "commitment")),
        # Credit facility operands
        ("drawn_outstanding", ("drawn",)),
        ("drawn_outstanding", ("outstanding",)),
        ("facility_commitment", ("facility", "commitment")),
        ("facility_commitment", ("aggregate", "commitment")),
        ("facility_commitment", ("revolving", "commitment")),
        # Equity
        ("unvested_rsu_count", ("unvested", "rsu")),
        ("unvested_rsu_count", ("unvested", "restricted")),
        ("per_share_price", ("per-share",)),
        ("per_share_price", ("per share", "price")),
        ("per_share_price", ("share price",)),
        # Buy-out
        ("buyout_multiple", ("ebitda", "multiple")),
        ("buyout_multiple", ("buy-out", "multiple")),
        # Acquirer / buyer revenue (for threshold tests against carve-outs)
        ("acquirer_revenue", ("acquirer", "revenue")),
        ("acquirer_revenue", ("buyer", "revenue")),
        ("acquirer_revenue", ("parent", "revenue")),
        ("acquirer_segment_revenue", ("acquirer", "segment", "revenue")),
        ("acquirer_segment_revenue", ("buyer", "segment", "revenue")),
        # Thresholds and limits
        ("threshold_amount", ("threshold",)),
        ("coverage_limit", ("coverage", "limit")),
        ("coverage_limit", ("aggregate limit",)),
    ]

    @staticmethod
    def _classify_operand_role(text: str) -> str:
        """Classify a financial operand string into a semantic role."""
        t = (text or "").lower()
        for role, keywords in RLMEngine._OPERAND_ROLE_PATTERNS:
            if all(k in t for k in keywords):
                return role
        return "unclassified"

    @staticmethod
    def _extract_operand_value_millions(text: str) -> "Optional[float]":
        """Extract the primary numeric value from an operand string (in millions)."""
        amounts = RLMEngine._money_amounts_in_millions(text)
        return amounts[0] if amounts else None

    @staticmethod
    def _extract_operand_subject(text: str, source_filename: str) -> str:
        """Extract the subject entity/contract label from an operand string."""
        t = text or ""
        for marker in ("attributable to ", "for ", "from "):
            idx = t.lower().find(marker)
            if idx >= 0:
                after = t[idx + len(marker):]
                label = after.split(":")[0].split(",")[0].split("(")[0].strip()
                if label and len(label) > 2:
                    return label
        return source_filename

    def _persist_contract_evidence(self, card: dict, filename: str) -> None:
        """Persist contract_card and its operands as typed evidence records.

        Materializes the contract_card into the graph so downstream
        deterministic passes can resolve cross-document operand links
        without relying on prompt text alone.
        """
        if self._matter_model is None:
            return
        te = self._matter_model.typed_evidence

        contract_name = card.get("contract_name") or filename
        counterparty = card.get("counterparty") or ""
        card_key = f"card:{filename}"
        te.upsert(
            "contract_card", card_key,
            payload={
                "contract_name": contract_name,
                "counterparty": counterparty,
                "filename": filename,
                "risk_rating": card.get("risk_rating_candidate") or "",
            },
            label=contract_name,
            document_id=filename,
            confidence=0.9,
        )

        fin_operands = card.get("financial_operands") or []
        if isinstance(fin_operands, list):
            for i, op_text in enumerate(fin_operands):
                if not isinstance(op_text, str) or not op_text.strip():
                    continue
                role = self._classify_operand_role(op_text)
                value = self._extract_operand_value_millions(op_text)
                subject = self._extract_operand_subject(op_text, filename)
                op_key = f"op:{filename}:{i}"
                te.upsert(
                    "calculation_operand", op_key,
                    payload={
                        "operand_role": role,
                        "value_millions": value,
                        "subject_label": subject,
                        "raw_text": op_text[:500],
                        "source_document": filename,
                        "source_priority": "body_stated",
                    },
                    label=f"{role}:{subject}",
                    document_id=filename,
                    confidence=0.85,
                )

        carve_outs = card.get("carve_outs") or []
        if isinstance(carve_outs, list):
            for i, co_text in enumerate(carve_outs):
                if not isinstance(co_text, str) or not co_text.strip():
                    continue
                co_key = f"prov:carve_out:{filename}:{i}"
                te.upsert(
                    "contract_provision", co_key,
                    payload={
                        "provision_kind": "carve_out",
                        "raw_text": co_text[:500],
                        "contract_name": contract_name,
                        "source_document": filename,
                    },
                    label=f"carve_out:{contract_name}",
                    document_id=filename,
                    confidence=0.85,
                )

        dep_rels = card.get("dependency_relationships") or []
        schedule_entries = card.get("schedule_entries") or []
        if isinstance(schedule_entries, list):
            for i, entry in enumerate(schedule_entries):
                if not isinstance(entry, dict):
                    continue
                target = entry.get("target_label") or ""
                sched_ref = entry.get("schedule_ref") or ""
                if not target and not sched_ref:
                    continue
                se_key = f"sched:{filename}:{i}"
                value_raw = entry.get("value") or ""
                value_m = self._extract_operand_value_millions(str(value_raw)) if value_raw else None
                te.upsert(
                    "schedule_entry", se_key,
                    payload={
                        "schedule_ref": sched_ref,
                        "row_index": entry.get("row_index", i),
                        "target_label": target,
                        "metric": entry.get("metric") or "",
                        "value_raw": str(value_raw)[:300],
                        "value_millions": value_m,
                        "period": entry.get("period") or "",
                        "linked_contract": entry.get("linked_contract") or "",
                        "source_document": filename,
                    },
                    label=f"{sched_ref}:{target}",
                    document_id=filename,
                    confidence=0.85,
                )
                if value_m and target:
                    metric = (entry.get("metric") or "").lower()
                    period = (entry.get("period") or "").lower()
                    if "revenue" in metric or "ttm" in period or "trailing" in period:
                        sched_role = "actual_counterparty_ttm_revenue"
                        tgt_lower = target.lower()
                        for dep in dep_rels:
                            if not isinstance(dep, dict):
                                continue
                            dc = (dep.get("component") or "").lower()
                            dh = (dep.get("host_product") or "").lower()
                            if dc and len(dc) > 2 and (dc in tgt_lower or tgt_lower in dc):
                                sched_role = "actual_product_line_ttm_revenue"
                                break
                            if dh and len(dh) > 2 and (dh in tgt_lower or tgt_lower in dh):
                                sched_role = "actual_product_line_ttm_revenue"
                                break
                        op_key = f"op:sched:{filename}:{i}"
                        te.upsert(
                            "calculation_operand", op_key,
                            payload={
                                "operand_role": sched_role,
                                "value_millions": value_m,
                                "subject_label": target,
                                "raw_text": f"{sched_ref} {target}: {value_raw}"[:500],
                                "source_document": filename,
                                "source_priority": "schedule_disclosed",
                            },
                            label=f"schedule_operand:{target}",
                            document_id=filename,
                            confidence=0.9,
                        )

        if isinstance(dep_rels, list):
            for i, dep in enumerate(dep_rels):
                if not isinstance(dep, dict):
                    continue
                component = dep.get("component") or ""
                host = dep.get("host_product") or ""
                if not component and not host:
                    continue
                dep_key = f"dep:{filename}:{i}"
                te.upsert(
                    "product_dependency", dep_key,
                    payload={
                        "component": component,
                        "host_product": host,
                        "relationship_type": dep.get("relationship_type") or "unknown",
                        "revenue_attribution": dep.get("revenue_attribution") or "",
                        "source_section": dep.get("source_section") or "",
                        "contract_or_vendor": dep.get("contract_or_vendor") or "",
                        "source_document": filename,
                        "contract_name": contract_name,
                    },
                    label=f"{component}->{host}",
                    document_id=filename,
                    confidence=0.85,
                )
                rev_attr = dep.get("revenue_attribution") or ""
                if rev_attr:
                    rev_val = self._extract_operand_value_millions(str(rev_attr))
                    if rev_val:
                        op_key = f"op:dep:{filename}:{i}"
                        te.upsert(
                            "calculation_operand", op_key,
                            payload={
                                "operand_role": "actual_product_line_ttm_revenue",
                                "value_millions": rev_val,
                                "subject_label": host or component,
                                "raw_text": f"{component} in {host}: {rev_attr}"[:500],
                                "source_document": filename,
                                "source_priority": "schedule_disclosed",
                            },
                            label=f"product_revenue:{host or component}",
                            document_id=filename,
                            confidence=0.85,
                        )

        downstream_risks = card.get("downstream_indirect_risks") or []
        if isinstance(downstream_risks, list):
            for i, risk in enumerate(downstream_risks):
                if isinstance(risk, str):
                    risk = {"re_trigger_scenario": risk}
                if not isinstance(risk, dict):
                    continue
                scenario = risk.get("re_trigger_scenario") or ""
                if not scenario:
                    continue
                dr_key = f"prov:downstream:{filename}:{i}"
                te.upsert(
                    "contract_provision", dr_key,
                    payload={
                        "provision_kind": "downstream_risk",
                        "contract_name": risk.get("contract_name") or contract_name,
                        "provision_section": risk.get("provision_section") or "",
                        "trigger_language": risk.get("trigger_language") or "",
                        "affected_actor_role": risk.get("affected_actor_role") or "acquirer",
                        "re_trigger_scenario": scenario[:500],
                        "source_document": filename,
                    },
                    label=f"downstream_risk:{risk.get('contract_name') or contract_name}",
                    document_id=filename,
                    confidence=0.85,
                )

    def _derive_provision_comparison_calculations(self) -> "list[str]":
        """Deterministic arithmetic on provision comparison data (Codex #3).

        Reads provision_comparison typed evidence and computes:
        - Rate/spread deltas × principal for dollar impact
        - Ratio threshold deltas (old - new covenant headroom)
        - Dollar cap reductions
        Emits [CALCULATED] facts that don't rely on LLM arithmetic.
        """
        if self._matter_model is None:
            return []
        try:
            rows = self._matter_model.typed_evidence.list_by_kind(
                "provision_comparison", limit=200,
            )
        except Exception:
            return []
        if not rows:
            return []

        by_provision: dict[str, dict[str, dict]] = {}
        for row in rows:
            payload = row.get("payload_json")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            prov = payload.get("provision", "")
            role = payload.get("source_role", "unknown")
            if not prov:
                continue
            if prov not in by_provision:
                by_provision[prov] = {}
            by_provision[prov][role] = payload

        results: list[str] = []

        def _extract_number(val_str: str) -> "Optional[float]":
            """Extract a numeric value from a provision value string.

            BPS values are converted to percentages (25 bps → 0.25).
            """
            if not val_str:
                return None
            import re
            val_str_clean = val_str.replace(",", "").replace("$", "").strip()
            m = re.search(r'([\d.]+)\s*(?:bps|basis\s*point)', val_str_clean, re.IGNORECASE)
            if m:
                return float(m.group(1)) / 100.0
            m = re.search(r'([\d.]+)\s*[%x]', val_str_clean)
            if m:
                return float(m.group(1))
            m = re.search(r'([\d.]+)\s*(M|million|mm)', val_str_clean, re.IGNORECASE)
            if m:
                return float(m.group(1)) * 1_000_000
            m = re.search(r'([\d.]+)', val_str_clean)
            if m:
                return float(m.group(1))
            return None

        def _is_percentage(val_str: str) -> bool:
            return "%" in val_str or "bps" in val_str.lower() or "basis" in val_str.lower()

        def _is_ratio(val_str: str) -> bool:
            return "x" in val_str.lower() and "%" not in val_str

        # Find facility size from facts for dollar impact calculations
        _facility_size: "Optional[float]" = None
        try:
            quant_rows = self._matter_model.quant.list_all(limit=100)
            for qr in quant_rows:
                ctx = (qr.get("context") or "").lower()
                raw = (qr.get("raw_text") or "").lower()
                if any(w in ctx + raw for w in ("facility", "commitment", "revolving", "term loan")):
                    val = qr.get("amount_value")
                    if val and val > 10_000_000:
                        _facility_size = float(val)
                        break
        except Exception:
            pass

        for prov, roles in by_provision.items():
            orig_data = roles.get("original", {})
            markup_data = roles.get("markup", {})
            if not orig_data or not markup_data:
                continue
            orig_val_str = orig_data.get("value", "")
            markup_val_str = markup_data.get("value", "")
            orig_num = _extract_number(orig_val_str)
            markup_num = _extract_number(markup_val_str)
            if orig_num is None or markup_num is None:
                continue

            delta = markup_num - orig_num
            if abs(delta) < 0.001:
                continue

            direction = "tightened" if delta > 0 else "loosened"
            if _is_percentage(orig_val_str):
                results.append(
                    f"[CALCULATED] {prov}: changed from {orig_val_str} to {markup_val_str} "
                    f"(delta: {'+' if delta > 0 else ''}{delta:.2f}%, {direction})"
                )
                if _facility_size and abs(delta) < 10:
                    annual_impact = _facility_size * abs(delta) / 100.0
                    results.append(
                        f"[CALCULATED] {prov} dollar impact: "
                        f"${_facility_size:,.0f} × {abs(delta):.2f}% = "
                        f"${annual_impact:,.0f}/year"
                    )
            elif _is_ratio(orig_val_str):
                results.append(
                    f"[CALCULATED] {prov}: changed from {orig_val_str} to {markup_val_str} "
                    f"(delta: {'+' if delta > 0 else ''}{delta:.2f}x, {direction})"
                )
            else:
                results.append(
                    f"[CALCULATED] {prov}: changed from {orig_val_str} to {markup_val_str} "
                    f"(delta: {'+' if delta > 0 else ''}{delta:,.0f})"
                )

        return results

    def _derive_regulatory_calculations(self) -> "list[str]":
        """Deterministic arithmetic on regulatory evidence data (Codex #3 extension).

        Reads regulatory_data typed evidence and computes:
        - HHI from market share data where available
        - Divestiture revenue vs cap analysis
        - Deal value ratios (breakup fee as % of consideration)
        """
        if self._matter_model is None:
            return []
        try:
            rows = self._matter_model.typed_evidence.list_by_kind(
                "regulatory_data", limit=300,
            )
        except Exception:
            return []
        if not rows:
            return []

        results: list[str] = []

        market_shares: dict[str, list[tuple[str, float]]] = {}
        remedy_values: list[dict] = []
        deal_value: "Optional[float]" = None
        breakup_fee: "Optional[float]" = None
        divestiture_cap: "Optional[float]" = None

        import re as _re_reg

        def _parse_pct(s: str) -> "Optional[float]":
            s = s.replace(",", "").strip()
            m = _re_reg.search(r'([\d.]+)\s*%', s)
            return float(m.group(1)) if m else None

        def _parse_dollar(s: str) -> "Optional[float]":
            s = s.replace(",", "").replace("$", "").strip()
            m = _re_reg.search(r'([\d.]+)\s*(B|billion)', s, _re_reg.IGNORECASE)
            if m:
                return float(m.group(1)) * 1_000_000_000
            m = _re_reg.search(r'([\d.]+)\s*(M|million|mm)', s, _re_reg.IGNORECASE)
            if m:
                return float(m.group(1)) * 1_000_000
            m = _re_reg.search(r'([\d.]+)', s)
            if m and float(m.group(1)) > 1000:
                return float(m.group(1))
            return None

        for row in rows:
            payload = row.get("payload_json")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            cat = (payload.get("category") or "").lower()
            entity = payload.get("entity") or ""
            value = payload.get("value") or ""

            if cat == "market_share":
                pct = _parse_pct(value)
                if pct is not None and pct > 0:
                    market_key = entity
                    for geo in ("Atlanta", "Savannah", "Charleston", "Greenville",
                                "Houston", "Jacksonville", "Tampa", "Birmingham",
                                "Charlotte", "Raleigh", "Nashville", "Memphis",
                                "New Orleans", "Baton Rouge", "Mobile"):
                        if geo.lower() in entity.lower():
                            market_key = geo
                            break
                    market_shares.setdefault(market_key, []).append((entity, pct))

            elif cat == "remedy":
                dollar = _parse_dollar(value)
                if dollar:
                    remedy_values.append({"entity": entity, "value": dollar, "raw": value})
                vl = value.lower()
                if "cap" in entity.lower() or "cap" in vl or "divestiture" in entity.lower():
                    d = _parse_dollar(value)
                    if d and d < 500_000_000:
                        divestiture_cap = d
                if "breakup" in entity.lower() or "breakup" in vl or "termination" in entity.lower():
                    d = _parse_dollar(value)
                    if d:
                        breakup_fee = d

        try:
            quant_rows = self._matter_model.quant.list_all(limit=100)
            for qr in quant_rows:
                ctx = (qr.get("context") or "").lower()
                raw = (qr.get("raw_text") or "").lower()
                val = qr.get("amount_value")
                if not val:
                    continue
                if any(w in ctx + raw for w in ("consideration", "deal value", "acquisition", "merger")):
                    if val > 100_000_000:
                        deal_value = float(val)
                if any(w in ctx + raw for w in ("breakup", "termination fee", "reverse")):
                    if val > 1_000_000 and val < 500_000_000:
                        breakup_fee = float(val)
                if any(w in ctx + raw for w in ("divestiture cap", "maximum divestiture", "revenue cap")):
                    if val > 1_000_000:
                        divestiture_cap = float(val)
        except Exception:
            pass

        for market, shares in market_shares.items():
            if len(shares) >= 2:
                total_hhi = sum(s ** 2 for _, s in shares)
                results.append(
                    f"[CALCULATED] HHI for {market}: {total_hhi:,.0f} "
                    f"(from {len(shares)} competitors: "
                    + ", ".join(f"{e} {s:.1f}%" for e, s in shares) + ")"
                )

        if divestiture_cap:
            total_divest_revenue = 0.0
            for rv in remedy_values:
                if rv["value"] > divestiture_cap * 0.5 and rv["value"] < divestiture_cap * 3:
                    el = rv["entity"].lower()
                    if "candidate" in el or "revenue" in el or "facility" in el:
                        total_divest_revenue = max(total_divest_revenue, rv["value"])
            if total_divest_revenue > 0:
                gap = total_divest_revenue - divestiture_cap
                results.append(
                    f"[CALCULATED] Divestiture candidate revenue ${total_divest_revenue:,.0f} "
                    f"vs cap ${divestiture_cap:,.0f} — "
                    f"{'EXCEEDS cap by ${:,.0f}'.format(gap) if gap > 0 else 'within cap by ${:,.0f}'.format(-gap)}"
                )

        if deal_value and breakup_fee:
            pct = (breakup_fee / deal_value) * 100
            results.append(
                f"[CALCULATED] Reverse breakup fee ${breakup_fee:,.0f} = "
                f"{pct:.1f}% of deal value ${deal_value:,.0f}"
            )

        return results

    def _resolve_operand_graph_calculations(
        self, facts: "Optional[list[str]]" = None,
    ) -> "list[str]":
        """Deterministic cross-document calculation pass using graph-stored operands.

        Queries the typed evidence store for contract_cards and
        calculation_operands, resolves schedule-disclosed operands to
        contracts, and computes revenue exposure / credit exposure
        without relying on the LLM to pick the correct operands.

        Also scans accumulated facts text for schedule-disclosed revenue
        patterns as a fallback when structured operands are incomplete.
        """
        if self._matter_model is None:
            return []
        te = self._matter_model.typed_evidence
        results: list[str] = []

        # Load all contract cards and operands from the graph
        contract_cards = te.list_by_kind("contract_card", limit=50)
        all_operands = te.list_by_kind("calculation_operand", limit=200)

        if not contract_cards:
            return []

        # Build contract registry: name/counterparty → card
        contract_registry: "dict[str, dict]" = {}
        for cc in contract_cards:
            payload = cc.get("payload_json")
            if isinstance(payload, str):
                import json as _json_mod
                try:
                    payload = _json_mod.loads(payload)
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            cname = (payload.get("contract_name") or "").lower().strip()
            cparty = (payload.get("counterparty") or "").lower().strip()
            fname = (payload.get("filename") or "").lower().strip()
            entry = {"payload": payload, "record": cc}
            if cname:
                contract_registry[cname] = entry
            if cparty:
                contract_registry[cparty] = entry
            if fname:
                contract_registry[fname] = entry

        # Parse all operands
        parsed_operands: "list[dict]" = []
        for op in all_operands:
            payload = op.get("payload_json")
            if isinstance(payload, str):
                import json as _json_mod
                try:
                    payload = _json_mod.loads(payload)
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            parsed_operands.append(payload)

        # Also check the quant store for schedule-disclosed revenue operands
        # that might not be in financial_operands but were in numeric_facts
        try:
            amount_quants = self._matter_model.quant.get_amounts(min_value=1.0)
            for qf in amount_quants:
                raw = (qf.get("raw_text") or "").lower()
                subj_id = qf.get("subject_id") or ""
                subj_type = (qf.get("subject_type") or "").lower()
                if ("ttm" in raw or "trailing" in raw or "attributable" in raw) and "revenue" in raw:
                    value_m = (qf.get("amount_value") or 0) / 1_000_000.0 if (qf.get("amount_value") or 0) > 1000 else qf.get("amount_value")
                    if value_m and value_m > 0.5:
                        parsed_operands.append({
                            "operand_role": "actual_counterparty_ttm_revenue",
                            "value_millions": value_m,
                            "subject_label": subj_id or self._extract_operand_subject(qf.get("raw_text", ""), ""),
                            "raw_text": qf.get("raw_text", "")[:200],
                            "source_document": "",
                            "source_priority": "schedule_disclosed",
                        })
                # Also catch revenue quants with entity-specific subject_id
                elif subj_type == "revenue" and subj_id and subj_id.lower() not in ("company", "total", "consolidated", ""):
                    amt = qf.get("amount_value") or 0
                    val_m = amt / 1_000_000.0 if amt > 1000 else amt
                    if val_m > 0.5:
                        parsed_operands.append({
                            "operand_role": "actual_counterparty_ttm_revenue",
                            "value_millions": val_m,
                            "subject_label": subj_id,
                            "raw_text": qf.get("raw_text", "")[:200],
                            "source_document": "",
                            "source_priority": "body_stated",
                        })
        except Exception:
            pass

        # Fallback: scan accumulated facts text for schedule-disclosed revenue
        # patterns. The LLM's numeric_fact metadata is unreliable, but the
        # fact text itself usually contains recognizable patterns.
        if facts:
            _rev_pattern = _re_engine.compile(
                r"(?:schedule|disclosure|attributable|ttm|trailing)"
                r".*?(?:revenue|sales).*?"
                r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*"
                r"(billion|bn|million|mm|m|thousand|k)?",
                _re_engine.IGNORECASE,
            )
            _subj_pattern = _re_engine.compile(
                r"(?:attributable to|for|from)\s+(?:the\s+)?([A-Z][A-Za-z\s]+?)(?:\s+(?:MSA|agreement|supply|contract|msa|Agreement)|\s*[,;:\(]|\s*$)",
            )
            for fact in facts:
                fl = fact.lower()
                if not ("schedule" in fl or "attributable" in fl or "ttm" in fl or "trailing" in fl):
                    continue
                if "revenue" not in fl and "sales" not in fl:
                    continue
                amounts = self._money_amounts_in_millions(fact)
                if not amounts:
                    continue
                subj_match = _subj_pattern.search(fact)
                subj_label = subj_match.group(1).strip() if subj_match else ""
                if not subj_label or subj_label.lower() in ("company", "total", "consolidated", "the"):
                    continue
                for amt in amounts[:1]:
                    if amt > 0.5:
                        parsed_operands.append({
                            "operand_role": "actual_counterparty_ttm_revenue",
                            "value_millions": amt,
                            "subject_label": subj_label.lower(),
                            "raw_text": fact[:200],
                            "source_document": "",
                            "source_priority": "schedule_disclosed",
                        })

        # Find company total TTM revenue (denominator)
        company_ttm: "Optional[float]" = None
        for op in parsed_operands:
            if op.get("operand_role") == "company_total_ttm_revenue":
                v = op.get("value_millions")
                if v and v > 0:
                    company_ttm = v
                    break

        # If not found in operands, try quant store
        if company_ttm is None:
            try:
                for qf in self._matter_model.quant.get_amounts(min_value=100.0):
                    raw = (qf.get("raw_text") or "").lower()
                    subj = (qf.get("subject_type") or "").lower()
                    # Exclude counterparty-specific revenue (has "attributable",
                    # entity names, "minimum", etc.) to avoid false company_ttm
                    if "attributable" in raw or "minimum" in raw or "commitment" in raw:
                        continue
                    if ("total" in raw or "company" in raw or "consolidated" in raw
                            or "estimated" in raw or "reported" in raw or "borrower" in raw) and "revenue" in raw:
                        val = qf.get("amount_value") or 0
                        val_m = val / 1_000_000.0 if val > 1000 else val
                        if val_m > 50:
                            company_ttm = val_m
                            break
            except Exception:
                pass
        # Facts-text fallback for company TTM
        if company_ttm is None and facts:
            _co_ttm_pattern = _re_engine.compile(
                r"(?:company|total|consolidated|aggregate|estimated|reported|borrower)\s+"
                r"(?:ttm|trailing|annual)?\s*"
                r"(?:twelve[- ]month\s+)?revenue.*?"
                r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*"
                r"(billion|bn|million|mm|m)?",
                _re_engine.IGNORECASE,
            )
            for fact in facts:
                fl = fact.lower()
                if "revenue" not in fl:
                    continue
                # Skip counterparty-specific lines
                if "attributable" in fl or "minimum" in fl or "commitment" in fl:
                    continue
                if not any(k in fl for k in (
                    "total", "company", "consolidated", "aggregate",
                    "estimated", "reported", "borrower",
                )):
                    continue
                m = _co_ttm_pattern.search(fact)
                if m:
                    amounts = self._money_amounts_in_millions(fact)
                    if amounts and amounts[0] > 50:
                        company_ttm = amounts[0]
                        break

        # Resolve product-line revenue through dependency graph.
        # If a product_dependency links component->host_product with revenue,
        # and a license contract references that component, the product-line
        # revenue becomes available as a numerator for that license's exposure.
        try:
            dep_records = te.list_by_kind("product_dependency", limit=50)
            for dep_rec in dep_records:
                dp = dep_rec.get("payload_json")
                if isinstance(dp, str):
                    import json as _json_mod
                    try:
                        dp = _json_mod.loads(dp)
                    except Exception:
                        continue
                if not isinstance(dp, dict):
                    continue
                rev_attr = dp.get("revenue_attribution") or ""
                host = (dp.get("host_product") or "").lower().strip()
                component = (dp.get("component") or "").lower().strip()
                src_doc = (dp.get("source_document") or "").lower().strip()
                if not rev_attr or not (host and component):
                    continue
                rev_val = self._extract_operand_value_millions(str(rev_attr))
                if not rev_val:
                    continue
                for contract_key, entry in contract_registry.items():
                    cp = entry["payload"]
                    cname = (cp.get("contract_name") or "").lower()
                    fname = (cp.get("filename") or "").lower()
                    matches_component = len(component) > 2 and (component in cname or component in fname)
                    matches_source = src_doc and (src_doc == fname or src_doc in contract_key)
                    if matches_component or matches_source:
                        subject_label = (cp.get("counterparty") or host or component).lower().strip()
                        already_exists = any(
                            o.get("operand_role") == "actual_product_line_ttm_revenue"
                            and o.get("value_millions") == rev_val
                            for o in parsed_operands
                        )
                        if already_exists:
                            break
                        parsed_operands.append({
                            "operand_role": "actual_product_line_ttm_revenue",
                            "value_millions": rev_val,
                            "subject_label": subject_label,
                            "raw_text": f"product-line revenue for {host} via {component}: {rev_attr}"[:500],
                            "source_document": src_doc,
                            "source_priority": "dependency_resolved",
                        })
                        break
        except Exception:
            pass

        # Compute revenue exposure for each counterparty
        # Group operands by subject_label and pick preferred operand per subject.
        # ONLY emit a calculation when we have actual_counterparty_ttm_revenue —
        # minimum_purchase_commitment alone is insufficient for accurate revenue
        # exposure and would produce wrong results (the LLM does better without
        # a wrong deterministic answer competing with correct free-form analysis).
        subject_operands: "dict[str, list[dict]]" = {}
        for op in parsed_operands:
            role = op.get("operand_role", "")
            if role in ("actual_counterparty_ttm_revenue", "actual_product_line_ttm_revenue",
                        "minimum_purchase_commitment"):
                subject = (op.get("subject_label") or "").lower().strip()
                if subject and subject not in ("", "null", "company", "total"):
                    subject_operands.setdefault(subject, []).append(op)

        for subject, ops in subject_operands.items():
            if not company_ttm:
                break
            # Only proceed if we have a high-confidence actual TTM operand
            ttm_ops = [o for o in ops if o.get("operand_role") in (
                "actual_counterparty_ttm_revenue", "actual_product_line_ttm_revenue")]
            if not ttm_ops:
                continue
            best = ttm_ops[0]
            value = best.get("value_millions")
            if not value or value <= 0 or company_ttm <= 0:
                continue
            # Sanity: numerator must be < denominator for revenue exposure
            if value >= company_ttm:
                continue
            pct = value / company_ttm * 100.0
            # Check if a minimum_commitment also exists for comparison
            min_ops = [o for o in ops if o.get("operand_role") == "minimum_purchase_commitment"]
            rejected_note = ""
            if min_ops and min_ops[0].get("value_millions"):
                rej_val = min_ops[0].get("value_millions")
                rejected_note = (
                    f" (used actual TTM {self._format_millions(value)}, "
                    f"NOT minimum commitment {self._format_millions(rej_val)})"
                )
            matched_contract = ""
            for reg_key, reg_entry in contract_registry.items():
                if subject in reg_key or reg_key in subject:
                    matched_contract = reg_entry["payload"].get("contract_name", "")
                    break
            contract_label = matched_contract or subject.title()
            results.append(
                f"[GRAPH-CALC] Revenue exposure for {contract_label}: "
                f"{self._format_millions(value)} / {self._format_millions(company_ttm)} "
                f"= {pct:.1f}%{rejected_note}"
            )
            try:
                calc_key = f"calc:revenue_exposure:{subject}"
                te.upsert(
                    "calculation_result", calc_key,
                    payload={
                        "calculation_type": "revenue_exposure_percent",
                        "result_value": round(pct, 2),
                        "numerator_value": value,
                        "denominator_value": company_ttm,
                        "numerator_role": best.get("operand_role"),
                        "subject_label": subject,
                        "contract_name": contract_label,
                    },
                    label=f"revenue_exposure:{subject}",
                    confidence=0.95,
                )
            except Exception:
                pass

        # Compute credit exposure (drawn vs commitment)
        drawn_ops = [o for o in parsed_operands if o.get("operand_role") == "drawn_outstanding"]
        commit_ops = [o for o in parsed_operands if o.get("operand_role") == "facility_commitment"]
        if drawn_ops and commit_ops:
            drawn_val = drawn_ops[0].get("value_millions")
            commit_val = commit_ops[0].get("value_millions")
            if drawn_val and commit_val and drawn_val != commit_val:
                results.append(
                    f"[GRAPH-CALC] Credit facility exposure: drawn/outstanding "
                    f"{self._format_millions(drawn_val)} on "
                    f"{self._format_millions(commit_val)} commitment. "
                    f"Mandatory prepayment exposure is {self._format_millions(drawn_val)}, "
                    f"not the {self._format_millions(commit_val)} facility maximum."
                )

        # Threshold tests from carve-out provisions.
        # Carve-outs may reference acquirer/parent revenue thresholds.
        # If the required subject's operand isn't in the documents,
        # emit blocked_missing_operand instead of substituting a wrong value.
        try:
            carve_out_provisions = te.list_by_kind("contract_provision", limit=50)
            _threshold_keywords = ("threshold", "exceed", "greater than", "less than",
                                   "more than", "not less than", "revenue of", "annual revenue")
            _acquirer_keywords = ("acquirer", "buyer", "purchasing party", "parent",
                                  "successor", "ultimate parent")
            for prov in carve_out_provisions:
                pp = prov.get("payload_json")
                if isinstance(pp, str):
                    import json as _json_mod
                    try:
                        pp = _json_mod.loads(pp)
                    except Exception:
                        continue
                if not isinstance(pp, dict):
                    continue
                if pp.get("provision_kind") != "carve_out":
                    continue
                raw = (pp.get("raw_text") or "").lower()
                if not any(k in raw for k in _threshold_keywords):
                    continue
                requires_acquirer = any(k in raw for k in _acquirer_keywords)
                if not requires_acquirer:
                    continue
                threshold_amounts = self._money_amounts_in_millions(pp.get("raw_text") or "")
                if not threshold_amounts:
                    continue
                threshold_val = threshold_amounts[0]
                contract_name_prov = pp.get("contract_name") or ""
                negated_less = any(k in raw for k in (
                    "not less than", "no less than", "at least",
                ))
                if negated_less:
                    use_less_than = False
                else:
                    use_less_than = any(
                        k in raw for k in (
                            "less than", "does not exceed", "not greater than",
                            "not more than", "below", "under", "fewer than",
                        )
                    )
                comparator_label = "<=" if use_less_than else ">="
                acquirer_ops = [
                    o for o in parsed_operands
                    if o.get("operand_role") in ("acquirer_revenue", "acquirer_segment_revenue")
                ]
                if acquirer_ops:
                    obs_val = acquirer_ops[0].get("value_millions")
                    if obs_val:
                        if use_less_than:
                            test_result = "PASS" if obs_val <= threshold_val else "FAIL"
                        else:
                            test_result = "PASS" if obs_val >= threshold_val else "FAIL"
                        results.append(
                            f"[THRESHOLD_TEST] {contract_name_prov} carve-out: acquirer revenue "
                            f"{self._format_millions(obs_val)} {comparator_label} threshold "
                            f"{self._format_millions(threshold_val)} — {test_result}"
                        )
                else:
                    results.append(
                        f"[BLOCKED_CALC] {contract_name_prov} carve-out threshold test: "
                        f"threshold is {self._format_millions(threshold_val)} but acquirer/parent "
                        f"revenue is NOT available in the reviewed documents. Cannot determine "
                        f"whether the carve-out applies. This is a missing-input gap."
                    )
                    try:
                        te.upsert(
                            "calculation_result",
                            f"calc:threshold_blocked:{contract_name_prov}",
                            payload={
                                "calculation_type": "threshold_test",
                                "status": "blocked_missing_operand",
                                "threshold_value": threshold_val,
                                "required_subject_role": "acquirer_revenue",
                                "contract_name": contract_name_prov,
                                "carve_out_text": (pp.get("raw_text") or "")[:300],
                            },
                            label=f"blocked_threshold:{contract_name_prov}",
                            confidence=0.9,
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        # Resolve downstream risk actor bindings.
        # Use transaction_context to bind generic roles (acquirer/target) to
        # actual entity names from the documents.
        try:
            actor_names: dict[str, str] = {}
            txn_records = te.list_by_kind("transaction_context", limit=10)
            for txn_rec in txn_records:
                tp = txn_rec.get("payload_json")
                if isinstance(tp, str):
                    import json as _json_mod
                    try:
                        tp = _json_mod.loads(tp)
                    except Exception:
                        continue
                if not isinstance(tp, dict):
                    continue
                for role_key in ("target", "acquirer", "merger_sub", "parent"):
                    name = (tp.get(role_key) or "").strip()
                    if name and role_key not in actor_names:
                        actor_names[role_key] = name

            all_provisions = te.list_by_kind("contract_provision", limit=50)
            for prov in all_provisions:
                pp = prov.get("payload_json")
                if isinstance(pp, str):
                    import json as _json_mod
                    try:
                        pp = _json_mod.loads(pp)
                    except Exception:
                        continue
                if not isinstance(pp, dict):
                    continue
                if pp.get("provision_kind") != "downstream_risk":
                    continue
                actor_role = pp.get("affected_actor_role") or "acquirer"
                actor_name = actor_names.get(actor_role, actor_role)
                contract_name_dr = pp.get("contract_name") or ""
                scenario = pp.get("re_trigger_scenario") or ""
                section = pp.get("provision_section") or ""
                if scenario:
                    results.append(
                        f"[DOWNSTREAM_RISK] {contract_name_dr}"
                        + (f" ({section})" if section else "")
                        + f": future ownership change of {actor_name} ({actor_role}) "
                        + f"could re-trigger this provision. {scenario[:200]}"
                    )
        except Exception:
            pass

        return results

    def _build_provision_comparison_summary(
        self, state: "InvestigationState",
    ) -> str:
        """Build a structured provision comparison table for comparison tasks.

        Aggregates provision_comparison typed evidence into a before/after
        table that synthesis can use directly for the deviation analysis.
        """
        _ql = (getattr(state, "query", "") or "").lower()
        _is_comp = any(w in _ql for w in (
            "markup", "redline", "compare", "comparison", "deviation",
            "counterparty", "credit facility", "credit agreement",
            "term sheet", "loan agreement", "commitment letter",
        ))
        if not _is_comp or self._matter_model is None:
            return ""
        try:
            rows = self._matter_model.typed_evidence.list_by_kind(
                "provision_comparison", limit=200,
            )
        except Exception:
            return ""
        if not rows:
            return ""
        by_provision: dict[str, dict[str, list[str]]] = {}
        for row in rows:
            payload = row.get("payload_json")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            prov = payload.get("provision", "")
            role = payload.get("source_role", "unknown")
            val = payload.get("value", "")
            sec = payload.get("section_ref", "")
            if not prov or not val:
                continue
            if prov not in by_provision:
                by_provision[prov] = {}
            entry = val
            if sec:
                entry += f" ({sec})"
            by_provision[prov].setdefault(role, []).append(entry)
        if not by_provision:
            return ""
        all_roles = sorted({r for roles in by_provision.values() for r in roles})
        if not all_roles:
            return ""
        header = "| Provision | " + " | ".join(r.replace("_", " ").title() for r in all_roles) + " |"
        sep = "|-----------|" + "|".join("-" * max(8, len(r) + 2) for r in all_roles) + "|"
        lines = [
            "PROVISION COMPARISON DATA (extracted from documents — use for deviation table):",
            header,
            sep,
        ]
        for prov, roles in sorted(by_provision.items()):
            cols = " | ".join("; ".join(roles.get(r, ["—"])) for r in all_roles)
            lines.append(f"| {prov} | {cols} |")
        lines.append("")
        lines.append(
            "Use this table as the BASIS for your deviation analysis. "
            "For each row where values differ between columns, produce a deviation finding "
            "with exact values, risk rating, dollar impact calculation, and recommendation."
        )
        return "\n".join(lines)

    def _build_regulatory_data_summary(
        self, state: "InvestigationState",
    ) -> str:
        """Build a structured regulatory evidence summary for antitrust/regulatory tasks.

        Aggregates regulatory_data typed evidence into organized categories
        that synthesis can use directly for risk assessment and strategy memos.
        """
        _ql = (getattr(state, "query", "") or "").lower()
        _is_reg = any(w in _ql for w in (
            "antitrust", "hsr", "merger review", "regulatory", "compliance",
            "market share", "hhi", "competitive effects",
        ))
        if not _is_reg or self._matter_model is None:
            return ""
        try:
            rows = self._matter_model.typed_evidence.list_by_kind(
                "regulatory_data", limit=300,
            )
        except Exception:
            return ""
        if not rows:
            return ""

        by_category: dict[str, list[dict]] = {}
        for row in rows:
            payload = row.get("payload_json")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            cat = (payload.get("category") or "unknown").lower()
            by_category.setdefault(cat, []).append(payload)

        if not by_category:
            return ""

        lines = ["REGULATORY EVIDENCE SUMMARY (extracted from documents):"]
        cat_order = ["market_share", "hhi", "hot_doc", "barrier",
                     "remedy", "timeline", "jurisdiction", "overlap",
                     "synergy", "accretion", "valuation", "framework",
                     "defense"]
        seen_cats = set()
        for cat in cat_order:
            if cat not in by_category:
                continue
            seen_cats.add(cat)
            items = by_category[cat]
            lines.append(f"\n### {cat.upper().replace('_', ' ')} ({len(items)} entries)")
            dedup: set[str] = set()
            for item in items:
                entity = item.get("entity", "")
                value = item.get("value", "")
                src = item.get("source_detail", "")
                key = f"{entity}:{value}"
                if key in dedup:
                    continue
                dedup.add(key)
                entry = f"- {entity}: {value}"
                if src:
                    entry += f" [{src}]"
                lines.append(entry)

        for cat, items in by_category.items():
            if cat in seen_cats:
                continue
            lines.append(f"\n### {cat.upper().replace('_', ' ')} ({len(items)} entries)")
            for item in items[:10]:
                entity = item.get("entity", "")
                value = item.get("value", "")
                lines.append(f"- {entity}: {value}")

        lines.append("")
        lines.append(
            "Use this evidence for your analysis. Cite specific data points "
            "with source references. Compute HHI where market shares are available."
        )
        return "\n".join(lines)

    def _build_adverse_evidence_summary(self, state: "InvestigationState") -> str:
        """Build adverse evidence table for synthesis context."""
        if self._matter_model is None:
            return ""
        try:
            rows = self._matter_model.typed_evidence.list_by_kind(
                "adverse_evidence", limit=50,
            )
        except Exception:
            return ""
        if not rows:
            return ""
        lines = [
            "ADVERSE EVIDENCE / HOT DOCUMENTS (flag these in your analysis):",
        ]
        for row in rows:
            payload = row.get("payload_json")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            quote = payload.get("quote", "")
            speaker = payload.get("speaker", "")
            sec = payload.get("section_ref", "")
            theory = payload.get("adverse_theory", "")
            doc_name = payload.get("source_document", "")
            if not quote:
                continue
            entry = f'- "{quote}"'
            if speaker:
                entry += f" — {speaker}"
            if doc_name:
                entry += f" [{doc_name}]"
            if sec:
                entry += f" ({sec})"
            if theory:
                entry += f" → {theory}"
            lines.append(entry)
        if len(lines) <= 1:
            return ""
        lines.append("")
        lines.append(
            "You MUST address each adverse item above in your analysis: "
            "identify the risk it creates, recommend how to handle it, "
            "and flag it prominently."
        )
        return "\n".join(lines)

    def _build_material_contract_coverage_section(
        self, state: "InvestigationState",
    ) -> str:
        """Build a mandatory checklist of all documents read, requiring synthesis to address each."""
        if not self._is_extraction_task(getattr(state, "query", "")):
            return ""
        doc_names: list[str] = []
        seen: set[str] = set()
        for c in getattr(state, "citations", []):
            name = getattr(c, "document", "") or ""
            if name and name.lower() not in seen:
                seen.add(name.lower())
                doc_names.append(name)
        if not doc_names:
            return ""
        lines = [
            "MATERIAL CONTRACT COVERAGE CHECKLIST (MANDATORY):",
            "The following documents were read during investigation.",
            "Your report MUST include a dedicated section for EACH document below.",
            "If a document has no relevant provisions, explicitly state that.",
            "",
        ]
        for i, name in enumerate(doc_names, 1):
            lines.append(
                f"  {i}. {name} — MUST address: provisions found, section numbers, "
                "triggers/thresholds, consent requirements, financial exposure, "
                "risk rating, and missing expected provisions"
            )
        lines.append(
            "\nDo NOT omit any document from the list above. Every document must appear "
            "as a named section in your report."
        )
        return "\n".join(lines)

    def _build_workflow_quality_section(self, state: InvestigationState) -> str:
        """Describe the active output contract for synthesis and repair."""
        if state.run_objective is None or state.working_set is None:
            self._initialize_workflow_state(state)
        objective = state.run_objective
        if objective is None:
            return ""

        lines = [
            "Workflow Quality Contract (mandatory):",
            f"- Workflow kind: {objective.workflow_kind}",
            f"- Output shape: {objective.output_shape}",
            f"- User goal: {objective.user_goal}",
            f"- Audience: {objective.audience}",
            f"- Policy audience: {objective.policy_audience}",
        ]
        output_contract = dict(
            getattr(getattr(state, "execution_contract", None), "output_contract", {})
            or {}
        )
        task_spec = dict(output_contract.get("task_spec") or {})
        if task_spec:
            lines.append("- Task ontology:")
            lines.append(f"  - task_type: {task_spec.get('task_type')}")
            lines.append(f"  - operation: {task_spec.get('operation')}")
            lines.append(f"  - answer_shape: {task_spec.get('answer_shape')}")
            required = task_spec.get("required_evidence") or []
            if required:
                lines.append(
                    "  - required_evidence: "
                    + ", ".join(str(item) for item in required[:10])
                )
            if task_spec.get("fresh_extraction_required"):
                lines.append(
                    "  - cached matter summaries are not sufficient by themselves"
                )
            if task_spec.get("operation") == "verify_absence":
                lines.append(
                    "  - absence vocabulary: searched_not_found, "
                    "false_premise_likely, out_of_matter, source_missing"
                )
        if objective.success_criteria:
            lines.append("- Success criteria:")
            for criterion in objective.success_criteria[:8]:
                lines.append(f"  - {criterion}")
        if objective.constraints:
            lines.append("- Constraints:")
            for constraint in objective.constraints[:8]:
                lines.append(f"  - {constraint}")

        if state.workflow_obligations:
            lines.append("- Obligations:")
            for obligation in state.workflow_obligations[:12]:
                validator = obligation.validator or obligation.obligation_type
                flags = []
                flags.append("required" if obligation.required else "optional")
                flags.append("blocking" if obligation.blocking else "advisory")
                lines.append(
                    f"  - [{validator}] {obligation.description} "
                    f"({', '.join(flags)})"
                )

        directives = self._workflow_validator_directives(state.workflow_obligations)
        if directives:
            lines.append("- Validator directives:")
            for directive in directives:
                lines.append(f"  - {directive}")

        working_set = state.working_set
        if working_set is not None:
            counts = [
                ("verified assertions", len(working_set.verified_assertion_ids)),
                ("candidate assertions", len(working_set.candidate_assertion_ids)),
                ("issues", len(working_set.issue_ids)),
                ("gaps", len(working_set.gap_ids)),
                ("documents", len(working_set.document_ids)),
                ("authorities", len(working_set.authority_ids)),
                ("assumptions", len(working_set.assumption_ids)),
            ]
            if any(count for _, count in counts) or working_set.dependency_manifest_hash:
                lines.append("- Working set:")
                for label, count in counts:
                    if count:
                        lines.append(f"  - {label}: {count}")
                if working_set.dependency_manifest_hash:
                    lines.append(
                        "  - dependency manifest hash: "
                        f"{working_set.dependency_manifest_hash}"
                    )

        return "\n".join(lines)

    @staticmethod
    def _workflow_validator_directives(
        obligations: list[Obligation],
    ) -> list[str]:
        validators = {
            item.validator or item.obligation_type
            for item in obligations
            if item.validator or item.obligation_type
        }
        directives: list[str] = []
        if "citation_floor" in validators:
            directives.append(
                "Material factual claims need record support; if support is "
                "insufficient, disclose the limitation instead of inventing "
                "citations."
            )
        if "gap_disclosure" in validators:
            directives.append(
                "Visible proof gaps, missing inputs, and unknowns must be "
                "disclosed rather than smoothed over."
            )
        if "draft_template" in validators:
            directives.append(
                "Use the selected work-product structure with clear sections, "
                "tables, or lists where appropriate."
            )
        if "assumption_labeling" in validators:
            directives.append(
                "Separate temporary assumptions from established matter facts."
            )
        if "human_review_required" in validators:
            directives.append(
                "Do not claim a draft is ready for filing, service, or external "
                "use before human review passes."
            )
        if "read_answerability" in validators:
            directives.append(
                "Answer only from existing eligible matter state unless the "
                "contract permits fresh extraction."
            )
        if "task_evidence_contract" in validators:
            directives.append(
                "Do the typed task requested; do not substitute a run delta, "
                "state table, generic memo, or unrelated diagnostic for the "
                "required answer shape."
            )
        if "required_evidence_objects" in validators:
            directives.append(
                "For each required evidence object, either use source-grounded "
                "support or explicitly label the object as not found after "
                "search; do not call all absent evidence missing documents."
            )
        if "absence_status" in validators:
            directives.append(
                "For premise checks, distinguish source_missing from "
                "searched_not_found, false_premise_likely, and out_of_matter."
            )
        if "trace_output_alignment" in validators:
            directives.append(
                "If required evidence was found during the run, do not claim "
                "the relevant document or object was unavailable in the final "
                "answer."
            )
        return directives

    def _emit_output(
        self,
        state: InvestigationState,
        output_text: str,
        *,
        emitter: str,
        dependency_manifest_hash: Optional[str] = None,
    ) -> OutputEnvelope:
        """Central output wrapper for workflow-aware user-facing text.

        The current service/UI still reads findings["final_output"], so
        this keeps that compatibility while also attaching an auditable
        envelope and validation results for the next UI/API layer.
        """
        if state.run_objective is None or state.working_set is None:
            self._initialize_workflow_state(state)
        validation_results = self._validate_workflow_output(
            state,
            output_text,
            emitter=emitter,
        )
        if validation_results:
            state.validation_results.extend(validation_results)
            self._apply_validation_results_to_obligations(
                state.workflow_obligations,
                validation_results,
            )

        objective = state.run_objective
        working_set = state.working_set
        output_envelope = OutputEnvelope.create(
            output_text=output_text,
            workflow_kind=(
                objective.workflow_kind if objective else WorkflowKind.ANALYSIS.value
            ),
            output_shape=objective.output_shape if objective else "answer",
            emitter=emitter,
            objective_id=objective.id if objective else None,
            dependency_manifest_hash=(
                dependency_manifest_hash
                or (working_set.dependency_manifest_hash if working_set else None)
            ),
            validation_results=validation_results,
            review_required=any(
                item.validator == "human_review_required"
                for item in state.workflow_obligations
            ),
        )
        state.output_envelope = output_envelope
        state.findings["output_envelope"] = output_envelope.to_dict()
        state.findings["final_output"] = output_text
        return output_envelope

    def _validate_workflow_output(
        self,
        state: InvestigationState,
        output_text: str,
        *,
        emitter: str,
    ) -> list[ValidationResult]:
        """Run first-pass structural validators against workflow obligations."""
        results: list[ValidationResult] = []
        contract = getattr(state, "execution_contract", None)
        output_contract = dict(getattr(contract, "output_contract", {}) or {})
        citation_floor = max(0, int(getattr(contract, "citation_floor", 0) or 0))
        if output_contract.get("requires_citations"):
            citation_floor = max(citation_floor, 1)

        seen_validators: set[str] = set()
        for obligation in state.workflow_obligations:
            validator = obligation.validator or obligation.obligation_type
            if validator in seen_validators:
                continue
            seen_validators.add(validator)
            result = self._run_workflow_validator(
                validator,
                state,
                output_text,
                citation_floor=citation_floor,
                emitter=emitter,
            )
            if result is not None:
                results.append(result)
        return results

    def _run_workflow_validator(
        self,
        validator: str,
        state: InvestigationState,
        output_text: str,
        *,
        citation_floor: int,
        emitter: str,
    ) -> Optional[ValidationResult]:
        matching_obligation_ids = [
            item.id
            for item in state.workflow_obligations
            if (item.validator or item.obligation_type) == validator
        ]

        if validator == "citation_floor":
            citations = [
                c for c in state.citations
                if getattr(c, "document", None)
            ]
            probe_citations = [
                c for c in (state.findings.get("sufficiency_probe_citations") or [])
                if isinstance(c, str) and c.strip()
            ]
            support_count = max(len(citations), len(probe_citations))
            passed = support_count >= citation_floor
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=1.0 if passed else 0.0,
                blocking_issues=[] if passed else [
                    f"citation support {support_count} < floor {citation_floor}"
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        if validator == "gap_disclosure":
            gap_count = self._open_gap_count()
            if gap_count <= 0:
                passed = True
            else:
                lowered = output_text.lower()
                passed = any(
                    marker in lowered
                    for marker in ("gap", "missing", "unresolved", "unknown")
                )
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=1.0 if passed else 0.0,
                blocking_issues=[] if passed else [
                    f"{gap_count} open gap(s) were not visibly disclosed"
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        if validator == "human_review_required":
            return ValidationResult(
                validator=validator,
                passed=False,
                score=0.0,
                warnings=["human review required before external use"],
                obligation_status={
                    oid: False for oid in matching_obligation_ids
                },
            )

        if validator == "draft_template":
            has_structure = any(marker in output_text for marker in ("#", "|", "\n- "))
            return ValidationResult(
                validator=validator,
                passed=bool(output_text.strip()) and has_structure,
                score=1.0 if output_text.strip() and has_structure else 0.4,
                warnings=[] if has_structure else [
                    "draft output has no visible section/table/list structure"
                ],
                obligation_status={
                    oid: bool(output_text.strip()) and has_structure
                    for oid in matching_obligation_ids
                },
            )

        if validator == "assumption_labeling":
            lowered = output_text.lower()
            mentions_assumption = "assum" in lowered
            return ValidationResult(
                validator=validator,
                passed=mentions_assumption,
                score=1.0 if mentions_assumption else 0.0,
                blocking_issues=[] if mentions_assumption else [
                    "temporary assumptions were not explicitly labeled"
                ],
                obligation_status={
                    oid: mentions_assumption for oid in matching_obligation_ids
                },
            )

        if validator == "task_evidence_contract":
            output_contract = dict(
                getattr(getattr(state, "execution_contract", None), "output_contract", {})
                or {}
            )
            task_spec = dict(output_contract.get("task_spec") or {})
            task_type = str(task_spec.get("task_type") or "")
            lowered = output_text.lower()
            forbidden_markers = []
            if task_type == "document_comparison":
                forbidden_markers.extend(("## what changed", "assertion delta"))
            if task_type in {
                "defined_term_inventory",
                "cross_document_reference_search",
                "signatory_extraction",
                "matter_subject_identification",
                "redaction_categorization",
                "deposition_extraction",
                "procedural_history",
            }:
                forbidden_markers.extend(("## list documents", "## list actors"))
            if task_type == "redaction_categorization":
                forbidden_markers.extend((
                    "redacted information because the operative text",
                    "redaction markers are completely absent",
                    "once the operative text is loaded",
                ))
            if task_type == "quantitative_reconciliation":
                forbidden_markers.extend((
                    "run diagnostics & safeguards",
                    "auto-generated by so-6",
                    "payment reconciliation (usd): invoiced $0.00",
                ))
            if task_type == "multi_document_synthesis":
                forbidden_markers.extend((
                    "has not been provided",
                    "currently absent from the provided facts",
                    "zero verified or candidate text",
                ))
            if task_type in {"premise_check", "out_of_matter_check"}:
                forbidden_markers.extend((
                    "once the operative text is loaded",
                    "until the operative text is loaded",
                ))
            violated = [m for m in forbidden_markers if m in lowered]
            passed = bool(output_text.strip()) and not violated
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=1.0 if passed else 0.0,
                blocking_issues=[] if passed else [
                    "output appears to substitute the wrong task artifact: "
                    + ", ".join(violated or ["empty output"])
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        if validator == "required_evidence_objects":
            lowered = output_text.lower()
            has_source_or_status = bool(state.citations) or any(
                marker in lowered
                for marker in (
                    "not found",
                    "searched_not_found",
                    "false_premise_likely",
                    "out_of_matter",
                    "source_missing",
                    "unverified",
                    "verified",
                    "citation",
                    "section",
                    "source",
                )
            )
            return ValidationResult(
                validator=validator,
                passed=bool(output_text.strip()) and has_source_or_status,
                score=1.0 if output_text.strip() and has_source_or_status else 0.0,
                blocking_issues=[] if has_source_or_status else [
                    "required evidence objects were neither used nor given an "
                    "explicit absence/status label"
                ],
                obligation_status={
                    oid: bool(output_text.strip()) and has_source_or_status
                    for oid in matching_obligation_ids
                },
            )

        if validator == "absence_status":
            lowered = output_text.lower().replace("-", "_")
            statuses = (
                "not_searched",
                "searched_not_found",
                "found_unverified",
                "found_verified",
                "conflicting_evidence",
                "out_of_matter",
                "false_premise_likely",
                "source_missing",
            )
            has_status = any(status in lowered for status in statuses)
            return ValidationResult(
                validator=validator,
                passed=has_status,
                score=1.0 if has_status else 0.0,
                blocking_issues=[] if has_status else [
                    "absence check did not label the result with a typed "
                    "absence status"
                ],
                obligation_status={
                    oid: has_status for oid in matching_obligation_ids
                },
            )

        if validator == "trace_output_alignment":
            manifest = state.findings.get("task_evidence_manifest") or {}
            found_targets = []
            if isinstance(manifest, dict):
                found_targets = [
                    str(item).strip().lower()
                    for item in (
                        manifest.get("found_targets")
                        or manifest.get("found_objects")
                        or ()
                    )
                    if str(item).strip()
                ]
            lowered = output_text.lower()
            denial_markers = (
                "has not been provided",
                "not provided",
                "not available",
                "currently absent",
                "zero verified or candidate text",
                "cannot see",
                "cannot access",
            )
            has_denial = any(marker in lowered for marker in denial_markers)
            contradicted_targets = [
                target for target in found_targets
                if target in lowered and has_denial
            ]
            passed = not contradicted_targets
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=1.0 if passed else 0.0,
                blocking_issues=[] if passed else [
                    "final answer contradicts evidence manifest for: "
                    + ", ".join(contradicted_targets[:5])
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        if validator == "comparison_min_deviations":
            import re
            lowered = output_text.lower()
            table_rows = len(re.findall(r"^\s*\|.*\|.*\|", output_text, re.MULTILINE))
            bullet_deviations = len(re.findall(
                r"(?:original|changed|current|proposed|deviation|difference)",
                lowered,
            ))
            deviation_count = max(table_rows, bullet_deviations // 2)
            passed = deviation_count >= 10
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=min(1.0, deviation_count / 10.0),
                blocking_issues=[] if passed else [
                    f"document comparison found only ~{deviation_count} deviations "
                    f"(minimum 10 required)"
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        if validator == "risk_rating_per_issue":
            import re
            lowered = output_text.lower()
            ratings_found = len(re.findall(
                r"\b(red|yellow|green|high\s*risk|medium\s*risk|low\s*risk)\b",
                lowered,
            ))
            issues_mentioned = len(re.findall(r"^#{1,3}\s+", output_text, re.MULTILINE))
            passed = ratings_found >= max(1, issues_mentioned // 2)
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=min(1.0, ratings_found / max(1, issues_mentioned)) if issues_mentioned else 0.5,
                blocking_issues=[] if passed else [
                    f"found {ratings_found} risk ratings for ~{issues_mentioned} "
                    f"issue sections"
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        if validator == "issue_coverage_matrix":
            manifest = state.findings.get("task_evidence_manifest") or {}
            if isinstance(manifest, dict):
                covered_kinds = set(manifest.get("covered_evidence_kinds") or [])
                missing_kinds = set(manifest.get("missing_evidence_kinds") or [])
                contract_obj = getattr(state, "execution_contract", None)
                _oc = dict(getattr(contract_obj, "output_contract", {}) or {})
                _ts = dict(_oc.get("task_spec") or {})
                required_kinds = set(_ts.get("required_evidence") or [])
                if required_kinds:
                    addressed = required_kinds & (covered_kinds | missing_kinds)
                    total = max(1, len(required_kinds))
                    passed = len(addressed) >= total * 0.7
                    score = len(addressed) / total
                else:
                    passed = True
                    score = 1.0
            else:
                passed = bool(output_text.strip())
                score = 1.0 if passed else 0.0
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=score,
                blocking_issues=[] if passed else [
                    f"issue coverage: evidence manifest missing required kinds"
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        if validator == "numeric_operand_coverage":
            import re
            numbers = re.findall(r"\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%|\d{1,3}(?:,\d{3})+", output_text)
            calculations = len(re.findall(
                r"(?:=|equals|total|sum|difference|ratio|margin|spread|basis points|bps)",
                output_text.lower(),
            ))
            passed = len(numbers) >= 3 and calculations >= 1
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=min(1.0, (len(numbers) + calculations) / 10.0),
                blocking_issues=[] if passed else [
                    f"quantitative task has {len(numbers)} numbers and "
                    f"{calculations} calculation markers (need more)"
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        if validator == "target_document_coverage":
            contract = getattr(state, "execution_contract", None)
            output_contract = dict(getattr(contract, "output_contract", {}) or {})
            target_docs = list(output_contract.get("target_documents") or [])
            lowered = output_text.lower()
            covered = 0
            missing = []
            for doc in target_docs:
                doc_lower = str(doc).lower().strip()
                name_parts = doc_lower.replace("_", " ").replace("-", " ").split()
                if doc_lower in lowered or any(
                    part in lowered for part in name_parts if len(part) > 4
                ):
                    covered += 1
                else:
                    missing.append(str(doc))
            total = max(1, len(target_docs))
            passed = covered >= total * 0.8
            return ValidationResult(
                validator=validator,
                passed=passed,
                score=covered / total,
                blocking_issues=[] if passed else [
                    f"target document coverage: {covered}/{total} addressed, "
                    f"missing: {', '.join(missing[:5])}"
                ],
                obligation_status={
                    oid: passed for oid in matching_obligation_ids
                },
            )

        return ValidationResult(
            validator=validator,
            passed=bool(output_text.strip()),
            score=1.0 if output_text.strip() else 0.0,
            blocking_issues=[] if output_text.strip() else [
                f"{validator} produced no output"
            ],
            obligation_status={
                oid: bool(output_text.strip()) for oid in matching_obligation_ids
            },
        )

    @staticmethod
    def _apply_validation_results_to_obligations(
        obligations: list[Obligation],
        validation_results: list[ValidationResult],
    ) -> None:
        by_id: dict[str, tuple[bool, str]] = {}
        for result in validation_results:
            note = (
                "; ".join(result.blocking_issues or result.warnings)
                or f"{result.validator}: {'passed' if result.passed else 'failed'}"
            )
            for obligation_id, passed in result.obligation_status.items():
                by_id[obligation_id] = (bool(passed), note)
        for obligation in obligations:
            if obligation.id in by_id:
                obligation.satisfied, obligation.status_note = by_id[obligation.id]

    def _open_gap_count(self) -> int:
        if self._matter_model is None:
            return 0
        try:
            rows = self._matter_model.gaps.open_gaps(limit=1000)
            return len(rows)
        except Exception:
            return 0

    @staticmethod
    def _repairable_workflow_results(
        validation_results: list[ValidationResult],
    ) -> list[ValidationResult]:
        fixable_validators = {
            "gap_disclosure",
            "draft_template",
            "assumption_labeling",
            "comparison_min_deviations",
            "risk_rating_per_issue",
        }
        return [
            result
            for result in validation_results
            if result.validator in fixable_validators
            and not result.passed
            and (result.blocking_issues or result.warnings)
        ]

    @staticmethod
    def _workflow_validation_issue_count(
        validation_results: list[ValidationResult],
    ) -> int:
        return sum(
            len(result.blocking_issues) + len(result.warnings)
            for result in validation_results
        )

    @staticmethod
    def _format_workflow_validation_issues(
        validation_results: list[ValidationResult],
    ) -> str:
        lines: list[str] = []
        for result in validation_results:
            issues = list(result.blocking_issues or []) + list(result.warnings or [])
            if result.passed and not issues:
                continue
            status = "passed" if result.passed else "failed"
            if issues:
                lines.append(
                    f"- {result.validator} ({status}): {'; '.join(issues)}"
                )
            else:
                lines.append(f"- {result.validator} ({status})")
        return "\n".join(lines) or "- No validator findings."

    async def _repair_output_if_needed(
        self,
        state: InvestigationState,
        output_text: str,
        *,
        emitter: str,
    ) -> str:
        """Run one focused repair pass for fixable workflow-output failures."""
        if state.run_objective is None or state.working_set is None:
            self._initialize_workflow_state(state)

        validation_results = self._validate_workflow_output(
            state,
            output_text,
            emitter=emitter,
        )
        repairable = self._repairable_workflow_results(validation_results)
        if not repairable:
            return output_text

        prompt = WORKFLOW_OUTPUT_REPAIR_PROMPT.format(
            workflow_section=self._build_workflow_quality_section(state),
            validation_issues=self._format_workflow_validation_issues(
                validation_results
            ),
            output_text=output_text,
        )
        try:
            state.llm_calls_required += 1
            repaired = await self.client.complete(
                prompt,
                tier=ModelTier.FLASH,
                timeout=self.config.synthesis_fallback_timeout,
                usage_label=f"{emitter}_workflow_repair",
                conversation_history=state.conversation_history,
            )
        except Exception:
            return output_text

        if not isinstance(repaired, str) or not repaired.strip():
            return output_text
        repaired = repaired.strip()

        repaired_results = self._validate_workflow_output(
            state,
            repaired,
            emitter=emitter,
        )
        original_issue_count = self._workflow_validation_issue_count(repairable)
        repaired_issue_count = self._workflow_validation_issue_count(
            self._repairable_workflow_results(repaired_results)
        )
        if repaired_issue_count <= original_issue_count:
            return repaired
        return output_text

    async def _complete_synthesis_with_fallback(
        self,
        state: InvestigationState,
        prompt: str,
    ) -> str:
        """Run final synthesis on PRO, falling back to FLASH on timeout.

        A PRO timeout should degrade answer polish, not discard the entire
        investigation. The fallback keeps the same source packet but asks for a
        concise answer so the cheaper model can finish inside an interactive
        request window.
        """
        try:
            return await self.client.complete(
                prompt,
                tier=ModelTier.PRO,
                timeout=self.config.synthesis_pro_timeout,
                usage_label="synthesis",
                conversation_history=state.conversation_history,
            )
        except TimeoutError as exc:
            timeout_note = (
                f"PRO synthesis timed out after {self.config.synthesis_pro_timeout}s; "
                "retrying with FLASH fallback."
            )
            state.findings["synthesis_timeout_fallback"] = {
                "model_tier": ModelTier.PRO.value,
                "fallback_tier": ModelTier.FLASH.value,
                "timeout_seconds": self.config.synthesis_pro_timeout,
                "error": str(exc),
            }
            self._emit_step(state, StepType.SYNTHESIS, timeout_note)
            logger.warning(timeout_note)

            fallback_prompt = (
                prompt
                + "\n\nFALLBACK SYNTHESIS INSTRUCTION:\n"
                + "The PRO synthesis call timed out. Produce the best concise "
                + "source-grounded answer from the provided context. Do not add "
                + "new investigation claims; preserve any uncertainty and gaps."
            )
            state.llm_calls_required += 1
            return await self.client.complete(
                fallback_prompt,
                tier=ModelTier.FLASH,
                timeout=self.config.synthesis_fallback_timeout,
                usage_label="synthesis_timeout_fallback",
                conversation_history=state.conversation_history,
            )

    async def investigate(
        self,
        query: str,
        repository_path: str | Path,
        research_mode: "str | None" = None,
        conversation_history: "list[dict[str, str]] | None" = None,
        execution_contract: "Any | None" = None,
    ) -> InvestigationState:
        """
        Run full recursive investigation.

        Args:
            query: The question to investigate
            repository_path: Path to document repository

        Returns:
            InvestigationState with all findings, citations, thinking trace
        """
        from ..matter.runtime import MatterRuntimeAdapter, NullMatterAdapter

        repo = MatterRepository(repository_path)
        # Always store the resolved absolute path so state.repository_path is stable
        # regardless of CWD changes (e.g., FastAPI background tasks).
        state = InvestigationState.create(
            query,
            str(repo.base_path),
            research_mode=research_mode,
            conversation_history=conversation_history,
        )
        # MVI-3: attach the cascade ExecutionContract so the
        # termination controller reads family-scoped stop rules.
        state.execution_contract = execution_contract
        self._initialize_workflow_state(state)

        # Adapt configuration based on repository size.
        # _doc_count update triggers semaphore recreation in _get_semaphore() so
        # the concurrency limit stays calibrated without nulling out mid-flight waiters.
        stats = repo.get_stats()
        self._adapt_config_for_repo_size(stats.total_files)
        # Reset per-run filename cache so a new investigation always gets a fresh snapshot.
        self._known_filenames = None

        # Build matter adapter — real or null depending on config + injected model
        if self.config.enable_matter_model and self._matter_model is not None:
            run_id = self._matter_model.start_run(
                query,
                research_mode=state.research_mode,
            )
            matter_adapter = MatterRuntimeAdapter(self._matter_model, run_id)
        else:
            run_id = None
            matter_adapter = NullMatterAdapter()

        # Store run_id on state so callers (e.g. UI stop button) can access it
        # during the investigation without waiting for it to complete.
        state._run_id = run_id
        state._matter_adapter = matter_adapter
        _usage_ctx = None
        if run_id is not None and self._matter_model is not None:
            _usage_ctx = self.client.begin_usage_context(
                matter_id=self._matter_model.matter_id,
                run_id=run_id,
                recorder=self._matter_model.record_llm_call,
            )

        self._emit_step(
            state,
            StepType.THINKING,
            f"Research mode: {self._research_mode_label(state.research_mode)}",
        )

        try:
            # Phase 1: Orientation — pass pre-computed stats to avoid a second glob walk
            await self._orient(state, repo, _stats=stats)
            if state.findings.get("_direct_repository_answer"):
                state.complete()
                if run_id is not None:
                    self._cleanup_checkpoints(state)
                    self._matter_model.complete_run(
                        run_id,
                        llm_calls_avoided=state.llm_calls_avoided,
                        llm_calls_required=state.llm_calls_required,
                    )
                return state

            # Phase 1.5: Document Ingestion — read all new/changed documents BEFORE
            # searching. This ensures the system understands what's in the repo
            # before generating search leads. Already-ingested files are skipped
            # (hot path via DocumentInventoryStore). Search becomes targeted
            # follow-up, not blind exploration.
            await self._ingest_documents(state, repo)

            # Phase 2: Iterative investigation loop
            await self._investigate_loop(state, repo)

            # If user stopped the run, skip verify/synthesis and mark interrupted.
            # Partial facts/citations/leads are preserved as-is for resume.
            _adapter = getattr(state, "_matter_adapter", None)
            if _adapter is not None and _adapter.is_stop_requested():
                self._emit_step(state, StepType.THINKING, "Stopped by user — partial state preserved")
                # Force checkpoint on stop so the resume route always has a file to use.
                # iteration=None because we are not at a clean periodic boundary.
                self._save_checkpoint(state, iteration=None)
                state.interrupt()
                if run_id is not None:
                    self._matter_model.interrupt_run(run_id)
                return state

            # Phase 2.5: Verify citations
            await self._verify_citations(state, repo)

            # Re-check stop after verify — _verify_citations() may have broken out early
            # without the caller knowing, leaving some citations unchecked. If stop was
            # requested, save checkpoint and interrupt rather than completing the run.
            _adapter_post_verify = getattr(state, "_matter_adapter", None)
            if _adapter_post_verify is not None and _adapter_post_verify.is_stop_requested():
                self._emit_step(
                    state, StepType.THINKING,
                    "Stopped by user during citation verification — partial state preserved",
                )
                self._save_checkpoint(state, iteration=None)
                state.interrupt()
                if run_id is not None:
                    self._matter_model.interrupt_run(run_id)
                return state

            # Phase 2.75: Detect gaps BEFORE synthesis so they appear in the memo (SO-7).
            # Running these here means _build_gap_summary() in _synthesize() finds them.
            # Each detector is isolated so one failure never suppresses the other.
            if run_id is not None:
                try:
                    # Detect numeric conflicts → gaps (SO-6 + SO-7)
                    self._matter_model.detect_quant_conflicts(run_id=run_id)
                except Exception as _qc_exc:
                    logger.warning("Quant conflict detection failed, continuing: %s", _qc_exc)
                try:
                    # Detect issues with zero supporting assertions → proof gaps (SO-7)
                    self._detect_proof_gaps()
                except Exception as _pg_exc:
                    logger.error("Proof gap detection failed — synthesis may miss unsupported issues: %s", _pg_exc)
                try:
                    # Background maintenance: mine contradictions (SO-2) — auto-discovers
                    # heuristic contradiction links and propagates belief state changes.
                    self._matter_model.mine_contradictions(run_id=run_id)
                except Exception as _mc_exc:
                    logger.warning("Contradiction mining failed, continuing: %s", _mc_exc)
                try:
                    # Background maintenance: detect document version chains (SO-1) —
                    # links versioned documents, persists family_id, and records gaps
                    # for missing base versions.
                    self._matter_model.refresh_document_families()
                except Exception as _vc_exc:
                    logger.warning(
                        "Version chain detection failed, continuing: %s", _vc_exc
                    )

            # Phase 3: Final synthesis (reads gaps via _build_gap_summary)
            await self._synthesize(state)

            # HIGH adv#035: use final_output PRESENCE rather than is_stop_requested()
            # to decide whether to interrupt. _synthesize() commits SO-1/SO-4 side
            # effects (authority extraction, proof state, cache) before the caller sees
            # the stop flag — checking is_stop_requested() here would interrupt a fully-
            # synthesized run and leave those side effects orphaned. Instead: if
            # _synthesize() returned early (stop was seen at its entry check → no output
            # written), interrupt cleanly. If final_output is present, synthesis
            # completed — commit the run regardless of a late stop signal.
            if "final_output" not in state.findings:
                self._emit_step(
                    state, StepType.THINKING,
                    "Stopped by user — partial state preserved",
                )
                self._save_checkpoint(state, iteration=None)
                state.interrupt()
                if run_id is not None:
                    self._matter_model.interrupt_run(run_id)
                return state

            state.complete()
            if run_id is not None:
                # adv#036 MEDIUM (r90 fix): check BEFORE complete_run() so the event is
                # committed while the run is still 'running' in the DB. Live streamers
                # stop polling once they see terminal status; writing after would race.
                try:
                    if self._matter_model.ledger.is_redirect_requested(run_id):
                        self._matter_model.ledger.clear_redirect(run_id)
                        from irys.matter.enums import LedgerEventType as _LET
                        self._matter_model.ledger.append_event(
                            run_id=run_id,
                            event_type=_LET.USER_REDIRECTED,
                            summary=(
                                "Redirect received too late — investigation completed "
                                "before it could be applied; resubmit on a new run"
                            ),
                            why="adv#036: late redirect cleared before run completion",
                        )
                except Exception:
                    pass
                self._cleanup_checkpoints(state)
                self._matter_model.complete_run(
                    run_id,
                    llm_calls_avoided=state.llm_calls_avoided,
                    llm_calls_required=state.llm_calls_required,
                )
                # Generate clarification questions from open gaps (SO-7)
                self._matter_model.generate_clarifications_from_gaps(
                    run_id=run_id,
                    top_n=3,
                    min_materiality=0.5,
                )
                # Attach pending clarifications to state (SO-7) so callers receive
                # them in the default workflow without a separate API call.
                try:
                    state.pending_clarifications = (
                        self._matter_model.clarifications.get_pending()
                    )
                except Exception:
                    pass  # non-fatal: endpoint still available at /clarifications
                # Attach durable reasoning trail to state (SO-3) so callers see the
                # full structured trace without a separate ledger query.
                try:
                    state.reasoning_trail = self._matter_model.ledger.get_events(run_id)
                except Exception:
                    pass  # non-fatal: in-memory thinking_steps still available

        except Exception as e:
            state.fail(str(e))
            if run_id is not None:
                self._cleanup_checkpoints(state)
                # adv#037 MEDIUM: mirror the resume path's fail_run() robustness.
                # If fail_run() raises (e.g. DB locked) the run would stay 'running'
                # and block future investigations on the matter. Attempt a bare
                # autocommit UPDATE as a last-resort fallback.
                try:
                    self._matter_model.fail_run(run_id, str(e))
                except Exception as _fail_exc:
                    logger.warning(
                        "fail_run(%s) raised during investigate() cleanup; "
                        "attempting direct status update fallback: %s",
                        run_id, _fail_exc,
                    )
                    try:
                        from datetime import datetime as _datetime, timezone as _tz
                        _now_iso = _datetime.now(_tz.utc).isoformat()
                        self._matter_model.ledger.db.execute(
                            "UPDATE run_session SET status='failed', completed_at=?,"
                            " next_action=NULL, stop_requested=0, redirect_requested=0"
                            " WHERE id=? AND matter_id=?",
                            (_now_iso, run_id, self._matter_model.matter_id),
                        )
                    except Exception:
                        pass
            raise
        finally:
            if _usage_ctx is not None:
                self.client.end_usage_context(_usage_ctx)

        return state

    @staticmethod
    def _compact_inventory_signal(text: str) -> str:
        return _re_date.sub(r"[^a-z0-9]+", "", str(text or "").lower())

    @staticmethod
    def _repository_inventory_target(query: str) -> tuple[str, str] | None:
        q = " ".join(str(query or "").lower().split())
        compact = RLMEngine._compact_inventory_signal(q)
        count_intent = any(
            phrase in q
            for phrase in (
                "how many",
                "number of",
                "count",
                "are there any",
            )
        )
        list_intent = any(
            phrase in q
            for phrase in (
                "list files",
                "list documents",
                "list filings",
                "which files",
                "which documents",
                "which filings",
                "what files",
                "what documents",
                "what filings",
            )
        )
        list_intent = list_intent or (
            q.startswith(("list 10", "list the 10", "what 10", "which 10"))
            and any(term in q for term in ("are there", "available", "exist"))
        )
        asks_inventory = count_intent or list_intent
        if not asks_inventory:
            return None
        content_terms = (
            "analyze",
            "compare",
            "contain",
            "contains",
            "disclose",
            "discuss",
            "explain",
            "income",
            "mention",
            "mentions",
            "missing",
            "revenue",
            "risk factor",
            "say",
            "says",
            "show",
            "summarize",
            "trend",
        )
        if any(term in q for term in content_terms):
            return None

        targets = (
            ("10k", "10-K"),
            ("10q", "10-Q"),
            ("8k", "8-K"),
            ("ex99", "EX-99"),
        )
        for compact_target, label in targets:
            if compact_target in compact:
                return compact_target, label
        return None

    def _try_answer_from_repository_inventory(
        self,
        state: InvestigationState,
        file_list: list[Any],
    ) -> bool:
        target = self._repository_inventory_target(state.query)
        if target is None:
            return False
        compact_target, label = target

        matches: list[str] = []
        for file_info in file_list:
            rel_path = str(getattr(file_info, "relative_path", "") or "")
            if compact_target in self._compact_inventory_signal(rel_path):
                matches.append(rel_path)
        matches = sorted(dict.fromkeys(matches))

        noun = "document" if len(matches) == 1 else "documents"
        lines = [f"There are {len(matches)} {label} {noun} in the repository."]
        if matches:
            shown = matches[:50]
            lines.append("")
            lines.extend(f"- `{path}`" for path in shown)
            if len(matches) > len(shown):
                lines.append(f"- ... {len(matches) - len(shown)} more")

        state.findings["repository_inventory_answer"] = {
            "target": label,
            "count": len(matches),
            "matched_paths": matches,
            "source": "repository_path_metadata",
        }
        state.findings["_direct_repository_answer"] = True
        state.early_terminate_reason = (
            f"Answered {label} inventory question from repository path metadata"
        )
        self._emit_output(
            state,
            "\n".join(lines),
            emitter="repository_inventory",
        )
        self._emit_step(
            state,
            StepType.FINDING,
            f"Answered from repository path metadata: {len(matches)} {label} {noun}; "
            "skipping profiling and deep read",
        )
        return True

    async def _orient(self, state: InvestigationState, repo: MatterRepository, _stats=None):
        """Phase 1: Understand repository and form initial hypothesis.

        _stats: pre-computed RepositoryStats from investigate() to avoid a second
        glob walk.  If None (e.g. direct callers in tests), stats are fetched here.
        """
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is not None and _adapter.is_stop_requested():
            return  # Stop was requested before orientation even started
        self._emit_step(state, StepType.THINKING, "Analyzing repository structure...")

        # Get repository overview — reuse pre-computed stats if available
        stats = _stats if _stats is not None else repo.get_stats()
        structure = repo.get_structure()

        structure_str = "\n".join(f"  {folder}: {count} files" for folder, count in structure.items())

        # Include actual filenames so the LLM can make informed decisions about
        # which documents are most likely relevant (e.g., "Master_Service_Agreement.pdf"
        # is clearly a contract, "Acorn_Invoice_2024.xlsx" is financial).
        # Limit to 100 filenames to avoid prompt bloat on large repos.
        file_list = repo.list_files()
        file_listing_lines = []
        for f in file_list[:100]:
            size_kb = f.size_bytes / 1024
            file_listing_lines.append(f"  {f.relative_path} ({f.file_type}, {size_kb:.0f}KB)")
        if len(file_list) > 100:
            file_listing_lines.append(f"  ... and {len(file_list) - 100} more files")
        file_listing_str = "\n".join(file_listing_lines) if file_listing_lines else "  (no supported files found)"

        # Cheap count/list questions can finish from filenames alone.
        if self._try_answer_from_repository_inventory(state, file_list):
            return

        # Read persisted matter state — activates SO-1 (reuse) and SO-4 (issue-driven)
        adapter = getattr(state, "_matter_adapter", None)
        matter_ctx = adapter.get_context() if adapter is not None else None

        # Seed InvestigationState with facts already in the matter model (SO-1 hot reuse)
        if matter_ctx is not None and matter_ctx.existing_assertion_count > 0:
            self._hydrate_from_matter_model(state)

        if self._matter_model is not None:
            try:
                state.cache_manifest_hash = (
                    self._matter_model.build_semantic_cache_manifest(
                        taint_class=self._resolve_taint_default(),
                    )
                )
            except Exception as exc:
                logger.warning("build_semantic_cache_manifest failed: %s", exc)
            if state.cache_manifest_hash and state.working_set:
                state.working_set.dependency_manifest_hash = (
                    state.cache_manifest_hash
                )

        # MVP.6: cap the durable matter_context block so new stores
        # can't silently inflate the orientation prompt. The repo file
        # listing stays uncapped — it's a direct structural signal the
        # orient prompt needs.
        _budget = self._get_packet_budget()
        _matter_ctx_str = _format_matter_context(matter_ctx)
        _matter_ctx_capped = self._cap_text_by_tokens(
            _matter_ctx_str, _budget.orientation_tokens
        )
        _orient_domain = self._resolve_active_domain(state)
        _orient_ctx = _DOMAIN_ORIENTATION_CONTEXT.get(
            _orient_domain, _DOMAIN_ORIENTATION_CONTEXT["legal"]
        )
        prompt = ORIENTATION_PROMPT.format(
            structure=structure_str,
            file_listing=file_listing_str,
            total_files=stats.total_files,
            query=state.query,
            matter_context=_matter_ctx_capped,
            research_alignment_guidance=RESEARCH_ALIGNMENT_GUIDANCE,
            domain_issue_types=_orient_ctx["issue_types"],
            domain_issue_type_descriptions=_orient_ctx["issue_type_descriptions"],
            domain_predicate_examples=_orient_ctx["predicate_examples"],
            domain_document_priorities=_orient_ctx["document_priorities"],
            domain_search_examples=_orient_ctx["search_examples"],
        )

        # Orientation cache key: sha256 of normalized query + total file count +
        # content fingerprints of answered clarifications, open issues, gaps, annotations,
        # and trust overrides — so any content change invalidates the cache, not just counts.
        import hashlib as _hashlib
        _ctx_fingerprint = ""
        if self._matter_model is not None:
            try:
                # Hash content (not just counts or IDs) so editing an issue/annotation/
                # answer invalidates the cache even when the count stays the same.
                _ans = sorted(
                    f"{q.get('question_text','')}:{q.get('answer_text','')}"
                    for q in self._matter_model.clarifications.get_answered(limit=100)
                )
                _iss = sorted(i["title"] for i in self._matter_model.issues.get_open_issues())
                # Hash gap descriptions (not just count) to detect content changes.
                # limit=100 prevents full-table scan on large corpora; any change in the
                # top-100 highest-materiality gaps invalidates the orientation cache.
                _gaps_fp = sorted(
                    g.get("description", "") for g in self._matter_model.gaps.open_gaps(limit=100)
                )
                _anns = sorted(
                    a.get("annotation_text", "") for a in self._matter_model.annotations.list_recent()
                )
                _trs = sorted(
                    f"{o.get('document_pattern', '')}:{o.get('trust_level', '')}"
                    for o in self._matter_model.trust_overrides.list_all()
                )
                _ctx_fingerprint = repr([_ans, _iss, _gaps_fp, _anns, _trs])
            except Exception:
                pass  # non-critical; fallback to query+file-count key
        # Include _ORIENTATION_CACHE_VERSION so prompt structure changes
        # (e.g. adding "predicates" field) automatically invalidate cached plans.
        # Include file listing digest (not just count) so replacing a file
        # with a same-named different file invalidates the cache.
        _file_digest = _hashlib.sha256(file_listing_str.encode()).hexdigest()[:16]
        _history_digest = _conversation_history_digest(state.conversation_history)
        _orient_key = _hashlib.sha256(
            f"{state.query.lower().strip()}\n{stats.total_files}\n{_file_digest}"
            f"\n{_history_digest}\n{_ctx_fingerprint}\nv{_ORIENTATION_CACHE_VERSION}"
            f"\ndomain:{_orient_domain}".encode()
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
            state.llm_calls_required += 1  # SO-1 telemetry
            response = await self.client.complete(
                prompt,
                tier=ModelTier.FLASH,
                json_mode=True,
                usage_label="orientation",
                conversation_history=state.conversation_history,
                temperature=0.0,
            )
            plan = self._parse_json_safe(response, _plan_defaults)
            # Persist for future warm runs
            if self._matter_model is not None:
                _mh = state.cache_manifest_hash
                if _mh:
                    self._matter_model.cache.put_brokered(
                        "orient", _orient_key, plan, manifest_hash=_mh,
                    )
                else:
                    self._matter_model.cache.put("orient", _orient_key, plan)
        else:
            state.llm_calls_avoided += 1  # SO-1 telemetry
            self._emit_step(
                state, StepType.THINKING, "Orientation cache hit — reusing prior plan"
            )

        _hyp = plan.get("hypothesis")
        state.hypothesis = _hyp if isinstance(_hyp, str) else None
        _plan_issues = plan.get("issues")
        state.findings["issues"] = _plan_issues if isinstance(_plan_issues, list) else []
        state.findings["initial_plan"] = plan

        # Record issues in matter model if enabled; collect new IDs so initial leads can
        # be linked to freshly-created issues even on the first run (SO-4 backbone fix).
        adapter = getattr(state, "_matter_adapter", None)
        # Collect all issue IDs produced by this orientation pass (new AND existing).
        # Used to build the predicate-lead pool and fallback lead targets.
        _orient_issue_ids: list[str] = []
        _raw_idx_to_issue_id: dict[int, str] = {}  # raw issues[] index → issue_id
        if adapter is not None and self._matter_model is not None:
            from ..matter.enums import IssueType
            _issue_type_map = {
                "claim": IssueType.CLAIM,
                "defense": IssueType.DEFENSE,
                "damages": IssueType.DAMAGES,
                "exposure": IssueType.DAMAGES,
                "contract_question": IssueType.CONTRACT_QUESTION,
                "interpretation": IssueType.CONTRACT_QUESTION,
                "procedural": IssueType.PROCEDURAL_BARRIER,
                "evidentiary": IssueType.EVIDENTIARY_BOTTLENECK,
                "condition_precedent": IssueType.CONDITION_PRECEDENT,
                "waiver": IssueType.WAIVER,
                "diligence_red_flag": IssueType.DILIGENCE_RED_FLAG,
                "compliance_failure": IssueType.COMPLIANCE_FAILURE,
            }
            _plan_issues_raw = plan.get("issues")
            for _raw_issue_idx, issue_item in enumerate(
                _plan_issues_raw if isinstance(_plan_issues_raw, list) else []
            ):
                # Accept both legacy string format and new {title, type} dict format
                if isinstance(issue_item, str):
                    issue_title = issue_item.strip()
                    issue_type = IssueType.CLAIM
                elif isinstance(issue_item, dict) and "title" in issue_item:
                    # Guard against None or non-string title from LLM
                    _raw_title = issue_item["title"]
                    if not isinstance(_raw_title, str):
                        continue
                    issue_title = _raw_title.strip()
                    _raw_type = issue_item.get("type")
                    issue_type = _issue_type_map.get(
                        (_raw_type.lower() if isinstance(_raw_type, str) else "claim"),
                        IssueType.CLAIM,
                    )
                else:
                    continue
                if not issue_title:
                    continue
                issue_id, _ = self._matter_model.issues.upsert_issue(
                    title=issue_title,
                    issue_type=issue_type,
                    salience=0.7,
                )
                _orient_issue_ids.append(issue_id)
                _raw_idx_to_issue_id[_raw_issue_idx] = issue_id
                adapter.log_step(
                    f"Issue identified ({issue_type.value}): {issue_title[:100]}",
                    why="From orientation analysis",
                )
                # Persist predicates for this issue (SO-4: issue predicate tree).
                # add_predicates_batch() is idempotent (INSERT OR IGNORE + unique index),
                # so safe on both new issues and warm-run cache hits that return existing IDs.
                # Guard against non-list LLM output (null, string, dict).
                if isinstance(issue_item, dict):
                    _preds_raw = issue_item.get("predicates")
                    if isinstance(_preds_raw, list):
                        self._matter_model.issues.add_predicates_batch(
                            issue_id=issue_id,
                            descriptions=_preds_raw[:4],
                        )

        # Checklist seeding: for comparison/regulatory tasks, ensure provision-level
        # issue objects exist even if orientation missed them. This prevents the coverage
        # planner from being blind to unmodeled issues.
        if adapter is not None and self._matter_model is not None:
            _existing_titles = set()
            for _eid in _orient_issue_ids:
                try:
                    _erow = self._matter_model.issues.get_issue(_eid)
                    if _erow:
                        _existing_titles.add(_erow.get("title", "").lower().strip())
                except Exception:
                    pass
            _checklist_issues = self._get_checklist_issues(state.query)
            for _cl_title, _cl_preds in _checklist_issues:
                _cl_lower = _cl_title.lower().strip()
                if any(_cl_lower in et or et in _cl_lower for et in _existing_titles if et):
                    continue
                try:
                    _cl_id, _ = self._matter_model.issues.upsert_issue(
                        title=_cl_title,
                        issue_type=IssueType.DILIGENCE_RED_FLAG,
                        salience=0.5,
                    )
                    _orient_issue_ids.append(_cl_id)
                    if _cl_preds:
                        self._matter_model.issues.add_predicates_batch(
                            issue_id=_cl_id, descriptions=_cl_preds[:3],
                        )
                    adapter.log_step(
                        f"Checklist issue seeded: {_cl_title[:80]}",
                        why="Provision checklist ensures completeness",
                    )
                except Exception:
                    pass

        # Create initial leads from plan — preserve raw search terms to bypass
        # _extract_search_term() token collapse (SO-4 issue-focused search).
        # Use weakest prior-run issue if it exists; otherwise rotate through freshly
        # created issues so run-1 facts are linked to issues from the start.
        weakest_id = matter_ctx.weakest_issue_id if matter_ctx else None
        _issue_pool = _orient_issue_ids  # fallback pool: distribute leads across orientation issues
        # SO-4: coverage-biased fallback pool — weakest issue first so the first
        # unannotated search targets the proof gap, then round-robin for the rest.
        # weakest_id is always included even if it is not in _orient_issue_ids (e.g. it is an
        # issue from a prior run that was not re-emitted by the orientation LLM this run).
        if weakest_id:
            _biased_pool = [weakest_id] + [i for i in _issue_pool if i != weakest_id]
        else:
            _biased_pool = list(_issue_pool)
        # SO-4 semantic gate: build text profiles for each issue in the pool once,
        # so unannotated searches can be validated against issue content (title +
        # predicates) rather than assigned purely structurally.  Profile building is
        # lightweight (DB lookups of short text rows) and skips gracefully on any error.
        _issue_profiles: "dict[str, str]" = (
            self._build_issue_profiles(_biased_pool) if len(_biased_pool) >= 2 else {}
        )
        # DB-verified IDs only — subset of _biased_pool where a profile was successfully built.
        # Used for round-robin fallback to avoid stale IDs that failed the profile lookup.
        _profile_pool: "list[str]" = list(_issue_profiles.keys())
        # Parse initial_searches: support new dict form {"term": "...", "issue_idx": N}
        # and legacy string form for backward compatibility.
        # Use `or []` to handle null from LLM (MEDIUM guard).
        _ps = plan.get("initial_searches")
        _raw_searches = (_ps if isinstance(_ps, list) else [])[:15]
        _initial_searches: list[tuple[str, int | None]] = []
        for _s in _raw_searches:
            if isinstance(_s, str) and _s.strip():
                _initial_searches.append((_s.strip(), None))
            elif isinstance(_s, dict):
                _term = _s.get("term", "")
                if isinstance(_term, str) and _term.strip():
                    _iidx = _s.get("issue_idx")
                    # Exclude booleans (bool is a subclass of int in Python).
                    _valid_idx = isinstance(_iidx, int) and not isinstance(_iidx, bool)
                    _initial_searches.append((_term.strip(), _iidx if _valid_idx else None))
        # Use a separate counter for bare-string (unannotated) searches so that LLM-annotated
        # entries don't shift the round-robin rotation for later unannotated entries.
        _bare_idx: int = 0
        for _idx, (_search_term, _lm_issue_idx) in enumerate(_initial_searches):
            # Assign focus_issue_id using priority order:
            # 1. LLM-specified issue_idx → raw issues[] position → issue_id via
            #    _raw_idx_to_issue_id (not filtered _orient_issue_ids, so skipped
            #    issues don't shift indices for later entries — MEDIUM fix).
            # 2. Semantic gate: Jaccard similarity of search term against issue profiles.
            #    Accepts the best match only if score > threshold AND margin > gap.
            #    Abstains (None) if no issue clears both thresholds.
            # 3. When semantic gate abstains (None), leave focus_issue_id=None —
            #    wrong attribution is worse than no attribution.
            #    Round-robin fallback is only used when profiles are unavailable
            #    (< 2 issues, or all profile builds failed), not on semantic abstention.
            if _lm_issue_idx is not None:
                _focus_id = _raw_idx_to_issue_id.get(_lm_issue_idx)
            elif _issue_profiles:
                # Semantic gate: accept best match or abstain (None)
                _focus_id = self._best_semantic_issue(_search_term, _issue_profiles)
                _bare_idx += 1
            elif _biased_pool:
                # No profiles available (< 2 issues or all profile builds failed)
                # — structural round-robin over full pool as last resort.
                # _profile_pool would always be empty here (it's a subset of _issue_profiles)
                # so use _biased_pool directly for the structural fallback.
                _focus_id = _biased_pool[_bare_idx % len(_biased_pool)]
                _bare_idx += 1
            else:
                _focus_id = None
            priority = 0.9 if (_focus_id and _idx == 0) else 0.8
            state.add_lead(
                description=f"Search for: {_search_term}",
                source="initial_plan",
                priority=priority,
                search_term=_search_term,
                focus_issue_id=_focus_id,
            )

        # Target documents: preserve specific filenames from orientation as
        # high-priority search leads. Semantically match each to the best issue
        # rather than always attributing to the first issue.
        _target_docs = plan.get("target_documents") or []
        for _td in (_target_docs if isinstance(_target_docs, list) else [])[:10]:
            if isinstance(_td, str) and _td.strip():
                _target_focus = None
                if _issue_profiles:
                    _target_focus = self._best_semantic_issue(_td.strip(), _issue_profiles)
                state.add_lead(
                    description=f"Target document: {_td.strip()}",
                    source="orientation_target",
                    priority=0.85,
                    search_term=_td.strip(),
                    focus_issue_id=_target_focus,
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

        # Add predicate-driven leads: one predicate per issue, capped by issue lane.
        # Coverage planner adds remaining predicates later based on live gaps.
        if self._matter_model is not None and _orient_issue_ids:
            _issue_lane = max(1, (self.config.max_leads_per_level + 1) // 2)
            _pred_budget = min(len(_orient_issue_ids), _issue_lane)
            _pred_added = 0
            _ordered_issues = []
            if weakest_id and weakest_id in _orient_issue_ids:
                _ordered_issues.append(weakest_id)
            for _oid in _orient_issue_ids:
                if _oid not in _ordered_issues:
                    _ordered_issues.append(_oid)
            for _pred_target_id in _ordered_issues:
                if _pred_added >= _pred_budget:
                    break
                _issue_predicates = self._matter_model.issues.get_predicates(_pred_target_id, limit=1)
                for _pred_row in _issue_predicates:
                    if _pred_added >= _pred_budget:
                        break
                    _pred_text = _pred_row.get("description", "").strip()
                    if _pred_text:
                        state.add_lead(
                            description=f"Evidence for: {_pred_text}",
                            source="predicate",
                            priority=0.75,
                            search_term=_pred_text,
                            focus_issue_id=_pred_target_id,
                        )
                        _pred_added += 1

        # Generate SPO predicate graph leads (SO-2 → SO-4): convert top assertion
        # predicate_keys to human-readable search terms. This directly uses the
        # assertion graph to drive retrieval — closing the loop between stored
        # structured knowledge and targeted search (read path, not write-only).
        # Only fire on runs after the first (matter has assertions from prior runs).
        if (
            matter_ctx is not None
            and matter_ctx.existing_assertion_count > 0
            and getattr(matter_ctx, "key_predicates", None)
        ):
            # Convert snake_case predicate keys to natural-language search terms.
            # Limit to 2 SPO leads to avoid overwhelming the lead queue.
            _spo_leads_added = 0
            for _pred_key in matter_ctx.key_predicates[:4]:
                if _spo_leads_added >= 2:
                    break
                # Convert snake_case → space-separated phrase (e.g. "agreed_to_pay" → "agreed to pay")
                _pred_phrase = _pred_key.replace("_", " ").strip()
                if not _pred_phrase or len(_pred_phrase) < 3:
                    continue
                # SO-4 semantic gate: try to match pred_phrase to the most relevant issue.
                # Abstain (None) if gate can't find a clear winner — wrong attribution
                # is worse than no attribution. Round-robin only when no profiles at all.
                _spo_focus: "Optional[str]" = None
                if _issue_profiles:
                    _spo_focus = self._best_semantic_issue(_pred_phrase, _issue_profiles)
                elif _biased_pool:
                    # No profiles — structural fallback to biased_pool
                    _spo_focus = _biased_pool[_spo_leads_added % len(_biased_pool)]
                state.add_lead(
                    description=f"SPO graph expansion: search for '{_pred_phrase}' relationships",
                    source="spo_graph",
                    priority=0.55,
                    search_term=_pred_phrase,
                    focus_issue_id=_spo_focus,
                )
                _spo_leads_added += 1

        self._emit_step(
            state,
            StepType.THINKING,
            f"Hypothesis: {state.hypothesis}",
            details=plan,
        )

        # Inject orientation target documents into output_contract for validators
        _orient_target_docs = plan.get("target_documents") or []
        if _orient_target_docs:
            _ec = getattr(state, "execution_contract", None)
            if _ec is not None:
                _oc = getattr(_ec, "output_contract", None) or {}
                if isinstance(_oc, dict) and "target_documents" not in _oc:
                    _oc["target_documents"] = [
                        str(d).strip() for d in _orient_target_docs
                        if isinstance(d, str) and d.strip()
                    ][:10]
                    _ec.output_contract = _oc

        # Sync workflow obligations after orientation enriched the contract
        self._sync_workflow_obligations_from_contract(state)

        # Log orientation summary to reasoning ledger (SO-3 user visibility)
        adapter = getattr(state, "_matter_adapter", None)
        if adapter is not None:
            _pif = plan.get("issues")
            issues_found = _pif if isinstance(_pif, list) else []
            _psf = plan.get("initial_searches")
            searches_planned = _psf if isinstance(_psf, list) else []
            adapter.log_step(
                f"Orientation complete: {len(issues_found)} issues, {len(searches_planned)} search leads",
                why=f"Hypothesis: {(state.hypothesis or '')[:200]}",
            )

    async def _investigate_loop(self, state: InvestigationState, repo: MatterRepository):
        """Phase 2: Iterative investigation with recursive lead following."""
        iteration = 0
        budget = self._get_research_profile(state)
        max_iterations = max(0, int(budget.max_iterations))
        contract = getattr(state, "execution_contract", None)
        if contract is not None:
            _contract_max = max(0, int(getattr(contract, "max_iter", max_iterations)))
            max_iterations = min(max_iterations, _contract_max)

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

            # Close-loop feedback: apply term boost/demotion once (not per-iteration
            # to avoid compounding — MEDIUM #6 from Codex review).
            if iteration == 0 and hasattr(state, 'apply_feedback_to_leads'):
                state.apply_feedback_to_leads()

            pending_leads = state.get_pending_leads()

            # P0.7: coverage-driven planner runs BEFORE the no-pending
            # break. If reactive leads are exhausted but the matter
            # still has weak material issues, the planner can inject
            # issue-targeted leads to keep investigation moving. The
            # planner reads coverage map once here; we reuse it below
            # to avoid a second SQL pass.
            # adv#11 review fix (#4 note): gate the early _cov_map
            # computation on the same investigate-family check as the
            # planner — don't pay the SQL round trip for a non-
            # investigate contract accidentally entering this loop.
            _cov_map: "dict[str, tuple[float, bool, int]]" = {}
            _contract = getattr(state, "execution_contract", None)
            _is_investigate = (
                _contract is None
                or getattr(_contract, "family", None) == "investigate"
            )
            if self._matter_model is not None and _is_investigate:
                # Mid-run issue discovery (Codex #5): scan accumulated facts
                # for issues that orientation missed, before coverage planner runs.
                _disc_count = self._discover_unmodeled_issues(state, iteration)
                if _disc_count > 0:
                    self._emit_step(
                        state, StepType.THINKING,
                        f"Discovered {_disc_count} new issue(s) from accumulated evidence",
                    )
                try:
                    _cov_map = self._get_issue_coverage_map()
                except Exception as _exc:
                    logger.warning("coverage_map failed: %s", _exc)
                    _cov_map = {}
                if _cov_map:
                    _planner_added = self._coverage_planner(state, _cov_map)
                    if _planner_added > 0:
                        self._emit_step(
                            state,
                            StepType.THINKING,
                            f"Coverage planner added {_planner_added} issue-targeted lead(s)",
                        )
                        pending_leads = state.get_pending_leads()

            if not pending_leads:
                self._emit_step(state, StepType.THINKING, "No more leads to investigate")
                break

            # SO-4 Leak-1+3: re-score leads by live issue coverage weakness each
            # iteration so weaker issues attract more budget as the run progresses.
            # P0.7: the planner above may have already populated _cov_map;
            # reuse it rather than paying a second SQL round-trip.
            if _cov_map and any(_l.focus_issue_id for _l in pending_leads):
                _ISSUE_BOOST = 0.35   # α — boost per unit weakness (1 - coverage_fraction)
                _GAP_BOOST   = 0.20   # β — extra boost when a proof gap is open
                _NEUTRAL_DAMP = 0.15  # κ — dampening for non-issue-anchored leads
                _reweighted: "list[tuple[float, object]]" = []
                for _lead in pending_leads:
                    if _lead.focus_issue_id and _lead.focus_issue_id in _cov_map:
                        _frac, _gap, _ = _cov_map[_lead.focus_issue_id]
                        _weakness = 1.0 - _frac
                        _adj = _lead.priority * (
                            1.0 + _ISSUE_BOOST * _weakness + (_GAP_BOOST if _gap else 0.0)
                        )
                    elif not _lead.focus_issue_id:
                        _adj = _lead.priority * (1.0 - _NEUTRAL_DAMP)
                    else:
                        _adj = _lead.priority  # issue-targeted, issue not yet in map
                    _reweighted.append((_adj, _lead))
                _reweighted.sort(key=lambda _x: _x[0], reverse=True)
                pending_leads = [_l for _, _l in _reweighted]

            # MVI-5 per-lead EV gating — stamp expected_cost_usd and
            # expected_coverage_gain on each pending lead so
            # _viable_leads has a real coverage-per-dollar signal. Cost
            # class is the same for all search leads today (LITE
            # extract + maybe FLASH reason ≈ $0.0015 after the
            # short-circuit); coverage gain is the target issue's
            # weakness times a small scalar, with a tiny default for
            # unanchored leads.
            if pending_leads:
                self._enrich_lead_ev(pending_leads, _cov_map)

            # SO-4 Leak-4: partition into issue-targeted and neutral to guarantee a
            # minimum issue-targeted quota — prevents neutral leads from crowding out
            # issue focus when the queue is dominated by generic follow-on searches.
            _i_pool = [_l for _l in pending_leads if _l.focus_issue_id]
            _n_pool = [_l for _l in pending_leads if not _l.focus_issue_id]
            _i_quota = max(1, (self.config.max_leads_per_level + 1) // 2)  # ceil(N/2)
            _i_sel = _i_pool[:_i_quota]
            _n_sel = _n_pool[:max(0, self.config.max_leads_per_level - len(_i_sel))]
            _budget = _i_sel + _n_sel

            leads_to_process = [_l for _l in _budget if _l.priority >= self.config.min_lead_priority]
            for _l in _budget:
                if _l.priority < self.config.min_lead_priority:
                    state.mark_lead_investigated(_l.id, "Skipped - low priority")

            # SO-4 Leak-6 emergency bootstrap: when a proof gap or weak issue exists
            # but no issue-targeted lead cleared the priority threshold, inject one
            # predicate-derived lead so coverage can advance even on neutral-heavy queues.
            if (not any(_l for _l in leads_to_process if _l.focus_issue_id)
                    and self._matter_model is not None and _cov_map):
                _gapped = [(iid, fr) for iid, (fr, gap, _) in _cov_map.items() if gap]
                _weak = [(iid, fr) for iid, (fr, _, _) in _cov_map.items() if fr < 0.3]
                _boot_candidates = _gapped or _weak
                if _boot_candidates:
                    _boot_id = min(_boot_candidates, key=lambda kv: kv[1])[0]
                    _boot_preds = self._matter_model.issues.get_predicates(_boot_id, limit=1)
                    if _boot_preds:
                        _boot_text = (_boot_preds[0].get("description") or "").strip()
                        if _boot_text:
                            _boot_lead = state.add_lead(
                                description=f"Gap bootstrap: {_boot_text}",
                                source="coverage_bootstrap",
                                priority=self.config.min_lead_priority + 0.01,
                                search_term=_boot_text,
                                focus_issue_id=_boot_id,
                            )
                            if _boot_lead is not None:
                                # Adv#11 Fix 2 (round 2): bootstrap lead must
                                # carry an EV score before the gate runs —
                                # otherwise it ships with ev_score=0 and
                                # _viable_leads treats it as a legacy lead
                                # gated only by priority, bypassing the
                                # coverage-per-dollar floor.
                                self._enrich_lead_ev([_boot_lead], _cov_map)
                                leads_to_process.append(_boot_lead)

            # Adv#11 Fix 2: MVI-5 EV gate was post-spend only (read
            # inside _should_continue_investigation). A below-floor lead
            # would still dispatch once before termination kicked in.
            # Apply _viable_leads here — AFTER bootstrap, before task
            # dispatch — so a cold-start batch with no EV coverage
            # (including an unenriched bootstrap injection) can't burn
            # tokens. When the caller doesn't supply a contract (legacy
            # path), _viable_leads falls back to priority-only
            # viability, which matches the previous behavior.
            _contract = getattr(state, "execution_contract", None)
            if _contract is not None and leads_to_process:
                _viable = self._viable_leads(leads_to_process, _contract)
                _viable_ids = {_l.id for _l in _viable}
                for _l in leads_to_process:
                    if _l.id not in _viable_ids:
                        state.mark_lead_investigated(
                            _l.id,
                            f"Skipped — below lead_ev_floor "
                            f"(ev={getattr(_l, 'ev_score', 0.0):.3f}, "
                            f"floor={getattr(_contract, 'lead_ev_floor', 0.5):.3f})",
                        )
                leads_to_process = _viable

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

            # Process leads in parallel with true cancellation support (SO-3).
            # asyncio.create_task() makes each lead a real Task so Task.cancel()
            # can inject CancelledError at in-flight LLM awaits, not just at the
            # next cooperative is_stop_requested() check.
            _lead_tasks = [
                asyncio.create_task(self._investigate_lead(state, repo, lead))
                for lead in leads_to_process
            ]
            results = await self._gather_with_cancellation(state, _lead_tasks)

            # Log any errors (None = task was cancelled by stop request — not an error)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Lead investigation failed: {leads_to_process[i].description}: {result}")

            # Track facts added this iteration for diminishing returns
            facts_after = len(state.findings.get("accumulated_facts", []))
            facts_added = facts_after - facts_before
            state.facts_per_iteration.append(facts_added)

            # MVI-3: sample governed-progress signals end-of-iteration
            # so the new termination checks can compute deltas. Sum of
            # issue coverage_fraction is the scalar answerability
            # proxy; open-gap count shows whether material
            # missingness is closing. Both cheap reads off the matter
            # model — no LLM.
            try:
                if self._matter_model is not None:
                    cov_rows = self._matter_model.get_issue_coverage_report(
                        policy_audience="internal",
                    )
                    coverage_sum = sum(
                        float(r.get("coverage_fraction") or 0.0)
                        for r in cov_rows
                    )
                    open_gaps = int(self._matter_model.gaps.count_open())
                else:
                    coverage_sum = 0.0
                    open_gaps = 0
            except Exception:
                coverage_sum = 0.0
                open_gaps = 0
            state.coverage_sum_per_iteration.append(coverage_sum)
            state.open_gap_count_per_iteration.append(open_gaps)

            # Flush belief revision for any new assertions added this iteration
            if adapter is not None:
                adapter.flush_revisions()

            iteration += 1

            # Inject clarification answers that arrived during this run (SO-3 active steering).
            # Each new answer becomes a high-priority search lead so the current run
            # immediately pursues the user-supplied context without restarting.
            if adapter is not None:
                new_answers = adapter.get_new_answered_clarifications()
                for answer in new_answers:
                    answer_text = (answer.get("answer_text") or "").strip()
                    question_text = (answer.get("question_text") or "").strip()
                    if answer_text:
                        # Resolve gap → issue so the lead closes the coverage gap (SO-4)
                        _cl_issue_id: Optional[str] = None
                        _gap_id = answer.get("gap_id")
                        if _gap_id and self._matter_model is not None:
                            _gl = self._matter_model.db.execute(
                                "SELECT affected_id FROM gap_link "
                                "WHERE gap_id=? AND affected_type='issue' LIMIT 1",
                                (_gap_id,),
                            ).fetchone()
                            if _gl:
                                _cl_issue_id = _gl["affected_id"]
                        state.add_lead(
                            description=f"User context: {answer_text[:80]}",
                            source="clarification_answer",
                            priority=0.9,
                            search_term=answer_text[:80],
                            focus_issue_id=_cl_issue_id,
                        )
                        adapter.log_step(
                            f"Injected clarification answer as active lead",
                            why=f"Q: {question_text[:60]} → A: {answer_text[:80]}",
                        )
                        self._emit_step(
                            state, StepType.REPLAN,
                            f"Steering from user context: {answer_text[:60]}",
                        )

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

            # Plan A: sufficiency probe every 2 iterations. Runs after
            # iter 2, 4, 6, ... so iter 1 doesn't false-positive on a
            # matter that just happens to have seed state. The probe
            # can stamp state.early_terminate_reason which overrides
            # contract.min_iter in _should_continue_investigation.
            if iteration >= 2 and iteration % self._SUFFICIENCY_PROBE_CADENCE == 0:
                try:
                    await self._run_sufficiency_probe(state)
                except Exception as _probe_exc:
                    logger.warning("sufficiency probe failed: %s", _probe_exc)

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

        P0.2: partitions hydrated rows into verified/candidate/
        stale/excluded buckets via TrustPolicy. Only verified +
        candidate enter accumulated_facts (with [VERIFIED]/[CANDIDATE]
        labels); stale/excluded live only in
        state.findings["hydrated_assertion_buckets"] so the engine can
        surface bucket counts without letting reviewed-out facts
        contaminate the synthesis prompt.
        """
        if self._matter_model is None:
            return
        try:
            recent = self._matter_model.assertions.list_recent_for_hydration(limit=2000)
        except Exception as _e:
            logger.warning("Matter model hydration failed — proceeding without prior facts: %s", _e)
            return
        if not recent:
            return

        from ..matter.trust import TrustBucket, TrustPolicy

        # P0.2 review fix #3: the store oversamples to limit*3 so
        # verified/candidate lanes are not starved at the chronological
        # head. The engine must enforce the real cap after
        # classification — without this, a matter with many stale rows
        # pushes 600+ facts into accumulated_facts and blows the prompt
        # budget. Cap is the same 200-slot budget the old code used.
        HYDRATION_CAP = 2000
        loaded = 0
        _strip_role_prefix = __import__("re").compile(r'^\[[A-Z_]+\]\s*').sub
        # P0.2: bucket partitioning. Engine consumers read the
        # buckets to surface counts; only verified/candidate enter
        # accumulated_facts.
        _buckets: dict[str, list[dict]] = {
            "verified": [], "candidate": [], "stale": [], "excluded": [],
        }
        _label_by_bucket = {
            "verified": "VERIFIED",
            "candidate": "CANDIDATE",
        }
        # P0.5 commit 3: route hydration through ContentPolicyGuard so
        # every decision is audited and the bucket map stays
        # consistent with the unified policy surface. We still read
        # belief_state + verification_status from the row payload;
        # privilege_flag is unknown at this layer (hydration query
        # doesn't join document_card), so the guard defaults to None
        # → fail-closed under clean audience for privileged targets,
        # which matches MVP.4 intent. Internal-audience hydration
        # bypasses privilege entirely.
        from ..matter.trust import ContentPurpose
        _guard = getattr(self._matter_model, "content_policy", None)
        for row in recent:
            prop = row.get("proposition_text", "")
            if not prop:
                continue
            classification = TrustPolicy.classify(
                assertion_verification_status=row.get("verification_status"),
                belief_state=row.get("belief_state"),
            )
            bucket_key = classification.bucket.value
            _buckets[bucket_key].append({
                "id": row.get("id"),
                "proposition_text": prop,
                "belief_state": row.get("belief_state"),
                "verification_status": row.get("verification_status"),
                "reason": classification.reason,
            })
            # Write one audit row per hydration decision (record=True
            # by default). Best-effort — a guard write failure cannot
            # break orientation.
            if _guard is not None and row.get("id"):
                try:
                    _guard.decide(
                        purpose=ContentPurpose.HYDRATION,
                        subject_kind="assertion",
                        subject_id=row["id"],
                        policy_audience="internal",
                        assertion_verification_status=row.get("verification_status"),
                        belief_state=row.get("belief_state"),
                    )
                except sqlite3.Error as _exc:
                    logger.warning(
                        "hydration: content_policy_audit write failed: %s",
                        _exc,
                    )
            # Stale/excluded never enter accumulated_facts — the user
            # has already expressed an opinion on them.
            if not classification.eligible:
                continue
            source_role = row.get("primary_source_role") or row.get("source_role") or "unknown"
            # Surface multi-source ambiguity (SO-5): when the same proposition appears in
            # both advocacy and operative documents, show all distinct roles so the LLM
            # can distinguish "contract clause alleged by plaintiff" from
            # "operative contract clause". Collapse only if truly single-source.
            _roles_csv = row.get("source_roles_csv") or ""
            _all_roles = [r for r in _roles_csv.split(",") if r] if _roles_csv else []
            if len(_all_roles) > 1:
                # Sort by trust descending for readability; de-dup preserving order
                _seen = set()
                _deduped = []
                for _r in _all_roles:
                    if _r not in _seen:
                        _seen.add(_r)
                        _deduped.append(_r)
                label = "MULTI-SOURCE[" + ",".join(r.upper() for r in _deduped) + "]"
            else:
                label = source_role.upper()
            # Strip any existing [ROLE] prefix to prevent double-labeling legacy rows
            prop_clean = _strip_role_prefix('', prop)
            # Prepend structured SPO annotation when the DB has typed fields (SO-2 read-back).
            # This makes structured assertions more useful to the LLM during orientation —
            # it sees not just the prose fact but the explicit subject/predicate/object.
            _subj_id = row.get("subject_ref_id")
            _pred = row.get("predicate_key")
            _obj_raw = row.get("object_json")
            _obj = None
            _obj_present = _obj_raw is not None
            if _obj_present:
                try:
                    _obj = json.loads(_obj_raw)
                except Exception:
                    _obj = _obj_raw
            if _subj_id or _pred or _obj_present:
                _spo_parts = []
                if _subj_id:
                    _spo_parts.append(f"SUBJ:{_subj_id}")
                if _pred:
                    _spo_parts.append(f"PRED:{_pred}")
                if _obj_present:
                    _spo_parts.append(f"OBJ:{str(_obj)[:60]}")
                prop_clean = f"[{' | '.join(_spo_parts)}] {prop_clean}"
            # P0.2: annotate with trust bucket so downstream consumers
            # (synthesis, UI) know whether to treat this as verified
            # matter-model support or a candidate lead awaiting review.
            _bucket_tag = _label_by_bucket.get(bucket_key, "CANDIDATE")
            _fact_str = f"[{label}][{_bucket_tag}] {prop_clean}"
            state.add_facts([_fact_str])
            if self.on_fact:
                self.on_fact(_fact_str)
            loaded += 1
            # P0.2 review fix #3: enforce the real cap after
            # classification. Stop once HYDRATION_CAP eligible rows
            # have been loaded even if the oversampled set is larger.
            if loaded >= HYDRATION_CAP:
                break

        # Expose bucket partitioning so engine consumers and tests can
        # assert on partitioning without re-deriving it from the
        # labeled fact strings.
        state.findings["hydrated_assertion_buckets"] = _buckets

        if loaded:
            _ver_n = len(_buckets["verified"])
            _cand_n = len(_buckets["candidate"])
            _stale_n = len(_buckets["stale"])
            _excl_n = len(_buckets["excluded"])
            self._emit_step(
                state,
                StepType.THINKING,
                f"Hydrated {loaded} facts ({_ver_n} verified, {_cand_n} candidate; "
                f"{_stale_n} stale + {_excl_n} excluded held back) from prior matter model run (SO-1 reuse)",
            )

    def _search_cached_assertions(
        self,
        queries: list[str],
        issue_id: Optional[str] = None,
        limit: int = 20,
    ) -> Optional[SearchResults]:
        """Search the assertion store for already-extracted intelligence.

        Converts assertion hits into pseudo-SearchResults so callers (like
        _investigate_lead) can treat them identically to raw repo grep hits.
        Returns None when no matter model or no hits.
        """
        if self._matter_model is None:
            return None
        try:
            rows = self._matter_model.search_assertions(
                queries=queries, issue_id=issue_id, limit=limit,
            )
        except Exception:
            return None
        if not rows:
            return None
        hits: list[SearchHit] = []
        _verified_count = 0
        _candidate_count = 0
        # P0.5 commit 3 (third surface): audit every cached search hit
        # that gets forwarded toward the LLM. The guard fires with
        # ContentPurpose.SEARCH_SNIPPETS_TO_LLM — a later audit can
        # answer "for this query, which assertions did Irys feed into
        # the next LLM turn?".
        _guard = getattr(self._matter_model, "content_policy", None)
        from ..matter.trust import ContentPurpose
        for r in rows:
            _fp = r.get("primary_document_id") or "assertion"
            _text = r.get("primary_raw_text") or r.get("proposition_text") or ""
            _score = float(r.get("term_matches", 1)) + (0.5 if r.get("issue_match") else 0)
            # P0.2: label candidate matches as leads so callers
            # cannot mistake them for source-text equivalents.
            _bucket = r.get("trust_bucket") or "candidate"
            if _bucket == "verified":
                _verified_count += 1
            else:
                _candidate_count += 1
                _text = f"[CANDIDATE LEAD — needs source confirmation] {_text}"
            if _guard is not None and r.get("id"):
                try:
                    _guard.decide(
                        purpose=ContentPurpose.SEARCH_SNIPPETS_TO_LLM,
                        subject_kind="assertion",
                        subject_id=r["id"],
                        policy_audience="internal",
                        assertion_verification_status=r.get("verification_status"),
                        belief_state=r.get("belief_state"),
                    )
                except sqlite3.Error as _exc:
                    logger.warning(
                        "cached search: content_policy_audit write failed: %s",
                        _exc,
                    )
            hits.append(SearchHit(
                file_path=_fp,
                filename=Path(_fp).name,
                page_num=0,
                line_num=0,
                match_text=_text,
                context_before=[],
                context_after=[],
                score=_score,
            ))
        sr = SearchResults(
            query=queries[0] if queries else "",
            hits=hits,
            files_searched=0,
            total_matches=len(hits),
        )
        sr._from_assertion_store = True  # type: ignore[attr-defined]
        sr._trust_bucket_counts = {  # type: ignore[attr-defined]
            "verified": _verified_count,
            "candidate": _candidate_count,
        }
        return sr

    def _build_lead_queries(
        self,
        lead: Lead,
        focus_issue_id: Optional[str],
        max_queries: int = 8,
    ) -> list[str]:
        """Build expanded search queries for a lead, issue-aware."""
        from ..core.search import expand_query
        search_term = lead.search_term or self._extract_search_term(lead.description)

        # Issue context enrichment — returns independent grep terms, not a mutated string
        issue_terms: list[str] = []
        if focus_issue_id and self._matter_model is not None:
            issue_terms = self._enrich_search_term_with_issue_context(
                search_term, focus_issue_id
            )

        # Gather context terms from active predicates for issue-aware expansion
        context_terms: list[str] = []
        if focus_issue_id and self._matter_model is not None:
            preds = self._matter_model.issues.get_predicates(focus_issue_id, limit=3)
            context_terms = [
                p.get("predicate_key") or p.get("description", "")
                for p in preds if p.get("predicate_key") or p.get("description")
            ][:2]

        queries = expand_query(search_term, max_expansions=max_queries - 1, context_terms=context_terms)
        # Append issue-derived terms as extra queries (separate, not concatenated)
        for it in issue_terms:
            if it not in queries:
                queries.append(it)
        return queries

    def _candidate_files_for_lead(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        lead: Lead,
        max_files: int = 120,
    ) -> Optional[list[str]]:
        """Return candidate file paths from document memory for targeted search.

        Returns None to signal full-repo fallback when no memory is available.
        """
        if self._matter_model is None:
            return None

        # Pass lead's search_term/description as query so candidates
        # are boosted by filename-token overlap (filename intelligence).
        _query = lead.search_term or lead.description or ""

        # Get candidates from document cards + inventory (highest salience first)
        candidates = self._matter_model.list_search_seed_docs(
            issue_id=lead.focus_issue_id,
            query=_query,
            limit=max_files,
        )
        if not candidates:
            return None

        return candidates

    def _select_deep_read_targets(
        self,
        state: InvestigationState,
        results,
        lead: Lead,
        focus_issue_id: Optional[str],
    ) -> list[str]:
        """Select which files from search results to deep-read.

        Prioritizes: issue-linked evidence gaps > new/unread docs >
        higher salience > unresolved card flags.
        Returns [] for assertion-backed result sets — those are already extracted.
        """
        # Assertion-backed results have line_num=0; skip deep-read
        if results.hits and all(h.line_num == 0 for h in results.hits):
            return []

        # Get files sorted by max hit score (existing behavior)
        top_files = sorted(
            results.by_file().keys(),
            key=lambda fp: max((h.score for h in results.by_file()[fp]), default=0),
            reverse=True,
        )

        # If we have document cards, re-rank by combining search score with card intelligence
        if self._matter_model is not None:
            scored: list[tuple[float, str]] = []
            for fp in top_files:
                search_score = max((h.score for h in results.by_file()[fp]), default=0)
                bonus = 0.0
                card = self._matter_model.get_document_card(relative_path=fp)
                if card:
                    # Boost docs with unresolved flags
                    flags = card.get("unresolved_flags") or []
                    if flags:
                        bonus += 0.3
                    # Boost operative documents
                    if card.get("operative_status") == "operative":
                        bonus += 0.2
                scored.append((search_score + bonus, fp))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_files = [fp for _, fp in scored]

        # Pairwise document lanes for comparison tasks (Codex #4):
        # Ensure at least one document from each role bucket is selected.
        _ql_dt = (state.query or "").lower()
        _is_comp_dt = any(w in _ql_dt for w in (
            "markup", "redline", "compare", "comparison", "deviation",
            "counterparty", "credit facility", "term sheet",
        ))
        if _is_comp_dt and len(top_files) > 1:
            _role_buckets: dict[str, list[str]] = {
                "original": [], "markup": [], "playbook": [], "projections": [], "other": [],
            }
            _role_keywords = {
                "original": ("original", "base", "initial", "agreed", "executed"),
                "markup": ("markup", "marked", "redline", "red-line", "counterparty", "lender", "revised"),
                "playbook": ("playbook", "negotiation", "instruction", "guideline", "partner"),
                "projections": ("projection", "financial", "model", "forecast", "budget", "pro forma"),
            }
            for fp in top_files:
                _fn = fp.lower()
                _assigned = False
                for role, kws in _role_keywords.items():
                    if any(k in _fn for k in kws):
                        _role_buckets[role].append(fp)
                        _assigned = True
                        break
                if not _assigned:
                    _role_buckets["other"].append(fp)

            _lane_selected: list[str] = []
            _seen = set()
            for role in ("original", "markup", "playbook", "projections"):
                if _role_buckets[role] and _role_buckets[role][0] not in _seen:
                    _lane_selected.append(_role_buckets[role][0])
                    _seen.add(_role_buckets[role][0])
            for fp in top_files:
                if fp not in _seen:
                    _lane_selected.append(fp)
                    _seen.add(fp)
            top_files = _lane_selected

        return top_files[:self.config.max_leads_per_level]

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

                # Build issue-aware queries with expansion
                queries = self._build_lead_queries(lead, lead.focus_issue_id)

                # Hot-path: if all documents already ingested, search cached
                # assertions first — avoids redundant raw file grep.
                # Hot-path: if all documents already ingested AND assertions
                # provide sufficient coverage (>=3 hits), use them instead of raw
                # grep. Below the threshold, fall through to repo search so we
                # don't suppress contradictory evidence from a single stale match.
                _ASSERTION_MIN_HITS = 3
                if state.findings.get("all_documents_ingested"):
                    cached = self._search_cached_assertions(
                        queries, issue_id=lead.focus_issue_id,
                    )
                    if cached and len(cached.hits) >= _ASSERTION_MIN_HITS:
                        results = cached
                        state.searches_performed += 1
                        self._emit_step(
                            state, StepType.FINDING,
                            f"Found {len(results.hits)} assertion matches for: {lead.description}",
                        )
                        await self._analyze_search_results(state, repo, results, lead)
                        _adp_post = getattr(state, "_matter_adapter", None)
                        if _adp_post is not None and _adp_post.is_stop_requested():
                            return
                        state.mark_lead_investigated(lead.id, f"Found {len(results.hits)} assertion matches")
                        return

                # Candidate-first: try memory-seeded file list before full repo
                candidates = self._candidate_files_for_lead(state, repo, lead)
                max_workers = min(4, max(1, self._doc_count))

                if candidates and len(queries) > 1:
                    # Stage 1: search only candidate files with expanded queries
                    results = repo.search_multi(
                        queries, file_paths=candidates, require_all=False,
                    )
                    state.searches_performed += 1
                    # Stage 2: if weak results, merge with full repo search
                    if len(results.hits) < 2:
                        fallback = repo.search(
                            queries[0], context_lines=3, max_workers=max_workers,
                        )
                        state.searches_performed += 1
                        # Merge: combine hits, dedup by (file_path, line_num)
                        seen = {(h.file_path, h.line_num) for h in results.hits}
                        merged_hits = list(results.hits)
                        for h in fallback.hits:
                            if (h.file_path, h.line_num) not in seen:
                                merged_hits.append(h)
                        results = type(results)(
                            query=queries[0],
                            hits=merged_hits,
                            files_searched=fallback.files_searched,
                            total_matches=len(merged_hits),
                        )
                elif candidates:
                    results = repo.search(
                        queries[0], context_lines=3, max_workers=max_workers,
                        file_paths=candidates,
                    )
                    state.searches_performed += 1
                    if len(results.hits) < 2:
                        fallback = repo.search(
                            queries[0], context_lines=3, max_workers=max_workers,
                        )
                        state.searches_performed += 1
                        seen = {(h.file_path, h.line_num) for h in results.hits}
                        merged_hits = list(results.hits)
                        for h in fallback.hits:
                            if (h.file_path, h.line_num) not in seen:
                                merged_hits.append(h)
                        results = type(results)(
                            query=queries[0],
                            hits=merged_hits,
                            files_searched=fallback.files_searched,
                            total_matches=len(merged_hits),
                        )
                else:
                    # No memory — full repo search (cold start)
                    results = repo.search(
                        queries[0], context_lines=3, max_workers=max_workers,
                    )
                    state.searches_performed += 1

                if not results.hits:
                    state.mark_lead_investigated(lead.id, "No results found")
                    # Record as a gap if this lead was targeting a specific issue (SO-7)
                    _adp = getattr(state, "_matter_adapter", None)
                    if _adp is not None and lead.focus_issue_id is not None:
                        from ..matter.enums import GapType
                        _adp.record_gap(
                            description=f"No documents found for search: '{queries[0]}'",
                            gap_type=GapType.MISSING_DOCUMENT,
                            expected_artifact=queries[0],
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

                # Do NOT mark investigated if stop fired inside the analysis — the lead
                # must remain pending so a resumed run can retry the full analysis.
                # (Mirrors the stop check at the top of this function.)
                _adp_post = getattr(state, "_matter_adapter", None)
                if _adp_post is not None and _adp_post.is_stop_requested():
                    return
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

        # Search-analysis cache (SO-1): same search term + same top-5 hits → skip FLASH call.
        # Cache key must cover everything that feeds the FLASH prompt: full search query,
        # full investigation query, full hypothesis, full results text, top-hit filenames,
        # AND a short hash of the prompt template itself so that prompt/model upgrades
        # automatically invalidate cached analysis from old template versions.
        import hashlib as _hl
        _focus_issue_id = lead.focus_issue_id if lead is not None else None
        _top_names = ",".join(sorted(h.filename for h in results.top(5)))
        # Build issue focus block + predicate allowlist in one call to avoid
        # a duplicate get_predicates() read on cache misses (perf fix).
        # _pred_allowlist: exact descriptions shown in Issue Focus — allowlist for resolution.
        # _pred_key_frag: included in cache key so changing open predicates invalidates cache.
        _issue_focus, _pred_allowlist = self._build_issue_focus_block(_focus_issue_id)
        _pred_key_frag = ",".join(_pred_allowlist)
        _analysis_key = _hl.sha256(
            f"{_ANALYZE_PROMPT_VER}\n{results.query}\n{state.query}\n{state.hypothesis or ''}"
            f"\n{_top_names}\n{results_text}\n{_focus_issue_id or ''}\n{_pred_key_frag}".encode()
        ).hexdigest()
        _cached_analysis = None
        if self._matter_model is not None:
            try:
                _cached_analysis = self._matter_model.cache.get("search_analysis", _analysis_key)
            except Exception:
                pass

        if _cached_analysis is not None:
            state.llm_calls_avoided += 1  # SO-1 telemetry: search-analysis cache hit
            analysis = _cached_analysis
        else:
            state.llm_calls_required += 1  # SO-1 telemetry: search-analysis cache miss
            # Re-check stop before the LLM call (SO-3 cooperative stop).
            _adp_pre = getattr(state, "_matter_adapter", None)
            if _adp_pre is not None and _adp_pre.is_stop_requested():
                return

            # Stage 1 — LITE extraction. Always fires. Reads the big
            # search_results input, emits compact structured output.
            # Codex review on b1ac54c: lightweight relevance hints
            # (query + hypothesis + first predicate line of the issue
            # focus) steer extraction toward investigation-relevant
            # facts without asking LITE to reason.
            _relevance_hint = "(none)"
            if _issue_focus and "Element to prove" in _issue_focus:
                for _ln in _issue_focus.splitlines():
                    if _ln.strip().startswith("Element to prove"):
                        _relevance_hint = _ln.strip()
                        break
            _extract_domain = self._resolve_active_domain(state)
            _extract_ex = _DOMAIN_EXTRACTION_EXAMPLES.get(
                _extract_domain, _DOMAIN_EXTRACTION_EXAMPLES["legal"]
            )
            stage1_prompt = EXTRACT_FINDINGS_PROMPT.format(
                query=state.query,
                hypothesis=state.hypothesis or "No hypothesis yet",
                relevance_hint=_relevance_hint,
                search_term=results.query,
                search_results=results_text,
                domain_subject_examples=_extract_ex["subject_examples"],
                domain_predicate_examples=_extract_ex["predicate_examples"],
                domain_object_examples=_extract_ex["object_examples"],
            )
            stage1_response = await self.client.complete(
                stage1_prompt,
                tier=ModelTier.LITE,
                json_mode=True,
                usage_label="search_extract",
                temperature=0.0,
            )
            stage1 = self._parse_json_safe(stage1_response, {
                "key_facts": [],
                "mentioned_leads": [],
                "mentioned_searches": [],
            })
            raw_facts = stage1.get("key_facts") or []
            raw_leads = stage1.get("mentioned_leads") or []
            raw_searches = stage1.get("mentioned_searches") or []

            # Short-circuit: if extraction found nothing worth reasoning
            # about, skip Stage 2 entirely. Saves the FLASH call on
            # low-signal searches (grep matches in boilerplate, etc.).
            has_signal = bool(raw_facts) or bool(raw_leads)
            if not has_signal:
                analysis = {
                    "key_facts": [],
                    "fact_relationships": [],
                    "new_leads": [],
                    "hypothesis_update": None,
                    "next_searches": raw_searches,
                    "predicates_satisfied": [],
                    "predicates_contested": [],
                }
            else:
                # Re-check stop between stages.
                if _adp_pre is not None and _adp_pre.is_stop_requested():
                    return

                # Stage 2 — FLASH reasoning. Compact input (Stage 1
                # output + matter context), no re-read of search_results.
                def _fmt_indexed(items, key=None):
                    if not items:
                        return "(none)"
                    lines = []
                    for i, item in enumerate(items):
                        if key is None:
                            lines.append(f"{i}: {item}")
                        else:
                            lines.append(f"{i}: {item.get(key, '')}")
                    return "\n".join(lines)

                # Defensive pre-normalization before Stage 2 and before
                # merge. Malformed items (non-dicts where dicts are
                # expected, non-numeric priorities, etc.) must NOT
                # abort the pipeline — the pre-split single-call code
                # silently tolerated junk, and so must we.
                normalized_facts = []
                for f in raw_facts:
                    if isinstance(f, dict):
                        normalized_facts.append(f)
                    elif f:
                        normalized_facts.append({"fact": str(f)})
                normalized_leads = []
                for item in raw_leads:
                    if isinstance(item, dict):
                        normalized_leads.append(item)
                    elif item:
                        normalized_leads.append({"desc": str(item)})
                normalized_searches = [str(s) for s in raw_searches if s]

                facts_block = _fmt_indexed(
                    [str(f.get("fact", "")) for f in normalized_facts],
                )
                leads_block = _fmt_indexed(normalized_leads, key="desc")
                searches_block = (
                    "\n".join(f"- {s}" for s in normalized_searches)
                    if normalized_searches else "(none)"
                )
                stage2_prompt = REASON_FINDINGS_PROMPT.format(
                    query=state.query,
                    hypothesis=state.hypothesis or "No hypothesis yet",
                    issue_focus=_issue_focus,
                    research_alignment_guidance=RESEARCH_ALIGNMENT_GUIDANCE,
                    facts_block=facts_block,
                    leads_block=leads_block,
                    searches_block=searches_block,
                )
                stage2_response = await self.client.complete(
                    stage2_prompt,
                    tier=ModelTier.FLASH,
                    json_mode=True,
                    usage_label="search_reason",
                    temperature=0.0,
                )
                stage2 = self._parse_json_safe(stage2_response, {
                    "fact_issue_relations": [],
                    "fact_relationships": [],
                    "hypothesis_update": None,
                    "predicates_satisfied": [],
                    "predicates_contested": [],
                    "lead_priorities": [],
                    "next_search_priorities": [],
                })

                # Merge Stage 1 extraction with Stage 2 reasoning into
                # the legacy `analysis` dict shape. All casts are guarded
                # so a malformed JSON-valid item can't abort the merge —
                # bad entries are silently dropped (same tolerance the
                # pre-split code had via its single _parse_json_safe).
                def _safe_int(val, default=-1):
                    try:
                        return int(val)
                    except (TypeError, ValueError):
                        return default

                def _safe_float(val, default=0.5):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return default

                relation_by_idx: dict[int, str] = {}
                for r in (stage2.get("fact_issue_relations") or []):
                    if not isinstance(r, dict):
                        continue
                    idx = _safe_int(r.get("fact_idx"), -1)
                    if idx < 0:
                        continue
                    rel = r.get("relation")
                    relation_by_idx[idx] = (
                        str(rel) if rel in ("supports", "attacks", "neutral") else "neutral"
                    )

                merged_facts = [
                    {**fact, "issue_relation": relation_by_idx.get(i, "neutral")}
                    for i, fact in enumerate(normalized_facts)
                ]

                priority_by_lead: dict[int, float] = {}
                for p in (stage2.get("lead_priorities") or []):
                    if not isinstance(p, dict):
                        continue
                    idx = _safe_int(p.get("lead_idx"), -1)
                    if idx < 0:
                        continue
                    priority_by_lead[idx] = _safe_float(p.get("priority"), 0.5)

                merged_leads = [
                    {**lead_dict, "priority": priority_by_lead.get(i, 0.5)}
                    for i, lead_dict in enumerate(normalized_leads)
                ]

                # next_searches: prefer Stage 2's ranked list; fall back
                # to Stage 1 if Stage 2 didn't rank any.
                ranked_pairs = [
                    (_safe_float(x.get("priority"), 0.0), str(x.get("term") or ""))
                    for x in (stage2.get("next_search_priorities") or [])
                    if isinstance(x, dict) and x.get("term")
                ]
                ranked_pairs.sort(key=lambda p: -p[0])
                next_searches = [term for _, term in ranked_pairs] or list(normalized_searches)

                analysis = {
                    "key_facts": merged_facts,
                    "fact_relationships": stage2.get("fact_relationships") or [],
                    "new_leads": merged_leads,
                    "hypothesis_update": stage2.get("hypothesis_update"),
                    "next_searches": next_searches,
                    "predicates_satisfied": stage2.get("predicates_satisfied") or [],
                    "predicates_contested": stage2.get("predicates_contested") or [],
                }

            # Cache the merged analysis for warm runs.
            if self._matter_model is not None:
                try:
                    _mh = state.cache_manifest_hash
                    if _mh:
                        self._matter_model.cache.put_brokered(
                            "search_analysis", _analysis_key, analysis,
                            manifest_hash=_mh,
                        )
                    else:
                        self._matter_model.cache.put("search_analysis", _analysis_key, analysis)
                except Exception:
                    pass

        # Initialize sentinel: facts actually persisted from this analysis pass.
        # Used by the predicate resolution block below; must always be bound.
        _search_assertion_ids: list[str] = []

        # Store key facts with per-fact source attribution (SO-5 provenance fix).
        # Facts from the LLM may be bare strings (legacy) or dicts with "fact" and
        # optional "source_file" keys. We use the per-fact source_file when present
        # so each fact is labeled and recorded against its actual source document
        # rather than always being attributed to the single top search hit.
        if analysis.get("key_facts"):
            # _infer_source_role is the module-level import; alias for readability here.
            _infer_role = _infer_source_role

            # Build a name→SearchHit lookup for all prompt-visible hits.
            # Keys: full file_path (always unique) + _unique_display_name suffix (cased
            # and lowercased).  _all_paths is deduplicated so the uniqueness check in
            # _unique_display_name is not confused by multiple hits from the same doc.
            _top_hits_for_lookup = list(results.top(10))
            _all_paths = list(dict.fromkeys(_h.file_path for _h in _top_hits_for_lookup))
            _hit_by_name: dict[str, object] = {}
            for _h in _top_hits_for_lookup:
                _hit_by_name[_h.file_path] = _h        # full path (always unique)
                # Register the same unique suffix the formatter showed the model
                _display = self._unique_display_name(_h.file_path, _all_paths)
                _hit_by_name[_display] = _h
                _hit_by_name[_display.lower()] = _h

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

            # Bare-string default: "neutral" — if the LLM returned a bare string without
            # issue_relation classification, we cannot infer the relation. Defaulting to
            # "supports" when issue-targeted inflates coverage with unclassified facts.
            # The dict-fact path already uses "neutral" when issue_relation is absent; this
            # is consistent with that. (SO-4 audit #021 finding)
            _bare_rel = "neutral"
            # facts_to_add: (text, src_label, doc_id, issue_relation, spo_dict|None)
            # spo_dict carries subject_ref_type/id, predicate_key, object_json for SO-2
            facts_to_add: list[tuple] = []
            for fact_item in analysis["key_facts"]:
                if isinstance(fact_item, str):
                    facts_to_add.append((fact_item, _fallback_src_label, _fallback_doc_id, _bare_rel, None))
                elif isinstance(fact_item, dict) and "fact" in fact_item:
                    fact_text = fact_item["fact"]
                    src_file = fact_item.get("source_file") or ""
                    _raw_rel = fact_item.get("issue_relation")
                    issue_rel = _raw_rel.lower().strip() if isinstance(_raw_rel, str) else "neutral"
                    if issue_rel not in ("supports", "attacks", "neutral"):
                        issue_rel = "neutral"
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
                    # Extract SPO triple when LLM provides it (SO-2 typed assertions)
                    _subj = fact_item.get("subject")
                    _pred = fact_item.get("predicate")
                    _obj = fact_item.get("object")
                    spo = None
                    if _subj or _pred or _obj:
                        spo = {
                            "subject_ref_type": "free_text" if _subj else None,
                            "subject_ref_id": str(_subj) if _subj else None,
                            "predicate_key": str(_pred).lower().replace(" ", "_") if _pred else None,
                            "object_json": json.dumps(str(_obj)) if _obj else None,
                        }
                    facts_to_add.append((fact_text, src_label, doc_id, issue_rel, spo))

            # SO-2 validation + retry: if primary extraction left any facts without SPO
            # triples, make one targeted FLASH retry to recover structured triples from
            # the already-extracted fact texts (no re-reading of source documents).
            # Threshold >= 1: fire for any unstructured fact, including small batches.
            # A single null-SPO fact is still a flat-fact violation; the FLASH retry is
            # cheap relative to the value of structured storage.
            if facts_to_add:
                _spo_count = sum(1 for _, _, _, _, _s in facts_to_add if _s is not None)
                if _spo_count < len(facts_to_add) and len(facts_to_add) >= 1:
                    self._emit_step(
                        state, StepType.REPLAN,
                        f"SPO extraction yielded {_spo_count}/{len(facts_to_add)} structured triples "
                        f"(search: '{results.query[:60]}'). Retrying for missing.",
                    )
                    _retry_texts = [txt for txt, _, _, _, _ in facts_to_add]
                    state.llm_calls_required += 1
                    _retry_spo = await self._retry_spo_extraction(_retry_texts)
                    if _retry_spo:
                        facts_to_add = [
                            (txt, lbl, doc, rel, _retry_spo.get(i) if spo is None else spo)
                            for i, (txt, lbl, doc, rel, spo) in enumerate(facts_to_add)
                        ]

            # Add to state with per-fact source-role prefix (SO-5)
            _new_facts = [f"[{lbl}] {txt}" for txt, lbl, _, _rel, _spo in facts_to_add]
            state.add_facts(_new_facts)
            if self.on_fact:
                for _f in _new_facts:
                    self.on_fact(_f)

            # Record into matter model with correct per-fact doc_id; collect assertion IDs
            # for graph-edge creation below (SO-2 assertion links in search analysis path).
            adapter = getattr(state, "_matter_adapter", None)
            _search_assertion_ids: list[str] = []
            if adapter is not None:
                issue_id = lead.focus_issue_id if lead is not None else None
                # Batch all facts into one outer transaction — inner per-fact transactions
                # become savepoints, collapsing N disk syncs into 1 (perf SO-1).
                # Use dict form for facts with SPO triples (SO-2), tuple form otherwise.
                _batch = []
                for fact_text, _lbl, doc_id, issue_rel, spo in facts_to_add:
                    if spo:
                        _batch.append({
                            "proposition_text": fact_text,
                            "document_id": doc_id,
                            "issue_link_type": issue_rel,
                            **spo,
                        })
                    else:
                        _batch.append((fact_text, doc_id, issue_rel))
                _search_assertion_ids = adapter.record_facts_batch(
                    _batch,
                    issue_id=issue_id,
                )
                if facts_to_add:
                    unique_docs = {d for _, _, d, _, _spo in facts_to_add}
                    adapter.log_step(
                        f"Recorded {len(facts_to_add)} facts from search: {results.query[:60]}",
                        why=f"Sources: {', '.join(sorted(unique_docs)[:3])}",
                    )
                # Build assertion dependency graph from LLM-identified relationships (SO-2).
                # Mirrors the deep-read path so warm runs (which skip deep reads) still produce
                # assertion edges from the search analysis pass.
                _rels = analysis.get("fact_relationships") or []
                for _rel in _rels[:5]:
                    if not isinstance(_rel, dict):
                        continue
                    _fi = _rel.get("from_idx")
                    _ti = _rel.get("to_idx")
                    _rt = _rel.get("relation", "")
                    if _rt and _rt not in _VALID_ASSERTION_LINK_TYPES:
                        # SO-3: invalid relation from LLM is dropped but must not be silently
                        # hidden.  Use debug-level Python logger to avoid spamming the user-
                        # visible reasoning ledger with LLM hallucinations.
                        logger.debug(
                            "Dropped invalid assertion relation '%s' from search analysis "
                            "(not in _VALID_ASSERTION_LINK_TYPES)", _rt
                        )
                    if (isinstance(_fi, int) and isinstance(_ti, int)
                            and 0 <= _fi < len(_search_assertion_ids)
                            and 0 <= _ti < len(_search_assertion_ids)
                            and _fi != _ti
                            and _search_assertion_ids[_fi]
                            and _search_assertion_ids[_ti]
                            and _search_assertion_ids[_fi] != _search_assertion_ids[_ti]
                            and _rt in _VALID_ASSERTION_LINK_TYPES):
                        adapter.record_assertion_link(
                            _search_assertion_ids[_fi],
                            _search_assertion_ids[_ti],
                            _rt,
                        )

        # MVP.2 SO-2: LLM-only paths cannot set issue_predicate.status='resolved'.
        # The prior implementation called IssueStore.resolve_predicate_by_description
        # on every predicate the LLM claimed to be satisfied, which let an
        # unverified AI output promote an element to resolved. That path is
        # removed. Assumption-gated blocking of the predicate is still
        # defensive (blocked is a narrowing transition, not an upgrade) and
        # stays in place until the human review queue lands in P0.3.
        _preds_satisfied = analysis.get("predicates_satisfied") or []
        if (isinstance(_preds_satisfied, list) and _focus_issue_id
                and self._matter_model is not None
                and any(_search_assertion_ids)
                and _pred_allowlist):
            _allowed = {
                d.strip('"').strip("'").strip().lower(): d
                for d in _pred_allowlist
            }
            for _ps in _preds_satisfied:
                if not isinstance(_ps, str):
                    continue
                _ps_key = _ps.strip().strip('"').strip("'").strip().lower()
                if _ps_key not in _allowed:
                    continue
                try:
                    _pred_rows = self._matter_model.issues.get_predicates(
                        _focus_issue_id, limit=20
                    )
                    _pred_match = next(
                        (p for p in _pred_rows
                         if (p.get("description") or "").strip().lower() == _ps_key),
                        None,
                    )
                    if _pred_match and self._matter_model.assumptions.has_blocking_assumptions(
                        "predicate", _pred_match["id"]
                    ):
                        self._matter_model.issues.set_predicate_status(
                            _pred_match["id"], "blocked"
                        )
                except Exception:
                    pass

        # Gap 3: handle contested predicates — LLM signals that evidence
        # supports both sides of an element.
        _preds_contested = analysis.get("predicates_contested") or []
        if (isinstance(_preds_contested, list) and _focus_issue_id
                and self._matter_model is not None and _pred_allowlist):
            _allowed_c = {
                d.strip('"').strip("'").strip().lower(): d
                for d in _pred_allowlist
            }
            for _pc in _preds_contested:
                if not isinstance(_pc, str):
                    continue
                _pc_key = _pc.strip().strip('"').strip("'").strip().lower()
                _orig_c = _allowed_c.get(_pc_key)
                if _orig_c:
                    try:
                        _pred_rows_c = self._matter_model.issues.get_predicates(
                            _focus_issue_id, limit=20
                        )
                        _pred_match_c = next(
                            (p for p in _pred_rows_c
                             if (p.get("description") or "").strip().lower() == _pc_key),
                            None
                        )
                        if _pred_match_c:
                            self._matter_model.issues.set_predicate_status(
                                _pred_match_c["id"], "contested"
                            )
                    except Exception:
                        pass

        # Lawyer-facing search analysis summary: what changed, what was found.
        _search_summary_parts = []
        if _search_assertion_ids:
            _n_recorded = sum(1 for a in _search_assertion_ids if a)
            _search_summary_parts.append(f"{_n_recorded} facts recorded")
        _n_satisfied = len([p for p in (_preds_satisfied or []) if isinstance(p, str)])
        _n_contested = len([p for p in (_preds_contested or []) if isinstance(p, str)])
        if _n_satisfied:
            _search_summary_parts.append(f"{_n_satisfied} elements satisfied")
        if _n_contested:
            _search_summary_parts.append(f"{_n_contested} elements contested")
        # Top source docs for the attorney
        _top_docs = [h.filename for h in results.top(3)]
        if _top_docs:
            _search_summary_parts.append(f"Sources: {', '.join(_top_docs)}")
        if _search_summary_parts:
            self._emit_step(
                state, StepType.FINDING,
                f"Search '{results.query[:50]}' — {'; '.join(_search_summary_parts)}",
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
        # SO-4 Leak-5 anchor tokens: parent's search term provides issue-specificity
        # signals.  A follow-on lead must share at least one meaningful (>3 char) token
        # with the parent to retain issue focus; otherwise it is demoted to neutral so it
        # cannot silently inflate issue coverage with generic searches.
        _anchor_tokens: "set[str]" = set()
        if lead is not None:
            _anchor_src = (lead.search_term or lead.description or "").lower()
            _anchor_tokens = {_w for _w in _anchor_src.split() if len(_w) > 3}
        for lead_data in analysis.get("new_leads", [])[:3]:
            if isinstance(lead_data, dict):
                desc = lead_data.get("description") or lead_data.get("desc")
                if desc:
                    if self._should_skip_follow_on_lead(desc, state):
                        continue
                    _validated_fid = _follow_on_issue_id
                    if _follow_on_issue_id and _anchor_tokens:
                        _desc_toks = {_w for _w in desc.lower().split() if len(_w) > 3}
                        if not (_anchor_tokens & _desc_toks):
                            _validated_fid = None  # demote — too generic to claim issue focus
                    state.add_lead(
                        description=desc,
                        source=f"Analysis of '{results.query}'",
                        priority=lead_data.get("priority", 0.5),
                        focus_issue_id=_validated_fid,
                    )

        # Convert next_searches (bare search terms from analysis) into leads.
        # These are lower priority than structured new_leads but still valuable
        # as targeted follow-up searches that maintain issue focus.
        for _ns in analysis.get("next_searches", [])[:6]:
            if isinstance(_ns, str) and _ns.strip():
                _ns_clean = _ns.strip()
                if self._should_skip_follow_on_lead(_ns_clean, state):
                    continue
                _validated_fid_ns = _follow_on_issue_id
                if _follow_on_issue_id and _anchor_tokens:
                    _ns_toks = {_w for _w in _ns_clean.lower().split() if len(_w) > 3}
                    if not (_anchor_tokens & _ns_toks):
                        _validated_fid_ns = None
                state.add_lead(
                    description=f"Follow-up search: {_ns_clean[:100]}",
                    source=f"Analysis of '{results.query}'",
                    priority=0.45,
                    search_term=_ns_clean,
                    focus_issue_id=_validated_fid_ns,
                )

        # Deep read top documents — use card-aware ranking when available (MEDIUM #5),
        # which boosts docs with unresolved flags and operative status.
        focus_issue_id = lead.focus_issue_id if lead is not None else None
        top_files = self._select_deep_read_targets(
            state, results, lead, focus_issue_id,
        )[:self.config.parallel_reads]
        if top_files:
            await self._batch_deep_read(state, repo, top_files, focus_issue_id=focus_issue_id)

    def _initial_deep_read_cap(self, state: InvestigationState, total_files: int) -> int:
        """Foreground cap for the first cold-path deep-read slice."""
        if total_files <= 0:
            return 0
        if self._query_requests_full_document_review(state.query):
            return total_files
        cap = max(1, int(self.config.max_initial_deep_read_documents or 1))
        mode = normalize_research_mode(getattr(state, "research_mode", None))
        if mode == ResearchMode.SIMPLE.value:
            cap = min(cap, 8)
        elif mode == ResearchMode.SEBIH_SPECIAL.value:
            cap = max(cap, 30)
        return min(total_files, cap)

    @staticmethod
    def _query_requests_full_document_review(query: str) -> bool:
        q = " ".join(str(query or "").lower().split())
        full_review_phrases = (
            "read every document",
            "read all documents",
            "read every file",
            "read all files",
            "ingest every document",
            "ingest all documents",
            "review every document",
            "review all documents",
            "review every file",
            "review all files",
            "summarize every document",
            "summarize all documents",
        )
        return any(phrase in q for phrase in full_review_phrases)

    @staticmethod
    def _query_is_finance_focused(query: str) -> bool:
        q = str(query or "").lower()
        finance_terms = (
            "finance",
            "financial",
            "financials",
            "revenue",
            "gross margin",
            "operating margin",
            "net income",
            "cash flow",
            "cashflow",
            "profit",
            "loss",
            "ebitda",
            "balance sheet",
            "statement of operations",
            "income statement",
            "10-k",
            "10k",
            "10-q",
            "10q",
            "sec filing",
            "annual report",
            "quarterly report",
            "earnings",
            "guidance",
            "arr",
        )
        return any(term in q for term in finance_terms)

    @staticmethod
    def _query_path_terms(query: str) -> set[str]:
        raw_terms = _re_date.split(r"[^a-z0-9]+", str(query or "").lower())
        stop = {
            "about", "after", "again", "against", "every", "file", "files",
            "from", "have", "into", "over", "that", "their", "there", "this",
            "what", "when", "where", "which", "with", "would", "your",
        }
        return {term for term in raw_terms if len(term) > 3 and term not in stop}

    @staticmethod
    def _is_simple_factual_lookup(query: str) -> bool:
        """Return True for short lookup questions that should finish on exact evidence."""
        q = " ".join(str(query or "").lower().split())
        if not q:
            return False
        starters = (
            "when did", "when was", "when is", "who did", "who was",
            "who is", "what is", "what was", "where is", "where was",
            "did ", "is ", "was ",
        )
        if not q.startswith(starters):
            return False
        words = q.split()
        if len(words) > 18:
            return False
        complex_terms = (
            "all ", "every ", "complete ", "compare", "comparison",
            "across", "analyze", "analysis", "validate", "draft",
            "memo", "risk", "risks", "issue list", "inventory",
            "explain in detail", "walk through", "summarize",
        )
        return not any(term in q for term in complex_terms)

    @staticmethod
    def _simple_lookup_anchor_terms(query: str) -> set[str]:
        """Extract rare-ish anchor terms for simple lookups.

        These are the terms that should dominate filename scoring and
        determine whether gathered evidence actually touched the target.
        """
        tokens = [
            t for t in _re_engine.findall(r"[a-z0-9][a-z0-9'-]{2,}", str(query or "").lower())
        ]
        stop = {
            "about", "after", "again", "against", "answer", "asked",
            "company", "confirm", "data", "date", "details", "did",
            "dog", "does", "employee", "employer", "event", "file",
            "files", "find", "from", "have", "into", "join", "joined",
            "joins", "joining", "list", "mention", "mentioned",
            "name", "person", "record", "records", "say", "says",
            "source", "sources", "speaker", "talk", "tell", "that",
            "their", "there", "this", "what", "when", "where", "which",
            "with", "work", "worked", "working",
        }
        anchors = {t.strip("-'") for t in tokens if len(t.strip("-'")) > 3 and t not in stop}
        return anchors

    @staticmethod
    def _evidence_has_date_signal(text: str) -> bool:
        if not text:
            return False
        if _re_engine.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
            return True
        if _re_engine.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text):
            return True
        month = (
            "january|february|march|april|may|june|july|august|"
            "september|october|november|december|jan\\.?|feb\\.?|"
            "mar\\.?|apr\\.?|jun\\.?|jul\\.?|aug\\.?|sep\\.?|"
            "sept\\.?|oct\\.?|nov\\.?|dec\\.?"
        )
        return bool(_re_engine.search(rf"\b({month})\s+\d{{1,2}},?\s+\d{{4}}\b", text, _re_engine.I))

    def _simple_lookup_answer_satisfied(
        self,
        state: InvestigationState,
    ) -> tuple[bool, str]:
        if not self._is_simple_factual_lookup(state.query):
            return False, ""
        anchors = self._simple_lookup_anchor_terms(state.query)
        if not anchors:
            return False, ""

        evidence_units: list[str] = []
        for fact in state.findings.get("accumulated_facts") or []:
            if isinstance(fact, dict):
                evidence_units.append(json.dumps(fact, sort_keys=True))
            else:
                evidence_units.append(str(fact))
        for citation in state.citations or []:
            evidence_units.append(
                " ".join(
                    str(part or "")
                    for part in (
                        getattr(citation, "document", ""),
                        getattr(citation, "text", ""),
                        getattr(citation, "context", ""),
                        getattr(citation, "relevance", ""),
                    )
                )
            )
        if not evidence_units:
            return False, ""

        matching_units = [
            unit for unit in evidence_units
            if any(anchor in unit.lower() for anchor in anchors)
        ]
        if not matching_units:
            return False, ""

        asks_when = " ".join(str(state.query or "").lower().split()).startswith("when ")
        if asks_when:
            # The exact-name source can arrive as a citation while the date sits
            # in an extracted fact from the same batch. Requiring both signals
            # prevents bare name hits from stopping date lookups too early.
            combined = "\n".join(evidence_units)
            if not self._evidence_has_date_signal(combined):
                return False, ""

        return True, (
            "Simple factual lookup satisfied by exact evidence for "
            f"{', '.join(sorted(anchors)[:3])}"
        )

    def _should_skip_follow_on_lead(
        self,
        candidate: str,
        state: InvestigationState,
    ) -> bool:
        """Drop low-value negative leads from irrelevant docs on simple lookups."""
        if not self._is_simple_factual_lookup(state.query):
            return False
        text = " ".join(str(candidate or "").lower().split())
        if not text:
            return True
        negative_irrelevance = (
            "cannot be answered from the provided document",
            "cannot be answered from this document",
            "cannot be answered by this document",
            "does not contain any information",
            "does not mention",
            "not mentioned",
            "no individual by that name",
            "query asks about",
            "provided document pages",
        )
        return any(phrase in text for phrase in negative_irrelevance)

    @staticmethod
    def _path_has_any(path_lower: str, needles: tuple[str, ...]) -> bool:
        return any(needle in path_lower for needle in needles)

    def _score_initial_deep_read_file(
        self,
        state: InvestigationState,
        file_info: Any,
        *,
        query_terms: set[str],
        simple_lookup_terms: set[str],
        plan_doc_types: list[str],
        plan_folders: list[str],
        target_documents: set[str],
        finance_focused: bool,
    ) -> tuple[float, list[str]]:
        """Score a file using only cheap foreground path/name signals."""
        from ..core.search import get_document_priority

        rel_path = str(getattr(file_info, "relative_path", "") or "")
        filename = str(getattr(file_info, "filename", "") or rel_path)
        path_lower = rel_path.replace("\\", "/").lower()
        name_lower = filename.lower()
        score = float(get_document_priority(filename))
        reasons = ["base document-priority score"]

        if path_lower in target_documents or name_lower in target_documents:
            score += 100.0
            reasons.append("orientation target document")

        for term in sorted(simple_lookup_terms):
            if term in path_lower or term in name_lower:
                score += 150.0
                reasons.append(f"simple lookup exact path match '{term}'")
                break

        for term in sorted(query_terms):
            if term in path_lower or term in name_lower:
                score += 1.75
                reasons.append(f"path matches query term '{term}'")
                break

        for rank, folder in enumerate(plan_folders[:8]):
            folder = folder.strip().replace("\\", "/").lower()
            if folder and folder in path_lower:
                score += max(0.4, 2.0 - rank * 0.15)
                reasons.append(f"orientation folder '{folder}'")
                break

        for rank, doc_type in enumerate(plan_doc_types[:8]):
            doc_type = doc_type.strip().lower()
            if doc_type and doc_type in path_lower:
                score += max(0.3, 1.5 - rank * 0.12)
                reasons.append(f"orientation document type '{doc_type}'")
                break

        if finance_focused:
            annual = self._path_has_any(
                path_lower,
                ("10-k", "10k", "annual-report", "annual_report", "annual report"),
            )
            quarterly = self._path_has_any(
                path_lower,
                ("10-q", "10q", "quarterly-report", "quarterly_report"),
            )
            earnings = self._path_has_any(
                path_lower,
                (
                    "earnings",
                    "ex-99",
                    "ex99",
                    "investor-relations",
                    "investor_relations",
                ),
            )
            if annual:
                score += 6.0
                reasons.append("finance query: annual/10-K source")
            if quarterly:
                score += 5.0
                reasons.append("finance query: quarterly/10-Q source")
            if earnings:
                score += 2.5
                reasons.append("finance query: earnings/investor exhibit")
            if self._path_has_any(
                path_lower,
                ("financial", "finance", "mda", "md&a"),
            ):
                score += 1.5
                reasons.append("finance query: financial path signal")
            if self._path_has_any(path_lower, ("8-k", "8k")) and not earnings:
                score += 0.3
                reasons.append("finance query: generic 8-K is secondary")
            if self._path_has_any(
                path_lower,
                ("def 14a", "def14a", "proxy", "form-4", "form_4", "13g", "s-8"),
            ):
                score -= 2.0
                reasons.append("finance query: lower-value filing family")
            if (
                self._path_has_any(
                    path_lower,
                    ("press-release", "press_release", "press/"),
                )
                and not earnings
            ):
                score -= 1.0
                reasons.append("finance query: generic press material")

        return score, reasons

    def _select_initial_deep_read_files(
        self,
        state: InvestigationState,
        files: list[Any],
    ) -> list[Any]:
        """Select the foreground deep-read slice from path/name signals.

        Large corpora should not cold-read every new file before the first
        answer. This keeps broad corpus/wiki maintenance separate from the
        user-facing foreground path.
        """
        if not files:
            return []
        cap = self._initial_deep_read_cap(state, len(files))
        if cap >= len(files):
            state.findings["initial_deep_read_selection"] = {
                "mode": "all_files",
                "selected_count": len(files),
                "skipped_count": 0,
                "total_new_files": len(files),
            }
            return list(files)

        plan = state.findings.get("initial_plan") or {}
        plan_doc_types = [
            str(item).lower() for item in (plan.get("document_priority") or [])
            if isinstance(item, str) and item.strip()
        ]
        plan_folders = [
            str(item).lower() for item in (plan.get("relevant_folders") or [])
            if isinstance(item, str) and item.strip()
        ]
        target_documents = {
            str(item).replace("\\", "/").lower()
            for item in (plan.get("target_documents") or [])
            if isinstance(item, str) and item.strip()
        }
        query_terms = self._query_path_terms(state.query)
        simple_lookup_terms = (
            self._simple_lookup_anchor_terms(state.query)
            if self._is_simple_factual_lookup(state.query)
            else set()
        )
        finance_focused = self._query_is_finance_focused(state.query)

        scored: list[tuple[float, str, Any, list[str]]] = []
        for file_info in files:
            score, reasons = self._score_initial_deep_read_file(
                state,
                file_info,
                query_terms=query_terms,
                simple_lookup_terms=simple_lookup_terms,
                plan_doc_types=plan_doc_types,
                plan_folders=plan_folders,
                target_documents=target_documents,
                finance_focused=finance_focused,
            )
            path_key = str(getattr(file_info, "relative_path", "") or "").lower()
            scored.append((score, path_key, file_info, reasons))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[:cap]
        selected_files = [item[2] for item in selected]
        state.findings["initial_deep_read_selection"] = {
            "mode": "path_scored",
            "selected_count": len(selected_files),
            "skipped_count": max(0, len(files) - len(selected_files)),
            "total_new_files": len(files),
            "cap": cap,
            "finance_focused": finance_focused,
            "simple_lookup_terms": sorted(simple_lookup_terms),
            "selected_paths": [
                str(getattr(item[2], "relative_path", "") or "") for item in selected
            ],
            "selection_reasons": {
                str(getattr(item[2], "relative_path", "") or ""): item[3][:4]
                for item in selected[:10]
            },
        }
        return selected_files

    async def _ingest_documents(
        self, state: InvestigationState, repo: MatterRepository
    ):
        """Phase 1.5: Read documents before searching.

        Simultaneously checks existing intelligence in .irys/ AND evaluates
        filenames against the query to decide what to read and in what order.
        Already-ingested files are skipped. New/changed files are scored by
        query relevance (filename match + document type priority) so the most
        informative documents get read first.
        """
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is not None and _adapter.is_stop_requested():
            return

        all_files = repo.list_files()
        if not all_files:
            self._emit_step(state, StepType.READING, "No documents found in folder")
            return

        # Check existing intelligence — which files are already ingested?
        ingested: set[str] = set()
        if self._matter_model is not None:
            ingested = set(self._matter_model.inventory.get_ingested_paths())

        new_files = [f for f in all_files if str(f.relative_path) not in ingested]
        cached_count = len(all_files) - len(new_files)
        selected_new_files = self._select_initial_deep_read_files(state, new_files)
        selected_new_paths = {str(f.relative_path) for f in selected_new_files}

        # Phase 1: Query-agnostic profiling for docs that haven't been profiled.
        # This populates document cards (type, source role, operative status)
        # WITHOUT needing a query or issue context (Priority 1: cold-path split).
        #
        # Runs even when new_files is empty — handles historical backfill for
        # existing DBs upgraded to v42 whose documents are already ingested but
        # lack document cards (Codex Tier 1 HIGH: backfill path).
        unprofiled: list[str] = []
        if self._matter_model is not None:
            _mm = self._matter_model
            # Lightweight pre-registration: create inventory rows for new files
            # using os.stat (no full-file read). SHA is computed later during
            # deep-read to avoid redundant I/O (Codex Tier 1 MEDIUM: perf).
            import os as _os
            for nf in new_files:
                _nf_path = str(nf.relative_path)
                _nf_abs = Path(repo.base_path) / _nf_path
                try:
                    _st = _nf_abs.stat()
                    _mm.inventory.upsert(
                        relative_path=_nf_path,
                        sha256="pending",  # placeholder — updated in _deep_read_document
                        size_bytes=_st.st_size,
                        file_type=Path(_nf_path).suffix.lstrip(".") or None,
                    )
                except Exception:
                    pass  # unreadable files will be skipped during profiling too

            unprofiled_rows = _mm.list_documents_needing_profile(limit=200)
            # Foreground profiling follows the same selected document slice as
            # foreground deep-read. Broad corpus-card maintenance belongs in a
            # background/batch path, not in the user's first answer latency.
            if selected_new_paths:
                unprofiled = [
                    r["relative_path"]
                    for r in unprofiled_rows
                    if r["relative_path"] in selected_new_paths
                ]
            else:
                # Historical backfill for already-ingested docs stays bounded.
                cap = self._initial_deep_read_cap(state, len(unprofiled_rows))
                unprofiled = [r["relative_path"] for r in unprofiled_rows[:cap]]

        if unprofiled:
            self._emit_step(
                state, StepType.READING,
                f"Profiling {len(unprofiled)} document{'s' if len(unprofiled) != 1 else ''}"
                + " — classifying types, roles, and structure",
            )
            await self._batch_profile(state, repo, unprofiled)

        if not new_files:
            self._emit_step(
                state, StepType.READING,
                f"All {len(all_files)} documents already ingested — using cached intelligence",
            )
            state.findings["all_documents_ingested"] = True
            return

        # Use only the foreground-selected slice for the first query-coupled read.
        file_paths = [str(f.relative_path) for f in selected_new_files]
        skipped_count = max(0, len(new_files) - len(selected_new_files))
        if skipped_count:
            self._emit_step(
                state,
                StepType.READING,
                f"Foreground selected {len(file_paths)} of {len(new_files)} new documents "
                "by path/name signals; deferred the rest for targeted search or background maintenance",
            )

        # Phase 2: Query-coupled deep read for fact extraction.
        # Only process docs that still need evidence extraction for current query.
        self._emit_step(
            state, StepType.READING,
            f"Reading {len(file_paths)} document{'s' if len(file_paths) != 1 else ''}"
            + (f" ({cached_count} already cached)" if cached_count else "")
            + f" — extracting evidence relevant to your query",
        )

        # Use the weakest issue (from orientation) as focus for fact extraction
        ctx = _adapter.get_context() if _adapter is not None else None
        focus_issue_id = ctx.weakest_issue_id if ctx else None

        await self._batch_deep_read(
            state, repo, file_paths, focus_issue_id=focus_issue_id
        )
        # Only mark fully ingested if the run was not stopped mid-batch
        _post_adapter = getattr(state, "_matter_adapter", None)
        if _post_adapter is None or not _post_adapter.is_stop_requested():
            state.findings["all_documents_ingested"] = skipped_count == 0

    async def _batch_profile(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_paths: list[str],
    ):
        """Profile multiple documents with controlled parallelism (query-agnostic)."""
        if not file_paths:
            return
        _adapter = getattr(state, "_matter_adapter", None)
        sem = asyncio.Semaphore(self.config.parallel_reads)

        async def limited_profile(fp: str):
            # Cooperative stop check (SO-3) before each profile
            if _adapter is not None and _adapter.is_stop_requested():
                return
            async with sem:
                return await self._profile_document(state, repo, fp)

        tasks = [limited_profile(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Document profile failed: {file_paths[i]}: {result}")

    async def _profile_document(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_path: str,
    ):
        """Query-agnostic document profiling: classify what a document IS.

        This is the first phase of the cold-path split (Priority 1). It reads
        the document and extracts structural metadata (type, source role,
        operative status, signatories, purpose) WITHOUT any query or issue
        context. The results are persisted as a document card.

        Unlike _deep_read_document which extracts issue-specific evidence,
        this only builds the document's identity in the matter model.
        """
        _mm = self._matter_model
        if _mm is None:
            return

        # Cooperative stop check (SO-3)
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is not None and _adapter.is_stop_requested():
            return

        _fp = Path(file_path)
        try:
            _rel_path = str(_fp.relative_to(repo.base_path))
        except ValueError:
            _rel_path = file_path

        # Skip if already profiled
        row = _mm.db.execute(
            "SELECT id, maintenance_status FROM document_inventory WHERE matter_id = ? AND relative_path = ?",
            (_mm.matter_id, _rel_path),
        ).fetchone()
        if row is None:
            return
        if row["maintenance_status"] not in ("pending",):
            return

        doc_id = row["id"]

        # Helper: ensure a minimal card exists (never overwrite rich cards).
        def _ensure_fallback_card():
            if _mm.document_cards.get_by_doc_id(doc_id) is None:
                _mm.document_cards.upsert(doc_id, doc_type="unknown")
                from ..core.search import get_document_priority
                _sal = min(1.0, get_document_priority(_rel_path) / 2.0)
                _mm.inventory.set_salience(doc_id, _sal)

        # --- Phase A: Read document ---
        # Status stays 'pending' during read. If read fails (transient I/O,
        # locked file, OR permanent parse error), the doc stays pending and
        # retries next run. Read failures are cheap (no LLM cost) and the
        # 200-doc batch cap prevents queue starvation.
        try:
            content = repo.read(file_path)
        except Exception as e:
            logger.debug("Profile read failed for %s: %s", _rel_path, e)
            try:
                _ensure_fallback_card()
            except Exception:
                pass
            # Leave as 'pending' — retries next run. No LLM cost incurred.
            return

        if not content or not content.full_text:
            _ensure_fallback_card()
            _mm.inventory.mark_profile_complete(doc_id)
            return

        # --- Phase B: LLM classification (mark started to prevent dupes) ---
        _mm.inventory.mark_profile_started(doc_id)
        try:
            excerpt = content.full_text[:3000]
            prompt = f"""Classify this document. Respond in JSON only.

Document: {_fp.name}
Content (excerpt):
{excerpt}

Privilege classification guidance (MVP.4 SO-5):
- privilege_flag=true when the document appears to be restricted or
  privileged (e.g. internal strategy memo, confidential correspondence,
  privileged work-product, material non-public information).
- privilege_flag=unknown when the document has attributes suggesting
  restricted handling but classification cannot be reliably determined
  (ambiguous internal memo, unclear provenance, unknown participants).
  Unknown is treated as contained in clean mode until human review.
- privilege_flag=false only when the document is plainly unrestricted
  (signed agreement, public filing, published report, third-party
  invoice, authoritative reference).

Return:
{{
    "doc_type": "contract|filing|correspondence|invoice|order|memo|report|notice|exhibit|other",
    "doc_subtype": "specific subtype (e.g. services_agreement, demand_letter, audit_report)",
    "title": "descriptive title",
    "doc_source_role": "advocacy|operative|authoritative|procedural|informal|draft|post_hoc|unknown",
    "author": "author name or null",
    "sender": "sender or null",
    "recipient": "recipient or null",
    "creation_date": "YYYY-MM-DD or null",
    "effective_date": "YYYY-MM-DD or null",
    "operative_status": "operative|superseded|draft|expired|disputed|unknown",
    "purpose": "one-sentence description (max 80 chars)",
    "rhetorical_posture": "neutral|adversarial|cooperative|protective|informational",
    "privilege_flag": "true|false|unknown",
    "privilege_basis": "one sentence explaining why, or null",
    "unresolved_flags": ["any open questions"]
}}"""

            state.llm_calls_required += 1
            response = await self.client.complete(
                prompt,
                tier=ModelTier.LITE,
                json_mode=True,
                usage_label="document_profile",
            )
            analysis = self._parse_json_safe(response, {
                "doc_type": "other",
                "doc_source_role": "unknown",
                "operative_status": "unknown",
            })

            _mm.upsert_document_profile(
                relative_path=_rel_path,
                analysis=analysis,
                run_id=getattr(state, "_run_id", None),
            )

            # Emit a brief profiling trace
            _doc_type = analysis.get("doc_type", "unknown")
            _purpose = analysis.get("purpose", "")
            self._emit_step(
                state, StepType.READING,
                f"Profiled {_fp.name}: {_doc_type}"
                + (f" — {_purpose[:60]}" if _purpose else ""),
            )

        except Exception as e:
            logger.debug("Profile LLM failed (transient) for %s: %s", _rel_path, e)
            try:
                _ensure_fallback_card()
            except Exception:
                pass
            # Reset to pending so transient LLM/network failures retry next
            # run. Deterministic read failures are handled above as terminal.
            _mm.db.execute(
                "UPDATE document_inventory SET maintenance_status='pending' WHERE id=?",
                (doc_id,),
            )

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
        _owns_in_progress = False  # track if THIS coroutine added the marker
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
            _owns_in_progress = True

            if _mm is not None:
                import hashlib as _hl
                try:
                    # Compute sha256 BEFORE the hot-path check so that content changes
                    # at the same path are always detected via upsert's mismatch logic.
                    # Reading raw bytes is cheap (no PDF parsing); we avoid that with repo.read().
                    _abs_fp = (Path(repo.base_path) / file_path) if not _fp.is_absolute() else _fp
                    _raw = _abs_fp.read_bytes()
                    _sha = _hl.sha256(_raw).hexdigest()
                    # Adversarial #8 fix: capture the OLD hash BEFORE
                    # inventory.upsert() overwrites it. Previously we
                    # called update_hash() after upsert and
                    # update_hash saw old==new and reported no change
                    # — so mark_document_stale never fired and the
                    # whole P0.4 invalidation chain was dead on the
                    # production path.
                    _pre_upsert_row = _mm.db.execute(
                        "SELECT id, sha256 FROM document_inventory "
                        "WHERE matter_id=? AND relative_path=?",
                        (_mm.matter_id, _rel_path),
                    ).fetchone()
                    _pre_upsert_sha: Optional[str] = (
                        _pre_upsert_row["sha256"] if _pre_upsert_row else None
                    )
                    _inv_id, _ = _mm.inventory.upsert(
                        relative_path=_rel_path,
                        sha256=_sha,
                        size_bytes=len(_raw),
                        file_type=_fp.suffix.lstrip(".") or None,
                    )
                    _inventory_doc_id = _inv_id
                    # Hash-change invalidation: compare pre-upsert to
                    # new hash. A flip between two real hashes fans
                    # out to mark_document_stale. The "pending"
                    # placeholder case (first real hash after
                    # pending) is not a content change.
                    try:
                        _had_real_old = bool(
                            _pre_upsert_sha and _pre_upsert_sha != "pending"
                        )
                        _hash_really_changed = (
                            _had_real_old
                            and _sha != "pending"
                            and _pre_upsert_sha != _sha
                        )
                        if _hash_really_changed:
                            _mm.mark_document_stale(
                                _inv_id,
                                reason=(
                                    f"document_hash_changed:"
                                    f"{(_pre_upsert_sha or '')[:12]}->{_sha[:12]}"
                                ),
                            )
                    except Exception as _hash_err:
                        logger.debug(
                            "P0.4 hash-change invalidation failed for %s: %s",
                            _rel_path, _hash_err,
                        )

                    # SO-1: Operative version enforcement.
                    # If this document is superseded by a newer version that is
                    # already fully ingested, skip redundant cold-path analysis and
                    # prefer the operative (HEAD) document.  This ensures reasoning
                    # reads the binding version, not an earlier draft.
                    try:
                        _operative_id = _mm.inventory.get_operative_version(_inventory_doc_id)
                        if _operative_id != _inventory_doc_id:
                            _op_row = _mm.inventory.get_doc_row(_operative_id)
                            _op_label = (_op_row["relative_path"] if _op_row else _operative_id)
                            self._emit_step(
                                state, StepType.READING,
                                f"SO-1: {_fp.name} is superseded — operative version: {_op_label}",
                            )
                            if not hasattr(state, "_superseded_docs"):
                                state._superseded_docs = {}
                            state._superseded_docs[_inventory_doc_id] = _operative_id
                            # If the operative version is already ingested, skip deep
                            # analysis of this superseded document entirely.
                            if _op_row and _op_row.get("ingest_status") == "complete":
                                state.documents_read += 1
                                state.documents_from_cache += 1
                                return
                    except Exception as _op_err:
                        logger.debug(
                            "Operative version check failed for %s: %s",
                            _rel_path, _op_err,
                        )

                    if _mm.inventory.is_ingested(_rel_path):
                        # HOT PATH: sha256 verified current; already fully ingested in a prior run.
                        # Assertion-to-issue linking for hot-path docs is intentionally omitted:
                        # bulk-linking all assertions to a new issue would inflate coverage metrics
                        # with unfiltered associations. False coverage masks gaps; gaps trigger
                        # targeted retrieval (SO-4). Only cold-path LLM analysis produces
                        # semantically filtered assertion-issue links.
                        state.documents_read += 1
                        state.documents_from_cache += 1  # SO-1: count hot-path hits
                        state.llm_calls_avoided += 1  # SO-1 telemetry: doc read LLM call avoided
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

            # Re-check stop before the LITE LLM call (SO-3 cooperative stop).
            # Increment documents_read AFTER the stop check so the counter only
            # reflects documents that were fully analyzed, not files we opened and skipped.
            _adp_dr = getattr(state, "_matter_adapter", None)
            if _adp_dr is not None and _adp_dr.is_stop_requested():
                return None

            state.documents_read += 1
            state.llm_calls_required += 1  # SO-1 telemetry: cold-path doc read

            _excerpt_chars = min(len(doc.full_text), 500_000)
            content = doc.get_excerpt(_excerpt_chars)

            _domain = self._resolve_active_domain(state)
            _vocab = _DOMAIN_DEEP_READ_VOCABULARY.get(_domain, _DOMAIN_DEEP_READ_VOCABULARY["legal"])
            _dr_ex = _DOMAIN_DEEP_READ_EXAMPLES.get(_domain, _DOMAIN_DEEP_READ_EXAMPLES["legal"])

            _is_mna = self._is_mna_change_control_task(state.query)
            _mna_section = _MNA_DEEP_READ_SECTION if _is_mna else ""
            _txn_ctx = (
                '"transaction_context": {{\n'
                '        "structure": "reverse triangular merger|forward merger|asset purchase|stock purchase|other|null",\n'
                '        "target": "target company name or null",\n'
                '        "acquirer": "acquiring company name or null",\n'
                '        "merger_sub": "merger subsidiary name or null",\n'
                '        "parent": "ultimate parent entity name or null",\n'
                '        "source_section": "Section or recital where transaction structure is described, or null"\n'
                '    }},\n    '
            ) if _is_mna else ""

            # Issue-specific deep read sections (Codex #2)
            _ql_dr = (state.query or "").lower()
            _is_comparison_dr = any(w in _ql_dr for w in (
                "markup", "redline", "compare", "comparison", "deviation",
                "counterparty", "credit facility", "credit agreement",
                "term sheet", "loan agreement", "commitment letter",
            ))
            _is_regulatory_dr = any(w in _ql_dr for w in (
                "antitrust", "hsr", "merger review", "regulatory", "compliance",
                "market share", "hhi", "competitive effects",
            ))
            _task_section = ""
            if _is_comparison_dr:
                _task_section = _COMPARISON_DEEP_READ_SECTION
            elif _is_regulatory_dr:
                _task_section = _REGULATORY_DEEP_READ_SECTION

            _cross_ref_ctx = ""
            _existing_facts = state.findings.get("accumulated_facts", [])
            if _existing_facts and len(_existing_facts) >= 3:
                if _is_comparison_dr:
                    _provision_facts = [f for f in _existing_facts if "[PROVISION]" in f]
                    _other_facts = [f for f in _existing_facts if "[PROVISION]" not in f][-40:]
                    _recent = _provision_facts + _other_facts
                else:
                    _recent = _existing_facts[-60:]
                _cross_ref_text = "\n".join(f"- {f[:250]}" for f in _recent)
                if len(_cross_ref_text) > 18000:
                    _cross_ref_text = _cross_ref_text[:18000] + "\n... (truncated)"
                _cross_ref_ctx = (
                    "\n\nCROSS-REFERENCE CONTEXT (facts already extracted from other documents):\n"
                    "Use these to identify CONNECTIONS, CONTRADICTIONS, and MISSING details.\n"
                    "When this document references the same terms, amounts, or provisions as below,\n"
                    "extract the EXACT values from THIS document for comparison.\n"
                    + _cross_ref_text
                    + "\n"
                )

            # Inject tracked change manifest for comparison tasks
            _tc_manifest = ""
            if _is_comparison_dr and hasattr(doc, "tracked_changes") and doc.tracked_changes:
                _tc_manifest = doc.get_tracked_change_manifest()
                if _tc_manifest:
                    content = _tc_manifest + "\n\n" + content

            # Build enhanced focus for comparison/regulatory tasks
            _base_focus = state.hypothesis or state.query
            if _is_comparison_dr:
                _base_focus = (
                    f"{_base_focus}\n\n"
                    "EXTRACTION PRIORITY: For every provision in this document, extract the EXACT "
                    "numeric value (not 'changed' or 'modified' — the actual number/percentage/threshold). "
                    "If this document contains BOTH original and proposed values, extract BOTH. "
                    "Pay special attention to: tier breakpoints in grids, step-down schedules with dates, "
                    "dollar caps on baskets, exact ratio thresholds, add-back caps with both annual and "
                    "lifetime limits, and any provisions that were ADDED or DELETED entirely.\n"
                    "TRACKED CHANGES: Look for [DELETED: ...] and [ADDED: ...] markers throughout the document. "
                    "These indicate specific text changes. [DELETED: X] means X was the original text that was "
                    "removed; [ADDED: Y] means Y is the new text that was inserted. EVERY such marker pair "
                    "represents a change that must be captured as a provision_comparison entry."
                )
            elif _is_regulatory_dr:
                _base_focus = (
                    f"{_base_focus}\n\n"
                    "EXTRACTION PRIORITY: Extract ALL quantitative data for competitive analysis — "
                    "EVERY market, EVERY entity, EVERY data point. Partial extraction is failure.\n"
                    "For market shares: exact percentages per company per geographic market (MSA). "
                    "If there are 9 MSAs, extract data for all 9.\n"
                    "For pricing: EVERY quote about competitive pricing, undercutting, margin impact, "
                    "or market disruption — with exact quote text, speaker name, and section reference.\n"
                    "For HHI: post-merger HHI AND delta for EVERY geographic market. "
                    "All company shares needed to compute HHI (share² × 10000 summed).\n"
                    "For hot documents: EVERY internal quote that could be used adversarially — "
                    "board presentations, strategy memos, emails — with speaker, section, slide number.\n"
                    "For remedies: EVERY divestiture candidate facility by name, location, and revenue. "
                    "Potential buyers by name. Contractual caps, exclusions, and limitations.\n"
                    "For deal timeline: EVERY date — signing, filing, waiting period, outside date, "
                    "extensions, fund term expirations, integration milestones.\n"
                    "For contractual provisions: exact dollar amounts for breakup fees, divestiture caps, "
                    "and any exclusions (e.g., ASU exclusion from divestitures)."
                )

            prompt = DEEP_READ_PROMPT.format(
                filename=doc.filename,
                page_range=f"1-{doc.page_count}",
                content=content,
                query=state.query,
                focus=_base_focus,
                domain_vocabulary=_vocab,
                mna_section=_mna_section + _task_section + _cross_ref_ctx,
                transaction_context_schema=_txn_ctx,
                domain_deep_read_examples=_dr_ex["deep_read_examples"],
                domain_numeric_subjects=_dr_ex["numeric_subjects"],
                domain_numeric_subject_id_example=_dr_ex["numeric_subject_id_example"],
            )

            from ..core.search import get_document_priority
            _doc_priority = get_document_priority(doc.filename)
            _read_tier = ModelTier.FLASH if _doc_priority >= 1.3 else ModelTier.LITE
            response = await self.client.complete(
                prompt,
                tier=_read_tier,
                json_mode=True,
                usage_label="document_deep_read",
                temperature=0.0,
            )

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
            for quote in analysis.get("quotes", [])[:25]:
                if isinstance(quote, dict) and "text" in quote:
                    citation = state.add_citation(
                        document=doc.filename,
                        page=quote.get("page"),
                        text=quote["text"],
                        context="",
                        relevance=quote.get("relevance", "Direct quote"),
                    )
                    if self.on_citation:
                        self.on_citation(citation)

            # Store findings with deduplication
            # key_facts can be strings or dicts with "fact" key
            # Always initialize these so numeric_facts / gap grounding below can reference them
            # even when key_facts is empty.
            # Bare-string default: "neutral" — consistent with the dict-fact fallback.
            # Bare strings lack issue_relation classification; crediting them as "supports"
            # inflates coverage with unclassified facts. (SO-4 audit #021 finding)
            _bare_rel_dr = "neutral"
            # facts_to_add: (text, issue_relation, effective_date, spo_dict|None)
            # spo_dict carries subject_ref_type/id, predicate_key, object_json for SO-2
            facts_to_add: list[tuple] = []
            _recorded_ids: list[str] = []
            if analysis.get("key_facts"):
                for fact_item in analysis["key_facts"]:
                    if isinstance(fact_item, str):
                        facts_to_add.append((fact_item, _bare_rel_dr, None, None))
                    elif isinstance(fact_item, dict) and "fact" in fact_item:
                        _raw_rel_dr = fact_item.get("issue_relation")
                        issue_rel = _raw_rel_dr.lower().strip() if isinstance(_raw_rel_dr, str) else "neutral"
                        if issue_rel not in ("supports", "attacks", "neutral"):
                            issue_rel = "neutral"
                        _raw_eff = fact_item.get("effective_date")
                        effective_date, _ = _normalize_date(
                            str(_raw_eff) if _raw_eff else "", _raw_eff, None
                        ) if _raw_eff else (None, "unknown")
                        # Extract SPO triple when LLM provides it (SO-2 typed assertions)
                        _subj = fact_item.get("subject")
                        _pred = fact_item.get("predicate")
                        _obj = fact_item.get("object")
                        spo = None
                        if _subj or _pred or _obj:
                            spo = {
                                "subject_ref_type": "free_text" if _subj else None,
                                "subject_ref_id": str(_subj) if _subj else None,
                                "predicate_key": str(_pred).lower().replace(" ", "_") if _pred else None,
                                "object_json": json.dumps(str(_obj)) if _obj else None,
                            }
                        facts_to_add.append((fact_item["fact"], issue_rel, effective_date, spo))

            # Flatten contract_card into additional facts for synthesis visibility.
            # This is OUTSIDE the key_facts block so cards persist even when
            # the model returns no key facts.
            _cc = analysis.get("contract_card")
            if isinstance(_cc, dict):
                _cc_lines = self._flatten_contract_card(_cc, doc.filename)
                for _cc_line in _cc_lines:
                    _cc_rel = "neutral" if "ABSENT" in _cc_line or "missing" in _cc_line.lower() else "supports"
                    facts_to_add.append((_cc_line, _cc_rel, None, None))

                _reinforce_lines = self._contract_card_provision_reinforcement(
                    _cc, doc.filename, content, state.query
                )
                for _r_line in _reinforce_lines:
                    facts_to_add.append((_r_line, "supports", None, None))
                self._persist_contract_evidence(_cc, doc.filename)

            # Persist transaction context (target/acquirer/merger_sub) as typed evidence
            _txn = analysis.get("transaction_context")
            if isinstance(_txn, dict) and self._matter_model is not None:
                _target = (_txn.get("target") or "").strip()
                _acquirer = (_txn.get("acquirer") or "").strip()
                _structure = (_txn.get("structure") or "").strip()
                if _target or _acquirer:
                    te = self._matter_model.typed_evidence
                    txn_key = f"txn:{doc.filename}"
                    te.upsert(
                        "transaction_context", txn_key,
                        payload={
                            "structure": _structure,
                            "target": _target,
                            "acquirer": _acquirer,
                            "merger_sub": (_txn.get("merger_sub") or "").strip(),
                            "parent": (_txn.get("parent") or "").strip(),
                            "source_section": (_txn.get("source_section") or "").strip(),
                            "source_document": doc.filename,
                        },
                        label=f"txn:{_target or 'unknown'}->{_acquirer or 'unknown'}",
                        document_id=doc.filename,
                        confidence=0.85,
                    )
                    if _target:
                        facts_to_add.append((
                            f"[TRANSACTION] Target: {_target}"
                            + (f", Acquirer: {_acquirer}" if _acquirer else "")
                            + (f", Structure: {_structure}" if _structure else ""),
                            "neutral", None, None
                        ))

            # Provision comparisons (from comparison-task deep reads)
            _prov_comps = analysis.get("provision_comparisons")
            if isinstance(_prov_comps, list) and _prov_comps:
                _mm_pc = self._matter_model
                for _pc in _prov_comps[:200]:
                    if not isinstance(_pc, dict):
                        continue
                    _prov = _pc.get("provision", "")
                    _val = _pc.get("value", "")
                    _sec = _pc.get("section_ref", "")
                    _role = _pc.get("source_role", "")
                    _vtype = _pc.get("value_type", "")
                    if not _prov or not _val:
                        continue
                    _pc_fact = f"[PROVISION] {_prov}: {_val}"
                    if _sec:
                        _pc_fact += f" ({_sec})"
                    if _role:
                        _pc_fact += f" [{_role}]"
                    facts_to_add.append((_pc_fact, "supports", None, {
                        "subject_ref_type": "free_text",
                        "subject_ref_id": _prov,
                        "predicate_key": f"has_{_vtype or 'value'}",
                        "object_json": json.dumps({"value": _val, "section": _sec, "source_role": _role}),
                    }))
                    if _mm_pc is not None:
                        _val_hash = _hashlib.md5(f"{_val}:{_sec}".encode()).hexdigest()[:8]
                        _mm_pc.typed_evidence.upsert(
                            "provision_comparison",
                            f"prov:{_prov}:{doc.filename}:{_role}:{_val_hash}",
                            payload={
                                "provision": _prov,
                                "value": _val,
                                "section_ref": _sec,
                                "source_role": _role,
                                "value_type": _vtype,
                                "source_document": doc.filename,
                            },
                            label=f"{_prov}: {_val}",
                            document_id=doc.filename,
                            confidence=0.9,
                        )

            # Regulatory data (from regulatory-task deep reads)
            _reg_data = analysis.get("regulatory_data")
            if isinstance(_reg_data, list) and _reg_data:
                _mm_rd = self._matter_model
                for _rd in _reg_data[:200]:
                    if not isinstance(_rd, dict):
                        continue
                    _cat = _rd.get("category", "")
                    _entity = _rd.get("entity", "")
                    _rdval = _rd.get("value", "")
                    _src_detail = _rd.get("source_detail", "")
                    _sig = _rd.get("significance", "")
                    if not _rdval:
                        continue
                    _rd_fact = f"[REGULATORY:{_cat.upper()}] {_entity}: {_rdval}"
                    if _src_detail:
                        _rd_fact += f" ({_src_detail})"
                    facts_to_add.append((_rd_fact, "supports", None, {
                        "subject_ref_type": "free_text",
                        "subject_ref_id": _entity or _cat,
                        "predicate_key": f"has_{_cat or 'data'}",
                        "object_json": json.dumps({"value": _rdval, "source": _src_detail}),
                    }))
                    if _mm_rd is not None:
                        _rdval_hash = _hashlib.md5(f"{_rdval}:{_src_detail}".encode()).hexdigest()[:8]
                        _mm_rd.typed_evidence.upsert(
                            "regulatory_data",
                            f"reg:{_cat}:{_entity}:{doc.filename}:{_rdval_hash}",
                            payload={
                                "category": _cat,
                                "entity": _entity,
                                "value": _rdval,
                                "source_detail": _src_detail,
                                "significance": _sig,
                                "source_document": doc.filename,
                            },
                            label=f"{_cat}: {_entity} = {_rdval}",
                            document_id=doc.filename,
                            confidence=0.9,
                        )

            # Adverse evidence (hot documents, admissions, problematic language)
            _adv_ev = analysis.get("adverse_evidence")
            if isinstance(_adv_ev, list) and _adv_ev:
                for _ae in _adv_ev[:30]:
                    if not isinstance(_ae, dict):
                        continue
                    _ae_quote = _ae.get("quote", "")
                    _ae_speaker = _ae.get("speaker", "")
                    _ae_sec = _ae.get("section_ref", "")
                    _ae_theory = _ae.get("adverse_theory", "")
                    if not _ae_quote:
                        continue
                    _ae_fact = f"[ADVERSE] \"{_ae_quote}\""
                    if _ae_speaker:
                        _ae_fact += f" — {_ae_speaker}"
                    if _ae_sec:
                        _ae_fact += f" ({_ae_sec})"
                    if _ae_theory:
                        _ae_fact += f" [Risk: {_ae_theory}]"
                    facts_to_add.append((_ae_fact, "attacks", None, {
                        "subject_ref_type": "free_text",
                        "subject_ref_id": _ae_speaker or "unknown",
                        "predicate_key": "admitted_or_stated",
                        "object_json": json.dumps({"quote": _ae_quote, "theory": _ae_theory}),
                    }))
                    if self._matter_model is not None:
                        _ae_hash = _hashlib.md5(_ae_quote[:100].encode()).hexdigest()[:8]
                        self._matter_model.typed_evidence.upsert(
                            "adverse_evidence",
                            f"adv:{doc.filename}:{_ae_hash}",
                            payload={
                                "quote": _ae_quote,
                                "speaker": _ae_speaker,
                                "section_ref": _ae_sec,
                                "adverse_theory": _ae_theory,
                                "source_document": doc.filename,
                            },
                            label=f"Adverse: {_ae_quote[:80]}",
                            document_id=doc.filename,
                            confidence=0.85,
                        )

            # Extraction completeness verification: if the LLM reports incomplete
            # extraction for any table/list, log a warning so we can track coverage.
            _ec = analysis.get("extraction_completeness")
            if isinstance(_ec, list):
                for _ecg in _ec:
                    if not isinstance(_ecg, dict):
                        continue
                    _grp = _ecg.get("group", "")
                    _src_n = _ecg.get("items_in_source")
                    _ext_n = _ecg.get("items_extracted")
                    _complete = _ecg.get("complete", True)
                    if _src_n and _ext_n and not _complete:
                        self._emit_step(
                            state, StepType.REPLAN,
                            f"Incomplete extraction: '{_grp}' has {_src_n} items "
                            f"in source but only {_ext_n} extracted (doc: '{doc.filename[:50]}')",
                        )

            # SO-2 validation: if any facts lack SPO triples, retry to recover them.
            # Threshold >= 1: fire even for single facts; FLASH retry is cheap.
            if facts_to_add:
                _dr_spo_count = sum(1 for _, _, _, _s in facts_to_add if _s is not None)
                if _dr_spo_count < len(facts_to_add) and len(facts_to_add) >= 1:
                    self._emit_step(
                        state, StepType.REPLAN,
                        f"SPO extraction yielded {_dr_spo_count}/{len(facts_to_add)} structured triples "
                        f"(deep-read: '{doc.filename[:60]}'). Retrying for missing.",
                    )
                    _dr_retry_texts = [f for f, _, _, _ in facts_to_add]
                    state.llm_calls_required += 1
                    _dr_retry_spo = await self._retry_spo_extraction(_dr_retry_texts)
                    if _dr_retry_spo:
                        facts_to_add = [
                            (f, rel, eff, _dr_retry_spo.get(i) if spo is None else spo)
                            for i, (f, rel, eff, spo) in enumerate(facts_to_add)
                        ]
            # SO-5: resolve source role using content-based classification first,
            # then fall back to filename heuristic.  The LLM classifies by document
            # content (not filename) so adversarial naming cannot spoof calibration.
            # _CONTENT_ROLE_MAP and _SourceRole are module-level constants.
            _llm_role_str = (analysis.get("doc_source_role") or "").lower().strip()
            _content_role = _CONTENT_ROLE_MAP.get(_llm_role_str, _SourceRole.UNKNOWN)
            # Effective role: content-based when available; filename heuristic as fallback
            _effective_role = (
                _content_role
                if _content_role != _SourceRole.UNKNOWN
                else _infer_source_role(doc.filename)
            )
            _src_label = _effective_role.value.upper()
            _new_facts = [f"[{_src_label}] {f}" for f, _, _d, _spo in facts_to_add]
            state.add_facts(_new_facts)
            if self.on_fact:
                for _f in _new_facts:
                    self.on_fact(_f)
            # Also record into matter model if enabled; pass issue_id if from targeted lead.
            # Use record_facts_batch() so N facts → 1 outer transaction (savepoints inside).
            adapter = getattr(state, "_matter_adapter", None)
            if adapter is not None:
                # _rel_path: repo-relative stable path (not basename) to prevent
                # same-name files in different dirs aliasing in assertion_occurrence.
                # Use dict form for facts with SPO triples (SO-2), tuple form otherwise.
                _dr_batch = []
                for f, issue_rel, eff_date, spo in facts_to_add:
                    if spo:
                        _dr_batch.append({
                            "proposition_text": f,
                            "document_id": _rel_path,
                            "issue_link_type": issue_rel,
                            "temporal_scope_start": eff_date,
                            **spo,
                        })
                    else:
                        _dr_batch.append((f, _rel_path, issue_rel, eff_date))
                _recorded_ids.extend(
                    adapter.record_facts_batch(
                        _dr_batch,
                        issue_id=focus_issue_id,
                        default_source_role=_effective_role,
                    )
                )

                # Build assertion dependency graph from LLM-identified relationships (SO-2)
                # Uses 0-based indices into facts_to_add / _recorded_ids
                _rels = analysis.get("fact_relationships") or []
                for _rel in _rels[:5]:  # cap to 5 edges per document
                    if not isinstance(_rel, dict):
                        continue
                    _fi = _rel.get("from_idx")
                    _ti = _rel.get("to_idx")
                    _rt = _rel.get("relation", "")
                    if _rt and _rt not in _VALID_ASSERTION_LINK_TYPES:
                        logger.debug(
                            "Dropped invalid assertion relation '%s' from deep-read "
                            "(not in _VALID_ASSERTION_LINK_TYPES)", _rt
                        )
                    if (isinstance(_fi, int) and isinstance(_ti, int)
                            and 0 <= _fi < len(_recorded_ids)
                            and 0 <= _ti < len(_recorded_ids)
                            and _fi != _ti
                            and _recorded_ids[_fi]
                            and _recorded_ids[_ti]
                            and _recorded_ids[_fi] != _recorded_ids[_ti]
                            and _rt in _VALID_ASSERTION_LINK_TYPES):
                        adapter.record_assertion_link(
                            _recorded_ids[_fi], _recorded_ids[_ti], _rt
                        )

            # Extract and store structured numeric facts (SO-6).
            # Ground each quant fact to an assertion_id by searching the already-recorded
            # key_facts for the raw numeric text. This gives provenance for reconciliation.
            # Batch into record_quants_batch() — one transaction for all numeric facts.
            if analysis.get("numeric_facts"):
                _adp = getattr(state, "_matter_adapter", None)
                if _adp is not None:
                    _quant_specs: list[dict] = []
                    for nf in analysis["numeric_facts"]:
                        if not isinstance(nf, dict):
                            continue
                        kind = nf.get("kind", "amount")
                        raw = nf.get("raw", "")
                        if not raw:
                            continue
                        value = nf.get("value")
                        try:
                            numeric_value = float(value) if value is not None else None
                        except (TypeError, ValueError, OverflowError):
                            numeric_value = None
                        amount = numeric_value if kind in ("amount", "balance", "count") else None
                        rate = numeric_value if kind == "rate" else None
                        unit = nf.get("unit") or ("count" if kind == "count" else None)
                        # Normalise dates to ISO YYYY-MM-DD with precision tracking (SO-6).
                        date_val: Optional[str] = None
                        date_end_val: Optional[str] = None
                        _date_precision: Optional[str] = None
                        if kind in ("date", "date_range"):
                            llm_val = nf.get("value") if isinstance(nf.get("value"), str) else None
                            llm_prec = nf.get("date_precision")
                            if kind == "date_range" and isinstance(llm_val, str) and "/" in llm_val:
                                parts = llm_val.split("/", 1)
                                date_val, _date_precision = _normalize_date(parts[0], parts[0], llm_prec)
                                date_end_val, _ = _normalize_date(parts[1], parts[1], llm_prec)
                            else:
                                date_val, _date_precision = _normalize_date(raw, llm_val, llm_prec)
                        # Ground to source assertion: prefer explicit assertion_idx from LLM
                        # (direct index into key_facts), fall back to string matching.
                        _nf_assertion_id: Optional[str] = None
                        _aidx = nf.get("assertion_idx")
                        if (isinstance(_aidx, int)
                                and 0 <= _aidx < len(_recorded_ids)):
                            _nf_assertion_id = _recorded_ids[_aidx]
                        else:
                            _raw_lower = raw.lower()
                            for (_ft, _frel, _fd, _spo), _fa in zip(facts_to_add, _recorded_ids):
                                if _raw_lower and _raw_lower in _ft.lower():
                                    _nf_assertion_id = _fa
                                    break
                        # Build span_id from page number if the LLM provided it (SO-6 grounding)
                        _nf_page = nf.get("page")
                        _nf_span_id = f"page:{_nf_page}" if isinstance(_nf_page, int) else None
                        _quant_specs.append({
                            "quant_kind": kind,
                            "raw_text": f"{raw} — {nf.get('context', '')}",
                            "amount_value": amount,
                            "currency": nf.get("currency"),
                            "date_value": date_val,
                            "date_end_value": date_end_val,
                            "rate_value": rate,
                            "subject_type": nf.get("subject"),
                            "subject_id": nf.get("subject_id"),
                            "assertion_id": _nf_assertion_id,
                            "span_id": _nf_span_id,
                            "date_precision": _date_precision,
                            "unit": unit,
                        })
                    if _quant_specs:
                        _adp.record_quants_batch(
                            _quant_specs, document_id=doc.filename
                        )
                    # Create calculation_operand typed evidence for revenue-related
                    # numeric facts so the graph resolver can link them to contracts.
                    if self._matter_model is not None and self._is_extraction_task(state.query):
                        _te = self._matter_model.typed_evidence
                        for _qi, _qs in enumerate(_quant_specs):
                            _q_raw = (_qs.get("raw_text") or "").lower()
                            _q_val = _qs.get("amount_value")
                            _q_subj_type = (_qs.get("subject_type") or "").lower()
                            _q_subj_id = _qs.get("subject_id") or ""
                            if not _q_val or _q_val <= 0:
                                continue
                            # Classify via keyword patterns first
                            _op_role = self._classify_operand_role(_q_raw)
                            # Metadata-aware fallback: if subject_type is revenue
                            # with a specific entity subject_id, classify by context
                            if _op_role == "unclassified" and _q_subj_type == "revenue":
                                _id_lower = _q_subj_id.lower()
                                if _id_lower and _id_lower not in ("company", "total", "consolidated", ""):
                                    _op_role = "actual_counterparty_ttm_revenue"
                                elif _id_lower in ("company", "total", "consolidated"):
                                    _op_role = "company_total_ttm_revenue"
                            # Also classify schedule-sourced revenue figures
                            if _op_role == "unclassified" and "revenue" in _q_raw:
                                if "schedule" in _q_raw or "disclosure" in _q_raw:
                                    if _q_subj_id and _q_subj_id.lower() not in ("company", "total", ""):
                                        _op_role = "actual_counterparty_ttm_revenue"
                            if _op_role in ("actual_counterparty_ttm_revenue",
                                            "company_total_ttm_revenue",
                                            "drawn_outstanding",
                                            "acquirer_revenue",
                                            "acquirer_segment_revenue",
                                            "actual_product_line_ttm_revenue"):
                                _val_m = _q_val / 1_000_000.0 if _q_val > 1000 else _q_val
                                if _val_m > 0.5:
                                    _subj_label = _q_subj_id or self._extract_operand_subject(_q_raw, doc.filename)
                                    _src_priority = (
                                        "schedule_disclosed"
                                        if "schedule" in _q_raw or "attributable" in _q_raw or "disclosure" in _q_raw
                                        else "body_stated"
                                    )
                                    try:
                                        _te.upsert(
                                            "calculation_operand",
                                            f"op:nf:{doc.filename}:{_qi}",
                                            payload={
                                                "operand_role": _op_role,
                                                "value_millions": _val_m,
                                                "subject_label": _subj_label,
                                                "raw_text": (_qs.get("raw_text") or "")[:300],
                                                "source_document": doc.filename,
                                                "source_priority": _src_priority,
                                            },
                                            label=f"{_op_role}:{_subj_label}",
                                            document_id=doc.filename,
                                            confidence=0.85,
                                        )
                                    except Exception:
                                        pass

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
                    if self._should_skip_follow_on_lead(concern, state):
                        continue
                    state.add_lead(
                        description=f"Investigate concern: {concern}",
                        source=doc.filename,
                        priority=0.6,
                    )

            # LLM-driven gap detection: check referenced documents against repo (SO-7)
            connections = analysis.get("connections", [])
            if connections:
                # Build filename lookup once per investigation run (not per document).
                # repo.list_files() does a recursive filesystem walk — doing it per-document
                # on a large matter is an avoidable O(docs × files) hotspot.
                if self._known_filenames is None:
                    self._known_filenames = {
                        f.filename.lower()
                        for f in repo.list_files()
                    }
                known_names = self._known_filenames
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

            # Persist document intelligence card (Change 6: write cards during deep-read)
            if _mm is not None and _inventory_doc_id is not None:
                try:
                    _mm.upsert_document_intelligence(
                        relative_path=_rel_path,
                        analysis=analysis,
                        focus_issue_id=focus_issue_id,
                        run_id=getattr(state, "_run_id", None),
                    )
                except Exception as _card_err:
                    logger.debug("Document card persistence failed for %s: %s", _rel_path, _card_err)

                # Persist quote spans for section-level read memory
                for quote in analysis.get("quotes", [])[:5]:
                    if isinstance(quote, dict) and quote.get("text"):
                        try:
                            _mm.add_doc_span(_inventory_doc_id, {
                                "span_type": "quote",
                                "span_text": quote["text"][:500],
                                "page_start": quote.get("page"),
                            })
                        except Exception:
                            pass  # non-fatal

            # Lawyer-facing deep-read summary: tell attorneys what we learned
            # from this document in plain language, not engineer metrics.
            _doc_name = Path(file_path).name
            _src_role = analysis.get("doc_source_role", "unknown")
            _doc_type = analysis.get("doc_type") or _src_role
            _n_facts = len(analysis.get("key_facts", []))
            _n_quotes = len(analysis.get("quotes", []))
            _concerns = analysis.get("concerns") or []
            _connections = analysis.get("connections") or []
            _unresolved = analysis.get("unresolved_flags") or []
            _purpose = analysis.get("purpose") or ""

            # Build a concise attorney-readable summary
            _summary_parts = [f"{_doc_name} ({_doc_type})"]
            if _purpose:
                _summary_parts.append(_purpose[:80])
            if _n_facts:
                _summary_parts.append(f"{_n_facts} key facts extracted")
            if _concerns:
                _summary_parts.append(f"Flags: {'; '.join(str(c) for c in _concerns[:2])}")
            if _connections:
                _summary_parts.append(f"References: {', '.join(str(c) for c in _connections[:2])}")
            if _unresolved:
                _summary_parts.append(f"Open questions: {'; '.join(str(u) for u in _unresolved[:2])}")
            self._emit_step(
                state, StepType.FINDING,
                " — ".join(_summary_parts),
            )

            # Mark document as fully ingested so future runs take the hot path (SO-1)
            if _mm is not None and _inventory_doc_id is not None:
                try:
                    _mm.inventory.mark_ingested(_inventory_doc_id)
                except Exception:
                    pass  # non-fatal

            # Background maintenance: after each new document, incrementally
            # refresh proof states and extract authority citations (SO-4).
            # Both are best-effort — failures must never block the pipeline.
            if _mm is not None:
                try:
                    if _recorded_ids:
                        # Targeted: only recompute proof states for issues linked to
                        # assertions from this document. Avoids recomputing all open
                        # issues on every document ingest (performance fix).
                        _issue_rows = _mm.db.execute(
                            "SELECT DISTINCT issue_id FROM assertion_issue_link"
                            " WHERE assertion_id IN ({})".format(
                                ",".join("?" * len(_recorded_ids))
                            ),
                            _recorded_ids,
                        ).fetchall()
                        if _issue_rows:
                            # Pre-fetch overrides once for all targeted issue recomputes
                            # rather than fetching inside each compute_and_store() call.
                            _ov_rows = _mm.db.execute(
                                """SELECT document_pattern, trust_level
                                   FROM document_trust_override
                                   WHERE matter_id=? AND trust_level != 'normal'
                                   ORDER BY LENGTH(document_pattern) DESC""",
                                (_mm.matter_id,),
                            ).fetchall()
                            _pre_ov = [
                                (r["document_pattern"], r["trust_level"])
                                for r in _ov_rows
                            ]
                            for _irow in _issue_rows:
                                _mm.proof_state.compute_and_store(
                                    _irow["issue_id"], _preloaded_overrides=_pre_ov
                                )
                    # If no assertions were recorded, proof state is unchanged — skip update.
                    # Post-synthesis compute_all() at end of run catches any remaining gaps.
                except Exception as exc:
                    _log.warning("proof_state compute_and_store failed: %s", exc)
                if analysis and analysis.get("quotes"):
                    # Extract authorities from the raw document text processed so far.
                    _quote_text = " ".join(
                        q.get("text", "") for q in analysis["quotes"][:10]
                        if isinstance(q, dict)
                    )
                    if _quote_text and _domain == "legal":
                        try:
                            self._extract_and_store_authorities(
                                _quote_text,
                                run_id=getattr(state, "_run_id", None),
                                source_document_ref=_rel_path,
                            )
                        except Exception:
                            pass

        except Exception as e:
            self._emit_step(state, StepType.ERROR, f"Failed to read {file_path}: {e}")
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
        finally:
            # Only clean up if THIS coroutine added the marker (ownership-safe).
            # Without this check, a second coroutine that returned early at the
            # guard would still discard the marker in its finally, allowing a
            # third coroutine to start a duplicate cold-path read.
            if _owns_in_progress and _rel_path is not None:
                state._reading_in_progress.discard(_rel_path)

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
        _stopped_early = False

        for citation in unverified:
            # Honour stop request mid-verification (SO-3 — adv#034 HIGH #3)
            if _adapter is not None and _adapter.is_stop_requested():
                _stopped_early = True
                break
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
        # LOW adv#035: only emit "complete" when the full list was processed; a
        # stop-interrupted verify loop must not log a false completion signal.
        if _stopped_early:
            self._emit_step(
                state,
                StepType.VERIFY,
                f"Verification interrupted: {stats['verified']} verified, "
                f"{stats['unverified']} unverified (partial — stopped by user)",
            )
        else:
            self._emit_step(
                state,
                StepType.VERIFY,
                f"Verification complete: {stats['verified']} verified, {stats['unverified']} unverified",
            )

    async def _synthesize(self, state: InvestigationState):
        """Phase 3: Final synthesis using Pro model.

        SYNTHESIS CONTEXT PRINCIPLE: Pass only answer ingredients (assertions,
        facts, findings, source refs). Do NOT include gap summaries,
        contradiction lists, issue status, proof state, or coverage reports
        unless the user's question explicitly asks about them. The
        investigative loop already handled those artifacts.
        """
        _adapter = getattr(state, "_matter_adapter", None)
        if _adapter is not None and _adapter.is_stop_requested():
            return  # Skip synthesis if user stopped the run
        self._emit_step(state, StepType.SYNTHESIS, "Synthesizing final analysis...")

        # Pre-synthesis refresh: rebuild accumulated_facts from the truth-maintained
        # assertion graph so that any belief-state revisions made during this run
        # (user corrections, superseded assertions) are reflected in synthesis. (SO-2)
        # All facts extracted during investigation are already persisted to the matter
        # model via record_facts_batch(), so the re-hydration is complete and correct.
        if self._matter_model is not None:
            state.findings["accumulated_facts"] = []
            self._hydrate_from_matter_model(state)

        # Log synthesis entry to reasoning ledger (SO-3)
        adapter = getattr(state, "_matter_adapter", None)
        if adapter is not None:
            facts = state.findings.get("accumulated_facts", [])
            adapter.log_step(
                f"Synthesis phase: {state.documents_read} docs, "
                f"{state.searches_performed} searches, {len(facts)} facts",
                why="All leads exhausted or investigation complete",
            )

        # Compile all findings, sorted by source trust (Gap 2: trust-aware synthesis).
        # Operative/authoritative facts appear first so the LLM weights them more heavily.
        facts = state.findings.get("accumulated_facts", [])
        facts = self._sort_facts_by_trust(facts)

        # Cross-document analysis pass: for extraction tasks, derive calculations,
        # flag inconsistencies, and identify missing provisions BEFORE synthesis.
        if self._is_extraction_task(state.query):
            graph_calcs = self._resolve_operand_graph_calculations(facts=facts)
            if graph_calcs:
                facts.extend(graph_calcs)

            # Deterministic provision comparison calculations (Codex #3)
            prov_calcs = self._derive_provision_comparison_calculations()
            if prov_calcs:
                facts.extend(prov_calcs)

            # Deterministic regulatory calculations (Codex #3 extension)
            reg_calcs = self._derive_regulatory_calculations()
            if reg_calcs:
                facts.extend(reg_calcs)

            if len(facts) > 2:
                _quant_ctx = self._build_quant_summary() if self._matter_model else ""
                derived_facts = await self._cross_document_analysis(state.query, facts, _quant_ctx)
                if derived_facts:
                    facts.extend(derived_facts)
                    state.findings["accumulated_facts"] = facts

        # Evidence relevance filter: LITE pass reads the full fact set and drops
        # only clearly irrelevant items. This replaces hard caps — the model decides
        # what matters based on the query, not an arbitrary number.
        if len(facts) > 50 and not self._is_extraction_task(state.query):
            facts = await self._filter_facts_for_relevance(state.query, facts)
        findings_text = "\n".join(f"• {fact}" for fact in facts)

        # Store citations and entities as structured metadata for UI panels.
        state.findings["metadata_citations"] = state.get_citations_formatted()
        state.findings["metadata_entities"] = state.get_entities_formatted()

        # Dynamically assemble the context packet — only include sections that
        # have real content. PRO gets exactly what's useful, nothing empty.
        context_build = await self._assemble_context_packet(state, findings_text)
        context_packet = context_build.text

        _synth_domain = self._resolve_active_domain(state)
        _synth_template = _compose_synthesis_prompt(_synth_domain)
        prompt = _synth_template.format(
            query=state.query,
            context_packet=context_packet,
        )

        # Synthesis cache (SO-1): same prompt → skip PRO LLM call on warm runs.
        # Key hashes the full prompt text (which captures facts, gaps, quant, citations).
        import hashlib as _sh
        _history_digest = _conversation_history_digest(state.conversation_history)
        _syn_key = _sh.sha256(f"{prompt}\n{_history_digest}".encode()).hexdigest()
        _cached_response = None
        if self._matter_model is not None:
            try:
                _cached_response = self._matter_model.cache.get("synthesis", _syn_key)
            except Exception:
                pass

        if _cached_response is not None:
            response = _cached_response
            state.llm_calls_avoided += 1
        else:
            # Use PRO for final synthesis, with FLASH fallback on timeout.
            state.llm_calls_required += 1
            response = await self._complete_synthesis_with_fallback(state, prompt)
            if self._matter_model is not None:
                try:
                    _mh = context_build.dependency_manifest_hash or state.cache_manifest_hash
                    if _mh:
                        self._matter_model.cache.put_brokered(
                            "synthesis", _syn_key, response,
                            manifest_hash=_mh,
                        )
                    else:
                        self._matter_model.cache.put("synthesis", _syn_key, response)
                except Exception:
                    pass

        # SO-5: Post-synthesis advocacy gate.
        # If any open issues rely exclusively on advocacy sources, force-append a
        # Source Calibration Advisory that names them. Prompt-level instruction alone
        # is advisory; this is a hard output mutation that cannot be LLM-bypassed.
        if self._matter_model is not None:
            try:
                _adv_enforced = self._enforce_advocacy_gate(response, state)
                if _adv_enforced is not None:
                    response = _adv_enforced
            except Exception as exc:
                logger.error(
                    "HARD GATE FAILURE: advocacy gate failed — output may contain "
                    "uncalibrated advocacy-source claims: %s", exc,
                )

        # SO-6: Post-synthesis quantitative threshold diagnostics.
        # Preserve high quantitative violations as side-channel findings, but do
        # not mutate the user-facing answer body. The answer must stay shaped by
        # the user's task contract; diagnostics belong in validator/UI surfaces.
        if self._matter_model is not None:
            try:
                self._enforce_quant_threshold_gate(response, state=state)
            except Exception as exc:
                logger.error(
                    "HARD GATE FAILURE: quant threshold gate failed — output may miss "
                    "critical financial exposure: %s", exc,
                )

        response = await self._repair_output_if_needed(
            state,
            response,
            emitter="synthesis",
        )
        self._emit_output(
            state, response, emitter="synthesis",
            dependency_manifest_hash=context_build.dependency_manifest_hash,
        )

        if self._matter_model is not None and _synth_domain == "legal":
            try:
                self._extract_and_store_authorities(
                    response,
                    run_id=getattr(state, "_run_id", None),
                    source_document_ref="synthesis:final",
                )
            except Exception:
                pass

        # Refresh proof state for all open issues (SO-4 proof-aware reasoning).
        if self._matter_model is not None:
            try:
                self._matter_model.proof_state.compute_all()
            except Exception as exc:
                _log.warning("proof_state compute_all failed: %s", exc)

        self._emit_step(state, StepType.SYNTHESIS, "Analysis complete")

    def _enforce_advocacy_gate(self, synthesis_output: str, state: "Optional[InvestigationState]" = None) -> Optional[str]:
        """Domain-calibrated reliance gate (SO-5).

        Post-synthesis hard gate that appends a Source Calibration Advisory when
        open issues rely exclusively on low-trust sources.  Domain-specific policy
        from _DOMAIN_RELIANCE_POLICY controls hedge markers, labels, and
        violation notes — legal uses advocacy markers, finance uses management-only
        markers, biomedical uses sponsor-only markers, etc.
        """
        domain = self._resolve_active_domain(state)
        policy = _DOMAIN_RELIANCE_POLICY.get(domain)
        if policy is None:
            logger.warning("No reliance policy for domain %r — falling back to legal", domain)
            policy = _DOMAIN_RELIANCE_POLICY["legal"]
        if self._matter_model is None:
            return None
        try:
            advocacy_issues = self._matter_model.proof_state.get_advocacy_only()
        except Exception as exc:
            _log.warning("_enforce_advocacy_gate: get_advocacy_only failed: %s", exc)
            return None
        if not advocacy_issues:
            return None

        # Restrict to open issues (same filter as _build_advocacy_gate_block).
        try:
            open_issue_ids = {
                row["id"]
                for row in self._matter_model.db.execute(
                    "SELECT id FROM issue WHERE matter_id=? AND status='open'",
                    (self._matter_model.matter_id,),
                ).fetchall()
            }
        except Exception:
            # On DB failure, open_issue_ids = None causes all advocacy issues
            # to be included — intentionally conservative (false-positive advisory
            # is preferable to a false-negative that misses an open issue).
            open_issue_ids = None

        active_advocacy = [
            ps for ps in advocacy_issues
            if open_issue_ids is None or ps.get("issue_id") in open_issue_ids
        ]
        if not active_advocacy:
            return None

        # Use module-level _ADVOCACY_MARKER_NAME for the injected section header.

        # Build issue index for human-readable names before structural check.
        issue_index: dict = {}
        try:
            report = self._matter_model.get_issue_coverage_report()
            issue_index = {
                item["id"]: item.get("title", "Untitled")
                for item in report
                if item.get("id")
            }
        except Exception:
            pass

        advocacy_titles = [
            issue_index.get(
                ps.get("issue_id", ""),
                str(ps.get("issue_id") or "")[:24],  # str() guard: issue_id may be non-string
            )
            for ps in active_advocacy
        ]

        _HEDGE_MARKERS = policy["hedge_markers"]
        _STRUCTURAL_VIOLATION = False

        # Normalize line endings: CRLF → LF so regex patterns that anchor on
        # \n work correctly on Windows-originated output or CRLF LLM responses.
        synthesis_output = synthesis_output.replace('\r\n', '\n')

        # Use module-level compiled patterns (_ADVOCACY_HDR_RE, _ADVOCACY_LIST_PAT,
        # _ADVOCACY_SECTIONS, _ADVOCACY_MARKER_PAT) — hoisted from per-call scope
        # to avoid repeated re.compile() on every synthesis invocation.

        def _extract_section(text: str, hdr: str, hdr_pat: "_re_engine.Pattern[str]") -> str:
            """Return text from hdr to the next header of equal or higher level."""
            m_hdr = hdr_pat.search(text)
            if not m_hdr:
                return ""
            start = m_hdr.start()
            if start > 0 and text[start] == '\n':
                start += 1  # skip leading newline — point to '#'
            level = min(3, len(hdr) - len(hdr.lstrip("#")))
            m_end = _ADVOCACY_HDR_RE[level].search(text, start + len(hdr))
            return text[start:(m_end.start() if m_end else len(text))]

        def _section_has_unhedged_title(
            section: str, titles: "list[str]", markers: "tuple[str, ...]"
        ) -> bool:
            """True if any title appears in a semantic unit without a hedge marker.

            Groups continuation lines into semantic units (bullet items / paragraphs)
            so that a hedge on a continuation line of the same bullet clears the title
            on the preceding line, and adjacent bullets cannot cross-contaminate.

            Titles shorter than 4 chars are skipped to avoid false matches.
            """
            lower = section.lower()
            # Build semantic units: group lines until a blank line or a new list item.
            units: "list[str]" = []
            buf: "list[str]" = []
            for ln in lower.split('\n'):
                ls = ln.lstrip()
                is_list_start = bool(_ADVOCACY_LIST_PAT.match(ls))
                if not ls:
                    if buf:
                        units.append(' '.join(buf))
                    buf = []
                elif is_list_start and buf:
                    units.append(' '.join(buf))
                    buf = [ln]
                else:
                    buf.append(ln)
            if buf:
                units.append(' '.join(buf))

            for title in titles:
                t_lower = title.lower()
                if len(t_lower) < 4:
                    continue
                for unit in units:
                    if t_lower in unit and not any(h in unit for h in markers):
                        return True
            return False

        # Check each candidate section (Key Findings, Factual Background) using
        # precompiled header patterns from _ADVOCACY_SECTIONS.
        _seen_hdrs: "set[str]" = set()
        for _chk_hdr, _chk_pat in _ADVOCACY_SECTIONS:
            if _chk_hdr in _seen_hdrs:
                continue  # skip colon-variant if base already matched
            _sec = _extract_section(synthesis_output, _chk_hdr, _chk_pat)
            if _sec:
                _seen_hdrs.add(_chk_hdr)
                if _section_has_unhedged_title(_sec, advocacy_titles, _HEDGE_MARKERS):
                    _STRUCTURAL_VIOLATION = True
                    break

        # If the advisory section header is already present AND no structural violation,
        # AND the advisory section actually references the active issues, gate is
        # satisfied.  An empty or wrong advisory header injected by the LLM must NOT
        # suppress the gate (LLM output is untrusted input).
        _marker_match = _ADVOCACY_MARKER_PAT.search(synthesis_output)
        if _marker_match and not _STRUCTURAL_VIOLATION:
            _advisory_section = synthesis_output[_marker_match.start():]
            _advisory_lower = _advisory_section.lower()
            _titles_covered = sum(
                1 for t in advocacy_titles
                if len(t) >= 4 and t.lower() in _advisory_lower
            )
            if _titles_covered >= len([t for t in advocacy_titles if len(t) >= 4]):
                return None

        violation_note = ""
        if _STRUCTURAL_VIOLATION:
            violation_note = (
                f"\n⚠ STRUCTURAL VIOLATION DETECTED: {policy['violation_note']}\n"
            )

        advisory_name = policy["advisory_name"]
        lines = [
            "",
            f"## {advisory_name}",
            f"*(Auto-generated by SO-5 reliance gate — the following issues lack "
            f"{policy['corroboration_label']} corroboration.)*",
            violation_note,
            f"The following issues are supported ONLY by {policy['source_description']}. "
            f"They must appear in ## {policy['section_label']}, NOT in "
            f"## Key Findings as established facts:",
            "",
        ]
        for ps in active_advocacy:
            issue_id = ps.get("issue_id", "?")
            title = issue_index.get(issue_id, str(issue_id or "?")[:24])
            tw = ps.get("trust_weighted_support", 0.0)
            lines.append(
                f"- **{title}** — {policy['source_label']}-only "
                f"(trust-weighted support: {tw:.2f})"
            )
        lines.append("")
        return synthesis_output + "\n".join(lines)

    def _enforce_quant_threshold_gate(
        self,
        synthesis_output: str,
        state: "Optional[InvestigationState]" = None,
    ) -> Optional[str]:
        """Record SO-6 financial diagnostics without mutating the answer body.

        Diagnostics remain available for UI/validator surfaces through
        state.findings, but the final answer stays shaped by the user's task.
        """
        if self._matter_model is None:
            return None
        try:
            violations = self._matter_model.compute_quant_thresholds()
        except Exception:
            return None

        high_violations = [v for v in violations if v.get("level") == "HIGH"]
        if not high_violations:
            return None  # no action when no HIGH violations

        # Content-based gate: check whether the specific violation figures we computed
        # are actually present in the output — not just whether a heading exists.
        # A heading without violation figures is a bypass, not genuine compliance.
        #
        # The violation descriptions are generated by compute_thresholds() and contain
        # specific amounts and percentages (e.g. "Positive exposure: USD 50,000.00").
        # They only appear verbatim if (a) we already injected them, or (b) the LLM
        # somehow replicated our exact phrasing — both mean the gate is satisfied.
        _violation_descs = [v.get("description", "") for v in high_violations]
        _has_violation_content = any(
            desc and desc[:60] in synthesis_output
            for desc in _violation_descs
        )
        if _has_violation_content:
            return None  # violation figures are present — gate is satisfied

        # Build the financial diagnostics block for side-panel / validator use.
        lines = [
            "## Financial Analysis",
            "*(Auto-generated by SO-6 threshold gate — LLM synthesis omitted this section.)*",
            "",
        ]
        for v in high_violations:
            desc = v.get("description", "")
            lines.append(f"- **[{v.get('level')}]** {desc}")

        # Add reconciliation summary if available
        try:
            chain = self._matter_model.reconcile_payment_chain()
            exp = chain.get("exposure", 0.0)
            inv = chain.get("invoiced", 0.0)
            paid = chain.get("paid", 0.0)
            ccy = chain.get("currency", "USD")
            if inv or paid:
                lines.append("")
                lines.append(
                    f"Payment Reconciliation ({ccy}): "
                    f"Invoiced ${inv:,.2f} — Paid ${paid:,.2f} — "
                    f"**Exposure ${exp:,.2f}**"
                )
        except Exception:
            pass

        diagnostics = "\n".join(lines)
        if state is not None:
            try:
                state.findings["quant_threshold_diagnostics"] = diagnostics
                state.findings["quant_threshold_high_count"] = len(high_violations)
            except Exception:
                pass
        return None

    def _extract_and_store_authorities(
        self,
        text: str,
        *,
        run_id: Optional[str] = None,
        source_document_ref: Optional[str] = None,
    ) -> None:
        """Extract legal citations from synthesis text and persist to AuthorityStore.

        Recognises the most common citation forms used in U.S. legal writing:
        - Case law: Smith v. Jones, 123 F.3d 456 (9th Cir. 2001)
        - U.S. Reports: 550 U.S. 544 (2007)
        - Federal statutes: 42 U.S.C. § 1983
        - Federal regulations: 29 C.F.R. § 825.100
        - State statutes: Cal. Civ. Code § 1750

        Citations are stored with weight='persuasive' by default (binding
        status requires jurisdictional analysis outside the engine).

        P0.1: every upsert carries a ProvenanceContext so an audit can
        trace each case/statute to the run that extracted it and, when
        ACTIVE_LLM_CALL is set, to the originating LLM call.
        """
        import re

        # Build a reusable provenance context for every authority
        # upsert in this extraction pass. The ContextVar bridge pulls
        # the most recent LLM call id/model/prompt_hash without having
        # to thread them through every call site.
        from ..core.models import ACTIVE_LLM_CALL
        from ..matter.models import ProvenanceContext
        _active = ACTIVE_LLM_CALL.get() or {}
        _auth_prov = ProvenanceContext(
            event_kind="authority_upsert",
            writer_name="AuthorityStore.upsert",
            run_id=run_id,
            model_id=_active.get("model_id"),
            model_tier=_active.get("model_tier"),
            prompt_version="SPEC.AUTHORITY_TREATMENT.v1",
            extractor_version="2026-04-17.p01.v1",
            llm_call_id=_active.get("call_id"),
            prompt_hash=_active.get("prompt_hash"),
            source_document_ref=source_document_ref,
            source_span_status=(
                "present" if source_document_ref else "not_applicable"
            ),
        )

        # Pattern: "Name v. Name, VolNo Reporter PageNo (Court Year)"
        # Captures full citation including optional court/year parenthetical.
        _CASE_PATTERN = re.compile(
            r"\b([A-Z][A-Za-z\s,'\.]+(?:Corp\.|Inc\.|LLC|Ltd\.)?)\s+v\.\s+"
            r"([A-Z][A-Za-z\s,'\.]+?),\s*"
            r"(\d+\s+[A-Za-z\.]+\s+\d+)"
            r"(?:\s+\([^)]{3,40}\))?",
            re.MULTILINE,
        )
        # Pattern: federal statute, 42 U.S.C. § 1983 or §§ 1331-1340
        _STATUTE_PATTERN = re.compile(
            r"\b(\d+)\s+(U\.S\.C\.|C\.F\.R\.|U\.S\.C\.A\.)\s+§{1,2}\s*([\d\-\.a-z]+)",
            re.IGNORECASE,
        )
        # Pattern: state code abbreviations, e.g., Cal. Civ. Code § 1750
        _STATE_STATUTE_PATTERN = re.compile(
            r"\b([A-Z][a-z]+\.(?:\s+[A-Z][a-z]+\.)+)\s+§{1,2}\s*([\d\-\.a-z]+)",
        )

        seen: set[str] = set()
        authority_store = self._matter_model.authority

        for m in _CASE_PATTERN.finditer(text):
            party1 = m.group(1).strip().rstrip(",")
            party2 = m.group(2).strip().rstrip(",")
            reporter = m.group(3).strip()
            citation = f"{party1} v. {party2}, {reporter}"
            # Normalise whitespace
            citation = " ".join(citation.split())
            if citation in seen:
                continue
            seen.add(citation)
            try:
                authority_store.upsert(
                    citation=citation,
                    authority_type="case",
                    weight="persuasive",
                    provenance=_auth_prov,
                )
            except Exception as exc:
                _log.debug("authority upsert failed for %r: %s", citation, exc)

        for m in _STATUTE_PATTERN.finditer(text):
            title = m.group(1).strip()
            code = m.group(2).strip()
            section = m.group(3).strip()
            citation = f"{title} {code} § {section}"
            citation = " ".join(citation.split())
            if citation in seen:
                continue
            seen.add(citation)
            auth_type = "regulation" if "C.F.R." in code else "statute"
            try:
                authority_store.upsert(
                    citation=citation,
                    authority_type=auth_type,
                    weight="binding",  # federal statutes and regulations are binding
                    provenance=_auth_prov,
                )
            except Exception as exc:
                _log.debug("authority upsert failed for %r: %s", citation, exc)

        for m in _STATE_STATUTE_PATTERN.finditer(text):
            code = m.group(1).strip()
            section = m.group(2).strip()
            citation = f"{code} § {section}"
            citation = " ".join(citation.split())
            if citation in seen or len(citation) < 8:
                continue
            seen.add(citation)
            try:
                authority_store.upsert(
                    citation=citation,
                    authority_type="statute",
                    weight="persuasive",  # state statutes — jurisdiction-dependent
                    provenance=_auth_prov,
                )
            except Exception as exc:
                _log.debug("authority upsert failed for %r: %s", citation, exc)

    def _build_issue_focus_block(
        self, focus_issue_id: Optional[str]
    ) -> tuple[str, list[str]]:
        """Build an issue-focus context block for the analysis prompt (SO-4).

        When a lead targets a specific issue, inject the issue title and first
        open predicate into the analysis prompt so the LLM prioritizes facts
        that address the issue's specific proof elements — not just query-token
        surface matches.

        Returns (block_str, pred_descriptions) so callers can use the predicate
        list as an allowlist for resolve_predicate_by_description() without a
        second get_predicates() call (eliminates duplicate DB read on cache misses).

        Returns ("", []) when no issue context is available.
        """
        if not focus_issue_id or self._matter_model is None:
            return "", []
        try:
            issue = self._matter_model.issues.get_issue(focus_issue_id)
            if not issue:
                return "", []
            title = issue.get("title", "")
            predicates = self._matter_model.issues.get_predicates(focus_issue_id, limit=2)
            pred_descs = [
                p.get("description", "")
                for p in predicates
                if p.get("status") == "open" and p.get("description")
            ]
            lines = [f"Issue Focus (SO-4 — prioritize facts addressing these elements):"]
            lines.append(f"  Issue: \"{title}\"")

            # Surface current coverage state so the strategist/lead-gen LLM sees
            # what is already supported vs. what needs more evidence. The canonical
            # source is get_issue_coverage_report(). The proof_state table declares
            # support_score, attack_score, coverage_fraction, and has_proof_gap
            # columns, but ProofStateStore.compute_and_store() writes only
            # trust_weighted_support, trust_weighted_attack, and proof_status — the
            # other four columns are a dead substrate pending MVP.5 store repair.
            # Until then, reading them from proof_state silently reported 0% for
            # every issue, which is why this block now goes through the coverage
            # report. proof_state.advocacy_only IS written and is read separately.
            report_row = None
            try:
                for row in self._matter_model.get_issue_coverage_report():
                    if row.get("id") == focus_issue_id:
                        report_row = row
                        break
            except Exception:
                report_row = None

            if report_row is not None:
                supporting = int(report_row.get("supporting_count") or 0)
                coverage = float(report_row.get("coverage_fraction") or 0.0)
                noun = "assertion" if supporting == 1 else "assertions"
                lines.append(
                    f"  Current support: {supporting} supporting {noun} | "
                    f"Coverage: {coverage:.0%}"
                )
                if report_row.get("has_proof_gap"):
                    lines.append(
                        "  ⚠ PROOF GAP: at least one required element has no evidence"
                    )
            # advocacy_only IS persisted on proof_state — surface it independently
            # even when the issue is absent from the coverage report (e.g. closed).
            try:
                ps = self._matter_model.proof_state.get(focus_issue_id)
                if ps and ps.get("advocacy_only"):
                    lines.append(
                        "  ⚠ ADVOCACY-ONLY: all supporting evidence is from advocacy sources"
                    )
            except Exception:
                pass  # proof state unavailable; advocacy flag simply not shown

            for desc in pred_descs:
                lines.append(f"  Element to prove: \"{desc}\"")

            # Gap 3: surface contested/blocked predicates so the LLM knows
            # which elements are disputed or assumption-gated.
            contested = self._matter_model.issues.get_predicates_by_status(
                focus_issue_id, statuses=("contested", "blocked")
            )
            for cp in contested[:3]:
                status = cp.get("status", "")
                desc = cp.get("description", "")
                if status == "contested":
                    lines.append(f"  ⚠ CONTESTED element: \"{desc}\" — parties disagree; "
                                 "present BOTH sides with supporting evidence.")
                elif status == "blocked":
                    lines.append(f"  🚫 BLOCKED element: \"{desc}\" — depends on an unresolved "
                                 "assumption; note the dependency, do not treat as established.")

            lines.append("  → Extract facts that support OR disprove these specific elements.")
            block = "\n".join(lines) + "\n"
            # Escape braces so the block is safe to pass through str.format() in the
            # ANALYZE_FINDINGS_PROMPT template — issue titles/predicates could contain
            # literal { } characters that would otherwise be misinterpreted as slots.
            return block.replace("{", "{{").replace("}", "}}"), pred_descs
        except Exception:
            return "", []

    def _enrich_search_term_with_issue_context(
        self, search_term: str, focus_issue_id: str
    ) -> list[str]:
        """Return additional grep terms derived from the issue's predicates (SO-4).

        Returns a list of independent search phrases drawn from predicate
        descriptions or the issue title. Each phrase is grep-compatible (no
        boolean operators). Returns [] on any error — enrichment is advisory,
        never blocks search.
        """
        if self._matter_model is None:
            return []
        try:
            predicates = self._matter_model.issues.get_predicates(focus_issue_id, limit=1)
            if predicates:
                ctx = predicates[0].get("description", "")
            else:
                issue_row = self._matter_model.issues.get_issue(focus_issue_id)
                ctx = (issue_row or {}).get("title", "")
            kws = [
                w.strip(".,;:()")
                for w in ctx.split()
                if len(w.strip(".,;:()")) > 3
            ][:3]
            if kws:
                enrichment = " ".join(kws)
                if enrichment.lower() not in search_term.lower():
                    return [enrichment]
        except Exception:
            pass  # enrichment is advisory; never block search
        return []

    def _build_issue_profiles(self, issue_ids: list[str]) -> "dict[str, str]":
        """Build text profiles for SO-4 semantic attribution gate.

        A profile is the issue title concatenated with its open predicate descriptions.
        Used by _best_semantic_issue() to validate structural attribution proposals
        before they are committed to lead.focus_issue_id.

        Returns an empty dict if the matter model is unavailable (gate skips gracefully).
        """
        if self._matter_model is None or not issue_ids:
            return {}
        profiles: dict[str, str] = {}
        for iid in issue_ids:
            try:
                issue_row = self._matter_model.issues.get_issue(iid)
                parts: list[str] = [((issue_row or {}).get("title") or "")]
                predicates = self._matter_model.issues.get_predicates(iid, limit=4)
                for p in predicates:
                    desc = (p.get("description") or "").strip()
                    if desc:
                        parts.append(desc)
                profile = " ".join(p for p in parts if p)
                if profile.strip():
                    profiles[iid] = profile
            except Exception:
                pass  # profile for this issue unavailable; skip
        return profiles

    @staticmethod
    def _best_semantic_issue(
        text: str,
        issue_profiles: "dict[str, str]",
        min_score: float = 0.05,
        min_margin: float = 0.03,
    ) -> "Optional[str]":
        """Return the best-matching issue_id if semantically confident, else None.

        Uses word-level Jaccard similarity between the input text and issue profiles.
        Returns None (abstain) when:
        - The pool has fewer than 2 issues (no comparison possible).
        - Top score is below min_score (no meaningful overlap at all).
        - Margin between best and second-best is below min_margin (ambiguous).

        Abstaining is safer than wrong attribution: a None focus_issue_id simply
        means the search runs without issue bias, not that it is assigned to a wrong issue.
        """
        if not text or not issue_profiles or len(issue_profiles) < 2:
            return None
        scored = sorted(
            ((iid, _jaccard_similarity(text, profile))
             for iid, profile in issue_profiles.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        best_id, best_score = scored[0]
        second_score = scored[1][1]
        if best_score >= min_score and (best_score - second_score) >= min_margin:
            return best_id
        return None  # Abstain — not confident enough to attribute

    def _build_advocacy_gate_block(self) -> str:
        """Build a mandatory hedging gate for issues supported only by advocacy sources (SO-5).

        When an issue has advocacy_only=True in proof_state, synthesis MUST present
        conclusions about that issue with explicit hedging language — not confident
        statements. This converts the advisory annotation into a hard synthesis gate:
        the instruction block appears at the TOP of the synthesis prompt, before source
        calibration, so the LLM reads the constraint before any facts.

        Returns empty string when no advocacy-only issues exist (no prompt pollution).
        """
        if self._matter_model is None:
            return ""
        try:
            # Use targeted query — hits ix_proof_state_advocacy(matter_id, advocacy_only)
            # instead of fetching all proof states and filtering in Python.
            ps_rows = self._matter_model.proof_state.get_advocacy_only()
        except Exception as exc:
            _log.warning("_build_advocacy_gate_block: get_advocacy_only failed: %s", exc)
            return ""

        if not ps_rows:
            return ""

        # Restrict to open issues only (Tier 1 correctness fix: stale closed-issue
        # proof states must not over-constrain synthesis).
        try:
            open_issue_ids = {
                row["id"]
                for row in self._matter_model.db.execute(
                    "SELECT id FROM issue WHERE matter_id=? AND status='open'",
                    (self._matter_model.matter_id,),
                ).fetchall()
            }
        except Exception:
            open_issue_ids = None  # fallback: do not filter

        advocacy_issues = [
            ps for ps in ps_rows
            if open_issue_ids is None or ps.get("issue_id") in open_issue_ids
        ]
        if not advocacy_issues:
            return ""

        issue_index: dict = {}
        try:
            report = self._matter_model.get_issue_coverage_report()
            issue_index = {
                item["id"]: item.get("title", "Untitled")
                for item in report
                if item.get("id")
            }
        except Exception:
            pass

        lines = [
            "⚠ ADVOCACY-ONLY GATE (SO-5 — HARD CONTROL — DO NOT OVERRIDE):",
            "The following issues have ZERO support from operative or authoritative sources.",
            "STRUCTURAL REQUIREMENT: these claims MUST NOT appear in ## Key Findings or",
            "  ## Factual Background as established facts.",
            "REQUIRED: place these claims ONLY in ## Unsubstantiated Claims (Advocacy Sources Only).",
            "Mandatory format: 'Plaintiff/Defendant alleges [X] [Source]. No operative evidence corroborates.'",
            "Prohibited patterns: 'The evidence shows [X]', 'Based on the facts [X]', '[X] is established.'",
            "",
            "Issues with advocacy-only support (MUST go in ## Unsubstantiated Claims, NOT ## Key Findings):",
        ]
        for ps in advocacy_issues:
            iid = ps.get("issue_id", "")
            title = issue_index.get(iid, iid[:12] if iid else "unknown")
            suf = float(ps.get("sufficiency", 0.0))
            lines.append(
                f"  • {title}: {int(suf * 100)}% sufficiency — ADVOCACY SOURCES ONLY"
            )
        lines.append("")  # blank line before next block
        return "\n".join(lines) + "\n"

    # Trust rank for sorting facts: lower = higher trust = appears first.
    _TRUST_RANK = {
        "OPERATIVE": 0, "AUTHORITATIVE": 1, "PROCEDURAL": 2,
        "INFORMAL": 3, "DRAFT": 4, "POST_HOC": 5, "ADVOCACY": 6,
        "UNKNOWN": 7,
    }
    _ROLE_PATTERN = __import__("re").compile(r'^\[([A-Z_]+(?:\[[^\]]*\])?)\]')

    def _sort_facts_by_trust(self, facts: list[str]) -> list[str]:
        """Sort facts by source-role trust rank (Gap 2: trust-aware synthesis).

        Operative/authoritative facts appear first, advocacy last. Multi-source
        facts use the highest-trust role. Unknown/unprefixed facts sort last.
        """
        def _rank(fact: str) -> int:
            m = self._ROLE_PATTERN.match(fact)
            if not m:
                return 99
            tag = m.group(1)
            # Handle MULTI-SOURCE[OPERATIVE,ADVOCACY] → use best role
            if tag.startswith("MULTI-SOURCE"):
                inner = tag[len("MULTI-SOURCE["):-1] if tag.endswith("]") else ""
                roles = [r.strip() for r in inner.split(",")]
                return min(self._TRUST_RANK.get(r, 99) for r in roles) if roles else 99
            return self._TRUST_RANK.get(tag, 99)

        return sorted(facts, key=_rank)

    async def _cross_document_analysis(
        self, query: str, facts: list[str],
        quantitative_context: str = "",
    ) -> list[str]:
        """Derive cross-document insights: calculations, inconsistencies, absences.

        Uses FLASH to analyze the full fact set and produce derived findings
        that individual document reads cannot generate (cross-references,
        timing comparisons, dollar calculations, risk assessments).
        """
        facts_text = "\n".join(f"- {f}" for f in facts)
        analysis_corpus = f"{facts_text}\n{quantitative_context or ''}"
        deterministic_findings = self._derive_mna_coc_findings(
            query, facts, analysis_corpus
        )
        quant_section = ""
        if quantitative_context:
            quant_section = f"\n\nQUANTITATIVE DATA EXTRACTED:\n{quantitative_context}\n"
        operand_locks = self._build_mna_coc_completion_checklist(
            query, analysis_corpus
        )
        operand_lock_section = ""
        if operand_locks:
            operand_lock_section = (
                "\n\nOPERAND LOCKS / MUST-KEEP FINDINGS:\n"
                f"{operand_locks}\n"
                "If a possible derived finding conflicts with these locks, omit the conflicting finding.\n"
            )
        is_mna = self._is_mna_change_control_task(query)
        _ql = query.lower()
        _is_comparison = any(w in _ql for w in ("markup", "redline", "compare", "comparison", "deviation", "counterparty"))
        _is_regulatory = any(w in _ql for w in ("antitrust", "hsr", "merger review", "regulatory", "compliance"))
        if is_mna:
            persona = (
                "You are a senior M&A attorney performing cross-document analysis on "
                "extracted contract provisions."
            )
        elif _is_comparison:
            persona = (
                "You are a senior banking/finance attorney performing deviation analysis "
                "comparing an original term sheet against a lender's markup."
            )
        elif _is_regulatory:
            persona = (
                "You are a senior antitrust attorney performing regulatory risk analysis "
                "across transaction documents, market data, and enforcement precedent."
            )
        else:
            persona = (
                "You are an expert analyst performing cross-document analysis on "
                "extracted findings."
            )
        mna_categories = ""
        mna_operand_discipline = ""
        comparison_categories = ""
        regulatory_categories = ""
        if is_mna:
            mna_categories = (
                "5. STRUCTURAL ANALYSIS:\n"
                "   - Identify the transaction structure (e.g., reverse triangular merger).\n"
                "   - In a reverse triangular merger, the TARGET survives as a wholly-owned subsidiary.\n"
                "     Entity survival means anti-assignment clauses may NOT be triggered because no 'assignment'\n"
                "     occurs — but this is JURISDICTION-DEPENDENT and must be flagged as uncertain.\n"
                "   - For EACH contract with 'assignment by operation of law' language, separately analyze\n"
                "     whether entity survival avoids the trigger, citing the specific section.\n"
                "6. DOWNSTREAM RISKS: For EACH contract with 'indirect' change of control or 'direct or\n"
                "   indirect' ownership language, flag that a future change of the ACQUIRER's ownership\n"
                "   could re-trigger the provision. Name the specific contract and section.\n"
                "7. LEGAL FRAMEWORK: For EACH supply agreement or MSA with an anti-assignment clause,\n"
                "   apply UCC § 2-210 SPECIFICALLY to that contract — distinguish assignment of rights\n"
                "   from delegation of duties. Do not emit generic UCC analysis.\n"
                "8. OWNERSHIP STAKES: Report exact ownership percentages from JV/partnership/subsidiary\n"
                "   agreements. Distinguish ownership stakes from voting-equity CoC thresholds.\n\n"
            )
            mna_operand_discipline = (
                "OPERAND DISCIPLINE FOR CALCULATIONS:\n"
                "   - RSU acceleration: use the EXACT unvested RSU count from the employment agreement.\n"
                "     Use per-share transaction price or implied share price — do NOT use JV buy-out EBITDA multiples.\n"
                "   - Credit facility: use the DRAWN/OUTSTANDING amount, not the commitment/facility maximum.\n"
                "   - Revenue exposure: pair counterparty-specific TTM revenue (from schedules/disclosures)\n"
                "     with company total TTM revenue. Do NOT use minimum purchase commitments as the numerator.\n"
                "   - Carve-outs: compare the acquirer's product-specific or segment revenue to the threshold;\n"
                "     state whether it exceeds the threshold or falls short, and the consequence.\n"
                "   - If a contract card says 'ABSENT' for a provision, note that explicitly.\n\n"
            )
        if _is_comparison:
            comparison_categories = (
                "5. PROVISION-BY-PROVISION DEVIATION TABLE:\n"
                "   For EACH provision that differs between the original and the markup, produce:\n"
                "   '[DEVIATION] Provision: <name>; Original: <exact value>; Markup: <exact value>; Impact: <description>'\n"
                "   Cover ALL of: interest rate/SOFR floor/margin grid, commitment fees, financial covenants\n"
                "   (leverage ratio, FCCR, interest coverage — EACH separately), EBITDA add-back caps,\n"
                "   synergy add-backs, acquisition baskets, restricted payments, ECF sweep, events of default\n"
                "   (cross-default thresholds), change of control, anti-layering/MFN, reinvestment period,\n"
                "   MAE definition, extension options, reporting requirements.\n"
                "6. DOLLAR IMPACT CALCULATIONS (MANDATORY for every changed financial term):\n"
                "   Use the ACTUAL facility size, not a sub-amount. For each deviation:\n"
                "   - State the formula: facility_size × rate_change = annual_cost_impact\n"
                "   - Compute the result with actual numbers from the documents\n"
                "   - Example: '$175,000,000 × 0.25% = $437,500/year additional interest cost'\n"
                "7. COVENANT HEADROOM ANALYSIS:\n"
                "   Compare actual/projected financial metrics to both old and new covenant thresholds.\n"
                "   Calculate the headroom reduction for each tightened covenant.\n"
                "8. RISK RATING: Assign Red/Yellow/Green to each deviation:\n"
                "   Red = material adverse change requiring renegotiation\n"
                "   Yellow = concerning, needs negotiation attention\n"
                "   Green = acceptable/market standard\n"
                "9. MISSING PROVISIONS: Identify provisions in the original that were deleted in the markup,\n"
                "   and provisions added by the lender that weren't in the original.\n\n"
            )
        if _is_regulatory:
            regulatory_categories = (
                "5. MARKET CONCENTRATION ANALYSIS (MANDATORY when market share data exists):\n"
                "   For EACH geographic market where overlap exists:\n"
                "   - State both parties' market shares\n"
                "   - Compute HHI = sum of (share%)² for all competitors (e.g., 30% → 900)\n"
                "   - Compute post-merger HHI and delta (change in HHI)\n"
                "   - Flag markets where post-merger HHI > 1,800 AND delta > 200 (structural presumption)\n"
                "   - Rank markets by severity (highest HHI/delta first)\n"
                "6. HOT DOCUMENT IDENTIFICATION:\n"
                "   Flag specific internal documents (emails, memos, presentations) that contain\n"
                "   language an enforcement agency would use as evidence of anticompetitive intent.\n"
                "   Include: author, date, specific language quoted, and why it's problematic.\n"
                "7. ENFORCEMENT PRECEDENT ANALYSIS:\n"
                "   Connect specific prior enforcement actions to this transaction's facts.\n"
                "   State the precedent case, what happened, and how this transaction compares.\n"
                "8. REMEDY ANALYSIS:\n"
                "   Identify what divestitures or behavioral remedies would likely be required.\n"
                "   Compute whether proposed divestiture caps are sufficient for the required remedies.\n"
                "9. TIMELINE ANALYSIS:\n"
                "   Map out regulatory milestones against contractual deadlines (outside date,\n"
                "   extension periods) and flag where timelines are inadequate.\n"
                "10. QUANTITATIVE DEFENSE ANALYSIS:\n"
                "    Assess efficiency defenses, failing firm defense, entry analysis with specific data.\n\n"
            )
        prompt = (
            f"{persona} Given the facts and numeric data below "
            "(extracted from multiple documents), produce DERIVED FINDINGS that "
            "require comparing across documents or performing calculations.\n\n"
            "Produce findings in these categories:\n"
            "1. MANDATORY CALCULATIONS (perform ALL that the data supports):\n"
            "   - revenue exposure percent = counterparty TTM revenue / company TTM revenue × 100\n"
            "     (compute for EVERY counterparty where both figures are available)\n"
            "   - Any other quantitative comparison the extracted data supports.\n"
            "   Do NOT state that operands are unavailable if they appear in EXTRACTED FACTS or QUANTITATIVE DATA.\n"
            "   CRITICAL: Use schedule-disclosed TTM revenue (not minimum purchase commitments from the contract body)\n"
            "   as the numerator for revenue exposure. If both appear in the facts, state both and use TTM for the %.\n"
            "2. INCONSISTENCIES: Flag where different documents define the same "
            "concept differently.\n"
            "3. TIMING CONFLICTS: Identify where deadlines across documents "
            "create sequencing problems.\n"
            "4. ABSENCES: Note where a document that SHOULD have a provision "
            "(based on its type) appears to lack it.\n"
            "5. ADVERSE EVIDENCE: Flag specific quotes, admissions, or language from "
            "internal documents (emails, memos, board decks, strategy docs) that could be "
            "used adversarially in litigation, negotiation, or regulatory proceedings. "
            "For each, state: exact quote, speaker/author, document, and why it's problematic.\n"
            "6. STRATEGIC RECOMMENDATIONS: For each material issue identified, propose at "
            "least TWO alternative courses of action (e.g., renegotiate vs. accept, fix-it-first "
            "vs. consent decree, litigate vs. settle) with specific pros/cons referencing the "
            "extracted evidence. Include deadlines, target provisions, and dollar impacts where data supports it.\n"
            f"{mna_categories}"
            f"{mna_operand_discipline}"
            f"{comparison_categories}"
            f"{regulatory_categories}"
            f"QUERY CONTEXT: {query}\n\n"
            f"EXTRACTED FACTS:\n{facts_text}\n"
            f"{quant_section}\n"
            f"{operand_lock_section}\n"
            "Return a JSON array of derived finding strings. Each should be a complete, "
            "self-contained statement with specific numbers and section references where known. "
            "Format: [\"finding 1\", \"finding 2\", ...]\n"
            "Return ONLY the JSON array."
        )
        findings: list[str] = list(deterministic_findings)
        try:
            response = await self.client.complete(
                prompt=prompt,
                tier=ModelTier.FLASH,
                timeout=120.0,
                usage_label="cross_document_analysis",
            )
            import json as _json
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            derived = _json.loads(text)
            if isinstance(derived, list):
                for f in derived:
                    if not isinstance(f, str) or not f.strip():
                        continue
                    if self._derived_finding_conflicts_with_operand_locks(
                        f, analysis_corpus
                    ):
                        continue
                    item = f"[DERIVED] {f}"
                    if item not in findings:
                        findings.append(item)
                if findings:
                    logger.info(
                        "Cross-document analysis produced %d derived findings",
                        len(findings),
                    )
                    return findings
        except Exception as exc:
            logger.warning(
                "Cross-document analysis failed (proceeding without): %s", exc,
            )
        return findings

    async def _filter_facts_for_relevance(
        self, query: str, facts: list[str],
    ) -> list[str]:
        """Use a LITE-tier LLM call to filter facts for relevance to the query.

        Receives the full fact set and returns only those that are relevant
        to answering the user's question. This replaces hard numeric caps —
        the model decides what matters based on semantic relevance, not arbitrary
        truncation. Keeps everything on failure (fail-open, never lose evidence).
        """
        numbered = "\n".join(f"{i}: {f}" for i, f in enumerate(facts))
        prompt = (
            "You are filtering evidence for a synthesis step. Given the user's query "
            "and a numbered list of facts extracted during investigation, return ONLY "
            "the indices of facts that are RELEVANT to answering the query. A fact is "
            "relevant if it directly supports, contradicts, or contextualizes the answer. "
            "Drop only facts that are clearly unrelated to the query.\n\n"
            "IMPORTANT: When in doubt, KEEP the fact. It is far better to include a "
            "marginally relevant fact than to drop a needed one.\n\n"
            f"USER QUERY: {query}\n\n"
            f"FACTS (numbered):\n{numbered}\n\n"
            "Return a JSON array of integer indices to KEEP. Example: [0, 1, 3, 5, 7]\n"
            "Return ONLY the JSON array, nothing else."
        )
        try:
            response = await self.client.complete(
                prompt=prompt,
                tier=ModelTier.LITE,
                timeout=30.0,
                usage_label="evidence_relevance_filter",
            )
            import json as _json
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            indices = _json.loads(text)
            if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                filtered = [facts[i] for i in indices if 0 <= i < len(facts)]
                if filtered:
                    logger.info(
                        "Evidence filter: %d/%d facts kept for query",
                        len(filtered), len(facts),
                    )
                    return filtered
        except Exception as exc:
            logger.warning(
                "Evidence relevance filter failed (keeping all %d facts): %s",
                len(facts), exc,
            )
        return facts

    # ------------------------------------------------------------------
    # PR.3: mandatory capped coverage and missingness for synthesis context
    # ------------------------------------------------------------------
    #
    # Token estimate uses len(text)//4, matching GeminiClient._parse_usage_metadata.
    # Caps are intentionally small: synthesis already gets evidence and candidate
    # sections, so the coverage and gap sections exist to force the LLM to reckon
    # with proof shape, not to dump the full matter model into the prompt.

    # MVP.6: coverage/gap caps moved onto RLMConfig.packet_budget.
    # The trim/count shape constants stay here because they govern
    # per-line rendering, not prompt-budget policy.
    _PACKET_ISSUE_TITLE_TRIM = 80
    _PACKET_GAP_DESC_TRIM = 160
    _PACKET_GAP_DEPS_PER_GAP = 2

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return (len(text) + 3) // 4 if text else 0

    def _get_packet_budget(self) -> "PacketBudget":
        """MVP.6: return the prompt budget object, with a fail-safe default
        for RLMEngine.__new__ test paths that skip __init__."""
        cfg = getattr(self, "config", None)
        if cfg is None:
            return PacketBudget()
        budget = getattr(cfg, "packet_budget", None)
        return budget if budget is not None else PacketBudget()

    def _cap_text_by_tokens(self, text: str, cap_tokens: int) -> str:
        """Deterministic line-wise truncation to a token cap.

        Line-wise preserves structural integrity of blocks with bullets
        or key:value lines; char-trim only kicks in when a single line
        does not fit. An ellipsis marker records omitted content.
        """
        if not text:
            return ""
        if cap_tokens <= 0 or self._estimate_tokens(text) <= cap_tokens:
            return text
        kept: list[str] = []
        used = 0
        lines = text.splitlines()
        for i, line in enumerate(lines):
            cost = self._estimate_tokens(line + "\n")
            if used + cost > cap_tokens:
                remaining = len(lines) - i
                kept.append(
                    f"… {remaining} more line(s) omitted under "
                    f"{cap_tokens}-token cap"
                )
                break
            kept.append(line)
            used += cost
        return "\n".join(kept)

    def _resolve_requested_issue_id(
        self, query: str, coverage_rows: "list[dict]"
    ) -> "Optional[str]":
        """Deterministically map the user query to an open issue id, or None.

        Reuses the existing semantic attribution gate. When only one open issue
        exists (below the gate's 2-issue floor), fall back to a conservative
        token-overlap check so PR.3 AC #5 ("requested issue cannot lose its
        material proof gap") still fires on single-issue fixtures.
        """
        if not query or not coverage_rows:
            return None
        open_ids = [row["id"] for row in coverage_rows if row.get("id")]
        if not open_ids:
            return None
        profiles = self._build_issue_profiles(open_ids)
        match = self._best_semantic_issue(query, profiles)
        if match:
            return match
        # Single-issue fallback: the semantic gate abstains when profile pool
        # has fewer than 2 entries. Only attach the query to that issue if it
        # genuinely shares a non-trivial token with the title.
        if len(open_ids) == 1:
            only_id = open_ids[0]
            title = next(
                (r.get("title") or "" for r in coverage_rows if r.get("id") == only_id),
                "",
            )
            q_tokens = {
                t.strip(".,;:()\"'").lower()
                for t in query.split()
                if len(t.strip(".,;:()\"'")) >= 4
            }
            t_tokens = {
                t.strip(".,;:()\"'").lower()
                for t in title.split()
                if len(t.strip(".,;:()\"'")) >= 4
            }
            if q_tokens & t_tokens:
                return only_id
        return None

    @staticmethod
    def _trim(text: str, limit: int) -> str:
        if text is None:
            return ""
        text = str(text)
        return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"

    def _build_capped_issue_coverage_section(self, query: str) -> str:
        """Capped, deterministic coverage section for the synthesis context packet.

        Mandatory for SO-4 — synthesis must see per-issue proof shape, not just
        supporting assertions. Ordering: requested issue first, then has_proof_gap,
        then weakest coverage, then highest materiality, then title, then id.
        """
        if self._matter_model is None:
            return ""
        try:
            rows = self._matter_model.get_issue_coverage_report()
        except Exception:
            return ""
        if not rows:
            return ""

        requested_id = self._resolve_requested_issue_id(query or "", rows)

        def _sort_key(item: dict) -> tuple:
            iid = item.get("id", "")
            return (
                0 if iid == requested_id else 1,
                0 if item.get("has_proof_gap") else 1,
                float(item.get("coverage_fraction") or 0.0),
                -float(item.get("materiality") or 0.0),
                (item.get("title") or "").lower(),
                iid,
            )

        ordered = sorted(rows, key=_sort_key)
        header = "Issue Coverage (proof shape across open issues):"
        lines = [header]
        used = self._estimate_tokens(header + "\n")
        cap = self._get_packet_budget().coverage_tokens
        shown = 0
        omitted = 0
        for item in ordered:
            title = self._trim(item.get("title") or "Untitled", self._PACKET_ISSUE_TITLE_TRIM)
            verified = int(item.get("verified_supporting_count") or 0)
            candidate = int(item.get("candidate_supporting_count") or 0)
            cnt = verified + candidate
            frac = float(item.get("coverage_fraction") or 0.0)
            v_frac = float(item.get("verified_coverage_fraction") or 0.0)
            gap_flag = " ⚠ PROOF GAP" if item.get("has_proof_gap") else ""
            requested_flag = " (requested)" if item.get("id") == requested_id else ""
            # P0.2: render verified and candidate lanes distinctly so
            # synthesis cannot conflate candidate support with verified
            # proof. Shape: [verified% / advisory%] title: V verified, C candidate
            line = (
                f"  [{v_frac:.0%} verified / {frac:.0%} advisory] {title}{requested_flag}: "
                f"{verified} verified, {candidate} candidate{gap_flag}"
            )
            cost = self._estimate_tokens(line + "\n")
            if cap > 0 and used + cost > cap and shown > 0:
                omitted = len(ordered) - shown
                break
            lines.append(line)
            used += cost
            shown += 1

        if omitted:
            lines.append(f"  … {omitted} more omitted under {cap}-token cap")
        return "\n".join(lines)

    def _build_capped_gap_section(
        self, query: str, requested_issue_id: "Optional[str]"
    ) -> str:
        """Capped, deterministic missingness section for the synthesis context packet.

        Mandatory for SO-7. High-materiality gaps must land in synthesis even
        when the LITE section selector would drop everything else. Bucketed
        selection guarantees that a material gap on the requested issue is
        never lost to a higher-materiality gap on an unrelated issue.
        """
        if self._matter_model is None:
            return ""
        try:
            all_gaps = self._matter_model.gaps.open_gaps(min_materiality=0.0)
        except Exception:
            return ""
        if not all_gaps:
            return ""

        def _affects_requested(gap: dict) -> bool:
            if not requested_issue_id:
                return False
            for dep in gap.get("dependencies") or []:
                if (
                    dep.get("affected_type") == "issue"
                    and dep.get("affected_id") == requested_issue_id
                ):
                    return True
            return False

        def _bucket(gap: dict) -> int:
            mat = float(gap.get("materiality_score") or 0.0)
            gtype = gap.get("gap_type") or ""
            if (
                gtype == "missing_issue_predicate"
                and _affects_requested(gap)
                and mat >= 0.4
            ):
                return 1
            if gtype == "missing_issue_predicate" and mat >= 0.7:
                return 2
            if mat >= 0.7:
                return 3
            if mat >= 0.4:
                return 4
            return 5  # filtered out

        def _sort_key(gap: dict) -> tuple:
            return (
                _bucket(gap),
                -float(gap.get("materiality_score") or 0.0),
                gap.get("gap_type") or "",
                (gap.get("description") or "").lower(),
                gap.get("id") or "",
            )

        candidates = [g for g in all_gaps if _bucket(g) <= 4]
        if not candidates:
            return ""
        candidates.sort(key=_sort_key)

        header = "Known Gaps (missingness that must be acknowledged):"
        lines = [header]
        used = self._estimate_tokens(header + "\n")
        cap = self._get_packet_budget().gap_tokens
        shown = 0

        def _render(gap: dict) -> list[str]:
            gtype = (gap.get("gap_type") or "unknown").replace("_", " ")
            desc = self._trim(gap.get("description") or "", self._PACKET_GAP_DESC_TRIM)
            mat = float(gap.get("materiality_score") or 0.0)
            label = "HIGH" if mat >= 0.7 else "MED" if mat >= 0.4 else "LOW"
            out = [f"  [{label}] {gtype}: {desc}"]
            deps = sorted(
                gap.get("dependencies") or [],
                key=lambda d: (
                    d.get("affected_type") or "",
                    d.get("affected_id") or "",
                ),
            )[: self._PACKET_GAP_DEPS_PER_GAP]
            if deps:
                dep_strs = [
                    f"{d.get('affected_type','?')}:{(d.get('affected_id','') or '')[:8]}"
                    for d in deps
                ]
                out.append(f"         Affects: {', '.join(dep_strs)}")
            return out

        # Special case: if the requested issue has a qualifying proof gap in
        # bucket 1, force-include it first and hard-trim until it fits.
        if requested_issue_id:
            forced = next(
                (g for g in candidates if _bucket(g) == 1),
                None,
            )
            if forced is not None:
                rendered = _render(forced)
                cost = sum(self._estimate_tokens(l + "\n") for l in rendered)
                if cap > 0 and used + cost > cap:
                    # Hard-trim the description to fit.
                    mat = float(forced.get("materiality_score") or 0.0)
                    label = "HIGH" if mat >= 0.7 else "MED" if mat >= 0.4 else "LOW"
                    gtype = (forced.get("gap_type") or "unknown").replace("_", " ")
                    desc = forced.get("description") or ""
                    budget_chars = max(20, (cap - used) * 4 - len(f"  [{label}] {gtype}: "))
                    rendered = [f"  [{label}] {gtype}: {self._trim(desc, budget_chars)}"]
                    cost = sum(self._estimate_tokens(l + "\n") for l in rendered)
                lines.extend(rendered)
                used += cost
                shown += 1
                candidates = [g for g in candidates if g.get("id") != forced.get("id")]

        for gap in candidates:
            rendered = _render(gap)
            cost = sum(self._estimate_tokens(l + "\n") for l in rendered)
            if cap > 0 and used + cost > cap:
                break
            lines.extend(rendered)
            used += cost
            shown += 1

        total = len([g for g in all_gaps if _bucket(g) <= 4])
        omitted = total - shown
        if omitted > 0:
            lines.append(f"  … {omitted} more omitted under {cap}-token cap")
        return "\n".join(lines)

    def _build_trust_abstention_block(self, query: str) -> str:
        """P0.2 AC #4: mandatory synthesis instruction that refuses
        definitive claims when support is candidate-only, stale-only,
        rejected-only, advocacy-only, or gap-blocked.

        The block is always present when the matter has any open
        issues so the LLM sees the abstention rule unconditionally,
        but the body lists per-issue constraints only for issues that
        fail the verified-support test (so a fully-verified matter
        does not get a pile of irrelevant warnings)."""
        if self._matter_model is None:
            return ""
        try:
            rows = self._matter_model.get_issue_coverage_report()
        except Exception:
            return ""
        if not rows:
            return ""
        lines = [
            "Trust and Abstention Rules (MANDATORY — do not override):",
            "  - Definitive claims require VERIFIED support. If an",
            "    issue's verified support is zero, frame findings as",
            "    provisional or unresolved — never as proven.",
            "  - Candidate support must be described as provisional,",
            "    e.g. \"candidate evidence suggests…\" — never as",
            "    established fact.",
            "  - Stale or rejected intelligence has been reviewed-out",
            "    by a human; do not resurrect it even if it appears",
            "    in the background.",
            "  - Advocacy-only support (complaints, briefs, demand",
            "    letters) must stay hedged; it is allegation, not",
            "    proof.",
        ]
        per_issue: list[str] = []
        for item in rows:
            title = self._trim(
                item.get("title") or "Untitled",
                self._PACKET_ISSUE_TITLE_TRIM,
            )
            v_cnt = int(item.get("verified_supporting_count") or 0)
            c_cnt = int(item.get("candidate_supporting_count") or 0)
            has_gap = bool(item.get("has_proof_gap"))
            if v_cnt > 0 and not has_gap:
                continue  # verified support — no per-issue warning needed
            if v_cnt == 0 and c_cnt > 0:
                per_issue.append(
                    f"    * {title}: candidate-only support "
                    f"({c_cnt}) — treat as provisional."
                )
            elif v_cnt == 0 and c_cnt == 0:
                per_issue.append(
                    f"    * {title}: no eligible support — treat as "
                    "unresolved proof gap."
                )
            elif has_gap:
                per_issue.append(
                    f"    * {title}: proof-gap blocked ({v_cnt} "
                    f"verified + {c_cnt} candidate) — call out the "
                    "gap explicitly."
                )
        if per_issue:
            lines.append("  Issue-specific abstention instructions:")
            lines.extend(per_issue)
        return "\n".join(lines)

    async def _assemble_context_packet(
        self, state: InvestigationState, findings_text: str,
        policy_audience: str = "clean",
    ) -> ContextPacketBuild:
        """Dynamically build the context packet for synthesis.

        Uses a LITE call to decide which OPTIONAL sections are relevant to
        the query: source_calibration, decision_context, entities,
        relationships, quantitative, and citations.

        The advocacy gate (SO-5) and the evidence block are always included.
        Issue coverage (SO-4) and gap summaries (SO-7) are also mandatory
        per PR.3: high-materiality proof gaps and per-issue coverage shape
        must not be selector-gated. Both are capped deterministically by
        token budget so a large matter cannot blow synthesis context.

        MVP.4 SO-5: under policy_audience='clean' (default), any content
        that references a privileged document's path or basename is
        scrubbed from the assembled packet. This is a defense-in-depth
        filter — substrate queries (coverage, gap, proof) already drop
        privileged assertions, but state.citations / state.entities and
        findings_text can still hydrate from upstream paths that don't
        know about privilege. The post-assembly scrub catches those.
        """
        # MVP.6: explicit opt-in registry for optional sections. New
        # store summaries default off; the baseline optional set below is
        # the set of sections that shipped before MVP.6. Adding a new
        # optional section requires registering its key here; silent
        # prompt inflation is not possible.
        # `is None` (not `or`) so an explicitly empty override means "no
        # optional sections", not "use the default". Empty frozenset is
        # falsy in Python; the bare `or` would silently fall through.
        _override = getattr(self, "_enabled_optional_sections", None)
        enabled_optional = (
            _override if _override is not None else _DEFAULT_OPTIONAL_SECTIONS
        )
        budget = self._get_packet_budget()

        # Gather all candidate optional sections (label → content).
        # Only sections with real content, already in the allowlist, and
        # within the per-section cap survive.
        candidates: dict[str, str] = {}

        def _add_optional(key: str, content: str) -> None:
            if key not in enabled_optional:
                return
            if not content or not content.strip():
                return
            capped = self._cap_text_by_tokens(
                content.rstrip(), budget.per_optional_section_tokens
            )
            if capped:
                candidates[key] = capped

        source_cal = self._build_source_calibration(state)
        if source_cal:
            _add_optional(
                "source_calibration",
                "Source Calibration (read before analyzing facts):\n" + source_cal,
            )
        decision_ctx = self._build_decision_context_block()
        if decision_ctx:
            _add_optional("decision_context", decision_ctx)
        entities_text = state.get_entities_formatted()
        if entities_text:
            _add_optional(
                "entities", "Key Entities Identified:\n" + entities_text
            )
        relationships = self._build_structured_relationships()
        if relationships:
            _add_optional(
                "relationships",
                "Structured Relationships (subject-predicate-object):\n" + relationships,
            )
        quant = self._build_quant_summary()
        _quant_is_mandatory = self._is_extraction_task(getattr(state, "query", "") or "")
        if quant and _quant_is_mandatory:
            pass  # handled below as mandatory section
        elif quant:
            _add_optional("quantitative", "Quantitative Summary:\n" + quant)
        citations_text = state.get_citations_formatted()
        _citations_block = ""
        if citations_text:
            _citations_block = self._cap_text_by_tokens(
                "Documentary Citations:\n" + citations_text,
                budget.per_optional_section_tokens * 2,
            )

        # Use LITE to decide which of the capped + allowed optional
        # sections are relevant to the query.
        selected_keys = list(candidates.keys())
        if candidates:
            try:
                selected_keys = await self._select_relevant_sections(
                    state.query, candidates
                )
            except Exception:
                pass  # on failure, include everything — safe default

        # Assemble in fixed priority order so the total-cap pass below
        # cannot crowd out mandatory sections.
        ordered: list[tuple[str, str, bool]] = []  # (key, text, mandatory)

        # Extraction-task detection: if the query asks for comprehensive extraction,
        # inject instructions that ensure exhaustive output with section refs and calculations.
        mna_checklist = self._build_mna_coc_completion_checklist(
            getattr(state, "query", "") or "",
            "\n".join(part for part in (findings_text, citations_text, quant) if part),
        )
        if mna_checklist:
            ordered.append(("mna_coc_completion_checklist", mna_checklist, True))

        extraction_instruction = self._build_extraction_instructions(
            getattr(state, "query", "") or ""
        )
        if extraction_instruction:
            ordered.append(("extraction_instructions", extraction_instruction, True))

        if quant and _quant_is_mandatory:
            ordered.append(("quantitative", "Quantitative Summary:\n" + quant, True))

        # Provision comparison summary for comparison tasks
        _prov_summary = self._build_provision_comparison_summary(state)
        if _prov_summary:
            ordered.append(("provision_comparisons", _prov_summary, True))

        # Regulatory data summary for antitrust/regulatory tasks
        _reg_summary = self._build_regulatory_data_summary(state)
        if _reg_summary:
            ordered.append(("regulatory_evidence", _reg_summary, True))

        # Adverse evidence / hot documents
        _adv_summary = self._build_adverse_evidence_summary(state)
        if _adv_summary:
            ordered.append(("adverse_evidence", _adv_summary, True))

        contract_coverage = self._build_material_contract_coverage_section(state)
        if contract_coverage:
            ordered.append(("material_contract_coverage", contract_coverage, True))

        workflow_quality = self._build_workflow_quality_section(state)
        if workflow_quality.strip():
            ordered.append(("workflow_quality", workflow_quality.rstrip(), True))

        advocacy_gate = self._build_advocacy_gate_block()
        if advocacy_gate.strip():
            ordered.append(("advocacy_gate", advocacy_gate.rstrip(), True))

        query = getattr(state, "query", "") or ""
        abstention = self._build_trust_abstention_block(query)
        if abstention.strip():
            ordered.append(("trust_abstention_gate", abstention.rstrip(), True))

        coverage_section = self._build_capped_issue_coverage_section(query)
        if coverage_section:
            ordered.append(("issue_coverage", coverage_section, True))
        requested_id: "Optional[str]" = None
        try:
            coverage_rows = (
                self._matter_model.get_issue_coverage_report()
                if self._matter_model else []
            )
            requested_id = self._resolve_requested_issue_id(query, coverage_rows)
        except Exception:
            requested_id = None
        gap_section = self._build_capped_gap_section(query, requested_id)
        if gap_section:
            ordered.append(("high_materiality_gaps", gap_section, True))

        # Evidence is mandatory and comes after mandatory proof framing.
        # Clean-mode findings scrub runs first.
        clean_findings = findings_text or "No specific findings accumulated"
        if policy_audience == "clean" and self._matter_model is not None:
            clean_findings = self._scrub_privileged_references(clean_findings)
        ordered.append((
            "evidence", "Evidence Gathered:\n" + clean_findings, True,
        ))

        # Citations are mandatory — synthesis needs document references for
        # grounded answers. Without them, findings lose provenance.
        if _citations_block:
            clean_citations = _citations_block
            if policy_audience == "clean" and self._matter_model is not None:
                clean_citations = self._scrub_privileged_references(clean_citations)
            ordered.append(("citations", clean_citations, True))

        # Optional sections in stable order behind mandatory ones so the
        # total-cap pass drops them first.
        _ORDER = [
            "source_calibration", "decision_context", "entities",
            "relationships", "quantitative",
        ]
        for key in _ORDER:
            if key in selected_keys and key in candidates:
                ordered.append((key, candidates[key], False))

        # Total-cap pass. When synthesis_total_tokens is 0 (unlimited),
        # include everything — let the model use its full context window.
        # Otherwise, mandatory sections always go in and optional sections
        # drop once the accumulator would exceed the cap.
        sections: list[str] = []
        used_tokens = 0
        omitted: list[str] = []
        total_cap = budget.synthesis_total_tokens
        for key, text, mandatory in ordered:
            cost = self._estimate_tokens(text + "\n\n")
            if total_cap <= 0 or mandatory or used_tokens + cost <= total_cap:
                sections.append(text)
                used_tokens += cost
            else:
                omitted.append(key)
        if omitted:
            # Record omissions in a shape future context_assembly_event
            # rows can consume directly.
            try:
                state.findings.setdefault("_packet_omissions", []).append({
                    "omitted_sections": list(omitted),
                    "synthesis_total_tokens": total_cap,
                    "used_tokens": used_tokens,
                })
            except Exception:
                pass
            # Also log once so an attentive operator can see drops.
            logger.info(
                "MVP.6 packet omitted %d optional section(s) under %d-token cap: %s",
                len(omitted), total_cap, ",".join(omitted),
            )

        packet = "\n\n".join(sections)
        # Same scrub over the whole packet catches any privileged
        # document paths that leaked into entity/citation/relationship
        # strings built from state.*.
        if policy_audience == "clean" and self._matter_model is not None:
            packet = self._scrub_privileged_references(packet)

        # Build per-output dependency manifest (SO-1, SO-5).
        selected_keys_set = tuple(
            k for k, _, mandatory in ordered if not mandatory or k in ("issue_coverage", "advocacy_gate")
        )
        manifest_hash: Optional[str] = None
        consumed_refs: list[tuple[str, str]] = []
        if self._matter_model is not None:
            try:
                if coverage_rows:
                    for row in coverage_rows:
                        rid = row.get("id")
                        if rid:
                            consumed_refs.append(("issue", str(rid)))
                assertion_ids = getattr(state, "_consumed_assertion_ids", None)
                if assertion_ids:
                    for aid in assertion_ids:
                        consumed_refs.append(("assertion", str(aid)))
                ns_keys = ["assertions", "issues", "evidence_edges"]
                if "source_calibration" in selected_keys:
                    ns_keys.append("proof_state")
                if "quantitative" in selected_keys:
                    ns_keys.append("quant_facts")
                manifest_hash = self._matter_model.build_output_dependency_manifest(
                    purpose="synthesis",
                    policy_audience=policy_audience,
                    taint_class=self._resolve_taint_default(),
                    object_refs=consumed_refs,
                    namespace_keys=ns_keys,
                )
            except Exception as exc:
                logger.warning("build_output_dependency_manifest failed: %s", exc)

        return ContextPacketBuild(
            text=packet,
            dependency_manifest_hash=manifest_hash,
            consumed_object_refs=tuple(consumed_refs),
            selected_sections=selected_keys_set,
            omitted_sections=tuple(omitted),
        )

    def _scrub_privileged_references(self, text: str) -> str:
        """Remove lines that mention a privileged document's relative path
        or basename. Conservative — a line is dropped when any privileged
        doc reference matches, replaced with a single "[withheld under
        clean policy]" marker so downstream readers see something was
        filtered rather than silence.
        """
        if not text or self._matter_model is None:
            return text
        try:
            rows = self._matter_model.db.execute(
                """SELECT di.relative_path
                   FROM document_card dc
                   JOIN document_inventory di ON di.id = dc.doc_id
                   WHERE di.matter_id=? AND dc.privilege_flag=1""",
                (self._matter_model.matter_id,),
            ).fetchall()
        except Exception:
            return text
        needles = set()
        for r in rows:
            path = (r["relative_path"] or "").strip()
            if not path:
                continue
            needles.add(path)
            # Add basename (last path segment) as additional needle so
            # references like "memo.docx" get caught even without the
            # full relative path.
            norm = path.replace("\\", "/")
            base = norm.rsplit("/", 1)[-1]
            if base and base != path:
                needles.add(base)
        if not needles:
            return text
        kept_lines = []
        withheld_emitted = False
        # P0.5 commit 3: audit the scrub decisions through the guard
        # so the content_policy_audit log records every
        # synthesis_context withhold event.
        _guard = getattr(self._matter_model, "content_policy", None)
        for line in text.splitlines():
            line_l = line.lower()
            matched = [n for n in needles if n.lower() in line_l]
            if matched:
                if _guard is not None:
                    # Use the first matched needle as the subject id
                    # so audit rows are keyed to the document, not
                    # to arbitrary line numbers.
                    try:
                        from ..matter.trust import ContentPurpose
                        _guard.decide(
                            purpose=ContentPurpose.SYNTHESIS_CONTEXT,
                            subject_kind="document",
                            subject_id=matched[0],
                            policy_audience="clean",
                            privilege_flag=True,
                            note="synthesis_packet_line_scrub",
                        )
                    except sqlite3.Error as _exc:
                        logger.warning(
                            "synthesis scrub: content_policy_audit write failed: %s",
                            _exc,
                        )
                if not withheld_emitted:
                    kept_lines.append("[withheld under clean policy]")
                    withheld_emitted = True
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines)

    async def _select_relevant_sections(
        self, query: str, candidates: dict[str, str]
    ) -> list[str]:
        """LITE call: given the query, decide which context sections are useful.

        Returns the keys from candidates that should be included in the
        synthesis context packet.
        """
        # Build a brief summary of each candidate for the selector
        summaries = []
        for key, content in candidates.items():
            # First 150 chars as preview
            preview = content[:150].replace("\n", " ")
            summaries.append(f"- {key}: {preview}...")

        prompt = (
            "You are preparing a context packet for a senior analyst who will "
            "analyze evidence and write a professional memorandum.\n\n"
            f"Query: {query}\n\n"
            "Available context sections:\n"
            + "\n".join(summaries)
            + "\n\nWhich sections are relevant to answering this query? "
            "Return ONLY a JSON array of the relevant section keys. "
            "Include a section if it would help the analyst reason about "
            "the query. Exclude sections that are irrelevant or would be noise.\n"
            "Example: [\"entities\", \"citations\"]"
        )

        response = await self.client.complete(
            prompt,
            tier=ModelTier.LITE,
            json_mode=True,
            usage_label="section_selection",
        )

        import json as _json
        selected = _json.loads(response)
        if isinstance(selected, list):
            # Validate keys
            return [k for k in selected if k in candidates]
        return list(candidates.keys())  # fallback

    def _build_source_calibration(self, state: InvestigationState) -> str:
        """
        Build a source-role calibration block for the synthesis prompt (SO-5).

        Queries the matter model for assertion counts grouped by source_role so
        the LLM knows which facts came from advocacy vs. operative sources.
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

        _domain = self._resolve_active_domain(state)
        _role_labels = _DOMAIN_ROLE_CALIBRATION_LABELS.get(_domain, _DOMAIN_ROLE_CALIBRATION_LABELS["legal"])

        lines = ["The following facts were extracted from documents with these source roles:"]
        for row in rows:
            role = row["source_role"] if row["source_role"] else "unknown"
            label = _role_labels.get(role, f"{role.upper()} — calibrate appropriately")
            lines.append(f"  • {row['cnt']} assertions from {label}")

        try:
            side_rows = self._matter_model.db.execute(
                """SELECT COALESCE(ao.source_side, 'neutral/unknown') AS side,
                          COUNT(DISTINCT a.id) AS cnt
                   FROM assertion a
                   JOIN assertion_occurrence ao ON ao.assertion_id = a.id
                   WHERE a.matter_id = ?
                   GROUP BY side ORDER BY cnt DESC""",
                (self._matter_model.matter_id,),
            ).fetchall()
            if side_rows:
                lines.append("\nStakeholder-side origin of extracted facts (may overlap):")
                for sr in side_rows:
                    lines.append(f"  • {sr['cnt']} assertions from {sr['side']} documents")
        except Exception:
            pass

        try:
            overrides = self._matter_model.trust_overrides.list_all()
            if overrides:
                lines.append("\nUser-set trust overrides (MANDATORY — respect these exactly):")
                for o in overrides:
                    pattern = o.get("document_pattern", "")
                    level = o.get("trust_level", "normal")
                    note = o.get("note", "")
                    if level == "low":
                        override_label = (
                            "LOW TRUST — present all facts from this document as ALLEGED "
                            "regardless of source role"
                        )
                    elif level == "high":
                        override_label = (
                            "HIGH TRUST — treat facts from this document as OPERATIVE/AUTHORITATIVE "
                            "even if source role would suggest lower trust"
                        )
                    else:
                        continue
                    line = f"  • [{level.upper()}] '{pattern}': {override_label}"
                    if note:
                        line += f" — Reason: {note}"
                    lines.append(line)
        except Exception:
            pass

        try:
            annotations = self._matter_model.annotations.list_recent(limit=8)
            if annotations:
                lines.append("\nUser strategic annotations for specific documents:")
                for ann in annotations:
                    doc = (ann.get("document_pattern") or "")
                    txt = (ann.get("annotation_text") or "")[:200]
                    ann_type = (ann.get("annotation_type") or "strategic").upper()
                    lines.append(f"  • [{ann_type}] '{doc}': {txt}")
        except Exception:
            pass

        lines.append(_DOMAIN_TRUST_HIERARCHY.get(_domain, _DOMAIN_TRUST_HIERARCHY["legal"]))
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

            chain = self._matter_model.reconcile_payment_chain()
            invoice_rows = self._matter_model.reconcile_invoice_chain()
            conflicts = self._matter_model.quant.get_conflicts()
            date_facts = self._matter_model.quant.get_by_kind("date", limit=8)
            rate_facts = self._matter_model.quant.get_by_kind("rate", limit=5)
        except Exception:
            return "Quantitative data unavailable."

        lines = [f"Extracted {total_count} numeric facts."]

        # SO-6: Hard threshold constraints — detected violations MUST appear in synthesis.
        # Run threshold computation (idempotent; records gaps to gap store as side effect).
        try:
            violations = self._matter_model.compute_quant_thresholds()
            if violations:
                lines.append(
                    "⚠ QUANTITATIVE THRESHOLD VIOLATIONS (SO-6 — MANDATORY in Financial Analysis):"
                )
                for v in violations:
                    level = v.get("level", "MED")
                    desc = v.get("description", "")
                    lines.append(f"  [{level}] {desc}")
                lines.append(
                    "  → These items MUST appear in the Financial Analysis section with source citations."
                )
        except Exception:
            pass

        # M&A diligence uses narrow operands that can be buried under larger
        # headline figures. Surface those operands before broad category totals.
        try:
            mna_keywords = [
                "revenue", "ttm", "outstanding", "drawn", "revolving",
                "prepayment", "rsu", "restricted stock", "share",
                "exchange ratio", "coverage", "aggregate limit", "run-off",
                "ebitda", "buy-out", "buyout", "termination fee",
                "credit", "required consents",
            ]
            # Derive entity keywords from extracted contract cards instead
            # of hardcoding benchmark-specific names.
            try:
                te_cards = self._matter_model.typed_evidence.list_by_kind(
                    "contract_card", limit=20
                )
                for _cc_row in te_cards:
                    _cc_p = _cc_row.get("payload_json")
                    if isinstance(_cc_p, str):
                        import json as _json_mod
                        try:
                            _cc_p = _json_mod.loads(_cc_p)
                        except Exception:
                            continue
                    if isinstance(_cc_p, dict):
                        for _field in ("counterparty", "contract_name"):
                            _val = (_cc_p.get(_field) or "").strip()
                            if _val and len(_val) > 2:
                                for _word in _val.lower().split():
                                    if len(_word) > 3 and _word not in mna_keywords:
                                        mna_keywords.append(_word)
            except Exception:
                pass
            quant_rows = self._matter_model.quant.list_all(limit=200)
            mna_rows: list[dict] = []
            seen_quant_rows: set[str] = set()
            for row in quant_rows:
                raw = str(row.get("raw_text") or "")
                subject = " ".join(
                    str(row.get(k) or "")
                    for k in ("subject_type", "subject_id", "unit")
                )
                haystack = f"{raw} {subject}".lower()
                if not any(k in haystack for k in mna_keywords):
                    continue
                key = (row.get("quant_kind"), row.get("subject_id"), raw[:160])
                key_s = repr(key)
                if key_s in seen_quant_rows:
                    continue
                seen_quant_rows.add(key_s)
                mna_rows.append(row)
                if len(mna_rows) >= 35:
                    break
            if mna_rows:
                lines.append(
                    "M&A calculation operand candidates (prefer legally narrower operands over broader headline figures):"
                )
                for row in mna_rows:
                    kind = row.get("quant_kind") or "quant"
                    subject = row.get("subject_id") or row.get("subject_type") or kind
                    raw = str(row.get("raw_text") or "")[:140]
                    value = row.get("amount_value")
                    if value is None:
                        value = row.get("rate_value")
                    if value is None:
                        value = row.get("date_value") or row.get("date_end_value")
                    if isinstance(value, (int, float)):
                        value_f = float(value)
                        value_s = (
                            f"{value_f:,.2f}" if abs(value_f) >= 1000
                            else f"{value_f:g}"
                        )
                    else:
                        value_s = str(value or "").strip()
                    label = f"{kind}:{subject}"
                    if value_s:
                        lines.append(f"  - {label}: {value_s} -- {raw}")
                    else:
                        lines.append(f"  - {label}: {raw}")
        except Exception:
            pass

        # SO-6: per-invoice breakdown when individual invoice data is available
        if invoice_rows:
            _ccy = (invoice_rows[0].get("currency") or "USD")
            lines.append(f"Per-invoice reconciliation ({_ccy}):")
            for inv in invoice_rows:
                _iid = inv.get("invoice_id") or "(unknown)"
                _iinv = inv.get("invoiced", 0.0)
                _ipaid = inv.get("paid", 0.0)
                _iout = inv.get("outstanding", 0.0)
                lines.append(
                    f"  Invoice {_iid}: invoiced ${_iinv:,.2f}  paid ${_ipaid:,.2f}"
                    f"  outstanding ${_iout:,.2f}"
                )
            # Aggregate totals follow
            _ccy = chain.get("currency", "USD")
            _inv = chain.get("invoiced", 0.0)
            _paid = chain.get("paid", 0.0)
            _disp = chain.get("disputed", 0.0)
            _exp = chain.get("exposure", 0.0)
            lines.append(
                f"  Total: invoiced ${_inv:,.2f}  payment ${_paid:,.2f}"
                f"  exposure ${_exp:,.2f}"
            )
            if _disp:
                lines.append(f"  Disputed (assertion belief_state): ${_disp:,.2f}")
            _spans = chain.get("source_spans") or []
            if _spans:
                lines.append(f"  Grounded in {len(_spans)} source span(s).")
        else:
            # SO-6: aggregate payment reconciliation (invoiced / paid / disputed / exposure)
            _ccy = chain.get("currency", "USD")
            _inv = chain.get("invoiced", 0.0)
            _paid = chain.get("paid", 0.0)
            _disp = chain.get("disputed", 0.0)
            _exp = chain.get("exposure", 0.0)
            if _inv or _paid or _disp:
                lines.append(f"Payment reconciliation ({_ccy}):")
                lines.append(f"  Invoiced:  ${_inv:>14,.2f}")
                lines.append(f"  Paid:      ${_paid:>14,.2f}")
                if _disp:
                    lines.append(f"  Disputed:  ${_disp:>14,.2f}")
                lines.append(f"  Exposure:  ${_exp:>14,.2f}  (invoiced − paid)")
                _spans = chain.get("source_spans") or []
                if _spans:
                    lines.append(f"  Grounded in {len(_spans)} source span(s).")

        # Show all non-invoice/payment categories so claims, damages, fees, etc.
        # are always visible regardless of whether a full chain was detected.
        _by_cat = chain.get("by_category") or {}
        _extra = {
            k: v for k, v in _by_cat.items()
            if k not in ("invoice", "payment", "unknown", None)
        }
        if _extra:
            lines.append("Other monetary amounts by category:")
            for subject, data in sorted(_extra.items(), key=lambda x: x[1]["total"], reverse=True):
                lines.append(f"  • {subject}: ${data['total']:,.2f} ({data['count']} entries)")
        elif not (_inv or _paid or _disp) and _by_cat:
            lines.append("Monetary amounts by category (USD unless noted):")
            for subject, data in sorted(_by_cat.items(), key=lambda x: x[1]["total"], reverse=True):
                lines.append(f"  • {subject}: ${data['total']:,.2f} ({data['count']} entries)")

        if conflicts:
            lines.append("NUMERIC CONFLICTS DETECTED (same category, different amounts):")
            for c in conflicts[:3]:
                subject = c.get("subject_type", "unknown")
                currency = c.get("currency", "")
                values = [f"${v:,.2f}" for v in (c.get("values") or [])[:4]]
                lines.append(f"  ⚠ {subject} ({currency}): {', '.join(values)} — UNRESOLVED DISCREPANCY")

        if date_facts:
            # Title avoids "chronological" since date_value may not be ISO-normalized.
            lines.append("Key dates extracted (by stored date value):")
            for df in date_facts:
                _dv = df.get("date_value") or df.get("raw_text", "")[:60]
                _ctx = df.get("raw_text", "")[:80]
                lines.append(f"  • {_dv} — {_ctx}" if _dv != _ctx else f"  • {_dv}")

        if rate_facts:
            lines.append("Rates and percentages:")
            for rf in rate_facts:
                _rv = rf.get("rate_value")
                _ctx = rf.get("raw_text", "")[:80]
                # Guard against non-numeric rate_value from corrupted/legacy rows.
                try:
                    _rate_str = f"{float(_rv):.4g}%" if _rv is not None else ""
                except (TypeError, ValueError, OverflowError):
                    _rate_str = ""
                lines.append(f"  • {_rate_str} — {_ctx}" if _rate_str else f"  • {_ctx}")

        # SO-6 fix #4: Show individual quant rows so synthesis has exact operands
        # for cross-document calculations (RSU counts, buyout multiples, lease fees).
        try:
            amount_rows = self._matter_model.quant.get_amounts()[:50]
            count_rows = self._matter_model.quant.get_by_kind("count", limit=30)
            if amount_rows:
                lines.append("High-signal extracted numeric facts:")
                for row in amount_rows:
                    subject = row.get("subject_id") or row.get("subject_type") or "amount"
                    raw = row.get("raw_text", "")[:120]
                    val = row.get("amount_value")
                    try:
                        lines.append(f"  - {subject}: ${float(val):,.2f} — {raw}")
                    except (TypeError, ValueError):
                        lines.append(f"  - {subject}: {val} — {raw}")
            if count_rows:
                for row in count_rows:
                    subject = row.get("subject_id") or row.get("subject_type") or "count"
                    raw = row.get("raw_text", "")[:120]
                    val = row.get("amount_value")
                    unit = row.get("unit") or ""
                    try:
                        lines.append(f"  - {subject}: {float(val):g} {unit} — {raw}")
                    except (TypeError, ValueError):
                        lines.append(f"  - {subject}: {val} {unit} — {raw}")
        except Exception:
            pass

        return "\n".join(lines)

    def _get_issue_coverage_map(self) -> "dict[str, tuple[float, bool, int]]":
        """Return {issue_id: (coverage_fraction, has_proof_gap, support_count)} from live DB.

        Called once per investigation iteration to drive dynamic lead reweighting (SO-4).

        get_issue_coverage_report() is canonical for both coverage_fraction and
        the base has_proof_gap flag. ProofStateStore rows only enrich the gap
        flag: contested, insufficient, or advocacy-only issues are elevated to
        has_proof_gap=True so they attract investigation budget. The weighted
        coverage_fraction from the report is preserved; ProofStateStore's raw
        assertion-count sufficiency is never substituted in (SO-4 audit #021).

        Returns empty dict when no matter model is available or the call fails.
        """
        if self._matter_model is None:
            return {}
        try:
            report = self._matter_model.get_issue_coverage_report()
        except Exception:
            return {}

        # Build base map from assertion-count report.
        base: "dict[str, tuple[float, bool, int]]" = {
            item["id"]: (
                float(item.get("coverage_fraction", 0.0)),
                bool(item.get("has_proof_gap", False)),
                int(item.get("supporting_count", 0)),
            )
            for item in report
            if item.get("id")
        }

        # Overlay proof_state metadata (advocacy_only, contested status) to enrich has_proof_gap.
        # We do NOT replace coverage_fraction with proof_state.sufficiency because
        # ProofStateStore uses raw assertion counts (not belief-state weights), which would
        # bypass the weighted coverage_fraction computed above (SO-4 audit #021 finding).
        try:
            ps_rows = self._matter_model.proof_state.get_all()
            for ps in ps_rows:
                iid = ps.get("issue_id")
                if iid not in base:
                    continue
                # Contested, insufficient, and advocacy-only issues all need more evidence.
                proof_status = ps.get("proof_status", "")
                has_gap = (
                    base[iid][1]
                    or proof_status in ("insufficient", "contested")
                    or bool(ps.get("advocacy_only"))
                )
                # Preserve the weighted coverage_fraction (base[iid][0]); only update has_gap.
                base[iid] = (base[iid][0], has_gap, base[iid][2])
        except Exception:
            pass  # fall back to base coverage if proof state unavailable

        return base

    def _build_issue_coverage_summary(self) -> str:
        """Build a per-issue evidence coverage block for the synthesis prompt (SO-4).

        Shows each open issue with its supporting assertion count, a coverage
        fraction, and whether a proof gap is present — so the LLM can surface
        which claims are well-evidenced vs. proof-gap-exposed rather than treating
        all claims uniformly.
        """
        if self._matter_model is None:
            return "No issue model available."
        try:
            report = self._matter_model.get_issue_coverage_report()
        except Exception:
            return "Issue coverage data unavailable."
        if not report:
            return "No open issues in matter model."

        # Index proof states by issue_id for richer annotations.
        proof_index: dict = {}
        try:
            if self._matter_model is not None:
                ps_rows = self._matter_model.proof_state.get_all()
                proof_index = {ps["issue_id"]: ps for ps in ps_rows}
        except Exception:
            pass

        lines = [f"{len(report)} open issue(s):"]
        for item in report:
            title = (item.get("title") or "Untitled")[:60]
            cnt = item.get("supporting_count", 0)
            issue_id = item.get("id", "")

            ps = proof_index.get(issue_id)
            if ps:
                sufficiency = float(ps.get("sufficiency", 0.0))
                proof_status = ps.get("proof_status", "insufficient")
                atk = int(ps.get("attacking_count", 0))
                pct = int(sufficiency * 100)
                gap_flag = ""
                if proof_status == "contested":
                    gap_flag = f" ⚠ CONTESTED ({atk} attacking)"
                elif proof_status == "insufficient" or item.get("has_proof_gap"):
                    gap_flag = " ⚠ PROOF GAP"
                if ps.get("advocacy_only"):
                    gap_flag += " ⚠ ADVOCACY-ONLY (no operative/authoritative support)"
                strength = (
                    "STRONG" if sufficiency >= 0.75
                    else "PARTIAL" if sufficiency >= 0.25
                    else "WEAK"
                )
                lines.append(
                    f"  [{strength}/{proof_status.upper()}] {title}: "
                    f"{cnt} supporting ({pct}% sufficiency){gap_flag}"
                )
            else:
                frac = item.get("coverage_fraction", 0.0)
                pct = int(frac * 100)
                gap_flag = " ⚠ PROOF GAP" if item.get("has_proof_gap") else ""
                strength = (
                    "STRONG" if frac >= 0.6
                    else "PARTIAL" if frac >= 0.3
                    else "WEAK"
                )
                lines.append(
                    f"  [{strength}] {title}: {cnt} supporting ({pct}%){gap_flag}"
                )
        return "\n".join(lines)

    def _build_gap_summary(self) -> str:
        """Build a structured gap block for the synthesis prompt (SO-7).

        Pulls open gaps from the matter model so the LLM is explicitly aware
        of what is missing and can surface them in the Gaps & Limitations section
        rather than silently skipping absent evidence.
        """
        if self._matter_model is None:
            return "No gap data available."
        try:
            all_gaps = self._matter_model.gaps.open_gaps(min_materiality=0.0)
            gaps = [g for g in all_gaps if g.get("materiality_score", 0.0) >= 0.3]
        except Exception:
            return "Gap data unavailable."
        total = len(all_gaps)
        if total == 0:
            return "No gaps identified."
        shown = min(len(gaps), 8)
        header = f"{total} open gap(s) total"
        if total > shown:
            header += f"; showing top {shown} by materiality (≥0.3) — {total - shown} lower-priority gap(s) omitted"
        lines = [header + ":"]
        for gap in gaps[:8]:  # cap to prevent prompt bloat
            gap_type = gap.get("gap_type", "unknown").replace("_", " ")
            description = gap.get("description", "")
            materiality = gap.get("materiality_score", 0.0)
            label = "HIGH" if materiality >= 0.7 else "MED" if materiality >= 0.4 else "LOW"
            lines.append(f"  [{label}] {gap_type}: {description}")
            # Surface dependency links so LLM knows what conclusions depend on this gap (SO-7)
            deps = gap.get("dependencies", [])[:3]  # cap at 3 to avoid prompt bloat
            if deps:
                dep_strs = [f"{d['affected_type']}:{d['affected_id'][:8]}" for d in deps]
                lines.append(f"         Affects: {', '.join(dep_strs)}")
        return "\n".join(lines)

    def _build_decision_context_block(self) -> str:
        """Build a decision-context framing block for the synthesis prompt.

        If a decision context has been set on the matter model, this block
        tells the LLM who the decision-maker is and what objective the analysis
        should serve.  This influences recommendation framing, prioritization,
        and output emphasis WITHOUT rewriting the record model or assertions.

        Returns empty string when no context is set (no-op for legacy runs).
        """
        if self._matter_model is None:
            return ""
        try:
            ctx = self._matter_model.decision_context.get()
        except Exception:
            return ""
        if ctx is None:
            return ""

        lines = ["DECISION CONTEXT (influences framing and prioritization — does not change the record):"]
        if ctx.get("decision_maker_type") and ctx["decision_maker_type"] != "unknown":
            line = f"  Decision-maker type: {ctx['decision_maker_type']}"
            if ctx.get("decision_maker_name"):
                line += f" ({ctx['decision_maker_name']})"
            lines.append(line)
        if ctx.get("objective") and ctx["objective"] != "unknown":
            lines.append(f"  Analysis objective: {ctx['objective'].replace('_', ' ')}")
        if ctx.get("scope_narrow"):
            lines.append("  Scope: NARROW — user requested focused, not exhaustive, output")
        if ctx.get("strategic_notes"):
            # Truncate to prevent prompt bloat
            notes = ctx["strategic_notes"][:400]
            lines.append(f"  Strategic context: {notes}")
        lines.append(
            "  Framing instruction: weight your analysis and recommendations toward the above "
            "objective and decision-maker. Do not alter factual findings or assertion grounding."
        )
        return "\n".join(lines)

    async def _retry_spo_extraction(self, fact_texts: list[str]) -> dict[int, dict]:
        """Best-effort SPO extraction retry (SO-2).

        Called when primary extraction yielded zero SPO triples. Makes one
        lightweight FLASH call with just the fact texts to extract structured
        subject/predicate/object triples without re-reading the source document.

        Returns a mapping of fact_index → spo_dict (only for facts where SPO
        was successfully extracted). Empty dict if retry fails or produces nothing.
        Non-fatal: caller continues with null-SPO behavior on any exception.
        """
        if not fact_texts:
            return {}
        lines = "\n".join(f"{i}: {t[:100]}" for i, t in enumerate(fact_texts))
        prompt = SPO_RETRY_PROMPT.format(facts=lines)
        try:
            response = await self.client.complete(
                prompt,
                tier=ModelTier.LITE,
                json_mode=True,
                usage_label="spo_retry",
            )
            # Direct parse — response must be a JSON array; _parse_json_safe is dict-only
            text = (response or "").strip()
            if "```" in text:
                start = text.find("```") + 3
                if text[start : start + 4] == "json":
                    start += 4
                end = text.find("```", start)
                text = text[start:end].strip() if end > start else text[start:].strip()
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return {}
            if not isinstance(parsed, list):
                return {}
            result: dict[int, dict] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= len(fact_texts):
                    continue
                subj = str(item.get("subject") or "").strip()
                pred = str(item.get("predicate") or "").strip()
                obj = str(item.get("object") or "").strip()
                if subj or pred or obj:
                    result[idx] = {
                        "subject_ref_type": "free_text" if subj else None,
                        "subject_ref_id": subj if subj else None,
                        "predicate_key": pred.lower().replace(" ", "_") if pred else None,
                        "object_json": json.dumps(obj) if obj else None,
                    }
            return result
        except Exception:
            return {}

    def _build_structured_relationships(self) -> str:
        """Build a structured assertion block for the synthesis prompt (SO-2).

        Pulls typed assertions that have populated subject_ref_id, predicate_key,
        and/or object_json from the matter model so the LLM can reason about
        explicit structured relationships rather than only prose text.

        Returns empty string if no typed assertions exist (no prompt bloat when
        SPO extraction was unavailable).
        """
        if self._matter_model is None:
            return ""
        try:
            # Fetch 60 most-recent typed assertions using the existing
            # ix_assertion_matter_created index (no full-scan), then sort
            # by SPO completeness in Python to avoid an unindexable ORDER BY
            # CASE expression that would force a full matter-scan.
            _candidates = self._matter_model.assertions.db.execute(
                """SELECT a.proposition_text, a.predicate_key, a.subject_ref_id,
                          a.object_json, a.belief_state,
                          (SELECT ao.source_role FROM assertion_occurrence ao
                           WHERE ao.assertion_id = a.id
                           ORDER BY CASE ao.source_role
                             WHEN 'authoritative' THEN 6 WHEN 'operative' THEN 5
                             WHEN 'procedural' THEN 4 WHEN 'post_hoc' THEN 3
                             WHEN 'informal' THEN 2 ELSE 1 END DESC
                           LIMIT 1) AS source_role
                   FROM assertion a
                   WHERE a.matter_id=?
                     AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                     AND (a.predicate_key IS NOT NULL OR a.subject_ref_id IS NOT NULL
                          OR a.object_json IS NOT NULL)
                   ORDER BY a.created_at DESC
                   LIMIT 60""",
                (self._matter_model.matter_id,),
            ).fetchall()
            # Sort by completeness score (count of non-null SPO fields) then keep top 30
            rows = sorted(
                _candidates,
                key=lambda r: (
                    (1 if r["predicate_key"] else 0) +
                    (1 if r["subject_ref_id"] else 0) +
                    (1 if r["object_json"] else 0)
                ),
                reverse=True,
            )[:30]
        except Exception:
            return ""

        if not rows:
            return ""

        lines = [f"Typed assertion graph ({len(rows)} structured relationships identified):"]
        for row in rows:
            pred = row["predicate_key"] or "?"
            subj = row["subject_ref_id"] or "?"
            try:
                obj = json.loads(row["object_json"]) if row["object_json"] else "?"
            except Exception:
                obj = row["object_json"] or "?"
            role = (row["source_role"] or "unknown").upper()
            belief = row["belief_state"] or "active"
            lines.append(
                f"  [{role}/{belief}] {subj} —[{pred}]→ {str(obj)[:80]}"
            )
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

        response = await self.client.complete(
            prompt,
            tier=ModelTier.LITE,
            json_mode=True,
            usage_label="entity_extraction",
        )

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

        response = await self.client.complete(
            prompt,
            tier=ModelTier.LITE,
            usage_label="contradiction_analysis",
        )

        defaults = {
            "is_contradiction": False,
            "contradiction_type": "none",
            "severity": "none",
            "explanation": "Unable to analyze",
            "reconciliation_possible": True,
            "reconciliation_theory": None,
            "significance": "Unknown",
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

        response = await self.client.complete(
            prompt,
            tier=ModelTier.LITE,
            usage_label="timeline_analysis",
        )

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

        response = await self.client.complete(
            prompt,
            tier=ModelTier.LITE,
            usage_label="evidence_classification",
        )

        defaults = {
            "claim": claim,
            "evidence_classification": {
                "direct": [],
                "circumstantial": [],
                "primary_sources": [],
                "secondary_sources": [],
                "reliability_concerns": [],
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
        budget = self._get_research_profile(state)

        # Leads completion
        total_leads = len(state.leads)
        investigated_leads = len([l for l in state.leads if l.status == "investigated"])
        factors["leads_complete"] = (investigated_leads / max(total_leads, 1)) * 100

        # Depth progress
        factors["depth_progress"] = (
            state.max_depth_reached / max(budget.max_depth, 1)
        ) * 100

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
            "research_mode": state.research_mode,
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
        """Emit a thinking step, call callback, and write to durable DB ledger (SO-3)."""
        step = state.add_step(step_type, content, details)
        if self.on_step:
            self.on_step(step)
        # Persist signal-bearing steps to durable reasoning ledger (SO-3).
        # THINKING steps are high-frequency and low-signal — kept in-memory only.
        # All other step types (SEARCH, READING, FINDING, REPLAN, VERIFY, SYNTHESIS, ERROR)
        # are written to the DB so the reasoning trail survives process restart.
        adapter = getattr(state, "_matter_adapter", None)
        if adapter is not None and step_type != StepType.THINKING:
            _summary = f"[{step_type.value.upper()}] {content}"
            try:
                if step_type == StepType.ERROR:
                    adapter.log_warning(_summary)
                else:
                    adapter.log_step(_summary)
            except Exception:
                pass  # Ledger write failure must not abort investigation
        # Also emit progress update
        self._emit_progress(state)

    def _emit_progress(self, state: InvestigationState):
        """Emit progress update."""
        if self.on_progress:
            self.on_progress(state.get_progress())

    def _calculate_effective_depth(self, state: InvestigationState) -> int:
        """Calculate effective max depth based on investigation progress."""
        budget = self._get_research_profile(state)
        if not self.config.adaptive_depth:
            return budget.max_depth

        base_depth = budget.max_depth

        # Reduce depth if we have many citations already
        if len(state.citations) >= budget.depth_citation_threshold:
            return max(budget.min_depth, base_depth - 2)

        # Reduce depth if confidence is high
        confidence = state.get_confidence_score()
        if confidence["score"] >= 70:
            return max(budget.min_depth, base_depth - 1)

        return base_depth

    # MVI-3: governed-progress epsilon — an iteration counts as
    # "material answerability delta" when the matter-wide sum of
    # issue coverage_fraction advances by at least this much OR any
    # open proof gap closes.
    _COVERAGE_DELTA_EPSILON = 0.05

    def _should_continue_investigation(self, state: InvestigationState) -> tuple[bool, str]:
        """MVI-3 cascade termination controller.

        Replaces the four legacy checks (confidence / all-docs /
        count-based-diminishing / count-based-productivity) with:
          1. target sufficiency  — every open high-materiality issue
             has coverage above its floor AND no blocking proof gap
          2. relevant scope exhausted — all docs linked to open issues
             have been read and no new pending leads reference
             unread docs
          3. no material answerability delta — two consecutive
             iterations produced no coverage advance and no gap close
          4. dead-loop fuse — 3 iterations of zero governed progress
             AND no viable leads (never fact-count based)
          5. viable-lead exhaustion — no pending lead above the EV
             floor (MVI-3 still uses the existing priority threshold;
             MVI-5 upgrades to a real EV signal)

        min_depth is now family-scoped via the ExecutionContract
        carried on state.execution_contract. Contracts from the
        cascade governor specify min_iter per family (0 for
        probe/read/query/trace; floor only for investigate).

        Returns (should_continue, reason).
        """
        budget = self._get_research_profile(state)
        contract = getattr(state, "execution_contract", None)

        # Plan A: sufficiency probe has priority over min_iter.
        # When the interim probe decides we can answer now, that
        # signal overrides the contract's minimum-iteration floor —
        # research modes otherwise mandate more iters than needed,
        # wasting tokens on matters where we already have the
        # answer. The loop itself sets state.findings['final_output']
        # to the probe answer before flipping this flag, so callers
        # downstream of the loop still see the synthesis output they
        # expect.
        _etr = getattr(state, "early_terminate_reason", None)
        if _etr:
            return False, f"Sufficiency probe: {_etr}"

        # Family-scoped min_depth gate. Contract wins if present.
        min_depth = budget.min_depth
        if contract is not None:
            # ExecutionContract carries min_iter semantics; translate
            # directly. Non-investigate contracts set min_iter=0 which
            # opens the door for early termination the moment the
            # target is answerable.
            min_depth = max(0, int(getattr(contract, "min_iter", min_depth)))
        simple_lookup_ok, simple_lookup_detail = self._simple_lookup_answer_satisfied(state)
        if simple_lookup_ok:
            state.findings["simple_lookup_satisfied"] = {
                "detail": simple_lookup_detail,
                "anchors": sorted(self._simple_lookup_anchor_terms(state.query)),
                "citations": len(state.citations),
                "facts": len(state.findings.get("accumulated_facts") or []),
            }
            return False, simple_lookup_detail
        if state.max_depth_reached < min_depth:
            return True, "Building minimum evidence base"

        mode_label = self._research_mode_label(budget.mode)

        # Check 1: target sufficiency (replaces the confidence/
        # citation threshold). "Every open high-materiality issue has
        # coverage at or above the contract's floor AND no blocking
        # proof gap." If no open issues exist, fall back to citation
        # floor so we don't spin forever on matters without an issue
        # tree yet.
        sufficiency_ok, sufficiency_detail = self._target_is_sufficient(
            state, contract,
        )
        if sufficiency_ok:
            return False, f"Target sufficient: {sufficiency_detail}"

        # Check 2: relevant scope exhausted (replaces "all docs
        # processed"). Only fires when we have issues + at least one
        # citation — otherwise we're still bootstrapping.
        if len(state.citations) >= 1 and self._relevant_scope_exhausted(state):
            return False, "Relevant document scope exhausted"

        # Check 3: no material answerability delta over last 2
        # iterations (replaces count-based diminishing returns).
        if self._no_material_answerability_delta(state):
            return False, (
                f"No coverage or proof-gap progress in last 2 iterations "
                f"(coverage_sum: {state.coverage_sum_per_iteration[-2:]})"
            )

        # Check 4: dead-loop fuse. 3 iterations of zero governed
        # progress AND no viable leads. Never fact-count based.
        pending = state.get_pending_leads()
        viable = self._viable_leads(pending, contract)
        if self._dead_loop_detected(state) and not viable:
            return False, (
                "Dead loop: 3 iterations without governed progress "
                "and no viable leads remaining"
            )

        # Check 5: viable-lead exhaustion (replaces no-high-priority).
        if not viable:
            return False, "No viable leads above EV floor"

        return True, (
            f"Continuing {mode_label} investigation "
            f"({len(viable)} viable leads)"
        )

    # ---- MVI-3 helpers ---------------------------------------------------

    def _target_is_sufficient(
        self, state: InvestigationState, contract: Any,
    ) -> tuple[bool, str]:
        """Codex master plan replacement for the confidence stop.
        Coverage over the target set AND no blocking proof gap."""
        if self._matter_model is None:
            # Fall back to the old citation floor when matter model
            # isn't wired (test harness / legacy callers).
            floor = 1
            if contract is not None:
                floor = max(floor, int(getattr(contract, "citation_floor", 1)))
            if len(state.citations) >= floor:
                return True, f"{len(state.citations)} citations >= floor {floor}"
            return False, ""
        try:
            coverage_rows = self._matter_model.get_issue_coverage_report(
                policy_audience="internal",
            )
        except Exception:
            return False, ""
        if not coverage_rows:
            # No open issues yet — fall back to a citation floor so we
            # don't spin forever on a matter whose issue tree is still
            # being seeded by orient.
            floor = 1
            if contract is not None:
                floor = max(floor, int(getattr(contract, "citation_floor", 1)))
            if len(state.citations) >= floor:
                return True, f"{len(state.citations)} citations, no open issues"
            return False, ""
        # Floor: high-materiality issues (>=0.5) must each have
        # coverage >= 0.85 AND no open proof gap to call the target
        # "sufficient". Non-material issues don't block.
        blocking = []
        for row in coverage_rows:
            if (row.get("materiality") or 0) < 0.5:
                continue
            frac = float(row.get("coverage_fraction") or 0.0)
            if row.get("has_proof_gap") or frac < 0.85:
                blocking.append(row.get("title", "?"))
        if not blocking:
            total_issues = sum(
                1 for r in coverage_rows
                if (r.get("materiality") or 0) >= 0.5
            )
            return True, f"{total_issues} high-materiality issues covered"
        return False, ""

    def _relevant_scope_exhausted(self, state: InvestigationState) -> bool:
        """All documents materially relevant to the investigation have
        been read and no pending lead points at unread material.
        Replaces the repo-global 'all docs processed' check."""
        pending = state.get_pending_leads()
        if pending:
            # If any pending lead still references unread material,
            # scope isn't exhausted.
            return False
        # Small-repo fallback: if we read everything in the repo and
        # have at least one citation, scope is by definition exhausted.
        if self._doc_count > 0 and state.documents_read >= self._doc_count:
            return True
        return False

    def _no_material_answerability_delta(
        self, state: InvestigationState,
    ) -> bool:
        """True when the last 2 iterations produced no coverage
        advance and no open-gap close. Replaces count-based
        diminishing returns."""
        cov = state.coverage_sum_per_iteration
        gaps = state.open_gap_count_per_iteration
        if len(cov) < 3 or len(gaps) < 3:
            return False
        # Look at the last two completed iterations: was there ANY
        # delta? cov[-1] is post-iter-N, cov[-2] is post-iter-N-1,
        # cov[-3] is post-iter-N-2.
        recent_cov = cov[-3:]
        recent_gaps = gaps[-3:]
        cov_delta_a = recent_cov[-1] - recent_cov[-2]
        cov_delta_b = recent_cov[-2] - recent_cov[-3]
        gap_delta_a = recent_gaps[-2] - recent_gaps[-1]  # gap CLOSED if positive
        gap_delta_b = recent_gaps[-3] - recent_gaps[-2]
        material_a = (
            cov_delta_a >= self._COVERAGE_DELTA_EPSILON or gap_delta_a > 0
        )
        material_b = (
            cov_delta_b >= self._COVERAGE_DELTA_EPSILON or gap_delta_b > 0
        )
        return not (material_a or material_b)

    def _dead_loop_detected(self, state: InvestigationState) -> bool:
        """3 iterations with zero governed progress. Fact counts are
        never consulted."""
        cov = state.coverage_sum_per_iteration
        gaps = state.open_gap_count_per_iteration
        if len(cov) < 4 or len(gaps) < 4:
            return False
        recent_cov = cov[-4:]
        recent_gaps = gaps[-4:]
        for i in range(1, 4):
            cov_delta = recent_cov[-i] - recent_cov[-i - 1]
            gap_delta = recent_gaps[-i - 1] - recent_gaps[-i]
            if cov_delta >= self._COVERAGE_DELTA_EPSILON or gap_delta > 0:
                return False
        return True

    # MVI-5 cost classes. Coarse — coverage-per-dollar only needs to
    # be directionally correct for the floor gate. Tune these after
    # the cost-visibility panel accumulates real per-call deltas.
    _LEAD_COST_SEARCH = 0.0015   # LITE extract + 35%-short-circuited FLASH reason
    _LEAD_COST_DEEP_READ = 0.002  # LITE deep_read
    # Coverage-gain coefficients — how much answerability advance a
    # lead is expected to produce. weakness * coefficient.
    _EV_ISSUE_GAIN_COEF = 0.15   # issue-targeted leads get weakness * 0.15
    _EV_NEUTRAL_GAIN = 0.03      # small default for leads with no focus_issue

    def _enrich_lead_ev(
        self,
        pending_leads: list,
        coverage_map: "dict[str, tuple[float, bool, int]]",
    ) -> None:
        """Stamp expected_cost_usd + expected_coverage_gain on every
        lead that doesn't have them yet. MVI-5 — feeds _viable_leads'
        coverage-per-dollar floor check."""
        for _lead in pending_leads:
            if _lead.expected_cost_usd > 0 and _lead.expected_coverage_gain > 0:
                continue  # already enriched (e.g. upstream lead planner)
            # Cost class heuristic — for MVI-5 every lead is a search
            # lead; deep_read leads carry a different search_term
            # shape that the engine handles separately.
            _lead.expected_cost_usd = self._LEAD_COST_SEARCH
            if (
                _lead.focus_issue_id
                and _lead.focus_issue_id in coverage_map
            ):
                frac, _gap, _ = coverage_map[_lead.focus_issue_id]
                weakness = max(0.0, 1.0 - float(frac))
                _lead.expected_coverage_gain = max(
                    0.01, weakness * self._EV_ISSUE_GAIN_COEF,
                )
            else:
                _lead.expected_coverage_gain = self._EV_NEUTRAL_GAIN

    # P0.7 coverage-driven lead planner — per-iteration and per-run
    # caps. Planner output never exceeds these; reactive leads
    # (follow-ons, user, clarification answers) have their own budget.
    _PLANNER_LEADS_PER_ITER = 4
    _PLANNER_LEADS_PER_RUN = 20

    _DISCOVERY_INTERVAL = 3
    _DISCOVERY_MAX_NEW_ISSUES = 5

    def _discover_unmodeled_issues(
        self,
        state: "InvestigationState",
        iteration: int,
    ) -> int:
        """Scan accumulated facts for issues that orientation missed (Codex #5).

        Runs every _DISCOVERY_INTERVAL iterations. Looks for:
        - [PROVISION] facts referencing provisions not covered by any issue
        - [REGULATORY:*] facts about topics without corresponding issues
        - Contract terms and numeric thresholds mentioned in facts

        Creates new issues so the coverage planner can generate leads for them.
        Returns the number of new issues created.
        """
        if self._matter_model is None:
            return 0
        if iteration < self._DISCOVERY_INTERVAL:
            return 0
        if iteration % self._DISCOVERY_INTERVAL != 0:
            return 0

        facts = state.findings.get("accumulated_facts", [])
        if len(facts) < 10:
            return 0

        try:
            existing_issues = self._matter_model.issues.list_issues(limit=50)
        except Exception:
            return 0
        existing_titles: set[str] = set()
        for iss in existing_issues:
            t = (iss.get("title") or "").lower().strip()
            if t:
                existing_titles.add(t)

        def _title_covered(candidate: str) -> bool:
            cl = candidate.lower().strip()
            for et in existing_titles:
                if cl in et or et in cl:
                    return True
                words_c = set(cl.split())
                words_e = set(et.split())
                if len(words_c & words_e) >= min(3, len(words_c)):
                    return True
            return False

        import re as _re_disc
        new_issues: list[tuple[str, list[str]]] = []

        _prov_pattern = _re_disc.compile(r'\[PROVISION\]\s*(.+?):\s*(.+)')
        _reg_pattern = _re_disc.compile(r'\[REGULATORY:(\w+)\]\s*(.+?):\s*(.+)')
        _calc_pattern = _re_disc.compile(r'\[CALCULATED\]\s*(.+?):\s*(.+)')

        discovered_provisions: dict[str, list[str]] = {}
        discovered_reg_topics: dict[str, list[str]] = {}

        for fact in facts:
            if not isinstance(fact, str):
                continue

            m = _prov_pattern.match(fact)
            if m:
                prov_name = m.group(1).strip()
                if not _title_covered(prov_name):
                    discovered_provisions.setdefault(prov_name, []).append(
                        m.group(2).strip()[:100]
                    )

            m = _reg_pattern.match(fact)
            if m:
                cat = m.group(1).strip()
                entity = m.group(2).strip()
                topic = f"{cat}: {entity}"
                if not _title_covered(topic) and not _title_covered(entity):
                    discovered_reg_topics.setdefault(topic, []).append(
                        m.group(3).strip()[:100]
                    )

        for prov_name, details in sorted(
            discovered_provisions.items(),
            key=lambda x: -len(x[1]),
        ):
            if len(new_issues) >= self._DISCOVERY_MAX_NEW_ISSUES:
                break
            if len(details) < 2:
                continue
            title = f"{prov_name} analysis"
            preds = [
                f"Extract exact {prov_name} terms from original document",
                f"Extract exact {prov_name} terms from markup/counterparty document",
                f"Compare before and after values for {prov_name}",
            ]
            new_issues.append((title, preds))

        for topic, details in sorted(
            discovered_reg_topics.items(),
            key=lambda x: -len(x[1]),
        ):
            if len(new_issues) >= self._DISCOVERY_MAX_NEW_ISSUES:
                break
            if len(details) < 2:
                continue
            title = f"Regulatory: {topic}"
            preds = [
                f"Gather all evidence related to {topic}",
                f"Assess regulatory risk implications of {topic}",
            ]
            new_issues.append((title, preds))

        created = 0
        for title, preds in new_issues:
            if _title_covered(title):
                continue
            try:
                iid, _ = self._matter_model.issues.upsert_issue(
                    title=title,
                    issue_type=IssueType.DILIGENCE_RED_FLAG,
                    salience=0.5,
                )
                if preds:
                    self._matter_model.issues.add_predicates_batch(
                        issue_id=iid, descriptions=preds[:3],
                    )
                existing_titles.add(title.lower().strip())
                created += 1
            except Exception as exc:
                logger.warning("discover_unmodeled_issues: upsert failed: %s", exc)
        return created

    def _coverage_planner(
        self,
        state: InvestigationState,
        coverage_map: "dict[str, tuple[float, bool, int]]",
    ) -> int:
        """P0.7.1 — proactively inject issue-targeted leads based on the
        matter model's coverage state, not the user's utterance.

        Contract (per Codex design gate):
         - Runs at the top of each _investigate_loop iteration, only
           for family='investigate' (or legacy contract=None).
         - Caps: 2 per iteration, 6 per run; never outranks reactive
           leads; fills the issue-targeted quota deficit only.
         - Lead shape: existing Lead with source='coverage_planner',
           focus_issue_id set, a literal search_term from the issue's
           first open predicate.
         - EV enrichment via the existing _enrich_lead_ev path.
         - Dedup: state.add_lead's string-similarity check already
           suppresses identical/near-identical descriptions.
         - NOT in P0.7.1: mid-loop clarification actions, quant-gap
           routing, authority retrieval, proof-lane predicate scoring.

        Returns the number of leads added this call.
        """
        if self._matter_model is None:
            return 0
        contract = getattr(state, "execution_contract", None)
        family = getattr(contract, "family", None) if contract is not None else None
        if family is not None and family != "investigate":
            return 0
        # Per-run cap.
        run_remaining = self._PLANNER_LEADS_PER_RUN - state.planner_leads_added
        if run_remaining <= 0:
            return 0
        # Per-iteration cap.
        iter_cap = min(self._PLANNER_LEADS_PER_ITER, run_remaining)
        # Issue-targeted lane deficit — planner fills when the issue
        # quota has capacity OR when uncovered issues exist. Always
        # allow at least 1 planner lead per iteration to ensure
        # exhaustive coverage across all orientation issues.
        pending = [l for l in state.get_pending_leads()]
        issue_pending = [l for l in pending if l.focus_issue_id]
        issue_quota = max(1, (self.config.max_leads_per_level + 1) // 2)
        deficit = issue_quota - len(issue_pending)
        capacity = min(iter_cap, max(1, deficit))
        if capacity <= 0:
            return 0
        # Signal pass: canonical coverage report + material open gaps.
        try:
            rows = self._matter_model.get_issue_coverage_report(
                policy_audience="internal",
            )
        except Exception as exc:
            logger.warning("coverage_planner: coverage report failed: %s", exc)
            return 0
        if not rows:
            return 0
        # Index gapped-issue links for the has_gap boost. Only material
        # gaps (>= 0.4) feed the planner; low-materiality noise stays
        # in the clarification end-of-run pass. adv#11 review fix #2:
        # filter to search-resolvable gap types — missing_user_context,
        # missing_authority, and missing_quantitative_input are NOT
        # resolvable by document search and shouldn't boost search
        # pressure. That's what P0.7.2's clarification-action path and
        # quant-specialization will handle.
        _SEARCHABLE_GAP_TYPES = {"missing_issue_predicate", "missing_document"}
        gapped_issue_ids: set[str] = set()
        try:
            for g in self._matter_model.gaps.open_gaps(min_materiality=0.4, limit=20):
                if g.get("gap_type") not in _SEARCHABLE_GAP_TYPES:
                    continue
                for dep in (g.get("dependencies") or []):
                    if dep.get("affected_type") == "issue" and dep.get("affected_id"):
                        gapped_issue_ids.add(dep["affected_id"])
        except (sqlite3.Error, ValueError, RuntimeError) as _exc:
            # adv#12 Finding #4: narrow the swallow and log. The base
            # coverage report still drives candidates, so losing the
            # gap boost degrades planner ranking but does not break
            # correctness.
            logger.warning("coverage_planner: open_gaps lookup failed: %s", _exc)
        # adv#11 review fix #1: dedup scope.
        #  - PENDING issue-focused leads (any source) block planner for
        #    that issue this iteration — the queue already has it.
        #  - Historical coverage_planner leads block same-issue re-
        #    planning only when the normalized search term matches.
        #    Prevents duplicate planner work while allowing a new angle.
        #  - Historical reactive/user leads do NOT permanently block
        #    planner — if a prior attempt didn't close the gap,
        #    another lead (with a different predicate) is worth
        #    trying.
        pending_issue_focus: set[str] = {
            l.focus_issue_id for l in state.get_pending_leads() if l.focus_issue_id
        }
        planner_issue_terms: set[tuple[str, str]] = {
            (l.focus_issue_id, " ".join((l.search_term or "").lower().split()))
            for l in state.leads
            if l.source == "coverage_planner" and l.focus_issue_id
        }
        planner_issue_ids: set[str] = {t[0] for t in planner_issue_terms}
        candidates: list[tuple[float, str, str, str]] = []
        for row in rows:
            iid = row.get("id")
            if not iid:
                continue
            # Pending issue-focused lead (any source) — planner waits.
            if iid in pending_issue_focus:
                continue
            materiality = float(row.get("materiality") or 0.0)
            if materiality < 0.4:
                continue
            frac, has_gap, _supp = coverage_map.get(iid, (1.0, False, 0))
            has_any_gap = bool(has_gap) or iid in gapped_issue_ids
            # Already-strong, no-gap issues — skip.
            if frac >= 0.85 and not has_any_gap:
                continue
            try:
                preds = self._matter_model.issues.get_predicates(iid, limit=8)
            except Exception:
                preds = []
            weakness = max(0.0, 1.0 - float(frac))
            base_score = weakness * 0.55 + (0.25 if has_any_gap else 0.0) + materiality * 0.20
            _added_any = False
            _had_preds = False
            for _pi, _pred in enumerate(preds):
                pred_text = (_pred.get("description") or "").strip()
                if not pred_text:
                    continue
                _had_preds = True
                norm_term = " ".join(pred_text.lower().split())
                if (iid, norm_term) in planner_issue_terms:
                    continue
                _pred_score = base_score - (_pi * 0.02)
                candidates.append((_pred_score, iid, row.get("title") or "", pred_text))
                _added_any = True
            if not _added_any:
                if _had_preds and iid in planner_issue_ids:
                    continue
                term = (row.get("title") or "").strip()
                if not term:
                    continue
                norm_term = " ".join(term.lower().split())
                if (iid, norm_term) in planner_issue_terms:
                    continue
                candidates.append((base_score, iid, row.get("title") or "", term))
        if not candidates:
            return 0
        # Round-robin by issue: take best predicate from each issue first,
        # then second-best from each, etc. Prevents one weak issue from
        # consuming all planner slots.
        from collections import defaultdict
        _by_issue: dict[str, list[tuple[float, str, str, str]]] = defaultdict(list)
        for cand in candidates:
            _by_issue[cand[1]].append(cand)
        for _iid_cands in _by_issue.values():
            _iid_cands.sort(reverse=True)
        _issue_order = sorted(_by_issue.keys(), key=lambda k: -_by_issue[k][0][0])
        _selected: list[tuple[float, str, str, str]] = []
        _round = 0
        while len(_selected) < capacity:
            _added_this_round = False
            for _iid_key in _issue_order:
                if len(_selected) >= capacity:
                    break
                _iid_cands = _by_issue[_iid_key]
                if _round < len(_iid_cands):
                    _selected.append(_iid_cands[_round])
                    _added_this_round = True
            if not _added_this_round:
                break
            _round += 1
        added = 0
        for score, iid, title, term in _selected:
            # Description includes the issue short-hash so
            # state.add_lead's string-similarity dedup (>0.8 word
            # overlap) can't collapse two genuinely-different planner
            # leads into one — titles and predicates may share the
            # same template boilerplate ("Plaintiff must show...").
            lead = state.add_lead(
                description=(
                    f"[plan:{iid[:8]}] {term[:80]} "
                    f"(issue: {title[:60]})"
                ),
                source="coverage_planner",
                priority=min(0.86, 0.55 + score),
                search_term=term,
                focus_issue_id=iid,
            )
            if lead is None:
                continue  # state.add_lead collapsed into an existing near-dupe
            # Same EV enrichment path as any other lead.
            self._enrich_lead_ev([lead], coverage_map)
            state.planner_leads_added += 1
            added += 1
        return added

    # Plan A: sufficiency probe. Cheap LITE-tier mid-loop check that
    # asks "can we answer the user's query right now with what we
    # have?". Runs every 2 iterations to avoid false positives from a
    # single noisy iter and to cap cost. When the probe says YES, the
    # engine stamps state.early_terminate_reason, writes the probe's
    # answer to state.findings['final_output'], and the termination
    # controller exits the loop — overriding contract.min_iter.
    _SUFFICIENCY_PROBE_PROMPT = """You are a cost-governance probe inside an investigation loop for a matter intelligence system. Your job is to decide whether the matter state ALREADY contains enough to answer the user's query, so the loop can stop early instead of running more expensive iterations.

User's query: {query}

Accumulated findings so far (may be partial):
{findings_summary}

Per-issue coverage snapshot:
{coverage_summary}

Decide:
- `can_answer`: true ONLY if you can produce a medium-or-higher-confidence answer that's grounded in the findings above, with at least one specific document citation.
- `answer`: your actual answer, 2-4 sentences, with the citation inline (e.g. "per contracts/msa.pdf, ...").
- `confidence`: "low" | "medium" | "high"
- `citations`: list of document names you used.
- `reason_not_yet`: empty string if can_answer=true; otherwise a short phrase naming what's still missing.

BIAS TOWARD finishing when a grounded answer exists — the next iteration costs real money. Say `can_answer: false` only when the findings genuinely can't support an answer at medium confidence.

Respond as JSON only:
{{
  "can_answer": true | false,
  "answer": "...",
  "confidence": "low" | "medium" | "high",
  "citations": ["doc.pdf"],
  "reason_not_yet": "..."
}}
"""

    _SUFFICIENCY_PROBE_CADENCE = 2  # run after every 2nd iteration
    _SUFFICIENCY_PROBE_MAX_PER_RUN = 3  # upper bound on probe cost

    async def _run_sufficiency_probe(
        self,
        state: InvestigationState,
    ) -> bool:
        """Run one interim sufficiency probe. Returns True when the
        probe decided we can answer now AND updated state accordingly
        (final_output set, early_terminate_reason stamped). Returns
        False to keep the loop running.

        Cheap — LITE tier, ~1k-token prompt. Budget bounded by
        _SUFFICIENCY_PROBE_MAX_PER_RUN.
        """
        _probes_used = int(state.findings.get("_probes_used", 0) or 0)
        if _probes_used >= self._SUFFICIENCY_PROBE_MAX_PER_RUN:
            return False
        import json as _json
        # Build compact findings summary. Keep this cheap — we only
        # need enough to let the probe judge answerability.
        _findings_text = ""
        _facts = state.findings.get("accumulated_facts") or []
        if _facts:
            _lines = []
            for f in _facts[:12]:
                txt = str(f.get("text") if isinstance(f, dict) else f).strip()
                if txt:
                    _lines.append(f"- {txt[:200]}")
            _findings_text = "\n".join(_lines) or "(no facts accumulated yet)"
        else:
            _findings_text = "(no facts accumulated yet)"
        _cit_lines: list[str] = []
        for _c in (state.citations or [])[:10]:
            _cit_lines.append(f"- {getattr(_c, 'document', '')}")
        if _cit_lines:
            _findings_text += "\n\nCitations:\n" + "\n".join(_cit_lines)
        # Coverage snapshot.
        _cov_text = ""
        if self._matter_model is not None:
            try:
                _cov_rows = self._matter_model.get_issue_coverage_report(
                    policy_audience="internal",
                )[:8]
                _cov_text = "\n".join(
                    f"- {r.get('title', '?')[:70]}: "
                    f"coverage {float(r.get('coverage_fraction') or 0.0):.0%}, "
                    f"{int(r.get('supporting_count') or 0)} supporting"
                    for r in _cov_rows
                ) or "(no open issues)"
            except Exception:
                _cov_text = "(coverage snapshot unavailable)"
        else:
            _cov_text = "(no matter model)"
        prompt = self._SUFFICIENCY_PROBE_PROMPT.format(
            query=state.query[:400],
            findings_summary=_findings_text[:2000],
            coverage_summary=_cov_text[:1000],
        )
        state.findings["_probes_used"] = _probes_used + 1
        try:
            from ..core.models import ModelTier as _ModelTier
            raw = await self.client.complete(
                prompt,
                tier=_ModelTier.LITE,
                json_mode=True,
                usage_label="sufficiency_probe",
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("sufficiency_probe call failed: %s", exc)
            return False
        try:
            parsed = _json.loads(raw or "{}")
        except Exception:
            return False
        if not parsed.get("can_answer"):
            return False
        confidence = str(parsed.get("confidence") or "low").lower()
        citations = [
            c.strip() for c in (parsed.get("citations") or [])
            if isinstance(c, str) and not isinstance(c, bool) and c.strip()
        ]
        answer = str(parsed.get("answer") or "").strip()
        if confidence not in {"medium", "high"} or not citations or not answer:
            # Don't trust "can_answer=true" with thin evidence — the
            # LLM may be eager. Require the same threshold the read
            # handler uses.
            return False
        # adv#14 Finding #1: cross-check the probe's cited docs
        # against documents we've ACTUALLY seen in this run. A LITE
        # probe can invent "ghost.pdf" to satisfy its own citation
        # floor — treating that as sufficient would terminate a
        # real investigation on fabricated evidence. Build the set
        # of known doc identifiers from state.citations plus
        # accumulated_facts, then require at least one probe
        # citation to match something real.
        known_docs: set[str] = set()
        for _c in (state.citations or []):
            _d = (getattr(_c, "document", "") or "").strip().lower()
            if _d:
                known_docs.add(_d)
                known_docs.add(_d.rsplit("/", 1)[-1])  # basename too
        for _f in (state.findings.get("accumulated_facts") or []):
            if isinstance(_f, dict):
                _d = str(_f.get("document") or _f.get("source_document") or "").strip().lower()
                if _d:
                    known_docs.add(_d)
                    known_docs.add(_d.rsplit("/", 1)[-1])
        matched = False
        for _cite in citations:
            _cl = _cite.strip().lower()
            if _cl in known_docs or _cl.rsplit("/", 1)[-1] in known_docs:
                matched = True
                break
        if not matched:
            logger.warning(
                "sufficiency_probe: rejected can_answer=true — "
                "none of %r match state.citations / accumulated_facts",
                citations,
            )
            return False
        # Stamp the early-terminate decision. The engine loop reads
        # this on the next _should_continue_investigation call and
        # breaks out.
        state.findings["sufficiency_probe_confidence"] = confidence
        state.findings["sufficiency_probe_citations"] = citations
        self._emit_output(state, answer, emitter="sufficiency_probe")
        state.early_terminate_reason = (
            f"matter already answers the query at {confidence} confidence "
            f"with {len(citations)} citation(s); skipping further iterations"
        )
        self._emit_step(
            state,
            StepType.THINKING,
            f"Sufficiency probe: can answer now ({confidence} confidence, "
            f"{len(citations)} citation(s)) — stopping loop.",
        )
        return True

    @staticmethod
    def _viable_leads(pending: list, contract: Any) -> list:
        """Lead viability check. Two-mode: when a lead carries a
        populated ev_score (MVI-5 enrichment ran), the floor is read
        as coverage-per-dollar. When ev_score is 0 (legacy path), the
        floor is read as a raw priority threshold — preserves old
        behavior for callers that haven't adopted EV yet.

        Adversarial #10 Fix E: validate lead_ev_floor rigorously.
        The old `or floor` silently coerced `0.0` back to the default
        0.5 (so a caller trying to disable gating got partial gating
        instead), and accepted NaN/±inf unchallenged. Now: explicit
        0.0 means "no floor, everything passes"; non-finite values
        are rejected and logged, defaulting to 0.5.
        """
        import math as _math
        floor = 0.5
        if contract is not None:
            raw = getattr(contract, "lead_ev_floor", floor)
            try:
                candidate = float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "lead_ev_floor=%r is not a number; defaulting to 0.5", raw,
                )
                candidate = 0.5
            if not _math.isfinite(candidate):
                logger.warning(
                    "lead_ev_floor=%r is non-finite; defaulting to 0.5", raw,
                )
                floor = 0.5
            else:
                floor = candidate
        viable = []
        for lead in pending:
            ev = getattr(lead, "ev_score", 0.0)
            if ev > 0:
                if ev >= floor:
                    viable.append(lead)
            else:
                if lead.priority >= floor:
                    viable.append(lead)
        return viable

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

        # 3. Domain-relevant terms
        domain_terms = {
            'contract', 'agreement', 'breach', 'damages', 'liability',
            'warranty', 'negligence', 'fraud', 'misrepresentation',
            'estimate', 'inspection', 'maintenance', 'invoice', 'payment',
            'revenue', 'compliance', 'specification', 'requirement',
            'finding', 'conclusion', 'diagnosis', 'assessment',
            'margin', 'debt', 'valuation', 'guidance', 'cash',
            'bug', 'error', 'latency', 'security', 'dependency', 'test',
            'trial', 'study', 'endpoint', 'cohort', 'sample', 'safety',
        }
        for w in words:
            if w.lower() in domain_terms:
                priority_words.append(w)

        # Use the highest priority word found, or fall back to first meaningful word
        if priority_words:
            return priority_words[0]

        return words[0]

    @staticmethod
    def _unique_display_name(file_path: str, all_paths: list) -> str:
        """Return shortest path suffix of file_path that is unique among all_paths.

        Starts from just the filename and adds parent components until unique.
        Falls back to the full path if all suffix depths collide (highly unlikely).
        """
        parts = Path(file_path).parts
        for depth in range(1, len(parts) + 1):
            candidate = "/".join(parts[-depth:])
            if sum(
                1 for p in all_paths
                if "/".join(Path(p).parts[-depth:]).lower() == candidate.lower()
            ) == 1:
                return candidate
        return file_path  # full path as ultimate fallback

    def _format_search_results(self, results: SearchResults, max_hits: int = 10) -> str:
        """Format search results for LLM consumption.

        Uses the shortest unique path suffix per hit so the model can return a
        stable, unambiguous file identifier even when basenames collide.
        """
        hits = list(results.top(max_hits))
        all_paths = list(dict.fromkeys(h.file_path for h in hits))
        lines = []
        for hit in hits:
            display_name = self._unique_display_name(hit.file_path, all_paths)
            lines.append(f"File: {display_name} (page {hit.page_num})")
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
            # Guard: LLM may return valid JSON that is not an object (null, [], "x", etc.)
            if not isinstance(result, dict):
                logger.warning(
                    "LLM returned non-dict JSON root (%s), using defaults",
                    type(result).__name__,
                )
                return defaults
            # Merge with defaults for any missing keys
            for key, value in defaults.items():
                if key not in result:
                    result[key] = value
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSON parse failed: {e}, response preview: {text[:200] if text else 'empty'}")
            return defaults

    def _detect_proof_gaps(self, policy_audience: str = "clean") -> None:
        """Record proof gaps for high-priority issues with no supporting assertions (SO-7).

        An issue that exists in the model but has zero supporting-assertion links is
        a 'proof gap' — the system recognised the claim but found no evidence for it.
        These are surfaced as GapType.MISSING_ISSUE_PREDICATE (the semantically correct
        type: a predicate/element required to satisfy the issue is unproven) with the
        issue linked so that generate_clarifications_from_gaps() can generate targeted
        questions.

        Also resolves previously-open proof gaps when an issue now has active support:
        a gap that was opened in a prior run is closed once new assertions fill it.

        MVP.4 (SO-5): under policy_audience='clean', assertions sourced from
        privileged documents don't count as support. Clean support must exist
        to close a gap, and a clean-support-empty / privileged-support-only
        issue still gets a missing_issue_predicate gap opened.

        Uses a single NOT EXISTS SQL query instead of two Python-level IN-list queries to:
          (a) avoid SQLite variable-count limits on large matters (>999 issues),
          (b) scope the existing-gap check to proof-gap type only, so an unrelated
              issue-linked gap (e.g. missing_document "exhibit A") does not suppress
              the zero-support proof gap.
        """
        if self._matter_model is None:
            return
        from ..matter.enums import GapType
        from datetime import datetime, timezone as _tz
        mid = self._matter_model.matter_id
        _ts = datetime.now(_tz.utc).isoformat()
        # MVP.4 clean-mode privilege filter. Applied to every assertion join
        # so privileged-only support cannot close or suppress a proof gap.
        priv_sql = ""
        if policy_audience == "clean":
            priv_sql = (
                " AND a.id NOT IN ("
                " SELECT DISTINCT ao.assertion_id FROM assertion_occurrence ao"
                " LEFT JOIN document_inventory di"
                "   ON di.id = ao.document_inventory_id"
                "   OR di.relative_path = ao.document_id"
                " JOIN document_card dc ON dc.doc_id = di.id"
                " WHERE di.matter_id = a.matter_id AND dc.privilege_flag = 1"
                ")"
            )

        # Resolve any proof gaps for issues that NOW have active supporting assertions.
        # This closes gaps that were opened in a prior iteration when the issue lacked support.
        self._matter_model.db.execute(
            f"""UPDATE gap SET status='resolved', updated_at=?
               WHERE matter_id=? AND status='open'
                 AND gap_type='missing_issue_predicate'
                 AND EXISTS (
                     SELECT 1 FROM gap_link gl
                     WHERE gl.gap_id=gap.id AND gl.affected_type='issue'
                       AND EXISTS (
                           SELECT 1 FROM assertion_issue_link ail
                           JOIN assertion a ON a.id=ail.assertion_id
                           LEFT JOIN verification_state vs
                             ON vs.target_kind='assertion'
                            AND vs.target_id=a.id
                            AND vs.matter_id=a.matter_id
                           WHERE ail.issue_id=gl.affected_id
                             AND ail.relation_type IN ('supports','establishes')
                             AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                             AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
                             {priv_sql}
                       )
                       -- MVP.3: an issue with any active edge-backed
                       -- support also closes the legacy gap.
                       OR EXISTS (
                           SELECT 1 FROM evidence_edge ee
                           JOIN assertion a ON a.id=ee.source_id
                           LEFT JOIN verification_state vs
                             ON vs.target_kind='assertion'
                            AND vs.target_id=a.id
                            AND vs.matter_id=a.matter_id
                           LEFT JOIN verification_state vs_edge
                             ON vs_edge.target_kind='evidence_edge'
                            AND vs_edge.target_id=ee.id
                            AND vs_edge.matter_id=ee.matter_id
                           WHERE ee.matter_id=? AND ee.target_kind='issue'
                             AND ee.target_id=gl.affected_id AND ee.active=1
                             AND ee.source_kind='assertion'
                             AND ee.relation_type IN ('supports','establishes')
                             AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                             AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
                             AND COALESCE(vs_edge.status, ee.verification_status, 'candidate') NOT IN ('rejected','stale')
                             {priv_sql}
                       )
                 )""",
            (_ts, mid, mid),
        )

        # MVP.3: an issue is "unsupported" for gap-detection purposes only
        # when it has no active support in EITHER substrate. This prevents
        # a missing-predicate gap from opening when the issue is supported
        # entirely through evidence_edge.
        # MVP.4 clean mode: privileged-only support counts as unsupported
        # for gap detection, so a privileged memo cannot silently satisfy
        # a claim.
        rows = self._matter_model.db.execute(
            f"""SELECT i.id, i.title, i.materiality
               FROM issue i
               WHERE i.matter_id=? AND i.status='open' AND i.materiality >= 0.4
                 AND NOT EXISTS (
                     SELECT 1 FROM assertion_issue_link ail
                     JOIN assertion a ON a.id=ail.assertion_id
                     LEFT JOIN verification_state vs
                       ON vs.target_kind='assertion'
                      AND vs.target_id=a.id
                      AND vs.matter_id=a.matter_id
                     WHERE ail.issue_id=i.id
                       AND ail.relation_type IN ('supports','establishes')
                       AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                       AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
                       {priv_sql}
                 )
                 AND NOT EXISTS (
                     SELECT 1 FROM evidence_edge ee
                     JOIN assertion a ON a.id=ee.source_id
                     LEFT JOIN verification_state vs
                       ON vs.target_kind='assertion'
                      AND vs.target_id=a.id
                      AND vs.matter_id=a.matter_id
                     LEFT JOIN verification_state vs_edge
                       ON vs_edge.target_kind='evidence_edge'
                      AND vs_edge.target_id=ee.id
                      AND vs_edge.matter_id=ee.matter_id
                     WHERE ee.matter_id=? AND ee.target_kind='issue'
                       AND ee.target_id=i.id AND ee.active=1
                       AND ee.source_kind='assertion'
                       AND ee.relation_type IN ('supports','establishes')
                       AND a.belief_state NOT IN ('disputed','withdrawn','superseded')
                       AND COALESCE(vs.status, 'candidate') NOT IN ('rejected','stale')
                       AND COALESCE(vs_edge.status, ee.verification_status, 'candidate') NOT IN ('rejected','stale')
                       {priv_sql}
                 )
                 AND NOT EXISTS (
                     SELECT 1 FROM gap g
                     JOIN gap_link gl ON gl.gap_id=g.id
                     WHERE g.matter_id=? AND g.status='open'
                       AND g.gap_type='missing_issue_predicate'
                       AND gl.affected_type='issue' AND gl.affected_id=i.id
                 )""",
            (mid, mid, mid),
        ).fetchall()

        if rows:
            self._matter_model.gaps.record_many([
                {
                    "gap_type": GapType.MISSING_ISSUE_PREDICATE,
                    "description": f"No supporting evidence found for issue: '{row['title']}'",
                    "expected_artifact": f"Evidence supporting: {row['title']}",
                    "materiality": row["materiality"] or 0.5,
                    "affected_type": "issue",
                    "affected_id": row["id"],
                }
                for row in rows
            ])

    def _cleanup_checkpoints(self, state: InvestigationState) -> None:
        """Delete all checkpoint files for this state on run completion/failure.

        Prevents unbounded disk accumulation: both per-iteration files
        (checkpoint_<state.id>_iter*.json) and the latest pointer
        (latest_<state.id>.json) are removed.

        Uses exact prefixes that match _save_checkpoint() naming to avoid
        matching unrelated JSON files that happen to contain the state id substring.
        """
        if not self.config.checkpoint_dir:
            return
        try:
            matter_id = (
                self._matter_model.matter_id
                if self._matter_model is not None
                else getattr(state, "_matter_id", "default")
            )
            ckpt_dir = Path(self.config.checkpoint_dir) / matter_id
            for pattern in (
                f"latest_{state.id}.json",
                f"checkpoint_{state.id}_iter*.json",
            ):
                for f in list(ckpt_dir.glob(pattern)):
                    f.unlink(missing_ok=True)
        except Exception:
            pass

    def _save_checkpoint(
        self, state: InvestigationState, iteration: "int | None" = None
    ):
        """Save investigation checkpoint.

        iteration=None is used on the forced stop path (no clean iteration boundary).
        In that case only the latest_<state.id>.json file is written (no per-iter file).
        Also writes the latest checkpoint path into run_session.next_action so the
        resume route can locate the file without scanning the filesystem.
        """
        if not self.config.checkpoint_dir:
            return

        # Use per-matter subdirectory so checkpoint files from different matters
        # cannot collide even if state.id (8-char) repeats across many matters.
        matter_id = (
            self._matter_model.matter_id if self._matter_model is not None else "default"
        )
        ckpt_dir = Path(self.config.checkpoint_dir) / matter_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Stamp matter_id onto state so the checkpoint payload carries its own identity.
        # Resume validation can then reject a checkpoint loaded for the wrong matter.
        if not getattr(state, "_matter_id", None) and self._matter_model is not None:
            state._matter_id = self._matter_model.matter_id

        if iteration is not None:
            checkpoint_path = ckpt_dir / f"checkpoint_{state.id}_iter{iteration}.json"
            state.save_checkpoint(checkpoint_path)
            logger.info("Saved checkpoint: %s", checkpoint_path)

        # Always write/overwrite the latest pointer
        latest_path = ckpt_dir / f"latest_{state.id}.json"
        state.save_checkpoint(latest_path)

        # Persist latest checkpoint path into run_session.next_action for resume
        run_id = getattr(state, "_run_id", None)
        if run_id is not None and self._matter_model is not None:
            try:
                self._matter_model.ledger.set_next_action(run_id, str(latest_path))
            except Exception as _sna_exc:
                # MEDIUM r81: log so operators know this run may be non-resumable
                logger.warning(
                    "set_next_action(%s, %s) failed during checkpoint — run may not "
                    "be resumable if this persists: %s",
                    run_id, latest_path, _sna_exc,
                )

    async def resume_investigation(
        self,
        checkpoint_path: str | Path,
        original_run_id: "str | None" = None,
        follow_up_query: "str | None" = None,
        research_mode: "str | None" = None,
        conversation_history: "list[dict[str, str]] | None" = None,
    ) -> InvestigationState:
        """
        Resume investigation from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            original_run_id: The interrupted run_session.id to check for a pending
                redirect (set by user via request_redirect() after stop). If present
                and redirect_requested=1, the redirect is propagated to the new run.
            follow_up_query: Optional new user query to continue from the checkpoint
                with a refined objective while keeping prior state.

        Returns:
            InvestigationState with completed investigation
        """
        state = InvestigationState.load_checkpoint(checkpoint_path)
        if conversation_history is not None:
            state.conversation_history = [
                {"query": str(turn.get("query") or "").strip(), "answer": str(turn.get("answer") or "").strip()}
                for turn in conversation_history
                if str(turn.get("query") or "").strip() or str(turn.get("answer") or "").strip()
            ]
        previous_mode = normalize_research_mode(getattr(state, "research_mode", None))
        state.research_mode = normalize_research_mode(
            research_mode,
            default=previous_mode,
        )
        mode_changed = state.research_mode != previous_mode
        _follow_up_query = (follow_up_query or "").strip()
        if _follow_up_query:
            _prior_query = state.query
            state.findings.setdefault("query_history", [])
            state.findings["query_history"].append(_prior_query)
            state.findings["continued_from_query"] = _prior_query
            state.findings["follow_up_query"] = _follow_up_query
            state.query = _follow_up_query
            state.add_lead(
                description=f"Follow-up query: {_follow_up_query}",
                source="follow_up_query",
                priority=1.0,
                search_term=_follow_up_query,
            )
        # MEDIUM r72/r74: scrub any stale final_output from non-terminal checkpoints.
        # Checkpoints are written BEFORE interrupt() flips state.status (state.py:1496),
        # so a checkpoint from an interrupted run serializes its pre-interrupt status
        # (typically "running"), never "interrupted". The correct guard is
        # "not completed/failed" — those are the only statuses that carry a legitimate
        # final memo. Completed/failed checkpoints must not be erased (r73).
        # Service/UI callers already reject non-interrupted runs via ledger status, but
        # guard at the engine level for direct callers too.
        if state.status not in ("completed", "failed"):
            state.findings.pop("final_output", None)
            state.findings.pop("output_envelope", None)
            state.output_envelope = None
        repo = MatterRepository(state.repository_path)

        self._emit_step(state, StepType.THINKING, "Resuming investigation from checkpoint")

        # Wire matter adapter so resumed runs get ledger entries + stop propagation
        from ..matter.runtime import MatterRuntimeAdapter, NullMatterAdapter
        run_id = None
        _orig_redirect_issue: "str | None" = None
        # HIGH r76: track whether this call actually won the CAS so the except block
        # can gate next_action/redirect restoration on only the winner.
        _claimed = False
        _usage_ctx = None

        try:
            # HIGH r75/r76: Concurrent-resume CAS — atomically claim the checkpoint by
            # clearing next_action BEFORE creating a new run row.  The entire setup
            # block lives inside this try so that any failure after the claim (e.g.
            # start_run() throws) lands in the except block and restores next_action.
            if (
                original_run_id is not None
                and self.config.enable_matter_model
                and self._matter_model is not None
            ):
                _claimed = self._matter_model.ledger.clear_next_action(original_run_id)
                if not _claimed:
                    raise ConcurrentResumeError(
                        f"Run '{original_run_id}' cannot be resumed: it is not in "
                        "'interrupted' status, or a concurrent resume already claimed it"
                    )
                # Fenced read: get redirect state AFTER claiming so we see any
                # concurrent redirect that arrived before the fence.
                orig = self._matter_model.ledger.get_run(original_run_id)
                if orig and orig.redirect_requested and orig.active_branch_issue_id:
                    _orig_redirect_issue = orig.active_branch_issue_id

            if self.config.enable_matter_model and self._matter_model is not None:
                run_id = self._matter_model.start_run(
                    f"Resume: {state.query[:120]}",
                    resumed_from=original_run_id,
                    research_mode=state.research_mode,
                )
                state._matter_adapter = MatterRuntimeAdapter(self._matter_model, run_id)
                _usage_ctx = self.client.begin_usage_context(
                    matter_id=self._matter_model.matter_id,
                    run_id=run_id,
                    recorder=self._matter_model.record_llm_call,
                )

                # Propagate the captured redirect to the new run (SO-3)
                if original_run_id is not None and _orig_redirect_issue is not None:
                    try:
                        self._matter_model.ledger.request_redirect(
                            run_id, _orig_redirect_issue
                        )
                        self._matter_model.ledger.clear_redirect(original_run_id)
                    except Exception:
                        pass
            else:
                state._matter_adapter = NullMatterAdapter()

            if mode_changed:
                state.findings["continued_from_research_mode"] = previous_mode
                state.findings["research_mode_override"] = state.research_mode
                self._emit_step(
                    state,
                    StepType.THINKING,
                    "Research mode changed from "
                    f"{self._research_mode_label(previous_mode)} to "
                    f"{self._research_mode_label(state.research_mode)}",
                )
            else:
                self._emit_step(
                    state,
                    StepType.THINKING,
                    f"Research mode: {self._research_mode_label(state.research_mode)}",
                )

            # Continue investigation loop if not already complete
            if state.status not in ("completed", "failed"):
                # Set run_id on state so periodic checkpoints write next_action correctly
                if run_id is not None:
                    state._run_id = run_id

                await self._investigate_loop(state, repo)

                # Mirror normal investigate() stop/interrupt branch
                _adapter = getattr(state, "_matter_adapter", None)
                if _adapter is not None and _adapter.is_stop_requested():
                    self._emit_step(
                        state, StepType.THINKING,
                        "Stopped by user during resumed investigation — partial state preserved",
                    )
                    self._save_checkpoint(state, iteration=None)
                    state.interrupt()
                    if run_id is not None:
                        self._matter_model.interrupt_run(run_id)
                    return state

                await self._verify_citations(state, repo)

                # Re-check stop after verify — same as investigate() path (HIGH r69)
                _adapter_post_verify = getattr(state, "_matter_adapter", None)
                if _adapter_post_verify is not None and _adapter_post_verify.is_stop_requested():
                    self._emit_step(
                        state, StepType.THINKING,
                        "Stopped by user during citation verification — partial state preserved",
                    )
                    self._save_checkpoint(state, iteration=None)
                    state.interrupt()
                    if run_id is not None:
                        self._matter_model.interrupt_run(run_id)
                    return state

                # Phase 2.75: Same maintenance block as normal investigate()
                if run_id is not None:
                    try:
                        self._matter_model.detect_quant_conflicts(run_id=run_id)
                    except Exception as _qc_exc:
                        logger.warning("Quant conflict detection failed, continuing: %s", _qc_exc)
                    try:
                        self._detect_proof_gaps()
                    except Exception as _pg_exc:
                        logger.warning("Proof gap detection failed, continuing: %s", _pg_exc)
                    try:
                        self._matter_model.mine_contradictions(run_id=run_id)
                    except Exception as _mc_exc:
                        logger.warning("Contradiction mining failed, continuing: %s", _mc_exc)
                    try:
                        self._matter_model.refresh_document_families()
                    except Exception as _vc_exc:
                        logger.warning("Version chain detection failed, continuing: %s", _vc_exc)

                await self._synthesize(state)

                # HIGH adv#035: same final_output presence check as investigate() path
                if "final_output" not in state.findings:
                    self._emit_step(
                        state, StepType.THINKING,
                        "Stopped by user — partial state preserved",
                    )
                    self._save_checkpoint(state, iteration=None)
                    state.interrupt()
                    if run_id is not None:
                        self._matter_model.interrupt_run(run_id)
                    return state

                state.complete()
                if run_id is not None:
                    # adv#036 MEDIUM (r90 fix): check BEFORE complete_run() — same
                    # rationale as investigate() path above.
                    try:
                        if self._matter_model.ledger.is_redirect_requested(run_id):
                            self._matter_model.ledger.clear_redirect(run_id)
                            from irys.matter.enums import LedgerEventType as _LET
                            self._matter_model.ledger.append_event(
                                run_id=run_id,
                                event_type=_LET.USER_REDIRECTED,
                                summary=(
                                    "Redirect received too late — investigation completed "
                                    "before it could be applied; resubmit on a new run"
                                ),
                                why="adv#036: late redirect cleared before run completion",
                            )
                    except Exception:
                        pass
                    self._cleanup_checkpoints(state)
                    self._matter_model.complete_run(
                        run_id,
                        llm_calls_avoided=state.llm_calls_avoided,
                        llm_calls_required=state.llm_calls_required,
                    )
                    # Mirror normal completion tail: clarifications + reasoning trail
                    try:
                        self._matter_model.generate_clarifications_from_gaps(
                            run_id=run_id, top_n=3, min_materiality=0.5
                        )
                        state.pending_clarifications = (
                            self._matter_model.clarifications.get_pending()
                        )
                    except Exception:
                        pass
                    try:
                        state.reasoning_trail = self._matter_model.ledger.get_events(run_id)
                    except Exception:
                        pass

        except ConcurrentResumeError:
            # CAS was rejected — next_action was never cleared, so no restore needed.
            # Propagate as-is; the service layer converts this to HTTP 409.
            raise

        except Exception as e:
            state.fail(str(e))
            # MEDIUM adv#035: restore BEFORE fail_run() so a DB failure in fail_run()
            # (e.g. lock/busy) cannot leave the original run orphaned after the CAS claim.
            # HIGH r69/r76: only restore when _claimed=True.
            if _claimed and original_run_id is not None and self._matter_model is not None:
                try:
                    self._matter_model.ledger.set_next_action(
                        original_run_id, str(checkpoint_path)
                    )
                except Exception as _restore_exc:
                    # MEDIUM r80: if restore write fails the original run is permanently
                    # non-resumable via normal routes. Log a WARNING with the run_id so
                    # operators can manually repair (set next_action via direct DB write).
                    logger.warning(
                        "CRITICAL: failed to restore next_action on original interrupted "
                        "run '%s' after resume failure — run may require manual repair. "
                        "checkpoint_path=%s error=%s",
                        original_run_id, checkpoint_path, _restore_exc,
                    )
                # Restore redirect flag (MEDIUM r70)
                if _orig_redirect_issue is not None:
                    try:
                        self._matter_model.ledger.request_redirect(
                            original_run_id, _orig_redirect_issue
                        )
                    except Exception:
                        pass
            if run_id is not None:
                # Do NOT clean up checkpoints on resume failure — the checkpoint
                # (state.id file) is the original interrupted run's checkpoint and
                # may still be valid for a re-resume attempt. (adv#034 MEDIUM)
                # MEDIUM r78: if fail_run() throws (e.g. DB locked), attempt a bare
                # autocommit UPDATE as a last-resort fallback so the new run doesn't
                # stay 'running' and block future resumes via the running-run guard.
                try:
                    self._matter_model.fail_run(run_id, str(e))
                except Exception as _fail_exc:
                    logger.warning(
                        "fail_run(%s) raised during resume cleanup; attempting direct "
                        "status update fallback: %s",
                        run_id, _fail_exc,
                    )
                    try:
                        # MEDIUM r79: also set completed_at + next_action=NULL to
                        # match fail_run() semantics (no event logged — best-effort)
                        # r94 LOW: also clear steering flags to match fail_run()
                        from datetime import datetime as _datetime, timezone as _tz
                        _now_iso = _datetime.now(_tz.utc).isoformat()
                        self._matter_model.ledger.db.execute(
                            "UPDATE run_session SET status='failed', completed_at=?,"
                            " next_action=NULL, stop_requested=0, redirect_requested=0"
                            " WHERE id=? AND matter_id=?",
                            (_now_iso, run_id, self._matter_model.matter_id),
                        )
                    except Exception:
                        pass  # Best-effort; manual cleanup may be needed
            raise
        finally:
            if _usage_ctx is not None:
                self.client.end_usage_context(_usage_ctx)

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
            query: The potentially compound query

        Returns:
            List of sub-queries with metadata:
            [{"query": "...", "priority": 0-1, "depends_on": None or query_id}]
        """
        prompt = f"""You are a research assistant. Analyze this query and determine if it should be broken into sub-queries.

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

        response = await self.client.complete(
            prompt,
            tier=ModelTier.LITE,
            usage_label="query_decomposition",
        )

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
            query: The query (may be compound)
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

