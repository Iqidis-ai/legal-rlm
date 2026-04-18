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
from .state import (
    InvestigationState,
    StepType,
    ThinkingStep,
    Citation,
    Lead,
    ResearchMode,
    normalize_research_mode,
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

    coverage_tokens: int = 256
    gap_tokens: int = 256
    orientation_tokens: int = 800
    per_optional_section_tokens: int = 384
    synthesis_total_tokens: int = 3000


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
    enable_matter_model: bool = True  # When True, persist facts to SQLite matter model
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
- Apply pragmatic legal judgment when choosing what to pursue next, including procedural posture, litigation risk, timing, remedy exposure, commercial realities, and business constraints.
- If the record points in a different direction than the user's apparent assumption, surface that clearly, but still organize the research around answering the user's question.
- Favor the highest-yield next step, not the most intellectually interesting one.
""".strip()

ORIENTATION_PROMPT = """You are an expert legal analyst conducting due diligence on a document repository.

Repository Structure:
{structure}

Document Listing:
{file_listing}

Total files: {total_files}

User Query: {query}
{matter_context}
Your task is to create a strategic research plan. Think like an experienced litigator or investigator.

{research_alignment_guidance}

Consider:
1. READ THE DOCUMENT LISTING CAREFULLY. File names reveal what each document IS (e.g., "Master_Service_Agreement.pdf" is a contract, "Complaint_Filed_2024.pdf" is a pleading, "Invoice_March.xlsx" is financial). Use file names to identify the MOST IMPORTANT documents.
2. What are the CORE legal issues that need to be established?
3. Which specific documents from the listing are MOST LIKELY to contain direct evidence? Name them explicitly in your search terms.
4. What SPECIFIC search terms will find relevant passages? Use party names, document-specific terms, and key phrases you expect to find IN those documents.
5. What is your preliminary hypothesis based on the query and the document names?

PRIORITIZE:
- Primary source documents (contracts, pleadings, agreements) over secondary (correspondence)
- Documents whose filenames suggest they contain key evidence for the query
- Documents with dates matching key events
- Files mentioning specific parties or amounts
- Existing Matter Intelligence about open gaps, missing evidence, and weakly supported issues
- If a PRIORITY FOCUS issue is listed in Existing Matter Intelligence, direct the first 2-3 `initial_searches` specifically toward that issue before broadening to general exploration

Respond in JSON format:
{{
    "issues": [
        {{
            "title": "issue description",
            "type": "claim|defense|damages|contract_question|procedural|evidentiary|condition_precedent|waiver|diligence_red_flag|compliance_failure",
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

For target_documents: list the EXACT filenames from the Document Listing above that you
believe are the highest-value retrieval targets. These should be specific files, not types.
Maximum 10 filenames. These will be used as durable retrieval targets throughout the investigation.

Issue types: claim=a party's primary legal claim, defense=an affirmative defense,
damages=a damages component or exposure, contract_question=a disputed contract interpretation,
procedural=a procedural barrier or threshold issue, evidentiary=an evidentiary bottleneck,
condition_precedent=a condition that must be satisfied, waiver=a waiver/estoppel defense,
diligence_red_flag=a due-diligence risk item, compliance_failure=a regulatory violation.

For each issue, include 2-4 "predicates": the specific testable elements that must be
established to prove or defeat that issue (e.g., for breach of contract: ["contract
existence and terms", "defendant's obligation", "failure to perform", "resulting damages"]).
Predicates drive targeted document search — make them concrete and searchable.

For initial_searches: each entry must include "term" (the search string) and "issue_idx"
(0-based index into the issues array above identifying which issue this search targets).
This enables the system to link discovered facts to the correct issue.

IMPORTANT — search term format:
Each "term" must be a simple literal phrase or exact filename that grep can match.
DO NOT use boolean operators (AND, OR, NOT), quotes as search syntax, or wildcards.
Good: "breach of contract", "Invoice_March.xlsx", "termination clause"
Bad: "breach AND contract", "\"termination\" OR \"cancellation\""
"""

# Bump this version string whenever ORIENTATION_PROMPT structure changes.
# Including it in the cache key ensures old cached plans (which may lack
# new fields like "predicates") are automatically invalidated after a
# prompt update (SO-1 stale-cache prevention).
_ORIENTATION_CACHE_VERSION = "8"


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

ANALYZE_FINDINGS_PROMPT = """You are a senior legal analyst extracting evidence from search results.

Query: {query}
Current Hypothesis: {hypothesis}
{issue_focus}
Search Results for "{search_term}":
{search_results}

{research_alignment_guidance}

ANALYZE THESE RESULTS CAREFULLY:

1. KEY FACTS: Extract ONLY the 10 most important specific facts (STRICT LIMIT: 10 maximum):
   - Format each fact as: {{"fact": "...", "source_file": "filename_if_determinable", "issue_relation": "supports|attacks|neutral", "subject": "Party A", "predicate": "agreed_to_pay", "object": "50000 USD by March 2023"}}
   - source_file: the file identifier exactly as shown in the search results (may be "filename.pdf" or "folder/filename.pdf" when multiple files share the same name)
   - issue_relation: whether this fact SUPPORTS the current hypothesis, ATTACKS/undermines it, or is NEUTRAL
   - subject: entity performing the action (person, company) — REQUIRED; provide best-effort even if uncertain (e.g. "plaintiff", "defendant", "contracting_party")
   - predicate: verb/action in snake_case — REQUIRED; describe the relationship (e.g. "agreed_to_pay", "was_employed_by", "executed_contract", "disputes_claim")
   - object: what the predicate applies to (amount, party, date, condition) — REQUIRED; include the key value or description
   - Omit subject/predicate/object ONLY when the fact is purely procedural with no entity relationship
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

4. NEXT SEARCHES: Suggest simple literal phrases that will:
   - Corroborate findings from multiple sources
   - Fill gaps in the evidence
   - Find contradictory evidence (for completeness)
   Each term must be a plain phrase (no AND/OR/NOT operators, no quoted sub-expressions).

5. PREDICATES SATISFIED (SO-4 — only if "Issue Focus" section appears above):
   - List the exact text of any "Element to prove" from the Issue Focus that is
     CLEARLY and DIRECTLY established by the extracted key_facts
   - Only include elements with direct evidence in these search results
   - Empty array if no Issue Focus above or no elements are clearly established

Respond in COMPACT JSON (keep under 3000 chars):
{{
    "key_facts": [{{"fact": "fact text", "source_file": "filename.pdf", "issue_relation": "supports", "subject": "Party A", "predicate": "agreed_to_pay", "object": "50000 USD"}}, ...],
    "fact_relationships": [{{"from_idx": 0, "to_idx": 1, "relation": "corroborates|contradicts|supersedes|supports"}}],
    "new_leads": [{{"desc": "...", "priority": 0.8}}],
    "hypothesis_update": "string or null",
    "next_searches": ["term1", "term2"],
    "predicates_satisfied": ["verbatim element text from Issue Focus, or empty array"],
    "predicates_contested": ["verbatim element text where evidence supports BOTH sides, or empty array"]
}}
"""

# Pre-computed template hash for search-analysis cache versioning.
# Including this in the cache key ensures that changing ANALYZE_FINDINGS_PROMPT
# automatically invalidates all cached analysis from the old template version.
import hashlib as _hashlib
_ANALYZE_PROMPT_VER = _hashlib.sha256(ANALYZE_FINDINGS_PROMPT.encode()).hexdigest()[:12]
del _hashlib  # avoid polluting module namespace

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
   - Format each fact as: {{"fact": "...", "page": N, "issue_relation": "supports|attacks|neutral", "effective_date": "YYYY-MM-DD or null", "subject": "Party A", "predicate": "agreed_to_pay", "object": "50000 USD by March 2023"}}
   - issue_relation: whether the fact SUPPORTS the investigation focus, ATTACKS/undermines it, or is NEUTRAL
   - effective_date: ISO date when this fact became effective/occurred (null if not temporally scoped)
   - subject: entity performing the action (person, company) — REQUIRED; provide best-effort (e.g. "plaintiff", "defendant", "contracting_party")
   - predicate: verb/action in snake_case — REQUIRED (e.g. "agreed_to_pay", "was_employed_by", "executed_contract", "disputes_claim")
   - object: what the predicate applies to (amount, party, date, condition) — REQUIRED; include the key value
   - Omit subject/predicate/object ONLY when the fact has no entity relationship (purely procedural)

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
   - kind: "amount" | "date" | "date_range" | "rate" | "balance" | "count"
   - subject: one-word subject type — "invoice" | "payment" | "fee" | "damages" | "balance" | "rate" | "deposit" | "penalty" | "other"
   - subject_id: specific identifier if present (e.g. "Invoice #1042", "Payment #3", null if none)
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

8. DOC SOURCE ROLE (SO-5 — classify this document by its content, NOT its filename):
   Choose exactly one of: advocacy, operative, authoritative, procedural, informal, draft, post_hoc, unknown
   - advocacy: pleadings, demand letters, briefs, position papers authored by a party to advance their interest
   - operative: signed contracts, executed agreements, court orders, deeds, leases with binding effect
   - authoritative: statutes, regulations, binding case law, official government publications
   - procedural: court filings, discovery materials, motions, notices, subpoenas
   - informal: emails, messages, notes, chats, texts, internal memos not constituting operative documents
   - draft: unsigned or unapproved versions — not yet operative
   - post_hoc: expert reports, declarations, analysis written after the events to explain or opine
   - unknown: cannot determine from document content alone

9. DOCUMENT CARD (classify this document for the matter model):
   - doc_type: broad category — "contract", "pleading", "correspondence", "invoice", "court_order", "memo", "report", "notice", "exhibit", "other"
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

Respond in COMPACT JSON (STRICT: under 4000 chars total):
{{
    "key_facts": [{{"fact": "...", "page": N, "issue_relation": "supports", "effective_date": "2023-03-15", "subject": "Party A", "predicate": "agreed_to_pay", "object": "50000 USD"}}],
    "quotes": [{{"text": "...", "page": N}}],
    "entities": {{"people": ["name1"], "dates": ["date1"], "amounts": ["$X"], "companies": ["co1"]}},
    "numeric_facts": [{{"kind": "amount", "subject": "invoice", "subject_id": "Invoice #1042", "raw": "$50,000", "value": 50000, "currency": "USD", "context": "payment due", "page": 3, "assertion_idx": 2}}, {{"kind": "date", "subject": "payment", "subject_id": null, "raw": "March 2023", "value": "2023-03-01", "date_precision": "month", "context": "payment due date", "page": 1, "assertion_idx": 0}}],
    "fact_relationships": [{{"from_idx": 0, "to_idx": 2, "relation": "supports"}}],
    "connections": ["doc reference 1"],
    "concerns": ["issue 1"],
    "doc_source_role": "advocacy|operative|authoritative|procedural|informal|draft|post_hoc|unknown",
    "doc_type": "contract|pleading|correspondence|invoice|court_order|memo|report|notice|exhibit|other",
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
    "unresolved_flags": ["any open questions about this document"]
}}
"""

# Used when the primary extraction returned zero SPO triples (SO-2 validated extraction).
# A single targeted retry extracts structured triples from the already-extracted fact texts,
# without re-reading the source document.
SPO_RETRY_PROMPT = """Extract subject-predicate-object triples from these legal facts.

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

SYNTHESIS_PROMPT = """You are a senior litigation partner with decades of experience drafting \
legal memoranda, advising on strategy, and presenting analysis to clients and courts.

Think like an experienced litigator. Reason through the evidence carefully. Distinguish \
what the record establishes from what is merely alleged. Identify the strongest and weakest \
points. Consider what an adversary would argue. Surface assumptions that carry the analysis. \
Write with the precision and authority expected of a top-tier law firm.

Original Query: {query}

{context_packet}

---

INSTRUCTIONS:

Analyze the evidence above and produce a comprehensive legal memorandum. Structure your \
analysis based on what the evidence actually warrants — include sections that are useful, \
omit sections that would be empty or speculative.

Your memorandum MUST include:
- An executive summary that leads with the conclusion (2-3 sentences)
- A thorough analysis of the evidence with citations to source documents
- An honest assessment of evidence strength (Strong / Moderate / Weak)
- Actionable recommendations

Your memorandum SHOULD include (when the evidence warrants):
- Factual background: chronological narrative grounded in operative sources
- Financial analysis: when monetary amounts, payments, or damages are at issue
- Contradictions or concerns: conflicting evidence, unresolved issues
- Unsubstantiated claims: allegations supported only by advocacy sources — present \
  as "plaintiff alleges" / "defendant contends," NEVER as established fact

Source-role treatment rules (MANDATORY):
  [ADVOCACY]: allegation or argument by a party — NEVER treat as established fact
  [OPERATIVE]: signed contract, court order, executed document — treat as established
  [AUTHORITATIVE]: statute, regulation, binding case law — treat as controlling
  [PROCEDURAL]: court filing, notice, docket entry — treat as procedurally established
  [INFORMAL]: email, note, draft — corroborative only, not standalone proof
  [UNKNOWN]: unverified source — flag explicitly

Write in formal legal memorandum style. Be precise and cite everything [Document, p. X]. \
Mark unverified citations with [UNVERIFIED]. Do not speculate beyond what evidence supports. \
Do not hide uncertainty — surface assumptions where they carry the analysis.
"""

# Override the legacy memo-only synthesis prompt with the current general legal
# work-product prompt. Keeping the assignment here avoids a large patch churn in
# a heavily edited file while updating the active prompt used by synthesis.
SYNTHESIS_PROMPT = """Role & Standard

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

Final Check Before Responding

Before finalizing, check:
- Did you answer the user's actual request?
- Did you adopt the right communication mode?
- Did you produce the requested artifact in the proper professional form if one was requested?
- Did you distinguish support, inference, and assumption correctly?
- Did you surface the real weaknesses and risks?
- Did you give the user the most useful next steps or clarifying question where needed?
- Is this strong enough that a demanding senior lawyer would trust it?

Original Query: {query}

{context_packet}
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
            return ResearchBudgetProfile(
                mode=mode,
                max_depth=min(self.config.max_depth, 2),
                min_depth=min(self.config.max_depth, 1),
                max_iterations=min(self.config.max_iterations, 5),
                depth_citation_threshold=min(self.config.depth_citation_threshold, 8),
                confidence_threshold=60,
                min_citations=5,
                diminishing_returns_fact_threshold=3,
                diminishing_returns_min_citations=2,
                diminishing_returns_min_confidence=35,
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

    async def investigate(
        self,
        query: str,
        repository_path: str | Path,
        research_mode: "str | None" = None,
        conversation_history: "list[dict[str, str]] | None" = None,
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
        state = InvestigationState.create(
            query,
            str(repo.base_path),
            research_mode=research_mode,
            conversation_history=conversation_history,
        )

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
                    logger.warning("Proof gap detection failed, continuing: %s", _pg_exc)
                try:
                    # Background maintenance: mine contradictions (SO-2) — auto-discovers
                    # heuristic contradiction links and propagates belief state changes.
                    self._matter_model.mine_contradictions(run_id=run_id)
                except Exception as _mc_exc:
                    logger.warning("Contradiction mining failed, continuing: %s", _mc_exc)
                try:
                    # Background maintenance: detect document version chains (SO-1) —
                    # links versioned documents and records gaps for missing base versions.
                    self._matter_model.detect_document_version_chains()
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

        # Read persisted matter state — activates SO-1 (reuse) and SO-4 (issue-driven)
        adapter = getattr(state, "_matter_adapter", None)
        matter_ctx = adapter.get_context() if adapter is not None else None

        # Seed InvestigationState with facts already in the matter model (SO-1 hot reuse)
        if matter_ctx is not None and matter_ctx.existing_assertion_count > 0:
            self._hydrate_from_matter_model(state)

        # MVP.6: cap the durable matter_context block so new stores
        # can't silently inflate the orientation prompt. The repo file
        # listing stays uncapped — it's a direct structural signal the
        # orient prompt needs.
        _budget = self._get_packet_budget()
        _matter_ctx_str = _format_matter_context(matter_ctx)
        _matter_ctx_capped = self._cap_text_by_tokens(
            _matter_ctx_str, _budget.orientation_tokens
        )
        prompt = ORIENTATION_PROMPT.format(
            structure=structure_str,
            file_listing=file_listing_str,
            total_files=stats.total_files,
            query=state.query,
            matter_context=_matter_ctx_capped,
            research_alignment_guidance=RESEARCH_ALIGNMENT_GUIDANCE,
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
            f"\n{_history_digest}\n{_ctx_fingerprint}\nv{_ORIENTATION_CACHE_VERSION}".encode()
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
            )
            plan = self._parse_json_safe(response, _plan_defaults)
            # Persist for future warm runs
            if self._matter_model is not None:
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
                "contract_question": IssueType.CONTRACT_QUESTION,
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
        _raw_searches = (_ps if isinstance(_ps, list) else [])[:5]
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
        # high-priority search leads. These are exact filename strings that the
        # LLM identified as highest-value retrieval targets from the listing.
        _target_docs = plan.get("target_documents") or []
        for _td in (_target_docs if isinstance(_target_docs, list) else [])[:10]:
            if isinstance(_td, str) and _td.strip():
                state.add_lead(
                    description=f"Target document: {_td.strip()}",
                    source="orientation_target",
                    priority=0.85,
                    search_term=_td.strip(),
                    focus_issue_id=_biased_pool[0] if _biased_pool else None,
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

        # Add predicate-driven leads for the weakest issue (SO-4).
        # Predicates are more specific than issue titles — each one is a concrete
        # searchable element (e.g. "failure to perform" vs "Breach of contract").
        # Limit to 3 predicates per orientation to stay within lead budget.
        if self._matter_model is not None and _orient_issue_ids:
            _pred_target_id = weakest_id or _orient_issue_ids[0]
            _issue_predicates = self._matter_model.issues.get_predicates(_pred_target_id, limit=3)
            for _pred_row in _issue_predicates:
                _pred_text = _pred_row.get("description", "").strip()
                if _pred_text:
                    state.add_lead(
                        description=f"Evidence for: {_pred_text}",
                        source="predicate",
                        priority=0.75,
                        search_term=_pred_text,
                        focus_issue_id=_pred_target_id,
                    )

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
        max_iterations = budget.max_iterations

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

            if not pending_leads:
                self._emit_step(state, StepType.THINKING, "No more leads to investigate")
                break

            # SO-4 Leak-1+3: re-score leads by live issue coverage weakness each
            # iteration so weaker issues attract more budget as the run progresses.
            # Guard: only query DB when at least one issue-targeted lead exists in
            # the queue — avoids 3 unnecessary SQL queries on neutral-only iterations.
            _cov_map: "dict[str, tuple[float, bool, int]]" = {}
            if (self._matter_model is not None
                    and any(_l.focus_issue_id for _l in pending_leads)):
                _cov_map = self._get_issue_coverage_map()
                if _cov_map:
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
                                leads_to_process.append(_boot_lead)

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
            recent = self._matter_model.assertions.list_recent_for_hydration(limit=200)
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
        HYDRATION_CAP = 200
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
            # Re-check stop before the FLASH LLM call (SO-3 cooperative stop).
            # If the user stopped the run while we were formatting results, skip the call.
            _adp_pre = getattr(state, "_matter_adapter", None)
            if _adp_pre is not None and _adp_pre.is_stop_requested():
                return
            prompt = ANALYZE_FINDINGS_PROMPT.format(
                query=state.query,
                hypothesis=state.hypothesis or "No hypothesis yet",
                issue_focus=_issue_focus,
                search_term=results.query,
                search_results=results_text,
                research_alignment_guidance=RESEARCH_ALIGNMENT_GUIDANCE,
            )
            # Use FLASH for analysis
            response = await self.client.complete(
                prompt,
                tier=ModelTier.FLASH,
                json_mode=True,
                usage_label="search_analysis",
            )
            analysis = self._parse_json_safe(response, {
                "key_facts": [],
                "new_leads": [],
                "hypothesis_update": None,
                "next_searches": [],
                "predicates_satisfied": [],
                "predicates_contested": [],
            })
            # Cache for warm runs
            if self._matter_model is not None:
                try:
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
                        f"(search: '{search_term[:60]}'). Retrying for missing.",
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
        for _ns in analysis.get("next_searches", [])[:2]:
            if isinstance(_ns, str) and _ns.strip():
                _ns_clean = _ns.strip()
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
            # Preserve salience ordering from list_needing_profile (highest first)
            unprofiled = [r["relative_path"] for r in unprofiled_rows]

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

        # Score new files by query relevance: filename keywords + document type priority.
        # Most informative documents get read first (contracts before misc, query-matching
        # filenames before generic ones).
        from ..core.search import get_document_priority
        query_terms = {w.lower() for w in state.query.split() if len(w) > 3}

        # Use orientation plan hints if available (document_priority, relevant_folders)
        plan = state.findings.get("initial_plan") or {}
        plan_doc_types = [d.lower() for d in (plan.get("document_priority") or [])]
        plan_folders = {f.lower() for f in (plan.get("relevant_folders") or [])}

        def _file_score(f) -> float:
            score = 0.0
            name_lower = f.filename.lower()
            path_lower = str(f.relative_path).lower()

            # Document type priority (contracts=1.5, emails=0.9, etc.)
            score += get_document_priority(f.filename)

            # Filename matches query terms (e.g., query "resignation" matches "Letters of resignation.pdf")
            for term in query_terms:
                if term in name_lower:
                    score += 2.0
                    break  # one match is enough signal

            # Orientation plan ranked this document type
            for rank, doc_type in enumerate(plan_doc_types):
                if doc_type in name_lower:
                    score += 1.0 - (rank * 0.1)  # higher-ranked types score more
                    break

            # File is in a folder the orientation flagged as relevant
            for folder in plan_folders:
                if folder in path_lower:
                    score += 0.5
                    break

            return score

        # Sort by score descending — most relevant first
        new_files.sort(key=_file_score, reverse=True)
        file_paths = [str(f.relative_path) for f in new_files]

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
            state.findings["all_documents_ingested"] = True

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
            prompt = f"""Classify this legal document. Respond in JSON only.

Document: {_fp.name}
Content (excerpt):
{excerpt}

Privilege classification guidance (MVP.4 SO-5):
- privilege_flag=true when the document appears to be attorney-client
  privileged or attorney work-product (e.g. internal legal memo,
  counsel-to-client email, litigation strategy note, settlement memo).
- privilege_flag=unknown when the document has attributes suggesting
  privilege but classification cannot be reliably determined (ambiguous
  internal memo, unsigned legal-looking memo, internal email of
  unknown participants). Unknown is treated as contained in clean
  mode until human review.
- privilege_flag=false only when the document is plainly non-privileged
  (signed contract, opposing-party pleading, public filing, invoice
  from a third party, authoritative statute).

Return:
{{
    "doc_type": "contract|pleading|correspondence|invoice|court_order|memo|report|notice|exhibit|other",
    "doc_subtype": "specific subtype (e.g. services_agreement, demand_letter)",
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

            # Critical documents (contracts, judgments, complaints, agreements) get
            # 3x the excerpt window so we don't lose key clauses to truncation.
            # Uses the same DOCUMENT_PRIORITY weights from search scoring.
            from ..core.search import get_document_priority
            _doc_priority = get_document_priority(doc.filename)
            if _doc_priority >= 1.3:
                _excerpt_chars = min(self.config.excerpt_chars * 3, len(doc.full_text))
            else:
                _excerpt_chars = self.config.excerpt_chars
            content = doc.get_excerpt(_excerpt_chars)

            prompt = DEEP_READ_PROMPT.format(
                filename=doc.filename,
                page_range=f"1-{doc.page_count}",
                content=content,
                query=state.query,
                focus=state.hypothesis or state.query,
            )

            # Use LITE for bulk reading; JSON mode forces valid JSON output
            response = await self.client.complete(
                prompt,
                tier=ModelTier.LITE,
                json_mode=True,
                usage_label="document_deep_read",
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
                            # SO-3: dropped relation must not be silently hidden.
                            # Debug-level to avoid spamming ledger with LLM noise.
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
                                # Same proposition text deduplicates to same assertion_id
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
                    for nf in analysis["numeric_facts"][:20]:  # limit to avoid noise
                        if not isinstance(nf, dict):
                            continue
                        kind = nf.get("kind", "amount")
                        raw = nf.get("raw", "")
                        if not raw:
                            continue
                        value = nf.get("value")
                        try:
                            amount = float(value) if kind == "amount" and value is not None else None
                            rate = float(value) if kind == "rate" and value is not None else None
                        except (TypeError, ValueError, OverflowError):
                            amount = None
                            rate = None
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
                        })
                    if _quant_specs:
                        _adp.record_quants_batch(
                            _quant_specs, document_id=doc.filename
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
                except Exception:
                    pass
                if analysis and analysis.get("quotes"):
                    # Extract authorities from the raw document text processed so far.
                    _quote_text = " ".join(
                        q.get("text", "") for q in analysis["quotes"][:10]
                        if isinstance(q, dict)
                    )
                    if _quote_text:
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
        """Phase 3: Final synthesis using Pro model."""
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
        findings_text = "\n".join(f"• {fact}" for fact in facts[:75])

        # Store citations and entities as structured metadata for UI panels.
        state.findings["metadata_citations"] = state.get_citations_formatted()
        state.findings["metadata_entities"] = state.get_entities_formatted()

        # Dynamically assemble the context packet — only include sections that
        # have real content. PRO gets exactly what's useful, nothing empty.
        context_packet = await self._assemble_context_packet(state, findings_text)

        prompt = SYNTHESIS_PROMPT.format(
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
            # Use PRO for final synthesis
            state.llm_calls_required += 1
            response = await self.client.complete(
                prompt,
                tier=ModelTier.PRO,
                usage_label="synthesis",
                conversation_history=state.conversation_history,
            )
            if self._matter_model is not None:
                try:
                    self._matter_model.cache.put("synthesis", _syn_key, response)
                except Exception:
                    pass

        state.findings["final_output"] = response

        # SO-5: Post-synthesis advocacy gate.
        # If any open issues rely exclusively on advocacy sources, force-append a
        # Source Calibration Advisory that names them. Prompt-level instruction alone
        # is advisory; this is a hard output mutation that cannot be LLM-bypassed.
        if self._matter_model is not None:
            try:
                _adv_enforced = self._enforce_advocacy_gate(response)
                if _adv_enforced is not None:
                    state.findings["final_output"] = _adv_enforced
                    response = _adv_enforced
            except Exception:
                pass  # gate is best-effort; never suppress synthesis

        # SO-6: Post-synthesis quantitative threshold gate.
        # If HIGH violations exist and the LLM skipped the Financial Analysis section,
        # force-append the structured numbers so the output always addresses critical
        # exposure. This is the hard behavioral gate (not just prompt advisory text).
        if self._matter_model is not None:
            try:
                _enforced = self._enforce_quant_threshold_gate(response)
                if _enforced is not None:
                    state.findings["final_output"] = _enforced
                    response = _enforced
            except Exception:
                pass  # gate is best-effort; never suppress synthesis

        # Persist legal citations found in synthesis output to authority store (SO-4).
        if self._matter_model is not None:
            try:
                self._extract_and_store_authorities(
                    response,
                    run_id=getattr(state, "_run_id", None),
                    source_document_ref="synthesis:final",
                )
            except Exception:
                pass  # best-effort; never block synthesis output

        # Refresh proof state for all open issues (SO-4 proof-aware reasoning).
        if self._matter_model is not None:
            try:
                self._matter_model.proof_state.compute_all()
            except Exception:
                pass

        self._emit_step(state, StepType.SYNTHESIS, "Analysis complete")

    def _enforce_advocacy_gate(self, synthesis_output: str) -> Optional[str]:
        """Behavioral gate: ensure advocacy-only issues are explicitly flagged (SO-5).

        Called after LLM synthesis. If any open issues have advocacy_only=True proof
        state AND the output doesn't already contain the Source Calibration Advisory
        marker, appends a structured block naming those issues and flagging them as
        unsupported by operative/authoritative sources.

        Returns augmented output, or None if no action needed.
        This is a hard behavioral output change, not just a prompt instruction.
        """
        if self._matter_model is None:
            return None
        try:
            advocacy_issues = self._matter_model.proof_state.get_advocacy_only()
        except Exception:
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

        # Structural violation check: detect advocacy-only issue content appearing in
        # ## Key Findings or ## Factual Background without hedging markers.
        # This is the hard gate: even if the advisory marker is already present, a
        # structural violation forces re-injection of the advisory block.
        #
        # Hedge check is PER-LINE: each Markdown bullet is checked independently so a
        # hedge phrase in an adjacent bullet cannot suppress a real violation on this line.
        _HEDGE_MARKERS = (
            "alleges", "alleged", "alleged that", "is alleged",
            "contends", "contended", "claims", "claimed",
            "asserts", "asserted", "according to",
            "plaintiff's", "defendant's", "per complaint", "per motion",
            "argued", "argued that", "per defense", "per plaintiff",
            "purportedly", "supposedly", "reportedly",
        )
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
        # gate is satisfied. Uses module-level _ADVOCACY_MARKER_PAT (anchored to
        # line-start + end-of-line; accepts ## and ###; case-insensitive).
        if _ADVOCACY_MARKER_PAT.search(synthesis_output) and not _STRUCTURAL_VIOLATION:
            return None

        violation_note = ""
        if _STRUCTURAL_VIOLATION:
            violation_note = (
                "\n⚠ STRUCTURAL VIOLATION DETECTED: advocacy-only claims found in "
                "## Key Findings or ## Factual Background without hedging. "
                "These are allegations only.\n"
            )

        lines = [
            "",
            f"## {_ADVOCACY_MARKER_NAME}",
            "*(Auto-generated by SO-5 advocacy gate — the following issues lack operative "
            "or authoritative corroboration.)*",
            violation_note,
            "The following issues are supported ONLY by advocacy-authored material "
            "(pleadings, briefs, demand letters). They must appear in "
            "## Unsubstantiated Claims, NOT in ## Key Findings as established facts:",
            "",
        ]
        for ps in active_advocacy:
            issue_id = ps.get("issue_id", "?")
            title = issue_index.get(issue_id, str(issue_id or "?")[:24])
            tw = ps.get("trust_weighted_support", 0.0)
            lines.append(
                f"- **{title}** — advocacy-only "
                f"(trust-weighted support: {tw:.2f})"
            )
        lines.append("")
        return synthesis_output + "\n".join(lines)

    def _enforce_quant_threshold_gate(self, synthesis_output: str) -> Optional[str]:
        """Behavioral gate: force-append financial data when HIGH violations exist (SO-6).

        Called after LLM synthesis. If there are HIGH quantitative threshold violations
        AND the synthesis output does not contain a Financial Analysis section, appends
        a structured block with the exact violation figures.

        Returns the (possibly augmented) output string, or None if no action was taken
        (caller keeps the original). This is a hard behavioral branch, not prompt text.
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

        # Force-append the financial analysis block.
        lines = [
            "",
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

        return synthesis_output + "\n".join(lines)

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
            except Exception:
                pass

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
            except Exception:
                pass

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
            except Exception:
                pass

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
        except Exception:
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
        if not text or cap_tokens <= 0:
            return ""
        if self._estimate_tokens(text) <= cap_tokens:
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
            if used + cost > cap and shown > 0:
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
                if used + cost > cap:
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
            if used + cost > cap:
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
    ) -> str:
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
        if quant:
            _add_optional("quantitative", "Quantitative Summary:\n" + quant)
        citations_text = state.get_citations_formatted()
        if citations_text:
            _add_optional(
                "citations", "Documentary Citations:\n" + citations_text
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

        # Optional sections in stable order behind mandatory ones so the
        # total-cap pass drops them first.
        _ORDER = [
            "source_calibration", "decision_context", "entities",
            "relationships", "quantitative", "citations",
        ]
        for key in _ORDER:
            if key in selected_keys and key in candidates:
                ordered.append((key, candidates[key], False))

        # Total-cap pass. Mandatory sections always go in. Optional
        # sections drop (recorded) once the accumulator would exceed
        # synthesis_total_tokens. If a single mandatory section alone
        # exceeds the cap, it still goes in — we never drop mandatory.
        sections: list[str] = []
        used_tokens = 0
        omitted: list[str] = []
        total_cap = budget.synthesis_total_tokens
        for key, text, mandatory in ordered:
            cost = self._estimate_tokens(text + "\n\n")
            if mandatory or used_tokens + cost <= total_cap:
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
        return packet

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
            "You are preparing a context packet for a senior attorney who will "
            "analyze evidence and write a legal memorandum.\n\n"
            f"Query: {query}\n\n"
            "Available context sections:\n"
            + "\n".join(summaries)
            + "\n\nWhich sections are relevant to answering this query? "
            "Return ONLY a JSON array of the relevant section keys. "
            "Include a section if it would help the attorney reason about "
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

        # Add litigation-side breakdown so the LLM knows whose documents produced facts (SO-5).
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
                # Note: a fact corroborated by documents from multiple sides is counted
                # once per side, so side totals may sum to more than total assertions.
                lines.append("\nLitigation-side origin of extracted facts (may overlap):")
                for sr in side_rows:
                    lines.append(f"  • {sr['cnt']} assertions from {sr['side']} documents")
        except Exception:
            pass

        # Incorporate user-set trust overrides so the LLM respects explicit calibration (SO-5).
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
                        continue  # 'normal' resets to auto; no special instruction needed
                    line = f"  • [{level.upper()}] '{pattern}': {override_label}"
                    if note:
                        line += f" — Reason: {note}"
                    lines.append(line)
        except Exception:
            pass

        # Incorporate user strategic annotations for named documents (SO-3 annotation).
        # These guide the LLM on how to interpret facts from specific documents.
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

        lines.append(
            "\n=== SOURCE TRUST HIERARCHY (MANDATORY — follow this ordering) ===\n"
            "1. OPERATIVE (contracts, signed agreements, court orders) — highest trust\n"
            "2. AUTHORITATIVE (statutes, regulations, published case law)\n"
            "3. PROCEDURAL (filings, docket entries, certificates of service)\n"
            "4. INFORMAL (emails, letters, meeting notes)\n"
            "5. DRAFT (unsigned drafts, redline versions, proposals)\n"
            "6. POST_HOC (post-hoc explanations, summaries written after events)\n"
            "7. ADVOCACY (complaints, briefs, demand letters, pleadings) — lowest trust\n"
            "\nANTI-AMPLIFICATION RULES:\n"
            "• NEVER present advocacy allegations as established fact.\n"
            "• ALWAYS qualify advocacy-sourced claims: 'Plaintiff alleges...', "
            "'According to the complaint...', 'Defendant argues...'.\n"
            "• When advocacy and operative sources conflict, the operative source controls.\n"
            "• Do NOT let advocacy material's confident tone inflate its weight.\n"
            "• If the ONLY source for a proposition is advocacy, explicitly note that "
            "it lacks independent corroboration.\n"
            "• Facts corroborated by multiple source types are stronger than single-source facts."
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

    def _should_continue_investigation(self, state: InvestigationState) -> tuple[bool, str]:
        """Determine if investigation should continue.

        Uses four criteria:
        1. Repository size - small repos terminate faster
        2. Research mode - simpler modes terminate faster
        3. Diminishing returns - stop if recent iterations add few new facts
        4. Verified citations - keep this requirement (per user preference)

        Returns:
            (should_continue, reason) - reason explains why we stopped/continue
        """
        budget = self._get_research_profile(state)
        # Always continue if minimum criteria not met
        if state.max_depth_reached < budget.min_depth:
            return True, "Building minimum evidence base"

        confidence = state.get_confidence_score()
        conf_threshold = budget.confidence_threshold
        min_citations = budget.min_citations

        # NEW: Adjust thresholds for small document sets
        # With fewer documents, we need fewer citations and can terminate earlier
        if self._doc_count <= 5:
            min_citations = min(min_citations, max(2, self._doc_count))
            conf_threshold = max(40, conf_threshold - 15)
        elif self._doc_count <= 10:
            min_citations = min(min_citations, self._doc_count)
            conf_threshold = max(45, conf_threshold - 10)

        mode_label = self._research_mode_label(budget.mode)

        # Check 1: Mode-aware confidence check
        if confidence["score"] >= conf_threshold and len(state.citations) >= min_citations:
            return False, (
                f"Sufficient evidence for {mode_label} mode "
                f"(confidence: {confidence['score']:.0f}%, {len(state.citations)} citations)"
            )

        # Check 2: For small repos, terminate if we've read all documents
        if self._doc_count > 0 and state.documents_read >= self._doc_count:
            if len(state.citations) >= 1:  # At least some evidence found
                return False, f"All {self._doc_count} documents processed"

        # Check 3: Diminishing returns - stop if last 2 iterations added < 3 facts each
        # For small repos, be more aggressive (< 2 facts)
        fact_threshold = budget.diminishing_returns_fact_threshold
        if len(state.facts_per_iteration) >= 2:
            recent_facts = state.facts_per_iteration[-2:]
            if all(f < fact_threshold for f in recent_facts):
                # Diminishing returns detected - but only stop if we have SOME evidence
                min_citations_for_stop = budget.diminishing_returns_min_citations
                if self._doc_count <= 5:
                    min_citations_for_stop = min(min_citations_for_stop, max(2, self._doc_count))
                if (
                    len(state.citations) >= min_citations_for_stop
                    and confidence["score"] >= budget.diminishing_returns_min_confidence
                ):
                    return False, (
                        f"Diminishing returns in {mode_label} mode "
                        f"(last 2 iterations: {recent_facts[0]}, {recent_facts[1]} new facts)"
                    )

        # Check 4: Extreme diminishing returns - 3 iterations with 0-1 facts each
        if len(state.facts_per_iteration) >= 3:
            recent_facts = state.facts_per_iteration[-3:]
            if all(f <= budget.very_low_productivity_max_facts for f in recent_facts):
                # Very low productivity - stop regardless
                return False, (
                    f"Very low productivity in {mode_label} mode "
                    f"(last 3 iterations: {recent_facts} new facts each)"
                )

        # Check if we have pending leads worth investigating
        pending = state.get_pending_leads()
        high_priority = [l for l in pending if l.priority >= 0.5]
        if not high_priority:
            return False, "No high-priority leads remaining"

        return True, (
            f"Continuing {mode_label} investigation "
            f"({len(high_priority)} leads, confidence: {confidence['score']:.0f}%)"
        )

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
                        self._matter_model.detect_document_version_chains()
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

